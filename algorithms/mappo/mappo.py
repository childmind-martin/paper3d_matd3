"""Standard cooperative MAPPO baseline with the current 7D raw-action interface."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from paper3d_train_optimized import OptimizedMATD3, _safe_tensor_to_numpy, parse_hidden_units

from .networks import (
    build_centralized_value_network,
    build_shared_mappo_actor_network,
    repeated_advantages,
    squashed_gaussian_log_prob_components,
    squashed_gaussian_log_prob,
    squashed_gaussian_sample,
)


class OptimizedMAPPO(OptimizedMATD3):
    """A standard on-policy MAPPO baseline that keeps the current environment/action protocol."""

    def __init__(self, n_agents, obs_shapes, action_dims, args):
        setattr(args, "matd3_use_dual_q", False)
        setattr(args, "matd3_use_separated_gradient", False)
        setattr(args, "matd3_use_hybrid_actor_objective", False)
        super().__init__(n_agents, obs_shapes, action_dims, args)
        self.algorithm_family = "ppo"
        self.use_dual_q = False
        self.use_separated_gradient = bool(getattr(self.args, "mappo_use_separated_gradient", False))
        self.use_hybrid_actor_objective = False
        self.actor_objective_mode = "ppo_clip_separated" if self.use_separated_gradient else "ppo_clip"
        self.rollout_length = int(getattr(self.args, "rollout_length", 1024))
        self.ppo_epochs = int(getattr(self.args, "ppo_epochs", 4))
        self.mini_batch_size = int(getattr(self.args, "mini_batch_size", 512))
        self.clip_ratio = float(getattr(self.args, "clip_ratio", 0.2))
        self.gae_lambda = float(getattr(self.args, "gae_lambda", 0.95))
        self.entropy_coef = float(getattr(self.args, "entropy_coef", 0.01))
        self.value_coef = float(getattr(self.args, "value_coef", 0.5))
        self.target_kl = float(getattr(self.args, "target_kl", 0.03))
        self.max_grad_norm = float(getattr(self.args, "max_grad_norm", getattr(self.args, "grad_clip_norm", 10.0)))
        self._last_action_log_probs = None
        self._last_values = None
        self._last_entropies = None
        self._last_pf_features = None
        self._last_actor_means = None
        self._last_update_stats = {}
        self._all_obs_same_shape = len({int(x) for x in self.obs_shapes}) <= 1
        self._shared_obs_dim = int(self.obs_shapes[0]) if self.obs_shapes else 0
        self._agent_index_offsets = np.arange(self.n_agents, dtype=np.int32)
        self.head_action_dim = min(3, int(self.action_dims[0])) if self.action_dims else 0
        self.tail_action_dim = max(0, int(self.action_dims[0]) - self.head_action_dim) if self.action_dims else 0
        self.action_force_ratio_var = tf.Variable(
            float(self.action_force_ratio_cached),
            trainable=False,
            dtype=tf.float32,
            name="mappo_action_force_ratio_var",
        )
        self.base_entropy_coef = float(self.entropy_coef)
        noise_boost = float(self.adaptive_learning.get("noise_boost_factor", 1.2))
        max_noise_scale = float(self.adaptive_learning.get("max_noise_scale", 2.0))
        self.adaptive_learning["entropy_boost_factor"] = float(noise_boost)
        self.adaptive_learning["max_entropy_coef"] = float(
            max(self.base_entropy_coef, self.base_entropy_coef * max_noise_scale)
        )
        self.adaptive_learning["min_entropy_coef"] = float(self.base_entropy_coef)

    def _init_networks(self):
        actor_hidden = parse_hidden_units(getattr(self.args, "actor_hidden", None)) or (256, 256, 256)
        critic_hidden = parse_hidden_units(getattr(self.args, "critic_hidden", None)) or (512, 512, 512)
        self.shared_actor = build_shared_mappo_actor_network(
            input_shape=(self.obs_shapes[0],),
            action_dim=self.action_dims[0],
            hidden_units=actor_hidden,
            use_residual=True,
            use_fr_feature=self.use_fr_feature_flag,
            use_pf_feature=self.use_pf_feature_flag,
            pf_feature_dim=self.pf_feature_dim,
        )
        self.actor_log_std = tf.Variable(
            np.full((self.action_dims[0],), -0.75, dtype=np.float32),
            trainable=True,
            dtype=tf.float32,
            name="mappo_actor_log_std",
        )
        critic_state_dim = int(sum(self.obs_shapes))
        self.value_critic = build_centralized_value_network(
            state_shape=(critic_state_dim,),
            hidden_units=critic_hidden,
            use_residual=True,
            use_fr_feature=self.use_fr_feature_flag,
            use_pf_feature=self.use_pf_feature_flag,
            pf_feature_dim=self.pf_feature_dim,
            n_agents=self.n_agents,
        )
        self.agents = []
        for _ in range(self.n_agents):
            self.agents.append(
                {
                    "actor": self.shared_actor,
                    "critic": self.value_critic,
                    "value_critic": self.value_critic,
                }
            )
        self.actors = [self.shared_actor for _ in range(self.n_agents)]

    def _init_optimizers(self):
        actor_lr = float(getattr(self.args, "learning_rate_actor", 3e-4))
        critic_lr = float(getattr(self.args, "learning_rate_critic", 5e-4))
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        for agent in self.agents:
            agent["actor_optimizer"] = self.actor_optimizer
            agent["critic_optimizer"] = self.value_optimizer
            agent["actor_lr_schedule"] = actor_lr
            agent["critic_lr_schedule"] = critic_lr
            agent["actor_lr_var"] = getattr(self.actor_optimizer, "learning_rate", None)
            agent["critic_lr_var"] = getattr(self.value_optimizer, "learning_rate", None)

    def _initialize_optimizers(self):
        dummy_obs = tf.zeros((1, self.obs_shapes[0]), dtype=tf.float32)
        actor_inputs = [dummy_obs]
        if self.use_fr_feature_flag:
            actor_inputs.append(tf.zeros((1, 1), dtype=tf.float32))
        if self.use_pf_feature_flag:
            actor_inputs.append(tf.zeros((1, self.pf_feature_dim), dtype=tf.float32))
        if len(actor_inputs) == 1:
            _ = self.shared_actor(actor_inputs[0], training=False)
        else:
            _ = self.shared_actor(actor_inputs, training=False)

        dummy_state = tf.zeros((1, int(sum(self.obs_shapes))), dtype=tf.float32)
        critic_inputs = [dummy_state]
        if self.use_fr_feature_flag:
            critic_inputs.append(tf.zeros((1, 1), dtype=tf.float32))
        if self.use_pf_feature_flag:
            critic_inputs.append(tf.zeros((1, self.pf_feature_dim * self.n_agents), dtype=tf.float32))
        _ = self.value_critic(critic_inputs, training=False)

        actor_vars = list(self.shared_actor.trainable_variables) + [self.actor_log_std]
        value_vars = list(self.value_critic.trainable_variables)
        try:
            if hasattr(self.actor_optimizer, "build"):
                self.actor_optimizer.build(actor_vars)
        except Exception:
            pass
        try:
            if hasattr(self.value_optimizer, "build"):
                self.value_optimizer.build(value_vars)
        except Exception:
            pass
        zero_actor = [tf.zeros_like(v) for v in actor_vars]
        zero_value = [tf.zeros_like(v) for v in value_vars]
        self.actor_optimizer.apply_gradients(zip(zero_actor, actor_vars))
        self.value_optimizer.apply_gradients(zip(zero_value, value_vars))

    def _adaptive_adjustment(self):
        """MAPPO版自适应调整：衰减学习率，并提升PPO探索强度（entropy系数）。"""

        def _safe_set_lr(opt, lr_var):
            base_opt = getattr(opt, "inner_optimizer", getattr(opt, "_optimizer", opt))
            try:
                lr_attr = getattr(base_opt, "learning_rate", None)
                if hasattr(lr_attr, "assign"):
                    lr_attr.assign(lr_var)
                else:
                    try:
                        setattr(base_opt, "learning_rate", lr_var)
                    except Exception:
                        setattr(base_opt, "learning_rate", float(lr_var.numpy()))
            except Exception:
                try:
                    setattr(base_opt, "lr", lr_var)
                except Exception:
                    try:
                        setattr(base_opt, "lr", float(lr_var.numpy()))
                    except Exception:
                        pass

        actor_lr_var = self.agents[0].get("actor_lr_var", None) if self.agents else None
        critic_lr_var = self.agents[0].get("critic_lr_var", None) if self.agents else None
        current_actor_lr = float(actor_lr_var.numpy()) if hasattr(actor_lr_var, "numpy") else float(getattr(self.args, "learning_rate_actor", 3e-4))
        current_critic_lr = float(critic_lr_var.numpy()) if hasattr(critic_lr_var, "numpy") else float(getattr(self.args, "learning_rate_critic", 5e-4))

        new_actor_lr = max(
            float(self.adaptive_learning.get("min_learning_rate", 2e-5)),
            current_actor_lr * float(self.adaptive_learning.get("lr_decay_factor", 0.95)),
        )
        new_critic_lr = max(
            float(self.adaptive_learning.get("min_learning_rate", 2e-5)),
            current_critic_lr * float(self.adaptive_learning.get("lr_decay_factor", 0.95)),
        )

        for agent in self.agents:
            if hasattr(agent.get("actor_lr_var", None), "assign"):
                agent["actor_lr_var"].assign(tf.cast(new_actor_lr, tf.float32))
            if hasattr(agent.get("critic_lr_var", None), "assign"):
                agent["critic_lr_var"].assign(tf.cast(new_critic_lr, tf.float32))
            agent["actor_lr"] = float(new_actor_lr)
            agent["critic_lr"] = float(new_critic_lr)

        _safe_set_lr(self.actor_optimizer, self.agents[0].get("actor_lr_var", tf.constant(new_actor_lr, dtype=tf.float32)))
        _safe_set_lr(self.value_optimizer, self.agents[0].get("critic_lr_var", tf.constant(new_critic_lr, dtype=tf.float32)))

        current_entropy = float(getattr(self, "entropy_coef", getattr(self.args, "entropy_coef", 0.01)))
        boost = float(self.adaptive_learning.get("noise_boost_factor", self.adaptive_learning.get("entropy_boost_factor", 1.2)))
        max_entropy = float(self.adaptive_learning.get("max_entropy_coef", max(current_entropy, self.base_entropy_coef)))
        min_entropy = float(self.adaptive_learning.get("min_entropy_coef", self.base_entropy_coef))
        if current_entropy >= max_entropy * 0.95:
            target_entropy = max(min_entropy, current_entropy * 0.7)
        else:
            target_entropy = min(max_entropy, max(min_entropy, current_entropy * boost))
        try:
            beta = float(os.getenv("ADAPTIVE_NOISE_SMOOTH", "0.3"))
        except Exception:
            beta = 0.3
        beta = max(0.0, min(1.0, beta))
        new_entropy = current_entropy + beta * (target_entropy - current_entropy)
        new_entropy = float(min(max_entropy, max(new_entropy, min_entropy)))
        self.entropy_coef = float(new_entropy)
        self.args.entropy_coef = float(new_entropy)

        print(
            f"MAPPO自适应调整: "
            f"Actor LR {current_actor_lr:.6f}->{new_actor_lr:.6f}, "
            f"Critic LR {current_critic_lr:.6f}->{new_critic_lr:.6f}, "
            f"Entropy {current_entropy:.4f}->{new_entropy:.4f} "
            f"(target={target_entropy:.4f}, beta={beta:.2f})"
        )

    def reset_hidden_states(self, batch_size=None):
        return None

    def _slice_agent_obs(self, states: tf.Tensor, agent_idx: int) -> tf.Tensor:
        obs_dim = self.obs_shapes[agent_idx] if agent_idx < len(self.obs_shapes) else self.obs_shapes[0]
        return states[:, agent_idx, :obs_dim]

    def _build_actor_inputs(
        self,
        agent_obs: tf.Tensor,
        fr_batch: Optional[tf.Tensor] = None,
        pf_batch: Optional[tf.Tensor] = None,
    ):
        inputs: List[tf.Tensor] = [agent_obs]
        if self.use_fr_feature_flag:
            if fr_batch is None:
                fr_batch = tf.zeros((tf.shape(agent_obs)[0], 1), dtype=agent_obs.dtype)
            inputs.append(fr_batch)
        if self.use_pf_feature_flag:
            if pf_batch is None:
                pf_batch = tf.zeros((tf.shape(agent_obs)[0], self.pf_feature_dim), dtype=agent_obs.dtype)
            inputs.append(pf_batch)
        return inputs[0] if len(inputs) == 1 else inputs

    def _build_value_inputs(
        self,
        global_state: tf.Tensor,
        fr_batch: Optional[tf.Tensor] = None,
        global_pf_batch: Optional[tf.Tensor] = None,
    ):
        inputs: List[tf.Tensor] = [global_state]
        if self.use_fr_feature_flag:
            if fr_batch is None:
                fr_batch = tf.zeros((tf.shape(global_state)[0], 1), dtype=global_state.dtype)
            inputs.append(fr_batch)
        if self.use_pf_feature_flag:
            if global_pf_batch is None:
                global_pf_batch = tf.zeros((tf.shape(global_state)[0], self.pf_feature_dim * self.n_agents), dtype=global_state.dtype)
            inputs.append(global_pf_batch)
        return inputs[0] if len(inputs) == 1 else inputs

    def _compute_global_state(self, states: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(states)[0]
        if len(set(self.obs_shapes)) <= 1:
            return tf.reshape(states[:, :, : self.obs_shapes[0]], [batch_size, self.n_agents * self.obs_shapes[0]])
        all_obs = [states[:, i, : self.obs_shapes[i]] for i in range(self.n_agents)]
        return tf.concat(all_obs, axis=1)

    def build_global_state_numpy(self, states) -> np.ndarray:
        states_np = np.asarray(states, dtype=np.float32)
        if states_np.ndim != 3:
            raise ValueError(f"states should be [num_envs, n_agents, obs_dim], got {states_np.shape}")
        if len(set(self.obs_shapes)) <= 1:
            return states_np[:, :, : self.obs_shapes[0]].reshape(states_np.shape[0], self.n_agents * self.obs_shapes[0])
        parts = [states_np[:, i, : self.obs_shapes[i]] for i in range(self.n_agents)]
        return np.concatenate(parts, axis=1).astype(np.float32, copy=False)

    def _compute_base_pf_features_tf(self, states: tf.Tensor) -> Tuple[tf.Tensor, Optional[Tuple[tf.Tensor, ...]], Optional[tf.Tensor]]:
        num_envs = tf.shape(states)[0]
        obs_dim = self.obs_shapes[0] if len(self.obs_shapes) > 0 else tf.shape(states)[2]
        if not (self.use_pf_feature_flag and self.use_tf_potential_field_cached and float(self.action_force_ratio_cached) > 0.0):
            zero = tf.zeros((num_envs, self.n_agents, max(3, self.pf_feature_dim)), dtype=tf.float32)
            return zero[:, :, : max(1, self.pf_feature_dim or 3)], None, None

        flat_obs = tf.reshape(states[:, :, :obs_dim], [-1, obs_dim])
        pf_obs_context = self._extract_pf_obs_context_compiled_tf(flat_obs)
        geometry_context = self._extract_pf_geometry_context_compiled_tf(
            pf_obs_context[1],
            pf_obs_context[2],
            pf_obs_context[4],
            pf_obs_context[5],
            pf_obs_context[6],
        )
        obs_valid_inputs = pf_obs_context[3]
        dummy_actions = tf.zeros((num_envs * self.n_agents, self.action_dims[0]), dtype=tf.float32)
        _, pf_force_flat = self._compute_pf_force_from_geometry_context_compiled_tf(
            dummy_actions,
            obs_valid_inputs,
            geometry_context,
        )
        pf_forces = tf.reshape(pf_force_flat, [num_envs, self.n_agents, 3])
        if self.pf_feature_dim > 3:
            pad = tf.zeros((num_envs, self.n_agents, self.pf_feature_dim - 3), dtype=pf_forces.dtype)
            pf_forces = tf.concat([pf_forces, pad], axis=-1)
        elif self.pf_feature_dim > 0:
            pf_forces = pf_forces[:, :, : self.pf_feature_dim]
        return pf_forces, geometry_context, obs_valid_inputs

    def _compute_shared_actor_outputs_batched(
        self,
        states: tf.Tensor,
        fr_batch: Optional[tf.Tensor],
        pf_features: tf.Tensor,
        add_noise: bool,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        num_envs = tf.shape(states)[0]
        actor_obs = tf.reshape(states[:, :, : self._shared_obs_dim], [num_envs * self.n_agents, self._shared_obs_dim])
        actor_fr = None
        if self.use_fr_feature_flag and fr_batch is not None:
            actor_fr = tf.repeat(fr_batch, repeats=self.n_agents, axis=0)
        actor_pf = None
        if self.use_pf_feature_flag and self.pf_feature_dim > 0:
            actor_pf = tf.reshape(pf_features[:, :, : self.pf_feature_dim], [num_envs * self.n_agents, self.pf_feature_dim])
        actor_inputs = self._build_actor_inputs(actor_obs, fr_batch=actor_fr, pf_batch=actor_pf)
        mean_action_flat = self.shared_actor(actor_inputs, training=False)
        if add_noise:
            sampled_action_flat, log_prob_flat, entropy_flat = squashed_gaussian_sample(mean_action_flat, self.actor_log_std)
        else:
            sampled_action_flat = mean_action_flat
            log_prob_flat, entropy_flat = squashed_gaussian_log_prob(mean_action_flat, self.actor_log_std, sampled_action_flat)
        raw_actor_means = tf.reshape(mean_action_flat, [num_envs, self.n_agents, self.action_dims[0]])
        sampled_actions = tf.reshape(sampled_action_flat, [num_envs, self.n_agents, self.action_dims[0]])
        log_probs = tf.reshape(log_prob_flat, [num_envs, self.n_agents])
        entropies = tf.reshape(entropy_flat, [num_envs, self.n_agents])
        return raw_actor_means, sampled_actions, log_probs, entropies

    def _sample_shared_actor_outputs_batched(
        self,
        states: tf.Tensor,
        fr_batch: Optional[tf.Tensor],
        pf_features: tf.Tensor,
        add_noise: bool,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        num_envs = tf.shape(states)[0]
        actor_obs = tf.reshape(states[:, :, : self._shared_obs_dim], [num_envs * self.n_agents, self._shared_obs_dim])
        actor_fr = None
        if self.use_fr_feature_flag and fr_batch is not None:
            actor_fr = tf.repeat(fr_batch, repeats=self.n_agents, axis=0)
        actor_pf = None
        if self.use_pf_feature_flag and self.pf_feature_dim > 0:
            actor_pf = tf.reshape(pf_features[:, :, : self.pf_feature_dim], [num_envs * self.n_agents, self.pf_feature_dim])
        actor_inputs = self._build_actor_inputs(actor_obs, fr_batch=actor_fr, pf_batch=actor_pf)
        mean_action_flat = self.shared_actor(actor_inputs, training=False)
        if add_noise:
            sampled_action_flat = tf.tanh(
                tf.random.normal(tf.shape(mean_action_flat), dtype=mean_action_flat.dtype) * tf.exp(tf.reshape(tf.clip_by_value(self.actor_log_std, -5.0, 1.0), [1, -1]))
                + tf.atanh(tf.clip_by_value(mean_action_flat, -0.999999, 0.999999))
            )
        else:
            sampled_action_flat = mean_action_flat
        raw_actor_means = tf.reshape(mean_action_flat, [num_envs, self.n_agents, self.action_dims[0]])
        sampled_actions = tf.reshape(sampled_action_flat, [num_envs, self.n_agents, self.action_dims[0]])
        return raw_actor_means, sampled_actions

    def _compute_shared_actor_outputs_loop(
        self,
        states: tf.Tensor,
        fr_batch: Optional[tf.Tensor],
        pf_features: tf.Tensor,
        add_noise: bool,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        raw_actor_means = []
        sampled_actions = []
        log_probs = []
        entropies = []
        for i in range(self.n_agents):
            agent_obs = self._slice_agent_obs(states, i)
            pf_batch = pf_features[:, i, :] if self.use_pf_feature_flag else None
            actor_inputs = self._build_actor_inputs(agent_obs, fr_batch=fr_batch, pf_batch=pf_batch)
            mean_action = self.shared_actor(actor_inputs, training=False)
            if add_noise:
                sampled_action, log_prob, entropy = squashed_gaussian_sample(mean_action, self.actor_log_std)
            else:
                sampled_action = mean_action
                log_prob, entropy = squashed_gaussian_log_prob(mean_action, self.actor_log_std, sampled_action)
            raw_actor_means.append(mean_action)
            sampled_actions.append(sampled_action)
            log_probs.append(log_prob)
            entropies.append(entropy)
        return (
            tf.stack(raw_actor_means, axis=1),
            tf.stack(sampled_actions, axis=1),
            tf.stack(log_probs, axis=1),
            tf.stack(entropies, axis=1),
        )

    def _sample_shared_actor_outputs_loop(
        self,
        states: tf.Tensor,
        fr_batch: Optional[tf.Tensor],
        pf_features: tf.Tensor,
        add_noise: bool,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        raw_actor_means = []
        sampled_actions = []
        for i in range(self.n_agents):
            agent_obs = self._slice_agent_obs(states, i)
            pf_batch = pf_features[:, i, :] if self.use_pf_feature_flag else None
            actor_inputs = self._build_actor_inputs(agent_obs, fr_batch=fr_batch, pf_batch=pf_batch)
            mean_action = self.shared_actor(actor_inputs, training=False)
            if add_noise:
                sampled_action = squashed_gaussian_sample(mean_action, self.actor_log_std)[0]
            else:
                sampled_action = mean_action
            raw_actor_means.append(mean_action)
            sampled_actions.append(sampled_action)
        return tf.stack(raw_actor_means, axis=1), tf.stack(sampled_actions, axis=1)

    def _cache_last_policy_outputs(
        self,
        raw_actor_outputs: tf.Tensor,
        pf_features_current: tf.Tensor,
        log_probs: tf.Tensor,
        values: tf.Tensor,
        entropies: tf.Tensor,
    ) -> None:
        self._last_action_log_probs = log_probs
        self._last_values = values
        self._last_entropies = entropies
        self._last_pf_features = pf_features_current
        self._last_actor_means = raw_actor_outputs

    def _sync_rollout_step_outputs(
        self,
        actions_for_storage: tf.Tensor,
        actions_for_execution: tf.Tensor,
        pf_features: Optional[tf.Tensor],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        num_envs = int(getattr(self.args, "num_envs", 1) or 1)
        action_dim = int(self.action_dims[0])
        pf_dim = int(self.pf_feature_dim) if self.use_pf_feature_flag else 0
        size_storage = num_envs * self.n_agents * action_dim
        size_execution = size_storage
        size_pf = num_envs * self.n_agents * pf_dim
        concat_parts = [
            tf.cast(tf.reshape(actions_for_storage, [-1]), tf.float32),
            tf.cast(tf.reshape(actions_for_execution, [-1]), tf.float32),
        ]
        if pf_dim > 0 and pf_features is not None:
            concat_parts.append(tf.cast(tf.reshape(pf_features[:, :, :pf_dim], [-1]), tf.float32))
        flat = np.ascontiguousarray(_safe_tensor_to_numpy(tf.concat(concat_parts, axis=0)), dtype=np.float32)
        cursor = 0
        actions_storage_np = flat[cursor:cursor + size_storage].reshape(num_envs, self.n_agents, action_dim)
        cursor += size_storage
        actions_exec_np = flat[cursor:cursor + size_execution].reshape(num_envs, self.n_agents, action_dim)
        cursor += size_execution
        pf_features_np = None
        if pf_dim > 0 and pf_features is not None:
            pf_features_np = flat[cursor:cursor + size_pf].reshape(num_envs, self.n_agents, pf_dim)
        return actions_storage_np, actions_exec_np, pf_features_np

    @tf.function(reduce_retracing=True)
    def _compute_rollout_policy_outputs(
        self,
        states: tf.Tensor,
        add_noise: bool,
    ):
        if not tf.is_tensor(states):
            states = tf.convert_to_tensor(states, dtype=tf.float32)
        elif states.dtype != tf.float32:
            states = tf.cast(states, tf.float32)

        num_envs = tf.shape(states)[0]
        current_force_ratio = tf.cast(self.action_force_ratio_cached, tf.float32)
        fr_batch = tf.fill((num_envs, 1), current_force_ratio) if self.use_fr_feature_flag else None
        pf_features, geometry_context, obs_valid_inputs = self._compute_base_pf_features_tf(states)
        pf_features = tf.cast(pf_features, tf.float32)
        if self._all_obs_same_shape:
            raw_actor_means, sampled_actions = self._sample_shared_actor_outputs_batched(
                states,
                fr_batch,
                pf_features,
                add_noise,
            )
        else:
            raw_actor_means, sampled_actions = self._sample_shared_actor_outputs_loop(
                states,
                fr_batch,
                pf_features,
                add_noise,
            )

        use_tf_potential_field = self.use_tf_potential_field_cached
        should_apply_pf = bool(use_tf_potential_field and geometry_context is not None and obs_valid_inputs is not None)
        if should_apply_pf and (float(self.action_force_ratio_cached) > 0.0):
            flat_actions = tf.reshape(sampled_actions, [-1, self.action_dims[0]])
            corrected_head_flat, pf_force_flat = self._apply_potential_field_correction_from_geometry_context_tf(
                flat_actions,
                current_force_ratio,
                obs_valid_inputs,
                geometry_context,
            )
            corrected_head = tf.reshape(corrected_head_flat, [num_envs, self.n_agents, 3])
            pf_force_network = tf.reshape(pf_force_flat, [num_envs, self.n_agents, 3])
            corrected_actions = tf.concat([corrected_head, sampled_actions[:, :, 3:]], axis=-1)
        else:
            corrected_actions = sampled_actions
            pf_force_network = tf.zeros((num_envs, self.n_agents, 3), dtype=tf.float32)

        return (
            sampled_actions,
            corrected_actions,
            pf_force_network,
            raw_actor_means,
            pf_features,
        )

    @tf.function(reduce_retracing=True)
    def _compute_policy_outputs(
        self,
        states: tf.Tensor,
        add_noise: bool,
    ):
        if not tf.is_tensor(states):
            states = tf.convert_to_tensor(states, dtype=tf.float32)
        elif states.dtype != tf.float32:
            states = tf.cast(states, tf.float32)

        num_envs = tf.shape(states)[0]
        current_force_ratio = tf.cast(self.action_force_ratio_cached, tf.float32)
        fr_batch = tf.fill((num_envs, 1), current_force_ratio) if self.use_fr_feature_flag else None
        pf_features, geometry_context, obs_valid_inputs = self._compute_base_pf_features_tf(states)
        pf_features = tf.cast(pf_features, tf.float32)
        if self._all_obs_same_shape:
            raw_actor_means, sampled_actions, log_probs, entropies = self._compute_shared_actor_outputs_batched(
                states,
                fr_batch,
                pf_features,
                add_noise,
            )
        else:
            raw_actor_means, sampled_actions, log_probs, entropies = self._compute_shared_actor_outputs_loop(
                states,
                fr_batch,
                pf_features,
                add_noise,
            )

        global_state = self._compute_global_state(states)
        global_pf = None
        if self.use_pf_feature_flag:
            global_pf = tf.reshape(pf_features, [num_envs, self.n_agents * self.pf_feature_dim])
        value_inputs = self._build_value_inputs(global_state, fr_batch=fr_batch, global_pf_batch=global_pf)
        values = tf.squeeze(self.value_critic(value_inputs, training=False), axis=-1)

        use_tf_potential_field = self.use_tf_potential_field_cached
        should_apply_pf = bool(use_tf_potential_field and geometry_context is not None and obs_valid_inputs is not None)
        if should_apply_pf and (float(self.action_force_ratio_cached) > 0.0):
            flat_actions = tf.reshape(sampled_actions, [-1, self.action_dims[0]])
            corrected_head_flat, pf_force_flat = self._apply_potential_field_correction_from_geometry_context_tf(
                flat_actions,
                current_force_ratio,
                obs_valid_inputs,
                geometry_context,
            )
            corrected_head = tf.reshape(corrected_head_flat, [num_envs, self.n_agents, 3])
            pf_force_network = tf.reshape(pf_force_flat, [num_envs, self.n_agents, 3])
            corrected_actions = tf.concat([corrected_head, sampled_actions[:, :, 3:]], axis=-1)
        else:
            corrected_actions = sampled_actions
            pf_force_network = tf.zeros((num_envs, self.n_agents, 3), dtype=tf.float32)

        return (
            sampled_actions,
            corrected_actions,
            pf_force_network,
            raw_actor_means,
            pf_features,
            log_probs,
            values,
            entropies,
        )

    @tf.function(reduce_retracing=True)
    def _predict_values_from_processed_states_tf(
        self,
        states: tf.Tensor,
        fr_batch: Optional[tf.Tensor] = None,
        pf_features: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        if states.dtype != tf.float32:
            states = tf.cast(states, tf.float32)
        num_envs = tf.shape(states)[0]
        global_state = self._compute_global_state(states)
        global_pf_batch = None
        if self.use_pf_feature_flag and pf_features is not None:
            global_pf_batch = tf.reshape(pf_features[:, :, : self.pf_feature_dim], [num_envs, self.n_agents * self.pf_feature_dim])
        critic_inputs = self._build_value_inputs(global_state, fr_batch=fr_batch, global_pf_batch=global_pf_batch)
        return tf.squeeze(self.value_critic(critic_inputs, training=False), axis=-1)

    @tf.function(reduce_retracing=True)
    def _predict_log_probs_from_rollout_obs_tf(
        self,
        actor_obs: tf.Tensor,
        actor_actions: tf.Tensor,
        actor_fr: Optional[tf.Tensor] = None,
        actor_pf: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        if actor_obs.dtype != tf.float32:
            actor_obs = tf.cast(actor_obs, tf.float32)
        if actor_actions.dtype != tf.float32:
            actor_actions = tf.cast(actor_actions, tf.float32)
        actor_inputs = self._prepare_actor_minibatch_inputs(actor_obs, actor_fr, actor_pf)
        current_mean_action = self.shared_actor(actor_inputs, training=False)
        log_probs, _ = squashed_gaussian_log_prob(current_mean_action, self.actor_log_std, actor_actions)
        return log_probs

    @tf.function(reduce_retracing=True)
    def _predict_log_prob_splits_from_rollout_obs_tf(
        self,
        actor_obs: tf.Tensor,
        actor_actions: tf.Tensor,
        actor_fr: Optional[tf.Tensor] = None,
        actor_pf: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        if actor_obs.dtype != tf.float32:
            actor_obs = tf.cast(actor_obs, tf.float32)
        if actor_actions.dtype != tf.float32:
            actor_actions = tf.cast(actor_actions, tf.float32)
        actor_inputs = self._prepare_actor_minibatch_inputs(actor_obs, actor_fr, actor_pf)
        current_mean_action = self.shared_actor(actor_inputs, training=False)
        log_prob_components, _ = squashed_gaussian_log_prob_components(
            current_mean_action,
            self.actor_log_std,
            actor_actions,
        )
        log_probs_head = tf.reduce_sum(log_prob_components[:, : self.head_action_dim], axis=-1)
        log_probs_tail = tf.reduce_sum(log_prob_components[:, self.head_action_dim :], axis=-1)
        return log_probs_head + log_probs_tail, log_probs_head, log_probs_tail

    @tf.function(reduce_retracing=True)
    def _predict_values_from_global_state_tf(
        self,
        global_state: tf.Tensor,
        fr_batch: Optional[tf.Tensor] = None,
        global_pf_batch: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        if global_state.dtype != tf.float32:
            global_state = tf.cast(global_state, tf.float32)
        critic_inputs = self._build_value_inputs(global_state, fr_batch=fr_batch, global_pf_batch=global_pf_batch)
        return tf.squeeze(self.value_critic(critic_inputs, training=False), axis=-1)

    def predict_values_vectorized(self, states, fr_values=None, pf_features=None):
        states_np = np.asarray(states, dtype=np.float32)
        states_tf = tf.convert_to_tensor(states_np, dtype=tf.float32)
        fr_batch = None
        if self.use_fr_feature_flag:
            if fr_values is None:
                fr_values = np.full((states_np.shape[0], 1), float(self.action_force_ratio_var.numpy()), dtype=np.float32)
            fr_batch = tf.convert_to_tensor(np.asarray(fr_values, dtype=np.float32).reshape(states_np.shape[0], 1), dtype=tf.float32)
        pf_features_tf = None
        if self.use_pf_feature_flag:
            if pf_features is None:
                if float(self.action_force_ratio_var.numpy()) > 0.0 and self.use_tf_potential_field_cached:
                    pf_features = self.compute_base_pf_forces_batch_numpy(states_np, float(self.action_force_ratio_var.numpy()))
                else:
                    pf_features = np.zeros((states_np.shape[0], self.n_agents, self.pf_feature_dim), dtype=np.float32)
            pf_np = np.asarray(pf_features, dtype=np.float32)
            if pf_np.ndim == 3:
                pf_features_tf = tf.convert_to_tensor(pf_np, dtype=tf.float32)
        values = self._predict_values_from_processed_states_tf(states_tf, fr_batch=fr_batch, pf_features=pf_features_tf)
        return np.asarray(_safe_tensor_to_numpy(values), dtype=np.float32)

    def predict_rollout_values_vectorized(self, global_states, fr_values=None, pf_features=None) -> np.ndarray:
        global_states_np = np.asarray(global_states, dtype=np.float32)
        flat_states = global_states_np.reshape(-1, global_states_np.shape[-1])
        global_state_tf = tf.convert_to_tensor(flat_states, dtype=tf.float32)
        fr_batch = None
        if self.use_fr_feature_flag and fr_values is not None:
            fr_np = np.asarray(fr_values, dtype=np.float32).reshape(-1, 1)
            fr_batch = tf.convert_to_tensor(fr_np, dtype=tf.float32)
        global_pf_batch = None
        if self.use_pf_feature_flag and pf_features is not None:
            pf_np = np.asarray(pf_features, dtype=np.float32)
            global_pf_batch = tf.convert_to_tensor(
                pf_np.reshape(pf_np.shape[0] * pf_np.shape[1], self.n_agents * self.pf_feature_dim),
                dtype=tf.float32,
            )
        values = self._predict_values_from_global_state_tf(global_state_tf, fr_batch=fr_batch, global_pf_batch=global_pf_batch)
        return np.asarray(_safe_tensor_to_numpy(values), dtype=np.float32).reshape(global_states_np.shape[:-1])

    def predict_rollout_log_probs_vectorized(self, obs, actions, fr_values=None, pf_features=None) -> np.ndarray:
        total_log_probs, _, _ = self.predict_rollout_log_prob_splits_vectorized(
            obs,
            actions,
            fr_values=fr_values,
            pf_features=pf_features,
        )
        return total_log_probs

    def predict_rollout_log_prob_splits_vectorized(self, obs, actions, fr_values=None, pf_features=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_np = np.asarray(obs, dtype=np.float32)
        actions_np = np.asarray(actions, dtype=np.float32)
        flat_obs = obs_np.reshape(-1, obs_np.shape[-1])
        flat_actions = actions_np.reshape(-1, actions_np.shape[-1])
        actor_obs_tf = tf.convert_to_tensor(flat_obs, dtype=tf.float32)
        actor_actions_tf = tf.convert_to_tensor(flat_actions, dtype=tf.float32)
        actor_fr_tf = None
        if self.use_fr_feature_flag and fr_values is not None:
            fr_np = np.asarray(fr_values, dtype=np.float32).reshape(-1, 1)
            actor_fr_tf = tf.convert_to_tensor(np.repeat(fr_np, self.n_agents, axis=0), dtype=tf.float32)
        actor_pf_tf = None
        if self.use_pf_feature_flag and pf_features is not None:
            pf_np = np.asarray(pf_features, dtype=np.float32)
            actor_pf_tf = tf.convert_to_tensor(pf_np.reshape(-1, self.pf_feature_dim), dtype=tf.float32)
        log_probs, log_probs_head, log_probs_tail = self._predict_log_prob_splits_from_rollout_obs_tf(
            actor_obs_tf,
            actor_actions_tf,
            actor_fr=actor_fr_tf,
            actor_pf=actor_pf_tf,
        )
        target_shape = obs_np.shape[:-1]
        return (
            np.asarray(_safe_tensor_to_numpy(log_probs), dtype=np.float32).reshape(target_shape),
            np.asarray(_safe_tensor_to_numpy(log_probs_head), dtype=np.float32).reshape(target_shape),
            np.asarray(_safe_tensor_to_numpy(log_probs_tail), dtype=np.float32).reshape(target_shape),
        )

    def batch_select_actions_vectorized(self, states, add_noise=True):
        (
            actions_for_storage,
            actions_for_execution,
            pf_forces,
            raw_actor_outputs,
            pf_features_current,
            log_probs,
            values,
            entropies,
        ) = self._compute_policy_outputs(states, add_noise=bool(add_noise))
        self._cache_last_policy_outputs(raw_actor_outputs, pf_features_current, log_probs, values, entropies)
        return actions_for_storage, actions_for_execution, pf_forces, raw_actor_outputs, pf_features_current

    def collect_rollout_step_vectorized(self, states, add_noise=True):
        (
            actions_for_storage,
            actions_for_execution,
            _pf_forces,
            raw_actor_outputs,
            pf_features_current,
        ) = self._compute_rollout_policy_outputs(states, add_noise=bool(add_noise))
        self._last_action_log_probs = None
        self._last_values = None
        self._last_entropies = None
        self._last_pf_features = pf_features_current
        self._last_actor_means = raw_actor_outputs
        return self._sync_rollout_step_outputs(
            actions_for_storage,
            actions_for_execution,
            pf_features_current if self.use_pf_feature_flag else None,
        )

    def _prepare_actor_minibatch_inputs(
        self,
        obs_mb: tf.Tensor,
        fr_mb: Optional[tf.Tensor],
        pf_mb: Optional[tf.Tensor],
    ):
        actor_inputs: List[tf.Tensor] = [obs_mb]
        if self.use_fr_feature_flag:
            actor_inputs.append(fr_mb)
        if self.use_pf_feature_flag:
            actor_inputs.append(pf_mb)
        return actor_inputs[0] if len(actor_inputs) == 1 else actor_inputs

    def _compute_separated_loss_weights(
        self,
        actor_fr_mb: Optional[tf.Tensor],
        batch_size: tf.Tensor,
        dtype: tf.dtypes.DType,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        if actor_fr_mb is None:
            fr_batch = tf.fill([batch_size, 1], tf.cast(self.action_force_ratio_cached, dtype))
        else:
            fr_batch = tf.cast(tf.reshape(actor_fr_mb, [-1, 1]), dtype)
        fr_batch = tf.clip_by_value(fr_batch, 0.0, 1.0)
        pf_params_fixed = tf.cast(tf.reduce_mean(tf.cast(self.c_pf_params_fixed, dtype)), dtype)
        traditional_apf_adjustment = pf_params_fixed * tf.cast(0.4, dtype)
        head_weight_base = (tf.cast(1.0, dtype) - fr_batch) + traditional_apf_adjustment
        tail_weight_base = fr_batch
        min_weight = tf.cast(0.1, dtype)
        head_weight = tf.maximum(head_weight_base, min_weight)
        tail_weight = tf.maximum(tail_weight_base, min_weight)
        weight_sum = tf.maximum(head_weight + tail_weight, tf.cast(1e-6, dtype))
        head_weight = tf.squeeze(head_weight / weight_sum, axis=-1)
        tail_weight = tf.squeeze(tail_weight / weight_sum, axis=-1)
        return head_weight, tail_weight

    @tf.function(reduce_retracing=True)
    def _ppo_minibatch_step(
        self,
        actor_obs_mb,
        actor_actions_mb,
        actor_old_log_probs_mb,
        actor_old_log_probs_head_mb,
        actor_old_log_probs_tail_mb,
        actor_advantages_mb,
        critic_states_mb,
        critic_returns_mb,
        actor_fr_mb,
        critic_fr_mb,
        actor_pf_mb,
        critic_pf_mb,
    ):
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            actor_inputs = self._prepare_actor_minibatch_inputs(actor_obs_mb, actor_fr_mb, actor_pf_mb)
            current_mean_action = self.shared_actor(actor_inputs, training=True)
            if self.use_separated_gradient:
                log_prob_components, entropy_components = squashed_gaussian_log_prob_components(
                    current_mean_action,
                    self.actor_log_std,
                    actor_actions_mb,
                )
                new_log_probs_head = tf.reduce_sum(log_prob_components[:, : self.head_action_dim], axis=-1)
                new_log_probs_tail = tf.reduce_sum(log_prob_components[:, self.head_action_dim :], axis=-1)
                new_log_probs = new_log_probs_head + new_log_probs_tail
                old_log_probs_head = tf.cast(actor_old_log_probs_head_mb, tf.float32)
                old_log_probs_tail = tf.cast(actor_old_log_probs_tail_mb, tf.float32)
                ratio_head = tf.exp(new_log_probs_head - old_log_probs_head)
                ratio_tail = tf.exp(new_log_probs_tail - old_log_probs_tail)
                clipped_ratio_head = tf.clip_by_value(ratio_head, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                clipped_ratio_tail = tf.clip_by_value(ratio_tail, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                surrogate_head = tf.minimum(ratio_head * actor_advantages_mb, clipped_ratio_head * actor_advantages_mb)
                surrogate_tail = tf.minimum(ratio_tail * actor_advantages_mb, clipped_ratio_tail * actor_advantages_mb)
                head_weight, tail_weight = self._compute_separated_loss_weights(
                    actor_fr_mb,
                    tf.shape(actor_obs_mb)[0],
                    actor_obs_mb.dtype,
                )
                policy_loss_head = -tf.reduce_mean(surrogate_head)
                policy_loss_tail = -tf.reduce_mean(surrogate_tail)
                policy_loss = -tf.reduce_mean(head_weight * surrogate_head + tail_weight * surrogate_tail)
                entropy = tf.reduce_sum(entropy_components, axis=-1)
                approx_kl_base = (old_log_probs_head + old_log_probs_tail) - new_log_probs
                clipfrac = tf.reduce_mean(
                    tf.cast(
                        tf.logical_or(
                            tf.abs(ratio_head - 1.0) > self.clip_ratio,
                            tf.abs(ratio_tail - 1.0) > self.clip_ratio,
                        ),
                        tf.float32,
                    )
                )
                mean_head_weight = tf.reduce_mean(head_weight)
                mean_tail_weight = tf.reduce_mean(tail_weight)
            else:
                new_log_probs, entropy = squashed_gaussian_log_prob(current_mean_action, self.actor_log_std, actor_actions_mb)
                ratio = tf.exp(new_log_probs - actor_old_log_probs_mb)
                clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                surrogate_1 = ratio * actor_advantages_mb
                surrogate_2 = clipped_ratio * actor_advantages_mb
                policy_loss = -tf.reduce_mean(tf.minimum(surrogate_1, surrogate_2))
                policy_loss_head = tf.constant(0.0, dtype=tf.float32)
                policy_loss_tail = tf.constant(0.0, dtype=tf.float32)
                approx_kl_base = actor_old_log_probs_mb - new_log_probs
                clipfrac = tf.reduce_mean(tf.cast(tf.abs(ratio - 1.0) > self.clip_ratio, tf.float32))
                mean_head_weight = tf.constant(0.0, dtype=tf.float32)
                mean_tail_weight = tf.constant(0.0, dtype=tf.float32)
            entropy_bonus = tf.reduce_mean(entropy)
            actor_loss = policy_loss - self.entropy_coef * entropy_bonus

            critic_inputs = self._build_value_inputs(critic_states_mb, fr_batch=critic_fr_mb, global_pf_batch=critic_pf_mb)
            values = tf.squeeze(self.value_critic(critic_inputs, training=True), axis=-1)
            value_loss = tf.reduce_mean(tf.square(critic_returns_mb - values))
            critic_loss = self.value_coef * value_loss

        actor_vars = list(self.shared_actor.trainable_variables) + [self.actor_log_std]
        actor_grads = actor_tape.gradient(actor_loss, actor_vars)
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, self.max_grad_norm)
        self.actor_optimizer.apply_gradients(zip(actor_grads, actor_vars))

        critic_vars = list(self.value_critic.trainable_variables)
        critic_grads = critic_tape.gradient(critic_loss, critic_vars)
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, self.max_grad_norm)
        self.value_optimizer.apply_gradients(zip(critic_grads, critic_vars))

        approx_kl = tf.reduce_mean(approx_kl_base)
        return (
            actor_loss,
            critic_loss,
            entropy_bonus,
            approx_kl,
            clipfrac,
            policy_loss,
            value_loss,
            policy_loss_head,
            policy_loss_tail,
            mean_head_weight,
            mean_tail_weight,
        )

    def update(self, rollout_buffer, batch_size=None):
        size = rollout_buffer.size()
        if size <= 0:
            return {}

        data = rollout_buffer.env_step_view()
        obs = data["obs"].reshape(size * rollout_buffer.num_envs, self.n_agents, rollout_buffer.obs_dim)
        actions = data["actions"].reshape(size * rollout_buffer.num_envs, self.n_agents, rollout_buffer.action_dim)
        old_log_probs = data["log_probs"].reshape(size * rollout_buffer.num_envs, self.n_agents)
        old_log_probs_head = data["log_probs_head"].reshape(size * rollout_buffer.num_envs, self.n_agents)
        old_log_probs_tail = data["log_probs_tail"].reshape(size * rollout_buffer.num_envs, self.n_agents)
        global_states = data["global_state"].reshape(size * rollout_buffer.num_envs, rollout_buffer.global_state_dim)
        returns = data["returns"].reshape(size * rollout_buffer.num_envs)
        advantages = data["advantages"].reshape(size * rollout_buffer.num_envs)
        fr_values = data["fr_values"].reshape(size * rollout_buffer.num_envs)
        pf_features = data["pf_features"]
        if pf_features is not None:
            pf_features = pf_features.reshape(size * rollout_buffer.num_envs, self.n_agents, self.pf_feature_dim)
            pf_global = pf_features.reshape(size * rollout_buffer.num_envs, self.n_agents * self.pf_feature_dim)
        else:
            pf_global = None

        adv_mean = float(np.mean(advantages))
        adv_std = float(np.std(advantages) + 1e-8)
        advantages = (advantages - adv_mean) / adv_std
        actor_advantages = repeated_advantages(advantages, self.n_agents)
        actor_obs = obs.reshape(size * rollout_buffer.num_envs * self.n_agents, rollout_buffer.obs_dim)
        actor_actions = actions.reshape(size * rollout_buffer.num_envs * self.n_agents, rollout_buffer.action_dim)
        actor_old_log_probs = old_log_probs.reshape(size * rollout_buffer.num_envs * self.n_agents)
        actor_old_log_probs_head = old_log_probs_head.reshape(size * rollout_buffer.num_envs * self.n_agents)
        actor_old_log_probs_tail = old_log_probs_tail.reshape(size * rollout_buffer.num_envs * self.n_agents)

        if self.use_fr_feature_flag:
            critic_fr = fr_values.reshape(-1, 1).astype(np.float32)
            actor_fr = np.repeat(critic_fr, self.n_agents, axis=0).astype(np.float32)
        else:
            critic_fr = None
            actor_fr = None

        if self.use_pf_feature_flag and pf_features is not None:
            actor_pf = pf_features.reshape(size * rollout_buffer.num_envs * self.n_agents, self.pf_feature_dim)
            critic_pf = pf_global.astype(np.float32)
        else:
            actor_pf = None
            critic_pf = None

        actor_obs_tf = tf.convert_to_tensor(actor_obs, dtype=tf.float32)
        actor_actions_tf = tf.convert_to_tensor(actor_actions, dtype=tf.float32)
        actor_old_log_probs_tf = tf.convert_to_tensor(actor_old_log_probs, dtype=tf.float32)
        actor_old_log_probs_head_tf = tf.convert_to_tensor(actor_old_log_probs_head, dtype=tf.float32)
        actor_old_log_probs_tail_tf = tf.convert_to_tensor(actor_old_log_probs_tail, dtype=tf.float32)
        actor_advantages_tf = tf.convert_to_tensor(actor_advantages, dtype=tf.float32)
        critic_states_tf = tf.convert_to_tensor(global_states, dtype=tf.float32)
        critic_returns_tf = tf.convert_to_tensor(returns, dtype=tf.float32)
        actor_fr_tf = tf.convert_to_tensor(actor_fr, dtype=tf.float32) if actor_fr is not None else None
        critic_fr_tf = tf.convert_to_tensor(critic_fr, dtype=tf.float32) if critic_fr is not None else None
        actor_pf_tf = tf.convert_to_tensor(actor_pf, dtype=tf.float32) if actor_pf is not None else None
        critic_pf_tf = tf.convert_to_tensor(critic_pf, dtype=tf.float32) if critic_pf is not None else None

        env_step_total = size * rollout_buffer.num_envs
        losses_actor_tf = []
        losses_critic_tf = []
        entropies_tf = []
        kls_tf = []
        clipfracs_tf = []
        policy_losses_tf = []
        value_losses_tf = []
        head_policy_losses_tf = []
        tail_policy_losses_tf = []
        head_weights_tf = []
        tail_weights_tf = []

        for _ in range(self.ppo_epochs):
            epoch_kls_tf = []
            for env_indices in rollout_buffer.iterate_env_minibatches(self.mini_batch_size, shuffle=True):
                env_indices = np.asarray(env_indices, dtype=np.int32)
                actor_indices = (env_indices[:, None] * self.n_agents + self._agent_index_offsets[None, :]).reshape(-1)
                actor_obs_mb = tf.gather(actor_obs_tf, actor_indices)
                actor_actions_mb = tf.gather(actor_actions_tf, actor_indices)
                actor_old_log_probs_mb = tf.gather(actor_old_log_probs_tf, actor_indices)
                actor_old_log_probs_head_mb = tf.gather(actor_old_log_probs_head_tf, actor_indices)
                actor_old_log_probs_tail_mb = tf.gather(actor_old_log_probs_tail_tf, actor_indices)
                actor_advantages_mb = tf.gather(actor_advantages_tf, actor_indices)
                critic_states_mb = tf.gather(critic_states_tf, env_indices)
                critic_returns_mb = tf.gather(critic_returns_tf, env_indices)
                actor_fr_mb = tf.gather(actor_fr_tf, actor_indices) if actor_fr_tf is not None else None
                critic_fr_mb = tf.gather(critic_fr_tf, env_indices) if critic_fr_tf is not None else None
                actor_pf_mb = tf.gather(actor_pf_tf, actor_indices) if actor_pf_tf is not None else None
                critic_pf_mb = tf.gather(critic_pf_tf, env_indices) if critic_pf_tf is not None else None

                (
                    actor_loss,
                    critic_loss,
                    entropy,
                    approx_kl,
                    clipfrac,
                    policy_loss,
                    value_loss,
                    policy_loss_head,
                    policy_loss_tail,
                    mean_head_weight,
                    mean_tail_weight,
                ) = self._ppo_minibatch_step(
                    actor_obs_mb,
                    actor_actions_mb,
                    actor_old_log_probs_mb,
                    actor_old_log_probs_head_mb,
                    actor_old_log_probs_tail_mb,
                    actor_advantages_mb,
                    critic_states_mb,
                    critic_returns_mb,
                    actor_fr_mb,
                    critic_fr_mb,
                    actor_pf_mb,
                    critic_pf_mb,
                )
                losses_actor_tf.append(actor_loss)
                losses_critic_tf.append(critic_loss)
                entropies_tf.append(entropy)
                kls_tf.append(approx_kl)
                clipfracs_tf.append(clipfrac)
                policy_losses_tf.append(policy_loss)
                value_losses_tf.append(value_loss)
                head_policy_losses_tf.append(policy_loss_head)
                tail_policy_losses_tf.append(policy_loss_tail)
                head_weights_tf.append(mean_head_weight)
                tail_weights_tf.append(mean_tail_weight)
                epoch_kls_tf.append(approx_kl)
            if self.target_kl > 0.0 and epoch_kls_tf:
                epoch_mean_kl = float(_safe_tensor_to_numpy(tf.reduce_mean(tf.stack(epoch_kls_tf, axis=0))))
                if epoch_mean_kl > self.target_kl:
                    break

        if losses_actor_tf:
            summary_tensor = tf.stack(
                [
                    tf.reduce_mean(tf.stack(losses_actor_tf, axis=0)),
                    tf.reduce_mean(tf.stack(losses_critic_tf, axis=0)),
                    tf.reduce_mean(tf.stack(entropies_tf, axis=0)),
                    tf.reduce_mean(tf.stack(kls_tf, axis=0)),
                    tf.reduce_mean(tf.stack(clipfracs_tf, axis=0)),
                    tf.reduce_mean(tf.stack(policy_losses_tf, axis=0)),
                    tf.reduce_mean(tf.stack(value_losses_tf, axis=0)),
                    tf.reduce_mean(tf.stack(head_policy_losses_tf, axis=0)),
                    tf.reduce_mean(tf.stack(tail_policy_losses_tf, axis=0)),
                    tf.reduce_mean(tf.stack(head_weights_tf, axis=0)),
                    tf.reduce_mean(tf.stack(tail_weights_tf, axis=0)),
                ],
                axis=0,
            )
            summary_stats = np.asarray(_safe_tensor_to_numpy(summary_tensor), dtype=np.float32)
        else:
            summary_stats = np.zeros((11,), dtype=np.float32)

        self.training_stats["train_steps"] = int(self.training_stats.get("train_steps", 0)) + 1
        self._last_update_stats = {
            "actor_loss": float(summary_stats[0]),
            "critic_loss": float(summary_stats[1]),
            "entropy": float(summary_stats[2]),
            "approx_kl": float(summary_stats[3]),
            "clipfrac": float(summary_stats[4]),
            "policy_loss": float(summary_stats[5]),
            "value_loss": float(summary_stats[6]),
            "policy_loss_head": float(summary_stats[7]),
            "policy_loss_tail": float(summary_stats[8]),
            "head_weight": float(summary_stats[9]),
            "tail_weight": float(summary_stats[10]),
        }
        return dict(self._last_update_stats)

    def save_models(self, path):
        os.makedirs(path, exist_ok=True)
        for i in range(self.n_agents):
            self.shared_actor.save_weights(os.path.join(path, f"actor_{i}.weights.h5"))
        self.value_critic.save_weights(os.path.join(path, "value_critic.weights.h5"))
        np.save(os.path.join(path, "actor_log_std.npy"), self.actor_log_std.numpy())
        meta = {
            "algorithm": "mappo",
            "n_agents": int(self.n_agents),
            "shared_actor": True,
            "separated_gradient": bool(self.use_separated_gradient),
        }
        with open(os.path.join(path, "mappo_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def load_models(self, path):
        actor_path = os.path.join(path, "actor_0.weights.h5")
        if not os.path.exists(actor_path):
            raise FileNotFoundError(f"缺少共享actor权重: {actor_path}")
        self.shared_actor.load_weights(actor_path)
        value_path = os.path.join(path, "value_critic.weights.h5")
        if os.path.exists(value_path):
            self.value_critic.load_weights(value_path)
        log_std_path = os.path.join(path, "actor_log_std.npy")
        if os.path.exists(log_std_path):
            self.actor_log_std.assign(np.load(log_std_path).astype(np.float32))
        for agent in self.agents:
            agent["actor"] = self.shared_actor
            agent["critic"] = self.value_critic
            agent["value_critic"] = self.value_critic
