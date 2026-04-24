"""On-policy rollout storage for MAPPO."""

from __future__ import annotations

import numpy as np


class MAPPORolloutBuffer:
    def __init__(
        self,
        rollout_length: int,
        num_envs: int,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        global_state_dim: int,
        pf_feature_dim: int = 0,
    ) -> None:
        self.rollout_length = int(rollout_length)
        self.num_envs = int(num_envs)
        self.n_agents = int(n_agents)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.global_state_dim = int(global_state_dim)
        self.pf_feature_dim = int(max(0, pf_feature_dim))

        self.obs = np.zeros((self.rollout_length, self.num_envs, self.n_agents, self.obs_dim), dtype=np.float32)
        self.global_state = np.zeros((self.rollout_length, self.num_envs, self.global_state_dim), dtype=np.float32)
        self.actions = np.zeros((self.rollout_length, self.num_envs, self.n_agents, self.action_dim), dtype=np.float32)
        self.log_probs = np.zeros((self.rollout_length, self.num_envs, self.n_agents), dtype=np.float32)
        self.log_probs_head = np.zeros((self.rollout_length, self.num_envs, self.n_agents), dtype=np.float32)
        self.log_probs_tail = np.zeros((self.rollout_length, self.num_envs, self.n_agents), dtype=np.float32)
        self.values = np.zeros((self.rollout_length, self.num_envs), dtype=np.float32)
        self.rewards = np.zeros((self.rollout_length, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.rollout_length, self.num_envs), dtype=np.float32)
        self.fr_values = np.zeros((self.rollout_length, self.num_envs), dtype=np.float32)
        self.pf_features = (
            np.zeros((self.rollout_length, self.num_envs, self.n_agents, self.pf_feature_dim), dtype=np.float32)
            if self.pf_feature_dim > 0
            else None
        )
        self.advantages = np.zeros((self.rollout_length, self.num_envs), dtype=np.float32)
        self.returns = np.zeros((self.rollout_length, self.num_envs), dtype=np.float32)
        self.ptr = 0
        self.full = False

    def reset(self) -> None:
        self.ptr = 0
        self.full = False

    def add_step(
        self,
        obs,
        global_state,
        actions,
        log_probs,
        values,
        rewards,
        dones,
        fr_values,
        pf_features=None,
        log_probs_head=None,
        log_probs_tail=None,
    ) -> None:
        if self.ptr >= self.rollout_length:
            raise IndexError("Rollout buffer is full")
        idx = self.ptr
        self.obs[idx] = np.asarray(obs, dtype=np.float32)
        self.global_state[idx] = np.asarray(global_state, dtype=np.float32)
        self.actions[idx] = np.asarray(actions, dtype=np.float32)
        self.log_probs[idx] = np.asarray(log_probs, dtype=np.float32)
        if log_probs_head is None:
            self.log_probs_head[idx].fill(0.0)
        else:
            self.log_probs_head[idx] = np.asarray(log_probs_head, dtype=np.float32)
        if log_probs_tail is None:
            self.log_probs_tail[idx].fill(0.0)
        else:
            self.log_probs_tail[idx] = np.asarray(log_probs_tail, dtype=np.float32)
        self.values[idx] = np.asarray(values, dtype=np.float32)
        self.rewards[idx] = np.asarray(rewards, dtype=np.float32)
        self.dones[idx] = np.asarray(dones, dtype=np.float32)
        self.fr_values[idx] = np.asarray(fr_values, dtype=np.float32)
        if self.pf_features is not None:
            if pf_features is None:
                self.pf_features[idx].fill(0.0)
            else:
                self.pf_features[idx] = np.asarray(pf_features, dtype=np.float32)
        self.ptr += 1
        self.full = self.ptr >= self.rollout_length

    def size(self) -> int:
        return int(self.ptr)

    def compute_gae_and_returns(self, last_values, gamma: float, gae_lambda: float) -> None:
        size = self.size()
        if size <= 0:
            return
        last_values = np.asarray(last_values, dtype=np.float32).reshape(self.num_envs)
        last_adv = np.zeros((self.num_envs,), dtype=np.float32)
        next_values = last_values
        for t in range(size - 1, -1, -1):
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + float(gamma) * next_values * not_done - self.values[t]
            last_adv = delta + float(gamma) * float(gae_lambda) * not_done * last_adv
            self.advantages[t] = last_adv
            self.returns[t] = self.advantages[t] + self.values[t]
            next_values = self.values[t]

    def iterate_env_minibatches(self, mini_batch_size: int, shuffle: bool = True):
        size = self.size()
        total = size * self.num_envs
        indices = np.arange(total, dtype=np.int32)
        if shuffle:
            np.random.shuffle(indices)
        mini_batch_size = max(1, int(mini_batch_size))
        for start in range(0, total, mini_batch_size):
            yield indices[start:start + mini_batch_size]

    def env_step_view(self):
        size = self.size()
        data = {
            "obs": self.obs[:size],
            "global_state": self.global_state[:size],
            "actions": self.actions[:size],
            "log_probs": self.log_probs[:size],
            "log_probs_head": self.log_probs_head[:size],
            "log_probs_tail": self.log_probs_tail[:size],
            "values": self.values[:size],
            "rewards": self.rewards[:size],
            "dones": self.dones[:size],
            "fr_values": self.fr_values[:size],
            "advantages": self.advantages[:size],
            "returns": self.returns[:size],
        }
        if self.pf_features is not None:
            data["pf_features"] = self.pf_features[:size]
        else:
            data["pf_features"] = None
        return data
