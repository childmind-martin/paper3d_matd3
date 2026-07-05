#!/usr/bin/env python3
"""Standalone Gazebo APF backend.

This module intentionally does not import or mutate the original Python APF
implementation. It re-computes the same runtime semantics from Gazebo-bound
state objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from gazebo_apf_state_provider import GazeboAPFSceneState, GazeboAPFStateProvider


def _safe_float(value: Any, default: float) -> float:
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return float(default)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _clip_by_norm(values: np.ndarray, max_norm: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    scale = np.minimum(1.0, float(max_norm) / np.maximum(norms, 1e-9))
    return arr * scale


@dataclass
class GazeboAPFConfig:
    goal_attraction: float = 26.0
    lambda_1_base: float = 8.5
    terrain_repulsion: float = 1600.0
    agent_influence_range: float = 150.0
    delta_k_att: float = 5.0
    delta_lambda_1: float = 2.2
    delta_k_rep: float = 600.0
    delta_radius: float = 80.0
    max_force_magnitude: float = 80.0
    terrain_safety_margin: float = 1.5
    action_range_x: float = 1.0
    action_range_y: float = 1.0
    action_range_z: float = 1.0
    z_action_bias: float = 0.0
    control_accel_gain: float = 1.0
    damping: float = 0.18
    simulation_dt: float = 0.08
    agent_accel: float = 3.6

    @classmethod
    def from_args(cls, args: Any) -> "GazeboAPFConfig":
        return cls(
            goal_attraction=_safe_float(getattr(args, "goal_attraction", None), 26.0),
            lambda_1_base=_safe_float(getattr(args, "lambda_1_base", None), 8.5),
            terrain_repulsion=_safe_float(getattr(args, "terrain_repulsion", None), 1600.0),
            agent_influence_range=_safe_float(getattr(args, "agent_influence_range", None), 150.0),
            delta_k_att=_safe_float(getattr(args, "delta_k_att", None), 5.0),
            delta_lambda_1=_safe_float(getattr(args, "delta_lambda_1", None), 2.2),
            delta_k_rep=_safe_float(getattr(args, "delta_k_rep", None), 600.0),
            delta_radius=_safe_float(getattr(args, "delta_radius", None), 80.0),
            max_force_magnitude=_safe_float(getattr(args, "max_force_magnitude", None), 80.0),
            terrain_safety_margin=_safe_float(getattr(args, "terrain_safety_margin", None), 1.5),
            action_range_x=_safe_float(getattr(args, "action_range_x", None), 1.0),
            action_range_y=_safe_float(getattr(args, "action_range_y", None), 1.0),
            action_range_z=_safe_float(getattr(args, "action_range_z", None), 1.0),
            z_action_bias=_safe_float(getattr(args, "z_action_bias", None), 0.0),
            control_accel_gain=_safe_float(getattr(args, "control_accel_gain", None), 1.0),
            damping=_safe_float(getattr(args, "damping", None), 0.18),
            simulation_dt=_safe_float(getattr(args, "simulation_dt", None), 0.08),
            agent_accel=_safe_float(getattr(args, "agent_accel", None), 3.6),
        )


@dataclass
class GazeboAPFResult:
    raw_actions: np.ndarray
    corrected_actions: np.ndarray
    corrected_accelerations: np.ndarray
    # Diagnostic one-step velocity prediction from Gazebo state + corrected acceleration.
    # The live adapter should send env-integrated agent velocities, not this field directly.
    cmd_vel: np.ndarray
    pf_forces: np.ndarray
    observations: np.ndarray
    debug: List[Dict[str, Any]]


class GazeboAPFCalculator:
    def __init__(self, config: GazeboAPFConfig, state_provider: GazeboAPFStateProvider) -> None:
        self.config = config
        self.state_provider = state_provider

    @classmethod
    def from_args(cls, args: Any, state_provider: GazeboAPFStateProvider) -> "GazeboAPFCalculator":
        return cls(GazeboAPFConfig.from_args(args), state_provider)

    def map_actor_pf_params(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float64)
        count = actions.shape[0]
        if actions.ndim != 2 or actions.shape[1] < 7:
            return np.zeros((count, 4), dtype=np.float64)
        pf_u = actions[:, 3:7]
        cfg = self.config
        k_att = cfg.goal_attraction + pf_u[:, 0:1] * cfg.delta_k_att
        lambda_1 = cfg.lambda_1_base + pf_u[:, 1:2] * cfg.delta_lambda_1
        k_rep = cfg.terrain_repulsion + pf_u[:, 2:3] * cfg.delta_k_rep
        radius = cfg.agent_influence_range + pf_u[:, 3:4] * cfg.delta_radius
        k_att = np.maximum(k_att, 0.1)
        lambda_1 = np.maximum(lambda_1, 2.0)
        k_rep = np.maximum(k_rep, max(cfg.terrain_repulsion * 0.5, 500.0))
        radius = np.maximum(radius, max(cfg.agent_influence_range * 0.5, 50.0))
        return np.concatenate([k_att, lambda_1, k_rep, radius], axis=1)

    def compute_base_pf_features(
        self,
        state: GazeboAPFSceneState,
        action_dim: int = 7,
        force_ratio: float = 1.0,
        observations: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if float(force_ratio) <= 0.0:
            return np.zeros((len(state.agents), 3), dtype=np.float32)
        dummy = np.zeros((len(state.agents), max(7, int(action_dim))), dtype=np.float32)
        result = self.correct_actions(dummy, state, force_ratio=1.0, observations=observations, mix_actions=False)
        return result.pf_forces.astype(np.float32, copy=False)

    def correct_actions(
        self,
        raw_actions: Sequence[Sequence[float]],
        state: GazeboAPFSceneState,
        force_ratio: float,
        observations: Optional[np.ndarray] = None,
        mix_actions: bool = True,
    ) -> GazeboAPFResult:
        raw = np.asarray(raw_actions, dtype=np.float64)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.shape[1] < 3:
            padded = np.zeros((raw.shape[0], 3), dtype=np.float64)
            padded[:, : raw.shape[1]] = raw
            raw = padded
        observations = (
            self.state_provider.build_observations(state)
            if observations is None
            else np.asarray(observations, dtype=np.float32)
        )
        count = min(raw.shape[0], len(state.agents), observations.shape[0])
        raw = raw[:count]
        observations = observations[:count]

        action_head = raw[:, :3].astype(np.float64, copy=True)
        valid_params = raw.shape[1] >= 7
        pf_params = self.map_actor_pf_params(raw) if valid_params else np.zeros((count, 4), dtype=np.float64)
        pf_params_valid = np.isfinite(pf_params).all(axis=1, keepdims=True) & (pf_params[:, 3:4] > 1e-6) & valid_params

        geometry = self._extract_geometry(state, observations)
        pf_forces, force_terms = self._compose_pf_actions(action_head, pf_params, pf_params_valid, geometry)
        if mix_actions:
            r = np.clip(float(force_ratio), 0.0, 1.0)
            corrected_head = action_head + r * (pf_forces - action_head)
        else:
            corrected_head = pf_forces.copy()
        corrected_head = np.where(np.isfinite(corrected_head), corrected_head, action_head)

        if raw.shape[1] > 3:
            corrected = np.concatenate([corrected_head, raw[:, 3:]], axis=1)
        else:
            corrected = corrected_head

        corrected_accel = self._normalized_action_to_acceleration(corrected_head, state)
        cmd_vel = self._acceleration_to_cmd_vel(corrected_accel, state)
        debug = self._build_debug(state, observations, pf_params, corrected, corrected_accel, cmd_vel, pf_forces, force_terms, geometry)
        return GazeboAPFResult(
            raw_actions=raw.astype(np.float32, copy=False),
            corrected_actions=corrected.astype(np.float32, copy=False),
            corrected_accelerations=corrected_accel.astype(np.float32, copy=False),
            cmd_vel=cmd_vel.astype(np.float32, copy=False),
            pf_forces=pf_forces.astype(np.float32, copy=False),
            observations=observations.astype(np.float32, copy=False),
            debug=debug,
        )

    def _extract_geometry(self, state: GazeboAPFSceneState, obs: np.ndarray) -> Dict[str, np.ndarray]:
        cfg = self.config
        obs = np.asarray(obs, dtype=np.float64)
        count = obs.shape[0]
        map_half = float(state.map_size) * 0.5
        pos = (obs[:, :3] + 1.0) * map_half
        terrain_info = obs[:, 9:41]
        obstacle_info = obs[:, 41:56].reshape(count, 3, 5)
        other_agents_info = obs[:, 63:75]

        goal_offset = 9 + 32 + 15
        goal_dir_norm = obs[:, goal_offset : goal_offset + 3]
        goal_dist_norm = obs[:, goal_offset + 3 : goal_offset + 4] * float(state.map_size)
        goal_abs_norm = obs[:, goal_offset + 4 : goal_offset + 7]
        dir_norm = np.linalg.norm(goal_dir_norm, axis=1, keepdims=True) + 1e-6
        valid_goal = dir_norm > 0.1
        goal_dir = np.where(valid_goal, goal_dir_norm / dir_norm, np.zeros_like(goal_dir_norm))
        goal_from_abs = (goal_abs_norm + 1.0) * map_half
        goal_from_rel = pos + goal_dir * goal_dist_norm
        abs_valid = np.sum(np.abs(goal_abs_norm), axis=1, keepdims=True) > 0.01
        goal_pos = np.where(abs_valid, goal_from_abs, np.where(valid_goal, goal_from_rel, pos))

        gx = np.clip(goal_pos[:, 0:1], 0.0, float(state.map_size) - 1.0)
        gy = np.clip(goal_pos[:, 1:2], 0.0, float(state.map_size) - 1.0)
        goal_terrain = np.asarray([state.terrain.height(float(x), float(y)) for x, y in zip(gx[:, 0], gy[:, 0])], dtype=np.float64).reshape(-1, 1)
        gz = np.clip(np.maximum(goal_pos[:, 2:3], goal_terrain + cfg.terrain_safety_margin), goal_terrain, float(state.map_size))
        goal_pos = np.concatenate([gx, gy, gz], axis=1)

        vec_to_goal = goal_pos - pos
        goal_dist = np.maximum(np.linalg.norm(vec_to_goal, axis=1, keepdims=True), 1e-6)
        goal_dir = vec_to_goal / goal_dist

        terrain_geom = self._terrain_geometry(pos, goal_pos, terrain_info)
        obstacle_geom = self._obstacle_geometry(obstacle_info, float(state.map_size))
        agent_geom = self._agent_geometry(pos, other_agents_info)
        geometry = {
            "pos": pos,
            "goal_pos": goal_pos,
            "goal_dir": goal_dir,
            "goal_dist": goal_dist,
        }
        geometry.update(terrain_geom)
        geometry.update(obstacle_geom)
        geometry.update(agent_geom)
        return geometry

    def _terrain_geometry(self, pos: np.ndarray, goal_pos: np.ndarray, terrain_info: np.ndarray) -> Dict[str, np.ndarray]:
        current_height = terrain_info[:, 1:2] * 100.0
        gradients = terrain_info[:, 2:6] * 10.0
        agent_height = pos[:, 2:3]
        clear_0 = agent_height - current_height
        forward_heights = terrain_info[:, 6:14] * 100.0
        clear_f = agent_height - forward_heights.reshape(pos.shape[0], 8)
        min_clear_forward = np.min(clear_f, axis=1, keepdims=True)
        r_min_candidate = np.minimum(clear_0, min_clear_forward)
        penetration = r_min_candidate < 0.0
        r_min = np.where(penetration, 0.05, np.maximum(r_min_candidate, 0.1))
        height_diff = r_min_candidate

        grad_x = (gradients[:, 0:1] - gradients[:, 2:3]) / 2.0
        grad_y = (gradients[:, 1:2] - gradients[:, 3:4]) / 2.0
        terrain_normal_base = np.concatenate([-grad_x, -grad_y, np.ones_like(grad_x)], axis=1)
        terrain_normal_base /= np.maximum(np.linalg.norm(terrain_normal_base, axis=1, keepdims=True), 1e-6)
        terrain_normal_safe = np.concatenate(
            [terrain_normal_base[:, 0:2], np.maximum(terrain_normal_base[:, 2:3], 0.1)],
            axis=1,
        )
        terrain_normal_safe /= np.maximum(np.linalg.norm(terrain_normal_safe, axis=1, keepdims=True), 1e-6)
        upward = np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64), (pos.shape[0], 1))
        terrain_normal = np.where(np.tile(penetration, (1, 3)), upward, terrain_normal_safe)
        terrain_normal = np.where(np.tile(terrain_normal[:, 2:3] < 0.0, (1, 3)), upward, terrain_normal)

        goal_dist_for_kappa = np.linalg.norm(goal_pos - pos, axis=1, keepdims=True)
        kappa = 1.0 / (1.0 + goal_dist_for_kappa / 500.0 + 1e-6)
        kappa = np.clip(kappa, 0.3, 1.0)
        inv_r_min = np.clip(1.0 / (r_min + 1e-6), 0.0, 20.0)
        penetration_sigmoid = 1.0 / (1.0 + np.exp(np.clip(-height_diff / 0.5, -50.0, 50.0)))
        terrain_penalty = 1.0 + 4.0 * (1.0 - penetration_sigmoid)
        deep_sigmoid = 1.0 / (1.0 + np.exp(np.clip(-(height_diff + 10.0) / 5.0, -50.0, 50.0)))
        deep_bonus = 5.0 * (1.0 - deep_sigmoid)
        terrain_penalty = np.where(height_diff < -10.0, terrain_penalty + deep_bonus, terrain_penalty)
        terrain_max = 100.0 + 400.0 * (1.0 - penetration_sigmoid)
        return {
            "terrain_normal": terrain_normal,
            "terrain_inv_r_min": inv_r_min,
            "terrain_kappa": kappa,
            "terrain_penalty_factor": terrain_penalty,
            "terrain_max_repulsion": terrain_max,
            "terrain_r_min_candidate": r_min_candidate,
            "terrain_current_height": current_height,
        }

    def _obstacle_geometry(self, obstacle_info: np.ndarray, map_size: float) -> Dict[str, np.ndarray]:
        surface_dist_norm = obstacle_info[:, :, 3:4]
        has_obstacle = surface_dist_norm < 0.99
        rel_pos = obstacle_info[:, :, 0:3] * map_size
        obstacle_radius = np.maximum(obstacle_info[:, :, 4:5] * map_size, 1.0)
        dist_center = np.linalg.norm(rel_pos, axis=2, keepdims=True)
        repulsion_dir = np.where(np.tile(has_obstacle, (1, 1, 3)), -rel_pos / (dist_center + 1e-6), np.zeros_like(rel_pos))
        raw_dist_surface = dist_center - obstacle_radius
        dist_surface = np.maximum(raw_dist_surface, 0.5)
        max_repulsion = np.where(raw_dist_surface < 0.0, 500.0, 100.0)
        return {
            "obstacle_has": has_obstacle,
            "obstacle_repulsion_dir": repulsion_dir,
            "obstacle_dist_surface": dist_surface,
            "obstacle_raw_dist_surface": raw_dist_surface,
            "obstacle_max_repulsion": max_repulsion,
        }

    def _agent_geometry(self, pos: np.ndarray, other_agents_info: np.ndarray) -> Dict[str, np.ndarray]:
        other1 = other_agents_info[:, 0:3] * 100.0
        other2 = other_agents_info[:, 6:9] * 100.0
        others_abs = np.stack([pos + other1, pos + other2], axis=1)
        diff = others_abs - pos[:, None, :]
        dist = np.maximum(np.linalg.norm(diff, axis=2, keepdims=True), 0.5)
        repulsion_dir = -diff / (dist + 1e-6)
        return {
            "agent_repulsion_dir": repulsion_dir,
            "agent_rel_dist": dist,
        }

    def _compose_pf_actions(
        self,
        action_head: np.ndarray,
        pf_params: np.ndarray,
        pf_params_valid: np.ndarray,
        geometry: Dict[str, np.ndarray],
    ) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        cfg = self.config
        k_att = pf_params[:, 0:1]
        lambda_1 = pf_params[:, 1:2]
        k_rep = pf_params[:, 2:3]
        radius = pf_params[:, 3:4]
        goal_dist = geometry["goal_dist"]

        d0 = np.clip(lambda_1, 3.0, 15.0)
        switch = _sigmoid(5.0 * (goal_dist - d0))
        goal_strength = k_att * d0 * (1.0 - switch) + 1.2 * k_att * goal_dist * switch
        goal_force = geometry["goal_dir"] * goal_strength

        inv_R_safe = 1.0 / (radius + 1e-6)
        base_repulsion = k_rep * (geometry["terrain_inv_r_min"] - inv_R_safe) * np.square(geometry["terrain_inv_r_min"]) * geometry["terrain_kappa"]
        base_repulsion = np.clip(base_repulsion, 0.0, 10000.0)
        terrain_strength = np.clip(
            base_repulsion * geometry["terrain_penalty_factor"],
            0.0,
            geometry["terrain_max_repulsion"],
        )
        terrain_force = geometry["terrain_normal"] * terrain_strength
        terrain_force[:, 2:3] = np.maximum(terrain_force[:, 2:3], 0.0)

        obstacle_dist = geometry["obstacle_dist_surface"]
        obstacle_inv_dist = np.clip(1.0 / (obstacle_dist + 1e-6), 0.0, 10.0)
        obstacle_inv_R = 1.0 / (radius[:, None, :] + 1e-6)
        obstacle_strength = (obstacle_inv_dist - obstacle_inv_R) * np.square(obstacle_inv_dist)
        obstacle_strength *= 1.0 / (1.0 + np.exp(np.clip(-(radius[:, None, :] - obstacle_dist) / 2.0, -50.0, 50.0)))
        obstacle_strength = np.clip(obstacle_strength, 0.0, geometry["obstacle_max_repulsion"])
        obstacle_strength = np.where(geometry["obstacle_has"], obstacle_strength, 0.0)
        obstacle_force = np.sum(geometry["obstacle_repulsion_dir"] * obstacle_strength, axis=1)

        agent_dist = geometry["agent_rel_dist"]
        agent_inv_dist = np.clip(1.0 / (agent_dist + 1e-6), 0.0, 10.0)
        agent_inv_R = 1.0 / (radius[:, None, :] + 1e-6)
        agent_strength = (agent_inv_dist - agent_inv_R) * np.square(agent_inv_dist)
        agent_strength *= 1.0 / (1.0 + np.exp(np.clip(-(radius[:, None, :] - agent_dist) / 2.0, -50.0, 50.0)))
        agent_strength = np.clip(agent_strength, 0.0, 15.0) * 3.0
        agent_force = np.sum(geometry["agent_repulsion_dir"] * agent_strength, axis=1)

        kappa_goal = np.exp(-goal_dist / 50.0)
        terrain_force = terrain_force * kappa_goal
        obstacle_force = obstacle_force * kappa_goal
        enhancement = 1.0 + 1.5 * _sigmoid((np.abs(terrain_force[:, 2:3]) - 0.5) * 4.0)
        terrain_force_enhanced = terrain_force * enhancement

        max_goal = cfg.max_force_magnitude * 3.0
        max_terrain = cfg.max_force_magnitude * 5.0
        max_other = cfg.max_force_magnitude * 2.0
        goal_limited = _clip_by_norm(goal_force, max_goal)
        terrain_limited = _clip_by_norm(terrain_force_enhanced, max_terrain)
        agent_limited = _clip_by_norm(agent_force, max_other)
        obstacle_limited = _clip_by_norm(obstacle_force, max_other)
        total = goal_limited + terrain_limited + agent_limited + obstacle_limited

        force_mag = np.linalg.norm(total, axis=1, keepdims=True)
        theoretical = np.sqrt(max_goal * max_goal + max_terrain * max_terrain + max_other * max_other * 2.0)
        norm_base = np.minimum(force_mag, theoretical)
        norm_base = np.maximum(norm_base, cfg.max_force_magnitude * 0.1)
        pf_action = total / (norm_base + 1e-6)
        pf_action = np.clip(np.where(np.isfinite(pf_action), pf_action, 0.0), -1.0, 1.0)
        finite = np.isfinite(pf_action).all(axis=1, keepdims=True)
        use_pf = pf_params_valid & finite
        pf_action = np.where(use_pf, pf_action, action_head)
        terms = {
            "goal_force": goal_force,
            "terrain_force": terrain_force_enhanced,
            "obstacle_force": obstacle_force,
            "agent_force": agent_force,
            "goal_force_limited": goal_limited,
            "terrain_force_limited": terrain_limited,
            "obstacle_force_limited": obstacle_limited,
            "agent_force_limited": agent_limited,
            "total_force_limited": total,
        }
        return pf_action, terms

    def _normalized_action_to_acceleration(self, corrected_head: np.ndarray, state: GazeboAPFSceneState) -> np.ndarray:
        cfg = self.config
        accel = np.asarray(corrected_head, dtype=np.float64).copy()
        accel[:, 0] *= cfg.action_range_x
        accel[:, 1] *= cfg.action_range_y
        accel[:, 2] = (accel[:, 2] + cfg.z_action_bias) * cfg.action_range_z * cfg.control_accel_gain
        agent_accels = np.asarray([agent.accel if np.isfinite(agent.accel) else cfg.agent_accel for agent in state.agents[: accel.shape[0]]], dtype=np.float64)
        accel *= agent_accels[:, None]
        return np.where(np.isfinite(accel), accel, 0.0)

    def _acceleration_to_cmd_vel(self, acceleration: np.ndarray, state: GazeboAPFSceneState) -> np.ndarray:
        cfg = self.config
        velocities = state.velocities[: acceleration.shape[0]].astype(np.float64, copy=True)
        masses = np.asarray([max(agent.mass, 1e-9) for agent in state.agents[: acceleration.shape[0]]], dtype=np.float64)
        cmd = velocities * (1.0 - cfg.damping) + (acceleration / masses[:, None]) * cfg.simulation_dt
        max_speeds = state.max_speeds[: cmd.shape[0]]
        speed = np.linalg.norm(cmd, axis=1, keepdims=True)
        scale = np.minimum(1.0, max_speeds[:, None] / np.maximum(speed, 1e-9))
        cmd = cmd * scale
        return np.where(np.isfinite(cmd), cmd, 0.0)

    def _build_debug(
        self,
        state: GazeboAPFSceneState,
        obs: np.ndarray,
        pf_params: np.ndarray,
        corrected: np.ndarray,
        corrected_accel: np.ndarray,
        cmd_vel: np.ndarray,
        pf_forces: np.ndarray,
        force_terms: Dict[str, np.ndarray],
        geometry: Dict[str, np.ndarray],
    ) -> List[Dict[str, Any]]:
        debug: List[Dict[str, Any]] = []
        for idx, agent in enumerate(state.agents[: corrected.shape[0]]):
            nearest = self.state_provider.nearest_obstacle(state, idx)
            terrain_clearance = self.state_provider.terrain_clearance(state, idx)
            total = force_terms["total_force_limited"][idx]
            norm = float(np.linalg.norm(total))
            direction = total / (norm + 1e-9)
            debug.append(
                {
                    "agent_id": int(idx),
                    "agent_name": agent.name,
                    "pose": agent.position.astype(float).tolist(),
                    "velocity": agent.velocity.astype(float).tolist(),
                    "goal": state.goals[idx].astype(float).tolist(),
                    "contact": bool(agent.contact),
                    "pf_params": pf_params[idx].astype(float).tolist() if idx < pf_params.shape[0] else None,
                    "corrected_action": corrected[idx].astype(float).tolist(),
                    "corrected_acceleration": corrected_accel[idx].astype(float).tolist(),
                    "cmd_vel": cmd_vel[idx].astype(float).tolist(),
                    "pf_force_action": pf_forces[idx].astype(float).tolist(),
                    "force_norm": norm,
                    "force_direction": direction.astype(float).tolist(),
                    "goal_force": force_terms["goal_force"][idx].astype(float).tolist(),
                    "terrain_force": force_terms["terrain_force"][idx].astype(float).tolist(),
                    "obstacle_force": force_terms["obstacle_force"][idx].astype(float).tolist(),
                    "agent_agent_repulsion": force_terms["agent_force"][idx].astype(float).tolist(),
                    "terrain_clearance": terrain_clearance,
                    "terrain_r_min_candidate": float(geometry["terrain_r_min_candidate"][idx, 0]),
                    "nearest_obstacle": nearest,
                }
            )
        return debug
