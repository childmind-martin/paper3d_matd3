#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("SUPPRESS_MA_PROMPT", "1")

from potential_field_corrector import ContinuousPotentialFieldCorrector, STATE_DIM, TERRAIN_DIM, OBSTACLE_DIM, GOAL_DIM, OTHER_AGENT_DIM, GOAL_OFFSET, OTHER_AGENT_OFFSET
from multiagent.scenarios.paper3d_terrain_weighted import Scenario as WeightedScenario


class TFPotentialFieldReference:
    """轻量版的TF势场参考实现，用于和NumPy版做数值对齐验证。"""

    def __init__(self, base_params, delta_params, map_size, max_force):
        self.map_size = tf.constant(map_size, dtype=tf.float32)
        self.map_half = self.map_size * 0.5
        self.max_force = tf.constant(max_force, dtype=tf.float32)

        # 将base/delta参数保存为tensor，便于广播
        self.base = {
            'goal_attraction': tf.constant(base_params['goal_attraction'], dtype=tf.float32),
            'lambda_1_base': tf.constant(base_params['lambda_1_base'], dtype=tf.float32),
            'terrain_repulsion': tf.constant(base_params['terrain_repulsion'], dtype=tf.float32),
            'agent_influence_range': tf.constant(base_params['agent_influence_range'], dtype=tf.float32),
        }
        self.delta = {
            'delta_k_att': tf.constant(delta_params['delta_k_att'], dtype=tf.float32),
            'delta_lambda_1': tf.constant(delta_params['delta_lambda_1'], dtype=tf.float32),
            'delta_k_rep': tf.constant(delta_params['delta_k_rep'], dtype=tf.float32),
            'delta_radius': tf.constant(delta_params['delta_radius'], dtype=tf.float32),
        }

    def _map_pf_params(self, pf_u):
        k_att = self.base['goal_attraction'] + pf_u[:, 0:1] * self.delta['delta_k_att']
        lambda_1 = self.base['lambda_1_base'] + pf_u[:, 1:2] * self.delta['delta_lambda_1']
        k_rep = self.base['terrain_repulsion'] + pf_u[:, 2:3] * self.delta['delta_k_rep']
        radius = self.base['agent_influence_range'] + pf_u[:, 3:4] * self.delta['delta_radius']

        k_att = tf.maximum(k_att, 0.1)
        lambda_1 = tf.maximum(lambda_1, 2.0)
        k_rep = tf.maximum(k_rep, 0.1)
        radius = tf.maximum(radius, 1.0)
        return k_att, lambda_1, k_rep, radius

    def _recover_positions(self, obs):
        norm_pos = obs[:, :3]
        return (norm_pos + 1.0) * self.map_half

    def _recover_goal(self, obs):
        goal_dir = obs[:, GOAL_OFFSET:GOAL_OFFSET + 3]
        goal_dist = obs[:, GOAL_OFFSET + 3:GOAL_OFFSET + 4] * self.map_size
        dir_norm = tf.norm(goal_dir, axis=1, keepdims=True)
        safe_dir = tf.where(dir_norm > 1e-6, goal_dir / dir_norm, tf.zeros_like(goal_dir))
        goal_pos = safe_dir * goal_dist
        return safe_dir, goal_dist, goal_pos

    def _calculate_goal_force(self, agent_pos, goal_pos, k_att, lambda_1):
        vec = goal_pos - agent_pos
        dist = tf.norm(vec, axis=1, keepdims=True)
        dist = tf.maximum(dist, tf.constant(1e-6, dtype=tf.float32))
        dir_to_goal = vec / dist

        d0 = tf.clip_by_value(lambda_1, 3.0, 15.0)
        dist_diff = dist - d0
        switch_factor = tf.sigmoid(5.0 * dist_diff)
        close = k_att * d0
        far = tf.cast(2.0, tf.float32) * k_att * dist
        attraction = close * (1.0 - switch_factor) + far * switch_factor
        attraction = tf.minimum(attraction, self.max_force)
        return dir_to_goal * attraction

    def _calculate_terrain_force(self, agent_pos, goal_pos, obs, k_rep, radius):
        terrain_info = obs[:, STATE_DIM:STATE_DIM + TERRAIN_DIM]
        current_height = terrain_info[:, 1:2] * 100.0
        gradients = terrain_info[:, 2:6] * 10.0

        agent_height = agent_pos[:, 2:3]
        height_diff = agent_height - current_height
        penetration_mask = height_diff < 0.0
        safe_r_min = tf.maximum(height_diff, 0.1)
        penetration_r_min = tf.maximum(-height_diff * 0.1, 0.05)
        r_min = tf.where(penetration_mask, penetration_r_min, safe_r_min)

        grad_x = (gradients[:, 0:1] + gradients[:, 2:3]) * 0.5
        grad_y = (gradients[:, 1:2] + gradients[:, 3:4]) * 0.5

        terrain_normal = tf.concat([-grad_x, -grad_y, tf.ones_like(grad_x)], axis=1)
        terrain_normal = tf.math.l2_normalize(terrain_normal, axis=1, epsilon=1e-6)
        upward = tf.constant([0.0, 0.0, 1.0], dtype=tf.float32)
        upward = tf.reshape(upward, [1, 3])
        terrain_normal = tf.where(
            tf.tile(penetration_mask, [1, 3]),
            tf.tile(upward, [tf.shape(agent_pos)[0], 1]),
            terrain_normal
        )

        goal_dist = tf.norm(goal_pos - agent_pos, axis=1, keepdims=True)
        kappa = tf.exp(-goal_dist / 50.0)

        inv_r_min = tf.clip_by_value(1.0 / (r_min + 1e-6), 0.0, 20.0)
        inv_R = 1.0 / (radius + 1e-6)
        base_repulsion = k_rep * (inv_r_min - inv_R) * tf.square(inv_r_min) * kappa
        penalty = tf.where(penetration_mask, 5.0, 1.0)
        repulsion_strength = base_repulsion * penalty
        max_strength = tf.where(penetration_mask, 50.0, 10.0)
        repulsion_strength = tf.clip_by_value(repulsion_strength, 0.0, max_strength)

        terrain_force = terrain_normal * repulsion_strength
        terrain_force_z = tf.abs(terrain_force[:, 2:3])
        terrain_force = tf.where(
            tf.tile(terrain_force_z > 0.5, [1, 3]),
            terrain_force * 2.5,
            terrain_force
        )
        return terrain_force

    def _calculate_obstacle_force(self, agent_pos, obs, radius):
        obstacle_info = obs[:, STATE_DIM + TERRAIN_DIM:STATE_DIM + TERRAIN_DIM + OBSTACLE_DIM]
        total = tf.zeros_like(agent_pos)
        map_size = self.map_size

        def _per_obstacle(direction, distance_norm):
            rel = direction * distance_norm * map_size
            return agent_pos + rel

        obstacle_positions = [
            _per_obstacle(obstacle_info[:, 0:3], obstacle_info[:, 3:4]),
            _per_obstacle(obstacle_info[:, 5:8], obstacle_info[:, 8:9]),
            _per_obstacle(obstacle_info[:, 10:13], obstacle_info[:, 13:14]),
        ]

        for obstacle_pos in obstacle_positions:
            vec = agent_pos - obstacle_pos
            dist = tf.norm(vec, axis=1, keepdims=True)
            dist = tf.maximum(dist, 0.5)
            rep_dir = vec / (dist + 1e-6)
            inv_dist = tf.clip_by_value(1.0 / (dist + 1e-6), 0.0, 10.0)
            inv_R = 1.0 / (radius + 1e-6)
            rep_strength = tf.where(
                dist < radius,
                (inv_dist - inv_R) * tf.square(inv_dist),
                tf.zeros_like(dist)
            )
            rep_strength = tf.clip_by_value(rep_strength, 0.0, 3.0)
            total += rep_dir * rep_strength
        return total

    def _calculate_agent_force(self, agent_pos, obs, radius):
        other_info = obs[:, OTHER_AGENT_OFFSET:OTHER_AGENT_OFFSET + OTHER_AGENT_DIM]
        rel_positions = [
            other_info[:, 0:3],
            other_info[:, 6:9]
        ]
        total = tf.zeros_like(agent_pos)
        for rel in rel_positions:
            other_abs = agent_pos + rel
            vec = agent_pos - other_abs
            dist = tf.norm(vec, axis=1, keepdims=True)
            dist = tf.maximum(dist, 0.5)
            rep_dir = vec / (dist + 1e-6)
            inv_dist = tf.clip_by_value(1.0 / (dist + 1e-6), 0.0, 10.0)
            inv_R = 1.0 / (radius + 1e-6)
            rep_strength = tf.where(
                dist < radius,
                (inv_dist - inv_R) * tf.square(inv_dist),
                tf.zeros_like(dist)
            )
            rep_strength = tf.clip_by_value(rep_strength, 0.0, 5.0)
            total += rep_dir * rep_strength
        return total

    def _mix(self, action_head, total_force, force_ratio):
        force_mag = tf.norm(total_force, axis=1, keepdims=True)
        clipped_force = tf.where(
            force_mag > self.max_force,
            total_force * (self.max_force / (force_mag + 1e-6)),
            total_force
        )
        force_mag = tf.minimum(force_mag, self.max_force)
        dir_pf = clipped_force / (tf.norm(clipped_force, axis=1, keepdims=True) + 1e-6)
        mag_pf_norm = tf.clip_by_value(force_mag / self.max_force, 0.0, 1.0)
        pf_action = dir_pf * mag_pf_norm
        fr = tf.clip_by_value(force_ratio, 0.0, 1.0)
        return tf.clip_by_value((1.0 - fr) * action_head + fr * pf_action, -1.0, 1.0)

    def apply(self, actions, observations, force_ratio):
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        observations = tf.convert_to_tensor(observations, dtype=tf.float32)
        batch = tf.shape(actions)[0]

        if tf.rank(force_ratio) == 0:
            fr = tf.fill([batch, 1], tf.cast(force_ratio, tf.float32))
        else:
            fr = tf.convert_to_tensor(force_ratio, dtype=tf.float32)
            if tf.shape(fr)[0] != batch:
                fr = tf.broadcast_to(fr, [batch, 1])

        pf_u = actions[:, 3:7]
        k_att, lambda_1, k_rep, radius = self._map_pf_params(pf_u)
        agent_pos = self._recover_positions(observations)
        goal_dir, goal_dist, rel_goal_pos = self._recover_goal(observations)
        goal_pos = agent_pos + rel_goal_pos

        goal_force = self._calculate_goal_force(agent_pos, goal_pos, k_att, lambda_1)
        terrain_force = self._calculate_terrain_force(agent_pos, goal_pos, observations, k_rep, radius)
        obstacle_force = self._calculate_obstacle_force(agent_pos, observations, radius)
        agent_force = self._calculate_agent_force(agent_pos, observations, radius)

        total_force = goal_force + terrain_force + obstacle_force + agent_force
        corrected_head = self._mix(actions[:, :3], total_force, fr)
        return corrected_head


def collect_samples(scenario, world, sample_count):
    samples = []
    while len(samples) < sample_count:
        scenario.reset_world(world)
        for agent in world.agents:
            obs = scenario.observation(agent, world)
            action = np.random.uniform(-1.0, 1.0, size=7).astype(np.float32)
            samples.append((obs.astype(np.float32), action))
            if len(samples) >= sample_count:
                break
    return samples


def main():
    parser = argparse.ArgumentParser(description="验证 TF 势场与 NumPy 势场输出是否一致")
    parser.add_argument("--samples", type=int, default=32, help="随机采样数量")
    parser.add_argument("--action-force-ratio", type=float, default=0.5, help="势场混合比例")
    parser.add_argument("--goal-attraction", type=float, default=2.5)
    parser.add_argument("--lambda-1-base", type=float, default=6.5)
    parser.add_argument("--terrain-repulsion", type=float, default=60.0)
    parser.add_argument("--agent-influence-range", type=float, default=30.0)
    parser.add_argument("--delta-k-att", type=float, default=1.8)
    parser.add_argument("--delta-lambda-1", type=float, default=4.0)
    parser.add_argument("--delta-k-rep", type=float, default=50.0)
    parser.add_argument("--delta-radius", type=float, default=6.0)
    parser.add_argument("--max-force", type=float, default=15.0)
    args = parser.parse_args()

    scenario = WeightedScenario()
    world = scenario.make_world()
    scenario.reset_world(world)

    samples = collect_samples(scenario, world, args.samples)
    observations = np.stack([s[0] for s in samples])
    actions = np.stack([s[1] for s in samples])

    base = {
        'goal_attraction': args.goal_attraction,
        'lambda_1_base': args.lambda_1_base,
        'terrain_repulsion': args.terrain_repulsion,
        'agent_influence_range': args.agent_influence_range,
    }
    delta = {
        'delta_k_att': args.delta_k_att,
        'delta_lambda_1': args.delta_lambda_1,
        'delta_k_rep': args.delta_k_rep,
        'delta_radius': args.delta_radius,
    }

    corrector = ContinuousPotentialFieldCorrector(
        terrain_data=getattr(scenario, "terrain", None),
        X=getattr(scenario, "X", None),
        Y=getattr(scenario, "Y", None),
        goal_attraction=base['goal_attraction'],
        lambda_1_base=base['lambda_1_base'],
        terrain_repulsion=base['terrain_repulsion'],
        influence_range=base['agent_influence_range'],
        max_force_magnitude=args.max_force,
        map_size=scenario.map_size,
        base_pf_params=base,
        delta_pf_params=delta,
    )

    numpy_head = np.stack([
        corrector.correct_from_observation(a, o, args.action_force_ratio)[:3]
        for a, o in zip(actions, observations)
    ])

    tf_reference = TFPotentialFieldReference(base, delta, scenario.map_size, args.max_force)
    tf_head = tf_reference.apply(actions, observations, args.action_force_ratio).numpy()

    diff = numpy_head - tf_head
    l2 = np.linalg.norm(diff, axis=1)

    print("==== 势场对齐验证结果 ====")
    print(f"样本数量: {args.samples}")
    print(f"最大绝对误差: {np.max(np.abs(diff)):.6f}")
    print(f"平均绝对误差: {np.mean(np.abs(diff)):.6f}")
    print(f"L2 误差统计: max={np.max(l2):.6f}, mean={np.mean(l2):.6f}")


if __name__ == "__main__":
    main()

