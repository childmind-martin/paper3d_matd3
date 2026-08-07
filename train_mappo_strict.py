#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standard external MAPPO baseline aligned with the current strict training protocol."""

from __future__ import annotations

import argparse
import functools
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from algorithms.mappo import OptimizedMAPPO
from algorithms.mappo.rollout_buffer import MAPPORolloutBuffer
from paper3d_train_optimized import (
    ParallelEnv,
    SingleEnvWrapper,
    _make_json_safe,
    configure_gpu,
    load_scenario_module,
    make_env_init,
)


def _str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _quiet() -> bool:
    return os.getenv("QUIET_OUTPUT", "1").lower() in ("1", "true", "yes", "on")


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _safe_float(value, default=0.0) -> float:
    try:
        value = float(value)
    except Exception:
        return float(default)
    return float(value) if np.isfinite(value) else float(default)


def _safe_int(value, default=0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _capture_training_environment_config(args) -> Dict[str, Any]:
    terrain_seed = _safe_int(os.getenv("SCENARIO_SEED", getattr(args, "terrain_seed", 67)), 67)
    terrain_base_seed_default = getattr(args, "terrain_base_seed", None)
    if terrain_base_seed_default is None:
        terrain_base_seed_default = terrain_seed
    terrain_base_seed = _safe_int(os.getenv("TERRAIN_BASE_SEED", terrain_base_seed_default), terrain_seed)
    training_env_sequence_seed_default = getattr(args, "training_env_sequence_seed", None)
    if training_env_sequence_seed_default is None:
        training_env_sequence_seed_default = terrain_base_seed
    training_env_sequence_seed = _safe_int(
        os.getenv("TRAIN_ENV_SEQUENCE_SEED", training_env_sequence_seed_default),
        terrain_base_seed,
    )
    semi_random = bool(
        getattr(args, "semi_random_terrain", False)
        or os.getenv("SEMI_RANDOM_TERRAIN", "0").lower() in ("1", "true", "yes", "on")
    )
    return {
        "schema_version": 1,
        "source": "train_mappo_strict",
        "use_fixed_positions": bool(getattr(args, "use_fixed_positions", False)),
        "use_dynamic_obstacles": bool(
            getattr(args, "use_dynamic_obstacles", False)
            or os.getenv("USE_DYNAMIC_OBSTACLES", "0").lower() in ("1", "true", "yes", "on")
        ),
        "random_terrain": bool(
            getattr(args, "random_terrain", False)
            or os.getenv("RANDOM_TERRAIN", "0").lower() in ("1", "true", "yes", "on")
        ),
        "semi_random_terrain": semi_random,
        "deterministic_env_sequence": bool(
            getattr(args, "deterministic_train_env_sequence", False)
            or os.getenv("DETERMINISTIC_TRAIN_ENV_SEQUENCE", "0").lower() in ("1", "true", "yes", "on")
        ),
        "terrain_seed": terrain_seed,
        "terrain_base_seed": terrain_base_seed,
        "training_env_sequence_seed": training_env_sequence_seed,
        "train_obstacle_sequence_mode": str(
            os.getenv(
                "TRAIN_OBSTACLE_SEQUENCE_MODE",
                str(getattr(args, "train_obstacle_sequence_mode", "legacy_linear") or "legacy_linear"),
            )
        ),
        "train_obstacle_sequence_namespace": str(
            os.getenv(
                "TRAIN_OBSTACLE_SEQUENCE_NAMESPACE",
                str(getattr(args, "train_obstacle_sequence_namespace", "train_obstacle") or "train_obstacle"),
            )
        ),
        "peak_jitter_range": float(os.getenv("PEAK_JITTER_RANGE", str(getattr(args, "peak_jitter_range", 0.0) or 0.0))),
        "peak_center_jitter_range": float(
            os.getenv("PEAK_CENTER_JITTER_RANGE", str(getattr(args, "peak_center_jitter_range", 0.0) or 0.0))
        ),
        "peak_height_jitter_ratio_min": float(
            os.getenv("PEAK_HEIGHT_JITTER_RATIO_MIN", str(getattr(args, "peak_height_jitter_ratio_min", 0.0) or 0.0))
        ),
        "peak_height_jitter_ratio_max": float(
            os.getenv("PEAK_HEIGHT_JITTER_RATIO_MAX", str(getattr(args, "peak_height_jitter_ratio_max", 0.0) or 0.0))
        ),
        "peak_height_max_scale": float(
            os.getenv("PEAK_HEIGHT_MAX_SCALE", str(getattr(args, "peak_height_max_scale", 1.0) or 1.0))
        ),
        "terrain_variant_noise_ratio": float(
            os.getenv("TERRAIN_VARIANT_NOISE_RATIO", str(getattr(args, "terrain_variant_noise_ratio", 0.0) or 0.0))
        ),
        "semi_random_hold_mode": str(
            os.getenv("SEMI_RANDOM_TERRAIN_HOLD_MODE", str(getattr(args, "semi_random_hold_mode", "episode") or "episode"))
        ),
        "semi_random_hold_episodes": int(
            os.getenv("SEMI_RANDOM_TERRAIN_HOLD_EPISODES", str(getattr(args, "semi_random_hold_episodes", 1) or 1))
        ),
        "semi_random_hold_min_episodes": int(
            os.getenv("SEMI_RANDOM_TERRAIN_HOLD_MIN_EPISODES", str(getattr(args, "semi_random_hold_min_episodes", 1) or 1))
        ),
        "semi_random_hold_max_episodes": int(
            os.getenv("SEMI_RANDOM_TERRAIN_HOLD_MAX_EPISODES", str(getattr(args, "semi_random_hold_max_episodes", 1) or 1))
        ),
    }


def _capture_training_hyperparameters_config(args) -> Dict[str, Any]:
    use_sep = bool(getattr(args, "mappo_use_separated_gradient", False))
    return {
        "algorithm_family": "ppo",
        "replay_buffer_size": 0,
        "action_force_ratio": float(getattr(args, "action_force_ratio", 0.0) or 0.0),
        "action_force_ratio_schedule_pct": str(getattr(args, "action_force_ratio_schedule_pct", "") or ""),
        "use_fr_feature": bool(getattr(args, "use_fr_feature", False)),
        "use_pf_feature": bool(getattr(args, "use_pf_feature", False)),
        "pf_feature_dim": int(getattr(args, "pf_feature_dim", 0) or 0),
        "actor_objective_mode": "ppo_clip_separated" if use_sep else "ppo_clip",
        "use_dual_q": False,
        "use_separated_gradient": use_sep,
        "use_hybrid_actor_objective": False,
        "hybrid_actor_alpha": 0.0,
        "rollout_length": int(getattr(args, "rollout_length", 1024)),
        "ppo_epochs": int(getattr(args, "ppo_epochs", 4)),
        "mini_batch_size": int(getattr(args, "mini_batch_size", getattr(args, "batch_size", 1024) or 1024)),
        "clip_ratio": float(getattr(args, "clip_ratio", 0.2)),
        "gae_lambda": float(getattr(args, "gae_lambda", 0.95)),
        "entropy_coef": float(getattr(args, "entropy_coef", 0.01)),
        "value_coef": float(getattr(args, "value_coef", 0.5)),
        "target_kl": float(getattr(args, "target_kl", 0.03)),
        "actor_learning_rate": float(getattr(args, "learning_rate_actor", 3e-4)),
        "critic_learning_rate": float(getattr(args, "learning_rate_critic", 5e-4)),
    }


def _resolve_seed(args) -> int:
    seed = getattr(args, "seed", None)
    if seed is None:
        try:
            seed = int(os.getenv("SEED", "1337"))
        except Exception:
            seed = 1337
    else:
        seed = int(seed)
    args.seed = int(seed)
    return int(seed)


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _compute_env_done(done_n: np.ndarray) -> np.ndarray:
    done_n = np.asarray(done_n).astype(bool)
    if done_n.ndim == 1:
        done_n = done_n[None, :]
    mode = os.getenv("EARLY_STOP_MODE", "all").strip().lower()
    if mode in ("never", "disabled"):
        return np.zeros(done_n.shape[0], dtype=bool)
    if mode == "any":
        return np.any(done_n, axis=1)
    if mode == "majority":
        ratio = _safe_float(os.getenv("EARLY_STOP_MAJORITY_RATIO", "0.5"), 0.5)
        need = max(1, int(np.ceil(ratio * done_n.shape[1])))
        return np.sum(done_n.astype(np.int32), axis=1) >= need
    return np.all(done_n, axis=1)


def _apply_force_ratio_schedule(args, trainer, episode_idx: int) -> float:
    if not hasattr(args, "_base_action_force_ratio"):
        args._base_action_force_ratio = float(getattr(args, "action_force_ratio", 0.0))
    pct_str = getattr(args, "action_force_ratio_schedule_pct", None)
    if (pct_str is None) or (str(pct_str).strip() == ""):
        pct_env = os.getenv("ACTION_FORCE_RATIO_SCHEDULE_PCT", "").strip()
        if pct_env and pct_env.upper() != "DISABLED":
            pct_str = pct_env
    if pct_str and str(pct_str).strip().upper() != "DISABLED":
        pairs = [p.strip() for p in str(pct_str).split(",") if p.strip()]
        schedule_points = []
        for item in pairs:
            if ":" not in item:
                continue
            key, value = item.split(":", 1)
            ks = key.strip().rstrip("%")
            try:
                kp = float(ks)
                if "%" in key:
                    kp = kp / 100.0
                schedule_points.append((kp, float(value)))
            except Exception:
                continue
        if schedule_points:
            schedule_points.sort(key=lambda x: x[0])
            total_episodes = max(1, int(getattr(args, "train_episodes", 1)))
            resume_start_episode = max(0, int(getattr(args, "_resume_fr_schedule_start_episode", 0) or 0))
            if resume_start_episode > 0:
                remaining = max(1, total_episodes - resume_start_episode)
                local_index = max(0, int(episode_idx) - resume_start_episode)
                progress = float(local_index + 1) / float(remaining)
            else:
                progress = float(episode_idx + 1) / float(total_episodes)
            progress = max(0.0, min(1.0, progress))
            if progress <= schedule_points[0][0]:
                new_ratio = schedule_points[0][1]
            elif progress >= schedule_points[-1][0]:
                new_ratio = schedule_points[-1][1]
            else:
                left = schedule_points[0]
                right = schedule_points[-1]
                for idx in range(1, len(schedule_points)):
                    if schedule_points[idx][0] >= progress:
                        left = schedule_points[idx - 1]
                        right = schedule_points[idx]
                        break
                span = max(right[0] - left[0], 1e-6)
                interp = (progress - left[0]) / span
                new_ratio = left[1] + interp * (right[1] - left[1])
            args.action_force_ratio = float(new_ratio)
    current = float(getattr(args, "action_force_ratio", 0.0))
    trainer.action_force_ratio_cached = current
    if hasattr(trainer, "action_force_ratio_var"):
        try:
            trainer.action_force_ratio_var.assign(current)
        except Exception:
            pass
    return current


def _collect_episode_summary(env) -> Dict[str, Any]:
    summaries: List[Dict[str, Any]] = []
    envs_to_check: List[Tuple[int, Any]] = []
    if hasattr(env, "envs") and isinstance(getattr(env, "envs"), (list, tuple)) and len(env.envs) > 0:
        for env_idx, single_env in enumerate(env.envs):
            envs_to_check.append((int(env_idx), single_env))
    elif hasattr(env, "env"):
        envs_to_check.append((0, env.env))
    elif hasattr(env, "world"):
        envs_to_check.append((0, env))

    for env_idx, single_env in envs_to_check:
        base_env = getattr(single_env, "env", None)
        if base_env is None:
            base_env = single_env
        world = getattr(base_env, "world", None)
        if world is None:
            continue
        summary = {
            "env_idx": int(env_idx),
            "agent_success_flags": list(getattr(world, "_episode_agent_success_flags", []) or []),
            "agent_reach_flags": list(getattr(world, "_episode_agent_reach_flags", []) or []),
            "agent_safe_flags": list(getattr(world, "_episode_agent_safe_flags", []) or []),
            "agent_collision_counts": [],
            "agent_goal_distances": [],
            "agent_min_distances": [],
            "terrain_collision_total": 0,
            "obstacle_collision_total": 0,
        }
        for agent in getattr(world, "agents", []):
            debug_info = getattr(agent, "debug_info", {}) if isinstance(getattr(agent, "debug_info", None), dict) else {}
            pen_count = _safe_int(debug_info.get("total_penetration_count", 0), 0)
            terrain_count = _safe_int(debug_info.get("terrain_penetration_count", 0), 0)
            obstacle_count = _safe_int(debug_info.get("obstacle_collision_count", 0), 0)
            d_min = debug_info.get("d_min_current", None)
            if d_min is None:
                d_min = getattr(agent, "last_min_distance", None)
            if isinstance(d_min, np.ndarray):
                d_min = float(d_min.reshape(-1)[-1]) if d_min.size > 0 else None
            elif d_min is not None:
                d_min = _safe_float(d_min, default=np.nan)
            goal_pos = None
            try:
                if hasattr(agent, "goal_a") and agent.goal_a is not None:
                    goal_pos = getattr(getattr(agent.goal_a, "state", None), "p_pos", None)
                elif getattr(getattr(base_env, "scenario", None), "goal_pos", None) is not None:
                    goal_pos = base_env.scenario.goal_pos
            except Exception:
                goal_pos = None
            pos = getattr(getattr(agent, "state", None), "p_pos", None)
            if goal_pos is None or pos is None:
                goal_dist = None
            else:
                try:
                    goal_dist = float(np.linalg.norm(np.asarray(pos, dtype=np.float32) - np.asarray(goal_pos, dtype=np.float32)))
                except Exception:
                    goal_dist = None
            summary["agent_collision_counts"].append(int(pen_count))
            summary["agent_goal_distances"].append(goal_dist)
            summary["agent_min_distances"].append(d_min)
            summary["terrain_collision_total"] += int(terrain_count)
            summary["obstacle_collision_total"] += int(obstacle_count)
        summaries.append(summary)

    agent_success_flags: List[int] = []
    agent_reach_flags: List[int] = []
    agent_safe_flags: List[int] = []
    agent_collision_counts: List[int] = []
    agent_goal_distances: List[Optional[float]] = []
    agent_min_distances: List[Optional[float]] = []
    terrain_collision_total = 0
    obstacle_collision_total = 0
    for summary in summaries:
        agent_success_flags.extend([_safe_int(v, 0) for v in summary["agent_success_flags"]])
        agent_reach_flags.extend([_safe_int(v, 0) for v in summary["agent_reach_flags"]])
        agent_safe_flags.extend([_safe_int(v, 0) for v in summary["agent_safe_flags"]])
        agent_collision_counts.extend([_safe_int(v, 0) for v in summary["agent_collision_counts"]])
        agent_goal_distances.extend(summary["agent_goal_distances"])
        agent_min_distances.extend(summary["agent_min_distances"])
        terrain_collision_total += int(summary["terrain_collision_total"])
        obstacle_collision_total += int(summary["obstacle_collision_total"])

    team_success_flag = 1 if agent_success_flags and all(int(v) == 1 for v in agent_success_flags) else 0
    min_distance_cleaned = [float(v) for v in agent_min_distances if v is not None and np.isfinite(v)]
    min_distance_payload = {
        "min_distance": float(min(min_distance_cleaned)) if min_distance_cleaned else None,
        "agent_min_distances": [
            (float(v) if v is not None and np.isfinite(v) else None) for v in agent_min_distances
        ],
    }
    return {
        "agent_success_flags": agent_success_flags,
        "agent_reach_flags": agent_reach_flags,
        "agent_safe_flags": agent_safe_flags,
        "team_success_flag": int(team_success_flag),
        "success_flag": int(team_success_flag),
        "agent_collision_counts": agent_collision_counts,
        "agent_goal_distances": [
            (float(v) if v is not None and np.isfinite(v) else None) for v in agent_goal_distances
        ],
        "total_collisions": int(sum(agent_collision_counts)),
        "terrain_collision_total": int(terrain_collision_total),
        "obstacle_collision_total": int(obstacle_collision_total),
        "min_distance_payload": min_distance_payload,
    }


def _save_checkpoint(
    trainer: OptimizedMAPPO,
    checkpoint_dir: str,
    episode_idx: int,
    episode_rewards: Sequence[float],
    episode_force_ratios: Sequence[float],
    best_reward: float,
    best_episode: int,
    best_team_success_rate: float,
    best_team_sr_episode: int,
    best_team_sr_force_ratio: float,
    best_team_sr_reward: float,
    success_flags: Sequence[int],
    agent_success_flags: Sequence[Sequence[int]],
    team_success_flags: Sequence[int],
    actor_losses_history: Sequence[float],
    critic_losses_history: Sequence[float],
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    trainer.save_models(checkpoint_dir)
    state_path = os.path.join(checkpoint_dir, "checkpoint_state.json")
    if not _env_flag_enabled("SAVE_TRAINING_RESUME_STATE", default=True):
        try:
            if os.path.exists(state_path):
                os.unlink(state_path)
        except Exception as exc:
            raise RuntimeError(f"清理已禁用的旧续训状态失败: {state_path}") from exc
        return
    state = {
        "episode": int(episode_idx),
        "episode_rewards": list(episode_rewards),
        "episode_force_ratios": list(episode_force_ratios),
        "best_reward": float(best_reward),
        "best_episode": int(best_episode),
        "best_episode_force_ratio": float(episode_force_ratios[best_episode]) if 0 <= best_episode < len(episode_force_ratios) else 0.0,
        "best_team_success_rate": float(best_team_success_rate),
        "best_team_sr_episode": int(best_team_sr_episode),
        "best_team_sr_force_ratio": float(best_team_sr_force_ratio),
        "best_team_sr_reward": float(best_team_sr_reward) if np.isfinite(best_team_sr_reward) else None,
        "success_flags": list(success_flags),
        "agent_success_flags": list(agent_success_flags),
        "team_success_flags": list(team_success_flags),
        "actor_losses_history": list(actor_losses_history),
        "critic_losses_history": list(critic_losses_history),
        "entropy_coef_history": list(getattr(trainer, "_entropy_coef_history", []) or []),
        "current_entropy_coef": float(getattr(trainer, "entropy_coef", getattr(trainer.args, "entropy_coef", 0.01))),
        "adaptive_learning_state": dict(getattr(trainer, "adaptive_learning", {}) or {}),
        "noise_scale_var_history": [],
        "train_episodes": int(getattr(trainer.args, "train_episodes", len(episode_rewards))),
        "exp_name": str(getattr(trainer.args, "exp_name", "")),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(_make_json_safe(state), f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="标准MAPPO外部基线训练脚本")
    parser.add_argument("--algo", type=str, default="mappo")
    parser.add_argument("--scenario", type=str, default="paper3d_terrain_energy")
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--episode-length", type=int, default=2800)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--exp-name", type=str, default="mappo_experiment")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument(
        "--save-interval",
        type=int,
        default=0,
        help="周期 epN 模型保存间隔；0 表示关闭中间回合快照",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)

    parser.add_argument("--learning-rate-actor", type=float, default=3e-4)
    parser.add_argument("--learning-rate-critic", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--actor-hidden", type=str, default=None)
    parser.add_argument("--critic-hidden", type=str, default=None)
    parser.add_argument("--rollout-length", type=int, default=1024)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--mini-batch-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--mappo-use-separated-gradient", type=_str2bool, default=False)
    parser.add_argument("--mem-debug", type=_str2bool, default=False)
    parser.add_argument("--debug-pf-forces", type=_str2bool, default=False)
    parser.add_argument("--profiling", type=_str2bool, default=False)
    parser.add_argument("--step-warn-s", type=float, default=5.0)
    parser.add_argument("--stack-dump-timeout", type=float, default=30.0)
    parser.add_argument("--auto-abort-on-nan", type=_str2bool, default=True)
    parser.add_argument("--restart-on-collapse", type=_str2bool, default=False)
    parser.add_argument("--collapse-patience", type=int, default=3)
    parser.add_argument("--collapse-loss-threshold", type=float, default=1000.0)
    parser.add_argument("--collapse-z-threshold", type=float, default=-50.0)

    parser.add_argument("--use-vectorization", type=_str2bool, default=True)
    parser.add_argument("--vectorized-rewards", type=_str2bool, default=True)
    parser.add_argument("--vectorized-observations", type=_str2bool, default=True)
    parser.add_argument("--xla-global", type=_str2bool, default=False)
    parser.add_argument("--jit-compile", type=_str2bool, default=False)
    parser.add_argument("--amp-mode", type=str, default="off")

    parser.add_argument("--action-force-ratio", type=float, default=0.7)
    parser.add_argument("--action-force-ratio-schedule-pct", type=str, default=None)
    parser.add_argument("--use-fr-feature", type=_str2bool, default=False)
    parser.add_argument("--use-pf-feature", type=_str2bool, default=False)
    parser.add_argument("--pf-feature-dim", type=int, default=3)
    parser.add_argument("--use-tf-potential-field", type=_str2bool, default=True)
    parser.add_argument("--goal-attraction", type=float, default=1.0)
    parser.add_argument("--lambda-1-base", type=float, default=5.0)
    parser.add_argument("--terrain-repulsion", type=float, default=80.0)
    parser.add_argument("--agent-influence-range", type=float, default=10.0)
    parser.add_argument("--delta-k-att", type=float, default=0.5)
    parser.add_argument("--delta-lambda-1", type=float, default=2.5)
    parser.add_argument("--delta-k-rep", type=float, default=40.0)
    parser.add_argument("--delta-radius", type=float, default=5.0)
    parser.add_argument("--max-force-magnitude", type=float, default=8.0)
    parser.add_argument("--max-weight-threshold", type=float, default=None)
    parser.add_argument("--weight-scaling-factor", type=float, default=None)
    parser.add_argument("--force-scale", type=float, default=5.0)
    parser.add_argument("--agent-repulsion", type=float, default=0.0)
    parser.add_argument("--terrain-sensing-mode", type=str, default="local")
    parser.add_argument("--success-count-mode", type=str, default=os.getenv("SUCCESS_COUNT_MODE", "any"))

    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--control-accel-gain", type=float, default=12.0)
    parser.add_argument("--damping", type=float, default=0.25)
    parser.add_argument("--action-range-x", type=float, default=2.0)
    parser.add_argument("--action-range-y", type=float, default=2.0)
    parser.add_argument("--action-range-z", type=float, default=1.0)
    parser.add_argument("--agent-max-speed", type=float, default=None)
    parser.add_argument("--agent-accel", type=float, default=None)
    parser.add_argument("--simulation-dt", type=float, default=0.08)
    parser.add_argument("--z-action-bias", type=float, default=0.0)
    parser.add_argument(
        "--quadrotor-attitude-response-time",
        type=float,
        default=float(os.getenv("QUADROTOR_ATTITUDE_RESPONSE_TIME", "0.0")),
    )
    parser.add_argument(
        "--quadrotor-psi-cmd",
        type=float,
        default=float(os.getenv("QUADROTOR_PSI_CMD", "0.0")),
    )
    parser.add_argument(
        "--use-quadrotor-dynamics",
        type=_str2bool,
        default=_env_flag_enabled("USE_QUADROTOR_DYNAMICS", False),
    )
    parser.add_argument("--pre-takeoff-start-radius", type=float, default=1.0)
    parser.add_argument("--pre-takeoff-airborne-threshold", type=float, default=0.5)

    parser.add_argument("--reward-pos-scale", type=float, default=1.0)
    parser.add_argument("--reward-neg-scale", type=float, default=1.0)
    parser.add_argument("--success-distance-threshold", type=float, default=2.0)
    parser.add_argument("--collision-distance-threshold", type=float, default=0.5)
    parser.add_argument("--collision-penalty-value", type=float, default=30.0)
    parser.add_argument("--success-reward-value", type=float, default=150.0)
    parser.add_argument("--no-collision-reward-value", type=float, default=0.0)
    parser.add_argument("--global-reward-mode", type=str, default="avg_progress")
    parser.add_argument("--shaping-gamma", type=float, default=0.95)
    parser.add_argument("--distance-weight", type=float, default=None)
    parser.add_argument("--exploration-weight", type=float, default=None)
    parser.add_argument("--stationary-weight", type=float, default=None)
    parser.add_argument("--direction-weight", type=float, default=None)
    parser.add_argument("--deviation-weight", type=float, default=None)
    parser.add_argument("--start-area-weight", type=float, default=None)
    parser.add_argument("--approach-weight", type=float, default=None)
    parser.add_argument("--energy-weight", type=float, default=None)
    parser.add_argument("--height-weight", type=float, default=None)
    parser.add_argument("--turn-smooth-weight", type=float, default=None)
    parser.add_argument("--height-reward-enabled", type=_str2bool, default=None)
    parser.add_argument("--height-ideal-min", type=float, default=None)
    parser.add_argument("--height-ideal-max", type=float, default=None)
    parser.add_argument("--lateral-weight", type=float, default=None)
    parser.add_argument("--clearance-weight", type=float, default=None)
    parser.add_argument("--clearance-d-max", type=float, default=None)
    parser.add_argument("--success-weight", type=float, default=None)
    parser.add_argument("--collision-weight", type=float, default=None)
    parser.add_argument("--collision-reduction-weight", type=float, default=None)
    parser.add_argument("--global-weight", type=float, default=None)
    parser.add_argument("--shaping-weight", type=float, default=None)
    parser.add_argument("--max-reward", type=float, default=None)
    parser.add_argument("--min-reward", type=float, default=None)
    parser.add_argument("--terrain-contact-eps", type=float, default=None)

    parser.add_argument("--use-fixed-positions", action="store_true")
    parser.add_argument("--positions-file", type=str, default="./saved_positions/5.json")
    parser.add_argument("--save-positions", action="store_true")
    parser.add_argument("--terrain-seed", type=int, default=None)
    parser.add_argument("--terrain-base-seed", type=int, default=None)
    parser.add_argument("--training-env-sequence-seed", type=int, default=None)
    parser.add_argument("--random-terrain", action="store_true")
    parser.add_argument("--per-env-terrain", type=_str2bool, default=None)
    parser.add_argument("--per-episode-terrain", type=_str2bool, default=None)
    parser.add_argument("--semi-random-terrain", type=_str2bool, default=None)
    parser.add_argument("--deterministic-train-env-sequence", type=_str2bool, default=None)
    parser.add_argument("--dynamic-first-time", action="store_true")
    parser.add_argument("--random-z0-positions", type=_str2bool, default=False)
    parser.add_argument("--use-dynamic-obstacles", type=_str2bool, default=None)
    parser.add_argument("--unlock-env-on-success", type=_str2bool, default=False)
    parser.add_argument("--unlock-env-on-plateau", type=_str2bool, default=False)
    parser.add_argument("--peak-jitter-range", type=float, default=None)
    parser.add_argument("--peak-center-jitter-range", type=float, default=None)
    parser.add_argument("--peak-height-jitter-ratio-min", type=float, default=None)
    parser.add_argument("--peak-height-jitter-ratio-max", type=float, default=None)
    parser.add_argument("--peak-height-max-scale", type=float, default=None)
    parser.add_argument("--terrain-variant-noise-ratio", type=float, default=None)
    parser.add_argument("--semi-random-hold-mode", type=str, default=None)
    parser.add_argument("--semi-random-hold-episodes", type=int, default=None)
    parser.add_argument("--semi-random-hold-min-episodes", type=int, default=None)
    parser.add_argument("--semi-random-hold-max-episodes", type=int, default=None)
    parser.add_argument("--terrain-complexity-level", type=int, default=3)
    parser.add_argument("--map-size", type=float, default=float(os.getenv("MAP_SIZE", "200")))
    parser.add_argument("--mountain-min-distance", type=float, default=None)
    parser.add_argument("--enable-reward-debug", type=_str2bool, default=False)
    parser.add_argument("--scenario-kw", action="append", default=None)
    parser.add_argument("--matd3-use-dual-q", type=_str2bool, default=False)
    parser.add_argument("--matd3-use-separated-gradient", type=_str2bool, default=False)
    parser.add_argument("--maddpg-use-dual-q", type=_str2bool, default=False)
    parser.add_argument("--maddpg-use-separated-gradient", type=_str2bool, default=False)

    # 保持 parse_known_args 兼容当前 run_optimized.sh。
    # shell 仍可能传入主线的 legacy off-policy 参数，这里直接忽略未知项。
    args, _unknown = parser.parse_known_args()
    if args.mini_batch_size is None:
        args.mini_batch_size = int(args.batch_size)
    args.training_environment_config = _capture_training_environment_config(args)
    args.training_hyperparameters_config = _capture_training_hyperparameters_config(args)
    return args


def train(args):
    quiet_output = _quiet()
    if not quiet_output:
        print("[MAPPO] starting training")

    configure_gpu()
    if bool(getattr(args, "xla_global", False)):
        try:
            os.environ.pop("XLA_FLAGS", None)
            os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
            tf.config.optimizer.set_jit(True)
        except Exception:
            pass

    resolved_seed = _resolve_seed(args)
    _set_random_seed(resolved_seed)
    os.environ["SUPPRESS_MA_PROMPT"] = "1"

    args_dict = vars(args).copy()
    if args.num_envs > 1:
        env_fns = [functools.partial(make_env_init, i, args_dict) for i in range(args.num_envs)]
        env = ParallelEnv(env_fns)
    else:
        env = SingleEnvWrapper(make_env_init(0, args_dict))

    probe_env = make_env_init(0, args_dict)
    n_agents = probe_env.n
    obs_shapes = [probe_env.observation_space[i].shape[0] for i in range(n_agents)]
    args.base_obs_shapes = list(obs_shapes)
    # Keep MAPPO aligned with the main training stack: PF features should only be
    # enabled when the resolved CLI/env flag says so, rather than being forced on
    # whenever external APF correction is active.
    args.use_pf_feature = bool(getattr(args, "use_pf_feature", False))
    args.pf_feature_dim = int(getattr(args, "pf_feature_dim", 3))
    args.training_hyperparameters_config = _capture_training_hyperparameters_config(args)
    action_dims = [7] * n_agents
    probe_env.close()

    trainer = OptimizedMAPPO(n_agents, obs_shapes, action_dims, args)
    if hasattr(env, "env") and hasattr(env.env, "scenario"):
        trainer.scenario_ref = env.env.scenario
        trainer.world_ref = env.env.world
        trainer.scenario = env.env.scenario
        trainer.update_terrain_cache(env.env.scenario)

    run_dir = os.path.join("logs", args.exp_name)
    model_root = os.path.join("models", args.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(model_root, exist_ok=True)

    episode_rewards: List[float] = []
    episode_force_ratios: List[float] = []
    actor_losses_history: List[float] = []
    critic_losses_history: List[float] = []
    entropy_coef_history: List[float] = []
    success_flags: List[int] = []
    agent_success_flags: List[List[int]] = []
    team_success_flags: List[int] = []
    collision_counts: List[int] = []
    terrain_collision_counts: List[int] = []
    obstacle_collision_counts: List[int] = []
    agent_collision_counts: List[List[int]] = []
    min_distances: List[Dict[str, Any]] = []
    loss_history: List[Dict[str, Any]] = []

    best_reward = -np.inf
    best_episode = -1
    best_team_success_rate = -1.0
    best_team_sr_episode = -1
    best_team_sr_force_ratio = 0.0
    best_team_sr_reward = -np.inf
    start_episode = 0
    resume_completed_episodes = 0

    resume_dir = None
    if args.checkpoint:
        resume_dir = args.checkpoint
    elif args.resume:
        resume_dir = os.path.join(model_root, "checkpoint")
    if resume_dir and os.path.isdir(resume_dir):
        state_path = os.path.join(resume_dir, "checkpoint_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            start_episode = _safe_int(state.get("episode", 0), 0)
            episode_rewards = list(state.get("episode_rewards", []) or [])
            episode_force_ratios = list(state.get("episode_force_ratios", []) or [])
            best_reward = _safe_float(state.get("best_reward", -np.inf), -np.inf)
            best_episode = _safe_int(state.get("best_episode", -1), -1)
            best_team_success_rate = _safe_float(state.get("best_team_success_rate", -1.0), -1.0)
            best_team_sr_episode = _safe_int(state.get("best_team_sr_episode", -1), -1)
            best_team_sr_force_ratio = _safe_float(state.get("best_team_sr_force_ratio", 0.0), 0.0)
            best_team_sr_reward = _safe_float(state.get("best_team_sr_reward", -np.inf), -np.inf)
            success_flags = list(state.get("success_flags", []) or [])
            agent_success_flags = [list(x) for x in (state.get("agent_success_flags", []) or [])]
            team_success_flags = list(state.get("team_success_flags", []) or [])
            actor_losses_history = list(state.get("actor_losses_history", []) or [])
            critic_losses_history = list(state.get("critic_losses_history", []) or [])
            entropy_coef_history = list(state.get("entropy_coef_history", []) or [])
            try:
                current_entropy_coef = state.get("current_entropy_coef", None)
                if current_entropy_coef is not None:
                    trainer.entropy_coef = float(current_entropy_coef)
                    args.entropy_coef = float(current_entropy_coef)
            except Exception:
                pass
            try:
                adaptive_state = state.get("adaptive_learning_state", None)
                if isinstance(adaptive_state, dict) and hasattr(trainer, "adaptive_learning"):
                    trainer.adaptive_learning.update(adaptive_state)
            except Exception:
                pass
            trainer._episode_success_flags = list(success_flags)
            trainer._episode_agent_success_flags = [list(x) for x in agent_success_flags]
            trainer._episode_team_success_flags = list(team_success_flags)
            trainer._entropy_coef_history = list(entropy_coef_history)
            completed_episode_candidates = [int(start_episode)]
            for seq in (
                episode_rewards,
                episode_force_ratios,
                team_success_flags,
                success_flags,
            ):
                try:
                    completed_episode_candidates.append(len(seq))
                except Exception:
                    pass
            resume_completed_episodes = max(0, max(completed_episode_candidates))
            if resume_completed_episodes != int(start_episode):
                print(
                    f"   - 续训对齐: 以已完成回合数 {resume_completed_episodes} 作为起始回合 "
                    f"(checkpoint episode={start_episode})"
                )
                start_episode = int(resume_completed_episodes)
            else:
                resume_completed_episodes = int(start_episode)
            if resume_completed_episodes > 0:
                setattr(args, "_resume_fr_schedule_start_episode", int(resume_completed_episodes))
                try:
                    args.action_force_ratio = float(
                        getattr(args, "_base_action_force_ratio", getattr(args, "action_force_ratio", 0.0))
                    )
                except Exception:
                    pass
                print(
                    f"   - FR续训策略: 从初始值重启，并按剩余 {max(1, int(args.train_episodes) - resume_completed_episodes)} 回合重新调度"
                )
        if os.path.exists(os.path.join(resume_dir, "actor_0.weights.h5")):
            trainer.load_models(resume_dir)

    rollout_buffer = MAPPORolloutBuffer(
        rollout_length=int(args.rollout_length),
        num_envs=int(args.num_envs),
        n_agents=int(n_agents),
        obs_dim=int(obs_shapes[0]),
        action_dim=int(action_dims[0]),
        global_state_dim=int(sum(obs_shapes)),
        pf_feature_dim=int(args.pf_feature_dim if args.use_pf_feature else 0),
    )

    tqdm_to_stdout = _env_flag_enabled("TQDM_TO_STDOUT", default=False)
    tqdm_file = sys.stdout if tqdm_to_stdout else sys.stderr
    tqdm_disable = _env_flag_enabled("TQDM_DISABLE", default=False)
    try:
        tqdm_mininterval = float(os.getenv("TQDM_MININTERVAL", "0.5"))
    except Exception:
        tqdm_mininterval = 0.5
    try:
        tqdm_ncols = int(os.getenv("TQDM_NCOLS", "100"))
    except Exception:
        tqdm_ncols = 100
    default_bar_fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
    tqdm_bar_format = os.getenv("TQDM_BAR_FORMAT", default_bar_fmt)
    use_light_progress = _env_flag_enabled("USE_LIGHT_PROGRESS", default=True)
    timing_enabled = _env_flag_enabled("TIMING_SUMMARY", default=False) or bool(getattr(args, "profiling", False))
    progress_output_file = sys.stdout if use_light_progress else tqdm_file

    def _new_timing_bucket() -> Dict[str, float]:
        return {
            "collect": 0.0,
            "env": 0.0,
            "obs": 0.0,
            "bootstrap_pf": 0.0,
            "bootstrap_value": 0.0,
            "update": 0.0,
        }

    class _LightProgress:
        def __init__(self, total, desc="", ncols=100, mininterval=0.5, initial=0, file_obj=None):
            self.total = int(total)
            self.desc = str(desc)
            self.ncols = int(ncols)
            self.mininterval = float(mininterval)
            self.current = int(max(0, initial))
            self._last = 0.0
            self._t0 = time.time()
            self._postfix = ""
            self._file = file_obj or sys.stdout
            if self.current > 0:
                self._draw(force=True)

        def set_postfix_str(self, s):
            self._postfix = s if isinstance(s, str) else str(s)

        def set_description(self, desc):
            self.desc = str(desc)

        @staticmethod
        def _fmt(sec):
            sec = max(0, int(sec))
            h = sec // 3600
            m = (sec % 3600) // 60
            s = sec % 60
            if h > 0:
                return f"{h}:{m:02d}:{s:02d}"
            return f"{m:02d}:{s:02d}"

        def _draw(self, force=False):
            now = time.time()
            if (not force) and ((now - self._last) < self.mininterval):
                return
            self._last = now
            frac = self.current / max(self.total, 1)
            bar_len = max(min(self.ncols - 30, 50), 10)
            fill = int(bar_len * frac)
            bar = "#" * fill + "-" * (bar_len - fill)
            elapsed = now - self._t0
            per_ep = elapsed / max(1, self.current)
            remaining = per_ep * max(0, self.total - self.current)
            total_est = per_ep * self.total
            eta_str = f"{self._fmt(elapsed)}<{self._fmt(remaining)} | total {self._fmt(total_est)}"
            msg = f"\r{self.desc} [{bar}] {self.current}/{self.total} ({eta_str}) {self._postfix}"
            self._file.write(msg)
            self._file.flush()

        def update(self, n=1):
            self.current = min(self.total, self.current + int(n))
            self._draw(force=False)

        def close(self):
            if self.current < self.total:
                self._draw(force=True)
            self._file.write("\n")
            self._file.flush()

    if use_light_progress:
        progress_bar = _LightProgress(
            total=int(args.train_episodes),
            desc="训练进度",
            ncols=tqdm_ncols,
            mininterval=tqdm_mininterval,
            initial=start_episode,
            file_obj=progress_output_file,
        )
    else:
        progress_bar = tqdm(
            total=int(args.train_episodes),
            initial=start_episode,
            position=0,
            leave=True,
            desc="训练进度",
            ncols=tqdm_ncols,
            mininterval=tqdm_mininterval,
            dynamic_ncols=True,
            file=tqdm_file,
            disable=tqdm_disable,
            unit="ep",
            bar_format=tqdm_bar_format,
        )
    episode_iterator = range(start_episode, int(args.train_episodes))

    def _prime_resume_env_sequence_state(training_env, completed_episodes):
        """在首次 reset 前对齐续训场景序列，保证 episode/障碍序列从断点接上。"""
        completed_episodes = int(completed_episodes or 0)
        if completed_episodes <= 0 or training_env is None:
            return

        def _apply(single_env):
            if single_env is None:
                return
            world = getattr(single_env, "world", None)
            scenario = getattr(single_env, "scenario", None)
            env_id = 0
            try:
                env_id = int(getattr(world, "env_id", getattr(scenario, "current_episode_env_id", 0)) or 0)
            except Exception:
                env_id = 0

            if world is not None:
                try:
                    world._episode_index_counter = int(completed_episodes)
                except Exception:
                    pass
                try:
                    world.episode_index = max(0, int(completed_episodes) - 1)
                except Exception:
                    pass
            if scenario is not None:
                try:
                    scenario.current_episode_index = int(completed_episodes)
                except Exception:
                    pass
                try:
                    scenario.current_episode_env_id = int(env_id)
                except Exception:
                    pass
                if (
                    bool(getattr(scenario, "use_dynamic_obstacles", False))
                    and not bool(getattr(scenario, "deterministic_train_env_sequence", False))
                ):
                    try:
                        scenario._obstacle_reset_count = int(completed_episodes)
                    except Exception:
                        pass

        try:
            if hasattr(training_env, "envs"):
                for single_env in list(getattr(training_env, "envs", []) or []):
                    _apply(single_env)
            elif hasattr(training_env, "env"):
                _apply(getattr(training_env, "env", None))
            else:
                _apply(training_env)
            if not quiet_output:
                print(
                    f"🔁 续训场景序列已对齐到 episode={completed_episodes} "
                    f"(首次新回合将从该索引继续)"
                )
        except Exception as resume_env_exc:
            if not quiet_output:
                print(f"[WARN] 续训场景序列对齐失败: {resume_env_exc}")

    def _progress_write(line: str) -> None:
        if use_light_progress:
            try:
                progress_output_file.write(f"\n{line}\n")
                progress_output_file.flush()
            except Exception:
                pass
        else:
            tqdm.write(line, file=progress_output_file)

    try:
        if resume_completed_episodes > 0:
            _prime_resume_env_sequence_state(env, resume_completed_episodes)
        for episode in episode_iterator:
            episode_start_time = time.time()
            current_fr = _apply_force_ratio_schedule(args, trainer, episode)
            skip_network_update = os.getenv("SKIP_NETWORK_UPDATE", "0") == "1"
            if hasattr(progress_bar, "set_description"):
                try:
                    progress_bar.set_description(f"训练进度 | FR={current_fr:.2f}")
                except Exception:
                    progress_bar.set_description("训练进度")

            obs_n = env.reset()
            if hasattr(env, "env") and hasattr(env.env, "scenario"):
                trainer.scenario_ref = env.env.scenario
                trainer.world_ref = env.env.world
                trainer.scenario = env.env.scenario
                trainer.update_terrain_cache(env.env.scenario)
            processed_obs = trainer.obs_processor.batch_process_observations_vectorized(obs_n)
            _obs_tensor_cached = tf.convert_to_tensor(processed_obs, dtype=tf.float32)
            episode_env_done = np.zeros((int(args.num_envs),), dtype=bool)
            per_env_cum_rewards = np.zeros((int(args.num_envs),), dtype=np.float32)
            step_count = 0
            rollout_buffer.reset()
            last_actor_loss = None
            last_critic_loss = None
            last_policy_loss = None
            last_value_loss = None
            last_entropy = None
            timing_ep = _new_timing_bucket() if timing_enabled else None

            while step_count < int(args.episode_length):
                t_collect0 = time.perf_counter() if timing_enabled else None
                actions_storage, actions_exec, pf_features_np = trainer.collect_rollout_step_vectorized(
                    _obs_tensor_cached,
                    add_noise=True,
                )
                if timing_enabled:
                    timing_ep["collect"] += time.perf_counter() - t_collect0

                t_env0 = time.perf_counter() if timing_enabled else None
                next_obs_n, rew_n, done_n, _infos = env.step(actions_exec)
                if timing_enabled:
                    timing_ep["env"] += time.perf_counter() - t_env0
                rew_eff = np.asarray(rew_n, dtype=np.float32).copy()
                rew_eff[episode_env_done, :] = 0.0
                team_rewards = np.mean(rew_eff, axis=1).astype(np.float32)
                env_done_now = _compute_env_done(done_n)
                global_state = trainer.build_global_state_numpy(processed_obs)

                rollout_buffer.add_step(
                    obs=processed_obs,
                    global_state=global_state,
                    actions=actions_storage,
                    log_probs=np.zeros((int(args.num_envs), int(n_agents)), dtype=np.float32),
                    values=np.zeros((int(args.num_envs),), dtype=np.float32),
                    rewards=team_rewards,
                    dones=env_done_now.astype(np.float32),
                    fr_values=np.full((int(args.num_envs),), current_fr, dtype=np.float32),
                    pf_features=pf_features_np,
                )

                per_env_cum_rewards += team_rewards
                t_obs0 = time.perf_counter() if timing_enabled else None
                processed_next_obs = trainer.obs_processor.batch_process_observations_vectorized(next_obs_n)
                if timing_enabled:
                    timing_ep["obs"] += time.perf_counter() - t_obs0
                step_count += 1

                all_done_now = bool(np.all(np.logical_or(episode_env_done, env_done_now)))
                if rollout_buffer.full or all_done_now or step_count == int(args.episode_length):
                    if skip_network_update:
                        rollout_buffer.reset()
                        processed_obs = processed_next_obs
                        _obs_tensor_cached = tf.convert_to_tensor(processed_obs, dtype=tf.float32)
                        episode_env_done = np.logical_or(episode_env_done, env_done_now)
                        if all_done_now:
                            break
                        continue
                    last_pf = None
                    if args.use_pf_feature:
                        if float(current_fr) > 0.0 and getattr(trainer, "use_tf_potential_field_cached", getattr(args, "use_tf_potential_field", True)):
                            t_boot_pf0 = time.perf_counter() if timing_enabled else None
                            last_pf = trainer.compute_base_pf_forces_batch_numpy(processed_next_obs, current_fr)
                            if timing_enabled:
                                timing_ep["bootstrap_pf"] += time.perf_counter() - t_boot_pf0
                            if last_pf.shape[-1] > int(args.pf_feature_dim):
                                last_pf = last_pf[:, :, : int(args.pf_feature_dim)]
                        else:
                            last_pf = np.zeros(
                                (int(args.num_envs), int(n_agents), int(args.pf_feature_dim)),
                                dtype=np.float32,
                            )
                    t_boot_v0 = time.perf_counter() if timing_enabled else None
                    last_values = trainer.predict_values_vectorized(
                        processed_next_obs,
                        fr_values=np.full((int(args.num_envs), 1), current_fr, dtype=np.float32) if args.use_fr_feature else None,
                        pf_features=last_pf,
                    )
                    rollout_data = rollout_buffer.env_step_view()
                    rollout_values = trainer.predict_rollout_values_vectorized(
                        rollout_data["global_state"],
                        fr_values=rollout_data["fr_values"] if args.use_fr_feature else None,
                        pf_features=rollout_data["pf_features"] if args.use_pf_feature else None,
                    )
                    rollout_buffer.values[: rollout_buffer.size()] = np.asarray(rollout_values, dtype=np.float32)
                    if bool(getattr(args, "mappo_use_separated_gradient", False)):
                        (
                            rollout_log_probs_total,
                            rollout_log_probs_head,
                            rollout_log_probs_tail,
                        ) = trainer.predict_rollout_log_prob_splits_vectorized(
                            rollout_data["obs"],
                            rollout_data["actions"],
                            fr_values=rollout_data["fr_values"] if args.use_fr_feature else None,
                            pf_features=rollout_data["pf_features"] if args.use_pf_feature else None,
                        )
                        rollout_buffer.log_probs[: rollout_buffer.size()] = np.asarray(rollout_log_probs_total, dtype=np.float32)
                        rollout_buffer.log_probs_head[: rollout_buffer.size()] = np.asarray(rollout_log_probs_head, dtype=np.float32)
                        rollout_buffer.log_probs_tail[: rollout_buffer.size()] = np.asarray(rollout_log_probs_tail, dtype=np.float32)
                    else:
                        rollout_log_probs = trainer.predict_rollout_log_probs_vectorized(
                            rollout_data["obs"],
                            rollout_data["actions"],
                            fr_values=rollout_data["fr_values"] if args.use_fr_feature else None,
                            pf_features=rollout_data["pf_features"] if args.use_pf_feature else None,
                        )
                        rollout_buffer.log_probs[: rollout_buffer.size()] = np.asarray(rollout_log_probs, dtype=np.float32)
                    if timing_enabled:
                        timing_ep["bootstrap_value"] += time.perf_counter() - t_boot_v0
                    rollout_buffer.compute_gae_and_returns(last_values, float(args.gamma), float(args.gae_lambda))
                    t_update0 = time.perf_counter() if timing_enabled else None
                    update_stats = trainer.update(rollout_buffer)
                    if timing_enabled:
                        timing_ep["update"] += time.perf_counter() - t_update0
                    if update_stats:
                        actor_loss = _safe_float(update_stats.get("actor_loss", 0.0), 0.0)
                        critic_loss = _safe_float(update_stats.get("critic_loss", 0.0), 0.0)
                        policy_loss = _safe_float(update_stats.get("policy_loss", actor_loss), actor_loss)
                        value_loss = _safe_float(update_stats.get("value_loss", critic_loss), critic_loss)
                        entropy = _safe_float(update_stats.get("entropy", 0.0), 0.0)
                        last_actor_loss = actor_loss
                        last_critic_loss = critic_loss
                        last_policy_loss = policy_loss
                        last_value_loss = value_loss
                        last_entropy = entropy
                        actor_losses_history.append(actor_loss)
                        critic_losses_history.append(critic_loss)
                        loss_record = {
                            "step": int(trainer.training_stats.get("train_steps", 0)),
                            "episode": int(episode),
                            "critic_loss": critic_loss,
                            "actor_loss": actor_loss,
                            "policy_loss": policy_loss,
                            "value_loss": value_loss,
                            "entropy": entropy,
                            "approx_kl": _safe_float(update_stats.get("approx_kl", 0.0), 0.0),
                            "clipfrac": _safe_float(update_stats.get("clipfrac", 0.0), 0.0),
                            "policy_loss_head": _safe_float(update_stats.get("policy_loss_head", 0.0), 0.0),
                            "policy_loss_tail": _safe_float(update_stats.get("policy_loss_tail", 0.0), 0.0),
                            "head_weight": _safe_float(update_stats.get("head_weight", 0.0), 0.0),
                            "tail_weight": _safe_float(update_stats.get("tail_weight", 0.0), 0.0),
                        }
                        loss_history.append(loss_record)
                    rollout_buffer.reset()

                processed_obs = processed_next_obs
                _obs_tensor_cached = tf.convert_to_tensor(processed_obs, dtype=tf.float32)
                episode_env_done = np.logical_or(episode_env_done, env_done_now)
                if all_done_now:
                    break

            avg_episode_reward = float(np.max(per_env_cum_rewards)) if per_env_cum_rewards.size > 0 else 0.0
            episode_rewards.append(avg_episode_reward)
            episode_force_ratios.append(float(current_fr))
            trainer.training_stats["episodes"] = int(episode + 1)
            trainer.training_stats["total_steps"] = int(trainer.training_stats.get("total_steps", 0)) + int(step_count)
            if hasattr(trainer, "total_steps_var"):
                trainer.total_steps_var.assign_add(tf.cast(step_count, tf.int64))

            summary = _collect_episode_summary(env)
            success_flags.append(int(summary["success_flag"]))
            agent_success_flags.append(list(summary["agent_success_flags"]))
            team_success_flags.append(int(summary["team_success_flag"]))
            collision_counts.append(int(summary["total_collisions"]))
            terrain_collision_counts.append(int(summary["terrain_collision_total"]))
            obstacle_collision_counts.append(int(summary["obstacle_collision_total"]))
            agent_collision_counts.append(list(summary["agent_collision_counts"]))
            min_distances.append(summary["min_distance_payload"])
            trainer._episode_success_flags = list(success_flags)
            trainer._episode_agent_success_flags = [list(x) for x in agent_success_flags]
            trainer._episode_team_success_flags = list(team_success_flags)
            trainer._episode_collision_counts = list(collision_counts)
            trainer._episode_terrain_collision_counts = list(terrain_collision_counts)
            trainer._episode_obstacle_collision_counts = list(obstacle_collision_counts)
            trainer._episode_agent_collision_counts = [list(x) for x in agent_collision_counts]
            trainer._episode_min_distances = list(min_distances)
            trainer._entropy_coef_history = list(entropy_coef_history)

            debug_episode_summary = os.getenv('DEBUG_EPISODE_SUMMARY', '1').lower() in ('1', 'true', 'yes', 'on')
            if debug_episode_summary:
                reach_info = []
                safe_info = []
                for i in range(len(summary["agent_success_flags"])):
                    reach_i = summary["agent_reach_flags"][i] if i < len(summary["agent_reach_flags"]) else 0
                    safe_i = summary["agent_safe_flags"][i] if i < len(summary["agent_safe_flags"]) else 0
                    reach_info.append(f"Agent{i}:{'✓' if reach_i else '✗'}")
                    safe_info.append(f"Agent{i}:{'✓' if safe_i else '✗'}")
                reach_str = ", ".join(reach_info)
                safe_str = ", ".join(safe_info)
                _progress_write(
                    f"[成功记录] 回合{episode+1}: 团队成功={bool(summary['team_success_flag'])} "
                    f"(Reach: {reach_str}, Safe: {safe_str}), "
                    f"智能体成功={summary['agent_success_flags']}, total_success_count={sum(team_success_flags)}"
                )
                all_reached = bool(summary["agent_reach_flags"]) and all(int(v) == 1 for v in summary["agent_reach_flags"])
                all_safe = bool(summary["agent_safe_flags"]) and all(int(v) == 1 for v in summary["agent_safe_flags"])
                if not bool(summary["team_success_flag"]):
                    if all_reached and not all_safe:
                        _progress_write("  [说明] 所有智能体都到达目标但未成功（原因：有碰撞），成功定义=到达目标∧无碰撞")
                    elif not all_reached and all_safe:
                        _progress_write("  [说明] 所有智能体都无碰撞但未成功（原因：未到达目标），成功定义=到达目标∧无碰撞")
                    elif not all_reached and not all_safe:
                        _progress_write("  [说明] 部分智能体未到达目标或有碰撞，成功定义=到达目标∧无碰撞")
                    if not all_reached:
                        dist_parts = []
                        unreached_distances = []
                        goal_distances = summary.get("agent_goal_distances", []) or []
                        for i in range(len(summary["agent_success_flags"])):
                            reach_i = summary["agent_reach_flags"][i] if i < len(summary["agent_reach_flags"]) else 0
                            dist_val = goal_distances[i] if i < len(goal_distances) else None
                            if dist_val is None or not np.isfinite(dist_val):
                                dist_repr = "NA"
                            else:
                                dist_repr = f"{float(dist_val):.2f}m"
                                if not reach_i:
                                    unreached_distances.append(float(dist_val))
                            suffix = "" if reach_i else "(未到达)"
                            dist_parts.append(f"Agent{i}={dist_repr}{suffix}")
                        if unreached_distances:
                            _progress_write(
                                f"[目标距离] 回合 {episode+1}: 各智能体终点距目标=[{', '.join(dist_parts)}], "
                                f"未到达智能体距离统计=min={min(unreached_distances):.2f}m, "
                                f"mean={float(np.mean(unreached_distances)):.2f}m, "
                                f"max={max(unreached_distances):.2f}m"
                            )

            debug_collision_summary = os.getenv('DEBUG_COLLISION_SUMMARY', '1').lower() in ('1', 'true', 'yes', 'on')
            if debug_collision_summary and ((episode + 1) % 10 == 0 or int(summary["total_collisions"]) > 0):
                _progress_write(
                    f"[碰撞统计] 回合 {episode+1}: 总碰撞次数={int(summary['total_collisions'])} "
                    f"(地形={int(summary['terrain_collision_total'])}, 球形障碍={int(summary['obstacle_collision_total'])}), "
                    f"各智能体碰撞次数={summary['agent_collision_counts']}, "
                    f"本回合成功={bool(summary['success_flag'])}, 累计成功={sum(team_success_flags)}"
                )

            trainer._adaptive_learning_check(avg_episode_reward)
            try:
                current_entropy_coef = float(getattr(trainer, "entropy_coef", getattr(args, "entropy_coef", 0.01)))
            except Exception:
                current_entropy_coef = float(getattr(args, "entropy_coef", 0.01))
            entropy_coef_history.append(current_entropy_coef)
            trainer._entropy_coef_history = list(entropy_coef_history)

            if episode >= 10:
                recent_rewards = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
                try:
                    is_stag = bool(trainer._detect_reward_stagnation(recent_rewards))
                except Exception:
                    is_stag = False
                prev = bool(getattr(trainer, "_stagnation_prev", False))
                trainer._reward_stagnation_detected = is_stag
                if is_stag != prev:
                    if is_stag:
                        _progress_write(f"⚠️ 检测到奖励停滞，已提升MAPPO探索强度与自适应学习率 (回合 {episode+1})")
                    else:
                        _progress_write(f"✅ 奖励恢复波动，MAPPO自适应探索恢复正常 (回合 {episode+1})")
                trainer._stagnation_prev = is_stag

            if avg_episode_reward > best_reward:
                best_reward = avg_episode_reward
                best_episode = episode
                if hasattr(trainer, "_reward_stagnation_detected"):
                    trainer._reward_stagnation_detected = False
                if args.save_model:
                    trainer.save_models(os.path.join(model_root, "best"))

            current_team_success_rate = (
                float(sum(team_success_flags) / len(team_success_flags)) if team_success_flags else 0.0
            )
            current_team_metric = (current_team_success_rate, float(avg_episode_reward))
            best_team_metric = (
                float(best_team_success_rate),
                float(best_team_sr_reward) if np.isfinite(best_team_sr_reward) else -np.inf,
            )
            if current_team_metric > best_team_metric:
                best_team_success_rate = current_team_success_rate
                best_team_sr_episode = episode
                best_team_sr_force_ratio = current_fr
                best_team_sr_reward = avg_episode_reward
                if args.save_model:
                    _save_checkpoint(
                        trainer,
                        os.path.join(model_root, "best_by_team_sr"),
                        episode,
                        episode_rewards,
                        episode_force_ratios,
                        best_reward,
                        best_episode,
                        best_team_success_rate,
                        best_team_sr_episode,
                        best_team_sr_force_ratio,
                        best_team_sr_reward,
                        success_flags,
                        agent_success_flags,
                        team_success_flags,
                        actor_losses_history,
                        critic_losses_history,
                    )

            if (
                args.save_model
                and int(args.save_interval) > 0
                and ((episode + 1) % int(args.save_interval) == 0)
            ):
                trainer.save_models(os.path.join(model_root, f"ep{episode + 1}"))
                if _env_flag_enabled("SAVE_TRAINING_RESUME_STATE", default=True):
                    _save_checkpoint(
                        trainer,
                        os.path.join(model_root, "checkpoint"),
                        episode + 1,
                        episode_rewards,
                        episode_force_ratios,
                        best_reward,
                        best_episode,
                        best_team_success_rate,
                        best_team_sr_episode,
                        best_team_sr_force_ratio,
                        best_team_sr_reward,
                        success_flags,
                        agent_success_flags,
                        team_success_flags,
                        actor_losses_history,
                        critic_losses_history,
                    )

            episode_time = time.time() - episode_start_time
            postfix_str = (
                f"R_ep={avg_episode_reward:,.0f} | "
                f"Best_R={best_reward:,.0f} | "
                f"SR={current_team_success_rate:.3f} | "
                f"FR={current_fr:.2f} | "
                f"Steps={step_count} | H={current_entropy_coef:.4f}"
            )
            if last_policy_loss is not None and last_value_loss is not None:
                postfix_str += f" | Pi={last_policy_loss:.2e} | V={last_value_loss:.2e}"
            elif last_actor_loss is not None and last_critic_loss is not None:
                postfix_str += f" | Loss a={last_actor_loss:.2e} c={last_critic_loss:.2e}"
            if hasattr(progress_bar, "set_postfix_str"):
                progress_bar.set_postfix_str(postfix_str)
            if hasattr(progress_bar, "update"):
                progress_bar.update(1)

            summary_line = (
                f"回合 {episode+1}/{args.train_episodes}: "
                f"奖励={avg_episode_reward:,.0f} | "
                f"最佳(回合 {best_episode+1})={best_reward:,.0f} | "
                f"Team_SR={current_team_success_rate:.3f} | "
                f"FR={current_fr:.2f} | "
                f"用时={episode_time:.1f}s"
            )
            _progress_write(summary_line)
            if timing_enabled and timing_ep is not None:
                step_denom = max(1, step_count)
                timing_line = (
                    f"[TIMING] ep={episode+1} | "
                    f"collect={timing_ep['collect'] * 1000.0 / step_denom:.1f}ms/step | "
                    f"env={timing_ep['env'] * 1000.0 / step_denom:.1f}ms/step | "
                    f"obs={timing_ep['obs'] * 1000.0 / step_denom:.1f}ms/step | "
                    f"boot_pf={timing_ep['bootstrap_pf'] * 1000.0:.1f}ms | "
                    f"boot_v={timing_ep['bootstrap_value'] * 1000.0:.1f}ms | "
                    f"update={timing_ep['update'] * 1000.0:.1f}ms"
                )
                _progress_write(timing_line)
            if not quiet_output:
                print(
                    f"[MAPPO] ep={episode + 1}/{args.train_episodes} reward={avg_episode_reward:.1f} "
                    f"team_sr={current_team_success_rate:.3f} fr={current_fr:.3f} steps={step_count}"
                )
    finally:
        if use_light_progress and hasattr(progress_bar, "close"):
            progress_bar.close()

    final_dir = os.path.join(model_root, "final")
    if args.save_model:
        trainer.save_models(final_dir)
        if _env_flag_enabled("SAVE_TRAINING_RESUME_STATE", default=True):
            _save_checkpoint(
                trainer,
                os.path.join(model_root, "checkpoint"),
                int(args.train_episodes),
                episode_rewards,
                episode_force_ratios,
                best_reward,
                best_episode,
                best_team_success_rate,
                best_team_sr_episode,
                best_team_sr_force_ratio,
                best_team_sr_reward,
                success_flags,
                agent_success_flags,
                team_success_flags,
                actor_losses_history,
                critic_losses_history,
            )

    num_episodes = len(agent_success_flags) if agent_success_flags else len(success_flags)
    agent_success_rates: List[float] = []
    if agent_success_flags and num_episodes > 0:
        max_agents = max(len(flags) for flags in agent_success_flags if flags) if agent_success_flags else 0
        for agent_idx in range(max_agents):
            success_count = sum(
                1
                for flags in agent_success_flags
                if len(flags) > agent_idx and _safe_int(flags[agent_idx], 0) == 1
            )
            agent_success_rates.append(float(success_count / max(1, num_episodes)))
    team_success_rate = float(sum(team_success_flags) / len(team_success_flags)) if team_success_flags else 0.0
    trainer._final_team_success_rate = float(team_success_rate)
    trainer._best_team_success_rate = float(best_team_success_rate)
    trainer._best_team_sr_episode = int(best_team_sr_episode)
    trainer._best_team_sr_force_ratio = float(best_team_sr_force_ratio)
    trainer._best_team_sr_reward = float(best_team_sr_reward) if np.isfinite(best_team_sr_reward) else None
    trainer._terrain_snapshot_artifacts = {}

    metrics_payload = {
        "episode_rewards": episode_rewards,
        "episode_force_ratios": episode_force_ratios,
        "collision_counts": collision_counts,
        "terrain_collision_counts": terrain_collision_counts,
        "obstacle_collision_counts": obstacle_collision_counts,
        "agent_collision_counts": agent_collision_counts,
        "min_distances_to_obstacle": min_distances,
        "entropy_coef_history": entropy_coef_history,
        "current_entropy_coef": float(getattr(trainer, "entropy_coef", getattr(args, "entropy_coef", 0.01))),
        "noise_scale_var_history": [],
        "success_flags": success_flags,
        "agent_success_flags": agent_success_flags,
        "team_success_flags": team_success_flags,
        "agent_success_rates": agent_success_rates,
        "team_success_rate": float(team_success_rate),
        "best_team_success_rate": float(best_team_success_rate),
        "best_team_sr_episode": int(best_team_sr_episode),
        "best_team_sr_force_ratio": float(best_team_sr_force_ratio),
        "best_team_sr_reward": float(best_team_sr_reward) if np.isfinite(best_team_sr_reward) else None,
        "train_episodes": int(args.train_episodes),
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    with open(os.path.join(run_dir, "episode_rewards.json"), "w", encoding="utf-8") as f:
        json.dump(_make_json_safe(metrics_payload), f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "loss_history.json"), "w", encoding="utf-8") as f:
        json.dump(_make_json_safe(loss_history), f, ensure_ascii=False, indent=2)

    training_environment = dict(args.training_environment_config)
    training_hyperparameters = dict(args.training_hyperparameters_config)
    results_args = dict(vars(args))
    results_args["algo"] = "mappo"
    results_args["algorithm"] = "mappo"
    results_args["mappo_use_separated_gradient"] = bool(getattr(args, "mappo_use_separated_gradient", False))
    results_args["scenario_name"] = str(getattr(args, "scenario", "paper3d_terrain_energy"))
    results_args["seed"] = int(resolved_seed)
    results_args["train_episodes"] = int(args.train_episodes)
    runtime_fallbacks = (
        ("simulation_dt", "SIMULATION_DT", float),
        ("z_action_bias", "Z_ACTION_BIAS", float),
        ("quadrotor_attitude_response_time", "QUADROTOR_ATTITUDE_RESPONSE_TIME", float),
        ("quadrotor_psi_cmd", "QUADROTOR_PSI_CMD", float),
        ("terrain_base_seed", "TERRAIN_BASE_SEED", int),
        ("peak_jitter_range", "PEAK_JITTER_RANGE", float),
        ("peak_center_jitter_range", "PEAK_CENTER_JITTER_RANGE", float),
        ("peak_height_jitter_ratio_min", "PEAK_HEIGHT_JITTER_RATIO_MIN", float),
        ("peak_height_jitter_ratio_max", "PEAK_HEIGHT_JITTER_RATIO_MAX", float),
        ("peak_height_max_scale", "PEAK_HEIGHT_MAX_SCALE", float),
        ("terrain_variant_noise_ratio", "TERRAIN_VARIANT_NOISE_RATIO", float),
        ("training_env_sequence_seed", "TRAIN_ENV_SEQUENCE_SEED", int),
        ("train_obstacle_sequence_mode", "TRAIN_OBSTACLE_SEQUENCE_MODE", str),
        ("train_obstacle_sequence_namespace", "TRAIN_OBSTACLE_SEQUENCE_NAMESPACE", str),
        ("semi_random_hold_episodes", "SEMI_RANDOM_TERRAIN_HOLD_EPISODES", int),
        ("semi_random_hold_min_episodes", "SEMI_RANDOM_TERRAIN_HOLD_MIN_EPISODES", int),
        ("semi_random_hold_max_episodes", "SEMI_RANDOM_TERRAIN_HOLD_MAX_EPISODES", int),
    )
    for arg_name, env_name, caster in runtime_fallbacks:
        if results_args.get(arg_name) is not None:
            continue
        env_value = os.getenv(env_name, "").strip()
        if not env_value:
            continue
        try:
            results_args[arg_name] = caster(env_value)
        except Exception:
            results_args[arg_name] = env_value

    results_args["terrain_seed"] = int(training_environment.get("terrain_seed", getattr(args, "terrain_seed", 67)))
    results_args["terrain_base_seed"] = int(training_environment.get("terrain_base_seed", results_args["terrain_seed"]))
    results_args["training_env_sequence_seed"] = int(
        training_environment.get("training_env_sequence_seed", results_args["terrain_base_seed"])
    )
    results_args["train_obstacle_sequence_mode"] = str(
        training_environment.get("train_obstacle_sequence_mode", results_args.get("train_obstacle_sequence_mode", "legacy_linear"))
    )
    results_args["train_obstacle_sequence_namespace"] = str(
        training_environment.get("train_obstacle_sequence_namespace", results_args.get("train_obstacle_sequence_namespace", "train_obstacle"))
    )
    results_args["semi_random_terrain"] = bool(training_environment.get("semi_random_terrain", False))
    results_args["deterministic_train_env_sequence"] = bool(training_environment.get("deterministic_env_sequence", False))
    results_args["random_terrain"] = bool(training_environment.get("random_terrain", False))
    results_args["use_dynamic_obstacles"] = bool(training_environment.get("use_dynamic_obstacles", False))
    if training_environment.get("semi_random_hold_mode") is not None:
        results_args["semi_random_hold_mode"] = str(training_environment.get("semi_random_hold_mode"))
    for key in (
        "peak_jitter_range",
        "peak_center_jitter_range",
        "peak_height_jitter_ratio_min",
        "peak_height_jitter_ratio_max",
        "peak_height_max_scale",
        "terrain_variant_noise_ratio",
        "semi_random_hold_episodes",
        "semi_random_hold_min_episodes",
        "semi_random_hold_max_episodes",
    ):
        if training_environment.get(key) is not None:
            results_args[key] = training_environment.get(key)
    results_args["use_quadrotor_dynamics"] = _env_flag_enabled(
        "USE_QUADROTOR_DYNAMICS",
        default=bool(getattr(args, "use_quadrotor_dynamics", False)),
    )

    results = {
        "episodes": len(episode_rewards),
        "rewards": episode_rewards,
        "algorithm": "mappo",
        "algo": "mappo",
        "scenario_name": str(getattr(args, "scenario", "paper3d_terrain_energy")),
        "episode_force_ratios": episode_force_ratios,
        "best_reward": float(max(episode_rewards)) if episode_rewards else 0.0,
        "best_episode": int(best_episode),
        "replay_buffer_size": 0,
        "actor_objective_mode": "ppo_clip_separated" if bool(getattr(args, "mappo_use_separated_gradient", False)) else "ppo_clip",
        "hybrid_actor_alpha": 0.0,
        "best_episode_force_ratio": (
            float(episode_force_ratios[best_episode]) if 0 <= best_episode < len(episode_force_ratios) else float(args.action_force_ratio)
        ),
        "last_episode_force_ratio": float(episode_force_ratios[-1]) if episode_force_ratios else float(args.action_force_ratio),
        "team_success_rate": float(team_success_rate),
        "best_team_success_rate": float(best_team_success_rate),
        "best_team_sr_episode": int(best_team_sr_episode),
        "best_team_sr_force_ratio": float(best_team_sr_force_ratio),
        "best_team_sr_reward": float(best_team_sr_reward) if np.isfinite(best_team_sr_reward) else None,
        "terrain_snapshot_artifacts": {},
        "training_environment_schema_version": int(training_environment.get("schema_version", 1)),
        "training_environment": training_environment,
        "training_hyperparameters": training_hyperparameters,
        "args": results_args,
    }
    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(_make_json_safe(results), f, ensure_ascii=False, indent=2)

    try:
        env.close()
    except Exception:
        pass

    return trainer, episode_rewards, run_dir, episode_force_ratios


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
