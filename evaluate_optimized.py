#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MADDPG优化版模型评估与可视化脚本
仿照1.0版本功能，支持模型加载、评估和可视化生成
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import traceback
import json
import time
import math
import shutil

# 设置环境变量抑制多智能体环境警告
os.environ['SUPPRESS_MA_PROMPT'] = '1'
# 🔧 抑制 TensorFlow/XLA 警告（如 cuFFT 注册警告）
# TF_CPP_MIN_LOG_LEVEL: 0=全部日志, 1=INFO及以上, 2=WARNING及以上, 3=ERROR及以上
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # 抑制警告，保留错误信息

# 可视化依赖（非交互后端）
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 导入优化的模块
from paper3d_train_optimized import (
    OptimizedMADDPG, 
    OptimizedMATD3,
    load_scenario_module, 
    configure_gpu, 
    try_apply_scenario_params, 
    build_continuous_action_network, 
    build_continuous_critic_network,
    build_continuous_critic_network_matd3,
    _save_vis_context_snapshot_artifacts,
)
try:
    from algorithms.mappo import OptimizedMAPPO
except ModuleNotFoundError:
    OptimizedMAPPO = None
from visualization.trajectory_visualizer import TrajectoryVisualizer
from utils.observation_processor import ObservationProcessor

# 导入环境
from multiagent.environment import MultiAgentEnv


def _finite_float_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _safe_mean(values):
    cleaned = [_finite_float_or_none(v) for v in values]
    cleaned = [v for v in cleaned if v is not None]
    if not cleaned:
        return None
    return float(np.mean(cleaned))


def _safe_std(values):
    cleaned = [_finite_float_or_none(v) for v in values]
    cleaned = [v for v in cleaned if v is not None]
    if not cleaned:
        return None
    return float(np.std(cleaned))


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _env_flag(name, default=False):
    try:
        return os.getenv(name, "1" if default else "0").lower() in ("1", "true", "yes", "on")
    except Exception:
        return bool(default)


def _env_int(name, default):
    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        return int(default)
    return value if value > 0 else int(default)


def _env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _episode_positions_filename(episode_idx, terrain_seed=None, terrain_variant_seed=None):
    episode_idx = int(episode_idx)
    terrain_seed = None if terrain_seed is None else int(terrain_seed)
    terrain_variant_seed = None if terrain_variant_seed is None else int(terrain_variant_seed)
    if terrain_variant_seed is not None:
        return f"episode_{episode_idx:03d}_seed_{terrain_seed}_variant_{terrain_variant_seed}.json"
    if terrain_seed is not None:
        return f"episode_{episode_idx:03d}_seed_{terrain_seed}.json"
    return f"episode_{episode_idx:03d}.json"


def _coerce_optional_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
    return bool(value)


def _coerce_optional_int(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def _coerce_optional_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _is_model_variant_dir_name(name):
    model_leaf = os.path.basename(os.path.normpath(name or ""))
    return model_leaf in (
        'final',
        'best',
        'best_by_team_sr',
        'best_by_strict_success',
        'checkpoint',
        'latest_ep',
    ) or model_leaf.startswith('ep')


def _sequence_implies_random_terrain(terrain_seed_sequence, terrain_variant_seed_sequence):
    if terrain_variant_seed_sequence:
        return True
    if not terrain_seed_sequence:
        return False
    try:
        normalized = [int(seed) for seed in terrain_seed_sequence if seed is not None]
    except Exception:
        normalized = [seed for seed in terrain_seed_sequence if seed is not None]
    return len(set(normalized)) > 1


def _iter_results_json_paths(model_path):
    if not model_path:
        return
    model_leaf = os.path.basename(os.path.normpath(model_path))
    if _is_model_variant_dir_name(model_leaf):
        model_base_dir = os.path.dirname(model_path)
    else:
        model_base_dir = model_path
    exp_name = os.path.basename(model_base_dir)
    potential_log_dirs = [
        os.path.join("logs", exp_name),
        model_base_dir,
        os.path.dirname(model_base_dir),
    ]
    seen = set()
    for log_dir in potential_log_dirs:
        if not os.path.isdir(log_dir):
            continue
        for root, dirs, files in os.walk(log_dir):
            if 'results.json' not in files:
                continue
            results_path = os.path.join(root, 'results.json')
            if results_path in seen:
                continue
            seen.add(results_path)
            yield results_path


def _load_training_alignment_snapshot(model_path):
    for results_path in _iter_results_json_paths(model_path):
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except Exception:
            continue

        training_args = results.get('args') if isinstance(results.get('args'), dict) else results
        if not isinstance(training_args, dict):
            continue

        scenario_name = (
            training_args.get('scenario_name')
            or training_args.get('scenario')
            or results.get('scenario_name')
            or results.get('scenario')
        )
        algorithm = (
            training_args.get('algorithm')
            or training_args.get('algo')
            or results.get('algorithm')
            or results.get('algo')
        )
        terrain_seed = training_args.get('terrain_seed')
        if terrain_seed is None:
            terrain_seed = training_args.get('scenario_seed', results.get('terrain_seed', results.get('scenario_seed')))

        snapshot = {
            'results_path': results_path,
            'scenario_name': scenario_name,
            'algorithm': algorithm,
            'random_terrain': _coerce_optional_bool(training_args.get('random_terrain', results.get('random_terrain'))),
            'use_dynamic_obstacles': _coerce_optional_bool(training_args.get('use_dynamic_obstacles', results.get('use_dynamic_obstacles'))),
            'terrain_seed': _coerce_optional_int(terrain_seed),
            'per_episode_terrain': _coerce_optional_bool(training_args.get('per_episode_terrain', results.get('per_episode_terrain'))),
            'per_env_terrain': _coerce_optional_bool(training_args.get('per_env_terrain', results.get('per_env_terrain'))),
            'semi_random_terrain': _coerce_optional_bool(training_args.get('semi_random_terrain', results.get('semi_random_terrain'))),
            'terrain_base_seed': _coerce_optional_int(training_args.get('terrain_base_seed', results.get('terrain_base_seed'))),
            'peak_jitter_range': _coerce_optional_float(training_args.get('peak_jitter_range', results.get('peak_jitter_range'))),
            'peak_center_jitter_range': _coerce_optional_float(training_args.get('peak_center_jitter_range', results.get('peak_center_jitter_range'))),
            'peak_height_jitter_ratio_min': _coerce_optional_float(training_args.get('peak_height_jitter_ratio_min', results.get('peak_height_jitter_ratio_min'))),
            'peak_height_jitter_ratio_max': _coerce_optional_float(training_args.get('peak_height_jitter_ratio_max', results.get('peak_height_jitter_ratio_max'))),
            'peak_height_max_scale': _coerce_optional_float(training_args.get('peak_height_max_scale', results.get('peak_height_max_scale'))),
            'terrain_variant_noise_ratio': _coerce_optional_float(training_args.get('terrain_variant_noise_ratio', results.get('terrain_variant_noise_ratio'))),
            'terrain_complexity_level': _coerce_optional_int(training_args.get('terrain_complexity_level', results.get('terrain_complexity_level'))),
            'map_size': _coerce_optional_float(training_args.get('map_size', results.get('map_size'))),
            'mountain_min_distance': _coerce_optional_float(training_args.get('mountain_min_distance', results.get('mountain_min_distance'))),
        }
        if snapshot['terrain_base_seed'] is None and snapshot['terrain_seed'] is not None:
            snapshot['terrain_base_seed'] = snapshot['terrain_seed']
        return snapshot
    return None


def _apply_training_alignment_to_args(args, snapshot, quiet=False):
    if args is None or not snapshot:
        return {}

    applied = {}

    def _apply(attr_name, value):
        if value is None:
            return
        setattr(args, attr_name, value)
        applied[attr_name] = value

    _apply('scenario_name', snapshot.get('scenario_name'))
    _apply('algorithm', snapshot.get('algorithm'))
    _apply('random_terrain', snapshot.get('random_terrain'))
    _apply('use_dynamic_obstacles', snapshot.get('use_dynamic_obstacles'))
    _apply('terrain_seed', snapshot.get('terrain_seed'))
    _apply('per_episode_terrain', snapshot.get('per_episode_terrain'))
    _apply('per_env_terrain', snapshot.get('per_env_terrain'))
    _apply('semi_random_terrain', snapshot.get('semi_random_terrain'))
    _apply('terrain_base_seed', snapshot.get('terrain_base_seed'))
    _apply('peak_jitter_range', snapshot.get('peak_jitter_range'))
    _apply('peak_center_jitter_range', snapshot.get('peak_center_jitter_range'))
    _apply('peak_height_jitter_ratio_min', snapshot.get('peak_height_jitter_ratio_min'))
    _apply('peak_height_jitter_ratio_max', snapshot.get('peak_height_jitter_ratio_max'))
    _apply('peak_height_max_scale', snapshot.get('peak_height_max_scale'))
    _apply('terrain_variant_noise_ratio', snapshot.get('terrain_variant_noise_ratio'))
    _apply('terrain_complexity_level', snapshot.get('terrain_complexity_level'))
    _apply('map_size', snapshot.get('map_size'))
    _apply('mountain_min_distance', snapshot.get('mountain_min_distance'))

    if applied and not quiet:
        pretty = ", ".join(f"{k}={applied[k]}" for k in sorted(applied.keys()))
        print(f"✅ 评估前按训练配置对齐环境参数: {pretty}")
        print(f"   来源: {snapshot.get('results_path')}")
    return applied


def _apply_runtime_env_overrides_from_args(args):
    """将依赖环境变量的隐藏运行时参数与args保持同步。"""
    runtime_pairs = (
        ("simulation_dt", "SIMULATION_DT"),
        ("z_action_bias", "Z_ACTION_BIAS"),
        ("quadrotor_attitude_response_time", "QUADROTOR_ATTITUDE_RESPONSE_TIME"),
        ("quadrotor_psi_cmd", "QUADROTOR_PSI_CMD"),
    )
    for attr_name, env_name in runtime_pairs:
        try:
            value = getattr(args, attr_name, None)
        except Exception:
            value = None
        if value is None:
            continue
        os.environ[env_name] = str(value)
    try:
        use_quadrotor_dynamics = getattr(args, "use_quadrotor_dynamics", None)
    except Exception:
        use_quadrotor_dynamics = None
    if use_quadrotor_dynamics is not None:
        os.environ["USE_QUADROTOR_DYNAMICS"] = "1" if bool(use_quadrotor_dynamics) else "0"

    terrain_bool_pairs = (
        ("use_dynamic_obstacles", "USE_DYNAMIC_OBSTACLES"),
        ("random_terrain", "RANDOM_TERRAIN"),
        ("per_episode_terrain", "PER_EPISODE_TERRAIN"),
        ("per_env_terrain", "PER_ENV_TERRAIN"),
        ("semi_random_terrain", "SEMI_RANDOM_TERRAIN"),
    )
    for attr_name, env_name in terrain_bool_pairs:
        try:
            value = getattr(args, attr_name, None)
        except Exception:
            value = None
        if value is None:
            continue
        os.environ[env_name] = "1" if bool(value) else "0"

    try:
        terrain_seed = getattr(args, "terrain_seed", None)
    except Exception:
        terrain_seed = None
    if terrain_seed is not None:
        os.environ["USE_SCENARIO_SEED"] = "1"
        os.environ["SCENARIO_SEED"] = str(int(terrain_seed))
        os.environ.setdefault("TERRAIN_BASE_SEED", str(int(terrain_seed)))

    try:
        terrain_base_seed = getattr(args, "terrain_base_seed", None)
    except Exception:
        terrain_base_seed = None
    if terrain_base_seed is not None:
        os.environ["TERRAIN_BASE_SEED"] = str(int(terrain_base_seed))

    terrain_numeric_env_pairs = (
        ("peak_jitter_range", "PEAK_JITTER_RANGE"),
        ("peak_center_jitter_range", "PEAK_CENTER_JITTER_RANGE"),
        ("peak_height_jitter_ratio_min", "PEAK_HEIGHT_JITTER_RATIO_MIN"),
        ("peak_height_jitter_ratio_max", "PEAK_HEIGHT_JITTER_RATIO_MAX"),
        ("peak_height_max_scale", "PEAK_HEIGHT_MAX_SCALE"),
        ("terrain_variant_noise_ratio", "TERRAIN_VARIANT_NOISE_RATIO"),
    )
    for attr_name, env_name in terrain_numeric_env_pairs:
        try:
            value = getattr(args, attr_name, None)
        except Exception:
            value = None
        if value is None:
            continue
        try:
            os.environ[env_name] = str(float(value))
        except Exception:
            continue

    numeric_env_pairs = (
        ("terrain_complexity_level", "TERRAIN_COMPLEXITY_LEVEL", int),
        ("map_size", "MAP_SIZE", float),
        ("mountain_min_distance", "MOUNTAIN_MIN_DISTANCE", float),
    )
    for attr_name, env_name, caster in numeric_env_pairs:
        try:
            value = getattr(args, attr_name, None)
        except Exception:
            value = None
        if value is None:
            continue
        try:
            os.environ[env_name] = str(caster(value))
        except Exception:
            continue


def _apply_terrain_runtime_params_to_scenario(scenario, world, args):
    """将地形关键参数显式下发到scenario/world，确保重建环境时真正生效。"""
    if scenario is None or args is None:
        return

    scalar_mappings = (
        ('use_dynamic_obstacles', bool, 'use_dynamic_obstacles'),
        ('random_terrain', bool, 'random_terrain'),
        ('per_episode_terrain', bool, 'per_episode_terrain'),
        ('per_env_terrain', bool, 'per_env_terrain'),
        ('semi_random_terrain', bool, 'use_semi_random_terrain'),
        ('peak_jitter_range', float, 'peak_jitter_range'),
        ('peak_center_jitter_range', float, 'peak_center_jitter_range'),
        ('peak_height_jitter_ratio_min', float, 'peak_height_jitter_ratio_min'),
        ('peak_height_jitter_ratio_max', float, 'peak_height_jitter_ratio_max'),
        ('peak_height_max_scale', float, 'peak_height_max_scale'),
        ('terrain_variant_noise_ratio', float, 'terrain_variant_noise_ratio'),
        ('terrain_complexity_level', int, 'terrain_complexity_level'),
        ('map_size', float, 'map_size'),
    )
    for arg_name, caster, scenario_attr in scalar_mappings:
        try:
            value = getattr(args, arg_name, None)
        except Exception:
            value = None
        if value is None:
            continue
        try:
            setattr(scenario, scenario_attr, caster(value))
        except Exception:
            continue

    try:
        terrain_seed = getattr(args, 'terrain_seed', None)
    except Exception:
        terrain_seed = None
    if terrain_seed is not None:
        try:
            terrain_seed_int = int(terrain_seed)
            scenario.seed = terrain_seed_int
            setattr(scenario, 'terrain_seed', terrain_seed_int)
            setattr(scenario, 'current_terrain_seed', terrain_seed_int)
            try:
                scenario.rng = np.random.RandomState(terrain_seed_int)
            except Exception:
                pass
        except Exception:
            pass

    try:
        terrain_base_seed = getattr(args, 'terrain_base_seed', None)
    except Exception:
        terrain_base_seed = None
    if terrain_base_seed is not None:
        try:
            scenario.terrain_base_seed = int(terrain_base_seed)
        except Exception:
            pass
    elif terrain_seed is not None:
        try:
            scenario.terrain_base_seed = int(terrain_seed)
        except Exception:
            pass

    try:
        terrain_variant_seed = getattr(args, 'terrain_variant_seed', None)
    except Exception:
        terrain_variant_seed = None
    if terrain_variant_seed is not None:
        try:
            scenario.terrain_variant_seed = int(terrain_variant_seed)
            setattr(scenario, 'current_terrain_variant_seed', int(terrain_variant_seed))
        except Exception:
            pass

    if world is None:
        return
    if terrain_seed is not None and hasattr(world, 'terrain_seed'):
        try:
            world.terrain_seed = int(terrain_seed)
        except Exception:
            pass
    try:
        map_size = getattr(args, 'map_size', None)
    except Exception:
        map_size = None
    if map_size is not None and hasattr(world, 'map_size'):
        try:
            world.map_size = float(map_size)
        except Exception:
            pass


def _apply_hidden_runtime_params_to_world(world, args, quiet_output=False):
    """将训练期的隐藏运行时参数显式下发到world，避免只改日志不改仿真。"""
    if world is None:
        return
    applied = []
    mapping = (
        ("simulation_dt", "dt"),
        ("z_action_bias", "z_action_bias"),
        ("quadrotor_attitude_response_time", "quadrotor_attitude_response_time"),
        ("quadrotor_psi_cmd", "quadrotor_psi_cmd"),
    )
    for arg_name, world_attr in mapping:
        try:
            value = getattr(args, arg_name, None)
        except Exception:
            value = None
        if value is None or not hasattr(world, world_attr):
            continue
        try:
            setattr(world, world_attr, float(value))
            applied.append((world_attr, float(value)))
        except Exception:
            continue
    if applied and not quiet_output:
        pretty = ", ".join(f"{name}={value}" for name, value in applied)
        print(f"✅ 已同步隐藏运行时参数到world: {pretty}")


def _normalize_vec3(vec):
    if vec is None:
        return None
    try:
        arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size < 3:
        return None
    arr = arr[:3]
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _distance_3d(a, b):
    va = _normalize_vec3(a)
    vb = _normalize_vec3(b)
    if va is None or vb is None:
        return None
    return float(np.linalg.norm(va - vb))


def _capture_agent_positions(agents):
    positions = []
    for agent in agents:
        pos = _normalize_vec3(getattr(getattr(agent, "state", None), "p_pos", None))
        positions.append(pos)
    return positions


def _extract_agent_goal_positions(world, scenario):
    goal_positions = []
    scenario_goal = getattr(scenario, "goal_pos", None)
    for agent in getattr(world, "agents", []):
        agent_goal = None
        try:
            if hasattr(agent, "goal_a") and hasattr(agent.goal_a, "state"):
                agent_goal = getattr(agent.goal_a.state, "p_pos", None)
        except Exception:
            agent_goal = None
        goal_positions.append(_normalize_vec3(agent_goal if agent_goal is not None else scenario_goal))
    return goal_positions


def _build_evaluation_summary(all_rewards, all_episodes_data, collision_distance_threshold=None):
    summary = {
        "episodes": int(len(all_episodes_data)),
        "avg_reward": _safe_mean(all_rewards),
        "std_reward": _safe_std(all_rewards),
        "max_reward": _finite_float_or_none(np.max(all_rewards)) if len(all_rewards) > 0 else None,
        "min_reward": _finite_float_or_none(np.min(all_rewards)) if len(all_rewards) > 0 else None,
        "success_episode_count": None,
        "team_success_rate": None,
        "agent_success_rates": [],
        "avg_steps": None,
        "std_steps": None,
        "avg_collision_count": None,
        "std_collision_count": None,
        "collision_free_rate": None,
        "avg_min_clearance_mean": None,
        "avg_min_clearance_min": None,
        "clearance_violation_rate": None,
        "clearance_violation_threshold": _finite_float_or_none(collision_distance_threshold),
        "avg_arrival_step_success_only": None,
        "avg_arrival_time_success_only": None,
        "avg_first_reach_step": None,
        "avg_first_reach_time": None,
        "avg_team_final_goal_distance": None,
        "avg_agent_final_goal_distance": None,
        "avg_team_min_goal_distance": None,
        "avg_agent_min_goal_distance": None,
        "avg_team_total_path_length": None,
        "avg_team_total_path_length_success_only": None,
        "std_team_total_path_length": None,
        "avg_agent_path_length": None,
        "avg_agent_path_length_success_only": None,
        "avg_team_path_efficiency": None,
        "avg_team_path_efficiency_success_only": None,
        "avg_agent_path_efficiency": None,
        "avg_agent_path_efficiency_success_only": None,
        "avg_penetration_count": None,
        "penetration_episode_rate": None,
        "max_penetration_depth": None,
        "mean_penetration_depth": None,
    }

    if not all_episodes_data:
        return summary

    success_flags = [int(ep.get("success", 0)) for ep in all_episodes_data]
    team_success_flags = [int(ep.get("team_success", ep.get("success", 0))) for ep in all_episodes_data]
    step_values = [ep.get("steps") for ep in all_episodes_data]
    collision_counts = [ep.get("collision_count", 0) for ep in all_episodes_data]
    team_path_lengths = [ep.get("path_length") for ep in all_episodes_data]
    team_path_lengths_success = [ep.get("path_length") for ep in all_episodes_data if ep.get("success", 0)]
    team_path_efficiencies = [ep.get("path_efficiency") for ep in all_episodes_data]
    team_path_efficiencies_success = [ep.get("path_efficiency") for ep in all_episodes_data if ep.get("success", 0)]
    first_reach_steps = [ep.get("first_reach_step") for ep in all_episodes_data]
    first_reach_times = [ep.get("first_reach_time") for ep in all_episodes_data]
    team_final_goal_distances = [ep.get("final_goal_distance") for ep in all_episodes_data]
    team_min_goal_distances = [ep.get("min_goal_distance") for ep in all_episodes_data]
    arrival_steps_success = [ep.get("arrival_step") for ep in all_episodes_data if ep.get("success", 0)]
    arrival_times_success = [ep.get("arrival_time") for ep in all_episodes_data if ep.get("success", 0)]

    min_distance_means = []
    min_distance_mins = []
    penetration_counts = []
    penetration_depths = []
    collision_free_count = 0
    violation_count = 0

    agent_success_lists = []
    agent_path_length_lists = []
    agent_path_length_success_lists = []
    agent_path_efficiency_lists = []
    agent_path_efficiency_success_lists = []
    agent_final_goal_distance_lists = []
    agent_min_goal_distance_lists = []

    for ep in all_episodes_data:
        min_distance = ep.get("min_distance")
        if isinstance(min_distance, dict):
            mean_clearance = _finite_float_or_none(min_distance.get("mean"))
            min_clearance = _finite_float_or_none(min_distance.get("min"))
            if mean_clearance is not None:
                min_distance_means.append(mean_clearance)
            if min_clearance is not None:
                min_distance_mins.append(min_clearance)
                if collision_distance_threshold is not None and min_clearance <= float(collision_distance_threshold):
                    violation_count += 1

        collision_count = ep.get("collision_count", 0)
        try:
            if int(collision_count) <= 0:
                collision_free_count += 1
        except Exception:
            pass

        penetration_stat = ep.get("penetration_stat")
        if isinstance(penetration_stat, dict):
            count = penetration_stat.get("count", 0)
            try:
                count = int(count)
            except Exception:
                count = 0
            penetration_counts.append(count)
            if count > 0:
                depth = _finite_float_or_none(penetration_stat.get("max_depth"))
                if depth is not None:
                    penetration_depths.append(depth)
        else:
            penetration_counts.append(0)

        agent_success_flags = ep.get("agent_success_flags", [])
        if isinstance(agent_success_flags, list):
            for idx, flag in enumerate(agent_success_flags):
                while len(agent_success_lists) <= idx:
                    agent_success_lists.append([])
                try:
                    agent_success_lists[idx].append(int(flag))
                except Exception:
                    agent_success_lists[idx].append(0)

        agent_path_lengths = ep.get("agent_path_lengths", [])
        if isinstance(agent_path_lengths, list):
            for idx, value in enumerate(agent_path_lengths):
                while len(agent_path_length_lists) <= idx:
                    agent_path_length_lists.append([])
                while len(agent_path_length_success_lists) <= idx:
                    agent_path_length_success_lists.append([])
                numeric = _finite_float_or_none(value)
                if numeric is not None:
                    agent_path_length_lists[idx].append(numeric)
                    if ep.get("success", 0):
                        agent_path_length_success_lists[idx].append(numeric)

        agent_path_efficiencies = ep.get("agent_path_efficiencies", [])
        if isinstance(agent_path_efficiencies, list):
            for idx, value in enumerate(agent_path_efficiencies):
                while len(agent_path_efficiency_lists) <= idx:
                    agent_path_efficiency_lists.append([])
                while len(agent_path_efficiency_success_lists) <= idx:
                    agent_path_efficiency_success_lists.append([])
                numeric = _finite_float_or_none(value)
                if numeric is not None:
                    agent_path_efficiency_lists[idx].append(numeric)
                    if ep.get("success", 0):
                        agent_path_efficiency_success_lists[idx].append(numeric)

        agent_final_goal_distances = ep.get("agent_final_goal_distances", [])
        if isinstance(agent_final_goal_distances, list):
            for idx, value in enumerate(agent_final_goal_distances):
                while len(agent_final_goal_distance_lists) <= idx:
                    agent_final_goal_distance_lists.append([])
                numeric = _finite_float_or_none(value)
                if numeric is not None:
                    agent_final_goal_distance_lists[idx].append(numeric)

        agent_min_goal_distances = ep.get("agent_min_goal_distances", [])
        if isinstance(agent_min_goal_distances, list):
            for idx, value in enumerate(agent_min_goal_distances):
                while len(agent_min_goal_distance_lists) <= idx:
                    agent_min_goal_distance_lists.append([])
                numeric = _finite_float_or_none(value)
                if numeric is not None:
                    agent_min_goal_distance_lists[idx].append(numeric)

    summary.update(
        {
            "success_episode_count": int(sum(success_flags)),
            "team_success_rate": _safe_mean(team_success_flags),
            "agent_success_rates": [_safe_mean(flags) for flags in agent_success_lists],
            "avg_steps": _safe_mean(step_values),
            "std_steps": _safe_std(step_values),
            "avg_collision_count": _safe_mean(collision_counts),
            "std_collision_count": _safe_std(collision_counts),
            "collision_free_rate": (
                float(collision_free_count / len(all_episodes_data)) if all_episodes_data else None
            ),
            "avg_min_clearance_mean": _safe_mean(min_distance_means),
            "avg_min_clearance_min": _safe_mean(min_distance_mins),
            "clearance_violation_rate": (
                float(violation_count / len(min_distance_mins)) if min_distance_mins else None
            ),
            "avg_arrival_step_success_only": _safe_mean(arrival_steps_success),
            "avg_arrival_time_success_only": _safe_mean(arrival_times_success),
            "avg_first_reach_step": _safe_mean(first_reach_steps),
            "avg_first_reach_time": _safe_mean(first_reach_times),
            "avg_team_final_goal_distance": _safe_mean(team_final_goal_distances),
            "avg_team_min_goal_distance": _safe_mean(team_min_goal_distances),
            "avg_team_total_path_length": _safe_mean(team_path_lengths),
            "avg_team_total_path_length_success_only": _safe_mean(team_path_lengths_success),
            "std_team_total_path_length": _safe_std(team_path_lengths),
            "avg_team_path_efficiency": _safe_mean(team_path_efficiencies),
            "avg_team_path_efficiency_success_only": _safe_mean(team_path_efficiencies_success),
            "avg_penetration_count": _safe_mean(penetration_counts),
            "penetration_episode_rate": (
                float(sum(1 for count in penetration_counts if count > 0) / len(penetration_counts))
                if penetration_counts else None
            ),
            "max_penetration_depth": (
                float(np.max(penetration_depths)) if penetration_depths else None
            ),
            "mean_penetration_depth": _safe_mean(penetration_depths),
        }
    )

    flattened_path_lengths = [value for values in agent_path_length_lists for value in values]
    flattened_path_lengths_success = [value for values in agent_path_length_success_lists for value in values]
    flattened_path_efficiencies = [value for values in agent_path_efficiency_lists for value in values]
    flattened_path_efficiencies_success = [value for values in agent_path_efficiency_success_lists for value in values]
    flattened_final_goal_distances = [value for values in agent_final_goal_distance_lists for value in values]
    flattened_min_goal_distances = [value for values in agent_min_goal_distance_lists for value in values]
    summary["avg_agent_path_length"] = _safe_mean(flattened_path_lengths)
    summary["avg_agent_path_length_success_only"] = _safe_mean(flattened_path_lengths_success)
    summary["avg_agent_path_efficiency"] = _safe_mean(flattened_path_efficiencies)
    summary["avg_agent_path_efficiency_success_only"] = _safe_mean(flattened_path_efficiencies_success)
    summary["avg_agent_final_goal_distance"] = _safe_mean(flattened_final_goal_distances)
    summary["avg_agent_min_goal_distance"] = _safe_mean(flattened_min_goal_distances)

    return summary

class ModelEvaluator:
    """模型评估器，仿照1.0版本的评估逻辑"""
    
    def __init__(self, args):
        self.args = args
        self._current_episode_terrain_info = {}
        self.training_alignment = _load_training_alignment_snapshot(getattr(args, 'load_model_path', None))
        if self.training_alignment:
            _apply_training_alignment_to_args(self.args, self.training_alignment)
            _apply_runtime_env_overrides_from_args(self.args)
        self.setup_environment()
        self.setup_visualizer()
        
    def setup_environment(self):
        """初始化环境"""
        print("初始化评估环境...")
        
        # 配置GPU
        gpu_configured = configure_gpu()
        try:
            physical_gpus = tf.config.list_physical_devices('GPU')
        except Exception:
            physical_gpus = []
        try:
            logical_gpus = tf.config.list_logical_devices('GPU')
        except Exception:
            logical_gpus = []
        cuda_visible = os.getenv('CUDA_VISIBLE_DEVICES', '<unset>')
        print(
            "[Eval Device] "
            f"python={sys.executable} | "
            f"CUDA_VISIBLE_DEVICES={cuda_visible} | "
            f"physical_gpus={len(physical_gpus)} | "
            f"logical_gpus={len(logical_gpus)} | "
            f"configure_gpu={'ok' if gpu_configured else 'fallback_cpu'}"
        )
        if not physical_gpus:
            print("⚠️  TensorFlow 当前未检测到可用 GPU，评估会回退到 CPU。")
        
        # 根据场景名称选择场景，支持新的场景选择逻辑
        scenario_name = self.args.scenario_name
        print(f"使用场景: {scenario_name}")
        
        # 加载场景
        self.scenario = load_scenario_module(scenario_name, self.args)
        if self.scenario is None:
            raise RuntimeError(f"无法加载场景: {scenario_name}")
        _apply_runtime_env_overrides_from_args(self.args)
        _apply_terrain_runtime_params_to_scenario(self.scenario, None, self.args)
        self.world = self.scenario.make_world()
        _apply_terrain_runtime_params_to_scenario(self.scenario, self.world, self.args)
        # 应用重力、控制增益与奖励缩放（仅在显式提供时覆盖）
        try:
            if hasattr(self.world, 'gravity') and getattr(self.args, 'gravity', None) is not None:
                self.world.gravity = float(self.args.gravity)
                print(f"已设置评估环境重力: gravity={self.world.gravity}")
            if hasattr(self.world, 'control_accel_gain') and getattr(self.args, 'control_accel_gain', None) is not None:
                self.world.control_accel_gain = float(self.args.control_accel_gain)
                print(f"已设置控制加速度增益: control_accel_gain={self.world.control_accel_gain}")
            if hasattr(self.world, 'reward_pos_scale') and getattr(self.args, 'reward_pos_scale', None) is not None:
                self.world.reward_pos_scale = float(self.args.reward_pos_scale)
            if hasattr(self.world, 'reward_neg_scale') and getattr(self.args, 'reward_neg_scale', None) is not None:
                self.world.reward_neg_scale = float(self.args.reward_neg_scale)
            if hasattr(self.world, 'damping') and getattr(self.args, 'damping', None) is not None:
                self.world.damping = float(self.args.damping)
                print(f"已设置评估环境阻尼: damping={self.world.damping}")
        except Exception as _e:
            print(f"评估环境设置物理/奖励缩放失败: {_e}")

        # 应用智能体速度/加速度（若提供）
        try:
            if getattr(self.args, 'agent_max_speed', None) is not None or getattr(self.args, 'agent_accel', None) is not None:
                for ag in getattr(self.world, 'agents', []):
                    if getattr(self.args, 'agent_max_speed', None) is not None and hasattr(ag, 'max_speed'):
                        ag.max_speed = float(self.args.agent_max_speed)
                    if getattr(self.args, 'agent_accel', None) is not None and hasattr(ag, 'accel'):
                        ag.accel = float(self.args.agent_accel)
        except Exception as _e:
            print(f"评估环境应用速度/加速度失败: {_e}")

        try:
            _apply_hidden_runtime_params_to_world(self.world, self.args, quiet_output=False)
        except Exception as _e:
            print(f"评估环境应用隐藏运行时参数失败: {_e}")

        # 将可能影响避障/检测的参数尽量下发到场景/世界（若存在对应属性）
        try:
            try_apply_scenario_params(self.scenario, self.world, self.args, tqdm_file=None)
        except Exception:
            pass
        
        # 🚨 关键修复：确保碰撞检测参数被正确设置到场景对象
        # 即使try_apply_scenario_params没有处理这些参数，也要手动设置
        try:
            if hasattr(self.args, 'collision_distance_threshold') and self.args.collision_distance_threshold is not None:
                if hasattr(self.scenario, 'collision_distance_threshold'):
                    self.scenario.collision_distance_threshold = float(self.args.collision_distance_threshold)
                    print(f"✅ 已设置碰撞距离阈值: {self.scenario.collision_distance_threshold}")
            if hasattr(self.args, 'collision_penalty_value') and self.args.collision_penalty_value is not None:
                if hasattr(self.scenario, 'collision_penalty_value'):
                    self.scenario.collision_penalty_value = float(self.args.collision_penalty_value)
                    print(f"✅ 已设置碰撞惩罚值: {self.scenario.collision_penalty_value}")
        except Exception as e:
            print(f"⚠️  设置碰撞检测参数失败: {e}")
        
        # 创建环境
        self.env = MultiAgentEnv(
            self.world,
            self.scenario.reset_world,
            self.scenario.reward,
            self.scenario.observation,
            done_callback=getattr(self.scenario, 'is_done', None),
            info_callback=None,
            shared_viewer=False
        )
        try:
            if hasattr(self.env, 'world') and self.env.world is not None:
                self.env.world.episode_length = int(getattr(self.args, 'episode_length', 2200) or 2200)
                self.env.world.current_step = 0
        except Exception:
            pass
        # 应用动作范围映射（仅在显式提供任一轴时覆盖）
        try:
            ax = getattr(self.args, 'action_range_x', None)
            ay = getattr(self.args, 'action_range_y', None)
            az = getattr(self.args, 'action_range_z', None)
            if any(v is not None for v in (ax, ay, az)) and hasattr(self.env, 'world'):
                current = getattr(self.env.world, 'action_range', None)
                if isinstance(current, (list, tuple)) and len(current) >= 3:
                    new_range = [float(current[0]), float(current[1]), float(current[2])]
                else:
                    new_range = [1.0, 1.0, 1.0]
                if ax is not None:
                    new_range[0] = float(ax)
                if ay is not None:
                    new_range[1] = float(ay)
                if az is not None:
                    new_range[2] = float(az)
                self.env.world.action_range = new_range
        except Exception:
            pass
        
        # 获取环境信息
        self.n_agents = self.env.n
        base_obs_shapes = [self.env.observation_space[i].shape[0] for i in range(self.n_agents)]
        
        # 🔧 关键修复：从训练配置（results.json）中读取训练时使用的观测维度
        # 优先从results.json读取，确保与训练时完全一致
        training_obs_shapes = None
        training_use_pf = None
        if hasattr(self.args, 'load_model_path') and self.args.load_model_path:
            try:
                # 尝试从模型路径找到results.json
                model_path = self.args.load_model_path
                # 移除 final/best/epXXX 等子目录
                model_leaf = os.path.basename(os.path.normpath(model_path))
                if _is_model_variant_dir_name(model_leaf):
                    model_base_dir = os.path.dirname(model_path)
                else:
                    model_base_dir = model_path
                
                exp_name = os.path.basename(model_base_dir)
                potential_log_dirs = [
                    os.path.join("logs", exp_name),
                    model_base_dir,
                    os.path.dirname(model_base_dir),
                ]
                
                for log_dir in potential_log_dirs:
                    # 查找results.json（可能在子目录中）
                    results_files = []
                    if os.path.isdir(log_dir):
                        # 在log_dir及其子目录中查找results.json
                        for root, dirs, files in os.walk(log_dir):
                            if 'results.json' in files:
                                results_files.append(os.path.join(root, 'results.json'))
                    
                    for results_file in results_files:
                        try:
                            with open(results_file, 'r', encoding='utf-8') as f:
                                results = json.load(f)
                            
                            # 🔧 关键修复：从args字典中读取配置（results.json格式：{'args': {...}}）
                            training_args = None
                            if 'args' in results and isinstance(results['args'], dict):
                                training_args = results['args']
                            elif isinstance(results, dict) and 'base_obs_shapes' in results:
                                # 向后兼容：如果args不在顶层，尝试从顶层读取
                                training_args = results
                            
                            if training_args is not None:
                                # 读取训练时的配置
                                if 'base_obs_shapes' in training_args:
                                    training_obs_shapes = training_args['base_obs_shapes']
                                    print(f"✅ 从训练配置读取观测维度: {training_obs_shapes}")
                                
                                if 'use_pf_feature' in training_args:
                                    training_use_pf = bool(training_args['use_pf_feature'])
                                    print(f"✅ 从训练配置读取PF特征标志: {training_use_pf}")
                            
                            if training_obs_shapes is not None:
                                break
                        except Exception as e:
                            continue
                    
                    if training_obs_shapes is not None:
                        break
            except Exception as e:
                print(f"⚠️  读取训练配置失败: {e}")
        
        # 🚨 关键修复：观测维度应该与训练时完全一致
        # 训练时PF特征是作为独立输入传递的，不是追加到观测中
        # 因此obs_shapes应该保持基础观测维度（81维），而不是84维
        if training_obs_shapes is not None:
            # 直接使用训练时的观测维度（训练时PF特征是独立输入，观测维度不包含PF特征）
            self.obs_shapes = training_obs_shapes
            print(f"✅ 使用训练时的观测维度: {self.obs_shapes} (PF特征作为独立输入，不包含在观测维度中)")
        else:
            # 回退到当前配置：使用基础观测维度（不含PF特征）
            # 因为PF特征是作为独立输入传递的，不应该追加到观测维度中
            self.obs_shapes = base_obs_shapes
            print(f"ℹ️  未找到训练配置，使用基础观测维度: {self.obs_shapes} (PF特征将作为独立输入传递)")
        
        self.action_dims = [7] * self.n_agents
        
        print(f"环境初始化完成:")
        print(f"  - 智能体数量: {self.n_agents}")
        print(f"  - 观察空间维度: {self.obs_shapes}")
        print(f"  - 动作空间维度: {self.action_dims}")
        
    def setup_visualizer(self):
        """初始化可视化器"""
        self.visualizer = TrajectoryVisualizer() if not self.args.disable_visualization else None

    def _rebuild_environment(self):
        """在场景被重新生成后重建world/env，并重新应用关键运行时参数。"""
        _apply_runtime_env_overrides_from_args(self.args)
        _apply_terrain_runtime_params_to_scenario(self.scenario, None, self.args)
        self.world = self.scenario.make_world()
        _apply_terrain_runtime_params_to_scenario(self.scenario, self.world, self.args)

        try:
            if hasattr(self.world, 'gravity') and getattr(self.args, 'gravity', None) is not None:
                self.world.gravity = float(self.args.gravity)
            if hasattr(self.world, 'control_accel_gain') and getattr(self.args, 'control_accel_gain', None) is not None:
                self.world.control_accel_gain = float(self.args.control_accel_gain)
            if hasattr(self.world, 'reward_pos_scale') and getattr(self.args, 'reward_pos_scale', None) is not None:
                self.world.reward_pos_scale = float(self.args.reward_pos_scale)
            if hasattr(self.world, 'reward_neg_scale') and getattr(self.args, 'reward_neg_scale', None) is not None:
                self.world.reward_neg_scale = float(self.args.reward_neg_scale)
            if hasattr(self.world, 'damping') and getattr(self.args, 'damping', None) is not None:
                self.world.damping = float(self.args.damping)
        except Exception:
            pass

        try:
            quiet_output = os.getenv("QUIET_OUTPUT", "1").lower() in ("1", "true", "yes", "on")
        except Exception:
            quiet_output = False
        try:
            _apply_hidden_runtime_params_to_world(self.world, self.args, quiet_output=quiet_output)
        except Exception:
            pass

        try:
            if getattr(self.args, 'agent_max_speed', None) is not None or getattr(self.args, 'agent_accel', None) is not None:
                for ag in getattr(self.world, 'agents', []):
                    if getattr(self.args, 'agent_max_speed', None) is not None and hasattr(ag, 'max_speed'):
                        ag.max_speed = float(self.args.agent_max_speed)
                    if getattr(self.args, 'agent_accel', None) is not None and hasattr(ag, 'accel'):
                        ag.accel = float(self.args.agent_accel)
        except Exception:
            pass

        try:
            try_apply_scenario_params(self.scenario, self.world, self.args, tqdm_file=None)
        except Exception:
            pass

        self.env = MultiAgentEnv(
            self.world,
            reset_callback=self.scenario.reset_world,
            reward_callback=self.scenario.reward,
            observation_callback=self.scenario.observation,
            done_callback=getattr(self.scenario, 'is_done', None),
            info_callback=None,
            shared_viewer=False
        )
        try:
            if hasattr(self.env, 'world') and self.env.world is not None:
                self.env.world.episode_length = int(getattr(self.args, 'episode_length', 2200) or 2200)
                self.env.world.current_step = 0
        except Exception:
            pass

        try:
            ax = getattr(self.args, 'action_range_x', None)
            ay = getattr(self.args, 'action_range_y', None)
            az = getattr(self.args, 'action_range_z', None)
            if any(v is not None for v in (ax, ay, az)) and hasattr(self.env, 'world'):
                current = getattr(self.env.world, 'action_range', None)
                if isinstance(current, (list, tuple)) and len(current) >= 3:
                    new_range = [float(current[0]), float(current[1]), float(current[2])]
                else:
                    new_range = [1.0, 1.0, 1.0]
                if ax is not None:
                    new_range[0] = float(ax)
                if ay is not None:
                    new_range[1] = float(ay)
                if az is not None:
                    new_range[2] = float(az)
                self.env.world.action_range = new_range
        except Exception:
            pass

        try:
            self.world.scenario = self.scenario
            self.env.scenario = self.scenario
        except Exception:
            pass

        try:
            terrain_sensing_mode = getattr(self.args, 'terrain_sensing_mode', 'local')
            if terrain_sensing_mode.startswith('oracle') and hasattr(self, 'maddpg'):
                self.maddpg.scenario_ref = self.scenario
                self.maddpg.world_ref = self.world
        except Exception:
            pass

    def _load_episode_positions(self, episode_idx, terrain_seed=None, terrain_variant_seed=None):
        """如果存在按回合保存的位置文件，则为当前episode加载。"""
        episode_positions_dir = os.getenv('EPISODE_POSITIONS_DIR', None)
        if not episode_positions_dir:
            return
        require_episode_positions = os.getenv('EVAL_REQUIRE_EPISODE_POSITIONS', '0').lower() in ('1', 'true', 'yes', 'on')

        try:
            from pathlib import Path

            positions_dir = Path(episode_positions_dir)
            candidates = []
            if terrain_seed is not None and terrain_variant_seed is not None:
                candidates.append(
                    positions_dir / _episode_positions_filename(
                        episode_idx,
                        terrain_seed=terrain_seed,
                        terrain_variant_seed=terrain_variant_seed,
                    )
                )
            if terrain_seed is not None:
                candidates.append(positions_dir / _episode_positions_filename(episode_idx, terrain_seed=terrain_seed))
            candidates.append(positions_dir / f"episode_{episode_idx:03d}.json")

            positions_file = None
            for candidate in candidates:
                if candidate.exists():
                    positions_file = candidate
                    break

            if positions_file is None:
                if require_episode_positions:
                    raise FileNotFoundError(
                        f"Episode {episode_idx + 1} 共享位置文件不存在: "
                        f"{positions_dir / _episode_positions_filename(episode_idx, terrain_seed=terrain_seed, terrain_variant_seed=terrain_variant_seed) if terrain_seed is not None else positions_dir}"
                    )
                self.scenario.fixed_positions = None
                self.scenario.use_fixed_positions = False
                self.scenario.positions_initialized = False
                print(f"⚠️  Episode {episode_idx + 1} 位置文件不存在，将使用动态生成")
                return

            with open(positions_file, 'r', encoding='utf-8') as f:
                positions_data = json.load(f)

            if 'agents' in positions_data and 'goal' in positions_data:
                self.scenario.fixed_positions = {
                    'agents': positions_data['agents'],
                    'goal': positions_data['goal']
                }
                self.scenario.use_fixed_positions = True
                self.scenario.positions_initialized = True
                print(f"✅ 加载Episode {episode_idx + 1}固定位置: {positions_file}")
                print(f"   智能体数量: {len(positions_data['agents'])}")
                print(f"   目标位置: {positions_data['goal']}")
                if hasattr(self.scenario, 'validate_and_adjust_fixed_positions'):
                    self.scenario.validate_and_adjust_fixed_positions()
                    print("   🔧 已根据当前地形调整智能体Z坐标，确保在地形上方")
            else:
                self.scenario.fixed_positions = None
                self.scenario.use_fixed_positions = False
                self.scenario.positions_initialized = False
                print(f"⚠️  Episode {episode_idx + 1}位置文件格式错误，将使用动态生成")
        except Exception as e:
            if require_episode_positions:
                raise
            self.scenario.fixed_positions = None
            self.scenario.use_fixed_positions = False
            self.scenario.positions_initialized = False
            print(f"⚠️  加载Episode {episode_idx + 1}位置文件失败: {e}，将使用动态生成")

    def _prepare_episode_terrain(
        self,
        episode_idx,
        terrain_seed_sequence=None,
        terrain_variant_seed_sequence=None,
        obstacle_seed_sequence=None,
    ):
        """为当前评估回合显式准备地形，确保每回合地形可控且不会被reset再次改写。"""
        try:
            quiet_output = os.getenv("QUIET_OUTPUT", "1").lower() in ("1", "true", "yes", "on")
        except Exception:
            quiet_output = True
        use_random_terrain = (
            bool(getattr(self.args, 'random_terrain', False))
            or _sequence_implies_random_terrain(terrain_seed_sequence, terrain_variant_seed_sequence)
        )
        if not use_random_terrain:
            obstacle_seed = None
            if obstacle_seed_sequence and episode_idx < len(obstacle_seed_sequence):
                obstacle_seed = int(obstacle_seed_sequence[episode_idx])
            try:
                self.scenario.current_episode_index = int(episode_idx)
                self.scenario.current_episode_env_id = 0
                self.scenario.current_episode_obstacle_seed_override = (
                    int(obstacle_seed) if obstacle_seed is not None else None
                )
            except Exception:
                pass
            try:
                if hasattr(self.env, 'scenario'):
                    self.env.scenario.current_episode_index = int(episode_idx)
                    self.env.scenario.current_episode_env_id = 0
                    self.env.scenario.current_episode_obstacle_seed_override = (
                        int(obstacle_seed) if obstacle_seed is not None else None
                    )
            except Exception:
                pass
            try:
                self.scenario.random_terrain = False
            except Exception:
                pass
            terrain_seed = getattr(self.scenario, 'current_terrain_seed', getattr(self.scenario, 'seed', None))
            terrain_variant_seed = getattr(
                self.scenario,
                'current_terrain_variant_seed',
                getattr(self.scenario, 'terrain_variant_seed', None),
            )
            self._current_episode_terrain_info = {
                'terrain_seed': terrain_seed,
                'terrain_variant_seed': terrain_variant_seed,
                'obstacle_seed': int(obstacle_seed) if obstacle_seed is not None else None,
            }
            return dict(self._current_episode_terrain_info)

        if terrain_seed_sequence and episode_idx < len(terrain_seed_sequence):
            terrain_seed = int(terrain_seed_sequence[episode_idx])
            if not quiet_output:
                print(f"🔧 使用预定义地形种子: {terrain_seed} (回合 {episode_idx + 1})")
        else:
            terrain_seed = int(np.random.randint(0, 1000000))
            if not quiet_output:
                print(f"🎲 为回合 {episode_idx + 1} 生成新地形种子: {terrain_seed}")

        terrain_variant_seed = None
        if terrain_variant_seed_sequence and episode_idx < len(terrain_variant_seed_sequence):
            terrain_variant_seed = int(terrain_variant_seed_sequence[episode_idx])
            if not quiet_output:
                print(f"🔧 使用同源扰动种子: {terrain_variant_seed} (回合 {episode_idx + 1})")

        obstacle_seed = None
        if obstacle_seed_sequence and episode_idx < len(obstacle_seed_sequence):
            obstacle_seed = int(obstacle_seed_sequence[episode_idx])
            if not quiet_output:
                print(f"🔧 使用共享障碍种子: {obstacle_seed} (回合 {episode_idx + 1})")

        if hasattr(self.scenario, 'regenerate_terrain'):
            if terrain_variant_seed is not None:
                self.scenario.regenerate_terrain(new_seed=terrain_seed, variant_seed=terrain_variant_seed)
            else:
                self.scenario.regenerate_terrain(new_seed=terrain_seed)
        else:
            try:
                self.scenario.seed = terrain_seed
                if hasattr(self.scenario, 'rng'):
                    self.scenario.rng = np.random.RandomState(terrain_seed)
            except Exception:
                pass

        self._rebuild_environment()
        self._load_episode_positions(episode_idx, terrain_seed, terrain_variant_seed)

        try:
            self.scenario.current_episode_index = int(episode_idx)
            self.scenario.current_episode_env_id = 0
            self.scenario.current_episode_obstacle_seed_override = (
                int(obstacle_seed) if obstacle_seed is not None else None
            )
        except Exception:
            pass
        try:
            if hasattr(self.env, 'scenario'):
                self.env.scenario.current_episode_index = int(episode_idx)
                self.env.scenario.current_episode_env_id = 0
                self.env.scenario.current_episode_obstacle_seed_override = (
                    int(obstacle_seed) if obstacle_seed is not None else None
                )
        except Exception:
            pass

        # 当前回合的地图已经显式准备完毕，禁用reset时的再次随机改图。
        try:
            self.scenario.random_terrain = False
            if hasattr(self.env, 'scenario'):
                self.env.scenario.random_terrain = False
        except Exception:
            pass

        self._current_episode_terrain_info = {
            'terrain_seed': getattr(self.scenario, 'current_terrain_seed', terrain_seed),
            'terrain_variant_seed': getattr(
                self.scenario,
                'current_terrain_variant_seed',
                terrain_variant_seed,
            ),
            'obstacle_seed': int(obstacle_seed) if obstacle_seed is not None else None,
        }
        return dict(self._current_episode_terrain_info)

    def _capture_episode_vis_context(self, episode_idx):
        """捕获当前评估回合的冻结可视化上下文，避免后续 live env/scenario 污染。"""
        ctx = {
            'terrain': None,
            'map_size': None,
            'goal_pos': None,
            'agent_goals': [],
            'obstacles': [],
            'terrain_source': 'episode_snapshot',
            'scenario_seed': None,
            'world_seed': None,
            'terrain_seed': None,
            'terrain_variant_seed': None,
            'terrain_params': {},
            'base_mountain_centers': [],
            'actual_mountain_centers': [],
            'episode': int(episode_idx) + 1,
        }
        try:
            if hasattr(self.scenario, 'build_terrain_snapshot'):
                snapshot = self.scenario.build_terrain_snapshot()
            else:
                snapshot = None
            if isinstance(snapshot, dict):
                terrain = snapshot.get('terrain')
                ctx['terrain'] = np.asarray(terrain, dtype=np.float32).copy() if terrain is not None else None
                ctx['map_size'] = snapshot.get('map_size', ctx['map_size'])
                ctx['goal_pos'] = snapshot.get('goal_pos', ctx['goal_pos'])
                ctx['agent_goals'] = snapshot.get('agent_goals', ctx['agent_goals']) or []
                ctx['obstacles'] = snapshot.get('obstacles', ctx['obstacles']) or []
                ctx['terrain_source'] = snapshot.get('terrain_source', ctx['terrain_source'])
                ctx['terrain_seed'] = snapshot.get('terrain_seed', ctx['terrain_seed'])
                ctx['terrain_variant_seed'] = snapshot.get('terrain_variant_seed', ctx['terrain_variant_seed'])
                ctx['terrain_params'] = snapshot.get('terrain_params', ctx['terrain_params']) or {}
                ctx['base_mountain_centers'] = snapshot.get('base_mountain_centers', ctx['base_mountain_centers']) or []
                ctx['actual_mountain_centers'] = snapshot.get('actual_mountain_centers', ctx['actual_mountain_centers']) or []
        except Exception:
            pass

        try:
            ctx['scenario_seed'] = getattr(self.scenario, 'seed', ctx['terrain_seed'])
            ctx['world_seed'] = getattr(self.world, 'terrain_seed', ctx['terrain_seed'])
        except Exception:
            pass

        if ctx.get('goal_pos') is None:
            try:
                if hasattr(self.scenario, 'goal_pos') and self.scenario.goal_pos is not None:
                    ctx['goal_pos'] = np.asarray(self.scenario.goal_pos, dtype=np.float32).copy()
            except Exception:
                pass

        if not ctx.get('agent_goals'):
            try:
                ctx['agent_goals'] = _extract_agent_goal_positions(self.world, self.scenario)
            except Exception:
                ctx['agent_goals'] = []

        if not ctx.get('obstacles'):
            try:
                ctx['obstacles'] = list(getattr(self.scenario, 'obstacles', []) or [])
            except Exception:
                ctx['obstacles'] = []

        return ctx
    
    def select_actions_eval(self, processed_obs, use_fr=False, use_pf=False):
        """
        评估时的动作选择，兼容MADDPG和MATD3
        
        Args:
            processed_obs: (n_agents, obs_dim) 或 (batch_size, n_agents, obs_dim)
            use_fr: 是否使用填充率特征（作为单独输入）
            use_pf: 是否使用势场特征（作为单独输入，如果启用）
        
        Returns:
            actions: (n_agents, action_dim) 或 (batch_size, n_agents, action_dim) - TensorFlow tensor
        """
        # 🔧 性能优化：使用tf.function装饰器加速（与训练代码一致）
        return self._select_actions_eval_tf(processed_obs, use_fr, use_pf)

    def _process_observations_for_eval(self, obs_n):
        """评估侧优先复用训练主线的向量化观测预处理，避免逐智能体 Python 循环。"""
        processor = getattr(getattr(self, 'maddpg', None), 'obs_processor', None)
        if processor is None:
            return np.asarray(obs_n, dtype=np.float32)

        processed = None
        try:
            if hasattr(processor, 'batch_process_observations_vectorized'):
                processed = processor.batch_process_observations_vectorized(obs_n)
            elif hasattr(processor, 'batch_process_observations_parallel'):
                processed = processor.batch_process_observations_parallel(obs_n)
            else:
                processed = processor.batch_process_observations(obs_n)
        except Exception:
            processed = processor.batch_process_observations(obs_n)

        processed = np.asarray(processed, dtype=np.float32)
        if processed.ndim == 3 and processed.shape[0] == 1:
            processed = processed[0]
        return processed

    def _get_base_obs_dim(self):
        """返回训练时的基础观测维度（PF 作为独立输入时不计入 obs）。"""
        try:
            if getattr(self, 'obs_shapes', None):
                obs_dims = [int(dim) for dim in self.obs_shapes if dim is not None]
                if obs_dims:
                    return max(obs_dims)
        except Exception:
            pass
        return 81

    def _get_pf_feature_dim(self):
        """返回 PF 特征维度，默认与训练配置保持 3 维。"""
        try:
            maddpg_args = getattr(getattr(self, 'maddpg', None), 'args', None)
            if maddpg_args is not None:
                pf_dim = int(getattr(maddpg_args, 'pf_feature_dim', 3))
                if pf_dim > 0:
                    return pf_dim
        except Exception:
            pass
        try:
            pf_dim = int(getattr(self.args, 'pf_feature_dim', 3))
            if pf_dim > 0:
                return pf_dim
        except Exception:
            pass
        return 3

    def _build_eval_actor_obs(self, processed_obs, use_pf, use_tf_potential_field, action_force_ratio):
        """
        生成评估时喂给 Actor 的输入底座：
        - 基础观测始终保持训练时的 base obs 维度
        - 若启用 PF 特征，则复用训练期的 base PF 特征定义并追加到 obs 尾部，
          供 Actor 单独切片使用
        """
        processed_obs = np.asarray(processed_obs, dtype=np.float32)
        if processed_obs.ndim != 2:
            return processed_obs

        base_obs_dim = self._get_base_obs_dim()
        base_obs = processed_obs[:, :base_obs_dim] if processed_obs.shape[1] > base_obs_dim else processed_obs

        if not use_pf:
            return base_obs

        pf_feature_dim = self._get_pf_feature_dim()
        pf_features = np.zeros((base_obs.shape[0], pf_feature_dim), dtype=np.float32)

        if use_tf_potential_field and action_force_ratio > 0.0:
            try:
                pf_forces = None

                # 与训练主线 batch_select_actions_vectorized 保持一致：
                # pf_input 应来自当前状态的 base PF 特征，而不是临时拼接出的其他语义。
                if hasattr(self.maddpg, 'compute_base_pf_forces_batch_numpy'):
                    pf_force_batch = self.maddpg.compute_base_pf_forces_batch_numpy(
                        np.expand_dims(base_obs, axis=0),
                        float(action_force_ratio),
                    )
                    if isinstance(pf_force_batch, np.ndarray) and pf_force_batch.ndim == 3 and pf_force_batch.shape[0] == 1:
                        pf_forces = np.asarray(pf_force_batch[0], dtype=np.float32)

                # 兼容旧模型/旧实现：若缺少统一 helper，则回退到 dummy_action 路径。
                if pf_forces is None:
                    dummy_actions_tf = tf.zeros((base_obs.shape[0], 7), dtype=tf.float32)
                    base_obs_tf = tf.convert_to_tensor(base_obs, dtype=tf.float32)
                    _, pf_forces_tf = self.maddpg._apply_potential_field_correction(
                        dummy_actions_tf, base_obs_tf, action_force_ratio
                    )
                    pf_forces = np.asarray(pf_forces_tf.numpy(), dtype=np.float32)

                if pf_forces.ndim == 2 and pf_forces.shape[0] == base_obs.shape[0]:
                    copy_dim = min(pf_feature_dim, pf_forces.shape[1])
                    pf_features[:, :copy_dim] = pf_forces[:, :copy_dim]
            except Exception:
                pass

        return np.concatenate([base_obs, pf_features], axis=1)

    @tf.function(reduce_retracing=True)
    def _select_actions_eval_tf(self, processed_obs, use_fr, use_pf):
        """
        评估时的动作选择（TensorFlow图模式，性能优化）
        
        Args:
            processed_obs: TensorFlow tensor (n_agents, obs_dim) 或 (batch_size, n_agents, obs_dim)
            use_fr: bool 是否使用填充率特征
            use_pf: bool 是否使用势场特征
        
        Returns:
            actions: TensorFlow tensor (n_agents, action_dim) 或 (batch_size, n_agents, action_dim)
        """
        # 🔧 修复：Actor网络期望的输入结构
        # 如果 use_pf_feature=True，Actor期望3个输入：[obs, fr_input, pf_input]
        # 如果 use_fr_feature=True 但 use_pf_feature=False，Actor期望2个输入：[obs, fr_input]
        # 如果两者都False，Actor期望1个输入：[obs]
        
        # 确保是tensor
        processed_obs = tf.convert_to_tensor(processed_obs, dtype=tf.float32)
        
        # 确保有批次维度
        if len(processed_obs.shape) == 2:  # (n_agents, obs_dim)
            processed_obs = tf.expand_dims(processed_obs, axis=0)  # (1, n_agents, obs_dim)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = tf.shape(processed_obs)[0]
        actions_list = []
        
        # 🔧 获取PF特征维度（与训练配置保持一致）
        pf_feature_dim = self._get_pf_feature_dim()
        
        # 🔧 关键修复：计算真实的FR值（action_force_ratio），而不是使用零向量
        # 训练时FR值会随时间变化（schedule），评估时应该使用固定的FR值
        # 🔧 性能优化：在tf.function外部缓存FR值，避免每次调用时获取
        if not hasattr(self, '_cached_fr_value'):
            action_force_ratio = getattr(self.args, 'action_force_ratio', 0.0)
            self._cached_fr_value = float(action_force_ratio)
        fr_value = tf.cast(self._cached_fr_value, tf.float32)
        
        # 🔧 性能优化：批量处理所有智能体（与训练代码一致）
        for i in range(self.n_agents):
            # 提取当前智能体的观测（基础观测，不包含PF特征）
            obs_dim = self.obs_shapes[i] if i < len(self.obs_shapes) else processed_obs.shape[2]
            agent_obs = processed_obs[:, i, :obs_dim]  # (batch_size, obs_dim)
            
            # 构建输入：obs + fr_input（如果启用）+ pf_input（如果启用）
            actor_inputs = [agent_obs]
            if use_fr:
                # 🔧 关键修复：使用真实的FR值，而不是零向量
                fr_input = tf.fill([batch_size, 1], fr_value)  # (batch_size, 1)
                actor_inputs.append(fr_input)
            if use_pf:
                # 评估侧的 processed_obs 已由 _build_eval_actor_obs 追加真实 PF 特征。
                # 若上游未追加完整维度，则在这里补零而不是错误地改变 obs 结构。
                pf_input = processed_obs[:, i, obs_dim:obs_dim + pf_feature_dim]
                pf_dim = tf.shape(pf_input)[1]
                pad_dim = tf.maximum(0, pf_feature_dim - pf_dim)
                pf_input = tf.pad(pf_input, [[0, 0], [0, pad_dim]])
                pf_input = pf_input[:, :pf_feature_dim]
                actor_inputs.append(pf_input)
            
            # 调用actor
            if len(actor_inputs) == 1:
                agent_actions = self.maddpg.agents[i]['actor'](actor_inputs[0], training=False)
            else:
                agent_actions = self.maddpg.agents[i]['actor'](actor_inputs, training=False)
            
            actions_list.append(agent_actions)
        
        # 堆叠为 (batch_size, n_agents, action_dim)
        actions = tf.stack(actions_list, axis=1)
        
        # 如果输入没有批次维度，移除输出的批次维度
        if squeeze_output:
            actions = actions[0]  # (n_agents, action_dim)
        
        # 🔧 性能优化：返回tensor而不是numpy，延迟转换到env.step时
        return actions
        
    def load_model(self):
        """加载训练好的模型"""
        # 从模型路径提取实验名称
        # 例如：models/调试分离梯度、无重力、无早停、预热、随机地图、高变FR低高低_exp_20251201_112141/final
        # 提取：调试分离梯度、无重力、无早停、预热、随机地图、高变FR低高低_exp_20251201_112141
        model_path = self.args.load_model_path
        model_leaf_dir = os.path.basename(os.path.normpath(model_path))
        # 移除 final/best/epXXX 等子目录
        if _is_model_variant_dir_name(model_leaf_dir):
            model_base_dir = os.path.dirname(model_path)
        else:
            model_base_dir = model_path
        exp_name = os.path.basename(model_base_dir)
        potential_log_dirs = [
            os.path.join("logs", exp_name),
            model_base_dir,
            os.path.dirname(model_base_dir),
        ]
        
        # 🔧 关键修复：从训练配置（results.json）中读取训练时使用的特征标志、ACTION_FORCE_RATIO和动作范围
        # 优先从results.json读取，确保与训练时完全一致
        training_use_fr = None
        training_use_pf = None
        training_pf_feature_dim = None
        selected_force_ratio = None
        selected_force_ratio_label = None
        selected_force_ratio_episode = None
        episode_force_ratios = None
        training_action_range_x = None
        training_action_range_y = None
        training_action_range_z = None
        training_episode_length = None  # 🚨 新增：读取训练时的episode_length
        training_actor_hidden = None  # 🚨 新增：读取训练时的actor_hidden
        training_critic_hidden = None  # 🚨 新增：读取训练时的critic_hidden
        training_matd3_use_dual_q = None
        training_matd3_use_separated_gradient = None
        training_maddpg_use_dual_q = None
        training_maddpg_use_separated_gradient = None
        training_gravity = None  # 🔧 新增：读取训练时的gravity
        training_damping = None  # 🔧 新增：读取训练时的damping
        training_reward_pos_scale = None
        training_reward_neg_scale = None
        training_agent_max_speed = None  # 🔧 新增：读取训练时的agent_max_speed
        training_agent_accel = None  # 🔧 新增：读取训练时的agent_accel
        training_control_accel_gain = None  # 🔧 新增：读取训练时的control_accel_gain
        training_simulation_dt = None
        training_z_action_bias = None
        training_quadrotor_attitude_response_time = None
        training_quadrotor_psi_cmd = None
        training_use_quadrotor_dynamics = None
        for log_dir in potential_log_dirs:
            # 查找results.json（可能在子目录中）
            results_files = []
            if os.path.isdir(log_dir):
                # 在log_dir及其子目录中查找results.json
                for root, dirs, files in os.walk(log_dir):
                    if 'results.json' in files:
                        results_files.append(os.path.join(root, 'results.json'))
            
            for results_file in results_files:
                try:
                    with open(results_file, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                    
                    if isinstance(results.get('episode_force_ratios'), list):
                        try:
                            episode_force_ratios = [float(v) for v in results['episode_force_ratios']]
                        except Exception:
                            episode_force_ratios = None

                    # 关键：FR 必须与实际评估的权重变体一致。
                    # - final -> 最后回合 FR
                    # - best_by_team_sr -> Team SR 最佳回合 FR
                    # - best_by_strict_success -> 严格成功最佳回合 FR
                    # - best -> reward 最佳回合 FR
                    # - epXXX/latest_ep -> 对应回合 FR（找不到则回退到最后回合 FR）
                    if model_leaf_dir == 'best_by_team_sr' and 'best_team_sr_force_ratio' in results:
                        selected_force_ratio = float(results['best_team_sr_force_ratio'])
                        if 'best_team_sr_episode' in results:
                            selected_force_ratio_episode = int(results['best_team_sr_episode']) + 1
                        selected_force_ratio_label = 'Team SR 最佳回合'
                        print(
                            f"✅ 从训练配置读取 Team SR 最佳回合的FR值: "
                            f"{selected_force_ratio} (回合 {selected_force_ratio_episode if selected_force_ratio_episode is not None else '?'})"
                        )
                    elif model_leaf_dir == 'best_by_strict_success' and 'best_strict_episode_force_ratio' in results:
                        selected_force_ratio = float(results['best_strict_episode_force_ratio'])
                        if 'best_strict_episode' in results:
                            selected_force_ratio_episode = int(results['best_strict_episode']) + 1
                        selected_force_ratio_label = '严格评估最佳回合'
                        print(
                            f"✅ 从训练配置读取严格最佳回合的FR值: "
                            f"{selected_force_ratio} (回合 {selected_force_ratio_episode if selected_force_ratio_episode is not None else '?'})"
                        )
                    elif model_leaf_dir == 'final':
                        if 'last_episode_force_ratio' in results:
                            selected_force_ratio = float(results['last_episode_force_ratio'])
                        elif 'args' in results and isinstance(results['args'], dict) and 'action_force_ratio' in results['args']:
                            selected_force_ratio = float(results['args']['action_force_ratio'])
                        elif 'action_force_ratio' in results:
                            selected_force_ratio = float(results['action_force_ratio'])
                        selected_force_ratio_label = '最终回合'
                        print(f"✅ 从训练配置读取最终回合的FR值: {selected_force_ratio}")
                    elif model_leaf_dir == 'checkpoint':
                        if 'last_episode_force_ratio' in results:
                            selected_force_ratio = float(results['last_episode_force_ratio'])
                        elif episode_force_ratios is not None and len(episode_force_ratios) > 0:
                            selected_force_ratio = float(episode_force_ratios[-1])
                            selected_force_ratio_episode = len(episode_force_ratios)
                        elif 'args' in results and isinstance(results['args'], dict) and 'action_force_ratio' in results['args']:
                            selected_force_ratio = float(results['args']['action_force_ratio'])
                        elif 'action_force_ratio' in results:
                            selected_force_ratio = float(results['action_force_ratio'])
                        selected_force_ratio_label = '检查点最新回合'
                        print(
                            f"✅ 从训练配置读取检查点最新回合的FR值: "
                            f"{selected_force_ratio}"
                            f"{f' (回合 {selected_force_ratio_episode})' if selected_force_ratio_episode is not None else ''}"
                        )
                    elif model_leaf_dir == 'best' and 'best_episode_force_ratio' in results:
                        selected_force_ratio = float(results['best_episode_force_ratio'])
                        if 'best_episode' in results:
                            selected_force_ratio_episode = int(results['best_episode']) + 1
                        selected_force_ratio_label = '奖励最佳回合'
                        print(
                            f"✅ 从训练配置读取奖励最佳回合的FR值: "
                            f"{selected_force_ratio} (回合 {selected_force_ratio_episode if selected_force_ratio_episode is not None else '?'})"
                        )
                    elif model_leaf_dir.startswith('ep'):
                        ep_idx = None
                        try:
                            ep_idx = max(int(''.join(ch for ch in model_leaf_dir if ch.isdigit())) - 1, 0)
                        except Exception:
                            ep_idx = None
                        if ep_idx is not None and episode_force_ratios is not None and ep_idx < len(episode_force_ratios):
                            selected_force_ratio = float(episode_force_ratios[ep_idx])
                            selected_force_ratio_episode = ep_idx + 1
                            selected_force_ratio_label = '指定回合'
                            print(f"✅ 从训练配置读取第 {selected_force_ratio_episode} 回合的FR值: {selected_force_ratio}")
                        elif 'last_episode_force_ratio' in results:
                            selected_force_ratio = float(results['last_episode_force_ratio'])
                            selected_force_ratio_label = '最终回合(回退)'
                            print(f"⚠️  未找到 {model_leaf_dir} 对应FR，回退到最终回合FR: {selected_force_ratio}")
                    
                    # 读取训练时的特征标志
                    if 'args' in results and isinstance(results['args'], dict):
                        # 从args字典中读取
                        if 'use_fr_feature' in results['args']:
                            training_use_fr = bool(results['args']['use_fr_feature'])
                        if 'use_pf_feature' in results['args']:
                            training_use_pf = bool(results['args']['use_pf_feature'])
                        # 🔧 关键修复：读取训练时的pf_feature_dim
                        if 'pf_feature_dim' in results['args']:
                            training_pf_feature_dim = int(results['args']['pf_feature_dim'])
                        # 🔧 关键修复：读取训练时的动作范围参数
                        if 'action_range_x' in results['args']:
                            training_action_range_x = float(results['args']['action_range_x'])
                        if 'action_range_y' in results['args']:
                            training_action_range_y = float(results['args']['action_range_y'])
                        if 'action_range_z' in results['args']:
                            training_action_range_z = float(results['args']['action_range_z'])
                        # 🚨 新增：读取训练时的episode_length
                        if 'episode_length' in results['args']:
                            training_episode_length = int(results['args']['episode_length'])
                        # 🚨 新增：读取训练时的网络结构配置
                        if 'actor_hidden' in results['args']:
                            training_actor_hidden = str(results['args']['actor_hidden'])
                        if 'critic_hidden' in results['args']:
                            training_critic_hidden = str(results['args']['critic_hidden'])
                        if 'matd3_use_dual_q' in results['args']:
                            training_matd3_use_dual_q = bool(results['args']['matd3_use_dual_q'])
                        if 'matd3_use_separated_gradient' in results['args']:
                            training_matd3_use_separated_gradient = bool(results['args']['matd3_use_separated_gradient'])
                        if 'maddpg_use_dual_q' in results['args']:
                            training_maddpg_use_dual_q = bool(results['args']['maddpg_use_dual_q'])
                        if 'maddpg_use_separated_gradient' in results['args']:
                            training_maddpg_use_separated_gradient = bool(results['args']['maddpg_use_separated_gradient'])
                        # 🔧 新增：读取训练时的物理参数（如果值为None，保持None以便后续报错）
                        if 'gravity' in results['args']:
                            val = results['args']['gravity']
                            training_gravity = float(val) if val is not None else None
                        if 'damping' in results['args']:
                            val = results['args']['damping']
                            training_damping = float(val) if val is not None else None
                        if 'reward_pos_scale' in results['args']:
                            val = results['args']['reward_pos_scale']
                            training_reward_pos_scale = float(val) if val is not None else None
                        if 'reward_neg_scale' in results['args']:
                            val = results['args']['reward_neg_scale']
                            training_reward_neg_scale = float(val) if val is not None else None
                        if 'agent_max_speed' in results['args']:
                            val = results['args']['agent_max_speed']
                            training_agent_max_speed = float(val) if val is not None else None
                        if 'agent_accel' in results['args']:
                            val = results['args']['agent_accel']
                            training_agent_accel = float(val) if val is not None else None
                        if 'control_accel_gain' in results['args']:
                            val = results['args']['control_accel_gain']
                            training_control_accel_gain = float(val) if val is not None else None
                        if 'simulation_dt' in results['args']:
                            val = results['args']['simulation_dt']
                            training_simulation_dt = float(val) if val is not None else None
                        if 'z_action_bias' in results['args']:
                            val = results['args']['z_action_bias']
                            training_z_action_bias = float(val) if val is not None else None
                        if 'quadrotor_attitude_response_time' in results['args']:
                            val = results['args']['quadrotor_attitude_response_time']
                            training_quadrotor_attitude_response_time = float(val) if val is not None else None
                        if 'quadrotor_psi_cmd' in results['args']:
                            val = results['args']['quadrotor_psi_cmd']
                            training_quadrotor_psi_cmd = float(val) if val is not None else None
                        if 'use_quadrotor_dynamics' in results['args']:
                            val = results['args']['use_quadrotor_dynamics']
                            training_use_quadrotor_dynamics = bool(val) if val is not None else None
                    else:
                        # 从顶层读取（向后兼容）
                        if 'use_fr_feature' in results:
                            training_use_fr = bool(results['use_fr_feature'])
                        if 'use_pf_feature' in results:
                            training_use_pf = bool(results['use_pf_feature'])
                        if 'pf_feature_dim' in results:
                            training_pf_feature_dim = int(results['pf_feature_dim'])
                        # 🔧 关键修复：从顶层读取动作范围参数（向后兼容）
                        if 'action_range_x' in results:
                            training_action_range_x = float(results['action_range_x'])
                        if 'action_range_y' in results:
                            training_action_range_y = float(results['action_range_y'])
                        if 'action_range_z' in results:
                            training_action_range_z = float(results['action_range_z'])
                        # 🚨 新增：从顶层读取网络结构配置（向后兼容）
                        if 'actor_hidden' in results:
                            training_actor_hidden = str(results['actor_hidden'])
                        if 'critic_hidden' in results:
                            training_critic_hidden = str(results['critic_hidden'])
                        if 'matd3_use_dual_q' in results:
                            training_matd3_use_dual_q = bool(results['matd3_use_dual_q'])
                        if 'matd3_use_separated_gradient' in results:
                            training_matd3_use_separated_gradient = bool(results['matd3_use_separated_gradient'])
                        if 'maddpg_use_dual_q' in results:
                            training_maddpg_use_dual_q = bool(results['maddpg_use_dual_q'])
                        if 'maddpg_use_separated_gradient' in results:
                            training_maddpg_use_separated_gradient = bool(results['maddpg_use_separated_gradient'])
                        # 🔧 新增：从顶层读取物理参数（向后兼容，如果值为None，保持None以便后续报错）
                        if 'gravity' in results:
                            val = results['gravity']
                            training_gravity = float(val) if val is not None else None
                        if 'damping' in results:
                            val = results['damping']
                            training_damping = float(val) if val is not None else None
                        if 'reward_pos_scale' in results:
                            val = results['reward_pos_scale']
                            training_reward_pos_scale = float(val) if val is not None else None
                        if 'reward_neg_scale' in results:
                            val = results['reward_neg_scale']
                            training_reward_neg_scale = float(val) if val is not None else None
                        if 'agent_max_speed' in results:
                            val = results['agent_max_speed']
                            training_agent_max_speed = float(val) if val is not None else None
                        if 'agent_accel' in results:
                            val = results['agent_accel']
                            training_agent_accel = float(val) if val is not None else None
                        if 'control_accel_gain' in results:
                            val = results['control_accel_gain']
                            training_control_accel_gain = float(val) if val is not None else None
                        if 'simulation_dt' in results:
                            val = results['simulation_dt']
                            training_simulation_dt = float(val) if val is not None else None
                        if 'z_action_bias' in results:
                            val = results['z_action_bias']
                            training_z_action_bias = float(val) if val is not None else None
                        if 'quadrotor_attitude_response_time' in results:
                            val = results['quadrotor_attitude_response_time']
                            training_quadrotor_attitude_response_time = float(val) if val is not None else None
                        if 'quadrotor_psi_cmd' in results:
                            val = results['quadrotor_psi_cmd']
                            training_quadrotor_psi_cmd = float(val) if val is not None else None
                        if 'use_quadrotor_dynamics' in results:
                            val = results['use_quadrotor_dynamics']
                            training_use_quadrotor_dynamics = bool(val) if val is not None else None
                    
                    if training_use_fr is not None and training_use_pf is not None:
                        print(f"✅ 从训练配置读取特征标志: use_fr_feature={training_use_fr}, use_pf_feature={training_use_pf}")
                        if training_pf_feature_dim is not None:
                            print(f"✅ 从训练配置读取PF特征维度: {training_pf_feature_dim}")
                        # 🔧 关键修复：打印动作范围参数
                        if training_action_range_x is not None or training_action_range_y is not None or training_action_range_z is not None:
                            print(f"✅ 从训练配置读取动作范围: X={training_action_range_x}, Y={training_action_range_y}, Z={training_action_range_z}")
                        break
                except Exception as e:
                    continue
            
            if training_use_fr is not None and training_use_pf is not None:
                break
        
        # 使用训练时的配置（如果找到），否则使用当前配置
        use_fr_feature = training_use_fr if training_use_fr is not None else int(os.getenv('USE_FR_FEATURE', getattr(self.args, 'use_fr_feature', 1))) > 0
        use_pf_feature = training_use_pf if training_use_pf is not None else int(os.getenv('USE_PF_FEATURE', getattr(self.args, 'use_pf_feature', 1))) > 0
        
        forced_eval_action_force_ratio = getattr(self.args, 'force_action_force_ratio', None)
        if forced_eval_action_force_ratio is not None:
            eval_action_force_ratio = float(forced_eval_action_force_ratio)
            if selected_force_ratio is not None:
                label = selected_force_ratio_label or "对应回合"
                ep_str = (
                    f"回合 {selected_force_ratio_episode}"
                    if selected_force_ratio_episode is not None
                    else label
                )
                print(
                    f"⚠️  控制实验：强制覆盖ACTION_FORCE_RATIO={eval_action_force_ratio} "
                    f"(原{label}FR={selected_force_ratio}, {ep_str})"
                )
            else:
                print(f"⚠️  控制实验：强制覆盖ACTION_FORCE_RATIO={eval_action_force_ratio}")
        elif selected_force_ratio is not None:
            eval_action_force_ratio = selected_force_ratio
            label = selected_force_ratio_label or "对应回合"
            ep_str = (
                f"回合 {selected_force_ratio_episode}"
                if selected_force_ratio_episode is not None
                else label
            )
            print(f"✅ 使用{label}的ACTION_FORCE_RATIO: {eval_action_force_ratio} ({ep_str})")
        else:
            error_msg = (
                "❌ 错误：无法找到与当前模型目录对应的FR值\n"
                "   测试时必须使用与当前模型变体一致的FR值，而不是使用默认值\n"
                "   final/checkpoint 应读取最后回合 FR；best_by_team_sr 应读取 best_team_sr_force_ratio；best 应读取 best_episode_force_ratio\n"
                "   如果训练配置不完整，评估结果将不准确，因此拒绝继续运行"
            )
            print(error_msg)
            raise ValueError("无法找到与当前模型变体一致的FR值，评估无法继续。请确保训练配置文件包含正确的 FR 字段。")
        
        # 更新args中的action_force_ratio，确保后续使用
        self.args.action_force_ratio = eval_action_force_ratio
        self._selected_checkpoint_force_ratio = (
            float(selected_force_ratio) if selected_force_ratio is not None else None
        )
        self._selected_checkpoint_force_ratio_label = selected_force_ratio_label
        self._forced_eval_action_force_ratio = (
            float(forced_eval_action_force_ratio)
            if forced_eval_action_force_ratio is not None
            else None
        )
        
        # 🔧 关键修复：必须使用训练时的动作范围参数实际值，不允许使用默认值
        # 如果找不到训练配置中的参数，报错并退出
        missing_action_ranges = []
        
        if training_action_range_x is None:
            missing_action_ranges.append("action_range_x")
        else:
            self.args.action_range_x = training_action_range_x
            print(f"✅ 使用训练时的ACTION_RANGE_X: {training_action_range_x}")
        
        if training_action_range_y is None:
            missing_action_ranges.append("action_range_y")
        else:
            self.args.action_range_y = training_action_range_y
            print(f"✅ 使用训练时的ACTION_RANGE_Y: {training_action_range_y}")
        
        if training_action_range_z is None:
            missing_action_ranges.append("action_range_z")
        else:
            self.args.action_range_z = training_action_range_z
            print(f"✅ 使用训练时的ACTION_RANGE_Z: {training_action_range_z}")
        
        # 🚨 如果缺少任何必需的动作范围参数，报错并退出
        if missing_action_ranges:
            error_msg = (
                f"❌ 错误：无法找到训练配置中的以下动作范围参数: {', '.join(missing_action_ranges)}\n"
                f"   测试时必须使用训练时实际使用的参数值，而不是使用默认值\n"
                f"   请检查训练配置文件 (results.json) 是否包含这些参数\n"
                f"   如果训练配置不完整，评估结果将不准确，因此拒绝继续运行"
            )
            print(error_msg)
            raise ValueError(f"无法找到训练配置中的动作范围参数: {', '.join(missing_action_ranges)}，评估无法继续。")
        
        # 🔧 关键修复：优先使用命令行参数，如果未指定则使用训练配置中的值
        # 这样支持消融实验使用不同的episode_length（如4000步）进行评估
        # 保存命令行参数的值（在覆盖之前）
        cmd_episode_length = getattr(self.args, 'episode_length', None)
        
        # 检查命令行参数是否明确指定（不是默认值2200）
        if cmd_episode_length is not None and cmd_episode_length != 2200:
            # 命令行参数明确指定了episode_length（不是默认值2200），使用命令行参数
            print(f"✅ 使用命令行指定的EPISODE_LENGTH: {cmd_episode_length} (覆盖训练配置中的 {training_episode_length if training_episode_length is not None else 'N/A'})")
            # 保持使用命令行参数，不覆盖
        elif training_episode_length is not None:
            # 使用训练配置中的episode_length
            self.args.episode_length = training_episode_length
            print(f"✅ 使用训练时的EPISODE_LENGTH: {training_episode_length}")
        else:
            # 既没有命令行参数，也没有训练配置，使用默认值
            error_msg = (
                "❌ 错误：无法找到训练配置中的episode_length参数\n"
                "   测试时必须使用训练时实际使用的episode_length值，而不是使用默认值\n"
                "   请检查训练配置文件 (results.json) 是否包含 episode_length 字段\n"
                "   或者通过 --episode-length 参数明确指定\n"
                "   如果训练配置不完整，评估结果将不准确，因此拒绝继续运行"
            )
            print(error_msg)
            raise ValueError("无法找到训练配置中的episode_length，评估无法继续。请确保训练配置文件包含 episode_length 字段，或通过 --episode-length 参数指定。")
        
        # 🔧 关键修复：必须使用训练时的物理参数实际值，不允许使用默认值
        # 如果找不到训练配置中的参数，报错并退出（因为使用默认值会导致评估不准确）
        missing_params = []
        
        if training_gravity is None:
            missing_params.append("gravity")
        else:
            self.args.gravity = training_gravity
            print(f"✅ 使用训练时的GRAVITY: {training_gravity}")
        
        if training_damping is None:
            missing_params.append("damping")
        else:
            self.args.damping = training_damping
            print(f"✅ 使用训练时的DAMPING: {training_damping}")
        
        if training_agent_max_speed is None:
            missing_params.append("agent_max_speed")
        else:
            self.args.agent_max_speed = training_agent_max_speed
            print(f"✅ 使用训练时的AGENT_MAX_SPEED: {training_agent_max_speed}")
        
        if training_agent_accel is None:
            missing_params.append("agent_accel")
        else:
            self.args.agent_accel = training_agent_accel
            print(f"✅ 使用训练时的AGENT_ACCEL: {training_agent_accel}")
        
        if training_control_accel_gain is None:
            missing_params.append("control_accel_gain")
        else:
            self.args.control_accel_gain = training_control_accel_gain
            print(f"✅ 使用训练时的CONTROL_ACCEL_GAIN: {training_control_accel_gain}")

        if training_reward_pos_scale is not None:
            self.args.reward_pos_scale = training_reward_pos_scale
            print(f"✅ 使用训练时的REWARD_POS_SCALE: {training_reward_pos_scale}")
        if training_reward_neg_scale is not None:
            self.args.reward_neg_scale = training_reward_neg_scale
            print(f"✅ 使用训练时的REWARD_NEG_SCALE: {training_reward_neg_scale}")
        
        # 🚨 如果缺少任何必需的物理参数，报错并退出
        if missing_params:
            error_msg = (
                f"❌ 错误：无法找到训练配置中的以下物理参数: {', '.join(missing_params)}\n"
                f"   测试时必须使用训练时实际使用的参数值，而不是使用默认值\n"
                f"   请检查训练配置文件 (results.json) 是否包含这些参数\n"
                f"   如果训练配置不完整，评估结果将不准确，因此拒绝继续运行"
            )
            print(error_msg)
            raise ValueError(f"无法找到训练配置中的物理参数: {', '.join(missing_params)}，评估无法继续。")

        runtime_use_quadrotor_dynamics = None
        runtime_use_quadrotor_env = str(os.getenv("USE_QUADROTOR_DYNAMICS", "")).strip()
        if runtime_use_quadrotor_env:
            runtime_use_quadrotor_dynamics = runtime_use_quadrotor_env.lower() in ("1", "true", "yes", "on")
        else:
            try:
                current_use_quadrotor_dynamics = getattr(self.args, "use_quadrotor_dynamics", None)
            except Exception:
                current_use_quadrotor_dynamics = None
            if current_use_quadrotor_dynamics is not None:
                runtime_use_quadrotor_dynamics = bool(current_use_quadrotor_dynamics)

        hidden_runtime_params = (
            ("simulation_dt", training_simulation_dt, "SIMULATION_DT"),
            ("z_action_bias", training_z_action_bias, "Z_ACTION_BIAS"),
            ("quadrotor_attitude_response_time", training_quadrotor_attitude_response_time, "QUADROTOR_ATTITUDE_RESPONSE_TIME"),
            ("quadrotor_psi_cmd", training_quadrotor_psi_cmd, "QUADROTOR_PSI_CMD"),
        )
        for attr_name, value, label in hidden_runtime_params:
            if value is None:
                continue
            setattr(self.args, attr_name, value)
            print(f"✅ 使用训练时的{label}: {value}")
        if training_use_quadrotor_dynamics is not None:
            training_use_quadrotor_dynamics = bool(training_use_quadrotor_dynamics)
            if (
                runtime_use_quadrotor_dynamics is not None
                and runtime_use_quadrotor_dynamics != training_use_quadrotor_dynamics
            ):
                self.args.use_quadrotor_dynamics = bool(runtime_use_quadrotor_dynamics)
                print(
                    "⚠️ USE_QUADROTOR_DYNAMICS 与训练配置记录不一致，"
                    f"保留当前运行时值: {self.args.use_quadrotor_dynamics} "
                    f"(训练记录为 {training_use_quadrotor_dynamics})"
                )
            else:
                self.args.use_quadrotor_dynamics = training_use_quadrotor_dynamics
                print(f"✅ 使用训练时的USE_QUADROTOR_DYNAMICS: {self.args.use_quadrotor_dynamics}")
        if self.training_alignment:
            _apply_training_alignment_to_args(self.args, self.training_alignment, quiet=True)
        _apply_runtime_env_overrides_from_args(self.args)
        self._rebuild_environment()
        print("✅ 已按训练配置重建评估环境")

        if training_use_fr is not None or training_use_pf is not None:
            print(f"✅ 使用特征标志: use_fr_feature={use_fr_feature}, use_pf_feature={use_pf_feature}")
        else:
            print(f"ℹ️  未找到训练配置，使用当前配置: use_fr_feature={use_fr_feature}, use_pf_feature={use_pf_feature}")
        
        # 🚨 关键修复：优先使用训练时的网络结构配置，确保与训练时完全一致
        # 如果找到训练配置，使用训练时的配置；否则使用命令行参数；最后才使用默认值
        actor_hidden = (
            training_actor_hidden if training_actor_hidden is not None else
            (getattr(self.args, 'actor_hidden', None) if getattr(self.args, 'actor_hidden', None) else
             '256,256,256')  # 默认值：与训练脚本一致
        )
        critic_hidden = (
            training_critic_hidden if training_critic_hidden is not None else
            (getattr(self.args, 'critic_hidden', None) if getattr(self.args, 'critic_hidden', None) else
             '256,256,256')  # 默认值：与训练脚本一致（3层×256）
        )
        
        # 打印使用的网络配置
        if training_actor_hidden is not None:
            print(f"✅ 使用训练时的Actor隐藏层配置: {actor_hidden}")
        else:
            print(f"ℹ️  未找到训练配置中的actor_hidden，使用: {actor_hidden}")
        if training_critic_hidden is not None:
            print(f"✅ 使用训练时的Critic隐藏层配置: {critic_hidden}")
        else:
            print(f"ℹ️  未找到训练配置中的critic_hidden，使用: {critic_hidden}")
        
        # 创建临时args用于MADDPG初始化
        maddpg_args = argparse.Namespace(
            learning_rate_actor=1e-4,
            learning_rate_critic=3e-4,
            gamma=0.95,
            tau=0.005,
            grad_clip_norm=10.0,
            huber_delta=1.0,
            noise_scale=0.0,  # 🔧 关键修复：评估时禁用噪声，确保使用纯策略
            noise_decay=1.0,  # 评估时不需要衰减
            noise_min=0.0,  # 评估时不需要最小噪声
            random_action_prob=0.0,  # 评估时禁用随机动作
            per_enabled=False,
            matd3_use_dual_q=training_matd3_use_dual_q if training_matd3_use_dual_q is not None else getattr(self.args, 'matd3_use_dual_q', True),
            matd3_use_separated_gradient=training_matd3_use_separated_gradient if training_matd3_use_separated_gradient is not None else getattr(self.args, 'matd3_use_separated_gradient', True),
            maddpg_use_dual_q=training_maddpg_use_dual_q if training_maddpg_use_dual_q is not None else getattr(self.args, 'maddpg_use_dual_q', False),
            maddpg_use_separated_gradient=training_maddpg_use_separated_gradient if training_maddpg_use_separated_gradient is not None else getattr(self.args, 'maddpg_use_separated_gradient', True),
            # 🚨 关键修复：使用从训练配置读取的网络结构
            actor_hidden=actor_hidden,
            critic_hidden=critic_hidden,
            # 🔧 新增：FR和PF特征标志
            use_fr_feature=use_fr_feature,
            use_pf_feature=use_pf_feature,
            # 🔧 关键修复：使用训练时的pf_feature_dim（确保Critic输入维度一致）
            pf_feature_dim=training_pf_feature_dim if training_pf_feature_dim is not None else getattr(self.args, 'pf_feature_dim', 3),
            # 🔧 关键修复：使用训练时的action_force_ratio（如果是apf_learnable，FR=1.0）
            action_force_ratio=eval_action_force_ratio,
            use_tf_potential_field=getattr(self.args, 'use_tf_potential_field', True),
            goal_attraction=getattr(self.args, 'goal_attraction', 1.0),
            lambda_1_base=getattr(self.args, 'lambda_1_base', 5.0),
            terrain_repulsion=getattr(self.args, 'terrain_repulsion', 80.0),
            agent_influence_range=getattr(self.args, 'agent_influence_range', 10.0),
            delta_k_att=getattr(self.args, 'delta_k_att', 0.5),
            delta_lambda_1=getattr(self.args, 'delta_lambda_1', 2.5),
            delta_k_rep=getattr(self.args, 'delta_k_rep', 40.0),
            delta_radius=getattr(self.args, 'delta_radius', 5.0),
        )
        
        # 🔧 新增：创建环境对象以支持Oracle模式
        # 注意：评估时使用SingleEnvWrapper，需要获取实际的env对象
        eval_env = MultiAgentEnv(
            self.world,
            self.scenario.reset_world,
            self.scenario.reward,
            self.scenario.observation,
            done_callback=getattr(self.scenario, 'is_done', None),
            info_callback=None,
            shared_viewer=False
        )
        try:
            if hasattr(eval_env, 'world') and eval_env.world is not None:
                eval_env.world.episode_length = int(getattr(self.args, 'episode_length', 2200) or 2200)
                eval_env.world.current_step = 0
        except Exception:
            pass
        # 将scenario引用挂到world（与训练路径保持一致）
        try:
            self.world.scenario = self.scenario
            eval_env.scenario = self.scenario
        except Exception:
            pass
        
        # 🔧 新增：设置terrain_sensing_mode（评估时使用，不影响Actor/Critic输入）
        terrain_sensing_mode = getattr(self.args, 'terrain_sensing_mode', 'local')
        maddpg_args.terrain_sensing_mode = terrain_sensing_mode
        print(f"[地形感知模式] {terrain_sensing_mode}")
        if terrain_sensing_mode.startswith('oracle'):
            print(f"  ⚠️  Oracle模式：仅用于APF地形力计算，Actor/Critic输入保持不变（使用观测）")
        
        # 初始化MADDPG或MATD3（根据算法选择）
        # 🔧 修复：移除env参数，因为训练脚本中已经移除了env参数支持
        # Oracle模式通过其他方式实现（在_calculate_terrain_forces_sphere_tf中直接访问scenario）
        algorithm = getattr(self.args, 'algorithm', 'matd3').lower()
        if algorithm == 'matd3':
            self.maddpg = OptimizedMATD3(self.n_agents, self.obs_shapes, self.action_dims, maddpg_args)
        elif algorithm == 'mappo':
            if OptimizedMAPPO is None:
                raise ImportError(
                    "当前仓库缺少 algorithms/mappo 模块，无法执行 MAPPO 评估。"
                    " 如果只复现 MATD3/MADDPG，可保持 --algorithm matd3；"
                    " 如果需要 MAPPO，请先将 algorithms/mappo/、train_mappo_strict.py、evaluate_mappo.py 提交到 GitHub。"
                )
            self.maddpg = OptimizedMAPPO(self.n_agents, self.obs_shapes, self.action_dims, maddpg_args)
        else:
            self.maddpg = OptimizedMADDPG(self.n_agents, self.obs_shapes, self.action_dims, maddpg_args)
        
        # 🔧 新增：为Oracle模式设置scenario引用（如果需要）
        # 注意：Oracle模式在评估时通过evaluate_optimized.py中的scenario对象访问
        # 训练脚本中的_calculate_terrain_forces_sphere_tf会通过其他方式获取scenario（如果需要）
        terrain_sensing_mode = getattr(self.args, 'terrain_sensing_mode', 'local')
        if terrain_sensing_mode.startswith('oracle'):
            # 将scenario引用存储到maddpg对象中，以便在APF计算时使用
            try:
                self.maddpg.scenario_ref = self.scenario
                self.maddpg.world_ref = self.world
                print(f"  ✅ Oracle模式：已设置scenario引用，scenario类型: {type(self.scenario).__name__}")
                # 验证scenario是否有get_terrain_height方法
                if hasattr(self.scenario, 'get_terrain_height'):
                    print(f"  ✅ Oracle模式：scenario具有get_terrain_height方法")
                else:
                    print(f"  ⚠️  Oracle模式：scenario缺少get_terrain_height方法，Oracle模式可能无法正常工作")
            except Exception as e:
                print(f"  ❌ Oracle模式：设置scenario引用失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 加载模型权重（带有效性检查与回退策略）
        def _is_valid_weights_dir(dir_path: str, n_agents: int) -> bool:
            """检查权重目录是否有效（支持MATD3 Twin Critic）"""
            try:
                if not os.path.isdir(dir_path):
                    return False
                algorithm = getattr(self.args, 'algorithm', 'matd3').lower()
                for i in range(n_agents):
                    ap = os.path.join(dir_path, f"actor_{i}.weights.h5")
                    if not os.path.isfile(ap) or os.path.getsize(ap) <= 0:
                        return False
                    # 🚨 标准MATD3：检查两个独立的Critic网络文件
                    if algorithm == 'matd3':
                        cp1 = os.path.join(dir_path, f"critic1_{i}.weights.h5")
                        cp2 = os.path.join(dir_path, f"critic2_{i}.weights.h5")
                        # 如果新格式文件不存在，尝试旧格式（兼容性）
                        if not (os.path.isfile(cp1) and os.path.getsize(cp1) > 0) and \
                           not (os.path.isfile(cp2) and os.path.getsize(cp2) > 0):
                            # 回退到旧格式检查
                            cp_old = os.path.join(dir_path, f"critic_{i}.weights.h5")
                            if not os.path.isfile(cp_old) or os.path.getsize(cp_old) <= 0:
                                return False
                    elif algorithm == 'mappo':
                        continue
                    else:
                        # MADDPG：单个Critic网络
                        cp = os.path.join(dir_path, f"critic_{i}.weights.h5")
                        if not os.path.isfile(cp) or os.path.getsize(cp) <= 0:
                            return False
                return True
            except Exception:
                return False

        def _find_fallback_dir(preferred_dir: str, n_agents: int) -> str:
            # 🔧 关键修复：支持中文路径，在传入路径下查找子目录
            # 首先尝试在传入路径下查找 final -> best -> 最新 ep*
            candidates = []
            for name in ("final", "best", "best_by_team_sr", "best_by_strict_success"):
                candidates.append(os.path.join(preferred_dir, name))
            try:
                if os.path.isdir(preferred_dir):
                    eps = [d for d in os.listdir(preferred_dir) if d.startswith("ep") and os.path.isdir(os.path.join(preferred_dir, d))]
                    # 按数字部分降序排序
                    def _ep_key(s):
                        import re
                        m = re.search(r"\d+", s)
                        return int(m.group(0)) if m else -1
                    eps_sorted = sorted(eps, key=_ep_key, reverse=True)
                    candidates.extend([os.path.join(preferred_dir, d) for d in eps_sorted])
            except Exception:
                pass
            
            # 🔧 关键修复：如果传入路径下找不到，再尝试在父目录下查找（兼容旧格式）
            if not any(_is_valid_weights_dir(c, n_agents) for c in candidates):
                parent = os.path.dirname(preferred_dir)
                for name in ("final", "best", "best_by_team_sr", "best_by_strict_success"):
                    candidates.append(os.path.join(parent, name))
                try:
                    if os.path.isdir(parent):
                        eps = [d for d in os.listdir(parent) if d.startswith("ep") and os.path.isdir(os.path.join(parent, d))]
                        def _ep_key(s):
                            import re
                            m = re.search(r"\d+", s)
                            return int(m.group(0)) if m else -1
                        eps_sorted = sorted(eps, key=_ep_key, reverse=True)
                        candidates.extend([os.path.join(parent, d) for d in eps_sorted])
                except Exception:
                    pass
            
            for c in candidates:
                if _is_valid_weights_dir(c, n_agents):
                    return c
            return None

        model_dir = self.args.load_model_path
        if not _is_valid_weights_dir(model_dir, self.n_agents):
            fb = _find_fallback_dir(model_dir, self.n_agents)
            if fb is not None:
                print(f"⚠️  检测到模型目录不完整，回退到: {fb}")
                model_dir = fb
            else:
                raise FileNotFoundError(f"找不到可用的权重文件，请检查目录: {self.args.load_model_path}")

        # 先以虚拟输入构建网络，确保变量已创建（为每个智能体的 actor/critic 都建图）
        try:
            critic_state_dim = sum(self.obs_shapes)
            use_fr = getattr(maddpg_args, 'use_fr_feature', False)
            use_pf = getattr(maddpg_args, 'use_pf_feature', False)
            pf_feature_dim = getattr(maddpg_args, 'pf_feature_dim', 3)
            matd3_use_dual_q = getattr(maddpg_args, 'matd3_use_dual_q', True)
            
            for i in range(self.n_agents):
                # 🔧 构建 actor：根据训练配置构建正确的输入结构
                # 如果 use_pf_feature=True，Actor期望3个输入：[obs, fr_input, pf_input]
                # 如果 use_fr_feature=True 但 use_pf_feature=False，Actor期望2个输入：[obs, fr_input]
                # 如果两者都False，Actor期望1个输入：[obs]
                dummy_obs = tf.zeros((1, self.obs_shapes[i]), dtype=tf.float32)
                actor_inputs = [dummy_obs]
                if use_fr:
                    actor_inputs.append(tf.zeros((1, 1), dtype=tf.float32))
                if use_pf:
                    # 🔧 关键修复：如果训练时启用了PF特征，Actor需要PF特征作为单独输入
                    actor_inputs.append(tf.zeros((1, pf_feature_dim), dtype=tf.float32))
                
                if len(actor_inputs) == 1:
                    _ = self.maddpg.agents[i]['actor'](actor_inputs[0], training=False)
                else:
                    _ = self.maddpg.agents[i]['actor'](actor_inputs, training=False)
                
                # 🚨 标准MATD3：构建两个独立的Critic网络（Twin Critic）
                # Critic需要PF特征作为单独输入（与Actor不同）
                dummy_state = tf.zeros((1, critic_state_dim), dtype=tf.float32)
                dummy_actions = tf.zeros((1, self.action_dims[i] * self.n_agents), dtype=tf.float32)
                critic_inputs = [dummy_state, dummy_actions]
                if use_fr:
                    critic_inputs.append(tf.zeros((1, 1), dtype=tf.float32))
                if use_pf:
                    critic_inputs.append(tf.zeros((1, pf_feature_dim * self.n_agents), dtype=tf.float32))
                
                if algorithm == 'matd3':
                    # 🚨 标准MATD3：构建critic1和critic2
                    critic1_output = self.maddpg.agents[i]['critic1'](critic_inputs, training=False)
                    critic2_output = self.maddpg.agents[i]['critic2'](critic_inputs, training=False)
                    if matd3_use_dual_q:
                        # 每个critic输出两个Q值（用于梯度分离）
                        assert isinstance(critic1_output, (list, tuple)) and len(critic1_output) == 2, \
                            f"MATD3 Critic1应该输出两个Q值，实际输出: {type(critic1_output)}"
                        assert isinstance(critic2_output, (list, tuple)) and len(critic2_output) == 2, \
                            f"MATD3 Critic2应该输出两个Q值，实际输出: {type(critic2_output)}"
                elif algorithm == 'mappo':
                    if i == 0:
                        value_inputs = [dummy_state]
                        if use_fr:
                            value_inputs.append(tf.zeros((1, 1), dtype=tf.float32))
                        if use_pf:
                            value_inputs.append(tf.zeros((1, pf_feature_dim * self.n_agents), dtype=tf.float32))
                        _ = self.maddpg.value_critic(value_inputs, training=False)
                else:
                    # MADDPG：单个Critic网络
                    critic_output = self.maddpg.agents[i]['critic'](critic_inputs, training=False)
        except Exception as e:
            print(f"⚠️ 网络构建警告: {e}")
            pass

        # 🔧 关键修复：手动加载权重，支持新格式权重文件
        def _manual_load_weights(model, weight_file: str):
            """手动从 HDF5 文件加载权重，支持新格式（layers/*/vars/*）"""
            try:
                import h5py
                with h5py.File(weight_file, 'r') as f:
                    # 检查是否是新格式（有 layers 组）
                    if 'layers' in f:
                        # 新格式：layers/dense/vars/0, layers/dense/vars/1
                        layer_weights = {}
                        def collect_weights(name, obj):
                            if isinstance(obj, h5py.Dataset):
                                # 提取层名（例如：layers/dense/vars/0 -> dense）
                                parts = name.split('/')
                                if len(parts) >= 3 and parts[0] == 'layers' and parts[2] == 'vars':
                                    layer_name = parts[1]
                                    var_idx = int(parts[3]) if len(parts) > 3 else 0
                                    if layer_name not in layer_weights:
                                        layer_weights[layer_name] = {}
                                    layer_weights[layer_name][var_idx] = obj[:]
                        f.visititems(collect_weights)
                        
                        # 将权重设置到模型中
                        loaded_count = 0
                        for layer in model.layers:
                            if layer.name in layer_weights:
                                weights_data = layer_weights[layer.name]
                                # 按索引排序（0=kernel, 1=bias 或 gamma, beta）
                                sorted_vars = [weights_data[i] for i in sorted(weights_data.keys())]
                                try:
                                    layer.set_weights(sorted_vars)
                                    loaded_count += 1
                                except Exception as e:
                                    # 如果形状不匹配，跳过
                                    pass
                        return loaded_count > 0
                    else:
                        # 旧格式，使用标准加载
                        model.load_weights(weight_file)
                        return True
            except Exception as e:
                return False
        
        # 安全加载：先常规加载，失败则尝试手动加载，最后 skip_mismatch 兜底
        def _safe_load(agent, path: str, kind: str):
            try:
                agent.load_weights(path)
                return True
            except Exception as e:
                # 🔧 关键修复：尝试手动加载（支持新格式权重文件）
                try:
                    if _manual_load_weights(agent, path):
                        print(f"✅ {kind} 使用手动加载方式成功加载: {os.path.basename(path)}")
                        return True
                except Exception as e_manual:
                    pass
                try:
                    # 兼容可能的细微层名差异，只使用skip_mismatch
                    agent.load_weights(path, skip_mismatch=True)
                    print(f"⚠️  {kind} 使用skip_mismatch方式加载: {os.path.basename(path)} | {e}")
                    return True
                except Exception as e2:
                    print(f"❌ 加载{kind}失败: {path} | {e2}")
                    return False

        print(f"正在从 {model_dir} 加载模型...")
        ok = True
        total_loaded_vars = 0
        total_vars = 0

        if algorithm == 'mappo':
            try:
                self.maddpg.load_models(model_dir)
                print("✅ MAPPO 模型加载完成!")
                return
            except Exception as e:
                raise RuntimeError(f"MAPPO模型加载失败: {e}") from e
        
        for i in range(self.n_agents):
            a_path = os.path.join(model_dir, f"actor_{i}.weights.h5")
            # 🚨 标准MATD3：对于MATD3，c_path仅用于MADDPG兼容性检查
            algorithm = getattr(self.args, 'algorithm', 'matd3').lower()
            c_path = os.path.join(model_dir, f"critic_{i}.weights.h5") if algorithm != 'matd3' else None
            # 加载前后变量快照用于统计覆盖比例
            def _snapshot_vars(model):
                return [v.numpy().copy() for v in model.trainable_variables]
            def _count_changed(before, after):
                import numpy as _np
                changed = 0
                total = min(len(before), len(after))
                for bi, ai in zip(before, after):
                    if bi.shape != ai.shape:
                        continue
                    if not _np.array_equal(bi, ai):
                        changed += 1
                return changed, total

            # actor
            a_before = _snapshot_vars(self.maddpg.agents[i]['actor'])
            ok = _safe_load(self.maddpg.agents[i]['actor'], a_path, f"actor[{i}]") and ok
            a_after = _snapshot_vars(self.maddpg.agents[i]['actor'])
            chg, tot = _count_changed(a_before, a_after)
            total_loaded_vars += chg
            total_vars += tot
            if tot > 0:
                ratio = (chg / tot) * 100.0
                print(f"actor[{i}] 覆盖变量: {chg}/{tot} ({ratio:.1f}%)")
                if ratio < 60.0:
                    print(f"⚠️  actor[{i}] 覆盖比例偏低，可能与训练结构不一致")

            # 🚨 标准MATD3：加载两个独立的Critic网络
            algorithm = getattr(self.args, 'algorithm', 'matd3').lower()
            if algorithm == 'matd3':
                # 加载critic1
                c1_path = os.path.join(model_dir, f"critic1_{i}.weights.h5")
                # 如果新格式文件不存在，尝试旧格式（兼容性）
                if not os.path.exists(c1_path):
                    c1_path = os.path.join(model_dir, f"critic_{i}.weights.h5")
                    if os.path.exists(c1_path):
                        print(f"⚠️  检测到旧格式critic文件，将同时加载到critic1和critic2...")
                
                c1_before = _snapshot_vars(self.maddpg.agents[i]['critic1'])
                ok = _safe_load(self.maddpg.agents[i]['critic1'], c1_path, f"critic1[{i}]") and ok
                c1_after = _snapshot_vars(self.maddpg.agents[i]['critic1'])
                chg1, tot1 = _count_changed(c1_before, c1_after)
                total_loaded_vars += chg1
                total_vars += tot1
                if tot1 > 0:
                    ratio1 = (chg1 / tot1) * 100.0
                    print(f"critic1[{i}] 覆盖变量: {chg1}/{tot1} ({ratio1:.1f}%)")
                    if ratio1 < 60.0:
                        print(f"⚠️  critic1[{i}] 覆盖比例偏低，可能与训练结构不一致")
                
                # 加载critic2（如果旧格式，使用相同文件；否则使用critic2文件）
                c2_path = os.path.join(model_dir, f"critic2_{i}.weights.h5")
                if not os.path.exists(c2_path):
                    c2_path = c1_path  # 使用旧格式文件
                
                c2_before = _snapshot_vars(self.maddpg.agents[i]['critic2'])
                ok = _safe_load(self.maddpg.agents[i]['critic2'], c2_path, f"critic2[{i}]") and ok
                c2_after = _snapshot_vars(self.maddpg.agents[i]['critic2'])
                chg2, tot2 = _count_changed(c2_before, c2_after)
                total_loaded_vars += chg2
                total_vars += tot2
                if tot2 > 0:
                    ratio2 = (chg2 / tot2) * 100.0
                    print(f"critic2[{i}] 覆盖变量: {chg2}/{tot2} ({ratio2:.1f}%)")
                    if ratio2 < 60.0:
                        print(f"⚠️  critic2[{i}] 覆盖比例偏低，可能与训练结构不一致")
                
                # 同步到目标网络
                try:
                    self.maddpg.agents[i]['target_actor'].set_weights(self.maddpg.agents[i]['actor'].get_weights())
                    self.maddpg.agents[i]['target_critic1'].set_weights(self.maddpg.agents[i]['critic1'].get_weights())
                    self.maddpg.agents[i]['target_critic2'].set_weights(self.maddpg.agents[i]['critic2'].get_weights())
                except Exception:
                    pass
            else:
                # MADDPG：单个Critic网络
                c_before = _snapshot_vars(self.maddpg.agents[i]['critic'])
                ok = _safe_load(self.maddpg.agents[i]['critic'], c_path, f"critic[{i}]") and ok
                c_after = _snapshot_vars(self.maddpg.agents[i]['critic'])
                chg, tot = _count_changed(c_before, c_after)
                total_loaded_vars += chg
                total_vars += tot
                if tot > 0:
                    ratio = (chg / tot) * 100.0
                    print(f"critic[{i}] 覆盖变量: {chg}/{tot} ({ratio:.1f}%)")
                    if ratio < 60.0:
                        print(f"⚠️  critic[{i}] 覆盖比例偏低，可能与训练结构不一致")
                # 同步到目标网络
                try:
                    self.maddpg.agents[i]['target_actor'].set_weights(self.maddpg.agents[i]['actor'].get_weights())
                    self.maddpg.agents[i]['target_critic'].set_weights(self.maddpg.agents[i]['critic'].get_weights())
                except Exception:
                    pass

        # 总体加载统计
        if total_vars > 0:
            overall_ratio = (total_loaded_vars / total_vars) * 100.0
            print(f"\n📊 总体模型加载统计:")
            print(f"   - 总变量数: {total_vars}")
            print(f"   - 成功加载: {total_loaded_vars}")
            print(f"   - 加载比例: {overall_ratio:.1f}%")
            
            if overall_ratio < 50.0:
                print(f"❌ 警告: 模型加载比例过低 ({overall_ratio:.1f}%)，可能使用的是随机权重!")
                print(f"   建议重新训练模型或检查模型文件完整性")
            elif overall_ratio < 80.0:
                print(f"⚠️  注意: 模型加载比例较低 ({overall_ratio:.1f}%)，部分权重可能未正确加载")
            else:
                print(f"✅ 模型加载比例良好 ({overall_ratio:.1f}%)")

        if not ok:
            raise RuntimeError("无法成功加载全部模型权重，请检查权重文件是否完整匹配。")

        print("✅ 模型加载完成!")
        
    def evaluate_single_episode(self, episode_idx):
        """评估单个回合，仿照1.0版本的逻辑"""
        try:
            quiet_output = os.getenv("QUIET_OUTPUT", "1").lower() in ("1", "true", "yes", "on")
        except Exception:
            quiet_output = True
        if not quiet_output:
            print(f"\n🚀 开始评估回合 {episode_idx + 1}")
        
        # 🔧 关键修复：在reset前记录固定位置信息，确保所有评估模式使用相同的起点
        if hasattr(self.scenario, 'use_fixed_positions') and self.scenario.use_fixed_positions:
            if hasattr(self.scenario, 'fixed_positions') and self.scenario.fixed_positions:
                agents_pos = self.scenario.fixed_positions.get('agents', [])
                goal_pos = self.scenario.fixed_positions.get('goal', None)
                if episode_idx == 0 and not quiet_output:  # 只在第一个回合打印
                    print(f"🔧 固定位置验证: 使用固定位置，{len(agents_pos)}个智能体")
                    if len(agents_pos) > 0:
                        print(f"   智能体0起点: [{agents_pos[0][0]:.2f}, {agents_pos[0][1]:.2f}, {agents_pos[0][2]:.2f}]")
                    if goal_pos:
                        print(f"   目标位置: [{goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f}]")
            else:
                if episode_idx == 0 and not quiet_output:
                    print(f"⚠️  警告: use_fixed_positions=True但fixed_positions为None，可能未正确加载位置文件")
        
        # 环境重置
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            obs_n, _ = reset_result
        else:
            obs_n = reset_result
        
        # 🔧 关键修复：在reset后验证实际起点位置，确保所有评估模式使用相同的起点
        if episode_idx == 0 and not quiet_output:  # 只在第一个回合打印
            actual_start_positions = []
            for i, agent in enumerate(self.world.agents):
                actual_start_positions.append(agent.state.p_pos.copy())
                if i < 3:  # 只打印前3个智能体
                    print(f"   实际智能体{i}起点: [{agent.state.p_pos[0]:.2f}, {agent.state.p_pos[1]:.2f}, {agent.state.p_pos[2]:.2f}]")

        initial_positions = _capture_agent_positions(self.world.agents)
        agent_goal_positions = _extract_agent_goal_positions(self.world, self.scenario)
        prev_positions = [pos.copy() if pos is not None else None for pos in initial_positions]
        agent_path_lengths = [0.0 for _ in initial_positions]
        direct_goal_distances = [
            _distance_3d(pos, goal) for pos, goal in zip(initial_positions, agent_goal_positions)
        ]
        agent_min_goal_distances = [
            float(dist) if dist is not None else None for dist in direct_goal_distances
        ]
        try:
            success_threshold = float(getattr(self.args, 'success_distance_threshold', 4.0))
        except Exception:
            success_threshold = 4.0
        try:
            simulation_dt = float(
                getattr(getattr(self.env, 'world', None), 'dt', os.getenv('SIMULATION_DT', '0.08'))
            )
        except Exception:
            simulation_dt = 0.08
        agent_first_reach_steps = []
        for pos, goal in zip(initial_positions, agent_goal_positions):
            dist_to_goal = _distance_3d(pos, goal)
            if dist_to_goal is not None and dist_to_goal <= success_threshold:
                agent_first_reach_steps.append(0)
            else:
                agent_first_reach_steps.append(None)
        team_first_reach_step = (
            0 if agent_first_reach_steps and all(step == 0 for step in agent_first_reach_steps) else None
        )
            
        episode_reward = 0
        try:
            light_mode = os.getenv("EVAL_LIGHT_MODE", "0").lower() in ("1", "true", "yes", "on")
        except Exception:
            light_mode = False
        save_team_success_html = _env_flag("SAVE_TEAM_SUCCESS_HTML", False)
        save_interactive_traj = _env_flag("SAVE_INTERACTIVE_TRAJ", True)
        save_all_episode_visualizations = _env_flag("SAVE_EVAL_ALL_EPISODES", False)
        save_best_traj = _env_flag("SAVE_BEST_TRAJ", True)
        save_trajectory_png = _env_flag("SAVE_EVAL_TRAJECTORY_PNG", False)
        save_actor_sequence = _env_flag("SAVE_EVAL_ACTOR_SEQUENCE", False)
        save_control_diagnostics = _env_flag("SAVE_EVAL_CONTROL_DIAGNOSTICS", False)
        trajectory_sample_interval = _env_int("EVAL_TRAJECTORY_SAMPLE_INTERVAL", 1)
        need_trajectory_artifacts = (
            not getattr(self.args, 'disable_visualization', False)
            and (
                save_all_episode_visualizations
                or save_best_traj
                or save_trajectory_png
                or save_interactive_traj
                or save_team_success_html
                or (not getattr(self.args, 'disable_gif', False))
            )
        )
        # 轻量模式下允许通过稀疏采样保留评估轨迹；若本轮不生成任何轨迹类产物，则完全跳过轨迹缓存。
        record_trajectory = bool(need_trajectory_artifacts)
        # 动作历史只服务于时序图/控制诊断；默认不再在纯HTML评估里额外记录。
        record_actions = save_actor_sequence or save_control_diagnostics
        episode_trajectory = []
        episode_actions_history = []  # 🔧 新增：记录动作历史（用于生成动作时序图）
        episode_executed_actions_history = []  # 记录实际送入环境的动作（含PF修正）
        episode_velocity_history = []  # 记录每步速度向量，判断是否“推不起来”
        episode_goal_distance_history = []  # 记录每步到目标距离，判断是否持续推进
        step_count = 0

        # 与产物需求对齐：若本轮不生成任何轨迹类文件，则连环境内部轨迹也一起关闭。
        try:
            self.env._disable_trajectory_recording = not bool(need_trajectory_artifacts)
        except Exception:
            pass
        
        # 处理观察数据
        processed_obs = self._process_observations_for_eval(obs_n)
        
        # 记录开始时间
        start_time = time.time()
        
        # 🚨 关键修复：确保episode_length正确设置
        episode_length = getattr(self.args, 'episode_length', 2200)
        if episode_length <= 0:
            episode_length = 2200
            if not quiet_output:
                print(f"⚠️  警告: episode_length无效，使用默认值: {episode_length}")
        
        # 🔧 新增：打印关键评估参数，便于调试
        action_force_ratio = getattr(self.args, 'action_force_ratio', 0.0)
        use_tf_potential_field = getattr(self.args, 'use_tf_potential_field', True)
        use_fr_feature = getattr(self.args, 'use_fr_feature', False)
        use_pf_feature = getattr(self.args, 'use_pf_feature', False)
        action_range_x = getattr(self.args, 'action_range_x', None)
        action_range_y = getattr(self.args, 'action_range_y', None)
        action_range_z = getattr(self.args, 'action_range_z', None)
        
        if not quiet_output:
            print(f"📊 评估配置:")
            print(f"   - episode_length={episode_length}")
            print(f"   - disable_early_termination={getattr(self.args, 'disable_early_termination', False)}")
            print(f"   - ACTION_FORCE_RATIO={action_force_ratio} {'⚠️ 为0，势场修正将不生效！' if action_force_ratio == 0.0 else ''}")
            print(f"   - USE_TF_POTENTIAL_FIELD={use_tf_potential_field}")
            print(f"   - use_fr_feature={use_fr_feature}")
            print(f"   - use_pf_feature={use_pf_feature}")
            print(f"   - action_range: X={action_range_x}, Y={action_range_y}, Z={action_range_z}")
            if use_tf_potential_field and action_force_ratio > 0.0:
                print(f"   ✅ 势场修正将生效 (FR={action_force_ratio})")
            else:
                print(f"   ⚠️  势场修正将不生效 (use_tf_potential_field={use_tf_potential_field}, FR={action_force_ratio})")
        
        # 🔧 新增：添加进度条显示评估进度
        # 🔧 修复：确保进度条正确显示，设置合适的更新频率
        try:
            tqdm_to_stdout = os.getenv("TQDM_TO_STDOUT", "1").lower() in ("1", "true", "yes", "on")
            tqdm_file = sys.stdout if tqdm_to_stdout else sys.stderr
        except Exception:
            tqdm_file = sys.stdout
        try:
            tqdm_disable = os.getenv("TQDM_DISABLE", "0").lower() in ("1", "true", "yes", "on")
        except Exception:
            tqdm_disable = False
        try:
            debug_action_steps = int(os.getenv("EVAL_DEBUG_ACTION_STEPS", "3"))
        except Exception:
            debug_action_steps = 3
        # 🚨 关键修复：设置position参数，避免多个进度条重叠显示
        # 问题：多个回合的进度条同时显示时，如果没有设置position，会导致显示混乱
        # 解决方案：每个回合使用position=0（单行显示），或者使用leave=False（完成后清除）
        # 注意：如果使用position，需要确保所有进度条使用相同的position，否则会显示多行
        # 🚨 关键优化：大幅降低进度条更新频率，避免并行执行时输出混乱
        # 从环境变量读取更新频率，默认值更合理
        try:
            mininterval = float(os.getenv('TQDM_MININTERVAL', '5.0'))  # 默认5秒更新一次（大幅降低）
        except Exception:
            mininterval = 5.0
        try:
            miniters = int(os.getenv('TQDM_MINITERS', '200'))  # 默认200步更新一次（大幅降低）
        except Exception:
            miniters = 200
        
        # 这里使用leave=False，让进度条完成后自动清除，避免累积
        pbar = tqdm(range(int(episode_length)), desc=f"回合 {episode_idx + 1}", unit="步",
                   ncols=120, leave=False, mininterval=mininterval, miniters=miniters,
                   file=tqdm_file, dynamic_ncols=False, ascii=False, disable=tqdm_disable,
                   position=0)  # 🔧 修复：设置position=0，确保所有进度条在同一行显示
        
        for step in pbar:
            use_tf_potential_field = getattr(self.args, 'use_tf_potential_field', True)
            action_force_ratio = getattr(self.args, 'action_force_ratio', 0.0)
            
            # 🔧 性能优化：大幅减少进度条更新频率（每200步更新一次，与训练脚本一致）
            # 训练脚本通常不使用步级进度条，评估时也减少更新频率以提升性能
            if step % 200 == 0 or step < 5:
                pbar.set_postfix({
                    '奖励': f'{episode_reward:.1f}',
                    '步数': f'{step + 1}/{episode_length}'
                })
                # 🔧 性能优化：移除pbar.refresh()，让tqdm自动管理刷新频率
            
            # 选择动作（评估时不加噪声）
            # 🔧 关键修复：使用训练配置的特征标志，而不是命令行参数
            # 确保与训练时的网络输入结构完全一致
            use_fr = getattr(self.maddpg.args, 'use_fr_feature', False)
            use_pf = getattr(self.maddpg.args, 'use_pf_feature', False)
            policy_obs = self._build_eval_actor_obs(
                processed_obs,
                use_pf=use_pf,
                use_tf_potential_field=use_tf_potential_field,
                action_force_ratio=action_force_ratio,
            )
            raw_actions_tf = self.select_actions_eval(policy_obs, use_fr=use_fr, use_pf=use_pf)
            # 🔧 性能优化：延迟numpy转换，尽量在tensor空间内操作
            raw_actions = raw_actions_tf.numpy() if isinstance(raw_actions_tf, tf.Tensor) else raw_actions_tf
            
            # 🔧 新增：记录Actor原始输出（用于生成动作时序图）
            # 🔧 性能优化：只在需要时记录，避免不必要的copy操作
            if record_actions:
                # 🔧 性能优化：使用列表推导式，避免循环中的多次copy
                episode_actions_history.append([action.copy() for action in raw_actions])
            
            # 🔧 关键修复：应用势场修正（与训练时一致）
            # 势场修正生效条件：USE_TF_POTENTIAL_FIELD=1 AND ACTION_FORCE_RATIO > 0.0
            use_tf_potential_field = getattr(self.args, 'use_tf_potential_field', True)
            action_force_ratio = getattr(self.args, 'action_force_ratio', 0.0)
            
            if use_tf_potential_field and action_force_ratio > 0.0:
                # 🔧 性能优化：批量应用势场修正，尽量在tensor空间内操作
                # 🔧 关键优化：如果raw_actions已经是tensor，避免重复转换
                if isinstance(raw_actions_tf, tf.Tensor):
                    raw_actions_tf_for_correction = raw_actions_tf
                else:
                    raw_actions_tf_for_correction = tf.convert_to_tensor(raw_actions, dtype=tf.float32)  # (n_agents, action_dim)
                
                # 势场修正始终基于基础观测进行，避免把 PF 特征误当作环境观测再次输入。
                base_obs_dim = self._get_base_obs_dim()
                base_obs_for_correction = processed_obs[:, :base_obs_dim] if processed_obs.shape[1] > base_obs_dim else processed_obs
                processed_obs_tf = tf.convert_to_tensor(base_obs_for_correction, dtype=tf.float32)  # (n_agents, obs_dim)
                    
                # 🔧 性能优化：使用训练代码的@tf.function装饰的势场修正函数
                corrected_head_tf, _ = self.maddpg._apply_potential_field_correction(
                    raw_actions_tf_for_correction, processed_obs_tf, action_force_ratio
                    )
                    
                # 获取action_dim（从tensor形状推断）
                action_dim = tf.shape(raw_actions_tf_for_correction)[1]
                if action_dim > 3:
                    corrected_actions_tf = tf.concat([corrected_head_tf, raw_actions_tf_for_correction[:, 3:]], axis=1)
                else:
                    corrected_actions_tf = corrected_head_tf
                
                # 🔧 性能优化：延迟numpy转换，只在env.step需要时转换
                actions = corrected_actions_tf.numpy()  # env.step需要numpy数组
            else:
                # 不使用势场修正，直接使用原始动作
                # 🔧 性能优化：如果raw_actions_tf是tensor，转换为numpy；否则直接使用
                if isinstance(raw_actions_tf, tf.Tensor):
                    actions = raw_actions_tf.numpy()
                else:
                    actions = raw_actions

            if record_actions:
                try:
                    if save_control_diagnostics:
                        episode_executed_actions_history.append([action.copy() for action in actions])
                except Exception:
                    pass
            
            # 🔧 性能优化：记录轨迹（只在需要可视化时记录）
            # 注意：环境中的agent._trajectory记录无法禁用（在environment.py的step方法中），但这里的额外记录可以禁用
            if record_trajectory and (step % trajectory_sample_interval == 0):
                try:
                    # 🔧 性能优化：使用列表推导式，一次性完成所有操作
                    positions = [agent.state.p_pos.copy() if hasattr(agent.state, 'p_pos') else [0, 0, 0] 
                                for agent in self.env.agents]
                    episode_trajectory.append(positions)
                except Exception as e:
                    # 🔧 性能优化：只在非quiet模式下输出警告
                    if not os.getenv("QUIET_OUTPUT", "1").lower() in ("1", "true", "yes", "on"):
                        tqdm.write(f"轨迹记录警告: {e}", file=tqdm_file)
                
            # 🔧 新增：在前几步打印动作值，便于调试（使用tqdm.write避免干扰进度条）
            if step < debug_action_steps:
                tqdm.write(f"   Step {step}: 动作值 (前3个智能体):", file=tqdm_file)
                for i in range(min(3, len(actions))):
                    tqdm.write(
                        f"      Agent {i}: {actions[i][:3]} (原始动作: {raw_actions[i][:3] if 'raw_actions' in locals() else 'N/A'})",
                        file=tqdm_file
                    )
                
            # 执行动作
            step_result = self.env.step(actions)
            if len(step_result) == 4:
                next_obs_n, rew_n, done_n, info_n = step_result
            elif len(step_result) == 5:
                next_obs_n, rew_n, terminated, truncated, info_n = step_result
                done_n = [t or tr for t, tr in zip(terminated, truncated)]
            else:
                raise ValueError(f"意外的环境step返回值: {len(step_result)}")

            # 累计奖励口径与训练保持一致：
            # 训练按“当前步所有智能体奖励的均值”累计，而不是直接对智能体求和。
            try:
                rew_arr = np.asarray(rew_n, dtype=np.float32)
                if rew_arr.size > 0:
                    if not np.all(np.isfinite(rew_arr)):
                        rew_arr = np.where(np.isfinite(rew_arr), rew_arr, -1000.0)
                    step_increment = float(np.mean(rew_arr))
                else:
                    step_increment = 0.0
            except Exception:
                try:
                    step_increment = float(np.mean(rew_n))
                except Exception:
                    step_increment = float(sum(rew_n)) if rew_n is not None else 0.0
            episode_reward += step_increment
            step_count += 1

            current_positions = _capture_agent_positions(self.env.agents)
            if save_control_diagnostics:
                current_velocities = []
                for agent in self.env.agents:
                    try:
                        vel = _normalize_vec3(getattr(getattr(agent, 'state', None), 'p_vel', None))
                    except Exception:
                        vel = None
                    current_velocities.append(vel)
                if current_velocities:
                    episode_velocity_history.append(
                        [vel.copy() if vel is not None else None for vel in current_velocities]
                    )

            if save_control_diagnostics and current_positions and agent_goal_positions:
                current_goal_distances = [
                    _distance_3d(pos, goal) for pos, goal in zip(current_positions, agent_goal_positions)
                ]
                episode_goal_distance_history.append(current_goal_distances)

            for agent_idx in range(min(len(agent_path_lengths), len(current_positions))):
                prev_pos = prev_positions[agent_idx]
                curr_pos = current_positions[agent_idx]
                if prev_pos is None or curr_pos is None:
                    continue
                step_distance = float(np.linalg.norm(curr_pos - prev_pos))
                if np.isfinite(step_distance):
                    agent_path_lengths[agent_idx] += step_distance
            prev_positions = [pos.copy() if pos is not None else None for pos in current_positions]

            if agent_goal_positions and current_positions:
                team_reached_now = True
                valid_reach_checks = 0
                for agent_idx in range(min(len(current_positions), len(agent_goal_positions))):
                    curr_pos = current_positions[agent_idx]
                    goal_pos = agent_goal_positions[agent_idx]
                    dist_to_goal = _distance_3d(curr_pos, goal_pos)
                    if dist_to_goal is None:
                        team_reached_now = False
                        continue
                    valid_reach_checks += 1
                    prev_min_goal_dist = agent_min_goal_distances[agent_idx]
                    if prev_min_goal_dist is None or dist_to_goal < prev_min_goal_dist:
                        agent_min_goal_distances[agent_idx] = float(dist_to_goal)
                    reached = dist_to_goal <= success_threshold
                    if reached and agent_first_reach_steps[agent_idx] is None:
                        agent_first_reach_steps[agent_idx] = step_count
                    if not reached:
                        team_reached_now = False
                if valid_reach_checks == 0:
                    team_reached_now = False
                if team_reached_now and team_first_reach_step is None:
                    team_first_reach_step = step_count
            
            # 更新观察
            processed_obs = self._process_observations_for_eval(next_obs_n)
            # 注意：势场力追加会在下一轮循环开始时进行
            
            # 检查结束条件（支持禁用提前终止）
            if all(done_n) and (not getattr(self.args, 'disable_early_termination', False)):
                # 🔧 性能优化：只在进度条启用时更新
                if not tqdm_disable:
                    pbar.set_postfix({
                        '奖励': f'{episode_reward:.1f}',
                        '状态': '提前结束',
                        '步数': f'{step + 1}/{episode_length}'
                    })
                    pbar.close()
                if not os.getenv("QUIET_OUTPUT", "1").lower() in ("1", "true", "yes", "on"):
                    tqdm.write(f"📍 回合在第 {step + 1}/{episode_length} 步自然结束（所有智能体done）", file=tqdm_file)
                break
                
        # 🔧 修复：在循环结束后更新最终状态
        # 🔧 性能优化：只在进度条启用时更新
        if not tqdm_disable:
            pbar.set_postfix({
                '奖励': f'{episode_reward:.1f}',
                '步数': f'{step_count}/{episode_length}',
                '状态': '完成'
            })
            pbar.close()
        
        # 计算回合统计
        episode_duration = time.time() - start_time
        avg_step_time = episode_duration / step_count if step_count > 0 else 0
        
        # 🔧 新增：收集碰撞次数和最小净空距离（与训练脚本一致）
        # 🚨 关键修复：确保使用与训练时相同的统计方式
        # 训练时使用 agent.current_episode_collision_count 和 agent.debug_info['total_penetration_count']
        # 两者应该同步更新，但优先使用 debug_info['total_penetration_count']（与训练脚本一致）
        episode_collision_counts = []
        episode_min_distances = []
        try:
            if hasattr(self.env, 'world') and hasattr(self.env.world, 'agents'):
                for agent in self.env.world.agents:
                    # 🚨 关键修复：确保debug_info已初始化
                    if not hasattr(agent, 'debug_info'):
                        agent.debug_info = {}
                    if not isinstance(agent.debug_info, dict):
                        agent.debug_info = {}
                    
                    # 🚨 关键修复：优先从debug_info读取，如果没有则从current_episode_collision_count读取
                    # 确保与训练脚本的统计方式完全一致
                    penetration_count = 0
                    if 'total_penetration_count' in agent.debug_info:
                        penetration_count = agent.debug_info.get('total_penetration_count', 0)
                    elif hasattr(agent, 'current_episode_collision_count'):
                        # 回退：如果debug_info中没有，使用current_episode_collision_count
                        penetration_count = agent.current_episode_collision_count
                    
                    try:
                        penetration_count = int(penetration_count) if np.isfinite(penetration_count) else 0
                    except (ValueError, TypeError, OverflowError):
                        penetration_count = 0
                    
                    episode_collision_counts.append(penetration_count)
                    
                    # 收集min_distance_to_obstacle
                    min_dist = None
                    if hasattr(agent, 'debug_info') and isinstance(agent.debug_info, dict):
                        min_dist = agent.debug_info.get('d_min_current', None)
                        if min_dist is not None:
                            try:
                                if isinstance(min_dist, np.ndarray):
                                    if min_dist.size > 0:
                                        min_dist = float(min_dist[-1] if min_dist.ndim > 0 else min_dist.item())
                                    else:
                                        min_dist = None
                                else:
                                    min_dist = float(min_dist)
                            except (ValueError, TypeError, AttributeError):
                                min_dist = None
                    
                    # 如果debug_info中没有，尝试从last_min_distance获取
                    if min_dist is None and hasattr(agent, 'last_min_distance') and agent.last_min_distance is not None:
                        try:
                            last_min_dist = agent.last_min_distance
                            if isinstance(last_min_dist, np.ndarray):
                                if last_min_dist.size > 0:
                                    min_dist = float(last_min_dist[-1] if last_min_dist.ndim > 0 else last_min_dist.item())
                                else:
                                    min_dist = None
                            else:
                                min_dist = float(last_min_dist)
                        except (ValueError, TypeError, AttributeError):
                            min_dist = None
                    
                    if min_dist is not None and np.isfinite(min_dist):
                        episode_min_distances.append(min_dist)
        except Exception as e:
            if not os.getenv("QUIET_OUTPUT", "1").lower() in ("1", "true", "yes", "on"):
                print(f"⚠️  收集碰撞统计信息时出错: {e}")
        
        # 🔧 新增：计算成功标志（与训练脚本一致）
        # 定义：Reach_i := 终止帧 ||p_T^i - g^i|| ≤ r_goal
        #       Safe_i := 整个回合期间无真实碰撞/真实穿透
        #       Succ_i := Reach_i ∧ Safe_i
        #       Succ_team := Λ_{i=1}^{M} Succ_i
        agent_success_flags = []
        team_success_flag = 1
        try:
            thr_success = float(getattr(self.args, 'success_distance_threshold', 4.0))
            if hasattr(self.env, 'world') and hasattr(self.env.world, 'agents'):
                scn = getattr(self.env, 'scenario', None)
                goal_pos = None
                if scn is not None:
                    goal_pos = getattr(scn, 'goal_pos', None)
                    if goal_pos is None and hasattr(scn, 'get_goal_pos'):
                        try:
                            goal_pos = scn.get_goal_pos()
                        except Exception:
                            goal_pos = None
                
                for agent_idx, agent in enumerate(self.env.world.agents):
                    # 检查是否到达目标（Reach_i）
                    pos = getattr(getattr(agent, 'state', None), 'p_pos', None)
                    if pos is None or len(pos) < 3:
                        agent_success_flags.append(0)
                        team_success_flag = 0
                        continue
                    
                    # 获取智能体的目标位置
                    ag_goal = None
                    if hasattr(agent, 'goal_a') and hasattr(agent.goal_a, 'state') and getattr(agent.goal_a.state, 'p_pos', None) is not None:
                        ag_goal = agent.goal_a.state.p_pos
                    if ag_goal is None:
                        ag_goal = goal_pos
                    
                    if ag_goal is None:
                        agent_success_flags.append(0)
                        team_success_flag = 0
                        continue
                    
                    # 计算到目标的距离
                    dx = pos[0] - ag_goal[0]
                    dy = pos[1] - ag_goal[1]
                    dz = pos[2] - ag_goal[2]
                    dist_goal_3d = (dx*dx + dy*dy + dz*dz) ** 0.5
                    # 统一阈值：评估侧不再乘 1.2
                    reach_i = (dist_goal_3d <= thr_success)
                    
                    # 检查是否无碰撞（Safe_i）
                    safe_i = True
                    try:
                        if getattr(agent, '_episode_has_collision', False):
                            safe_i = False
                    except Exception:
                        pass
                    try:
                        if getattr(agent, '_had_obstacle_collision', False):
                            safe_i = False
                    except Exception:
                        pass
                    try:
                        if getattr(agent, '_had_terrain_contact_or_penetration', False):
                            safe_i = False
                    except Exception:
                        pass

                    pen_count = None
                    if agent_idx < len(episode_collision_counts):
                        pen_count = episode_collision_counts[agent_idx]
                    if pen_count is None and hasattr(agent, 'debug_info') and isinstance(agent.debug_info, dict):
                        pen_count = agent.debug_info.get('total_penetration_count', 0)
                    try:
                        pen_count = int(pen_count) if np.isfinite(pen_count) else 0
                    except Exception:
                        pen_count = None
                    if pen_count is None or pen_count > 0:
                        safe_i = False
                    
                    # 单智能体成功 = 到达目标 AND 无碰撞
                    succ_i = 1 if (reach_i and safe_i) else 0
                    agent_success_flags.append(succ_i)
                    
                    if succ_i == 0:
                        team_success_flag = 0
        except Exception as e:
            if not os.getenv("QUIET_OUTPUT", "1").lower() in ("1", "true", "yes", "on"):
                print(f"⚠️  计算成功标志时出错: {e}")
            agent_success_flags = [0] * len(episode_collision_counts) if episode_collision_counts else []
            team_success_flag = 0
        
        # 计算汇总统计
        total_collisions = sum(episode_collision_counts) if episode_collision_counts else 0
        try:
            total_collisions = int(total_collisions) if np.isfinite(total_collisions) else 0
        except (ValueError, TypeError, OverflowError):
            total_collisions = 0
        
        # 计算最小净空距离的统计（与训练脚本格式一致）
        min_distance_stat = None
        if episode_min_distances:
            try:
                min_distance_stat = {
                    'mean': float(np.mean(episode_min_distances)),
                    'min': float(np.min(episode_min_distances))
                }
            except Exception:
                min_distance_stat = None
        
        episode_success = (team_success_flag == 1)
        success_flag = 1 if episode_success else 0

        agent_path_efficiencies = []
        for direct_dist, path_len in zip(direct_goal_distances, agent_path_lengths):
            if direct_dist is None or path_len is None or path_len <= 1e-9:
                agent_path_efficiencies.append(None)
                continue
            efficiency = direct_dist / max(path_len, 1e-9)
            agent_path_efficiencies.append(float(np.clip(efficiency, 0.0, 1.0)))

        valid_direct_dists = [dist for dist in direct_goal_distances if dist is not None]
        team_direct_distance = float(np.sum(valid_direct_dists)) if valid_direct_dists else None
        team_total_path_length = float(np.sum(agent_path_lengths)) if agent_path_lengths else 0.0
        path_efficiency = None
        if team_direct_distance is not None and team_total_path_length > 1e-9:
            path_efficiency = float(np.clip(team_direct_distance / team_total_path_length, 0.0, 1.0))

        final_positions_for_stats = _capture_agent_positions(self.env.agents)
        agent_final_goal_distances = [
            _distance_3d(pos, goal) for pos, goal in zip(final_positions_for_stats, agent_goal_positions)
        ]
        valid_final_goal_distances = [dist for dist in agent_final_goal_distances if dist is not None]
        final_goal_distance = (
            float(np.sum(valid_final_goal_distances)) if valid_final_goal_distances else None
        )
        valid_min_goal_distances = [dist for dist in agent_min_goal_distances if dist is not None]
        min_goal_distance = (
            float(np.sum(valid_min_goal_distances)) if valid_min_goal_distances else None
        )

        first_reach_step = team_first_reach_step
        first_reach_time = (
            float(team_first_reach_step * simulation_dt) if team_first_reach_step is not None else None
        )

        # 到达时间/步数按“首次团队到达”记录；arrival_* 保持与 success 绑定，便于 success-only 汇总
        arrival_step = first_reach_step if success_flag == 1 else None
        arrival_time = first_reach_time if success_flag == 1 else None
        
        # 🔧 新增：计算穿透深度统计（从min_distance中的负值提取）
        penetration_depths = []
        penetration_count_episodes = 0
        if episode_min_distances:
            for min_dist in episode_min_distances:
                if min_dist is not None and np.isfinite(min_dist):
                    if min_dist < 0:  # 负值表示穿透
                        penetration_depths.append(abs(min_dist))
                        penetration_count_episodes += 1
        
        penetration_stat = None
        if penetration_depths:
            penetration_stat = {
                'count': penetration_count_episodes,
                'max_depth': float(np.max(penetration_depths)),
                'mean_depth': float(np.mean(penetration_depths)),
                'min_depth': float(np.min(penetration_depths))
            }

        if record_trajectory:
            try:
                final_positions = [
                    agent.state.p_pos.copy() if hasattr(agent.state, 'p_pos') else [0, 0, 0]
                    for agent in self.env.agents
                ]
                episode_trajectory.append(final_positions)
            except Exception as e:
                if not quiet_output:
                    print(f"⚠️  追加最终轨迹点失败: {e}")
        
        if not quiet_output:
            print(f"✅ 回合 {episode_idx + 1} 完成:")
            print(f"   - 奖励: {episode_reward:.2f}")
            print(f"   - 步数: {step_count}/{episode_length} (完成度: {step_count/episode_length*100:.1f}%)")
            print(f"   - 用时: {episode_duration:.2f}秒")
            print(f"   - 平均步时: {avg_step_time:.4f}秒/步")
            print(f"   - 碰撞次数: {total_collisions} (智能体: {episode_collision_counts})")
            print(f"   - 成功率: 团队={success_flag}, 智能体={agent_success_flags}")
            if arrival_step is not None:
                print(f"   - 成功首达步数: {arrival_step}")
            elif first_reach_step is not None:
                print(f"   - 首次到达步数(未计为成功): {first_reach_step}")
            print(f"   - 路径长度: 团队={team_total_path_length:.2f}, 智能体={['{:.2f}'.format(v) for v in agent_path_lengths]}")
            if path_efficiency is not None:
                print(f"   - 路径效率: 团队={path_efficiency:.3f}")
            if final_goal_distance is not None:
                print(f"   - 终止帧目标距离: 团队={final_goal_distance:.2f}, 智能体={['{:.2f}'.format(v) if v is not None else 'N/A' for v in agent_final_goal_distances]}")
            if min_goal_distance is not None:
                print(f"   - 回合最小目标距离: 团队={min_goal_distance:.2f}, 智能体={['{:.2f}'.format(v) if v is not None else 'N/A' for v in agent_min_goal_distances]}")
            if min_distance_stat is not None:
                print(f"   - 最小净空距离: 均值={min_distance_stat['mean']:.2f}, 最小值={min_distance_stat['min']:.2f}")
            if penetration_stat is not None:
                print(f"   - 穿透统计: 次数={penetration_stat['count']}, 最大深度={penetration_stat['max_depth']:.2f}, 平均深度={penetration_stat['mean_depth']:.2f}")
            if step_count < episode_length:
                print(f"   ⚠️  注意: 回合提前结束（可能由于done=True或提前终止）")

        vis_context = self._capture_episode_vis_context(episode_idx)
        
        return {
            'episode': episode_idx,
            'reward': episode_reward,
            'steps': step_count,
            'trajectory': episode_trajectory if record_trajectory else [],
            'actions_history': episode_actions_history if record_actions else [],
            'executed_actions_history': episode_executed_actions_history if save_control_diagnostics else [],
            'velocity_history': episode_velocity_history if save_control_diagnostics else [],
            'goal_distance_history': episode_goal_distance_history if save_control_diagnostics else [],
            'duration': episode_duration,
            # 🔧 新增：返回碰撞和成功指标（与训练脚本一致）
            'collision_count': total_collisions,
            'agent_collision_counts': episode_collision_counts,
            'min_distance': min_distance_stat,
            'success': success_flag,
            'agent_success_flags': agent_success_flags,
            'team_success': team_success_flag,
            # 🔧 新增：到达时间/步数
            'arrival_step': arrival_step,
            'arrival_time': arrival_time,
            'first_reach_step': first_reach_step,
            'first_reach_time': first_reach_time,
            'path_length': team_total_path_length,
            'agent_path_lengths': [float(v) for v in agent_path_lengths],
            'direct_distance': team_direct_distance,
            'agent_direct_distances': [float(v) if v is not None else None for v in direct_goal_distances],
            'final_goal_distance': final_goal_distance,
            'agent_final_goal_distances': [float(v) if v is not None else None for v in agent_final_goal_distances],
            'min_goal_distance': min_goal_distance,
            'agent_min_goal_distances': [float(v) if v is not None else None for v in agent_min_goal_distances],
            'agent_first_reach_steps': [int(v) if v is not None else None for v in agent_first_reach_steps],
            'path_efficiency': path_efficiency,
            'agent_path_efficiencies': agent_path_efficiencies,
            # 🔧 新增：穿透深度统计
            'penetration_stat': penetration_stat,
            'vis_context': vis_context,
        }
        
    def generate_visualization(self, episode_data, is_best=False):
        """生成可视化结果，仿照1.0版本
        
        Args:
            episode_data: 回合数据
            is_best: 是否为最佳回合（用于文件名标识）
        """
        if not self.visualizer or not episode_data['trajectory']:
            return {}
            
        print("🎨 正在生成可视化结果...")
        
        # 创建保存目录
        os.makedirs(self.args.save_viz_path, exist_ok=True)
        
        # 仿照1.0版本的可视化参数
        viz_args = argparse.Namespace(
            save_gifs=True,
            save_trajectory_images=True,
            exp_name=os.path.basename(self.args.save_viz_path),
            save_dir=self.args.save_viz_path
        )
        
        generated_files = {}
        try:
            vis_context = episode_data.get('vis_context')
            scenario_for_viz = self.scenario
            goal_positions_snapshot = None
            if isinstance(vis_context, dict):
                scenario_for_viz = argparse.Namespace(**vis_context)
                goal_positions_snapshot = {
                    'goal_pos': vis_context.get('goal_pos'),
                    'agent_goals': vis_context.get('agent_goals', []) or [],
                }
                snapshot_prefix = f"episode_{int(episode_data.get('episode', 0)) + 1:03d}"
                snapshot_artifacts = _save_vis_context_snapshot_artifacts(
                    self.args.save_viz_path,
                    snapshot_prefix,
                    vis_context,
                )
                if snapshot_artifacts:
                    generated_files.update({
                        'terrain_snapshot_json_path': snapshot_artifacts.get('terrain_snapshot_json_path'),
                        'terrain_npy_path': snapshot_artifacts.get('terrain_npy_path'),
                    })

            def _extract_action_head_norms(history):
                if not history:
                    return []
                norms = []
                for step_actions in history:
                    step_norms = []
                    for action in step_actions:
                        try:
                            arr = np.asarray(action, dtype=np.float32).reshape(-1)
                            if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
                                step_norms.append(float(np.linalg.norm(arr[:3])))
                            else:
                                step_norms.append(None)
                        except Exception:
                            step_norms.append(None)
                    norms.append(step_norms)
                return norms

            def _extract_xy_action_norms(history):
                if not history:
                    return []
                norms = []
                for step_actions in history:
                    step_norms = []
                    for action in step_actions:
                        try:
                            arr = np.asarray(action, dtype=np.float32).reshape(-1)
                            if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
                                step_norms.append(float(np.linalg.norm(arr[:2])))
                            else:
                                step_norms.append(None)
                        except Exception:
                            step_norms.append(None)
                    norms.append(step_norms)
                return norms

            def _extract_z_component(history):
                if not history:
                    return []
                values = []
                for step_actions in history:
                    step_values = []
                    for action in step_actions:
                        try:
                            arr = np.asarray(action, dtype=np.float32).reshape(-1)
                            step_values.append(float(arr[2]) if arr.size >= 3 and np.isfinite(arr[2]) else None)
                        except Exception:
                            step_values.append(None)
                    values.append(step_values)
                return values

            def _extract_speed_history(velocity_history):
                if not velocity_history:
                    return []
                speed_history = []
                for step_velocities in velocity_history:
                    step_speeds = []
                    for vel in step_velocities:
                        try:
                            arr = np.asarray(vel, dtype=np.float32).reshape(-1)
                            if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
                                step_speeds.append(float(np.linalg.norm(arr[:3])))
                            else:
                                step_speeds.append(None)
                        except Exception:
                            step_speeds.append(None)
                    speed_history.append(step_speeds)
                return speed_history

            def _plot_series(ax, history, title, ylabel, *, styles=None, threshold=None):
                if not history:
                    ax.set_title(title)
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
                    return
                steps = np.arange(len(history), dtype=np.int32)
                n_agents = max((len(step_values) for step_values in history), default=0)
                colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange', 'tab:brown']
                styles = styles or {}
                for agent_idx in range(n_agents):
                    series = []
                    for step_values in history:
                        value = step_values[agent_idx] if agent_idx < len(step_values) else None
                        series.append(np.nan if value is None else float(value))
                    label = styles.get(agent_idx, {}).get('label', f'Agent {agent_idx}')
                    linestyle = styles.get(agent_idx, {}).get('linestyle', '-')
                    alpha = styles.get(agent_idx, {}).get('alpha', 0.9)
                    color = styles.get(agent_idx, {}).get('color', colors[agent_idx % len(colors)])
                    ax.plot(steps, series, label=label, linestyle=linestyle, alpha=alpha, color=color, linewidth=1.5)
                if threshold is not None:
                    ax.axhline(float(threshold), color='black', linestyle='--', linewidth=1.0, alpha=0.8, label='Threshold')
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.set_xlabel('Step')
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, ncol=2)

            def _generate_control_diagnostics_plot():
                raw_history = episode_data.get('actions_history') or []
                executed_history = episode_data.get('executed_actions_history') or []
                velocity_history = episode_data.get('velocity_history') or []
                goal_distance_history = episode_data.get('goal_distance_history') or []
                if not raw_history and not executed_history and not velocity_history and not goal_distance_history:
                    return None

                diagnostics_path = os.path.join(
                    self.args.save_viz_path,
                    f"trajectory_ep{episode_num}_control_diagnostics.png"
                )
                fig, axes = plt.subplots(2, 2, figsize=(16, 10))

                speed_history = _extract_speed_history(velocity_history)
                _plot_series(
                    axes[0, 0],
                    speed_history,
                    'Speed Magnitude',
                    'm/s',
                    threshold=getattr(self.args, 'agent_max_speed', None),
                )

                _plot_series(
                    axes[0, 1],
                    goal_distance_history,
                    'Distance To Goal',
                    'm',
                    threshold=getattr(self.args, 'success_distance_threshold', None),
                )

                raw_xy_history = _extract_xy_action_norms(raw_history)
                executed_xy_history = _extract_xy_action_norms(executed_history)
                xy_history = []
                for idx in range(max(len(raw_xy_history), len(executed_xy_history))):
                    step_values = []
                    raw_step = raw_xy_history[idx] if idx < len(raw_xy_history) else []
                    exec_step = executed_xy_history[idx] if idx < len(executed_xy_history) else []
                    n_agents = max(len(raw_step), len(exec_step))
                    for agent_idx in range(n_agents):
                        raw_v = raw_step[agent_idx] if agent_idx < len(raw_step) else None
                        exec_v = exec_step[agent_idx] if agent_idx < len(exec_step) else None
                        step_values.extend([raw_v, exec_v])
                    xy_history.append(step_values)
                xy_styles = {}
                colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange', 'tab:brown']
                n_agents_xy = max(
                    max((len(step) for step in raw_xy_history), default=0),
                    max((len(step) for step in executed_xy_history), default=0),
                )
                for agent_idx in range(n_agents_xy):
                    xy_styles[2 * agent_idx] = {
                        'label': f'Agent {agent_idx} raw XY',
                        'linestyle': '--',
                        'alpha': 0.6,
                        'color': colors[agent_idx % len(colors)],
                    }
                    xy_styles[2 * agent_idx + 1] = {
                        'label': f'Agent {agent_idx} exec XY',
                        'linestyle': '-',
                        'alpha': 0.9,
                        'color': colors[agent_idx % len(colors)],
                    }
                _plot_series(axes[1, 0], xy_history, 'XY Action Norm (Raw vs Executed)', 'norm', styles=xy_styles)

                raw_z_history = _extract_z_component(raw_history)
                executed_z_history = _extract_z_component(executed_history)
                z_history = []
                for idx in range(max(len(raw_z_history), len(executed_z_history))):
                    step_values = []
                    raw_step = raw_z_history[idx] if idx < len(raw_z_history) else []
                    exec_step = executed_z_history[idx] if idx < len(executed_z_history) else []
                    n_agents = max(len(raw_step), len(exec_step))
                    for agent_idx in range(n_agents):
                        raw_v = raw_step[agent_idx] if agent_idx < len(raw_step) else None
                        exec_v = exec_step[agent_idx] if agent_idx < len(exec_step) else None
                        step_values.extend([raw_v, exec_v])
                    z_history.append(step_values)
                z_styles = {}
                for agent_idx in range(n_agents_xy):
                    z_styles[2 * agent_idx] = {
                        'label': f'Agent {agent_idx} raw Z',
                        'linestyle': '--',
                        'alpha': 0.6,
                        'color': colors[agent_idx % len(colors)],
                    }
                    z_styles[2 * agent_idx + 1] = {
                        'label': f'Agent {agent_idx} exec Z',
                        'linestyle': '-',
                        'alpha': 0.9,
                        'color': colors[agent_idx % len(colors)],
                    }
                _plot_series(axes[1, 1], z_history, 'Z Action (Raw vs Executed)', 'action', styles=z_styles)

                reward_value = float(episode_data.get('reward', 0.0))
                fig.suptitle(
                    f'Episode {episode_num} Control Diagnostics | reward={reward_value:.1f}',
                    fontsize=15,
                    fontweight='bold',
                )
                plt.tight_layout()
                plt.savefig(diagnostics_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                return diagnostics_path

            # 生成轨迹图像（同时传入场景提取的目标信息，避免空字典提示）
            episode_num = int(episode_data.get('episode', 0)) + 1
            terrain_level = episode_data.get('terrain_complexity_level', 'unknown')
            image_path = os.path.join(
                self.args.save_viz_path,
                f"trajectory_ep{episode_num}_level{terrain_level}_r{episode_data['reward']:.0f}.png"
            )
            goal_positions_img = None
            try:
                if goal_positions_snapshot is not None:
                    goal_positions_img = goal_positions_snapshot
                else:
                    goal_positions_img = self._get_goal_positions_from_scenario()
            except Exception:
                goal_positions_img = None
            
            save_trajectory_png = _env_flag('SAVE_EVAL_TRAJECTORY_PNG', True)
            save_actor_sequence = _env_flag('SAVE_EVAL_ACTOR_SEQUENCE', False)
            save_control_diagnostics = _env_flag('SAVE_EVAL_CONTROL_DIAGNOSTICS', False)

            # 原有轨迹图（只有显式开启调试产物时才生成 actor 时序图）
            actor_outputs_history = None
            if save_actor_sequence and 'actions_history' in episode_data and episode_data['actions_history']:
                try:
                    # 将动作历史转换为numpy数组格式（与训练时一致）
                    # 格式: (steps, n_agents, action_dim)
                    actor_outputs_history = np.array(episode_data['actions_history'])
                    print(f"✅ 检测到动作历史数据，长度: {len(actor_outputs_history)} 步")
                    
                    # 🔧 关键修复：评估时每步都记录，但时序图期望采样数据（每10步一个点）
                    # 为了与训练时一致，需要按10步间隔采样
                    # 注意：os已在文件开头导入，不需要重复导入
                    actor_output_interval = int(os.getenv('ACTOR_OUTPUT_SAMPLE_INTERVAL', '10'))
                    if actor_output_interval <= 0:
                        actor_output_interval = 10
                    
                    # 按间隔采样，与训练时保持一致
                    if len(actor_outputs_history) > actor_output_interval:
                        sampled_indices = list(range(0, len(actor_outputs_history), actor_output_interval))
                        actor_outputs_history = actor_outputs_history[sampled_indices]
                        print(f"✅ 动作历史已采样：原始{len(episode_data['actions_history'])}步 → 采样后{len(actor_outputs_history)}步（间隔={actor_output_interval}）")
                except Exception as e:
                    print(f"⚠️ 动作历史数据转换失败: {e}")
                    import traceback
                    traceback.print_exc()
                    actor_outputs_history = None
            
            # 🔧 关键修复：传递env_instance参数，确保能正确绘制地形和目标位置
            # 如果没有env_instance，visualizer无法获取正确的地形数据，导致图片为空
            # 🚨 关键修复：确保goal_positions不为None，如果获取失败则使用场景中的目标位置
            if goal_positions_img is None:
                try:
                    if goal_positions_snapshot is not None:
                        goal_positions_img = goal_positions_snapshot
                    else:
                        goal_positions_img = self._get_goal_positions_from_scenario()
                    if goal_positions_img and goal_positions_img.get('goal_pos') is None:
                        # 如果仍然没有目标位置，尝试从world.landmarks获取
                        if hasattr(self.world, 'landmarks') and len(self.world.landmarks) > 0:
                            landmark = self.world.landmarks[0]
                            if hasattr(landmark, 'state') and hasattr(landmark.state, 'p_pos'):
                                # 🚨 关键修复：保留已有的agent_goals，只补充中央目标
                                if goal_positions_img is None:
                                    goal_positions_img = {'goal_pos': None, 'agent_goals': []}
                                goal_positions_img['goal_pos'] = landmark.state.p_pos.tolist()
                                print(f"✅ 从world.landmarks获取目标位置: {goal_positions_img['goal_pos']}")
                                print(f"✅ 保留各智能体目标: {len(goal_positions_img.get('agent_goals', []))}个")
                except Exception as e:
                    print(f"⚠️  获取目标位置失败: {e}")
            
            if save_trajectory_png:
                generated_files['image_path'] = image_path
                self.visualizer.generate_trajectory_image(
                        trajectories=episode_data['trajectory'],
                        scenario=scenario_for_viz,
                        save_path=image_path,
                        episode_num=episode_data['episode'],
                        reward=episode_data['reward'],
                        episode_type='evaluation',
                        goal_positions=goal_positions_img,
                        env_instance=None if goal_positions_snapshot is not None else self.env,
                        actor_outputs_history=actor_outputs_history,  # 🔧 传入动作历史（如果存在）
                        title_step_note=f"Total Steps: {int(episode_data.get('steps', len(episode_data.get('trajectory', []))))}"
                )
            elif save_actor_sequence and actor_outputs_history is not None and len(actor_outputs_history) > 0:
                try:
                    # 时序图本质上与静态3D轨迹图是独立分析产物；当用户明确关闭PNG时，仍然保留时序图。
                    self.visualizer._generate_actor_outputs_sequence_image(
                        actor_outputs_history=actor_outputs_history,
                        episode_num=episode_num,
                        reward=float(episode_data.get('reward', 0.0)),
                        episode_type='best' if is_best else 'evaluation',
                        original_save_path=image_path,
                    )
                except Exception as actor_seq_err:
                    print(f"⚠️ 独立Actor时序图生成失败: {actor_seq_err}")
            else:
                print("⏭️ 跳过静态PNG轨迹图生成（SAVE_EVAL_TRAJECTORY_PNG=0）")

            actor_sequence_path = f"{os.path.splitext(image_path)[0]}_actor_sequence.png"
            if save_actor_sequence and os.path.exists(actor_sequence_path):
                generated_files['actor_sequence_path'] = actor_sequence_path

            diagnostics_path = None
            if save_control_diagnostics:
                diagnostics_path = _generate_control_diagnostics_plot()
                if diagnostics_path and os.path.exists(diagnostics_path):
                    generated_files['control_diagnostics_path'] = diagnostics_path

            # 叠加障碍/地形等信息的增强版图（默认禁用）
            enable_overlay = getattr(self.args, 'enable_overlay', False) and not getattr(self.args, 'disable_overlay', False)
            if enable_overlay:
                overlay_path = os.path.join(
                    self.args.save_viz_path,
                    f"trajectory_ep{episode_num}_level{terrain_level}_overlay.png"
                )
                self._generate_overlay_image(episode_data, overlay_path)
                generated_files['overlay_path'] = overlay_path
                print(f"✅ Overlay图片已保存: {overlay_path}")
            else:
                print(f"⏭️ 跳过overlay图片生成（已禁用）")
            
            # 默认每个回合都生成交互式HTML；可用 SAVE_INTERACTIVE_TRAJ=0 关闭
            enable_interactive_traj = os.getenv('SAVE_INTERACTIVE_TRAJ', '1').lower() in ('1','true','yes','on')
            enable_html = enable_interactive_traj
            html_path = None
            if enable_html:
                html_path = os.path.join(
                    self.args.save_viz_path,
                    f"trajectory_ep{episode_num}_interactive.html"
                )
                generated_files['html_path'] = html_path
                # 🚨 关键修复：确保交互式HTML中显示目标点
                goal_positions_html = None
                try:
                    # 优先使用之前获取的目标位置
                    if goal_positions_img is not None:
                        goal_positions_html = goal_positions_img
                    else:
                        # 尝试从环境获取
                        if hasattr(self.env, 'get_goal_positions'):
                            goal_positions_html = self.env.get_goal_positions(0)
                        if not isinstance(goal_positions_html, dict) or goal_positions_html.get('goal_pos') is None:
                            goal_positions_html = self._get_goal_positions_from_scenario()
                        # 如果仍然没有，尝试从world.landmarks获取
                        if (not goal_positions_html or goal_positions_html.get('goal_pos') is None) and hasattr(self.world, 'landmarks') and len(self.world.landmarks) > 0:
                            landmark = self.world.landmarks[0]
                            if hasattr(landmark, 'state') and hasattr(landmark.state, 'p_pos'):
                                # 🚨 关键修复：保留已有的agent_goals，只补充中央目标
                                if goal_positions_html is None:
                                    goal_positions_html = {'goal_pos': None, 'agent_goals': []}
                                goal_positions_html['goal_pos'] = landmark.state.p_pos.tolist()
                                print(f"✅ 交互式HTML: 从world.landmarks获取目标位置: {goal_positions_html['goal_pos']}")
                                print(f"✅ 保留各智能体目标: {len(goal_positions_html.get('agent_goals', []))}个")
                except Exception as e:
                    print(f"⚠️  获取交互式HTML目标位置失败: {e}")
                    goal_positions_html = None
                
                self.visualizer.generate_trajectory_interactive(
                    trajectories=episode_data['trajectory'],
                    save_path=html_path,
                    title=f"Evaluation Episode {episode_num} (reward={episode_data['reward']:.1f})",
                    goal_positions=goal_positions_html,
                    scenario=scenario_for_viz,
                    env_instance=None if goal_positions_snapshot is not None else self.env
                )

            # GIF仍只在显式允许时生成，默认由shell脚本禁用
            if is_best and len(episode_data['trajectory']) > 10 and not getattr(self.args, 'disable_gif', False):
                gif_path = os.path.join(
                    self.args.save_viz_path,
                    f"best_trajectory_ep{episode_num}_animation.gif"
                )
                generated_files['gif_path'] = gif_path
                self.visualizer.generate_trajectory_gif(
                    trajectories=episode_data['trajectory'],
                    scenario=scenario_for_viz,
                    save_path=gif_path,
                    episode_num=episode_data['episode'],
                    reward=episode_data['reward'],
                    goal_positions=goal_positions_img,
                    gif_max_frames=getattr(self.args, 'gif_max_frames', 60)
                )
            
            # 如禁用HTML
            if not enable_html:
                print(f"⏭️ 跳过HTML轨迹图生成（已禁用）")
            
            print(f"✅ 可视化结果已保存到: {self.args.save_viz_path}")
            return generated_files
            
        except Exception as e:
            print(f"⚠️ 可视化生成失败: {e}")
            traceback.print_exc()
            return generated_files
            
    def run_evaluation(self):
        """运行完整评估流程"""
        print("="*60)
        print("🔬 MADDPG模型评估开始")
        print("="*60)
        
        # 加载模型
        self.load_model()
        
        # 评估统计
        all_rewards = []
        all_episodes_data = []
        
        # 🔧 新增：跟踪最佳回合（与训练脚本逻辑一致）
        best_reward = -np.inf
        best_episode = 0
        best_episode_data = None
        best_trajectory = None
        best_actor_outputs_history = None
        best_success_reward = -np.inf
        best_success_episode = None
        best_success_episode_data = None
        best_success_trajectory = None
        best_success_actor_outputs_history = None
        save_all_episode_visualizations = os.getenv('SAVE_EVAL_ALL_EPISODES', '1').lower() in ('1','true','yes','on')
        episode_visualizations = []
        try:
            quiet_output = os.getenv("QUIET_OUTPUT", "1").lower() in ("1", "true", "yes", "on")
        except Exception:
            quiet_output = True
        
        # 🔧 新增：检查是否保存最佳回合可视化（与训练脚本一致）
        enable_best_traj = os.getenv('SAVE_BEST_TRAJ', '1').lower() in ('1','true','yes','on')
        enable_interactive = os.getenv('SAVE_INTERACTIVE_TRAJ', '1').lower() in ('1','true','yes','on')
        save_team_success_html = _env_flag('SAVE_TEAM_SUCCESS_HTML', False)
        persist_episode_trajectories = _env_flag('SAVE_EVAL_TRAJECTORY_JSON', True)
        save_actor_sequence = _env_flag('SAVE_EVAL_ACTOR_SEQUENCE', False)
        save_control_diagnostics = _env_flag('SAVE_EVAL_CONTROL_DIAGNOSTICS', False)

        def _normalize_generated_files(generated_files):
            if not isinstance(generated_files, dict):
                return {}
            normalized = {}
            for key, value in generated_files.items():
                if value is None:
                    continue
                try:
                    normalized[key] = os.path.basename(str(value))
                except Exception:
                    normalized[key] = str(value)
            return normalized

        def _copy_alias(source_path, alias_name):
            if not source_path:
                return None
            try:
                source_str = str(source_path)
                if not os.path.exists(source_str):
                    return None
                alias_path = os.path.join(self.args.save_viz_path, alias_name)
                shutil.copyfile(source_str, alias_path)
                return alias_path
            except Exception as alias_e:
                print(f"⚠️  复制可视化别名失败 ({alias_name}): {alias_e}")
                return None

        if not quiet_output:
            print(f"\n📊 开始评估 {self.args.eval_episodes} 个回合")
            print(f"🏔️ 地形模式: {'随机地形' if self.args.random_terrain else '固定地形'}")
            if self.args.terrain_complexity_level is not None:
                print(f"🏔️ 地形复杂度等级: {self.args.terrain_complexity_level}")
            else:
                print(f"🏔️ 地形复杂度等级: 随机选择 (1-4)")
            print(f"📸 保存每回合可视化: {'是' if save_all_episode_visualizations else '否'}")
            print(f"🏆 最佳回合单独可视化: {'是' if (enable_best_traj and not save_all_episode_visualizations) else '否'}")
            print(f"📊 保存交互式HTML: {'是' if enable_interactive else '否'}")
        
        # 🔧 关键修复：从环境变量读取地形种子序列，确保所有评估模式使用相同的地图顺序
        terrain_seed_sequence = None
        terrain_seed_str = os.getenv('TERRAIN_SEED_SEQUENCE', '')
        if terrain_seed_str:
            try:
                terrain_seed_sequence = [int(s.strip()) for s in terrain_seed_str.split(',') if s.strip()]
                if not quiet_output:
                    print(f"🔧 使用预定义地形种子序列（共{len(terrain_seed_sequence)}个）: {terrain_seed_sequence[:5]}... (前5个)")
            except Exception as e:
                print(f"⚠️  解析地形种子序列失败: {e}，将使用随机地形")
                terrain_seed_sequence = None
        terrain_variant_seed_sequence = None
        terrain_variant_seed_str = os.getenv('TERRAIN_VARIANT_SEED_SEQUENCE', '')
        if terrain_variant_seed_str:
            try:
                terrain_variant_seed_sequence = [int(s.strip()) for s in terrain_variant_seed_str.split(',') if s.strip()]
                if not quiet_output:
                    print(
                        f"🔧 使用预定义同源扰动种子序列（共{len(terrain_variant_seed_sequence)}个）: "
                        f"{terrain_variant_seed_sequence[:5]}... (前5个)"
                    )
            except Exception as e:
                print(f"⚠️  解析同源扰动种子序列失败: {e}，将忽略 variant 序列")
                terrain_variant_seed_sequence = None
        obstacle_seed_sequence = None
        obstacle_seed_str = os.getenv('OBSTACLE_SEED_SEQUENCE', '')
        if obstacle_seed_str:
            try:
                obstacle_seed_sequence = [int(s.strip()) for s in obstacle_seed_str.split(',') if s.strip()]
                if not quiet_output:
                    print(
                        f"🔧 使用共享障碍种子序列（共{len(obstacle_seed_sequence)}个）: "
                        f"{obstacle_seed_sequence[:5]}... (前5个)"
                    )
            except Exception as e:
                print(f"⚠️  解析共享障碍种子序列失败: {e}，将忽略 obstacle 序列")
                obstacle_seed_sequence = None

        terrain_family_override = os.getenv('POST_EVAL_TERRAIN_FAMILY', '').strip()
        position_family_override = os.getenv('POST_EVAL_POSITION_FAMILY', '').strip()
        runtime_random_terrain = (
            _env_flag('RANDOM_TERRAIN', bool(getattr(self.args, 'random_terrain', False)))
            or _sequence_implies_random_terrain(terrain_seed_sequence, terrain_variant_seed_sequence)
        )
        setup_semi_random_terrain = _env_flag('SEMI_RANDOM_TERRAIN', False)
        effective_use_quadrotor_dynamics = getattr(self.args, 'use_quadrotor_dynamics', None)
        if effective_use_quadrotor_dynamics is None:
            effective_use_quadrotor_dynamics = _env_flag('USE_QUADROTOR_DYNAMICS', False)
        effective_simulation_dt = _finite_float_or_none(getattr(self.args, 'simulation_dt', None))
        if effective_simulation_dt is None:
            effective_simulation_dt = _env_float('SIMULATION_DT', 0.08)
        effective_z_action_bias = _finite_float_or_none(getattr(self.args, 'z_action_bias', None))
        if effective_z_action_bias is None:
            effective_z_action_bias = _env_float('Z_ACTION_BIAS', 0.0)
        effective_quadrotor_attitude_response_time = _finite_float_or_none(
            getattr(self.args, 'quadrotor_attitude_response_time', None)
        )
        if effective_quadrotor_attitude_response_time is None:
            effective_quadrotor_attitude_response_time = _env_float('QUADROTOR_ATTITUDE_RESPONSE_TIME', 0.0)
        effective_quadrotor_psi_cmd = _finite_float_or_none(getattr(self.args, 'quadrotor_psi_cmd', None))
        if effective_quadrotor_psi_cmd is None:
            effective_quadrotor_psi_cmd = _env_float('QUADROTOR_PSI_CMD', 0.0)
        evaluation_setup = {
            'terrain_family': terrain_family_override or (
                'similar_unseen' if setup_semi_random_terrain else (
                    'train_match'
                    if not runtime_random_terrain
                    else 'random_unseen'
                )
            ),
            'semi_random_terrain': setup_semi_random_terrain,
            'terrain_seed': _env_int(
                'TERRAIN_BASE_SEED' if setup_semi_random_terrain else 'SCENARIO_SEED',
                getattr(self.args, 'terrain_seed', 67) if getattr(self.args, 'terrain_seed', None) is not None else 67,
            ),
            'terrain_base_seed': _env_int('TERRAIN_BASE_SEED', getattr(self.args, 'terrain_seed', 67) if getattr(self.args, 'terrain_seed', None) is not None else 67),
            'peak_jitter_range': _env_float('PEAK_JITTER_RANGE', 15.0),
            'peak_center_jitter_range': _env_float('PEAK_CENTER_JITTER_RANGE', min(_env_float('PEAK_JITTER_RANGE', 15.0), 3.0)),
            'peak_height_jitter_ratio_min': _env_float('PEAK_HEIGHT_JITTER_RATIO_MIN', 0.20),
            'peak_height_jitter_ratio_max': _env_float('PEAK_HEIGHT_JITTER_RATIO_MAX', 0.40),
            'peak_height_max_scale': _env_float('PEAK_HEIGHT_MAX_SCALE', 1.30),
            'terrain_variant_noise_ratio': _env_float('TERRAIN_VARIANT_NOISE_RATIO', 0.15),
            'position_family': position_family_override or os.getenv('HELDOUT_POSITION_MODE', 'train_match'),
            'reference_positions_file': os.getenv('HELDOUT_REFERENCE_POSITIONS_FILE', ''),
            'start_center_jitter': _env_float('HELDOUT_START_CENTER_JITTER', 12.0),
            'agent_local_jitter': _env_float('HELDOUT_AGENT_LOCAL_JITTER', 3.0),
            'goal_region_radius': _env_float('HELDOUT_GOAL_REGION_RADIUS', 18.0),
            'terrain_complexity': int(getattr(self.args, 'terrain_complexity_level', 0) or 0),
            'map_size': _env_int('MAP_SIZE', 200),
            'mountain_min_distance': _env_int('MOUNTAIN_MIN_DISTANCE', 55),
            'random_terrain': bool(runtime_random_terrain),
            'shared_obstacle_seed_sequence': bool(obstacle_seed_sequence),
            'use_dynamic_obstacles': _env_flag('USE_DYNAMIC_OBSTACLES', True),
            'use_quadrotor_dynamics': bool(effective_use_quadrotor_dynamics),
            'gravity': _finite_float_or_none(getattr(self.args, 'gravity', None)),
            'control_accel_gain': _finite_float_or_none(getattr(self.args, 'control_accel_gain', None)),
            'damping': _finite_float_or_none(getattr(self.args, 'damping', None)),
            'agent_max_speed': _finite_float_or_none(getattr(self.args, 'agent_max_speed', None)),
            'agent_accel': _finite_float_or_none(getattr(self.args, 'agent_accel', None)),
            'action_range_x': _finite_float_or_none(getattr(self.args, 'action_range_x', None)),
            'action_range_y': _finite_float_or_none(getattr(self.args, 'action_range_y', None)),
            'action_range_z': _finite_float_or_none(getattr(self.args, 'action_range_z', None)),
            'simulation_dt': effective_simulation_dt,
            'z_action_bias': effective_z_action_bias,
            'quadrotor_attitude_response_time': effective_quadrotor_attitude_response_time,
            'quadrotor_psi_cmd': effective_quadrotor_psi_cmd,
            'episode_length': _safe_int(getattr(self.args, 'episode_length', None)),
            'requested_episode_length_multiplier': _env_float(
                'EVAL_EPISODE_LENGTH_MULTIPLIER',
                _env_float('POST_EVAL_EPISODE_LENGTH_MULTIPLIER', 1.0),
            ),
            'action_force_ratio': _finite_float_or_none(getattr(self.args, 'action_force_ratio', None)),
            'action_force_ratio_source': (
                'forced_override'
                if getattr(self.args, 'force_action_force_ratio', None) is not None
                else (
                    getattr(self, '_selected_checkpoint_force_ratio_label', None)
                    or 'checkpoint_variant'
                )
            ),
            'checkpoint_action_force_ratio': _finite_float_or_none(
                getattr(self, '_selected_checkpoint_force_ratio', None)
            ),
            'forced_action_force_ratio': _finite_float_or_none(
                getattr(self, '_forced_eval_action_force_ratio', None)
            ),
        }
        
        for episode in range(self.args.eval_episodes):
            if not quiet_output:
                print(f"\n🚀 开始评估回合 {episode + 1}/{self.args.eval_episodes}")
            
            # 为每个回合随机选择地形复杂度等级（如果未指定）
            if self.args.terrain_complexity_level is None:
                terrain_level = np.random.randint(1, 5)  # 1-4
                if not quiet_output:
                    print(f"🎲 随机选择地形复杂度等级: {terrain_level}")
            else:
                terrain_level = self.args.terrain_complexity_level
                if not quiet_output:
                    print(f"🏔️ 使用指定地形复杂度等级: {terrain_level}")
            self.scenario.terrain_complexity_level = terrain_level

            terrain_info = self._prepare_episode_terrain(
                episode,
                terrain_seed_sequence,
                terrain_variant_seed_sequence,
                obstacle_seed_sequence,
            )
            terrain_seed = terrain_info.get('terrain_seed') if isinstance(terrain_info, dict) else terrain_info
            terrain_variant_seed = (
                terrain_info.get('terrain_variant_seed') if isinstance(terrain_info, dict) else None
            )
            obstacle_seed = (
                terrain_info.get('obstacle_seed') if isinstance(terrain_info, dict) else None
            )
            if terrain_seed is not None:
                if not quiet_output:
                    print(f"🗺️ 当前回合地形种子: base={terrain_seed}, variant={terrain_variant_seed}, obstacle={obstacle_seed}")
            
            # 🔧 关键修复：添加异常处理，确保单个回合失败不会导致整个评估失败
            try:
                episode_data = self.evaluate_single_episode(episode)
                if episode_data is None:
                    print(f"⚠️  回合 {episode + 1} 评估返回空数据，跳过")
                    continue
                
                # 验证episode_data是否包含必要字段
                if 'reward' not in episode_data or 'trajectory' not in episode_data:
                    print(f"⚠️  回合 {episode + 1} 评估数据不完整，跳过")
                    print(f"    episode_data keys: {list(episode_data.keys())}")
                    continue
                
                episode_data['terrain_complexity_level'] = terrain_level
                episode_data['terrain_seed'] = terrain_seed
                episode_data['terrain_variant_seed'] = terrain_variant_seed
                episode_data['obstacle_seed'] = obstacle_seed
                all_rewards.append(episode_data['reward'])
                
                # 🔧 修改：跟踪最佳回合（与训练脚本逻辑一致）
                if episode_data['reward'] > best_reward:
                    best_reward = episode_data['reward']
                    best_episode = episode
                    best_episode_data = episode_data.copy()
                    best_trajectory = episode_data.get('trajectory', [])
                    best_actor_outputs_history = episode_data.get('actions_history', None)
                    print(f"✅ 更新最佳回合: Episode {episode + 1}, Reward = {best_reward:.2f}")

                if episode_data.get('team_success', 0) == 1 and episode_data['reward'] > best_success_reward:
                    best_success_reward = episode_data['reward']
                    best_success_episode = episode
                    best_success_episode_data = episode_data.copy()
                    best_success_trajectory = episode_data.get('trajectory', [])
                    best_success_actor_outputs_history = episode_data.get('actions_history', None)
                    print(f"✅ 更新最佳成功回合: Episode {episode + 1}, Reward = {best_success_reward:.2f}")

                # 评估汇总只需要轻量统计字段；较大的轨迹/动作历史仅保留给最佳回合和显式可视化流程。
                stored_episode_data = dict(episode_data)
                if not persist_episode_trajectories:
                    stored_episode_data['trajectory'] = []
                if not save_actor_sequence:
                    stored_episode_data['actions_history'] = []
                if not save_control_diagnostics:
                    stored_episode_data['executed_actions_history'] = []
                    stored_episode_data['velocity_history'] = []
                    stored_episode_data['goal_distance_history'] = []
                stored_episode_data['vis_context'] = None
                all_episodes_data.append(stored_episode_data)
                
                if save_all_episode_visualizations and not self.args.disable_visualization:
                    try:
                        generated_episode_files = self.generate_visualization(episode_data, is_best=False) or {}
                        normalized_files = _normalize_generated_files(generated_episode_files)
                        if normalized_files:
                            episode_data['visualization_files'] = normalized_files
                            episode_visualizations.append(
                                {
                                    'episode': episode_data.get('episode'),
                                    'reward': float(episode_data.get('reward', 0.0)),
                                    'team_success': int(episode_data.get('team_success', episode_data.get('success', 0))),
                                    'files': normalized_files,
                                }
                            )
                    except Exception as viz_e:
                        print(f"⚠️  回合 {episode + 1} 可视化生成失败: {viz_e}")
                        traceback.print_exc()
            except Exception as ep_e:
                print(f"❌ 回合 {episode + 1} 评估失败: {ep_e}")
                traceback.print_exc()
                # 继续下一个回合，不中断整个评估流程
                continue
                
        # 评估总结
        print("\n" + "="*60)
        print("📈 评估结果总结")
        print("="*60)
        
        # 🔧 关键修复：检查是否有有效的评估结果
        if len(all_rewards) == 0:
            print("❌ 警告: 没有成功完成任何评估回合！")
            print("   可能的原因:")
            print("   1. 模型加载失败（维度不匹配）")
            print("   2. 环境初始化失败")
            print("   3. 势场修正配置不匹配")
            print("   4. 评估过程中出现异常")
            print("")
            print("   建议检查:")
            print("   - 模型路径是否正确")
            print("   - 训练配置（use_pf_feature, use_fr_feature）是否与评估时一致")
            print("   - DELTA_*参数是否与训练时一致（传统APF需要DELTA_*=0.0）")
            print("   - 查看上方的错误信息")
            # 即使没有结果，也保存一个空的评估结果文件，便于调试
            results = {
                'model_path': self.args.load_model_path,
                'scenario': self.args.scenario_name,
                'episodes': 0,
                'avg_reward': None,
                'std_reward': None,
                'max_reward': None,
                'min_reward': None,
                'all_rewards': [],
                'summary': _build_evaluation_summary([], [], getattr(self.args, 'collision_distance_threshold', None)),
                'evaluation_setup': evaluation_setup,
                'episode_details': [],
                'visualization_artifacts': {},
                'terrain_seed_sequence': terrain_seed_sequence or [],
                'terrain_variant_seed_sequence': terrain_variant_seed_sequence or [],
                'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'error': 'No episodes completed successfully'
            }
            results_path = os.path.join(self.args.save_viz_path, 'evaluation_results.json')
            os.makedirs(self.args.save_viz_path, exist_ok=True)
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"⚠️  已保存空的评估结果文件: {results_path}")
            return results
        
        avg_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        max_reward = np.max(all_rewards)
        min_reward = np.min(all_rewards)
        summary = _build_evaluation_summary(
            all_rewards,
            all_episodes_data,
            collision_distance_threshold=getattr(self.args, 'collision_distance_threshold', None),
        )
        
        print(f"平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"最高奖励: {max_reward:.2f}")
        print(f"最低奖励: {min_reward:.2f}")
        print(f"总回合数: {len(all_rewards)}")
        if summary.get('team_success_rate') is not None:
            print(f"团队成功率: {summary['team_success_rate']:.3f}")
        if summary.get('avg_team_total_path_length') is not None:
            print(f"平均团队路径长度: {summary['avg_team_total_path_length']:.2f}")
        if summary.get('avg_team_total_path_length_success_only') is not None:
            print(f"成功回合平均团队路径长度: {summary['avg_team_total_path_length_success_only']:.2f}")
        if summary.get('avg_arrival_step_success_only') is not None:
            print(f"成功回合平均首达步数: {summary['avg_arrival_step_success_only']:.1f}")
        if summary.get('avg_collision_count') is not None:
            print(f"平均碰撞次数: {summary['avg_collision_count']:.2f}")

        visualization_artifacts = {
            'episode_visualizations': episode_visualizations,
        }
        best_generated_files = {}
        if best_episode_data is not None and not self.args.disable_visualization and (save_all_episode_visualizations or enable_best_traj):
            try:
                best_episode_for_viz = best_episode_data.copy()
                best_episode_for_viz['trajectory'] = best_trajectory
                if best_actor_outputs_history is not None:
                    best_episode_for_viz['actions_history'] = best_actor_outputs_history
                if save_all_episode_visualizations:
                    best_episode_num = int(best_episode_for_viz.get('episode', 0)) + 1
                    best_image_path = os.path.join(
                        self.args.save_viz_path,
                        f"trajectory_ep{best_episode_num}_level{best_episode_for_viz.get('terrain_complexity_level', 'unknown')}_r{best_episode_for_viz['reward']:.0f}.png"
                    )
                    best_generated_files = {
                        'image_path': best_image_path,
                        'html_path': os.path.join(
                            self.args.save_viz_path,
                            f"trajectory_ep{best_episode_num}_interactive.html"
                        ),
                        'actor_sequence_path': os.path.join(
                            self.args.save_viz_path,
                            f"trajectory_ep{best_episode_num}_level{best_episode_for_viz.get('terrain_complexity_level', 'unknown')}_r{best_episode_for_viz['reward']:.0f}_actor_sequence.png"
                        ),
                        'control_diagnostics_path': os.path.join(
                            self.args.save_viz_path,
                            f"trajectory_ep{best_episode_num}_control_diagnostics.png"
                        ),
                    }
                else:
                    print(f"\n🎨 正在生成最佳回合可视化 (Episode {best_episode + 1}, Reward = {best_reward:.2f})...")
                    best_generated_files = self.generate_visualization(best_episode_for_viz, is_best=True) or {}
                    print(f"✅ 最佳回合可视化已生成")
            except Exception as viz_e:
                print(f"⚠️  最佳回合可视化生成失败: {viz_e}")
                traceback.print_exc()
                best_generated_files = {}

        best_html_alias = _copy_alias(best_generated_files.get('html_path'), 'best_reward_interactive.html')
        best_png_alias = _copy_alias(best_generated_files.get('image_path'), 'best_reward.png')
        best_actor_sequence_alias = _copy_alias(best_generated_files.get('actor_sequence_path'), 'best_reward_actor_sequence.png')
        best_control_diag_alias = _copy_alias(best_generated_files.get('control_diagnostics_path'), 'best_reward_control_diagnostics.png')
        if best_html_alias:
            visualization_artifacts['best_reward_html'] = best_html_alias
        if best_png_alias:
            visualization_artifacts['best_reward_png'] = best_png_alias
        if best_actor_sequence_alias:
            visualization_artifacts['best_reward_actor_sequence'] = best_actor_sequence_alias
        if best_control_diag_alias:
            visualization_artifacts['best_reward_control_diagnostics'] = best_control_diag_alias

        if save_team_success_html and best_success_episode_data is not None and not self.args.disable_visualization:
            reuse_best_reward_visualization = (
                bool(best_generated_files)
                and best_episode_data is not None
                and best_episode is not None
                and best_success_episode is not None
                and int(best_episode) == int(best_success_episode)
            )
            if reuse_best_reward_visualization:
                print(
                    f"\n♻️  最佳奖励回合同时也是最佳团队成功回合 "
                    f"(Episode {best_success_episode + 1})，复用已有可视化..."
                )
            else:
                print(
                    f"\n🌐 正在生成最佳团队成功回合HTML "
                    f"(Episode {best_success_episode + 1}, Reward = {best_success_reward:.2f})..."
                )
            try:
                if reuse_best_reward_visualization:
                    generated_success_files = dict(best_generated_files)
                else:
                    best_success_episode_for_viz = best_success_episode_data.copy()
                    best_success_episode_for_viz['trajectory'] = best_success_trajectory
                    if best_success_actor_outputs_history is not None:
                        best_success_episode_for_viz['actions_history'] = best_success_actor_outputs_history
                    if save_all_episode_visualizations:
                        best_success_episode_num = int(best_success_episode_for_viz.get('episode', 0)) + 1
                        success_image_path = os.path.join(
                            self.args.save_viz_path,
                            f"trajectory_ep{best_success_episode_num}_level{best_success_episode_for_viz.get('terrain_complexity_level', 'unknown')}_r{best_success_episode_for_viz['reward']:.0f}.png"
                        )
                        generated_success_files = {
                            'image_path': success_image_path,
                            'html_path': os.path.join(
                                self.args.save_viz_path,
                                f"trajectory_ep{best_success_episode_num}_interactive.html"
                            ),
                            'actor_sequence_path': os.path.join(
                                self.args.save_viz_path,
                                f"trajectory_ep{best_success_episode_num}_level{best_success_episode_for_viz.get('terrain_complexity_level', 'unknown')}_r{best_success_episode_for_viz['reward']:.0f}_actor_sequence.png"
                            ),
                            'control_diagnostics_path': os.path.join(
                                self.args.save_viz_path,
                                f"trajectory_ep{best_success_episode_num}_control_diagnostics.png"
                            ),
                        }
                    else:
                        generated_success_files = self.generate_visualization(best_success_episode_for_viz, is_best=False) or {}
                success_html_path = generated_success_files.get('html_path')
                success_png_path = generated_success_files.get('image_path')
                success_actor_sequence_path = generated_success_files.get('actor_sequence_path')
                success_control_diag_path = generated_success_files.get('control_diagnostics_path')
                if success_html_path and os.path.exists(success_html_path):
                    success_html_alias = os.path.join(self.args.save_viz_path, 'team_success_best_interactive.html')
                    shutil.copyfile(success_html_path, success_html_alias)
                    visualization_artifacts['team_success_best_html'] = success_html_alias
                if success_png_path and os.path.exists(success_png_path):
                    success_png_alias = os.path.join(self.args.save_viz_path, 'team_success_best.png')
                    shutil.copyfile(success_png_path, success_png_alias)
                    visualization_artifacts['team_success_best_png'] = success_png_alias
                if success_actor_sequence_path and os.path.exists(success_actor_sequence_path):
                    success_actor_alias = os.path.join(self.args.save_viz_path, 'team_success_best_actor_sequence.png')
                    shutil.copyfile(success_actor_sequence_path, success_actor_alias)
                    visualization_artifacts['team_success_best_actor_sequence'] = success_actor_alias
                if success_control_diag_path and os.path.exists(success_control_diag_path):
                    success_control_alias = os.path.join(self.args.save_viz_path, 'team_success_best_control_diagnostics.png')
                    shutil.copyfile(success_control_diag_path, success_control_alias)
                    visualization_artifacts['team_success_best_control_diagnostics'] = success_control_alias
            except Exception as success_viz_e:
                print(f"⚠️  团队成功回合HTML生成失败: {success_viz_e}")
                traceback.print_exc()
        elif save_team_success_html and best_success_episode_data is None:
            print("⏭️  未生成团队成功HTML（本次评估没有团队成功回合）")

        # 保存评估结果
        results = {
            'model_path': self.args.load_model_path,
            'scenario': self.args.scenario_name,
            'episodes': len(all_rewards),
            'avg_reward': float(avg_reward),
            'std_reward': float(std_reward),
            'max_reward': float(max_reward),
            'min_reward': float(min_reward),
            'all_rewards': [float(r) for r in all_rewards],
            'summary': summary,
            'evaluation_setup': evaluation_setup,
            'visualization_artifacts': visualization_artifacts,
            'terrain_seed_sequence': terrain_seed_sequence or [],
            'terrain_variant_seed_sequence': terrain_variant_seed_sequence or [],
            'episode_details': [
                {
                    'episode': ep['episode'],
                    'reward': float(ep['reward']),
                    'steps': ep['steps'],
                    'terrain_complexity_level': ep.get('terrain_complexity_level', 'unknown'),
                    'terrain_seed': ep.get('terrain_seed', None),
                    'terrain_variant_seed': ep.get('terrain_variant_seed', None),
                    'duration': ep['duration'],
                    'trajectory': ep.get('trajectory', []) if persist_episode_trajectories else [],
                    # 🔧 新增：保存碰撞和成功指标（与训练脚本一致）
                    'collision_count': ep.get('collision_count', 0),
                    'agent_collision_counts': ep.get('agent_collision_counts', []),
                    'min_distance': ep.get('min_distance', None),
                    'success': ep.get('success', 0),
                    'agent_success_flags': ep.get('agent_success_flags', []),
                    'team_success': ep.get('team_success', 0),
                    # 🔧 新增：到达时间/步数
                    'arrival_step': ep.get('arrival_step', None),
                    'arrival_time': ep.get('arrival_time', None),
                    'first_reach_step': ep.get('first_reach_step', None),
                    'first_reach_time': ep.get('first_reach_time', None),
                    'path_length': ep.get('path_length', None),
                    'agent_path_lengths': ep.get('agent_path_lengths', []),
                    'direct_distance': ep.get('direct_distance', None),
                    'agent_direct_distances': ep.get('agent_direct_distances', []),
                    'final_goal_distance': ep.get('final_goal_distance', None),
                    'agent_final_goal_distances': ep.get('agent_final_goal_distances', []),
                    'min_goal_distance': ep.get('min_goal_distance', None),
                    'agent_min_goal_distances': ep.get('agent_min_goal_distances', []),
                    'agent_first_reach_steps': ep.get('agent_first_reach_steps', []),
                    'path_efficiency': ep.get('path_efficiency', None),
                    'agent_path_efficiencies': ep.get('agent_path_efficiencies', []),
                    # 🔧 新增：穿透深度统计
                    'penetration_stat': ep.get('penetration_stat', None),
                    'visualization_files': ep.get('visualization_files', {}),
                } for ep in all_episodes_data
            ],
            'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        results_path = os.path.join(self.args.save_viz_path, 'evaluation_results.json')
        
        # 🔧 修复：将numpy数组转换为列表，确保JSON可序列化
        def convert_to_json_serializable(obj):
            """递归地将numpy数组和其他不可序列化对象转换为JSON可序列化的格式"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            elif obj is None:
                return None
            elif isinstance(obj, (bool, int, float, str)):
                return obj
            else:
                # 对于其他类型，尝试转换为字符串
                try:
                    return str(obj)
                except Exception:
                    return None
        
        # 转换结果数据
        results_serializable = convert_to_json_serializable(results)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
            
        print(f"✅ 评估结果已保存: {results_path}")
        
        # 如果已经为每个回合保存了可视化，则最佳回合对应文件已包含在内，仅保留best别名文件。
        if save_all_episode_visualizations:
            print("⏭️  跳过额外的最佳回合可视化生成（每回合可视化已保存）")
        elif not enable_best_traj:
            print(f"⏭️  跳过最佳回合可视化生成（SAVE_BEST_TRAJ=0）")
        elif self.args.disable_visualization:
            print(f"⏭️  跳过最佳回合可视化生成（--disable-visualization）")
        
        # 显示生成的文件信息
        print(f"\n📁 生成的文件:")
        print(f"  📊 评估统计: {results_path}")
        if enable_best_traj and best_episode_data is not None:
            print(f"  🏆 最佳回合: Episode {best_episode + 1}, Reward = {best_reward:.2f}")
        if visualization_artifacts.get('best_reward_html'):
            print(f"  🌐 最佳奖励HTML: {visualization_artifacts['best_reward_html']}")
        if visualization_artifacts.get('best_reward_actor_sequence'):
            print(f"  📈 最佳奖励动作时序图: {visualization_artifacts['best_reward_actor_sequence']}")
        if visualization_artifacts.get('best_reward_control_diagnostics'):
            print(f"  📊 最佳奖励控制诊断图: {visualization_artifacts['best_reward_control_diagnostics']}")
        if visualization_artifacts.get('team_success_best_html'):
            print(f"  🌐 团队成功HTML: {visualization_artifacts['team_success_best_html']}")
        if visualization_artifacts.get('team_success_best_actor_sequence'):
            print(f"  📈 团队成功动作时序图: {visualization_artifacts['team_success_best_actor_sequence']}")
        if visualization_artifacts.get('team_success_best_control_diagnostics'):
            print(f"  📊 团队成功控制诊断图: {visualization_artifacts['team_success_best_control_diagnostics']}")
        
        # 列出生成的图片和HTML文件
        if os.path.exists(self.args.save_viz_path):
            print(f"  🖼️  生成的图片:")
            png_files = [f for f in os.listdir(self.args.save_viz_path) if f.endswith('.png')]
            for png_file in png_files[:5]:  # 只显示前5个
                file_path = os.path.join(self.args.save_viz_path, png_file)
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"     {png_file} ({file_size:.1f}KB)")
            
            print(f"  🎬 生成的动画:")
            gif_files = [f for f in os.listdir(self.args.save_viz_path) if f.endswith('.gif')]
            for gif_file in gif_files[:3]:  # 只显示前3个
                file_path = os.path.join(self.args.save_viz_path, gif_file)
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"     {gif_file} ({file_size:.1f}KB)")
            
            print(f"  🌐 生成的HTML交互图:")
            html_files = [f for f in os.listdir(self.args.save_viz_path) if f.endswith('.html')]
            for html_file in html_files[:5]:  # 只显示前5个
                file_path = os.path.join(self.args.save_viz_path, html_file)
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"     {html_file} ({file_size:.1f}KB)")
        
        print(f"\n💡 查看结果:")
        print(f"   cd {self.args.save_viz_path} && ls -la")
        print(f"   python -m http.server 8000  # 启动HTTP服务器查看HTML文件")
        
        return results

    # ============= 可视化增强：障碍/地形/安全区叠加 ============= #
    def _get_extent_from_world(self):
        """尽量从world/terrain获取绘图区间；失败则返回None"""
        try:
            terrain = getattr(self.world, 'terrain', None)
            if terrain is not None and hasattr(terrain, 'extent'):
                return terrain.extent  # (xmin, xmax, ymin, ymax)
        except Exception:
            pass
        return None

    def _derive_extent_from_trajectory(self, traj):
        try:
            xs, ys = [], []
            for step_pos in traj:
                for p in step_pos:
                    if len(p) >= 2:
                        xs.append(float(p[0])); ys.append(float(p[1]))
            if not xs:
                return None
            pad = 5.0
            return (min(xs)-pad, max(xs)+pad, min(ys)-pad, max(ys)+pad)
        except Exception:
            return None

    def _plot_terrain_and_obstacles(self, ax, extent):
        """叠加地形等高线/障碍掩膜/圆形障碍"""
        # 地形高度图/等高线
        try:
            terrain = getattr(self.world, 'terrain', None)
            if terrain is not None and hasattr(terrain, 'height_map') and not self.args.no_plot_terrain:
                hmap = np.asarray(terrain.height_map)
                if hmap.ndim == 2 and hmap.size > 0:
                    ax.contourf(hmap, levels=20, cmap='Greys', alpha=0.25, extent=extent)
            if terrain is not None and hasattr(terrain, 'obstacle_mask') and not self.args.no_plot_obstacles:
                mask = np.asarray(terrain.obstacle_mask).astype(float)
                if mask.ndim == 2 and mask.size > 0:
                    ax.imshow(mask, cmap='Reds', alpha=0.25, extent=extent, origin='lower')
        except Exception:
            pass

        # 圆形/实体障碍
        try:
            if not self.args.no_plot_obstacles:
                obs_list = getattr(self.world, 'obstacles', None)
                if isinstance(obs_list, (list, tuple)):
                    for ob in obs_list:
                        pos = getattr(getattr(ob, 'state', None), 'p_pos', None)
                        r = getattr(ob, 'radius', None) or getattr(ob, 'size', None)
                        if pos is None or r is None:
                            continue
                        x, y = float(pos[0]), float(pos[1])
                        circ = plt.Circle((x, y), float(r), color='red', alpha=0.25, lw=1.0)
                        ax.add_patch(circ)
        except Exception:
            pass

    def _plot_trajectories(self, ax, traj):
        colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']
        n_agents = self.n_agents
        steps = len(traj)
        for i_agent in range(n_agents):
            xs, ys = [], []
            for s in range(steps):
                p = traj[s][i_agent]
                xs.append(float(p[0])); ys.append(float(p[1]))
            ax.plot(xs, ys, '-', lw=2, color=colors[i_agent % len(colors)], label=f'agent{i_agent}')
            # 起点/终点
            ax.plot(xs[0], ys[0], 'o', color=colors[i_agent % len(colors)], ms=5, alpha=0.9)
            ax.plot(xs[-1], ys[-1], 's', color=colors[i_agent % len(colors)], ms=5, alpha=0.9)

    def _compute_min_inter_agent_distance(self, traj):
        min_d = math.inf
        cnt_below = 0
        thr = getattr(self.args, 'minimum_clearance', None)
        for step_pos in traj:
            for i in range(len(step_pos)):
                for j in range(i+1, len(step_pos)):
                    dx = float(step_pos[i][0]) - float(step_pos[j][0])
                    dy = float(step_pos[i][1]) - float(step_pos[j][1])
                    d = math.hypot(dx, dy)
                    if d < min_d:
                        min_d = d
                    if thr is not None and d < float(thr):
                        cnt_below += 1
        return (min_d if min_d != math.inf else None), cnt_below

    def _generate_overlay_image(self, episode_data, save_path):
        """生成overlay图片，包含地形、障碍、目标点和轨迹"""
        traj = episode_data['trajectory']
        if not traj:
            return
        extent = self._get_extent_from_world()
        if extent is None:
            extent = self._derive_extent_from_trajectory(traj)
        fig, ax = plt.subplots(figsize=(10, 8))
        if extent is not None:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

        # 地形/障碍叠加
        self._plot_terrain_and_obstacles(ax, extent)
        
        # 🔧 关键修复：绘制目标点（中央目标和各智能体目标）
        try:
            goal_positions = self._get_goal_positions_from_scenario()
            if goal_positions:
                # 绘制中央目标
                if 'goal_pos' in goal_positions and goal_positions['goal_pos'] is not None:
                    g = goal_positions['goal_pos']
                    try:
                        import numpy as _np
                        g = _np.asarray(g, dtype=_np.float32).reshape(-1)
                        if len(g) >= 2:
                            gx, gy = float(g[0]), float(g[1])
                            ax.scatter(gx, gy, color='yellow', marker='*', s=500, 
                                      edgecolors='red', linewidth=2, zorder=1000, 
                                      label='Goal', alpha=0.9)
                            ax.text(gx, gy + 5.0, "GOAL", color='red', fontsize=14,
                                   fontweight='bold', ha='center', va='bottom', zorder=1000)
                    except Exception as e:
                        print(f"⚠️ 绘制中央目标失败: {e}")
                
                # 绘制各智能体目标
                if 'agent_goals' in goal_positions and isinstance(goal_positions['agent_goals'], list):
                    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
                    for idx, gp in enumerate(goal_positions['agent_goals']):
                        if gp is None:
                            continue
                        try:
                            import numpy as _np
                            gpa = _np.asarray(gp, dtype=_np.float32).reshape(-1)
                            if len(gpa) >= 2:
                                gx, gy = float(gpa[0]), float(gpa[1])
                                c = colors[idx % len(colors)]
                                ax.scatter(gx, gy, color=c, marker='^', s=200, zorder=900, 
                                          alpha=0.9, label=f'Agent {idx} Target')
                                ax.text(gx, gy + 3.0, f"Agent {idx}", color=c, fontsize=10,
                                       ha='center', va='bottom', zorder=900, fontweight='bold')
                        except Exception as e:
                            print(f"⚠️ 绘制智能体{idx}目标失败: {e}")
        except Exception as e:
            print(f"⚠️ 获取目标位置失败: {e}")

        # 轨迹
        self._plot_trajectories(ax, traj)

        # 安全距离统计
        min_d, cnt_below = self._compute_min_inter_agent_distance(traj)
        subtitle = f"min inter-agent d: {min_d:.2f}" if min_d is not None else "min inter-agent d: N/A"
        if getattr(self.args, 'minimum_clearance', None) is not None:
            subtitle += f", <thr count: {cnt_below} (thr={self.args.minimum_clearance})"

        ax.set_title(f"Trajectory Overlay - ep {episode_data['episode']} | reward {episode_data['reward']:.1f}\n{subtitle}")
        ax.set_xlabel('X'); ax.set_ylabel('Y')
        ax.grid(True, ls='--', alpha=0.3)
        ax.legend(loc='best')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _get_goal_positions_from_scenario(self):
        """从场景中获取目标位置信息"""
        try:
            result = {'goal_pos': None, 'agent_goals': []}
            
            # 获取中央目标位置
            if hasattr(self.scenario, 'goal_pos') and self.scenario.goal_pos is not None:
                result['goal_pos'] = np.asarray(self.scenario.goal_pos, dtype=np.float32)
                print(f"✅ 找到中央目标位置: {result['goal_pos']}")
            else:
                print("⚠️ 场景中没有找到中央目标位置")
            
            # 获取每个智能体的目标位置
            if hasattr(self.env, 'world') and hasattr(self.env.world, 'agents'):
                agents = self.env.world.agents
                for i, agent in enumerate(agents):
                    if hasattr(agent, 'goal_a') and hasattr(agent.goal_a, 'state'):
                        agent_goal = np.asarray(agent.goal_a.state.p_pos, dtype=np.float32)
                        result['agent_goals'].append(agent_goal)
                        print(f"✅ 找到智能体{i}目标位置: {agent_goal}")
                    else:
                        result['agent_goals'].append(None)
                        print(f"⚠️ 智能体{i}没有独立目标位置")
            
            return result
            
        except Exception as e:
            print(f"⚠️ 从场景获取目标信息失败: {e}")
            return {'goal_pos': None, 'agent_goals': []}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="MADDPG优化版模型评估脚本",
        epilog="""
示例用法:
  # 基本评估（生成PNG、GIF和HTML文件）
  python3 evaluate_optimized.py --load-model-path models/optimized_exp/best --eval-episodes 5
  
  # 禁用HTML生成
  python3 evaluate_optimized.py --load-model-path models/optimized_exp/best --disable-html
  
  # 禁用所有可视化
  python3 evaluate_optimized.py --load-model-path models/optimized_exp/best --disable-visualization
  
  # 使用固定位置
  python3 evaluate_optimized.py --load-model-path models/optimized_exp/best --use-fixed-positions --positions-file ./saved_positions/my_positions.json
  
  # 调整势场参数
  python3 evaluate_optimized.py --load-model-path models/optimized_exp/best --action-force-ratio 0.8 --influence-range 3.0
  
  # 禁用势场修正
  python3 evaluate_optimized.py --load-model-path models/optimized_exp/best --enable-action-correction false
  
  # 启用overlay图片（包含地形信息）
  python3 evaluate_optimized.py --load-model-path models/optimized_exp/best --enable-overlay

生成的文件:
  - trajectory_ep{episode}_r{reward}.png: 静态轨迹图
  - trajectory_ep{episode}_overlay.png: 带地形信息的轨迹图（需要--enable-overlay）
  - trajectory_ep{episode}_animation.gif: 轨迹动画
  - trajectory_ep{episode}_interactive.html: 可交互3D轨迹图（需要plotly）
  - evaluation_results.json: 评估统计结果

HTML交互式轨迹图功能:
  - 支持3D视角拖拽和缩放
  - 显示智能体轨迹、目标位置和地形信息
  - 每个评估回合都会生成独立的HTML文件
  - 需要安装plotly: pip install plotly
  - 可通过--disable-html参数禁用
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 环境参数
    parser.add_argument("--scenario-name", type=str, default="paper3d_terrain_weighted", 
                       help="场景名称 (paper3d_terrain_weighted, paper3d_terrain_vectorized, paper3d_terrain_energy)")
    parser.add_argument("--episode-length", type=int, default=2200, 
                       help="每回合最大步数（默认2200，与训练脚本一致）")
    parser.add_argument("--eval-episodes", type=int, default=3, 
                       help="评估回合数（将随机生成不同复杂度的地图）")
    parser.add_argument("--terrain-complexity-level", type=int, default=None, 
                       help="地形复杂度等级 (1-4)，None表示随机选择")
    parser.add_argument("--random-terrain", action="store_true", default=False,
                       help="使用随机地形（默认启用）")
    # 🔧 修复：与训练脚本保持一致的默认参数
    parser.add_argument("--gravity", type=float, default=0.0, help="环境重力加速度（作用于 -Z 方向），默认0.0（无重力）")
    parser.add_argument("--control-accel-gain", type=float, default=1.0, help="动作到物理加速度的控制增益，默认1.0")
    parser.add_argument("--reward-pos-scale", type=float, default=1.5, help="正向奖励缩放系数，默认1.5")
    parser.add_argument("--reward-neg-scale", type=float, default=2.5, help="负向奖励缩放系数，默认2.5")
    parser.add_argument("--agent-max-speed", type=float, default=37.5, help="智能体最大速度，默认37.5")
    parser.add_argument("--agent-accel", type=float, default=3.6, help="智能体加速度，默认3.6")
    parser.add_argument("--action-range-x", type=float, default=3.5, help="动作X轴映射范围系数（将网络输出乘以该系数），默认3.5")
    parser.add_argument("--action-range-y", type=float, default=3.5, help="动作Y轴映射范围系数（将网络输出乘以该系数），默认3.5")
    parser.add_argument("--action-range-z", type=float, default=3.0, help="动作Z轴映射范围系数（将网络输出乘以该系数），默认3.0")
    parser.add_argument("--damping", type=float, default=0.18, help="速度阻尼系数，默认0.18")
    parser.add_argument("--simulation-dt", type=float, default=float(os.getenv('SIMULATION_DT', '0.08')),
                       help="仿真步长dt（秒），默认从SIMULATION_DT读取")
    parser.add_argument("--z-action-bias", type=float, default=float(os.getenv('Z_ACTION_BIAS', '0.0')),
                       help="Z轴动作偏置，默认从Z_ACTION_BIAS读取")
    parser.add_argument("--use-quadrotor-dynamics", type=lambda x: str(x).lower() in ('1', 'true', 'yes', 'on'),
                       default=os.getenv('USE_QUADROTOR_DYNAMICS', '0').lower() in ('1', 'true', 'yes', 'on'),
                       help="是否启用四旋翼动力学，默认从USE_QUADROTOR_DYNAMICS读取")
    parser.add_argument("--quadrotor-attitude-response-time", type=float, default=float(os.getenv('QUADROTOR_ATTITUDE_RESPONSE_TIME', '0.0')),
                       help="四旋翼姿态响应时间，默认从QUADROTOR_ATTITUDE_RESPONSE_TIME读取")
    parser.add_argument("--quadrotor-psi-cmd", type=float, default=float(os.getenv('QUADROTOR_PSI_CMD', '0.0')),
                       help="四旋翼偏航指令，默认从QUADROTOR_PSI_CMD读取")
    
    # 势场/动作修正相关参数
    parser.add_argument("--enable-action-correction", type=lambda x: (str(x).lower() == 'true'), default=True, 
                       help="启用势场/混合动作修正（如集成时生效）")
    parser.add_argument("--correction-type", type=str, default="potential_field", 
                       choices=["potential_field", "hybrid", "none"], help="修正类型")
    parser.add_argument("--influence-range", type=float, default=2.5, help="Potential field influence range")
    parser.add_argument("--force-param-ratio", type=float, default=0.8, help="Potential field parameter adjustment base coefficient")
    
    # 势场力参数范围映射
    parser.add_argument("--force-param-goal-attraction-range", type=float, nargs=2, default=[0.5, 3.0], 
                       help="势场力参数：目标吸引力范围 [min, max]，网络输出p[0]映射到此范围")
    parser.add_argument("--force-param-lambda-1-range", type=float, nargs=2, default=[0.1, 2.0], 
                       help="势场力参数：lambda_1范围 [min, max]，网络输出p[1]映射到此范围")
    parser.add_argument("--force-param-terrain-repulsion-range", type=float, nargs=2, default=[0.1, 1.5], 
                       help="势场力参数：地形排斥力范围 [min, max]，网络输出p[2]映射到此范围")
    parser.add_argument("--force-param-detection-radius-range", type=float, nargs=2, default=[2.0, 10.0], 
                       help="势场力参数：检测半径范围 [min, max]，网络输出p[3]映射到此范围")
    
    # 网络动作和势场动作混合比例
    parser.add_argument("--action-force-ratio", type=float, default=0.75, 
                       help="网络动作和势场动作的混合比例 (0.0=完全网络动作, 1.0=完全势场动作, 默认0.75=75%%势场+25%%网络)")
    parser.add_argument(
        "--force-action-force-ratio",
        type=float,
        default=None,
        help="可选：强制覆盖评估时使用的FR，仅用于敏感性/控制实验；默认仍按checkpoint对应FR评估",
    )
    
    # 势场修正版本选择
    parser.add_argument("--use-tf-potential-field", type=lambda x: (str(x).lower() in ('1','true','yes','on')), default=True,
                       help="是否使用TensorFlow版本的势场修正 (1=TF版本, 0=原版)")
    
    # 🔧 新增：地形感知模式参数（仅用于评估时APF地形力计算）
    parser.add_argument("--terrain-sensing-mode", type=str, default="local",
                        choices=["local", "oracle_same_probes", "oracle_dense"],
                        help="地形感知模式: local=使用观测中的地形信息, oracle_same_probes=Oracle真值(相同probe布局), oracle_dense=Oracle真值(密集探测)")
    
    # 🔧 新增：FR和PF特征标志
    parser.add_argument("--use-fr-feature", type=lambda x: (str(x).lower() in ('1','true','yes','on')), default=True,
                       help="Enable FR feature (Force Ratio as separate input)")
    parser.add_argument("--use-pf-feature", type=lambda x: (str(x).lower() in ('1','true','yes','on')), default=True,
                       help="Enable PF feature (Potential field force appended to obs)")
    
    # 🔧 修复：与训练脚本保持一致的势场参数默认值
    parser.add_argument("--goal-attraction", type=float, default=15.0, help="Goal attraction force，默认15.0")
    parser.add_argument("--lambda-1-base", type=float, default=8.5, help="Lambda_1 base value，默认8.5")
    parser.add_argument("--terrain-repulsion", type=float, default=3800.0, help="Terrain repulsion force，默认3800.0")
    parser.add_argument("--agent-influence-range", type=float, default=10.0, help="Agent influence range，默认10.0")
    parser.add_argument("--delta-k-att", type=float, default=0.5, help="Delta K_att，默认0.5")
    parser.add_argument("--delta-lambda-1", type=float, default=2.5, help="Delta Lambda_1，默认2.5")
    parser.add_argument("--delta-k-rep", type=float, default=40.0, help="Delta K_rep，默认40.0")
    parser.add_argument("--delta-radius", type=float, default=5.0, help="Delta Radius，默认5.0")
    
    # 🔧 新增：算法选择
    parser.add_argument("--algorithm", type=str, default="matd3", choices=["maddpg", "matd3", "mappo"],
                       help="Training algorithm selection (maddpg, matd3 or mappo)")
    
    # 分项加权奖励参数（如果使用加权场景）
    parser.add_argument("--distance-weight", type=float, default=None, help="距离奖励权重")
    parser.add_argument("--exploration-weight", type=float, default=None, help="探索奖励权重")
    parser.add_argument("--stationary-weight", type=float, default=None, help="停滞惩罚权重")
    parser.add_argument("--direction-weight", type=float, default=None, help="方向一致性奖励权重")
    parser.add_argument("--deviation-weight", type=float, default=None, help="偏离奖励权重")
    parser.add_argument("--start-area-weight", type=float, default=None, help="起始区域奖励权重")
    parser.add_argument("--approach-weight", type=float, default=None, help="接近目标奖励权重")
    parser.add_argument("--energy-weight", type=float, default=None, help="能量效率奖励权重")
    parser.add_argument("--height-weight", type=float, default=None, help="高度适应性奖励权重")
    parser.add_argument("--height-reward-enabled", type=lambda x: (str(x).lower() in ('1','true','yes','on')), default=None, help="是否启用高度奖励")
    parser.add_argument("--height-ideal-min", type=float, default=None, help="理想高度下限")
    parser.add_argument("--height-ideal-max", type=float, default=None, help="理想高度上限")
    parser.add_argument("--lateral-weight", type=float, default=None, help="侧向/绕行奖励权重")
    parser.add_argument("--clearance-weight", type=float, default=None, help="净空/最小距离增益奖励权重")
    parser.add_argument("--clearance-d-max", type=float, default=None, help="净空奖励归一化因子")
    parser.add_argument("--success-weight", type=float, default=None, help="成功奖励权重")
    parser.add_argument("--collision-weight", type=float, default=None, help="碰撞惩罚权重")
    parser.add_argument("--collision-reduction-weight", type=float, default=None, help="碰撞次数减少奖励权重")
    parser.add_argument("--global-weight", type=float, default=None, help="全局奖励权重")
    parser.add_argument("--shaping-weight", type=float, default=None, help="潜势函数 shaping 权重")
    parser.add_argument("--max-reward", type=float, default=None, help="最大奖励值")
    parser.add_argument("--min-reward", type=float, default=None, help="最小奖励值")
    parser.add_argument("--success-reward-value", type=float, default=None, help="成功一次性奖励值")
    parser.add_argument("--no-collision-reward-value", type=float, default=None, help="无碰撞奖励值")
    parser.add_argument("--success-distance-threshold", type=float, default=None, help="成功判定距离阈值")
    parser.add_argument("--collision-penalty-value", type=float, default=None, help="碰撞惩罚绝对值")
    parser.add_argument("--collision-distance-threshold", type=float, default=None, help="碰撞/接触距离阈值")
    parser.add_argument("--global-reward-mode", type=str, default=None, help="全局奖励模式")
    parser.add_argument("--shaping-gamma", type=float, default=None, help="潜势函数 gamma")
    
    # 模型和保存路径
    default_model_path = os.getenv('MODEL_PATH', 'models/optimized_exp')
    parser.add_argument("--load-model-path", type=str, default=default_model_path,
                       help="要加载的模型权重文件夹路径（默认从环境变量MODEL_PATH或models/optimized_exp/best）")
    parser.add_argument("--save-viz-path", type=str, default="evaluation_results", 
                       help="可视化结果保存路径")
    
    # 可视化控制
    parser.add_argument("--disable-visualization", action="store_true", 
                       help="禁用可视化生成")
    parser.add_argument("--enable-overlay", action="store_true", default=False,
                       help="启用overlay图片生成（包含地形和障碍物信息）")
    parser.add_argument("--disable-overlay", action="store_true",
                       help="禁用overlay图片生成（默认禁用）")
    parser.add_argument("--disable-gif", action="store_true",
                       help="禁用GIF生成（避免长时间阻塞或大文件）")
    parser.add_argument("--gif-max-frames", type=int, default=60,
                       help="限制GIF的最大帧数（默认60帧）")
    
    # 场景兼容性参数（为了与原版兼容）
    parser.add_argument("--terrain-seed", type=int, default=None, help="地形种子")
    parser.add_argument("--use-fixed-positions", action="store_true", help="使用固定位置")
    parser.add_argument("--positions-file", type=str, default="./saved_positions/5.json", 
                       help="固定位置文件路径")
    parser.add_argument("--dynamic-first-time", action="store_true", help="动态首次运行")
    parser.add_argument("--disable-early-termination", action="store_true", 
                       help="禁用提前终止，强制运行完整的episode_length步数")
    # 与训练一致的隐藏层配置（可选），用于构建相同拓扑以加载权重
    parser.add_argument("--actor-hidden", type=str, default=None, help="Actor隐藏层，例如: 384,256,128,64")
    parser.add_argument("--critic-hidden", type=str, default=None, help="Critic隐藏层，例如: 512,256,128,64")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    disable_viz_env = _env_flag("EVAL_DISABLE_VISUALIZATION", False)
    light_mode_env = _env_flag("EVAL_LIGHT_MODE", False)
    save_trajectory_png = _env_flag("SAVE_EVAL_TRAJECTORY_PNG", True)
    save_team_success_html = _env_flag("SAVE_TEAM_SUCCESS_HTML", False)
    save_actor_sequence = _env_flag("SAVE_EVAL_ACTOR_SEQUENCE", False)
    save_control_diagnostics = _env_flag("SAVE_EVAL_CONTROL_DIAGNOSTICS", False)
    keep_viz_artifacts_in_light_mode = (
        save_trajectory_png or save_team_success_html or save_actor_sequence or save_control_diagnostics
    )
    if disable_viz_env or (light_mode_env and not keep_viz_artifacts_in_light_mode):
        args.disable_visualization = True
        args.disable_gif = True
        setattr(args, 'disable_html', True)
    elif light_mode_env:
        args.disable_gif = True
    # 在任何 TensorFlow 操作之前优先配置GPU，避免已初始化后再设内存增长
    try:
        configure_gpu()
    except Exception:
        pass
    
    # 🔧 新增：启用XLA加速（如果环境变量XLA_GLOBAL=1）
    # 与训练脚本保持一致，使用XLA Global模式加速评估
    # 注意：必须在TensorFlow图构建之前启用XLA
    xla_global = os.getenv('XLA_GLOBAL', '1').lower() in ('1', 'true', 'yes', 'on')  # 默认启用
    if xla_global:
        try:
            # 🔧 关键修复：在启用XLA前设置稳定的XLA配置，避免CUDA错误
            # 与训练脚本保持一致，使用稳定的XLA配置
            existing_xla_flags = os.environ.get('XLA_FLAGS', '')
            # 移除所有可能导致问题的flag
            flags_to_remove = [
                '--xla_gpu_enable_triton_gemm=false',
                '--xla_gpu_enable_triton_gemm=true',
                '--xla_gpu_force_compilation_parallelism=1',
                '--xla_gpu_force_compilation_parallelism=0',
            ]
            cleaned_flags = existing_xla_flags
            for flag in flags_to_remove:
                cleaned_flags = cleaned_flags.replace(flag, '')
            cleaned_flags = ' '.join(cleaned_flags.split())  # 清理多余空格
            
            # 设置稳定的XLA配置
            stable_xla_flags = [
                '--xla_gpu_autotune_level=1',  # 降低autotune级别，减少kernel搜索空间
                '--xla_gpu_deterministic_ops=true',  # 强制确定性操作
            ]
            
            # 合并配置
            if cleaned_flags:
                stable_xla_flags_str = cleaned_flags + ' ' + ' '.join(stable_xla_flags)
            else:
                stable_xla_flags_str = ' '.join(stable_xla_flags)
            os.environ['XLA_FLAGS'] = stable_xla_flags_str
            
            # 启用XLA Global JIT编译（与训练脚本一致）
            tf.config.optimizer.set_jit(True)
            print("✅ XLA加速已启用（Global JIT模式）")
            print("   💡 提示：XLA首次编译可能需要一些时间，后续运行会更快")
        except Exception as e:
            print(f"⚠️  XLA加速启用失败: {e}")
            print("   💡 提示：如果遇到问题，可以设置XLA_GLOBAL=0禁用XLA")
    else:
        print("ℹ️  XLA加速未启用（设置XLA_GLOBAL=1以启用）")

    # 设置随机种子以确保每次评估都有不同的随机性
    import time
    import random
    current_time = int(time.time() * 1000000) % 2**32
    random.seed(current_time)
    np.random.seed(current_time)
    tf.random.set_seed(current_time)
    print(f"🎲 设置随机种子: {current_time} (确保每次评估的随机性)")
    
    # 显示HTML生成状态
    enable_html = getattr(args, 'enable_html', True) and not getattr(args, 'disable_html', False)
    if enable_html:
        print("🌐 HTML交互式轨迹图生成: 启用")
        print("💡 提示: 如果HTML生成失败，请安装plotly: pip install plotly")
    else:
        print("🌐 HTML交互式轨迹图生成: 禁用")
    
    try:
        # 创建评估器
        evaluator = ModelEvaluator(args)
        
        # 运行评估
        results = evaluator.run_evaluation()
        
        print("\n🎉 评估完成!")
        
        # 显示HTML文件查看提示
        if enable_html:
            print(f"\n🌐 查看HTML交互式轨迹图:")
            print(f"   cd {args.save_viz_path}")
            print(f"   python -m http.server 8000")
            print(f"   然后在浏览器中打开: http://localhost:8000")
        
    except KeyboardInterrupt:
        print("\n⚠️ 评估被用户中断")
    except Exception as e:
        print(f"\n❌ 评估出错: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
