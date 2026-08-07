#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MADDPG优化版模型评估与可视化脚本
仿照1.0版本功能，支持模型加载、评估和可视化生成
"""

import os
import sys
from pathlib import Path


def _bootstrap_gpu_ld_path_before_tensorflow():
    """Ensure TensorFlow sees WSL/conda CUDA libraries before it is imported."""
    entrypoint_name = Path(sys.argv[0]).name
    if __name__ != '__main__' and entrypoint_name not in {'evaluate_mappo.py'}:
        return
    if os.environ.get('CUDA_VISIBLE_DEVICES') == '':
        return
    if os.environ.get('MATD3_SKIP_GPU_LD_BOOTSTRAP', '').lower() in ('1', 'true', 'yes', 'on'):
        return
    if os.environ.get('MATD3_EVAL_GPU_LD_BOOTSTRAPPED') == '1':
        return

    current_parts = [p for p in os.environ.get('LD_LIBRARY_PATH', '').split(':') if p]
    current_set = set(current_parts)

    candidate_paths = []
    wsl_cuda_lib = Path('/usr/lib/wsl/lib')
    if wsl_cuda_lib.is_dir():
        candidate_paths.append(str(wsl_cuda_lib))

    env_prefix = os.environ.get('CONDA_PREFIX') or sys.prefix
    if env_prefix:
        prefix = Path(env_prefix)
        conda_lib = prefix / 'lib'
        if conda_lib.is_dir():
            candidate_paths.append(str(conda_lib))

        py_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
        nvidia_root = prefix / 'lib' / py_tag / 'site-packages' / 'nvidia'
        if nvidia_root.is_dir():
            candidate_paths.extend(str(path) for path in sorted(nvidia_root.glob('*/lib')) if path.is_dir())

    prepend_paths = []
    for path in candidate_paths:
        if path not in current_set and path not in prepend_paths:
            prepend_paths.append(path)

    if not prepend_paths:
        os.environ['MATD3_EVAL_GPU_LD_BOOTSTRAPPED'] = '1'
        return

    os.environ['LD_LIBRARY_PATH'] = ':'.join(prepend_paths + current_parts)
    os.environ['MATD3_EVAL_GPU_LD_BOOTSTRAPPED'] = '1'
    print(
        "[Eval GPU Bootstrap] added CUDA library paths before TensorFlow import: "
        + ':'.join(prepend_paths),
        flush=True,
    )
    os.execve(sys.executable, [sys.executable] + sys.argv, os.environ.copy())


_bootstrap_gpu_ld_path_before_tensorflow()

import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import traceback
import json
import time
import math
import shutil
import csv
import re
import copy
import shlex
import signal
import subprocess
from collections import deque
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from experiment_runtime_config import (
    SCIENTIFIC_RUNTIME_ENV_FIELDS,
    add_runtime_environment_arguments,
    apply_runtime_environment,
    find_training_runtime_manifest,
    load_training_runtime_manifest,
    runtime_environment_from_manifest,
)
from multiagent.scenarios.obstacle_observation import normalize_obstacle_observation_mode

_EVAL_NOISE_TYPE = "gaussian"
_EVAL_NOISE_STREAM_MODE = "per_episode_seedsequence_pcg64_v1"

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


_REWARD_DECOMPOSITION_FIELDS = [
    'reward_progress',
    'reward_clearance',
    'reward_height',
    'reward_stagnation',
    'reward_sync',
    'reward_dense_success',
    'reward_energy',
    'reward_safety_penalty',
    'reward_collision_penalty',
    'reward_terrain_penalty',
    'reward_obstacle_penalty',
    'reward_inter_agent_penalty',
    'reward_boundary_penalty',
    'reward_terminal_success',
    'reward_terminal_failure',
    'reward_terminal_quality',
    'reward_apf_or_action_cost_if_any',
]


def _new_eval_diagnostics_accumulator():
    return {
        'reward_components': {name: 0.0 for name in _REWARD_DECOMPOSITION_FIELDS},
        'reward_diag_step_count': 0,
        'reward_diag_missing_steps': 0,
        'raw_action_norm_sum': 0.0,
        'corr_action_norm_sum': 0.0,
        'action_delta_norm_sum': 0.0,
        'pf_force_norm_sum': 0.0,
        'force_ratio_sum': 0.0,
        'action_diag_step_count': 0,
        'speed_sum': 0.0,
        'speed_step_count': 0,
        'semantic_gap_sum': 0.0,
        'semantic_gap_max': 0.0,
        'semantic_gap_step_count': 0,
        'reward_total_before_clip_sum': 0.0,
        'reward_clip_delta_sum': 0.0,
    }


def _mean_head_norm(values):
    try:
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        head_dim = min(3, int(arr.shape[-1]))
        if head_dim <= 0:
            return None
        norms = np.linalg.norm(arr[..., :head_dim], axis=-1)
        finite = norms[np.isfinite(norms)]
        if finite.size == 0:
            return None
        return float(np.mean(finite))
    except Exception:
        return None


def _add_optional_mean(acc, key, value):
    try:
        if value is None:
            return
        value = float(value)
        if not np.isfinite(value):
            return
        acc[key] += value
    except Exception:
        pass


def _get_vectorized_reward_diagnostics(env):
    try:
        scenario = getattr(env, 'scenario', None)
        for attr in ('vectorized_calculator', 'vectorized_reward_calculator', 'reward_calculator'):
            calc = getattr(scenario, attr, None) if scenario is not None else None
            diag = getattr(calc, 'last_reward_diagnostics', None) if calc is not None else None
            if isinstance(diag, dict):
                return diag
    except Exception:
        return None
    return None


def _get_world_success_snapshot(env, expected_count=None):
    try:
        world = getattr(env, 'world', None)
        if world is None:
            return None
        reach = list(getattr(world, '_episode_agent_reach_flags', []))
        safe = list(getattr(world, '_episode_agent_safe_flags', []))
        succ = list(getattr(world, '_episode_agent_success_flags', []))
        if expected_count is not None:
            expected_count = int(expected_count)
            if expected_count > 0 and (
                len(reach) != expected_count or len(safe) != expected_count or len(succ) != expected_count
            ):
                return None
        if not succ:
            return None
        team = int(getattr(world, '_episode_team_success_flag', 1 if all(int(v) == 1 for v in succ) else 0))
        return {
            'reach_flags': [int(v) for v in reach],
            'safe_flags': [int(v) for v in safe],
            'success_flags': [int(v) for v in succ],
            'team_success': 1 if team == 1 else 0,
            'all_reached': 1 if bool(getattr(world, '_episode_all_reached', False)) else 0,
            'done_reason': getattr(world, '_episode_done_reason', None),
        }
    except Exception:
        return None


_MISSING_GAZEBO_LIVE_STATE = object()


def _copy_gazebo_live_state_value(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _snapshot_attr(obj, attr):
    if obj is None or not hasattr(obj, attr):
        return _MISSING_GAZEBO_LIVE_STATE
    return _copy_gazebo_live_state_value(getattr(obj, attr))


def _restore_attr(obj, attr, value):
    if obj is None:
        return
    if value is _MISSING_GAZEBO_LIVE_STATE:
        try:
            if hasattr(obj, attr):
                delattr(obj, attr)
        except Exception:
            pass
        return
    try:
        setattr(obj, attr, _copy_gazebo_live_state_value(value))
    except Exception:
        pass


def _snapshot_gazebo_live_reward_state(env):
    """Capture mutable reward/done bookkeeping before Python env.step prediction."""
    world = getattr(env, 'world', None)
    agents = list(getattr(env, 'agents', []) or getattr(world, 'policy_agents', []) or [])
    env_attrs = (
        '_agent_done_flags',
        '_agent_done_steps',
        '_termination_reasons',
    )
    world_attrs = (
        '_global_reward_given',
        '_global_reward_step_cache',
        '_team_sync_step_cache',
        '_team_sync_state',
        '_episode_success',
        '_episode_all_reached',
        '_episode_agent_reach_flags',
        '_episode_agent_safe_flags',
        '_episode_agent_success_flags',
        '_episode_team_success_flag',
        '_episode_success_thr_snapshot',
        '_episode_terminal',
        '_episode_done_reason',
        '_termination_reasons',
        '_all_agents_reached_logged',
    )
    agent_attrs = (
        'last_position',
        'last_velocity',
        'last_goal_dist',
        'stationary_count',
        'visited_cells',
        'debug_info',
        '_success_state',
        '_episode_has_collision',
        '_had_obstacle_collision',
        '_had_terrain_contact_or_penetration',
        '_had_penetration_or_collision',
        '_ever_reached_goal',
        'current_episode_collision_count',
        'previous_episode_collision_count',
        'collision_reduction_reward_given',
        'last_min_distance',
        'out_of_bounds_info',
        '_rc_th',
    )
    return {
        'env': {attr: _snapshot_attr(env, attr) for attr in env_attrs},
        'world': {attr: _snapshot_attr(world, attr) for attr in world_attrs},
        'agents': [
            {attr: _snapshot_attr(agent, attr) for attr in agent_attrs}
            for agent in agents
        ],
    }


def _restore_gazebo_live_reward_state(env, snapshot):
    if not isinstance(snapshot, dict):
        return
    world = getattr(env, 'world', None)
    for attr, value in snapshot.get('env', {}).items():
        _restore_attr(env, attr, value)
    for attr, value in snapshot.get('world', {}).items():
        _restore_attr(world, attr, value)
    agents = list(getattr(env, 'agents', []) or getattr(world, 'policy_agents', []) or [])
    for agent, agent_state in zip(agents, snapshot.get('agents', [])):
        if isinstance(agent_state, dict):
            for attr, value in agent_state.items():
                _restore_attr(agent, attr, value)


def _clear_eval_observation_cache(env):
    try:
        callback_owner = getattr(env.observation_callback, "__self__", None)
        if callback_owner is not None:
            callback_owner._obs_step_cache_key = None
            callback_owner._obs_step_cache = {}
    except Exception:
        pass


def _mean_step_reward_increment(rew_n):
    try:
        rew_arr = np.asarray(rew_n, dtype=np.float32)
        if rew_arr.size > 0:
            if not np.all(np.isfinite(rew_arr)):
                rew_arr = np.where(np.isfinite(rew_arr), rew_arr, -1000.0)
            return float(np.mean(rew_arr))
        return 0.0
    except Exception:
        try:
            return float(np.mean(rew_n))
        except Exception:
            return float(sum(rew_n)) if rew_n is not None else 0.0


def _recompute_step_outputs_from_current_state(env, fallback_info_n=None):
    """Recompute obs/reward/done after Gazebo state feedback becomes authoritative."""
    world = getattr(env, 'world', None)
    agents = list(getattr(env, 'agents', []) or getattr(world, 'policy_agents', []) or [])
    agent_count = len(agents)
    _clear_eval_observation_cache(env)

    try:
        obs_n = env._get_obs_batch(agents)
    except Exception:
        default_obs_dim = env._get_default_obs_dim()
        obs_n = [np.zeros(default_obs_dim, dtype=np.float32) for _ in range(agent_count)]

    reward_pos_scale = float(getattr(world, 'reward_pos_scale', 1.0))
    reward_neg_scale = float(getattr(world, 'reward_neg_scale', 1.0))
    reward_values = None
    reward_owner = getattr(env.reward_callback, '__self__', None) if getattr(env, 'reward_callback', None) is not None else None
    if reward_owner is not None and hasattr(reward_owner, '_compute_batch_rewards'):
        try:
            if hasattr(reward_owner, '_ensure_world_reward_initialized'):
                reward_owner._ensure_world_reward_initialized(world)
            reward_batch = reward_owner._compute_batch_rewards([agents], [world], cache_key=None)
            if (
                isinstance(reward_batch, np.ndarray)
                and reward_batch.ndim == 2
                and reward_batch.shape[0] >= 1
                and reward_batch.shape[1] >= agent_count
            ):
                reward_values = np.asarray(reward_batch[0], dtype=np.float32)
        except Exception:
            reward_values = None

    rew_n = []
    for i, agent in enumerate(agents):
        try:
            if reward_values is not None and i < reward_values.shape[0]:
                r = float(reward_values[i])
            else:
                r = float(env._get_reward(agent))
            rew_n.append(r * reward_pos_scale if r >= 0 else r * reward_neg_scale)
        except Exception:
            rew_n.append(0.0)

    done_n = []
    for i, agent in enumerate(agents):
        try:
            agent_key = getattr(agent, 'name', f'agent_{i}')
            if getattr(env, '_agent_done_flags', {}).get(agent_key, False):
                done_n.append(True)
            else:
                d = bool(env._get_done(agent))
                if d:
                    env._agent_done_flags[agent_key] = True
                    env._agent_done_steps[agent_key] = int(getattr(env, '_current_step', 0))
                done_n.append(d)
        except Exception:
            done_n.append(False)

    try:
        env._sync_world_team_success_snapshot()
        if bool(getattr(world, '_episode_all_reached', False)):
            done_n = [True] * agent_count
            for i, agent in enumerate(agents):
                agent_key = getattr(agent, 'name', f'agent_{i}')
                env._agent_done_flags[agent_key] = True
                env._agent_done_steps[agent_key] = int(getattr(world, 'current_step', getattr(env, '_current_step', 0)))
    except Exception:
        pass

    info_list = []
    fallback_list = []
    if isinstance(fallback_info_n, dict):
        fallback_list = fallback_info_n.get('n', []) or []
    for i, agent in enumerate(agents):
        try:
            if getattr(env, '_eval_light_info', False):
                info = {}
            else:
                info = env._get_info(agent)
                if not isinstance(info, dict):
                    info = {}
        except Exception:
            info = fallback_list[i] if i < len(fallback_list) and isinstance(fallback_list[i], dict) else {}
        info_list.append(info)

    return obs_n, rew_n, done_n, {'n': info_list}


def _terminate_process_group(proc, timeout=5.0):
    if proc is None:
        return
    try:
        if proc.poll() is not None:
            return
    except Exception:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=timeout)
        return
    except Exception:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=1.0)
    except Exception:
        pass


def _export_and_launch_gazebo_live_world(scenario, world, args, episode_idx, quiet_output=True):
    if not (_env_flag("GAZEBO_LIVE_SYNC", False) and _env_flag("GAZEBO_LIVE_AUTOLAUNCH", False)):
        return None

    output_root_raw = os.getenv("GAZEBO_LIVE_EXPORT_DIR", "").strip()
    if output_root_raw:
        output_dir = Path(output_root_raw).expanduser().resolve()
        if int(episode_idx) != 0:
            output_dir = output_dir / f"episode_{int(episode_idx) + 1:03d}"
    else:
        output_dir = Path(getattr(args, "save_viz_path", "evaluation_results")).expanduser().resolve() / f"gazebo_live_ep{int(episode_idx) + 1:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    dense_default = int(round(float(getattr(scenario, "map_size", 200.0) or 200.0)))
    dense_resolution = max(2, _env_int("GAZEBO_LIVE_DENSE_RESOLUTION", dense_default))
    visual_resolution_env = _env_int("GAZEBO_LIVE_VISUAL_RESOLUTION", 0)
    visual_resolution = visual_resolution_env if visual_resolution_env >= 2 else None
    coarse_resolution = max(2, _env_int("GAZEBO_LIVE_COARSE_COLLISION_RESOLUTION", 80))
    use_coarse_collision = _env_flag("GAZEBO_LIVE_USE_COARSE_COLLISION", False)
    reuse_cache = _env_flag("GAZEBO_LIVE_EXPORT_CACHE", False)
    collision_mode = _gazebo_live_collision_mode()
    physical_collision_enabled = collision_mode != "nonblocking"

    from scenario_exporter import export_scenario_snapshot
    from gazebo_terrain_exporter import export_gazebo_scene
    from gazebo_dynamic_replay_exporter import export_gazebo_live_world

    scenario_snapshot = export_scenario_snapshot(
        scenario=scenario,
        world=world,
        output_dir=output_dir,
        scenario_name=getattr(args, "scenario_name", "paper3d_terrain_vectorized"),
        dense_resolution=dense_resolution,
        generate_html=_env_flag("GAZEBO_LIVE_EXPORT_HTML", False),
    )
    scenario_json = Path(scenario_snapshot["export_paths"]["scenario_json"]).resolve()
    gazebo_scene = export_gazebo_scene(
        scenario_json=scenario_json,
        output_dir=output_dir,
        visual_resolution=visual_resolution,
        coarse_collision_resolution=coarse_resolution,
        use_coarse_collision=use_coarse_collision,
        terrain_collision=physical_collision_enabled,
        obstacle_collision=physical_collision_enabled,
        reuse_cache=reuse_cache,
    )
    gazebo_live = export_gazebo_live_world(
        scenario_json=scenario_json,
        output_dir=output_dir,
        base_world_sdf=Path(gazebo_scene["world_sdf"]),
        collision_mode=collision_mode,
        reuse_cache=reuse_cache,
    )
    gazebo_live["collision_mode"] = collision_mode
    gazebo_live["physical_collision_enabled"] = bool(physical_collision_enabled)

    model_parent_dir = Path(gazebo_live["model_parent_dir"]).resolve()
    current_resource_path = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    resource_parts = [str(model_parent_dir)]
    if current_resource_path:
        resource_parts.append(current_resource_path)
    os.environ["GZ_SIM_RESOURCE_PATH"] = ":".join(resource_parts)
    os.environ["GAZEBO_LIVE_WORLD"] = str(gazebo_live["world_name"])
    if gazebo_live.get("agent_prefix"):
        os.environ["GAZEBO_LIVE_AGENT_PREFIX"] = str(gazebo_live["agent_prefix"])
    os.environ["GAZEBO_LIVE_STATE_FILE"] = str(output_dir / "gazebo_live_state.json")
    os.environ["GAZEBO_LIVE_CONTACT_FLAG_FILE"] = str(output_dir / "gazebo_live_contact.flag")

    if not _env_flag("GAZEBO_LIVE_AUTOLAUNCH_START", True):
        gazebo_live.update(
            {
                "autolaunch_started": False,
                "output_dir": str(output_dir),
                "scenario_json": str(scenario_json),
            }
        )
        return gazebo_live

    home_dir = os.getenv("GAZEBO_LIVE_HOME", "/tmp/matd3_gz_home")
    stdout_path = output_dir / "gz_server.stdout.log"
    stderr_path = output_dir / "gz_server.stderr.log"
    launch_gui = _env_flag("GAZEBO_LIVE_AUTOLAUNCH_GUI", False)
    launch_run = _env_flag("GAZEBO_LIVE_AUTOLAUNCH_RUN", False)
    gz_command = " ".join(
        part for part in (
            "gz",
            "sim",
            "-s",
            "-r" if launch_run else "",
            shlex.quote(str(gazebo_live["world_live_sdf"])),
        )
        if part
    )
    gazebo_shell_prefix = (
        # Keep Gazebo out of the active conda runtime; otherwise gz transport / Qt
        # may load conda's libstdc++ and crash when opening the GUI.
        "unset LD_LIBRARY_PATH; "
        "unset PYTHONPATH; "
        "unset CONDA_PREFIX; "
        "unset CONDA_DEFAULT_ENV; "
        "unset QT_PLUGIN_PATH; "
        "unset QT_QPA_PLATFORM_PLUGIN_PATH; "
        "unset QT_QPA_PLATFORM; "
        "unset OPENCV_QT_PLUGIN_PATH; "
        "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; "
        "export XDG_RUNTIME_DIR=/tmp/runtime-$USER; "
        "mkdir -p \"$XDG_RUNTIME_DIR\"; "
        "chmod 700 \"$XDG_RUNTIME_DIR\"; "
        "source /opt/ros/jazzy/setup.bash; "
        f"export HOME={shlex.quote(home_dir)}; "
        f"export GZ_SIM_RESOURCE_PATH={shlex.quote(str(model_parent_dir))}:$GZ_SIM_RESOURCE_PATH; "
    )
    shell_cmd = gazebo_shell_prefix + f"exec {gz_command}"
    stdout_f = stdout_path.open("w", encoding="utf-8")
    stderr_f = stderr_path.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            ["bash", "-lc", shell_cmd],
            stdout=stdout_f,
            stderr=stderr_f,
            text=True,
            start_new_session=True,
        )
    finally:
        stdout_f.close()
        stderr_f.close()

    wait_seconds = max(0.0, _env_float("GAZEBO_LIVE_AUTOLAUNCH_WAIT", 2.0))
    if wait_seconds > 0.0:
        time.sleep(wait_seconds)
    if proc.poll() is not None:
        err_tail = ""
        try:
            err_tail = stderr_path.read_text(encoding="utf-8", errors="ignore")[-2000:]
        except Exception:
            pass
        launch_error = f"Gazebo live autolaunch exited early with code {proc.returncode}: {err_tail}"
        gazebo_live.update(
            {
                "autolaunch_started": False,
                "autolaunch_error": launch_error,
                "output_dir": str(output_dir),
                "scenario_json": str(scenario_json),
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
                "process_returncode": int(proc.returncode) if proc.returncode is not None else None,
                "_process": None,
                "_gui_process": None,
            }
        )
        if _env_flag("GAZEBO_LIVE_REQUIRED", False) or _env_flag("GAZEBO_LIVE_AUTOLAUNCH_REQUIRED", False):
            raise RuntimeError(launch_error)
        return gazebo_live

    gui_proc = None
    gui_error = None
    gui_stdout_path = output_dir / "gz_gui.stdout.log"
    gui_stderr_path = output_dir / "gz_gui.stderr.log"
    if launch_gui:
        gui_command = f"gz sim -g {shlex.quote(str(gazebo_live['world_live_sdf']))}"
        gui_shell_cmd = gazebo_shell_prefix + f"exec {gui_command}"
        gui_stdout_f = gui_stdout_path.open("w", encoding="utf-8")
        gui_stderr_f = gui_stderr_path.open("w", encoding="utf-8")
        try:
            gui_proc = subprocess.Popen(
                ["bash", "-lc", gui_shell_cmd],
                stdout=gui_stdout_f,
                stderr=gui_stderr_f,
                text=True,
                start_new_session=True,
            )
        finally:
            gui_stdout_f.close()
            gui_stderr_f.close()
        gui_wait_seconds = max(0.0, _env_float("GAZEBO_LIVE_GUI_WAIT", 2.0))
        if gui_wait_seconds > 0.0:
            time.sleep(gui_wait_seconds)
        if gui_proc.poll() is not None:
            try:
                gui_error = gui_stderr_path.read_text(encoding="utf-8", errors="ignore")[-2000:]
            except Exception:
                gui_error = f"Gazebo GUI exited early with code {gui_proc.returncode}"
            if _env_flag("GAZEBO_LIVE_GUI_REQUIRED", False):
                _terminate_process_group(proc)
                raise RuntimeError(f"Gazebo live GUI exited early with code {gui_proc.returncode}: {gui_error}")

    gazebo_live.update(
        {
            "autolaunch_started": True,
            "autolaunch_gui": bool(launch_gui),
            "autolaunch_run": bool(launch_run),
            "gui_error": gui_error,
            "output_dir": str(output_dir),
            "scenario_json": str(scenario_json),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "gui_stdout_log": str(gui_stdout_path) if launch_gui else None,
            "gui_stderr_log": str(gui_stderr_path) if launch_gui else None,
            "process_pid": int(proc.pid),
            "gui_process_pid": int(gui_proc.pid) if gui_proc is not None else None,
            "_process": proc,
            "_gui_process": gui_proc,
        }
    )
    if not quiet_output:
        print(f"✅ Gazebo live world 已同源导出并启动: {gazebo_live['world_live_sdf']}")
    return gazebo_live


def _reward_diag_component_mean(weighted_components, reward_names, selected_names, negative_only=False):
    try:
        name_to_idx = {str(name): idx for idx, name in enumerate(reward_names)}
        indices = [name_to_idx[name] for name in selected_names if name in name_to_idx]
        if not indices:
            return 0.0
        arr = np.asarray(weighted_components, dtype=np.float32)
        if arr.ndim == 2:
            selected = arr[:, indices]
        elif arr.ndim == 3:
            selected = arr[:, :, indices]
        else:
            return 0.0
        per_agent = np.sum(selected, axis=-1)
        if negative_only:
            per_agent = np.minimum(per_agent, 0.0)
        finite = per_agent[np.isfinite(per_agent)]
        if finite.size == 0:
            return 0.0
        return float(np.mean(finite))
    except Exception:
        return 0.0


def _reward_diag_array_mean(diag, key):
    try:
        arr = np.asarray(diag.get(key), dtype=np.float32)
        if arr.size == 0:
            return 0.0
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return 0.0
        return float(np.mean(finite))
    except Exception:
        return 0.0


def _accumulate_reward_decomposition(acc, env):
    diag = _get_vectorized_reward_diagnostics(env)
    if not diag:
        acc['reward_diag_missing_steps'] += 1
        return
    weighted_components = diag.get('weighted_components')
    reward_names = diag.get('reward_names', [])
    if weighted_components is None or not reward_names:
        acc['reward_diag_missing_steps'] += 1
        return

    components = acc['reward_components']
    components['reward_progress'] += _reward_diag_component_mean(
        weighted_components, reward_names, ('distance', 'approach', 'exploration', 'shaping')
    )
    components['reward_clearance'] += _reward_diag_component_mean(weighted_components, reward_names, ('clearance',))
    components['reward_height'] += _reward_diag_component_mean(weighted_components, reward_names, ('height',))
    components['reward_stagnation'] += _reward_diag_component_mean(weighted_components, reward_names, ('stationary',))
    components['reward_sync'] += _reward_diag_component_mean(weighted_components, reward_names, ('global',))
    components['reward_dense_success'] += _reward_diag_component_mean(weighted_components, reward_names, ('success',))
    components['reward_energy'] += _reward_diag_component_mean(weighted_components, reward_names, ('energy',))
    components['reward_collision_penalty'] += _reward_diag_component_mean(weighted_components, reward_names, ('collision',))
    components['reward_safety_penalty'] += _reward_diag_component_mean(
        weighted_components, reward_names, ('clearance', 'height', 'lateral'), negative_only=True
    )
    components['reward_terminal_success'] += _reward_diag_array_mean(diag, 'terminal_success')
    components['reward_terminal_failure'] += (
        _reward_diag_array_mean(diag, 'terminal_failure')
        + _reward_diag_array_mean(diag, 'terminal_unsafe_arrival')
    )
    components['reward_terminal_quality'] += _reward_diag_array_mean(diag, 'terminal_quality')
    acc['reward_total_before_clip_sum'] += _reward_diag_array_mean(diag, 'total_before_clip')
    acc['reward_clip_delta_sum'] += _reward_diag_array_mean(diag, 'clip_delta')
    acc['reward_diag_step_count'] += 1


def _accumulate_eval_step_diagnostics(acc, env, actions=None, raw_actions=None, pf_forces=None, action_force_ratio=None):
    _accumulate_reward_decomposition(acc, env)
    raw_norm = _mean_head_norm(raw_actions)
    corr_norm = _mean_head_norm(actions)
    pf_norm = _mean_head_norm(pf_forces)
    delta_norm = None
    try:
        if actions is not None and raw_actions is not None:
            arr_corr = np.asarray(actions, dtype=np.float32)
            arr_raw = np.asarray(raw_actions, dtype=np.float32)
            if arr_corr.shape == arr_raw.shape and arr_corr.size > 0:
                delta_norm = _mean_head_norm(arr_corr - arr_raw)
    except Exception:
        delta_norm = None

    if any(v is not None for v in (raw_norm, corr_norm, delta_norm, pf_norm)) or action_force_ratio is not None:
        _add_optional_mean(acc, 'raw_action_norm_sum', raw_norm)
        _add_optional_mean(acc, 'corr_action_norm_sum', corr_norm)
        _add_optional_mean(acc, 'action_delta_norm_sum', delta_norm)
        _add_optional_mean(acc, 'pf_force_norm_sum', pf_norm)
        _add_optional_mean(acc, 'force_ratio_sum', action_force_ratio)
        acc['action_diag_step_count'] += 1

    try:
        speeds = []
        for agent in getattr(env, 'agents', []):
            vel = _normalize_vec3(getattr(getattr(agent, 'state', None), 'p_vel', None))
            if vel is not None:
                speed = float(np.linalg.norm(vel))
                if np.isfinite(speed):
                    speeds.append(speed)
        if speeds:
            acc['speed_sum'] += float(np.mean(speeds))
            acc['speed_step_count'] += 1
    except Exception:
        pass


def _finalize_eval_diagnostics(acc):
    action_count = max(int(acc.get('action_diag_step_count', 0)), 1)
    speed_count = max(int(acc.get('speed_step_count', 0)), 1)
    semantic_count = max(int(acc.get('semantic_gap_step_count', 0)), 1)
    return {
        'reward_components': {
            key: float(value) for key, value in acc.get('reward_components', {}).items()
        },
        'diagnostic_metrics': {
            'mean_semantic_gap': float(acc.get('semantic_gap_sum', 0.0) / semantic_count),
            'max_semantic_gap': float(acc.get('semantic_gap_max', 0.0)),
            'mean_force_ratio': float(acc.get('force_ratio_sum', 0.0) / action_count),
            'avg_speed': float(acc.get('speed_sum', 0.0) / speed_count),
            'avg_action_norm': float(acc.get('raw_action_norm_sum', 0.0) / action_count),
            'avg_corr_action_norm': float(acc.get('corr_action_norm_sum', 0.0) / action_count),
            'avg_action_delta_norm': float(acc.get('action_delta_norm_sum', 0.0) / action_count),
            'mean_pf_force_norm': float(acc.get('pf_force_norm_sum', 0.0) / action_count),
            'reward_diag_step_count': int(acc.get('reward_diag_step_count', 0)),
            'reward_diag_missing_steps': int(acc.get('reward_diag_missing_steps', 0)),
            'reward_total_before_clip_sum': float(acc.get('reward_total_before_clip_sum', 0.0)),
            'reward_clip_delta_sum': float(acc.get('reward_clip_delta_sum', 0.0)),
        },
    }


def _infer_eval_method_name(args):
    text = f"{getattr(args, 'save_viz_path', '')} {getattr(args, 'load_model_path', '')}"
    known = [
        ('ds_matd3_fixed_bucket', 'DS Fixed Bucket'),
        ('ds_matd3_recent', 'DS Recent'),
        ('ds_matd3_uniform', 'DS Uniform'),
        ('ds_matd3_legacy_per', 'DS Legacy PER'),
        ('ds_matd3_original', 'DS Original'),
    ]
    for needle, label in known:
        if needle in text:
            return label
    model_path = str(getattr(args, 'load_model_path', '') or '')
    leaf = os.path.basename(os.path.dirname(model_path)) if model_path.endswith(os.sep) else os.path.basename(model_path)
    return leaf or 'unknown'


def _parse_eval_int_from_paths(args, patterns):
    text = f"{getattr(args, 'save_viz_path', '')} {getattr(args, 'load_model_path', '')}"
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None
    return None


def _agent_flags_any_two(agent_flags):
    flags = []
    try:
        flags = [int(v) for v in (agent_flags or [])]
    except Exception:
        flags = []
    return int(any(flags)), int(sum(flags) >= 2)


def _infer_episode_done_reason(success_flag, first_reach_step, step_count, episode_length, total_collisions):
    try:
        if int(success_flag) == 1:
            return 'team_success'
        if first_reach_step is not None:
            return 'all_reached_without_safe_team_success'
        if int(step_count) >= int(episode_length):
            return 'time_limit'
        if int(total_collisions or 0) > 0:
            return 'collision_or_safety_early_done'
    except Exception:
        pass
    return 'early_done'


def _write_reward_decomposition_eval_csv(output_dir, episodes, args):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'reward_decomposition_eval.csv')
    method = _infer_eval_method_name(args)
    train_seed = _parse_eval_int_from_paths(args, (r'trainseed(\d+)', r'__seed(\d+)', r'seed(\d+)'))
    eval_seed = _parse_eval_int_from_paths(args, (r'testseed(\d+)', r'eval[_-]?seed(\d+)'))
    fieldnames = [
        'method', 'train_seed', 'eval_seed', 'episode_id',
        'success', 'any_arrival', 'two_arrival', 'collision_free',
        'final_goal_distance', 'episode_length', 'total_collisions',
        'reward_total',
        *_REWARD_DECOMPOSITION_FIELDS,
        'n_terrain_collisions', 'n_obstacle_collisions',
        'n_inter_agent_collisions', 'n_boundary_violations',
        'mean_semantic_gap', 'max_semantic_gap', 'mean_force_ratio',
        'path_length', 'avg_speed', 'avg_action_norm', 'avg_corr_action_norm',
        'avg_action_delta_norm', 'mean_pf_force_norm',
        'reward_diag_step_count', 'reward_diag_missing_steps',
        'reward_total_before_clip_sum', 'reward_clip_delta_sum',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep in episodes:
            any_arrival, two_arrival = _agent_flags_any_two(ep.get('agent_success_flags', []))
            total_collisions = int(ep.get('collision_count', 0) or 0)
            inter_collisions = int(ep.get('inter_agent_collision_count', 0) or 0)
            comps = dict(ep.get('reward_decomposition') or {})
            metrics = dict(ep.get('diagnostic_metrics') or {})
            row = {
                'method': method,
                'train_seed': train_seed,
                'eval_seed': eval_seed,
                'episode_id': ep.get('episode'),
                'success': int(ep.get('team_success', ep.get('success', 0)) or 0),
                'any_arrival': any_arrival,
                'two_arrival': two_arrival,
                'collision_free': int(total_collisions == 0 and inter_collisions == 0),
                'final_goal_distance': ep.get('final_goal_distance'),
                'episode_length': ep.get('steps'),
                'total_collisions': total_collisions,
                'reward_total': ep.get('reward'),
                'n_terrain_collisions': int(ep.get('terrain_collision_count', 0) or 0),
                'n_obstacle_collisions': int(ep.get('obstacle_collision_count', 0) or 0),
                'n_inter_agent_collisions': inter_collisions,
                'n_boundary_violations': int(ep.get('boundary_violation_count', 0) or 0),
                'mean_semantic_gap': metrics.get('mean_semantic_gap', 0.0),
                'max_semantic_gap': metrics.get('max_semantic_gap', 0.0),
                'mean_force_ratio': metrics.get('mean_force_ratio', 0.0),
                'path_length': ep.get('path_length'),
                'avg_speed': metrics.get('avg_speed', 0.0),
                'avg_action_norm': metrics.get('avg_action_norm', 0.0),
                'avg_corr_action_norm': metrics.get('avg_corr_action_norm', 0.0),
                'avg_action_delta_norm': metrics.get('avg_action_delta_norm', 0.0),
                'mean_pf_force_norm': metrics.get('mean_pf_force_norm', 0.0),
                'reward_diag_step_count': metrics.get('reward_diag_step_count', 0),
                'reward_diag_missing_steps': metrics.get('reward_diag_missing_steps', 0),
                'reward_total_before_clip_sum': metrics.get('reward_total_before_clip_sum', 0.0),
                'reward_clip_delta_sum': metrics.get('reward_clip_delta_sum', 0.0),
            }
            for field in _REWARD_DECOMPOSITION_FIELDS:
                row[field] = comps.get(field, 0.0)
            writer.writerow(row)
    return path


def _write_success_reward_consistency_report(output_dir, episodes, args):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'success_reward_consistency_report.txt')
    method = _infer_eval_method_name(args)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"method={method}\n")
        f.write("team_success_reward is true when reward_terminal_success is non-zero in the eval reward diagnostics.\n")
        f.write(
            "episode_id\tteam_success_eval\tteam_success_reward\t"
            "terminal_success_bonus_applied\tsafe_flags\tfinal_distances\t"
            "collisions\tepisode_done_reason\treward_total\n"
        )
        for ep in episodes:
            comps = ep.get('reward_decomposition') or {}
            terminal_success = float(comps.get('reward_terminal_success', 0.0) or 0.0)
            applied = bool(abs(terminal_success) > 1e-8)
            f.write(
                f"{ep.get('episode')}\t"
                f"{int(ep.get('team_success', ep.get('success', 0)) or 0)}\t"
                f"{int(applied)}\t"
                f"{int(applied)}\t"
                f"{ep.get('agent_safe_flags', [])}\t"
                f"{ep.get('agent_final_goal_distances', [])}\t"
                f"{ep.get('agent_collision_counts', [])}\t"
                f"{ep.get('episode_done_reason', 'unknown')}\t"
                f"{ep.get('reward')}\n"
            )
    return path


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


def _env_int_or_default(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


def _env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _gazebo_live_collision_mode():
    raw = os.getenv("GAZEBO_LIVE_COLLISION_MODE", "hard").strip().lower()
    if raw in (
        "nonblocking",
        "non_blocking",
        "visual_only",
        "visual-only",
        "soft",
        "python_soft",
        "python-soft",
        "transfer_equivalence",
        "none",
        "disabled",
        "off",
        "0",
        "false",
        "no",
    ):
        return "nonblocking"
    return "hard"


def _env_optional_float(name):
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except Exception:
        return None
    return value if np.isfinite(value) else None


def _apply_fast_artifact_env_defaults():
    if not _env_flag("FAST_ARTIFACTS", False):
        return
    defaults = {
        "SAVE_INTERACTIVE_TRAJ": "0",
        "SAVE_EVAL_ALL_EPISODES": "0",
        "SAVE_BEST_TRAJ": "0",
        "SAVE_TEAM_SUCCESS_HTML": "0",
        "SAVE_EVAL_TRAJECTORY_JSON": "0",
        "SAVE_TRAJECTORY_SNAPSHOT": "0",
        "SAVE_GAZEBO_REPLAY": "0",
        "SAVE_GAZEBO_DYNAMIC_REPLAY": "0",
        "SAVE_GAZEBO_FAST_REPLAY": "0",
        "COMPILE_GAZEBO_FAST_REPLAY": "0",
        "GAZEBO_LIVE_SYNC": "0",
        "SAVE_EVAL_TRAJECTORY_PNG": "0",
        "SAVE_EVAL_ACTOR_SEQUENCE": "0",
        "SAVE_EVAL_CONTROL_DIAGNOSTICS": "0",
        "DISABLE_TRAJECTORY_RECORDING": "1",
        "EVAL_LIGHT_ACTION_PATH": "1",
        "EVAL_LIGHT_INFO": "1",
        "EVAL_ACTOR_ONLY": "1",
        "EVAL_VERIFY_WEIGHT_COVERAGE": "0",
        "TIMING_DETAIL": "0",
        "TIMING_LEVEL": "1",
        "DEBUG_EPISODE_SUMMARY": "0",
        "DEBUG_COLLISION_SUMMARY": "0",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _artifact_filename(filename):
    tag = str(os.getenv("EVAL_ARTIFACT_FILENAME_TAG", "") or "").strip()
    if not tag:
        return filename
    clean = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in tag)
    if not clean:
        return filename
    stem, ext = os.path.splitext(filename)
    return f"{stem}_{clean}{ext}"


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


def _model_path_has_placeholder(model_path):
    text = str(model_path or '')
    placeholders = (
        'YOUR_MODEL_DIR',
        '<MODEL_DIR>',
        '<YOUR_MODEL_DIR>',
        'MODEL_DIR_PLACEHOLDER',
    )
    return any(token in text for token in placeholders)


def _validate_load_model_path_arg(args):
    model_path = str(getattr(args, 'load_model_path', '') or '').strip()
    if not model_path:
        raise ValueError("--load-model-path 不能为空，请指定包含 actor_*.weights.h5 的模型目录")
    if _model_path_has_placeholder(model_path):
        raise ValueError(
            "--load-model-path 仍然包含示例占位符 YOUR_MODEL_DIR；"
            "请替换为真实模型目录，例如 /home/tang/matd3/models/<实验目录>/best_by_team_sr"
        )


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


def _set_scenario_terrain_complexity(scenario, terrain_level):
    """Apply an episode complexity level and refresh all derived parameters."""
    if scenario is None:
        raise ValueError("scenario is required when applying terrain complexity")
    terrain_level = max(1, min(4, int(terrain_level)))
    scenario.terrain_complexity_level = terrain_level
    setup_complexity = getattr(scenario, '_setup_complexity_parameters', None)
    if callable(setup_complexity):
        setup_complexity()
    return terrain_level


def _scenario_terrain_variant_seed(scenario, fallback=None):
    """Return a variant seed only when the scenario actually uses variants."""
    if scenario is None or not bool(getattr(scenario, 'use_semi_random_terrain', False)):
        return None
    value = getattr(
        scenario,
        'current_terrain_variant_seed',
        getattr(scenario, 'terrain_variant_seed', fallback),
    )
    return None if value is None else int(value)


def _iter_results_json_paths(model_path):
    if not model_path:
        return
    if _model_path_has_placeholder(model_path):
        return
    model_leaf = os.path.basename(os.path.normpath(model_path))
    if _is_model_variant_dir_name(model_leaf):
        model_base_dir = os.path.dirname(model_path)
    else:
        model_base_dir = model_path
    exp_name = os.path.basename(model_base_dir)
    potential_log_dirs = []
    seen_dirs = set()

    def _add_candidate_dir(path, allow_broad_root=False):
        if not path:
            return
        norm = os.path.normpath(path)
        if norm in seen_dirs or not os.path.isdir(norm):
            return
        leaf = os.path.basename(norm)
        if not allow_broad_root and leaf in ('models', 'logs'):
            return
        seen_dirs.add(norm)
        potential_log_dirs.append(norm)

    _add_candidate_dir(model_base_dir)
    # The mirrored model-local record is identity-bound to the checkpoint root.
    # Only fall back to the logs tree for legacy runs without that mirror.
    _add_candidate_dir(os.path.join("logs", exp_name))

    parent_dir = os.path.dirname(os.path.normpath(model_base_dir))
    # Some legacy layouts store results.json one level above a checkpoint
    # container. Never recurse through broad roots like ./models, because that
    # can bind an invalid model path to an unrelated experiment config.
    if os.path.isfile(os.path.join(parent_dir, 'results.json')):
        _add_candidate_dir(parent_dir, allow_broad_root=True)
    else:
        _add_candidate_dir(parent_dir)

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


_TRAINING_RUNTIME_REWARD_ENV_FIELDS = (
    ('reward_version', 'REWARD_VERSION', 'str'),
    ('reward_terminal_order_fix', 'REWARD_TERMINAL_ORDER_FIX', 'bool'),
    ('goal_ring_individual_scale', 'GOAL_RING_INDIVIDUAL_SCALE', 'float'),
    ('goal_ring_team_gated', 'GOAL_RING_TEAM_GATED', 'bool'),
    ('goal_ring_require_agent_safe', 'GOAL_RING_REQUIRE_AGENT_SAFE', 'bool'),
    ('progress_distance_state_scale', 'PROGRESS_DISTANCE_STATE_SCALE', 'float'),
    ('progress_reward_scale', 'PROGRESS_REWARD_SCALE', 'float'),
    ('team_progress_bottleneck_only', 'TEAM_PROGRESS_BOTTLENECK_ONLY', 'bool'),
    ('team_progress_non_bottleneck_scale', 'TEAM_PROGRESS_NON_BOTTLENECK_SCALE', 'float'),
    ('team_progress_bottleneck_eps', 'TEAM_PROGRESS_BOTTLENECK_EPS', 'float'),
    ('team_success_bonus', 'TEAM_SUCCESS_BONUS', 'float'),
    ('unsafe_arrival_penalty', 'UNSAFE_ARRIVAL_PENALTY', 'float'),
    ('non_success_terminal_guard_enabled', 'NON_SUCCESS_TERMINAL_GUARD_ENABLED', 'bool'),
    ('non_success_terminal_penalty_base', 'NON_SUCCESS_TERMINAL_PENALTY_BASE', 'float'),
    ('non_success_terminal_penalty_per_meter', 'NON_SUCCESS_TERMINAL_PENALTY_PER_METER', 'float'),
    ('non_success_terminal_penalty_max', 'NON_SUCCESS_TERMINAL_PENALTY_MAX', 'float'),
    ('terminal_failure_penalty_base', 'TERMINAL_FAILURE_PENALTY_BASE', 'float'),
    ('terminal_failure_penalty_per_meter', 'TERMINAL_FAILURE_PENALTY_PER_METER', 'float'),
    ('terminal_failure_penalty_max', 'TERMINAL_FAILURE_PENALTY_MAX', 'float'),
    ('clearance_quality_bonus_weight', 'CLEARANCE_QUALITY_BONUS_WEIGHT', 'float'),
    ('efficiency_bonus_weight', 'EFFICIENCY_BONUS_WEIGHT', 'float'),
    ('team_sync_reward_enabled', 'TEAM_SYNC_REWARD_ENABLED', 'bool'),
    ('team_goal_occupancy_scale', 'TEAM_GOAL_OCCUPANCY_SCALE', 'float'),
    ('team_bottleneck_progress_scale', 'TEAM_BOTTLENECK_PROGRESS_SCALE', 'float'),
    ('team_waiting_scale', 'TEAM_WAITING_SCALE', 'float'),
    ('team_bottleneck_delta_clip', 'TEAM_BOTTLENECK_DELTA_CLIP', 'float'),
    ('clearance_dense_positive_scale', 'CLEARANCE_DENSE_POSITIVE_SCALE', 'float'),
    ('height_dense_positive_scale', 'HEIGHT_DENSE_POSITIVE_SCALE', 'float'),
)


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
        training_environment = results.get('training_environment', {}) if isinstance(results, dict) else {}
        if not isinstance(training_environment, dict):
            training_environment = {}

        model_candidate = Path(str(model_path or '')).expanduser()
        if _is_model_variant_dir_name(model_candidate.name):
            model_exp_name = model_candidate.parent.name
        else:
            model_exp_name = model_candidate.name
        exp_name = str(training_args.get('exp_name') or model_exp_name or '').strip()
        explicit_manifest_path = str(results.get('training_manifest_path') or '').strip()
        expected_manifest_sha256 = str(results.get('training_manifest_sha256') or '').strip()
        runtime_manifest_path = find_training_runtime_manifest(
            Path(__file__).resolve().parent,
            exp_name,
            explicit_manifest_path or None,
        )
        manifest_runtime_environment = {}
        if runtime_manifest_path is not None:
            runtime_manifest = load_training_runtime_manifest(
                runtime_manifest_path,
                exp_name=exp_name,
                expected_content_sha256=expected_manifest_sha256 or None,
            )
            manifest_runtime_environment = runtime_environment_from_manifest(runtime_manifest)

        def _metadata_value(key):
            if key in training_environment and training_environment.get(key) is not None:
                return training_environment.get(key)
            if key in training_args and training_args.get(key) is not None:
                return training_args.get(key)
            if results.get(key) is not None:
                return results.get(key)
            return manifest_runtime_environment.get(key)

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
        terrain_seed = _metadata_value('terrain_seed')
        if terrain_seed is None:
            terrain_seed = training_args.get('scenario_seed', results.get('terrain_seed', results.get('scenario_seed')))

        snapshot = {
            'results_path': results_path,
            'training_runtime_manifest_path': (
                str(runtime_manifest_path) if runtime_manifest_path is not None else ''
            ),
            'scenario_name': scenario_name,
            'algorithm': algorithm,
            'random_terrain': _coerce_optional_bool(_metadata_value('random_terrain')),
            'use_dynamic_obstacles': _coerce_optional_bool(_metadata_value('use_dynamic_obstacles')),
            'terrain_seed': _coerce_optional_int(terrain_seed),
            'per_episode_terrain': _coerce_optional_bool(training_args.get('per_episode_terrain', results.get('per_episode_terrain'))),
            'per_env_terrain': _coerce_optional_bool(training_args.get('per_env_terrain', results.get('per_env_terrain'))),
            'semi_random_terrain': _coerce_optional_bool(_metadata_value('semi_random_terrain')),
            'terrain_base_seed': _coerce_optional_int(_metadata_value('terrain_base_seed')),
            'peak_jitter_range': _coerce_optional_float(_metadata_value('peak_jitter_range')),
            'peak_center_jitter_range': _coerce_optional_float(_metadata_value('peak_center_jitter_range')),
            'peak_height_jitter_ratio_min': _coerce_optional_float(_metadata_value('peak_height_jitter_ratio_min')),
            'peak_height_jitter_ratio_max': _coerce_optional_float(_metadata_value('peak_height_jitter_ratio_max')),
            'peak_height_max_scale': _coerce_optional_float(_metadata_value('peak_height_max_scale')),
            'terrain_variant_noise_ratio': _coerce_optional_float(_metadata_value('terrain_variant_noise_ratio')),
            'terrain_contact_eps': _coerce_optional_float(_metadata_value('terrain_contact_eps')),
            'agent_size': _coerce_optional_float(_metadata_value('agent_size')),
            'obstacle_observation_mode': normalize_obstacle_observation_mode(
                _metadata_value('obstacle_observation_mode')
            ),
            'obstacle_risk_velocity_forward_weight': _coerce_optional_float(_metadata_value('obstacle_risk_velocity_forward_weight')),
            'obstacle_risk_goal_along_weight': _coerce_optional_float(_metadata_value('obstacle_risk_goal_along_weight')),
            'terrain_complexity_level': _coerce_optional_int(training_args.get('terrain_complexity_level', results.get('terrain_complexity_level'))),
            'map_size': _coerce_optional_float(training_args.get('map_size', results.get('map_size'))),
            'mountain_min_distance': _coerce_optional_float(training_args.get('mountain_min_distance', results.get('mountain_min_distance'))),
            'max_reward': _coerce_optional_float(_metadata_value('max_reward')),
            'min_reward': _coerce_optional_float(_metadata_value('min_reward')),
            'success_reward_value': _coerce_optional_float(_metadata_value('success_reward_value')),
            'no_collision_reward_value': _coerce_optional_float(_metadata_value('no_collision_reward_value')),
            'success_distance_threshold': _coerce_optional_float(_metadata_value('success_distance_threshold')),
            'collision_penalty_value': _coerce_optional_float(_metadata_value('collision_penalty_value')),
            'collision_distance_threshold': _coerce_optional_float(_metadata_value('collision_distance_threshold')),
            'global_reward_mode': _metadata_value('global_reward_mode'),
            'shaping_gamma': _coerce_optional_float(_metadata_value('shaping_gamma')),
        }
        for runtime_field in SCIENTIFIC_RUNTIME_ENV_FIELDS:
            raw_value = _metadata_value(runtime_field.attr)
            if runtime_field.kind == 'float':
                snapshot[runtime_field.attr] = _coerce_optional_float(raw_value)
            elif runtime_field.kind == 'int':
                snapshot[runtime_field.attr] = _coerce_optional_int(raw_value)
            elif runtime_field.kind == 'bool':
                snapshot[runtime_field.attr] = _coerce_optional_bool(raw_value)
            else:
                snapshot[runtime_field.attr] = None if raw_value is None else str(raw_value)
        for field_name, _env_name, value_kind in _TRAINING_RUNTIME_REWARD_ENV_FIELDS:
            raw_value = _metadata_value(field_name)
            if value_kind == 'float':
                snapshot[field_name] = _coerce_optional_float(raw_value)
            elif value_kind == 'bool':
                snapshot[field_name] = _coerce_optional_bool(raw_value)
            else:
                snapshot[field_name] = None if raw_value is None else str(raw_value)
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
    _apply('terrain_contact_eps', snapshot.get('terrain_contact_eps'))
    _apply('agent_size', snapshot.get('agent_size'))
    _apply('obstacle_observation_mode', snapshot.get('obstacle_observation_mode'))
    _apply('obstacle_risk_velocity_forward_weight', snapshot.get('obstacle_risk_velocity_forward_weight'))
    _apply('obstacle_risk_goal_along_weight', snapshot.get('obstacle_risk_goal_along_weight'))
    _apply('terrain_complexity_level', snapshot.get('terrain_complexity_level'))
    _apply('map_size', snapshot.get('map_size'))
    _apply('mountain_min_distance', snapshot.get('mountain_min_distance'))
    _apply('max_reward', snapshot.get('max_reward'))
    _apply('min_reward', snapshot.get('min_reward'))
    _apply('success_reward_value', snapshot.get('success_reward_value'))
    _apply('no_collision_reward_value', snapshot.get('no_collision_reward_value'))
    _apply('success_distance_threshold', snapshot.get('success_distance_threshold'))
    _apply('collision_penalty_value', snapshot.get('collision_penalty_value'))
    _apply('collision_distance_threshold', snapshot.get('collision_distance_threshold'))
    _apply('global_reward_mode', snapshot.get('global_reward_mode'))
    _apply('shaping_gamma', snapshot.get('shaping_gamma'))
    for runtime_field in SCIENTIFIC_RUNTIME_ENV_FIELDS:
        _apply(runtime_field.attr, snapshot.get(runtime_field.attr))
    for field_name, _env_name, _value_kind in _TRAINING_RUNTIME_REWARD_ENV_FIELDS:
        _apply(field_name, snapshot.get(field_name))

    if applied and not quiet:
        pretty = ", ".join(f"{k}={applied[k]}" for k in sorted(applied.keys()))
        print(f"✅ 评估前按训练配置对齐环境参数: {pretty}")
        print(f"   来源: {snapshot.get('results_path')}")
    return applied


def _apply_runtime_env_overrides_from_args(args):
    """将依赖环境变量的隐藏运行时参数与args保持同步。"""
    apply_runtime_environment(args)
    runtime_pairs = (
        ("simulation_dt", "SIMULATION_DT"),
        ("z_action_bias", "Z_ACTION_BIAS"),
        ("quadrotor_attitude_response_time", "QUADROTOR_ATTITUDE_RESPONSE_TIME"),
        ("quadrotor_psi_cmd", "QUADROTOR_PSI_CMD"),
        ("agent_size", "AGENT_SIZE"),
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
        obstacle_observation_mode = getattr(args, "obstacle_observation_mode", None)
    except Exception:
        obstacle_observation_mode = None
    if obstacle_observation_mode is not None:
        obstacle_observation_mode = normalize_obstacle_observation_mode(obstacle_observation_mode)
        args.obstacle_observation_mode = obstacle_observation_mode
        os.environ["OBSTACLE_OBSERVATION_MODE"] = obstacle_observation_mode
        os.environ["OBSTACLE_OBS_MODE"] = obstacle_observation_mode

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
        ("terrain_contact_eps", "TERRAIN_CONTACT_EPS"),
        ("obstacle_risk_velocity_forward_weight", "OBSTACLE_RISK_VELOCITY_FORWARD_WEIGHT"),
        ("obstacle_risk_goal_along_weight", "OBSTACLE_RISK_GOAL_ALONG_WEIGHT"),
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

    for attr_name, env_name, value_kind in _TRAINING_RUNTIME_REWARD_ENV_FIELDS:
        value = getattr(args, attr_name, None)
        if value is None:
            continue
        if value_kind == 'bool':
            os.environ[env_name] = '1' if bool(value) else '0'
        else:
            os.environ[env_name] = str(value)

    numeric_env_pairs = (
        ("terrain_complexity_level", "TERRAIN_COMPLEXITY_LEVEL", int),
        ("map_size", "MAP_SIZE", float),
        ("mountain_min_distance", "MOUNTAIN_MIN_DISTANCE", int),
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


def _apply_terrain_runtime_params_to_scenario(
    scenario,
    world,
    args,
    *,
    preserve_episode_terrain=False,
):
    """将地形关键参数显式下发到scenario/world。

    ``args.terrain_seed`` 是训练/评估的基础配置，不一定是当前回合已经显式
    生成的地形种子。重建某个回合的 world 时必须保留 scenario 中已生成的
    seed、variant 和 RNG 状态，否则会形成“地形数组来自新种子、元数据和
    后续位置随机流却来自基础种子”的混合环境。
    """
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
        ('terrain_contact_eps', float, '_terrain_contact_eps'),
        ('terrain_contact_eps', float, 'terrain_contact_eps'),
        ('terrain_complexity_level', int, 'terrain_complexity_level'),
        ('map_size', float, 'map_size'),
        ('obstacle_observation_mode', str, 'obstacle_observation_mode'),
        ('obstacle_risk_velocity_forward_weight', float, 'obstacle_risk_velocity_forward_weight'),
        ('obstacle_risk_goal_along_weight', float, 'obstacle_risk_goal_along_weight'),
    )
    for arg_name, caster, scenario_attr in scalar_mappings:
        try:
            value = getattr(args, arg_name, None)
        except Exception:
            value = None
        if value is None:
            continue
        if scenario_attr == 'obstacle_observation_mode':
            setattr(scenario, scenario_attr, normalize_obstacle_observation_mode(value))
            continue
        try:
            setattr(scenario, scenario_attr, caster(value))
        except Exception:
            continue

    try:
        terrain_seed = getattr(args, 'terrain_seed', None)
    except Exception:
        terrain_seed = None
    if terrain_seed is not None and not preserve_episode_terrain:
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
    if terrain_variant_seed is not None and not preserve_episode_terrain:
        try:
            scenario.terrain_variant_seed = int(terrain_variant_seed)
            setattr(scenario, 'current_terrain_variant_seed', int(terrain_variant_seed))
        except Exception:
            pass

    if world is None:
        return
    if hasattr(world, 'terrain_seed'):
        try:
            effective_terrain_seed = (
                getattr(scenario, 'current_terrain_seed', getattr(scenario, 'seed', None))
                if preserve_episode_terrain
                else terrain_seed
            )
            if effective_terrain_seed is not None:
                world.terrain_seed = int(effective_terrain_seed)
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


def _normalize_vecn(vec, length, default=None):
    if vec is None:
        if default is None:
            return None
        return np.asarray(default, dtype=np.float64)[:length]
    try:
        arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    except Exception:
        if default is None:
            return None
        return np.asarray(default, dtype=np.float64)[:length]
    if arr.size < length:
        if default is None:
            return None
        out = np.asarray(default, dtype=np.float64).reshape(-1)
        if out.size < length:
            return None
        return out[:length]
    arr = arr[:length]
    if not np.all(np.isfinite(arr)):
        if default is None:
            return None
        return np.asarray(default, dtype=np.float64)[:length]
    return arr


def _normalize_quat_wxyz(quat):
    arr = _normalize_vecn(quat, 4, default=[1.0, 0.0, 0.0, 0.0])
    if arr is None:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 1e-9:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return arr / norm


def _copy_action_frame(actions):
    if actions is None:
        return []
    frame = []
    try:
        for action in actions:
            arr = np.asarray(action, dtype=np.float64).reshape(-1)
            frame.append(arr.tolist())
    except Exception:
        return []
    return frame


def _capture_agent_dynamic_states(agents):
    frames = []
    for agent in agents:
        state = getattr(agent, "state", None)
        pos = _normalize_vec3(getattr(state, "p_pos", None))
        vel = _normalize_vec3(getattr(state, "p_vel", None))
        acc = _normalize_vec3(getattr(state, "p_acc", None))
        angular_vel = _normalize_vec3(getattr(state, "angular_vel", None))
        orientation = _normalize_quat_wxyz(getattr(state, "orientation", None))
        motors = _normalize_vecn(getattr(state, "motor_speeds", None), 4, default=[0.0, 0.0, 0.0, 0.0])
        frames.append(
            {
                "position": (pos.tolist() if pos is not None else [0.0, 0.0, 0.0]),
                "velocity": (vel.tolist() if vel is not None else [0.0, 0.0, 0.0]),
                "acceleration": (acc.tolist() if acc is not None else [0.0, 0.0, 0.0]),
                "orientation": orientation.tolist(),
                "angular_velocity": (angular_vel.tolist() if angular_vel is not None else [0.0, 0.0, 0.0]),
                "motor_speeds": (motors.tolist() if motors is not None else [0.0, 0.0, 0.0, 0.0]),
            }
        )
    return frames


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


def _capture_agent_velocities(agents):
    velocities = []
    for agent in agents:
        vel = _normalize_vec3(getattr(getattr(agent, "state", None), "p_vel", None))
        velocities.append(vel)
    return velocities


def _apply_agent_velocity_frame(agents, velocities):
    try:
        arr = np.asarray(velocities, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
    except Exception:
        return False
    for idx, agent in enumerate(list(agents)[: arr.shape[0]]):
        state = getattr(agent, "state", None)
        if state is None or arr.shape[1] < 3:
            continue
        vel = arr[idx, :3]
        if not np.all(np.isfinite(vel)):
            continue
        state.p_vel = vel.astype(np.float64, copy=True)
    return True


def _positions_from_gazebo_state_data(data, agent_count):
    if not isinstance(data, dict):
        return None
    entries = data.get("agents", [])
    if not isinstance(entries, list) or len(entries) < int(agent_count):
        return None
    positions = []
    for agent_idx in range(int(agent_count)):
        entry = entries[agent_idx]
        if not isinstance(entry, dict) or not bool(entry.get("seen", False)):
            return None
        pos = _normalize_vec3(entry.get("position"))
        if pos is None:
            return None
        positions.append(pos)
    return positions


def _vec_frames_to_lists(values):
    out = []
    for value in values or []:
        if value is None:
            out.append(None)
        else:
            out.append(np.asarray(value, dtype=np.float64).reshape(-1).tolist())
    return out


def _json_safe_eval_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe_eval_value(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe_eval_value(value.tolist())
    if isinstance(value, dict):
        return {str(k): _json_safe_eval_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_eval_value(v) for v in value]
    try:
        return float(value) if np.isfinite(value) else None
    except Exception:
        return str(value)


def _array_row_to_list(values, idx, limit=None):
    try:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if idx >= arr.shape[0]:
            return None
        row = arr[idx]
        if limit is not None:
            row = row[: int(limit)]
        return _json_safe_eval_value(row)
    except Exception:
        return None


def _norm3_or_none(values):
    try:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)[:3]
        if arr.size < 3 or not np.all(np.isfinite(arr)):
            return None
        return float(np.linalg.norm(arr))
    except Exception:
        return None


def _gazebo_contact_pair_class(pair):
    if not isinstance(pair, dict):
        return "invalid"
    collision1 = str(pair.get("collision1", "") or "")
    collision2 = str(pair.get("collision2", "") or "")
    text = f"{collision1} {collision2}".lower()
    if not collision1 and not collision2:
        return "topic_marker_only"
    has_agent_envelope = "python_collision_envelope_collision" in text
    if not has_agent_envelope:
        return "unrelated_collision_pair"
    if "obstacle" in text:
        return "agent_vs_obstacle"
    if "terrain" in text or "matd3_terrain" in text:
        return "agent_vs_terrain"
    if text.count("dynamic_agent") >= 2:
        return "agent_vs_agent"
    return "other_agent_envelope_contact"


def _gazebo_contact_pair_is_real(pair):
    return _gazebo_contact_pair_class(pair) in (
        "agent_vs_obstacle",
        "agent_vs_terrain",
        "agent_vs_agent",
    )


def _gazebo_contact_pair_agent_index(pair, fallback_prefix=None):
    if not isinstance(pair, dict):
        return None
    texts = [
        str(pair.get("agent", "") or ""),
        str(pair.get("collision1", "") or ""),
        str(pair.get("collision2", "") or ""),
        str(pair.get("topic", "") or ""),
    ]
    patterns = []
    if fallback_prefix:
        patterns.append(re.escape(str(fallback_prefix)) + r"(\d+)")
    patterns.extend([r"dynamic_agent[^:\s/]*_(\d+)", r"agent[_-]?(\d+)"])
    for text in texts:
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return int(match.group(1))
                except Exception:
                    continue
    return None


def _compact_gazebo_contact_pair(pair, agent_prefix=None):
    if not isinstance(pair, dict):
        return {}
    out = {
        "topic": pair.get("topic"),
        "agent": pair.get("agent"),
        "agent_id": _gazebo_contact_pair_agent_index(pair, fallback_prefix=agent_prefix),
        "collision1": pair.get("collision1"),
        "collision2": pair.get("collision2"),
        "contacts": pair.get("contacts"),
        "class": _gazebo_contact_pair_class(pair),
    }
    return {k: v for k, v in out.items() if v not in (None, "", [])}


def _real_gazebo_contact_pairs(contact_pairs, agent_prefix=None):
    real_pairs = []
    for pair in contact_pairs or []:
        if _gazebo_contact_pair_is_real(pair):
            real_pairs.append(_compact_gazebo_contact_pair(pair, agent_prefix=agent_prefix))
    return real_pairs


def _mark_gazebo_contact_collision_on_agent(agent, contact_classes, step_count=None):
    try:
        agent._episode_has_collision = True
        agent._had_penetration_or_collision = True
    except Exception:
        pass
    if not hasattr(agent, "debug_info") or not isinstance(getattr(agent, "debug_info", None), dict):
        try:
            agent.debug_info = {}
        except Exception:
            pass
    debug = getattr(agent, "debug_info", {}) if isinstance(getattr(agent, "debug_info", None), dict) else {}
    classes = set(str(v) for v in (contact_classes or []) if v)
    if "agent_vs_terrain" in classes:
        try:
            agent._had_terrain_contact_or_penetration = True
        except Exception:
            pass
    if "agent_vs_obstacle" in classes:
        try:
            agent._had_obstacle_collision = True
        except Exception:
            pass
    try:
        debug["gazebo_contact_collision_count"] = int(debug.get("gazebo_contact_collision_count", 0) or 0) + 1
        debug["total_penetration_count"] = int(debug.get("total_penetration_count", 0) or 0) + 1
        if "agent_vs_terrain" in classes:
            debug["terrain_penetration_count"] = int(debug.get("terrain_penetration_count", 0) or 0) + 1
        if "agent_vs_obstacle" in classes:
            debug["obstacle_collision_count"] = int(debug.get("obstacle_collision_count", 0) or 0) + 1
        if "agent_vs_agent" in classes:
            debug["agent_agent_contact_count"] = int(debug.get("agent_agent_contact_count", 0) or 0) + 1
        if step_count is not None and debug.get("first_gazebo_contact_step") is None:
            debug["first_gazebo_contact_step"] = int(step_count)
    except Exception:
        pass
    try:
        if not hasattr(agent, "current_episode_collision_count"):
            agent.current_episode_collision_count = 0
        agent.current_episode_collision_count = int(agent.current_episode_collision_count) + 1
    except Exception:
        pass


def _apf_step_debug_frame(
    step_count,
    raw_actions,
    original_corrected,
    original_pf,
    gazebo_result,
    comparison_records,
    cmd_vel,
    gazebo_pose,
    nominal_cmd_vel=None,
    safety_filter_records=None,
):
    comp_by_agent = {}
    for record in comparison_records or []:
        try:
            comp_by_agent[int(record.get("agent_id"))] = record
        except Exception:
            continue
    safety_by_agent = {}
    for record in safety_filter_records or []:
        try:
            safety_by_agent[int(record.get("agent_id"))] = record
        except Exception:
            continue
    count = 0
    try:
        count = max(count, int(np.asarray(raw_actions).shape[0]))
    except Exception:
        pass
    if gazebo_result is not None:
        try:
            count = max(count, int(gazebo_result.raw_actions.shape[0]))
        except Exception:
            pass
        try:
            count = max(count, len(gazebo_result.debug))
        except Exception:
            pass
    frame = {
        "step": int(step_count),
        "raw_actions": _copy_action_frame(raw_actions),
        "python_apf_outputs": [],
        "gazebo_apf_outputs": [],
        "nominal_cmd_vel": _vec_frames_to_lists(nominal_cmd_vel),
        "cmd_vel": _vec_frames_to_lists(cmd_vel),
        "gazebo_pose": _vec_frames_to_lists(gazebo_pose),
        "agents": [],
    }
    for idx in range(count):
        debug = {}
        if gazebo_result is not None and idx < len(getattr(gazebo_result, "debug", [])):
            debug = gazebo_result.debug[idx] or {}
        nearest = debug.get("nearest_obstacle") or {}
        original_corr = _array_row_to_list(original_corrected, idx)
        original_pf_vec = _array_row_to_list(original_pf, idx, limit=3)
        gazebo_corr = debug.get("corrected_action")
        gazebo_pf_vec = debug.get("pf_force_action")
        comp = comp_by_agent.get(idx, {})
        safety = safety_by_agent.get(idx, {})
        py_out = {
            "agent_id": idx,
            "corrected_action": original_corr,
            "pf_vector": original_pf_vec,
            "pf_norm": _norm3_or_none(original_pf_vec),
        }
        gz_out = {
            "agent_id": idx,
            "corrected_action": _json_safe_eval_value(gazebo_corr),
            "corrected_acceleration": _json_safe_eval_value(debug.get("corrected_acceleration")),
            "pf_vector": _json_safe_eval_value(gazebo_pf_vec),
            "pf_norm": _norm3_or_none(gazebo_pf_vec),
            "cmd_vel": _json_safe_eval_value(debug.get("cmd_vel")),
            "nearest_obstacle": _json_safe_eval_value(nearest),
            "terrain_clearance": _json_safe_eval_value(debug.get("terrain_clearance")),
        }
        agent_row = {
            "agent_id": idx,
            "raw_action": _array_row_to_list(raw_actions, idx),
            "python_apf": py_out,
            "gazebo_apf": gz_out,
            "comparison": _json_safe_eval_value(comp),
            "nearest_obstacle_id": nearest.get("name"),
            "surface_distance": nearest.get("surface_distance"),
            "nearest_obstacle_clearance": nearest.get("clearance"),
            "terrain_clearance": debug.get("terrain_clearance"),
            "nominal_cmd_vel": _array_row_to_list(nominal_cmd_vel, idx, limit=3),
            "cmd_vel": _array_row_to_list(cmd_vel, idx, limit=3),
            "gazebo_pose": _array_row_to_list(gazebo_pose, idx, limit=3),
            "safety_filter": _json_safe_eval_value(safety),
        }
        frame["python_apf_outputs"].append(py_out)
        frame["gazebo_apf_outputs"].append(gz_out)
        frame["agents"].append(_json_safe_eval_value(agent_row))
    return _json_safe_eval_value(frame)


def _filter_contact_debug_window(records, start_step, end_step, agent_id=None, obstacle_name=None):
    out = []
    for frame in records or []:
        try:
            step = int(frame.get("step"))
        except Exception:
            continue
        if step < int(start_step) or step > int(end_step):
            continue
        agents = []
        for agent in frame.get("agents", []) or []:
            try:
                current_agent_id = int(agent.get("agent_id"))
            except Exception:
                current_agent_id = None
            if agent_id is not None and current_agent_id != int(agent_id):
                continue
            nearest = agent.get("nearest_obstacle_id")
            if obstacle_name and nearest != obstacle_name:
                continue
            agents.append(agent)
        if agents:
            row = dict(frame)
            row["agents"] = agents
            out.append(row)
    return out


def _surface_distance_trace(debug_window, agent_id=None, obstacle_name=None):
    trace = []
    for frame in debug_window or []:
        step = frame.get("step")
        for agent in frame.get("agents", []) or []:
            if agent_id is not None and agent.get("agent_id") != agent_id:
                continue
            if obstacle_name and agent.get("nearest_obstacle_id") != obstacle_name:
                continue
            trace.append(
                {
                    "step": step,
                    "agent_id": agent.get("agent_id"),
                    "nearest_obstacle_id": agent.get("nearest_obstacle_id"),
                    "surface_distance": agent.get("surface_distance"),
                    "nearest_obstacle_clearance": agent.get("nearest_obstacle_clearance"),
                    "terrain_clearance": agent.get("terrain_clearance"),
                }
            )
    return trace


def _write_contact_debug_window_artifacts(
    output_dir,
    episode_idx,
    debug_window,
    first_contact_step=None,
    first_contact_pair=None,
    agent_id=None,
    obstacle_name=None,
):
    if not output_dir or not debug_window:
        return {}
    try:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return {}

    safe_obstacle = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(obstacle_name or "obstacle"))
    safe_agent = "all" if agent_id is None else str(agent_id)
    stem = f"hard_contact_debug_ep{int(episode_idx):03d}_agent{safe_agent}_{safe_obstacle}"
    payload = {
        "episode": int(episode_idx),
        "agent_id": agent_id,
        "obstacle_name": obstacle_name,
        "first_contact_step": int(first_contact_step) if first_contact_step is not None else None,
        "first_contact_pair": first_contact_pair,
        "window": debug_window,
    }
    paths = {}
    try:
        json_path = out_dir / f"{stem}.json"
        json_path.write_text(json.dumps(_json_safe_eval_value(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        paths["json"] = str(json_path)
    except Exception:
        pass

    rows = []
    for frame in debug_window or []:
        for agent in frame.get("agents", []) or []:
            py = agent.get("python_apf", {}) or {}
            gz = agent.get("gazebo_apf", {}) or {}
            comp = agent.get("comparison", {}) or {}
            py_pf = py.get("pf_vector") or []
            gz_pf = gz.get("pf_vector") or []
            py_corr = py.get("corrected_action") or []
            gz_corr = gz.get("corrected_action") or []
            nominal_cmd = agent.get("nominal_cmd_vel") or []
            cmd = agent.get("cmd_vel") or []
            pose = agent.get("gazebo_pose") or []
            safety = agent.get("safety_filter") or {}
            rows.append(
                {
                    "step": frame.get("step"),
                    "agent_id": agent.get("agent_id"),
                    "nearest_obstacle_id": agent.get("nearest_obstacle_id"),
                    "surface_distance": agent.get("surface_distance"),
                    "terrain_clearance": agent.get("terrain_clearance"),
                    "python_corr_ax": py_corr[0] if len(py_corr) > 0 else "",
                    "python_corr_ay": py_corr[1] if len(py_corr) > 1 else "",
                    "python_corr_az": py_corr[2] if len(py_corr) > 2 else "",
                    "gazebo_corr_ax": gz_corr[0] if len(gz_corr) > 0 else "",
                    "gazebo_corr_ay": gz_corr[1] if len(gz_corr) > 1 else "",
                    "gazebo_corr_az": gz_corr[2] if len(gz_corr) > 2 else "",
                    "python_pf_x": py_pf[0] if len(py_pf) > 0 else "",
                    "python_pf_y": py_pf[1] if len(py_pf) > 1 else "",
                    "python_pf_z": py_pf[2] if len(py_pf) > 2 else "",
                    "python_pf_norm": py.get("pf_norm"),
                    "gazebo_pf_x": gz_pf[0] if len(gz_pf) > 0 else "",
                    "gazebo_pf_y": gz_pf[1] if len(gz_pf) > 1 else "",
                    "gazebo_pf_z": gz_pf[2] if len(gz_pf) > 2 else "",
                    "gazebo_pf_norm": gz.get("pf_norm"),
                    "corrected_action_error": comp.get("corrected_action_error"),
                    "pf_force_error": comp.get("pf_force_error"),
                    "direction_cosine_similarity": comp.get("direction_cosine_similarity"),
                    "difference_reason": comp.get("difference_reason"),
                    "filter_mode": safety.get("mode"),
                    "filter_active": safety.get("filter_active"),
                    "filter_trigger_reason": safety.get("filter_trigger_reason"),
                    "inward_velocity_before_filter": safety.get("inward_velocity_before_filter"),
                    "inward_velocity_after_filter": safety.get("inward_velocity_after_filter"),
                    "relative_inward_velocity_before_filter": safety.get("relative_inward_velocity_before_filter"),
                    "relative_inward_velocity_after_filter": safety.get("relative_inward_velocity_after_filter"),
                    "current_relative_inward_velocity": safety.get("current_relative_inward_velocity"),
                    "closing_inward_velocity_for_stopping": safety.get("closing_inward_velocity_for_stopping"),
                    "outward_speed_applied": safety.get("outward_speed_applied"),
                    "cmd_delta_norm": safety.get("cmd_delta_norm"),
                    "goal_distance": safety.get("goal_distance"),
                    "goal_projection_before_filter": safety.get("goal_projection_before_filter"),
                    "goal_projection_after_filter": safety.get("goal_projection_after_filter"),
                    "tangential_velocity_before_filter": safety.get("tangential_velocity_before_filter"),
                    "tangential_velocity_after_filter": safety.get("tangential_velocity_after_filter"),
                    "tangential_velocity_kept_ratio": safety.get("tangential_velocity_kept_ratio"),
                    "outward_velocity_added": safety.get("outward_velocity_added"),
                    "filter_invasiveness": safety.get("filter_invasiveness"),
                    "boundary_dwell_steps": safety.get("boundary_dwell_steps"),
                    "line_to_goal_blocked": safety.get("line_to_goal_blocked"),
                    "nearest_agent_distance": safety.get("nearest_agent_distance"),
                    "formation_error": safety.get("formation_error"),
                    "avoidance_state": safety.get("avoidance_state"),
                    "halfspace_projection_delta_norm": safety.get("halfspace_projection_delta_norm"),
                    "tangent_recovery_applied": safety.get("tangent_recovery_applied"),
                    "goal_projection_recovery_applied": safety.get("goal_projection_recovery_applied"),
                    "nominal_cmd_vx": nominal_cmd[0] if len(nominal_cmd) > 0 else "",
                    "nominal_cmd_vy": nominal_cmd[1] if len(nominal_cmd) > 1 else "",
                    "nominal_cmd_vz": nominal_cmd[2] if len(nominal_cmd) > 2 else "",
                    "cmd_vx": cmd[0] if len(cmd) > 0 else "",
                    "cmd_vy": cmd[1] if len(cmd) > 1 else "",
                    "cmd_vz": cmd[2] if len(cmd) > 2 else "",
                    "pose_x": pose[0] if len(pose) > 0 else "",
                    "pose_y": pose[1] if len(pose) > 1 else "",
                    "pose_z": pose[2] if len(pose) > 2 else "",
                }
            )
    try:
        if rows:
            csv_path = out_dir / f"{stem}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            paths["csv"] = str(csv_path)
    except Exception:
        pass

    try:
        if rows:
            steps = np.asarray([float(r["step"]) for r in rows], dtype=np.float64)
            surface = np.asarray(
                [float(r["surface_distance"]) if r["surface_distance"] not in (None, "") else np.nan for r in rows],
                dtype=np.float64,
            )
            py_norm = np.asarray(
                [float(r["python_pf_norm"]) if r["python_pf_norm"] not in (None, "") else np.nan for r in rows],
                dtype=np.float64,
            )
            gz_norm = np.asarray(
                [float(r["gazebo_pf_norm"]) if r["gazebo_pf_norm"] not in (None, "") else np.nan for r in rows],
                dtype=np.float64,
            )
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            ax = axes[0]
            ax.plot(steps, surface, label="surface_distance", color="tab:red")
            ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
            if first_contact_step is not None:
                ax.axvline(float(first_contact_step), color="tab:orange", linewidth=1.2, label="first_contact")
            ax.set_xlabel("step")
            ax.set_ylabel("distance")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

            ax = axes[1]
            poses = np.asarray(
                [
                    [r["pose_x"], r["pose_y"]]
                    for r in rows
                    if r["pose_x"] not in (None, "") and r["pose_y"] not in (None, "")
                ],
                dtype=np.float64,
            )
            if poses.size:
                ax.plot(poses[:, 0], poses[:, 1], color="black", linewidth=1.0, label="gazebo pose")
                ax.scatter([poses[0, 0]], [poses[0, 1]], color="green", s=25, label="window start")
                ax.scatter([poses[-1, 0]], [poses[-1, 1]], color="red", s=25, label="window end")
            first_nearest = None
            for frame in debug_window:
                agents = frame.get("agents", []) or []
                if agents:
                    first_nearest = (agents[0].get("gazebo_apf", {}) or {}).get("nearest_obstacle")
                    if first_nearest:
                        break
            if isinstance(first_nearest, dict) and first_nearest.get("center") is not None:
                center = np.asarray(first_nearest.get("center"), dtype=np.float64).reshape(-1)
                radius = float(first_nearest.get("radius", 0.0) or 0.0)
                if center.size >= 2:
                    ax.add_patch(plt.Circle((center[0], center[1]), radius, fill=False, color="tab:red", linewidth=1.2))
                    ax.scatter([center[0]], [center[1]], color="tab:red", s=20, label=first_nearest.get("name", "obstacle"))
            if poses.size:
                stride = max(1, len(rows) // 12)
                for r in rows[::stride]:
                    try:
                        x = float(r["pose_x"])
                        y = float(r["pose_y"])
                        pfx = float(r["python_pf_x"])
                        pfy = float(r["python_pf_y"])
                        gfx = float(r["gazebo_pf_x"])
                        gfy = float(r["gazebo_pf_y"])
                        ax.arrow(x, y, pfx * 3.0, pfy * 3.0, color="tab:blue", width=0.05, alpha=0.6)
                        ax.arrow(x, y, gfx * 3.0, gfy * 3.0, color="tab:orange", width=0.05, alpha=0.6)
                    except Exception:
                        continue
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("blue=python PF, orange=gazebo PF")
            ax.axis("equal")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

            fig.suptitle(f"Hard contact debug ep={episode_idx}, agent={agent_id}, obstacle={obstacle_name}")
            fig.tight_layout()
            plot_path = out_dir / f"{stem}.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            paths["plot"] = str(plot_path)
            paths["pf_norm_series"] = {
                "step": steps.astype(float).tolist(),
                "python_pf_norm": py_norm.astype(float).tolist(),
                "gazebo_pf_norm": gz_norm.astype(float).tolist(),
            }
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
    return _json_safe_eval_value(paths)


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


def _compute_inter_agent_collision_snapshot(agents):
    agents = list(agents or [])
    positions = _capture_agent_positions(agents)
    per_agent_counts = [0] * len(agents)
    pair_count = 0
    min_clearance = None

    for i in range(len(agents)):
        pos_i = positions[i]
        if pos_i is None:
            continue
        try:
            size_i = float(getattr(agents[i], "size", 0.0) or 0.0)
        except Exception:
            size_i = 0.0
        for j in range(i + 1, len(agents)):
            pos_j = positions[j]
            if pos_j is None:
                continue
            try:
                size_j = float(getattr(agents[j], "size", 0.0) or 0.0)
            except Exception:
                size_j = 0.0
            center_dist = float(np.linalg.norm(pos_i - pos_j))
            clearance = center_dist - (size_i + size_j)
            if min_clearance is None or clearance < min_clearance:
                min_clearance = float(clearance)
            if clearance < 0.0:
                pair_count += 1
                per_agent_counts[i] += 1
                per_agent_counts[j] += 1

    return pair_count, per_agent_counts, min_clearance


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
        "agent_reach_rates": [],
        "agent_safe_rates": [],
        "done_reason_counts": {},
        "all_reached_without_safe_team_success_count": 0,
        "all_reached_without_safe_team_success_rate": None,
        "two_success_not_team_count": 0,
        "two_success_not_team_rate": None,
        "all_safe_not_team_count": 0,
        "all_safe_not_team_rate": None,
        "unsafe_reached_agent_slot_count": 0,
        "unsafe_reached_agent_slot_rate": None,
        "avg_steps": None,
        "std_steps": None,
        "avg_collision_count": None,
        "std_collision_count": None,
        "collision_free_rate": None,
        "avg_terrain_collision_count": None,
        "std_terrain_collision_count": None,
        "avg_obstacle_collision_count": None,
        "std_obstacle_collision_count": None,
        "avg_inter_agent_collision_count": None,
        "std_inter_agent_collision_count": None,
        "inter_agent_collision_free_rate": None,
        "avg_min_clearance_mean": None,
        "avg_min_clearance_min": None,
        "avg_min_inter_agent_clearance": None,
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
    terrain_collision_counts = [ep.get("terrain_collision_count", 0) for ep in all_episodes_data]
    obstacle_collision_counts = [ep.get("obstacle_collision_count", 0) for ep in all_episodes_data]
    inter_agent_collision_counts = [ep.get("inter_agent_collision_count", 0) for ep in all_episodes_data]
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
    min_inter_agent_clearances = []
    penetration_counts = []
    penetration_depths = []
    collision_free_count = 0
    inter_agent_collision_free_count = 0
    violation_count = 0

    agent_success_lists = []
    agent_reach_lists = []
    agent_safe_lists = []
    agent_path_length_lists = []
    agent_path_length_success_lists = []
    agent_path_efficiency_lists = []
    agent_path_efficiency_success_lists = []
    agent_final_goal_distance_lists = []
    agent_min_goal_distance_lists = []
    done_reason_counts = {}
    all_reached_without_safe_team_success_count = 0
    two_success_not_team_count = 0
    all_safe_not_team_count = 0
    unsafe_reached_agent_slot_count = 0
    reached_agent_slot_count = 0

    def _episode_flag(value):
        try:
            return 1 if int(value) != 0 else 0
        except Exception:
            return 0

    def _episode_reach_flags(ep):
        explicit_flags = ep.get("agent_reach_flags")
        if isinstance(explicit_flags, list) and explicit_flags:
            return [_episode_flag(flag) for flag in explicit_flags]
        reach_steps = ep.get("agent_first_reach_steps", [])
        flags = []
        if isinstance(reach_steps, list):
            for step in reach_steps:
                numeric = _finite_float_or_none(step)
                flags.append(1 if numeric is not None and numeric >= 0 else 0)
        return flags

    for ep in all_episodes_data:
        team_success_flag = int(ep.get("team_success", ep.get("success", 0)) or 0)
        done_reason = ep.get("episode_done_reason") or ep.get("done_reason") or "unknown"
        done_reason_counts[str(done_reason)] = done_reason_counts.get(str(done_reason), 0) + 1

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

        inter_agent_collision_count = ep.get("inter_agent_collision_count", 0)
        try:
            if int(inter_agent_collision_count) <= 0:
                inter_agent_collision_free_count += 1
        except Exception:
            pass

        min_inter_agent_clearance = _finite_float_or_none(ep.get("min_inter_agent_clearance"))
        if min_inter_agent_clearance is not None:
            min_inter_agent_clearances.append(min_inter_agent_clearance)

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

        agent_reach_flags = _episode_reach_flags(ep)
        if agent_reach_flags:
            for idx, flag in enumerate(agent_reach_flags):
                while len(agent_reach_lists) <= idx:
                    agent_reach_lists.append([])
                agent_reach_lists[idx].append(_episode_flag(flag))

        agent_safe_flags = ep.get("agent_safe_flags", [])
        if isinstance(agent_safe_flags, list):
            for idx, flag in enumerate(agent_safe_flags):
                while len(agent_safe_lists) <= idx:
                    agent_safe_lists.append([])
                agent_safe_lists[idx].append(_episode_flag(flag))

        success_count = 0
        if isinstance(agent_success_flags, list):
            success_count = sum(_episode_flag(flag) for flag in agent_success_flags)
        reach_count = sum(_episode_flag(flag) for flag in agent_reach_flags)
        safe_count = 0
        if isinstance(agent_safe_flags, list):
            safe_count = sum(_episode_flag(flag) for flag in agent_safe_flags)
        n_agents_diag = max(
            len(agent_success_flags) if isinstance(agent_success_flags, list) else 0,
            len(agent_reach_flags),
            len(agent_safe_flags) if isinstance(agent_safe_flags, list) else 0,
        )
        all_reached = bool(n_agents_diag > 0 and reach_count >= n_agents_diag)
        if team_success_flag == 0 and (
            str(done_reason) == "all_reached_without_safe_team_success" or all_reached
        ):
            all_reached_without_safe_team_success_count += 1
        if team_success_flag == 0 and success_count >= min(2, max(n_agents_diag, 1)):
            two_success_not_team_count += 1
        if team_success_flag == 0 and n_agents_diag > 0 and safe_count >= n_agents_diag:
            all_safe_not_team_count += 1
        if agent_reach_flags:
            reached_agent_slot_count += reach_count
            safe_for_slots = agent_safe_flags if isinstance(agent_safe_flags, list) else []
            for idx, reach_flag in enumerate(agent_reach_flags):
                if _episode_flag(reach_flag) == 0:
                    continue
                safe_flag = _episode_flag(safe_for_slots[idx]) if idx < len(safe_for_slots) else 0
                if safe_flag == 0:
                    unsafe_reached_agent_slot_count += 1

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
            "agent_reach_rates": [_safe_mean(flags) for flags in agent_reach_lists],
            "agent_safe_rates": [_safe_mean(flags) for flags in agent_safe_lists],
            "done_reason_counts": dict(sorted(done_reason_counts.items())),
            "all_reached_without_safe_team_success_count": int(
                all_reached_without_safe_team_success_count
            ),
            "all_reached_without_safe_team_success_rate": (
                float(all_reached_without_safe_team_success_count / len(all_episodes_data))
                if all_episodes_data else None
            ),
            "two_success_not_team_count": int(two_success_not_team_count),
            "two_success_not_team_rate": (
                float(two_success_not_team_count / len(all_episodes_data))
                if all_episodes_data else None
            ),
            "all_safe_not_team_count": int(all_safe_not_team_count),
            "all_safe_not_team_rate": (
                float(all_safe_not_team_count / len(all_episodes_data))
                if all_episodes_data else None
            ),
            "unsafe_reached_agent_slot_count": int(unsafe_reached_agent_slot_count),
            "unsafe_reached_agent_slot_rate": (
                float(unsafe_reached_agent_slot_count / reached_agent_slot_count)
                if reached_agent_slot_count > 0 else None
            ),
            "avg_steps": _safe_mean(step_values),
            "std_steps": _safe_std(step_values),
            "avg_collision_count": _safe_mean(collision_counts),
            "std_collision_count": _safe_std(collision_counts),
            "collision_free_rate": (
                float(collision_free_count / len(all_episodes_data)) if all_episodes_data else None
            ),
            "avg_terrain_collision_count": _safe_mean(terrain_collision_counts),
            "std_terrain_collision_count": _safe_std(terrain_collision_counts),
            "avg_obstacle_collision_count": _safe_mean(obstacle_collision_counts),
            "std_obstacle_collision_count": _safe_std(obstacle_collision_counts),
            "avg_inter_agent_collision_count": _safe_mean(inter_agent_collision_counts),
            "std_inter_agent_collision_count": _safe_std(inter_agent_collision_counts),
            "inter_agent_collision_free_rate": (
                float(inter_agent_collision_free_count / len(all_episodes_data)) if all_episodes_data else None
            ),
            "avg_min_clearance_mean": _safe_mean(min_distance_means),
            "avg_min_clearance_min": _safe_mean(min_distance_mins),
            "avg_min_inter_agent_clearance": _safe_mean(min_inter_agent_clearances),
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


def _generate_evaluation_summary_plot(all_episodes_data, summary, save_path):
    if not all_episodes_data:
        return None

    episodes = [int(ep.get("episode", idx)) + 1 for idx, ep in enumerate(all_episodes_data)]
    rewards = [_finite_float_or_none(ep.get("reward")) for ep in all_episodes_data]
    team_success = [int(ep.get("team_success", ep.get("success", 0)) or 0) for ep in all_episodes_data]
    collision_counts = [_finite_float_or_none(ep.get("collision_count", 0)) for ep in all_episodes_data]
    terrain_collision_counts = [_finite_float_or_none(ep.get("terrain_collision_count", 0)) for ep in all_episodes_data]
    obstacle_collision_counts = [_finite_float_or_none(ep.get("obstacle_collision_count", 0)) for ep in all_episodes_data]
    inter_agent_collision_counts = [
        _finite_float_or_none(ep.get("inter_agent_collision_count", 0)) for ep in all_episodes_data
    ]
    final_goal_distances = [_finite_float_or_none(ep.get("final_goal_distance")) for ep in all_episodes_data]
    min_inter_agent_clearances = [
        _finite_float_or_none(ep.get("min_inter_agent_clearance")) for ep in all_episodes_data
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax_reward, ax_collision, ax_goal, ax_summary = axes.flatten()

    x = np.arange(len(episodes), dtype=np.int32)

    reward_values = [r if r is not None else np.nan for r in rewards]
    ax_reward.plot(episodes, reward_values, color="#1f77b4", linewidth=2.0, marker="o", markersize=4)
    success_idx = [idx for idx, flag in enumerate(team_success) if flag == 1]
    if success_idx:
        ax_reward.scatter(
            [episodes[idx] for idx in success_idx],
            [reward_values[idx] for idx in success_idx],
            color="#2ca02c",
            s=40,
            zorder=3,
            label="Team Success",
        )
        ax_reward.legend(loc="best")
    ax_reward.set_title("Reward Per Episode")
    ax_reward.set_xlabel("Episode")
    ax_reward.set_ylabel("Reward")
    ax_reward.grid(True, ls="--", alpha=0.3)

    collision_vals = [v if v is not None else 0.0 for v in collision_counts]
    terrain_vals = [v if v is not None else 0.0 for v in terrain_collision_counts]
    obstacle_vals = [v if v is not None else 0.0 for v in obstacle_collision_counts]
    inter_agent_vals = [v if v is not None else 0.0 for v in inter_agent_collision_counts]
    ax_collision.bar(
        x - 0.2,
        terrain_vals,
        width=0.38,
        color="#8c564b",
        alpha=0.85,
        label="Terrain Collision Count",
    )
    ax_collision.bar(
        x - 0.2,
        obstacle_vals,
        width=0.38,
        bottom=terrain_vals,
        color="#ff7f0e",
        alpha=0.85,
        label="Obstacle Collision Count",
    )
    ax_collision.bar(
        x + 0.2,
        inter_agent_vals,
        width=0.4,
        color="#d62728",
        alpha=0.85,
        label="Inter-Agent Collision Count",
    )
    ax_collision.set_xticks(x)
    ax_collision.set_xticklabels(episodes)
    ax_collision.set_title("Collision Statistics")
    ax_collision.set_xlabel("Episode")
    ax_collision.set_ylabel("Count")
    ax_collision.grid(True, axis="y", ls="--", alpha=0.3)
    ax_collision.legend(loc="best")

    goal_vals = [v if v is not None else np.nan for v in final_goal_distances]
    clearance_vals = [v if v is not None else np.nan for v in min_inter_agent_clearances]
    ax_goal.plot(episodes, goal_vals, color="#9467bd", linewidth=2.0, marker="o", markersize=4, label="Team Final Goal Distance")
    ax_goal.set_title("Goal Distance / Inter-Agent Clearance")
    ax_goal.set_xlabel("Episode")
    ax_goal.set_ylabel("Goal Distance")
    ax_goal.grid(True, ls="--", alpha=0.3)
    ax_goal_2 = ax_goal.twinx()
    ax_goal_2.plot(
        episodes,
        clearance_vals,
        color="#2ca02c",
        linewidth=2.0,
        marker="s",
        markersize=4,
        label="Min Inter-Agent Clearance",
    )
    ax_goal_2.set_ylabel("Min Inter-Agent Clearance")
    lines_1, labels_1 = ax_goal.get_legend_handles_labels()
    lines_2, labels_2 = ax_goal_2.get_legend_handles_labels()
    ax_goal.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    ax_summary.axis("off")
    summary_lines = [
        "Official Evaluation Summary",
        f"Episodes: {summary.get('episodes')}",
        f"Team success rate: {summary.get('team_success_rate'):.3f}" if summary.get("team_success_rate") is not None else "Team success rate: N/A",
        f"Avg reward: {summary.get('avg_reward'):.2f}" if summary.get("avg_reward") is not None else "Avg reward: N/A",
        f"Avg collision count: {summary.get('avg_collision_count'):.2f}" if summary.get("avg_collision_count") is not None else "Avg collision count: N/A",
        f"Avg terrain collision count: {summary.get('avg_terrain_collision_count'):.2f}" if summary.get("avg_terrain_collision_count") is not None else "Avg terrain collision count: N/A",
        f"Avg obstacle collision count: {summary.get('avg_obstacle_collision_count'):.2f}" if summary.get("avg_obstacle_collision_count") is not None else "Avg obstacle collision count: N/A",
        f"Avg inter-agent collision count: {summary.get('avg_inter_agent_collision_count'):.2f}" if summary.get("avg_inter_agent_collision_count") is not None else "Avg inter-agent collision count: N/A",
        f"Inter-agent collision-free rate: {summary.get('inter_agent_collision_free_rate'):.3f}" if summary.get("inter_agent_collision_free_rate") is not None else "Inter-agent collision-free rate: N/A",
        f"Avg team final goal distance: {summary.get('avg_team_final_goal_distance'):.2f}" if summary.get("avg_team_final_goal_distance") is not None else "Avg team final goal distance: N/A",
        f"Avg min inter-agent clearance: {summary.get('avg_min_inter_agent_clearance'):.2f}" if summary.get("avg_min_inter_agent_clearance") is not None else "Avg min inter-agent clearance: N/A",
    ]
    ax_summary.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f7f7", edgecolor="#cccccc"),
    )

    fig.suptitle("Official Evaluation Statistics", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return save_path

class ModelEvaluator:
    """模型评估器，仿照1.0版本的评估逻辑"""
    
    def __init__(self, args):
        self.args = args
        self._current_episode_terrain_info = {}
        self._eval_noise_scale = max(0.0, float(getattr(args, 'eval_noise_scale', 0.0) or 0.0))
        self._eval_random_action_prob = min(
            1.0,
            max(0.0, float(getattr(args, 'eval_random_action_prob', 0.0) or 0.0)),
        )
        eval_noise_seed = getattr(args, 'eval_noise_seed', None)
        self._eval_noise_seed = int(eval_noise_seed) if eval_noise_seed is not None else None
        self.training_alignment = _load_training_alignment_snapshot(getattr(args, 'load_model_path', None))
        if self.training_alignment:
            _apply_training_alignment_to_args(self.args, self.training_alignment)
            _apply_runtime_env_overrides_from_args(self.args)
        if self._eval_noise_scale > 0.0 or self._eval_random_action_prob > 0.0:
            print(
                "[EvalNoise] "
                f"type={_EVAL_NOISE_TYPE}, stream={_EVAL_NOISE_STREAM_MODE}, "
                f"noise_scale={self._eval_noise_scale:.6g}, "
                f"random_action_prob={self._eval_random_action_prob:.6g}, "
                f"seed={self._eval_noise_seed}"
            )
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
        ld_parts = [p for p in os.getenv('LD_LIBRARY_PATH', '').split(':') if p]
        conda_prefix = os.getenv('CONDA_PREFIX', '')
        conda_lib = str(Path(conda_prefix) / 'lib') if conda_prefix else ''
        cuda_ld_entries = [
            p for p in ld_parts
            if (
                p == '/usr/lib/wsl/lib'
                or (conda_lib and p == conda_lib)
                or '/site-packages/nvidia/' in p
            )
        ]
        self._eval_device_info = {
            'python': sys.executable,
            'cuda_visible_devices': cuda_visible,
            'physical_gpus': len(physical_gpus),
            'logical_gpus': len(logical_gpus),
            'physical_gpu_names': [getattr(gpu, 'name', str(gpu)) for gpu in physical_gpus],
            'logical_gpu_names': [getattr(gpu, 'name', str(gpu)) for gpu in logical_gpus],
            'configure_gpu': 'ok' if gpu_configured else 'fallback_cpu',
            'require_gpu': _env_flag('MATD3_REQUIRE_GPU', False),
            'gpu_ld_bootstrapped': os.getenv('MATD3_EVAL_GPU_LD_BOOTSTRAPPED') == '1',
            'ld_library_path_has_wsl_cuda': '/usr/lib/wsl/lib' in ld_parts,
            'ld_library_path_has_conda_lib': bool(conda_lib and conda_lib in ld_parts),
            'ld_library_path_cuda_entries': cuda_ld_entries,
        }
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
            if _env_flag('MATD3_REQUIRE_GPU', False):
                raise RuntimeError(
                    "MATD3_REQUIRE_GPU=1，但 TensorFlow 未检测到 GPU；已拒绝继续用 CPU 跑评估。"
                )
        
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

        # 应用智能体尺寸/速度/加速度（若提供）
        try:
            if (
                getattr(self.args, 'agent_size', None) is not None
                or getattr(self.args, 'agent_max_speed', None) is not None
                or getattr(self.args, 'agent_accel', None) is not None
            ):
                for ag in getattr(self.world, 'agents', []):
                    if getattr(self.args, 'agent_size', None) is not None and hasattr(ag, 'size'):
                        ag.size = float(self.args.agent_size)
                    if getattr(self.args, 'agent_max_speed', None) is not None and hasattr(ag, 'max_speed'):
                        ag.max_speed = float(self.args.agent_max_speed)
                    if getattr(self.args, 'agent_accel', None) is not None and hasattr(ag, 'accel'):
                        ag.accel = float(self.args.agent_accel)
        except Exception as _e:
            print(f"评估环境应用尺寸/速度/加速度失败: {_e}")

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

    def _rebuild_environment(self, *, preserve_episode_terrain=False):
        """在场景被重新生成后重建world/env，并重新应用关键运行时参数。"""
        _apply_runtime_env_overrides_from_args(self.args)
        _apply_terrain_runtime_params_to_scenario(
            self.scenario,
            None,
            self.args,
            preserve_episode_terrain=preserve_episode_terrain,
        )
        self.world = self.scenario.make_world()
        _apply_terrain_runtime_params_to_scenario(
            self.scenario,
            self.world,
            self.args,
            preserve_episode_terrain=preserve_episode_terrain,
        )

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
            if (
                getattr(self.args, 'agent_size', None) is not None
                or getattr(self.args, 'agent_max_speed', None) is not None
                or getattr(self.args, 'agent_accel', None) is not None
            ):
                for ag in getattr(self.world, 'agents', []):
                    if getattr(self.args, 'agent_size', None) is not None and hasattr(ag, 'size'):
                        ag.size = float(self.args.agent_size)
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
            terrain_variant_seed = _scenario_terrain_variant_seed(self.scenario)
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

        self._rebuild_environment(preserve_episode_terrain=True)
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
            'terrain_variant_seed': _scenario_terrain_variant_seed(
                self.scenario,
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

    def _capture_episode_vis_context_for_context(self, eval_ctx, episode_idx):
        """在不永久切换 live evaluator 状态的前提下捕获某个 batch 子环境的可视化上下文。"""
        old_scenario = getattr(self, 'scenario', None)
        old_world = getattr(self, 'world', None)
        old_env = getattr(self, 'env', None)
        try:
            self.scenario = eval_ctx.scenario
            self.world = eval_ctx.world
            self.env = eval_ctx.env
            return self._capture_episode_vis_context(episode_idx)
        finally:
            self.scenario = old_scenario
            self.world = old_world
            self.env = old_env

    def _load_episode_positions_into_scenario(self, scenario, episode_idx, terrain_seed=None, terrain_variant_seed=None):
        """为给定 scenario 加载 official episode 位置文件，避免 batch 子环境共享 self.scenario。"""
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
                scenario.fixed_positions = None
                scenario.use_fixed_positions = False
                scenario.positions_initialized = False
                print(f"⚠️  Episode {episode_idx + 1} 位置文件不存在，将使用动态生成")
                return

            with open(positions_file, 'r', encoding='utf-8') as f:
                positions_data = json.load(f)

            if 'agents' in positions_data and 'goal' in positions_data:
                scenario.fixed_positions = {
                    'agents': positions_data['agents'],
                    'goal': positions_data['goal']
                }
                scenario.use_fixed_positions = True
                scenario.positions_initialized = True
                if hasattr(scenario, 'validate_and_adjust_fixed_positions'):
                    scenario.validate_and_adjust_fixed_positions()
            else:
                scenario.fixed_positions = None
                scenario.use_fixed_positions = False
                scenario.positions_initialized = False
                print(f"⚠️  Episode {episode_idx + 1}位置文件格式错误，将使用动态生成")
        except Exception as e:
            if require_episode_positions:
                raise
            scenario.fixed_positions = None
            scenario.use_fixed_positions = False
            scenario.positions_initialized = False
            print(f"⚠️  加载Episode {episode_idx + 1}位置文件失败: {e}，将使用动态生成")

    def _apply_eval_context_runtime_params(
        self,
        scenario,
        world,
        env=None,
        *,
        preserve_episode_terrain=False,
    ):
        """把评估运行时参数应用到 batch 子环境，保持与单环境评估一致。"""
        _apply_terrain_runtime_params_to_scenario(
            scenario,
            world,
            self.args,
            preserve_episode_terrain=preserve_episode_terrain,
        )

        try:
            if hasattr(world, 'gravity') and getattr(self.args, 'gravity', None) is not None:
                world.gravity = float(self.args.gravity)
            if hasattr(world, 'control_accel_gain') and getattr(self.args, 'control_accel_gain', None) is not None:
                world.control_accel_gain = float(self.args.control_accel_gain)
            if hasattr(world, 'reward_pos_scale') and getattr(self.args, 'reward_pos_scale', None) is not None:
                world.reward_pos_scale = float(self.args.reward_pos_scale)
            if hasattr(world, 'reward_neg_scale') and getattr(self.args, 'reward_neg_scale', None) is not None:
                world.reward_neg_scale = float(self.args.reward_neg_scale)
            if hasattr(world, 'damping') and getattr(self.args, 'damping', None) is not None:
                world.damping = float(self.args.damping)
        except Exception:
            pass

        try:
            quiet_output = os.getenv("QUIET_OUTPUT", "1").lower() in ("1", "true", "yes", "on")
        except Exception:
            quiet_output = True
        try:
            _apply_hidden_runtime_params_to_world(world, self.args, quiet_output=quiet_output)
        except Exception:
            pass

        try:
            if (
                getattr(self.args, 'agent_size', None) is not None
                or getattr(self.args, 'agent_max_speed', None) is not None
                or getattr(self.args, 'agent_accel', None) is not None
            ):
                for ag in getattr(world, 'agents', []):
                    if getattr(self.args, 'agent_size', None) is not None and hasattr(ag, 'size'):
                        ag.size = float(self.args.agent_size)
                    if getattr(self.args, 'agent_max_speed', None) is not None and hasattr(ag, 'max_speed'):
                        ag.max_speed = float(self.args.agent_max_speed)
                    if getattr(self.args, 'agent_accel', None) is not None and hasattr(ag, 'accel'):
                        ag.accel = float(self.args.agent_accel)
        except Exception:
            pass

        try:
            try_apply_scenario_params(scenario, world, self.args, tqdm_file=None)
        except Exception:
            pass

        try:
            if hasattr(self.args, 'collision_distance_threshold') and self.args.collision_distance_threshold is not None:
                if hasattr(scenario, 'collision_distance_threshold'):
                    scenario.collision_distance_threshold = float(self.args.collision_distance_threshold)
            if hasattr(self.args, 'collision_penalty_value') and self.args.collision_penalty_value is not None:
                if hasattr(scenario, 'collision_penalty_value'):
                    scenario.collision_penalty_value = float(self.args.collision_penalty_value)
        except Exception:
            pass

        try:
            ax = getattr(self.args, 'action_range_x', None)
            ay = getattr(self.args, 'action_range_y', None)
            az = getattr(self.args, 'action_range_z', None)
            if any(v is not None for v in (ax, ay, az)):
                current = getattr(world, 'action_range', None)
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
                world.action_range = new_range
        except Exception:
            pass

        try:
            world.episode_length = int(getattr(self.args, 'episode_length', 2200) or 2200)
            world.current_step = 0
            world.scenario = scenario
        except Exception:
            pass
        if env is not None:
            try:
                env.scenario = scenario
            except Exception:
                pass

    def _build_episode_eval_context(
        self,
        episode_idx,
        terrain_level,
        terrain_seed_sequence=None,
        terrain_variant_seed_sequence=None,
        obstacle_seed_sequence=None,
    ):
        """为 batch episode 创建独立 scenario/world/env，不污染 self.scenario。"""
        scenario = load_scenario_module(self.args.scenario_name, self.args)
        if scenario is None:
            raise RuntimeError(f"无法加载场景: {self.args.scenario_name}")
        _apply_runtime_env_overrides_from_args(self.args)
        _apply_terrain_runtime_params_to_scenario(scenario, None, self.args)
        terrain_level = _set_scenario_terrain_complexity(scenario, terrain_level)

        use_random_terrain = (
            bool(getattr(self.args, 'random_terrain', False))
            or _sequence_implies_random_terrain(terrain_seed_sequence, terrain_variant_seed_sequence)
        )
        obstacle_seed = None
        if obstacle_seed_sequence and episode_idx < len(obstacle_seed_sequence):
            obstacle_seed = int(obstacle_seed_sequence[episode_idx])

        terrain_seed = getattr(scenario, 'current_terrain_seed', getattr(scenario, 'seed', None))
        terrain_variant_seed = _scenario_terrain_variant_seed(scenario)

        if use_random_terrain:
            if terrain_seed_sequence and episode_idx < len(terrain_seed_sequence):
                terrain_seed = int(terrain_seed_sequence[episode_idx])
            else:
                terrain_seed = int(np.random.randint(0, 1000000))

            terrain_variant_seed = None
            if terrain_variant_seed_sequence and episode_idx < len(terrain_variant_seed_sequence):
                terrain_variant_seed = int(terrain_variant_seed_sequence[episode_idx])

            if hasattr(scenario, 'regenerate_terrain'):
                if terrain_variant_seed is not None:
                    scenario.regenerate_terrain(new_seed=terrain_seed, variant_seed=terrain_variant_seed)
                else:
                    scenario.regenerate_terrain(new_seed=terrain_seed)
            else:
                try:
                    scenario.seed = terrain_seed
                    if hasattr(scenario, 'rng'):
                        scenario.rng = np.random.RandomState(terrain_seed)
                except Exception:
                    pass
        else:
            try:
                scenario.random_terrain = False
            except Exception:
                pass

        world = scenario.make_world()
        env = MultiAgentEnv(
            world,
            reset_callback=scenario.reset_world,
            reward_callback=scenario.reward,
            observation_callback=scenario.observation,
            done_callback=getattr(scenario, 'is_done', None),
            info_callback=None,
            shared_viewer=False
        )
        self._apply_eval_context_runtime_params(
            scenario,
            world,
            env,
            preserve_episode_terrain=use_random_terrain,
        )

        if use_random_terrain:
            self._load_episode_positions_into_scenario(
                scenario,
                episode_idx,
                terrain_seed=terrain_seed,
                terrain_variant_seed=terrain_variant_seed,
            )

        try:
            scenario.current_episode_index = int(episode_idx)
            scenario.current_episode_env_id = 0
            scenario.current_episode_obstacle_seed_override = (
                int(obstacle_seed) if obstacle_seed is not None else None
            )
            env.scenario.current_episode_index = int(episode_idx)
            env.scenario.current_episode_env_id = 0
            env.scenario.current_episode_obstacle_seed_override = (
                int(obstacle_seed) if obstacle_seed is not None else None
            )
        except Exception:
            pass

        if use_random_terrain:
            try:
                scenario.random_terrain = False
                env.scenario.random_terrain = False
            except Exception:
                pass
        else:
            try:
                world._episode_index_counter = int(episode_idx)
                world.episode_index = int(episode_idx)
            except Exception:
                pass

        terrain_info = {
            'terrain_seed': getattr(scenario, 'current_terrain_seed', terrain_seed),
            'terrain_variant_seed': _scenario_terrain_variant_seed(scenario, terrain_variant_seed),
            'obstacle_seed': int(obstacle_seed) if obstacle_seed is not None else None,
        }
        return argparse.Namespace(
            episode_idx=int(episode_idx),
            terrain_level=terrain_level,
            terrain_info=terrain_info,
            scenario=scenario,
            world=world,
            env=env,
        )
    
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

    def _make_eval_noise_streams(self, episode_idx):
        """Create independent, topology-invariant streams for one episode.

        Gaussian perturbations, random-action masks, and replacement actions use
        separate PCG64 streams.  Consequently the Gaussian sequence is identical
        between Gaussian-only and combined modes, and one episode ending early
        cannot advance another episode's random state.
        """
        eval_noise_seed = getattr(self, '_eval_noise_seed', None)
        eval_noise_scale = float(getattr(self, '_eval_noise_scale', 0.0) or 0.0)
        eval_random_action_prob = float(
            getattr(self, '_eval_random_action_prob', 0.0) or 0.0
        )
        if eval_noise_seed is None:
            if eval_noise_scale > 0.0 or eval_random_action_prob > 0.0:
                raise RuntimeError("启用评估动作扰动时必须先解析 eval_noise_seed")
            return None
        base_seed = int(eval_noise_seed)
        episode_seed = int(episode_idx)

        def _stream(stream_id):
            seed_sequence = np.random.SeedSequence(
                [
                    base_seed & 0xFFFFFFFF,
                    (base_seed >> 32) & 0xFFFFFFFF,
                    episode_seed & 0xFFFFFFFF,
                    (episode_seed >> 32) & 0xFFFFFFFF,
                    int(stream_id),
                ]
            )
            return np.random.default_rng(seed_sequence)

        return {
            "gaussian": _stream(1),
            "random_mask": _stream(2),
            "random_action": _stream(3),
        }

    def _apply_eval_action_noise(self, actions, episode_streams):
        """Apply recorded Gaussian/random-action perturbations to raw actions."""
        eval_noise_scale = float(getattr(self, '_eval_noise_scale', 0.0) or 0.0)
        eval_random_action_prob = float(
            getattr(self, '_eval_random_action_prob', 0.0) or 0.0
        )
        if eval_noise_scale <= 0.0 and eval_random_action_prob <= 0.0:
            return actions
        if episode_streams is None:
            raise RuntimeError("评估动作扰动缺少 episode 独立随机流")

        actions_tensor = tf.convert_to_tensor(actions)
        squeeze_output = len(actions_tensor.shape) == 2
        batched_actions = tf.expand_dims(actions_tensor, axis=0) if squeeze_output else actions_tensor
        if len(batched_actions.shape) != 3:
            raise ValueError(f"评估动作必须是2维或3维，实际shape={batched_actions.shape}")

        streams = [episode_streams] if isinstance(episode_streams, dict) else list(episode_streams)
        batch_size = batched_actions.shape[0]
        agent_count = batched_actions.shape[1]
        action_dim = batched_actions.shape[2]
        if batch_size is None or agent_count is None or action_dim is None:
            dynamic_shape = tf.shape(batched_actions).numpy().tolist()
            batch_size, agent_count, action_dim = map(int, dynamic_shape)
        else:
            batch_size = int(batch_size)
            agent_count = int(agent_count)
            action_dim = int(action_dim)
        if len(streams) != batch_size:
            raise ValueError(
                f"episode随机流数量与动作batch不一致: streams={len(streams)}, batch={batch_size}"
            )

        result = batched_actions
        if eval_noise_scale > 0.0:
            gaussian_noise = np.stack(
                [
                    stream["gaussian"].normal(
                        loc=0.0,
                        scale=eval_noise_scale,
                        size=(agent_count, min(3, action_dim)),
                    )
                    for stream in streams
                ],
                axis=0,
            ).astype(result.dtype.as_numpy_dtype, copy=False)
            noisy_head = tf.clip_by_value(
                result[:, :, :3] + tf.convert_to_tensor(gaussian_noise, dtype=result.dtype),
                tf.cast(-1.0, result.dtype),
                tf.cast(1.0, result.dtype),
            )
            result = tf.concat([noisy_head, result[:, :, 3:]], axis=2)

        if eval_random_action_prob > 0.0:
            random_mask = np.stack(
                [
                    stream["random_mask"].random(agent_count) < eval_random_action_prob
                    for stream in streams
                ],
                axis=0,
            )
            random_actions = np.stack(
                [
                    stream["random_action"].uniform(-1.0, 1.0, size=(agent_count, action_dim))
                    for stream in streams
                ],
                axis=0,
            ).astype(result.dtype.as_numpy_dtype, copy=False)
            result = tf.where(
                tf.expand_dims(tf.convert_to_tensor(random_mask, dtype=tf.bool), axis=2),
                tf.convert_to_tensor(random_actions, dtype=result.dtype),
                result,
            )

        return result[0] if squeeze_output else result

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
        if processed_obs.ndim not in (2, 3):
            return processed_obs

        base_obs_dim = self._get_base_obs_dim()
        had_batch_dim = processed_obs.ndim == 3
        if had_batch_dim:
            base_obs = (
                processed_obs[:, :, :base_obs_dim]
                if processed_obs.shape[2] > base_obs_dim
                else processed_obs
            )
        else:
            base_obs = processed_obs[:, :base_obs_dim] if processed_obs.shape[1] > base_obs_dim else processed_obs

        if not use_pf:
            return base_obs

        pf_feature_dim = self._get_pf_feature_dim()
        if had_batch_dim:
            pf_features = np.zeros((base_obs.shape[0], base_obs.shape[1], pf_feature_dim), dtype=np.float32)
        else:
            pf_features = np.zeros((base_obs.shape[0], pf_feature_dim), dtype=np.float32)

        if use_tf_potential_field and action_force_ratio > 0.0:
            try:
                pf_forces = None

                # 与训练主线 batch_select_actions_vectorized 保持一致：
                # pf_input 应来自当前状态的 base PF 特征，而不是临时拼接出的其他语义。
                if hasattr(self.maddpg, 'compute_base_pf_forces_batch_numpy'):
                    pf_input_batch = base_obs if had_batch_dim else np.expand_dims(base_obs, axis=0)
                    pf_force_batch = self.maddpg.compute_base_pf_forces_batch_numpy(
                        pf_input_batch,
                        float(action_force_ratio),
                    )
                    if isinstance(pf_force_batch, np.ndarray) and pf_force_batch.ndim == 3:
                        pf_forces = np.asarray(pf_force_batch if had_batch_dim else pf_force_batch[0], dtype=np.float32)

                # 兼容旧模型/旧实现：若缺少统一 helper，则回退到 dummy_action 路径。
                if pf_forces is None:
                    flat_base_obs = base_obs.reshape((-1, base_obs.shape[-1])) if had_batch_dim else base_obs
                    dummy_actions_tf = tf.zeros((flat_base_obs.shape[0], 7), dtype=tf.float32)
                    base_obs_tf = tf.convert_to_tensor(flat_base_obs, dtype=tf.float32)
                    _, pf_forces_tf = self.maddpg._apply_potential_field_correction(
                        dummy_actions_tf, base_obs_tf, action_force_ratio
                    )
                    pf_forces = np.asarray(pf_forces_tf.numpy(), dtype=np.float32)
                    if had_batch_dim:
                        pf_forces = pf_forces.reshape((base_obs.shape[0], base_obs.shape[1], -1))

                if had_batch_dim:
                    if pf_forces.ndim == 3 and pf_forces.shape[:2] == base_obs.shape[:2]:
                        copy_dim = min(pf_feature_dim, pf_forces.shape[2])
                        pf_features[:, :, :copy_dim] = pf_forces[:, :, :copy_dim]
                else:
                    if pf_forces.ndim == 2 and pf_forces.shape[0] == base_obs.shape[0]:
                        copy_dim = min(pf_feature_dim, pf_forces.shape[1])
                        pf_features[:, :copy_dim] = pf_forces[:, :copy_dim]
            except Exception:
                pass

        return np.concatenate([base_obs, pf_features], axis=2 if had_batch_dim else 1)

    def _build_eval_actor_obs_tf_with_context(self, processed_obs, use_pf, use_tf_potential_field, action_force_ratio):
        """
        TensorFlow 版评估 Actor 输入构造。
        返回 (policy_obs_tf, base_obs_tf, pf_shared_context)，其中 pf_shared_context 可被
        后续 PF correction 复用，避免同一步重复解析 observation geometry。
        """
        processed_np = np.asarray(processed_obs, dtype=np.float32)
        if processed_np.ndim != 3:
            policy_obs_np = self._build_eval_actor_obs(
                processed_np,
                use_pf=use_pf,
                use_tf_potential_field=use_tf_potential_field,
                action_force_ratio=action_force_ratio,
            )
            base_obs_dim = self._get_base_obs_dim()
            base_obs_np = processed_np[:, :base_obs_dim] if processed_np.ndim == 2 and processed_np.shape[1] > base_obs_dim else processed_np
            return tf.convert_to_tensor(policy_obs_np, dtype=tf.float32), tf.convert_to_tensor(base_obs_np, dtype=tf.float32), None

        base_obs_dim = self._get_base_obs_dim()
        base_obs_np = processed_np[:, :, :base_obs_dim] if processed_np.shape[2] > base_obs_dim else processed_np
        base_obs_tf = tf.convert_to_tensor(base_obs_np, dtype=tf.float32)
        if not use_pf:
            return base_obs_tf, base_obs_tf, None

        pf_feature_dim = self._get_pf_feature_dim()
        pf_features_tf = tf.zeros(
            [tf.shape(base_obs_tf)[0], tf.shape(base_obs_tf)[1], pf_feature_dim],
            dtype=tf.float32,
        )
        pf_shared_context = None

        if use_tf_potential_field and action_force_ratio > 0.0:
            try:
                obs_shapes = [int(v) for v in getattr(self, 'obs_shapes', []) if v is not None]
                action_dims = [int(v) for v in getattr(self, 'action_dims', []) if v is not None]
                uniform_obs_dims = bool(obs_shapes) and len(set(obs_shapes)) <= 1
                uniform_action_dims = bool(action_dims) and len(set(action_dims)) <= 1
                has_shared_helpers = all(
                    hasattr(self.maddpg, name)
                    for name in (
                        '_extract_pf_obs_context_compiled_tf',
                        '_extract_pf_geometry_context_compiled_tf',
                        '_compute_base_pf_forces_from_shared_context_tf',
                    )
                )
                if uniform_obs_dims and uniform_action_dims and has_shared_helpers:
                    correct_obs_dim = obs_shapes[0]
                    act_dim = action_dims[0]
                    flat_base_obs = tf.ensure_shape(
                        tf.reshape(base_obs_tf[:, :, :correct_obs_dim], [-1, correct_obs_dim]),
                        [None, correct_obs_dim],
                    )
                    obs_ctx = self.maddpg._extract_pf_obs_context_compiled_tf(flat_base_obs)
                    geometry_ctx = self.maddpg._extract_pf_geometry_context_compiled_tf(
                        obs_ctx[1],
                        obs_ctx[2],
                        obs_ctx[4],
                        obs_ctx[5],
                        obs_ctx[6],
                    )
                    pf_tensor = self.maddpg._compute_base_pf_forces_from_shared_context_tf(
                        obs_ctx[3],
                        geometry_ctx,
                        float(action_force_ratio),
                        tf.shape(base_obs_tf)[0],
                        act_dim,
                    )
                    copy_dim = min(int(pf_feature_dim), 3)
                    if copy_dim > 0:
                        pf_features_tf = tf.concat(
                            [
                                pf_tensor[:, :, :copy_dim],
                                tf.zeros(
                                    [tf.shape(base_obs_tf)[0], tf.shape(base_obs_tf)[1], pf_feature_dim - copy_dim],
                                    dtype=tf.float32,
                                ),
                            ],
                            axis=2,
                        ) if copy_dim < pf_feature_dim else pf_tensor[:, :, :pf_feature_dim]
                    pf_shared_context = (obs_ctx[3], geometry_ctx)
                else:
                    raise RuntimeError("shared PF helpers unavailable for current shapes")
            except Exception:
                policy_obs_np = self._build_eval_actor_obs(
                    processed_np,
                    use_pf=use_pf,
                    use_tf_potential_field=use_tf_potential_field,
                    action_force_ratio=action_force_ratio,
                )
                return tf.convert_to_tensor(policy_obs_np, dtype=tf.float32), base_obs_tf, None

        return tf.concat([base_obs_tf, pf_features_tf], axis=2), base_obs_tf, pf_shared_context

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

        def _coerce_float_list(values):
            if not isinstance(values, list):
                return None
            coerced = []
            try:
                for item in values:
                    coerced.append(float(item))
            except Exception:
                return None
            return coerced if coerced else None

        def _load_checkpoint_episode_force_ratios():
            candidates = [
                Path(model_path) / "checkpoint_state.json",
                Path(model_base_dir) / "checkpoint_state.json",
                Path(model_base_dir) / "checkpoint" / "checkpoint_state.json",
                Path(model_base_dir) / "best_by_team_sr" / "checkpoint_state.json",
            ]
            seen = set()
            for candidate in candidates:
                try:
                    resolved = candidate.resolve()
                except Exception:
                    resolved = candidate
                if resolved in seen or not candidate.exists():
                    continue
                seen.add(resolved)
                try:
                    with open(candidate, 'r', encoding='utf-8') as f:
                        state = json.load(f)
                except Exception:
                    continue
                ratios = _coerce_float_list(state.get('episode_force_ratios') if isinstance(state, dict) else None)
                if ratios is not None:
                    print(f"✅ 从检查点状态读取逐回合FR序列: {candidate} (n={len(ratios)})")
                    return ratios
            return None

        checkpoint_episode_force_ratios = _load_checkpoint_episode_force_ratios()
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
                    
                    results_episode_force_ratios = _coerce_float_list(results.get('episode_force_ratios'))
                    if results_episode_force_ratios is not None:
                        episode_force_ratios = results_episode_force_ratios
                    elif episode_force_ratios is None and checkpoint_episode_force_ratios is not None:
                        episode_force_ratios = checkpoint_episode_force_ratios

                    # 关键：FR 必须与实际评估的权重变体一致。
                    # - final -> 最后回合 FR
                    # - best_by_team_sr -> Team SR 最佳回合 FR
                    # - best_by_strict_success -> 严格成功最佳回合 FR
                    # - best -> reward 最佳回合 FR
                    # - epXXX/latest_ep -> 对应回合 FR；找不到逐回合 FR 时拒绝评估，不能回退到最终回合 FR
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
                        else:
                            print(
                                f"⚠️  未找到 {model_leaf_dir} 对应逐回合FR，"
                                f"拒绝回退到最终回合FR以避免评估错配"
                            )
                    
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
                            training_use_quadrotor_dynamics = _coerce_optional_bool(results['args']['use_quadrotor_dynamics'])
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
                            training_use_quadrotor_dynamics = _coerce_optional_bool(results['use_quadrotor_dynamics'])
                    
                    if training_use_fr is not None and training_use_pf is not None:
                        print(f"✅ 从训练配置读取特征标志: use_fr_feature={training_use_fr}, use_pf_feature={training_use_pf}")
                        if training_pf_feature_dim is not None:
                            print(f"✅ 从训练配置读取PF特征维度: {training_pf_feature_dim}")
                        # 🔧 关键修复：打印动作范围参数
                        if training_action_range_x is not None or training_action_range_y is not None or training_action_range_z is not None:
                            print(f"✅ 从训练配置读取动作范围: X={training_action_range_x}, Y={training_action_range_y}, Z={training_action_range_z}")
                        if selected_force_ratio is not None:
                            break
                except Exception as e:
                    continue
            
            if training_use_fr is not None and training_use_pf is not None and selected_force_ratio is not None:
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
            self.args.use_quadrotor_dynamics = training_use_quadrotor_dynamics
            if (
                runtime_use_quadrotor_dynamics is not None
                and runtime_use_quadrotor_dynamics != training_use_quadrotor_dynamics
            ):
                print(
                    "⚠️ USE_QUADROTOR_DYNAMICS 与训练配置记录不一致，"
                    f"严格评估模式下已覆盖运行时值 {runtime_use_quadrotor_dynamics} "
                    f"-> 训练记录值 {training_use_quadrotor_dynamics}"
                )
            else:
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
            eval_actor_only=_env_flag('EVAL_ACTOR_ONLY', True),
            goal_attraction=getattr(self.args, 'goal_attraction', 26.0),
            lambda_1_base=getattr(self.args, 'lambda_1_base', 8.5),
            terrain_repulsion=getattr(self.args, 'terrain_repulsion', 1600.0),
            agent_influence_range=getattr(self.args, 'agent_influence_range', 150.0),
            delta_k_att=getattr(self.args, 'delta_k_att', 5.0),
            delta_lambda_1=getattr(self.args, 'delta_lambda_1', 2.2),
            delta_k_rep=getattr(self.args, 'delta_k_rep', 600.0),
            delta_radius=getattr(self.args, 'delta_radius', 80.0),
            max_force_magnitude=getattr(self.args, 'max_force_magnitude', 80.0),
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
        eval_actor_only = bool(getattr(maddpg_args, 'eval_actor_only', False)) and algorithm in ('matd3', 'maddpg')
        if eval_actor_only:
            print("⚡ EVAL_ACTOR_ONLY=1：评估仅构建/加载Actor，跳过Critic、Target网络和优化器。")
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
                    if eval_actor_only:
                        continue
                    # 🚨 标准MATD3：检查两个独立的Critic网络文件
                    if algorithm == 'matd3':
                        cp1 = os.path.join(dir_path, f"critic1_{i}.weights.h5")
                        cp2 = os.path.join(dir_path, f"critic2_{i}.weights.h5")
                        cp_old = os.path.join(dir_path, f"critic_{i}.weights.h5")
                        has_twin_critics = (
                            os.path.isfile(cp1) and os.path.getsize(cp1) > 0
                            and os.path.isfile(cp2) and os.path.getsize(cp2) > 0
                        )
                        has_legacy_critic = os.path.isfile(cp_old) and os.path.getsize(cp_old) > 0
                        if not has_twin_critics and not has_legacy_critic:
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
            # 只在显式传入“实验根目录”时解析其子目录。若调用方已经指定
            # final/best/epXXX 等具体 checkpoint，则不允许静默改用兄弟目录，
            # 否则结果中的模型身份会与请求不一致。
            if _is_model_variant_dir_name(preferred_dir):
                return None
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

        # 先以虚拟输入构建网络，确保变量已创建。评估轻量模式只构建 Actor。
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
                
                if eval_actor_only:
                    continue

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
            raise RuntimeError(f"评估网络构建失败，拒绝继续加载或运行: {e}") from e

        # 实验评估必须使用拓扑完全匹配的权重。旧实现的手动逐层加载和
        # skip_mismatch 会在只加载部分变量时仍返回成功，进而把随机初始化参数
        # 混入正式结果。标准 load_weights 失败即判定该 checkpoint 不兼容。
        def _safe_load(agent, path: str, kind: str):
            try:
                agent.load_weights(path)
                return True
            except Exception as e:
                print(f"❌ 严格加载{kind}失败: {path} | {e}")
                return False

        print(f"正在从 {model_dir} 加载模型...")
        ok = True
        total_loaded_vars = 0
        total_vars = 0
        verify_weight_coverage = _env_flag('EVAL_VERIFY_WEIGHT_COVERAGE', False)
        if not verify_weight_coverage:
            print("⚡ EVAL_VERIFY_WEIGHT_COVERAGE=0：跳过加载前后权重全量拷贝检查，降低并发内存峰值。")

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
            if verify_weight_coverage:
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
            else:
                ok = _safe_load(self.maddpg.agents[i]['actor'], a_path, f"actor[{i}]") and ok
                tot = len(getattr(self.maddpg.agents[i]['actor'], 'trainable_variables', []) or [])
                total_loaded_vars += tot
                total_vars += tot
                print(f"actor[{i}] 已加载: {os.path.basename(a_path)} ({tot} trainable vars)")

            if eval_actor_only:
                continue

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

    def _init_batched_episode_state(self, eval_ctx):
        """初始化一个 batch 子回合的统计状态。"""
        episode_idx = int(eval_ctx.episode_idx)
        env = eval_ctx.env
        world = eval_ctx.world
        scenario = eval_ctx.scenario

        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs_n, _ = reset_result
        else:
            obs_n = reset_result

        # reset_world 才会根据本回合上下文最终生成动态障碍。这里必须在
        # reset 之后回填实际种子，否则 batch 路径会把已使用的障碍种子
        # 永久记录为 None，后续无法证明不同模型是否使用了同一组环境。
        try:
            eval_ctx.terrain_info['terrain_seed'] = getattr(
                scenario,
                'current_terrain_seed',
                eval_ctx.terrain_info.get('terrain_seed'),
            )
            eval_ctx.terrain_info['terrain_variant_seed'] = _scenario_terrain_variant_seed(
                scenario,
                eval_ctx.terrain_info.get('terrain_variant_seed'),
            )
            actual_obstacle_seed = getattr(scenario, 'current_episode_obstacle_seed', None)
            if actual_obstacle_seed is not None:
                eval_ctx.terrain_info['obstacle_seed'] = int(actual_obstacle_seed)
        except Exception:
            pass

        initial_positions = _capture_agent_positions(world.agents)
        agent_goal_positions = _extract_agent_goal_positions(world, scenario)
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
            simulation_dt = float(getattr(world, 'dt', os.getenv('SIMULATION_DT', '0.08')))
        except Exception:
            simulation_dt = 0.08

        agent_first_reach_steps = []
        for pos, goal in zip(initial_positions, agent_goal_positions):
            dist_to_goal = _distance_3d(pos, goal)
            agent_first_reach_steps.append(0 if dist_to_goal is not None and dist_to_goal <= success_threshold else None)
        team_first_reach_step = (
            0 if agent_first_reach_steps and all(step == 0 for step in agent_first_reach_steps) else None
        )

        save_team_success_html = _env_flag("SAVE_TEAM_SUCCESS_HTML", False)
        save_interactive_traj = _env_flag("SAVE_INTERACTIVE_TRAJ", True)
        save_gazebo_replay = _env_flag("SAVE_GAZEBO_REPLAY", False)
        save_gazebo_dynamic_replay = _env_flag("SAVE_GAZEBO_DYNAMIC_REPLAY", False)
        save_trajectory_snapshot = _env_flag("SAVE_TRAJECTORY_SNAPSHOT", save_gazebo_replay or save_gazebo_dynamic_replay)
        save_all_episode_visualizations = _env_flag("SAVE_EVAL_ALL_EPISODES", False)
        save_best_traj = _env_flag("SAVE_BEST_TRAJ", True)
        save_trajectory_png = _env_flag("SAVE_EVAL_TRAJECTORY_PNG", False)
        save_actor_sequence = _env_flag("SAVE_EVAL_ACTOR_SEQUENCE", False)
        save_control_diagnostics = _env_flag("SAVE_EVAL_CONTROL_DIAGNOSTICS", save_gazebo_dynamic_replay)
        trajectory_sample_interval = _env_int("EVAL_TRAJECTORY_SAMPLE_INTERVAL", 1)
        need_trajectory_artifacts = (
            not getattr(self.args, 'disable_visualization', False)
            and (
                save_all_episode_visualizations
                or save_best_traj
                or save_trajectory_png
                or save_interactive_traj
                or save_team_success_html
                or save_trajectory_snapshot
                or save_gazebo_replay
                or save_gazebo_dynamic_replay
                or (not getattr(self.args, 'disable_gif', False))
            )
        )
        record_trajectory = bool(need_trajectory_artifacts)
        record_actions = save_actor_sequence or save_control_diagnostics or save_gazebo_dynamic_replay
        try:
            env._disable_trajectory_recording = not bool(need_trajectory_artifacts)
        except Exception:
            pass

        return {
            'ctx': eval_ctx,
            'episode': episode_idx,
            'processed_obs': self._process_observations_for_eval(obs_n),
            'done': False,
            'failed': False,
            'error': None,
            'wall_start': time.perf_counter(),
            'start_time': time.time(),
            'episode_reward': 0.0,
            'step_count': 0,
            'prev_positions': [pos.copy() if pos is not None else None for pos in initial_positions],
            'agent_path_lengths': [0.0 for _ in initial_positions],
            'direct_goal_distances': direct_goal_distances,
            'agent_goal_positions': agent_goal_positions,
            'agent_min_goal_distances': agent_min_goal_distances,
            'success_threshold': success_threshold,
            'simulation_dt': simulation_dt,
            'agent_first_reach_steps': agent_first_reach_steps,
            'team_first_reach_step': team_first_reach_step,
            'episode_trajectory': [],
            'episode_actions_history': [],
            'episode_executed_actions_history': [],
            'episode_velocity_history': [],
            'episode_goal_distance_history': [],
            'episode_dynamic_state_history': (
                [_capture_agent_dynamic_states(env.agents)] if save_gazebo_dynamic_replay else []
            ),
            'episode_dynamic_time_history': ([0.0] if save_gazebo_dynamic_replay else []),
            'episode_dynamic_step_indices': ([0] if save_gazebo_dynamic_replay else []),
            'episode_dynamic_raw_action_history': [],
            'episode_dynamic_executed_action_history': [],
            'episode_dynamic_pf_force_history': [],
            'record_trajectory': record_trajectory,
            'record_actions': record_actions,
            'save_control_diagnostics': save_control_diagnostics,
            'save_gazebo_dynamic_replay': save_gazebo_dynamic_replay,
            'trajectory_sample_interval': trajectory_sample_interval,
            'episode_inter_agent_collision_counts': [0] * len(getattr(env, 'agents', []) or []),
            'episode_inter_agent_collision_pair_count': 0,
            'episode_min_inter_agent_clearance': None,
            'episode_length': int(getattr(self.args, 'episode_length', 2200) or 2200),
            'eval_diagnostics': _new_eval_diagnostics_accumulator(),
            'eval_noise_streams': self._make_eval_noise_streams(episode_idx),
        }

    def _prepare_batched_episode_step(self, state, actions, raw_actions, step):
        """记录 step 前的可选轨迹/动作 artifact。"""
        env = state['ctx'].env
        if state.get('record_actions') and raw_actions is not None:
            try:
                state['episode_actions_history'].append([action.copy() for action in raw_actions])
            except Exception:
                pass
        if state.get('record_trajectory') and (step % int(state.get('trajectory_sample_interval', 1)) == 0):
            try:
                positions = [
                    agent.state.p_pos.copy() if hasattr(agent.state, 'p_pos') else [0, 0, 0]
                    for agent in env.agents
                ]
                state['episode_trajectory'].append(positions)
            except Exception:
                pass
        try:
            if state.get('record_actions') and state.get('save_control_diagnostics'):
                state['episode_executed_actions_history'].append([action.copy() for action in actions])
        except Exception:
            pass

    def _step_batched_episode_env(self, state, actions):
        """只执行单个 batch 子回合的 env.step，便于在线程池中并行。"""
        env = state['ctx'].env
        step_result = env.step(actions)
        if len(step_result) == 4:
            next_obs_n, rew_n, done_n, info_n = step_result
        elif len(step_result) == 5:
            next_obs_n, rew_n, terminated, truncated, info_n = step_result
            done_n = [t or tr for t, tr in zip(terminated, truncated)]
        else:
            raise ValueError(f"意外的环境step返回值: {len(step_result)}")
        return next_obs_n, rew_n, done_n, info_n

    def _complete_batched_episode_step(
        self,
        state,
        next_obs_n,
        rew_n,
        done_n,
        info_n,
        actions=None,
        raw_actions=None,
        pf_forces=None,
    ):
        """合并单个 batch 子回合的 step 后统计，保持原有结果语义。"""
        env = state['ctx'].env
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
        state['episode_reward'] += step_increment
        state['step_count'] += 1
        _accumulate_eval_step_diagnostics(
            state['eval_diagnostics'],
            env,
            actions=actions,
            raw_actions=raw_actions,
            pf_forces=pf_forces,
            action_force_ratio=getattr(self.args, 'action_force_ratio', 0.0),
        )
        if state.get('save_gazebo_dynamic_replay'):
            state['episode_dynamic_state_history'].append(_capture_agent_dynamic_states(env.agents))
            state['episode_dynamic_time_history'].append(float(state['step_count'] * state.get('simulation_dt', 0.08)))
            state['episode_dynamic_step_indices'].append(int(state['step_count']))
            state['episode_dynamic_raw_action_history'].append(_copy_action_frame(raw_actions))
            state['episode_dynamic_executed_action_history'].append(_copy_action_frame(actions))
            state['episode_dynamic_pf_force_history'].append(_copy_action_frame(pf_forces))

        current_positions = _capture_agent_positions(env.agents)
        if state.get('save_control_diagnostics'):
            current_velocities = []
            for agent in env.agents:
                try:
                    vel = _normalize_vec3(getattr(getattr(agent, 'state', None), 'p_vel', None))
                except Exception:
                    vel = None
                current_velocities.append(vel)
            if current_velocities:
                state['episode_velocity_history'].append(
                    [vel.copy() if vel is not None else None for vel in current_velocities]
                )

        agent_goal_positions = state['agent_goal_positions']
        if state.get('save_control_diagnostics') and current_positions and agent_goal_positions:
            state['episode_goal_distance_history'].append(
                [_distance_3d(pos, goal) for pos, goal in zip(current_positions, agent_goal_positions)]
            )

        for agent_idx in range(min(len(state['agent_path_lengths']), len(current_positions))):
            prev_pos = state['prev_positions'][agent_idx]
            curr_pos = current_positions[agent_idx]
            if prev_pos is None or curr_pos is None:
                continue
            step_distance = float(np.linalg.norm(curr_pos - prev_pos))
            if np.isfinite(step_distance):
                state['agent_path_lengths'][agent_idx] += step_distance
        state['prev_positions'] = [pos.copy() if pos is not None else None for pos in current_positions]

        try:
            pair_count_step, per_agent_step_counts, min_clearance_step = _compute_inter_agent_collision_snapshot(
                getattr(env, 'agents', [])
            )
            state['episode_inter_agent_collision_pair_count'] += int(pair_count_step)
            if min_clearance_step is not None:
                cur_min = state['episode_min_inter_agent_clearance']
                if cur_min is None or min_clearance_step < cur_min:
                    state['episode_min_inter_agent_clearance'] = float(min_clearance_step)
            counts = state['episode_inter_agent_collision_counts']
            if len(counts) < len(per_agent_step_counts):
                counts.extend([0] * (len(per_agent_step_counts) - len(counts)))
            for idx, per_agent_count in enumerate(per_agent_step_counts):
                counts[idx] += int(per_agent_count)
        except Exception:
            pass

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
                prev_min_goal_dist = state['agent_min_goal_distances'][agent_idx]
                if prev_min_goal_dist is None or dist_to_goal < prev_min_goal_dist:
                    state['agent_min_goal_distances'][agent_idx] = float(dist_to_goal)
                reached = dist_to_goal <= state['success_threshold']
                if reached and state['agent_first_reach_steps'][agent_idx] is None:
                    state['agent_first_reach_steps'][agent_idx] = state['step_count']
                if not reached:
                    team_reached_now = False
            if valid_reach_checks == 0:
                team_reached_now = False
            if team_reached_now and state['team_first_reach_step'] is None:
                state['team_first_reach_step'] = state['step_count']

        state['processed_obs'] = self._process_observations_for_eval(next_obs_n)
        if all(done_n) and (not getattr(self.args, 'disable_early_termination', False)):
            state['done'] = True
        return info_n

    def _advance_batched_episode_state(self, state, actions, raw_actions, step, pf_forces=None):
        """推进 batch 子回合一步，并更新轻量统计。"""
        self._prepare_batched_episode_step(state, actions, raw_actions, step)
        next_obs_n, rew_n, done_n, info_n = self._step_batched_episode_env(state, actions)
        return self._complete_batched_episode_step(
            state,
            next_obs_n,
            rew_n,
            done_n,
            info_n,
            actions=actions,
            raw_actions=raw_actions,
            pf_forces=pf_forces,
        )

    def _finalize_batched_episode_state(self, state):
        """将 batch 子回合状态转换为 evaluation_results.json 兼容的 episode_data。"""
        eval_ctx = state['ctx']
        env = eval_ctx.env
        episode_idx = int(state['episode'])
        episode_duration = time.time() - state['start_time']

        episode_collision_counts = []
        episode_min_distances = []
        episode_terrain_total = 0
        episode_obstacle_total = 0
        try:
            for agent in getattr(getattr(env, 'world', None), 'agents', []):
                if not hasattr(agent, 'debug_info') or not isinstance(agent.debug_info, dict):
                    agent.debug_info = {}
                penetration_count = agent.debug_info.get(
                    'total_penetration_count',
                    getattr(agent, 'current_episode_collision_count', 0),
                )
                try:
                    penetration_count = int(penetration_count) if np.isfinite(penetration_count) else 0
                except Exception:
                    penetration_count = 0
                episode_collision_counts.append(penetration_count)

                try:
                    terrain_collision_count = agent.debug_info.get('terrain_penetration_count', 0)
                    obstacle_collision_count = agent.debug_info.get('obstacle_collision_count', 0)
                    terrain_collision_count = int(terrain_collision_count) if np.isfinite(terrain_collision_count) else 0
                    obstacle_collision_count = int(obstacle_collision_count) if np.isfinite(obstacle_collision_count) else 0
                except Exception:
                    terrain_collision_count = 0
                    obstacle_collision_count = 0
                episode_terrain_total += terrain_collision_count
                episode_obstacle_total += obstacle_collision_count

                min_dist = agent.debug_info.get('d_min_current', None)
                if min_dist is None and hasattr(agent, 'last_min_distance'):
                    min_dist = agent.last_min_distance
                try:
                    if isinstance(min_dist, np.ndarray):
                        min_dist = float(min_dist[-1] if min_dist.size > 0 and min_dist.ndim > 0 else min_dist.item())
                    elif min_dist is not None:
                        min_dist = float(min_dist)
                except Exception:
                    min_dist = None
                if min_dist is not None and np.isfinite(min_dist):
                    episode_min_distances.append(min_dist)
        except Exception:
            pass

        agent_success_flags = []
        agent_safe_flags = []
        team_success_flag = 1
        try:
            thr_success = float(getattr(self.args, 'success_distance_threshold', 4.0))
            scn = getattr(env, 'scenario', None)
            goal_pos = getattr(scn, 'goal_pos', None) if scn is not None else None
            if goal_pos is None and scn is not None and hasattr(scn, 'get_goal_pos'):
                try:
                    goal_pos = scn.get_goal_pos()
                except Exception:
                    goal_pos = None
            for agent_idx, agent in enumerate(getattr(getattr(env, 'world', None), 'agents', [])):
                pos = getattr(getattr(agent, 'state', None), 'p_pos', None)
                if pos is None or len(pos) < 3:
                    agent_success_flags.append(0)
                    team_success_flag = 0
                    continue
                ag_goal = None
                if hasattr(agent, 'goal_a') and hasattr(agent.goal_a, 'state') and getattr(agent.goal_a.state, 'p_pos', None) is not None:
                    ag_goal = agent.goal_a.state.p_pos
                if ag_goal is None:
                    ag_goal = goal_pos
                if ag_goal is None:
                    agent_success_flags.append(0)
                    team_success_flag = 0
                    continue
                dx = pos[0] - ag_goal[0]
                dy = pos[1] - ag_goal[1]
                dz = pos[2] - ag_goal[2]
                reach_i = ((dx * dx + dy * dy + dz * dz) ** 0.5) <= thr_success
                safe_i = True
                for attr in ('_episode_has_collision', '_had_obstacle_collision', '_had_terrain_contact_or_penetration'):
                    try:
                        if getattr(agent, attr, False):
                            safe_i = False
                    except Exception:
                        pass
                pen_count = episode_collision_counts[agent_idx] if agent_idx < len(episode_collision_counts) else None
                if pen_count is None and hasattr(agent, 'debug_info') and isinstance(agent.debug_info, dict):
                    pen_count = agent.debug_info.get('total_penetration_count', 0)
                try:
                    pen_count = int(pen_count) if np.isfinite(pen_count) else 0
                except Exception:
                    pen_count = None
                if pen_count is None or pen_count > 0:
                    safe_i = False
                succ_i = 1 if (reach_i and safe_i) else 0
                agent_safe_flags.append(1 if safe_i else 0)
                agent_success_flags.append(succ_i)
                if succ_i == 0:
                    team_success_flag = 0
        except Exception:
            agent_success_flags = [0] * len(episode_collision_counts) if episode_collision_counts else []
            agent_safe_flags = [0] * len(episode_collision_counts) if episode_collision_counts else []
            team_success_flag = 0

        world_snapshot = _get_world_success_snapshot(env, expected_count=len(episode_collision_counts))
        if world_snapshot is not None:
            agent_safe_flags = world_snapshot['safe_flags']
            agent_success_flags = world_snapshot['success_flags']
            team_success_flag = world_snapshot['team_success']

        total_collisions = sum(episode_collision_counts) if episode_collision_counts else 0
        try:
            total_collisions = int(total_collisions) if np.isfinite(total_collisions) else 0
        except Exception:
            total_collisions = 0
        min_distance_stat = None
        if episode_min_distances:
            try:
                min_distance_stat = {
                    'mean': float(np.mean(episode_min_distances)),
                    'min': float(np.min(episode_min_distances)),
                }
            except Exception:
                min_distance_stat = None

        direct_goal_distances = state['direct_goal_distances']
        agent_path_lengths = state['agent_path_lengths']
        agent_path_efficiencies = []
        for direct_dist, path_len in zip(direct_goal_distances, agent_path_lengths):
            if direct_dist is None or path_len is None or path_len <= 1e-9:
                agent_path_efficiencies.append(None)
            else:
                agent_path_efficiencies.append(float(np.clip(direct_dist / max(path_len, 1e-9), 0.0, 1.0)))
        valid_direct_dists = [dist for dist in direct_goal_distances if dist is not None]
        team_direct_distance = float(np.sum(valid_direct_dists)) if valid_direct_dists else None
        team_total_path_length = float(np.sum(agent_path_lengths)) if agent_path_lengths else 0.0
        path_efficiency = None
        if team_direct_distance is not None and team_total_path_length > 1e-9:
            path_efficiency = float(np.clip(team_direct_distance / team_total_path_length, 0.0, 1.0))

        final_positions_for_stats = _capture_agent_positions(env.agents)
        agent_goal_positions = state['agent_goal_positions']
        agent_final_goal_distances = [
            _distance_3d(pos, goal) for pos, goal in zip(final_positions_for_stats, agent_goal_positions)
        ]
        valid_final_goal_distances = [dist for dist in agent_final_goal_distances if dist is not None]
        final_goal_distance = float(np.sum(valid_final_goal_distances)) if valid_final_goal_distances else None
        valid_min_goal_distances = [dist for dist in state['agent_min_goal_distances'] if dist is not None]
        min_goal_distance = float(np.sum(valid_min_goal_distances)) if valid_min_goal_distances else None

        first_reach_step = state['team_first_reach_step']
        first_reach_time = float(first_reach_step * state['simulation_dt']) if first_reach_step is not None else None
        success_flag = 1 if team_success_flag == 1 else 0
        arrival_step = first_reach_step if success_flag == 1 else None
        arrival_time = first_reach_time if success_flag == 1 else None

        penetration_depths = []
        penetration_count_episodes = 0
        for min_dist in episode_min_distances:
            if min_dist is not None and np.isfinite(min_dist) and min_dist < 0:
                penetration_depths.append(abs(min_dist))
                penetration_count_episodes += 1
        penetration_stat = None
        if penetration_depths:
            penetration_stat = {
                'count': penetration_count_episodes,
                'max_depth': float(np.max(penetration_depths)),
                'mean_depth': float(np.mean(penetration_depths)),
                'min_depth': float(np.min(penetration_depths)),
            }

        if state.get('record_trajectory'):
            try:
                final_positions = [
                    agent.state.p_pos.copy() if hasattr(agent.state, 'p_pos') else [0, 0, 0]
                    for agent in env.agents
                ]
                state['episode_trajectory'].append(final_positions)
            except Exception:
                pass

        eval_diag = _finalize_eval_diagnostics(state.get('eval_diagnostics', _new_eval_diagnostics_accumulator()))
        reward_decomposition = eval_diag['reward_components']
        diagnostic_metrics = eval_diag['diagnostic_metrics']
        terminal_success_bonus_applied = bool(
            abs(float(reward_decomposition.get('reward_terminal_success', 0.0) or 0.0)) > 1e-8
        )
        episode_done_reason = _infer_episode_done_reason(
            success_flag,
            first_reach_step,
            state['step_count'],
            state.get('episode_length', getattr(self.args, 'episode_length', 2200)),
            total_collisions,
        )
        if world_snapshot is not None and world_snapshot.get('done_reason'):
            episode_done_reason = world_snapshot['done_reason']

        return {
            'episode': episode_idx,
            'reward': state['episode_reward'],
            'steps': state['step_count'],
            'trajectory': state['episode_trajectory'] if state.get('record_trajectory') else [],
            'actions_history': state['episode_actions_history'] if state.get('record_actions') else [],
            'executed_actions_history': state['episode_executed_actions_history'] if state.get('save_control_diagnostics') else [],
            'velocity_history': state['episode_velocity_history'] if state.get('save_control_diagnostics') else [],
            'goal_distance_history': state['episode_goal_distance_history'] if state.get('save_control_diagnostics') else [],
            'dynamic_state_history': state['episode_dynamic_state_history'] if state.get('save_gazebo_dynamic_replay') else [],
            'dynamic_time_history': state['episode_dynamic_time_history'] if state.get('save_gazebo_dynamic_replay') else [],
            'dynamic_step_indices': state['episode_dynamic_step_indices'] if state.get('save_gazebo_dynamic_replay') else [],
            'dynamic_raw_action_history': state['episode_dynamic_raw_action_history'] if state.get('save_gazebo_dynamic_replay') else [],
            'dynamic_executed_action_history': state['episode_dynamic_executed_action_history'] if state.get('save_gazebo_dynamic_replay') else [],
            'dynamic_pf_force_history': state['episode_dynamic_pf_force_history'] if state.get('save_gazebo_dynamic_replay') else [],
            'duration': episode_duration,
            'wall_time_seconds': float(time.perf_counter() - state['wall_start']),
            'collision_count': total_collisions,
            'terrain_collision_count': int(episode_terrain_total),
            'obstacle_collision_count': int(episode_obstacle_total),
            'agent_collision_counts': episode_collision_counts,
            'inter_agent_collision_count': int(state['episode_inter_agent_collision_pair_count']),
            'agent_inter_agent_collision_counts': [int(v) for v in state['episode_inter_agent_collision_counts']],
            'min_inter_agent_clearance': (
                float(state['episode_min_inter_agent_clearance']) if state['episode_min_inter_agent_clearance'] is not None else None
            ),
            'min_distance': min_distance_stat,
            'success': success_flag,
            'agent_success_flags': agent_success_flags,
            'agent_safe_flags': agent_safe_flags,
            'team_success': team_success_flag,
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
            'agent_min_goal_distances': [float(v) if v is not None else None for v in state['agent_min_goal_distances']],
            'agent_first_reach_steps': [int(v) if v is not None else None for v in state['agent_first_reach_steps']],
            'path_efficiency': path_efficiency,
            'agent_path_efficiencies': agent_path_efficiencies,
            'penetration_stat': penetration_stat,
            'reward_decomposition': reward_decomposition,
            'diagnostic_metrics': diagnostic_metrics,
            'terminal_success_bonus_applied': terminal_success_bonus_applied,
            'episode_done_reason': episode_done_reason,
            'vis_context': self._capture_episode_vis_context_for_context(eval_ctx, episode_idx),
            'terrain_complexity_level': eval_ctx.terrain_level,
            'terrain_seed': eval_ctx.terrain_info.get('terrain_seed'),
            'terrain_variant_seed': eval_ctx.terrain_info.get('terrain_variant_seed'),
            'obstacle_seed': eval_ctx.terrain_info.get('obstacle_seed'),
        }

    def _evaluate_episode_batch(self, episode_contexts):
        """方案A：单进程内多 episode 上下文，batch actor/PF 推理，可选并行 env.step。"""
        states = [self._init_batched_episode_state(eval_ctx) for eval_ctx in episode_contexts]
        try:
            episode_length = int(getattr(self.args, 'episode_length', 2200) or 2200)
        except Exception:
            episode_length = 2200
        episode_length = max(1, episode_length)

        use_tf_potential_field = getattr(self.args, 'use_tf_potential_field', True)
        action_force_ratio = getattr(self.args, 'action_force_ratio', 0.0)
        use_fr = getattr(self.maddpg.args, 'use_fr_feature', False)
        use_pf = getattr(self.maddpg.args, 'use_pf_feature', False)
        base_obs_dim = self._get_base_obs_dim()
        try:
            eval_env_step_threads = int(
                getattr(self, '_eval_env_step_threads', getattr(self.args, 'eval_env_step_threads', 1)) or 1
            )
        except Exception:
            eval_env_step_threads = 1
        eval_env_step_threads = max(1, min(eval_env_step_threads, len(states)))
        step_executor = (
            ThreadPoolExecutor(max_workers=eval_env_step_threads, thread_name_prefix="eval-env-step")
            if eval_env_step_threads > 1
            else None
        )

        try:
            for step in range(episode_length):
                active_states = [state for state in states if not state.get('done') and not state.get('failed')]
                if not active_states:
                    break
                try:
                    processed_batch = np.stack([state['processed_obs'] for state in active_states], axis=0).astype(np.float32)
                    policy_obs, base_obs_for_correction_tf, pf_shared_context = self._build_eval_actor_obs_tf_with_context(
                        processed_batch,
                        use_pf=use_pf,
                        use_tf_potential_field=use_tf_potential_field,
                        action_force_ratio=action_force_ratio,
                    )
                    raw_actions_tf = self.select_actions_eval(policy_obs, use_fr=use_fr, use_pf=use_pf)
                    raw_actions_tf = self._apply_eval_action_noise(
                        raw_actions_tf,
                        [state['eval_noise_streams'] for state in active_states],
                    )
                    if isinstance(raw_actions_tf, tf.Tensor):
                        raw_actions_tensor = raw_actions_tf
                    else:
                        raw_actions_tensor = tf.convert_to_tensor(raw_actions_tf, dtype=tf.float32)
                    try:
                        action_dim = int(raw_actions_tensor.shape[-1])
                    except Exception:
                        action_dim = int(tf.shape(raw_actions_tensor)[-1].numpy())
                    raw_actions = raw_actions_tensor.numpy()
                    pf_forces_batch = None

                    if use_tf_potential_field and action_force_ratio > 0.0:
                        flat_actions_tf = tf.reshape(raw_actions_tensor, [-1, action_dim])
                        if (
                            pf_shared_context is not None
                            and hasattr(self.maddpg, '_apply_potential_field_correction_from_geometry_context_tf')
                        ):
                            corrected_head_tf, pf_force_flat_tf = self.maddpg._apply_potential_field_correction_from_geometry_context_tf(
                                flat_actions_tf,
                                float(action_force_ratio),
                                pf_shared_context[0],
                                pf_shared_context[1],
                            )
                        else:
                            try:
                                correction_obs_dim = int(base_obs_for_correction_tf.shape[-1])
                            except Exception:
                                correction_obs_dim = base_obs_dim
                            flat_obs_tf = tf.reshape(base_obs_for_correction_tf, [-1, correction_obs_dim])
                            corrected_head_tf, pf_force_flat_tf = self.maddpg._apply_potential_field_correction(
                                flat_actions_tf,
                                flat_obs_tf,
                                action_force_ratio,
                            )
                        try:
                            pf_forces_batch = pf_force_flat_tf.numpy().reshape(
                                (len(active_states), self.n_agents, 3)
                            )
                        except Exception:
                            pf_forces_batch = None
                        if action_dim > 3:
                            corrected_actions_tf = tf.concat([corrected_head_tf, flat_actions_tf[:, 3:]], axis=1)
                        else:
                            corrected_actions_tf = corrected_head_tf
                        actions_batch = corrected_actions_tf.numpy().reshape(
                            (len(active_states), self.n_agents, action_dim)
                        )
                    else:
                        actions_batch = raw_actions_tensor.numpy()
                except Exception as action_e:
                    for state in active_states:
                        state['failed'] = True
                        state['error'] = action_e
                    continue

                step_items = [
                    (
                        state,
                        np.asarray(actions_batch[idx], dtype=np.float32),
                        None if raw_actions is None else np.asarray(raw_actions[idx], dtype=np.float32),
                        None if pf_forces_batch is None else np.asarray(pf_forces_batch[idx], dtype=np.float32),
                    )
                    for idx, state in enumerate(active_states)
                ]

                if step_executor is not None and len(step_items) > 1:
                    runnable_items = []
                    for state, actions_np, raw_actions_np, pf_forces_np in step_items:
                        try:
                            self._prepare_batched_episode_step(state, actions_np, raw_actions_np, step)
                            runnable_items.append((state, actions_np, raw_actions_np, pf_forces_np))
                        except Exception as step_e:
                            state['failed'] = True
                            state['error'] = step_e
                    futures = [
                        (state, actions_np, raw_actions_np, pf_forces_np, step_executor.submit(self._step_batched_episode_env, state, actions_np))
                        for state, actions_np, raw_actions_np, pf_forces_np in runnable_items
                    ]
                    for state, actions_np, raw_actions_np, pf_forces_np, future in futures:
                        try:
                            next_obs_n, rew_n, done_n, info_n = future.result()
                            self._complete_batched_episode_step(
                                state,
                                next_obs_n,
                                rew_n,
                                done_n,
                                info_n,
                                actions=actions_np,
                                raw_actions=raw_actions_np,
                                pf_forces=pf_forces_np,
                            )
                        except Exception as step_e:
                            state['failed'] = True
                            state['error'] = step_e
                else:
                    for state, actions_np, raw_actions_np, pf_forces_np in step_items:
                        try:
                            self._advance_batched_episode_state(
                                state,
                                actions=actions_np,
                                raw_actions=raw_actions_np,
                                step=step + 1,
                                pf_forces=pf_forces_np,
                            )
                        except Exception as step_e:
                            state['failed'] = True
                            state['error'] = step_e
        finally:
            if step_executor is not None:
                step_executor.shutdown(wait=True)

        episode_data = []
        for state in states:
            if state.get('failed'):
                print(f"❌ 回合 {state['episode'] + 1} batch评估失败: {state.get('error')}")
                continue
            episode_data.append(self._finalize_batched_episode_state(state))
        return sorted(episode_data, key=lambda item: int(item.get('episode', 0)))
        
    def evaluate_single_episode(self, episode_idx):
        """评估单个回合，仿照1.0版本的逻辑"""
        try:
            quiet_output = os.getenv("QUIET_OUTPUT", "1").lower() in ("1", "true", "yes", "on")
        except Exception:
            quiet_output = True
        eval_noise_streams = self._make_eval_noise_streams(episode_idx)
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
        
        validation_active = _env_flag("GAZEBO_LIVE_VALIDATION", False)
        validation_root = None
        if validation_active:
            try:
                from gazebo_live_validation import validation_root_from_args
                validation_root = validation_root_from_args(self.args)
            except Exception:
                validation_root = Path(os.getenv("GAZEBO_LIVE_VALIDATION_DIR", "results/gazebo_live_validation")).resolve()

        # 环境重置
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            obs_n, _ = reset_result
        else:
            obs_n = reset_result

        gazebo_live_launch = None
        gazebo_live_launch_error = None
        try:
            gazebo_live_launch = _export_and_launch_gazebo_live_world(
                self.scenario,
                self.world,
                self.args,
                episode_idx,
                quiet_output=quiet_output,
            )
        except Exception as launch_err:
            gazebo_live_launch_error = str(launch_err)
            if _env_flag("GAZEBO_LIVE_REQUIRED", False):
                raise
            if not quiet_output:
                print(f"⚠️ Gazebo live world 同源导出/启动失败，将继续普通评估: {gazebo_live_launch_error}")

        gazebo_live_scene_check = None
        if isinstance(gazebo_live_launch, dict) and gazebo_live_launch.get("autolaunch_error"):
            gazebo_live_launch_error = str(gazebo_live_launch.get("autolaunch_error"))
        if validation_active and isinstance(gazebo_live_launch, dict):
            try:
                from gazebo_live_validation import run_scene_binding_check
                gazebo_live_scene_check = run_scene_binding_check(
                    scenario=self.scenario,
                    world=self.world,
                    gazebo_live=gazebo_live_launch,
                    episode_idx=episode_idx,
                    terrain_info=getattr(self, "_current_episode_terrain_info", {}) or {},
                    validation_root=validation_root,
                )
                if (
                    not bool(gazebo_live_scene_check.get("ok", False))
                    and _env_flag("GAZEBO_LIVE_SCENE_CHECK_REQUIRED", True)
                ):
                    failed = [
                        item.get("name")
                        for item in gazebo_live_scene_check.get("checks", [])
                        if not item.get("ok")
                    ]
                    raise RuntimeError(f"Gazebo live scene binding check failed: {failed}")
            except Exception as scene_check_err:
                gazebo_live_launch_error = str(scene_check_err)
                if _env_flag("GAZEBO_LIVE_REQUIRED", False) or _env_flag("GAZEBO_LIVE_SCENE_CHECK_REQUIRED", validation_active):
                    raise
                if not quiet_output:
                    print(f"⚠️ Gazebo live scene binding check failed: {gazebo_live_launch_error}")

        python_scene_signature_record = None
        if validation_active:
            try:
                from gazebo_live_validation import build_python_scene_signature
                python_scene_signature_record = build_python_scene_signature(
                    scenario=self.scenario,
                    world=self.world,
                    terrain_info=getattr(self, "_current_episode_terrain_info", {}) or {},
                )
            except Exception as python_scene_sig_err:
                python_scene_signature_record = {"error": str(python_scene_sig_err)}
        
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
        save_gazebo_replay = _env_flag("SAVE_GAZEBO_REPLAY", False)
        save_gazebo_dynamic_replay = _env_flag("SAVE_GAZEBO_DYNAMIC_REPLAY", False)
        save_trajectory_snapshot = _env_flag("SAVE_TRAJECTORY_SNAPSHOT", save_gazebo_replay or save_gazebo_dynamic_replay)
        save_all_episode_visualizations = _env_flag("SAVE_EVAL_ALL_EPISODES", False)
        save_best_traj = _env_flag("SAVE_BEST_TRAJ", True)
        save_trajectory_png = _env_flag("SAVE_EVAL_TRAJECTORY_PNG", False)
        save_actor_sequence = _env_flag("SAVE_EVAL_ACTOR_SEQUENCE", False)
        save_control_diagnostics = _env_flag("SAVE_EVAL_CONTROL_DIAGNOSTICS", save_gazebo_dynamic_replay)
        trajectory_sample_interval = _env_int("EVAL_TRAJECTORY_SAMPLE_INTERVAL", 1)
        need_trajectory_artifacts = (
            validation_active
            or (
                not getattr(self.args, 'disable_visualization', False)
                and (
                    save_all_episode_visualizations
                    or save_best_traj
                    or save_trajectory_png
                    or save_interactive_traj
                    or save_team_success_html
                    or save_trajectory_snapshot
                    or save_gazebo_replay
                    or save_gazebo_dynamic_replay
                    or (not getattr(self.args, 'disable_gif', False))
                )
            )
        )
        # 轻量模式下允许通过稀疏采样保留评估轨迹；若本轮不生成任何轨迹类产物，则完全跳过轨迹缓存。
        record_trajectory = bool(need_trajectory_artifacts)
        # 动作历史只服务于时序图/控制诊断；默认不再在纯HTML评估里额外记录。
        record_actions = save_actor_sequence or save_control_diagnostics or save_gazebo_dynamic_replay or validation_active
        episode_trajectory = []
        episode_actions_history = []  # 🔧 新增：记录动作历史（用于生成动作时序图）
        episode_executed_actions_history = []  # 记录实际送入环境的动作（含PF修正）
        episode_velocity_history = []  # 记录每步速度向量，判断是否“推不起来”
        episode_goal_distance_history = []  # 记录每步到目标距离，判断是否持续推进
        episode_dynamic_state_history = []
        episode_dynamic_time_history = []
        episode_dynamic_step_indices = []
        episode_dynamic_raw_action_history = []
        episode_dynamic_executed_action_history = []
        episode_dynamic_pf_force_history = []
        step_count = 0
        episode_inter_agent_collision_counts = [0] * len(getattr(self.env, 'agents', []) or [])
        episode_inter_agent_collision_pair_count = 0
        episode_min_inter_agent_clearance = None
        eval_diagnostics = _new_eval_diagnostics_accumulator()

        # 与产物需求对齐：若本轮不生成任何轨迹类文件，则连环境内部轨迹也一起关闭。
        try:
            self.env._disable_trajectory_recording = not bool(need_trajectory_artifacts)
        except Exception:
            pass
        
        # 处理观察数据
        processed_obs = self._process_observations_for_eval(obs_n)
        if save_gazebo_dynamic_replay:
            episode_dynamic_state_history.append(_capture_agent_dynamic_states(self.env.agents))
            episode_dynamic_time_history.append(0.0)
            episode_dynamic_step_indices.append(0)

        gazebo_live_client = None
        gazebo_live_sync_active = False
        gazebo_live_sync_error = None
        gazebo_live_sync_every = max(1, _env_int("GAZEBO_LIVE_SYNC_EVERY", 1))
        gazebo_live_control_mode = os.getenv("GAZEBO_LIVE_CONTROL_MODE", "pose").strip().lower()
        if gazebo_live_control_mode in ("cmd_vel", "twist"):
            gazebo_live_control_mode = "velocity"
        if gazebo_live_control_mode not in ("pose", "velocity"):
            gazebo_live_control_mode = "pose"
        gazebo_live_consistency_mode = os.getenv("GAZEBO_LIVE_CONSISTENCY_MODE", "gazebo_authoritative").strip().lower()
        if gazebo_live_consistency_mode in ("python", "python_only", "python_authoritative", "mirror", "follow_python"):
            gazebo_live_consistency_mode = "python_authoritative"
        else:
            gazebo_live_consistency_mode = "gazebo_authoritative"
        gazebo_live_python_authoritative = gazebo_live_consistency_mode == "python_authoritative"
        gazebo_live_semantic_mode = os.getenv("GAZEBO_LIVE_SEMANTIC_MODE", "transfer_equivalence").strip().lower()
        gazebo_live_collision_mode = _gazebo_live_collision_mode()
        gazebo_live_physical_collision_enabled = gazebo_live_collision_mode != "nonblocking"
        gazebo_live_physical_contact_semantics = gazebo_live_semantic_mode in (
            "physical",
            "physical_robustness",
            "contact_authoritative",
            "gazebo_contact_authoritative",
        )
        gazebo_live_command_sleep = max(0.0, _env_float("GAZEBO_LIVE_COMMAND_SLEEP", 0.0))
        gazebo_live_step_iterations = max(0, _env_int("GAZEBO_LIVE_STEP_ITERATIONS", 0))
        gazebo_live_state_feedback = _env_flag("GAZEBO_LIVE_STATE_FEEDBACK", False)
        gazebo_live_state_feedback_timeout = max(0.0, _env_float("GAZEBO_LIVE_STATE_FEEDBACK_TIMEOUT", 0.2))
        gazebo_live_authoritative_feedback = (
            gazebo_live_control_mode == "velocity"
            and gazebo_live_state_feedback
            and not gazebo_live_python_authoritative
            and _env_flag("GAZEBO_LIVE_AUTHORITATIVE_FEEDBACK", True)
        )
        gazebo_live_contact_authoritative = _env_flag(
            "GAZEBO_LIVE_CONTACT_AUTHORITATIVE",
            gazebo_live_physical_contact_semantics and not gazebo_live_python_authoritative,
        )
        gazebo_live_contact_marks_collision = _env_flag(
            "GAZEBO_LIVE_CONTACT_MARKS_COLLISION",
            gazebo_live_contact_authoritative,
        )
        gazebo_live_contact_terminates = _env_flag("GAZEBO_LIVE_CONTACT_TERMINATES", False)
        gazebo_live_pose_correction = _env_flag(
            "GAZEBO_LIVE_POSE_CORRECTION",
            gazebo_live_python_authoritative,
        )
        gazebo_live_state_feedback_updates = 0
        gazebo_live_state_feedback_misses = 0
        gazebo_live_authoritative_feedback_updates = 0
        gazebo_live_authoritative_feedback_errors = 0
        gazebo_contact_detected = False
        gazebo_contact_raw_flag_count = 0
        gazebo_contact_false_positive_count = 0
        gazebo_contact_step = None
        gazebo_first_contact_step = None
        gazebo_first_contact_pair = None
        gazebo_first_contact_position = None
        gazebo_first_contact_agent_id = None
        gazebo_contact_count = 0
        gazebo_contact_agent_indices = []
        gazebo_contact_pairs = []
        gazebo_real_contact_pairs = []
        gazebo_contact_pair_class_counts = {}
        gazebo_contact_debug_records = []
        gazebo_contact_debug_tail = deque(maxlen=20)
        gazebo_contact_debug_window_start = _env_int_or_default("GAZEBO_LIVE_CONTACT_DEBUG_WINDOW_START", 380)
        gazebo_contact_debug_post_steps = max(0, _env_int_or_default("GAZEBO_LIVE_CONTACT_DEBUG_POST_STEPS", 5))
        gazebo_contact_debug_agent_id = _env_int_or_default("GAZEBO_LIVE_CONTACT_DEBUG_AGENT_ID", 2)
        gazebo_contact_debug_obstacle = os.getenv("GAZEBO_LIVE_CONTACT_DEBUG_OBSTACLE", "obstacle_1")
        gazebo_first_contact_pending_snapshot = False
        validation_trace_tail = deque(maxlen=20)
        gazebo_live_client_stats = {
            "semantic_mode": gazebo_live_semantic_mode,
            "feedback_velocity_mode": os.getenv("GAZEBO_LIVE_FEEDBACK_VELOCITY_MODE", "clamp").strip().lower(),
            "feedback_acceleration_mode": os.getenv("GAZEBO_LIVE_FEEDBACK_ACCELERATION_MODE", "estimate").strip().lower(),
            "pose_jump_reject_count": 0,
            "max_pose_jump_observed": 0.0,
            "max_feedback_speed_observed": 0.0,
            "max_feedback_accel_observed": 0.0,
            "cmd_vel_publish_count": 0,
            "pose_publish_count": 0,
            "bridge_ack_enabled": _env_flag("GAZEBO_LIVE_WAIT_ACK", False),
            "bridge_ack_count": 0,
            "bridge_ack_timeout_count": 0,
            "bridge_last_ack_state_frame": None,
            "pre_step_sleep_ms": _env_int("GAZEBO_LIVE_PRE_STEP_SLEEP_MS", 0),
            "post_step_sleep_ms": _env_int("GAZEBO_LIVE_POST_STEP_SLEEP_MS", 0),
            "wall_time_step_ms": _env_int("GAZEBO_LIVE_WALL_TIME_STEP_MS", 0),
            "pause_for_step": _env_flag("GAZEBO_LIVE_PAUSE_FOR_STEP", False),
            "world_name": None,
            "agent_prefix": None,
            "bridge_running": False,
            "state_file": os.getenv("GAZEBO_LIVE_STATE_FILE", None),
            "contact_flag_file": os.getenv("GAZEBO_LIVE_CONTACT_FLAG_FILE", None),
            "contact_feedback_enabled": _env_flag("GAZEBO_LIVE_CONTACT_FEEDBACK", True),
            "contact_feedback_armed": False,
            "contact_topics": [],
        }
        apf_backend = str(getattr(self.args, "apf_backend", "python_original") or "python_original").strip().lower()
        if apf_backend not in ("python_original", "gazebo_apf"):
            apf_backend = "python_original"
        gazebo_apf_enabled = apf_backend == "gazebo_apf"
        gazebo_apf_provider = None
        gazebo_apf_calculator = None
        gazebo_apf_validator = None
        gazebo_apf_metrics = None
        gazebo_apf_error = None
        gazebo_velocity_filter = None
        gazebo_velocity_filter_mode = "off"
        gazebo_velocity_filter_enabled = False
        gazebo_velocity_filter_error = None
        gazebo_velocity_filter_records = []
        gazebo_velocity_filter_summary = None
        gazebo_velocity_filter_artifacts = {}
        if gazebo_apf_enabled:
            try:
                from gazebo_apf import GazeboAPFCalculator
                from gazebo_apf_state_provider import GazeboAPFStateProvider
                from gazebo_apf_validator import GazeboAPFValidator

                scenario_json = (
                    gazebo_live_launch.get("scenario_json")
                    if isinstance(gazebo_live_launch, dict)
                    else None
                )
                gazebo_apf_provider = GazeboAPFStateProvider.from_runtime(
                    scenario_json=scenario_json,
                    scenario=self.scenario,
                    world=self.world,
                    agent_count=self.n_agents,
                    state_file=os.getenv("GAZEBO_LIVE_STATE_FILE", None),
                    contact_flag_file=os.getenv("GAZEBO_LIVE_CONTACT_FLAG_FILE", None),
                    agent_prefix=os.getenv("GAZEBO_LIVE_AGENT_PREFIX", "dynamic_agent_"),
                    state_feedback_dt=float(getattr(self.args, "simulation_dt", os.getenv("SIMULATION_DT", "0.08"))),
                )
                gazebo_apf_calculator = GazeboAPFCalculator.from_args(self.args, gazebo_apf_provider)
                gazebo_apf_validator = GazeboAPFValidator(
                    output_dir=getattr(self.args, "gazebo_apf_output_dir", "results/gazebo_apf_validation"),
                    mismatch_threshold=float(getattr(self.args, "gazebo_apf_consistency_threshold", 0.05)),
                    direction_threshold=float(os.getenv("GAZEBO_APF_DIRECTION_THRESHOLD", "0.995")),
                    max_cases=int(os.getenv("GAZEBO_APF_MAX_MISMATCH_CASES", "50")),
                )
                validation_active = True
                if not quiet_output:
                    print(
                        "✅ Gazebo APF backend enabled: "
                        f"scenario_json={scenario_json}, output={gazebo_apf_validator.output_dir}"
                    )
            except Exception as apf_err:
                gazebo_apf_error = str(apf_err)
                if _env_flag("GAZEBO_APF_REQUIRED", True):
                    raise
                gazebo_apf_enabled = False
                if not quiet_output:
                    print(f"⚠️ Gazebo APF backend init failed; fallback to python_original: {gazebo_apf_error}")
        if gazebo_apf_enabled:
            try:
                from gazebo_velocity_safety_filter import GazeboObstacleVelocitySafetyFilter

                gazebo_velocity_filter = GazeboObstacleVelocitySafetyFilter.from_env()
                gazebo_velocity_filter_mode = gazebo_velocity_filter.config.mode
                gazebo_velocity_filter_enabled = bool(gazebo_velocity_filter.config.enabled)
                monitor_enabled = _env_flag("GAZEBO_LIVE_OBSTACLE_SAFETY_MONITOR", True)
                if not gazebo_velocity_filter_enabled and not monitor_enabled:
                    gazebo_velocity_filter = None
                if not quiet_output and gazebo_velocity_filter is not None:
                    print(
                        "✅ Gazebo obstacle velocity safety filter/monitor: "
                        f"mode={gazebo_velocity_filter_mode}, "
                        f"enabled={gazebo_velocity_filter_enabled}, "
                        f"margin={gazebo_velocity_filter.config.safety_margin}, "
                        f"stopping_margin={gazebo_velocity_filter.config.stopping_margin}"
                    )
            except Exception as filter_err:
                gazebo_velocity_filter_error = str(filter_err)
                if _env_flag("GAZEBO_LIVE_OBSTACLE_SAFETY_REQUIRED", gazebo_apf_enabled):
                    raise
                gazebo_velocity_filter = None
        if _env_flag("GAZEBO_LIVE_SYNC", False):
            try:
                from gazebo_live_pose_client import make_live_pose_client_from_env

                gazebo_live_client = make_live_pose_client_from_env(len(getattr(self.env, "agents", []) or []))
                if gazebo_live_client is not None:
                    gazebo_live_client.start()
                    gazebo_live_client.send_agents(self.env.agents)
                    try:
                        gazebo_live_client.clear_contact_flag()
                    except Exception:
                        pass
                    gazebo_live_sync_active = True
                    gazebo_live_client_stats["world_name"] = getattr(gazebo_live_client, "world", None)
                    gazebo_live_client_stats["agent_prefix"] = getattr(gazebo_live_client, "agent_prefix", None)
                    gazebo_live_client_stats["bridge_running"] = bool(gazebo_live_client.is_running())
                    gazebo_live_client_stats["state_file"] = str(getattr(gazebo_live_client, "state_file", "")) or None
                    gazebo_live_client_stats["contact_flag_file"] = str(getattr(gazebo_live_client, "contact_flag_file", "")) or None
                    gazebo_live_client_stats["contact_feedback_enabled"] = bool(getattr(gazebo_live_client, "contact_feedback", False))
                    gazebo_live_client_stats["contact_feedback_armed"] = bool(gazebo_live_client.contact_feedback_armed())
                    gazebo_live_client_stats["contact_topics"] = gazebo_live_client.expected_contact_topics()
                    if not quiet_output:
                        print(
                            f"✅ Gazebo live sync 已启动: world={gazebo_live_client.world}, "
                            f"agents={gazebo_live_client.agent_count}, mode={gazebo_live_control_mode}, "
                            f"every={gazebo_live_sync_every} step, gz_steps={gazebo_live_step_iterations}, "
                            f"state_feedback={gazebo_live_state_feedback}, "
                            f"authoritative_feedback={gazebo_live_authoritative_feedback}, "
                            f"semantic_mode={gazebo_live_semantic_mode}, "
                            f"contact_marks_collision={gazebo_live_contact_marks_collision}, "
                            f"agent_prefix={gazebo_live_client.agent_prefix}, "
                            f"feedback_vel={gazebo_live_client.state_velocity_mode}, "
                            f"feedback_acc={getattr(gazebo_live_client, 'state_acceleration_mode', None)}"
                        )
            except Exception as live_sync_err:
                gazebo_live_sync_error = str(live_sync_err)
                if _env_flag("GAZEBO_LIVE_REQUIRED", False):
                    raise
                gazebo_live_client = None
                if not quiet_output:
                    print(f"⚠️ Gazebo live sync 启动失败，将继续普通评估: {gazebo_live_sync_error}")
        
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
            print(f"   - apf_backend={apf_backend}")
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
            gazebo_apf_state = None
            gazebo_apf_base_obs = None
            gazebo_apf_result = None
            gazebo_apf_comparison_records = []
            original_corrected_head_np = None
            original_corrected_full_np = None
            original_pf_forces_np = None
            if gazebo_apf_enabled and gazebo_apf_provider is not None:
                try:
                    if gazebo_live_client is not None:
                        gazebo_apf_state = gazebo_apf_provider.read_live_state(
                            client=gazebo_live_client,
                            agents=self.env.agents,
                            timeout=max(0.0, min(gazebo_live_state_feedback_timeout, 0.05)),
                            require_seen=True,
                        )
                    if gazebo_apf_state is None:
                        gazebo_apf_state = gazebo_apf_provider.build_state(
                            agents=self.env.agents,
                            source="python_fallback_before_gazebo_feedback",
                        )
                    if gazebo_apf_state is not None:
                        gazebo_apf_provider.apply_state_to_agents(gazebo_apf_state, self.env.agents)
                        try:
                            _clear_eval_observation_cache(self.env)
                        except Exception:
                            pass
                        gazebo_apf_base_obs = gazebo_apf_provider.build_observations(gazebo_apf_state)
                        processed_obs = self._process_observations_for_eval(gazebo_apf_base_obs)
                except Exception as state_err:
                    gazebo_apf_error = str(state_err)
                    if _env_flag("GAZEBO_APF_REQUIRED", True):
                        raise
                    gazebo_apf_state = None
                    gazebo_apf_base_obs = None

            if gazebo_apf_enabled and gazebo_apf_calculator is not None and gazebo_apf_state is not None:
                base_obs_dim = self._get_base_obs_dim()
                base_obs_for_policy = processed_obs[:, :base_obs_dim] if processed_obs.shape[1] > base_obs_dim else processed_obs
                if use_pf:
                    try:
                        gazebo_pf_features = gazebo_apf_calculator.compute_base_pf_features(
                            gazebo_apf_state,
                            action_dim=self.action_dims[0] if self.action_dims else 7,
                            force_ratio=action_force_ratio,
                            observations=base_obs_for_policy,
                        )
                        pf_feature_dim = self._get_pf_feature_dim()
                        if gazebo_pf_features.shape[1] < pf_feature_dim:
                            pad = np.zeros((gazebo_pf_features.shape[0], pf_feature_dim - gazebo_pf_features.shape[1]), dtype=np.float32)
                            gazebo_pf_features = np.concatenate([gazebo_pf_features, pad], axis=1)
                        policy_obs = np.concatenate(
                            [base_obs_for_policy, gazebo_pf_features[:, :pf_feature_dim]],
                            axis=1,
                        ).astype(np.float32, copy=False)
                    except Exception:
                        if _env_flag("GAZEBO_APF_REQUIRED", True):
                            raise
                        policy_obs = self._build_eval_actor_obs(
                            processed_obs,
                            use_pf=use_pf,
                            use_tf_potential_field=use_tf_potential_field,
                            action_force_ratio=action_force_ratio,
                        )
                else:
                    policy_obs = base_obs_for_policy
            else:
                policy_obs = self._build_eval_actor_obs(
                    processed_obs,
                    use_pf=use_pf,
                    use_tf_potential_field=use_tf_potential_field,
                    action_force_ratio=action_force_ratio,
                )
            raw_actions_tf = self.select_actions_eval(policy_obs, use_fr=use_fr, use_pf=use_pf)
            raw_actions_tf = self._apply_eval_action_noise(raw_actions_tf, eval_noise_streams)
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
                pf_forces = None
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

                # The original APF is still called here for default backend and
                # for gazebo_apf consistency measurement. It is not modified.
                corrected_head_tf, pf_forces_tf = self.maddpg._apply_potential_field_correction(
                    raw_actions_tf_for_correction, processed_obs_tf, action_force_ratio
                )
                try:
                    original_pf_forces_np = pf_forces_tf.numpy()
                except Exception:
                    original_pf_forces_np = None
                try:
                    original_corrected_head_np = corrected_head_tf.numpy()
                    if raw_actions.shape[1] > 3:
                        original_corrected_full_np = np.concatenate(
                            [original_corrected_head_np, raw_actions[:, 3:]],
                            axis=1,
                        )
                    else:
                        original_corrected_full_np = original_corrected_head_np
                except Exception:
                    original_corrected_head_np = None
                    original_corrected_full_np = None

                if (
                    gazebo_apf_enabled
                    and gazebo_apf_calculator is not None
                    and gazebo_apf_provider is not None
                    and gazebo_apf_state is not None
                ):
                    try:
                        gazebo_apf_result = gazebo_apf_calculator.correct_actions(
                            raw_actions,
                            gazebo_apf_state,
                            force_ratio=action_force_ratio,
                            observations=base_obs_for_correction,
                        )
                        actions = gazebo_apf_result.corrected_actions
                        pf_forces = gazebo_apf_result.pf_forces

                        if (
                            gazebo_apf_validator is not None
                            and original_pf_forces_np is not None
                            and original_corrected_full_np is not None
                        ):
                            comparison = gazebo_apf_validator.compare(
                                original_corrected_actions=original_corrected_full_np,
                                original_pf_forces=original_pf_forces_np,
                                gazebo_result=gazebo_apf_result,
                                state=gazebo_apf_state,
                                state_provider=gazebo_apf_provider,
                                raw_actions=raw_actions,
                                original_observations=base_obs_for_correction,
                                gazebo_observations=gazebo_apf_result.observations,
                                episode=episode_idx,
                                step=step,
                                seed=(
                                    getattr(self.scenario, "current_terrain_seed", None)
                                    if hasattr(self, "scenario")
                                    else None
                                ),
                                force_ratio=action_force_ratio,
                            )
                            gazebo_apf_comparison_records = comparison.get("records", [])
                    except Exception as apf_step_err:
                        gazebo_apf_error = str(apf_step_err)
                        if _env_flag("GAZEBO_APF_REQUIRED", True):
                            raise
                        # Fallback preserves existing behavior if the new backend is optional.
                        pf_forces = original_pf_forces_np
                        action_dim = int(raw_actions.shape[1]) if hasattr(raw_actions, "shape") and len(raw_actions.shape) > 1 else 0
                        if action_dim > 3:
                            corrected_actions_tf = tf.concat([corrected_head_tf, raw_actions_tf_for_correction[:, 3:]], axis=1)
                        else:
                            corrected_actions_tf = corrected_head_tf
                        actions = corrected_actions_tf.numpy()
                else:
                    pf_forces = original_pf_forces_np
                    # 获取action_dim（从tensor形状推断）
                    action_dim = int(raw_actions.shape[1]) if hasattr(raw_actions, "shape") and len(raw_actions.shape) > 1 else 0
                    if action_dim > 3:
                        corrected_actions_tf = tf.concat([corrected_head_tf, raw_actions_tf_for_correction[:, 3:]], axis=1)
                    else:
                        corrected_actions_tf = corrected_head_tf

                    # 🔧 性能优化：延迟numpy转换，只在env.step需要时转换
                    actions = corrected_actions_tf.numpy()  # env.step需要numpy数组
            else:
                pf_forces = None
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

            semantic_filter_context = {}
            if gazebo_apf_enabled and gazebo_apf_calculator is not None and gazebo_apf_state is not None:
                try:
                    raw_actor_accel_np = gazebo_apf_calculator._normalized_action_to_acceleration(
                        np.asarray(raw_actions, dtype=np.float64)[:, :3],
                        gazebo_apf_state,
                    )
                    shadow_raw_cmd_vel_np = gazebo_apf_calculator._acceleration_to_cmd_vel(
                        raw_actor_accel_np,
                        gazebo_apf_state,
                    )
                    corrected_accel_np = (
                        np.asarray(gazebo_apf_result.corrected_accelerations, dtype=np.float64)
                        if gazebo_apf_result is not None
                        else gazebo_apf_calculator._normalized_action_to_acceleration(
                            np.asarray(actions, dtype=np.float64)[:, :3],
                            gazebo_apf_state,
                        )
                    )
                    semantic_filter_context = {
                        "raw_actions": raw_actions,
                        "corrected_actions": actions,
                        "raw_actor_accel": raw_actor_accel_np,
                        "corrected_accel": corrected_accel_np,
                        "shadow_raw_cmd_vel": shadow_raw_cmd_vel_np,
                    }
                except Exception as semantic_context_err:
                    gazebo_velocity_filter_error = str(semantic_context_err)
                    if _env_flag("GAZEBO_LIVE_OBSTACLE_SAFETY_REQUIRED", False):
                        raise
                
            authoritative_reward_snapshot = (
                _snapshot_gazebo_live_reward_state(self.env)
                if gazebo_live_authoritative_feedback
                else None
            )

            # 执行动作；velocity live 模式下这一步提供策略动作到速度命令的同源映射，
            # 随后的 Gazebo 状态反馈会覆盖 agent state，并重算本步输出。
            step_result = self.env.step(actions)
            if len(step_result) == 4:
                next_obs_n, rew_n, done_n, info_n = step_result
            elif len(step_result) == 5:
                next_obs_n, rew_n, terminated, truncated, info_n = step_result
                done_n = [t or tr for t, tr in zip(terminated, truncated)]
            else:
                raise ValueError(f"意外的环境step返回值: {len(step_result)}")

            step_count += 1
            gazebo_state_feedback_applied = False
            validation_python_pose = _capture_agent_positions(self.env.agents) if validation_active else []
            validation_nominal_cmd_vel = _capture_agent_velocities(self.env.agents)
            validation_cmd_vel = validation_nominal_cmd_vel
            validation_gazebo_pose = None
            gazebo_velocity_filter_step_records = []
            if gazebo_velocity_filter is not None:
                try:
                    can_filter_velocity = (
                        gazebo_live_control_mode == "velocity"
                        and not gazebo_live_python_authoritative
                        and gazebo_apf_state is not None
                    )
                    if can_filter_velocity:
                        if not isinstance(semantic_filter_context, dict):
                            semantic_filter_context = {}
                        semantic_filter_context = dict(semantic_filter_context)
                        semantic_filter_context.update({
                            "step": int(step_count),
                            "episode_length": int(episode_length),
                            "simulation_dt": float(simulation_dt),
                        })
                        filter_result = gazebo_velocity_filter.apply(
                            gazebo_apf_state,
                            validation_nominal_cmd_vel,
                            diagnostics_context=semantic_filter_context,
                        )
                        gazebo_velocity_filter_step_records = []
                        for record in filter_result.records:
                            if not isinstance(record, dict):
                                continue
                            item = dict(record)
                            item["episode"] = int(episode_idx)
                            item["step"] = int(step_count)
                            item["hard_contact_seen_before_step"] = bool(gazebo_first_contact_step is not None)
                            gazebo_velocity_filter_step_records.append(_json_safe_eval_value(item))
                        gazebo_velocity_filter_records.extend(gazebo_velocity_filter_step_records)
                        if gazebo_velocity_filter_enabled:
                            _apply_agent_velocity_frame(self.env.agents, filter_result.final_cmd_vel)
                        validation_cmd_vel = _capture_agent_velocities(self.env.agents)
                    elif gazebo_velocity_filter_enabled:
                        gazebo_velocity_filter_error = (
                            "obstacle velocity safety filter requires Gazebo-authoritative velocity control "
                            "and a Gazebo APF state"
                        )
                        if _env_flag("GAZEBO_LIVE_OBSTACLE_SAFETY_REQUIRED", True):
                            raise RuntimeError(gazebo_velocity_filter_error)
                except Exception as filter_step_err:
                    gazebo_velocity_filter_error = str(filter_step_err)
                    if _env_flag("GAZEBO_LIVE_OBSTACLE_SAFETY_REQUIRED", gazebo_velocity_filter_enabled):
                        raise
                    validation_cmd_vel = validation_nominal_cmd_vel

            if gazebo_live_client is not None and (step_count % gazebo_live_sync_every == 0):
                try:
                    if gazebo_live_control_mode == "velocity":
                        gazebo_live_client.send_velocity_agents(self.env.agents)
                    else:
                        gazebo_live_client.send_agents(self.env.agents)
                    if gazebo_live_command_sleep > 0.0:
                        time.sleep(gazebo_live_command_sleep)
                    if gazebo_live_control_mode == "velocity" and gazebo_live_state_feedback:
                        if gazebo_live_python_authoritative:
                            min_frame = getattr(gazebo_live_client, "_pending_state_min_frame", None)
                            state_data = gazebo_live_client.read_state(
                                min_frame=min_frame,
                                timeout=gazebo_live_state_feedback_timeout,
                            )
                            gazebo_positions = _positions_from_gazebo_state_data(
                                state_data,
                                len(getattr(self.env, "agents", []) or []),
                            )
                            if gazebo_positions is not None:
                                gazebo_live_state_feedback_updates += 1
                                if validation_active:
                                    validation_gazebo_pose = gazebo_positions
                                try:
                                    frame = int(state_data.get("frame", -1))
                                    gazebo_live_client._last_state_frame = frame
                                    gazebo_live_client._last_state_positions = np.asarray(gazebo_positions, dtype=np.float64)
                                    if getattr(gazebo_live_client, "_pending_state_min_frame", None) is not None and frame >= int(gazebo_live_client._pending_state_min_frame):
                                        gazebo_live_client._pending_state_min_frame = None
                                except Exception:
                                    pass
                            else:
                                gazebo_live_state_feedback_misses += 1
                                if _env_flag("GAZEBO_LIVE_STATE_FEEDBACK_REQUIRED", False):
                                    raise RuntimeError("Gazebo live state feedback did not arrive")
                            if gazebo_live_pose_correction:
                                gazebo_live_client.send_agents(self.env.agents)
                        else:
                            state_ok = gazebo_live_client.apply_state_to_agents(
                                self.env.agents,
                                timeout=gazebo_live_state_feedback_timeout,
                            )
                            if state_ok:
                                gazebo_state_feedback_applied = True
                                gazebo_live_state_feedback_updates += 1
                                if validation_active:
                                    validation_gazebo_pose = _capture_agent_positions(self.env.agents)
                                if gazebo_live_authoritative_feedback and authoritative_reward_snapshot is not None:
                                    _restore_gazebo_live_reward_state(self.env, authoritative_reward_snapshot)
                                try:
                                    _clear_eval_observation_cache(self.env)
                                    next_obs_n = self.env._get_obs_batch(self.env.agents)
                                except Exception:
                                    pass
                            else:
                                gazebo_live_state_feedback_misses += 1
                                if _env_flag("GAZEBO_LIVE_STATE_FEEDBACK_REQUIRED", False):
                                    raise RuntimeError("Gazebo live state feedback did not arrive")
                except Exception as live_sync_err:
                    gazebo_live_sync_error = str(live_sync_err)
                    if _env_flag("GAZEBO_LIVE_REQUIRED", False):
                        raise
                    try:
                        gazebo_live_client.close()
                    except Exception:
                        pass
                    gazebo_live_client = None
                    gazebo_live_sync_active = False
                    if not quiet_output:
                        tqdm.write(
                            f"⚠️ Gazebo live sync 中断，将继续普通评估: {gazebo_live_sync_error}",
                            file=tqdm_file,
                        )

            if gazebo_live_client is not None:
                try:
                    if gazebo_live_client.contact_detected():
                        gazebo_contact_raw_flag_count += 1
                        contact_pairs = []
                        try:
                            contact_pairs = gazebo_live_client.contact_pairs()
                        except Exception:
                            contact_pairs = []
                        for pair in contact_pairs or []:
                            pair_class = _gazebo_contact_pair_class(pair)
                            gazebo_contact_pair_class_counts[pair_class] = int(
                                gazebo_contact_pair_class_counts.get(pair_class, 0) or 0
                            ) + 1
                        real_pairs = _real_gazebo_contact_pairs(
                            contact_pairs,
                            agent_prefix=getattr(gazebo_live_client, "agent_prefix", None),
                        )
                        if not real_pairs:
                            gazebo_contact_false_positive_count += 1
                            try:
                                gazebo_live_client.clear_contact_flag()
                            except Exception:
                                pass
                        else:
                            gazebo_contact_detected = True
                            gazebo_contact_count += 1
                            gazebo_contact_pairs.extend(real_pairs)
                            gazebo_real_contact_pairs.extend(real_pairs)
                            contact_agents = sorted(
                                {
                                    int(pair["agent_id"])
                                    for pair in real_pairs
                                    if pair.get("agent_id") is not None
                                }
                            )
                            if not contact_agents:
                                contact_agents = gazebo_live_client.contact_agent_indices()
                            if not contact_agents:
                                contact_agents = list(range(len(getattr(self.env, "agents", []) or [])))
                            gazebo_contact_agent_indices = sorted(set(gazebo_contact_agent_indices).union(contact_agents))
                            if gazebo_contact_step is None:
                                gazebo_contact_step = int(step_count)
                                gazebo_first_contact_step = int(step_count)
                                gazebo_first_contact_pair = real_pairs[0] if real_pairs else None
                                gazebo_first_contact_agent_id = int(contact_agents[0]) if contact_agents else None
                                gazebo_first_contact_pending_snapshot = True

                            classes_by_agent = {}
                            for pair in real_pairs:
                                agent_idx = pair.get("agent_id")
                                if agent_idx is None:
                                    continue
                                try:
                                    classes_by_agent.setdefault(int(agent_idx), set()).add(str(pair.get("class") or ""))
                                except Exception:
                                    continue
                            if gazebo_live_contact_marks_collision:
                                for contact_agent_idx in contact_agents:
                                    try:
                                        agent = self.env.agents[contact_agent_idx]
                                    except Exception:
                                        continue
                                    _mark_gazebo_contact_collision_on_agent(
                                        agent,
                                        classes_by_agent.get(int(contact_agent_idx), []),
                                        step_count=step_count,
                                    )
                            if gazebo_live_contact_terminates and len(done_n) > 0:
                                for done_idx in range(len(done_n)):
                                    done_n[done_idx] = True
                                try:
                                    if hasattr(self.env, "world"):
                                        self.env.world._episode_done_reason = "gazebo_hard_contact"
                                except Exception:
                                    pass
                            try:
                                if hasattr(self.env, "_sync_world_team_success_snapshot"):
                                    self.env._sync_world_team_success_snapshot()
                                if gazebo_live_contact_terminates and hasattr(self.env, "world"):
                                    self.env.world._episode_done_reason = "gazebo_hard_contact"
                            except Exception:
                                pass
                            try:
                                gazebo_live_client.clear_contact_flag()
                            except Exception:
                                pass
                            if not quiet_output:
                                tqdm.write(
                                    f"⚠️ Gazebo hard contact at step {step_count}: "
                                    f"agents={contact_agents}, pair={gazebo_first_contact_pair}",
                                    file=tqdm_file,
                                )
                except Exception as contact_err:
                    gazebo_live_sync_error = str(contact_err)
                    if _env_flag("GAZEBO_LIVE_REQUIRED", False):
                        raise

            if gazebo_live_authoritative_feedback and gazebo_state_feedback_applied:
                try:
                    next_obs_n, rew_n, done_n, info_n = _recompute_step_outputs_from_current_state(
                        self.env,
                        fallback_info_n=info_n,
                    )
                    if gazebo_contact_detected and gazebo_live_contact_terminates and len(done_n) > 0:
                        done_n = [True] * len(done_n)
                        try:
                            if hasattr(self.env, "world"):
                                self.env.world._episode_done_reason = "gazebo_hard_contact"
                        except Exception:
                            pass
                    gazebo_live_authoritative_feedback_updates += 1
                except Exception as authoritative_err:
                    gazebo_live_authoritative_feedback_errors += 1
                    gazebo_live_sync_error = str(authoritative_err)
                    if _env_flag("GAZEBO_LIVE_REQUIRED", False) or _env_flag("GAZEBO_LIVE_AUTHORITATIVE_FEEDBACK_REQUIRED", False):
                        raise
            elif gazebo_live_authoritative_feedback and not gazebo_state_feedback_applied:
                if _env_flag("GAZEBO_LIVE_AUTHORITATIVE_FEEDBACK_REQUIRED", False):
                    raise RuntimeError("Gazebo authoritative feedback was requested but no Gazebo state was applied")

            # 累计奖励口径与训练保持一致：
            # 训练按“当前步所有智能体奖励的均值”累计，而不是直接对智能体求和。
            step_increment = _mean_step_reward_increment(rew_n)
            episode_reward += step_increment
            _accumulate_eval_step_diagnostics(
                eval_diagnostics,
                self.env,
                actions=actions,
                raw_actions=raw_actions,
                pf_forces=pf_forces,
                action_force_ratio=action_force_ratio,
            )
            if save_gazebo_dynamic_replay:
                episode_dynamic_state_history.append(_capture_agent_dynamic_states(self.env.agents))
                episode_dynamic_time_history.append(float(step_count * simulation_dt))
                episode_dynamic_step_indices.append(int(step_count))
                episode_dynamic_raw_action_history.append(_copy_action_frame(raw_actions))
                episode_dynamic_executed_action_history.append(_copy_action_frame(actions))
                episode_dynamic_pf_force_history.append(_copy_action_frame(pf_forces))

            current_positions = _capture_agent_positions(self.env.agents)
            current_feedback_velocities = _capture_agent_velocities(self.env.agents)
            if gazebo_first_contact_pending_snapshot:
                try:
                    idx = gazebo_first_contact_agent_id
                    if idx is not None and 0 <= int(idx) < len(current_positions):
                        pos = current_positions[int(idx)]
                        gazebo_first_contact_position = (
                            pos.astype(float).tolist() if pos is not None else None
                        )
                except Exception:
                    gazebo_first_contact_position = None
                gazebo_first_contact_pending_snapshot = False
            if validation_active:
                try:
                    from gazebo_live_validation import capture_pose_metrics
                    validation_metrics = capture_pose_metrics(
                        scenario=self.scenario,
                        positions=current_positions,
                        goals=agent_goal_positions,
                    )
                except Exception as validation_metric_err:
                    validation_metrics = {"error": str(validation_metric_err)}
                validation_trace_tail.append(
                    {
                        "step": int(step_count),
                        "actions": _copy_action_frame(actions),
                        "raw_actions": _copy_action_frame(raw_actions),
                        "nominal_cmd_vel": _vec_frames_to_lists(validation_nominal_cmd_vel),
                        "cmd_vel": _vec_frames_to_lists(validation_cmd_vel),
                        "feedback_vel": _vec_frames_to_lists(current_feedback_velocities),
                        "python_pose": _vec_frames_to_lists(validation_python_pose),
                        "gazebo_pose": _vec_frames_to_lists(
                            validation_gazebo_pose if validation_gazebo_pose is not None else current_positions
                        ),
                        "safety_filter": _json_safe_eval_value(gazebo_velocity_filter_step_records),
                        "metrics": validation_metrics,
                    }
                )
            if gazebo_apf_enabled:
                try:
                    debug_pose = validation_gazebo_pose if validation_gazebo_pose is not None else current_positions
                    debug_frame = _apf_step_debug_frame(
                        step_count=step_count,
                        raw_actions=raw_actions,
                        original_corrected=original_corrected_full_np,
                        original_pf=original_pf_forces_np,
                        gazebo_result=gazebo_apf_result,
                        comparison_records=gazebo_apf_comparison_records,
                        cmd_vel=validation_cmd_vel,
                        gazebo_pose=debug_pose,
                        nominal_cmd_vel=validation_nominal_cmd_vel,
                        safety_filter_records=gazebo_velocity_filter_step_records,
                    )
                    gazebo_contact_debug_tail.append(debug_frame)
                    debug_window_end = (
                        gazebo_first_contact_step + gazebo_contact_debug_post_steps
                        if gazebo_first_contact_step is not None
                        else None
                    )
                    if int(step_count) >= int(gazebo_contact_debug_window_start) and (
                        debug_window_end is None or int(step_count) <= int(debug_window_end)
                    ):
                        gazebo_contact_debug_records.append(debug_frame)
                except Exception as debug_frame_err:
                    gazebo_apf_error = str(debug_frame_err)
                    if _env_flag("GAZEBO_APF_REQUIRED", True):
                        raise
            if gazebo_apf_enabled and gazebo_apf_validator is not None and gazebo_apf_result is not None and gazebo_apf_state is not None:
                try:
                    gazebo_apf_validator.record_live_step(
                        gazebo_result=gazebo_apf_result,
                        state=gazebo_apf_state,
                        episode=episode_idx,
                        step=step_count,
                        comparison_records=gazebo_apf_comparison_records,
                        nominal_cmd_vel=validation_nominal_cmd_vel,
                        sent_cmd_vel=validation_cmd_vel,
                        feedback_velocities=current_feedback_velocities,
                        safety_filter_records=gazebo_velocity_filter_step_records,
                    )
                except Exception as live_metric_err:
                    gazebo_apf_error = str(live_metric_err)
                    if _env_flag("GAZEBO_APF_REQUIRED", True):
                        raise
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

            try:
                pair_count_step, per_agent_step_counts, min_clearance_step = _compute_inter_agent_collision_snapshot(
                    getattr(self.env, 'agents', [])
                )
                episode_inter_agent_collision_pair_count += int(pair_count_step)
                if min_clearance_step is not None:
                    if (
                        episode_min_inter_agent_clearance is None
                        or min_clearance_step < episode_min_inter_agent_clearance
                    ):
                        episode_min_inter_agent_clearance = float(min_clearance_step)
                if len(episode_inter_agent_collision_counts) < len(per_agent_step_counts):
                    episode_inter_agent_collision_counts.extend(
                        [0] * (len(per_agent_step_counts) - len(episode_inter_agent_collision_counts))
                    )
                for idx, per_agent_count in enumerate(per_agent_step_counts):
                    episode_inter_agent_collision_counts[idx] += int(per_agent_count)
            except Exception:
                pass

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

        if gazebo_apf_validator is not None:
            try:
                gazebo_apf_metrics = gazebo_apf_validator.finalize(first_contact_step=gazebo_first_contact_step)
            except Exception as apf_finalize_err:
                gazebo_apf_error = str(apf_finalize_err)
                if _env_flag("GAZEBO_APF_REQUIRED", True):
                    raise

        if gazebo_live_client is not None:
            try:
                gazebo_live_client_stats = {
                    "semantic_mode": gazebo_live_semantic_mode,
                    "feedback_velocity_mode": getattr(gazebo_live_client, "state_velocity_mode", None),
                    "feedback_acceleration_mode": getattr(gazebo_live_client, "state_acceleration_mode", None),
                    "pose_jump_reject_count": int(getattr(gazebo_live_client, "pose_jump_reject_count", 0) or 0),
                    "max_pose_jump_observed": float(getattr(gazebo_live_client, "max_pose_jump_observed", 0.0) or 0.0),
                    "max_feedback_speed_observed": float(getattr(gazebo_live_client, "max_feedback_speed_observed", 0.0) or 0.0),
                    "max_feedback_accel_observed": float(getattr(gazebo_live_client, "max_feedback_accel_observed", 0.0) or 0.0),
                    "cmd_vel_publish_count": int(getattr(gazebo_live_client, "sent_twist_frames", 0) or 0),
                    "pose_publish_count": int(getattr(gazebo_live_client, "sent_frames", 0) or 0),
                    "bridge_ack_enabled": bool(getattr(gazebo_live_client, "wait_ack", False)),
                    "bridge_ack_count": int(getattr(gazebo_live_client, "ack_count", 0) or 0),
                    "bridge_ack_timeout_count": int(getattr(gazebo_live_client, "ack_timeout_count", 0) or 0),
                    "bridge_last_ack_state_frame": getattr(gazebo_live_client, "last_ack_state_frame", None),
                    "pre_step_sleep_ms": int(getattr(gazebo_live_client, "pre_step_sleep_ms", 0) or 0),
                    "post_step_sleep_ms": int(getattr(gazebo_live_client, "post_step_sleep_ms", 0) or 0),
                    "wall_time_step_ms": int(getattr(gazebo_live_client, "wall_time_step_ms", 0) or 0),
                    "pause_for_step": bool(getattr(gazebo_live_client, "pause_for_step", False)),
                    "world_name": getattr(gazebo_live_client, "world", None),
                    "agent_prefix": getattr(gazebo_live_client, "agent_prefix", None),
                    "bridge_running": bool(gazebo_live_client.is_running()),
                    "state_file": str(getattr(gazebo_live_client, "state_file", "")) or None,
                    "contact_flag_file": str(getattr(gazebo_live_client, "contact_flag_file", "")) or None,
                    "contact_feedback_enabled": bool(getattr(gazebo_live_client, "contact_feedback", False)),
                    "contact_feedback_armed": bool(gazebo_live_client.contact_feedback_armed()),
                    "contact_topics": gazebo_live_client.expected_contact_topics(),
                }
                gazebo_live_client.close()
            except Exception as live_sync_close_err:
                gazebo_live_sync_error = str(live_sync_close_err)
                if _env_flag("GAZEBO_LIVE_REQUIRED", False):
                    raise
            finally:
                gazebo_live_client = None
        if isinstance(gazebo_live_launch, dict) and gazebo_live_launch.get("_process") is not None:
            if not _env_flag("GAZEBO_LIVE_KEEP_ALIVE", False):
                _terminate_process_group(gazebo_live_launch.get("_gui_process"))
                _terminate_process_group(gazebo_live_launch.get("_process"))
        
        # 计算回合统计
        episode_duration = time.time() - start_time
        avg_step_time = episode_duration / step_count if step_count > 0 else 0
        
        # 🔧 新增：收集碰撞次数和最小净空距离（与训练脚本一致）
        # 🚨 关键修复：确保使用与训练时相同的统计方式
        # 训练时使用 agent.current_episode_collision_count 和 agent.debug_info['total_penetration_count']
        # 两者应该同步更新，但优先使用 debug_info['total_penetration_count']（与训练脚本一致）
        episode_collision_counts = []
        episode_min_distances = []
        episode_terrain_total = 0
        episode_obstacle_total = 0
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

                    terrain_collision_count = 0
                    obstacle_collision_count = 0
                    try:
                        if hasattr(agent, 'debug_info') and isinstance(agent.debug_info, dict):
                            terrain_collision_count = agent.debug_info.get('terrain_penetration_count', 0)
                            obstacle_collision_count = agent.debug_info.get('obstacle_collision_count', 0)
                        terrain_collision_count = (
                            int(terrain_collision_count) if np.isfinite(terrain_collision_count) else 0
                        )
                        obstacle_collision_count = (
                            int(obstacle_collision_count) if np.isfinite(obstacle_collision_count) else 0
                        )
                    except (ValueError, TypeError, OverflowError):
                        terrain_collision_count = 0
                        obstacle_collision_count = 0
                    episode_terrain_total += terrain_collision_count
                    episode_obstacle_total += obstacle_collision_count
                    
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
        agent_safe_flags = []
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
                    agent_safe_flags.append(1 if safe_i else 0)
                    agent_success_flags.append(succ_i)
                    
                    if succ_i == 0:
                        team_success_flag = 0
        except Exception as e:
            if not os.getenv("QUIET_OUTPUT", "1").lower() in ("1", "true", "yes", "on"):
                print(f"⚠️  计算成功标志时出错: {e}")
            agent_success_flags = [0] * len(episode_collision_counts) if episode_collision_counts else []
            agent_safe_flags = [0] * len(episode_collision_counts) if episode_collision_counts else []
            team_success_flag = 0

        world_snapshot = _get_world_success_snapshot(self.env, expected_count=len(episode_collision_counts))
        if world_snapshot is not None:
            agent_safe_flags = world_snapshot['safe_flags']
            agent_success_flags = world_snapshot['success_flags']
            team_success_flag = world_snapshot['team_success']
        
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

        eval_diag = _finalize_eval_diagnostics(eval_diagnostics)
        reward_decomposition = eval_diag['reward_components']
        diagnostic_metrics = eval_diag['diagnostic_metrics']
        terminal_success_bonus_applied = bool(
            abs(float(reward_decomposition.get('reward_terminal_success', 0.0) or 0.0)) > 1e-8
        )
        episode_done_reason = _infer_episode_done_reason(
            success_flag,
            first_reach_step,
            step_count,
            episode_length,
            total_collisions,
        )
        if world_snapshot is not None and world_snapshot.get('done_reason'):
            episode_done_reason = world_snapshot['done_reason']
        if gazebo_first_contact_step is not None and gazebo_live_contact_terminates:
            episode_done_reason = "gazebo_hard_contact"
        
        if not quiet_output:
            print(f"✅ 回合 {episode_idx + 1} 完成:")
            print(f"   - 奖励: {episode_reward:.2f}")
            print(f"   - 步数: {step_count}/{episode_length} (完成度: {step_count/episode_length*100:.1f}%)")
            print(f"   - 用时: {episode_duration:.2f}秒")
            print(f"   - 平均步时: {avg_step_time:.4f}秒/步")
            print(
                f"   - 碰撞次数: {total_collisions} "
                f"(地形={episode_terrain_total}, 障碍物={episode_obstacle_total}; "
                f"智能体: {episode_collision_counts})"
            )
            print(
                f"   - 队间碰撞: 成对计数={episode_inter_agent_collision_pair_count} "
                f"(智能体: {episode_inter_agent_collision_counts})"
            )
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
            if episode_min_inter_agent_clearance is not None:
                print(f"   - 最小队间净空: {episode_min_inter_agent_clearance:.2f}")
            if penetration_stat is not None:
                print(f"   - 穿透统计: 次数={penetration_stat['count']}, 最大深度={penetration_stat['max_depth']:.2f}, 平均深度={penetration_stat['mean_depth']:.2f}")
            if step_count < episode_length:
                print(f"   ⚠️  注意: 回合提前结束（可能由于done=True或提前终止）")

        vis_context = self._capture_episode_vis_context(episode_idx)
        
        gazebo_live_cmd_vel_publish_count = int(gazebo_live_client_stats.get("cmd_vel_publish_count", 0) or 0)
        gazebo_live_feedback_update_ratio = float(
            gazebo_live_authoritative_feedback_updates / max(gazebo_live_cmd_vel_publish_count, 1)
        )
        gazebo_live_state_feedback_update_ratio = float(
            gazebo_live_state_feedback_updates / max(gazebo_live_cmd_vel_publish_count, 1)
        )
        gazebo_contact_debug_window_end_requested = (
            int(gazebo_first_contact_step) + int(gazebo_contact_debug_post_steps)
            if gazebo_first_contact_step is not None
            else int(step_count) + int(gazebo_contact_debug_post_steps)
        )
        gazebo_contact_debug_window = _filter_contact_debug_window(
            gazebo_contact_debug_records,
            gazebo_contact_debug_window_start,
            gazebo_contact_debug_window_end_requested,
            agent_id=gazebo_contact_debug_agent_id,
            obstacle_name=gazebo_contact_debug_obstacle,
        )
        pre_contact_debug_window = [
            frame for frame in gazebo_contact_debug_window
            if gazebo_first_contact_step is None or int(frame.get("step", 0) or 0) <= int(gazebo_first_contact_step)
        ]
        pre_contact_surface_distance_trace = _surface_distance_trace(
            pre_contact_debug_window,
            agent_id=gazebo_contact_debug_agent_id,
            obstacle_name=gazebo_contact_debug_obstacle,
        )
        gazebo_contact_debug_window_end_available = None
        if gazebo_contact_debug_window:
            try:
                gazebo_contact_debug_window_end_available = max(
                    int(frame.get("step", 0) or 0) for frame in gazebo_contact_debug_window
                )
            except Exception:
                gazebo_contact_debug_window_end_available = None
        gazebo_contact_debug_tail_list = list(gazebo_contact_debug_tail)
        last_20_raw_actions = [frame.get("raw_actions", []) for frame in gazebo_contact_debug_tail_list]
        last_20_python_apf_outputs = [
            frame.get("python_apf_outputs", []) for frame in gazebo_contact_debug_tail_list
        ]
        last_20_gazebo_apf_outputs = [
            frame.get("gazebo_apf_outputs", []) for frame in gazebo_contact_debug_tail_list
        ]
        last_20_nominal_cmd_vel = [frame.get("nominal_cmd_vel", []) for frame in gazebo_contact_debug_tail_list]
        last_20_cmd_vel = [frame.get("cmd_vel", []) for frame in gazebo_contact_debug_tail_list]
        last_20_gazebo_pose = [frame.get("gazebo_pose", []) for frame in gazebo_contact_debug_tail_list]
        last_20_safety_filter_outputs = [
            [agent.get("safety_filter", {}) for agent in frame.get("agents", [])]
            for frame in gazebo_contact_debug_tail_list
        ]
        gazebo_contact_debug_artifacts = {}
        if gazebo_apf_validator is not None and gazebo_contact_debug_window:
            gazebo_contact_debug_artifacts = _write_contact_debug_window_artifacts(
                output_dir=gazebo_apf_validator.output_dir,
                episode_idx=episode_idx,
                debug_window=gazebo_contact_debug_window,
                first_contact_step=gazebo_first_contact_step,
                first_contact_pair=gazebo_first_contact_pair,
                agent_id=gazebo_contact_debug_agent_id,
                obstacle_name=gazebo_contact_debug_obstacle,
            )
        if gazebo_velocity_filter_records:
            try:
                from gazebo_velocity_safety_filter import (
                    summarize_velocity_filter_records,
                    write_velocity_filter_debug_artifacts,
                )

                gazebo_velocity_filter_summary = summarize_velocity_filter_records(gazebo_velocity_filter_records)
                velocity_filter_output_dir = (
                    gazebo_apf_validator.output_dir
                    if gazebo_apf_validator is not None
                    else getattr(self.args, "gazebo_apf_output_dir", "results/gazebo_apf_validation")
                )
                gazebo_velocity_filter_artifacts = write_velocity_filter_debug_artifacts(
                    output_dir=velocity_filter_output_dir,
                    episode_idx=episode_idx,
                    records=gazebo_velocity_filter_records,
                    first_contact_step=gazebo_first_contact_step,
                    hard_contact=bool(gazebo_first_contact_step is not None),
                    agent_id=gazebo_contact_debug_agent_id,
                    obstacle_name=gazebo_contact_debug_obstacle,
                    window_start=gazebo_contact_debug_window_start,
                    window_end=gazebo_contact_debug_window_end_requested,
                )
            except Exception as filter_artifact_err:
                gazebo_velocity_filter_error = str(filter_artifact_err)
                if _env_flag("GAZEBO_LIVE_OBSTACLE_SAFETY_REQUIRED", False):
                    raise

        episode_data = {
            'episode': episode_idx,
            'reward': episode_reward,
            'steps': step_count,
            'episode_length': int(episode_length),
            'trajectory': episode_trajectory if record_trajectory else [],
            'actions_history': episode_actions_history if record_actions else [],
            'executed_actions_history': episode_executed_actions_history if save_control_diagnostics else [],
            'velocity_history': episode_velocity_history if save_control_diagnostics else [],
            'goal_distance_history': episode_goal_distance_history if save_control_diagnostics else [],
            'dynamic_state_history': episode_dynamic_state_history if save_gazebo_dynamic_replay else [],
            'dynamic_time_history': episode_dynamic_time_history if save_gazebo_dynamic_replay else [],
            'dynamic_step_indices': episode_dynamic_step_indices if save_gazebo_dynamic_replay else [],
            'dynamic_raw_action_history': episode_dynamic_raw_action_history if save_gazebo_dynamic_replay else [],
            'dynamic_executed_action_history': episode_dynamic_executed_action_history if save_gazebo_dynamic_replay else [],
            'dynamic_pf_force_history': episode_dynamic_pf_force_history if save_gazebo_dynamic_replay else [],
            'gazebo_live_sync_active': bool(gazebo_live_sync_active),
            'gazebo_live_sync_error': gazebo_live_sync_error,
            'apf_backend': apf_backend,
            'gazebo_apf_enabled': bool(gazebo_apf_enabled),
            'gazebo_apf_error': gazebo_apf_error,
            'gazebo_apf_metrics': gazebo_apf_metrics,
            'gazebo_apf_adapter_metrics': (
                gazebo_apf_metrics.get('adapter_metrics')
                if isinstance(gazebo_apf_metrics, dict)
                else None
            ),
            'gazebo_obstacle_velocity_filter_mode': gazebo_velocity_filter_mode,
            'gazebo_obstacle_velocity_filter_enabled': bool(gazebo_velocity_filter_enabled),
            'gazebo_obstacle_velocity_filter_error': gazebo_velocity_filter_error,
            'gazebo_obstacle_velocity_filter_summary': gazebo_velocity_filter_summary,
            'gazebo_obstacle_velocity_filter_artifacts': gazebo_velocity_filter_artifacts,
            'gazebo_obstacle_velocity_filter_records_tail': _json_safe_eval_value(
                gazebo_velocity_filter_records[-60:]
            ),
            'gazebo_apf_validation_dir': (
                str(gazebo_apf_validator.output_dir)
                if gazebo_apf_validator is not None
                else None
            ),
            'gazebo_live_launch_error': gazebo_live_launch_error,
            'gazebo_live_autolaunched': bool(isinstance(gazebo_live_launch, dict) and gazebo_live_launch.get('autolaunch_started', False)),
            'gazebo_live_autolaunch_gui': bool(isinstance(gazebo_live_launch, dict) and gazebo_live_launch.get('autolaunch_gui', False)),
            'gazebo_live_autolaunch_run': bool(isinstance(gazebo_live_launch, dict) and gazebo_live_launch.get('autolaunch_run', False)),
            'gazebo_live_gui_error': (
                gazebo_live_launch.get('gui_error') if isinstance(gazebo_live_launch, dict) else None
            ),
            'gazebo_live_world_sdf': (
                gazebo_live_launch.get('world_live_sdf') if isinstance(gazebo_live_launch, dict) else None
            ),
            'gazebo_live_scenario_json': (
                gazebo_live_launch.get('scenario_json') if isinstance(gazebo_live_launch, dict) else None
            ),
            'gazebo_live_export_dir': (
                gazebo_live_launch.get('output_dir') if isinstance(gazebo_live_launch, dict) else None
            ),
            'gazebo_live_consistency_mode': gazebo_live_consistency_mode,
            'gazebo_live_semantic_mode': gazebo_live_client_stats.get("semantic_mode", gazebo_live_semantic_mode),
            'gazebo_live_collision_mode': (
                gazebo_live_launch.get('collision_mode')
                if isinstance(gazebo_live_launch, dict) and gazebo_live_launch.get('collision_mode')
                else gazebo_live_collision_mode
            ),
            'gazebo_live_physical_collision_enabled': bool(
                gazebo_live_launch.get('physical_collision_enabled')
                if isinstance(gazebo_live_launch, dict) and 'physical_collision_enabled' in gazebo_live_launch
                else gazebo_live_physical_collision_enabled
            ),
            'gazebo_live_python_authoritative': bool(gazebo_live_python_authoritative),
            'gazebo_live_contact_authoritative': bool(gazebo_live_contact_authoritative),
            'gazebo_live_contact_marks_collision': bool(gazebo_live_contact_marks_collision),
            'gazebo_live_contact_terminates': bool(gazebo_live_contact_terminates),
            'gazebo_live_pose_correction': bool(gazebo_live_pose_correction),
            'gazebo_live_control_mode': gazebo_live_control_mode,
            'gazebo_live_command_sleep': float(gazebo_live_command_sleep),
            'gazebo_live_step_iterations': int(gazebo_live_step_iterations),
            'gazebo_live_state_feedback': bool(gazebo_live_state_feedback),
            'gazebo_live_contact_feedback': bool(gazebo_live_client_stats.get("contact_feedback_enabled", _env_flag("GAZEBO_LIVE_CONTACT_FEEDBACK", True))),
            'gazebo_live_bridge_running': bool(gazebo_live_client_stats.get("bridge_running", False)),
            'gazebo_live_contact_feedback_armed': bool(gazebo_live_client_stats.get("contact_feedback_armed", False)),
            'gazebo_live_contact_topics': gazebo_live_client_stats.get("contact_topics", []),
            'gazebo_live_state_file': gazebo_live_client_stats.get("state_file"),
            'gazebo_live_contact_flag_file': gazebo_live_client_stats.get("contact_flag_file"),
            'gazebo_live_feedback_velocity_mode': gazebo_live_client_stats.get("feedback_velocity_mode"),
            'gazebo_live_feedback_acceleration_mode': gazebo_live_client_stats.get("feedback_acceleration_mode"),
            'gazebo_live_world_name': gazebo_live_client_stats.get("world_name") or (
                gazebo_live_launch.get("world_name") if isinstance(gazebo_live_launch, dict) else None
            ),
            'gazebo_live_agent_prefix': gazebo_live_client_stats.get("agent_prefix") or (
                gazebo_live_launch.get("agent_prefix") if isinstance(gazebo_live_launch, dict) else None
            ),
            'gazebo_live_cmd_vel_publish_count': gazebo_live_cmd_vel_publish_count,
            'gazebo_live_pose_publish_count': int(gazebo_live_client_stats.get("pose_publish_count", 0) or 0),
            'gazebo_live_bridge_ack_enabled': bool(gazebo_live_client_stats.get("bridge_ack_enabled", False)),
            'gazebo_live_bridge_ack_count': int(gazebo_live_client_stats.get("bridge_ack_count", 0) or 0),
            'gazebo_live_bridge_ack_timeout_count': int(gazebo_live_client_stats.get("bridge_ack_timeout_count", 0) or 0),
            'gazebo_live_bridge_last_ack_state_frame': gazebo_live_client_stats.get("bridge_last_ack_state_frame"),
            'gazebo_live_pre_step_sleep_ms': int(gazebo_live_client_stats.get("pre_step_sleep_ms", 0) or 0),
            'gazebo_live_post_step_sleep_ms': int(gazebo_live_client_stats.get("post_step_sleep_ms", 0) or 0),
            'gazebo_live_wall_time_step_ms': int(gazebo_live_client_stats.get("wall_time_step_ms", 0) or 0),
            'gazebo_live_pause_for_step': bool(gazebo_live_client_stats.get("pause_for_step", False)),
            'gazebo_live_pose_jump_reject_count': int(gazebo_live_client_stats.get("pose_jump_reject_count", 0) or 0),
            'gazebo_live_max_pose_jump_observed': float(gazebo_live_client_stats.get("max_pose_jump_observed", 0.0) or 0.0),
            'gazebo_live_max_feedback_speed_observed': float(gazebo_live_client_stats.get("max_feedback_speed_observed", 0.0) or 0.0),
            'gazebo_live_max_feedback_accel_observed': float(gazebo_live_client_stats.get("max_feedback_accel_observed", 0.0) or 0.0),
            'gazebo_live_state_feedback_updates': int(gazebo_live_state_feedback_updates),
            'gazebo_live_state_feedback_misses': int(gazebo_live_state_feedback_misses),
            'gazebo_live_state_feedback_update_ratio': gazebo_live_state_feedback_update_ratio,
            'gazebo_live_authoritative_feedback': bool(gazebo_live_authoritative_feedback),
            'gazebo_live_authoritative_feedback_updates': int(gazebo_live_authoritative_feedback_updates),
            'gazebo_live_authoritative_feedback_errors': int(gazebo_live_authoritative_feedback_errors),
            'gazebo_live_feedback_update_ratio': gazebo_live_feedback_update_ratio,
            'gazebo_contact_detected': bool(gazebo_contact_detected),
            'gazebo_contact_count': int(gazebo_contact_count),
            'gazebo_contact_raw_flag_count': int(gazebo_contact_raw_flag_count),
            'gazebo_contact_false_positive_count': int(gazebo_contact_false_positive_count),
            'gazebo_contact_step': int(gazebo_contact_step) if gazebo_contact_step is not None else None,
            'gazebo_first_contact_step': int(gazebo_first_contact_step) if gazebo_first_contact_step is not None else None,
            'first_contact_step': int(gazebo_first_contact_step) if gazebo_first_contact_step is not None else None,
            'first_contact_pair': gazebo_first_contact_pair,
            'first_contact_position': gazebo_first_contact_position,
            'first_contact_agent_id': (
                int(gazebo_first_contact_agent_id) if gazebo_first_contact_agent_id is not None else None
            ),
            'gazebo_contact_agent_indices': [int(v) for v in gazebo_contact_agent_indices],
            'gazebo_contact_pairs': gazebo_contact_pairs,
            'gazebo_real_contact_pairs': gazebo_real_contact_pairs,
            'gazebo_contact_pair_class_counts': gazebo_contact_pair_class_counts,
            'pre_contact_surface_distance_trace': pre_contact_surface_distance_trace,
            'last_20_raw_actions': last_20_raw_actions,
            'last_20_python_apf_outputs': last_20_python_apf_outputs,
            'last_20_gazebo_apf_outputs': last_20_gazebo_apf_outputs,
            'last_20_nominal_cmd_vel': last_20_nominal_cmd_vel,
            'last_20_cmd_vel': last_20_cmd_vel,
            'last_20_safety_filter_outputs': last_20_safety_filter_outputs,
            'last_20_gazebo_pose': last_20_gazebo_pose,
            'gazebo_contact_debug_window': gazebo_contact_debug_window,
            'gazebo_contact_debug_window_start': int(gazebo_contact_debug_window_start),
            'gazebo_contact_debug_window_end_requested': int(gazebo_contact_debug_window_end_requested),
            'gazebo_contact_debug_window_end_available': (
                int(gazebo_contact_debug_window_end_available)
                if gazebo_contact_debug_window_end_available is not None
                else None
            ),
            'gazebo_contact_debug_agent_id': int(gazebo_contact_debug_agent_id),
            'gazebo_contact_debug_obstacle': gazebo_contact_debug_obstacle,
            'gazebo_contact_debug_artifacts': gazebo_contact_debug_artifacts,
            'gazebo_live_scene_check': gazebo_live_scene_check,
            'python_scene_signature': (
                python_scene_signature_record.get('python_scene_signature')
                if isinstance(python_scene_signature_record, dict)
                else None
            ),
            'python_scene_signature_parts': (
                python_scene_signature_record.get('python_scene_signature_parts')
                if isinstance(python_scene_signature_record, dict)
                else None
            ),
            'scene_signature': (
                gazebo_live_scene_check.get('scene_signature')
                if isinstance(gazebo_live_scene_check, dict)
                else None
            ),
            'validation_trace_tail': list(validation_trace_tail) if validation_active else [],
            'duration': episode_duration,
            # 🔧 新增：返回碰撞和成功指标（与训练脚本一致）
            'collision_count': total_collisions,
            'terrain_collision_count': int(episode_terrain_total),
            'obstacle_collision_count': int(episode_obstacle_total),
            'agent_collision_counts': episode_collision_counts,
            'inter_agent_collision_count': int(episode_inter_agent_collision_pair_count),
            'agent_inter_agent_collision_counts': [int(v) for v in episode_inter_agent_collision_counts],
            'min_inter_agent_clearance': (
                float(episode_min_inter_agent_clearance) if episode_min_inter_agent_clearance is not None else None
            ),
            'min_distance': min_distance_stat,
            'success': success_flag,
            'agent_success_flags': agent_success_flags,
            'agent_safe_flags': agent_safe_flags,
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
            'reward_decomposition': reward_decomposition,
            'diagnostic_metrics': diagnostic_metrics,
            'terminal_success_bonus_applied': terminal_success_bonus_applied,
            'episode_done_reason': episode_done_reason,
            'vis_context': vis_context,
        }
        if gazebo_velocity_filter_records:
            try:
                from gazebo_velocity_safety_filter import write_progress_diagnostics_artifacts

                progress_output_dir = (
                    gazebo_apf_validator.output_dir
                    if gazebo_apf_validator is not None
                    else getattr(self.args, "gazebo_apf_output_dir", "results/gazebo_apf_validation")
                )
                gazebo_progress_artifacts = write_progress_diagnostics_artifacts(
                    output_dir=progress_output_dir,
                    episode_idx=episode_idx,
                    records=gazebo_velocity_filter_records,
                    episode_data=episode_data,
                )
                episode_data['gazebo_progress_diagnostics_artifacts'] = gazebo_progress_artifacts
                episode_data['gazebo_progress_failure_summary'] = (
                    gazebo_progress_artifacts.get('episode_progress_summary')
                    if isinstance(gazebo_progress_artifacts, dict)
                    else None
                )
                episode_data['gazebo_progress_diagnostics_csv'] = (
                    gazebo_progress_artifacts.get('gazebo_progress_diagnostics_csv')
                    if isinstance(gazebo_progress_artifacts, dict)
                    else None
                )
                episode_data['progress_failure_summary_json'] = (
                    gazebo_progress_artifacts.get('progress_failure_summary_json')
                    if isinstance(gazebo_progress_artifacts, dict)
                    else None
                )
            except Exception as progress_artifact_err:
                episode_data['gazebo_progress_diagnostics_error'] = str(progress_artifact_err)
                if _env_flag("GAZEBO_LIVE_OBSTACLE_SAFETY_REQUIRED", False):
                    raise
        if validation_active:
            try:
                from gazebo_live_validation import build_sync_health_record
                episode_data['gazebo_live_sync_health'] = build_sync_health_record(
                    episode_data=episode_data,
                    backend=getattr(self.args, 'eval_backend', os.getenv('EVAL_BACKEND', 'python_only')),
                    validation_root=validation_root,
                )
            except Exception as sync_health_err:
                episode_data['gazebo_live_sync_health_error'] = str(sync_health_err)
        return episode_data
        
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

            save_gazebo_replay = _env_flag('SAVE_GAZEBO_REPLAY', False)
            save_gazebo_dynamic_replay = _env_flag('SAVE_GAZEBO_DYNAMIC_REPLAY', False)
            save_gazebo_fast_replay = _env_flag('SAVE_GAZEBO_FAST_REPLAY', False)
            compile_gazebo_fast_replay = _env_flag('COMPILE_GAZEBO_FAST_REPLAY', save_gazebo_fast_replay)
            save_trajectory_snapshot = _env_flag('SAVE_TRAJECTORY_SNAPSHOT', save_gazebo_replay or save_gazebo_dynamic_replay)
            if save_trajectory_snapshot:
                try:
                    from pathlib import Path
                    from trajectory_snapshot_exporter import write_dynamic_replay_snapshot, write_trajectory_snapshot

                    replay_dir = (
                        os.path.join(self.args.save_viz_path, f"gazebo_replay_ep{episode_num:03d}")
                        if (save_gazebo_replay or save_gazebo_dynamic_replay)
                        else self.args.save_viz_path
                    )
                    snapshot_prefix = f"episode_{episode_num:03d}"
                    gazebo_replay_cache = _env_flag('GAZEBO_REPLAY_CACHE', True)
                    gazebo_replay_compact = _env_flag('GAZEBO_REPLAY_COMPACT', save_gazebo_replay)
                    snapshot_files = write_trajectory_snapshot(
                        output_dir=Path(replay_dir),
                        prefix=snapshot_prefix,
                        episode_data=episode_data,
                        vis_context=vis_context if isinstance(vis_context, dict) else None,
                        args=self.args,
                        generated_files=generated_files,
                        compact=gazebo_replay_compact,
                        reuse_cache=gazebo_replay_cache,
                    )
                    generated_files.update({k: v for k, v in snapshot_files.items() if v})

                    if save_gazebo_replay or save_gazebo_dynamic_replay:
                        from gazebo_terrain_exporter import export_gazebo_scene
                        from gazebo_trajectory_exporter import export_gazebo_trajectory_replay

                        scenario_json_path = snapshot_files.get('scenario_json_path')
                        trajectory_json_path = snapshot_files.get('trajectory_snapshot_path')
                        if not scenario_json_path or not trajectory_json_path:
                            raise RuntimeError("Gazebo replay export requires both scenario_json_path and trajectory_snapshot_path")

                        visual_resolution_raw = os.getenv('GAZEBO_TERRAIN_VISUAL_RESOLUTION', '').strip()
                        visual_resolution = int(visual_resolution_raw) if visual_resolution_raw else None
                        coarse_collision_resolution = _env_int('GAZEBO_COARSE_COLLISION_RESOLUTION', 80)
                        gazebo_scene = export_gazebo_scene(
                            scenario_json=Path(scenario_json_path),
                            output_dir=Path(replay_dir),
                            visual_resolution=visual_resolution,
                            coarse_collision_resolution=coarse_collision_resolution,
                            use_coarse_collision=_env_flag('GAZEBO_USE_COARSE_COLLISION', False),
                            uav_marker_scale=_env_optional_float('GAZEBO_UAV_MARKER_SCALE'),
                            show_agent_collision_envelope=not _env_flag('GAZEBO_HIDE_AGENT_COLLISION_ENVELOPE', False),
                            reuse_cache=gazebo_replay_cache,
                        )
                        gazebo_replay = export_gazebo_trajectory_replay(
                            scenario_json=Path(scenario_json_path),
                            trajectory_json=Path(trajectory_json_path),
                            output_dir=Path(replay_dir),
                            base_world_sdf=Path(gazebo_scene['world_sdf']),
                            path_stride=_env_int('GAZEBO_REPLAY_PATH_STRIDE', 2),
                            path_radius=_env_float('GAZEBO_REPLAY_PATH_RADIUS', 0.18),
                            tube_sides=_env_int('GAZEBO_REPLAY_TUBE_SIDES', 10),
                            ghost_stride=_env_int('GAZEBO_REPLAY_GHOST_STRIDE', 80),
                            ghost_radius_scale=_env_float('GAZEBO_REPLAY_GHOST_RADIUS_SCALE', 1.0),
                            reuse_cache=gazebo_replay_cache,
                        )
                        generated_files.update({
                            'gazebo_world_sdf': gazebo_scene.get('world_sdf'),
                            'gazebo_world_replay_sdf': gazebo_replay.get('world_replay_sdf'),
                            'gazebo_model_parent_dir': gazebo_replay.get('model_parent_dir'),
                            'gazebo_resource_path_command': gazebo_replay.get('resource_path_command'),
                        })
                        print(f"✅ Gazebo静态轨迹回放已生成: {gazebo_replay.get('world_replay_sdf')}")
                        if save_gazebo_dynamic_replay:
                            from gazebo_dynamic_replay_exporter import export_gazebo_dynamic_replay

                            dynamic_snapshot_files = write_dynamic_replay_snapshot(
                                output_dir=Path(replay_dir),
                                prefix=snapshot_prefix,
                                episode_data=episode_data,
                                args=self.args,
                                trajectory_snapshot_path=Path(trajectory_json_path),
                                scenario_json_path=Path(scenario_json_path),
                                compact=gazebo_replay_compact,
                                reuse_cache=gazebo_replay_cache,
                            )
                            dynamic_replay = export_gazebo_dynamic_replay(
                                scenario_json=Path(scenario_json_path),
                                trajectory_json=Path(trajectory_json_path),
                                dynamic_json=Path(dynamic_snapshot_files['dynamic_replay_json_path']),
                                output_dir=Path(replay_dir),
                                base_world_sdf=Path(gazebo_replay['world_replay_sdf']),
                                uav_marker_scale=_env_optional_float('GAZEBO_UAV_MARKER_SCALE'),
                                reuse_cache=gazebo_replay_cache,
                                build_fast_replay=save_gazebo_fast_replay or compile_gazebo_fast_replay,
                                compile_fast_replay=compile_gazebo_fast_replay,
                            )
                            generated_files.update({k: v for k, v in dynamic_snapshot_files.items() if v})
                            generated_files.update({
                                'gazebo_dynamic_world_sdf': dynamic_replay.get('world_dynamic_replay_sdf'),
                                'gazebo_dynamic_player': dynamic_replay.get('player_script'),
                                'gazebo_dynamic_replay_json': dynamic_replay.get('dynamic_replay_json'),
                                'gazebo_dynamic_replay_npz': dynamic_replay.get('dynamic_replay_npz'),
                                'gazebo_fast_replay_meta': dynamic_replay.get('fast_replay', {}).get('meta_path') if isinstance(dynamic_replay.get('fast_replay'), dict) else None,
                                'gazebo_fast_replay_executable': dynamic_replay.get('fast_replay', {}).get('executable') if isinstance(dynamic_replay.get('fast_replay'), dict) else None,
                                'gazebo_fast_replay_binary': dynamic_replay.get('fast_replay', {}).get('binary', {}).get('binary_path') if isinstance(dynamic_replay.get('fast_replay'), dict) else None,
                            })
                            print(f"✅ Gazebo动态轨迹回放已生成: {dynamic_replay.get('world_dynamic_replay_sdf')}")
                except Exception as gazebo_replay_err:
                    print(f"⚠️ Gazebo静态轨迹回放导出失败: {gazebo_replay_err}")
                    traceback.print_exc()

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
        selected_episode_visualizations = set()
        for raw_value, one_based in (
            (os.getenv('SAVE_EVAL_EPISODE_INDICES', ''), False),
            (os.getenv('SAVE_EVAL_EPISODE_NUMBERS', ''), True),
        ):
            for token in str(raw_value or '').replace(';', ',').replace(' ', ',').split(','):
                token = token.strip()
                if not token:
                    continue
                try:
                    episode_id = int(token)
                except Exception:
                    continue
                if one_based:
                    episode_id -= 1
                if episode_id >= 0:
                    selected_episode_visualizations.add(int(episode_id))

        def _should_save_episode_visualization(episode_idx):
            return save_all_episode_visualizations or int(episode_idx) in selected_episode_visualizations

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
        persist_dynamic_replay_json = _env_flag('SAVE_EVAL_DYNAMIC_REPLAY_JSON', False)
        log_episode_timing = _env_flag("EVAL_LOG_EPISODE_TIMING", True)

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
        
        terrain_level_sequence = None
        terrain_level_str = os.getenv('TERRAIN_COMPLEXITY_LEVEL_SEQUENCE', '')
        if terrain_level_str:
            try:
                terrain_level_sequence = [
                    max(1, min(4, int(s.strip())))
                    for s in terrain_level_str.split(',')
                    if s.strip()
                ]
                if not quiet_output:
                    print(
                        f"🔧 使用预定义地形复杂度序列（共{len(terrain_level_sequence)}个）: "
                        f"{terrain_level_sequence[:5]}... (前5个)"
                    )
            except Exception as e:
                print(f"⚠️  解析地形复杂度序列失败: {e}，将使用随机复杂度")
                terrain_level_sequence = None

        def _select_episode_terrain_level(episode_index):
            # An explicit evaluation sequence is the episode protocol and must
            # take precedence over the model's single training-time level.
            if terrain_level_sequence:
                return int(terrain_level_sequence[int(episode_index) % len(terrain_level_sequence)])
            if self.args.terrain_complexity_level is not None:
                return int(self.args.terrain_complexity_level)
            return int(np.random.randint(1, 5))

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
        reward_runtime = getattr(self.scenario, 'vectorized_calculator', None)

        def _actual_reward_float(attribute_name, env_name, default_value):
            value = getattr(reward_runtime, attribute_name, None) if reward_runtime is not None else None
            numeric = _finite_float_or_none(value)
            return numeric if numeric is not None else _env_float(env_name, default_value)

        def _actual_reward_bool(attribute_name, env_name, default_value):
            if reward_runtime is not None and hasattr(reward_runtime, attribute_name):
                return bool(getattr(reward_runtime, attribute_name))
            return _env_flag(env_name, default_value)

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
            'terrain_contact_eps': _env_float('TERRAIN_CONTACT_EPS', _finite_float_or_none(getattr(self.args, 'terrain_contact_eps', None)) or 0.2),
            'obstacle_observation_mode': normalize_obstacle_observation_mode(
                os.getenv(
                    'OBSTACLE_OBSERVATION_MODE',
                    os.getenv('OBSTACLE_OBS_MODE', getattr(self.args, 'obstacle_observation_mode', 'nearest_surface')),
                )
            ),
            'obstacle_risk_velocity_forward_weight': _env_float(
                'OBSTACLE_RISK_VELOCITY_FORWARD_WEIGHT',
                4.0
                if _finite_float_or_none(getattr(self.args, 'obstacle_risk_velocity_forward_weight', None)) is None
                else _finite_float_or_none(getattr(self.args, 'obstacle_risk_velocity_forward_weight', None)),
            ),
            'obstacle_risk_goal_along_weight': _env_float(
                'OBSTACLE_RISK_GOAL_ALONG_WEIGHT',
                3.0
                if _finite_float_or_none(getattr(self.args, 'obstacle_risk_goal_along_weight', None)) is None
                else _finite_float_or_none(getattr(self.args, 'obstacle_risk_goal_along_weight', None)),
            ),
            'reward_pos_scale': _finite_float_or_none(getattr(self.args, 'reward_pos_scale', None)),
            'reward_neg_scale': _finite_float_or_none(getattr(self.args, 'reward_neg_scale', None)),
            'distance_weight': _finite_float_or_none(getattr(self.args, 'distance_weight', None)),
            'exploration_weight': _finite_float_or_none(getattr(self.args, 'exploration_weight', None)),
            'stationary_weight': _finite_float_or_none(getattr(self.args, 'stationary_weight', None)),
            'direction_weight': _finite_float_or_none(getattr(self.args, 'direction_weight', None)),
            'deviation_weight': _finite_float_or_none(getattr(self.args, 'deviation_weight', None)),
            'start_area_weight': _finite_float_or_none(getattr(self.args, 'start_area_weight', None)),
            'approach_weight': _finite_float_or_none(getattr(self.args, 'approach_weight', None)),
            'energy_weight': _finite_float_or_none(getattr(self.args, 'energy_weight', None)),
            'height_weight': _finite_float_or_none(getattr(self.args, 'height_weight', None)),
            'height_reward_enabled': bool(getattr(self.args, 'height_reward_enabled', True)),
            'height_ideal_min': _finite_float_or_none(getattr(self.args, 'height_ideal_min', None)),
            'height_ideal_max': _finite_float_or_none(getattr(self.args, 'height_ideal_max', None)),
            'lateral_weight': _finite_float_or_none(getattr(self.args, 'lateral_weight', None)),
            'clearance_weight': _finite_float_or_none(getattr(self.args, 'clearance_weight', None)),
            'clearance_d_max': _finite_float_or_none(getattr(self.args, 'clearance_d_max', None)),
            'success_weight': _finite_float_or_none(getattr(self.args, 'success_weight', None)),
            'collision_weight': _finite_float_or_none(getattr(self.args, 'collision_weight', None)),
            'collision_reduction_weight': _finite_float_or_none(
                getattr(self.args, 'collision_reduction_weight', None)
            ),
            'global_weight': _finite_float_or_none(getattr(self.args, 'global_weight', None)),
            'shaping_weight': _finite_float_or_none(getattr(self.args, 'shaping_weight', None)),
            'max_reward': _finite_float_or_none(getattr(self.args, 'max_reward', None)),
            'min_reward': _finite_float_or_none(getattr(self.args, 'min_reward', None)),
            'success_reward_value': _finite_float_or_none(getattr(self.args, 'success_reward_value', None)),
            'no_collision_reward_value': _finite_float_or_none(
                getattr(self.args, 'no_collision_reward_value', None)
            ),
            'success_distance_threshold': _finite_float_or_none(
                getattr(self.args, 'success_distance_threshold', None)
            ),
            'collision_penalty_value': _finite_float_or_none(
                getattr(self.args, 'collision_penalty_value', None)
            ),
            'collision_distance_threshold': _finite_float_or_none(
                getattr(self.args, 'collision_distance_threshold', None)
            ),
            'global_reward_mode': str(getattr(self.args, 'global_reward_mode', 'success_rate')),
            'shaping_gamma': _finite_float_or_none(getattr(self.args, 'shaping_gamma', None)),
            'reward_version': str(os.getenv('REWARD_VERSION', os.getenv('reward_version', 'v1')) or 'v1').strip(),
            'reward_terminal_order_fix': bool(
                getattr(getattr(self, 'env', None), '_reward_terminal_order_fix_enabled', True)
            ),
            'goal_ring_individual_scale': _actual_reward_float(
                'goal_ring_individual_scale', 'GOAL_RING_INDIVIDUAL_SCALE', 0.25
            ),
            'goal_ring_team_gated': _actual_reward_bool(
                'goal_ring_team_gated', 'GOAL_RING_TEAM_GATED', False
            ),
            'goal_ring_require_agent_safe': _actual_reward_bool(
                'goal_ring_require_agent_safe', 'GOAL_RING_REQUIRE_AGENT_SAFE', True
            ),
            'progress_distance_state_scale': _actual_reward_float(
                'progress_distance_state_scale', 'PROGRESS_DISTANCE_STATE_SCALE', 0.0
            ),
            'progress_reward_scale': _actual_reward_float(
                'progress_reward_scale', 'PROGRESS_REWARD_SCALE', 1.0
            ),
            'team_progress_bottleneck_only': _actual_reward_bool(
                'team_progress_bottleneck_only', 'TEAM_PROGRESS_BOTTLENECK_ONLY', False
            ),
            'team_progress_non_bottleneck_scale': _actual_reward_float(
                'team_progress_non_bottleneck_scale', 'TEAM_PROGRESS_NON_BOTTLENECK_SCALE', 1.0
            ),
            'team_progress_bottleneck_eps': _actual_reward_float(
                'team_progress_bottleneck_eps', 'TEAM_PROGRESS_BOTTLENECK_EPS', 1.0
            ),
            'team_success_bonus': _actual_reward_float(
                'team_success_bonus', 'TEAM_SUCCESS_BONUS', 3000.0
            ),
            'unsafe_arrival_penalty': _actual_reward_float(
                'unsafe_arrival_penalty', 'UNSAFE_ARRIVAL_PENALTY', 1200.0
            ),
            'non_success_terminal_guard_enabled': _actual_reward_bool(
                'non_success_terminal_guard_enabled', 'NON_SUCCESS_TERMINAL_GUARD_ENABLED', True
            ),
            'non_success_terminal_penalty_base': _actual_reward_float(
                'non_success_terminal_penalty_base', 'NON_SUCCESS_TERMINAL_PENALTY_BASE', 250.0
            ),
            'non_success_terminal_penalty_per_meter': _actual_reward_float(
                'non_success_terminal_penalty_per_meter', 'NON_SUCCESS_TERMINAL_PENALTY_PER_METER', 900.0
            ),
            'non_success_terminal_penalty_max': _actual_reward_float(
                'non_success_terminal_penalty_max', 'NON_SUCCESS_TERMINAL_PENALTY_MAX', 1200.0
            ),
            'terminal_failure_penalty_base': _actual_reward_float(
                'terminal_failure_penalty_base', 'TERMINAL_FAILURE_PENALTY_BASE', 30.0
            ),
            'terminal_failure_penalty_per_meter': _actual_reward_float(
                'terminal_failure_penalty_per_meter', 'TERMINAL_FAILURE_PENALTY_PER_METER', 120.0
            ),
            'terminal_failure_penalty_max': _actual_reward_float(
                'terminal_failure_penalty_max', 'TERMINAL_FAILURE_PENALTY_MAX', 180.0
            ),
            'clearance_quality_bonus_weight': _actual_reward_float(
                'clearance_quality_bonus_weight', 'CLEARANCE_QUALITY_BONUS_WEIGHT', 800.0
            ),
            'efficiency_bonus_weight': _actual_reward_float(
                'efficiency_bonus_weight', 'EFFICIENCY_BONUS_WEIGHT', 800.0
            ),
            'team_sync_reward_enabled': _actual_reward_bool(
                'team_sync_enabled', 'TEAM_SYNC_REWARD_ENABLED', True
            ),
            'team_goal_occupancy_scale': _actual_reward_float(
                'team_goal_occupancy_scale', 'TEAM_GOAL_OCCUPANCY_SCALE', 1.0
            ),
            'team_bottleneck_progress_scale': _actual_reward_float(
                'team_bottleneck_progress_scale', 'TEAM_BOTTLENECK_PROGRESS_SCALE', 4.0
            ),
            'team_waiting_scale': _actual_reward_float(
                'team_waiting_scale', 'TEAM_WAITING_SCALE', 0.6
            ),
            'team_bottleneck_delta_clip': _actual_reward_float(
                'team_bottleneck_delta_clip', 'TEAM_BOTTLENECK_DELTA_CLIP', 1.0
            ),
            'clearance_dense_positive_scale': _actual_reward_float(
                'clearance_dense_positive_scale', 'CLEARANCE_DENSE_POSITIVE_SCALE', 0.0
            ),
            'height_dense_positive_scale': _actual_reward_float(
                'height_dense_positive_scale', 'HEIGHT_DENSE_POSITIVE_SCALE', 0.0
            ),
            'position_family': position_family_override or os.getenv('HELDOUT_POSITION_MODE', 'train_match'),
            'reference_positions_file': os.getenv('HELDOUT_REFERENCE_POSITIONS_FILE', ''),
            'use_fixed_positions': bool(getattr(self.args, 'use_fixed_positions', False)),
            'positions_file': str(getattr(self.args, 'positions_file', '') or ''),
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
            'agent_size': _finite_float_or_none(getattr(self.args, 'agent_size', None)),
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
            'apf_backend': str(getattr(self.args, 'apf_backend', 'python_original')),
            'gazebo_apf_output_dir': str(getattr(self.args, 'gazebo_apf_output_dir', 'results/gazebo_apf_validation')),
            'use_tf_potential_field': bool(getattr(self.args, 'use_tf_potential_field', True)),
            'use_fr_feature': bool(getattr(self.args, 'use_fr_feature', False)),
            'use_pf_feature': bool(getattr(self.args, 'use_pf_feature', False)),
            'terrain_sensing_mode': str(getattr(self.args, 'terrain_sensing_mode', 'local')),
            'goal_attraction': _finite_float_or_none(getattr(self.args, 'goal_attraction', None)),
            'lambda_1_base': _finite_float_or_none(getattr(self.args, 'lambda_1_base', None)),
            'terrain_repulsion': _finite_float_or_none(getattr(self.args, 'terrain_repulsion', None)),
            'agent_influence_range': _finite_float_or_none(getattr(self.args, 'agent_influence_range', None)),
            'delta_k_att': _finite_float_or_none(getattr(self.args, 'delta_k_att', None)),
            'delta_lambda_1': _finite_float_or_none(getattr(self.args, 'delta_lambda_1', None)),
            'delta_k_rep': _finite_float_or_none(getattr(self.args, 'delta_k_rep', None)),
            'delta_radius': _finite_float_or_none(getattr(self.args, 'delta_radius', None)),
            'max_force_magnitude': _finite_float_or_none(getattr(self.args, 'max_force_magnitude', None)),
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
            'eval_noise_scale': _finite_float_or_none(getattr(self.args, 'eval_noise_scale', 0.0)),
            'eval_random_action_prob': _finite_float_or_none(getattr(self.args, 'eval_random_action_prob', 0.0)),
            'eval_noise_seed': _safe_int(getattr(self.args, 'eval_noise_seed', None)),
            'eval_noise_type': _EVAL_NOISE_TYPE,
            'eval_noise_stream_mode': _EVAL_NOISE_STREAM_MODE,
            'training_runtime_manifest_path': str(
                os.getenv('TRAINING_RUNTIME_MANIFEST_PATH', '') or ''
            ),
            'eval_device': getattr(self, '_eval_device_info', {}),
        }
        for runtime_field in SCIENTIFIC_RUNTIME_ENV_FIELDS:
            runtime_value = getattr(self.args, runtime_field.attr, None)
            if runtime_field.kind == 'float':
                runtime_value = _finite_float_or_none(runtime_value)
            elif runtime_field.kind == 'int':
                runtime_value = _safe_int(runtime_value)
            elif runtime_field.kind == 'bool' and runtime_value is not None:
                runtime_value = bool(runtime_value)
            evaluation_setup[runtime_field.attr] = runtime_value

        def _log_episode_timing(episode_idx, episode_data, wall_seconds, failed=False):
            if not log_episode_timing:
                return
            episode_data = episode_data or {}
            steps = _safe_int(episode_data.get('steps'))
            rollout_seconds = _finite_float_or_none(episode_data.get('duration'))
            reward = _finite_float_or_none(episode_data.get('reward'))
            team_success = episode_data.get('team_success', episode_data.get('success', None))
            step_rate = None
            if steps and wall_seconds is not None and wall_seconds > 0:
                step_rate = float(steps) / float(wall_seconds)
            parts = [
                f"[EvalTiming] episode={episode_idx + 1}/{self.args.eval_episodes}",
                f"wall_s={wall_seconds:.2f}",
            ]
            if rollout_seconds is not None:
                parts.append(f"rollout_s={rollout_seconds:.2f}")
            if steps is not None:
                parts.append(f"steps={steps}")
            if step_rate is not None:
                parts.append(f"steps_per_s={step_rate:.2f}")
            if reward is not None:
                parts.append(f"reward={reward:.2f}")
            if team_success is not None:
                parts.append(f"team_success={team_success}")
            if failed:
                parts.append("status=failed")
            print(" | ".join(parts))

        eval_episode_parallelism = max(
            1,
            int(
                getattr(
                    self.args,
                    'eval_episode_parallelism',
                    _env_int('EVAL_EPISODE_PARALLELISM', 1),
                )
                or 1
            ),
        )
        eval_episode_count = max(1, int(self.args.eval_episodes))
        eval_episode_start_index = max(0, _env_int("EVAL_EPISODE_START_INDEX", 0))
        eval_episode_stop_index = eval_episode_start_index + eval_episode_count
        eval_episode_parallelism = min(eval_episode_parallelism, eval_episode_count)
        use_batched_episode_eval = eval_episode_parallelism > 1
        if use_batched_episode_eval and str(getattr(self.args, 'apf_backend', 'python_original')).strip().lower() == 'gazebo_apf':
            print("⚠️  apf_backend=gazebo_apf 需要逐步读取Gazebo反馈，已回退串行评估。")
            use_batched_episode_eval = False
        if use_batched_episode_eval and str(getattr(self.args, 'terrain_sensing_mode', 'local')).startswith('oracle'):
            print("⚠️  EVAL_EPISODE_PARALLELISM>1 但 terrain_sensing_mode=oracle*，为避免 scenario_ref 污染，回退串行评估。")
            use_batched_episode_eval = False
        try:
            eval_env_step_threads = int(
                getattr(
                    self.args,
                    'eval_env_step_threads',
                    _env_int('EVAL_ENV_STEP_THREADS', 1),
                )
                or 1
            )
        except Exception:
            eval_env_step_threads = 1
        eval_env_step_threads = max(1, eval_env_step_threads)
        if use_batched_episode_eval:
            eval_env_step_threads = min(eval_env_step_threads, eval_episode_parallelism)
        else:
            eval_env_step_threads = 1
        self._eval_env_step_threads = int(eval_env_step_threads)
        evaluation_setup['eval_episode_parallelism'] = int(eval_episode_parallelism if use_batched_episode_eval else 1)
        evaluation_setup['eval_episode_parallelism_mode'] = 'inprocess_batch' if use_batched_episode_eval else 'serial'
        evaluation_setup['eval_env_step_threads'] = int(self._eval_env_step_threads)

        serial_episode_range = range(eval_episode_start_index, eval_episode_stop_index)
        if use_batched_episode_eval:
            print(
                f"[EvalBatch] 启用方案A: episode_parallelism={eval_episode_parallelism}, "
                f"env_step_threads={self._eval_env_step_threads}，模型单次加载，batch actor/PF"
            )
            for batch_start in range(eval_episode_start_index, eval_episode_stop_index, eval_episode_parallelism):
                batch_end = min(eval_episode_stop_index, batch_start + eval_episode_parallelism)
                episode_contexts = []
                for episode in range(batch_start, batch_end):
                    terrain_level = _select_episode_terrain_level(episode)
                    episode_contexts.append(
                        self._build_episode_eval_context(
                            episode,
                            terrain_level,
                            terrain_seed_sequence,
                            terrain_variant_seed_sequence,
                            obstacle_seed_sequence,
                        )
                    )

                try:
                    batch_episode_data = self._evaluate_episode_batch(episode_contexts)
                except Exception as batch_e:
                    print(f"❌ batch评估失败，批次 {batch_start + 1}-{batch_end}: {batch_e}")
                    traceback.print_exc()
                    batch_episode_data = []

                for episode_data in batch_episode_data:
                    if episode_data is None:
                        print("⚠️  batch评估返回空episode数据，跳过")
                        continue
                    episode = int(episode_data.get('episode', 0))
                    if 'reward' not in episode_data or 'trajectory' not in episode_data:
                        _log_episode_timing(episode, episode_data, episode_data.get('wall_time_seconds', 0.0), failed=True)
                        print(f"⚠️  回合 {episode + 1} batch评估数据不完整，跳过")
                        print(f"    episode_data keys: {list(episode_data.keys())}")
                        continue

                    episode_wall_seconds = float(episode_data.get('wall_time_seconds', episode_data.get('duration', 0.0)) or 0.0)
                    all_rewards.append(episode_data['reward'])
                    _log_episode_timing(episode, episode_data, episode_wall_seconds)

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

                    stored_episode_data = dict(episode_data)
                    if not persist_episode_trajectories:
                        stored_episode_data['trajectory'] = []
                    if not save_actor_sequence:
                        stored_episode_data['actions_history'] = []
                    if not save_control_diagnostics:
                        stored_episode_data['executed_actions_history'] = []
                        stored_episode_data['velocity_history'] = []
                        stored_episode_data['goal_distance_history'] = []
                    if not persist_dynamic_replay_json:
                        stored_episode_data['dynamic_state_history'] = []
                        stored_episode_data['dynamic_time_history'] = []
                        stored_episode_data['dynamic_step_indices'] = []
                        stored_episode_data['dynamic_raw_action_history'] = []
                        stored_episode_data['dynamic_executed_action_history'] = []
                        stored_episode_data['dynamic_pf_force_history'] = []
                    stored_episode_data['vis_context'] = None
                    all_episodes_data.append(stored_episode_data)

                    if _should_save_episode_visualization(episode) and not self.args.disable_visualization:
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
            serial_episode_range = range(0)

        for episode in serial_episode_range:
            episode_wall_start = time.perf_counter()
            if not quiet_output:
                print(f"\n🚀 开始评估回合 {episode + 1}/{self.args.eval_episodes}")
            
            # 显式评估序列优先；无序列时再使用固定等级或随机等级。
            terrain_level = _select_episode_terrain_level(episode)
            if terrain_level_sequence:
                if not quiet_output:
                    print(f"🔧 使用预定义地形复杂度等级: {terrain_level}")
            elif self.args.terrain_complexity_level is None:
                if not quiet_output:
                    print(f"🎲 随机选择地形复杂度等级: {terrain_level}")
            else:
                if not quiet_output:
                    print(f"🏔️ 使用指定地形复杂度等级: {terrain_level}")
            terrain_level = _set_scenario_terrain_complexity(self.scenario, terrain_level)

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
                    _log_episode_timing(episode, {}, time.perf_counter() - episode_wall_start, failed=True)
                    print(f"⚠️  回合 {episode + 1} 评估返回空数据，跳过")
                    continue
                
                # 验证episode_data是否包含必要字段
                if 'reward' not in episode_data or 'trajectory' not in episode_data:
                    _log_episode_timing(episode, episode_data, time.perf_counter() - episode_wall_start, failed=True)
                    print(f"⚠️  回合 {episode + 1} 评估数据不完整，跳过")
                    print(f"    episode_data keys: {list(episode_data.keys())}")
                    continue
                
                episode_wall_seconds = time.perf_counter() - episode_wall_start
                actual_terrain_seed = terrain_seed
                if actual_terrain_seed is None:
                    actual_terrain_seed = getattr(
                        self.scenario,
                        'current_terrain_seed',
                        getattr(self.scenario, 'seed', None),
                    )
                actual_terrain_variant_seed = _scenario_terrain_variant_seed(
                    self.scenario,
                    terrain_variant_seed,
                )
                actual_obstacle_seed = obstacle_seed
                if actual_obstacle_seed is None:
                    actual_obstacle_seed = getattr(self.scenario, 'current_episode_obstacle_seed', None)

                episode_data['wall_time_seconds'] = float(episode_wall_seconds)
                episode_data['terrain_complexity_level'] = terrain_level
                episode_data['terrain_seed'] = actual_terrain_seed
                episode_data['terrain_variant_seed'] = actual_terrain_variant_seed
                episode_data['obstacle_seed'] = actual_obstacle_seed
                all_rewards.append(episode_data['reward'])
                _log_episode_timing(episode, episode_data, episode_wall_seconds)
                
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
                if not persist_dynamic_replay_json:
                    stored_episode_data['dynamic_state_history'] = []
                    stored_episode_data['dynamic_time_history'] = []
                    stored_episode_data['dynamic_step_indices'] = []
                    stored_episode_data['dynamic_raw_action_history'] = []
                    stored_episode_data['dynamic_executed_action_history'] = []
                    stored_episode_data['dynamic_pf_force_history'] = []
                stored_episode_data['vis_context'] = None
                all_episodes_data.append(stored_episode_data)
                
                if _should_save_episode_visualization(episode) and not self.args.disable_visualization:
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
                _log_episode_timing(episode, {}, time.perf_counter() - episode_wall_start, failed=True)
                print(f"❌ 回合 {episode + 1} 评估失败: {ep_e}")
                traceback.print_exc()
                # 继续下一个回合，不中断整个评估流程
                continue
                
        actual_episode_indices = []
        for episode_data in all_episodes_data:
            try:
                actual_episode_indices.append(int(episode_data.get('episode')))
            except Exception:
                pass
        expected_episode_indices = list(range(eval_episode_start_index, eval_episode_stop_index))
        if sorted(actual_episode_indices) != expected_episode_indices or len(all_rewards) != eval_episode_count:
            missing = sorted(set(expected_episode_indices) - set(actual_episode_indices))
            duplicates = sorted(
                episode_idx
                for episode_idx in set(actual_episode_indices)
                if actual_episode_indices.count(episode_idx) > 1
            )
            raise RuntimeError(
                "评估回合不完整，拒绝写入可复用结果: "
                f"completed={len(all_rewards)}/{eval_episode_count}, "
                f"missing={missing}, duplicates={duplicates}"
            )

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
                'terrain_complexity_level_sequence': terrain_level_sequence or [],
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
        os.makedirs(self.args.save_viz_path, exist_ok=True)
        evaluation_setup['gazebo_live_sync_enabled'] = _env_flag('GAZEBO_LIVE_SYNC', False)
        evaluation_setup['eval_backend'] = getattr(self.args, 'eval_backend', None) or os.getenv('EVAL_BACKEND', 'python_only')
        evaluation_setup['gazebo_live_validation'] = _env_flag('GAZEBO_LIVE_VALIDATION', False)
        evaluation_setup['gazebo_live_validation_dir'] = os.getenv('GAZEBO_LIVE_VALIDATION_DIR', None)
        evaluation_setup['gazebo_live_autolaunch'] = _env_flag('GAZEBO_LIVE_AUTOLAUNCH', False)
        evaluation_setup['gazebo_live_autolaunch_gui'] = _env_flag('GAZEBO_LIVE_AUTOLAUNCH_GUI', False)
        evaluation_setup['gazebo_live_autolaunch_run'] = _env_flag('GAZEBO_LIVE_AUTOLAUNCH_RUN', False)
        evaluation_setup['gazebo_live_keep_alive'] = _env_flag('GAZEBO_LIVE_KEEP_ALIVE', False)
        evaluation_setup['gazebo_live_export_dir'] = os.getenv('GAZEBO_LIVE_EXPORT_DIR', None)
        evaluation_setup['gazebo_live_world'] = os.getenv('GAZEBO_LIVE_WORLD', 'matd3_static_scene')
        evaluation_setup['gazebo_live_agent_prefix'] = os.getenv('GAZEBO_LIVE_AGENT_PREFIX', 'dynamic_agent_')
        evaluation_setup['gazebo_live_consistency_mode'] = os.getenv('GAZEBO_LIVE_CONSISTENCY_MODE', 'gazebo_authoritative')
        gazebo_live_setup_semantic_mode = os.getenv('GAZEBO_LIVE_SEMANTIC_MODE', 'transfer_equivalence').strip().lower()
        gazebo_live_setup_physical_contact = gazebo_live_setup_semantic_mode in (
            'physical',
            'physical_robustness',
            'contact_authoritative',
            'gazebo_contact_authoritative',
        )
        evaluation_setup['gazebo_live_semantic_mode'] = gazebo_live_setup_semantic_mode
        gazebo_live_setup_collision_mode = _gazebo_live_collision_mode()
        evaluation_setup['gazebo_live_collision_mode'] = gazebo_live_setup_collision_mode
        evaluation_setup['gazebo_live_physical_collision_enabled'] = gazebo_live_setup_collision_mode != 'nonblocking'
        evaluation_setup['gazebo_live_contact_authoritative'] = _env_flag('GAZEBO_LIVE_CONTACT_AUTHORITATIVE', gazebo_live_setup_physical_contact)
        evaluation_setup['gazebo_live_contact_marks_collision'] = _env_flag('GAZEBO_LIVE_CONTACT_MARKS_COLLISION', _env_flag('GAZEBO_LIVE_CONTACT_AUTHORITATIVE', gazebo_live_setup_physical_contact))
        evaluation_setup['gazebo_live_contact_terminates'] = _env_flag('GAZEBO_LIVE_CONTACT_TERMINATES', False)
        evaluation_setup['gazebo_live_pose_correction'] = _env_flag('GAZEBO_LIVE_POSE_CORRECTION', False)
        evaluation_setup['gazebo_live_sync_every'] = _env_int('GAZEBO_LIVE_SYNC_EVERY', 1)
        evaluation_setup['gazebo_live_control_mode'] = os.getenv('GAZEBO_LIVE_CONTROL_MODE', 'pose')
        evaluation_setup['gazebo_live_command_sleep'] = _env_float('GAZEBO_LIVE_COMMAND_SLEEP', 0.0)
        evaluation_setup['gazebo_live_step_iterations'] = _env_int('GAZEBO_LIVE_STEP_ITERATIONS', 0)
        evaluation_setup['gazebo_live_wait_ack'] = _env_flag('GAZEBO_LIVE_WAIT_ACK', False)
        evaluation_setup['gazebo_live_ack_timeout'] = _env_float('GAZEBO_LIVE_ACK_TIMEOUT', 2.0)
        evaluation_setup['gazebo_live_pre_step_sleep_ms'] = _env_int('GAZEBO_LIVE_PRE_STEP_SLEEP_MS', 0)
        evaluation_setup['gazebo_live_post_step_sleep_ms'] = _env_int('GAZEBO_LIVE_POST_STEP_SLEEP_MS', 0)
        evaluation_setup['gazebo_live_wall_time_step_ms'] = _env_int('GAZEBO_LIVE_WALL_TIME_STEP_MS', 0)
        evaluation_setup['gazebo_live_pause_for_step'] = _env_flag('GAZEBO_LIVE_PAUSE_FOR_STEP', False)
        evaluation_setup['gazebo_live_state_feedback'] = _env_flag('GAZEBO_LIVE_STATE_FEEDBACK', False)
        evaluation_setup['gazebo_live_feedback_velocity_mode'] = os.getenv('GAZEBO_LIVE_FEEDBACK_VELOCITY_MODE', 'clamp')
        evaluation_setup['gazebo_live_feedback_acceleration_mode'] = os.getenv('GAZEBO_LIVE_FEEDBACK_ACCELERATION_MODE', 'estimate')
        evaluation_setup['gazebo_live_max_pose_jump'] = _env_float('GAZEBO_LIVE_MAX_POSE_JUMP', 100.0)
        evaluation_setup['gazebo_uav_visual_scale_multiplier'] = _env_float('GAZEBO_UAV_VISUAL_SCALE_MULTIPLIER', _env_float('GAZEBO_LIVE_UAV_VISUAL_SCALE_MULTIPLIER', 3.0))
        evaluation_setup['gazebo_live_authoritative_feedback'] = _env_flag('GAZEBO_LIVE_AUTHORITATIVE_FEEDBACK', True)
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
        if summary.get('avg_terrain_collision_count') is not None:
            print(f"平均地形碰撞次数: {summary['avg_terrain_collision_count']:.2f}")
        if summary.get('avg_obstacle_collision_count') is not None:
            print(f"平均障碍物碰撞次数: {summary['avg_obstacle_collision_count']:.2f}")
        if summary.get('avg_inter_agent_collision_count') is not None:
            print(f"平均队间碰撞次数: {summary['avg_inter_agent_collision_count']:.2f}")
        if summary.get('avg_min_inter_agent_clearance') is not None:
            print(f"平均最小队间净空: {summary['avg_min_inter_agent_clearance']:.2f}")
        summary_plot_file = os.path.join(self.args.save_viz_path, _artifact_filename('evaluation_summary.png'))
        print(f"统计图: {summary_plot_file}")

        visualization_artifacts = {
            'episode_visualizations': episode_visualizations,
        }
        summary_plot_path = None
        try:
            summary_plot_path = _generate_evaluation_summary_plot(
                all_episodes_data,
                summary,
                summary_plot_file,
            )
        except Exception as summary_plot_err:
            print(f"⚠️  评估统计图生成失败: {summary_plot_err}")
            traceback.print_exc()
            summary_plot_path = None
        if summary_plot_path and os.path.exists(summary_plot_path):
            visualization_artifacts['evaluation_summary_plot'] = summary_plot_path
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

        best_html_alias = _copy_alias(best_generated_files.get('html_path'), _artifact_filename('best_reward_interactive.html'))
        best_png_alias = _copy_alias(best_generated_files.get('image_path'), _artifact_filename('best_reward.png'))
        best_actor_sequence_alias = _copy_alias(best_generated_files.get('actor_sequence_path'), _artifact_filename('best_reward_actor_sequence.png'))
        best_control_diag_alias = _copy_alias(best_generated_files.get('control_diagnostics_path'), _artifact_filename('best_reward_control_diagnostics.png'))
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
                    success_html_alias = os.path.join(self.args.save_viz_path, _artifact_filename('team_success_best_interactive.html'))
                    shutil.copyfile(success_html_path, success_html_alias)
                    visualization_artifacts['team_success_best_html'] = success_html_alias
                if success_png_path and os.path.exists(success_png_path):
                    success_png_alias = os.path.join(self.args.save_viz_path, _artifact_filename('team_success_best.png'))
                    shutil.copyfile(success_png_path, success_png_alias)
                    visualization_artifacts['team_success_best_png'] = success_png_alias
                if success_actor_sequence_path and os.path.exists(success_actor_sequence_path):
                    success_actor_alias = os.path.join(self.args.save_viz_path, _artifact_filename('team_success_best_actor_sequence.png'))
                    shutil.copyfile(success_actor_sequence_path, success_actor_alias)
                    visualization_artifacts['team_success_best_actor_sequence'] = success_actor_alias
                if success_control_diag_path and os.path.exists(success_control_diag_path):
                    success_control_alias = os.path.join(self.args.save_viz_path, _artifact_filename('team_success_best_control_diagnostics.png'))
                    shutil.copyfile(success_control_diag_path, success_control_alias)
                    visualization_artifacts['team_success_best_control_diagnostics'] = success_control_alias
            except Exception as success_viz_e:
                print(f"⚠️  团队成功回合HTML生成失败: {success_viz_e}")
                traceback.print_exc()
        elif save_team_success_html and best_success_episode_data is None:
            print("⏭️  未生成团队成功HTML（本次评估没有团队成功回合）")

        try:
            reward_decomp_csv_path = _write_reward_decomposition_eval_csv(
                self.args.save_viz_path,
                all_episodes_data,
                self.args,
            )
            visualization_artifacts['reward_decomposition_eval_csv'] = reward_decomp_csv_path
            print(f"✅ Reward decomposition CSV已保存: {reward_decomp_csv_path}")
        except Exception as reward_diag_e:
            print(f"⚠️  Reward decomposition CSV生成失败: {reward_diag_e}")
            traceback.print_exc()

        try:
            success_consistency_path = _write_success_reward_consistency_report(
                self.args.save_viz_path,
                all_episodes_data,
                self.args,
            )
            visualization_artifacts['success_reward_consistency_report'] = success_consistency_path
            print(f"✅ Success/reward一致性报告已保存: {success_consistency_path}")
        except Exception as success_diag_e:
            print(f"⚠️  Success/reward一致性报告生成失败: {success_diag_e}")
            traceback.print_exc()

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
            # 结果文件记录“实际执行”的逐回合环境，而不是只记录调用方是否
            # 显式传入过序列。否则自动生成的地形/障碍在结果中会显示为空，
            # 无法用于跨模型配对审计。
            'terrain_complexity_level_sequence': [
                ep.get('terrain_complexity_level') for ep in all_episodes_data
            ],
            'terrain_seed_sequence': [ep.get('terrain_seed') for ep in all_episodes_data],
            'terrain_variant_seed_sequence': [
                ep.get('terrain_variant_seed') for ep in all_episodes_data
            ],
            'obstacle_seed_sequence': [ep.get('obstacle_seed') for ep in all_episodes_data],
            'episode_details': [
                {
                    'episode': ep['episode'],
                    'reward': float(ep['reward']),
                    'steps': ep['steps'],
                    'episode_length': ep.get('episode_length', getattr(self.args, 'episode_length', None)),
                    'terrain_complexity_level': ep.get('terrain_complexity_level', 'unknown'),
                    'terrain_seed': ep.get('terrain_seed', None),
                    'terrain_variant_seed': ep.get('terrain_variant_seed', None),
                    'obstacle_seed': ep.get('obstacle_seed', None),
                    'duration': ep['duration'],
                    'wall_time_seconds': ep.get('wall_time_seconds', None),
                    'trajectory': ep.get('trajectory', []) if persist_episode_trajectories else [],
                    # 🔧 新增：保存碰撞和成功指标（与训练脚本一致）
                    'collision_count': ep.get('collision_count', 0),
                    'terrain_collision_count': ep.get('terrain_collision_count', 0),
                    'obstacle_collision_count': ep.get('obstacle_collision_count', 0),
                    'agent_collision_counts': ep.get('agent_collision_counts', []),
                    'inter_agent_collision_count': ep.get('inter_agent_collision_count', 0),
                    'agent_inter_agent_collision_counts': ep.get('agent_inter_agent_collision_counts', []),
                    'min_inter_agent_clearance': ep.get('min_inter_agent_clearance', None),
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
                    'agent_safe_flags': ep.get('agent_safe_flags', []),
                    'reward_decomposition': ep.get('reward_decomposition', {}),
                    'diagnostic_metrics': ep.get('diagnostic_metrics', {}),
                    'terminal_success_bonus_applied': ep.get('terminal_success_bonus_applied', False),
                    'episode_done_reason': ep.get('episode_done_reason', None),
                    'apf_backend': ep.get('apf_backend', 'python_original'),
                    'gazebo_apf_enabled': ep.get('gazebo_apf_enabled', False),
                    'gazebo_apf_error': ep.get('gazebo_apf_error', None),
                    'gazebo_apf_metrics': ep.get('gazebo_apf_metrics', None),
                    'gazebo_apf_adapter_metrics': ep.get('gazebo_apf_adapter_metrics', None),
                    'gazebo_obstacle_velocity_filter_mode': ep.get('gazebo_obstacle_velocity_filter_mode', 'off'),
                    'gazebo_obstacle_velocity_filter_enabled': ep.get('gazebo_obstacle_velocity_filter_enabled', False),
                    'gazebo_obstacle_velocity_filter_error': ep.get('gazebo_obstacle_velocity_filter_error', None),
                    'gazebo_obstacle_velocity_filter_summary': ep.get('gazebo_obstacle_velocity_filter_summary', None),
                    'gazebo_obstacle_velocity_filter_artifacts': ep.get('gazebo_obstacle_velocity_filter_artifacts', {}),
                    'gazebo_obstacle_velocity_filter_records_tail': ep.get('gazebo_obstacle_velocity_filter_records_tail', []),
                    'gazebo_progress_diagnostics_artifacts': ep.get('gazebo_progress_diagnostics_artifacts', {}),
                    'gazebo_progress_failure_summary': ep.get('gazebo_progress_failure_summary', None),
                    'gazebo_progress_diagnostics_csv': ep.get('gazebo_progress_diagnostics_csv', None),
                    'progress_failure_summary_json': ep.get('progress_failure_summary_json', None),
                    'gazebo_progress_diagnostics_error': ep.get('gazebo_progress_diagnostics_error', None),
                    'gazebo_apf_validation_dir': ep.get('gazebo_apf_validation_dir', None),
                    'gazebo_live_sync_active': ep.get('gazebo_live_sync_active', False),
                    'gazebo_live_sync_error': ep.get('gazebo_live_sync_error', None),
                    'gazebo_live_launch_error': ep.get('gazebo_live_launch_error', None),
                    'gazebo_live_autolaunched': ep.get('gazebo_live_autolaunched', False),
                    'gazebo_live_autolaunch_gui': ep.get('gazebo_live_autolaunch_gui', False),
                    'gazebo_live_autolaunch_run': ep.get('gazebo_live_autolaunch_run', False),
                    'gazebo_live_gui_error': ep.get('gazebo_live_gui_error', None),
                    'gazebo_live_world_sdf': ep.get('gazebo_live_world_sdf', None),
                    'gazebo_live_scenario_json': ep.get('gazebo_live_scenario_json', None),
                    'gazebo_live_export_dir': ep.get('gazebo_live_export_dir', None),
                    'gazebo_live_control_mode': ep.get('gazebo_live_control_mode', None),
                    'gazebo_live_command_sleep': ep.get('gazebo_live_command_sleep', None),
                    'gazebo_live_step_iterations': ep.get('gazebo_live_step_iterations', None),
                    'gazebo_live_state_feedback': ep.get('gazebo_live_state_feedback', False),
                    'gazebo_live_contact_feedback': ep.get('gazebo_live_contact_feedback', False),
                    'gazebo_live_contact_authoritative': ep.get('gazebo_live_contact_authoritative', True),
                    'gazebo_live_contact_marks_collision': ep.get('gazebo_live_contact_marks_collision', True),
                    'gazebo_live_contact_terminates': ep.get('gazebo_live_contact_terminates', False),
                    'gazebo_live_bridge_running': ep.get('gazebo_live_bridge_running', False),
                    'gazebo_live_contact_feedback_armed': ep.get('gazebo_live_contact_feedback_armed', False),
                    'gazebo_live_contact_topics': ep.get('gazebo_live_contact_topics', []),
                    'gazebo_live_state_file': ep.get('gazebo_live_state_file', None),
                    'gazebo_live_contact_flag_file': ep.get('gazebo_live_contact_flag_file', None),
                    'gazebo_live_feedback_velocity_mode': ep.get('gazebo_live_feedback_velocity_mode', None),
                    'gazebo_live_world_name': ep.get('gazebo_live_world_name', None),
                    'gazebo_live_agent_prefix': ep.get('gazebo_live_agent_prefix', None),
                    'gazebo_live_cmd_vel_publish_count': ep.get('gazebo_live_cmd_vel_publish_count', 0),
                    'gazebo_live_pose_publish_count': ep.get('gazebo_live_pose_publish_count', 0),
                    'gazebo_live_bridge_ack_enabled': ep.get('gazebo_live_bridge_ack_enabled', False),
                    'gazebo_live_bridge_ack_count': ep.get('gazebo_live_bridge_ack_count', 0),
                    'gazebo_live_bridge_ack_timeout_count': ep.get('gazebo_live_bridge_ack_timeout_count', 0),
                    'gazebo_live_bridge_last_ack_state_frame': ep.get('gazebo_live_bridge_last_ack_state_frame', None),
                    'gazebo_live_pre_step_sleep_ms': ep.get('gazebo_live_pre_step_sleep_ms', 0),
                    'gazebo_live_post_step_sleep_ms': ep.get('gazebo_live_post_step_sleep_ms', 0),
                    'gazebo_live_wall_time_step_ms': ep.get('gazebo_live_wall_time_step_ms', 0),
                    'gazebo_live_pause_for_step': ep.get('gazebo_live_pause_for_step', False),
                    'gazebo_live_pose_jump_reject_count': ep.get('gazebo_live_pose_jump_reject_count', 0),
                    'gazebo_live_max_pose_jump_observed': ep.get('gazebo_live_max_pose_jump_observed', 0.0),
                    'gazebo_live_max_feedback_speed_observed': ep.get('gazebo_live_max_feedback_speed_observed', 0.0),
                    'gazebo_live_state_feedback_updates': ep.get('gazebo_live_state_feedback_updates', 0),
                    'gazebo_live_state_feedback_misses': ep.get('gazebo_live_state_feedback_misses', 0),
                    'gazebo_live_state_feedback_update_ratio': ep.get('gazebo_live_state_feedback_update_ratio', 0.0),
                    'gazebo_live_authoritative_feedback': ep.get('gazebo_live_authoritative_feedback', False),
                    'gazebo_live_authoritative_feedback_updates': ep.get('gazebo_live_authoritative_feedback_updates', 0),
                    'gazebo_live_authoritative_feedback_errors': ep.get('gazebo_live_authoritative_feedback_errors', 0),
                    'gazebo_live_feedback_update_ratio': ep.get('gazebo_live_feedback_update_ratio', 0.0),
                    'gazebo_contact_detected': ep.get('gazebo_contact_detected', False),
                    'gazebo_contact_count': ep.get('gazebo_contact_count', 0),
                    'gazebo_contact_raw_flag_count': ep.get('gazebo_contact_raw_flag_count', 0),
                    'gazebo_contact_false_positive_count': ep.get('gazebo_contact_false_positive_count', 0),
                    'gazebo_contact_step': ep.get('gazebo_contact_step', None),
                    'gazebo_first_contact_step': ep.get('gazebo_first_contact_step', None),
                    'first_contact_step': ep.get('first_contact_step', None),
                    'first_contact_pair': ep.get('first_contact_pair', None),
                    'first_contact_position': ep.get('first_contact_position', None),
                    'first_contact_agent_id': ep.get('first_contact_agent_id', None),
                    'gazebo_contact_agent_indices': ep.get('gazebo_contact_agent_indices', []),
                    'gazebo_contact_pairs': ep.get('gazebo_contact_pairs', []),
                    'gazebo_real_contact_pairs': ep.get('gazebo_real_contact_pairs', []),
                    'gazebo_contact_pair_class_counts': ep.get('gazebo_contact_pair_class_counts', {}),
                    'pre_contact_surface_distance_trace': ep.get('pre_contact_surface_distance_trace', []),
                    'last_20_raw_actions': ep.get('last_20_raw_actions', []),
                    'last_20_python_apf_outputs': ep.get('last_20_python_apf_outputs', []),
                    'last_20_gazebo_apf_outputs': ep.get('last_20_gazebo_apf_outputs', []),
                    'last_20_nominal_cmd_vel': ep.get('last_20_nominal_cmd_vel', []),
                    'last_20_cmd_vel': ep.get('last_20_cmd_vel', []),
                    'last_20_safety_filter_outputs': ep.get('last_20_safety_filter_outputs', []),
                    'last_20_gazebo_pose': ep.get('last_20_gazebo_pose', []),
                    'gazebo_contact_debug_window': ep.get('gazebo_contact_debug_window', []),
                    'gazebo_contact_debug_window_start': ep.get('gazebo_contact_debug_window_start', None),
                    'gazebo_contact_debug_window_end_requested': ep.get('gazebo_contact_debug_window_end_requested', None),
                    'gazebo_contact_debug_window_end_available': ep.get('gazebo_contact_debug_window_end_available', None),
                    'gazebo_contact_debug_agent_id': ep.get('gazebo_contact_debug_agent_id', None),
                    'gazebo_contact_debug_obstacle': ep.get('gazebo_contact_debug_obstacle', None),
                    'gazebo_contact_debug_artifacts': ep.get('gazebo_contact_debug_artifacts', {}),
                    'gazebo_live_scene_check': ep.get('gazebo_live_scene_check', None),
                    'gazebo_live_sync_health': ep.get('gazebo_live_sync_health', None),
                    'python_scene_signature': ep.get('python_scene_signature', None),
                    'python_scene_signature_parts': ep.get('python_scene_signature_parts', None),
                    'scene_signature': ep.get('scene_signature', None),
                    'validation_trace_tail': ep.get('validation_trace_tail', []),
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
    parser.add_argument("--config-resolve-only", action="store_true", default=False,
                       help="完成训练配置对齐与参数解析后输出JSON并退出，不构建环境或加载模型")
    parser.add_argument("--eval-episode-parallelism", type=int, default=_env_int('EVAL_EPISODE_PARALLELISM', 1),
                       help="方案A：单进程内同时推进的评估回合数；1为旧串行路径")
    parser.add_argument("--eval-env-step-threads", type=int, default=_env_int('EVAL_ENV_STEP_THREADS', 1),
                       help="方案A：batch内并行env.step的线程数；1为串行env.step")
    parser.add_argument("--eval-process-shards", type=int, default=_env_int('EVAL_PROCESS_SHARDS', 1),
                       help="方案B：将评估回合切成多个独立子进程并行运行；1为关闭")
    parser.add_argument("--eval-process-workers", type=int, default=_env_int('EVAL_PROCESS_WORKERS', 0),
                       help="方案B：同时运行的子进程数；0表示等于eval-process-shards")
    parser.add_argument("--eval-shard-episode-parallelism", type=int, default=_env_int('EVAL_SHARD_EPISODE_PARALLELISM', 0),
                       help="方案B：每个子进程内部的episode_parallelism；0表示按总parallelism自动均分")
    parser.add_argument("--eval-shard-env-step-threads", type=int, default=_env_int('EVAL_SHARD_ENV_STEP_THREADS', 0),
                       help="方案B：每个子进程内部的env_step_threads；0表示按总threads自动均分")
    parser.add_argument("--eval-backend", "--eval_backend", type=str,
                       default=os.getenv("EVAL_BACKEND", None),
                       choices=["python_only", "gazebo_live", "both"],
                       help="显式启用验证评估后端: python_only=原Python评估, gazebo_live=Gazebo live反馈评估, both=同seed成对对照")
    parser.add_argument("--validation-output-dir", type=str,
                       default=os.getenv("GAZEBO_LIVE_VALIDATION_DIR", "results/gazebo_live_validation"),
                       help="Gazebo-live验证结果输出目录")
    parser.add_argument("--gazebo-live-gui", action="store_true", default=False,
                       help="Gazebo-live后端启动Gazebo GUI；等价于设置GAZEBO_LIVE_AUTOLAUNCH_GUI=1")
    parser.add_argument("--gazebo-live-gui-required", action="store_true", default=False,
                       help="Gazebo GUI启动失败时直接中断；等价于设置GAZEBO_LIVE_GUI_REQUIRED=1")
    parser.add_argument("--terrain-contact-eps", type=float, default=float(os.getenv('TERRAIN_CONTACT_EPS', '0.2')),
                       help="地形接触/碰撞高度容差，默认从TERRAIN_CONTACT_EPS读取")
    parser.add_argument("--obstacle-observation-mode", "--obstacle-obs-mode", dest="obstacle_observation_mode",
                       type=str,
                       default=os.getenv('OBSTACLE_OBSERVATION_MODE', os.getenv('OBSTACLE_OBS_MODE', 'nearest_surface')),
                       help="障碍物15维槽位选择方式: nearest_surface 或 risk_lite_v2")
    parser.add_argument("--obstacle-risk-velocity-forward-weight", type=float,
                       default=float(os.getenv('OBSTACLE_RISK_VELOCITY_FORWARD_WEIGHT', os.getenv('OBSTACLE_OBS_VEL_FORWARD_WEIGHT', '4.0'))),
                       help="risk_lite_v2 中速度方向前向距离惩罚权重")
    parser.add_argument("--obstacle-risk-goal-along-weight", type=float,
                       default=float(os.getenv('OBSTACLE_RISK_GOAL_ALONG_WEIGHT', os.getenv('OBSTACLE_OBS_GOAL_ALONG_WEIGHT', '3.0'))),
                       help="risk_lite_v2 中目标走廊沿程距离惩罚权重")
    parser.add_argument("--terrain-complexity-level", type=int, default=None, 
                       help="地形复杂度等级 (1-4)，None表示随机选择")
    parser.add_argument("--random-terrain", action="store_true", default=False,
                       help="使用随机地形（默认启用）")
    # 🔧 修复：与训练脚本保持一致的默认参数
    parser.add_argument("--gravity", type=float, default=0.0, help="环境重力加速度（作用于 -Z 方向），默认0.0（无重力）")
    parser.add_argument("--control-accel-gain", type=float, default=1.0, help="动作到物理加速度的控制增益，默认1.0")
    parser.add_argument("--reward-pos-scale", type=float, default=1.5, help="正向奖励缩放系数，默认1.5")
    parser.add_argument("--reward-neg-scale", type=float, default=2.5, help="负向奖励缩放系数，默认2.5")
    parser.add_argument("--agent-size", type=float, default=float(os.getenv('AGENT_SIZE', '0.5')), help="智能体物理半径/可视化半径，默认从AGENT_SIZE读取")
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
    parser.add_argument("--apf-backend", "--apf_backend", dest="apf_backend", type=str,
                       default=os.getenv("APF_BACKEND", "python_original"),
                       choices=["python_original", "gazebo_apf"],
                       help="APF backend: python_original keeps the existing path; gazebo_apf uses Gazebo-authoritative state")
    parser.add_argument("--gazebo-apf-output-dir", type=str,
                       default=os.getenv("GAZEBO_APF_VALIDATION_DIR", "results/gazebo_apf_validation"),
                       help="Gazebo APF validation output directory")
    parser.add_argument("--gazebo-apf-consistency-threshold", type=float,
                       default=float(os.getenv("GAZEBO_APF_CONSISTENCY_THRESHOLD", "0.05")),
                       help="Mismatch threshold for Gazebo APF consistency validation")
    
    # 🔧 新增：地形感知模式参数（仅用于评估时APF地形力计算）
    parser.add_argument("--terrain-sensing-mode", type=str, default="local",
                        choices=["local", "oracle_same_probes", "oracle_dense"],
                        help="地形感知模式: local=使用观测中的地形信息, oracle_same_probes=Oracle真值(相同probe布局), oracle_dense=Oracle真值(密集探测)")
    
    # 🔧 新增：FR和PF特征标志
    parser.add_argument("--use-fr-feature", type=lambda x: (str(x).lower() in ('1','true','yes','on')), default=True,
                       help="Enable FR feature (Force Ratio as separate input)")
    parser.add_argument("--use-pf-feature", type=lambda x: (str(x).lower() in ('1','true','yes','on')), default=True,
                       help="Enable PF feature (Potential field force appended to obs)")
    parser.add_argument("--pf-feature-dim", type=int, default=int(os.getenv("PF_FEATURE_DIM", "3")),
                       help="PF feature dimension used by actor/critic auxiliary inputs")
    
    # 🔧 修复：与当前run_optimized.sh默认值保持一致；run_evaluation.sh会优先回读训练结果JSON覆盖这些兜底值
    parser.add_argument("--goal-attraction", type=float, default=float(os.getenv("GOAL_ATTRACTION", "26.0")),
                        help="Goal attraction force，默认26.0（与当前run_optimized.sh一致）")
    parser.add_argument("--lambda-1-base", type=float, default=float(os.getenv("LAMBDA_1_BASE", "8.5")),
                        help="Lambda_1 base value，默认8.5（与当前run_optimized.sh一致）")
    parser.add_argument("--terrain-repulsion", type=float, default=float(os.getenv("TERRAIN_REPULSION", "1600.0")),
                        help="Terrain repulsion force，默认1600.0（与当前run_optimized.sh一致）")
    parser.add_argument("--agent-influence-range", type=float, default=float(os.getenv("AGENT_INFLUENCE_RANGE", "150.0")),
                        help="Agent influence range，默认150.0（与当前run_optimized.sh一致）")
    parser.add_argument("--delta-k-att", type=float, default=float(os.getenv("DELTA_K_ATT", "5.0")),
                        help="Delta K_att，默认5.0（与当前run_optimized.sh一致）")
    parser.add_argument("--delta-lambda-1", type=float, default=float(os.getenv("DELTA_LAMBDA_1", "2.2")),
                        help="Delta Lambda_1，默认2.2（与当前run_optimized.sh一致）")
    parser.add_argument("--delta-k-rep", type=float, default=float(os.getenv("DELTA_K_REP", "600.0")),
                        help="Delta K_rep，默认600.0（与当前run_optimized.sh一致）")
    parser.add_argument("--delta-radius", type=float, default=float(os.getenv("DELTA_RADIUS", "80.0")),
                        help="Delta Radius，默认80.0（与当前run_optimized.sh一致）")
    parser.add_argument("--max-force-magnitude", type=float, default=float(os.getenv("MAX_FORCE_MAGNITUDE", "80.0")),
                       help="最大势场力幅值，默认80.0（与当前run_optimized.sh训练默认一致）")
    
    # 🔧 新增：算法选择
    parser.add_argument("--algorithm", type=str, default="matd3", choices=["maddpg", "matd3", "mappo"],
                       help="Training algorithm selection (maddpg, matd3 or mappo)")
    parser.add_argument("--eval-noise-scale", type=float, default=0.0,
                       help="评估时加到Actor原始动作前3维的高斯噪声标准差；默认0，保持纯策略评估")
    parser.add_argument("--eval-random-action-prob", type=float, default=0.0,
                       help="评估时用均匀随机raw action替换Actor输出的概率；默认0")
    parser.add_argument("--eval-noise-seed", type=int, default=None,
                       help="评估扰动基础种子；按全局episode拆分独立随机流，不受分片和提前终止影响")
    
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
    parser.add_argument("--max-reward", type=float, default=1000.0, help="最大奖励值")
    parser.add_argument("--min-reward", type=float, default=-2500.0, help="最小奖励值")
    parser.add_argument("--success-reward-value", type=float, default=150.0, help="成功一次性奖励值")
    parser.add_argument("--no-collision-reward-value", type=float, default=0.0, help="无碰撞奖励值")
    parser.add_argument("--success-distance-threshold", type=float, default=2.0, help="成功判定距离阈值")
    parser.add_argument("--collision-penalty-value", type=float, default=30.0, help="碰撞惩罚绝对值")
    parser.add_argument("--collision-distance-threshold", type=float, default=0.5, help="碰撞/接触距离阈值")
    parser.add_argument("--global-reward-mode", type=str, default="success_rate", help="全局奖励模式")
    parser.add_argument("--shaping-gamma", type=float, default=0.95, help="潜势函数 gamma")
    
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

    add_runtime_environment_arguments(parser)
    return parser.parse_args()


@contextmanager
def _temporary_env(updates):
    old_values = {}
    for key, value in (updates or {}).items():
        old_values[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[str(key)] = str(value)
    try:
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _validation_backend_env(args, backend):
    validation_dir = str(Path(getattr(args, "validation_output_dir", "results/gazebo_live_validation")).expanduser().resolve())
    try:
        sim_dt = float(getattr(args, "simulation_dt", None) or os.getenv("SIMULATION_DT", "0.08"))
    except Exception:
        sim_dt = 0.08
    gazebo_step_iterations = os.getenv("GAZEBO_LIVE_STEP_ITERATIONS", "0")
    default_wall_time_step_ms = (
        str(max(1, int(round(float(sim_dt) * 1000.0))))
        if str(gazebo_step_iterations).strip() in ("", "0")
        else "0"
    )
    default_command_sleep = "0.0"
    consistency_mode = os.getenv("GAZEBO_LIVE_CONSISTENCY_MODE", "gazebo_authoritative").strip().lower()
    if consistency_mode in ("python", "python_only", "python_authoritative", "mirror", "follow_python"):
        consistency_mode = "python_authoritative"
    else:
        consistency_mode = "gazebo_authoritative"
    python_authoritative = consistency_mode == "python_authoritative"
    common = {
        "EVAL_BACKEND": backend,
        "GAZEBO_LIVE_VALIDATION": "1",
        "GAZEBO_LIVE_VALIDATION_DIR": validation_dir,
        "SAVE_EVAL_TRAJECTORY_JSON": "1",
        "EVAL_EPISODE_PARALLELISM": "1",
        "EVAL_ENV_STEP_THREADS": "1",
    }
    if backend == "gazebo_live":
        common.update(
            {
                "GAZEBO_LIVE_SYNC": "1",
                "GAZEBO_LIVE_AUTOLAUNCH": "1",
                "GAZEBO_LIVE_AUTOLAUNCH_RUN": os.getenv("GAZEBO_LIVE_AUTOLAUNCH_RUN", "0"),
                "GAZEBO_LIVE_SEMANTIC_MODE": os.getenv("GAZEBO_LIVE_SEMANTIC_MODE", "transfer_equivalence"),
                "GAZEBO_LIVE_CONSISTENCY_MODE": consistency_mode,
                "GAZEBO_LIVE_CONTROL_MODE": "velocity",
                "GAZEBO_LIVE_STATE_FEEDBACK": "1",
                "GAZEBO_LIVE_FEEDBACK_VELOCITY_MODE": os.getenv("GAZEBO_LIVE_FEEDBACK_VELOCITY_MODE", "clamp"),
                "GAZEBO_LIVE_FEEDBACK_ACCELERATION_MODE": os.getenv("GAZEBO_LIVE_FEEDBACK_ACCELERATION_MODE", "estimate"),
                "GAZEBO_LIVE_AUTHORITATIVE_FEEDBACK": os.getenv(
                    "GAZEBO_LIVE_AUTHORITATIVE_FEEDBACK",
                    "0" if python_authoritative else "1",
                ),
                "GAZEBO_LIVE_CONTACT_FEEDBACK": "1",
                "GAZEBO_LIVE_CONTACT_AUTHORITATIVE": os.getenv(
                    "GAZEBO_LIVE_CONTACT_AUTHORITATIVE",
                    "0",
                ),
                "GAZEBO_LIVE_CONTACT_MARKS_COLLISION": os.getenv(
                    "GAZEBO_LIVE_CONTACT_MARKS_COLLISION",
                    "0",
                ),
                "GAZEBO_LIVE_CONTACT_TERMINATES": os.getenv("GAZEBO_LIVE_CONTACT_TERMINATES", "0"),
                "GAZEBO_LIVE_POSE_CORRECTION": os.getenv(
                    "GAZEBO_LIVE_POSE_CORRECTION",
                    "1" if python_authoritative else "0",
                ),
                "GAZEBO_LIVE_STEP_ITERATIONS": gazebo_step_iterations,
                "GAZEBO_LIVE_COMMAND_SLEEP": os.getenv("GAZEBO_LIVE_COMMAND_SLEEP", default_command_sleep),
                "GAZEBO_LIVE_WALL_TIME_STEP_MS": os.getenv("GAZEBO_LIVE_WALL_TIME_STEP_MS", default_wall_time_step_ms),
                "GAZEBO_LIVE_WAIT_ACK": os.getenv("GAZEBO_LIVE_WAIT_ACK", "1"),
                "GAZEBO_LIVE_ACK_TIMEOUT": os.getenv("GAZEBO_LIVE_ACK_TIMEOUT", "2.0"),
                "GAZEBO_LIVE_PAUSE_FOR_STEP": os.getenv("GAZEBO_LIVE_PAUSE_FOR_STEP", "0"),
                "GAZEBO_LIVE_PRE_STEP_SLEEP_MS": os.getenv("GAZEBO_LIVE_PRE_STEP_SLEEP_MS", "20"),
                "GAZEBO_LIVE_POST_STEP_SLEEP_MS": os.getenv("GAZEBO_LIVE_POST_STEP_SLEEP_MS", "20"),
                "GAZEBO_LIVE_STATE_FEEDBACK_DT": os.getenv("GAZEBO_LIVE_STATE_FEEDBACK_DT", str(sim_dt)),
                "GAZEBO_LIVE_SCENE_CHECK_REQUIRED": "1",
                "EVAL_EPISODE_PARALLELISM": "1",
                "EVAL_ENV_STEP_THREADS": "1",
            }
        )
    else:
        common.update(
            {
                "GAZEBO_LIVE_SYNC": "0",
                "GAZEBO_LIVE_AUTOLAUNCH": "0",
                "GAZEBO_LIVE_CONTROL_MODE": "pose",
            }
        )
    return common


def _ensure_paired_seed_env(args):
    updates = {}
    episode_count = max(1, int(getattr(args, "eval_episodes", 1) or 1))
    try:
        base_seed = int(
            getattr(args, "terrain_seed", None)
            if getattr(args, "terrain_seed", None) is not None
            else os.getenv("SCENARIO_SEED", os.getenv("TERRAIN_BASE_SEED", "88"))
        )
    except Exception:
        base_seed = 88

    random_terrain_requested = (
        bool(getattr(args, "random_terrain", False))
        or os.getenv("RANDOM_TERRAIN", "0").lower() in ("1", "true", "yes", "on")
        or bool(os.getenv("TERRAIN_SEED_SEQUENCE", "").strip())
    )
    if random_terrain_requested and not os.getenv("TERRAIN_SEED_SEQUENCE", "").strip():
        updates["TERRAIN_SEED_SEQUENCE"] = ",".join(str(base_seed + idx) for idx in range(episode_count))
    if getattr(args, "terrain_complexity_level", None) is None and not os.getenv("TERRAIN_COMPLEXITY_LEVEL_SEQUENCE", "").strip():
        rng = np.random.default_rng(base_seed + 20000)
        updates["TERRAIN_COMPLEXITY_LEVEL_SEQUENCE"] = ",".join(
            str(int(v)) for v in rng.integers(1, 5, size=episode_count)
        )
    auto_obstacle_sequence = os.getenv(
        "GAZEBO_LIVE_AUTO_OBSTACLE_SEED_SEQUENCE",
        os.getenv("VALIDATION_AUTO_OBSTACLE_SEED_SEQUENCE", "0"),
    ).strip().lower() in ("1", "true", "yes", "on")
    if auto_obstacle_sequence and not os.getenv("OBSTACLE_SEED_SEQUENCE", "").strip():
        updates["OBSTACLE_SEED_SEQUENCE"] = ",".join(str(base_seed + 10000 + idx) for idx in range(episode_count))
    updates.setdefault("SCENARIO_SEED", str(base_seed))
    updates.setdefault("TERRAIN_BASE_SEED", str(base_seed))
    return updates


def _validation_base_seed(args):
    try:
        return int(os.getenv("GAZEBO_LIVE_VALIDATION_RANDOM_SEED", "").strip())
    except Exception:
        pass
    try:
        if getattr(args, "terrain_seed", None) is not None:
            return int(getattr(args, "terrain_seed"))
    except Exception:
        pass
    try:
        return int(os.getenv("SCENARIO_SEED", os.getenv("TERRAIN_BASE_SEED", "88")))
    except Exception:
        return 88


def _reset_validation_random_state(args):
    seed = int(_validation_base_seed(args)) % (2**32)
    try:
        import random as _random
        _random.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass
    return seed


def _run_single_backend_evaluation(args, backend, save_viz_path=None):
    run_args = copy.deepcopy(args)
    run_args.eval_backend = backend
    if save_viz_path is not None:
        run_args.save_viz_path = str(save_viz_path)
    run_args.eval_episode_parallelism = 1
    run_args.eval_env_step_threads = 1
    with _temporary_env(_validation_backend_env(run_args, backend)):
        _reset_validation_random_state(run_args)
        evaluator = ModelEvaluator(run_args)
        return evaluator.run_evaluation()


def _run_paired_backend_evaluation(args):
    from gazebo_live_validation import write_validation_outputs

    validation_root = Path(getattr(args, "validation_output_dir", "results/gazebo_live_validation")).expanduser().resolve()
    validation_root.mkdir(parents=True, exist_ok=True)
    seed_updates = _ensure_paired_seed_env(args)
    results_by_backend = {}
    with _temporary_env(seed_updates):
        for backend in ("python_only", "gazebo_live"):
            backend_dir = validation_root / backend
            backend_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[GazeboLiveValidation] running backend={backend}, output={backend_dir}")
            results_by_backend[backend] = _run_single_backend_evaluation(
                args,
                backend,
                save_viz_path=backend_dir,
            )
    validation = write_validation_outputs(results_by_backend=results_by_backend, output_dir=validation_root)
    print(f"[GazeboLiveValidation] episode_metrics={validation_root / 'episode_metrics.csv'}")
    print(f"[GazeboLiveValidation] summary={validation_root / 'summary.json'}")
    print(f"[GazeboLiveValidation] difference_cases={validation_root / 'difference_cases.json'}")
    return {
        "results_by_backend": results_by_backend,
        "validation": validation,
    }


def _split_eval_episode_shards(start_index, episode_count, shard_count):
    shard_count = max(1, min(int(shard_count), int(episode_count)))
    base_count = int(episode_count) // shard_count
    remainder = int(episode_count) % shard_count
    shards = []
    cursor = int(start_index)
    for shard_idx in range(shard_count):
        count = base_count + (1 if shard_idx < remainder else 0)
        if count <= 0:
            continue
        shards.append(
            {
                "index": int(shard_idx),
                "start": int(cursor),
                "count": int(count),
            }
        )
        cursor += count
    return shards


def _append_arg_override(argv, name, value):
    return list(argv) + [str(name), str(value)]


def _eval_shard_worker_tuning(args, shard_count):
    total_episode_parallelism = max(1, int(getattr(args, "eval_episode_parallelism", 1) or 1))
    total_env_threads = max(1, int(getattr(args, "eval_env_step_threads", 1) or 1))
    requested_parallelism = int(getattr(args, "eval_shard_episode_parallelism", 0) or 0)
    requested_threads = int(getattr(args, "eval_shard_env_step_threads", 0) or 0)
    if requested_parallelism > 0:
        worker_parallelism = requested_parallelism
    else:
        worker_parallelism = max(1, int(math.ceil(float(total_episode_parallelism) / max(1, shard_count))))
    if requested_threads > 0:
        worker_threads = requested_threads
    else:
        worker_threads = max(1, int(math.ceil(float(total_env_threads) / max(1, shard_count))))
    worker_threads = min(worker_threads, worker_parallelism)
    return int(worker_parallelism), int(worker_threads)


def _merge_process_shard_results(args, shard_specs, shard_root, output_dir, started_at):
    shard_root = Path(shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_results = []
    for spec in shard_specs:
        shard_dir = Path(spec["dir"])
        result_path = shard_dir / "evaluation_results.json"
        if not result_path.exists():
            raise FileNotFoundError(f"分片结果不存在: {result_path}")
        with result_path.open("r", encoding="utf-8") as f:
            shard_result = json.load(f)
        expected_shard_episodes = list(
            range(int(spec["start"]), int(spec["start"]) + int(spec["count"]))
        )
        shard_details = shard_result.get("episode_details", []) or []
        try:
            actual_shard_episodes = [int(ep.get("episode")) for ep in shard_details]
        except Exception as exc:
            raise RuntimeError(f"分片 episode_details 无法解析: {result_path} | {exc}") from exc
        if sorted(actual_shard_episodes) != expected_shard_episodes:
            raise RuntimeError(
                f"分片 episode 范围不完整: {result_path} | "
                f"got={actual_shard_episodes}, expected={expected_shard_episodes}"
            )
        try:
            recorded_shard_count = int(shard_result.get("episodes", 0) or 0)
        except Exception:
            recorded_shard_count = 0
        if recorded_shard_count != int(spec["count"]):
            raise RuntimeError(
                f"分片 episodes 不匹配: {result_path} | "
                f"got={recorded_shard_count}, expected={spec['count']}"
            )
        shard_results.append(
            {
                "spec": dict(spec),
                "path": str(result_path),
                "result": shard_result,
            }
        )

    details_by_episode = {}
    for item in shard_results:
        for ep in item["result"].get("episode_details", []) or []:
            try:
                episode_idx = int(ep.get("episode"))
            except Exception:
                continue
            if episode_idx in details_by_episode:
                raise RuntimeError(f"分片结果中出现重复episode: {episode_idx}")
            details_by_episode[episode_idx] = ep

    expected_start = max(0, _env_int("EVAL_EPISODE_START_INDEX", 0))
    expected_count = max(1, int(getattr(args, "eval_episodes", 1) or 1))
    expected_episodes = list(range(expected_start, expected_start + expected_count))
    missing_episodes = [ep for ep in expected_episodes if ep not in details_by_episode]
    if missing_episodes:
        raise RuntimeError(f"分片结果缺少episode: {missing_episodes}")

    episode_details = [details_by_episode[ep] for ep in expected_episodes]
    all_rewards = [float(ep.get("reward", 0.0) or 0.0) for ep in episode_details]
    summary = _build_evaluation_summary(
        all_rewards,
        episode_details,
        getattr(args, "collision_distance_threshold", None),
    )

    template = copy.deepcopy(shard_results[0]["result"]) if shard_results else {}
    evaluation_setup = copy.deepcopy(template.get("evaluation_setup", {}) or {})
    evaluation_setup["eval_episode_parallelism_mode"] = "process_shards"
    evaluation_setup["eval_process_shards"] = int(len(shard_specs))
    evaluation_setup["eval_process_workers"] = int(getattr(args, "eval_process_workers", 0) or len(shard_specs))
    evaluation_setup["eval_process_shard_root"] = str(shard_root)
    evaluation_setup["eval_process_started_at"] = started_at
    evaluation_setup["eval_process_finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    evaluation_setup["eval_process_shard_specs"] = [
        {
            "index": int(spec["index"]),
            "start": int(spec["start"]),
            "count": int(spec["count"]),
            "dir": str(spec["dir"]),
            "log": str(spec["log"]),
            "eval_noise_seed": spec.get("eval_noise_seed"),
        }
        for spec in shard_specs
    ]
    evaluation_setup["eval_noise_seed_base"] = getattr(args, "eval_noise_seed", None)
    try:
        evaluation_setup["eval_episode_parallelism"] = int(getattr(args, "eval_episode_parallelism", 1) or 1)
        evaluation_setup["eval_env_step_threads"] = int(getattr(args, "eval_env_step_threads", 1) or 1)
    except Exception:
        pass

    merged = copy.deepcopy(template)
    merged["model_path"] = getattr(args, "load_model_path", template.get("model_path", None))
    merged["scenario"] = getattr(args, "scenario_name", template.get("scenario", None))
    merged["episodes"] = int(len(episode_details))
    merged["avg_reward"] = float(np.mean(all_rewards)) if all_rewards else None
    merged["std_reward"] = float(np.std(all_rewards)) if all_rewards else None
    merged["max_reward"] = float(np.max(all_rewards)) if all_rewards else None
    merged["min_reward"] = float(np.min(all_rewards)) if all_rewards else None
    merged["all_rewards"] = all_rewards
    merged["summary"] = summary
    merged["evaluation_setup"] = evaluation_setup
    merged["episode_details"] = episode_details
    merged["terrain_complexity_level_sequence"] = [
        ep.get("terrain_complexity_level") for ep in episode_details
    ]
    merged["terrain_seed_sequence"] = [ep.get("terrain_seed") for ep in episode_details]
    merged["terrain_variant_seed_sequence"] = [
        ep.get("terrain_variant_seed") for ep in episode_details
    ]
    merged["obstacle_seed_sequence"] = [
        ep.get("obstacle_seed") for ep in episode_details
    ]

    def _resolve_shard_artifact(value, shard_dir):
        if value is None or not str(value).strip():
            return None
        candidate = Path(str(value))
        if candidate.is_absolute() and candidate.exists():
            return str(candidate)
        direct = Path(shard_dir) / candidate
        if direct.exists():
            return str(direct.resolve())
        by_name = Path(shard_dir) / candidate.name
        if by_name.exists():
            return str(by_name.resolve())
        return str(candidate)

    episode_visualizations = []
    for item in shard_results:
        shard_dir = Path(item["spec"]["dir"])
        shard_artifacts = item["result"].get("visualization_artifacts", {})
        if not isinstance(shard_artifacts, dict):
            continue
        for entry in shard_artifacts.get("episode_visualizations", []) or []:
            if not isinstance(entry, dict):
                continue
            normalized_entry = copy.deepcopy(entry)
            files = normalized_entry.get("files", {})
            if isinstance(files, dict):
                normalized_entry["files"] = {
                    key: _resolve_shard_artifact(value, shard_dir)
                    for key, value in files.items()
                }
            episode_visualizations.append(normalized_entry)
    episode_visualizations.sort(key=lambda entry: int(entry.get("episode", -1)))

    visualization_artifacts = {
        "process_shard_root": str(shard_root),
        "process_shard_results": [
            {
                "index": int(item["spec"]["index"]),
                "start": int(item["spec"]["start"]),
                "count": int(item["spec"]["count"]),
                "result": item["path"],
            }
            for item in shard_results
        ],
        "episode_visualizations": episode_visualizations,
    }

    def _shard_for_episode(episode_idx):
        for item in shard_results:
            start = int(item["spec"]["start"])
            stop = start + int(item["spec"]["count"])
            if start <= int(episode_idx) < stop:
                return item
        return None

    best_reward_episode = max(
        episode_details,
        key=lambda ep: float(ep.get("reward", -float("inf"))),
    )
    best_reward_shard = _shard_for_episode(best_reward_episode["episode"])
    best_success_candidates = [ep for ep in episode_details if int(ep.get("team_success", 0) or 0) == 1]
    best_success_shard = None
    if best_success_candidates:
        best_success_episode = max(
            best_success_candidates,
            key=lambda ep: float(ep.get("reward", -float("inf"))),
        )
        best_success_shard = _shard_for_episode(best_success_episode["episode"])

    def _copy_selected_artifacts(item, prefix):
        if item is None:
            return
        shard_artifacts = item["result"].get("visualization_artifacts", {})
        if not isinstance(shard_artifacts, dict):
            return
        shard_dir = Path(item["spec"]["dir"])
        for key, value in shard_artifacts.items():
            if key.startswith(prefix):
                visualization_artifacts[key] = _resolve_shard_artifact(value, shard_dir)

    _copy_selected_artifacts(best_reward_shard, "best_reward_")
    _copy_selected_artifacts(best_success_shard, "team_success_best_")
    merged["visualization_artifacts"] = visualization_artifacts
    merged["evaluation_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "evaluation_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe_eval_value(merged), f, indent=2, ensure_ascii=False)
    print(f"✅ 分片评估结果已合并: {results_path}")

    index_path = shard_root / "shard_results_index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(
            _json_safe_eval_value(
                {
                    "output_results": str(results_path),
                    "shards": evaluation_setup["eval_process_shard_specs"],
                    "summary": summary,
                }
            ),
            f,
            indent=2,
            ensure_ascii=False,
        )
    return merged


def _terminate_process_shard_workers(active_workers):
    for active in active_workers:
        proc = active.get("proc")
        if proc is not None and proc.poll() is None:
            proc.terminate()
    for active in active_workers:
        proc = active.get("proc")
        if proc is not None and proc.poll() is None:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        log_fh = active.get("log_fh")
        if log_fh is not None and not log_fh.closed:
            log_fh.close()


def _run_process_sharded_evaluation(args):
    if os.getenv("MATD3_EVAL_PROCESS_SHARD_WORKER") == "1":
        return None
    shard_count = int(getattr(args, "eval_process_shards", 1) or 1)
    if shard_count <= 1:
        return None
    requested_backend = getattr(args, "eval_backend", None)
    if requested_backend in ("both", "gazebo_live"):
        raise RuntimeError("--eval-process-shards 暂不支持 gazebo_live/both 后端")

    eval_episode_count = max(1, int(getattr(args, "eval_episodes", 1) or 1))
    eval_episode_start = max(0, _env_int("EVAL_EPISODE_START_INDEX", 0))
    shard_specs = _split_eval_episode_shards(eval_episode_start, eval_episode_count, shard_count)
    shard_count = len(shard_specs)
    process_workers = int(getattr(args, "eval_process_workers", 0) or 0)
    if process_workers <= 0:
        process_workers = shard_count
    process_workers = max(1, min(process_workers, shard_count))
    worker_parallelism, worker_threads = _eval_shard_worker_tuning(args, shard_count)

    output_dir = Path(getattr(args, "save_viz_path", "evaluation_results")).expanduser().resolve()
    shard_root = output_dir / "_episode_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    print(
        "[EvalProcessShards] "
        f"shards={shard_count}, workers={process_workers}, "
        f"worker_episode_parallelism={worker_parallelism}, worker_env_step_threads={worker_threads}, "
        f"episode_range={eval_episode_start}-{eval_episode_start + eval_episode_count - 1}"
    )

    base_argv = list(sys.argv[1:])
    script_path = Path(__file__).resolve()
    pending = []
    for spec in shard_specs:
        shard_dir = shard_root / f"shard_{spec['index']:02d}_start{spec['start']:03d}_count{spec['count']:03d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        log_path = shard_dir / "worker.log"
        worker_argv = list(base_argv)
        worker_argv = _append_arg_override(worker_argv, "--eval-process-shards", 1)
        worker_argv = _append_arg_override(worker_argv, "--eval-process-workers", 1)
        worker_argv = _append_arg_override(worker_argv, "--eval-episodes", spec["count"])
        worker_argv = _append_arg_override(worker_argv, "--eval-episode-parallelism", min(worker_parallelism, spec["count"]))
        worker_argv = _append_arg_override(worker_argv, "--eval-env-step-threads", min(worker_threads, spec["count"]))
        worker_argv = _append_arg_override(worker_argv, "--save-viz-path", str(shard_dir))
        base_noise_seed = getattr(args, "eval_noise_seed", None)
        if base_noise_seed is not None:
            # Every worker receives the same base.  The evaluator folds the
            # global episode index into independent per-episode PCG64 streams,
            # so changing shard boundaries cannot change action perturbations.
            worker_argv = _append_arg_override(worker_argv, "--eval-noise-seed", int(base_noise_seed))
            spec["eval_noise_seed"] = int(base_noise_seed)
        command = [sys.executable, str(script_path)] + worker_argv
        worker_env = os.environ.copy()
        worker_env["MATD3_EVAL_PROCESS_SHARD_WORKER"] = "1"
        worker_env["EVAL_EPISODE_START_INDEX"] = str(spec["start"])
        worker_env["EVAL_PROCESS_SHARD_INDEX"] = str(spec["index"])
        worker_env["EVAL_PROCESS_SHARD_COUNT"] = str(shard_count)
        worker_env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        worker_env.setdefault("OMP_NUM_THREADS", str(max(1, worker_threads)))
        worker_env.setdefault("TF_NUM_INTRAOP_THREADS", str(max(1, worker_threads)))
        worker_env.setdefault("TF_NUM_INTEROP_THREADS", "1")
        spec["dir"] = str(shard_dir)
        spec["log"] = str(log_path)
        pending.append(
            {
                "spec": spec,
                "command": command,
                "env": worker_env,
                "log_path": log_path,
            }
        )

    running = []
    launched = 0
    try:
        while pending or running:
            while pending and len(running) < process_workers:
                item = pending.pop(0)
                log_fh = open(item["log_path"], "w", encoding="utf-8")
                proc = subprocess.Popen(
                    item["command"],
                    cwd=str(Path.cwd()),
                    env=item["env"],
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                launched += 1
                print(
                    "[EvalProcessShards] launched "
                    f"shard={item['spec']['index']} start={item['spec']['start']} "
                    f"count={item['spec']['count']} pid={proc.pid} log={item['log_path']}"
                )
                running.append({"item": item, "proc": proc, "log_fh": log_fh})

            time.sleep(2.0)
            still_running = []
            failed_active = None
            for active in running:
                ret = active["proc"].poll()
                if ret is None:
                    still_running.append(active)
                    continue
                active["log_fh"].close()
                spec = active["item"]["spec"]
                if ret != 0:
                    failed_active = active
                    break
                print(
                    "[EvalProcessShards] finished "
                    f"shard={spec['index']} start={spec['start']} count={spec['count']}"
                )
            if failed_active is not None:
                failed_spec = failed_active["item"]["spec"]
                _terminate_process_shard_workers(running)
                running = []
                raise RuntimeError(
                    f"评估分片失败: shard={failed_spec['index']} "
                    f"retcode={failed_active['proc'].returncode} "
                    f"log={failed_active['item']['log_path']}"
                )
            running = still_running
    except KeyboardInterrupt:
        _terminate_process_shard_workers(running)
        running = []
        raise
    finally:
        if running:
            _terminate_process_shard_workers(running)

    if launched != len(shard_specs):
        raise RuntimeError(f"分片启动数量异常: launched={launched}, expected={len(shard_specs)}")
    return _merge_process_shard_results(args, shard_specs, shard_root, output_dir, started_at)


def main():
    """主函数"""
    _apply_fast_artifact_env_defaults()
    args = parse_args()
    try:
        _validate_load_model_path_arg(args)
    except ValueError as e:
        print(f"\n❌ 参数错误: {e}")
        sys.exit(2)
    if bool(getattr(args, "config_resolve_only", False)):
        alignment = _load_training_alignment_snapshot(getattr(args, 'load_model_path', None))
        if alignment:
            _apply_training_alignment_to_args(args, alignment, quiet=True)
        _apply_runtime_env_overrides_from_args(args)
        resolved_payload = {
            "schema_version": 1,
            "mode": "config_resolve_only",
            "args": dict(vars(args)),
            "training_results_path": alignment.get("results_path") if alignment else None,
            "training_runtime_manifest_path": (
                alignment.get("training_runtime_manifest_path") if alignment else None
            ),
        }
        print(
            "RESOLVED_EVAL_CONFIG_JSON="
            + json.dumps(resolved_payload, ensure_ascii=False, sort_keys=True, default=str)
        )
        return
    if bool(getattr(args, "gazebo_live_gui", False)):
        os.environ["GAZEBO_LIVE_AUTOLAUNCH_GUI"] = "1"
    if bool(getattr(args, "gazebo_live_gui_required", False)):
        os.environ["GAZEBO_LIVE_GUI_REQUIRED"] = "1"
    disable_viz_env = _env_flag("EVAL_DISABLE_VISUALIZATION", False)
    light_mode_env = _env_flag("EVAL_LIGHT_MODE", False)
    save_interactive_traj = _env_flag("SAVE_INTERACTIVE_TRAJ", True)
    save_all_episode_visualizations = _env_flag("SAVE_EVAL_ALL_EPISODES", False)
    save_best_traj = _env_flag("SAVE_BEST_TRAJ", True)
    save_trajectory_png = _env_flag("SAVE_EVAL_TRAJECTORY_PNG", True)
    save_team_success_html = _env_flag("SAVE_TEAM_SUCCESS_HTML", False)
    save_actor_sequence = _env_flag("SAVE_EVAL_ACTOR_SEQUENCE", False)
    save_control_diagnostics = _env_flag("SAVE_EVAL_CONTROL_DIAGNOSTICS", False)
    save_gazebo_replay = _env_flag("SAVE_GAZEBO_REPLAY", False)
    save_gazebo_dynamic_replay = _env_flag("SAVE_GAZEBO_DYNAMIC_REPLAY", False)
    save_trajectory_snapshot = _env_flag("SAVE_TRAJECTORY_SNAPSHOT", save_gazebo_replay or save_gazebo_dynamic_replay)
    needs_interactive_visualization = save_interactive_traj and (
        save_all_episode_visualizations
        or save_best_traj
        or save_team_success_html
    )
    keep_viz_artifacts_in_light_mode = (
        needs_interactive_visualization
        or save_trajectory_png
        or save_team_success_html
        or save_actor_sequence
        or save_control_diagnostics
        or save_gazebo_replay
        or save_gazebo_dynamic_replay
        or save_trajectory_snapshot
    )
    if disable_viz_env or (light_mode_env and not keep_viz_artifacts_in_light_mode):
        args.disable_visualization = True
        args.disable_gif = True
        setattr(args, 'disable_html', True)
    elif light_mode_env:
        args.disable_gif = True

    noise_active = (
        float(getattr(args, 'eval_noise_scale', 0.0) or 0.0) > 0.0
        or float(getattr(args, 'eval_random_action_prob', 0.0) or 0.0) > 0.0
    )
    if noise_active and getattr(args, 'eval_noise_seed', None) is None:
        args.eval_noise_seed = int(time.time_ns() % (2**31 - 1))
        print(f"[EvalNoise] 未指定seed，已生成并记录基础seed={args.eval_noise_seed}")

    sharded_results = _run_process_sharded_evaluation(args)
    if sharded_results is not None:
        print("\n🎉 分片评估完成!")
        return

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

    # 设置随机种子；噪声对照实验可通过 --eval-noise-seed 固定随机流
    import time
    import random
    eval_noise_seed = getattr(args, 'eval_noise_seed', None)
    if eval_noise_seed is not None:
        current_seed = int(eval_noise_seed) % 2**32
        seed_reason = "固定评估噪声/随机流"
    else:
        current_seed = int(time.time() * 1000000) % 2**32
        seed_reason = "确保每次评估的随机性"
    random.seed(current_seed)
    np.random.seed(current_seed)
    tf.random.set_seed(current_seed)
    print(f"🎲 设置随机种子: {current_seed} ({seed_reason})")
    
    # 显示HTML生成状态
    enable_html = getattr(args, 'enable_html', True) and not getattr(args, 'disable_html', False)
    if enable_html:
        print("🌐 HTML交互式轨迹图生成: 启用")
        print("💡 提示: 如果HTML生成失败，请安装plotly: pip install plotly")
    else:
        print("🌐 HTML交互式轨迹图生成: 禁用")
    
    try:
        requested_backend = getattr(args, "eval_backend", None)
        if requested_backend == "both":
            results = _run_paired_backend_evaluation(args)
        elif requested_backend in ("python_only", "gazebo_live"):
            results = _run_single_backend_evaluation(args, requested_backend)
            try:
                from gazebo_live_validation import write_validation_outputs, validation_root_from_args
                validation_root = validation_root_from_args(args)
                write_validation_outputs(
                    results_by_backend={requested_backend: results},
                    output_dir=validation_root,
                )
                print(f"[GazeboLiveValidation] episode_metrics={validation_root / 'episode_metrics.csv'}")
                print(f"[GazeboLiveValidation] summary={validation_root / 'summary.json'}")
            except Exception as validation_err:
                print(f"⚠️ Gazebo-live validation summary failed: {validation_err}")
        else:
            evaluator = ModelEvaluator(args)
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
        raise
    except Exception as e:
        print(f"\n❌ 评估出错: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
