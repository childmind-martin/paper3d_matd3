#!/usr/bin/env python3
"""Recover a completed training unit after the known results-postamble failure.

This command is deliberately narrow.  It only accepts a frozen immutable
manifest whose exact model identity has:

* a complete ``episode_rewards.json`` for the requested horizon,
* a readable non-empty ``loss_history.json``,
* complete and readable final MATD3 weights,
* launcher-log proof that the matching run reached N/N on GPU, closed the
  environment, and then failed with the known ``training_device_info``
  NameError while constructing ``results.json``.

It does not resume training and it refuses to overwrite an existing result.
Without ``--apply`` it performs the full audit but writes nothing.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

KNOWN_POSTAMBLE_ERROR = "NameError: name 'training_device_info' is not defined"
RECOVERY_SCHEMA_VERSION = 1
PER_EPISODE_KEYS = (
    "episode_rewards",
    "collision_counts",
    "min_distances_to_obstacle",
    "noise_scale_var_history",
    "actual_noise_std_history",
    "success_flags",
    "agent_success_flags",
    "team_success_flags",
    "episode_rewards_per_env",
    "team_success_flags_per_env",
    "agent_success_flags_per_env",
    "collision_counts_per_env",
    "agent_collision_counts_per_env",
    "cross_ref_labeled_env_counts",
    "cross_ref_labeled_transition_counts",
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    replace_existing_recovery: bool = False,
) -> None:
    if path.exists() and not replace_existing_recovery:
        raise FileExistsError(f"拒绝覆盖已有结果: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.recovery-{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _parse_training_device_evidence(
    launcher_text: str,
    *,
    launcher_path: Path,
    log_dir: Path,
    repo_root: Path,
    expected_episodes: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Bind the last matching GPU device line to the exact completed run."""
    normalized_text = launcher_text.replace("\r\n", "\n").replace("\r", "\n")
    relative_log_dir = _relative_or_absolute(log_dir, repo_root)
    completion_marker = f"可视化输出目录: {relative_log_dir}"
    completion_offset = normalized_text.rfind(completion_marker)
    if completion_offset < 0:
        raise RuntimeError(
            f"launcher 日志没有匹配当前 run_dir 的完成标记: {completion_marker}"
        )

    device_marker = "[Train Device]"
    device_offset = normalized_text.rfind(
        device_marker,
        0,
        completion_offset,
    )
    if device_offset < 0:
        raise RuntimeError("launcher 日志在当前 run 前没有 [Train Device] 记录")
    next_device_offset = normalized_text.find(
        device_marker,
        device_offset + len(device_marker),
        completion_offset,
    )
    if next_device_offset >= 0:
        raise RuntimeError("当前 run 的设备证据边界不唯一")

    line_end = normalized_text.find("\n", device_offset)
    if line_end < 0:
        line_end = len(normalized_text)
    device_line = normalized_text[device_offset:line_end].strip()
    match = re.fullmatch(
        r"\[Train Device\] "
        r"python=(?P<python>.*?) \| "
        r"CUDA_VISIBLE_DEVICES=(?P<cuda>.*?) \| "
        r"physical_gpus=(?P<physical>\d+) \| "
        r"logical_gpus=(?P<logical>\d+) \| "
        r"require_gpu=(?P<required>True|False)",
        device_line,
    )
    if match is None:
        raise RuntimeError(f"无法解析训练设备记录: {device_line}")

    run_segment = normalized_text[device_offset:]
    progress_pattern = re.compile(
        rf"(?:训练进度 .*?|回合 ){int(expected_episodes)}/"
        rf"{int(expected_episodes)}"
    )
    if progress_pattern.search(run_segment) is None:
        raise RuntimeError(
            f"当前 run 没有达到 {expected_episodes}/{expected_episodes}"
        )
    if KNOWN_POSTAMBLE_ERROR not in run_segment:
        raise RuntimeError("当前 run 不是已知的 results postamble NameError")
    error_offset = run_segment.find(KNOWN_POSTAMBLE_ERROR)
    relative_completion_offset = completion_offset - device_offset
    if error_offset <= relative_completion_offset:
        raise RuntimeError("已知 NameError 不是发生在当前 run 完成标记之后")
    if "训练出错:" in run_segment:
        training_error_lines = [
            line.strip()
            for line in run_segment.splitlines()
            if line.strip().startswith("训练出错:")
        ]
        expected_training_error = (
            "训练出错: name 'training_device_info' is not defined"
        )
        if training_error_lines != [expected_training_error]:
            raise RuntimeError(
                "当前 run 除已知 postamble NameError 外还有其他训练错误: "
                + repr(training_error_lines)
            )

    actor_input_matches = re.findall(
        r"\[MATD3网络\] 智能体(?P<agent>\d+) - "
        r"Actor输入: (?P<obs_dim>\d+),",
        run_segment,
    )
    actor_obs_dims: Dict[int, int] = {}
    for agent_text, obs_dim_text in actor_input_matches:
        agent_index = int(agent_text)
        obs_dim = int(obs_dim_text)
        if (
            agent_index in actor_obs_dims
            and actor_obs_dims[agent_index] != obs_dim
        ):
            raise RuntimeError("当前 run 的 Actor 输入维度记录不一致")
        actor_obs_dims[agent_index] = obs_dim
    expected_agent_indices = list(range(len(actor_obs_dims)))
    if sorted(actor_obs_dims) != expected_agent_indices or not actor_obs_dims:
        raise RuntimeError(
            "当前 run 缺少连续的逐 agent Actor 输入维度记录"
        )
    replay_obs_matches = [
        int(value)
        for value in re.findall(r"- 观察维度: (\d+)", run_segment)
    ]
    replay_action_matches = [
        int(value)
        for value in re.findall(r"- 动作维度: (\d+)", run_segment)
    ]
    if len(set(replay_obs_matches)) != 1 or not replay_obs_matches:
        raise RuntimeError("当前 run 的 ReplayBuffer 观察维度证据不唯一")
    if len(set(replay_action_matches)) != 1 or not replay_action_matches:
        raise RuntimeError("当前 run 的 ReplayBuffer 动作维度证据不唯一")
    replay_obs_dim = replay_obs_matches[0]
    replay_action_dim = replay_action_matches[0]
    base_obs_shapes = [
        actor_obs_dims[index] for index in expected_agent_indices
    ]
    if any(value != replay_obs_dim for value in base_obs_shapes):
        raise RuntimeError("Actor 与 ReplayBuffer 的观察维度记录不一致")
    base_action_dims = [
        replay_action_dim for _ in expected_agent_indices
    ]

    physical_gpus = int(match.group("physical"))
    logical_gpus = int(match.group("logical"))
    require_gpu = match.group("required") == "True"
    if not require_gpu or physical_gpus < 1 or logical_gpus < 1:
        raise RuntimeError(
            "launcher 日志不能证明 MATD3_REQUIRE_GPU=1 下的物理和逻辑 GPU"
        )

    device_info = {
        "python": match.group("python"),
        "cuda_visible_devices": match.group("cuda"),
        "physical_gpus": physical_gpus,
        "logical_gpus": logical_gpus,
        "configure_gpu": "ok",
        "require_gpu": True,
    }
    line_number = normalized_text.count("\n", 0, device_offset) + 1
    evidence = {
        "launcher_log": str(launcher_path.resolve()),
        "launcher_log_sha256": _file_sha256(launcher_path),
        "training_device_line": int(line_number),
        "training_device_record": device_line,
        "completed_run_dir": str(log_dir.resolve()),
        "completed_progress": (
            f"{int(expected_episodes)}/{int(expected_episodes)}"
        ),
        "terminal_error": KNOWN_POSTAMBLE_ERROR,
        "base_obs_shapes": base_obs_shapes,
        "base_action_dims": base_action_dims,
    }
    return device_info, evidence


def _validate_metrics(
    metrics: Mapping[str, Any],
    *,
    log_dir: Path,
    expected_episodes: int,
    expected_num_envs: int,
) -> List[float]:
    if not isinstance(metrics, Mapping):
        raise RuntimeError("episode_rewards.json 根节点不是对象")
    rewards = metrics.get("episode_rewards")
    if not isinstance(rewards, list) or len(rewards) != expected_episodes:
        raise RuntimeError(
            "episode_rewards 长度不匹配: "
            f"got={len(rewards) if isinstance(rewards, list) else 'invalid'}, "
            f"expected={expected_episodes}"
        )
    try:
        normalized_rewards = [float(value) for value in rewards]
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("episode_rewards 包含非数值") from exc
    if not all(math.isfinite(value) for value in normalized_rewards):
        raise RuntimeError("episode_rewards 包含 NaN/Inf")

    for key in PER_EPISODE_KEYS:
        values = metrics.get(key)
        if not isinstance(values, list) or len(values) != expected_episodes:
            raise RuntimeError(
                f"{key} 长度不匹配: "
                f"got={len(values) if isinstance(values, list) else 'invalid'}, "
                f"expected={expected_episodes}"
            )
    for key in (
        "episode_rewards_per_env",
        "team_success_flags_per_env",
        "agent_success_flags_per_env",
        "collision_counts_per_env",
        "agent_collision_counts_per_env",
    ):
        values = metrics[key]
        bad_rows = [
            index
            for index, row in enumerate(values)
            if not isinstance(row, list) or len(row) != expected_num_envs
        ]
        if bad_rows:
            raise RuntimeError(
                f"{key} 的环境维度错误，首个异常 episode={bad_rows[0]}"
            )

    if int(metrics.get("train_episodes", 0) or 0) != expected_episodes:
        raise RuntimeError("metrics.train_episodes 与 manifest 不一致")
    if str(metrics.get("timestamp", "")) != log_dir.name:
        raise RuntimeError(
            "metrics.timestamp 与日志目录不一致: "
            f"{metrics.get('timestamp')!r} != {log_dir.name!r}"
        )
    expected_parallelism = {
        "num_envs": expected_num_envs,
        "synchronous_iterations": expected_episodes,
        "environment_trajectories": expected_num_envs * expected_episodes,
        "reward_aggregation": "equal_mean_across_environments",
        "success_aggregation": "equal_mean_across_environments",
        "worker_seed_derivation": "base_seed_plus_env_id_times_100003",
        "episode_audit_snapshot_schema_version": 1,
    }
    if metrics.get("training_parallelism") != expected_parallelism:
        raise RuntimeError(
            "metrics.training_parallelism 与 4 环境同步训练契约不一致"
        )
    return normalized_rewards


def _verify_h5(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"权重文件缺失或为空: {path}")
    import h5py

    dataset_count = 0
    with h5py.File(path, "r") as handle:
        def count_dataset(_name: str, value: Any) -> None:
            nonlocal dataset_count
            if isinstance(value, h5py.Dataset):
                dataset_count += 1

        handle.visititems(count_dataset)
    if dataset_count <= 0:
        raise RuntimeError(f"HDF5 权重没有数据集: {path}")


def _validate_final_model(
    final_dir: Path,
    *,
    args: argparse.Namespace,
    expected_agents: int,
) -> Dict[str, str]:
    if not final_dir.is_dir():
        raise RuntimeError(f"final 模型目录不存在: {final_dir}")
    required_names = {
        name
        for agent_index in range(expected_agents)
        for name in (
            f"actor_{agent_index}.weights.h5",
            f"critic1_{agent_index}.weights.h5",
            f"critic2_{agent_index}.weights.h5",
            f"target_actor_{agent_index}.weights.h5",
            f"target_critic1_{agent_index}.weights.h5",
            f"target_critic2_{agent_index}.weights.h5",
        )
    }
    actual_core_names = {
        path.name
        for path in final_dir.glob("*.weights.h5")
        if path.name.startswith(
            (
                "actor_",
                "critic1_",
                "critic2_",
                "target_actor_",
                "target_critic1_",
                "target_critic2_",
            )
        )
    }
    missing = sorted(required_names - actual_core_names)
    unexpected = sorted(actual_core_names - required_names)
    if missing or unexpected:
        raise RuntimeError(
            f"final 核心权重清单错误: missing={missing}, unexpected={unexpected}"
        )
    for name in sorted(required_names):
        _verify_h5(final_dir / name)

    from cross_agent_reference_selector import (
        ADVANTAGE_SELECTOR_MODES,
        MODE_SHARED_TWIN_HEAD_TAIL,
        selector_state_errors,
    )

    selector_mode = str(
        getattr(args, "cross_agent_reference_selector_mode", "hard") or "hard"
    ).strip().lower()
    if selector_mode in ADVANTAGE_SELECTOR_MODES:
        state_path = final_dir / "cross_agent_reference_state.json"
        state = _load_json(state_path)
        state_errors = selector_state_errors(
            state,
            expected_mode=selector_mode,
            require_null_input_dim=(
                selector_mode != MODE_SHARED_TWIN_HEAD_TAIL
            ),
        )
        if state_errors:
            raise RuntimeError(
                "final selector state 无效: " + "; ".join(state_errors)
            )
        if int(state.get("selector_update_count", 0) or 0) <= 0:
            raise RuntimeError("final selector state 的 update_count 为 0")
    if selector_mode == MODE_SHARED_TWIN_HEAD_TAIL:
        _verify_h5(final_dir / "reference_selector_shared.weights.h5")

    return {
        name: _file_sha256(final_dir / name)
        for name in sorted(required_names)
        if name.startswith("actor_")
    }


def _parse_manifest_args(
    manifest: Mapping[str, Any],
) -> Tuple[argparse.Namespace, Any]:
    exec_env = manifest.get("exec_env")
    argv = manifest.get("argv")
    if not isinstance(exec_env, Mapping):
        raise RuntimeError("manifest.exec_env 无效")
    if not isinstance(argv, list) or not all(
        isinstance(value, str) for value in argv
    ):
        raise RuntimeError("manifest.argv 无效")
    if "--resume" in argv or "--checkpoint" in argv:
        raise RuntimeError("恢复工具只接受从 episode 0 开始的新训练")
    for key, value in exec_env.items():
        os.environ[str(key)] = str(value)
    os.environ.setdefault("SUPPRESS_MA_PROMPT", "1")

    import paper3d_train_optimized as training

    previous_argv = sys.argv
    try:
        sys.argv = [str(manifest.get("python_script", "train.py")), *argv]
        parsed = training.parse_args()
    finally:
        sys.argv = previous_argv
    training._apply_runtime_env_overrides_from_args(parsed)
    return parsed, training


def _force_ratio_history(
    args: argparse.Namespace,
    *,
    training: Any,
    episodes: int,
) -> List[float]:
    schedule_args = copy.copy(args)
    if hasattr(schedule_args, "_base_action_force_ratio"):
        delattr(schedule_args, "_base_action_force_ratio")
    return [
        float(
            training._compute_force_ratio_schedule_value(
                schedule_args,
                episode_idx=index,
                resume_start_episode=0,
            )
        )
        for index in range(episodes)
    ]


def _build_training_hyperparameters(
    args: argparse.Namespace,
    *,
    initial_results_args: Mapping[str, Any],
    reward_version: str,
    reward_terminal_order_fix: bool,
) -> Dict[str, Any]:
    keys = (
        "learning_rate_actor",
        "learning_rate_critic",
        "huber_delta",
        "lr_decay_enabled",
        "lr_decay_steps",
        "lr_decay_rate",
        "lr_min_actor",
        "lr_min_critic",
        "actor_hidden",
        "critic_hidden",
        "buffer_dtype",
        "update_rate",
        "actor_update_delay",
        "policy_noise",
        "noise_clip",
        "policy_freq",
        "per_replace",
        "per_uniform_mix",
        "per_td_weight",
        "per_reward_weight",
        "per_age_decay",
        "noise_scale",
        "noise_decay",
        "noise_decay_enabled",
        "noise_decay_steps",
        "noise_staircase",
        "noise_min",
        "random_action_prob",
        "random_action_prob_training",
        "reward_scale",
        "q_clip_value",
        "critic_q_reg",
        "action_reg_coef",
        "neg_z_reg_coef",
        "action_force_ratio",
        "action_force_ratio_schedule_pct",
        "cross_agent_reference_enabled",
        "cross_agent_reference_coef",
        "cross_agent_reference_start_episode",
        "cross_agent_reference_actor_start_episode",
        "cross_agent_reference_actor_ramp_episodes",
        "cross_agent_reference_actor_require_success",
        "cross_agent_reference_update_interval",
        "cross_agent_reference_pairs_per_agent",
        "cross_agent_reference_progress_threshold",
        "cross_agent_reference_margin",
        "cross_agent_reference_head_weight",
        "cross_agent_reference_tail_weight",
        "cross_agent_reference_use_clean_label",
        "cross_agent_reference_target_semantics",
        "cross_agent_reference_exclude_random",
        "cross_agent_reference_quality_gate",
        "cross_agent_reference_gate_mode",
        "cross_agent_reference_selector_enabled",
        "cross_agent_reference_selector_mode",
        "cross_agent_reference_selector_train_in_graph",
        "cross_agent_reference_selector_lr",
        "cross_agent_reference_selector_hidden",
        "cross_agent_reference_selector_init_logit",
        "cross_agent_reference_selector_adv_clip",
        "cross_agent_reference_advantage_ema_decay",
        "cross_agent_reference_advantage_epsilon",
        "cross_agent_reference_advantage_initial_scale",
    )
    hyperparameters = {
        key: copy.deepcopy(initial_results_args.get(key))
        for key in keys
        if key in initial_results_args
    }
    use_dual_q = bool(getattr(args, "matd3_use_dual_q", False))
    use_separated = bool(
        getattr(args, "matd3_use_separated_gradient", False)
    )
    use_hybrid = bool(
        getattr(args, "matd3_use_hybrid_actor_objective", False)
    )
    if not use_dual_q:
        actor_objective_mode = "single_q_joint"
    elif use_separated:
        actor_objective_mode = "hybrid" if use_hybrid else "separated"
    else:
        actor_objective_mode = "unified"
    hyperparameters.update(
        {
            "replay_buffer_size": int(getattr(args, "buffer_size", 0) or 0),
            "actor_objective_mode": actor_objective_mode,
            "use_dual_q": use_dual_q,
            "use_separated_gradient": use_separated,
            "use_hybrid_actor_objective": use_hybrid,
            "hybrid_actor_alpha": float(
                getattr(args, "matd3_hybrid_actor_alpha", 0.0) or 0.0
            ),
            "action_semantics_mode": str(
                getattr(args, "matd3_action_semantics_mode", "dual") or "dual"
            ),
            "reconstruct_corrected_target": bool(
                getattr(args, "matd3_reconstruct_corrected_target", True)
            ),
            "reward_version": reward_version,
            "reward_terminal_order_fix": reward_terminal_order_fix,
        }
    )
    return hyperparameters


def _discover_completed_log_dir(
    repo_root: Path,
    exp_name: str,
    *,
    expected_episodes: int,
) -> Tuple[Path, Dict[str, Any]]:
    log_root = repo_root / "logs" / exp_name
    candidates: List[Tuple[Path, Dict[str, Any]]] = []
    for metrics_path in sorted(log_root.glob("*/episode_rewards.json")):
        try:
            metrics = _load_json(metrics_path)
            rewards = metrics.get("episode_rewards")
            if (
                isinstance(rewards, list)
                and len(rewards) == expected_episodes
                and str(metrics.get("timestamp", "")) == metrics_path.parent.name
            ):
                candidates.append((metrics_path.parent.resolve(), metrics))
        except Exception:
            continue
    if len(candidates) != 1:
        raise RuntimeError(
            "必须唯一定位完整日志目录: "
            f"exp_name={exp_name}, candidates={[str(item[0]) for item in candidates]}"
        )
    return candidates[0]


def _recover_one(
    manifest_path: Path,
    *,
    repo_root: Path,
    apply: bool,
    repair_existing_recovery: bool,
) -> Dict[str, Any]:
    import ablation_dual_q_separated_gradient as ablation
    import experiment_runtime_config

    manifest_path = manifest_path.expanduser().resolve()
    manifest = ablation._load_manifest(manifest_path)
    meta = manifest.get("meta")
    if not isinstance(meta, Mapping):
        raise RuntimeError("manifest.meta 无效")
    exp_name = str(meta.get("exp_name_with_timestamp", "") or "").strip()
    label = str(meta.get("label", "") or "").strip()
    expected_episodes = int(meta.get("episodes", 0) or 0)
    expected_seed = int(meta.get("seed"))
    expected_num_envs = int(meta.get("num_envs", 0) or 0)
    if not exp_name or not label:
        raise RuntimeError("manifest 缺少 exp_name_with_timestamp/label")
    if expected_episodes <= 0 or expected_num_envs <= 0:
        raise RuntimeError("manifest 的 episodes/num_envs 无效")

    args, training = _parse_manifest_args(manifest)
    if str(getattr(args, "exp_name", "")) != exp_name:
        raise RuntimeError("parse_args 后的 exp_name 与 manifest 不一致")
    if int(getattr(args, "seed", -1)) != expected_seed:
        raise RuntimeError("parse_args 后的 seed 与 manifest 不一致")
    if int(getattr(args, "train_episodes", 0)) != expected_episodes:
        raise RuntimeError("parse_args 后的 train_episodes 与 manifest 不一致")
    if int(getattr(args, "num_envs", 0)) != expected_num_envs:
        raise RuntimeError("parse_args 后的 num_envs 与 manifest 不一致")
    if str(getattr(args, "algo", "")).strip().lower() != "matd3":
        raise RuntimeError("恢复工具当前只接受 MATD3 完成单元")

    log_dir, metrics = _discover_completed_log_dir(
        repo_root,
        exp_name,
        expected_episodes=expected_episodes,
    )
    rewards = _validate_metrics(
        metrics,
        log_dir=log_dir,
        expected_episodes=expected_episodes,
        expected_num_envs=expected_num_envs,
    )
    metrics_path = log_dir / "episode_rewards.json"
    loss_history_path = log_dir / "loss_history.json"
    loss_history = _load_json(loss_history_path)
    if not isinstance(loss_history, list) or not loss_history:
        raise RuntimeError(f"loss_history 无效: {loss_history_path}")

    model_root = (repo_root / "models" / exp_name).resolve()
    model_results_path = model_root / "results.json"
    log_results_path = log_dir / "results.json"
    existing_recovery = None
    if model_results_path.exists() or log_results_path.exists():
        if not repair_existing_recovery:
            raise FileExistsError(
                "恢复只允许补齐缺失结果，拒绝覆盖: "
                f"{model_results_path}, {log_results_path}"
            )
        if not model_results_path.is_file() or not log_results_path.is_file():
            raise RuntimeError("已有恢复结果没有同时镜像到 model/log")
        model_existing = _load_json(model_results_path)
        log_existing = _load_json(log_results_path)
        if model_existing != log_existing:
            raise RuntimeError("已有 model/log 恢复结果内容不一致")
        existing_recovery = model_existing
    actor_hashes = _validate_final_model(
        model_root / "final",
        args=args,
        expected_agents=3,
    )

    parent_batch_dir = manifest_path.parents[3]
    launcher_path = (
        parent_batch_dir
        / "launcher_logs"
        / f"seed_{expected_seed}__{label}.log"
    ).resolve()
    launcher_text = launcher_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    training_device, device_evidence = _parse_training_device_evidence(
        launcher_text,
        launcher_path=launcher_path,
        log_dir=log_dir,
        repo_root=repo_root,
        expected_episodes=expected_episodes,
    )

    force_ratios = metrics.get("episode_force_ratios")
    if force_ratios is None:
        force_ratios = _force_ratio_history(
            args,
            training=training,
            episodes=expected_episodes,
        )
        force_ratio_source = "recomputed_from_frozen_manifest_schedule"
    else:
        if not isinstance(force_ratios, list) or len(force_ratios) != expected_episodes:
            raise RuntimeError("metrics.episode_force_ratios 长度无效")
        force_ratios = [float(value) for value in force_ratios]
        force_ratio_source = "episode_rewards_json"

    best_team_sr_episode = int(metrics.get("best_team_sr_episode", -1))
    if 0 <= best_team_sr_episode < expected_episodes:
        recorded_best_fr = float(
            metrics.get("best_team_sr_force_ratio", float("nan"))
        )
        expected_best_fr = float(force_ratios[best_team_sr_episode])
        if not math.isclose(
            recorded_best_fr,
            expected_best_fr,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "FR 日程重建与 metrics.best_team_sr_force_ratio 不一致"
            )

    manifest_training_environment = meta.get("training_environment")
    if not isinstance(manifest_training_environment, Mapping):
        raise RuntimeError("manifest.meta.training_environment 无效")
    training_environment = dict(manifest_training_environment)
    training_environment.update(
        experiment_runtime_config.runtime_environment_from_manifest(
            manifest,
            include_code_defaults=True,
        )
    )
    training_environment["schema_version"] = int(
        manifest_training_environment.get("schema_version", 1)
    )
    training_environment["source"] = str(
        manifest_training_environment.get(
            "source",
            "ablation_resolved_setup",
        )
    )

    initial_results_args = {
        str(key): copy.deepcopy(value)
        for key, value in vars(args).items()
        if not str(key).startswith("_")
    }
    reward_version = str(
        manifest.get("exec_env", {}).get(
            "REWARD_VERSION",
            manifest.get("exec_env", {}).get("reward_version", "v1"),
        )
        or "v1"
    ).strip()
    terminal_order_raw = str(
        manifest.get("exec_env", {}).get(
            "REWARD_TERMINAL_ORDER_FIX",
            manifest.get("exec_env", {}).get(
                "reward_terminal_order_fix",
                "",
            ),
        )
        or ""
    ).strip()
    reward_terminal_order_fix = (
        terminal_order_raw.lower() in ("1", "true", "yes", "on")
        if terminal_order_raw
        else True
    )
    training_hyperparameters = _build_training_hyperparameters(
        args,
        initial_results_args=initial_results_args,
        reward_version=reward_version,
        reward_terminal_order_fix=reward_terminal_order_fix,
    )

    results_args = dict(initial_results_args)
    results_args["action_force_ratio"] = float(force_ratios[-1])
    results_args["base_obs_shapes"] = list(
        device_evidence["base_obs_shapes"]
    )
    results_args["base_action_dims"] = list(
        device_evidence["base_action_dims"]
    )
    for key, value in training_environment.items():
        arg_key = (
            "deterministic_train_env_sequence"
            if key == "deterministic_env_sequence"
            else key
        )
        if arg_key != "source" and arg_key != "schema_version":
            results_args[arg_key] = copy.deepcopy(value)
    results_args["reward_version"] = reward_version
    results_args["reward_terminal_order_fix"] = reward_terminal_order_fix

    best_episode = rewards.index(max(rewards))
    manifest_hash = str(meta.get("content_sha256", "") or "").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_hash):
        raise RuntimeError("manifest content_sha256 无效")
    recovery_provenance = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "reason": "known_results_postamble_name_error_after_completed_training",
        "force_ratio_source": force_ratio_source,
        "manifest_path": str(manifest_path),
        "manifest_content_sha256": manifest_hash,
        "episode_metrics_path": str(metrics_path.resolve()),
        "episode_metrics_sha256": _file_sha256(metrics_path),
        "loss_history_path": str(loss_history_path.resolve()),
        "loss_history_sha256": _file_sha256(loss_history_path),
        "final_actor_sha256": actor_hashes,
        "device_evidence": device_evidence,
    }
    results = {
        "episodes": expected_episodes,
        "rewards": rewards,
        "best_reward": max(rewards),
        "best_episode": best_episode,
        "replay_buffer_size": int(
            training_hyperparameters.get("replay_buffer_size", 0) or 0
        ),
        "actor_objective_mode": str(
            training_hyperparameters.get("actor_objective_mode", "")
        ),
        "hybrid_actor_alpha": float(
            training_hyperparameters.get("hybrid_actor_alpha", 0.0) or 0.0
        ),
        "episode_force_ratios": force_ratios,
        "best_episode_force_ratio": float(force_ratios[best_episode]),
        "last_episode_force_ratio": float(force_ratios[-1]),
        "team_success_rate": float(metrics.get("team_success_rate", 0.0)),
        "best_team_success_rate": float(
            metrics.get("best_team_success_rate", -1.0)
        ),
        "best_team_sr_episode": best_team_sr_episode,
        "best_team_sr_force_ratio": float(
            metrics.get("best_team_sr_force_ratio", 0.0)
        ),
        "best_team_sr_reward": metrics.get("best_team_sr_reward"),
        "terrain_snapshot_artifacts": {},
        "training_manifest_sha256": manifest_hash,
        "training_manifest_path": str(manifest_path),
        "training_environment_schema_version": int(
            training_environment.get("schema_version", 1)
        ),
        "training_environment": training_environment,
        "training_device": training_device,
        "training_parallelism": copy.deepcopy(
            metrics["training_parallelism"]
        ),
        "training_hyperparameters": training_hyperparameters,
        "args": results_args,
        "recovery_provenance": recovery_provenance,
    }
    safe_results = training._make_json_safe(results)
    candidate_errors = experiment_runtime_config._result_completion_errors(
        safe_results,
        model_name=exp_name,
        expected_episodes=expected_episodes,
        expected_seed=expected_seed,
        expected_num_envs=expected_num_envs,
        require_gpu=True,
    )
    if candidate_errors:
        raise RuntimeError(
            "恢复候选结果不满足完成契约: " + "; ".join(candidate_errors)
        )

    if existing_recovery is not None:
        existing_provenance = existing_recovery.get("recovery_provenance")
        if not isinstance(existing_provenance, Mapping):
            raise RuntimeError("已有结果不是本工具生成的恢复结果")
        immutable_recovery_fields = (
            "reason",
            "manifest_path",
            "manifest_content_sha256",
            "episode_metrics_path",
            "episode_metrics_sha256",
            "loss_history_path",
            "loss_history_sha256",
        )
        mismatches = [
            key
            for key in immutable_recovery_fields
            if existing_provenance.get(key) != recovery_provenance.get(key)
        ]
        if mismatches:
            raise RuntimeError(
                "已有恢复结果的不可变证据与当前输入不一致: "
                + ", ".join(mismatches)
            )

    if apply:
        replace_existing = existing_recovery is not None
        _atomic_write_json(
            log_results_path,
            safe_results,
            replace_existing_recovery=replace_existing,
        )
        _atomic_write_json(
            model_results_path,
            safe_results,
            replace_existing_recovery=replace_existing,
        )
        completion_errors = (
            experiment_runtime_config.training_unit_completion_errors(
                model_root,
                expected_episodes,
                repo_root=repo_root,
                expected_agents=3,
                expected_seed=expected_seed,
                expected_num_envs=expected_num_envs,
                require_gpu=True,
            )
        )
        if completion_errors:
            raise RuntimeError(
                "写入后完整单元校验失败: " + "; ".join(completion_errors)
            )

    return {
        "label": label,
        "seed": expected_seed,
        "episodes": expected_episodes,
        "num_envs": expected_num_envs,
        "log_dir": str(log_dir),
        "model_root": str(model_root),
        "training_device": training_device,
        "best_reward": float(max(rewards)),
        "team_success_rate": float(metrics.get("team_success_rate", 0.0)),
        "action": (
            "repaired_existing_recovery"
            if apply and existing_recovery is not None
            else "recovered"
            if apply
            else "dry_run_repair_pass"
            if existing_recovery is not None
            else "dry_run_pass"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="严格恢复已完成训练的 results postamble"
    )
    parser.add_argument(
        "manifests",
        nargs="+",
        help="一个或多个 immutable resolved manifest",
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="通过全部校验后原子写入 model/log results.json",
    )
    parser.add_argument(
        "--repair-existing-recovery",
        action="store_true",
        help="只允许替换本工具生成且不可变证据完全相同的恢复结果",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    summaries = [
        _recover_one(
            Path(manifest),
            repo_root=repo_root,
            apply=bool(args.apply),
            repair_existing_recovery=bool(args.repair_existing_recovery),
        )
        for manifest in args.manifests
    ]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    if not args.apply:
        print("[DRY RUN] 全部恢复审计通过；未写入任何结果文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
