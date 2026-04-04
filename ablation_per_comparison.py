#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PER ablation around the current run_optimized mainline.

Default setup:
- Baseline: completely uniform replay sampling (PER disabled)
- Optimized: current improved PER mainline from run_optimized defaults
- Single-seed run with fixed environment for fair comparison

Outputs:
- reward / loss / training-dashboard plots
- optional interactive reward comparison
- summary json / csv
- resolved manifests for reproducibility
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Avoid matplotlib cache warnings in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from ablation_action_pf_comparison import (
    find_latest_log_dir,
    generate_interactive_comparison,
    load_metrics,
    plot_comparison_losses,
    plot_comparison_rewards,
    plot_comparison_success_collision_clearance,
)
from ablation_dual_q_separated_gradient import (
    _list_label_run_dirs,
    _load_manifest,
    _resolve_current_run_log_dir,
    _resolve_log_dir_from_manifest,
    _resolve_training_python,
    _save_json,
    generate_fixed_positions,
    setup_base_env_vars,
)


DEFAULT_PER_SETTINGS = {
    "PER_ENABLED": "1",
    "PER_REPLACE": "0",
    "PER_UNIFORM_MIX": "0.40",
    "PER_TD_WEIGHT": "0.80",
    "PER_REWARD_WEIGHT": "0.10",
    "PER_AGE_DECAY": "0.95",
}


def _bool_arg(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _safe_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _tail_mean(values: Sequence[Any], tail_n: int) -> Optional[float]:
    if not values:
        return None
    arr = np.asarray(values[-min(len(values), int(tail_n)):], dtype=np.float64)
    if arr.size == 0:
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _derive_scenario_name(args) -> str:
    if args.scenario_name:
        scenario_name = str(args.scenario_name).strip()
        if scenario_name == "paper3d_terrain_energy":
            args.use_weighted_reward = False
            args.vectorized_scenario = False
            return scenario_name
        if scenario_name == "paper3d_terrain_weighted":
            args.use_weighted_reward = True
            args.vectorized_scenario = False
            return scenario_name
        if scenario_name == "paper3d_terrain_vectorized":
            args.use_weighted_reward = True
            args.vectorized_scenario = True
            return scenario_name
        raise ValueError(
            "当前脚本仅支持 run_optimized 主线可选的 3 个场景族："
            "paper3d_terrain_energy / paper3d_terrain_weighted / paper3d_terrain_vectorized。"
        )
    if not bool(args.use_weighted_reward):
        return "paper3d_terrain_energy"
    return "paper3d_terrain_vectorized" if bool(args.vectorized_scenario) else "paper3d_terrain_weighted"


def _build_experiment_configs(args) -> List[Dict[str, Any]]:
    common_env = {
        "ALGORITHM": str(args.algorithm),
        "USE_LITE_BUFFER": "1",
        "VECTORIZED_SCENARIO": str(int(bool(args.vectorized_scenario))),
    }
    uniform_env = {
        **common_env,
        "PER_ENABLED": "0",
    }
    improved_env = {
        **common_env,
        "PER_ENABLED": "1",
        "PER_REPLACE": str(int(bool(args.per_replace))),
        "PER_UNIFORM_MIX": f"{float(args.per_uniform_mix):.6g}",
        "PER_TD_WEIGHT": f"{float(args.per_td_weight):.6g}",
        "PER_REWARD_WEIGHT": f"{float(args.per_reward_weight):.6g}",
        "PER_AGE_DECAY": f"{float(args.per_age_decay):.6g}",
    }
    return [
        {
            "label": "per_uniform_baseline",
            "name": "Uniform Replay Baseline",
            "name_en": "Uniform Replay Baseline",
            "description": "Lite replay buffer with completely uniform sampling (PER disabled).",
            "env": uniform_env,
            "expected": {
                "per_enabled": False,
            },
        },
        {
            "label": "per_improved_mainline",
            "name": "Improved PER Mainline",
            "name_en": "Improved PER Mainline",
            "description": "Current run_optimized PER mainline: PER + uniform mix + TD/reward/age priority.",
            "env": improved_env,
            "expected": {
                "per_enabled": True,
                "per_replace": bool(args.per_replace),
                "per_uniform_mix": float(args.per_uniform_mix),
                "per_td_weight": float(args.per_td_weight),
                "per_reward_weight": float(args.per_reward_weight),
                "per_age_decay": float(args.per_age_decay),
            },
        },
    ]


def _build_batch_dir(args, scenario_name: str, positions_file: Path) -> Path:
    batch_root = Path(args.output_root)
    batch_root.mkdir(parents=True, exist_ok=True)
    batch_id = f"batch_per_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_dir = batch_root / batch_id
    for subdir in ("plots", "results", "manifests"):
        (batch_dir / subdir).mkdir(parents=True, exist_ok=True)
    config_payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": str(Path(args.script).resolve()),
        "episodes": int(args.episodes),
        "batch_size": int(args.batch_size),
        "algorithm": str(args.algorithm),
        "seed": int(args.seed),
        "scenario_seed": int(args.scenario_seed),
        "config_mode": str(args.config_mode),
        "env_isolation": str(args.env_isolation),
        "use_weighted_reward": bool(args.use_weighted_reward),
        "vectorized_scenario": bool(args.vectorized_scenario),
        "scenario_name": scenario_name,
        "positions_file": str(positions_file),
        "smooth_window": int(args.smooth_window),
        "fit_method": str(args.fit_method),
        "per_mainline": {
            "per_replace": bool(args.per_replace),
            "per_uniform_mix": float(args.per_uniform_mix),
            "per_td_weight": float(args.per_td_weight),
            "per_reward_weight": float(args.per_reward_weight),
            "per_age_decay": float(args.per_age_decay),
        },
    }
    _save_json(batch_dir / "config.json", config_payload)
    return batch_dir


def _resolve_experiment_manifest(
    cfg: Dict[str, Any],
    positions_file: Path,
    args,
    manifest_dir: Path,
) -> Tuple[Dict[str, Any], Path]:
    env = setup_base_env_vars(
        positions_file,
        env_isolation=args.env_isolation,
        config_mode=args.config_mode,
        scenario_seed=int(args.scenario_seed),
    )
    env.update(cfg.get("env", {}))
    env["SEED"] = str(int(args.seed))
    env["TRAIN_PYTHON_BIN"] = str(args.training_python)
    manifest_path = manifest_dir / f"{cfg['label']}_resolved_manifest.json"
    env["ABLATION_RESOLVE_ONLY"] = "1"
    env["ABLATION_MANIFEST_PATH"] = str(manifest_path)
    cmd = [
        "/bin/bash",
        args.script,
        str(args.episodes),
        str(args.batch_size),
        cfg["label"],
        str(int(bool(args.use_weighted_reward))),
        str(cfg["env"].get("ALGORITHM", args.algorithm)),
    ]
    subprocess.run(cmd, env=env, cwd=Path(args.script).resolve().parent, check=True)
    manifest = _load_manifest(manifest_path)
    manifest.setdefault("meta", {})
    manifest["meta"].update(
        {
            "label": cfg["label"],
            "seed": int(args.seed),
            "scenario_seed": int(args.scenario_seed),
            "config_mode": str(args.config_mode),
            "scenario_name": str(args.scenario_name),
        }
    )
    _save_json(manifest_path, manifest)
    return manifest, manifest_path


def _validate_result(cfg: Dict[str, Any], log_dir: str, metrics: Dict[str, Any], args) -> List[str]:
    errors: List[str] = []
    rewards = metrics.get("episode_rewards", [])
    if not isinstance(rewards, list) or len(rewards) == 0:
        return ["episode_rewards 为空"]
    if len(rewards) != int(args.episodes):
        errors.append(f"episode_rewards 长度不匹配: got={len(rewards)}, expected={int(args.episodes)}")
    try:
        rewards_np = np.asarray(rewards, dtype=np.float64)
        if not np.all(np.isfinite(rewards_np)):
            errors.append("episode_rewards 含 NaN/Inf")
    except Exception:
        errors.append("episode_rewards 不是可解析数值数组")

    results_path = Path(log_dir) / "results.json"
    if not results_path.exists():
        errors.append("缺少 results.json")
        return errors
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            results_data = json.load(f)
    except Exception as exc:
        return [f"读取 results.json 失败: {exc}"]

    run_args = results_data.get("args", {})
    if not isinstance(run_args, dict):
        errors.append("results.json 缺少 args")
        return errors

    got_seed = _safe_int(run_args.get("seed"))
    if got_seed != int(args.seed):
        errors.append(f"seed 不匹配: got={got_seed}, expected={int(args.seed)}")

    got_algo = str(run_args.get("algo", "")).strip().lower()
    if got_algo != str(args.algorithm).strip().lower():
        errors.append(f"algo 不匹配: got={got_algo}, expected={str(args.algorithm).strip().lower()}")

    got_train_ep = _safe_int(run_args.get("train_episodes"))
    if got_train_ep != int(args.episodes):
        errors.append(f"train_episodes 不匹配: got={got_train_ep}, expected={int(args.episodes)}")

    expected = cfg.get("expected", {})
    expected_per_enabled = bool(expected.get("per_enabled", False))
    got_per_enabled = _bool_arg(run_args.get("per_enabled", False))
    if got_per_enabled != expected_per_enabled:
        errors.append(f"per_enabled 不匹配: got={got_per_enabled}, expected={expected_per_enabled}")

    if expected_per_enabled:
        bool_fields = (
            ("per_replace", bool(expected.get("per_replace", False))),
        )
        float_fields = (
            ("per_uniform_mix", _safe_float(expected.get("per_uniform_mix"))),
            ("per_td_weight", _safe_float(expected.get("per_td_weight"))),
            ("per_reward_weight", _safe_float(expected.get("per_reward_weight"))),
            ("per_age_decay", _safe_float(expected.get("per_age_decay"))),
        )
        for field_name, expected_bool in bool_fields:
            got_value = _bool_arg(run_args.get(field_name, False))
            if got_value != expected_bool:
                errors.append(f"{field_name} 不匹配: got={got_value}, expected={expected_bool}")
        for field_name, expected_float in float_fields:
            if expected_float is None:
                continue
            got_value = _safe_float(run_args.get(field_name))
            if got_value is None or not math.isclose(got_value, expected_float, rel_tol=1e-6, abs_tol=1e-6):
                errors.append(f"{field_name} 不匹配: got={got_value}, expected={expected_float}")

    got_scenario_seed = _safe_int(run_args.get("scenario_seed"))
    if got_scenario_seed is not None and got_scenario_seed != int(args.scenario_seed):
        errors.append(
            f"scenario_seed 不匹配: got={got_scenario_seed}, expected={int(args.scenario_seed)}"
        )

    return errors


def _run_experiment(
    cfg: Dict[str, Any],
    args,
    batch_dir: Path,
    positions_file: Path,
) -> Dict[str, Any]:
    project_logs_root = Path(args.script).resolve().parent / "logs"
    manifest_dir = batch_dir / "manifests"
    manifest, manifest_path = _resolve_experiment_manifest(cfg, positions_file, args, manifest_dir)
    python_bin = manifest.get("python_executable") or args.training_python
    cmd = [
        str(python_bin),
        str(manifest.get("python_script")),
        *list(manifest.get("argv", [])),
    ]
    before_run_dirs = {d.name for d in _list_label_run_dirs(project_logs_root, cfg["label"])}

    print(f"\n{'=' * 72}")
    print(f"[PER消融] 运行实验: {cfg['name_en']}")
    print(f"[PER消融] Label: {cfg['label']}")
    print(f"[PER消融] Seed: {args.seed}")
    print(f"[PER消融] 解析配置: {manifest_path}")
    print(f"{'=' * 72}")

    subprocess.run(
        cmd,
        env=dict(manifest.get("exec_env", {})),
        cwd=manifest.get("cwd", str(Path(args.script).resolve().parent)),
        check=True,
    )

    log_dir = _resolve_log_dir_from_manifest(Path(manifest_path), project_logs_root)
    if not log_dir:
        log_dir = _resolve_current_run_log_dir(project_logs_root, cfg["label"], before_run_dirs)
    if not log_dir:
        log_dir = find_latest_log_dir(cfg["label"], str(project_logs_root))

    metrics = load_metrics(log_dir)
    validation_errors = _validate_result(cfg, log_dir, metrics, args)
    if validation_errors:
        raise RuntimeError(
            "[PER消融-{}] 结果校验失败: {}".format(cfg["label"], " | ".join(validation_errors))
        )

    return {
        "label": cfg["label"],
        "name": cfg.get("name", cfg["label"]),
        "name_en": cfg.get("name_en", cfg.get("name", cfg["label"])),
        "description": cfg.get("description", ""),
        "log_dir": str(log_dir),
        "manifest_path": str(manifest_path),
        "metrics": metrics,
        "success": True,
        "expected": cfg.get("expected", {}),
    }


def _summarize_series(series: List[Dict[str, Any]], tail_n: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in series:
        metrics = item.get("metrics", {})
        rewards = metrics.get("episode_rewards", []) or []
        success_flags = metrics.get("success_flags", []) or []
        collision_counts = metrics.get("collision_counts", []) or []
        clearances = metrics.get("min_distances_to_obstacle", []) or []
        row = {
            "label": item.get("label"),
            "name_en": item.get("name_en"),
            "log_dir": item.get("log_dir"),
            "episodes": len(rewards),
            "best_reward": float(np.max(np.asarray(rewards, dtype=np.float64))) if rewards else None,
            "final_reward": _safe_float(rewards[-1]) if rewards else None,
            f"tail{tail_n}_mean_reward": _tail_mean(rewards, tail_n),
            f"tail{tail_n}_mean_success": _tail_mean(success_flags, tail_n),
            f"tail{tail_n}_mean_collisions": _tail_mean(collision_counts, tail_n),
            f"tail{tail_n}_mean_clearance": _tail_mean(clearances, tail_n),
            "team_success_rate": _safe_float(metrics.get("team_success_rate")),
        }
        rows.append(row)
    return rows


def _write_summary(batch_dir: Path, args, series: List[Dict[str, Any]]) -> Tuple[Path, Path]:
    summary_rows = _summarize_series(series, tail_n=int(args.summary_tail))
    summary_json = batch_dir / "results" / "summary.json"
    summary_csv = batch_dir / "results" / "summary.csv"
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "episodes": int(args.episodes),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "scenario_seed": int(args.scenario_seed),
        "algorithm": str(args.algorithm),
        "summary_tail": int(args.summary_tail),
        "series": summary_rows,
    }
    _save_json(summary_json, payload)
    fieldnames = [
        "label",
        "name_en",
        "episodes",
        "best_reward",
        "final_reward",
        f"tail{int(args.summary_tail)}_mean_reward",
        f"tail{int(args.summary_tail)}_mean_success",
        f"tail{int(args.summary_tail)}_mean_collisions",
        f"tail{int(args.summary_tail)}_mean_clearance",
        "team_success_rate",
        "log_dir",
    ]
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    return summary_json, summary_csv


def _render_outputs(batch_dir: Path, args, series: List[Dict[str, Any]]) -> Dict[str, str]:
    output_dir = batch_dir / "plots"
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    files: Dict[str, str] = {}

    reward_png = output_dir / f"per_reward_comparison_{timestamp}.png"
    plot_comparison_rewards(
        series,
        title="PER Ablation - Reward Comparison",
        output_path=reward_png,
        smooth_window=int(args.smooth_window),
        fit_method=str(args.fit_method),
    )
    files["reward_comparison"] = reward_png.name

    loss_png = output_dir / f"per_loss_comparison_{timestamp}.png"
    plot_comparison_losses(
        series,
        title="PER Ablation",
        output_path=loss_png,
    )
    files["loss_comparison"] = loss_png.name

    dashboard_png = output_dir / f"per_training_dashboard_{timestamp}.png"
    plot_comparison_success_collision_clearance(
        series,
        title="PER Ablation - Training Metrics",
        output_path=dashboard_png,
        smooth_window=int(args.smooth_window),
        fit_method=str(args.fit_method),
    )
    files["training_dashboard"] = dashboard_png.name

    html_path = output_dir / f"per_reward_comparison_{timestamp}.html"
    generate_interactive_comparison(
        series,
        title="PER Ablation",
        output_path=html_path,
        smooth_window=int(args.smooth_window),
        fit_method=str(args.fit_method),
    )
    if html_path.exists():
        files["reward_comparison_interactive"] = html_path.name

    return files


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Current run_optimized PER ablation (single-seed).")
    parser.add_argument("--script", type=str, default="./run_optimized.sh", help="Path to run_optimized.sh")
    parser.add_argument("--episodes", type=int, default=200, help="Training episodes for each run")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size passed to run_optimized")
    parser.add_argument("--use-weighted-reward", type=_bool_arg, default=True, help="Whether to use weighted reward scenario")
    parser.add_argument("--algorithm", type=str, default="matd3", choices=["matd3", "maddpg"], help="Algorithm passed to run_optimized")
    parser.add_argument("--seed", type=int, default=936487, help="Shared random seed for both experiments")
    parser.add_argument("--scenario-seed", type=int, default=88, help="Shared terrain/scenario seed")
    parser.add_argument("--config-mode", type=str, default="strict_ablation", choices=["strict_ablation", "run_optimized_default", "legacy_ablation"], help="Environment control mode")
    parser.add_argument("--env-isolation", type=str, default="strict", choices=["strict", "inherit"], help="Child process env isolation mode")
    parser.add_argument("--output-root", type=str, default="ablation_experiments", help="Root directory for ablation outputs")
    parser.add_argument("--positions-file", type=str, default="", help="Optional fixed positions file path")
    parser.add_argument("--scenario-name", type=str, default="", help="Optional scenario module override")
    parser.add_argument("--vectorized-scenario", type=_bool_arg, default=True, help="Use vectorized weighted scenario when weighted reward is enabled")
    parser.add_argument("--smooth-window", type=int, default=10, help="Reward/dashboard smoothing window")
    parser.add_argument("--fit-method", type=str, default="moving_average", choices=["moving_average"], help="Curve fitting method")
    parser.add_argument("--summary-tail", type=int, default=50, help="Tail window used in summary table")

    parser.add_argument("--per-replace", type=_bool_arg, default=False, help="Improved PER setting: sampling with replacement")
    parser.add_argument("--per-uniform-mix", type=float, default=float(DEFAULT_PER_SETTINGS["PER_UNIFORM_MIX"]), help="Improved PER setting: uniform-mix ratio")
    parser.add_argument("--per-td-weight", type=float, default=float(DEFAULT_PER_SETTINGS["PER_TD_WEIGHT"]), help="Improved PER setting: TD-error weight")
    parser.add_argument("--per-reward-weight", type=float, default=float(DEFAULT_PER_SETTINGS["PER_REWARD_WEIGHT"]), help="Improved PER setting: reward weight")
    parser.add_argument("--per-age-decay", type=float, default=float(DEFAULT_PER_SETTINGS["PER_AGE_DECAY"]), help="Improved PER setting: age decay")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.script = str(Path(args.script).resolve())
    if not Path(args.script).exists():
        raise FileNotFoundError(f"run_optimized.sh not found: {args.script}")

    args.training_python = _resolve_training_python()
    if not args.training_python:
        raise RuntimeError(
            "未找到可用的 TensorFlow 训练解释器。"
            " 请先确认训练环境可用，或通过 TRAIN_PYTHON_BIN 提供解释器。"
        )

    scenario_name = _derive_scenario_name(args)
    args.scenario_name = scenario_name

    if args.positions_file:
        positions_file = Path(args.positions_file).resolve()
    else:
        positions_file = (Path("./saved_positions") / f"per_ablation_seed{int(args.scenario_seed)}.json").resolve()

    if args.config_mode in ("strict_ablation", "legacy_ablation"):
        generate_fixed_positions(
            positions_file=positions_file,
            scenario_seed=int(args.scenario_seed),
            scenario_name=scenario_name,
            terrain_complexity_level=3,
        )
    else:
        positions_file.parent.mkdir(parents=True, exist_ok=True)

    batch_dir = _build_batch_dir(args, scenario_name, positions_file)
    configs = _build_experiment_configs(args)
    _save_json(batch_dir / "results" / "experiment_configs.json", {"experiments": configs})

    print(f"{'=' * 72}")
    print("PER 消融实验")
    print(f"输出目录: {batch_dir}")
    print(f"训练脚本: {args.script}")
    print(f"训练解释器: {args.training_python}")
    print(f"算法: {args.algorithm}")
    print(f"Seed: {args.seed}")
    print(f"Scenario seed: {args.scenario_seed}")
    print(f"Scenario: {scenario_name}")
    print(f"Config mode: {args.config_mode}")
    print(f"{'=' * 72}")

    series: List[Dict[str, Any]] = []
    for cfg in configs:
        result = _run_experiment(cfg, args, batch_dir, positions_file)
        series.append(result)

    output_files = _render_outputs(batch_dir, args, series)
    summary_json, summary_csv = _write_summary(batch_dir, args, series)
    _save_json(
        batch_dir / "results" / "outputs.json",
        {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "batch_dir": str(batch_dir),
            "plots": output_files,
            "summary_json": str(summary_json),
            "summary_csv": str(summary_csv),
        },
    )

    print("\nPER 消融完成")
    print(f"  批次目录: {batch_dir}")
    print(f"  Summary JSON: {summary_json}")
    print(f"  Summary CSV: {summary_csv}")
    for key, value in output_files.items():
        print(f"  {key}: {batch_dir / 'plots' / value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
