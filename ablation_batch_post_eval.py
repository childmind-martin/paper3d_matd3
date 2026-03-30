#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为已完成的单-seed 消融 batch 补跑后评估，并生成测试对比图。

设计目标：
1. 不重新训练
2. 不依赖临时 symlink
3. 严格按 batch manifests 绑定到对应的历史模型
4. 直接把测试结果与测试图补写回原 batch 目录
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

os.environ.setdefault("SUPPRESS_MA_PROMPT", "1")
os.environ.setdefault("SUPPRESS_TERRAIN_OUTPUT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

from ablation_action_pf_comparison import load_metrics
from ablation_dual_q_separated_gradient import (
    DEFAULT_POST_EVAL_EPISODES,
    EXPERIMENT_CONFIGS,
    _build_post_eval_spec,
    _collect_post_eval_series,
    _evaluate_claims,
    _load_manifest,
    _post_eval_enabled,
    _resolve_post_eval_seed,
    _resolve_run_log_dir,
    _run_post_training_evaluation,
    _save_json,
    _sort_experiment_configs,
    _validate_loaded_result,
    _write_post_eval_summary_text,
    _plot_post_eval_arrival_path_comparison,
    _plot_post_eval_summary_dashboard,
    plot_comparison_rewards_dualq,
    plot_comparison_success_collision_clearance,
    plot_comparison_success_rate_and_clearance,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BATCH_ROOT = REPO_ROOT / "ablation_experiments"
DEFAULT_LOGS_ROOT = REPO_ROOT / "logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对已完成的消融 batch 补跑共享测试集评估，并生成测试图"
    )
    parser.add_argument(
        "batch",
        type=str,
        help="batch 目录名或完整路径，例如 batch_groupA_20260329_162853",
    )
    parser.add_argument(
        "--batch-root",
        type=str,
        default=str(DEFAULT_BATCH_ROOT),
        help="batch 根目录；当 batch 仅提供目录名时，会在这里查找",
    )
    parser.add_argument(
        "--logs-root",
        type=str,
        default=str(DEFAULT_LOGS_ROOT),
        help="训练日志根目录",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=None,
        choices=[cfg["label"] for cfg in EXPERIMENT_CONFIGS],
        help="只补测指定实验；默认使用 batch config 里的 experiments",
    )
    parser.add_argument(
        "--post-eval-episodes",
        type=int,
        default=None,
        help="测试回合数；默认优先读取 batch config，其次使用脚本默认值",
    )
    parser.add_argument(
        "--post-eval-seed",
        type=int,
        default=None,
        help="共享测试集种子；默认优先读取 batch config，否则由 scenario-seed 推导",
    )
    parser.add_argument(
        "--post-eval-mode",
        type=str,
        default=None,
        choices=["heldout_shared", "match_train_env"],
        help="后评估模式；默认优先读取 batch config",
    )
    parser.add_argument(
        "--post-eval-model-variant",
        type=str,
        default=None,
        choices=["auto", "final", "best", "best_by_team_sr", "latest_ep"],
        help="后评估使用的模型检查点变体；默认优先读取 batch config",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="测试曲线平滑窗口",
    )
    parser.add_argument(
        "--fit-method",
        type=str,
        default="moving_average",
        choices=["moving_average", "spline", "poly"],
        help="测试曲线拟合方式",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="只补跑后评估与 summary，不生成图片",
    )
    parser.add_argument(
        "--allow-fallback-latest",
        action="store_true",
        help="manifest 指向的精确日志缺失时，允许回退到同标签最新日志；默认关闭",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅解析 batch、manifest 和日志目录，不实际执行评估",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"JSON 格式错误: {path}")
    return data


def _resolve_batch_dir(batch_arg: str, batch_root: Path) -> Path:
    raw = Path(batch_arg)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((Path.cwd() / raw).resolve())
        candidates.append((REPO_ROOT / raw).resolve())
        candidates.append((batch_root / raw.name).resolve())

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"未找到 batch 目录: {batch_arg}\n"
        f"已尝试: {[str(candidate) for candidate in candidates]}"
    )


def _resolve_repo_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _resolve_experiment_configs(selected_labels: List[str]) -> List[Dict[str, Any]]:
    config_map = {cfg["label"]: cfg for cfg in EXPERIMENT_CONFIGS}
    missing = [label for label in selected_labels if label not in config_map]
    if missing:
        raise RuntimeError(f"未知实验标签: {missing}")
    return _sort_experiment_configs([config_map[label] for label in selected_labels])


def _resolve_log_dir_from_manifest_strict(
    manifest_path: Path,
    logs_root: Path,
    allow_fallback_latest: bool = False,
) -> str:
    manifest = _load_manifest(manifest_path)
    meta = manifest.get("meta", {}) if isinstance(manifest.get("meta"), dict) else {}
    exp_name_with_timestamp = str(meta.get("exp_name_with_timestamp", "")).strip()
    label = str(meta.get("label", manifest_path.stem.replace("_resolved_manifest", ""))).strip()

    if exp_name_with_timestamp:
        run_root = logs_root / exp_name_with_timestamp
        if run_root.exists():
            resolved = _resolve_run_log_dir(logs_root, exp_name_with_timestamp)
            if resolved:
                return resolved
            raise RuntimeError(f"日志根存在但无法定位具体运行目录: {run_root}")
        if not allow_fallback_latest:
            raise RuntimeError(
                f"manifest 指向的精确日志目录不存在: {run_root}\n"
                f"label={label}, exp_name_with_timestamp={exp_name_with_timestamp}"
            )

    if allow_fallback_latest:
        from ablation_action_pf_comparison import find_latest_log_dir

        return find_latest_log_dir(label, str(logs_root))

    raise RuntimeError(f"manifest 缺少 exp_name_with_timestamp: {manifest_path}")


def _build_runtime_args(batch_config: Dict[str, Any]) -> SimpleNamespace:
    scenario_seed = int(batch_config.get("scenario_seed", 88))
    args = SimpleNamespace()
    args.disable_post_eval = False
    args.env_isolation = "strict"
    args.config_mode = str(batch_config.get("config_mode", "strict_ablation"))
    args.resolved_scenario_seed = scenario_seed
    args.use_dynamic_obstacles = bool(batch_config.get("use_dynamic_obstacles", False))
    args.post_eval_episodes = int(
        batch_config.get("post_eval_episodes", DEFAULT_POST_EVAL_EPISODES)
    )
    args.post_eval_seed = batch_config.get("post_eval_seed")
    args.post_eval_mode = str(batch_config.get("post_eval_mode", "heldout_shared"))
    args.post_eval_model_variant = str(batch_config.get("post_eval_model_variant", "final"))
    args.smooth_window = 5
    args.fit_method = "moving_average"
    args.resolved_post_eval_seed = _resolve_post_eval_seed(args)
    return args


def _build_post_eval_only_summary(
    series: List[Dict[str, Any]],
    args: SimpleNamespace,
    batch_dir: Path,
    positions_file: Path,
    batch_seed: int,
    experiment_group: str,
    group_desc: str,
    post_eval_spec: Dict[str, Any],
    output_files: Dict[str, str],
    timestamp: str,
) -> Dict[str, Any]:
    claims_report = _evaluate_claims(series, {item["label"] for item in series})
    summary = {
        "summary_mode": "post_eval_only",
        "timestamp": timestamp,
        "batch_dir": str(batch_dir),
        "seed": int(batch_seed),
        "experiment_group": experiment_group,
        "experiment_group_desc": group_desc,
        "positions_file": str(positions_file),
        "post_eval_enabled": _post_eval_enabled(args),
        "post_eval_mode": str(post_eval_spec["mode"]),
        "post_eval_episodes": int(post_eval_spec["episodes"]),
        "post_eval_seed": int(post_eval_spec["seed"]),
        "post_eval_model_variant": str(post_eval_spec["model_variant"]),
        "claims_report": claims_report,
        "experiments": [
            {
                "label": item["label"],
                "name": item.get("name", item["label"]),
                "name_en": item.get("name_en", item.get("name", item["label"])),
                "description": item.get("description", ""),
                "log_dir": item.get("log_dir", ""),
                "manifest_path": item.get("manifest_path", ""),
                "post_eval": {
                    "enabled": item.get("post_eval_summary") is not None,
                    "mode": item.get("post_eval_spec", {}).get("mode")
                    if isinstance(item.get("post_eval_spec"), dict)
                    else None,
                    "episodes": item.get("post_eval_episode_count"),
                    "seed": item.get("post_eval_spec", {}).get("seed")
                    if isinstance(item.get("post_eval_spec"), dict)
                    else None,
                    "model_variant": item.get("post_eval_spec", {}).get("model_variant")
                    if isinstance(item.get("post_eval_spec"), dict)
                    else None,
                    "eval_dir": item.get("post_eval_dir", ""),
                    "results_path": item.get("post_eval_results_path", ""),
                    "log_path": item.get("post_eval_log_path", ""),
                    "spec_path": item.get("post_eval_spec_path", ""),
                    "summary": item.get("post_eval_summary"),
                },
            }
            for item in series
        ],
        "output_files": output_files,
    }
    return summary


def _write_post_eval_only_outputs(
    series: List[Dict[str, Any]],
    args: SimpleNamespace,
    batch_dir: Path,
    positions_file: Path,
    batch_seed: int,
    experiment_group: str,
    group_desc: str,
    post_eval_spec: Dict[str, Any],
) -> Dict[str, Any]:
    output_dir = batch_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    post_eval_series = _collect_post_eval_series(series)
    if not post_eval_series:
        raise RuntimeError("没有可用于绘图的 post_eval 数据")

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    eval_title = f"Post-training Evaluation - Group {experiment_group} ({group_desc})"
    output_files: Dict[str, str] = {}

    if not getattr(args, "skip_plots", False):
        eval_reward_png = output_dir / f"post_eval_reward_comparison_{timestamp}.png"
        plot_comparison_rewards_dualq(
            post_eval_series,
            eval_title,
            eval_reward_png,
            smooth_window=max(1, min(int(args.smooth_window), 5)),
            fit_method=args.fit_method,
        )
        output_files["post_eval_reward_comparison"] = eval_reward_png.name

        eval_success_png = output_dir / f"post_eval_success_collision_clearance_comparison_{timestamp}.png"
        plot_comparison_success_collision_clearance(
            post_eval_series,
            eval_title,
            eval_success_png,
            smooth_window=max(1, min(int(args.smooth_window), 5)),
            fit_method=args.fit_method,
        )
        output_files["post_eval_success_collision_clearance_comparison"] = eval_success_png.name

        eval_clearance_png = output_dir / f"post_eval_success_rate_and_clearance_comparison_{timestamp}.png"
        plot_comparison_success_rate_and_clearance(
            post_eval_series,
            eval_title,
            eval_clearance_png,
            smooth_window=max(1, min(int(args.smooth_window), 5)),
            fit_method=args.fit_method,
        )
        output_files["post_eval_success_rate_and_clearance_comparison"] = eval_clearance_png.name

        eval_arrival_path_png = output_dir / f"post_eval_arrival_path_comparison_{timestamp}.png"
        _plot_post_eval_arrival_path_comparison(post_eval_series, eval_title, eval_arrival_path_png)
        output_files["post_eval_arrival_path_comparison"] = eval_arrival_path_png.name

        eval_dashboard_png = output_dir / f"post_eval_summary_dashboard_{timestamp}.png"
        _plot_post_eval_summary_dashboard(post_eval_series, eval_title, eval_dashboard_png)
        output_files["post_eval_summary_dashboard"] = eval_dashboard_png.name

        eval_summary_txt = output_dir / f"post_eval_summary_{timestamp}.txt"
        _write_post_eval_summary_text(post_eval_series, eval_summary_txt)
        output_files["post_eval_summary_text"] = eval_summary_txt.name

    summary = _build_post_eval_only_summary(
        series=series,
        args=args,
        batch_dir=batch_dir,
        positions_file=positions_file,
        batch_seed=batch_seed,
        experiment_group=experiment_group,
        group_desc=group_desc,
        post_eval_spec=post_eval_spec,
        output_files=output_files,
        timestamp=timestamp,
    )
    summary_path = output_dir / f"post_eval_only_summary_{timestamp}.json"
    latest_summary_path = output_dir / "latest_post_eval_only_summary.json"
    _save_json(summary_path, summary)
    _save_json(latest_summary_path, summary)

    print(f"\n{'=' * 70}")
    print("Post-eval backfill complete")
    print(f"Batch directory: {batch_dir}")
    print(f"Summary file: {summary_path}")
    print(f"Latest summary: {latest_summary_path}")
    for key, filename in output_files.items():
        print(f"{key}: {filename}")
    print(f"{'=' * 70}")

    summary["summary_path"] = str(summary_path)
    summary["latest_summary_path"] = str(latest_summary_path)
    return summary


def main() -> int:
    args = parse_args()

    batch_root = Path(args.batch_root).resolve()
    logs_root = Path(args.logs_root).resolve()
    batch_dir = _resolve_batch_dir(args.batch, batch_root=batch_root)
    config_path = batch_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"缺少 batch config: {config_path}")

    batch_config = _load_json(config_path)
    runtime_args = _build_runtime_args(batch_config)
    runtime_args.skip_plots = bool(args.skip_plots)
    runtime_args.smooth_window = int(args.smooth_window)
    runtime_args.fit_method = str(args.fit_method)

    if args.post_eval_episodes is not None:
        runtime_args.post_eval_episodes = int(args.post_eval_episodes)
    if args.post_eval_seed is not None:
        runtime_args.post_eval_seed = int(args.post_eval_seed)
    if args.post_eval_mode is not None:
        runtime_args.post_eval_mode = str(args.post_eval_mode)
    if args.post_eval_model_variant is not None:
        runtime_args.post_eval_model_variant = str(args.post_eval_model_variant)
    runtime_args.resolved_post_eval_seed = _resolve_post_eval_seed(runtime_args)

    selected_labels = (
        list(args.experiments)
        if args.experiments
        else list(batch_config.get("experiments", []))
    )
    if not selected_labels:
        raise RuntimeError("batch config 中没有 experiments，且命令行也未指定 --experiments")
    configs_to_run = _resolve_experiment_configs(selected_labels)

    positions_file = _resolve_repo_path(str(batch_config.get("positions_file", "")))
    if not positions_file.exists():
        raise FileNotFoundError(f"positions_file 不存在: {positions_file}")

    batch_seed = int(batch_config.get("seed"))
    scenario_seed = int(batch_config.get("scenario_seed", 88))
    experiment_group = str(batch_config.get("experiment_group", "A"))
    group_desc = str(
        batch_config.get(
            "experiment_group_desc",
            "Fixed Map" if experiment_group == "A" else "Random Obstacles",
        )
    )

    post_eval_spec = _build_post_eval_spec(runtime_args, batch_dir, positions_file)
    if post_eval_spec is None:
        raise RuntimeError("post-eval 当前被禁用，无法执行补测")

    manifest_dir = batch_dir / "manifests"
    if not manifest_dir.exists():
        raise FileNotFoundError(f"缺少 manifests 目录: {manifest_dir}")

    print(f"{'=' * 70}")
    print("Post-eval backfill")
    print(f"Batch: {batch_dir}")
    print(f"Logs root: {logs_root}")
    print(f"Experiments: {[cfg['label'] for cfg in configs_to_run]}")
    print(
        f"Post-eval: mode={post_eval_spec['mode']}, episodes={post_eval_spec['episodes']}, "
        f"seed={post_eval_spec['seed']}, model_variant={post_eval_spec['model_variant']}"
    )
    if post_eval_spec.get("episode_positions_dir"):
        print(f"Heldout testset dir: {post_eval_spec['episode_positions_dir']}")
    print(f"{'=' * 70}")

    series: List[Dict[str, Any]] = []
    for cfg in configs_to_run:
        label = cfg["label"]
        manifest_path = manifest_dir / f"{label}_resolved_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"缺少 resolved manifest: {manifest_path}")

        log_dir = _resolve_log_dir_from_manifest_strict(
            manifest_path=manifest_path,
            logs_root=logs_root,
            allow_fallback_latest=bool(args.allow_fallback_latest),
        )
        metrics = load_metrics(log_dir)
        validation_errors = _validate_loaded_result(
            cfg=cfg,
            log_dir=log_dir,
            metrics=metrics,
            expected_episodes=int(batch_config.get("episodes")),
            positions_file=positions_file,
            expected_terrain_seed=scenario_seed,
            batch_seed=batch_seed,
        )
        if validation_errors:
            raise RuntimeError(
                f"[{label}] 训练结果有效性校验失败: {' | '.join(validation_errors)}"
            )

        result = {
            "label": label,
            "name": cfg.get("name", label),
            "name_en": cfg.get("name_en", cfg.get("name", label)),
            "description": cfg.get("description", ""),
            "log_dir": log_dir,
            "manifest_path": str(manifest_path),
            "metrics": metrics,
            "success": True,
        }

        print(f"[{label}] log_dir = {log_dir}")
        if args.dry_run:
            series.append(result)
            continue

        result = _run_post_training_evaluation(
            result=result,
            cfg=cfg,
            positions_file=positions_file,
            args=runtime_args,
            batch_dir=batch_dir,
            post_eval_spec=post_eval_spec,
        )
        series.append(result)

    if args.dry_run:
        print("\nDry-run complete: manifest、log_dir 和 post-eval 配置解析成功，未执行评估。")
        return 0

    _write_post_eval_only_outputs(
        series=series,
        args=runtime_args,
        batch_dir=batch_dir,
        positions_file=positions_file,
        batch_seed=batch_seed,
        experiment_group=experiment_group,
        group_desc=group_desc,
        post_eval_spec=post_eval_spec,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断。", file=sys.stderr)
        raise SystemExit(130)
