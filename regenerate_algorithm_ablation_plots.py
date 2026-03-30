#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用当前规范名称重新生成算法消融实验图表并回写 summary。"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ablation_action_pf_comparison import (
    find_latest_log_dir,
    load_metrics,
    plot_comparison_success_collision_clearance,
    plot_comparison_success_rate_and_clearance,
    resolve_metric_file,
    resolve_current_collision_threshold,
)
from ablation_dual_q_separated_gradient import (
    EXPERIMENT_CONFIGS,
    OPTIONAL_REFERENCE_EXPERIMENT_LABELS,
    _evaluate_claims,
    plot_comparison_losses_dualq,
    plot_comparison_rewards_dualq,
)


DISPLAY_BY_LABEL: Dict[str, Dict[str, str]] = {
    cfg["label"]: {
        "name": cfg.get("name", cfg["label"]),
        "name_en": cfg.get("name_en", cfg.get("name", cfg["label"])),
        "description": cfg.get("description", ""),
    }
    for cfg in EXPERIMENT_CONFIGS
}
CONFIG_BY_LABEL: Dict[str, Dict[str, Any]] = {
    cfg["label"]: cfg for cfg in EXPERIMENT_CONFIGS
}


def canonicalize_experiment(exp: Dict) -> Dict:
    patched = dict(exp)
    label = patched.get("label")
    display = DISPLAY_BY_LABEL.get(label)
    if display:
        patched.update(display)
    return patched


def iter_summary_files(root: Path, patterns: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    unique_files = sorted({path.resolve() for path in files if path.is_file()})
    return unique_files


def build_title(summary_data: Dict) -> str:
    base = "MATD3 Separated-Skeleton / Actor-Objective Ablation"
    experiment_group = summary_data.get("experiment_group")
    if experiment_group:
        group_desc = summary_data.get("experiment_group_desc") or (
            "Fixed Map" if str(experiment_group).upper() == "A" else "Random Obstacles"
        )
        return f"{base} - Group {experiment_group} ({group_desc})"
    return base


def load_series(experiments: List[Dict], fallback_collision_threshold: float) -> List[Dict]:
    series: List[Dict] = []
    for exp in experiments:
        log_dir = exp.get("log_dir")
        if not log_dir:
            continue
        log_path = Path(log_dir)
        if not log_path.is_absolute():
            log_path = (Path.cwd() / log_path).resolve()
        if not log_path.exists():
            print(f"[Skip] log_dir 不存在: {log_path}")
            continue
        metrics = load_metrics(str(log_path), fallback_collision_threshold=fallback_collision_threshold)
        series.append(
            {
                "label": exp.get("label"),
                "name": exp.get("name", exp.get("label", "Unknown")),
                "name_en": exp.get("name_en", exp.get("name", exp.get("label", "Unknown"))),
                "description": exp.get("description", ""),
                "log_dir": str(log_path),
                "manifest_path": exp.get("manifest_path"),
                "metrics": metrics,
                "success": True,
            }
        )
    return series


def refresh_summary_experiments(summary_data: Dict, series: List[Dict]) -> Dict:
    updated = deepcopy(summary_data)
    by_label = {item["label"]: item for item in series}
    refreshed_experiments: List[Dict] = []
    for exp in updated.get("experiments", []):
        patched = canonicalize_experiment(exp)
        series_item = by_label.get(patched.get("label"))
        if series_item:
            metrics = series_item["metrics"]
            patched["collision_distance_threshold"] = metrics.get("collision_distance_threshold")
            patched["collision_threshold_source"] = metrics.get("collision_threshold_source", "unknown")
        refreshed_experiments.append(patched)
    updated["experiments"] = refreshed_experiments
    return updated


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_log_dir_from_manifest(manifest_path: Path) -> Optional[str]:
    manifest = load_json(manifest_path)
    meta = manifest.get("meta", {}) if isinstance(manifest.get("meta"), dict) else {}
    label = meta.get("label") or manifest_path.stem.replace("_resolved_manifest", "")
    exp_name_with_timestamp = meta.get("exp_name_with_timestamp")
    if exp_name_with_timestamp:
        candidate_root = Path.cwd() / "logs" / str(exp_name_with_timestamp)
        metric_file = (
            resolve_metric_file(str(candidate_root), "episode_rewards.json")
            or resolve_metric_file(str(candidate_root), "results.json")
        )
        if metric_file is not None:
            return str(metric_file.parent)

    fallback_root = Path.cwd() / "logs"
    return find_latest_log_dir(str(label), str(fallback_root))


def build_series_from_batch(batch_dir: Path, fallback_collision_threshold: float) -> tuple[Dict[str, Any], List[Dict]]:
    config_path = batch_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"批次目录缺少 config.json: {batch_dir}")

    batch_config = load_json(config_path)
    labels = list(batch_config.get("experiments", []) or [])
    manifest_dir = batch_dir / "manifests"

    series: List[Dict] = []
    for label in labels:
        cfg = CONFIG_BY_LABEL.get(label)
        if cfg is None:
            print(f"[Skip] 未知实验标签: {label}")
            continue

        manifest_path = manifest_dir / f"{label}_resolved_manifest.json"
        log_dir = None
        if manifest_path.exists():
            log_dir = resolve_log_dir_from_manifest(manifest_path)
        if not log_dir:
            fallback_root = Path.cwd() / "logs"
            log_dir = find_latest_log_dir(label, str(fallback_root))
        if not log_dir:
            print(f"[Skip] 未找到实验日志: {label}")
            continue

        metrics = load_metrics(log_dir, fallback_collision_threshold=fallback_collision_threshold)
        series.append(
            {
                "label": label,
                "name": cfg.get("name", label),
                "name_en": cfg.get("name_en", cfg.get("name", label)),
                "description": cfg.get("description", ""),
                "log_dir": log_dir,
                "manifest_path": str(manifest_path) if manifest_path.exists() else "",
                "metrics": metrics,
                "success": bool(metrics.get("episode_rewards")),
            }
        )
    return batch_config, series


def write_summary_and_plots(
    batch_dir: Path,
    batch_config: Dict[str, Any],
    series: List[Dict],
    smooth_window: int,
    fit_method: str,
) -> Path:
    if not series:
        raise RuntimeError(f"批次 {batch_dir} 中没有可用于绘图的实验数据")

    plots_dir = batch_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    experiment_group = batch_config.get("experiment_group")
    group_desc = batch_config.get("experiment_group_desc") or (
        "Fixed Map" if str(experiment_group).upper() == "A" else "Random Obstacles"
    )
    title = f"MATD3 Separated-Skeleton / Actor-Objective Ablation - Group {experiment_group} ({group_desc})"
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    reward_png = plots_dir / f"reward_comparison_{timestamp}.png"
    loss_png = plots_dir / f"loss_comparison_{timestamp}.png"
    success_collision_png = plots_dir / f"success_collision_clearance_comparison_{timestamp}.png"
    success_rate_clearance_png = plots_dir / f"success_rate_and_clearance_comparison_{timestamp}.png"

    plot_comparison_rewards_dualq(
        series,
        title,
        reward_png,
        smooth_window=smooth_window,
        fit_method=fit_method,
    )
    plot_comparison_losses_dualq(series, title, loss_png)
    plot_comparison_success_collision_clearance(
        series,
        title,
        success_collision_png,
        smooth_window=max(10, smooth_window),
        fit_method=fit_method,
    )
    plot_comparison_success_rate_and_clearance(
        series,
        title,
        success_rate_clearance_png,
        smooth_window=max(10, smooth_window),
    )

    selected_labels = {item.get("label") for item in series if item.get("label")}
    claims_report = _evaluate_claims(series, selected_labels)

    summary = {
        "timestamp": timestamp,
        "experiment_group": experiment_group,
        "experiment_group_desc": group_desc,
        "use_dynamic_obstacles": bool(batch_config.get("use_dynamic_obstacles", False)),
        "strict_validity_enabled": str(batch_config.get("config_mode", "")).strip().lower() == "strict_ablation",
        "claims_report": claims_report,
        "experiments": [
            {
                "label": item["label"],
                "name": item.get("name", item["label"]),
                "name_en": item.get("name_en", item.get("name", item["label"])),
                "description": item.get("description", ""),
                "log_dir": item.get("log_dir", ""),
                "manifest_path": item.get("manifest_path", ""),
                "final_reward": item["metrics"].get("episode_rewards", [])[-1] if item["metrics"].get("episode_rewards") else None,
                "avg_reward": float(sum(item["metrics"].get("episode_rewards", [])) / len(item["metrics"].get("episode_rewards", [])))
                if item["metrics"].get("episode_rewards") else None,
                "max_reward": max(item["metrics"].get("episode_rewards", [])) if item["metrics"].get("episode_rewards") else None,
                "collision_distance_threshold": item["metrics"].get("collision_distance_threshold"),
                "collision_threshold_source": item["metrics"].get("collision_threshold_source", "unknown"),
            }
            for item in series
        ],
        "output_files": {
            "reward_comparison": reward_png.name,
            "loss_comparison": loss_png.name,
            "success_collision_clearance_comparison": success_collision_png.name,
            "success_rate_and_clearance_comparison": success_rate_clearance_png.name,
        },
    }

    summary_path = plots_dir / f"summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    latest_summary_path = plots_dir / "latest_summary.json"
    latest_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def find_latest_batch(root: Path, experiment_group: Optional[str], require_reference: bool) -> Path:
    def _batch_log_mtime(batch_dir: Path) -> float:
        manifest_dir = batch_dir / "manifests"
        latest_mtime = 0.0
        for manifest_path in manifest_dir.glob("*_resolved_manifest.json"):
            try:
                log_dir = resolve_log_dir_from_manifest(manifest_path)
            except Exception:
                log_dir = None
            if not log_dir:
                continue
            for filename in ("episode_rewards.json", "results.json"):
                metric_path = resolve_metric_file(log_dir, filename)
                if metric_path is not None and metric_path.exists():
                    try:
                        latest_mtime = max(latest_mtime, metric_path.stat().st_mtime)
                    except OSError:
                        pass
        if latest_mtime > 0.0:
            return latest_mtime
        try:
            return batch_dir.stat().st_mtime
        except OSError:
            return 0.0

    candidates: List[tuple[float, Path]] = []
    for config_path in root.glob("batch*/config.json"):
        batch_dir = config_path.parent
        try:
            config = load_json(config_path)
        except Exception:
            continue
        if str(config.get("config_mode", "")).strip().lower() != "strict_ablation":
            continue
        if experiment_group and str(config.get("experiment_group", "")).upper() != str(experiment_group).upper():
            continue
        labels = set(config.get("experiments", []) or [])
        if require_reference and not (labels & set(OPTIONAL_REFERENCE_EXPERIMENT_LABELS)):
            continue
        candidates.append((_batch_log_mtime(batch_dir), batch_dir))
    if not candidates:
        raise FileNotFoundError(
            f"未找到匹配的算法消融批次: root={root}, group={experiment_group}, require_reference={require_reference}"
        )
    candidates.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    return candidates[0][1]


def output_path_from_summary(plots_dir: Path, output_files: Dict, key: str, timestamp: Optional[str]) -> Path:
    filename = output_files.get(key)
    if filename:
        return plots_dir / filename
    suffix = {
        "reward_comparison": "reward_comparison",
        "loss_comparison": "loss_comparison",
        "success_collision_clearance_comparison": "success_collision_clearance_comparison",
        "success_rate_and_clearance_comparison": "success_rate_and_clearance_comparison",
    }[key]
    ts = timestamp or "regenerated"
    return plots_dir / f"{suffix}_{ts}.png"


def regenerate_summary(summary_path: Path, smooth_window: int, fit_method: str) -> None:
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    canonical_experiments = [canonicalize_experiment(exp) for exp in summary_data.get("experiments", [])]
    fallback_collision_threshold = resolve_current_collision_threshold()
    series = load_series(canonical_experiments, fallback_collision_threshold=fallback_collision_threshold)
    plots_dir = summary_path.parent
    if not series:
        updated_summary = deepcopy(summary_data)
        updated_summary["experiments"] = canonical_experiments
        summary_path.write_text(json.dumps(updated_summary, indent=2, ensure_ascii=False), encoding="utf-8")

        latest_summary_path = plots_dir / "latest_summary.json"
        if latest_summary_path.exists():
            latest_data = json.loads(latest_summary_path.read_text(encoding="utf-8"))
            latest_data["experiments"] = [canonicalize_experiment(exp) for exp in latest_data.get("experiments", [])]
            latest_summary_path.write_text(json.dumps(latest_data, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"[Skip] 没有可用实验数据，仅统一 summary 命名: {summary_path}")
        return

    output_files = dict(summary_data.get("output_files", {}) or {})
    timestamp = summary_data.get("timestamp")
    title = build_title(summary_data)

    reward_png = output_path_from_summary(plots_dir, output_files, "reward_comparison", timestamp)
    loss_png = output_path_from_summary(plots_dir, output_files, "loss_comparison", timestamp)
    success_collision_png = output_path_from_summary(
        plots_dir,
        output_files,
        "success_collision_clearance_comparison",
        timestamp,
    )
    success_rate_clearance_png = output_path_from_summary(
        plots_dir,
        output_files,
        "success_rate_and_clearance_comparison",
        timestamp,
    )

    print(f"[Replot] {summary_path}")
    plot_comparison_rewards_dualq(
        series,
        title,
        reward_png,
        smooth_window=smooth_window,
        fit_method=fit_method,
    )
    plot_comparison_losses_dualq(series, title, loss_png)
    plot_comparison_success_collision_clearance(
        series,
        title,
        success_collision_png,
        smooth_window=max(10, smooth_window),
    )
    plot_comparison_success_rate_and_clearance(
        series,
        title,
        success_rate_clearance_png,
        smooth_window=max(10, smooth_window),
    )

    output_files.update(
        {
            "reward_comparison": reward_png.name,
            "loss_comparison": loss_png.name,
            "success_collision_clearance_comparison": success_collision_png.name,
            "success_rate_and_clearance_comparison": success_rate_clearance_png.name,
        }
    )

    updated_summary = refresh_summary_experiments(summary_data, series)
    updated_summary["output_files"] = output_files
    summary_path.write_text(json.dumps(updated_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    latest_summary_path = plots_dir / "latest_summary.json"
    if latest_summary_path.exists():
        latest_data = json.loads(latest_summary_path.read_text(encoding="utf-8"))
        if latest_data.get("timestamp") == summary_data.get("timestamp"):
            latest_data = deepcopy(updated_summary)
        else:
            latest_data = refresh_summary_experiments(latest_data, series)
        latest_summary_path.write_text(json.dumps(latest_data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="重新生成算法消融实验图表并统一图例命名")
    parser.add_argument(
        "--root",
        type=str,
        default="ablation_experiments",
        help="算法消融批次目录根路径",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=[
            "batch_202603*/plots/summary_202603*.json",
            "batch_groupA_202603*/plots/summary_202603*.json",
            "batch_groupB_202603*/plots/summary_202603*.json",
        ],
        help="要处理的 summary 文件 glob 模式",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=10,
        help="奖励图平滑窗口；成功率类图至少使用 10",
    )
    parser.add_argument(
        "--fit-method",
        type=str,
        default="moving_average",
        choices=["moving_average", "spline", "poly"],
        help="奖励曲线拟合方法",
    )
    parser.add_argument(
        "--complete-latest-batch",
        action="store_true",
        help="不依赖现有 summary，直接从最近一次 strict_ablation 批次的 manifests/logs 补齐 plots 和 summary",
    )
    parser.add_argument(
        "--batch-dir",
        type=str,
        default=None,
        help="指定要补齐的算法消融批次目录；提供后将优先于 --complete-latest-batch",
    )
    parser.add_argument(
        "--experiment-group",
        type=str,
        default=None,
        help="与 --complete-latest-batch 联用时，限定批次分组（如 A / B）",
    )
    parser.add_argument(
        "--require-reference",
        action="store_true",
        help="与 --complete-latest-batch 联用时，要求该批次包含参考实验（如 MADDPG 家族）",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if args.batch_dir or args.complete_latest_batch:
        if args.batch_dir:
            batch_dir = Path(args.batch_dir)
        else:
            batch_dir = find_latest_batch(
                root=root,
                experiment_group=args.experiment_group,
                require_reference=bool(args.require_reference),
            )
        fallback_collision_threshold = resolve_current_collision_threshold()
        batch_config, series = build_series_from_batch(
            batch_dir=batch_dir,
            fallback_collision_threshold=fallback_collision_threshold,
        )
        summary_path = write_summary_and_plots(
            batch_dir=batch_dir,
            batch_config=batch_config,
            series=series,
            smooth_window=args.smooth_window,
            fit_method=args.fit_method,
        )
        print(f"[Done] 已补齐批次: {batch_dir}")
        print(f"[Done] Summary: {summary_path}")
        return 0

    summary_files = iter_summary_files(root, args.patterns)
    if not summary_files:
        print(f"[Error] 未找到任何 summary 文件，root={root}")
        return 1

    processed = 0
    for summary_path in summary_files:
        try:
            regenerate_summary(summary_path, smooth_window=args.smooth_window, fit_method=args.fit_method)
            processed += 1
        except Exception as exc:
            print(f"[Error] 重绘失败: {summary_path}: {exc}")

    print(f"[Done] 已处理 {processed}/{len(summary_files)} 个 summary")
    return 0 if processed > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
