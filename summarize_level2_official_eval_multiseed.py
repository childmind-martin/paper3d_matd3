#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize_level2_official_eval import (
    COLOR_MAP,
    DEFAULT_LABELS,
    DISPLAY_NAME_MAP,
)


RUN_METRIC_KEYS = [
    "team_success_rate",
    "avg_reward",
    "avg_collision_count",
    "avg_terrain_collision_count",
    "avg_obstacle_collision_count",
    "avg_inter_agent_collision_count",
    "collision_free_rate",
    "inter_agent_collision_free_rate",
    "avg_team_final_goal_distance",
    "avg_agent_final_goal_distance",
    "avg_min_inter_agent_clearance",
    "avg_steps",
]


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _display_name(label: str) -> str:
    return DISPLAY_NAME_MAP.get(label, label)


def _find_latest_eval_result(log_root: Path, label: str, seed: int) -> Optional[Path]:
    pattern = f"level2_retrain_{label}_seed{seed}_*/evaluation_official*/evaluation_results.json"
    candidates = list(log_root.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_result_row(label: str, seed: int, result_path: Path) -> Dict[str, Any]:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
    row = {
        "label": label,
        "display_name": _display_name(label),
        "seed": seed,
        "episodes": _safe_int(summary.get("episodes", data.get("episodes"))),
        "results_path": str(result_path),
    }
    for key in RUN_METRIC_KEYS:
        row[key] = _safe_float(summary.get(key))
    return row


def _aggregate_rows(rows: Sequence[Dict[str, Any]], labels: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)

    aggregated: List[Dict[str, Any]] = []
    for label in labels:
        items = grouped.get(label, [])
        if not items:
            continue
        agg = {
            "label": label,
            "display_name": _display_name(label),
            "seed_count": len(items),
            "seeds": sorted(int(item["seed"]) for item in items if item.get("seed") is not None),
        }
        for key in RUN_METRIC_KEYS:
            values = np.asarray(
                [float(item[key]) for item in items if item.get(key) is not None and np.isfinite(item.get(key))],
                dtype=np.float64,
            )
            agg[f"{key}_mean"] = float(np.mean(values)) if values.size else None
            agg[f"{key}_std"] = float(np.std(values)) if values.size else None
        aggregated.append(agg)
    return aggregated


def _write_csv(rows: Sequence[Dict[str, Any]], path: Path, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _annotate_bars(ax: plt.Axes, bars: Iterable[Any], values: Sequence[Optional[float]], *, percent: bool = False) -> None:
    for bar, value in zip(bars, values):
        if value is None or not math.isfinite(value):
            continue
        label = f"{value * 100:.1f}%" if percent else f"{value:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), label, ha="center", va="bottom", fontsize=8)


def _plot_dashboard(aggregated: Sequence[Dict[str, Any]], output_path: Path) -> None:
    if not aggregated:
        return

    labels = [item["display_name"] for item in aggregated]
    colors = [COLOR_MAP.get(item["label"], "#4C78A8") for item in aggregated]
    x = np.arange(len(aggregated))

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    axes = axes.flatten()

    avg_rewards = [item.get("avg_reward_mean") or 0.0 for item in aggregated]
    reward_err = [item.get("avg_reward_std") or 0.0 for item in aggregated]
    bars_reward = axes[0].bar(x, avg_rewards, yerr=reward_err, color=colors, capsize=4, alpha=0.9)
    axes[0].set_title("Average Reward (Mean ± Std)", fontsize=13, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].grid(True, axis="y", alpha=0.25, linestyle="--")
    _annotate_bars(axes[0], bars_reward, avg_rewards)

    success = [item.get("team_success_rate_mean") or 0.0 for item in aggregated]
    success_err = [item.get("team_success_rate_std") or 0.0 for item in aggregated]
    collision_free = [item.get("collision_free_rate_mean") or 0.0 for item in aggregated]
    collision_free_err = [item.get("collision_free_rate_std") or 0.0 for item in aggregated]
    width = 0.38
    bars1 = axes[1].bar(x - width / 2.0, success, width=width, yerr=success_err, color="#2CA02C", capsize=4, label="Team Success")
    bars2 = axes[1].bar(x + width / 2.0, collision_free, width=width, yerr=collision_free_err, color="#FF7F0E", capsize=4, label="Collision-Free")
    axes[1].set_title("Rates (Mean ± Std)", fontsize=13, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, axis="y", alpha=0.25, linestyle="--")
    axes[1].legend(fontsize=9)
    _annotate_bars(axes[1], bars1, success, percent=True)
    _annotate_bars(axes[1], bars2, collision_free, percent=True)

    terrain = np.asarray([item.get("avg_terrain_collision_count_mean") or 0.0 for item in aggregated], dtype=np.float64)
    obstacle = np.asarray([item.get("avg_obstacle_collision_count_mean") or 0.0 for item in aggregated], dtype=np.float64)
    inter_agent = np.asarray([item.get("avg_inter_agent_collision_count_mean") or 0.0 for item in aggregated], dtype=np.float64)
    axes[2].bar(x, terrain, color="#8C6D31", label="Terrain")
    axes[2].bar(x, obstacle, bottom=terrain, color="#E6550D", label="Obstacle")
    axes[2].bar(x, inter_agent, bottom=terrain + obstacle, color="#C44E52", label="Inter-Agent")
    axes[2].set_title("Collision Breakdown (Mean)", fontsize=13, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=25, ha="right")
    axes[2].grid(True, axis="y", alpha=0.25, linestyle="--")
    axes[2].legend(fontsize=9)

    goal_distance = [item.get("avg_team_final_goal_distance_mean") or 0.0 for item in aggregated]
    goal_err = [item.get("avg_team_final_goal_distance_std") or 0.0 for item in aggregated]
    bars_goal = axes[3].bar(x, goal_distance, yerr=goal_err, color=colors, capsize=4, alpha=0.9)
    axes[3].set_title("Avg Team Final Goal Distance", fontsize=13, fontweight="bold")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=25, ha="right")
    axes[3].grid(True, axis="y", alpha=0.25, linestyle="--")
    _annotate_bars(axes[3], bars_goal, goal_distance)

    inter_clearance = [item.get("avg_min_inter_agent_clearance_mean") or 0.0 for item in aggregated]
    inter_clearance_err = [item.get("avg_min_inter_agent_clearance_std") or 0.0 for item in aggregated]
    bars_clearance = axes[4].bar(x, inter_clearance, yerr=inter_clearance_err, color=colors, capsize=4, alpha=0.9)
    axes[4].set_title("Avg Min Inter-Agent Clearance", fontsize=13, fontweight="bold")
    axes[4].set_xticks(x)
    axes[4].set_xticklabels(labels, rotation=25, ha="right")
    axes[4].grid(True, axis="y", alpha=0.25, linestyle="--")
    _annotate_bars(axes[4], bars_clearance, inter_clearance)

    ranking_rows = sorted(
        aggregated,
        key=lambda item: (
            -(item.get("team_success_rate_mean") or float("-inf")),
            -(item.get("avg_reward_mean") or float("-inf")),
            item.get("avg_team_final_goal_distance_mean") if item.get("avg_team_final_goal_distance_mean") is not None else float("inf"),
        ),
    )
    summary_lines = ["Multi-seed Ranking"]
    for idx, item in enumerate(ranking_rows, start=1):
        summary_lines.append(
            f"{idx}. {item['display_name']}: "
            f"SR={((item.get('team_success_rate_mean') or 0.0) * 100):.1f}% | "
            f"R={item.get('avg_reward_mean') or 0.0:.1f} | "
            f"D={item.get('avg_team_final_goal_distance_mean') or 0.0:.1f} | "
            f"n={item.get('seed_count', 0)}"
        )
    axes[5].axis("off")
    axes[5].set_title("Cross-Seed Summary", fontsize=13, fontweight="bold")
    axes[5].text(0.02, 0.98, "\n".join(summary_lines), va="top", ha="left", fontsize=10, family="monospace")

    fig.suptitle("Level2 Official Evaluation Multi-Seed Summary", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate official Level2 evaluation results across seeds and algorithms.")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--labels", nargs="*", default=list(DEFAULT_LABELS))
    parser.add_argument("--log-root", type=Path, default=Path("/home/tang/matd3/logs"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path(
        f"/home/tang/matd3/diagnostics/level2_official_eval_multiseed_summary_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    for seed in args.seeds:
        for label in args.labels:
            result_path = _find_latest_eval_result(args.log_root, label, seed)
            if result_path is None:
                missing.append({"seed": seed, "label": label})
                continue
            rows.append(_load_result_row(label, seed, result_path))

    if missing and args.strict:
        pairs = ", ".join(f"{item['label']}@{item['seed']}" for item in missing)
        raise SystemExit(f"Missing official evaluation results: {pairs}")
    if not rows:
        raise SystemExit("No official evaluation results found.")

    label_index = {label: idx for idx, label in enumerate(args.labels)}
    rows.sort(key=lambda row: (int(row["seed"]), label_index.get(row["label"], 10**9)))
    aggregated = _aggregate_rows(rows, args.labels)

    runs_csv = output_dir / "official_eval_multiseed_all_runs.csv"
    agg_csv = output_dir / "official_eval_multiseed_aggregated.csv"
    summary_json = output_dir / "official_eval_multiseed_summary.json"
    dashboard_png = output_dir / "official_eval_multiseed_dashboard.png"

    run_fieldnames = ["label", "display_name", "seed", "episodes", *RUN_METRIC_KEYS, "results_path"]
    _write_csv(rows, runs_csv, run_fieldnames)

    agg_fieldnames = ["label", "display_name", "seed_count", "seeds"]
    for key in RUN_METRIC_KEYS:
        agg_fieldnames.extend([f"{key}_mean", f"{key}_std"])
    _write_csv(aggregated, agg_csv, agg_fieldnames)

    summary_payload = {
        "seeds": list(args.seeds),
        "labels": list(args.labels),
        "missing": missing,
        "runs": rows,
        "aggregated": aggregated,
        "artifacts": {
            "all_runs_csv": str(runs_csv),
            "aggregated_csv": str(agg_csv),
            "summary_json": str(summary_json),
            "dashboard_png": str(dashboard_png),
        },
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _plot_dashboard(aggregated, dashboard_png)

    print(f"输出目录: {output_dir}")
    print(f"All-runs CSV: {runs_csv}")
    print(f"Aggregated CSV: {agg_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Dashboard PNG: {dashboard_png}")
    if missing:
        print("缺失结果:")
        for item in missing:
            print(f"  - {item['label']} @ seed {item['seed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
