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

from summarize_level2_official_eval import COLOR_MAP, DISPLAY_NAME_MAP, eval_config_errors, _filename_suffix


DEFAULT_LABELS = [
    "matd3_full_dual_semantic",
    "matd3_cross_agent_ref_agent_success",
    "matd3_cross_agent_ref_agent_quality",
    "matd3_cross_agent_ref_soft_advantage",
    "matd3_cross_agent_ref_selector_mix",
    "matd3_cross_agent_ref_reward_to_success_selector_tail0",
    "matd3_cross_agent_ref_reward_to_success_selector_tail01",
    "matd3_cross_agent_ref_reward_to_success_selector",
    "matd3_cross_agent_ref_reward_to_success_selector_tail10",
    "matd3_cross_agent_ref_progress_gate",
    "matd3_cross_agent_ref_agent_success_behavior_label",
    "matd3_full_dual_semantic_cross_agent_ref",
    "matd3_cross_agent_ref_no_quality_gate",
    "matd3_cross_agent_ref_behavior_label",
    "matd3_collapsed_replay",
    "matd3_no_corrected_target_reconstruction",
]

RUN_METRIC_KEYS = [
    "team_success_rate",
    "agent_success_rate_any",
    "agent_success_rate_two_or_more",
    "all_reached_without_safe_team_success_rate",
    "two_success_not_team_rate",
    "all_safe_not_team_rate",
    "unsafe_reached_agent_slot_rate",
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
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _partial_success_rates_from_details(data: Dict[str, Any]) -> Dict[str, float]:
    """Compute partial-arrival rates from per-episode agent success flags."""
    episode_details = data.get("episode_details")
    if not isinstance(episode_details, list) or not episode_details:
        return {}

    any_success = 0
    two_or_more_success = 0
    counted = 0
    for episode in episode_details:
        if not isinstance(episode, dict):
            continue
        flags = episode.get("agent_success_flags")
        if not isinstance(flags, list) or not flags:
            continue

        parsed_flags: List[int] = []
        for flag in flags:
            parsed = _safe_int(flag)
            if parsed is None:
                continue
            parsed_flags.append(1 if parsed > 0 else 0)
        if not parsed_flags:
            continue

        success_count = sum(parsed_flags)
        any_success += int(success_count >= 1)
        two_or_more_success += int(success_count >= 2)
        counted += 1

    if counted <= 0:
        return {}
    return {
        "agent_success_rate_any": float(any_success / counted),
        "agent_success_rate_two_or_more": float(two_or_more_success / counted),
    }


def _display_name(label: str) -> str:
    return DISPLAY_NAME_MAP.get(label, label)


def _result_path(log_root: Path, run_tag: str, label: str, train_seed: int, eval_seed: int) -> Path:
    return (
        log_root
        / f"{run_tag}_{label}_trainseed{int(train_seed)}_testseed{int(eval_seed)}"
        / "evaluation_official"
        / "evaluation_results.json"
    )


def _load_result_row(
    *,
    label: str,
    train_seed: int,
    eval_seed: int,
    result_path: Path,
) -> Dict[str, Any]:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
    setup = data.get("evaluation_setup", {}) if isinstance(data.get("evaluation_setup"), dict) else {}
    config_errors = eval_config_errors(data)
    row: Dict[str, Any] = {
        "label": label,
        "display_name": _display_name(label),
        "train_seed": int(train_seed),
        "eval_seed": int(eval_seed),
        "episodes": _safe_int(summary.get("episodes", data.get("episodes"))),
        "success_episode_count": _safe_int(summary.get("success_episode_count")),
        "config_valid": not config_errors,
        "config_errors": "; ".join(config_errors),
        "action_force_ratio_source": str(setup.get("action_force_ratio_source", "") or ""),
        "use_quadrotor_dynamics": setup.get("use_quadrotor_dynamics"),
        "use_dynamic_obstacles": setup.get("use_dynamic_obstacles"),
        "results_path": str(result_path),
    }
    for key in RUN_METRIC_KEYS:
        row[key] = _safe_float(summary.get(key))
    for key, value in _partial_success_rates_from_details(data).items():
        if row.get(key) is None:
            row[key] = value
    return row


def _aggregate_rows(rows: Sequence[Dict[str, Any]], labels: Sequence[str], group_key: Optional[str] = None) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["label"], row.get(group_key) if group_key else None)
        grouped[key].append(row)

    aggregated: List[Dict[str, Any]] = []
    for label in labels:
        keys = [key for key in grouped if key[0] == label]
        keys.sort(key=lambda item: (-1 if item[1] is None else int(item[1])))
        for key in keys:
            items = grouped.get(key, [])
            if not items:
                continue
            agg: Dict[str, Any] = {
                "label": label,
                "display_name": _display_name(label),
                "run_count": len(items),
                "train_seeds": sorted({int(item["train_seed"]) for item in items}),
                "eval_seeds": sorted({int(item["eval_seed"]) for item in items}),
            }
            if group_key:
                agg[group_key] = key[1]
            for metric in RUN_METRIC_KEYS:
                values = np.asarray(
                    [float(item[metric]) for item in items if item.get(metric) is not None],
                    dtype=np.float64,
                )
                agg[f"{metric}_mean"] = float(np.mean(values)) if values.size else None
                agg[f"{metric}_std"] = float(np.std(values)) if values.size else None
            aggregated.append(agg)
    return aggregated


def _write_csv(rows: Sequence[Dict[str, Any]], path: Path, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _annotate_bars(ax: plt.Axes, bars: Iterable[Any], values: Sequence[Optional[float]], *, percent: bool = False) -> None:
    for bar, value in zip(bars, values):
        if value is None or not math.isfinite(value):
            continue
        text = f"{value * 100:.1f}%" if percent else f"{value:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), text, ha="center", va="bottom", fontsize=8)


def _plot_dashboard(aggregated: Sequence[Dict[str, Any]], output_path: Path) -> None:
    if not aggregated:
        return
    labels = [item["display_name"] for item in aggregated]
    colors = [COLOR_MAP.get(item["label"], "#4C78A8") for item in aggregated]
    x = np.arange(len(aggregated))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    team_success = [item.get("team_success_rate_mean") or 0.0 for item in aggregated]
    team_success_err = [item.get("team_success_rate_std") or 0.0 for item in aggregated]
    bars = axes[0].bar(x, team_success, yerr=team_success_err, color=colors, capsize=4)
    axes[0].set_title("Team Success Rate", fontweight="bold")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].grid(True, axis="y", alpha=0.25, linestyle="--")
    _annotate_bars(axes[0], bars, team_success, percent=True)

    collision_free = [item.get("collision_free_rate_mean") or 0.0 for item in aggregated]
    collision_free_err = [item.get("collision_free_rate_std") or 0.0 for item in aggregated]
    bars = axes[1].bar(x, collision_free, yerr=collision_free_err, color=colors, capsize=4)
    axes[1].set_title("Collision-Free Rate", fontweight="bold")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25, linestyle="--")
    _annotate_bars(axes[1], bars, collision_free, percent=True)

    distance = [item.get("avg_team_final_goal_distance_mean") or 0.0 for item in aggregated]
    distance_err = [item.get("avg_team_final_goal_distance_std") or 0.0 for item in aggregated]
    bars = axes[2].bar(x, distance, yerr=distance_err, color=colors, capsize=4)
    axes[2].set_title("Team Final Goal Distance", fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20, ha="right")
    axes[2].grid(True, axis="y", alpha=0.25, linestyle="--")
    _annotate_bars(axes[2], bars, distance)

    rewards = [item.get("avg_reward_mean") or 0.0 for item in aggregated]
    reward_err = [item.get("avg_reward_std") or 0.0 for item in aggregated]
    bars = axes[3].bar(x, rewards, yerr=reward_err, color=colors, capsize=4)
    axes[3].set_title("Average Reward", fontweight="bold")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=20, ha="right")
    axes[3].grid(True, axis="y", alpha=0.25, linestyle="--")
    _annotate_bars(axes[3], bars, rewards)

    terrain = np.asarray([item.get("avg_terrain_collision_count_mean") or 0.0 for item in aggregated])
    obstacle = np.asarray([item.get("avg_obstacle_collision_count_mean") or 0.0 for item in aggregated])
    inter_agent = np.asarray([item.get("avg_inter_agent_collision_count_mean") or 0.0 for item in aggregated])
    axes[4].bar(x, terrain, color="#6BAED6", label="Terrain")
    axes[4].bar(x, obstacle, bottom=terrain, color="#FD8D3C", label="Obstacle")
    axes[4].bar(x, inter_agent, bottom=terrain + obstacle, color="#74C476", label="Inter-Agent")
    axes[4].set_title("Collision Breakdown", fontweight="bold")
    axes[4].set_xticks(x)
    axes[4].set_xticklabels(labels, rotation=20, ha="right")
    axes[4].grid(True, axis="y", alpha=0.25, linestyle="--")
    axes[4].legend(fontsize=9)

    any_success = [item.get("agent_success_rate_any_mean") or 0.0 for item in aggregated]
    two_success = [item.get("agent_success_rate_two_or_more_mean") or 0.0 for item in aggregated]
    width = 0.36
    bars_any = axes[5].bar(x - width / 2, any_success, width=width, color="#2CA02C", label="Any-Agent")
    bars_two = axes[5].bar(x + width / 2, two_success, width=width, color="#9467BD", label="Two-Agent")
    axes[5].set_title("Partial Arrival Rates", fontweight="bold")
    axes[5].set_ylim(0.0, 1.05)
    axes[5].set_xticks(x)
    axes[5].set_xticklabels(labels, rotation=20, ha="right")
    axes[5].grid(True, axis="y", alpha=0.25, linestyle="--")
    axes[5].legend(fontsize=9)
    _annotate_bars(axes[5], bars_any, any_success, percent=True)
    _annotate_bars(axes[5], bars_two, two_success, percent=True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize 10-seed x 30-episode dual-semantics eval sweep.")
    parser.add_argument("--log-root", default="/home/tang/matd3/logs")
    parser.add_argument("--run-tag", default="level2_dual_semantics_eval10x30")
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[101, 202, 936487])
    parser.add_argument(
        "--eval-seeds",
        type=int,
        nargs="+",
        default=[30088, 30188, 30288, 30388, 30488, 30588, 30688, 30788, 30888, 30988],
    )
    parser.add_argument("--expected-episodes", type=int, default=30)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--allow-forced-fr",
        action="store_true",
        help="Accept evaluation runs with action_force_ratio_source=forced_override; use for fixed-FR control evals.",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--filename-tag", default="", help="Optional suffix for output files, e.g. model_fr")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_root = Path(args.log_root).resolve()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path(
        f"/home/tang/matd3/diagnostics/{args.run_tag}_summary_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    for label in args.labels:
        for train_seed in args.train_seeds:
            for eval_seed in args.eval_seeds:
                result_path = _result_path(log_root, args.run_tag, label, int(train_seed), int(eval_seed))
                if not result_path.exists():
                    missing.append(
                        {
                            "label": label,
                            "train_seed": int(train_seed),
                            "eval_seed": int(eval_seed),
                            "reason": "missing_result",
                            "expected_path": str(result_path),
                        }
                    )
                    continue
                row = _load_result_row(label=label, train_seed=int(train_seed), eval_seed=int(eval_seed), result_path=result_path)
                if args.allow_forced_fr and row.get("config_errors"):
                    errors = [
                        item
                        for item in str(row.get("config_errors") or "").split("; ")
                        if item and item != "forced_override action_force_ratio_source"
                    ]
                    row["config_errors"] = "; ".join(errors)
                    row["config_valid"] = not errors
                if row.get("episodes") != int(args.expected_episodes):
                    missing.append(
                        {
                            "label": label,
                            "train_seed": int(train_seed),
                            "eval_seed": int(eval_seed),
                            "reason": f"episodes={row.get('episodes')}",
                            "expected_episodes": int(args.expected_episodes),
                            "results_path": str(result_path),
                        }
                    )
                if not row.get("config_valid", False):
                    missing.append(
                        {
                            "label": label,
                            "train_seed": int(train_seed),
                            "eval_seed": int(eval_seed),
                            "reason": row.get("config_errors"),
                            "results_path": str(result_path),
                        }
                    )
                rows.append(row)

    if missing and args.strict:
        print(f"[strict] missing/incomplete runs: {len(missing)}")
        for item in missing[:20]:
            print(item)
        raise SystemExit(2)

    aggregate_all = _aggregate_rows(rows, args.labels)
    aggregate_by_train_seed = _aggregate_rows(rows, args.labels, group_key="train_seed")

    run_fieldnames = [
        "label",
        "display_name",
        "train_seed",
        "eval_seed",
        "episodes",
        "success_episode_count",
        "config_valid",
        "config_errors",
        "action_force_ratio_source",
        "use_quadrotor_dynamics",
        "use_dynamic_obstacles",
        *RUN_METRIC_KEYS,
        "results_path",
    ]
    agg_fieldnames = [
        "label",
        "display_name",
        "run_count",
        "train_seeds",
        "eval_seeds",
    ]
    train_agg_fieldnames = [
        "label",
        "display_name",
        "train_seed",
        "run_count",
        "train_seeds",
        "eval_seeds",
    ]
    for metric in RUN_METRIC_KEYS:
        agg_fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
        train_agg_fieldnames.extend([f"{metric}_mean", f"{metric}_std"])

    suffix = _filename_suffix(args.filename_tag)
    runs_csv = output_dir / f"dual_semantics_eval10x30_all_runs{suffix}.csv"
    agg_csv = output_dir / f"dual_semantics_eval10x30_aggregated{suffix}.csv"
    train_csv = output_dir / f"dual_semantics_eval10x30_by_train_seed{suffix}.csv"
    summary_json = output_dir / f"dual_semantics_eval10x30_summary{suffix}.json"
    dashboard_png = output_dir / f"dual_semantics_eval10x30_dashboard{suffix}.png"

    _write_csv(rows, runs_csv, run_fieldnames)
    _write_csv(aggregate_all, agg_csv, agg_fieldnames)
    _write_csv(aggregate_by_train_seed, train_csv, train_agg_fieldnames)
    _plot_dashboard(aggregate_all, dashboard_png)

    payload = {
        "run_tag": args.run_tag,
        "labels": list(args.labels),
        "train_seeds": [int(seed) for seed in args.train_seeds],
        "eval_seeds": [int(seed) for seed in args.eval_seeds],
        "expected_episodes": int(args.expected_episodes),
        "missing": missing,
        "runs": rows,
        "aggregated": aggregate_all,
        "aggregated_by_train_seed": aggregate_by_train_seed,
        "artifacts": {
            "runs_csv": str(runs_csv),
            "aggregated_csv": str(agg_csv),
            "by_train_seed_csv": str(train_csv),
            "summary_json": str(summary_json),
            "dashboard_png": str(dashboard_png),
        },
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Loaded runs: {len(rows)}")
    print(f"Missing/incomplete: {len(missing)}")
    print(f"Summary dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
