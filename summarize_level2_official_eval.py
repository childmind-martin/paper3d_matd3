#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LABELS = [
    "matd3_single_q",
    "matd3_dual_q",
    "matd3_separated_gradient",
    "matd3_separated_hybrid_actor",
    "matd3_separated_hybrid_actor_alpha20",
    "maddpg_baseline",
    "maddpg_dual_q",
    "maddpg_separated_gradient",
    "mappo_baseline",
    "mappo_fusion_only",
    "mappo_separated_gradient",
]

DUAL_SEMANTICS_LABELS = [
    "matd3_full_dual_semantic",
    "matd3_cross_agent_ref_agent_success",
    "matd3_cross_agent_ref_agent_quality",
    "matd3_cross_agent_ref_soft_advantage",
    "matd3_cross_agent_ref_selector_mix",
    "matd3_cross_agent_ref_reward_to_success_selector",
    "matd3_cross_agent_ref_progress_gate",
    "matd3_cross_agent_ref_agent_success_behavior_label",
    "matd3_full_dual_semantic_cross_agent_ref",
    "matd3_cross_agent_ref_no_quality_gate",
    "matd3_cross_agent_ref_behavior_label",
    "matd3_collapsed_replay",
    "matd3_no_corrected_target_reconstruction",
]


DISPLAY_NAME_MAP = {
    "matd3_single_q": "MATD3 Single-Q",
    "matd3_dual_q": "MATD3 Dual-Q",
    "matd3_separated_gradient": "MATD3 Sep-Grad",
    "matd3_separated_hybrid_actor": "MATD3 Hybrid alpha=0.80",
    "matd3_separated_hybrid_actor_alpha20": "MATD3 Hybrid alpha=0.20",
    "maddpg_baseline": "MADDPG Baseline",
    "maddpg_dual_q": "MADDPG Dual-Q",
    "maddpg_separated_gradient": "MADDPG Sep-Grad",
    "mappo_baseline": "MAPPO Baseline",
    "mappo_fusion_only": "MAPPO Fusion-Only",
    "mappo_separated_gradient": "MAPPO Sep-Grad",
    "matd3_full_dual_semantic": "Full Dual-Semantic",
    "matd3_collapsed_replay": "Collapsed Replay",
    "matd3_no_corrected_target_reconstruction": "No Corrected Target Recon",
    "matd3_cross_agent_ref_agent_success": "CrossRef Agent Success",
    "matd3_cross_agent_ref_agent_quality": "CrossRef Agent Quality",
    "matd3_cross_agent_ref_soft_advantage": "CrossRef Soft Advantage",
    "matd3_cross_agent_ref_selector_mix": "CrossRef Selector Mix",
    "matd3_cross_agent_ref_reward_to_success_selector": "CrossRef Reward-to-Success Selector",
    "matd3_cross_agent_ref_progress_gate": "CrossRef Progress Gate",
    "matd3_cross_agent_ref_agent_success_behavior_label": "CrossRef Success Behavior Label",
    "matd3_full_dual_semantic_cross_agent_ref": "Full DS + CrossRef",
    "matd3_cross_agent_ref_no_quality_gate": "CrossRef No Gate",
    "matd3_cross_agent_ref_behavior_label": "CrossRef Behavior Label",
}


COLOR_MAP = {
    "matd3_single_q": "#4C78A8",
    "matd3_dual_q": "#2E5EAA",
    "matd3_separated_gradient": "#1F77B4",
    "matd3_separated_hybrid_actor": "#0B559F",
    "matd3_separated_hybrid_actor_alpha20": "#08306B",
    "maddpg_baseline": "#54A24B",
    "maddpg_dual_q": "#2CA02C",
    "maddpg_separated_gradient": "#1B7F1B",
    "mappo_baseline": "#F58518",
    "mappo_fusion_only": "#E45756",
    "mappo_separated_gradient": "#C83E4D",
    "matd3_full_dual_semantic": "#1F77B4",
    "matd3_collapsed_replay": "#B279A2",
    "matd3_no_corrected_target_reconstruction": "#E45756",
    "matd3_cross_agent_ref_agent_success": "#009E73",
    "matd3_cross_agent_ref_agent_quality": "#56B4E9",
    "matd3_cross_agent_ref_soft_advantage": "#CC79A7",
    "matd3_cross_agent_ref_selector_mix": "#D55E00",
    "matd3_cross_agent_ref_reward_to_success_selector": "#6A3D9A",
    "matd3_cross_agent_ref_progress_gate": "#2CA02C",
    "matd3_cross_agent_ref_agent_success_behavior_label": "#8C564B",
    "matd3_full_dual_semantic_cross_agent_ref": "#2CA02C",
    "matd3_cross_agent_ref_no_quality_gate": "#FF7F0E",
    "matd3_cross_agent_ref_behavior_label": "#7F7F7F",
}


CSV_FIELDNAMES = [
    "label",
    "display_name",
    "seed",
    "episodes",
    "team_success_rate",
    "success_episode_count",
    "avg_reward",
    "std_reward",
    "max_reward",
    "min_reward",
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
    "config_valid",
    "config_errors",
    "action_force_ratio_source",
    "use_quadrotor_dynamics",
    "use_dynamic_obstacles",
    "results_path",
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


def _filename_suffix(tag: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(tag or "").strip())
    return f"_{clean}" if clean else ""


def _to_bool_optional(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None


def eval_config_errors(data: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    setup = data.get("evaluation_setup") if isinstance(data, dict) else None
    if not isinstance(setup, dict):
        return ["missing evaluation_setup"]

    fr_source = str(setup.get("action_force_ratio_source", "") or "").strip()
    if not fr_source:
        errors.append("missing action_force_ratio_source")
    elif fr_source == "forced_override":
        errors.append("forced_override action_force_ratio_source")

    use_quad = _to_bool_optional(setup.get("use_quadrotor_dynamics"))
    if use_quad is not True:
        errors.append(f"use_quadrotor_dynamics={setup.get('use_quadrotor_dynamics')}")

    use_dynamic_obstacles = _to_bool_optional(setup.get("use_dynamic_obstacles"))
    if use_dynamic_obstacles is not True:
        errors.append(f"use_dynamic_obstacles={setup.get('use_dynamic_obstacles')}")

    for key in (
        "gravity",
        "control_accel_gain",
        "agent_max_speed",
        "agent_accel",
        "damping",
        "simulation_dt",
        "quadrotor_attitude_response_time",
        "quadrotor_psi_cmd",
        "action_range_x",
        "action_range_y",
        "action_range_z",
    ):
        if _safe_float(setup.get(key)) is None:
            errors.append(f"missing {key}")
    return errors


def _find_latest_eval_result(log_root: Path, label: str, seed: int, run_tag: str = "") -> Optional[Path]:
    if run_tag:
        patterns = [
            f"{run_tag}_{label}_seed{seed}_*/evaluation_official/evaluation_results.json",
        ]
    else:
        patterns = [
            f"level2_ms_official_{label}_seed{seed}_*/evaluation_official/evaluation_results.json",
            f"level2_retrain_{label}_seed{seed}_*/evaluation_official/evaluation_results.json",
            f"*{label}_seed{seed}_*/evaluation_official/evaluation_results.json",
        ]
    candidates: List[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for candidate in log_root.glob(pattern):
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(candidate)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_result_row(label: str, seed: int, result_path: Path) -> Dict[str, Any]:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
    setup = data.get("evaluation_setup", {}) if isinstance(data.get("evaluation_setup"), dict) else {}
    config_errors = eval_config_errors(data)
    row = {
        "label": label,
        "display_name": _display_name(label),
        "seed": seed,
        "episodes": _safe_int(summary.get("episodes", data.get("episodes"))),
        "team_success_rate": _safe_float(summary.get("team_success_rate")),
        "success_episode_count": _safe_int(summary.get("success_episode_count")),
        "avg_reward": _safe_float(summary.get("avg_reward", data.get("avg_reward"))),
        "std_reward": _safe_float(summary.get("std_reward", data.get("std_reward"))),
        "max_reward": _safe_float(summary.get("max_reward", data.get("max_reward"))),
        "min_reward": _safe_float(summary.get("min_reward", data.get("min_reward"))),
        "avg_collision_count": _safe_float(summary.get("avg_collision_count")),
        "avg_terrain_collision_count": _safe_float(summary.get("avg_terrain_collision_count")),
        "avg_obstacle_collision_count": _safe_float(summary.get("avg_obstacle_collision_count")),
        "avg_inter_agent_collision_count": _safe_float(summary.get("avg_inter_agent_collision_count")),
        "collision_free_rate": _safe_float(summary.get("collision_free_rate")),
        "inter_agent_collision_free_rate": _safe_float(summary.get("inter_agent_collision_free_rate")),
        "avg_team_final_goal_distance": _safe_float(summary.get("avg_team_final_goal_distance")),
        "avg_agent_final_goal_distance": _safe_float(summary.get("avg_agent_final_goal_distance")),
        "avg_min_inter_agent_clearance": _safe_float(summary.get("avg_min_inter_agent_clearance")),
        "avg_steps": _safe_float(summary.get("avg_steps")),
        "config_valid": not config_errors,
        "config_errors": "; ".join(config_errors),
        "action_force_ratio_source": str(setup.get("action_force_ratio_source", "") or ""),
        "use_quadrotor_dynamics": setup.get("use_quadrotor_dynamics"),
        "use_dynamic_obstacles": setup.get("use_dynamic_obstacles"),
        "results_path": str(result_path),
    }
    return row


def _write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_FIELDNAMES})


def _write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _annotate_bars(ax: plt.Axes, bars: Iterable[Any], values: Sequence[Optional[float]], *, percent: bool = False) -> None:
    for bar, value in zip(bars, values):
        if value is None or not math.isfinite(value):
            continue
        label = f"{value * 100:.1f}%" if percent else f"{value:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )


def _plot_dashboard(rows: Sequence[Dict[str, Any]], output_path: Path, *, seed: int) -> None:
    if not rows:
        return

    labels = [row["display_name"] for row in rows]
    colors = [COLOR_MAP.get(row["label"], "#4C78A8") for row in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    axes = axes.flatten()

    avg_rewards = [row.get("avg_reward") or 0.0 for row in rows]
    reward_bars = axes[0].bar(x, avg_rewards, color=colors, alpha=0.9)
    axes[0].set_title("Average Reward", fontsize=13, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].grid(True, axis="y", alpha=0.25, linestyle="--")
    _annotate_bars(axes[0], reward_bars, avg_rewards)

    success_rates = [row.get("team_success_rate") or 0.0 for row in rows]
    collision_free_rates = [row.get("collision_free_rate") or 0.0 for row in rows]
    width = 0.38
    bars1 = axes[1].bar(x - width / 2.0, success_rates, width=width, color="#2CA02C", label="Team Success Rate")
    bars2 = axes[1].bar(x + width / 2.0, collision_free_rates, width=width, color="#FF7F0E", label="Collision-Free Rate")
    axes[1].set_title("Success / Collision-Free Rates", fontsize=13, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, axis="y", alpha=0.25, linestyle="--")
    axes[1].legend(fontsize=9)
    _annotate_bars(axes[1], bars1, success_rates, percent=True)
    _annotate_bars(axes[1], bars2, collision_free_rates, percent=True)

    terrain = np.asarray([row.get("avg_terrain_collision_count") or 0.0 for row in rows], dtype=np.float64)
    obstacle = np.asarray([row.get("avg_obstacle_collision_count") or 0.0 for row in rows], dtype=np.float64)
    inter_agent = np.asarray([row.get("avg_inter_agent_collision_count") or 0.0 for row in rows], dtype=np.float64)
    axes[2].bar(x, terrain, color="#8C6D31", label="Terrain")
    axes[2].bar(x, obstacle, bottom=terrain, color="#E6550D", label="Obstacle")
    axes[2].bar(x, inter_agent, bottom=terrain + obstacle, color="#C44E52", label="Inter-Agent")
    axes[2].set_title("Average Collision Breakdown", fontsize=13, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=25, ha="right")
    axes[2].grid(True, axis="y", alpha=0.25, linestyle="--")
    axes[2].legend(fontsize=9)

    goal_distance = [row.get("avg_team_final_goal_distance") or 0.0 for row in rows]
    bars_goal = axes[3].bar(x, goal_distance, color=colors, alpha=0.9)
    axes[3].set_title("Average Team Final Goal Distance", fontsize=13, fontweight="bold")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=25, ha="right")
    axes[3].grid(True, axis="y", alpha=0.25, linestyle="--")
    _annotate_bars(axes[3], bars_goal, goal_distance)

    inter_clearance = [row.get("avg_min_inter_agent_clearance") or 0.0 for row in rows]
    bars_clearance = axes[4].bar(x, inter_clearance, color=colors, alpha=0.9)
    axes[4].set_title("Average Min Inter-Agent Clearance", fontsize=13, fontweight="bold")
    axes[4].set_xticks(x)
    axes[4].set_xticklabels(labels, rotation=25, ha="right")
    axes[4].grid(True, axis="y", alpha=0.25, linestyle="--")
    _annotate_bars(axes[4], bars_clearance, inter_clearance)

    ranking_rows = sorted(
        rows,
        key=lambda row: (
            -(row.get("team_success_rate") or float("-inf")),
            -(row.get("avg_reward") or float("-inf")),
            row.get("avg_team_final_goal_distance") if row.get("avg_team_final_goal_distance") is not None else float("inf"),
        ),
    )
    summary_lines = [
        f"Seed: {seed}",
        f"Algorithms: {len(rows)}",
        "",
        "Ranking",
    ]
    for idx, row in enumerate(ranking_rows, start=1):
        summary_lines.append(
            f"{idx}. {row['display_name']}: "
            f"SR={((row.get('team_success_rate') or 0.0) * 100):.1f}% | "
            f"R={row.get('avg_reward') or 0.0:.1f} | "
            f"D={row.get('avg_team_final_goal_distance') or 0.0:.1f}"
        )
    axes[5].axis("off")
    axes[5].set_title("Cross-Algorithm Summary", fontsize=13, fontweight="bold")
    axes[5].text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    fig.suptitle(f"Level2 Official Evaluation Cross-Algorithm Summary (seed={seed})", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_terminal_lines(rows: Sequence[Dict[str, Any]]) -> List[str]:
    header = (
        f"{'label':<34} "
        f"{'succ%':>7} "
        f"{'reward':>12} "
        f"{'coll':>10} "
        f"{'terr':>10} "
        f"{'obs':>10} "
        f"{'team_d':>10}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['label']:<34} "
            f"{((row.get('team_success_rate') or 0.0) * 100):>6.1f}% "
            f"{(row.get('avg_reward') or 0.0):>12.2f} "
            f"{(row.get('avg_collision_count') or 0.0):>10.2f} "
            f"{(row.get('avg_terrain_collision_count') or 0.0):>10.2f} "
            f"{(row.get('avg_obstacle_collision_count') or 0.0):>10.2f} "
            f"{(row.get('avg_team_final_goal_distance') or 0.0):>10.2f}"
        )
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate official Level2 evaluation results across algorithms.")
    parser.add_argument("--seed", type=int, required=True, help="Seed id, e.g. 202")
    parser.add_argument("--log-root", type=Path, default=Path("/home/tang/matd3/logs"))
    parser.add_argument("--labels", nargs="*", default=list(DEFAULT_LABELS))
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for csv/json/png")
    parser.add_argument("--run-tag", default="", help="Optional log prefix, e.g. level2_ms_checkpoint_fr_eval")
    parser.add_argument("--filename-tag", default="", help="Optional suffix for output files, e.g. model_fr")
    parser.add_argument("--strict", action="store_true", help="Fail if any label is missing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path(
        f"/home/tang/matd3/diagnostics/level2_official_eval_summary_seed{args.seed}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for label in args.labels:
        result_path = _find_latest_eval_result(args.log_root, label, args.seed, args.run_tag)
        if result_path is None:
            missing.append(label)
            continue
        row = _load_result_row(label, args.seed, result_path)
        rows.append(row)
        if not row.get("config_valid", False):
            missing.append(f"{label}({row.get('config_errors')})")

    if missing and args.strict:
        raise SystemExit(f"Missing official evaluation results for: {', '.join(missing)}")
    if not rows:
        raise SystemExit("No official evaluation results found.")

    order_index = {label: idx for idx, label in enumerate(args.labels)}
    rows.sort(key=lambda row: order_index.get(row["label"], 10**9))

    suffix = _filename_suffix(args.filename_tag)
    csv_path = output_dir / f"official_eval_cross_algo_summary{suffix}.csv"
    json_path = output_dir / f"official_eval_cross_algo_summary{suffix}.json"
    png_path = output_dir / f"official_eval_cross_algo_summary{suffix}.png"

    _write_csv(rows, csv_path)
    _write_json(
        {
            "seed": args.seed,
            "labels": list(args.labels),
            "missing_labels": missing,
            "rows": rows,
            "artifacts": {
                "csv": str(csv_path),
                "json": str(json_path),
                "png": str(png_path),
            },
        },
        json_path,
    )
    _plot_dashboard(rows, png_path, seed=args.seed)

    print(f"输出目录: {output_dir}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"PNG: {png_path}")
    if missing:
        print(f"缺失算法: {', '.join(missing)}")
    print()
    for line in _build_terminal_lines(rows):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
