#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TRIO = [
    "matd3_separated_gradient",
    "matd3_dual_q",
    "matd3_separated_hybrid_actor",
]

STYLE = {
    "matd3_separated_gradient": {"abbr": "M3-Sep", "color": "#1f77b4", "marker": "o"},
    "matd3_dual_q": {"abbr": "M3-Uni", "color": "#d62728", "marker": "s"},
    "matd3_separated_hybrid_actor": {"abbr": "M3-H80", "color": "#ff7f0e", "marker": "D"},
}


DEFAULT_SCENARIOS = [
    {
        "name": "Group A\nFixed map",
        "train_summary": "/home/tang/matd3/ablation_experiments/multi_seed_groupA_20260412_205350/plots/summary_20260414_221440.json",
        "post_summary": "/home/tang/matd3/ablation_experiments/multi_seed_groupA_20260412_205350/plots/summary_20260414_221440.json",
    },
    {
        "name": "Group B\nRandom obstacles +\nsemi-random terrain",
        "train_summary": "/home/tang/matd3/ablation_experiments/multi_seed_groupB_20260406_230829/plots/summary_20260414_234950.json",
        "post_summary": "/home/tang/matd3/ablation_experiments/multi_seed_groupB_20260406_230829/plots/summary_20260414_234950.json",
    },
    {
        "name": "Group B Testset-2\nHeld-out evaluation",
        "train_summary": "/home/tang/matd3/ablation_experiments/multi_seed_groupB_20260331_220752_testset2_20260409/plots/training_summary_from_logs_20260413_085818.json",
        "post_summary": "/home/tang/matd3/ablation_experiments/multi_seed_groupB_20260331_220752_testset2_20260409/plots/summary_20260413_032743.json",
    },
]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_by_label(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {item["label"]: item for item in summary.get("aggregated_experiments", [])}


def _finite_values(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            out.append(numeric)
    return out


def _mean_std(values: Iterable[Any]) -> Tuple[float | None, float | None, int]:
    usable = _finite_values(values)
    if not usable:
        return None, None, 0
    arr = np.asarray(usable, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr)), int(arr.size)


def _metric_from_seed_values(
    row: Dict[str, Any],
    key: str,
    *,
    nested_key: str | None = None,
) -> Tuple[float | None, float | None, int]:
    values = []
    for seed_value in row.get("seed_values", []) or []:
        source = seed_value
        if nested_key is not None:
            nested = seed_value.get(nested_key)
            source = nested if isinstance(nested, dict) else {}
        values.append(source.get(key))
    return _mean_std(values)


def _scenario_rows(train_summary_path: Path, post_summary_path: Path) -> Dict[str, Dict[str, float | int | None]]:
    train_summary = _summary_by_label(_load_json(train_summary_path))
    post_summary = _summary_by_label(_load_json(post_summary_path))

    rows: Dict[str, Dict[str, float | int | None]] = {}
    for label in TRIO:
        train_row = train_summary[label]
        post_row = post_summary[label]

        train_success_mean, train_success_std, train_success_n = _metric_from_seed_values(
            train_row, "tail100_success_mean"
        )
        train_reward_mean, train_reward_std, _ = _metric_from_seed_values(
            train_row, "tail100_reward_mean"
        )
        post_success_mean, post_success_std, post_success_n = _metric_from_seed_values(
            post_row, "team_success_rate", nested_key="post_eval_summary"
        )
        post_reward_mean, post_reward_std, _ = _metric_from_seed_values(
            post_row, "avg_reward", nested_key="post_eval_summary"
        )
        post_collision_free_mean, post_collision_free_std, _ = _metric_from_seed_values(
            post_row, "collision_free_rate", nested_key="post_eval_summary"
        )
        post_goal_mean, post_goal_std, _ = _metric_from_seed_values(
            post_row, "avg_team_final_goal_distance", nested_key="post_eval_summary"
        )
        post_path_eff_mean, post_path_eff_std, _ = _metric_from_seed_values(
            post_row, "avg_team_path_efficiency", nested_key="post_eval_summary"
        )
        post_arrival_mean, post_arrival_std, post_arrival_n = _metric_from_seed_values(
            post_row, "avg_arrival_step_success_only", nested_key="post_eval_summary"
        )

        rows[label] = {
            "train_tail_success_mean": train_success_mean,
            "train_tail_success_std": train_success_std,
            "train_tail_success_n": train_success_n,
            "train_tail_reward_mean": train_reward_mean,
            "train_tail_reward_std": train_reward_std,
            "post_success_mean": post_success_mean,
            "post_success_std": post_success_std,
            "post_success_n": post_success_n,
            "post_reward_mean": post_reward_mean,
            "post_reward_std": post_reward_std,
            "post_collision_free_mean": post_collision_free_mean,
            "post_collision_free_std": post_collision_free_std,
            "post_goal_mean": post_goal_mean,
            "post_goal_std": post_goal_std,
            "post_path_eff_mean": post_path_eff_mean,
            "post_path_eff_std": post_path_eff_std,
            "post_arrival_success_only_mean": post_arrival_mean,
            "post_arrival_success_only_std": post_arrival_std,
            "post_arrival_success_only_n": post_arrival_n,
            "success_gap": (
                None
                if train_success_mean is None or post_success_mean is None
                else post_success_mean - train_success_mean
            ),
        }
    return rows


def _write_csv(
    scenarios: list[Dict[str, str]],
    scenario_metrics: Dict[str, Dict[str, Dict[str, float | int | None]]],
    output_path: Path,
) -> None:
    fieldnames = [
        "scenario",
        "label",
        "abbr",
        "train_tail_success_mean",
        "train_tail_success_std",
        "train_tail_success_n",
        "train_tail_reward_mean",
        "train_tail_reward_std",
        "post_success_mean",
        "post_success_std",
        "post_success_n",
        "post_reward_mean",
        "post_reward_std",
        "post_collision_free_mean",
        "post_collision_free_std",
        "post_goal_mean",
        "post_goal_std",
        "post_path_eff_mean",
        "post_path_eff_std",
        "post_arrival_success_only_mean",
        "post_arrival_success_only_std",
        "post_arrival_success_only_n",
        "success_gap",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scenario in scenarios:
            scenario_name = scenario["name"]
            scenario_csv_name = scenario_name.replace("\n", " ")
            for label in TRIO:
                row = {"scenario": scenario_csv_name, "label": label, "abbr": STYLE[label]["abbr"]}
                row.update(scenario_metrics[scenario_name][label])
                writer.writerow(row)


def _format_percent(value: float | None) -> str:
    return "NA" if value is None else f"{value * 100:.1f}%"


def _format_float(value: float | None, digits: int = 2) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _plot(
    scenarios: list[Dict[str, str]],
    scenario_metrics: Dict[str, Dict[str, Dict[str, float | int | None]]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, len(scenarios), figsize=(18.0, 9.4), constrained_layout=True)
    offsets = {
        "matd3_separated_gradient": -0.08,
        "matd3_dual_q": 0.00,
        "matd3_separated_hybrid_actor": 0.08,
    }

    for col, scenario in enumerate(scenarios):
        scenario_name = scenario["name"]
        rows = scenario_metrics[scenario_name]

        top_ax = axes[0, col]
        for label in TRIO:
            style = STYLE[label]
            x = np.asarray([0.0, 1.0], dtype=np.float64) + offsets[label]
            train_mean = rows[label]["train_tail_success_mean"]
            post_mean = rows[label]["post_success_mean"]
            train_std = rows[label]["train_tail_success_std"] or 0.0
            post_std = rows[label]["post_success_std"] or 0.0
            if train_mean is None or post_mean is None:
                continue
            y = np.asarray([train_mean, post_mean], dtype=np.float64)
            yerr = np.asarray([train_std, post_std], dtype=np.float64)
            top_ax.plot(
                x,
                y,
                color=style["color"],
                marker=style["marker"],
                linewidth=2.2,
                markersize=7,
                label=style["abbr"],
                zorder=3,
            )
            top_ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="none",
                ecolor=style["color"],
                elinewidth=1.2,
                capsize=3,
                zorder=2,
            )
            top_ax.text(
                x[1] + 0.05,
                y[1],
                f"{style['abbr']}\n{post_mean * 100:.1f}%",
                fontsize=8,
                color=style["color"],
                va="center",
                ha="left",
            )
        top_ax.set_title(scenario_name, fontsize=12, fontweight="bold")
        top_ax.set_xticks([0.0, 1.0], ["Tail-100 train", "Post-eval"])
        top_ax.set_ylim(-0.02, 1.05)
        top_ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        if col == 0:
            top_ax.set_ylabel("Team success rate")

        bottom_ax = axes[1, col]
        positions = np.arange(len(TRIO), dtype=np.float64)
        heights = []
        errors = []
        colors = []
        labels = []
        for label in TRIO:
            style = STYLE[label]
            colors.append(style["color"])
            labels.append(style["abbr"])
            heights.append(rows[label]["post_goal_mean"] or 0.0)
            errors.append(rows[label]["post_goal_std"] or 0.0)
        bars = bottom_ax.bar(
            positions,
            heights,
            yerr=errors,
            color=colors,
            edgecolor="#303030",
            linewidth=0.8,
            capsize=4,
            alpha=0.92,
        )
        ymax = max((h + e for h, e in zip(heights, errors)), default=1.0)
        bottom_ax.set_ylim(0.0, max(1.0, ymax * 1.32))
        bottom_ax.set_xticks(positions, labels)
        bottom_ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        if col == 0:
            bottom_ax.set_ylabel("Post-eval final goal distance (m)")

        for bar, label in zip(bars, TRIO):
            row = rows[label]
            cf_text = _format_percent(row["post_collision_free_mean"])
            eta_text = _format_float(row["post_path_eff_mean"])
            goal_text = _format_float(row["post_goal_mean"])
            note = f"{goal_text} m\neta={eta_text}\nCF={cf_text}"
            bottom_ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + (row["post_goal_std"] or 0.0) + ymax * 0.03,
                note,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=STYLE[label]["color"],
            marker=STYLE[label]["marker"],
            linewidth=2.2,
            markersize=7,
            label=STYLE[label]["abbr"],
        )
        for label in TRIO
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(
        "MATD3 Trio Audit: Success Generalization Gap and Post-eval Goal Residual",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Bottom-panel annotations show post-eval path efficiency (eta) and collision-free rate (CF).",
        ha="center",
        fontsize=10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit MATD3 trio training/post-eval summaries.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/tang/matd3/ablation_experiments/analysis_20260423"),
        help="Directory for exported plot and CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenarios = DEFAULT_SCENARIOS
    scenario_metrics: Dict[str, Dict[str, Dict[str, float | int | None]]] = {}
    for scenario in scenarios:
        scenario_metrics[scenario["name"]] = _scenario_rows(
            Path(scenario["train_summary"]),
            Path(scenario["post_summary"]),
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "matd3_trio_metric_audit.csv"
    png_path = output_dir / "matd3_trio_generalization_gap_and_goal_residual.png"

    _write_csv(scenarios, scenario_metrics, csv_path)
    _plot(scenarios, scenario_metrics, png_path)

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote plot: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
