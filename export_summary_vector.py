#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


STYLE_MAP: dict[str, dict[str, Any]] = {
    "matd3_separated_gradient": {"abbr": "Sep.", "color": "#1f77b4", "linestyle": "-", "marker": "o", "hatch": ""},
    "matd3_dual_q": {"abbr": "Uni.", "color": "#d62728", "linestyle": "--", "marker": "s", "hatch": "//"},
    "matd3_separated_hybrid_actor": {"abbr": "Hyb.", "color": "#ff7f0e", "linestyle": "-.", "marker": "D", "hatch": "xx"},
    "matd3_separated_hybrid_actor_alpha20": {"abbr": "Hyb-0.2", "color": "#bcbd22", "linestyle": "--", "marker": "P", "hatch": "++"},
    "matd3_single_q": {"abbr": "Base.", "color": "#2ca02c", "linestyle": ":", "marker": "^", "hatch": ".."},
    "maddpg_separated_gradient": {"abbr": "DPG-Sep", "color": "#9467bd", "linestyle": "-", "marker": "v", "hatch": ""},
    "maddpg_dual_q": {"abbr": "DPG-Uni", "color": "#17becf", "linestyle": "--", "marker": ">", "hatch": "\\\\"},
    "maddpg_baseline": {"abbr": "DPG-Base", "color": "#7f7f7f", "linestyle": "-.", "marker": "<", "hatch": "--"},
    "mappo_baseline": {"abbr": "PPO-Std", "color": "#e377c2", "linestyle": ":", "marker": "h", "hatch": "oo"},
    "mappo_fusion_only": {"abbr": "PPO-Fus", "color": "#6a51a3", "linestyle": "--", "marker": "p", "hatch": "OO"},
    "mappo_separated_gradient": {"abbr": "PPO-Sep", "color": "#008b8b", "linestyle": "-", "marker": "X", "hatch": "**"},
}

MULTI_FAMILY_ABBR: dict[str, str] = {
    "matd3_separated_gradient": "M3-Sep",
    "matd3_dual_q": "M3-Uni",
    "matd3_separated_hybrid_actor": "M3-Hyb",
    "matd3_separated_hybrid_actor_alpha20": "M3-Hyb-0.2",
    "matd3_single_q": "M3-Base",
    "maddpg_separated_gradient": "DPG-Sep",
    "maddpg_dual_q": "DPG-Uni",
    "maddpg_baseline": "DPG-Base",
    "mappo_baseline": "PPO-Std",
    "mappo_fusion_only": "PPO-Fus",
    "mappo_separated_gradient": "PPO-Sep",
}


TRAIN_METRICS = [
    ("reward", "reward_mean_curve", "reward_std_curve", "Reward", "Reward"),
    ("success", "success_mean_curve", "success_std_curve", "Team Success Rate", "Success Rate"),
    ("collision", "collision_mean_curve", "collision_std_curve", "Collision Count", "Collisions"),
    ("clearance", "clearance_mean_curve", "clearance_std_curve", "Average Clearance (m)", "Clearance (m)"),
]

POST_EVAL_METRICS = [
    ("team_success_rate", "Team Success Rate", "Success Rate"),
    ("avg_reward", "Average Reward", "Reward"),
    ("avg_collision_count", "Average Collisions", "Collisions"),
    ("avg_min_clearance_min", "Average Min Clearance (m)", "Clearance (m)"),
    ("avg_team_total_path_length", "Average Team Path Length (m)", "Path Length (m)"),
    ("avg_arrival_step_success_only", "Success Arrival Step", "Step"),
]


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _item_by_label(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["label"]: item for item in summary["aggregated_experiments"]}


def _style(label: str) -> dict[str, Any]:
    return STYLE_MAP.get(
        label,
        {"abbr": label, "color": "#444444", "linestyle": "-", "marker": "o", "hatch": ""},
    )


def _abbr(label: str, *, multi_family: bool) -> str:
    if multi_family:
        return MULTI_FAMILY_ABBR.get(label, _style(label)["abbr"])
    return _style(label)["abbr"]


def _resolve_reward_detail_window(reward_curves: list[np.ndarray]) -> dict[str, tuple[float, float]] | None:
    usable = []
    for curve in reward_curves:
        curve = curve[np.isfinite(curve)]
        if curve.size:
            usable.append(curve)
    if not usable:
        return None

    min_len = min(curve.size for curve in usable)
    if min_len < 8:
        return None

    x_start_idx = min(max(int(min_len * 0.30), 0), max(0, min_len - 2))
    tail_start_idx = min(max(int(min_len * 0.80), 0), max(0, min_len - 1))
    tail_scores = []
    for curve in usable:
        tail = curve[tail_start_idx:min_len]
        tail = tail[np.isfinite(tail)]
        tail_scores.append(float(np.mean(tail)) if tail.size else float("-inf"))
    focused = [
        curve
        for _, curve in sorted(zip(tail_scores, usable), key=lambda pair: pair[0], reverse=True)[: min(5, len(usable))]
    ]
    segment = np.concatenate([curve[x_start_idx:min_len] for curve in focused if curve[x_start_idx:min_len].size])
    segment = segment[np.isfinite(segment)]
    if segment.size == 0:
        return None

    y_low = float(np.quantile(segment, 0.02))
    y_high = float(np.quantile(segment, 0.98))
    if not np.isfinite(y_low) or not np.isfinite(y_high):
        return None
    if y_high <= y_low:
        y_low = float(np.min(segment))
        y_high = float(np.max(segment))
    span = max(y_high - y_low, 1.0)
    pad = span * 0.12
    return {"xlim": (x_start_idx + 1, min_len), "ylim": (y_low - pad, y_high + pad)}


def _add_reward_curve_inset(ax: plt.Axes, reward_specs: list[dict[str, Any]]) -> None:
    curves = [spec["curve"] for spec in reward_specs]
    zoom = _resolve_reward_detail_window(curves)
    if not zoom:
        return

    inset_ax = ax.inset_axes([0.08, 0.12, 0.36, 0.30])
    for spec in reward_specs:
        curve = spec["curve"]
        xs = np.arange(1, curve.size + 1)
        inset_ax.plot(xs, curve, color=spec["color"], linewidth=1.5, alpha=0.95, linestyle=spec["linestyle"])
    inset_ax.set_xlim(*zoom["xlim"])
    inset_ax.set_ylim(*zoom["ylim"])
    inset_ax.grid(True, alpha=0.22, linestyle="--")
    inset_ax.tick_params(axis="both", labelsize=7, pad=1)

    x0, x1 = zoom["xlim"]
    y0, y1 = zoom["ylim"]
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=1.15,
            edgecolor="#444444",
            facecolor="none",
            linestyle="--",
            alpha=0.95,
            zorder=6,
        )
    )

    target_x = x0 + 0.62 * (x1 - x0)
    target_y_top = y0 + 0.80 * (y1 - y0)
    target_y_bottom = y0 + 0.20 * (y1 - y0)
    ax.annotate(
        "",
        xy=(target_x, target_y_top),
        xycoords="data",
        xytext=(1.0, 0.88),
        textcoords=inset_ax.transAxes,
        arrowprops=dict(arrowstyle="->", color="#444444", lw=1.2),
        zorder=7,
    )
    ax.annotate(
        "",
        xy=(target_x, target_y_bottom),
        xycoords="data",
        xytext=(1.0, 0.12),
        textcoords=inset_ax.transAxes,
        arrowprops=dict(arrowstyle="->", color="#444444", lw=1.2),
        zorder=7,
    )


def _resolve_reward_bar_window(values: np.ndarray, errors: np.ndarray) -> dict[str, tuple[float, float]] | None:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return None
    y_low = float(np.quantile(finite, 0.05))
    y_high = float(np.quantile(finite, 0.95))
    if y_high <= y_low:
        return None
    span = y_high - y_low
    pad = max(span * 0.20, 1.0)
    lower_err = values - errors
    upper_err = values + errors
    finite_low = lower_err[np.isfinite(lower_err)]
    finite_high = upper_err[np.isfinite(upper_err)]
    if finite_low.size:
        y_low = min(y_low, float(np.min(finite_low)))
    if finite_high.size:
        y_high = max(y_high, float(np.max(finite_high)))
    return {"xlim": (-0.5, len(values) - 0.5), "ylim": (y_low - pad, y_high + pad)}


def _add_reward_bar_inset(
    ax: plt.Axes,
    positions: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    colors: list[str],
    hatches: list[str],
    labels: list[str],
) -> None:
    zoom = _resolve_reward_bar_window(means, stds)
    if not zoom:
        return

    inset_ax = ax.inset_axes([0.08, 0.12, 0.40, 0.30])
    bars = inset_ax.bar(positions, means, color=colors, edgecolor="#444444", linewidth=0.8)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    inset_ax.errorbar(positions, means, yerr=stds, fmt="none", ecolor="#222222", elinewidth=1.0, capsize=3)
    inset_ax.set_xlim(*zoom["xlim"])
    inset_ax.set_ylim(*zoom["ylim"])
    inset_ax.set_xticks(positions)
    inset_ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=7)
    inset_ax.grid(True, axis="y", alpha=0.22, linestyle="--")
    inset_ax.tick_params(axis="y", labelsize=7, pad=1)

    x0, x1 = zoom["xlim"]
    y0, y1 = zoom["ylim"]
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=1.10,
            edgecolor="#444444",
            facecolor="none",
            linestyle="--",
            alpha=0.90,
            zorder=6,
        )
    )
    ax.annotate(
        "",
        xy=(x1 - 0.4, y0 + 0.78 * (y1 - y0)),
        xycoords="data",
        xytext=(1.0, 0.84),
        textcoords=inset_ax.transAxes,
        arrowprops=dict(arrowstyle="->", color="#444444", lw=1.2),
        zorder=7,
    )
    ax.annotate(
        "",
        xy=(x1 - 0.4, y0 + 0.22 * (y1 - y0)),
        xycoords="data",
        xytext=(1.0, 0.16),
        textcoords=inset_ax.transAxes,
        arrowprops=dict(arrowstyle="->", color="#444444", lw=1.2),
        zorder=7,
    )


def _resolve_horizontal_limits(metric_key: str, means: np.ndarray, stds: np.ndarray) -> tuple[float, float]:
    finite_means = means[np.isfinite(means)]
    finite_stds = stds[np.isfinite(stds)]
    if finite_means.size == 0:
        return (0.0, 1.0)

    if metric_key == "team_success_rate":
        upper = float(np.nanmax(means + stds))
        upper = max(0.10, upper)
        upper = min(1.0, upper * 1.55 + 0.025)
        return (0.0, upper)

    lower = float(np.nanmin(means - stds))
    upper = float(np.nanmax(means + stds))
    span = max(upper - lower, 1.0)

    if upper <= 0:
        right_pad = span * 0.08
        left_pad = span * 0.24
        return (lower - left_pad, upper + right_pad)

    if lower >= 0:
        left_pad = span * 0.06
        right_pad = span * 0.30
        return (max(0.0, lower - left_pad), upper + right_pad)

    left_pad = span * 0.16
    right_pad = span * 0.22
    return (lower - left_pad, upper + right_pad)


def _format_metric_label(metric_key: str, mean: float) -> str:
    if metric_key == "team_success_rate":
        return f"{mean * 100:.1f}%"
    if metric_key == "avg_reward":
        return f"{mean / 1000:.1f}k" if abs(mean) >= 1000 else f"{mean:.0f}"
    return f"{mean:.2f}" if abs(mean) < 100 else f"{mean:.0f}"


def _resolve_horizontal_label_position(
    mean: float,
    std: float,
    x_limits: tuple[float, float],
    value_offset: float,
) -> tuple[float, str]:
    x_min, x_max = x_limits
    axis_span = max(x_max - x_min, 1e-9)
    edge_padding = axis_span * 0.04
    error_extent = max(float(std), 0.0) if np.isfinite(std) else 0.0
    error_left = mean - error_extent
    error_right = mean + error_extent

    if mean >= 0:
        candidate = max(mean, error_right) + value_offset
        if candidate <= x_max - edge_padding:
            return candidate, "left"
        fallback = min(mean, error_left) - value_offset
        return max(fallback, x_min + edge_padding), "right"

    candidate = min(mean, error_left) - value_offset
    if candidate >= x_min + edge_padding:
        return candidate, "right"
    fallback = max(mean, error_right) + value_offset
    return min(fallback, x_max - edge_padding), "left"


def export_training_figure(summary_path: Path, output_base: Path, title: str, labels: list[str]) -> None:
    summary = _load_summary(summary_path)
    items = _item_by_label(summary)
    selected = [items[label] for label in labels if label in items]
    if not selected:
        raise ValueError("No requested labels found in summary.")
    multi_family = len(selected) > 4

    if len(selected) <= 3:
        fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.9), sharex=False)
        panels = TRAIN_METRICS[:2]
        legend_anchor = (0.985, 0.92)
        rect = [0.02, 0.03, 0.88, 0.92]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.5), sharex=False)
        axes = axes.flatten()
        panels = TRAIN_METRICS
        legend_anchor = (0.985, 0.50)
        rect = [0.02, 0.03, 0.88, 0.96]

    axes = np.asarray(axes).reshape(-1)
    reward_specs: list[dict[str, Any]] = []
    line_width = 1.8 if len(selected) > 8 else 2.1
    band_alpha = 0.08 if len(selected) > 8 else 0.11

    for ax, (_metric_name, mean_key, std_key, title_text, ylabel) in zip(axes, panels):
        for item in selected:
            label = item["label"]
            style = _style(label)
            mean_curve = np.asarray(item.get(mean_key, []), dtype=np.float64)
            std_curve = np.asarray(item.get(std_key, []), dtype=np.float64)
            xs = np.arange(1, mean_curve.size + 1)
            ax.plot(
                xs,
                mean_curve,
                color=style["color"],
                linewidth=line_width,
                alpha=0.95,
                linestyle=style["linestyle"],
                label=_abbr(label, multi_family=multi_family),
            )
            if std_curve.size:
                ax.fill_between(xs, mean_curve - std_curve, mean_curve + std_curve, color=style["color"], alpha=band_alpha)
            if mean_key == "reward_mean_curve":
                reward_specs.append({"curve": mean_curve, "color": style["color"], "linestyle": style["linestyle"]})
        ax.set_title(title_text, fontsize=12.5, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.28, linestyle="--")
        if mean_key == "success_mean_curve":
            ax.set_ylim([-0.10, 0.80 if len(selected) > 3 else 1.05])
        if mean_key == "collision_mean_curve":
            ax.set_ylim(bottom=-100 if len(selected) <= 4 else -1000)
        if mean_key == "clearance_mean_curve":
            ax.set_ylim(bottom=0)

    _add_reward_curve_inset(axes[0], reward_specs)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="center right", bbox_to_anchor=legend_anchor, frameon=True, fontsize=9)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.985)
    fig.tight_layout(rect=rect)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def export_posteval_figure(summary_path: Path, output_base: Path, title: str, labels: list[str]) -> None:
    summary = _load_summary(summary_path)
    items = _item_by_label(summary)
    selected = [items[label] for label in labels if label in items]
    if not selected:
        raise ValueError("No requested labels found in summary.")
    multi_family = len(selected) > 4

    if len(selected) <= 3:
        fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.9), sharex=False)
        panels = POST_EVAL_METRICS[:2]
        legend_anchor = (0.985, 0.92)
        rect = [0.02, 0.03, 0.88, 0.92]
    else:
        fig, axes = plt.subplots(2, 3, figsize=(17.4, 10.8), sharex=False)
        axes = axes.flatten()
        panels = POST_EVAL_METRICS
        legend_anchor = None
        rect = [0.075, 0.045, 0.985, 0.955]

    axes = np.asarray(axes).reshape(-1)
    positions = np.arange(len(selected))
    reward_means = reward_stds = None
    reward_colors: list[str] = []
    reward_hatches: list[str] = []
    reward_labels: list[str] = []

    for ax, (metric_key, title_text, ylabel) in zip(axes, panels):
        means = []
        stds = []
        colors = []
        hatches = []
        ticklabels = []
        for item in selected:
            style = _style(item["label"])
            stats = item.get("post_eval_metric_stats", {}).get(metric_key, {})
            means.append(stats.get("mean", np.nan))
            stds.append(stats.get("std", 0.0))
            colors.append(style["color"])
            hatches.append(style["hatch"])
            ticklabels.append(_abbr(item["label"], multi_family=multi_family))
        means_arr = np.asarray(means, dtype=np.float64)
        stds_arr = np.asarray(stds, dtype=np.float64)
        if len(selected) > 4:
            bars = ax.barh(positions, means_arr, color=colors, edgecolor="#444444", linewidth=0.8)
            for bar, hatch in zip(bars, hatches):
                bar.set_hatch(hatch)
            ax.errorbar(means_arr, positions, xerr=stds_arr, fmt="none", ecolor="#222222", elinewidth=1.1, capsize=3)
            ax.set_title(title_text, fontsize=12.0, fontweight="bold")
            ax.set_xlabel(ylabel)
            ax.set_yticks(positions)
            ax.set_yticklabels(ticklabels, fontsize=9)
            ax.invert_yaxis()
            ax.grid(True, axis="x", alpha=0.28, linestyle="--")
            ax.set_xlim(*_resolve_horizontal_limits(metric_key, means_arr, stds_arr))
        else:
            bars = ax.bar(positions, means_arr, color=colors, edgecolor="#444444", linewidth=0.8)
            for bar, hatch in zip(bars, hatches):
                bar.set_hatch(hatch)
            ax.errorbar(positions, means_arr, yerr=stds_arr, fmt="none", ecolor="#222222", elinewidth=1.1, capsize=4)
            ax.set_title(title_text, fontsize=12.0, fontweight="bold")
            ax.set_ylabel(ylabel)
            ax.set_xticks(positions)
            ax.set_xticklabels(ticklabels, rotation=28, ha="right", fontsize=9)
            ax.grid(True, axis="y", alpha=0.28, linestyle="--")
            if metric_key == "team_success_rate":
                ax.set_ylim(0, 1.0)
        if metric_key == "avg_reward":
            reward_means = means_arr
            reward_stds = stds_arr
            reward_colors = colors
            reward_hatches = hatches
            reward_labels = ticklabels
        finite_means = means_arr[np.isfinite(means_arr)]
        value_span = float(np.ptp(finite_means)) if finite_means.size > 1 else max(1.0, float(np.nanmax(np.abs(means_arr))))
        for xpos, mean, std in zip(positions, means_arr, stds_arr):
            if np.isfinite(mean):
                label_text = _format_metric_label(metric_key, mean)
                if len(selected) > 4:
                    x_limits = ax.get_xlim()
                    axis_span = x_limits[1] - x_limits[0]
                    value_offset = max(
                        value_span * 0.025,
                        axis_span * 0.015,
                        0.018 if metric_key == "team_success_rate" else 1.5,
                    )
                    text_x, ha = _resolve_horizontal_label_position(mean, std, x_limits, value_offset)
                    ax.text(text_x, xpos, label_text, ha=ha, va="center", fontsize=8.5, clip_on=False)
                else:
                    offset = (
                        0.03 * max(1.0, np.nanmax(np.abs(means_arr)))
                        if metric_key == "avg_reward"
                        else 0.02 * max(1.0, np.nanmax(means_arr + stds_arr))
                    )
                    va = "bottom" if mean >= 0 else "top"
                    text_y = mean + offset if mean >= 0 else mean - offset
                    ax.text(xpos, text_y, label_text, ha="center", va=va, fontsize=8)

    if reward_means is not None and reward_stds is not None and len(selected) <= 4:
        _add_reward_bar_inset(axes[1], positions, reward_means, reward_stds, reward_colors, reward_hatches, reward_labels)

    if legend_anchor is not None:
        legend_handles = []
        legend_labels = []
        for item in selected:
            style = _style(item["label"])
            legend_handles.append(Line2D([0], [0], color=style["color"], marker=style["marker"], linestyle="", markersize=8))
            legend_labels.append(_abbr(item["label"], multi_family=multi_family))
        fig.legend(legend_handles, legend_labels, loc="center right", bbox_to_anchor=legend_anchor, frameon=True, fontsize=9)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.985)
    fig.tight_layout(rect=rect)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export vector training or post-eval figures from summary JSON.")
    parser.add_argument("summary_json", type=Path, help="Path to summary_*.json or latest_summary.json")
    parser.add_argument("output_base", type=Path, help="Output path without extension")
    parser.add_argument("--mode", choices=["train", "posteval"], required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Summary labels to include, e.g. matd3_separated_gradient matd3_dual_q matd3_separated_hybrid_actor",
    )
    args = parser.parse_args()

    if args.mode == "train":
        export_training_figure(args.summary_json, args.output_base, args.title, args.labels)
    else:
        export_posteval_figure(args.summary_json, args.output_base, args.title, args.labels)


if __name__ == "__main__":
    main()
