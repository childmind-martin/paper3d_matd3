#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle


METRIC_SPECS = [
    ("reward", "Reward", "Reward"),
    ("success", "Team Success Rate", "Success Rate"),
    ("collision", "Collision Count", "Collisions"),
    ("clearance", "Average Clearance (m)", "Clearance (m)"),
]


def _parse_linestyle(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return tuple(value)
    if not isinstance(value, str):
        return value or "-"
    value = value.strip()
    if value.startswith("(") and value.endswith(")"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (tuple, list)):
                return tuple(parsed)
        except Exception:
            pass
    return value or "-"


def _resolve_reward_detail_window(reward_curves: List[Dict[str, Any]]) -> Optional[Dict[str, tuple[float, float]]]:
    usable = []
    for item in reward_curves:
        curve = np.asarray(item.get("curve", []), dtype=np.float64)
        curve = curve[np.isfinite(curve)]
        if curve.size:
            usable.append({"curve": curve})
    if not usable:
        return None

    min_len = min(item["curve"].size for item in usable)
    if min_len < 8:
        return None

    x_start_idx = min(max(int(min_len * 0.30), 0), max(0, min_len - 2))
    tail_start_idx = min(max(int(min_len * 0.80), 0), max(0, min_len - 1))
    tail_scores = []
    for item in usable:
        tail_segment = item["curve"][tail_start_idx:min_len]
        tail_segment = tail_segment[np.isfinite(tail_segment)]
        tail_scores.append(float(np.mean(tail_segment)) if tail_segment.size else float("-inf"))
    tail_scores_arr = np.asarray(tail_scores, dtype=np.float64)
    if tail_scores_arr.size == 0 or not np.any(np.isfinite(tail_scores_arr)):
        return None

    focused = [
        item
        for _, item in sorted(
            zip(tail_scores, usable),
            key=lambda pair: pair[0],
            reverse=True,
        )[: min(5, len(usable))]
    ]
    segment = np.concatenate(
        [item["curve"][x_start_idx:min_len] for item in focused if item["curve"][x_start_idx:min_len].size]
    )
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

    return {
        "xlim": (x_start_idx + 1, min_len),
        "ylim": (y_low - pad, y_high + pad),
    }


def _add_reward_detail_inset(ax: plt.Axes, reward_curves: List[Dict[str, Any]]) -> None:
    zoom = _resolve_reward_detail_window(reward_curves)
    if not zoom:
        return

    inset_ax = ax.inset_axes([0.08, 0.12, 0.36, 0.30])
    for item in reward_curves:
        curve = np.asarray(item.get("curve", []), dtype=np.float64)
        if curve.size == 0:
            continue
        xs = np.arange(1, curve.size + 1)
        inset_ax.plot(
            xs,
            curve,
            color=item["color"],
            linewidth=1.5,
            alpha=0.95,
            linestyle=item["linestyle"],
        )
    inset_ax.set_xlim(*zoom["xlim"])
    inset_ax.set_ylim(*zoom["ylim"])
    inset_ax.grid(True, alpha=0.22, linestyle="--")
    inset_ax.tick_params(axis="both", labelsize=7, pad=1)

    x0, x1 = zoom["xlim"]
    y0, y1 = zoom["ylim"]
    rect = Rectangle(
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
    ax.add_patch(rect)

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


def _load_curve_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def plot_curve_data_vector(curve_json_path: Path, output_base: Path, title: str) -> None:
    data = _load_curve_json(curve_json_path)
    labels = data["labels"]
    abbr_by_label = dict(zip(data["labels"], data["abbrs"]))
    style_by_label = {item["label"]: item for item in data["styles"]}

    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.5), sharex=False)
    axes = axes.flatten()
    reward_curves = []

    line_width = 1.8 if len(labels) > 8 else 2.1
    band_alpha = 0.08 if len(labels) > 8 else 0.11

    for ax, (metric_key, title_text, ylabel) in zip(axes, METRIC_SPECS):
        metric = data["metrics"][metric_key]
        xs = np.asarray(metric["x"], dtype=np.float64)
        for label in labels:
            style = style_by_label[label]
            mean_curve = np.asarray(metric["mean"][label], dtype=np.float64)
            std_curve = np.asarray(metric["std"][label], dtype=np.float64)
            color = style["color"]
            linestyle = _parse_linestyle(style.get("linestyle", "-"))
            ax.plot(
                xs,
                mean_curve,
                color=color,
                linewidth=line_width,
                alpha=0.95,
                linestyle=linestyle,
                label=abbr_by_label[label],
            )
            ax.fill_between(xs, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=band_alpha)
            if metric_key == "reward":
                reward_curves.append(
                    {
                        "curve": mean_curve,
                        "color": color,
                        "linestyle": linestyle,
                    }
                )
        ax.set_title(title_text, fontsize=12.5, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.28, linestyle="--")
        if metric_key == "success":
            ax.set_ylim([-0.10, 0.80])
        if metric_key == "collision":
            ax.set_ylim(bottom=-2000)
        if metric_key == "clearance":
            ax.set_ylim([-40, 120])

    _add_reward_detail_inset(axes[0], reward_curves)

    handles, labels_text = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_text, loc="center right", bbox_to_anchor=(0.985, 0.50), frameon=True, fontsize=9)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0.02, 0.03, 0.88, 0.96])

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export vector ablation plots from multi-seed curve-data JSON.")
    parser.add_argument("curve_json", type=Path, help="Path to multi_seed_curve_data_*.json")
    parser.add_argument("output_base", type=Path, help="Output path without extension")
    parser.add_argument(
        "--title",
        default="Extended Multi-seed Mean Ablation Comparison",
        help="Figure title",
    )
    args = parser.parse_args()
    plot_curve_data_vector(args.curve_json, args.output_base, args.title)


if __name__ == "__main__":
    main()
