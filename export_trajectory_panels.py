#!/usr/bin/env python3
"""Export selected Plotly trajectory HTML files to paper-ready PDF/SVG panels.

This script reads existing Plotly HTML exports, extracts the serialized figure
data/layout, applies paper-oriented cleanup, and writes vector outputs that can
be inserted into the LaTeX manuscript.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable


ROOT = Path("/home/tang")
PAPER_DIR = ROOT / "paper"


@dataclass(frozen=True)
class PanelSpec:
    key: str
    title: str
    html_path: Path


@dataclass(frozen=True)
class ObstacleDiskSpec:
    cx: float
    cy: float
    radius: float


PANEL_SPECS: list[PanelSpec] = [
    PanelSpec(
        key="traj_l2_sep_seed202",
        title="Level 2 / M3-Sep",
        html_path=ROOT
        / "matd3/logs/matd3_separated_gradient__seed202__batch_groupB_seed202_20260331_220752_20260401_181851/20260401_181853/best_trajectory_ep107_interactive.html",
    ),
    PanelSpec(
        key="traj_l2_uni_seed202",
        title="Level 2 / M3-Uni",
        html_path=ROOT
        / "matd3/logs/matd3_dual_q__seed202__batch_groupB_seed202_20260331_220752_20260401_220552/20260401_220554/best_trajectory_ep192_interactive.html",
    ),
    PanelSpec(
        key="traj_l2_hyb_seed202",
        title="Level 2 / M3-Hyb",
        html_path=ROOT
        / "matd3/logs/matd3_separated_hybrid_actor__seed202__batch_groupB_seed202_20260331_220752_20260406_230620/20260406_230623/best_trajectory_ep146_interactive.html",
    ),
    PanelSpec(
        key="traj_l3_sep_seed202",
        title="Level 3 / M3-Sep",
        html_path=ROOT
        / "matd3/logs/matd3_separated_gradient__seed202__batch_groupB_seed202_20260406_230829_20260406_230909/20260406_230915/best_trajectory_ep162_interactive.html",
    ),
    PanelSpec(
        key="traj_l3_uni_seed202",
        title="Level 3 / M3-Uni",
        html_path=ROOT
        / "matd3/logs/matd3_dual_q__seed202__batch_groupB_seed202_20260406_230829_20260407_113359/20260407_113402/best_trajectory_ep919_interactive.html",
    ),
    PanelSpec(
        key="traj_l3_hyb_seed202",
        title="Level 3 / M3-Hyb",
        html_path=ROOT
        / "matd3/logs/matd3_separated_hybrid_actor__seed202__batch_groupB_seed202_20260406_230829_20260408_113923/20260408_113925/best_trajectory_ep579_interactive.html",
    ),
    PanelSpec(
        key="traj_l2_test_sep_seed202",
        title="Level 2 Test / M3-Sep",
        html_path=ROOT
        / "matd3/ablation_experiments/multi_seed_groupB_20260331_220752/seed_batches/batch_groupB_seed202_20260331_220752/results/post_eval/matd3_separated_gradient/best_by_team_sr/team_success_best_interactive.html",
    ),
    PanelSpec(
        key="traj_l2_test_sep_ep4_seed202",
        title="Level 2 Test E4 / M3-Sep",
        html_path=ROOT
        / "matd3/ablation_experiments/multi_seed_groupB_20260331_220752/seed_batches/batch_groupB_seed202_20260331_220752/results/post_eval/matd3_separated_gradient/best_by_team_sr/trajectory_ep4_interactive.html",
    ),
    PanelSpec(
        key="traj_l2_test_uni_ep4_seed202",
        title="Level 2 Test E4 / M3-Uni",
        html_path=ROOT
        / "matd3/ablation_experiments/multi_seed_groupB_20260331_220752/seed_batches/batch_groupB_seed202_20260331_220752/results/post_eval/matd3_dual_q/best_by_team_sr/trajectory_ep4_interactive.html",
    ),
    PanelSpec(
        key="traj_l2_test_success_uni_seed936487",
        title="Level 2 Successful Test / Unified",
        html_path=ROOT
        / "matd3/ablation_experiments/multi_seed_groupB_20260331_220752_testset2_20260409/seed_batches/batch_groupB_seed936487_20260331_220752/results/post_eval/matd3_dual_q/matched_validation/team_success_best_interactive.html",
    ),
    PanelSpec(
        key="traj_l2_test_success_uni_seed202_latest",
        title="Level 2 Successful Test / Unified",
        html_path=ROOT
        / "matd3/ablation_experiments/multi_seed_groupB_20260331_220752_testset2_20260409/seed_batches/batch_groupB_seed202_20260331_220752/results/post_eval/matd3_dual_q/matched_validation/team_success_best_interactive.html",
    ),
    PanelSpec(
        key="traj_l3_test_hyb_seed202_ep7",
        title="Level 3 Test / M3-Hyb",
        html_path=ROOT
        / "matd3/ablation_experiments/multi_seed_groupB_20260406_230829/seed_batches/batch_groupB_seed202_20260406_230829/results/post_eval/matd3_separated_hybrid_actor/best_by_team_sr/trajectory_ep11_interactive.html",
    ),
    PanelSpec(
        key="traj_l3_test_uni_ep18_seed202",
        title="Level 3 Test E18 / M3-Uni",
        html_path=ROOT
        / "matd3/ablation_experiments/multi_seed_groupB_20260406_230829/seed_batches/batch_groupB_seed202_20260406_230829/results/post_eval/matd3_dual_q/best_by_team_sr/trajectory_ep18_interactive.html",
    ),
    PanelSpec(
        key="traj_l3_test_hyb_ep18_seed202",
        title="Level 3 Test E18 / M3-Hyb",
        html_path=ROOT
        / "matd3/ablation_experiments/multi_seed_groupB_20260406_230829/seed_batches/batch_groupB_seed202_20260406_230829/results/post_eval/matd3_separated_hybrid_actor/best_by_team_sr/trajectory_ep18_interactive.html",
    ),
    PanelSpec(
        key="traj_l3_test_success_sep_seed202",
        title="Level 3 Successful Test / Separated",
        html_path=ROOT
        / "matd3/ablation_experiments/multi_seed_groupB_20260406_230829/seed_batches/batch_groupB_seed202_20260406_230829/results/post_eval/matd3_separated_gradient/best_by_team_sr/team_success_best_interactive.html",
    ),
    PanelSpec(
        key="traj_l3_test_success_hyb_seed202",
        title="Level 3 Successful Test / Hybrid",
        html_path=ROOT
        / "matd3/ablation_experiments/multi_seed_groupB_20260406_230829/seed_batches/batch_groupB_seed202_20260406_230829/results/post_eval/matd3_separated_hybrid_actor/matched_validation/team_success_best_interactive.html",
    ),
    PanelSpec(
        key="traj_l3_test_success_sep_seed101",
        title="Level 3 Successful Test / Separated",
        html_path=ROOT
        / "matd3/ablation_experiments/multi_seed_groupB_20260406_230829/seed_batches/batch_groupB_seed101_20260406_230829/results/post_eval/matd3_separated_gradient/matched_validation/team_success_best_interactive.html",
    ),
    PanelSpec(
        key="traj_l3_test_success_uni_seed101",
        title="Level 3 Successful Test / Unified",
        html_path=ROOT
        / "matd3/ablation_experiments/multi_seed_groupB_20260406_230829/seed_batches/batch_groupB_seed101_20260406_230829/results/post_eval/matd3_dual_q/matched_validation/team_success_best_interactive.html",
    ),
    PanelSpec(
        key="traj_l3_test_success_hyb_seed101",
        title="Level 3 Successful Test / Hybrid",
        html_path=ROOT
        / "matd3/ablation_experiments/multi_seed_groupB_20260406_230829/seed_batches/batch_groupB_seed101_20260406_230829/results/post_eval/matd3_separated_hybrid_actor/matched_validation/team_success_best_interactive.html",
    ),
]


def _extract_plotly_payload(html_text: str) -> tuple[Any, Any, Any]:
    marker = "Plotly.newPlot("
    start = html_text.rfind(marker)
    if start < 0:
        raise ValueError("Plotly.newPlot(...) not found in HTML.")

    decoder = json.JSONDecoder()
    cursor = start + len(marker)

    # Skip the first argument (div id string).
    first_quote = html_text.find('"', cursor)
    if first_quote < 0:
        raise ValueError("Could not locate Plotly div id.")
    second_quote = html_text.find('"', first_quote + 1)
    if second_quote < 0:
        raise ValueError("Could not parse Plotly div id.")
    cursor = second_quote + 1

    # Skip comma/whitespace before data payload.
    while cursor < len(html_text) and html_text[cursor] in " \t\r\n,":
        cursor += 1
    data, consumed = decoder.raw_decode(html_text[cursor:])
    cursor += consumed

    while cursor < len(html_text) and html_text[cursor] in " \t\r\n,":
        cursor += 1
    layout, consumed = decoder.raw_decode(html_text[cursor:])
    cursor += consumed

    while cursor < len(html_text) and html_text[cursor] in " \t\r\n,":
        cursor += 1
    config, _ = decoder.raw_decode(html_text[cursor:])
    return data, layout, config


def _make_paper_ready_figure(spec: PanelSpec) -> go.Figure:
    html_text = spec.html_path.read_text(encoding="utf-8", errors="ignore")
    data, layout, _config = _extract_plotly_payload(html_text)
    fig = go.Figure(data=data, layout=layout)

    for trace in fig.data:
        trace_type = getattr(trace, "type", "")

        if trace_type == "surface":
            trace.showscale = False
            trace.hoverinfo = "skip"

        if trace_type == "scatter3d":
            name = str(getattr(trace, "name", ""))
            trace.showlegend = False

            if "起点" in name:
                trace.mode = "markers"
                trace.text = None
                if getattr(trace, "marker", None):
                    trace.marker.size = 8
            elif "终点" in name:
                if getattr(trace, "marker", None):
                    trace.marker.size = 8
            elif "中央目标" in name or "目标" in name:
                if getattr(trace, "marker", None):
                    trace.marker.size = 8
            elif "智能体 " in name:
                if getattr(trace, "line", None):
                    trace.line.width = 9

    scene = fig.layout.scene.to_plotly_json() if fig.layout.scene else {}
    scene.update(
        dict(
            bgcolor="rgb(255,255,255)",
            xaxis=dict(
                scene.get("xaxis", {}),
                title=dict(text="X", font=dict(size=24)),
                tickfont=dict(size=15),
            ),
            yaxis=dict(
                scene.get("yaxis", {}),
                title=dict(text="Y", font=dict(size=24)),
                tickfont=dict(size=15),
            ),
            zaxis=dict(
                scene.get("zaxis", {}),
                title=dict(text="Z", font=dict(size=24)),
                tickfont=dict(size=15),
            ),
            camera=dict(eye=dict(x=1.38, y=1.36, z=0.96), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1)),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=0.48),
        )
    )

    fig.update_layout(
        title=dict(text="", x=0.5, font=dict(size=18, color="rgb(0, 0, 0)")),
        scene=scene,
        width=1040,
        height=760,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
    )

    return fig


def _as_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _scatter_color(trace: Any, fallback: str = "rgb(30,30,30)") -> str:
    line = getattr(trace, "line", None)
    if line is not None:
        color = getattr(line, "color", None)
        if color:
            return str(color)
    marker = getattr(trace, "marker", None)
    if marker is not None:
        color = getattr(marker, "color", None)
        if color:
            return str(color)
    return fallback


def _finite_xy(trace: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if not hasattr(trace, "x") or not hasattr(trace, "y"):
        return None
    x = _as_array(trace.x).reshape(-1)
    y = _as_array(trace.y).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return None
    return x[finite], y[finite]


def _make_obstacle_disk(trace: Any) -> ObstacleDiskSpec | None:
    finite_xy = _finite_xy(trace)
    if finite_xy is None:
        return None

    x, y = finite_xy
    cx = float((np.nanmin(x) + np.nanmax(x)) * 0.5)
    cy = float((np.nanmin(y) + np.nanmax(y)) * 0.5)
    radius = float(max(np.nanmax(x) - np.nanmin(x), np.nanmax(y) - np.nanmin(y)) * 0.5)
    if not np.isfinite(radius) or radius <= 0:
        return None

    return ObstacleDiskSpec(cx=cx, cy=cy, radius=radius)


def _resolve_axis_range(values: np.ndarray, *, pad_ratio: float = 0.0, min_pad: float = 0.0) -> list[float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return [-10.0, 215.0]
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    span = max(upper - lower, 1.0)
    pad = max(span * pad_ratio, min_pad)
    return [lower - pad, upper + pad]


def _mpl_color(color: Any, *, alpha: float | None = None) -> tuple[float, float, float, float]:
    rgba: tuple[float, float, float, float]
    if isinstance(color, str):
        stripped = color.strip()
        if stripped.startswith("rgba(") and stripped.endswith(")"):
            parts = [part.strip() for part in stripped[5:-1].split(",")]
            rgba = (
                float(parts[0]) / 255.0,
                float(parts[1]) / 255.0,
                float(parts[2]) / 255.0,
                float(parts[3]),
            )
        elif stripped.startswith("rgb(") and stripped.endswith(")"):
            parts = [part.strip() for part in stripped[4:-1].split(",")]
            rgba = (
                float(parts[0]) / 255.0,
                float(parts[1]) / 255.0,
                float(parts[2]) / 255.0,
                1.0,
            )
        else:
            rgba = mcolors.to_rgba(stripped)
    else:
        rgba = mcolors.to_rgba(color)

    if alpha is not None:
        return (rgba[0], rgba[1], rgba[2], alpha)
    return rgba


def _colorscale_to_cmap(colorscale: Any) -> mcolors.Colormap:
    default_scale = [
        [0.0, "rgb(220, 220, 180)"],
        [0.3, "rgb(180, 200, 120)"],
        [0.5, "rgb(120, 160, 100)"],
        [0.7, "rgb(100, 120, 80)"],
        [0.85, "rgb(139, 137, 137)"],
        [1.0, "rgb(255, 255, 255)"],
    ]
    source_scale = colorscale or default_scale
    cmap_points: list[tuple[float, tuple[float, float, float, float]]] = []
    for item in source_scale:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        cmap_points.append((float(item[0]), _mpl_color(item[1])))
    if not cmap_points:
        cmap_points = [(point[0], _mpl_color(point[1])) for point in default_scale]
    cmap_points.sort(key=lambda item: item[0])
    return mcolors.LinearSegmentedColormap.from_list("trajectory_top_view", cmap_points)


def _make_top_view_figure(spec: PanelSpec) -> plt.Figure:
    html_text = spec.html_path.read_text(encoding="utf-8", errors="ignore")
    data, layout, _config = _extract_plotly_payload(html_text)
    source = go.Figure(data=data, layout=layout)

    terrain = None
    terrain_x_axis: np.ndarray | None = None
    terrain_y_axis: np.ndarray | None = None
    obstacle_disks: list[ObstacleDiskSpec] = []
    trajectory_traces: list[tuple[np.ndarray, np.ndarray, str]] = []
    start_traces: list[tuple[np.ndarray, np.ndarray, str]] = []
    goal_by_name: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}

    for trace in source.data:
        trace_type = getattr(trace, "type", "")
        name = str(getattr(trace, "name", ""))

        if trace_type == "surface" and name == "地形" and terrain is None:
            x = _as_array(trace.x)
            y = _as_array(trace.y)
            z = _as_array(trace.z)
            terrain = (x, y, z, getattr(trace, "colorscale", None))
            terrain_x_axis = x[0, :] if x.ndim == 2 else x
            terrain_y_axis = y[:, 0] if y.ndim == 2 else y
            continue

        if trace_type == "surface" and name == "障碍":
            disk = _make_obstacle_disk(trace)
            if disk is not None:
                obstacle_disks.append(disk)
            continue

        if trace_type == "scatter3d":
            finite_xy = _finite_xy(trace)
            if finite_xy is None:
                continue
            x, y = finite_xy
            color = _scatter_color(trace)
            if name.startswith("智能体 "):
                trajectory_traces.append((x, y, color))
            elif name.startswith("起点"):
                start_traces.append((x, y, color))
            elif "目标" in name and "智能体" in name:
                # Keep the last target trace for each agent because it uses
                # the trajectory-matched color in the current HTML exports.
                goal_by_name[name] = (x, y, color)

    if terrain is None or terrain_x_axis is None or terrain_y_axis is None:
        raise ValueError(f"Terrain surface not found in {spec.html_path}")

    _, _, z, colorscale = terrain
    x_range = _resolve_axis_range(terrain_x_axis)
    y_range = _resolve_axis_range(terrain_y_axis)
    extent = [
        float(np.min(terrain_x_axis)),
        float(np.max(terrain_x_axis)),
        float(np.min(terrain_y_axis)),
        float(np.max(terrain_y_axis)),
    ]
    cmap = _colorscale_to_cmap(colorscale)
    z_min = float(np.nanmin(z))
    z_max = float(np.nanmax(z))
    contour_levels = np.linspace(z_min, z_max, 13)

    fig, ax = plt.subplots(figsize=(5.1, 4.55), constrained_layout=False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    terrain_im = ax.imshow(
        z,
        extent=extent,
        origin="lower",
        cmap=cmap,
        interpolation="bilinear",
        aspect="equal",
        zorder=0,
    )
    ax.contour(
        terrain_x_axis,
        terrain_y_axis,
        z,
        levels=contour_levels,
        colors=[(1.0, 1.0, 1.0, 0.45)],
        linewidths=0.55,
        zorder=1,
    )

    for disk in obstacle_disks:
        ax.add_patch(
            Circle(
                (disk.cx, disk.cy),
                disk.radius,
                facecolor=_mpl_color("rgba(220,40,35,0.22)"),
                edgecolor=_mpl_color("rgba(220,40,35,0.55)"),
                linewidth=0.8,
                zorder=2,
            )
        )

    for x, y, color in trajectory_traces:
        ax.plot(x, y, color=_mpl_color(color), linewidth=2.2, solid_capstyle="round", zorder=3)

    for x, y, color in start_traces:
        ax.scatter(
            x,
            y,
            s=30,
            marker="s",
            c=[_mpl_color(color)],
            edgecolors="white",
            linewidths=0.9,
            zorder=4,
            clip_on=False,
        )
    for key in sorted(goal_by_name):
        x, y, color = goal_by_name[key]
        ax.scatter(
            x,
            y,
            s=38,
            marker="D",
            c=[_mpl_color(color)],
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
            clip_on=False,
        )

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0)
    ax.set_xlabel("X", fontsize=11, labelpad=0.5)
    ax.set_ylabel("Y", fontsize=11, labelpad=0.5)
    ax.tick_params(axis="both", labelsize=8, length=2.5, pad=1.0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.0%", pad=0.018)
    cbar = fig.colorbar(terrain_im, cax=cax)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7, length=0, pad=0.8)
    cbar.ax.set_title("Height (m)", fontsize=7, pad=1.5, loc="left")

    fig.subplots_adjust(left=0.085, right=0.94, bottom=0.10, top=0.995)
    return fig


def _save_top_view_figure(spec: PanelSpec, pdf_path: Path, svg_path: Path, png_path: Path | None = None) -> None:
    fig = _make_top_view_figure(spec)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.01)
    if png_path is not None:
        fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def export_all() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for spec in PANEL_SPECS:
        if not spec.html_path.exists():
            print(f"skipped missing source: {spec.html_path}")
            continue

        fig = _make_paper_ready_figure(spec)
        pdf_path = PAPER_DIR / f"{spec.key}.pdf"
        svg_path = PAPER_DIR / f"{spec.key}.svg"
        fig.write_image(pdf_path)
        fig.write_image(svg_path)
        print(f"exported: {pdf_path}")
        print(f"exported: {svg_path}")

        top_pdf_path = PAPER_DIR / f"{spec.key}_top.pdf"
        top_svg_path = PAPER_DIR / f"{spec.key}_top.svg"
        _save_top_view_figure(spec, top_pdf_path, top_svg_path)
        print(f"exported: {top_pdf_path}")
        print(f"exported: {top_svg_path}")


if __name__ == "__main__":
    export_all()
