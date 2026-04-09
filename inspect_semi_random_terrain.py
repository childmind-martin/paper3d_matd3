#!/usr/bin/env python3
"""
Export one semi-random terrain family baseline and four perturbed variants as standalone HTML files.

The default baseline is the "family base" map:
- same terrain base seed
- same semi-random generation pipeline
- zero local peak shift
- zero peak-height jitter
- zero variant noise

This is the most useful reference when inspecting what the current semi-random terrain logic
actually changes relative to the shared base terrain family.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.image as mpimg
import numpy as np

os.environ.setdefault("SUPPRESS_MA_PROMPT", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matd3_mplconfig")

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap

from multiagent.scenarios.paper3d_terrain_energy import Scenario


DEFAULT_VARIANT_SEEDS = [101, 202, 303, 404]
DEFAULT_MAP_SIZE = 200
DEFAULT_COMPLEXITY = 3

# Current Group-B semi-random training defaults.
DEFAULT_PEAK_JITTER_RANGE = 12.0
DEFAULT_PEAK_CENTER_JITTER_RANGE = 2.0
DEFAULT_HEIGHT_JITTER_MIN = 0.15
DEFAULT_HEIGHT_JITTER_MAX = 0.30
DEFAULT_HEIGHT_CAP = 1.20
DEFAULT_VARIANT_NOISE_RATIO = 0.10
DEFAULT_PAPER_CASE_LABELS = "base_family_seed_88,semi_random_base_88_variant_101,semi_random_base_88_variant_202"
DEFAULT_GOAL_POSITION_FILE = (
    Path(__file__).resolve().parent / "saved_positions" / "strict_ablation_seed88_groupB.json"
)
PLOTLY_TERRAIN_COLORSCALE = [
    (0.00, (220, 220, 180)),
    (0.30, (180, 200, 120)),
    (0.50, (120, 160, 100)),
    (0.70, (100, 120, 80)),
    (0.85, (139, 137, 137)),
    (1.00, (255, 255, 255)),
]


@contextmanager
def _temporary_env(updates: Dict[str, Optional[str]]):
    original: Dict[str, Optional[str]] = {}
    for key, value in updates.items():
        original[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _parse_variant_seeds(raw: str) -> List[int]:
    values = [chunk.strip() for chunk in str(raw).split(",")]
    seeds = [int(chunk) for chunk in values if chunk]
    if not seeds:
        raise ValueError("variant seed list is empty")
    return seeds


def _as_float_triplets(items: Sequence[Sequence[float]]) -> List[List[float]]:
    out: List[List[float]] = []
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        out.append([float(item[0]), float(item[1]), float(item[2])])
    return out


def _compute_peak_shift_stats(
    base_centers: Sequence[Sequence[float]],
    actual_centers: Sequence[Sequence[float]],
) -> Dict[str, Optional[float]]:
    if not base_centers or not actual_centers or len(base_centers) != len(actual_centers):
        return {
            "mean_peak_center_shift": None,
            "max_peak_center_shift": None,
            "mean_peak_height_sample_delta": None,
            "max_peak_height_sample_delta": None,
        }

    center_shifts: List[float] = []
    height_deltas: List[float] = []
    for base_row, actual_row in zip(base_centers, actual_centers):
        bx, by, bz = float(base_row[0]), float(base_row[1]), float(base_row[2])
        ax, ay, az = float(actual_row[0]), float(actual_row[1]), float(actual_row[2])
        center_shifts.append(float(np.hypot(ax - bx, ay - by)))
        height_deltas.append(float(abs(az - bz)))

    return {
        "mean_peak_center_shift": float(np.mean(center_shifts)) if center_shifts else None,
        "max_peak_center_shift": float(np.max(center_shifts)) if center_shifts else None,
        "mean_peak_height_sample_delta": float(np.mean(height_deltas)) if height_deltas else None,
        "max_peak_height_sample_delta": float(np.max(height_deltas)) if height_deltas else None,
    }


def _compute_samples(map_size: int, sample_rate: int, terrain_shape: Tuple[int, int]) -> Tuple[List[int], List[int]]:
    if sample_rate <= 1:
        x_samples = list(range(int(terrain_shape[1])))
        y_samples = list(range(int(terrain_shape[0])))
        return x_samples, y_samples

    x_samples = list(np.arange(0, map_size, sample_rate, dtype=np.int32).tolist())
    y_samples = list(np.arange(0, map_size, sample_rate, dtype=np.int32).tolist())
    if (map_size - 1) % sample_rate != 0:
        x_samples.append(int(map_size - 1))
        y_samples.append(int(map_size - 1))

    x_samples = x_samples[: terrain_shape[1]]
    y_samples = y_samples[: terrain_shape[0]]
    return x_samples, y_samples


def _build_case(
    *,
    label: str,
    base_seed: int,
    map_size: int,
    terrain_complexity_level: int,
    use_semi_random: bool,
    variant_seed: Optional[int],
    peak_jitter_range: float,
    peak_center_jitter_range: float,
    peak_height_jitter_ratio_min: float,
    peak_height_jitter_ratio_max: float,
    peak_height_max_scale: float,
    terrain_variant_noise_ratio: float,
    baseline_mode: str,
) -> Dict[str, object]:
    env_updates = {
        "SUPPRESS_MA_PROMPT": "1",
        "SUPPRESS_TERRAIN_OUTPUT": "1",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": "/tmp/matd3_mplconfig",
        "USE_DYNAMIC_OBSTACLES": "0",
        "USE_SCENARIO_SEED": "1",
        "SCENARIO_SEED": str(base_seed),
        "MAP_SIZE": str(map_size),
        "TERRAIN_COMPLEXITY_LEVEL": str(terrain_complexity_level),
        "SEMI_RANDOM_TERRAIN": "1" if use_semi_random else "0",
        "TERRAIN_BASE_SEED": str(base_seed),
        "PEAK_JITTER_RANGE": str(peak_jitter_range),
        "PEAK_CENTER_JITTER_RANGE": str(peak_center_jitter_range),
        "PEAK_HEIGHT_JITTER_RATIO_MIN": str(peak_height_jitter_ratio_min),
        "PEAK_HEIGHT_JITTER_RATIO_MAX": str(peak_height_jitter_ratio_max),
        "PEAK_HEIGHT_MAX_SCALE": str(peak_height_max_scale),
        "TERRAIN_VARIANT_NOISE_RATIO": str(terrain_variant_noise_ratio),
        "TERRAIN_VARIANT_SEED": str(variant_seed if variant_seed is not None else base_seed),
    }

    with _temporary_env(env_updates):
        scenario = Scenario(
            seed=base_seed,
            random_terrain=False,
            use_fixed_positions=False,
            terrain_complexity_level=terrain_complexity_level,
            map_size=map_size,
        )
        scenario.generate_terrain()
        snapshot = scenario.build_terrain_snapshot()
        terrain = np.asarray(snapshot.get("terrain"), dtype=np.float32)
        if terrain.ndim != 2:
            raise RuntimeError(f"{label}: terrain array is invalid with shape {terrain.shape}")
        base_centers = _as_float_triplets(snapshot.get("base_mountain_centers") or [])
        actual_centers = _as_float_triplets(snapshot.get("actual_mountain_centers") or [])
        sample_rate = int(getattr(scenario, "terrain_sample_rate", 1) or 1)
        x_samples, y_samples = _compute_samples(int(map_size), sample_rate, terrain.shape)
        start_area = dict(getattr(scenario, "start_area", {}) or {})
        terrain_params = dict(snapshot.get("terrain_params") or {})

    stats = {
        "label": label,
        "baseline_mode": baseline_mode,
        "map_size": int(map_size),
        "terrain_shape": list(terrain.shape),
        "sample_rate": int(sample_rate),
        "base_seed": int(base_seed),
        "variant_seed": int(variant_seed) if variant_seed is not None else None,
        "use_semi_random": bool(use_semi_random),
        "peak_count": int(len(actual_centers)),
        "height_min": float(np.min(terrain)),
        "height_mean": float(np.mean(terrain)),
        "height_std": float(np.std(terrain)),
        "height_max": float(np.max(terrain)),
        "terrain_params": terrain_params,
        "start_area": start_area,
        "base_mountain_centers": base_centers,
        "actual_mountain_centers": actual_centers,
        "x_samples": x_samples,
        "y_samples": y_samples,
        "terrain": terrain,
    }
    stats.update(_compute_peak_shift_stats(base_centers, actual_centers))
    return stats


def _apply_reference_delta(case: Dict[str, object], reference_terrain: np.ndarray) -> None:
    terrain = np.asarray(case["terrain"], dtype=np.float32)
    if terrain.shape != reference_terrain.shape:
        case["mean_abs_height_delta_vs_reference"] = None
        case["max_abs_height_delta_vs_reference"] = None
        return
    delta = np.abs(terrain - reference_terrain)
    case["mean_abs_height_delta_vs_reference"] = float(np.mean(delta))
    case["max_abs_height_delta_vs_reference"] = float(np.max(delta))


def _build_shift_line_segments(
    base_centers: Sequence[Sequence[float]],
    actual_centers: Sequence[Sequence[float]],
) -> List[List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    if len(base_centers) != len(actual_centers):
        return [xs, ys, zs]
    for base_row, actual_row in zip(base_centers, actual_centers):
        xs.extend([float(base_row[0]), float(actual_row[0]), None])
        ys.extend([float(base_row[1]), float(actual_row[1]), None])
        zs.extend([float(base_row[2]), float(actual_row[2]), None])
    return [xs, ys, zs]


def _format_stat(value: Optional[float], fmt: str = "{:.2f}") -> str:
    if value is None:
        return "N/A"
    return fmt.format(float(value))


def _plotly_colorscale_json() -> List[List[object]]:
    return [
        [float(stop), f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"]
        for stop, rgb in PLOTLY_TERRAIN_COLORSCALE
    ]


def _plotly_terrain_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "plotly_terrain_surface",
        [(float(stop), tuple(channel / 255.0 for channel in rgb)) for stop, rgb in PLOTLY_TERRAIN_COLORSCALE],
    )


def _paper_case_title(case: Dict[str, object]) -> str:
    variant_seed = case.get("variant_seed")
    base_seed = case.get("base_seed")
    if variant_seed is None:
        return f"Raw Seed {base_seed}"
    if int(variant_seed) == int(base_seed) and str(case.get("label", "")).startswith("base_family"):
        return "Family Base"
    return f"Variant {variant_seed}"


def _resolve_optional_path(raw_path: Optional[str]) -> Optional[Path]:
    if raw_path is None:
        return None
    raw_str = str(raw_path).strip()
    if not raw_str:
        return None
    path = Path(raw_str).expanduser()
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent / path).resolve()
    return path


def _load_goal_position(goal_position_file: Optional[str]) -> Optional[np.ndarray]:
    goal_path = _resolve_optional_path(goal_position_file)
    if goal_path is None or not goal_path.is_file():
        return None

    payload = json.loads(goal_path.read_text(encoding="utf-8"))
    goal = None
    if isinstance(payload, dict):
        goal = payload.get("goal")
    elif isinstance(payload, list) and payload:
        goal = payload[-1]

    if goal is None:
        return None

    goal_pos = np.asarray(goal, dtype=np.float32).reshape(-1)
    if goal_pos.size < 2:
        return None
    if goal_pos.size == 2:
        goal_pos = np.asarray([goal_pos[0], goal_pos[1], 0.0], dtype=np.float32)
    return goal_pos[:3]


def _select_paper_cases(
    cases: Sequence[Dict[str, object]],
    requested_labels: Optional[str],
    max_cases: int = 3,
) -> List[Dict[str, object]]:
    requested = [chunk.strip() for chunk in str(requested_labels or "").split(",") if chunk.strip()]
    if not requested:
        return list(cases[:max_cases])

    lookup = {str(case["label"]): case for case in cases}
    selected: List[Dict[str, object]] = []
    missing: List[str] = []
    for label in requested:
        case = lookup.get(label)
        if case is None:
            missing.append(label)
            continue
        selected.append(case)

    if missing:
        raise ValueError(f"Unknown paper case label(s): {missing}")
    if not selected:
        raise ValueError("No paper cases selected.")
    return selected[:max_cases]


def _sample_case_height(case: Dict[str, object], x_coord: float, y_coord: float) -> float:
    terrain = np.asarray(case["terrain"], dtype=np.float32)
    x_samples = np.asarray(case["x_samples"], dtype=np.float32)
    y_samples = np.asarray(case["y_samples"], dtype=np.float32)
    x_index = int(np.argmin(np.abs(x_samples - float(x_coord))))
    y_index = int(np.argmin(np.abs(y_samples - float(y_coord))))
    return float(terrain[y_index, x_index])


def _build_case_plotly_figure(
    case: Dict[str, object],
    goal_pos: Optional[np.ndarray] = None,
    *,
    title: Optional[str] = None,
    show_scale: bool = False,
    show_legend: bool = False,
    show_peak_markers: bool = True,
    show_peak_shift: bool = True,
    show_goal_marker: bool = True,
    camera_eye: Optional[Dict[str, float]] = None,
    axis_title_size: int = 12,
    axis_tick_size: int = 10,
):
    import plotly.graph_objects as go

    terrain = np.asarray(case["terrain"], dtype=np.float32)
    x_samples = list(case["x_samples"])
    y_samples = list(case["y_samples"])
    base_centers = list(case.get("base_mountain_centers") or [])
    actual_centers = list(case.get("actual_mountain_centers") or [])
    shift_lines = _build_shift_line_segments(base_centers, actual_centers)

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=x_samples,
            y=y_samples,
            z=terrain,
            colorscale=_plotly_colorscale_json(),
            name="terrain",
            showscale=show_scale,
            opacity=1.0,
            colorbar=(
                dict(
                    title="Height (m)",
                    len=0.78,
                    thickness=14,
                    x=1.02,
                    xanchor="left",
                )
                if show_scale else None
            ),
            lighting=dict(
                ambient=0.62,
                diffuse=0.82,
                specular=0.18,
                roughness=0.55,
            ),
            contours=dict(
                z=dict(
                    show=True,
                    usecolormap=True,
                    highlightcolor="limegreen",
                    project=dict(z=False),
                )
            ),
        )
    )

    if show_peak_markers and base_centers:
        fig.add_trace(
            go.Scatter3d(
                x=[row[0] for row in base_centers],
                y=[row[1] for row in base_centers],
                z=[row[2] + 1.0 for row in base_centers],
                mode="markers",
                name="base peaks",
                marker=dict(size=4, color="royalblue", symbol="diamond"),
            )
        )

    if show_peak_markers and actual_centers:
        fig.add_trace(
            go.Scatter3d(
                x=[row[0] for row in actual_centers],
                y=[row[1] for row in actual_centers],
                z=[row[2] + 1.5 for row in actual_centers],
                mode="markers",
                name="actual peaks",
                marker=dict(size=5, color="orangered", symbol="circle"),
            )
        )

    if show_peak_shift and shift_lines[0]:
        fig.add_trace(
            go.Scatter3d(
                x=shift_lines[0],
                y=shift_lines[1],
                z=shift_lines[2],
                mode="lines",
                name="peak shift",
                line=dict(color="rgba(40, 40, 40, 0.45)", width=4),
                hoverinfo="skip",
            )
        )

    if show_goal_marker and goal_pos is not None and goal_pos.size >= 2:
        goal_x = float(np.clip(goal_pos[0], 0.0, float(case.get("map_size", 200.0))))
        goal_y = float(np.clip(goal_pos[1], 0.0, float(case.get("map_size", 200.0))))
        goal_z = _sample_case_height(case, goal_x, goal_y) + 1.5
        fig.add_trace(
            go.Scatter3d(
                x=[goal_x],
                y=[goal_y],
                z=[goal_z],
                mode="markers",
                name="goal",
                marker=dict(size=7, color="crimson", symbol="circle"),
            )
        )

    fig.update_layout(
        title=(dict(text=title, x=0.5, xanchor="center") if title else None),
        margin=dict(l=0, r=0, t=18 if title else 4, b=0),
        paper_bgcolor="#ffffff",
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.98,
            xanchor="left",
            x=0.02,
        ),
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="X", titlefont=dict(size=axis_title_size), tickfont=dict(size=axis_tick_size)),
            yaxis=dict(title="Y", titlefont=dict(size=axis_title_size), tickfont=dict(size=axis_tick_size)),
            zaxis=dict(title="Z", titlefont=dict(size=axis_title_size), tickfont=dict(size=axis_tick_size)),
            camera=dict(eye=(camera_eye or dict(x=1.55, y=1.55, z=0.9))),
        ),
    )
    return fig


def _crop_plotly_export_whitespace(image_path: Path, padding: int = 8) -> Path:
    from PIL import Image, ImageChops

    image = Image.open(image_path).convert("RGB")
    background = Image.new("RGB", image.size, (255, 255, 255))
    diff = ImageChops.difference(image, background)
    bbox = diff.getbbox()
    if bbox is None:
        return image_path

    left = max(0, bbox[0] - padding)
    top = max(0, bbox[1] - padding)
    right = min(image.width, bbox[2] + padding)
    bottom = min(image.height, bbox[3] + padding)
    image.crop((left, top, right, bottom)).save(image_path)
    return image_path


def _export_plotly_case_image(
    case: Dict[str, object],
    output_path: Path,
    goal_pos: Optional[np.ndarray] = None,
) -> Path:
    fig = _build_case_plotly_figure(
        case,
        goal_pos=goal_pos,
        title=None,
        show_scale=False,
        show_legend=False,
        show_peak_markers=False,
        show_peak_shift=False,
        show_goal_marker=False,
        camera_eye=dict(x=1.38, y=1.34, z=0.78),
        axis_title_size=24,
        axis_tick_size=18,
    )
    # Export the Plotly panel at a much higher native resolution and keep that
    # resolution intact in the final paper figure. The previous pipeline looked
    # blurry because Matplotlib resampled the top-row images down to figure-dpi
    # resolution before embedding them into PDF.
    fig.write_image(str(output_path), width=1200, height=820, scale=3)
    return _crop_plotly_export_whitespace(output_path, padding=2)


def _write_paper_terrain_figure(
    cases: Sequence[Dict[str, object]],
    output_stem: Path,
    goal_pos: Optional[np.ndarray] = None,
) -> List[Path]:
    selected_cases = list(cases[:3])
    if not selected_cases:
        raise ValueError("At least one case is required to generate the paper terrain figure.")

    top_row_images: List[Optional[np.ndarray]] = [None for _ in selected_cases]
    plotly_export_error: Optional[Exception] = None
    try:
        with tempfile.TemporaryDirectory(prefix="terrain_paper_plotly_", dir=str(output_stem.parent)) as temp_dir:
            for index, case in enumerate(selected_cases):
                plotly_path = Path(temp_dir) / f"plotly_top_{index}.png"
                _export_plotly_case_image(case, plotly_path, goal_pos=goal_pos)
                top_row_images[index] = mpimg.imread(str(plotly_path))
    except Exception as exc:
        plotly_export_error = exc

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    # A higher figure dpi helps preserve any remaining raster artists when the
    # combined panel is written to PDF/SVG, while the contour row stays vector.
    fig = plt.figure(figsize=(14.2, 8.4), facecolor="white", dpi=400)
    grid = gridspec.GridSpec(
        2,
        len(selected_cases),
        figure=fig,
        height_ratios=[1.2, 1.04],
        hspace=0.04,
        wspace=0.08,
    )
    fig.subplots_adjust(left=0.022, right=0.995, top=0.965, bottom=0.045)
    cmap = _plotly_terrain_cmap()

    for index, case in enumerate(selected_cases):
        terrain = np.asarray(case["terrain"], dtype=np.float32)
        x_samples = np.asarray(case["x_samples"], dtype=np.float32)
        y_samples = np.asarray(case["y_samples"], dtype=np.float32)
        grid_x, grid_y = np.meshgrid(x_samples, y_samples, indexing="xy")
        map_limit = float(case.get("map_size", max(np.max(x_samples), np.max(y_samples)) + 1))
        column_title = _paper_case_title(case)
        case_vmin = float(np.min(terrain))
        case_vmax = float(np.max(terrain))
        case_z_padding = max(8.0, 0.08 * max(case_vmax - case_vmin, 1.0))
        levels = np.linspace(case_vmin + 0.10 * (case_vmax - case_vmin), case_vmax, 5)

        top_image = top_row_images[index]
        if top_image is not None:
            ax_top = fig.add_subplot(grid[0, index])
            # `interpolation="none"` is the key fix here: on PDF/SVG backends it
            # passes the native pixel buffer through instead of resampling it to
            # the canvas dpi, which previously collapsed the Plotly exports to
            # about 100 ppi inside the final PDF.
            ax_top.imshow(top_image, interpolation="none", resample=False)
            ax_top.set_axis_off()
            ax_top.set_title(column_title, pad=4)
        else:
            ax3d = fig.add_subplot(grid[0, index], projection="3d")
            ax3d.plot_surface(
                grid_x,
                grid_y,
                terrain,
                cmap=cmap,
                vmin=case_vmin,
                vmax=case_vmax,
                linewidth=0.0,
                antialiased=False,
                rstride=1,
                cstride=1,
                shade=False,
            )
            ax3d.contour(
                grid_x,
                grid_y,
                terrain,
                levels=levels,
                colors="#67ff67",
                linewidths=0.82,
            )
            ax3d.contour(
                grid_x,
                grid_y,
                terrain,
                zdir="x",
                offset=0.0,
                levels=levels,
                colors="#2f2f2f",
                linewidths=0.42,
                alpha=0.35,
            )
            ax3d.view_init(elev=34, azim=-64)
            ax3d.set_box_aspect((1.0, 1.0, 0.72))
            ax3d.set_xlim(0, map_limit)
            ax3d.set_ylim(0, map_limit)
            ax3d.set_zlim(case_vmin, case_vmax + case_z_padding)
            ax3d.xaxis.pane.set_facecolor((0.96, 0.96, 0.96, 0.30))
            ax3d.yaxis.pane.set_facecolor((0.96, 0.96, 0.96, 0.20))
            ax3d.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
                axis._axinfo["grid"]["color"] = (0.72, 0.72, 0.72, 0.55)
                axis._axinfo["grid"]["linewidth"] = 0.5
            ax3d.set_xlabel("X", fontsize=16, labelpad=-1)
            ax3d.set_ylabel("Y", fontsize=16, labelpad=-1)
            ax3d.set_zlabel("Z", fontsize=16, labelpad=0)
            ax3d.set_xticks(np.linspace(0, map_limit, 5))
            ax3d.set_yticks(np.linspace(0, map_limit, 5))
            ax3d.set_title(column_title, pad=2)
            ax3d.tick_params(axis="both", pad=0, labelsize=11.5)

        ax2d = fig.add_subplot(grid[1, index])
        mesh = ax2d.pcolormesh(
            grid_x,
            grid_y,
            terrain,
            cmap=cmap,
            vmin=case_vmin,
            vmax=case_vmax,
            shading="nearest",
        )
        ax2d.contour(
            grid_x,
            grid_y,
            terrain,
            levels=levels,
            colors="#edf0de",
            linewidths=0.48,
            alpha=0.62,
        )
        ax2d.set_aspect("equal", adjustable="box")
        ax2d.set_xlim(0, map_limit)
        ax2d.set_ylim(0, map_limit)
        ax2d.set_xlabel("X", fontsize=17, labelpad=4)
        ax2d.set_ylabel("Y", fontsize=17, labelpad=1)
        ax2d.set_xticks(np.linspace(0, map_limit, 5))
        ax2d.set_yticks(np.linspace(0, map_limit, 5))
        ax2d.tick_params(axis="both", labelsize=13.5)

        colorbar = fig.colorbar(
            mesh,
            ax=ax2d,
            fraction=0.046,
            pad=0.018,
            ticks=np.linspace(case_vmin, case_vmax, 5),
        )
        colorbar.ax.tick_params(labelsize=11.5, pad=1)
        colorbar.ax.set_title("Z", fontsize=13, pad=4)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    svg_path = output_stem.with_suffix(".svg")
    fig.savefig(png_path, dpi=500, bbox_inches="tight", pad_inches=0.005, facecolor="white")
    fig.savefig(pdf_path, dpi=500, bbox_inches="tight", pad_inches=0.005, facecolor="white")
    fig.savefig(svg_path, dpi=500, bbox_inches="tight", pad_inches=0.005, facecolor="white")
    plt.close(fig)
    if plotly_export_error is not None:
        print(f"[paper figure] Plotly export unavailable, used fallback top row: {plotly_export_error}")
    return [png_path, pdf_path, svg_path]


def _write_case_html(case: Dict[str, object], output_path: Path) -> None:
    terrain = np.asarray(case["terrain"], dtype=np.float32)
    x_samples = list(case["x_samples"])
    y_samples = list(case["y_samples"])
    base_centers = list(case["base_mountain_centers"])
    actual_centers = list(case["actual_mountain_centers"])
    shift_lines = _build_shift_line_segments(base_centers, actual_centers)
    terrain_params = dict(case.get("terrain_params") or {})
    start_area = dict(case.get("start_area") or {})

    info_lines = [
        f"<div class='stat'><span class='label'>Label:</span> {case['label']}</div>",
        f"<div class='stat'><span class='label'>Base Seed:</span> {case['base_seed']}</div>",
        f"<div class='stat'><span class='label'>Variant Seed:</span> {case['variant_seed'] if case['variant_seed'] is not None else 'base'}</div>",
        f"<div class='stat'><span class='label'>Peak Count:</span> {case['peak_count']}</div>",
        f"<div class='stat'><span class='label'>Max Height:</span> {_format_stat(case.get('height_max'))} m</div>",
        f"<div class='stat'><span class='label'>Mean Height:</span> {_format_stat(case.get('height_mean'))} m</div>",
        f"<div class='stat'><span class='label'>Mean |Δz| vs Base:</span> {_format_stat(case.get('mean_abs_height_delta_vs_reference'))} m</div>",
        f"<div class='stat'><span class='label'>Max |Δz| vs Base:</span> {_format_stat(case.get('max_abs_height_delta_vs_reference'))} m</div>",
        f"<div class='stat'><span class='label'>Mean Peak Shift:</span> {_format_stat(case.get('mean_peak_center_shift'))} m</div>",
        f"<div class='stat'><span class='label'>Max Peak Shift:</span> {_format_stat(case.get('max_peak_center_shift'))} m</div>",
        f"<div class='stat'><span class='label'>Start Area Source:</span> {start_area.get('source', 'N/A')}</div>",
        f"<div class='stat'><span class='label'>Center Jitter:</span> {_format_stat(terrain_params.get('peak_center_jitter_range'))} m</div>",
        f"<div class='stat'><span class='label'>Height Jitter:</span> {terrain_params.get('peak_height_jitter_ratio_range', 'N/A')}</div>",
        f"<div class='stat'><span class='label'>Variant Noise Ratio:</span> {_format_stat(terrain_params.get('terrain_variant_noise_ratio'))}</div>",
    ]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{case['label']}</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body {{
      margin: 0;
      padding: 18px;
      font-family: Arial, sans-serif;
      background: #eef1f5;
      color: #1d2430;
    }}
    #info {{
      background: #ffffff;
      border-radius: 10px;
      padding: 16px 18px;
      box-shadow: 0 8px 24px rgba(21, 33, 56, 0.10);
      margin-bottom: 12px;
      line-height: 1.6;
    }}
    #plot {{
      width: 100%;
      height: 88vh;
      background: #ffffff;
      border-radius: 10px;
      box-shadow: 0 8px 24px rgba(21, 33, 56, 0.10);
    }}
    .stat {{
      display: inline-block;
      margin: 0 18px 6px 0;
      white-space: nowrap;
    }}
    .label {{
      font-weight: 700;
      color: #243447;
    }}
  </style>
</head>
<body>
  <div id="info">
    <h2 style="margin: 0 0 10px 0;">{case['label']}</h2>
    {''.join(info_lines)}
  </div>
  <div id="plot"></div>
  <script>
    const terrainZ = {json.dumps(terrain.tolist())};
    const xSamples = {json.dumps(x_samples)};
    const ySamples = {json.dumps(y_samples)};
    const baseCenters = {json.dumps(base_centers)};
    const actualCenters = {json.dumps(actual_centers)};
    const shiftLineX = {json.dumps(shift_lines[0])};
    const shiftLineY = {json.dumps(shift_lines[1])};
    const shiftLineZ = {json.dumps(shift_lines[2])};

    const terrainTrace = {{
      type: 'surface',
      x: xSamples,
      y: ySamples,
      z: terrainZ,
      colorscale: {json.dumps(_plotly_colorscale_json())},
      name: 'terrain',
      showscale: true,
      opacity: 1.0,
      colorbar: {{
        title: 'Height (m)',
        len: 0.78,
        thickness: 16,
        x: 1.02,
        xanchor: 'left'
      }},
      lighting: {{
        ambient: 0.62,
        diffuse: 0.82,
        specular: 0.18,
        roughness: 0.55
      }},
      contours: {{
        z: {{
          show: true,
          usecolormap: true,
          highlightcolor: 'limegreen',
          project: {{z: false}}
        }}
      }}
    }};

    const traces = [terrainTrace];

    if (baseCenters.length > 0) {{
      traces.push({{
        type: 'scatter3d',
        mode: 'markers',
        name: 'base peaks',
        x: baseCenters.map(row => row[0]),
        y: baseCenters.map(row => row[1]),
        z: baseCenters.map(row => row[2] + 1.0),
        marker: {{
          size: 5,
          color: 'royalblue',
          symbol: 'diamond'
        }}
      }});
    }}

    if (actualCenters.length > 0) {{
      traces.push({{
        type: 'scatter3d',
        mode: 'markers',
        name: 'actual peaks',
        x: actualCenters.map(row => row[0]),
        y: actualCenters.map(row => row[1]),
        z: actualCenters.map(row => row[2] + 1.5),
        marker: {{
          size: 5,
          color: 'orangered',
          symbol: 'circle'
        }}
      }});
    }}

    if (shiftLineX.length > 0) {{
      traces.push({{
        type: 'scatter3d',
        mode: 'lines',
        name: 'peak shift',
        x: shiftLineX,
        y: shiftLineY,
        z: shiftLineZ,
        line: {{
          color: 'rgba(40, 40, 40, 0.45)',
          width: 4
        }},
        hoverinfo: 'skip'
      }});
    }}

    const layout = {{
      title: {{
        text: '{case["label"]}',
        x: 0.5,
        xanchor: 'center'
      }},
      margin: {{l: 0, r: 0, t: 52, b: 0}},
      paper_bgcolor: '#ffffff',
      scene: {{
        aspectmode: 'data',
        xaxis: {{title: 'X'}},
        yaxis: {{title: 'Y'}},
        zaxis: {{title: 'Height'}},
        camera: {{
          eye: {{x: 1.55, y: 1.55, z: 0.9}}
        }}
      }},
      legend: {{
        orientation: 'h',
        yanchor: 'bottom',
        y: 0.98,
        xanchor: 'left',
        x: 0.02
      }}
    }};

    Plotly.newPlot('plot', traces, layout, {{
      responsive: true,
      displaylogo: false,
      scrollZoom: true
    }});
  </script>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export semi-random terrain family previews as HTML.")
    parser.add_argument("--base-seed", type=int, default=88, help="Base terrain seed.")
    parser.add_argument(
        "--variant-seeds",
        type=str,
        default=",".join(str(seed) for seed in DEFAULT_VARIANT_SEEDS),
        help="Comma-separated list of four variant seeds.",
    )
    parser.add_argument(
        "--baseline-mode",
        type=str,
        default="family_base",
        choices=["family_base", "raw_seed"],
        help="family_base=semi-random family base with zero perturbation; raw_seed=plain fixed terrain from the seed.",
    )
    parser.add_argument("--output-dir", type=str, default="terrain_previews", help="Output directory.")
    parser.add_argument("--map-size", type=int, default=DEFAULT_MAP_SIZE, help="Map size.")
    parser.add_argument(
        "--terrain-complexity-level",
        type=int,
        default=DEFAULT_COMPLEXITY,
        help="Terrain complexity level.",
    )
    parser.add_argument("--peak-jitter-range", type=float, default=DEFAULT_PEAK_JITTER_RANGE)
    parser.add_argument("--peak-center-jitter-range", type=float, default=DEFAULT_PEAK_CENTER_JITTER_RANGE)
    parser.add_argument("--peak-height-jitter-ratio-min", type=float, default=DEFAULT_HEIGHT_JITTER_MIN)
    parser.add_argument("--peak-height-jitter-ratio-max", type=float, default=DEFAULT_HEIGHT_JITTER_MAX)
    parser.add_argument("--peak-height-max-scale", type=float, default=DEFAULT_HEIGHT_CAP)
    parser.add_argument("--terrain-variant-noise-ratio", type=float, default=DEFAULT_VARIANT_NOISE_RATIO)
    parser.add_argument(
        "--paper-figure-stem",
        type=str,
        default="terrain_family_paper",
        help="If non-empty, save a matched static paper figure as both PNG and PDF using this output stem.",
    )
    parser.add_argument(
        "--paper-case-labels",
        type=str,
        default=DEFAULT_PAPER_CASE_LABELS,
        help="Comma-separated case labels used in the paper figure (default: family base + variants 101 and 202).",
    )
    parser.add_argument(
        "--goal-position-file",
        type=str,
        default=str(DEFAULT_GOAL_POSITION_FILE),
        help="Optional JSON file that provides the shared goal position overlay for the paper figure.",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    variant_seeds = _parse_variant_seeds(args.variant_seeds)
    if len(variant_seeds) != 4:
        raise ValueError(f"Expected exactly 4 variant seeds, got {len(variant_seeds)}: {variant_seeds}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.baseline_mode == "family_base":
        base_case = _build_case(
            label=f"base_family_seed_{args.base_seed}",
            base_seed=int(args.base_seed),
            map_size=int(args.map_size),
            terrain_complexity_level=int(args.terrain_complexity_level),
            use_semi_random=True,
            variant_seed=int(args.base_seed),
            peak_jitter_range=float(args.peak_jitter_range),
            peak_center_jitter_range=0.0,
            peak_height_jitter_ratio_min=0.0,
            peak_height_jitter_ratio_max=0.0,
            peak_height_max_scale=1.0,
            terrain_variant_noise_ratio=0.0,
            baseline_mode=args.baseline_mode,
        )
    else:
        base_case = _build_case(
            label=f"raw_seed_{args.base_seed}",
            base_seed=int(args.base_seed),
            map_size=int(args.map_size),
            terrain_complexity_level=int(args.terrain_complexity_level),
            use_semi_random=False,
            variant_seed=None,
            peak_jitter_range=float(args.peak_jitter_range),
            peak_center_jitter_range=float(args.peak_center_jitter_range),
            peak_height_jitter_ratio_min=float(args.peak_height_jitter_ratio_min),
            peak_height_jitter_ratio_max=float(args.peak_height_jitter_ratio_max),
            peak_height_max_scale=float(args.peak_height_max_scale),
            terrain_variant_noise_ratio=float(args.terrain_variant_noise_ratio),
            baseline_mode=args.baseline_mode,
        )

    reference_terrain = np.asarray(base_case["terrain"], dtype=np.float32)
    _apply_reference_delta(base_case, reference_terrain)

    cases: List[Dict[str, object]] = [base_case]
    for variant_seed in variant_seeds:
        case = _build_case(
            label=f"semi_random_base_{args.base_seed}_variant_{variant_seed}",
            base_seed=int(args.base_seed),
            map_size=int(args.map_size),
            terrain_complexity_level=int(args.terrain_complexity_level),
            use_semi_random=True,
            variant_seed=int(variant_seed),
            peak_jitter_range=float(args.peak_jitter_range),
            peak_center_jitter_range=float(args.peak_center_jitter_range),
            peak_height_jitter_ratio_min=float(args.peak_height_jitter_ratio_min),
            peak_height_jitter_ratio_max=float(args.peak_height_jitter_ratio_max),
            peak_height_max_scale=float(args.peak_height_max_scale),
            terrain_variant_noise_ratio=float(args.terrain_variant_noise_ratio),
            baseline_mode=args.baseline_mode,
        )
        _apply_reference_delta(case, reference_terrain)
        cases.append(case)

    summary_rows: List[Dict[str, object]] = []
    for case in cases:
        file_name = f"{case['label']}.html"
        html_path = output_dir / file_name
        _write_case_html(case, html_path)

        summary_rows.append(
            {
                "label": case["label"],
                "file": file_name,
                "base_seed": case["base_seed"],
                "variant_seed": case["variant_seed"],
                "use_semi_random": case["use_semi_random"],
                "height_mean": case["height_mean"],
                "height_max": case["height_max"],
                "mean_abs_height_delta_vs_reference": case.get("mean_abs_height_delta_vs_reference"),
                "max_abs_height_delta_vs_reference": case.get("max_abs_height_delta_vs_reference"),
                "mean_peak_center_shift": case.get("mean_peak_center_shift"),
                "max_peak_center_shift": case.get("max_peak_center_shift"),
                "start_area": case.get("start_area"),
                "terrain_params": case.get("terrain_params"),
            }
        )

    summary = {
        "baseline_mode": args.baseline_mode,
        "base_seed": int(args.base_seed),
        "variant_seeds": variant_seeds,
        "map_size": int(args.map_size),
        "terrain_complexity_level": int(args.terrain_complexity_level),
        "peak_jitter_range": float(args.peak_jitter_range),
        "peak_center_jitter_range": float(args.peak_center_jitter_range),
        "peak_height_jitter_ratio_min": float(args.peak_height_jitter_ratio_min),
        "peak_height_jitter_ratio_max": float(args.peak_height_jitter_ratio_max),
        "peak_height_max_scale": float(args.peak_height_max_scale),
        "terrain_variant_noise_ratio": float(args.terrain_variant_noise_ratio),
        "cases": summary_rows,
    }
    summary_path = output_dir / "terrain_family_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    paper_figure_paths: List[Path] = []
    paper_figure_stem = _resolve_optional_path(args.paper_figure_stem)
    if paper_figure_stem is not None:
        selected_paper_cases = _select_paper_cases(cases, args.paper_case_labels)
        goal_pos = _load_goal_position(args.goal_position_file)
        paper_figure_paths = _write_paper_terrain_figure(
            selected_paper_cases,
            paper_figure_stem,
            goal_pos=goal_pos,
        )

    print(f"Saved terrain previews to: {output_dir}")
    for row in summary_rows:
        print(
            f"- {row['file']}: variant={row['variant_seed']} | "
            f"mean_abs_dz={row['mean_abs_height_delta_vs_reference']:.3f} | "
            f"mean_peak_shift={row['mean_peak_center_shift'] if row['mean_peak_center_shift'] is not None else 'N/A'}"
        )
    print(f"Summary JSON: {summary_path}")
    for path in paper_figure_paths:
        print(f"Paper figure: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
