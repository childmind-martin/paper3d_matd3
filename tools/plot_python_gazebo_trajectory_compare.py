#!/usr/bin/env python3
"""Create a Python-vs-Gazebo trajectory comparison figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


AGENT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]


def _load_json(path: Path) -> Dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as f:
        return json.load(f)


def _episode(results: Dict[str, Any]) -> Dict[str, Any]:
    details = results.get("episode_details") or []
    if not details:
        raise ValueError("evaluation_results.json has no episode_details")
    return details[0]


def _trajectory(ep: Dict[str, Any]) -> np.ndarray:
    traj = np.asarray(ep.get("trajectory") or [], dtype=np.float64)
    if traj.ndim != 3 or traj.shape[2] < 3:
        raise ValueError("episode trajectory must have shape [frame, agent, xyz]")
    return traj[:, :, :3]


def _positions_from_items(items: Sequence[Dict[str, Any]], key: str = "position") -> np.ndarray:
    out: List[List[float]] = []
    for item in items or []:
        pos = item.get(key)
        if isinstance(pos, Sequence) and len(pos) >= 3:
            out.append([float(pos[0]), float(pos[1]), float(pos[2])])
    return np.asarray(out, dtype=np.float64)


def _scenario_paths(scenario: Dict[str, Any], scenario_json: Path) -> Tuple[Optional[Path], float]:
    terrain = scenario.get("terrain") if isinstance(scenario.get("terrain"), dict) else {}
    dense = terrain.get("dense_npy") or terrain.get("sampled_npy")
    dense_path = Path(dense).expanduser() if dense else None
    if dense_path is not None and not dense_path.is_absolute():
        dense_path = scenario_json.parent / dense_path
    map_size = float(scenario.get("map_size", 200.0) or 200.0)
    return dense_path, map_size


def _load_terrain(scenario: Dict[str, Any], scenario_json: Path) -> Tuple[Optional[np.ndarray], float]:
    terrain_path, map_size = _scenario_paths(scenario, scenario_json)
    if terrain_path is None or not terrain_path.exists():
        return None, map_size
    arr = np.asarray(np.load(terrain_path), dtype=np.float64)
    if arr.ndim != 2:
        return None, map_size
    return arr, map_size


def _axis_limits(*trajectories: np.ndarray, map_size: float) -> Tuple[float, float, float, float]:
    points = []
    for traj in trajectories:
        if traj.size:
            points.append(traj[:, :, :2].reshape(-1, 2))
    if not points:
        return 0.0, map_size, 0.0, map_size
    pts = np.vstack(points)
    finite = pts[np.all(np.isfinite(pts), axis=1)]
    if finite.size == 0:
        return 0.0, map_size, 0.0, map_size
    margin = 12.0
    xmin = max(0.0, float(np.min(finite[:, 0])) - margin)
    xmax = min(map_size, float(np.max(finite[:, 0])) + margin)
    ymin = max(0.0, float(np.min(finite[:, 1])) - margin)
    ymax = min(map_size, float(np.max(finite[:, 1])) + margin)
    return xmin, xmax, ymin, ymax


def _draw_scene(
    ax: plt.Axes,
    scenario: Dict[str, Any],
    terrain: Optional[np.ndarray],
    map_size: float,
    limits: Tuple[float, float, float, float],
) -> None:
    if terrain is not None:
        image = ax.imshow(
            terrain.T,
            extent=[0.0, map_size, 0.0, map_size],
            origin="lower",
            cmap="terrain",
            alpha=0.34,
            interpolation="bilinear",
            zorder=0,
        )
        image.set_rasterized(True)
    for obstacle in scenario.get("obstacles") or []:
        center = obstacle.get("center") or []
        if len(center) < 2:
            continue
        radius = float(obstacle.get("radius", 0.0) or 0.0)
        circle = Circle(
            (float(center[0]), float(center[1])),
            radius,
            facecolor="#d62728",
            edgecolor="#8b0000",
            alpha=0.22,
            linewidth=1.0,
            zorder=1,
        )
        ax.add_patch(circle)
    starts = _positions_from_items(scenario.get("start_positions") or [])
    goals = _positions_from_items(scenario.get("agent_goals") or [])
    for idx, start in enumerate(starts):
        color = AGENT_COLORS[idx % len(AGENT_COLORS)]
        ax.scatter(start[0], start[1], marker="o", s=42, color=color, edgecolor="white", linewidth=0.7, zorder=5)
    for idx, goal in enumerate(goals):
        color = AGENT_COLORS[idx % len(AGENT_COLORS)]
        ax.scatter(goal[0], goal[1], marker="*", s=130, color=color, edgecolor="black", linewidth=0.6, zorder=6)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="white", linewidth=0.5, alpha=0.35)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _plot_traj(ax: plt.Axes, traj: np.ndarray, *, linestyle: str, label_prefix: str, alpha: float = 0.95) -> None:
    frame_count, agent_count, _ = traj.shape
    stride = max(1, frame_count // 650)
    for agent_idx in range(agent_count):
        color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]
        xy = traj[::stride, agent_idx, :2]
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            linestyle=linestyle,
            linewidth=2.0,
            color=color,
            alpha=alpha,
            label=f"{label_prefix} agent{agent_idx}",
            zorder=3,
        )
        ax.scatter(xy[-1, 0], xy[-1, 1], marker="s", s=34, color=color, edgecolor="black", linewidth=0.5, zorder=7)


def _metrics_text(name: str, ep: Dict[str, Any], traj: np.ndarray) -> str:
    finals = ep.get("agent_final_goal_distances") or []
    finals_txt = ", ".join(f"a{i}={float(v):.2f}" for i, v in enumerate(finals[:3]))
    return (
        f"{name}\n"
        f"success={int(ep.get('team_success', ep.get('success', 0)) or 0)}  "
        f"steps={int(ep.get('steps', max(traj.shape[0] - 1, 0)) or 0)}\n"
        f"final_team={float(ep.get('final_goal_distance', np.nan)):.2f}\n"
        f"{finals_txt}"
    )


def _annotate(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#777777", "alpha": 0.86},
        zorder=10,
    )


def build_figure(
    python_results: Path,
    gazebo_results: Path,
    scenario_json: Path,
    output: Path,
) -> Path:
    python_data = _load_json(python_results)
    gazebo_data = _load_json(gazebo_results)
    scenario = _load_json(scenario_json)
    py_ep = _episode(python_data)
    gz_ep = _episode(gazebo_data)
    py_traj = _trajectory(py_ep)
    gz_traj = _trajectory(gz_ep)
    terrain, map_size = _load_terrain(scenario, scenario_json)
    limits = _axis_limits(py_traj, gz_traj, map_size=map_size)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.4), constrained_layout=True)
    titles = ["Python-only trajectory", "Gazebo-authoritative trajectory", "Overlay comparison"]
    for ax, title in zip(axes, titles):
        _draw_scene(ax, scenario, terrain, map_size, limits)
        ax.set_title(title)

    _plot_traj(axes[0], py_traj, linestyle="-", label_prefix="Python")
    _annotate(axes[0], _metrics_text("Python-only", py_ep, py_traj))

    _plot_traj(axes[1], gz_traj, linestyle="-", label_prefix="Gazebo")
    _annotate(axes[1], _metrics_text("Gazebo-live", gz_ep, gz_traj))

    _plot_traj(axes[2], py_traj, linestyle="--", label_prefix="Python", alpha=0.72)
    _plot_traj(axes[2], gz_traj, linestyle="-", label_prefix="Gazebo", alpha=0.95)
    axes[2].legend(loc="lower right", fontsize=8, framealpha=0.88)

    fig.suptitle(
        "MATD3 Python vs Gazebo trajectory comparison | terrain_seed=88 variant=387032 obstacle_seed=12088",
        fontsize=13,
    )
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-results", required=True, type=Path)
    parser.add_argument("--gazebo-results", required=True, type=Path)
    parser.add_argument("--scenario-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    path = build_figure(
        python_results=args.python_results,
        gazebo_results=args.gazebo_results,
        scenario_json=args.scenario_json,
        output=args.output,
    )
    print(path)


if __name__ == "__main__":
    main()
