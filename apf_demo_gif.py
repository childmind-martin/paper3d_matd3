#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APF demo that uses the same ContinuousPotentialFieldCorrector as training.
Generates frames (top-down + altitude subplot) and stitches them into a GIF.

Usage:
  python apf_demo_gif.py --frames-dir apf_frames --gif apf_demo.gif --steps 200 --fps 12
"""

import argparse
import math
import os
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    import imageio.v2 as imageio  # noqa: E402
except Exception as e:  # pragma: no cover - import guard
    raise SystemExit(f"imageio is required (pip install imageio). Error: {e}")

from potential_field_corrector import ContinuousPotentialFieldCorrector


def find_repo_terrain() -> Optional[np.ndarray]:
    """Try to load a saved terrain from saved_terrains/*.npy; return None if missing."""
    terrain_dir = Path("saved_terrains")
    if not terrain_dir.exists():
        return None
    npy_files = sorted(terrain_dir.glob("*.npy"))
    for f in npy_files:
        try:
            arr = np.load(f)
            if arr.ndim == 2:
                return arr.astype(np.float32)
        except Exception:
            continue
    return None


def generate_heightmap(map_size: int, seed: int) -> np.ndarray:
    """Fallback synthetic terrain."""
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.linspace(-1, 1, map_size), np.linspace(-1, 1, map_size))
    hills = (
        2.8 * np.exp(-6 * ((xx + 0.25) ** 2 + (yy + 0.1) ** 2))
        + 2.0 * np.exp(-10 * ((xx - 0.25) ** 2 + (yy - 0.25) ** 2))
    )
    noise = 0.12 * rng.standard_normal((map_size, map_size))
    terrain = hills + noise
    terrain -= terrain.min() - 0.2  # shift positive
    return terrain.astype(np.float32)


def load_or_generate_terrain(map_size: int, seed: int) -> Tuple[np.ndarray, int]:
    terrain = find_repo_terrain()
    if terrain is not None:
        return terrain, terrain.shape[0]
    generated = generate_heightmap(map_size, seed)
    return generated, generated.shape[0]


def pick_start_goal(terrain: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    size = terrain.shape[0]
    margin = max(5, int(size * 0.05))

    def sample_point() -> np.ndarray:
        x = rng.integers(margin, size - margin)
        y = rng.integers(margin, size - margin)
        z = float(terrain[int(y), int(x)]) + rng.uniform(1.5, 3.0)
        return np.array([float(x), float(y), z], dtype=np.float32)

    start = sample_point()
    goal = sample_point()
    for _ in range(50):
        goal = sample_point()
        if np.linalg.norm(goal[:2] - start[:2]) > size * 0.35:
            break
    return start, goal


def simulate(args):
    frames_dir = Path(args.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    terrain, map_size = load_or_generate_terrain(args.map_size, args.seed)
    rng = np.random.default_rng(args.seed + 5)

    corrector = ContinuousPotentialFieldCorrector(
        terrain_data=terrain,
        goal_attraction=5.0,
        lambda_1_base=6.0,
        terrain_repulsion=60.0,
        agent_repulsion=1.0,
        influence_range=15.0,
        max_force_magnitude=8.8,
        sphere_detection_radius=10.0,
        min_height_above_terrain=1.2,
        terrain_safety_margin=1.5,
        debug_mode=False,
    )

    # Obstacles aligned with corrector logic (same format as training).
    corrector.obstacles = [
        {"center": [map_size * 0.42, map_size * 0.6, 0.0], "radius": 7.0},
        {"center": [map_size * 0.7, map_size * 0.32, 0.0], "radius": 6.0},
    ]

    pos, goal = pick_start_goal(terrain, rng)
    vel = np.zeros(3, dtype=np.float32)
    fr = float(args.action_force_ratio)
    path = [pos.copy()]
    altitude_trace = [pos[2]]
    terrain_trace = [corrector.get_terrain_height(pos[0], pos[1])]

    frames = []
    grid_skip = max(1, map_size // 60)  # downsample for 3D surface speed
    grid_x = np.arange(0, map_size, grid_skip)
    grid_y = np.arange(0, map_size, grid_skip)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    grid_z = terrain[::grid_skip, ::grid_skip]
    for step in range(args.steps):
        # "Policy" action: simple PD toward goal, shaped to [-1,1].
        to_goal = goal - pos
        desired_vel = to_goal * 0.04
        net_action = np.clip(desired_vel, -1.0, 1.0)

        # PF force (same scaling as training path: normalized by max_force_magnitude).
        goal_force = corrector.calculate_goal_attraction_force(pos, goal)
        terrain_force = corrector.calculate_terrain_forces_sphere(pos, goal)
        pf_force = goal_force + terrain_force  # no other agents here
        pf_action = np.clip(pf_force / max(corrector.max_force_magnitude, 1e-6), -1.0, 1.0)

        corrected_action = (1.0 - fr) * net_action + fr * pf_action
        corrected_action = np.clip(corrected_action, -1.0, 1.0)

        # Integrate with mild damping.
        vel = 0.90 * vel + 0.60 * corrected_action
        speed = np.linalg.norm(vel)
        if speed > 5.0:
            vel *= 5.0 / speed
        pos = pos + vel * 0.5

        # Clamp above terrain.
        terrain_h = corrector.get_terrain_height(pos[0], pos[1])
        pos[2] = max(terrain_h + 1.0, pos[2])

        path.append(pos.copy())
        altitude_trace.append(pos[2])
        terrain_trace.append(terrain_h)

        # Plot frame (top-down + altitude)
        fig = plt.figure(figsize=(10, 5), dpi=130)
        ax_xy = fig.add_subplot(1, 2, 1)
        ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
        extent = [0, map_size, 0, map_size]
        im = ax_xy.imshow(terrain, origin="lower", extent=extent, cmap="terrain")
        fig.colorbar(im, ax=ax_xy, fraction=0.046, pad=0.04, label="height")

        for ob in corrector.obstacles:
            circ = plt.Circle(
                (ob["center"][0], ob["center"][1]), ob["radius"], color="red", alpha=0.35
            )
            ax_xy.add_patch(circ)

        path_np = np.array(path)
        ax_xy.plot(path_np[:, 0], path_np[:, 1], color="blue", linewidth=2, label="trajectory")
        ax_xy.scatter(goal[0], goal[1], color="gold", marker="*", s=140, edgecolor="k", label="goal")
        ax_xy.scatter(pos[0], pos[1], color="black", s=28, label="agent")

        # Force arrows: PF action (orange) vs policy action (cyan)
        arrow_scale = 6.0
        ax_xy.arrow(
            pos[0],
            pos[1],
            pf_action[0] * arrow_scale,
            pf_action[1] * arrow_scale,
            head_width=1.4,
            head_length=2.2,
            fc="orange",
            ec="orange",
            alpha=0.9,
            length_includes_head=True,
            label="pf_action",
        )
        ax_xy.arrow(
            pos[0],
            pos[1],
            net_action[0] * arrow_scale,
            net_action[1] * arrow_scale,
            head_width=1.4,
            head_length=2.2,
            fc="cyan",
            ec="cyan",
            alpha=0.8,
            length_includes_head=True,
            label="policy_action",
        )

        ax_xy.set_xlim(0, map_size)
        ax_xy.set_ylim(0, map_size)
        ax_xy.set_title(f"APF step {step+1}/{args.steps}")
        ax_xy.set_xlabel("X")
        ax_xy.set_ylabel("Y")
        ax_xy.legend(loc="upper left")
        ax_xy.set_aspect("equal")

        # 3D terrain + path
        ax_3d.plot_surface(
            grid_xx,
            grid_yy,
            grid_z,
            cmap="terrain",
            linewidth=0,
            antialiased=False,
            alpha=0.85,
            rstride=1,
            cstride=1,
        )
        path_np = np.array(path)
        ax_3d.plot(
            path_np[:, 0],
            path_np[:, 1],
            path_np[:, 2],
            color="blue",
            linewidth=2.0,
            label="trajectory_3d",
        )
        ax_3d.scatter(goal[0], goal[1], goal[2], color="gold", marker="*", s=120, edgecolor="k", label="goal")
        ax_3d.scatter(pos[0], pos[1], pos[2], color="black", s=30, label="agent")
        ax_3d.set_xlim(0, map_size)
        ax_3d.set_ylim(0, map_size)
        z_max = max(float(grid_z.max()), float(max(altitude_trace) + 1.0))
        ax_3d.set_zlim(0, z_max)
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.view_init(elev=35, azim=-45)
        ax_3d.legend(loc="upper right")

        fig.tight_layout()
        frame_path = frames_dir / f"frame_{step:04d}.png"
        # 保持尺寸一致，避免 mimsave 时报 shape 不一致
        fig.savefig(frame_path)
        plt.close(fig)
        frames.append(frame_path)

    return frames


def frames_to_gif(frames, gif_path, fps):
    images = []
    for p in frames:
        try:
            images.append(imageio.imread(p))
        except Exception as e:
            print(f"Skip frame {p}: {e}")
    if not images:
        raise RuntimeError("No frames to write into GIF.")
    duration = 1.0 / max(fps, 1e-3)
    imageio.mimsave(gif_path, images, duration=duration)
    print(f"Wrote GIF: {gif_path} ({len(images)} frames @ {fps} fps)")


def parse_args():
    parser = argparse.ArgumentParser(description="APF demo GIF generator (uses repo corrector).")
    parser.add_argument("--frames-dir", type=str, default="apf_frames", help="Directory to store frames.")
    parser.add_argument("--gif", type=str, default="apf_demo.gif", help="Output GIF path.")
    parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps.")
    parser.add_argument("--fps", type=float, default=12.0, help="GIF frame rate.")
    parser.add_argument("--map-size", type=int, default=120, help="Generated terrain size if no saved terrain found.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--action-force-ratio",
        type=float,
        default=0.7,
        help="Mixing ratio between policy action and APF action (same semantics as training).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    frames = simulate(args)
    frames_to_gif(frames, args.gif, args.fps)


if __name__ == "__main__":
    main()
