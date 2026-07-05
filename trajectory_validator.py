#!/usr/bin/env python3
"""Validate MATD3 trajectory snapshots against exported Gazebo replay geometry."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _trajectory_array(snapshot: Dict[str, Any]) -> np.ndarray:
    npz_path = snapshot.get("trajectory_npz_path") or snapshot.get("export_paths", {}).get("trajectory_npz")
    if npz_path and Path(npz_path).exists():
        with np.load(Path(npz_path)) as data:
            key = "trajectory" if "trajectory" in data.files else data.files[0]
            arr = data[key].astype(np.float64)
    else:
        npy_path = snapshot.get("trajectory_npy_path") or snapshot.get("export_paths", {}).get("trajectory_npy")
        if npy_path and Path(npy_path).exists():
            arr = np.load(Path(npy_path)).astype(np.float64)
        else:
            arr = np.asarray(snapshot.get("trajectory", []), dtype=np.float64)
    if arr.ndim != 3 or arr.shape[0] < 1 or arr.shape[1] < 1 or arr.shape[2] < 3:
        raise ValueError("trajectory snapshot must have shape [frame][agent][xyz]")
    arr = arr[:, :, :3]
    if not np.all(np.isfinite(arr)):
        raise ValueError("trajectory contains non-finite positions")
    return arr


def _load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path) as data:
            key = "trajectory" if "trajectory" in data.files else data.files[0]
            return data[key].astype(np.float64)
    return np.load(path).astype(np.float64)


def _bilinear_height(xs: np.ndarray, ys: np.ndarray, z: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xi = np.interp(x, xs, np.arange(xs.size, dtype=np.float64))
    yi = np.interp(y, ys, np.arange(ys.size, dtype=np.float64))
    x0 = np.clip(np.floor(xi).astype(int), 0, xs.size - 1)
    y0 = np.clip(np.floor(yi).astype(int), 0, ys.size - 1)
    x1 = np.clip(x0 + 1, 0, xs.size - 1)
    y1 = np.clip(y0 + 1, 0, ys.size - 1)
    wx = xi - x0
    wy = yi - y0
    z00 = z[y0, x0]
    z10 = z[y0, x1]
    z01 = z[y1, x0]
    z11 = z[y1, x1]
    return (1.0 - wx) * (1.0 - wy) * z00 + wx * (1.0 - wy) * z10 + (1.0 - wx) * wy * z01 + wx * wy * z11


def _sphere_clearance(points: np.ndarray, obstacle: Dict[str, Any]) -> np.ndarray:
    center = np.asarray(obstacle["center"], dtype=np.float64)
    return np.linalg.norm(points - center.reshape(1, 3), axis=1) - float(obstacle["radius"])


def _box_clearance(points: np.ndarray, obstacle: Dict[str, Any]) -> np.ndarray:
    center = np.asarray(obstacle["center"], dtype=np.float64)
    half = np.asarray(obstacle["size"], dtype=np.float64) / 2.0
    local = np.abs(points - center.reshape(1, 3)) - half.reshape(1, 3)
    outside = np.maximum(local, 0.0)
    outside_dist = np.linalg.norm(outside, axis=1)
    inside_depth = np.minimum(np.max(local, axis=1), 0.0)
    return outside_dist + inside_depth


def _cylinder_clearance(points: np.ndarray, obstacle: Dict[str, Any]) -> np.ndarray:
    center = np.asarray(obstacle["center"], dtype=np.float64)
    radius = float(obstacle["radius"])
    half_len = float(obstacle["length"]) / 2.0
    local = points - center.reshape(1, 3)
    radial = np.linalg.norm(local[:, :2], axis=1)
    radial_excess = radial - radius
    z_excess = np.abs(local[:, 2]) - half_len
    outside = np.stack([np.maximum(radial_excess, 0.0), np.maximum(z_excess, 0.0)], axis=1)
    outside_dist = np.linalg.norm(outside, axis=1)
    inside_depth = np.minimum(np.maximum(radial_excess, z_excess), 0.0)
    return outside_dist + inside_depth


def _obstacle_clearance(points: np.ndarray, obstacles: Sequence[Dict[str, Any]]) -> np.ndarray:
    clearances = np.full((points.shape[0],), np.inf, dtype=np.float64)
    for obstacle in obstacles:
        obs_type = obstacle.get("type")
        if obs_type == "sphere":
            cur = _sphere_clearance(points, obstacle)
        elif obs_type == "box":
            cur = _box_clearance(points, obstacle)
        elif obs_type == "cylinder":
            cur = _cylinder_clearance(points, obstacle)
        else:
            continue
        clearances = np.minimum(clearances, cur)
    return clearances


def _world_has_replay_include(world_sdf: Path) -> bool:
    root = ET.parse(world_sdf).getroot()
    world = root.find("world")
    if world is None:
        return False
    for include in world.findall("include"):
        if include.findtext("name") == "matd3_trajectory_replay":
            return True
        if include.findtext("uri") == "model://matd3_trajectory_replay":
            return True
    return False


def _path_stride_indices(frame_count: int, stride: int) -> List[int]:
    indices = list(range(0, int(frame_count), max(1, int(stride))))
    if indices[-1] != int(frame_count) - 1:
        indices.append(int(frame_count) - 1)
    return indices


def validate_trajectory_replay(
    scenario_json: Path,
    trajectory_json: Path,
    start_tolerance: float = 1e-4,
    centerline_tolerance: float = 1e-4,
) -> Dict[str, Any]:
    scenario_json = Path(scenario_json).resolve()
    trajectory_json = Path(trajectory_json).resolve()
    scenario = _load_json(scenario_json)
    trajectory_snapshot = _load_json(trajectory_json)
    trajectory = _trajectory_array(trajectory_snapshot)

    terrain = scenario["terrain"]
    dense = np.load(Path(terrain["dense_npy"])).astype(np.float64)
    dense_x = np.asarray(terrain["dense_coordinates"]["x"], dtype=np.float64)
    dense_y = np.asarray(terrain["dense_coordinates"]["y"], dtype=np.float64)

    flat_points = trajectory.reshape(-1, 3)
    terrain_h = _bilinear_height(dense_x, dense_y, dense, flat_points[:, 0], flat_points[:, 1])
    terrain_clearance = flat_points[:, 2] - terrain_h
    obstacle_clearance = _obstacle_clearance(flat_points, scenario.get("obstacles", []))
    min_clearance = np.minimum(terrain_clearance, obstacle_clearance)
    threshold = float(scenario.get("collision", {}).get("collision_distance_threshold", trajectory_snapshot.get("collision_distance_threshold", 0.5)))
    collision_mask = (min_clearance < threshold) | (min_clearance < 0.0)

    starts = np.asarray([item["position"] for item in scenario.get("start_positions", [])], dtype=np.float64)
    start_errors: List[float] = []
    if starts.shape == trajectory[0].shape:
        start_errors = np.linalg.norm(trajectory[0] - starts, axis=1).tolist()
        start_passed = bool(np.max(start_errors) <= start_tolerance) if start_errors else False
    else:
        start_passed = False

    replay = trajectory_snapshot.get("gazebo_replay", {})
    centerline_reports = []
    centerline_passed = True
    for meta in replay.get("agent_paths", []):
        agent_idx = int(meta["agent_index"])
        centerline_path = Path(meta["centerline_npy"])
        path_stride = int(meta.get("path_stride", replay.get("path_stride", 1)))
        exported = _load_array(centerline_path)
        expected = trajectory[_path_stride_indices(trajectory.shape[0], path_stride), agent_idx, :]
        # The exporter also removes exact duplicate consecutive points.
        deduped = [expected[0]]
        for point in expected[1:]:
            if np.linalg.norm(point - deduped[-1]) > 1e-8:
                deduped.append(point)
        expected = np.asarray(deduped, dtype=np.float64)
        if exported.shape == expected.shape:
            errors = np.linalg.norm(exported - expected, axis=1)
            max_error = float(np.max(errors)) if errors.size else 0.0
        else:
            max_error = float("inf")
        passed = bool(np.isfinite(max_error) and max_error <= centerline_tolerance)
        centerline_passed = centerline_passed and passed
        centerline_reports.append(
            {
                "agent_index": agent_idx,
                "exported_shape": list(exported.shape),
                "expected_shape": list(expected.shape),
                "max_position_error": max_error,
                "passed": passed,
            }
        )

    world_replay_sdf = replay.get("world_replay_sdf")
    world_include_passed = bool(world_replay_sdf and Path(world_replay_sdf).exists() and _world_has_replay_include(Path(world_replay_sdf)))

    report = {
        "scenario_json": str(scenario_json),
        "trajectory_json": str(trajectory_json),
        "frame_count": int(trajectory.shape[0]),
        "agent_count": int(trajectory.shape[1]),
        "start_check": {
            "start_tolerance": float(start_tolerance),
            "errors": start_errors,
            "passed": start_passed,
        },
        "clearance": {
            "collision_distance_threshold": threshold,
            "min_terrain_clearance": float(np.min(terrain_clearance)),
            "min_obstacle_clearance": float(np.min(obstacle_clearance)) if np.any(np.isfinite(obstacle_clearance)) else None,
            "min_clearance": float(np.min(min_clearance)),
            "collision_sample_count": int(np.count_nonzero(collision_mask)),
            "collision_sample_rate": float(np.count_nonzero(collision_mask) / max(1, flat_points.shape[0])),
        },
        "gazebo_replay": {
            "world_replay_sdf": world_replay_sdf,
            "world_include_passed": world_include_passed,
            "centerlines": centerline_reports,
            "centerline_passed": centerline_passed,
        },
        "passed": bool(start_passed and centerline_passed and world_include_passed),
    }
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a MATD3 Gazebo trajectory replay export.")
    parser.add_argument("--scenario-json", required=True)
    parser.add_argument("--trajectory-json", required=True)
    parser.add_argument("--start-tolerance", type=float, default=1e-4)
    parser.add_argument("--centerline-tolerance", type=float, default=1e-4)
    parser.add_argument("--report-path", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = validate_trajectory_replay(
        scenario_json=Path(args.scenario_json),
        trajectory_json=Path(args.trajectory_json),
        start_tolerance=args.start_tolerance,
        centerline_tolerance=args.centerline_tolerance,
    )
    report_path = Path(args.report_path) if args.report_path else Path(args.trajectory_json).resolve().parent / "trajectory_validation_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[trajectory_validator] report={report_path}")
    print(
        "[trajectory_validator] passed="
        f"{report['passed']}, start passed={report['start_check']['passed']}, "
        f"centerline passed={report['gazebo_replay']['centerline_passed']}, "
        f"world include passed={report['gazebo_replay']['world_include_passed']}, "
        f"collision samples={report['clearance']['collision_sample_count']}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
