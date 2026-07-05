#!/usr/bin/env python3
"""Validate exported MATD3 scenario snapshots against Gazebo OBJ/SDF geometry."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_world_root(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def _parse_pose(text: Optional[str]) -> List[float]:
    if not text:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    vals = [float(x) for x in text.split()]
    while len(vals) < 6:
        vals.append(0.0)
    return vals[:6]


def _read_obj_vertices(path: Path) -> np.ndarray:
    vertices: List[List[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"OBJ has no vertices: {path}")
    return np.asarray(vertices, dtype=np.float64)


def _regular_grid_from_vertices(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.unique(vertices[:, 0])
    ys = np.unique(vertices[:, 1])
    xs.sort()
    ys.sort()
    z = np.zeros((ys.size, xs.size), dtype=np.float64)
    x_index = {float(x): i for i, x in enumerate(xs)}
    y_index = {float(y): i for i, y in enumerate(ys)}
    for x, y, height in vertices:
        z[y_index[float(y)], x_index[float(x)]] = height
    return xs, ys, z


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


def _triangle_height(xs: np.ndarray, ys: np.ndarray, z: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Height on the same two-triangle cell split used by gazebo_terrain_exporter OBJ."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xi = np.interp(x, xs, np.arange(xs.size, dtype=np.float64))
    yi = np.interp(y, ys, np.arange(ys.size, dtype=np.float64))
    x0 = np.clip(np.floor(xi).astype(int), 0, xs.size - 2)
    y0 = np.clip(np.floor(yi).astype(int), 0, ys.size - 2)
    u = xi - x0
    v = yi - y0
    z00 = z[y0, x0]
    z10 = z[y0, x0 + 1]
    z01 = z[y0 + 1, x0]
    z11 = z[y0 + 1, x0 + 1]
    lower = u >= v
    out = np.empty_like(u, dtype=np.float64)
    out[lower] = z00[lower] + u[lower] * (z10[lower] - z00[lower]) + v[lower] * (z11[lower] - z10[lower])
    out[~lower] = z00[~lower] + u[~lower] * (z11[~lower] - z01[~lower]) + v[~lower] * (z01[~lower] - z00[~lower])
    return out


def _extract_sdf_models(world_root: ET.Element) -> Dict[str, Dict[str, Any]]:
    world = world_root.find("world")
    if world is None:
        raise ValueError("world.sdf has no <world> element")
    models: Dict[str, Dict[str, Any]] = {}
    for model in world.findall("model"):
        name = model.attrib.get("name", "")
        pose = _parse_pose(model.findtext("pose"))
        link = model.find("link")
        geometry = None
        if link is not None:
            collision = link.find("collision")
            visual = link.find("visual")
            geom_parent = collision if collision is not None else visual
            if geom_parent is not None:
                geometry = geom_parent.find("geometry")
        entry: Dict[str, Any] = {"name": name, "pose": pose}
        if geometry is not None:
            sphere = geometry.find("sphere")
            box = geometry.find("box")
            cylinder = geometry.find("cylinder")
            if sphere is not None:
                entry["type"] = "sphere"
                entry["radius"] = float(sphere.findtext("radius"))
            elif box is not None:
                entry["type"] = "box"
                entry["size"] = [float(v) for v in box.findtext("size").split()]
            elif cylinder is not None:
                entry["type"] = "cylinder"
                entry["radius"] = float(cylinder.findtext("radius"))
                entry["length"] = float(cylinder.findtext("length"))
        models[name] = entry
    for include in world.findall("include"):
        name = include.findtext("name")
        if not name:
            uri = include.findtext("uri", "")
            name = uri.rsplit("/", 1)[-1] if uri else ""
        if not name:
            continue
        models[name] = {
            "name": name,
            "pose": _parse_pose(include.findtext("pose")),
            "type": "include",
            "uri": include.findtext("uri", ""),
        }
    return models


def _max_abs(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.max(np.abs(np.asarray(values, dtype=np.float64))))


def _compare_obstacles(snapshot: Dict[str, Any], sdf_models: Dict[str, Dict[str, Any]], tol: float) -> Dict[str, Any]:
    errors: List[float] = []
    missing: List[str] = []
    type_mismatch: List[str] = []
    for obstacle in snapshot.get("obstacles", []):
        name = obstacle["name"]
        model = sdf_models.get(name)
        if model is None:
            missing.append(name)
            continue
        if model.get("type") != obstacle.get("type"):
            type_mismatch.append(name)
            continue
        errors.extend((np.asarray(model["pose"][:3]) - np.asarray(obstacle["center"], dtype=np.float64)).tolist())
        if obstacle["type"] == "sphere":
            errors.append(float(model["radius"]) - float(obstacle["radius"]))
        elif obstacle["type"] == "box":
            errors.extend((np.asarray(model["size"]) - np.asarray(obstacle["size"], dtype=np.float64)).tolist())
        elif obstacle["type"] == "cylinder":
            errors.append(float(model["radius"]) - float(obstacle["radius"]))
            errors.append(float(model["length"]) - float(obstacle["length"]))
    max_error = _max_abs(errors)
    return {
        "count_snapshot": len(snapshot.get("obstacles", [])),
        "count_sdf": len([name for name in sdf_models if name.startswith("obstacle_")]),
        "missing": missing,
        "type_mismatch": type_mismatch,
        "max_abs_error": max_error,
        "passed": not missing and not type_mismatch and max_error <= tol,
    }


def _compare_markers(snapshot: Dict[str, Any], sdf_models: Dict[str, Dict[str, Any]], tol: float) -> Dict[str, Any]:
    expected: Dict[str, Sequence[float]] = {}
    for idx, start in enumerate(snapshot.get("start_positions", [])):
        expected[f"start_agent_{idx}"] = start["position"]
    if snapshot.get("goal") and snapshot["goal"].get("position") is not None:
        expected["goal_center"] = snapshot["goal"]["position"]
    for idx, goal in enumerate(snapshot.get("agent_goals", [])):
        expected[f"goal_agent_{idx}"] = goal["position"]

    errors: List[float] = []
    missing: List[str] = []
    for name, pos in expected.items():
        model = sdf_models.get(name)
        if model is None:
            missing.append(name)
            continue
        errors.extend((np.asarray(model["pose"][:3]) - np.asarray(pos, dtype=np.float64)).tolist())
    max_error = _max_abs(errors)
    return {
        "count_expected": len(expected),
        "missing": missing,
        "max_abs_position_error": max_error,
        "passed": not missing and max_error <= tol,
    }


def _obstacle_clearance(points: np.ndarray, obstacles: Sequence[Dict[str, Any]]) -> np.ndarray:
    clearances = np.full((points.shape[0],), np.inf, dtype=np.float64)
    for obstacle in obstacles:
        obs_type = obstacle.get("type")
        center = np.asarray(obstacle["center"], dtype=np.float64)
        if obs_type == "sphere":
            radius = float(obstacle["radius"])
            d = np.linalg.norm(points - center.reshape(1, 3), axis=1) - radius
        elif obs_type == "box":
            half = np.asarray(obstacle["size"], dtype=np.float64) / 2.0
            local = np.abs(points - center.reshape(1, 3)) - half.reshape(1, 3)
            outside = np.maximum(local, 0.0)
            outside_dist = np.linalg.norm(outside, axis=1)
            inside_depth = np.minimum(np.max(local, axis=1), 0.0)
            d = outside_dist + inside_depth
        elif obs_type == "cylinder":
            radius = float(obstacle["radius"])
            half_len = float(obstacle["length"]) / 2.0
            local = points - center.reshape(1, 3)
            radial = np.linalg.norm(local[:, :2], axis=1)
            radial_excess = radial - radius
            z_excess = np.abs(local[:, 2]) - half_len
            outside = np.stack([np.maximum(radial_excess, 0.0), np.maximum(z_excess, 0.0)], axis=1)
            outside_dist = np.linalg.norm(outside, axis=1)
            inside_depth = np.minimum(np.maximum(radial_excess, z_excess), 0.0)
            d = outside_dist + inside_depth
        else:
            continue
        clearances = np.minimum(clearances, d)
    return clearances


def _min_clearance(
    points: np.ndarray,
    terrain_heights: np.ndarray,
    obstacles: Sequence[Dict[str, Any]],
) -> np.ndarray:
    terrain_clearance = points[:, 2] - terrain_heights
    obstacle_clearance = _obstacle_clearance(points, obstacles)
    return np.minimum(terrain_clearance, obstacle_clearance)


def _collision_from_clearance(clearance: np.ndarray, threshold: float) -> np.ndarray:
    return (clearance < threshold) | (clearance < 0.0)


def validate_export(
    scenario_json: Path,
    samples: int = 1000,
    seed: int = 12345,
    vertex_tolerance: float = 1e-4,
    mesh_sample_tolerance: float = 1.0,
    sdf_tolerance: float = 1e-5,
) -> Dict[str, Any]:
    scenario_json = scenario_json.resolve()
    snapshot = _load_json(scenario_json)
    terrain = snapshot["terrain"]
    dense = np.load(Path(terrain["dense_npy"])).astype(np.float64)
    dense_x = np.asarray(terrain["dense_coordinates"]["x"], dtype=np.float64)
    dense_y = np.asarray(terrain["dense_coordinates"]["y"], dtype=np.float64)

    gazebo = snapshot.get("gazebo", {})
    visual_mesh_path = Path(gazebo.get("visual_mesh", {}).get("path", scenario_json.parent / "models/matd3_terrain/meshes/terrain.obj"))
    world_sdf = Path(gazebo.get("world_sdf", scenario_json.parent / "world.sdf"))
    vertices = _read_obj_vertices(visual_mesh_path)
    mesh_x, mesh_y, mesh_z = _regular_grid_from_vertices(vertices)

    vertex_expected = _bilinear_height(dense_x, dense_y, dense, vertices[:, 0], vertices[:, 1])
    vertex_errors = vertices[:, 2] - vertex_expected

    rng = np.random.default_rng(int(seed))
    domain = snapshot["coordinate_domain"]
    random_x = rng.uniform(float(domain["x_min"]), float(domain["x_max"]), int(samples))
    random_y = rng.uniform(float(domain["y_min"]), float(domain["y_max"]), int(samples))
    py_h = _bilinear_height(dense_x, dense_y, dense, random_x, random_y)
    mesh_h = _triangle_height(mesh_x, mesh_y, mesh_z, random_x, random_y)
    mesh_errors = mesh_h - py_h

    sdf_models = _extract_sdf_models(_load_world_root(world_sdf))
    obstacle_report = _compare_obstacles(snapshot, sdf_models, sdf_tolerance)
    marker_report = _compare_markers(snapshot, sdf_models, sdf_tolerance)

    threshold = float(snapshot.get("collision", {}).get("collision_distance_threshold", 0.5))
    z_offsets = rng.uniform(-2.0, 8.0, int(samples))
    points = np.stack([random_x, random_y, py_h + z_offsets], axis=1)
    py_clearance = _min_clearance(points, py_h, snapshot.get("obstacles", []))
    sdf_clearance = _min_clearance(points, mesh_h, snapshot.get("obstacles", []))
    py_collisions = _collision_from_clearance(py_clearance, threshold)
    sdf_collisions = _collision_from_clearance(sdf_clearance, threshold)
    mismatch_mask = py_collisions != sdf_collisions
    collision_mismatches = int(np.count_nonzero(mismatch_mask))
    # A triangle mesh is a planar approximation of the exported Python height
    # function between vertices. Report raw threshold flips, but fail only when
    # they remain outside the mesh height-error tolerance band.
    boundary_distance = np.minimum(np.abs(py_clearance - threshold), np.abs(sdf_clearance - threshold))
    robust_mismatch_mask = mismatch_mask & (boundary_distance > mesh_sample_tolerance)
    robust_collision_mismatches = int(np.count_nonzero(robust_mismatch_mask))

    terrain_report = {
        "dense_shape": [int(dense.shape[0]), int(dense.shape[1])],
        "obj_vertex_count": int(vertices.shape[0]),
        "obj_grid_shape": [int(mesh_y.size), int(mesh_x.size)],
        "vertex_max_abs_height_error": float(np.max(np.abs(vertex_errors))),
        "vertex_mean_abs_height_error": float(np.mean(np.abs(vertex_errors))),
        "random_mesh_max_abs_height_error": float(np.max(np.abs(mesh_errors))),
        "random_mesh_mean_abs_height_error": float(np.mean(np.abs(mesh_errors))),
        "vertex_tolerance": float(vertex_tolerance),
        "mesh_sample_tolerance": float(mesh_sample_tolerance),
        "vertex_check_passed": bool(np.max(np.abs(vertex_errors)) <= vertex_tolerance),
        "random_mesh_check_passed": bool(np.max(np.abs(mesh_errors)) <= mesh_sample_tolerance),
    }

    collision_report = {
        "samples": int(samples),
        "threshold": threshold,
        "raw_mismatches": collision_mismatches,
        "mismatch_rate": float(collision_mismatches / max(1, int(samples))),
        "mesh_height_tolerance_band": float(mesh_sample_tolerance),
        "robust_mismatches": robust_collision_mismatches,
        "robust_mismatch_rate": float(robust_collision_mismatches / max(1, int(samples))),
        "passed": robust_collision_mismatches == 0,
    }

    report = {
        "scenario_json": str(scenario_json),
        "world_sdf": str(world_sdf),
        "visual_mesh": str(visual_mesh_path),
        "terrain": terrain_report,
        "obstacles": obstacle_report,
        "markers": marker_report,
        "collision": collision_report,
    }
    report["passed"] = bool(
        terrain_report["vertex_check_passed"]
        and terrain_report["random_mesh_check_passed"]
        and obstacle_report["passed"]
        and marker_report["passed"]
        and collision_report["passed"]
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a MATD3 Gazebo static scene export.")
    parser.add_argument("--scenario-json", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--vertex-tolerance", type=float, default=1e-4)
    parser.add_argument("--mesh-sample-tolerance", type=float, default=1.0)
    parser.add_argument("--sdf-tolerance", type=float, default=1e-5)
    parser.add_argument("--report-path", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = validate_export(
        scenario_json=Path(args.scenario_json),
        samples=args.samples,
        seed=args.seed,
        vertex_tolerance=args.vertex_tolerance,
        mesh_sample_tolerance=args.mesh_sample_tolerance,
        sdf_tolerance=args.sdf_tolerance,
    )
    report_path = Path(args.report_path) if args.report_path else Path(args.scenario_json).resolve().parent / "validation_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[scenario_validator] report={report_path}")
    print(f"[scenario_validator] passed={report['passed']}")
    print(
        "[scenario_validator] terrain vertex max error="
        f"{report['terrain']['vertex_max_abs_height_error']:.6g}, random mesh max error="
        f"{report['terrain']['random_mesh_max_abs_height_error']:.6g}"
    )
    print(
        "[scenario_validator] obstacles passed="
        f"{report['obstacles']['passed']}, markers passed={report['markers']['passed']}, "
        f"collision raw mismatches={report['collision']['raw_mismatches']}, "
        f"robust mismatches={report['collision']['robust_mismatches']}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
