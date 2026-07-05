#!/usr/bin/env python3
"""Write MATD3 trajectory snapshots and scene snapshots for Gazebo replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scenario_exporter import _as_float_list, _json_safe, normalize_obstacle
except Exception:  # pragma: no cover - fallback for standalone use
    def _json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return _json_safe(value.tolist())
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(v) for v in value]
        return str(value)

    def _as_float_list(value: Any, length: Optional[int] = None) -> Optional[List[float]]:
        try:
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if length is not None and arr.size < length:
            return None
        if length is not None:
            arr = arr[:length]
        return [float(v) for v in arr.tolist()]

    def normalize_obstacle(raw: Any, index: int) -> Dict[str, Any]:
        if isinstance(raw, dict):
            center = raw.get("center", raw.get("position", raw.get("pos")))
            radius = raw.get("radius", raw.get("r", raw.get("size")))
            c = _as_float_list(center, 3)
            if c is not None and radius is not None:
                return {
                    "name": str(raw.get("name", f"obstacle_{index}")),
                    "type": str(raw.get("type", "sphere")),
                    "type_source": "fallback",
                    "center": c,
                    "radius": float(radius),
                }
        raise ValueError(f"unsupported obstacle at index {index}")


FORMAT_VERSION = 1


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if not raw.strip():
        return bool(default)
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def _array_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(b"|")
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(b"|")
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _load_sidecar(path: Path) -> Optional[Dict[str, Any]]:
    meta_path = _sidecar_path(path)
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None
    return meta if isinstance(meta, dict) else None


def _write_sidecar(path: Path, meta: Dict[str, Any]) -> None:
    meta_path = _sidecar_path(path)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(meta), f, ensure_ascii=False, indent=2)


def _array_artifact_meta(path: Path, array: np.ndarray, storage: str, sha256: str, cache_hit: bool) -> Dict[str, Any]:
    arr = np.asarray(array)
    return {
        "path": str(path),
        "sidecar_meta": str(_sidecar_path(path)),
        "storage": storage,
        "sha256": sha256,
        "shape": [int(v) for v in arr.shape],
        "dtype": str(arr.dtype),
        "cache_hit": bool(cache_hit),
    }


def _save_npy_cached(path: Path, array: np.ndarray, reuse_cache: bool = True) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(array)
    sha256 = _array_sha256(arr)
    base_meta = _array_artifact_meta(path, arr, "npy", sha256, False)
    if reuse_cache and path.exists():
        old_meta = _load_sidecar(path)
        if (
            old_meta
            and old_meta.get("sha256") == sha256
            and old_meta.get("shape") == base_meta["shape"]
            and old_meta.get("dtype") == base_meta["dtype"]
            and old_meta.get("storage") == "npy"
        ):
            meta = dict(old_meta)
            meta["path"] = str(path)
            meta["sidecar_meta"] = str(_sidecar_path(path))
            meta["cache_hit"] = True
            return meta
    np.save(path, arr)
    _write_sidecar(path, {k: v for k, v in base_meta.items() if k != "cache_hit"})
    return base_meta


def _save_npz_cached(path: Path, array_name: str, array: np.ndarray, reuse_cache: bool = True) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(array)
    sha256 = _array_sha256(arr)
    base_meta = _array_artifact_meta(path, arr, "npz", sha256, False)
    base_meta["array_name"] = str(array_name)
    if reuse_cache and path.exists():
        old_meta = _load_sidecar(path)
        if (
            old_meta
            and old_meta.get("sha256") == sha256
            and old_meta.get("shape") == base_meta["shape"]
            and old_meta.get("dtype") == base_meta["dtype"]
            and old_meta.get("storage") == "npz"
            and old_meta.get("array_name") == array_name
        ):
            meta = dict(old_meta)
            meta["path"] = str(path)
            meta["sidecar_meta"] = str(_sidecar_path(path))
            meta["cache_hit"] = True
            return meta
    np.savez_compressed(path, **{array_name: arr})
    _write_sidecar(path, {k: v for k, v in base_meta.items() if k != "cache_hit"})
    return base_meta


def _arrays_bundle_sha256(arrays: Dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for name in sorted(arrays):
        arr = np.ascontiguousarray(arrays[name])
        h.update(str(name).encode("utf-8"))
        h.update(b"|")
        h.update(str(arr.shape).encode("utf-8"))
        h.update(b"|")
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(b"|")
        h.update(arr.tobytes(order="C"))
        h.update(b"\n")
    return h.hexdigest()


def _save_npz_bundle_cached(path: Path, arrays: Dict[str, np.ndarray], reuse_cache: bool = True) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {str(k): np.ascontiguousarray(v) for k, v in arrays.items()}
    sha256 = _arrays_bundle_sha256(normalized)
    base_meta = {
        "path": str(path),
        "sidecar_meta": str(_sidecar_path(path)),
        "storage": "npz",
        "sha256": sha256,
        "arrays": {
            name: {
                "shape": [int(v) for v in arr.shape],
                "dtype": str(arr.dtype),
            }
            for name, arr in sorted(normalized.items())
        },
        "cache_hit": False,
    }
    if reuse_cache and path.exists():
        old_meta = _load_sidecar(path)
        if (
            old_meta
            and old_meta.get("sha256") == sha256
            and old_meta.get("storage") == "npz"
            and old_meta.get("arrays") == base_meta["arrays"]
        ):
            meta = dict(old_meta)
            meta["path"] = str(path)
            meta["sidecar_meta"] = str(_sidecar_path(path))
            meta["cache_hit"] = True
            return meta
    np.savez_compressed(path, **normalized)
    _write_sidecar(path, {k: v for k, v in base_meta.items() if k != "cache_hit"})
    return base_meta


def _float_attr(args: Any, name: str, env_name: str, default: Optional[float]) -> Optional[float]:
    value = getattr(args, name, None) if args is not None else None
    if value is None:
        raw = os.getenv(env_name, "")
        if raw.strip():
            value = raw
    if value is None:
        return default
    try:
        value = float(value)
    except Exception:
        return default
    if not np.isfinite(value):
        return default
    return float(value)


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        return int(default)
    return value if value > 0 else int(default)


def _trajectory_array(trajectory: Any) -> np.ndarray:
    arr = np.asarray(trajectory, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[0] < 1 or arr.shape[1] < 1 or arr.shape[2] < 3:
        raise ValueError("trajectory must have shape [frame][agent][xyz]")
    arr = arr[:, :, :3]
    if not np.all(np.isfinite(arr)):
        raise ValueError("trajectory contains non-finite positions")
    return arr


def _terrain_coordinate_vectors(map_size: float, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    max_coord = float(map_size) - 1.0
    xs = np.linspace(0.0, max_coord, int(width), dtype=np.float64) if width > 1 else np.asarray([0.0], dtype=np.float64)
    ys = np.linspace(0.0, max_coord, int(height), dtype=np.float64) if height > 1 else np.asarray([0.0], dtype=np.float64)
    return xs, ys


def _height_from_grid(terrain: np.ndarray, map_size: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    terrain = np.asarray(terrain, dtype=np.float64)
    if terrain.ndim != 2:
        raise ValueError("terrain must be a 2D height field")
    height, width = terrain.shape
    xs, ys = _terrain_coordinate_vectors(map_size, width, height)
    x = np.clip(np.asarray(x, dtype=np.float64), xs[0], xs[-1])
    y = np.clip(np.asarray(y, dtype=np.float64), ys[0], ys[-1])
    xi = np.interp(x, xs, np.arange(width, dtype=np.float64))
    yi = np.interp(y, ys, np.arange(height, dtype=np.float64))
    x0 = np.clip(np.floor(xi).astype(int), 0, width - 1)
    y0 = np.clip(np.floor(yi).astype(int), 0, height - 1)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    wx = xi - x0
    wy = yi - y0
    z00 = terrain[y0, x0]
    z10 = terrain[y0, x1]
    z01 = terrain[y1, x0]
    z11 = terrain[y1, x1]
    return (1.0 - wx) * (1.0 - wy) * z00 + wx * (1.0 - wy) * z10 + (1.0 - wx) * wy * z01 + wx * wy * z11


def _sample_dense_terrain(terrain: np.ndarray, map_size: float, resolution: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolution = int(resolution)
    if resolution < 2:
        raise ValueError("dense terrain resolution must be at least 2")
    domain_max = float(map_size) - 1.0
    xs = np.linspace(0.0, domain_max, resolution, dtype=np.float64)
    ys = np.linspace(0.0, domain_max, resolution, dtype=np.float64)
    dense = np.zeros((resolution, resolution), dtype=np.float32)
    for yi, y in enumerate(ys):
        dense[yi, :] = _height_from_grid(terrain, map_size, xs, np.full_like(xs, y))
    return xs, ys, dense


def _normalize_goal_positions(raw_goals: Any, terrain: np.ndarray, map_size: float) -> List[Dict[str, Any]]:
    goals: List[Dict[str, Any]] = []
    if not raw_goals:
        return goals
    for idx, raw in enumerate(raw_goals):
        if isinstance(raw, dict):
            pos = _as_float_list(raw.get("position", raw.get("pos", raw.get("center", raw))), 3)
            name = str(raw.get("name", f"agent_goal_{idx}"))
        else:
            pos = _as_float_list(raw, 3)
            name = f"agent_goal_{idx}"
        if pos is None:
            continue
        terrain_h = float(_height_from_grid(terrain, map_size, np.asarray([pos[0]]), np.asarray([pos[1]]))[0])
        goals.append(
            {
                "name": name,
                "position": pos,
                "terrain_height": terrain_h,
                "height_above_terrain": float(pos[2] - terrain_h),
            }
        )
    return goals


def _start_positions_from_trajectory(
    first_frame: np.ndarray,
    terrain: np.ndarray,
    map_size: float,
    agent_size: float,
    agent_max_speed: Optional[float],
    agent_accel: Optional[float],
) -> List[Dict[str, Any]]:
    starts: List[Dict[str, Any]] = []
    for idx, pos_arr in enumerate(first_frame):
        pos = [float(v) for v in pos_arr[:3].tolist()]
        terrain_h = float(_height_from_grid(terrain, map_size, np.asarray([pos[0]]), np.asarray([pos[1]]))[0])
        starts.append(
            {
                "name": f"agent_{idx}",
                "position": pos,
                "agent_size": float(agent_size),
                "max_speed": float(agent_max_speed) if agent_max_speed is not None else None,
                "accel": float(agent_accel) if agent_accel is not None else None,
                "terrain_height": terrain_h,
                "height_above_terrain": float(pos[2] - terrain_h),
            }
        )
    return starts


def export_replay_scenario_snapshot(
    output_dir: Path,
    prefix: str,
    trajectory: np.ndarray,
    vis_context: Dict[str, Any],
    args: Any = None,
    html_path: Optional[str] = None,
    image_path: Optional[str] = None,
    reuse_cache: bool = True,
) -> Dict[str, Any]:
    terrain_raw = vis_context.get("terrain") if isinstance(vis_context, dict) else None
    if terrain_raw is None:
        raise ValueError("vis_context terrain is required to export a Gazebo replay scenario")
    terrain_sampled = np.asarray(terrain_raw, dtype=np.float32)
    if terrain_sampled.ndim != 2:
        raise ValueError("vis_context terrain must be a 2D height field")

    map_size_raw = vis_context.get("map_size", None)
    map_size = int(map_size_raw if map_size_raw is not None else os.getenv("MAP_SIZE", "200"))
    if map_size < 2:
        raise ValueError("map_size must be at least 2")

    dense_resolution = _int_env("GAZEBO_REPLAY_DENSE_RESOLUTION", map_size)
    dense_x, dense_y, terrain_dense = _sample_dense_terrain(terrain_sampled, map_size, dense_resolution)
    terrain_h, terrain_w = terrain_sampled.shape
    sample_x, sample_y = _terrain_coordinate_vectors(map_size, terrain_w, terrain_h)

    terrain_sampled_path = output_dir / f"{prefix}_terrain_sampled.npy"
    terrain_dense_path = output_dir / f"{prefix}_terrain_dense.npy"
    sampled_artifact = _save_npy_cached(terrain_sampled_path, terrain_sampled, reuse_cache=reuse_cache)
    dense_artifact = _save_npy_cached(terrain_dense_path, terrain_dense, reuse_cache=reuse_cache)

    agent_size = _float_attr(args, "agent_size", "AGENT_SIZE", 0.5)
    collision_threshold = _float_attr(args, "collision_distance_threshold", "COLLISION_DISTANCE_THRESHOLD", 0.5)
    agent_max_speed = _float_attr(args, "agent_max_speed", "AGENT_MAX_SPEED", None)
    agent_accel = _float_attr(args, "agent_accel", "AGENT_ACCEL", None)

    starts = _start_positions_from_trajectory(
        trajectory[0],
        terrain_sampled,
        float(map_size),
        float(agent_size if agent_size is not None else 0.5),
        agent_max_speed,
        agent_accel,
    )

    goal_pos = _as_float_list(vis_context.get("goal_pos") if isinstance(vis_context, dict) else None, 3)
    central_goal = None
    if goal_pos is not None:
        terrain_h_goal = float(_height_from_grid(terrain_sampled, map_size, np.asarray([goal_pos[0]]), np.asarray([goal_pos[1]]))[0])
        central_goal = {
            "name": "center_goal",
            "position": goal_pos,
            "terrain_height": terrain_h_goal,
            "height_above_terrain": float(goal_pos[2] - terrain_h_goal),
        }

    agent_goals = _normalize_goal_positions(vis_context.get("agent_goals", []), terrain_sampled, float(map_size))
    obstacles = [normalize_obstacle(ob, i) for i, ob in enumerate(vis_context.get("obstacles", []) or [])]

    scenario_json_path = output_dir / f"{prefix}_scenario.json"
    snapshot: Dict[str, Any] = {
        "export_version": FORMAT_VERSION,
        "scenario_name": str(vis_context.get("scenario_name", "paper3d_terrain_energy") if isinstance(vis_context, dict) else "paper3d_terrain_energy"),
        "snapshot_source": "trajectory_snapshot_exporter.vis_context",
        "map_size": int(map_size),
        "seed": vis_context.get("scenario_seed") if isinstance(vis_context, dict) else None,
        "terrain_seed": vis_context.get("terrain_seed") if isinstance(vis_context, dict) else None,
        "terrain_variant_seed": vis_context.get("terrain_variant_seed") if isinstance(vis_context, dict) else None,
        "terrain_complexity_level": int(getattr(args, "terrain_complexity_level", 0) or 0) if args is not None else 0,
        "terrain_params": _json_safe(vis_context.get("terrain_params", {}) if isinstance(vis_context, dict) else {}),
        "coordinate_domain": {
            "x_min": 0.0,
            "x_max": float(map_size - 1),
            "y_min": 0.0,
            "y_max": float(map_size - 1),
            "domain_source": "[0, map_size-1]",
        },
        "terrain": {
            "source": "vis_context.terrain with Scenario.get_terrain_height-compatible bilinear interpolation",
            "sampled_npy": str(terrain_sampled_path),
            "dense_npy": str(terrain_dense_path),
            "sampled_sha256": sampled_artifact["sha256"],
            "dense_sha256": dense_artifact["sha256"],
            "sampled_shape": [int(terrain_h), int(terrain_w)],
            "dense_shape": [int(terrain_dense.shape[0]), int(terrain_dense.shape[1])],
            "terrain_downsampled": bool(terrain_sampled.shape[0] != map_size or terrain_sampled.shape[1] != map_size),
            "terrain_sample_rate": max(1, int(round(float(map_size) / max(1, max(terrain_h, terrain_w))))),
            "stored_sample_coordinates": {
                "description": "coordinates implied by vis_context.terrain samples",
                "x": [float(x) for x in sample_x.tolist()],
                "y": [float(y) for y in sample_y.tolist()],
            },
            "effective_interpolation_coordinates": {
                "description": "coordinates used for bilinear interpolation in the exported replay scene",
                "x": [float(x) for x in sample_x.tolist()],
                "y": [float(y) for y in sample_y.tolist()],
            },
            "dense_coordinates": {
                "x": [float(x) for x in dense_x.tolist()],
                "y": [float(y) for y in dense_y.tolist()],
            },
            "height_stats": {
                "sampled_min": float(np.min(terrain_sampled)),
                "sampled_max": float(np.max(terrain_sampled)),
                "sampled_mean": float(np.mean(terrain_sampled)),
                "dense_min": float(np.min(terrain_dense)),
                "dense_max": float(np.max(terrain_dense)),
                "dense_mean": float(np.mean(terrain_dense)),
            },
            "artifacts": {
                "sampled": sampled_artifact,
                "dense": dense_artifact,
                "reuse_cache": bool(reuse_cache),
            },
        },
        "mountains": {
            "base_mountain_centers": _json_safe(vis_context.get("base_mountain_centers", []) if isinstance(vis_context, dict) else []),
            "actual_mountain_centers": _json_safe(vis_context.get("actual_mountain_centers", []) if isinstance(vis_context, dict) else []),
        },
        "start_positions": starts,
        "goal": central_goal,
        "agent_goals": agent_goals,
        "obstacles": obstacles,
        "collision": {
            "collision_distance_threshold": float(collision_threshold if collision_threshold is not None else 0.5),
            "terrain_collision_rule": "z - terrain_height < threshold",
            "sphere_collision_rule": "norm(position - center) - radius < threshold",
        },
        "python_reference_html": html_path,
        "python_reference_image": image_path,
        "export_paths": {
            "output_dir": str(output_dir),
            "scenario_json": str(scenario_json_path),
            "terrain_sampled_npy": str(terrain_sampled_path),
            "terrain_dense_npy": str(terrain_dense_path),
        },
        "artifact_cache": {
            "reuse_cache": bool(reuse_cache),
            "terrain_sampled_cache_hit": bool(sampled_artifact.get("cache_hit", False)),
            "terrain_dense_cache_hit": bool(dense_artifact.get("cache_hit", False)),
        },
    }
    with scenario_json_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(snapshot), f, ensure_ascii=False, indent=2)
    return snapshot


def write_trajectory_snapshot(
    output_dir: Path,
    prefix: str,
    episode_data: Dict[str, Any],
    vis_context: Optional[Dict[str, Any]] = None,
    args: Any = None,
    generated_files: Optional[Dict[str, Any]] = None,
    source: str = "evaluate_optimized.generate_visualization",
    compact: bool = False,
    reuse_cache: bool = True,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = dict(generated_files or {})
    trajectory = _trajectory_array(episode_data.get("trajectory", []))
    trajectory_f32 = trajectory.astype(np.float32)

    trajectory_npy_path = output_dir / f"{prefix}_trajectory.npy"
    trajectory_npz_path = output_dir / f"{prefix}_trajectory.npz"
    if compact:
        trajectory_artifact = _save_npz_cached(trajectory_npz_path, "trajectory", trajectory_f32, reuse_cache=reuse_cache)
        trajectory_storage = "npz"
        trajectory_payload: List[Any] = []
        trajectory_npy_value = None
        trajectory_npz_value = str(trajectory_npz_path)
    else:
        trajectory_artifact = _save_npy_cached(trajectory_npy_path, trajectory_f32, reuse_cache=reuse_cache)
        trajectory_storage = "json+npy"
        trajectory_payload = trajectory.tolist()
        trajectory_npy_value = str(trajectory_npy_path)
        trajectory_npz_value = None

    scenario_snapshot = None
    scenario_json_path = None
    if isinstance(vis_context, dict) and vis_context.get("terrain") is not None:
        scenario_snapshot = export_replay_scenario_snapshot(
            output_dir=output_dir,
            prefix=prefix,
            trajectory=trajectory,
            vis_context=vis_context,
            args=args,
            html_path=generated_files.get("html_path"),
            image_path=generated_files.get("image_path"),
            reuse_cache=reuse_cache,
        )
        scenario_json_path = scenario_snapshot.get("export_paths", {}).get("scenario_json")

    simulation_dt = _float_attr(args, "simulation_dt", "SIMULATION_DT", 0.08)
    sample_interval = _int_env("EVAL_TRAJECTORY_SAMPLE_INTERVAL", 1)
    agent_size = _float_attr(args, "agent_size", "AGENT_SIZE", 0.5)
    collision_threshold = _float_attr(args, "collision_distance_threshold", "COLLISION_DISTANCE_THRESHOLD", 0.5)

    snapshot_path = output_dir / f"{prefix}_trajectory_snapshot.json"
    snapshot: Dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "source": source,
        "episode": int(episode_data.get("episode", 0)) + 1,
        "reward": float(episode_data.get("reward", 0.0)),
        "steps": int(episode_data.get("steps", max(0, trajectory.shape[0] - 1))),
        "frame_count": int(trajectory.shape[0]),
        "agent_count": int(trajectory.shape[1]),
        "trajectory_layout": "[frame][agent][x,y,z]",
        "trajectory": trajectory_payload,
        "trajectory_embedded": bool(not compact),
        "trajectory_storage": trajectory_storage,
        "trajectory_npy_path": trajectory_npy_value,
        "trajectory_npz_path": trajectory_npz_value,
        "trajectory_sha256": trajectory_artifact["sha256"],
        "trajectory_artifact": trajectory_artifact,
        "sample_interval_steps": int(sample_interval),
        "simulation_dt": float(simulation_dt if simulation_dt is not None else 0.08),
        "sample_dt": float((simulation_dt if simulation_dt is not None else 0.08) * sample_interval),
        "agent_size": float(agent_size if agent_size is not None else 0.5),
        "collision_distance_threshold": float(collision_threshold if collision_threshold is not None else 0.5),
        "start_positions": trajectory[0].tolist(),
        "final_positions": trajectory[-1].tolist(),
        "episode_metrics": {
            "team_success": int(episode_data.get("team_success", episode_data.get("success", 0)) or 0),
            "collision_count": int(episode_data.get("collision_count", 0) or 0),
            "terrain_collision_count": int(episode_data.get("terrain_collision_count", 0) or 0),
            "obstacle_collision_count": int(episode_data.get("obstacle_collision_count", 0) or 0),
            "inter_agent_collision_count": int(episode_data.get("inter_agent_collision_count", 0) or 0),
            "path_length": episode_data.get("path_length"),
            "agent_path_lengths": episode_data.get("agent_path_lengths", []),
            "episode_done_reason": episode_data.get("episode_done_reason"),
        },
        "scenario_json_path": scenario_json_path,
        "terrain_snapshot_json_path": generated_files.get("terrain_snapshot_json_path"),
        "terrain_npy_path": generated_files.get("terrain_npy_path"),
        "html_path": generated_files.get("html_path"),
        "image_path": generated_files.get("image_path"),
        "export_paths": {
            "output_dir": str(output_dir),
            "trajectory_snapshot_json": str(snapshot_path),
            "trajectory_npy": trajectory_npy_value,
            "trajectory_npz": trajectory_npz_value,
            "scenario_json": scenario_json_path,
        },
        "artifact_cache": {
            "reuse_cache": bool(reuse_cache),
            "compact": bool(compact),
            "trajectory_cache_hit": bool(trajectory_artifact.get("cache_hit", False)),
        },
    }
    with snapshot_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(snapshot), f, ensure_ascii=False, indent=2)

    return {
        "trajectory_snapshot_path": str(snapshot_path),
        "trajectory_npy_path": trajectory_npy_value,
        "trajectory_npz_path": trajectory_npz_value,
        "scenario_json_path": str(scenario_json_path) if scenario_json_path else None,
    }


def _vec_or_default(value: Any, length: int, default: Sequence[float]) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        arr = np.asarray(default, dtype=np.float64).reshape(-1)
    if arr.size < length:
        arr = np.asarray(default, dtype=np.float64).reshape(-1)
    arr = arr[:length]
    if not np.all(np.isfinite(arr)):
        arr = np.asarray(default, dtype=np.float64).reshape(-1)[:length]
    return arr.astype(np.float32)


def _dynamic_state_arrays(episode_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    frames = episode_data.get("dynamic_state_history") or []
    if not frames:
        raise ValueError("episode_data.dynamic_state_history is required for Gazebo dynamic replay")
    if not isinstance(frames, list) or not isinstance(frames[0], list) or not frames[0]:
        raise ValueError("dynamic_state_history must have shape [frame][agent]{state}")
    frame_count = len(frames)
    agent_count = max(len(frame) for frame in frames)
    if agent_count < 1:
        raise ValueError("dynamic_state_history has no agents")

    positions = np.zeros((frame_count, agent_count, 3), dtype=np.float32)
    velocities = np.zeros((frame_count, agent_count, 3), dtype=np.float32)
    accelerations = np.zeros((frame_count, agent_count, 3), dtype=np.float32)
    orientations = np.zeros((frame_count, agent_count, 4), dtype=np.float32)
    angular_velocities = np.zeros((frame_count, agent_count, 3), dtype=np.float32)
    motor_speeds = np.zeros((frame_count, agent_count, 4), dtype=np.float32)
    orientations[:, :, 0] = 1.0

    for frame_idx, frame in enumerate(frames):
        if not isinstance(frame, list):
            continue
        for agent_idx in range(min(agent_count, len(frame))):
            state = frame[agent_idx] if isinstance(frame[agent_idx], dict) else {}
            positions[frame_idx, agent_idx, :] = _vec_or_default(state.get("position"), 3, [0.0, 0.0, 0.0])
            velocities[frame_idx, agent_idx, :] = _vec_or_default(state.get("velocity"), 3, [0.0, 0.0, 0.0])
            accelerations[frame_idx, agent_idx, :] = _vec_or_default(state.get("acceleration"), 3, [0.0, 0.0, 0.0])
            quat = _vec_or_default(state.get("orientation"), 4, [1.0, 0.0, 0.0, 0.0]).astype(np.float64)
            q_norm = float(np.linalg.norm(quat))
            if np.isfinite(q_norm) and q_norm > 1e-9:
                quat = quat / q_norm
            else:
                quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            orientations[frame_idx, agent_idx, :] = quat.astype(np.float32)
            angular_velocities[frame_idx, agent_idx, :] = _vec_or_default(state.get("angular_velocity"), 3, [0.0, 0.0, 0.0])
            motor_speeds[frame_idx, agent_idx, :] = _vec_or_default(state.get("motor_speeds"), 4, [0.0, 0.0, 0.0, 0.0])

    if not np.all(np.isfinite(positions)):
        raise ValueError("dynamic positions contain non-finite values")
    times_raw = episode_data.get("dynamic_time_history") or []
    if len(times_raw) == frame_count:
        times = np.asarray(times_raw, dtype=np.float32).reshape(-1)
    else:
        times = np.arange(frame_count, dtype=np.float32)
    if not np.all(np.isfinite(times)):
        times = np.arange(frame_count, dtype=np.float32)
    steps_raw = episode_data.get("dynamic_step_indices") or []
    if len(steps_raw) == frame_count:
        step_indices = np.asarray(steps_raw, dtype=np.int32).reshape(-1)
    else:
        step_indices = np.arange(frame_count, dtype=np.int32)

    return {
        "positions": positions,
        "velocities": velocities,
        "accelerations": accelerations,
        "orientations_wxyz": orientations,
        "angular_velocities": angular_velocities,
        "motor_speeds": motor_speeds,
        "times": times.astype(np.float32),
        "step_indices": step_indices.astype(np.int32),
    }


def _history_to_array(history: Any, fallback_agent_count: int) -> np.ndarray:
    if not history:
        return np.zeros((0, int(fallback_agent_count), 0), dtype=np.float32)
    frames = history if isinstance(history, list) else []
    frame_count = len(frames)
    agent_count = max([len(frame) for frame in frames if isinstance(frame, list)] + [int(fallback_agent_count)])
    dim = 0
    for frame in frames:
        if not isinstance(frame, list):
            continue
        for item in frame:
            try:
                dim = max(dim, int(np.asarray(item, dtype=np.float32).reshape(-1).size))
            except Exception:
                pass
    arr = np.full((frame_count, agent_count, dim), np.nan, dtype=np.float32)
    if dim == 0:
        return arr
    for frame_idx, frame in enumerate(frames):
        if not isinstance(frame, list):
            continue
        for agent_idx, item in enumerate(frame[:agent_count]):
            try:
                vals = np.asarray(item, dtype=np.float32).reshape(-1)
            except Exception:
                continue
            vals = vals[:dim]
            if vals.size:
                arr[frame_idx, agent_idx, : vals.size] = vals
    return arr


def write_dynamic_replay_snapshot(
    output_dir: Path,
    prefix: str,
    episode_data: Dict[str, Any],
    args: Any = None,
    trajectory_snapshot_path: Optional[Path] = None,
    scenario_json_path: Optional[Path] = None,
    source: str = "evaluate_optimized.generate_visualization",
    compact: bool = True,
    reuse_cache: bool = True,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = _dynamic_state_arrays(episode_data)
    agent_count = int(arrays["positions"].shape[1])
    arrays["raw_actions"] = _history_to_array(episode_data.get("dynamic_raw_action_history"), agent_count)
    arrays["executed_actions"] = _history_to_array(episode_data.get("dynamic_executed_action_history"), agent_count)
    arrays["potential_field_forces"] = _history_to_array(episode_data.get("dynamic_pf_force_history"), agent_count)

    dynamic_npz_path = output_dir / f"{prefix}_dynamic_replay.npz"
    dynamic_artifact = _save_npz_bundle_cached(dynamic_npz_path, arrays, reuse_cache=reuse_cache)

    simulation_dt = _float_attr(args, "simulation_dt", "SIMULATION_DT", 0.08)
    sample_dt = float(simulation_dt if simulation_dt is not None else 0.08)
    if arrays["times"].shape[0] >= 2:
        dt_candidates = np.diff(arrays["times"].astype(np.float64))
        dt_candidates = dt_candidates[np.isfinite(dt_candidates) & (dt_candidates > 0)]
        if dt_candidates.size:
            sample_dt = float(np.median(dt_candidates))
    agent_size = _float_attr(args, "agent_size", "AGENT_SIZE", 0.5)
    collision_threshold = _float_attr(args, "collision_distance_threshold", "COLLISION_DISTANCE_THRESHOLD", 0.5)
    identity = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    has_non_identity_attitude = bool(np.any(np.abs(arrays["orientations_wxyz"] - identity.reshape(1, 1, 4)) > 1e-5))

    snapshot_path = output_dir / f"{prefix}_dynamic_replay.json"
    snapshot = {
        "format_version": FORMAT_VERSION,
        "source": source,
        "episode": int(episode_data.get("episode", 0)) + 1,
        "reward": float(episode_data.get("reward", 0.0)),
        "steps": int(episode_data.get("steps", max(0, arrays["positions"].shape[0] - 1))),
        "frame_count": int(arrays["positions"].shape[0]),
        "agent_count": agent_count,
        "dynamic_layout": {
            "positions": "[frame][agent][x,y,z]",
            "velocities": "[frame][agent][vx,vy,vz]",
            "accelerations": "[frame][agent][ax,ay,az]",
            "orientations_wxyz": "[frame][agent][w,x,y,z]",
            "angular_velocities": "[frame][agent][wx,wy,wz]",
            "motor_speeds": "[frame][agent][m0,m1,m2,m3]",
            "raw_actions": "[transition][agent][action_dim]",
            "executed_actions": "[transition][agent][action_dim]",
            "potential_field_forces": "[transition][agent][force_dim]",
        },
        "dynamic_storage": "npz",
        "dynamic_npz_path": str(dynamic_npz_path),
        "dynamic_artifact": dynamic_artifact,
        "simulation_dt": float(simulation_dt if simulation_dt is not None else 0.08),
        "sample_dt": sample_dt,
        "agent_size": float(agent_size if agent_size is not None else 0.5),
        "collision_distance_threshold": float(collision_threshold if collision_threshold is not None else 0.5),
        "attitude_source": "recorded_quadrotor_state" if has_non_identity_attitude else "identity_or_unavailable",
        "collision_handling": "Gazebo replay sets model poses from recorded Python states; physics contact response is not fed back into the policy.",
        "trajectory_snapshot_json_path": str(Path(trajectory_snapshot_path).resolve()) if trajectory_snapshot_path else None,
        "scenario_json_path": str(Path(scenario_json_path).resolve()) if scenario_json_path else None,
        "export_paths": {
            "output_dir": str(output_dir),
            "dynamic_replay_json": str(snapshot_path),
            "dynamic_replay_npz": str(dynamic_npz_path),
            "trajectory_snapshot_json": str(Path(trajectory_snapshot_path).resolve()) if trajectory_snapshot_path else None,
            "scenario_json": str(Path(scenario_json_path).resolve()) if scenario_json_path else None,
        },
        "artifact_cache": {
            "reuse_cache": bool(reuse_cache),
            "compact": bool(compact),
            "dynamic_cache_hit": bool(dynamic_artifact.get("cache_hit", False)),
        },
    }
    with snapshot_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(snapshot), f, ensure_ascii=False, indent=2)

    return {
        "dynamic_replay_json_path": str(snapshot_path),
        "dynamic_replay_npz_path": str(dynamic_npz_path),
    }


def _reference_trajectory_from_scenario(scenario_json: Path, steps: int) -> Tuple[Dict[str, Any], np.ndarray]:
    with Path(scenario_json).open("r", encoding="utf-8") as f:
        scenario = json.load(f)
    starts = [entry["position"] for entry in scenario.get("start_positions", [])]
    goals = [entry["position"] for entry in scenario.get("agent_goals", [])]
    if not starts:
        raise ValueError("scenario has no start_positions")
    if not goals:
        goals = starts
    steps = max(2, int(steps))
    frames = []
    for frame_idx in range(steps):
        alpha = frame_idx / float(steps - 1)
        frame = []
        for idx, start in enumerate(starts):
            s = np.asarray(start, dtype=np.float64)
            g = np.asarray(goals[idx if idx < len(goals) else -1], dtype=np.float64)
            frame.append(((1.0 - alpha) * s + alpha * g).tolist())
        frames.append(frame)
    return scenario, np.asarray(frames, dtype=np.float64)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a trajectory snapshot for Gazebo replay.")
    parser.add_argument("--scenario-json", required=True, help="Existing scenario.json used for a reference trajectory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="reference")
    parser.add_argument("--reference-steps", type=int, default=120)
    parser.add_argument("--compact", action="store_true", default=_env_flag("GAZEBO_REPLAY_COMPACT", False), help="Store trajectory in compressed .npz and omit the full trajectory array from JSON.")
    parser.add_argument("--no-cache", action="store_true", help="Rewrite array artifacts even when matching sidecar metadata exists.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    scenario, trajectory = _reference_trajectory_from_scenario(Path(args.scenario_json), args.reference_steps)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_npy = out_dir / f"{args.prefix}_trajectory.npy"
    traj_npz = out_dir / f"{args.prefix}_trajectory.npz"
    trajectory_f32 = trajectory.astype(np.float32)
    reuse_cache = not bool(args.no_cache)
    if args.compact:
        artifact = _save_npz_cached(traj_npz, "trajectory", trajectory_f32, reuse_cache=reuse_cache)
        trajectory_payload: List[Any] = []
        trajectory_npy_value = None
        trajectory_npz_value = str(traj_npz)
        trajectory_storage = "npz"
    else:
        artifact = _save_npy_cached(traj_npy, trajectory_f32, reuse_cache=reuse_cache)
        trajectory_payload = trajectory.tolist()
        trajectory_npy_value = str(traj_npy)
        trajectory_npz_value = None
        trajectory_storage = "json+npy"
    snapshot_path = out_dir / f"{args.prefix}_trajectory_snapshot.json"
    snapshot = {
        "format_version": FORMAT_VERSION,
        "source": "trajectory_snapshot_exporter.reference_from_scenario",
        "episode": 1,
        "reward": 0.0,
        "steps": int(trajectory.shape[0] - 1),
        "frame_count": int(trajectory.shape[0]),
        "agent_count": int(trajectory.shape[1]),
        "trajectory_layout": "[frame][agent][x,y,z]",
        "trajectory": trajectory_payload,
        "trajectory_embedded": bool(not args.compact),
        "trajectory_storage": trajectory_storage,
        "trajectory_npy_path": trajectory_npy_value,
        "trajectory_npz_path": trajectory_npz_value,
        "trajectory_sha256": artifact["sha256"],
        "trajectory_artifact": artifact,
        "sample_interval_steps": 1,
        "simulation_dt": 0.08,
        "sample_dt": 0.08,
        "agent_size": float((scenario.get("start_positions") or [{}])[0].get("agent_size", 0.5)),
        "collision_distance_threshold": float(scenario.get("collision", {}).get("collision_distance_threshold", 0.5)),
        "start_positions": trajectory[0].tolist(),
        "final_positions": trajectory[-1].tolist(),
        "scenario_json_path": str(Path(args.scenario_json).resolve()),
        "export_paths": {
            "output_dir": str(out_dir),
            "trajectory_snapshot_json": str(snapshot_path),
            "trajectory_npy": trajectory_npy_value,
            "trajectory_npz": trajectory_npz_value,
            "scenario_json": str(Path(args.scenario_json).resolve()),
        },
        "artifact_cache": {
            "reuse_cache": bool(reuse_cache),
            "compact": bool(args.compact),
            "trajectory_cache_hit": bool(artifact.get("cache_hit", False)),
        },
    }
    with snapshot_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(snapshot), f, ensure_ascii=False, indent=2)
    print(f"[trajectory_snapshot_exporter] trajectory_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
