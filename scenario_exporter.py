#!/usr/bin/env python3
"""Export a reset MATD3 scenario as an auditable static scene snapshot."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


EXPORT_VERSION = 1


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


def _bool_from_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _terrain_coordinate_vectors(map_size: float, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """Coordinates used by the current Scenario.get_terrain_height interpolation."""
    max_coord = float(map_size) - 1.0
    if width <= 1:
        xs = np.asarray([0.0], dtype=np.float64)
    else:
        xs = np.linspace(0.0, max_coord, int(width), dtype=np.float64)
    if height <= 1:
        ys = np.asarray([0.0], dtype=np.float64)
    else:
        ys = np.linspace(0.0, max_coord, int(height), dtype=np.float64)
    return xs, ys


def _stored_sample_vectors(map_size: float, sample_rate: int) -> Tuple[List[float], List[float]]:
    """Coordinates originally sampled during terrain downsampling."""
    m = int(round(float(map_size)))
    if m <= 0:
        return [], []
    xs = list(np.arange(0, m, int(sample_rate), dtype=np.float64))
    ys = list(np.arange(0, m, int(sample_rate), dtype=np.float64))
    last = float(m - 1)
    if xs and xs[-1] != last:
        xs.append(last)
    if ys and ys[-1] != last:
        ys.append(last)
    return [float(x) for x in xs], [float(y) for y in ys]


def _sample_dense_terrain(
    scenario: Any,
    map_size: float,
    resolution: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolution = int(resolution)
    if resolution < 2:
        raise ValueError("dense terrain resolution must be at least 2")
    domain_max = float(map_size) - 1.0
    xs = np.linspace(0.0, domain_max, resolution, dtype=np.float64)
    ys = np.linspace(0.0, domain_max, resolution, dtype=np.float64)
    dense = np.zeros((resolution, resolution), dtype=np.float32)
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            dense[yi, xi] = float(scenario.get_terrain_height(float(x), float(y)))
    return xs, ys, dense


def normalize_obstacle(raw: Any, index: int) -> Dict[str, Any]:
    """Normalize current and future obstacle records without silently changing geometry."""
    name = f"obstacle_{index}"
    source = "unknown"
    if isinstance(raw, dict):
        name = str(raw.get("name", name))
        declared_type = str(raw.get("type", raw.get("shape", ""))).strip().lower()
        center = raw.get("center", raw.get("pos", raw.get("position")))
        source = "dict"
        if declared_type in ("sphere", "ball"):
            radius = raw.get("radius", raw.get("r", raw.get("size")))
            c = _as_float_list(center, 3)
            if c is None or radius is None:
                raise ValueError(f"{name}: sphere obstacle requires center and radius")
            return {
                "name": name,
                "type": "sphere",
                "type_source": "declared",
                "center": c,
                "radius": float(radius),
            }
        if declared_type in ("box", "cube"):
            size = raw.get("size", raw.get("dimensions", raw.get("extent")))
            c = _as_float_list(center, 3)
            s = _as_float_list(size, 3)
            if c is None or s is None:
                raise ValueError(f"{name}: box obstacle requires center and 3D size")
            return {
                "name": name,
                "type": "box",
                "type_source": "declared",
                "center": c,
                "size": s,
            }
        if declared_type in ("cylinder", "cyl"):
            radius = raw.get("radius", raw.get("r"))
            length = raw.get("length", raw.get("height"))
            c = _as_float_list(center, 3)
            if c is None or radius is None or length is None:
                raise ValueError(f"{name}: cylinder obstacle requires center, radius and length")
            return {
                "name": name,
                "type": "cylinder",
                "type_source": "declared",
                "center": c,
                "radius": float(radius),
                "length": float(length),
            }
        if center is not None and ("radius" in raw or "r" in raw):
            radius = raw.get("radius", raw.get("r"))
            c = _as_float_list(center, 3)
            if c is None:
                raise ValueError(f"{name}: inferred sphere obstacle has invalid center")
            return {
                "name": name,
                "type": "sphere",
                "type_source": "inferred_from_center_radius",
                "center": c,
                "radius": float(radius),
            }
        raise ValueError(f"{name}: unsupported obstacle geometry keys={sorted(raw.keys())}")

    state = getattr(raw, "state", None)
    center = getattr(state, "p_pos", None)
    radius = getattr(raw, "radius", getattr(raw, "r", getattr(raw, "size", None)))
    c = _as_float_list(center, 3)
    if c is None or radius is None:
        raise ValueError(f"{name}: unsupported obstacle object from {source}")
    return {
        "name": str(getattr(raw, "name", name)),
        "type": "sphere",
        "type_source": "inferred_from_landmark_size",
        "center": c,
        "radius": float(radius),
    }


def _world_start_positions(world: Any, scenario: Any) -> List[Dict[str, Any]]:
    starts = []
    for idx, agent in enumerate(getattr(world, "agents", []) or []):
        state = getattr(agent, "state", None)
        pos = _as_float_list(getattr(state, "p_pos", None), 3)
        if pos is None:
            continue
        vel = _as_float_list(getattr(state, "p_vel", None), 3)
        orientation = _as_float_list(getattr(state, "orientation", None), 4)
        terrain_h = float(scenario.get_terrain_height(pos[0], pos[1]))
        starts.append(
            {
                "name": str(getattr(agent, "name", f"agent_{idx}")),
                "position": pos,
                "initial_velocity": vel,
                "initial_orientation_wxyz": orientation,
                "agent_size": float(getattr(agent, "size", 0.05)),
                "max_speed": float(getattr(agent, "max_speed", 0.0)) if getattr(agent, "max_speed", None) is not None else None,
                "accel": float(getattr(agent, "accel", 0.0)) if getattr(agent, "accel", None) is not None else None,
                "terrain_height": terrain_h,
                "height_above_terrain": float(pos[2] - terrain_h),
            }
        )
    return starts


def _world_agent_goals(world: Any, scenario: Any) -> List[Dict[str, Any]]:
    raw_goals: List[Any] = []
    if hasattr(world, "agent_goals") and getattr(world, "agent_goals") is not None:
        raw_goals = list(getattr(world, "agent_goals"))
    if not raw_goals:
        for agent in getattr(world, "agents", []) or []:
            goal = getattr(agent, "goal_a", None)
            raw_goals.append(getattr(getattr(goal, "state", None), "p_pos", None))

    goals = []
    for idx, raw in enumerate(raw_goals):
        pos = _as_float_list(raw, 3)
        if pos is None:
            continue
        terrain_h = float(scenario.get_terrain_height(pos[0], pos[1]))
        goals.append(
            {
                "name": f"agent_goal_{idx}",
                "position": pos,
                "terrain_height": terrain_h,
                "height_above_terrain": float(pos[2] - terrain_h),
            }
        )
    return goals


def _make_reference_trajectories(
    starts: Sequence[Dict[str, Any]],
    goals: Sequence[Dict[str, Any]],
    steps: int = 60,
) -> List[List[List[float]]]:
    """Return time-major straight reference paths for the existing HTML renderer."""
    if not starts:
        return []
    count = len(starts)
    steps = max(2, int(steps))
    time_major: List[List[List[float]]] = []
    for t in range(steps):
        alpha = float(t) / float(steps - 1)
        frame: List[List[float]] = []
        for i in range(count):
            start = np.asarray(starts[i]["position"], dtype=np.float64)
            if i < len(goals):
                goal = np.asarray(goals[i]["position"], dtype=np.float64)
            else:
                goal = start.copy()
            p = (1.0 - alpha) * start + alpha * goal
            frame.append([float(p[0]), float(p[1]), float(p[2])])
        time_major.append(frame)
    return time_major


def _generate_python_html(
    output_dir: Path,
    scenario: Any,
    starts: Sequence[Dict[str, Any]],
    central_goal: Optional[Dict[str, Any]],
    agent_goals: Sequence[Dict[str, Any]],
) -> Optional[str]:
    try:
        from visualization.trajectory_visualizer import TrajectoryVisualizer
    except Exception as exc:
        print(f"[scenario_exporter] Plotly HTML skipped: failed to import visualizer: {exc}")
        return None

    trajectories = _make_reference_trajectories(starts, agent_goals, steps=80)
    if not trajectories:
        return None
    goal_positions = {
        "goal_pos": central_goal["position"] if central_goal is not None else None,
        "agent_goals": [g["position"] for g in agent_goals],
    }
    html_path = output_dir / "python_reference_scene_interactive.html"
    ok = TrajectoryVisualizer(verbose=False).generate_trajectory_interactive(
        trajectories=trajectories,
        save_path=str(html_path),
        title="MATD3 Python Reference Scene",
        goal_positions=goal_positions,
        scenario=scenario,
        env_instance=None,
    )
    return str(html_path) if ok else None


def load_scenario_class(scenario_name: str) -> Any:
    module_name = scenario_name
    if "." not in module_name:
        module_name = f"multiagent.scenarios.{scenario_name}"
    module = importlib.import_module(module_name)
    return getattr(module, "Scenario")


def export_scenario_snapshot(
    scenario: Any,
    world: Any,
    output_dir: Path,
    scenario_name: str,
    dense_resolution: int,
    generate_html: bool = True,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    terrain_sampled = np.asarray(getattr(scenario, "terrain", None), dtype=np.float32)
    if terrain_sampled.size == 0:
        raise RuntimeError("scenario.terrain is empty after reset_world")
    if terrain_sampled.ndim != 2:
        raise RuntimeError(f"scenario.terrain must be 2D, got shape={terrain_sampled.shape}")

    map_size = float(getattr(scenario, "map_size"))
    terrain_h, terrain_w = terrain_sampled.shape
    effective_x, effective_y = _terrain_coordinate_vectors(map_size, terrain_w, terrain_h)
    sample_rate = int(getattr(scenario, "terrain_sample_rate", 1))
    stored_sample_x, stored_sample_y = _stored_sample_vectors(map_size, sample_rate)
    dense_x, dense_y, terrain_dense = _sample_dense_terrain(scenario, map_size, dense_resolution)

    sampled_path = output_dir / "terrain_sampled.npy"
    dense_path = output_dir / "terrain_dense.npy"
    np.save(sampled_path, terrain_sampled)
    np.save(dense_path, terrain_dense)

    starts = _world_start_positions(world, scenario)
    central_goal_pos = _as_float_list(getattr(world, "goal_pos", None), 3)
    if central_goal_pos is None:
        central_goal_pos = _as_float_list(getattr(scenario, "goal_pos", None), 3)
    central_goal = None
    if central_goal_pos is not None:
        central_terrain_h = float(scenario.get_terrain_height(central_goal_pos[0], central_goal_pos[1]))
        central_goal = {
            "name": "center_goal",
            "position": central_goal_pos,
            "terrain_height": central_terrain_h,
            "height_above_terrain": float(central_goal_pos[2] - central_terrain_h),
        }
    agent_goals = _world_agent_goals(world, scenario)

    raw_obstacles = list(getattr(scenario, "obstacles", []) or [])
    obstacles = [normalize_obstacle(ob, i) for i, ob in enumerate(raw_obstacles)]
    actual_obstacle_seed = getattr(scenario, "current_episode_obstacle_seed", None)
    try:
        actual_obstacle_seed = int(actual_obstacle_seed) if actual_obstacle_seed is not None else None
    except Exception:
        actual_obstacle_seed = None

    html_path = None
    if generate_html:
        html_path = _generate_python_html(output_dir, scenario, starts, central_goal, agent_goals)

    try:
        collision_threshold = float(os.getenv("COLLISION_DISTANCE_THRESHOLD", getattr(scenario, "collision_distance_threshold", 0.5)))
    except Exception:
        collision_threshold = 0.5
    agent_sizes = []
    for start in starts:
        try:
            agent_sizes.append(float(start.get("agent_size")))
        except Exception:
            pass
    representative_agent_size = float(agent_sizes[0]) if agent_sizes else float(getattr(scenario, "agent_size", 0.5))

    snapshot: Dict[str, Any] = {
        "export_version": EXPORT_VERSION,
        "scenario_name": scenario_name,
        "map_size": map_size,
        "seed": int(getattr(scenario, "seed", -1)) if getattr(scenario, "seed", None) is not None else None,
        "terrain_seed": int(getattr(scenario, "current_terrain_seed", getattr(scenario, "seed", -1)))
        if getattr(scenario, "current_terrain_seed", getattr(scenario, "seed", None)) is not None
        else None,
        "terrain_variant_seed": int(getattr(scenario, "current_terrain_variant_seed", getattr(scenario, "terrain_variant_seed", 0)))
        if getattr(scenario, "current_terrain_variant_seed", getattr(scenario, "terrain_variant_seed", None)) is not None
        else None,
        "obstacle_seed": actual_obstacle_seed,
        "obstacle_seed_source": "scenario.current_episode_obstacle_seed",
        "terrain_complexity_level": int(getattr(scenario, "terrain_complexity_level", -1)),
        "terrain_params": _json_safe(getattr(scenario, "terrain_params", {}) or {}),
        "coordinate_domain": {
            "x_min": 0.0,
            "x_max": map_size - 1.0,
            "y_min": 0.0,
            "y_max": map_size - 1.0,
            "domain_source": "[0, map_size-1]",
        },
        "terrain": {
            "source": "scenario.get_terrain_height",
            "sampled_npy": str(sampled_path),
            "dense_npy": str(dense_path),
            "sampled_shape": [int(terrain_h), int(terrain_w)],
            "dense_shape": [int(terrain_dense.shape[0]), int(terrain_dense.shape[1])],
            "terrain_downsampled": bool(getattr(scenario, "terrain_downsampled", False)),
            "terrain_sample_rate": sample_rate,
            "stored_sample_coordinates": {
                "description": "coordinates used when scenario.terrain was originally downsampled",
                "x": stored_sample_x,
                "y": stored_sample_y,
            },
            "effective_interpolation_coordinates": {
                "description": "coordinates implied by current scenario.get_terrain_height and current Plotly HTML",
                "x": [float(x) for x in effective_x.tolist()],
                "y": [float(y) for y in effective_y.tolist()],
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
        },
        "mountains": {
            "base_mountain_centers": _json_safe(getattr(scenario, "base_mountain_centers", []) or []),
            "actual_mountain_centers": _json_safe(getattr(scenario, "actual_mountain_centers", []) or []),
        },
        "start_positions": starts,
        "goal": central_goal,
        "agent_goals": agent_goals,
        "obstacles": obstacles,
        "collision": {
            "collision_distance_threshold": collision_threshold,
            "agent_size": representative_agent_size,
            "agent_sizes": agent_sizes,
            "gazebo_physical_collision_radius_default": representative_agent_size,
            "terrain_collision_rule": "z - agent_size - terrain_height < collision_distance_threshold",
            "sphere_collision_rule": "norm(position - center) - obstacle_radius - agent_size < collision_distance_threshold",
            "gazebo_physical_collision_rule": "Gazebo dynamic UAV collision radius defaults to start_positions[].agent_size; Python safety threshold is recomputed from Gazebo pose.",
        },
        "python_reference_html": html_path,
        "export_paths": {
            "output_dir": str(output_dir),
            "scenario_json": str(output_dir / "scenario.json"),
            "terrain_sampled_npy": str(sampled_path),
            "terrain_dense_npy": str(dense_path),
        },
    }

    scenario_json_path = output_dir / "scenario.json"
    with scenario_json_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(snapshot), f, ensure_ascii=False, indent=2)
    return snapshot


def create_scenario_from_args(args: argparse.Namespace) -> Tuple[Any, Any]:
    if args.start_altitude_offset is not None:
        os.environ["START_ALTITUDE_OFFSET"] = str(args.start_altitude_offset)
    if args.goal_altitude is not None:
        os.environ["GOAL_ALTITUDE"] = str(args.goal_altitude)
    if args.agent_size is not None:
        os.environ["AGENT_SIZE"] = str(float(args.agent_size))
    os.environ.setdefault("QUIET_OUTPUT", "1")
    os.environ.setdefault("SUPPRESS_TERRAIN_OUTPUT", "1")

    scenario_cls = load_scenario_class(args.scenario_name)
    scenario = scenario_cls(
        seed=args.seed,
        map_size=args.map_size,
        terrain_complexity_level=args.terrain_complexity_level,
        random_terrain=args.random_terrain,
        use_fixed_positions=False,
        use_dynamic_obstacles=args.use_dynamic_obstacles,
        agent_size=args.agent_size,
    )
    world = scenario.make_world()
    world.is_main_env = True
    world.env_id = int(args.env_id)
    world.current_episode = int(args.episode_index)
    scenario.reset_world(world)
    return scenario, world


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a reset MATD3 scenario snapshot.")
    parser.add_argument("--output-dir", required=True, help="Directory for scenario.json, npy files and reference HTML.")
    parser.add_argument("--scenario-name", default="paper3d_terrain_vectorized")
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--map-size", type=float, default=200.0)
    parser.add_argument("--terrain-complexity-level", type=int, default=3)
    parser.add_argument("--dense-resolution", type=int, default=200)
    parser.add_argument("--start-altitude-offset", type=float, default=12.0)
    parser.add_argument("--goal-altitude", type=float, default=25.0)
    parser.add_argument("--agent-size", type=float, default=None, help="Agent physical / visualization radius in scene units; defaults to AGENT_SIZE or scenario default.")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--random-terrain", action="store_true")
    parser.add_argument("--use-dynamic-obstacles", action="store_true")
    parser.add_argument("--no-html", action="store_true", help="Do not generate the Python Plotly reference HTML.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    scenario, world = create_scenario_from_args(args)
    snapshot = export_scenario_snapshot(
        scenario=scenario,
        world=world,
        output_dir=output_dir,
        scenario_name=args.scenario_name,
        dense_resolution=args.dense_resolution,
        generate_html=not args.no_html,
    )
    print(f"[scenario_exporter] scenario_json={snapshot['export_paths']['scenario_json']}")
    if snapshot.get("python_reference_html"):
        print(f"[scenario_exporter] python_reference_html={snapshot['python_reference_html']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
