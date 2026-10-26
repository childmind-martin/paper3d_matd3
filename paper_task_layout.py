#!/usr/bin/env python3
"""Versioned task-layout contract for paper-grade navigation experiments.

The training and evaluation stacks historically recorded start/goal distance
parameters without proving that the resolved episode layout satisfied them.
This module provides one deterministic, serializable contract shared by the
scenario, evaluators, tests, and paper result exporters.

The reference path is a conservative 2.5-D grid proxy: spherical obstacles are
inflated in XY and terrain height contributes to segment length/climb.  It is
used to reject disconnected layouts and to stratify task difficulty; it is not
presented as the trajectory executed by the learned controller.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


TASK_LAYOUT_SCHEMA_VERSION = "paper_task_layout_v5"
REFERENCE_PATH_SCHEMA_VERSION = "surface_astar_v4"


class TaskLayoutError(ValueError):
    """Raised when a resolved task cannot satisfy the scientific contract."""


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(value)
    return value


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash JSON content after removing self-referential digest fields."""
    content = dict(_json_ready(payload))
    content.pop("content_sha256", None)
    encoded = json.dumps(
        content,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _terrain_sha256(terrain: Optional[np.ndarray]) -> Optional[str]:
    if terrain is None:
        return None
    array = np.asarray(terrain, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise TaskLayoutError("terrain must be a finite 2-D array")
    digest = hashlib.sha256()
    digest.update(str(tuple(int(value) for value in array.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(array.astype("<f4", copy=False)).tobytes())
    return digest.hexdigest()


def _serialized_obstacles(obstacles: Optional[Iterable[Any]]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for obstacle in list(obstacles or []):
        if isinstance(obstacle, Mapping):
            center = obstacle.get("center")
            radius = obstacle.get("radius", obstacle.get("size"))
        else:
            state = getattr(obstacle, "state", None)
            center = getattr(state, "p_pos", None)
            radius = getattr(obstacle, "size", None)
        try:
            center_array = np.asarray(center, dtype=np.float64).reshape(-1)
            radius_value = float(radius)
        except Exception:
            continue
        if center_array.size < 3 or not np.isfinite(center_array[:3]).all():
            continue
        if not math.isfinite(radius_value) or radius_value < 0.0:
            continue
        serialized.append(
            {
                "center": [float(value) for value in center_array[:3]],
                "radius": radius_value,
            }
        )
    return serialized


def task_identity_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return physical task identity, excluding run/episode bookkeeping."""
    return {
        "schema_version": payload.get("schema_version"),
        "map_size": payload.get("map_size"),
        "terrain_sha256": payload.get("terrain_sha256"),
        "obstacles": payload.get("obstacles"),
        "contract": payload.get("contract"),
        "agent_starts": payload.get("agent_starts"),
        "central_goal": payload.get("central_goal"),
        "agent_goals": payload.get("agent_goals"),
        "difficulty_bin": payload.get("difficulty_bin"),
    }


def task_id_for_payload(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(task_identity_payload(payload))[:24]


def atomic_write_json(path: os.PathLike[str] | str, payload: Mapping[str, Any]) -> Path:
    """Atomically persist a task artifact so workers never expose partial JSON."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    normalized = _json_ready(payload)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(normalized, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target


def _positions(value: Sequence[Sequence[float]], *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 3 or array.shape[0] <= 0:
        raise TaskLayoutError(f"{name} must have shape [N, >=3], got {array.shape}")
    array = array[:, :3]
    if not np.isfinite(array).all():
        raise TaskLayoutError(f"{name} contains non-finite coordinates")
    return array


def horizontal_distances(
    starts: Sequence[Sequence[float]], goals: Sequence[Sequence[float]]
) -> np.ndarray:
    starts_array = _positions(starts, name="agent_starts")
    goals_array = _positions(goals, name="agent_goals")
    if starts_array.shape[0] != goals_array.shape[0]:
        raise TaskLayoutError(
            "agent_starts and agent_goals must contain the same number of agents"
        )
    return np.linalg.norm(starts_array[:, :2] - goals_array[:, :2], axis=1)


def validate_distance_bounds(
    starts: Sequence[Sequence[float]],
    goals: Sequence[Sequence[float]],
    *,
    min_distance: float,
    max_distance: float,
    tolerance: float = 1e-5,
) -> np.ndarray:
    min_distance = float(min_distance)
    max_distance = float(max_distance)
    if not math.isfinite(min_distance) or not math.isfinite(max_distance):
        raise TaskLayoutError("start/goal distance bounds must be finite")
    if min_distance < 0.0 or max_distance <= min_distance:
        raise TaskLayoutError(
            f"invalid start/goal distance interval [{min_distance}, {max_distance}]"
        )
    distances = horizontal_distances(starts, goals)
    invalid = np.flatnonzero(
        (distances < min_distance - tolerance)
        | (distances > max_distance + tolerance)
    )
    if invalid.size:
        details = ", ".join(
            f"agent_{int(index)}={float(distances[index]):.6f}m"
            for index in invalid
        )
        raise TaskLayoutError(
            f"resolved task violates horizontal start/goal distance "
            f"[{min_distance:.6f}, {max_distance:.6f}]m: {details}"
        )
    return distances


def _terrain_height(terrain: Optional[np.ndarray], map_size: float, x: float, y: float) -> float:
    if terrain is None:
        return 0.0
    grid = np.asarray(terrain)
    if grid.ndim != 2 or grid.shape[0] <= 0 or grid.shape[1] <= 0:
        return 0.0
    x_scale = (grid.shape[1] - 1) / max(float(map_size) - 1.0, 1.0)
    y_scale = (grid.shape[0] - 1) / max(float(map_size) - 1.0, 1.0)
    xi = int(np.clip(round(float(x) * x_scale), 0, grid.shape[1] - 1))
    yi = int(np.clip(round(float(y) * y_scale), 0, grid.shape[0] - 1))
    value = float(grid[yi, xi])
    return value if math.isfinite(value) else 0.0


def _obstacle_geometry(obstacles: Optional[Iterable[Any]]) -> Tuple[np.ndarray, np.ndarray]:
    centers: List[List[float]] = []
    radii: List[float] = []
    for obstacle in list(obstacles or []):
        center = None
        radius = None
        if isinstance(obstacle, Mapping):
            center = obstacle.get("center")
            radius = obstacle.get("radius", obstacle.get("size"))
        else:
            state = getattr(obstacle, "state", None)
            center = getattr(state, "p_pos", None)
            radius = getattr(obstacle, "size", None)
        try:
            center_array = np.asarray(center, dtype=np.float64).reshape(-1)
            radius_value = float(radius)
        except Exception:
            continue
        if center_array.size < 2 or not np.isfinite(center_array[:2]).all():
            continue
        if not math.isfinite(radius_value) or radius_value < 0.0:
            continue
        centers.append([float(center_array[0]), float(center_array[1])])
        radii.append(radius_value)
    if not centers:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return np.asarray(centers, dtype=np.float64), np.asarray(radii, dtype=np.float64)


def _polyline_obstacle_surface_clearance(
    points_xy: np.ndarray,
    centers_xy: np.ndarray,
    radii: np.ndarray,
) -> Optional[float]:
    """Return exact XY surface clearance over every polyline segment."""
    points = np.asarray(points_xy, dtype=np.float64)
    centers = np.asarray(centers_xy, dtype=np.float64)
    obstacle_radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    if centers.shape[0] == 0:
        return None
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 1:
        raise TaskLayoutError("reference polyline must contain finite XY points")
    if not np.isfinite(points).all():
        raise TaskLayoutError("reference polyline contains non-finite XY points")
    if points.shape[0] == 1:
        distances = np.linalg.norm(
            points[0][None, :] - centers, axis=1
        ) - obstacle_radii
        return float(np.min(distances))
    segment_starts = points[:-1]
    segment_vectors = points[1:] - segment_starts
    squared_lengths = np.sum(segment_vectors * segment_vectors, axis=1)
    relative = centers[None, :, :] - segment_starts[:, None, :]
    projections = np.sum(
        relative * segment_vectors[:, None, :], axis=2
    )
    projections = np.divide(
        projections,
        squared_lengths[:, None],
        out=np.zeros_like(projections),
        where=squared_lengths[:, None] > 1e-12,
    )
    projections = np.clip(projections, 0.0, 1.0)
    closest = (
        segment_starts[:, None, :]
        + projections[:, :, None] * segment_vectors[:, None, :]
    )
    distances = np.linalg.norm(
        centers[None, :, :] - closest, axis=2
    ) - obstacle_radii[None, :]
    return float(np.min(distances))


@dataclass(frozen=True)
class ReferencePathMetrics:
    feasible: bool
    path_length: Optional[float]
    detour_ratio: Optional[float]
    maximum_climb: Optional[float]
    minimum_corridor_clearance: Optional[float]
    minimum_boundary_clearance: Optional[float]
    minimum_obstacle_clearance: Optional[float]
    grid_resolution: float
    expanded_nodes: int

    def to_dict(self) -> Dict[str, Any]:
        return _json_ready(
            {
                "schema_version": REFERENCE_PATH_SCHEMA_VERSION,
                "feasible": bool(self.feasible),
                "path_length": self.path_length,
                "detour_ratio": self.detour_ratio,
                "maximum_climb": self.maximum_climb,
                "minimum_corridor_clearance": self.minimum_corridor_clearance,
                "minimum_boundary_clearance": self.minimum_boundary_clearance,
                "minimum_obstacle_clearance": self.minimum_obstacle_clearance,
                "grid_resolution": float(self.grid_resolution),
                "expanded_nodes": int(self.expanded_nodes),
            }
        )


@dataclass(frozen=True)
class ReferencePathPlan:
    """One deterministic reference corridor and the metrics derived from it."""

    metrics: ReferencePathMetrics
    waypoints: Tuple[Tuple[float, float, float], ...]


def _reference_path_plan(
    start: Sequence[float],
    goal: Sequence[float],
    *,
    map_size: float,
    terrain: Optional[np.ndarray] = None,
    obstacles: Optional[Iterable[Any]] = None,
    grid_resolution: float = 4.0,
    obstacle_inflation: float = 1.5,
    boundary_clearance: float = 0.0,
    surface_clearance: float = 4.5,
) -> ReferencePathPlan:
    """Compute one deterministic 2.5-D A* corridor for one agent."""
    start_array = np.asarray(start, dtype=np.float64).reshape(-1)
    goal_array = np.asarray(goal, dtype=np.float64).reshape(-1)
    if start_array.size < 3 or goal_array.size < 3:
        raise TaskLayoutError("reference path endpoints must be 3-D")
    if not np.isfinite(start_array[:3]).all() or not np.isfinite(goal_array[:3]).all():
        raise TaskLayoutError("reference path endpoints must be finite")
    map_size = float(map_size)
    grid_resolution = float(grid_resolution)
    obstacle_inflation = max(0.0, float(obstacle_inflation))
    boundary_clearance = max(0.0, float(boundary_clearance))
    surface_clearance = float(surface_clearance)
    if (
        map_size <= 0.0
        or grid_resolution <= 0.0
        or not math.isfinite(surface_clearance)
        or surface_clearance <= 0.0
    ):
        raise TaskLayoutError(
            "map_size, grid_resolution, and surface_clearance must be positive"
        )

    grid_count = max(2, int(math.ceil(map_size / grid_resolution)) + 1)

    def to_cell(position: np.ndarray) -> Tuple[int, int]:
        return (
            int(np.clip(round(float(position[0]) / grid_resolution), 0, grid_count - 1)),
            int(np.clip(round(float(position[1]) / grid_resolution), 0, grid_count - 1)),
        )

    def to_xy(cell: Tuple[int, int]) -> Tuple[float, float]:
        # Use the exact physical endpoints as graph nodes.  The old planner
        # silently measured from rounded grid centres, so its reported length
        # did not exactly describe the corridor later presented to the agent.
        if cell == start_cell:
            return float(start_array[0]), float(start_array[1])
        if cell == goal_cell:
            return float(goal_array[0]), float(goal_array[1])
        return (
            min(float(cell[0]) * grid_resolution, map_size - 1e-6),
            min(float(cell[1]) * grid_resolution, map_size - 1e-6),
        )

    start_cell = to_cell(start_array)
    goal_cell = to_cell(goal_array)
    centers, radii = _obstacle_geometry(obstacles)

    blocked = np.zeros((grid_count, grid_count), dtype=bool)
    xs = np.minimum(
        np.arange(grid_count, dtype=np.float64) * grid_resolution,
        map_size - 1e-6,
    )
    ys = xs.copy()
    if boundary_clearance > 0.0:
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
        blocked |= (
            (grid_x < boundary_clearance)
            | (grid_y < boundary_clearance)
            | ((map_size - grid_x) < boundary_clearance)
            | ((map_size - grid_y) < boundary_clearance)
        )
    if centers.shape[0] > 0:
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
        for center, radius in zip(centers, radii):
            inflated = float(radius) + obstacle_inflation
            blocked |= (grid_x - center[0]) ** 2 + (grid_y - center[1]) ** 2 <= inflated ** 2
    blocked[start_cell] = False
    blocked[goal_cell] = False

    height_cache: Dict[Tuple[int, int], float] = {}

    def height(cell: Tuple[int, int]) -> float:
        cached = height_cache.get(cell)
        if cached is not None:
            return cached
        x, y = to_xy(cell)
        value = _terrain_height(terrain, map_size, x, y)
        height_cache[cell] = value
        return value

    def heuristic(cell: Tuple[int, int]) -> float:
        x, y = to_xy(cell)
        gx, gy = to_xy(goal_cell)
        return math.hypot(gx - x, gy - y)

    def edge_is_clear(source: Tuple[int, int], target: Tuple[int, int]) -> bool:
        """Reject graph edges that cut through an inflated spherical obstacle."""
        if centers.shape[0] == 0:
            return True
        start_xy = np.asarray(to_xy(source), dtype=np.float64)
        end_xy = np.asarray(to_xy(target), dtype=np.float64)
        edge = end_xy - start_xy
        squared_length = float(np.dot(edge, edge))
        if squared_length <= 1e-12:
            closest = np.repeat(start_xy[None, :], centers.shape[0], axis=0)
        else:
            projection = np.sum((centers - start_xy) * edge, axis=1)
            projection = np.clip(projection / squared_length, 0.0, 1.0)
            closest = start_xy[None, :] + projection[:, None] * edge[None, :]
        distances = np.linalg.norm(centers - closest, axis=1)
        return bool(np.all(distances > radii + obstacle_inflation))

    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    frontier: List[Tuple[float, float, Tuple[int, int]]] = [(heuristic(start_cell), 0.0, start_cell)]
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    best_cost: Dict[Tuple[int, int], float] = {start_cell: 0.0}
    expanded = 0
    found = start_cell == goal_cell

    while frontier and not found:
        _, current_cost, cell = heapq.heappop(frontier)
        if current_cost > best_cost.get(cell, math.inf) + 1e-12:
            continue
        expanded += 1
        for dx, dy in neighbors:
            nxt = (cell[0] + dx, cell[1] + dy)
            if not (0 <= nxt[0] < grid_count and 0 <= nxt[1] < grid_count):
                continue
            if blocked[nxt]:
                continue
            if not edge_is_clear(cell, nxt):
                continue
            horizontal = grid_resolution * (math.sqrt(2.0) if dx and dy else 1.0)
            dz = height(nxt) - height(cell)
            segment_cost = math.hypot(horizontal, dz)
            candidate = current_cost + segment_cost
            if candidate + 1e-12 >= best_cost.get(nxt, math.inf):
                continue
            best_cost[nxt] = candidate
            parent[nxt] = cell
            if nxt == goal_cell:
                found = True
                break
            heapq.heappush(frontier, (candidate + heuristic(nxt), candidate, nxt))

    if not found:
        return ReferencePathPlan(
            metrics=ReferencePathMetrics(
                feasible=False,
                path_length=None,
                detour_ratio=None,
                maximum_climb=None,
                minimum_corridor_clearance=None,
                minimum_boundary_clearance=None,
                minimum_obstacle_clearance=None,
                grid_resolution=grid_resolution,
                expanded_nodes=expanded,
            ),
            waypoints=(),
        )

    cells = [goal_cell]
    while cells[-1] != start_cell:
        previous = parent.get(cells[-1])
        if previous is None:
            return ReferencePathPlan(
                metrics=ReferencePathMetrics(
                    False,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    grid_resolution,
                    expanded,
                ),
                waypoints=(),
            )
        cells.append(previous)
    cells.reverse()

    corridor_xy = np.asarray([to_xy(cell) for cell in cells], dtype=np.float64)
    corridor_heights = np.asarray(
        [height(cell) + surface_clearance for cell in cells],
        dtype=np.float64,
    )
    # Persist a physically executable 3-D reference: exact start -> safe
    # terrain-relative corridor -> exact goal.  The earlier surface-only metric
    # omitted the potentially dominant climb from the actual UAV start height.
    xy_path = np.concatenate(
        [start_array[None, :2], corridor_xy, goal_array[None, :2]], axis=0
    )
    path_heights = np.concatenate(
        [
            np.asarray([start_array[2]], dtype=np.float64),
            corridor_heights,
            np.asarray([goal_array[2]], dtype=np.float64),
        ]
    )
    keep = np.ones(xy_path.shape[0], dtype=bool)
    if xy_path.shape[0] > 1:
        duplicate_3d = (
            np.linalg.norm(np.diff(xy_path, axis=0), axis=1) <= 1e-12
        ) & (np.abs(np.diff(path_heights)) <= 1e-12)
        keep[1:] = ~duplicate_3d
        xy_path = xy_path[keep]
        path_heights = path_heights[keep]
    if xy_path.shape[0] == 1:
        path_length = 0.0
        maximum_climb = 0.0
    else:
        dxy = np.linalg.norm(np.diff(xy_path, axis=0), axis=1)
        dz = np.diff(path_heights)
        path_length = float(np.sum(np.hypot(dxy, dz)))
        maximum_climb = float(np.sum(np.maximum(dz, 0.0)))

    euclidean_3d = float(np.linalg.norm(start_array[:3] - goal_array[:3]))
    detour_ratio = path_length / max(euclidean_3d, 1e-6)
    boundary_clearance = float(
        np.min(
            np.stack(
                [
                    xy_path[:, 0],
                    xy_path[:, 1],
                    map_size - xy_path[:, 0],
                    map_size - xy_path[:, 1],
                ],
                axis=1,
            )
        )
    )
    obstacle_clearance = None
    minimum_clearance = boundary_clearance
    if centers.shape[0] > 0:
        obstacle_clearance = _polyline_obstacle_surface_clearance(
            xy_path, centers, radii
        )
        minimum_clearance = min(minimum_clearance, obstacle_clearance)

    waypoints = tuple(
        (float(xy[0]), float(xy[1]), float(surface_z))
        for xy, surface_z in zip(xy_path, path_heights)
    )
    return ReferencePathPlan(
        metrics=ReferencePathMetrics(
            feasible=True,
            path_length=path_length,
            detour_ratio=float(detour_ratio),
            maximum_climb=maximum_climb,
            minimum_corridor_clearance=float(minimum_clearance),
            minimum_boundary_clearance=float(boundary_clearance),
            minimum_obstacle_clearance=obstacle_clearance,
            grid_resolution=grid_resolution,
            expanded_nodes=expanded,
        ),
        waypoints=waypoints,
    )


def reference_path_plan(
    start: Sequence[float],
    goal: Sequence[float],
    *,
    map_size: float,
    terrain: Optional[np.ndarray] = None,
    obstacles: Optional[Iterable[Any]] = None,
    grid_resolution: float = 4.0,
    obstacle_inflation: float = 1.5,
    boundary_clearance: float = 0.0,
    surface_clearance: float = 4.5,
) -> ReferencePathPlan:
    """Return the exact path used to derive the published task metrics."""
    return _reference_path_plan(
        start,
        goal,
        map_size=map_size,
        terrain=terrain,
        obstacles=obstacles,
        grid_resolution=grid_resolution,
        obstacle_inflation=obstacle_inflation,
        boundary_clearance=boundary_clearance,
        surface_clearance=surface_clearance,
    )


def reference_path_metrics(
    start: Sequence[float],
    goal: Sequence[float],
    *,
    map_size: float,
    terrain: Optional[np.ndarray] = None,
    obstacles: Optional[Iterable[Any]] = None,
    grid_resolution: float = 4.0,
    obstacle_inflation: float = 1.5,
    boundary_clearance: float = 0.0,
    surface_clearance: float = 4.5,
) -> ReferencePathMetrics:
    """Compute deterministic 2.5-D A* task metrics for one agent."""
    return reference_path_plan(
        start,
        goal,
        map_size=map_size,
        terrain=terrain,
        obstacles=obstacles,
        grid_resolution=grid_resolution,
        obstacle_inflation=obstacle_inflation,
        boundary_clearance=boundary_clearance,
        surface_clearance=surface_clearance,
    ).metrics


def build_task_layout(
    *,
    agent_starts: Sequence[Sequence[float]],
    agent_goals: Sequence[Sequence[float]],
    central_goal: Sequence[float],
    map_size: float,
    min_start_goal_dist: float,
    max_start_goal_dist: float,
    terrain: Optional[np.ndarray] = None,
    obstacles: Optional[Iterable[Any]] = None,
    terrain_seed: Optional[int] = None,
    terrain_variant_seed: Optional[int] = None,
    obstacle_seed: Optional[int] = None,
    position_seed: Optional[int] = None,
    episode_index: Optional[int] = None,
    env_id: Optional[int] = None,
    source: str = "scenario_reset",
    difficulty_bin: Optional[str] = None,
    grid_resolution: float = 4.0,
    obstacle_inflation: float = 1.5,
    reference_surface_clearance: float = 4.5,
    success_distance_threshold: float = 2.0,
    collision_distance_threshold: float = 0.5,
    goal_formation_radius: Optional[float] = None,
    goal_boundary_clearance: Optional[float] = None,
    deadline_policy: Optional[Mapping[str, Any]] = None,
    simulation_dt_seconds: float = 0.08,
    require_reference_feasible: bool = True,
) -> Dict[str, Any]:
    """Build and validate one resolved episode task artifact."""
    starts = _positions(agent_starts, name="agent_starts")
    goals = _positions(agent_goals, name="agent_goals")
    if starts.shape[0] != goals.shape[0]:
        raise TaskLayoutError("agent_starts and agent_goals count mismatch")
    central = np.asarray(central_goal, dtype=np.float64).reshape(-1)
    if central.size < 3 or not np.isfinite(central[:3]).all():
        raise TaskLayoutError("central_goal must be one finite 3-D coordinate")
    distances_xy = validate_distance_bounds(
        starts,
        goals,
        min_distance=min_start_goal_dist,
        max_distance=max_start_goal_dist,
    )
    distances_3d = np.linalg.norm(starts - goals, axis=1)
    obstacle_list = list(obstacles or [])
    success_distance_threshold = float(success_distance_threshold)
    collision_distance_threshold = float(collision_distance_threshold)
    if (
        not math.isfinite(success_distance_threshold)
        or success_distance_threshold <= 0.0
        or not math.isfinite(collision_distance_threshold)
        or collision_distance_threshold < 0.0
    ):
        raise TaskLayoutError(
            "success/collision distance thresholds are invalid"
        )

    reference_metrics = [
        reference_path_metrics(
            starts[index],
            goals[index],
            map_size=map_size,
            terrain=terrain,
            obstacles=obstacle_list,
            grid_resolution=grid_resolution,
            obstacle_inflation=obstacle_inflation,
            boundary_clearance=(
                0.0 if goal_boundary_clearance is None else goal_boundary_clearance
            ),
            surface_clearance=reference_surface_clearance,
        )
        for index in range(starts.shape[0])
    ]
    infeasible = [index for index, metrics in enumerate(reference_metrics) if not metrics.feasible]
    if require_reference_feasible and infeasible:
        raise TaskLayoutError(
            "reference planner found disconnected agent tasks: "
            + ", ".join(f"agent_{index}" for index in infeasible)
        )

    task = {
        "schema_version": TASK_LAYOUT_SCHEMA_VERSION,
        "source": str(source),
        "episode_index": None if episode_index is None else int(episode_index),
        "env_id": None if env_id is None else int(env_id),
        "map_size": float(map_size),
        "terrain_sha256": _terrain_sha256(terrain),
        "obstacles": _serialized_obstacles(obstacle_list),
        "seeds": {
            "terrain": None if terrain_seed is None else int(terrain_seed),
            "terrain_variant": None if terrain_variant_seed is None else int(terrain_variant_seed),
            "obstacle": None if obstacle_seed is None else int(obstacle_seed),
            "position": None if position_seed is None else int(position_seed),
        },
        "contract": {
            "distance_semantics": "per_agent_horizontal_xy",
            "min_start_goal_dist": float(min_start_goal_dist),
            "max_start_goal_dist": float(max_start_goal_dist),
            "require_reference_feasible": bool(require_reference_feasible),
            "reference_grid_resolution": float(grid_resolution),
            "reference_obstacle_inflation": float(obstacle_inflation),
            "reference_surface_clearance": float(
                reference_surface_clearance
            ),
            "success_distance_threshold": success_distance_threshold,
            "collision_distance_threshold": collision_distance_threshold,
            "reference_boundary_clearance": (
                0.0 if goal_boundary_clearance is None else float(goal_boundary_clearance)
            ),
            "goal_formation_radius": (
                None if goal_formation_radius is None else float(goal_formation_radius)
            ),
            "goal_boundary_clearance": (
                None if goal_boundary_clearance is None else float(goal_boundary_clearance)
            ),
        },
        "agent_starts": starts,
        "central_goal": central[:3],
        "agent_goals": goals,
        "metrics": {
            "horizontal_distances": distances_xy,
            "distances_3d": distances_3d,
            "reference_paths": [metrics.to_dict() for metrics in reference_metrics],
            "all_reference_feasible": not infeasible,
        },
        "difficulty_bin": difficulty_bin,
    }
    difficulty_summary = task_difficulty_summary(task)
    task["metrics"]["difficulty_summary"] = difficulty_summary
    if deadline_policy is not None:
        task["contract"]["deadline"] = task_deadline_contract(
            difficulty_summary["maximum_reference_path_length_m"],
            simulation_dt_seconds=simulation_dt_seconds,
            policy=deadline_policy,
        )
    task = _json_ready(task)
    task["task_id"] = task_id_for_payload(task)
    task["content_sha256"] = canonical_sha256(task)
    return task


def validate_task_layout_payload(payload: Mapping[str, Any]) -> List[str]:
    """Return all schema/content errors without mutating the artifact."""
    errors: List[str] = []
    if not isinstance(payload, Mapping):
        return ["task layout must be a JSON object"]
    if payload.get("schema_version") != TASK_LAYOUT_SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {TASK_LAYOUT_SCHEMA_VERSION!r}, got {payload.get('schema_version')!r}"
        )
    expected_hash = canonical_sha256(payload)
    if payload.get("content_sha256") != expected_hash:
        errors.append("content_sha256 mismatch")
    task_id = payload.get("task_id")
    if not isinstance(task_id, str) or len(task_id) != 24:
        errors.append("task_id must be the 24-character canonical task identifier")
    elif task_id != task_id_for_payload(payload):
        errors.append("task_id physical identity mismatch")
    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        errors.append("contract must be an object")
        return errors
    try:
        reference_surface_clearance = float(
            contract["reference_surface_clearance"]
        )
        success_distance_threshold = float(
            contract["success_distance_threshold"]
        )
        collision_distance_threshold = float(
            contract["collision_distance_threshold"]
        )
        if (
            not math.isfinite(reference_surface_clearance)
            or reference_surface_clearance <= 0.0
            or not math.isfinite(success_distance_threshold)
            or success_distance_threshold <= 0.0
            or not math.isfinite(collision_distance_threshold)
            or collision_distance_threshold < 0.0
        ):
            raise ValueError("values must satisfy clearance>0, success>0, collision>=0")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"invalid task semantics contract: {exc}")
    try:
        validate_distance_bounds(
            payload.get("agent_starts", []),
            payload.get("agent_goals", []),
            min_distance=float(contract["min_start_goal_dist"]),
            max_distance=float(contract["max_start_goal_dist"]),
        )
    except Exception as exc:
        errors.append(str(exc))
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        errors.append("metrics must be an object")
    elif bool(contract.get("require_reference_feasible", False)) and not bool(
        metrics.get("all_reference_feasible", False)
    ):
        errors.append("reference feasibility contract is not satisfied")
    if isinstance(metrics, Mapping):
        try:
            expected_summary = task_difficulty_summary(payload)
            if metrics.get("difficulty_summary") != expected_summary:
                errors.append("difficulty_summary mismatch")
        except Exception as exc:
            errors.append(f"invalid difficulty summary inputs: {exc}")
    return errors


def task_difficulty_summary(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Derive compact, deterministic difficulty features from one resolved task."""
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TaskLayoutError("task metrics must be an object")
    paths = list(metrics.get("reference_paths") or [])
    if not paths:
        raise TaskLayoutError("task requires reference path metrics")
    if any(not bool(path.get("feasible", False)) for path in paths):
        raise TaskLayoutError("difficulty summary requires feasible reference paths")

    def values(name: str) -> List[float]:
        result = []
        for index, path in enumerate(paths):
            raw = path.get(name)
            if raw is None:
                raise TaskLayoutError(f"reference_paths[{index}].{name} is missing")
            value = float(raw)
            if not math.isfinite(value):
                raise TaskLayoutError(f"reference_paths[{index}].{name} is non-finite")
            result.append(value)
        return result

    raw_horizontal = metrics.get("horizontal_distances")
    horizontal = (
        []
        if raw_horizontal is None
        else [float(value) for value in list(raw_horizontal)]
    )
    if len(horizontal) != len(paths) or not all(math.isfinite(value) for value in horizontal):
        raise TaskLayoutError("horizontal distance metrics are missing or misaligned")
    path_lengths = values("path_length")
    detours = values("detour_ratio")
    climbs = values("maximum_climb")
    boundary = values("minimum_boundary_clearance")
    obstacle_values = [
        float(path["minimum_obstacle_clearance"])
        for path in paths
        if path.get("minimum_obstacle_clearance") is not None
    ]
    if obstacle_values and not all(math.isfinite(value) for value in obstacle_values):
        raise TaskLayoutError("obstacle clearance metrics contain non-finite values")
    return {
        "agent_count": len(paths),
        "obstacle_count": len(list(payload.get("obstacles") or [])),
        "minimum_horizontal_distance_m": float(min(horizontal)),
        "maximum_horizontal_distance_m": float(max(horizontal)),
        "mean_reference_path_length_m": float(np.mean(path_lengths)),
        "maximum_reference_path_length_m": float(max(path_lengths)),
        "maximum_detour_ratio": float(max(detours)),
        "maximum_cumulative_climb_m": float(max(climbs)),
        "minimum_boundary_clearance_m": float(min(boundary)),
        "minimum_obstacle_clearance_m": (
            None if not obstacle_values else float(min(obstacle_values))
        ),
    }


def validate_task_difficulty_acceptance(
    payload: Mapping[str, Any], acceptance: Mapping[str, Any]
) -> List[str]:
    """Validate one task against an explicit stage difficulty envelope."""
    if not isinstance(acceptance, Mapping):
        return ["difficulty_acceptance must be an object"]
    bounds = acceptance.get("summary_bounds")
    if not isinstance(bounds, Mapping) or not bounds:
        return ["difficulty_acceptance.summary_bounds must be a non-empty object"]
    try:
        summary = task_difficulty_summary(payload)
    except Exception as exc:
        return [str(exc)]
    errors: List[str] = []
    for metric_name, raw_bound in bounds.items():
        if metric_name not in summary:
            errors.append(f"unsupported difficulty metric={metric_name!r}")
            continue
        if not isinstance(raw_bound, Mapping):
            errors.append(f"difficulty bound for {metric_name} must be an object")
            continue
        value = summary[metric_name]
        if value is None:
            errors.append(f"difficulty metric {metric_name} is unavailable")
            continue
        if "minimum" in raw_bound:
            minimum = float(raw_bound["minimum"])
            if float(value) < minimum:
                errors.append(f"{metric_name}={value} below minimum={minimum}")
        if "maximum" in raw_bound:
            maximum = float(raw_bound["maximum"])
            if float(value) > maximum:
                errors.append(f"{metric_name}={value} above maximum={maximum}")
        if "minimum" not in raw_bound and "maximum" not in raw_bound:
            errors.append(f"difficulty bound for {metric_name} has no minimum/maximum")
    return errors


def task_deadline_contract(
    maximum_reference_path_length_m: float,
    *,
    simulation_dt_seconds: float,
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    """Convert a frozen reference path into a bounded finite-horizon deadline."""
    if not isinstance(policy, Mapping):
        raise TaskLayoutError("task deadline policy must be an object")
    policy_id = str(policy.get("policy_id", "")).strip()
    if policy_id != "reference_path_slack_v1":
        raise TaskLayoutError(f"unsupported task deadline policy={policy_id!r}")
    path_length = float(maximum_reference_path_length_m)
    dt = float(simulation_dt_seconds)
    nominal_speed = float(policy.get("nominal_speed_mps"))
    slack = float(policy.get("path_time_slack"))
    overhead = float(policy.get("fixed_overhead_seconds"))
    minimum_seconds = float(policy.get("minimum_seconds"))
    maximum_seconds = float(policy.get("maximum_seconds"))
    values = (path_length, dt, nominal_speed, slack, overhead, minimum_seconds, maximum_seconds)
    if not all(math.isfinite(value) for value in values):
        raise TaskLayoutError("task deadline policy contains non-finite values")
    if (
        path_length <= 0.0
        or dt <= 0.0
        or nominal_speed <= 0.0
        or slack <= 0.0
        or overhead < 0.0
        or minimum_seconds <= 0.0
        or maximum_seconds < minimum_seconds
    ):
        raise TaskLayoutError(f"invalid task deadline policy={dict(policy)!r}")
    raw_seconds = overhead + (path_length / nominal_speed) * slack
    bounded_seconds = float(np.clip(raw_seconds, minimum_seconds, maximum_seconds))
    deadline_steps = max(1, int(math.ceil(bounded_seconds / dt)))
    return {
        "policy_id": policy_id,
        "nominal_speed_mps": nominal_speed,
        "path_time_slack": slack,
        "fixed_overhead_seconds": overhead,
        "minimum_seconds": minimum_seconds,
        "maximum_seconds": maximum_seconds,
        "simulation_dt_seconds": dt,
        "maximum_reference_path_length_m": path_length,
        "unclipped_deadline_seconds": float(raw_seconds),
        "deadline_steps": deadline_steps,
        "deadline_seconds": float(deadline_steps * dt),
    }


def validate_task_deadline_contract(
    payload: Mapping[str, Any], expected_policy: Mapping[str, Any]
) -> List[str]:
    """Validate the persisted per-task deadline against the bank policy."""
    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        return ["task contract is missing"]
    actual = contract.get("deadline")
    if not isinstance(actual, Mapping):
        return ["task contract deadline is missing"]
    try:
        expected_dt = float(expected_policy.get("simulation_dt_seconds"))
        expected = task_deadline_contract(
            task_difficulty_summary(payload)["maximum_reference_path_length_m"],
            simulation_dt_seconds=expected_dt,
            policy=expected_policy,
        )
    except Exception as exc:
        return [f"invalid expected deadline policy: {exc}"]
    return [] if dict(actual) == expected else ["task deadline contract mismatch"]


__all__ = [
    "TASK_LAYOUT_SCHEMA_VERSION",
    "REFERENCE_PATH_SCHEMA_VERSION",
    "TaskLayoutError",
    "ReferencePathPlan",
    "ReferencePathMetrics",
    "atomic_write_json",
    "build_task_layout",
    "canonical_sha256",
    "horizontal_distances",
    "reference_path_plan",
    "reference_path_metrics",
    "task_identity_payload",
    "task_id_for_payload",
    "task_difficulty_summary",
    "task_deadline_contract",
    "validate_distance_bounds",
    "validate_task_difficulty_acceptance",
    "validate_task_deadline_contract",
    "validate_task_layout_payload",
]
