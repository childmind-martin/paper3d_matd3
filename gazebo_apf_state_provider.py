#!/usr/bin/env python3
"""Gazebo-authoritative state binding for the standalone APF backend."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _as_vec3(value: Any, default: Optional[Sequence[float]] = None) -> np.ndarray:
    if default is None:
        default = (0.0, 0.0, 0.0)
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
            return arr[:3].astype(np.float64, copy=True)
    except Exception:
        pass
    return np.asarray(default, dtype=np.float64).reshape(3).copy()


def _as_quat_wxyz(value: Any) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size >= 4 and np.all(np.isfinite(arr[:4])):
            norm = float(np.linalg.norm(arr[:4]))
            if norm > 1e-9:
                return (arr[:4] / norm).astype(np.float64, copy=True)
    except Exception:
        pass
    return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _safe_float(value: Any, default: float) -> float:
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return float(default)


@dataclass
class GazeboAPFAgentState:
    name: str
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    orientation_wxyz: np.ndarray
    contact: bool
    radius: float
    max_speed: float
    accel: float
    mass: float = 1.0


@dataclass
class GazeboAPFObstacle:
    name: str
    type: str
    center: np.ndarray
    radius: float
    raw: Dict[str, Any]


@dataclass
class GazeboAPFSceneState:
    agents: List[GazeboAPFAgentState]
    goals: np.ndarray
    obstacles: List[GazeboAPFObstacle]
    map_size: float
    terrain: "ScenarioTerrainSampler"
    collision_distance_threshold: float
    scenario_seed: Optional[int] = None
    terrain_seed: Optional[int] = None
    obstacle_seed: Optional[int] = None
    frame: Optional[int] = None
    source: str = "unknown"

    @property
    def positions(self) -> np.ndarray:
        return np.asarray([agent.position for agent in self.agents], dtype=np.float64)

    @property
    def velocities(self) -> np.ndarray:
        return np.asarray([agent.velocity for agent in self.agents], dtype=np.float64)

    @property
    def accelerations(self) -> np.ndarray:
        return np.asarray([agent.acceleration for agent in self.agents], dtype=np.float64)

    @property
    def agent_radii(self) -> np.ndarray:
        return np.asarray([agent.radius for agent in self.agents], dtype=np.float64)

    @property
    def max_speeds(self) -> np.ndarray:
        return np.asarray([max(agent.max_speed, 1e-6) for agent in self.agents], dtype=np.float64)


class ScenarioTerrainSampler:
    """Bilinear terrain sampler backed by exported scenario data or a live scenario."""

    def __init__(
        self,
        terrain: Optional[np.ndarray],
        map_size: float,
        x_coords: Optional[Sequence[float]] = None,
        y_coords: Optional[Sequence[float]] = None,
        scenario: Optional[Any] = None,
    ) -> None:
        self.terrain = None if terrain is None else np.asarray(terrain, dtype=np.float32)
        self.map_size = float(map_size)
        self.x_coords = None if x_coords is None else np.asarray(x_coords, dtype=np.float64)
        self.y_coords = None if y_coords is None else np.asarray(y_coords, dtype=np.float64)
        self.scenario = scenario

    @classmethod
    def from_scenario_json(cls, scenario_json: Optional[Path], scenario: Optional[Any] = None) -> "ScenarioTerrainSampler":
        if scenario_json is None:
            map_size = _safe_float(getattr(scenario, "map_size", 200.0), 200.0)
            terrain = getattr(scenario, "terrain", None)
            return cls(terrain=terrain, map_size=map_size, scenario=scenario)

        scenario_json = Path(scenario_json).expanduser().resolve()
        with scenario_json.open("r", encoding="utf-8") as f:
            snapshot = json.load(f)
        map_size = _safe_float(snapshot.get("map_size"), _safe_float(getattr(scenario, "map_size", 200.0), 200.0))
        terrain_meta = snapshot.get("terrain", {}) if isinstance(snapshot.get("terrain"), dict) else {}
        dense_path = terrain_meta.get("dense_npy") or terrain_meta.get("sampled_npy")
        terrain = None
        if dense_path:
            terrain_path = Path(str(dense_path))
            if not terrain_path.is_absolute():
                terrain_path = scenario_json.parent / terrain_path
            if terrain_path.exists():
                terrain = np.load(terrain_path)
        dense_coords = terrain_meta.get("dense_coordinates", {}) if isinstance(terrain_meta.get("dense_coordinates"), dict) else {}
        x_coords = dense_coords.get("x")
        y_coords = dense_coords.get("y")
        return cls(terrain=terrain, map_size=map_size, x_coords=x_coords, y_coords=y_coords, scenario=scenario)

    def height(self, x: float, y: float) -> float:
        if self.scenario is not None and hasattr(self.scenario, "get_terrain_height"):
            try:
                return float(self.scenario.get_terrain_height(float(x), float(y)))
            except Exception:
                pass
        if self.terrain is None or self.terrain.size == 0:
            return 0.0
        arr = self.terrain
        h, w = arr.shape[:2]
        if self.x_coords is not None and self.x_coords.size == w and w > 1:
            ix = np.interp(float(x), self.x_coords, np.arange(w, dtype=np.float64))
        else:
            ix = float(x) * float(max(w - 1, 0)) / max(self.map_size - 1.0, 1e-9)
        if self.y_coords is not None and self.y_coords.size == h and h > 1:
            iy = np.interp(float(y), self.y_coords, np.arange(h, dtype=np.float64))
        else:
            iy = float(y) * float(max(h - 1, 0)) / max(self.map_size - 1.0, 1e-9)
        ix = float(np.clip(ix, 0.0, max(w - 1, 0)))
        iy = float(np.clip(iy, 0.0, max(h - 1, 0)))
        x0 = int(np.floor(ix))
        x1 = int(np.ceil(ix))
        y0 = int(np.floor(iy))
        y1 = int(np.ceil(iy))
        wx = ix - x0
        wy = iy - y0
        v00 = float(arr[y0, x0])
        v10 = float(arr[y0, x1])
        v01 = float(arr[y1, x0])
        v11 = float(arr[y1, x1])
        return float((1.0 - wx) * (1.0 - wy) * v00 + wx * (1.0 - wy) * v10 + (1.0 - wx) * wy * v01 + wx * wy * v11)

    def batch_height(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        if self.scenario is not None and hasattr(self.scenario, "batch_get_terrain_height"):
            try:
                return np.asarray(self.scenario.batch_get_terrain_height(coords[:, :2]), dtype=np.float32).reshape(-1)
            except Exception:
                pass
        return np.asarray([self.height(float(x), float(y)) for x, y in coords[:, :2]], dtype=np.float32)


class GazeboAPFStateProvider:
    """Build APF state from Gazebo feedback plus exported scenario binding."""

    _GRADIENT_OFFSETS = np.asarray([[3.0, 0.0], [0.0, 3.0], [-3.0, 0.0], [0.0, -3.0]], dtype=np.float32)
    _FORWARD_DISTANCES = np.asarray([2, 4, 6, 10, 15, 20, 25, 30], dtype=np.float32)
    _DIRECTION_PAIRS = np.asarray(
        [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)],
        dtype=np.float32,
    )
    _NEAR_DISTANCE = 5.0
    _FAR_DISTANCE = 12.0

    def __init__(
        self,
        scenario_json: Optional[Path] = None,
        scenario: Optional[Any] = None,
        world: Optional[Any] = None,
        agent_count: Optional[int] = None,
        state_file: Optional[Path] = None,
        contact_flag_file: Optional[Path] = None,
        agent_prefix: str = "dynamic_agent_",
        state_feedback_dt: float = 0.08,
        feedback_velocity_mode: Optional[str] = None,
        feedback_acceleration_mode: Optional[str] = None,
    ) -> None:
        self.scenario_json = Path(scenario_json).expanduser().resolve() if scenario_json else None
        self.scenario = scenario
        self.world = world
        self.agent_count = int(agent_count) if agent_count is not None else None
        self.state_file = Path(state_file).expanduser().resolve() if state_file else None
        self.contact_flag_file = Path(contact_flag_file).expanduser().resolve() if contact_flag_file else None
        self.agent_prefix = str(agent_prefix or "dynamic_agent_")
        self.state_feedback_dt = max(1e-9, float(state_feedback_dt))
        vel_mode = str(feedback_velocity_mode or os.getenv("GAZEBO_LIVE_FEEDBACK_VELOCITY_MODE", "clamp")).strip().lower()
        if vel_mode not in ("preserve", "estimate", "clamp"):
            vel_mode = "clamp"
        self.feedback_velocity_mode = vel_mode
        acc_mode = str(feedback_acceleration_mode or os.getenv("GAZEBO_LIVE_FEEDBACK_ACCELERATION_MODE", "estimate")).strip().lower()
        if acc_mode not in ("preserve", "estimate", "zero"):
            acc_mode = "estimate"
        self.feedback_acceleration_mode = acc_mode
        self.feedback_max_speed_scale = max(0.01, _safe_float(os.getenv("GAZEBO_LIVE_FEEDBACK_MAX_SPEED_SCALE"), 1.0))
        self.feedback_max_accel_scale = max(0.0, _safe_float(os.getenv("GAZEBO_LIVE_FEEDBACK_MAX_ACCEL_SCALE"), 0.0))
        self._last_frame: Optional[int] = None
        self._last_positions: Optional[np.ndarray] = None
        self._last_velocities: Optional[np.ndarray] = None
        self._last_time: Optional[float] = None
        self._snapshot: Dict[str, Any] = {}
        if self.scenario_json is not None and self.scenario_json.exists():
            with self.scenario_json.open("r", encoding="utf-8") as f:
                self._snapshot = json.load(f)
        self.map_size = _safe_float(
            self._snapshot.get("map_size"),
            _safe_float(getattr(scenario, "map_size", getattr(world, "map_size", 200.0)), 200.0),
        )
        self.terrain = ScenarioTerrainSampler.from_scenario_json(self.scenario_json, scenario=scenario)
        self.obstacles = self._load_obstacles()
        self.goals = self._load_goals()
        self.start_agents = self._load_start_agents()
        self.collision_distance_threshold = _safe_float(
            (self._snapshot.get("collision") or {}).get("collision_distance_threshold")
            if isinstance(self._snapshot.get("collision"), dict)
            else None,
            _safe_float(getattr(scenario, "collision_distance_threshold", 0.5), 0.5),
        )

    @classmethod
    def from_runtime(
        cls,
        scenario_json: Optional[Any] = None,
        scenario: Optional[Any] = None,
        world: Optional[Any] = None,
        agent_count: Optional[int] = None,
        state_file: Optional[Any] = None,
        contact_flag_file: Optional[Any] = None,
        agent_prefix: Optional[str] = None,
        state_feedback_dt: Optional[float] = None,
        feedback_velocity_mode: Optional[str] = None,
        feedback_acceleration_mode: Optional[str] = None,
    ) -> "GazeboAPFStateProvider":
        return cls(
            scenario_json=Path(scenario_json) if scenario_json else None,
            scenario=scenario,
            world=world,
            agent_count=agent_count,
            state_file=Path(state_file) if state_file else Path(os.getenv("GAZEBO_LIVE_STATE_FILE", "")) if os.getenv("GAZEBO_LIVE_STATE_FILE") else None,
            contact_flag_file=Path(contact_flag_file) if contact_flag_file else Path(os.getenv("GAZEBO_LIVE_CONTACT_FLAG_FILE", "")) if os.getenv("GAZEBO_LIVE_CONTACT_FLAG_FILE") else None,
            agent_prefix=agent_prefix or os.getenv("GAZEBO_LIVE_AGENT_PREFIX", "dynamic_agent_"),
            state_feedback_dt=state_feedback_dt if state_feedback_dt is not None else _safe_float(os.getenv("GAZEBO_LIVE_STATE_FEEDBACK_DT"), 0.08),
            feedback_velocity_mode=feedback_velocity_mode,
            feedback_acceleration_mode=feedback_acceleration_mode,
        )

    def _load_start_agents(self) -> List[Dict[str, Any]]:
        starts = self._snapshot.get("start_positions", [])
        if isinstance(starts, list) and starts:
            return [s for s in starts if isinstance(s, dict)]
        agents = list(getattr(self.world, "agents", []) or [])
        records = []
        for idx, agent in enumerate(agents):
            state = getattr(agent, "state", None)
            records.append(
                {
                    "name": str(getattr(agent, "name", f"agent_{idx}")),
                    "position": _as_vec3(getattr(state, "p_pos", None)).tolist(),
                    "initial_velocity": _as_vec3(getattr(state, "p_vel", None)).tolist(),
                    "initial_orientation_wxyz": _as_quat_wxyz(getattr(state, "orientation", None)).tolist(),
                    "agent_size": _safe_float(getattr(agent, "size", 0.5), 0.5),
                    "max_speed": _safe_float(getattr(agent, "max_speed", 37.5), 37.5),
                    "accel": _safe_float(getattr(agent, "accel", 3.6), 3.6),
                    "mass": _safe_float(getattr(agent, "mass", 1.0), 1.0),
                }
            )
        return records

    def _load_goals(self) -> np.ndarray:
        goals: List[np.ndarray] = []
        raw_goals = self._snapshot.get("agent_goals", [])
        if isinstance(raw_goals, list):
            for item in raw_goals:
                if isinstance(item, dict) and item.get("position") is not None:
                    goals.append(_as_vec3(item.get("position")))
        if not goals and isinstance(self._snapshot.get("goal"), dict):
            center_goal = _as_vec3(self._snapshot["goal"].get("position"))
            count = self.agent_count or len(self._load_start_agents()) or 1
            goals = [center_goal.copy() for _ in range(count)]
        if not goals and self.world is not None:
            for agent in getattr(self.world, "agents", []) or []:
                goal = getattr(agent, "goal_a", None)
                goals.append(_as_vec3(getattr(getattr(goal, "state", None), "p_pos", None)))
        if not goals:
            count = self.agent_count or 1
            goals = [np.zeros(3, dtype=np.float64) for _ in range(count)]
        return np.asarray(goals, dtype=np.float64)

    def _load_obstacles(self) -> List[GazeboAPFObstacle]:
        obstacles: List[GazeboAPFObstacle] = []
        raw_obstacles = self._snapshot.get("obstacles", [])
        if not isinstance(raw_obstacles, list) and self.scenario is not None:
            raw_obstacles = list(getattr(self.scenario, "obstacles", []) or [])
        for idx, raw in enumerate(raw_obstacles or []):
            try:
                if not isinstance(raw, dict):
                    center = _as_vec3(getattr(getattr(raw, "state", None), "p_pos", None))
                    radius = _safe_float(getattr(raw, "radius", getattr(raw, "size", 1.0)), 1.0)
                    name = str(getattr(raw, "name", f"obstacle_{idx}"))
                    obstacles.append(GazeboAPFObstacle(name=name, type="sphere", center=center, radius=radius, raw={}))
                    continue
                name = str(raw.get("name", f"obstacle_{idx}"))
                obs_type = str(raw.get("type", "sphere")).lower()
                center = _as_vec3(raw.get("center", raw.get("position", raw.get("pos"))))
                if obs_type == "box":
                    size = np.asarray(raw.get("size", [1.0, 1.0, 1.0]), dtype=np.float64).reshape(-1)
                    radius = float(np.linalg.norm(size[:3]) * 0.5) if size.size >= 3 else 1.0
                elif obs_type == "cylinder":
                    radius = _safe_float(raw.get("radius"), 1.0)
                    length = _safe_float(raw.get("length", raw.get("height", 0.0)), 0.0)
                    radius = float(np.sqrt(radius * radius + (0.5 * length) ** 2))
                else:
                    radius = _safe_float(raw.get("radius", raw.get("r", raw.get("size"))), 1.0)
                obstacles.append(GazeboAPFObstacle(name=name, type=obs_type, center=center, radius=max(radius, 1e-6), raw=dict(raw)))
            except Exception:
                continue
        return obstacles

    def _agent_template(self, idx: int, agents: Optional[Sequence[Any]]) -> Dict[str, Any]:
        if idx < len(self.start_agents):
            template = dict(self.start_agents[idx])
        else:
            template = {}
        if agents is not None and idx < len(agents):
            agent = agents[idx]
            state = getattr(agent, "state", None)
            template.setdefault("name", str(getattr(agent, "name", f"agent_{idx}")))
            template.setdefault("position", _as_vec3(getattr(state, "p_pos", None)).tolist())
            template.setdefault("initial_velocity", _as_vec3(getattr(state, "p_vel", None)).tolist())
            template.setdefault("initial_orientation_wxyz", _as_quat_wxyz(getattr(state, "orientation", None)).tolist())
            template["agent_size"] = _safe_float(getattr(agent, "size", template.get("agent_size", 0.5)), 0.5)
            template["max_speed"] = _safe_float(getattr(agent, "max_speed", template.get("max_speed", 37.5)), 37.5)
            template["accel"] = _safe_float(getattr(agent, "accel", template.get("accel", 3.6)), 3.6)
            template["mass"] = _safe_float(getattr(agent, "mass", template.get("mass", 1.0)), 1.0)
        template.setdefault("name", f"agent_{idx}")
        template.setdefault("position", [0.0, 0.0, 0.0])
        template.setdefault("initial_velocity", [0.0, 0.0, 0.0])
        template.setdefault("initial_orientation_wxyz", [1.0, 0.0, 0.0, 0.0])
        template.setdefault("agent_size", 0.5)
        template.setdefault("max_speed", 37.5)
        template.setdefault("accel", 3.6)
        template.setdefault("mass", 1.0)
        return template

    def _agent_max_speed(self, idx: int, agents: Optional[Sequence[Any]]) -> float:
        template = self._agent_template(idx, agents)
        return max(_safe_float(template.get("max_speed"), 0.0), 0.0)

    def _agent_accel_limit(self, idx: int, agents: Optional[Sequence[Any]]) -> float:
        template = self._agent_template(idx, agents)
        return max(_safe_float(template.get("accel"), 0.0), 0.0)

    def _agent_vectors(self, agents: Optional[Sequence[Any]], field: str, count: int) -> Optional[np.ndarray]:
        if agents is None:
            return None
        values = []
        for agent in list(agents)[:count]:
            values.append(_as_vec3(getattr(getattr(agent, "state", None), field, None)))
        if len(values) < count:
            return None
        return np.asarray(values, dtype=np.float64)

    def _apply_velocity_mode(self, raw_velocities: np.ndarray, agents: Optional[Sequence[Any]]) -> np.ndarray:
        velocities = np.asarray(raw_velocities, dtype=np.float64).copy()
        if self.feedback_velocity_mode == "clamp":
            for idx in range(velocities.shape[0]):
                limit = self._agent_max_speed(idx, agents) * self.feedback_max_speed_scale
                speed = float(np.linalg.norm(velocities[idx]))
                if limit > 0.0 and np.isfinite(speed) and speed > limit:
                    velocities[idx] = velocities[idx] / max(speed, 1e-9) * limit
        return np.where(np.isfinite(velocities), velocities, 0.0)

    def _apply_acceleration_mode(self, raw_accelerations: np.ndarray, agents: Optional[Sequence[Any]]) -> np.ndarray:
        accelerations = np.asarray(raw_accelerations, dtype=np.float64).copy()
        if self.feedback_acceleration_mode == "zero":
            return np.zeros_like(accelerations)
        if self.feedback_max_accel_scale > 0.0:
            for idx in range(accelerations.shape[0]):
                limit = self._agent_accel_limit(idx, agents) * self.feedback_max_accel_scale
                norm = float(np.linalg.norm(accelerations[idx]))
                if limit > 0.0 and np.isfinite(norm) and norm > limit:
                    accelerations[idx] = accelerations[idx] / max(norm, 1e-9) * limit
        return np.where(np.isfinite(accelerations), accelerations, 0.0)

    def _contact_indices(self) -> List[int]:
        if self.contact_flag_file is None:
            return []
        try:
            if not self.contact_flag_file.exists() or self.contact_flag_file.stat().st_size <= 0:
                return []
            text = self.contact_flag_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []
        indices = set()
        pattern = re.compile(re.escape(self.agent_prefix) + r"(\d+)")
        for match in pattern.finditer(text):
            try:
                idx = int(match.group(1))
            except Exception:
                continue
            if self.agent_count is None or 0 <= idx < self.agent_count:
                indices.add(idx)
        return sorted(indices)

    def read_gazebo_state_data(
        self,
        client: Optional[Any] = None,
        min_frame: Optional[int] = None,
        timeout: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        if client is not None and hasattr(client, "read_state"):
            try:
                return client.read_state(min_frame=min_frame, timeout=timeout)
            except Exception:
                return None
        if self.state_file is None:
            return None
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            try:
                if self.state_file.exists() and self.state_file.stat().st_size > 0:
                    data = json.loads(self.state_file.read_text(encoding="utf-8"))
                    frame = int(data.get("frame", -1))
                    if min_frame is None or frame >= int(min_frame):
                        return data
            except Exception:
                pass
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.005)

    def build_state(
        self,
        agents: Optional[Sequence[Any]] = None,
        positions: Optional[Sequence[Sequence[float]]] = None,
        velocities: Optional[Sequence[Sequence[float]]] = None,
        accelerations: Optional[Sequence[Sequence[float]]] = None,
        orientations_wxyz: Optional[Sequence[Sequence[float]]] = None,
        contacts: Optional[Sequence[bool]] = None,
        frame: Optional[int] = None,
        source: str = "python_fallback",
    ) -> GazeboAPFSceneState:
        if positions is None and agents is not None:
            positions = [_as_vec3(getattr(getattr(agent, "state", None), "p_pos", None)) for agent in agents]
        if positions is None:
            positions = [record.get("position", [0.0, 0.0, 0.0]) for record in self.start_agents]
        pos_arr = np.asarray([_as_vec3(v) for v in positions], dtype=np.float64)
        count = int(self.agent_count or pos_arr.shape[0] or len(self.start_agents) or self.goals.shape[0])
        if pos_arr.shape[0] < count:
            pad = np.zeros((count - pos_arr.shape[0], 3), dtype=np.float64)
            pos_arr = np.vstack([pos_arr, pad])
        pos_arr = pos_arr[:count]

        if velocities is None and agents is not None:
            velocities = [_as_vec3(getattr(getattr(agent, "state", None), "p_vel", None)) for agent in agents]
        vel_values = [] if velocities is None else list(velocities)
        vel_arr = np.asarray([_as_vec3(v) for v in vel_values], dtype=np.float64) if velocities is not None else np.zeros_like(pos_arr)
        if vel_arr.shape[0] < count:
            vel_arr = np.vstack([vel_arr, np.zeros((count - vel_arr.shape[0], 3), dtype=np.float64)])
        vel_arr = vel_arr[:count]

        if accelerations is None and agents is not None:
            accelerations = [_as_vec3(getattr(getattr(agent, "state", None), "p_acc", None)) for agent in agents]
        acc_values = [] if accelerations is None else list(accelerations)
        acc_arr = np.asarray([_as_vec3(v) for v in acc_values], dtype=np.float64) if accelerations is not None else np.zeros_like(pos_arr)
        if acc_arr.shape[0] < count:
            acc_arr = np.vstack([acc_arr, np.zeros((count - acc_arr.shape[0], 3), dtype=np.float64)])
        acc_arr = acc_arr[:count]

        if orientations_wxyz is None and agents is not None:
            orientations_wxyz = [_as_quat_wxyz(getattr(getattr(agent, "state", None), "orientation", None)) for agent in agents]
        quat_values = [] if orientations_wxyz is None else list(orientations_wxyz)
        quat_arr = [_as_quat_wxyz(v) for v in quat_values]
        while len(quat_arr) < count:
            quat_arr.append(_as_quat_wxyz(None))

        contact_set = set(self._contact_indices())
        if contacts is not None:
            for idx, contact in enumerate(contacts):
                if contact:
                    contact_set.add(idx)

        goal_arr = np.asarray(self.goals, dtype=np.float64)
        if goal_arr.shape[0] < count:
            last_goal = goal_arr[-1] if goal_arr.size else np.zeros(3, dtype=np.float64)
            goal_arr = np.vstack([goal_arr, np.tile(last_goal.reshape(1, 3), (count - goal_arr.shape[0], 1))])
        goal_arr = goal_arr[:count]

        agent_states: List[GazeboAPFAgentState] = []
        for idx in range(count):
            template = self._agent_template(idx, agents)
            agent_states.append(
                GazeboAPFAgentState(
                    name=str(template.get("name", f"agent_{idx}")),
                    position=pos_arr[idx].astype(np.float64, copy=True),
                    velocity=vel_arr[idx].astype(np.float64, copy=True),
                    acceleration=acc_arr[idx].astype(np.float64, copy=True),
                    orientation_wxyz=quat_arr[idx].astype(np.float64, copy=True),
                    contact=idx in contact_set,
                    radius=_safe_float(template.get("agent_size"), 0.5),
                    max_speed=_safe_float(template.get("max_speed"), 37.5),
                    accel=_safe_float(template.get("accel"), 3.6),
                    mass=max(_safe_float(template.get("mass"), 1.0), 1e-9),
                )
            )

        return GazeboAPFSceneState(
            agents=agent_states,
            goals=goal_arr,
            obstacles=list(self.obstacles),
            map_size=float(self.map_size),
            terrain=self.terrain,
            collision_distance_threshold=float(self.collision_distance_threshold),
            scenario_seed=self._snapshot.get("seed"),
            terrain_seed=self._snapshot.get("terrain_seed"),
            obstacle_seed=self._snapshot.get("obstacle_seed"),
            frame=frame,
            source=source,
        )

    def read_live_state(
        self,
        client: Optional[Any] = None,
        agents: Optional[Sequence[Any]] = None,
        timeout: float = 0.0,
        require_seen: bool = True,
    ) -> Optional[GazeboAPFSceneState]:
        min_frame = None
        data = self.read_gazebo_state_data(client=client, min_frame=min_frame, timeout=timeout)
        if not isinstance(data, dict):
            return None
        entries = data.get("agents", [])
        if not isinstance(entries, list):
            return None
        count = int(self.agent_count or len(entries))
        if len(entries) < count:
            return None
        positions: List[np.ndarray] = []
        orientations: List[np.ndarray] = []
        for idx in range(count):
            entry = entries[idx]
            if not isinstance(entry, dict):
                return None
            if require_seen and not bool(entry.get("seen", False)):
                return None
            pos = _as_vec3(entry.get("position"), default=None)
            if not np.all(np.isfinite(pos)):
                return None
            positions.append(pos)
            orientations.append(_as_quat_wxyz(entry.get("orientation_wxyz")))
        frame = int(data.get("frame", -1))
        pos_arr = np.asarray(positions, dtype=np.float64)
        velocities = None
        accelerations = None
        reused_frame = self._last_frame is not None and frame <= int(self._last_frame)
        if reused_frame:
            velocities = self._agent_vectors(agents, "p_vel", count)
            accelerations = self._agent_vectors(agents, "p_acc", count)
        elif self._last_positions is not None and self._last_positions.shape == pos_arr.shape:
            dt = self.state_feedback_dt
            raw_velocities = (pos_arr - self._last_positions) / max(dt, 1e-9)
            if self.feedback_velocity_mode == "preserve":
                velocities = self._agent_vectors(agents, "p_vel", count)
            else:
                velocities = self._apply_velocity_mode(raw_velocities, agents)
            if velocities is not None:
                if self.feedback_acceleration_mode == "preserve":
                    accelerations = self._agent_vectors(agents, "p_acc", count)
                elif self.feedback_acceleration_mode == "zero":
                    accelerations = np.zeros_like(velocities)
                elif self._last_velocities is not None and self._last_velocities.shape == velocities.shape:
                    raw_accelerations = (velocities - self._last_velocities) / max(dt, 1e-9)
                    accelerations = self._apply_acceleration_mode(raw_accelerations, agents)
        elif agents is not None:
            velocities = [_as_vec3(getattr(getattr(agent, "state", None), "p_vel", None)) for agent in agents]
            accelerations = [_as_vec3(getattr(getattr(agent, "state", None), "p_acc", None)) for agent in agents]
        state = self.build_state(
            agents=agents,
            positions=pos_arr,
            velocities=velocities,
            accelerations=accelerations,
            orientations_wxyz=orientations,
            frame=frame,
            source="gazebo_feedback_reused" if reused_frame else "gazebo_feedback",
        )
        self._last_frame = frame
        self._last_positions = pos_arr.copy()
        self._last_velocities = state.velocities.copy()
        self._last_time = time.monotonic()
        return state

    def apply_state_to_agents(self, state: GazeboAPFSceneState, agents: Sequence[Any]) -> None:
        for idx, agent_state in enumerate(state.agents[: len(agents)]):
            state_obj = getattr(agents[idx], "state", None)
            if state_obj is None:
                continue
            state_obj.p_pos = agent_state.position.astype(np.float64, copy=True)
            state_obj.p_vel = agent_state.velocity.astype(np.float64, copy=True)
            state_obj.p_acc = agent_state.acceleration.astype(np.float64, copy=True)
            state_obj.orientation = agent_state.orientation_wxyz.astype(np.float64, copy=True)

    def build_observations(self, state: GazeboAPFSceneState) -> np.ndarray:
        positions = state.positions.astype(np.float32)
        velocities = state.velocities.astype(np.float32)
        accelerations = state.accelerations.astype(np.float32)
        num_agents = positions.shape[0]
        map_half = max(float(state.map_size) * 0.5, 1e-6)
        max_speeds = np.maximum(state.max_speeds.astype(np.float32), 1e-6)

        state_info = np.concatenate(
            [
                positions / map_half - 1.0,
                velocities / max_speeds[:, None],
                np.clip(accelerations / 10.0, -1.0, 1.0),
            ],
            axis=1,
        ).astype(np.float32)

        goal_info = np.zeros((num_agents, 7), dtype=np.float32)
        for idx in range(num_agents):
            goal = state.goals[idx].astype(np.float32)
            rel = goal - positions[idx]
            dist = float(np.linalg.norm(rel))
            direction = rel / (dist + 1e-6)
            goal_info[idx, :3] = direction
            goal_info[idx, 3] = dist / max(float(state.map_size), 1e-6)
            goal_info[idx, 4:] = goal / map_half - 1.0

        terrain_info = self._build_terrain_info(state, positions, velocities)
        obstacle_info = self._build_obstacle_info(state, positions)
        other_info = self._build_other_agents_info(state, positions, velocities, max_speeds)
        env_info = np.asarray([num_agents / 10.0, -0.8, -0.4, 0.0, 0.4, 0.8], dtype=np.float32)
        obs = np.concatenate(
            [
                state_info,
                terrain_info,
                obstacle_info,
                goal_info,
                other_info,
                np.broadcast_to(env_info[None, :], (num_agents, env_info.size)),
            ],
            axis=1,
        )
        return obs.astype(np.float32, copy=False)

    def _build_terrain_info(self, state: GazeboAPFSceneState, positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        num_agents = positions.shape[0]
        terrain_info = np.zeros((num_agents, 32), dtype=np.float32)
        current_xy = positions[:, :2].astype(np.float32, copy=False)
        map_max = np.float32(max(float(state.map_size) - 1.0, 0.0))
        sample_coords = np.zeros((num_agents, 29, 2), dtype=np.float32)
        sample_coords[:, 0, :] = current_xy
        sample_coords[:, 1:5, :] = np.clip(current_xy[:, None, :] + self._GRADIENT_OFFSETS[None, :, :], 0.0, map_max)

        vel_norm = np.linalg.norm(velocities, axis=1, keepdims=True)
        vel_dir = np.zeros_like(velocities, dtype=np.float32)
        vel_dir[:, 0] = 1.0
        moving = vel_norm[:, 0] > 1e-6
        if np.any(moving):
            vel_dir[moving] = velocities[moving] / vel_norm[moving]

        forward_raw = current_xy[:, None, :] + self._FORWARD_DISTANCES[None, :, None] * vel_dir[:, None, :2]
        forward_valid = np.logical_and.reduce(
            (
                forward_raw[:, :, 0] >= 0.0,
                forward_raw[:, :, 0] < state.map_size,
                forward_raw[:, :, 1] >= 0.0,
                forward_raw[:, :, 1] < state.map_size,
            )
        )
        sample_coords[:, 5:13, :] = np.where(forward_valid[:, :, None], forward_raw, 0.0)

        near_raw = current_xy[:, None, :] + self._DIRECTION_PAIRS[None, :, :] * self._NEAR_DISTANCE
        far_raw = current_xy[:, None, :] + self._DIRECTION_PAIRS[None, :, :] * self._FAR_DISTANCE
        near_valid = np.logical_and.reduce(
            (
                near_raw[:, :, 0] >= 0.0,
                near_raw[:, :, 0] < state.map_size,
                near_raw[:, :, 1] >= 0.0,
                near_raw[:, :, 1] < state.map_size,
            )
        )
        far_valid = np.logical_and.reduce(
            (
                far_raw[:, :, 0] >= 0.0,
                far_raw[:, :, 0] < state.map_size,
                far_raw[:, :, 1] >= 0.0,
                far_raw[:, :, 1] < state.map_size,
            )
        )
        sample_coords[:, 13:29:2, :] = np.where(near_valid[:, :, None], near_raw, 0.0)
        sample_coords[:, 14:29:2, :] = np.where(far_valid[:, :, None], far_raw, 0.0)

        heights = state.terrain.batch_height(sample_coords.reshape(-1, 2)).reshape(num_agents, 29)
        current_height = heights[:, 0]
        grad_heights = heights[:, 1:5]
        forward_heights = heights[:, 5:13]
        surround_heights = heights[:, 13:29]

        terrain_info[:, 0] = (positions[:, 2] - current_height) / 20.0
        terrain_info[:, 1] = current_height / 100.0
        terrain_info[:, 2] = (grad_heights[:, 0] - current_height) / 10.0
        terrain_info[:, 3] = (grad_heights[:, 1] - current_height) / 10.0
        terrain_info[:, 4] = (grad_heights[:, 2] - current_height) / 10.0
        terrain_info[:, 5] = (grad_heights[:, 3] - current_height) / 10.0
        terrain_info[:, 6:14] = np.where(forward_valid, forward_heights / 100.0, 0.0).astype(np.float32)

        surround_valid = np.empty((num_agents, 16), dtype=bool)
        surround_valid[:, 0::2] = near_valid
        surround_valid[:, 1::2] = far_valid
        terrain_info[:, 14:30] = np.where(surround_valid, surround_heights / 100.0, 0.0).astype(np.float32)

        counts = np.maximum(np.sum(near_valid, axis=1).astype(np.float32), 1.0)
        near_heights = surround_heights[:, 0::2].astype(np.float32, copy=False)
        valid_heights = np.where(near_valid, near_heights, 0.0)
        mean_heights = np.sum(valid_heights, axis=1) / counts
        centered = np.where(near_valid, near_heights - mean_heights[:, None], 0.0)
        terrain_info[:, 30] = np.sqrt(np.sum(centered * centered, axis=1) / counts) / 20.0

        px = (grad_heights[:, 0] - grad_heights[:, 2]) / 6.0
        py = (grad_heights[:, 1] - grad_heights[:, 3]) / 6.0
        dz = positions[:, 2] - current_height
        terrain_info[:, 31] = np.clip(dz / np.sqrt(1.0 + px * px + py * py) / 20.0, -1.0, 1.0)
        return terrain_info.astype(np.float32, copy=False)

    def _build_obstacle_info(self, state: GazeboAPFSceneState, positions: np.ndarray) -> np.ndarray:
        num_agents = positions.shape[0]
        obstacle_info = np.zeros((num_agents, 15), dtype=np.float32)
        obstacle_info[:, 3::5] = 1.0
        if not state.obstacles:
            return obstacle_info
        centers = np.asarray([ob.center for ob in state.obstacles], dtype=np.float32)
        radii = np.asarray([ob.radius for ob in state.obstacles], dtype=np.float32)
        rel = centers[None, :, :] - positions[:, None, :]
        d_center = np.linalg.norm(rel, axis=2)
        d_surface = np.maximum(0.0, d_center - radii[None, :])
        top_k = min(3, centers.shape[0])
        if centers.shape[0] > top_k:
            nearest = np.argpartition(d_surface, kth=top_k - 1, axis=1)[:, :top_k]
            nearest_d = np.take_along_axis(d_surface, nearest, axis=1)
            sorted_idx = np.take_along_axis(nearest, np.argsort(nearest_d, axis=1), axis=1)
        else:
            sorted_idx = np.argsort(d_surface, axis=1)
        rows = np.arange(num_agents)
        max_dist = max(float(state.map_size), 1e-6)
        for k in range(top_k):
            idx = sorted_idx[:, k]
            off = k * 5
            obstacle_info[:, off : off + 3] = rel[rows, idx] / max_dist
            obstacle_info[:, off + 3] = np.minimum(d_surface[rows, idx] / max_dist, 1.0)
            obstacle_info[:, off + 4] = np.minimum(radii[idx] / max_dist, 1.0)
        return obstacle_info.astype(np.float32, copy=False)

    def _build_other_agents_info(
        self,
        state: GazeboAPFSceneState,
        positions: np.ndarray,
        velocities: np.ndarray,
        max_speeds: np.ndarray,
    ) -> np.ndarray:
        num_agents = positions.shape[0]
        other = np.zeros((num_agents, 12), dtype=np.float32)
        for idx in range(num_agents):
            others = [j for j in range(num_agents) if j != idx]
            if not others:
                continue
            if len(others) == 1:
                others = [others[0], others[0]]
            for slot, other_idx in enumerate(others[:2]):
                base = slot * 6
                other[idx, base : base + 3] = (positions[other_idx] - positions[idx]) / 100.0
                other[idx, base + 3 : base + 6] = velocities[other_idx] / max(max_speeds[idx], 1e-6)
        return other.astype(np.float32, copy=False)

    def nearest_obstacle(self, state: GazeboAPFSceneState, agent_idx: int) -> Optional[Dict[str, Any]]:
        if not state.obstacles or agent_idx >= len(state.agents):
            return None
        pos = state.agents[agent_idx].position
        best = None
        for obstacle in state.obstacles:
            diff = obstacle.center - pos
            center_distance = float(np.linalg.norm(diff))
            surface_distance = center_distance - float(obstacle.radius)
            clearance = surface_distance - float(state.agents[agent_idx].radius)
            item = {
                "name": obstacle.name,
                "type": obstacle.type,
                "center": obstacle.center.astype(float).tolist(),
                "radius": float(obstacle.radius),
                "center_distance": center_distance,
                "surface_distance": surface_distance,
                "clearance": clearance,
            }
            if best is None or surface_distance < best["surface_distance"]:
                best = item
        return best

    def terrain_clearance(self, state: GazeboAPFSceneState, agent_idx: int) -> float:
        agent = state.agents[agent_idx]
        terrain_h = float(state.terrain.height(agent.position[0], agent.position[1]))
        return float(agent.position[2] - agent.radius - terrain_h)
