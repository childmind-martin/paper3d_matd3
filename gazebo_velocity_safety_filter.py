#!/usr/bin/env python3
"""Obstacle-aware velocity safety filter for Gazebo-authoritative evaluation.

This module is intentionally separate from the actor and Python APF code.  It
only post-processes the velocity setpoint that is about to be sent to Gazebo.
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gazebo_apf_state_provider import GazeboAPFSceneState


def _safe_float(value: Any, default: float) -> float:
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return float(default)


def _safe_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        return float(value) if np.isfinite(value) else None
    except Exception:
        return str(value)


def _as_cmd_array(values: Any, count: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    out = np.zeros((int(count), 3), dtype=np.float64)
    rows = min(out.shape[0], arr.shape[0])
    cols = min(3, arr.shape[1] if arr.ndim > 1 else 0)
    if rows > 0 and cols > 0:
        out[:rows, :cols] = arr[:rows, :cols]
    return np.where(np.isfinite(out), out, 0.0)


def _as_optional_cmd_array(values: Any, count: int) -> Optional[np.ndarray]:
    if values is None:
        return None
    try:
        return _as_cmd_array(values, count)
    except Exception:
        return None


def _norm_or_zero(values: Any) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)[:3]
        if arr.size >= 3 and np.all(np.isfinite(arr)):
            return arr.astype(np.float64, copy=True)
    except Exception:
        pass
    return np.zeros(3, dtype=np.float64)


@dataclass
class VelocitySafetyFilterConfig:
    LEGACY_MODES = ("off", "safety_margin", "velocity_filter")
    PROJECTION_MODES = (
        "velocity_filter_projection",
        "velocity_filter_tangent",
        "velocity_filter_tangent_hysteresis",
        "velocity_filter_goal_projection_recovery",
    )

    mode: str = "off"
    safety_margin: float = 1.0
    stopping_margin: float = 1.0
    brake_accel: float = 3.6
    inward_scale: float = 0.0
    outward_speed: float = 0.0
    max_outward_speed: float = 0.5
    clamp_to_max_speed: bool = True
    projection_alpha: float = 1.0
    tangent_gain: float = 0.45
    tangent_speed_floor: float = 0.25
    recovery_goal_gain: float = 0.6
    min_goal_projection: float = 0.15
    d_enter: float = float("nan")
    d_exit: float = float("nan")
    resume_steps: int = 8
    boundary_band: float = 0.25
    boundary_speed_threshold: float = 0.25
    line_block_margin: float = 0.5
    boundary_escape_dwell_steps: int = 48
    boundary_escape_goal_projection_threshold: float = 0.45
    boundary_escape_tangent_speed_floor: float = 1.0
    boundary_escape_tangent_speed_max: float = 1.4
    formation_relaxation_mode: str = "off"
    arrived_agent_hold: bool = False
    agent_arrival_radius: float = 1.0
    agent_arrival_release_radius: float = 1.8
    arrived_hold_max_speed: float = 0.2
    arrived_hold_kp: float = 0.5
    laggard_goal_floor: bool = False
    goal_progress_watchdog_window: int = 80
    min_progress_rate: float = 0.02
    goal_progress_floor: float = 0.5
    adaptive_goal_floor_by_remaining_time: bool = False
    min_goal_progress_floor: float = 0.5
    max_goal_progress_floor: float = 2.0
    goal_progress_finish_margin: float = 0.3
    single_laggard_max_goal_floor: float = 2.0
    laggard_margin: float = 20.0
    safety_relax_clearance: float = 1.2
    arrived_agent_soft_margin: float = 3.0
    single_laggard_finish: bool = False
    candidate_arbitration: bool = False
    candidate_tangent_floors: Tuple[float, ...] = (0.6, 0.8, 1.0)
    candidate_terrain_guard: bool = True
    candidate_terrain_clearance_min: float = 1.5
    candidate_prediction_dt: float = 0.08
    candidate_rollout_enabled: bool = False
    candidate_rollout_horizon: float = 1.5
    candidate_rollout_dt: float = 0.2
    candidate_soft_penalty_weight: float = 1.0
    candidate_disable_tangent_on_terrain_block: bool = True
    micro_waypoint_fan: bool = False
    w_goal: float = 1.0
    w_filter_delta: float = 0.05
    w_smooth: float = 0.05
    w_agent_closing: float = 0.5
    w_formation: float = 0.02
    agent_agent_constraint_distance: float = 3.0

    @classmethod
    def from_env(cls) -> "VelocitySafetyFilterConfig":
        raw = (
            os.getenv("GAZEBO_LIVE_OBSTACLE_SAFETY_MODE")
            or os.getenv("GAZEBO_LIVE_OBSTACLE_SAFETY_FILTER")
            or os.getenv("GAZEBO_LIVE_OBSTACLE_FILTER_MODE")
            or "off"
        )
        mode = str(raw).strip().lower()
        if mode in ("0", "false", "no", "off", "none", "disabled"):
            mode = "off"
        elif mode in ("1", "true", "yes", "on", "velocity", "filter"):
            mode = "velocity_filter"
        elif mode in ("margin", "safety", "safety-margin"):
            mode = "safety_margin"
        elif mode in ("projection", "velocity-projection", "velocity_filter_halfspace"):
            mode = "velocity_filter_projection"
        elif mode in ("tangent", "velocity-tangent", "velocity_filter_tangent"):
            mode = "velocity_filter_tangent"
        elif mode in ("tangent_hysteresis", "tangent-hysteresis", "velocity_filter_hysteresis"):
            mode = "velocity_filter_tangent_hysteresis"
        elif mode in ("goal_projection_recovery", "goal-recovery", "velocity_filter_recovery"):
            mode = "velocity_filter_goal_projection_recovery"
        elif mode not in cls.LEGACY_MODES and mode not in cls.PROJECTION_MODES:
            mode = "off"

        relaxation_mode = str(os.getenv("GAZEBO_LIVE_FORMATION_RELAXATION_MODE", "off") or "off").strip().lower()
        if relaxation_mode not in (
            "off",
            "arrived_hold",
            "laggard_goal_floor",
            "arrived_hold_laggard_goal_floor",
        ):
            relaxation_mode = "off"
        arrived_hold = _safe_bool("GAZEBO_LIVE_ARRIVED_AGENT_HOLD", False) or relaxation_mode in (
            "arrived_hold",
            "arrived_hold_laggard_goal_floor",
        )
        laggard_goal_floor = _safe_bool("GAZEBO_LIVE_LAGGARD_GOAL_FLOOR", False) or relaxation_mode in (
            "laggard_goal_floor",
            "arrived_hold_laggard_goal_floor",
        )
        candidate_floors: List[float] = []
        raw_floors = os.getenv("GAZEBO_LIVE_CANDIDATE_TANGENT_FLOORS", "0.6,0.8,1.0")
        for part in str(raw_floors).split(","):
            value = _safe_float(part.strip(), float("nan"))
            if np.isfinite(value) and value > 0.0:
                candidate_floors.append(float(value))
        if not candidate_floors:
            candidate_floors = [0.6, 0.8, 1.0]

        agent_agent_constraint_distance = max(
            0.0,
            _safe_float(os.getenv("GAZEBO_LIVE_AGENT_AGENT_CONSTRAINT_DISTANCE"), 3.0),
        )
        default_outward = 0.0 if mode in ("off", "safety_margin") else 0.15
        return cls(
            mode=mode,
            safety_margin=max(0.0, _safe_float(os.getenv("GAZEBO_LIVE_OBSTACLE_SAFETY_MARGIN"), 1.0)),
            stopping_margin=max(0.0, _safe_float(os.getenv("GAZEBO_LIVE_OBSTACLE_STOPPING_MARGIN"), 1.0)),
            brake_accel=max(1e-6, _safe_float(os.getenv("GAZEBO_LIVE_OBSTACLE_BRAKE_ACCEL"), 3.6)),
            inward_scale=float(np.clip(_safe_float(os.getenv("GAZEBO_LIVE_OBSTACLE_INWARD_SCALE"), 0.0), 0.0, 1.0)),
            outward_speed=max(0.0, _safe_float(os.getenv("GAZEBO_LIVE_OBSTACLE_OUTWARD_SPEED"), default_outward)),
            max_outward_speed=max(0.0, _safe_float(os.getenv("GAZEBO_LIVE_OBSTACLE_MAX_OUTWARD_SPEED"), 0.5)),
            clamp_to_max_speed=_safe_bool("GAZEBO_LIVE_OBSTACLE_CLAMP_TO_MAX_SPEED", True),
            projection_alpha=max(0.0, _safe_float(os.getenv("GAZEBO_LIVE_OBSTACLE_PROJECTION_ALPHA"), 1.0)),
            tangent_gain=max(0.0, _safe_float(os.getenv("GAZEBO_LIVE_OBSTACLE_TANGENT_GAIN"), 0.45)),
            tangent_speed_floor=max(0.0, _safe_float(os.getenv("GAZEBO_LIVE_OBSTACLE_TANGENT_SPEED_FLOOR"), 0.25)),
            recovery_goal_gain=max(0.0, _safe_float(os.getenv("GAZEBO_LIVE_OBSTACLE_RECOVERY_GOAL_GAIN"), 0.6)),
            min_goal_projection=max(0.0, _safe_float(os.getenv("GAZEBO_LIVE_OBSTACLE_MIN_GOAL_PROJECTION"), 0.15)),
            d_enter=_safe_float(os.getenv("GAZEBO_LIVE_OBSTACLE_D_ENTER"), float("nan")),
            d_exit=_safe_float(os.getenv("GAZEBO_LIVE_OBSTACLE_D_EXIT"), float("nan")),
            resume_steps=max(1, int(_safe_float(os.getenv("GAZEBO_LIVE_OBSTACLE_RESUME_STEPS"), 8))),
            boundary_band=max(0.0, _safe_float(os.getenv("GAZEBO_LIVE_OBSTACLE_BOUNDARY_BAND"), 0.25)),
            boundary_speed_threshold=max(
                0.0,
                _safe_float(os.getenv("GAZEBO_LIVE_OBSTACLE_BOUNDARY_SPEED_THRESHOLD"), 0.25),
            ),
            line_block_margin=max(0.0, _safe_float(os.getenv("GAZEBO_LIVE_OBSTACLE_LINE_BLOCK_MARGIN"), 0.5)),
            boundary_escape_dwell_steps=max(
                0,
                int(_safe_float(os.getenv("GAZEBO_LIVE_OBSTACLE_BOUNDARY_ESCAPE_DWELL_STEPS"), 48)),
            ),
            boundary_escape_goal_projection_threshold=max(
                0.0,
                _safe_float(os.getenv("GAZEBO_LIVE_OBSTACLE_BOUNDARY_ESCAPE_GOAL_PROJECTION_THRESHOLD"), 0.45),
            ),
            boundary_escape_tangent_speed_floor=max(
                0.0,
                _safe_float(os.getenv("GAZEBO_LIVE_OBSTACLE_BOUNDARY_ESCAPE_TANGENT_SPEED_FLOOR"), 1.0),
            ),
            boundary_escape_tangent_speed_max=max(
                0.0,
                _safe_float(os.getenv("GAZEBO_LIVE_OBSTACLE_BOUNDARY_ESCAPE_TANGENT_SPEED_MAX"), 1.4),
            ),
            formation_relaxation_mode=relaxation_mode,
            arrived_agent_hold=arrived_hold,
            agent_arrival_radius=max(0.0, _safe_float(os.getenv("GAZEBO_LIVE_AGENT_ARRIVAL_RADIUS"), 1.0)),
            agent_arrival_release_radius=max(
                0.0,
                _safe_float(os.getenv("GAZEBO_LIVE_AGENT_ARRIVAL_RELEASE_RADIUS"), 1.8),
            ),
            arrived_hold_max_speed=max(0.0, _safe_float(os.getenv("GAZEBO_LIVE_ARRIVED_HOLD_MAX_SPEED"), 0.2)),
            arrived_hold_kp=max(0.0, _safe_float(os.getenv("GAZEBO_LIVE_ARRIVED_HOLD_KP"), 0.5)),
            laggard_goal_floor=laggard_goal_floor,
            goal_progress_watchdog_window=max(
                2,
                int(_safe_float(os.getenv("GAZEBO_LIVE_GOAL_PROGRESS_WATCHDOG_WINDOW"), 80)),
            ),
            min_progress_rate=max(0.0, _safe_float(os.getenv("GAZEBO_LIVE_MIN_PROGRESS_RATE"), 0.02)),
            goal_progress_floor=max(0.0, _safe_float(os.getenv("GAZEBO_LIVE_GOAL_PROGRESS_FLOOR"), 0.5)),
            adaptive_goal_floor_by_remaining_time=_safe_bool(
                "GAZEBO_LIVE_ADAPTIVE_GOAL_FLOOR_BY_REMAINING_TIME",
                False,
            ),
            min_goal_progress_floor=max(
                0.0,
                _safe_float(os.getenv("GAZEBO_LIVE_MIN_GOAL_PROGRESS_FLOOR"), 0.5),
            ),
            max_goal_progress_floor=max(
                0.0,
                _safe_float(os.getenv("GAZEBO_LIVE_MAX_GOAL_PROGRESS_FLOOR"), 2.0),
            ),
            goal_progress_finish_margin=max(
                0.0,
                _safe_float(os.getenv("GAZEBO_LIVE_GOAL_PROGRESS_FINISH_MARGIN"), 0.3),
            ),
            single_laggard_max_goal_floor=max(
                0.0,
                _safe_float(os.getenv("GAZEBO_LIVE_SINGLE_LAGGARD_MAX_GOAL_FLOOR"), 2.0),
            ),
            laggard_margin=max(0.0, _safe_float(os.getenv("GAZEBO_LIVE_LAGGARD_MARGIN"), 20.0)),
            safety_relax_clearance=max(0.0, _safe_float(os.getenv("GAZEBO_LIVE_SAFETY_RELAX_CLEARANCE"), 1.2)),
            arrived_agent_soft_margin=max(
                0.0,
                _safe_float(os.getenv("GAZEBO_LIVE_ARRIVED_AGENT_SOFT_MARGIN"), agent_agent_constraint_distance),
            ),
            single_laggard_finish=_safe_bool("GAZEBO_LIVE_SINGLE_LAGGARD_FINISH", False),
            candidate_arbitration=_safe_bool("GAZEBO_LIVE_CANDIDATE_ARBITRATION", False),
            candidate_tangent_floors=tuple(candidate_floors),
            candidate_terrain_guard=_safe_bool("GAZEBO_LIVE_CANDIDATE_TERRAIN_GUARD", True),
            candidate_terrain_clearance_min=max(
                0.0,
                _safe_float(os.getenv("GAZEBO_LIVE_CANDIDATE_TERRAIN_CLEARANCE_MIN"), 1.5),
            ),
            candidate_prediction_dt=max(
                1e-6,
                _safe_float(os.getenv("GAZEBO_LIVE_CANDIDATE_PREDICTION_DT"), _safe_float(os.getenv("SIMULATION_DT"), 0.08)),
            ),
            candidate_rollout_enabled=_safe_bool("GAZEBO_LIVE_CANDIDATE_ROLLOUT", False),
            candidate_rollout_horizon=max(
                1e-6,
                _safe_float(os.getenv("GAZEBO_LIVE_CANDIDATE_ROLLOUT_HORIZON"), 1.5),
            ),
            candidate_rollout_dt=max(
                1e-6,
                _safe_float(os.getenv("GAZEBO_LIVE_CANDIDATE_ROLLOUT_DT"), 0.2),
            ),
            candidate_soft_penalty_weight=max(
                0.0,
                _safe_float(os.getenv("GAZEBO_LIVE_CANDIDATE_SOFT_PENALTY_WEIGHT"), 1.0),
            ),
            candidate_disable_tangent_on_terrain_block=_safe_bool(
                "GAZEBO_LIVE_CANDIDATE_DISABLE_TANGENT_ON_TERRAIN_BLOCK",
                True,
            ),
            micro_waypoint_fan=_safe_bool("GAZEBO_LIVE_MICRO_WAYPOINT_FAN", False),
            w_goal=_safe_float(os.getenv("GAZEBO_LIVE_CANDIDATE_W_GOAL"), 1.0),
            w_filter_delta=_safe_float(os.getenv("GAZEBO_LIVE_CANDIDATE_W_FILTER_DELTA"), 0.05),
            w_smooth=_safe_float(os.getenv("GAZEBO_LIVE_CANDIDATE_W_SMOOTH"), 0.05),
            w_agent_closing=_safe_float(os.getenv("GAZEBO_LIVE_CANDIDATE_W_AGENT_CLOSING"), 0.5),
            w_formation=_safe_float(os.getenv("GAZEBO_LIVE_CANDIDATE_W_FORMATION"), 0.02),
            agent_agent_constraint_distance=agent_agent_constraint_distance,
        )

    @property
    def enabled(self) -> bool:
        return self.mode in ("safety_margin", "velocity_filter", *self.PROJECTION_MODES)

    @property
    def projection_enabled(self) -> bool:
        return self.mode in self.PROJECTION_MODES

    @property
    def tangent_enabled(self) -> bool:
        return self.mode in (
            "velocity_filter_tangent",
            "velocity_filter_tangent_hysteresis",
            "velocity_filter_goal_projection_recovery",
        )

    @property
    def hysteresis_enabled(self) -> bool:
        return self.mode in ("velocity_filter_tangent_hysteresis", "velocity_filter_goal_projection_recovery")

    @property
    def formation_relaxation_enabled(self) -> bool:
        return bool(self.arrived_agent_hold or self.laggard_goal_floor or self.candidate_arbitration)


class AgentLivenessMonitor:
    def __init__(self, config: VelocitySafetyFilterConfig) -> None:
        self.config = config
        self._goal_history: Dict[int, Deque[float]] = {}
        self._filter_history: Dict[int, Deque[float]] = {}
        self._boundary_history: Dict[int, Deque[float]] = {}
        self._goal_projection_history: Dict[int, Deque[float]] = {}
        self._goal_floor_conflict_history: Dict[int, Deque[float]] = {}
        self._arrived: Dict[int, bool] = {}
        self._stalled_steps: Dict[int, int] = {}

    def pre_step(self, state: GazeboAPFSceneState) -> Dict[int, Dict[str, Any]]:
        window = int(self.config.goal_progress_watchdog_window)
        goal_distances: List[float] = []
        for idx, agent in enumerate(state.agents):
            goal = state.goals[idx] if idx < len(state.goals) else None
            dist = self._goal_distance(agent.position, goal)
            goal_distances.append(dist)
            hist = self._goal_history.setdefault(idx, deque(maxlen=max(window + 1, 2)))
            hist.append(dist)
            arrived = bool(self._arrived.get(idx, False))
            if np.isfinite(dist):
                if arrived:
                    arrived = dist <= float(self.config.agent_arrival_release_radius)
                else:
                    arrived = dist <= float(self.config.agent_arrival_radius)
            else:
                arrived = False
            self._arrived[idx] = arrived

        finite = [d for d in goal_distances if np.isfinite(d)]
        min_team_goal_distance = float(min(finite)) if finite else None
        arrived_count = int(sum(1 for value in self._arrived.values() if bool(value)))
        agent_count = int(len(goal_distances))
        ranked = sorted(
            [
                (idx, dist)
                for idx, dist in enumerate(goal_distances)
                if np.isfinite(dist) and not bool(self._arrived.get(idx, False))
            ],
            key=lambda item: item[1],
            reverse=True,
        )
        rank_by_agent = {idx: rank + 1 for rank, (idx, _) in enumerate(ranked)}

        out: Dict[int, Dict[str, Any]] = {}
        for idx, dist in enumerate(goal_distances):
            progress = self._progress_rate(idx)
            arrived = bool(self._arrived.get(idx, False))
            stalled = int(self._stalled_steps.get(idx, 0) or 0)
            if arrived:
                stalled = 0
            elif progress < float(self.config.min_progress_rate):
                stalled += 1
            else:
                stalled = 0
            self._stalled_steps[idx] = stalled
            out[idx] = {
                "agent_arrived": arrived,
                "agent_laggard_rank": rank_by_agent.get(idx),
                "arrived_count": arrived_count,
                "agent_count": agent_count,
                "single_laggard_finish_candidate": bool(arrived_count >= max(agent_count - 1, 0) and not arrived),
                "goal_progress_rate": progress,
                "stalled_steps": stalled,
                "recent_filter_active_ratio": self._recent_ratio(self._filter_history.get(idx)),
                "recent_boundary_dwell_ratio": self._recent_ratio(self._boundary_history.get(idx)),
                "recent_goal_projection": self._recent_mean(self._goal_projection_history.get(idx)),
                "recent_goal_floor_safety_conflict_ratio": self._recent_ratio(
                    self._goal_floor_conflict_history.get(idx)
                ),
                "min_team_goal_distance": min_team_goal_distance,
            }
        return out

    def post_step(self, records: Sequence[Dict[str, Any]]) -> None:
        window = int(self.config.goal_progress_watchdog_window)
        for record in records or []:
            try:
                idx = int(record.get("agent_id"))
            except Exception:
                continue
            fh = self._filter_history.setdefault(idx, deque(maxlen=max(window, 2)))
            bh = self._boundary_history.setdefault(idx, deque(maxlen=max(window, 2)))
            gh = self._goal_projection_history.setdefault(idx, deque(maxlen=max(window, 2)))
            ch = self._goal_floor_conflict_history.setdefault(idx, deque(maxlen=max(window, 2)))
            fh.append(1.0 if bool(record.get("filter_active", False)) else 0.0)
            bh.append(1.0 if _safe_float(record.get("boundary_dwell_steps"), 0.0) > 0.0 else 0.0)
            ch.append(1.0 if bool(record.get("goal_floor_safety_conflict", False)) else 0.0)
            gp = _safe_float(record.get("goal_projection_after_filter"), float("nan"))
            if np.isfinite(gp):
                gh.append(gp)

    def _progress_rate(self, idx: int) -> float:
        hist = list(self._goal_history.get(idx) or [])
        if len(hist) < 2:
            return 0.0
        first = _safe_float(hist[0], float("nan"))
        last = _safe_float(hist[-1], float("nan"))
        if not np.isfinite(first) or not np.isfinite(last):
            return 0.0
        return float((first - last) / max(len(hist) - 1, 1))

    @staticmethod
    def _goal_distance(position: Any, goal: Any) -> float:
        try:
            p = np.asarray(position, dtype=np.float64).reshape(-1)[:3]
            g = np.asarray(goal, dtype=np.float64).reshape(-1)[:3]
            if p.size >= 3 and g.size >= 3 and np.all(np.isfinite(p)) and np.all(np.isfinite(g)):
                return float(np.linalg.norm(g - p))
        except Exception:
            pass
        return float("nan")

    @staticmethod
    def _recent_ratio(values: Optional[Deque[float]]) -> float:
        vals = [float(v) for v in list(values or []) if np.isfinite(v)]
        return float(np.mean(vals)) if vals else 0.0

    @staticmethod
    def _recent_mean(values: Optional[Deque[float]]) -> Optional[float]:
        vals = [float(v) for v in list(values or []) if np.isfinite(v)]
        return float(np.mean(vals)) if vals else None


@dataclass
class VelocitySafetyFilterResult:
    nominal_cmd_vel: np.ndarray
    final_cmd_vel: np.ndarray
    records: List[Dict[str, Any]]

    def summary(self) -> Dict[str, Any]:
        return summarize_velocity_filter_records(self.records)


class GazeboObstacleVelocitySafetyFilter:
    def __init__(self, config: VelocitySafetyFilterConfig) -> None:
        self.config = config
        self._avoidance_state: Dict[Tuple[int, str], Dict[str, Any]] = {}
        self._liveness = AgentLivenessMonitor(config)
        self._last_cmd_vel: Dict[int, np.ndarray] = {}

    @classmethod
    def from_env(cls) -> "GazeboObstacleVelocitySafetyFilter":
        return cls(VelocitySafetyFilterConfig.from_env())

    def apply(
        self,
        state: GazeboAPFSceneState,
        nominal_cmd_vel: Any,
        diagnostics_context: Optional[Dict[str, Any]] = None,
    ) -> VelocitySafetyFilterResult:
        count = len(state.agents)
        nominal = _as_cmd_array(nominal_cmd_vel, count)
        diagnostics_context = diagnostics_context or {}
        raw_actor_accel = _as_optional_cmd_array(diagnostics_context.get("raw_actor_accel"), count)
        corrected_accel = _as_optional_cmd_array(diagnostics_context.get("corrected_accel"), count)
        shadow_raw_cmd_vel = _as_optional_cmd_array(diagnostics_context.get("shadow_raw_cmd_vel"), count)
        raw_actions = diagnostics_context.get("raw_actions")
        corrected_actions = diagnostics_context.get("corrected_actions")
        runtime_step = int(_safe_float(diagnostics_context.get("step"), _safe_float(state.frame, 0.0)))
        episode_length = int(_safe_float(diagnostics_context.get("episode_length"), 0.0))
        simulation_dt = max(
            1e-9,
            _safe_float(
                diagnostics_context.get("simulation_dt"),
                _safe_float(os.getenv("SIMULATION_DT"), self.config.candidate_prediction_dt),
            ),
        )
        liveness = self._liveness.pre_step(state)
        final = nominal.copy()
        records: List[Dict[str, Any]] = []
        for idx, agent in enumerate(state.agents):
            nearest = self._nearest_obstacle(state, idx)
            cmd = nominal[idx].astype(np.float64, copy=True)
            filtered = cmd.copy()
            record = self._base_record(state, idx, agent, nearest, cmd)
            record.update(
                {
                    "runtime_step": runtime_step,
                    "episode_length": episode_length,
                    "simulation_dt": simulation_dt,
                }
            )
            self._attach_semantic_context(
                record,
                idx,
                raw_actor_accel=raw_actor_accel,
                corrected_accel=corrected_accel,
                shadow_raw_cmd_vel=shadow_raw_cmd_vel,
                raw_actions=raw_actions,
                corrected_actions=corrected_actions,
                liveness=liveness.get(idx, {}),
            )
            if nearest is not None:
                filtered, update = self._filter_cmd(state, agent, nearest, cmd, record)
                record.update(update)
            else:
                record.update(
                    {
                        "filter_active": False,
                        "filter_trigger_reason": "no_inward_velocity" if nearest is not None else "no_obstacle",
                        "allowed_inward_velocity": record["inward_velocity_before_filter"],
                        "outward_speed_applied": 0.0,
                        "halfspace_projection_delta_norm": 0.0,
                    }
                )
                filtered, extra_update = self._apply_liveness_without_obstacle(state, agent, filtered, cmd, record)
                record.update(extra_update)
            filtered = self._clamp_speed(filtered, agent.max_speed)
            final[idx] = filtered
            self._finalize_record(record, cmd, filtered)
            records.append(_json_safe(record))
            self._last_cmd_vel[int(idx)] = filtered.astype(np.float64, copy=True)
        self._liveness.post_step(records)
        return VelocitySafetyFilterResult(
            nominal_cmd_vel=nominal.astype(np.float32, copy=False),
            final_cmd_vel=final.astype(np.float32, copy=False),
            records=records,
        )

    def _nearest_obstacle(self, state: GazeboAPFSceneState, agent_idx: int) -> Optional[Dict[str, Any]]:
        if not state.obstacles or agent_idx >= len(state.agents):
            return None
        agent = state.agents[agent_idx]
        best = None
        for obstacle in state.obstacles:
            diff = np.asarray(obstacle.center, dtype=np.float64) - np.asarray(agent.position, dtype=np.float64)
            center_distance = float(np.linalg.norm(diff))
            surface_distance = center_distance - float(obstacle.radius)
            clearance = surface_distance - float(agent.radius)
            unit = diff / max(center_distance, 1e-9)
            item = {
                "name": obstacle.name,
                "type": obstacle.type,
                "center": np.asarray(obstacle.center, dtype=np.float64).astype(float).tolist(),
                "radius": float(obstacle.radius),
                "center_distance": center_distance,
                "surface_distance": surface_distance,
                "clearance": clearance,
                "unit_to_obstacle": unit.astype(float).tolist(),
            }
            if best is None or surface_distance < best["surface_distance"]:
                best = item
        return best

    def _goal_geometry(self, state: GazeboAPFSceneState, agent_idx: int, position: np.ndarray) -> Dict[str, Any]:
        goal = None
        try:
            goals = np.asarray(state.goals, dtype=np.float64)
            if goals.ndim >= 2 and agent_idx < goals.shape[0] and goals.shape[1] >= 3:
                candidate = goals[agent_idx, :3]
                if np.all(np.isfinite(candidate)):
                    goal = candidate.astype(np.float64, copy=True)
        except Exception:
            goal = None
        if goal is None:
            return {
                "goal": None,
                "goal_distance": None,
                "goal_direction": None,
            }
        delta = goal - position
        dist = float(np.linalg.norm(delta))
        direction = delta / max(dist, 1e-9)
        return {
            "goal": goal.astype(float).tolist(),
            "goal_distance": dist,
            "goal_direction": direction.astype(float).tolist(),
        }

    def _nearest_agent_and_formation_error(self, state: GazeboAPFSceneState, agent_idx: int) -> Dict[str, Any]:
        try:
            positions = np.asarray([agent.position for agent in state.agents], dtype=np.float64)
            velocities = np.asarray([agent.velocity for agent in state.agents], dtype=np.float64)
        except Exception:
            return {}
        if positions.ndim != 2 or positions.shape[0] <= 1 or agent_idx >= positions.shape[0]:
            return {}
        current = positions[agent_idx, :3]
        current_vel = velocities[agent_idx, :3] if velocities.ndim == 2 and agent_idx < velocities.shape[0] else np.zeros(3)
        nearest_id = None
        nearest_distance = None
        nearest_rel_pos = None
        nearest_rel_vel = None
        nearest_goal_distance = None
        formation_errors: List[float] = []
        goals = None
        try:
            goals = np.asarray(state.goals, dtype=np.float64)
            if goals.ndim != 2 or goals.shape[0] < positions.shape[0] or goals.shape[1] < 3:
                goals = None
        except Exception:
            goals = None
        for other_idx in range(positions.shape[0]):
            if other_idx == agent_idx:
                continue
            other = positions[other_idx, :3]
            if not np.all(np.isfinite(other)) or not np.all(np.isfinite(current)):
                continue
            distance = float(np.linalg.norm(other - current))
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_id = int(other_idx)
                nearest_rel_pos = (other - current).astype(float).tolist()
                if velocities.ndim == 2 and other_idx < velocities.shape[0]:
                    nearest_rel_vel = (velocities[other_idx, :3] - current_vel).astype(float).tolist()
                if goals is not None:
                    nearest_goal_distance = float(np.linalg.norm(goals[other_idx, :3] - goals[agent_idx, :3]))
            if goals is not None:
                desired = float(np.linalg.norm(goals[other_idx, :3] - goals[agent_idx, :3]))
                if np.isfinite(desired):
                    formation_errors.append(abs(distance - desired))
        formation_error = float(np.mean(formation_errors)) if formation_errors else None
        return {
            "nearest_agent_id": nearest_id,
            "nearest_agent_distance": nearest_distance,
            "nearest_agent_relative_position": nearest_rel_pos,
            "nearest_agent_relative_velocity": nearest_rel_vel,
            "nearest_agent_goal_distance": nearest_goal_distance,
            "formation_error": formation_error,
        }

    def _line_to_goal_blocked(
        self,
        position: np.ndarray,
        goal: Optional[Sequence[float]],
        nearest: Optional[Dict[str, Any]],
        agent_radius: float,
    ) -> Tuple[bool, Optional[float]]:
        if goal is None or nearest is None:
            return False, None
        try:
            start = np.asarray(position, dtype=np.float64).reshape(-1)[:3]
            end = np.asarray(goal, dtype=np.float64).reshape(-1)[:3]
            center = np.asarray(nearest.get("center"), dtype=np.float64).reshape(-1)[:3]
            if start.size < 3 or end.size < 3 or center.size < 3:
                return False, None
            segment = end - start
            seg_len_sq = float(np.dot(segment, segment))
            if seg_len_sq <= 1e-9:
                return False, None
            t = float(np.clip(np.dot(center - start, segment) / seg_len_sq, 0.0, 1.0))
            closest = start + t * segment
            center_distance = float(np.linalg.norm(center - closest))
            inflated_radius = float(nearest.get("radius", 0.0) or 0.0) + float(agent_radius) + self.config.line_block_margin
            line_clearance = center_distance - inflated_radius
            return bool(0.0 < t < 1.0 and line_clearance <= 0.0), line_clearance
        except Exception:
            return False, None

    def _direct_line_feasibility(
        self,
        state: GazeboAPFSceneState,
        agent_idx: int,
        position: np.ndarray,
        goal: Optional[Sequence[float]],
        agent_radius: float,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "line_to_goal_blocked_obstacle": False,
            "line_to_goal_blocked_terrain": False,
            "line_to_goal_blocked_agent": False,
            "line_to_goal_min_obstacle_clearance": None,
            "line_to_goal_min_terrain_clearance": None,
            "line_to_goal_min_agent_clearance": None,
        }
        if goal is None:
            return result
        try:
            start = np.asarray(position, dtype=np.float64).reshape(-1)[:3]
            end = np.asarray(goal, dtype=np.float64).reshape(-1)[:3]
            if start.size < 3 or end.size < 3 or not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
                return result
            segment = end - start
            seg_len = float(np.linalg.norm(segment))
            seg_len_sq = float(np.dot(segment, segment))
            if seg_len <= 1e-9 or seg_len_sq <= 1e-12:
                return result
        except Exception:
            return result

        obstacle_min = float("inf")
        for obstacle in state.obstacles or []:
            try:
                center = np.asarray(obstacle.center, dtype=np.float64).reshape(-1)[:3]
                if center.size < 3 or not np.all(np.isfinite(center)):
                    continue
                t = float(np.clip(np.dot(center - start, segment) / seg_len_sq, 0.0, 1.0))
                closest = start + t * segment
                clearance = float(np.linalg.norm(center - closest) - float(obstacle.radius) - float(agent_radius))
                obstacle_min = min(obstacle_min, clearance)
                if 0.0 < t < 1.0 and clearance <= float(self.config.line_block_margin):
                    result["line_to_goal_blocked_obstacle"] = True
            except Exception:
                continue
        if np.isfinite(obstacle_min):
            result["line_to_goal_min_obstacle_clearance"] = obstacle_min

        terrain_min = float("inf")
        try:
            sample_count = int(np.clip(np.ceil(seg_len / 5.0) + 1, 2, 80))
            for alpha in np.linspace(0.0, 1.0, sample_count):
                point = start + float(alpha) * segment
                terrain_height = float(state.terrain.height(float(point[0]), float(point[1])))
                clearance = float(point[2] - float(agent_radius) - terrain_height)
                terrain_min = min(terrain_min, clearance)
                if clearance <= 0.0:
                    result["line_to_goal_blocked_terrain"] = True
        except Exception:
            pass
        if np.isfinite(terrain_min):
            result["line_to_goal_min_terrain_clearance"] = terrain_min

        agent_min = float("inf")
        for other_idx, other in enumerate(state.agents or []):
            if other_idx == agent_idx:
                continue
            try:
                other_pos = np.asarray(other.position, dtype=np.float64).reshape(-1)[:3]
                if other_pos.size < 3 or not np.all(np.isfinite(other_pos)):
                    continue
                t = float(np.clip(np.dot(other_pos - start, segment) / seg_len_sq, 0.0, 1.0))
                closest = start + t * segment
                other_radius = float(getattr(other, "radius", 0.0) or 0.0)
                physical_clearance = float(np.linalg.norm(other_pos - closest) - float(agent_radius) - other_radius)
                agent_min = min(agent_min, physical_clearance)
                soft_clearance = self._agent_agent_soft_clearance_threshold(
                    state,
                    other_idx,
                    float(agent_radius),
                    other_radius,
                )
                if 0.0 < t < 1.0 and physical_clearance <= soft_clearance:
                    result["line_to_goal_blocked_agent"] = True
            except Exception:
                continue
        if np.isfinite(agent_min):
            result["line_to_goal_min_agent_clearance"] = agent_min
        return result

    def _agent_agent_soft_clearance_threshold(
        self,
        state: GazeboAPFSceneState,
        other_idx: int,
        self_radius: float,
        other_radius: float,
    ) -> float:
        default_clearance = max(
            0.0,
            float(self.config.agent_agent_constraint_distance) - float(self_radius) - float(other_radius),
        )
        if bool(self._liveness._arrived.get(int(other_idx), False)):
            return max(0.0, min(default_clearance, float(self.config.arrived_agent_soft_margin)))
        return default_clearance

    def _base_record(
        self,
        state: GazeboAPFSceneState,
        idx: int,
        agent: Any,
        nearest: Optional[Dict[str, Any]],
        cmd: np.ndarray,
    ) -> Dict[str, Any]:
        unit = nearest.get("unit_to_obstacle") if nearest else None
        inward = self._inward_velocity(cmd, unit)
        obstacle_velocity = np.zeros(3, dtype=np.float64)
        if nearest and nearest.get("velocity") is not None:
            try:
                raw_obstacle_velocity = np.asarray(nearest.get("velocity"), dtype=np.float64).reshape(-1)
                if raw_obstacle_velocity.size >= 3 and np.all(np.isfinite(raw_obstacle_velocity[:3])):
                    obstacle_velocity = raw_obstacle_velocity[:3].astype(np.float64, copy=True)
            except Exception:
                obstacle_velocity = np.zeros(3, dtype=np.float64)
        relative_velocity = cmd - obstacle_velocity
        relative_inward = self._inward_velocity(relative_velocity, unit)
        current_relative_velocity = np.asarray(agent.velocity, dtype=np.float64).reshape(-1)[:3] - obstacle_velocity
        current_relative_inward = self._inward_velocity(current_relative_velocity, unit)
        closing_inward_for_stopping = max(0.0, inward, relative_inward, current_relative_inward)
        stopping_distance = 0.0
        if closing_inward_for_stopping > 0.0:
            brake = max(float(getattr(agent, "accel", self.config.brake_accel) or self.config.brake_accel), self.config.brake_accel, 1e-6)
            stopping_distance = float((closing_inward_for_stopping * closing_inward_for_stopping) / (2.0 * brake))
        position = np.asarray(agent.position, dtype=np.float64).reshape(-1)[:3]
        terrain_height = None
        terrain_clearance = None
        try:
            terrain_height = float(state.terrain.height(float(position[0]), float(position[1])))
            terrain_clearance = float(position[2] - float(agent.radius) - terrain_height)
        except Exception:
            terrain_height = None
            terrain_clearance = None
        goal_info = self._goal_geometry(state, idx, position)
        goal_dir = goal_info.get("goal_direction")
        goal_projection_before = self._goal_projection(cmd, goal_dir)
        line_blocked, line_clearance = self._line_to_goal_blocked(
            position,
            goal_info.get("goal"),
            nearest,
            float(agent.radius),
        )
        direct_line = self._direct_line_feasibility(state, idx, position, goal_info.get("goal"), float(agent.radius))
        direct_line["line_to_goal_blocked_any_direct"] = bool(
            direct_line.get("line_to_goal_blocked_obstacle", False)
            or direct_line.get("line_to_goal_blocked_terrain", False)
            or direct_line.get("line_to_goal_blocked_agent", False)
        )
        agent_geometry = self._nearest_agent_and_formation_error(state, idx)
        try:
            positions = np.asarray([ag.position for ag in state.agents], dtype=np.float64)
            team_centroid = np.mean(positions[:, :3], axis=0).astype(float).tolist() if positions.ndim == 2 and positions.size else None
        except Exception:
            team_centroid = None
        nearest_agent_distance = agent_geometry.get("nearest_agent_distance")
        formation_error = agent_geometry.get("formation_error")
        agent_agent_constraint_active = False
        if nearest_agent_distance is not None:
            agent_agent_constraint_active = float(nearest_agent_distance) <= float(self.config.agent_agent_constraint_distance)
        return {
            "agent_id": int(idx),
            "agent_name": agent.name,
            "mode": self.config.mode,
            "enabled": bool(self.config.enabled),
            "frame": state.frame,
            "source": state.source,
            "pose": np.asarray(agent.position, dtype=np.float64).astype(float).tolist(),
            "velocity": np.asarray(agent.velocity, dtype=np.float64).astype(float).tolist(),
            "nominal_cmd_vel": cmd.astype(float).tolist(),
            "goal": goal_info.get("goal"),
            "goal_distance": goal_info.get("goal_distance"),
            "goal_direction": goal_dir,
            "goal_projection_before_filter": goal_projection_before,
            "obstacle_velocity": obstacle_velocity.astype(float).tolist(),
            "relative_velocity_before_filter": relative_velocity.astype(float).tolist(),
            "current_relative_velocity": current_relative_velocity.astype(float).tolist(),
            "nearest_obstacle": None if nearest is None else {k: v for k, v in nearest.items() if k != "unit_to_obstacle"},
            "nearest_obstacle_id": nearest.get("name") if nearest else None,
            "obstacle_center": nearest.get("center") if nearest else None,
            "obstacle_radius": nearest.get("radius") if nearest else None,
            "surface_distance": nearest.get("surface_distance") if nearest else None,
            "agent_radius": float(agent.radius),
            "terrain_height": terrain_height,
            "terrain_clearance": terrain_clearance,
            "collision_distance_threshold": float(state.collision_distance_threshold),
            "clearance": nearest.get("clearance") if nearest else None,
            "unit_to_obstacle": unit,
            "safety_margin": float(self.config.safety_margin),
            "stopping_margin": float(self.config.stopping_margin),
            "brake_accel": float(self.config.brake_accel),
            "stopping_distance": stopping_distance,
            "inward_velocity_before_filter": inward,
            "relative_inward_velocity_before_filter": relative_inward,
            "current_relative_inward_velocity": current_relative_inward,
            "closing_inward_velocity_for_stopping": closing_inward_for_stopping,
            "tangential_speed_before_filter": self._tangential_speed(cmd, unit),
            "tangential_velocity_before_filter": self._tangential_speed(cmd, unit),
            "line_to_goal_blocked": bool(line_blocked),
            "line_to_goal_clearance": line_clearance,
            **direct_line,
            "nearest_agent_distance": nearest_agent_distance,
            "nearest_agent_id": agent_geometry.get("nearest_agent_id"),
            "nearest_agent_relative_position": agent_geometry.get("nearest_agent_relative_position"),
            "nearest_agent_relative_velocity": agent_geometry.get("nearest_agent_relative_velocity"),
            "nearest_agent_goal_distance": agent_geometry.get("nearest_agent_goal_distance"),
            "agent_agent_constraint_active": bool(agent_agent_constraint_active),
            "formation_error": formation_error,
            "formation_error_agent": formation_error,
            "team_centroid": team_centroid,
        }

    def _attach_semantic_context(
        self,
        record: Dict[str, Any],
        idx: int,
        raw_actor_accel: Optional[np.ndarray],
        corrected_accel: Optional[np.ndarray],
        shadow_raw_cmd_vel: Optional[np.ndarray],
        raw_actions: Any,
        corrected_actions: Any,
        liveness: Dict[str, Any],
    ) -> None:
        goal_dir = record.get("goal_direction")
        raw_accel = raw_actor_accel[idx] if raw_actor_accel is not None and idx < raw_actor_accel.shape[0] else None
        corr_accel = corrected_accel[idx] if corrected_accel is not None and idx < corrected_accel.shape[0] else None
        shadow_cmd = (
            shadow_raw_cmd_vel[idx]
            if shadow_raw_cmd_vel is not None and idx < shadow_raw_cmd_vel.shape[0]
            else None
        )
        record["raw_actor_accel"] = None if raw_accel is None else np.asarray(raw_accel, dtype=np.float64).astype(float).tolist()
        record["corrected_accel"] = None if corr_accel is None else np.asarray(corr_accel, dtype=np.float64).astype(float).tolist()
        record["shadow_raw_cmd_vel"] = None if shadow_cmd is None else np.asarray(shadow_cmd, dtype=np.float64).astype(float).tolist()
        record["raw_action"] = self._row_or_none(raw_actions, idx)
        record["corrected_action"] = self._row_or_none(corrected_actions, idx)
        record["raw_goal_projection"] = self._goal_projection(_norm_or_zero(shadow_cmd), goal_dir) if shadow_cmd is not None else None
        record["apf_goal_projection"] = self._goal_projection(_norm_or_zero(record.get("nominal_cmd_vel")), goal_dir)
        if record.get("raw_goal_projection") is not None and record.get("apf_goal_projection") is not None:
            record["apf_goal_projection_delta"] = float(record["apf_goal_projection"] - record["raw_goal_projection"])
        else:
            record["apf_goal_projection_delta"] = None
        for key, value in (liveness or {}).items():
            record[key] = value

    @staticmethod
    def _row_or_none(values: Any, idx: int) -> Optional[List[float]]:
        try:
            arr = np.asarray(values, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if idx < arr.shape[0]:
                row = arr[idx].reshape(-1)
                if np.all(np.isfinite(row)):
                    return row.astype(float).tolist()
        except Exception:
            pass
        return None

    def _apply_liveness_without_obstacle(
        self,
        state: GazeboAPFSceneState,
        agent: Any,
        filtered: np.ndarray,
        cmd: np.ndarray,
        record: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        update: Dict[str, Any] = {
            "arrived_hold_active": False,
            "goal_floor_active": False,
            "goal_floor_safety_conflict": False,
            "single_laggard_finish_active": False,
            "finish_safety_conflict": False,
            "selected_candidate_name": "nominal",
            "candidate_scores": {},
            "candidate_goal_projections": {},
        }
        out = filtered.copy()
        if self.config.arrived_agent_hold and bool(record.get("agent_arrived", False)):
            hold, hold_update = self._arrived_hold_cmd(agent, record)
            out = hold
            update.update(hold_update)
            update["filter_active"] = True
            update["filter_trigger_reason"] = "arrived_hold"
            update["selected_candidate_name"] = "arrived_hold"
        elif self._should_apply_goal_floor(record):
            out, floor_update = self._apply_goal_floor(out, normal=None, record=record)
            update.update(floor_update)
            if floor_update.get("goal_floor_active"):
                update["filter_active"] = True
                update["filter_trigger_reason"] = "laggard_goal_floor"
                update["selected_candidate_name"] = "goal_progress_floor"
        if self.config.single_laggard_finish and bool(record.get("single_laggard_finish_candidate", False)):
            out, finish_update = self._apply_single_laggard_finish(
                state=state,
                agent=agent,
                filtered=out,
                cmd=cmd,
                normal=None,
                surface_distance=None,
                d_safe=None,
                record=record,
                state_key=(int(record.get("agent_id", -1)), str(record.get("nearest_obstacle_id") or "none")),
            )
            update.update(finish_update)
            if finish_update.get("single_laggard_finish_active"):
                update["filter_active"] = True
                update["filter_trigger_reason"] = "single_laggard_finish"
                update["selected_candidate_name"] = (
                    str(finish_update.get("micro_waypoint_selected") or "single_laggard_finish")
                    if finish_update.get("micro_waypoint_fan_active")
                    else "single_laggard_finish"
                )
        record.update(update)
        if self.config.candidate_arbitration:
            out, arbitration = self._arbitrate_candidates(
                base=out,
                cmd=cmd,
                normal=None,
                surface_distance=None,
                d_safe=None,
                agent=agent,
                record=record,
                state_key=(int(record.get("agent_id", -1)), str(record.get("nearest_obstacle_id") or "none")),
                state=state,
            )
            update.update(arbitration)
            update["filter_active"] = bool(update.get("filter_active", False) or arbitration.get("candidate_arbitration_active", False))
            if arbitration.get("candidate_arbitration_active"):
                update["filter_trigger_reason"] = str(update.get("filter_trigger_reason") or "candidate_arbitration")
        return out, update

    def _filter_cmd(
        self,
        state: GazeboAPFSceneState,
        agent: Any,
        nearest: Dict[str, Any],
        cmd: np.ndarray,
        record: Dict[str, Any],
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        cfg = self.config
        unit = np.asarray(record.get("unit_to_obstacle"), dtype=np.float64).reshape(-1)[:3]
        if unit.size < 3 or not np.all(np.isfinite(unit)):
            return cmd.copy(), {
                "filter_active": False,
                "filter_trigger_reason": "invalid_obstacle_direction",
                "allowed_inward_velocity": record["inward_velocity_before_filter"],
                "outward_speed_applied": 0.0,
                "halfspace_projection_delta_norm": 0.0,
            }
        inward = float(record["inward_velocity_before_filter"])
        surface_distance = float(nearest["surface_distance"])
        clearance = float(nearest["clearance"])
        safety_surface_limit = float(agent.radius) + cfg.safety_margin
        stopping_surface_limit = float(agent.radius) + cfg.stopping_margin + float(record["stopping_distance"])
        d_enter = float(cfg.d_enter) if np.isfinite(cfg.d_enter) else safety_surface_limit
        d_exit = (
            float(cfg.d_exit)
            if np.isfinite(cfg.d_exit)
            else d_enter + max(0.5, cfg.boundary_band * 2.0, cfg.safety_margin * 0.5)
        )

        trigger_reasons = []
        if surface_distance <= safety_surface_limit:
            trigger_reasons.append("safety_margin")
        if cfg.mode == "velocity_filter" and surface_distance <= stopping_surface_limit:
            trigger_reasons.append("stopping_distance")
        if cfg.projection_enabled and surface_distance <= stopping_surface_limit:
            trigger_reasons.append("stopping_distance")

        state_key = (int(record.get("agent_id", -1)), str(nearest.get("name") or "obstacle"))
        enter_avoid = surface_distance < d_enter or (float(record.get("stopping_distance") or 0.0) + cfg.stopping_margin > surface_distance)
        clear_avoid = surface_distance > d_exit and not bool(record.get("line_to_goal_blocked", False))
        avoidance = self._transition_avoidance_state(state_key, enter_avoid, clear_avoid)
        record.update(
            {
                "d_enter": d_enter,
                "d_exit": d_exit,
                "avoidance_state": avoidance["state"],
                "avoidance_clear_count": int(avoidance.get("clear_count", 0) or 0),
            }
        )

        if not cfg.enabled:
            return cmd.copy(), {
                "filter_active": False,
                "filter_trigger_reason": "disabled",
                "allowed_inward_velocity": inward,
                "outward_speed_applied": 0.0,
                "safety_surface_limit": safety_surface_limit,
                "stopping_surface_limit": stopping_surface_limit,
                "d_enter": d_enter,
                "d_exit": d_exit,
                "halfspace_projection_delta_norm": 0.0,
                "avoidance_state": avoidance["state"],
                "avoidance_clear_count": int(avoidance.get("clear_count", 0) or 0),
            }

        if not cfg.projection_enabled:
            if inward <= 0.0:
                return cmd.copy(), {
                    "filter_active": False,
                    "filter_trigger_reason": "no_inward_velocity",
                    "allowed_inward_velocity": inward,
                    "outward_speed_applied": 0.0,
                    "safety_surface_limit": safety_surface_limit,
                    "stopping_surface_limit": stopping_surface_limit,
                    "d_enter": d_enter,
                    "d_exit": d_exit,
                    "halfspace_projection_delta_norm": 0.0,
                    "avoidance_state": avoidance["state"],
                    "avoidance_clear_count": int(avoidance.get("clear_count", 0) or 0),
                }

            active = bool(trigger_reasons)
            if not active:
                return cmd.copy(), {
                    "filter_active": False,
                    "filter_trigger_reason": "outside_trigger_region",
                    "allowed_inward_velocity": inward,
                    "outward_speed_applied": 0.0,
                    "safety_surface_limit": safety_surface_limit,
                    "stopping_surface_limit": stopping_surface_limit,
                    "d_enter": d_enter,
                    "d_exit": d_exit,
                    "halfspace_projection_delta_norm": 0.0,
                    "avoidance_state": avoidance["state"],
                    "avoidance_clear_count": int(avoidance.get("clear_count", 0) or 0),
                }

            tangential = cmd - inward * unit
            if cfg.mode == "safety_margin":
                allowed_inward = inward * cfg.inward_scale
            else:
                free_clearance = max(0.0, clearance - cfg.stopping_margin)
                allowed_inward = float(np.sqrt(max(0.0, 2.0 * cfg.brake_accel * free_clearance)))
                allowed_inward = min(inward, allowed_inward)
                if "safety_margin" in trigger_reasons:
                    allowed_inward = min(allowed_inward, inward * cfg.inward_scale)

            outward = 0.0
            if cfg.outward_speed > 0.0:
                deficit = max(0.0, cfg.safety_margin - clearance)
                intensity = 1.0 if cfg.safety_margin <= 1e-9 else float(np.clip(deficit / cfg.safety_margin, 0.0, 1.0))
                if "stopping_distance" in trigger_reasons and "safety_margin" not in trigger_reasons:
                    intensity = max(intensity, 0.25)
                outward = min(float(cfg.outward_speed) * max(intensity, 0.0), float(cfg.max_outward_speed))

            filtered = tangential + allowed_inward * unit - outward * unit
            return filtered, {
                "filter_active": True,
                "filter_trigger_reason": ",".join(trigger_reasons),
                "allowed_inward_velocity": allowed_inward,
                "outward_speed_applied": outward,
                "safety_surface_limit": safety_surface_limit,
                "stopping_surface_limit": stopping_surface_limit,
                "d_enter": d_enter,
                "d_exit": d_exit,
                "halfspace_projection_delta_norm": 0.0,
                "avoidance_state": avoidance["state"],
                "avoidance_clear_count": int(avoidance.get("clear_count", 0) or 0),
            }

        liveness_active = bool(
            (cfg.arrived_agent_hold and bool(record.get("agent_arrived", False)))
            or self._should_apply_goal_floor(record)
            or cfg.candidate_arbitration
        )
        active = bool(trigger_reasons or avoidance["state"] != "FREE" or liveness_active)
        if not active:
            return cmd.copy(), {
                "filter_active": False,
                "filter_trigger_reason": "outside_trigger_region",
                "allowed_inward_velocity": inward,
                "outward_speed_applied": 0.0,
                "safety_surface_limit": safety_surface_limit,
                "stopping_surface_limit": stopping_surface_limit,
                "d_enter": d_enter,
                "d_exit": d_exit,
                "halfspace_projection_delta_norm": 0.0,
                "avoidance_state": avoidance["state"],
                "avoidance_clear_count": int(avoidance.get("clear_count", 0) or 0),
            }

        normal = -unit
        filtered, projection_update = self._project_halfspace(
            cmd,
            normal=normal,
            surface_distance=surface_distance,
            d_safe=safety_surface_limit,
        )
        if projection_update["halfspace_projection_delta_norm"] > 1e-9:
            trigger_reasons.append("halfspace_projection")

        tangent_update: Dict[str, Any] = {}
        if cfg.tangent_enabled:
            filtered, tangent_update = self._apply_tangent_recovery(
                filtered,
                cmd,
                normal=normal,
                state_key=state_key,
                avoidance=avoidance,
                agent=agent,
                record=record,
            )
            if tangent_update.get("tangent_recovery_applied"):
                trigger_reasons.append("tangent_recovery")

        recovery_update: Dict[str, Any] = {}
        if cfg.mode == "velocity_filter_goal_projection_recovery":
            filtered, recovery_update = self._apply_goal_projection_recovery(
                filtered,
                normal=normal,
                surface_distance=surface_distance,
                d_safe=safety_surface_limit,
                record=record,
            )
            if recovery_update.get("goal_projection_recovery_applied"):
                trigger_reasons.append("goal_projection_recovery")

        outward_added = self._outward_component(filtered - cmd, normal)
        if cfg.outward_speed > 0.0 and surface_distance <= safety_surface_limit:
            deficit = max(0.0, cfg.safety_margin - clearance)
            intensity = 1.0 if cfg.safety_margin <= 1e-9 else float(np.clip(deficit / cfg.safety_margin, 0.0, 1.0))
            extra_outward = min(float(cfg.outward_speed) * max(intensity, 0.0), float(cfg.max_outward_speed))
            if extra_outward > 0.0:
                filtered = filtered + extra_outward * normal
                outward_added += extra_outward
                trigger_reasons.append("outward_bias")

        liveness_update: Dict[str, Any] = {
            "arrived_hold_active": False,
            "goal_floor_active": False,
            "goal_floor_safety_conflict": False,
        }
        if cfg.arrived_agent_hold and bool(record.get("agent_arrived", False)):
            filtered, hold_update = self._arrived_hold_cmd(agent, record)
            filtered, safety_update = self._safety_project_candidate(filtered, normal, surface_distance, safety_surface_limit)
            liveness_update.update(hold_update)
            liveness_update.update({"arrived_hold_safety_projection_delta_norm": safety_update.get("delta_norm")})
            trigger_reasons.append("arrived_hold")
        elif self._should_apply_goal_floor(record):
            filtered, floor_update = self._apply_goal_floor(filtered, normal=normal, record=record)
            filtered, safety_update = self._safety_project_candidate(filtered, normal, surface_distance, safety_surface_limit)
            after_floor_safety = self._goal_projection(filtered, record.get("goal_direction"))
            conflict = (
                floor_update.get("goal_floor_active")
                and after_floor_safety is not None
                and after_floor_safety + 1e-6 < float(floor_update.get("goal_floor_target", self.config.goal_progress_floor))
            )
            floor_update["goal_floor_safety_conflict"] = bool(conflict)
            floor_update["goal_floor_after_safety_projection"] = after_floor_safety
            floor_update["goal_floor_safety_projection_delta_norm"] = safety_update.get("delta_norm")
            liveness_update.update(floor_update)
            if floor_update.get("goal_floor_active"):
                trigger_reasons.append("laggard_goal_floor")
        record.update(liveness_update)

        if cfg.single_laggard_finish and bool(record.get("single_laggard_finish_candidate", False)):
            filtered, finish_update = self._apply_single_laggard_finish(
                state=state,
                agent=agent,
                filtered=filtered,
                cmd=cmd,
                normal=normal,
                surface_distance=surface_distance,
                d_safe=safety_surface_limit,
                record=record,
                state_key=state_key,
            )
            liveness_update.update(finish_update)
            record.update(finish_update)
            if finish_update.get("single_laggard_finish_active"):
                trigger_reasons.append("single_laggard_finish")

        arbitration_update: Dict[str, Any] = {
            "candidate_arbitration_active": False,
            "selected_candidate_name": "adaptive_escape",
            "candidate_scores": {},
            "candidate_goal_projections": {},
        }
        if cfg.candidate_arbitration:
            filtered, arbitration_update = self._arbitrate_candidates(
                base=filtered,
                cmd=cmd,
                normal=normal,
                surface_distance=surface_distance,
                d_safe=safety_surface_limit,
                agent=agent,
                record=record,
                state_key=state_key,
                state=state,
            )
            if arbitration_update.get("candidate_arbitration_active"):
                trigger_reasons.append("candidate_arbitration")

        trigger_text = ",".join(dict.fromkeys(trigger_reasons)) if trigger_reasons else "projection_monitor"
        return filtered, {
            "filter_active": True,
            "filter_trigger_reason": trigger_text,
            "allowed_inward_velocity": self._inward_velocity(filtered, unit),
            "outward_speed_applied": outward_added,
            "outward_velocity_added": outward_added,
            "safety_surface_limit": safety_surface_limit,
            "stopping_surface_limit": stopping_surface_limit,
            "d_enter": d_enter,
            "d_exit": d_exit,
            "avoidance_state": avoidance["state"],
            "avoidance_clear_count": int(avoidance.get("clear_count", 0) or 0),
            **projection_update,
            **tangent_update,
            **recovery_update,
            **liveness_update,
            **arbitration_update,
        }

    def _transition_avoidance_state(self, key: Tuple[int, str], enter: bool, clear: bool) -> Dict[str, Any]:
        entry = self._avoidance_state.setdefault(
            key,
            {
                "state": "FREE",
                "clear_count": 0,
                "boundary_dwell_steps": 0,
                "cached_tangent": None,
            },
        )
        state = str(entry.get("state") or "FREE")
        clear_count = int(entry.get("clear_count", 0) or 0)
        if enter:
            state = "AVOID"
            clear_count = 0
        elif state in ("AVOID", "RECOVER", "RESUME"):
            if clear:
                clear_count += 1
            else:
                clear_count = 0
            if clear_count >= int(self.config.resume_steps):
                if state == "AVOID":
                    state = "RECOVER"
                elif state == "RECOVER":
                    state = "RESUME"
                else:
                    state = "FREE"
                clear_count = 0
        else:
            state = "FREE"
            clear_count = 0
        entry["state"] = state
        entry["clear_count"] = clear_count
        return entry

    def _project_halfspace(
        self,
        cmd: np.ndarray,
        normal: np.ndarray,
        surface_distance: float,
        d_safe: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        n = np.asarray(normal, dtype=np.float64).reshape(-1)[:3]
        n_norm = float(np.linalg.norm(n))
        if n_norm <= 1e-9 or not np.all(np.isfinite(n)):
            return cmd.copy(), {
                "halfspace_projection_delta_norm": 0.0,
                "halfspace_lower_bound": None,
                "normal_velocity_before_projection": None,
                "normal_velocity_after_projection": None,
            }
        n = n / n_norm
        before = float(np.dot(cmd, n))
        lower = float(-self.config.projection_alpha * (float(surface_distance) - float(d_safe)))
        projected = cmd.copy()
        if before < lower:
            projected = projected + (lower - before) * n
        after = float(np.dot(projected, n))
        return projected, {
            "halfspace_projection_delta_norm": float(np.linalg.norm(projected - cmd)),
            "halfspace_lower_bound": lower,
            "normal_velocity_before_projection": before,
            "normal_velocity_after_projection": after,
        }

    def _apply_tangent_recovery(
        self,
        filtered: np.ndarray,
        cmd: np.ndarray,
        normal: np.ndarray,
        state_key: Tuple[int, str],
        avoidance: Dict[str, Any],
        agent: Any,
        record: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        state = str(avoidance.get("state") or "FREE")
        line_blocked = bool(record.get("line_to_goal_blocked", False))
        if state == "FREE" and not line_blocked:
            return filtered, {
                "tangent_recovery_applied": False,
                "tangent_direction": None,
                "tangent_target_speed": 0.0,
            }
        tangent = self._tangent_direction(state_key, normal, cmd, record)
        if tangent is None:
            return filtered, {
                "tangent_recovery_applied": False,
                "tangent_direction": None,
                "tangent_target_speed": 0.0,
            }
        speed_ref = min(
            float(np.linalg.norm(cmd)),
            float(getattr(agent, "max_speed", 0.0) or np.linalg.norm(cmd) or 0.0),
        )
        target_speed = max(float(self.config.tangent_speed_floor), float(self.config.tangent_gain) * speed_ref)
        escape_update = self._boundary_escape_tangent_target(filtered, avoidance, record)
        if escape_update["boundary_escape_tangent_boost_applied"]:
            target_speed = max(target_speed, float(escape_update["boundary_escape_tangent_target_speed"]))
        current_speed = float(np.dot(filtered, tangent))
        if current_speed >= target_speed:
            return filtered, {
                "tangent_recovery_applied": False,
                "tangent_direction": tangent.astype(float).tolist(),
                "tangent_target_speed": target_speed,
                "tangent_speed_before_recovery": current_speed,
                "tangent_speed_after_recovery": current_speed,
                **escape_update,
            }
        out = filtered + (target_speed - current_speed) * tangent
        return out, {
            "tangent_recovery_applied": True,
            "tangent_direction": tangent.astype(float).tolist(),
            "tangent_target_speed": target_speed,
            "tangent_speed_before_recovery": current_speed,
            "tangent_speed_after_recovery": float(np.dot(out, tangent)),
            **escape_update,
        }

    def _boundary_escape_tangent_target(
        self,
        filtered: np.ndarray,
        avoidance: Dict[str, Any],
        record: Dict[str, Any],
    ) -> Dict[str, Any]:
        cfg = self.config
        dwell_threshold = int(cfg.boundary_escape_dwell_steps)
        if dwell_threshold <= 0 or cfg.boundary_escape_tangent_speed_floor <= 0.0:
            return {
                "boundary_escape_tangent_boost_applied": False,
                "boundary_escape_tangent_target_speed": 0.0,
                "boundary_escape_dwell_threshold": dwell_threshold,
            }
        state = str(avoidance.get("state") or "FREE")
        if state not in ("AVOID", "RECOVER"):
            return {
                "boundary_escape_tangent_boost_applied": False,
                "boundary_escape_tangent_target_speed": 0.0,
                "boundary_escape_dwell_threshold": dwell_threshold,
            }
        dwell_steps = int(avoidance.get("boundary_dwell_steps", 0) or 0)
        surface = _safe_float(record.get("surface_distance"), float("nan"))
        d_enter = _safe_float(record.get("d_enter"), float("nan"))
        near_boundary = (
            np.isfinite(surface)
            and np.isfinite(d_enter)
            and surface <= d_enter + max(0.0, float(cfg.boundary_band))
        )
        goal_projection = self._goal_projection(filtered, record.get("goal_direction"))
        slow_goal_progress = (
            goal_projection is None
            or _safe_float(goal_projection, float("nan")) <= float(cfg.boundary_escape_goal_projection_threshold)
        )
        boost = bool(dwell_steps >= dwell_threshold and near_boundary and slow_goal_progress)
        target = 0.0
        if boost:
            excess = max(0, dwell_steps - dwell_threshold)
            ramp = min(1.0, float(excess) / max(float(dwell_threshold), 1.0))
            target = float(cfg.boundary_escape_tangent_speed_floor) * (1.0 + 0.25 * ramp)
            if cfg.boundary_escape_tangent_speed_max > 0.0:
                target = min(target, float(cfg.boundary_escape_tangent_speed_max))
        return {
            "boundary_escape_tangent_boost_applied": boost,
            "boundary_escape_tangent_target_speed": target,
            "boundary_escape_dwell_threshold": dwell_threshold,
            "boundary_escape_goal_projection": goal_projection,
        }

    def _arrived_hold_cmd(self, agent: Any, record: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        pos = _norm_or_zero(record.get("pose"))
        goal = _norm_or_zero(record.get("goal"))
        error = goal - pos
        cmd = float(self.config.arrived_hold_kp) * error
        speed = float(np.linalg.norm(cmd))
        limit = float(self.config.arrived_hold_max_speed)
        if limit > 0.0 and speed > limit:
            cmd = cmd / max(speed, 1e-9) * limit
        return cmd, {
            "arrived_hold_active": True,
            "arrived_hold_cmd_vel": cmd.astype(float).tolist(),
            "arrived_hold_error_norm": float(np.linalg.norm(error)),
        }

    def _should_apply_goal_floor(self, record: Dict[str, Any]) -> bool:
        if not self.config.laggard_goal_floor:
            return False
        if bool(record.get("agent_arrived", False)):
            return False
        clearance = _safe_float(record.get("clearance"), float("nan"))
        if np.isfinite(clearance) and clearance < float(self.config.safety_relax_clearance):
            return False
        terrain_clearance = _safe_float(record.get("terrain_clearance"), float("nan"))
        if np.isfinite(terrain_clearance) and terrain_clearance < float(self.config.safety_relax_clearance):
            return False
        boundary_ratio = _safe_float(record.get("recent_boundary_dwell_ratio"), 0.0)
        if boundary_ratio >= 0.20:
            return False
        goal_distance = _safe_float(record.get("goal_distance"), float("nan"))
        min_team = _safe_float(record.get("min_team_goal_distance"), float("nan"))
        single_laggard = bool(record.get("single_laggard_finish_candidate", False))
        laggard_by_margin = False
        if np.isfinite(goal_distance) and np.isfinite(min_team):
            laggard_by_margin = bool(goal_distance >= min_team + float(self.config.laggard_margin))
        elif single_laggard:
            laggard_by_margin = True
        if not laggard_by_margin and not single_laggard:
            return False
        goal_dir = self._vec3_or_none(record.get("goal_direction"))
        if goal_dir is None:
            return False
        floor_info = self._goal_floor_target_info(record, single_laggard=single_laggard)
        target = _safe_float(floor_info.get("goal_floor_target"), float(self.config.goal_progress_floor))
        current_projection = _safe_float(
            record.get("filter_goal_projection", record.get("apf_goal_projection", record.get("goal_projection_before_filter"))),
            float("nan"),
        )
        progress_rate = _safe_float(record.get("goal_progress_rate"), 0.0)
        if self.config.adaptive_goal_floor_by_remaining_time or single_laggard:
            return bool((not np.isfinite(current_projection)) or current_projection < target)
        return progress_rate < float(self.config.min_progress_rate)

    def _goal_floor_target_info(self, record: Dict[str, Any], single_laggard: bool = False) -> Dict[str, Any]:
        fixed_floor = float(self.config.goal_progress_floor)
        info: Dict[str, Any] = {
            "goal_floor_target": fixed_floor,
            "adaptive_goal_floor_active": False,
            "remaining_time": None,
            "goal_progress_required_speed": None,
        }
        if not (self.config.adaptive_goal_floor_by_remaining_time or single_laggard):
            return info
        step = int(_safe_float(record.get("runtime_step", record.get("step")), 0.0))
        episode_length = int(_safe_float(record.get("episode_length"), 0.0))
        dt = max(1e-9, _safe_float(record.get("simulation_dt"), _safe_float(os.getenv("SIMULATION_DT"), 0.08)))
        goal_distance = _safe_float(record.get("goal_distance"), float("nan"))
        if episode_length <= 0 or step < 0 or not np.isfinite(goal_distance):
            return info
        remaining_steps = max(int(episode_length) - int(step), 1)
        remaining_time = float(remaining_steps) * dt
        required = float(goal_distance) / max(remaining_time, 1e-6)
        max_floor = (
            float(self.config.single_laggard_max_goal_floor)
            if single_laggard
            else float(self.config.max_goal_progress_floor)
        )
        min_floor = float(self.config.min_goal_progress_floor)
        if max_floor < min_floor:
            max_floor = min_floor
        target = float(np.clip(required + float(self.config.goal_progress_finish_margin), min_floor, max_floor))
        info.update(
            {
                "goal_floor_target": target,
                "adaptive_goal_floor_active": True,
                "remaining_time": remaining_time,
                "goal_progress_required_speed": required,
            }
        )
        return info

    def _apply_goal_floor(
        self,
        cmd: np.ndarray,
        normal: Optional[np.ndarray],
        record: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        goal_dir = self._vec3_or_none(record.get("goal_direction"))
        if goal_dir is None:
            return cmd.copy(), {"goal_floor_active": False, "goal_floor_safety_conflict": False}
        before = float(np.dot(cmd, goal_dir))
        target_info = self._goal_floor_target_info(
            record,
            single_laggard=bool(record.get("single_laggard_finish_candidate", False)),
        )
        floor = float(target_info.get("goal_floor_target", self.config.goal_progress_floor))
        if before >= floor:
            return cmd.copy(), {
                "goal_floor_active": False,
                "goal_floor_before_projection": before,
                "goal_floor_target": floor,
                "goal_floor_safety_conflict": False,
                **target_info,
            }
        projected = cmd + (floor - before) * goal_dir
        return projected, {
            "goal_floor_active": True,
            "goal_floor_before_projection": before,
            "goal_floor_after_projection": float(np.dot(projected, goal_dir)),
            "goal_floor_target": floor,
            "goal_floor_safety_conflict": False,
            **target_info,
        }

    def _safety_project_candidate(
        self,
        candidate: np.ndarray,
        normal: Optional[np.ndarray],
        surface_distance: Optional[float],
        d_safe: Optional[float],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if normal is None or surface_distance is None or d_safe is None:
            return candidate.copy(), {"delta_norm": 0.0, "safe": True}
        projected, update = self._project_halfspace(candidate, normal, float(surface_distance), float(d_safe))
        lower = update.get("halfspace_lower_bound")
        after = update.get("normal_velocity_after_projection")
        safe = True
        if lower is not None and after is not None:
            safe = bool(float(after) + 1e-6 >= float(lower))
        return projected, {
            "delta_norm": update.get("halfspace_projection_delta_norm", 0.0),
            "safe": safe,
            "halfspace_lower_bound": lower,
            "normal_velocity_after_projection": after,
        }

    def _limit_speed_accel(
        self,
        candidate: np.ndarray,
        agent: Any,
        record: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        out = np.asarray(candidate, dtype=np.float64).reshape(-1)[:3].copy()
        before_speed = float(np.linalg.norm(out))
        max_speed = float(getattr(agent, "max_speed", 0.0) or 0.0)
        speed_limited = False
        if max_speed > 0.0 and before_speed > max_speed:
            out = out / max(before_speed, 1e-9) * max_speed
            speed_limited = True

        accel_limited = False
        prev = self._last_cmd_vel.get(int(record.get("agent_id", -1)))
        if prev is not None:
            prev_arr = np.asarray(prev, dtype=np.float64).reshape(-1)[:3]
            if prev_arr.size >= 3 and np.all(np.isfinite(prev_arr)):
                dt = max(1e-9, _safe_float(record.get("simulation_dt"), _safe_float(os.getenv("SIMULATION_DT"), 0.08)))
                accel_limit = float(getattr(agent, "accel", 0.0) or 0.0)
                delta = out - prev_arr
                delta_norm = float(np.linalg.norm(delta))
                max_delta = accel_limit * dt if accel_limit > 0.0 else 0.0
                if max_delta > 0.0 and delta_norm > max_delta:
                    out = prev_arr + delta / max(delta_norm, 1e-9) * max_delta
                    accel_limited = True
        out = self._clamp_speed(out, max_speed)
        return out, {
            "speed_limited": bool(speed_limited),
            "accel_limited": bool(accel_limited),
            "candidate_speed_before_limit": before_speed,
            "candidate_speed_after_limit": float(np.linalg.norm(out)),
        }

    def _rollout_candidate_safety(
        self,
        candidate: np.ndarray,
        state: GazeboAPFSceneState,
        agent: Any,
        record: Dict[str, Any],
    ) -> Dict[str, Any]:
        agent_idx = int(record.get("agent_id", -1))
        pos0 = np.asarray(agent.position, dtype=np.float64).reshape(-1)[:3]
        vel = np.asarray(candidate, dtype=np.float64).reshape(-1)[:3]
        if pos0.size < 3 or vel.size < 3 or not np.all(np.isfinite(pos0)) or not np.all(np.isfinite(vel)):
            return {
                "hard_violation": True,
                "hard_violation_reason": "projection_failed",
                "min_obstacle_clearance": None,
                "min_terrain_clearance": None,
                "min_agent_clearance": None,
                "soft_penalty": 0.0,
                "predicted_goal_distance": None,
            }
        if self.config.candidate_rollout_enabled:
            horizon = float(self.config.candidate_rollout_horizon)
            dt = float(self.config.candidate_rollout_dt)
        else:
            horizon = float(self.config.candidate_prediction_dt)
            dt = float(self.config.candidate_prediction_dt)
        dt = max(dt, 1e-6)
        steps = max(1, int(np.ceil(max(horizon, dt) / dt)))
        agent_radius = float(getattr(agent, "radius", record.get("agent_radius", 0.0)) or 0.0)
        min_obstacle = float("inf")
        min_terrain = float("inf")
        min_agent = float("inf")
        soft_penalty = 0.0
        hard_reasons: List[str] = []
        for step_idx in range(1, steps + 1):
            t = min(float(step_idx) * dt, max(horizon, dt))
            pos = pos0 + vel * t
            for obstacle in state.obstacles or []:
                try:
                    center = np.asarray(obstacle.center, dtype=np.float64).reshape(-1)[:3]
                    if center.size < 3 or not np.all(np.isfinite(center)):
                        continue
                    clearance = float(np.linalg.norm(pos - center) - float(obstacle.radius) - agent_radius)
                    min_obstacle = min(min_obstacle, clearance)
                    if clearance < 0.0:
                        hard_reasons.append("obstacle")
                except Exception:
                    continue
            try:
                terrain_h = float(state.terrain.height(float(pos[0]), float(pos[1])))
                clearance = float(pos[2] - agent_radius - terrain_h)
                min_terrain = min(min_terrain, clearance)
                if clearance < 0.0:
                    hard_reasons.append("terrain")
                elif self.config.candidate_terrain_guard:
                    soft_penalty += max(0.0, float(self.config.candidate_terrain_clearance_min) - clearance)
            except Exception:
                pass
            for other_idx, other in enumerate(state.agents or []):
                if other_idx == agent_idx:
                    continue
                try:
                    other_pos = np.asarray(other.position, dtype=np.float64).reshape(-1)[:3]
                    other_vel = np.asarray(other.velocity, dtype=np.float64).reshape(-1)[:3]
                    if other_pos.size < 3 or not np.all(np.isfinite(other_pos)):
                        continue
                    if other_vel.size >= 3 and np.all(np.isfinite(other_vel)):
                        other_pos = other_pos + other_vel * t
                    other_radius = float(getattr(other, "radius", 0.0) or 0.0)
                    clearance = float(np.linalg.norm(pos - other_pos) - agent_radius - other_radius)
                    min_agent = min(min_agent, clearance)
                    if clearance < 0.0:
                        hard_reasons.append("agent")
                    else:
                        soft_threshold = self._agent_agent_soft_clearance_threshold(
                            state,
                            other_idx,
                            agent_radius,
                            other_radius,
                        )
                        soft_penalty += max(0.0, soft_threshold - clearance)
                except Exception:
                    continue
        goal = self._raw_vec3_or_none(record.get("goal"))
        predicted_goal_distance = None
        if goal is not None:
            predicted_goal_distance = float(np.linalg.norm(goal - (pos0 + vel * max(horizon, dt))))
        unique_hard = list(dict.fromkeys(hard_reasons))
        return {
            "hard_violation": bool(unique_hard),
            "hard_violation_reason": ",".join(unique_hard) if unique_hard else None,
            "min_obstacle_clearance": min_obstacle if np.isfinite(min_obstacle) else None,
            "min_terrain_clearance": min_terrain if np.isfinite(min_terrain) else None,
            "min_agent_clearance": min_agent if np.isfinite(min_agent) else None,
            "soft_penalty": float(soft_penalty),
            "predicted_goal_distance": predicted_goal_distance,
        }

    def _candidate_reject_category(self, reason: Optional[str]) -> str:
        text = str(reason or "").lower()
        if "terrain" in text:
            return "terrain"
        if "obstacle" in text:
            return "obstacle"
        if "agent" in text:
            return "agent"
        if "speed" in text:
            return "speed"
        if "accel" in text:
            return "accel"
        return "projection_failed"

    def _apply_single_laggard_finish(
        self,
        state: GazeboAPFSceneState,
        agent: Any,
        filtered: np.ndarray,
        cmd: np.ndarray,
        normal: Optional[np.ndarray],
        surface_distance: Optional[float],
        d_safe: Optional[float],
        record: Dict[str, Any],
        state_key: Tuple[int, str],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        update: Dict[str, Any] = {
            "single_laggard_finish_active": False,
            "finish_goal_projection_before_safety": None,
            "finish_goal_projection_after_safety": None,
            "finish_safety_conflict": False,
            "finish_line_to_goal_blocked": bool(record.get("line_to_goal_blocked", False)),
            "finish_selected_cmd_vel": None,
        }
        if not self.config.single_laggard_finish or not bool(record.get("single_laggard_finish_candidate", False)):
            return filtered, update
        goal_dir = self._vec3_or_none(record.get("goal_direction"))
        if goal_dir is None:
            update["finish_safety_conflict"] = True
            return filtered, update
        target_info = self._goal_floor_target_info(record, single_laggard=True)
        finish_speed = max(float(target_info.get("goal_floor_target", 0.0) or 0.0), float(self.config.goal_progress_floor))
        max_speed = float(getattr(agent, "max_speed", 0.0) or 0.0)
        if max_speed > 0.0:
            finish_speed = min(finish_speed, max_speed)
        finish = finish_speed * goal_dir
        finish, limit_update = self._limit_speed_accel(finish, agent, record)
        before_safety = self._goal_projection(finish, goal_dir)
        projected, safety_update = self._safety_project_candidate(finish, normal, surface_distance, d_safe)
        rollout = self._rollout_candidate_safety(projected, state, agent, record)
        after_safety = self._goal_projection(projected, goal_dir)
        line_blocked = bool(
            record.get("line_to_goal_blocked_obstacle", False)
            or record.get("line_to_goal_blocked_terrain", False)
            or record.get("line_to_goal_blocked_agent", False)
        )
        conflict = bool(rollout.get("hard_violation", False))
        selected = projected
        micro_update: Dict[str, Any] = {"micro_waypoint_fan_active": False}
        if self.config.micro_waypoint_fan and (
            line_blocked
            or _safe_float(record.get("recent_goal_floor_safety_conflict_ratio"), 0.0) > 0.5
        ):
            selected, micro_update = self._micro_waypoint_fan_candidate(
                state=state,
                agent=agent,
                cmd=filtered,
                normal=normal,
                surface_distance=surface_distance,
                d_safe=d_safe,
                record=record,
                state_key=state_key,
                finish_speed=finish_speed,
            )
            conflict = bool(micro_update.get("micro_waypoint_fan_conflict", conflict))
            after_safety = self._goal_projection(selected, goal_dir)
        elif conflict:
            selected = filtered
        floor_target = float(target_info.get("goal_floor_target", self.config.goal_progress_floor))
        if after_safety is not None and after_safety + 1e-6 < floor_target:
            conflict = True
        update.update(
            {
                "single_laggard_finish_active": True,
                "single_laggard_finish_speed": finish_speed,
                "finish_goal_projection_before_safety": before_safety,
                "finish_goal_projection_after_safety": after_safety,
                "finish_safety_conflict": bool(conflict),
                "finish_line_to_goal_blocked": line_blocked,
                "finish_selected_cmd_vel": selected.astype(float).tolist(),
                "finish_rollout_hard_violation_reason": rollout.get("hard_violation_reason"),
                "finish_rollout_min_obstacle_clearance": rollout.get("min_obstacle_clearance"),
                "finish_rollout_min_terrain_clearance": rollout.get("min_terrain_clearance"),
                "finish_rollout_min_agent_clearance": rollout.get("min_agent_clearance"),
                **target_info,
                **{f"finish_{k}": v for k, v in limit_update.items()},
                "finish_safety_projection_delta_norm": safety_update.get("delta_norm"),
                **micro_update,
            }
        )
        return selected, update

    def _micro_waypoint_fan_candidate(
        self,
        state: GazeboAPFSceneState,
        agent: Any,
        cmd: np.ndarray,
        normal: Optional[np.ndarray],
        surface_distance: Optional[float],
        d_safe: Optional[float],
        record: Dict[str, Any],
        state_key: Tuple[int, str],
        finish_speed: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        goal_dir = self._vec3_or_none(record.get("goal_direction"))
        if goal_dir is None:
            return cmd.copy(), {
                "micro_waypoint_fan_active": True,
                "micro_waypoint_fan_conflict": True,
                "micro_waypoint_selected": None,
            }
        tangent = self._tangent_direction(state_key, normal, cmd, record) if normal is not None else None
        if tangent is None:
            tangent = np.asarray([-goal_dir[1], goal_dir[0], 0.0], dtype=np.float64)
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm <= 1e-9:
                tangent = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            else:
                tangent = tangent / tangent_norm
        up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        raw_dirs: Dict[str, np.ndarray] = {
            "goal": goal_dir,
            "goal_plus_tangent": 0.8 * goal_dir + 0.6 * tangent,
            "goal_minus_tangent": 0.8 * goal_dir - 0.6 * tangent,
            "tangent": tangent,
            "minus_tangent": -tangent,
            "goal_plus_up": goal_dir + 0.3 * up,
        }
        evaluated: Dict[str, Dict[str, Any]] = {}
        safe: Dict[str, np.ndarray] = {}
        for name, direction in raw_dirs.items():
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-9:
                continue
            candidate = finish_speed * direction / norm
            candidate, limit_update = self._limit_speed_accel(candidate, agent, record)
            projected, safety_update = self._safety_project_candidate(candidate, normal, surface_distance, d_safe)
            rollout = self._rollout_candidate_safety(projected, state, agent, record)
            evaluated[name] = {
                "goal_projection": self._goal_projection(projected, goal_dir),
                "predicted_goal_distance": rollout.get("predicted_goal_distance"),
                "hard_violation": rollout.get("hard_violation"),
                "hard_violation_reason": rollout.get("hard_violation_reason"),
                "min_obstacle_clearance": rollout.get("min_obstacle_clearance"),
                "min_terrain_clearance": rollout.get("min_terrain_clearance"),
                "min_agent_clearance": rollout.get("min_agent_clearance"),
                "speed_limited": limit_update.get("speed_limited"),
                "accel_limited": limit_update.get("accel_limited"),
                "projection_delta_norm": safety_update.get("delta_norm"),
            }
            if not bool(rollout.get("hard_violation", False)):
                safe[name] = projected
        if not safe:
            return cmd.copy(), {
                "micro_waypoint_fan_active": True,
                "micro_waypoint_fan_conflict": True,
                "micro_waypoint_selected": None,
                "micro_waypoint_candidates": evaluated,
            }
        selected_name = min(
            safe.keys(),
            key=lambda key: _safe_float(evaluated.get(key, {}).get("predicted_goal_distance"), float("inf")),
        )
        return safe[selected_name], {
            "micro_waypoint_fan_active": True,
            "micro_waypoint_fan_conflict": False,
            "micro_waypoint_selected": selected_name,
            "micro_waypoint_candidates": evaluated,
        }

    def _terrain_candidate_safety(
        self,
        candidate: np.ndarray,
        state: GazeboAPFSceneState,
        agent: Any,
        record: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.config.candidate_terrain_guard:
            return {"safe": True, "predicted_terrain_clearance": None}
        clearance_min = float(self.config.candidate_terrain_clearance_min)
        if clearance_min <= 0.0:
            return {"safe": True, "predicted_terrain_clearance": None}
        try:
            pos = self._vec3_or_none(record.get("pose"))
            cmd = np.asarray(candidate, dtype=np.float64).reshape(-1)[:3]
            if pos is None or cmd.size < 3 or not np.all(np.isfinite(cmd)):
                return {"safe": True, "predicted_terrain_clearance": None}
            dt = float(self.config.candidate_prediction_dt)
            predicted = pos + cmd * dt
            terrain_h = float(state.terrain.height(float(predicted[0]), float(predicted[1])))
            agent_radius = float(getattr(agent, "radius", record.get("agent_radius", 0.0)) or 0.0)
            clearance = float(predicted[2] - agent_radius - terrain_h)
            if clearance + 1e-6 < clearance_min:
                return {
                    "safe": False,
                    "reason": "unsafe_terrain_clearance",
                    "predicted_terrain_clearance": clearance,
                    "terrain_clearance_min": clearance_min,
                }
            return {
                "safe": True,
                "predicted_terrain_clearance": clearance,
                "terrain_clearance_min": clearance_min,
            }
        except Exception:
            return {"safe": True, "predicted_terrain_clearance": None}

    def _arbitrate_candidates(
        self,
        base: np.ndarray,
        cmd: np.ndarray,
        normal: Optional[np.ndarray],
        surface_distance: Optional[float],
        d_safe: Optional[float],
        agent: Any,
        record: Dict[str, Any],
        state_key: Tuple[int, str],
        state: GazeboAPFSceneState,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        base_name = "current_adaptive_escape"
        if bool(record.get("single_laggard_finish_active", False)):
            base_name = "single_laggard_finish"
        elif bool(record.get("arrived_hold_active", False)):
            base_name = "arrived_hold"
        elif bool(record.get("goal_floor_active", False)):
            base_name = "goal_progress_floor"
        candidates: Dict[str, np.ndarray] = {base_name: base.copy()}
        tangent = self._tangent_direction(state_key, normal, cmd, record) if normal is not None else None
        tangent_blocked_by_terrain = bool(
            self.config.candidate_disable_tangent_on_terrain_block
            and bool(record.get("line_to_goal_blocked_terrain", False))
        )
        if tangent is not None and not tangent_blocked_by_terrain:
            for floor in self.config.candidate_tangent_floors:
                current = float(np.dot(base, tangent))
                candidates[f"tangent_floor_{floor:g}"] = base + max(0.0, float(floor) - current) * tangent
        if self._should_apply_goal_floor(record):
            floor_cmd, floor_update = self._apply_goal_floor(base, normal=normal, record=record)
            if floor_update.get("goal_floor_active"):
                candidates["goal_progress_floor"] = floor_cmd
        if self.config.arrived_agent_hold and bool(record.get("agent_arrived", False)):
            hold_cmd, _ = self._arrived_hold_cmd(agent, record)
            candidates["arrived_hold"] = hold_cmd
        terrain_clearance_now = _safe_float(record.get("terrain_clearance"), float("nan"))
        terrain_recovery_threshold = max(
            float(self.config.candidate_terrain_clearance_min) * 2.0,
            float(self.config.safety_relax_clearance) + 1.0,
        )
        if (
            self.config.candidate_rollout_enabled
            and np.isfinite(terrain_clearance_now)
            and terrain_clearance_now < terrain_recovery_threshold
        ):
            recovery = base.copy()
            deficit = max(0.0, terrain_recovery_threshold - terrain_clearance_now)
            recovery[2] = max(float(recovery[2]), min(float(getattr(agent, "max_speed", 2.0) or 2.0), 0.8 + 0.5 * deficit))
            candidates["terrain_recovery_up"] = recovery
        prev = self._last_cmd_vel.get(int(record.get("agent_id", -1)), cmd)
        scores: Dict[str, float] = {}
        projections: Dict[str, Optional[float]] = {}
        rejected: Dict[str, str] = {}
        terrain_clearances: Dict[str, Optional[float]] = {}
        safety_clearances: Dict[str, Dict[str, Optional[float]]] = {}
        rollout_goal_distances: Dict[str, Optional[float]] = {}
        safe_candidates: Dict[str, np.ndarray] = {}
        reject_counts = {
            "terrain": 0,
            "obstacle": 0,
            "agent": 0,
            "speed": 0,
            "accel": 0,
            "projection_failed": 0,
        }
        for name, candidate in candidates.items():
            projected, safety = self._safety_project_candidate(candidate, normal, surface_distance, d_safe)
            if not bool(safety.get("safe", True)):
                rejected[name] = "unsafe_after_projection"
                reject_counts["projection_failed"] += 1
                continue
            projected, limit_update = self._limit_speed_accel(projected, agent, record)
            rollout = self._rollout_candidate_safety(projected, state, agent, record)
            terrain_clearances[name] = rollout.get("min_terrain_clearance")
            rollout_goal_distances[name] = rollout.get("predicted_goal_distance")
            safety_clearances[name] = {
                "obstacle": rollout.get("min_obstacle_clearance"),
                "terrain": rollout.get("min_terrain_clearance"),
                "agent": rollout.get("min_agent_clearance"),
                "soft_penalty": rollout.get("soft_penalty"),
            }
            if bool(rollout.get("hard_violation", False)):
                reason = str(rollout.get("hard_violation_reason") or "projection_failed")
                rejected[name] = reason
                reject_counts[self._candidate_reject_category(reason)] += 1
                continue
            safe_candidates[name] = projected
            score, projection = self._candidate_score(projected, cmd, prev, record)
            score -= float(self.config.candidate_soft_penalty_weight) * _safe_float(rollout.get("soft_penalty"), 0.0)
            if bool(limit_update.get("speed_limited", False)):
                score -= 0.02
            if bool(limit_update.get("accel_limited", False)):
                score -= 0.02
            scores[name] = score
            projections[name] = projection

        arbitration_active = len(candidates) > 1
        sorted_scores = sorted(scores.values(), reverse=True)
        score_margin = (
            float(sorted_scores[0] - sorted_scores[1])
            if len(sorted_scores) >= 2
            else None
        )
        accepted_count = int(len(safe_candidates))
        if not safe_candidates:
            fallback = None
            fallback_update: Dict[str, Any] = {}
            if (
                self.config.candidate_rollout_enabled
                and any(self._candidate_reject_category(reason) == "terrain" for reason in rejected.values())
                and np.isfinite(terrain_clearance_now)
                and terrain_clearance_now > 0.0
            ):
                recovery = base.copy()
                recovery[2] = max(float(recovery[2]), min(float(getattr(agent, "max_speed", 2.0) or 2.0), 1.2))
                fallback, fallback_update = self._limit_speed_accel(recovery, agent, record)
                scores["terrain_recovery_fallback"] = -1e6
                projections["terrain_recovery_fallback"] = self._goal_projection(
                    fallback,
                    record.get("goal_direction"),
                )
                safety_clearances["terrain_recovery_fallback"] = {
                    "obstacle": None,
                    "terrain": terrain_clearance_now,
                    "agent": None,
                    "soft_penalty": None,
                }
            return (fallback.copy() if fallback is not None else base.copy()), {
                "candidate_arbitration_active": arbitration_active,
                "selected_candidate_name": "terrain_recovery_fallback" if fallback is not None else base_name,
                "candidate_scores": scores,
                "candidate_goal_projections": projections,
                "candidate_rejections": rejected,
                "candidate_terrain_clearances": terrain_clearances,
                "candidate_safety_clearances": safety_clearances,
                "candidate_rollout_predicted_goal_distances": rollout_goal_distances,
                "accepted_candidate_count": accepted_count,
                "candidate_pool_collapsed": accepted_count <= 1,
                "candidate_tangent_suppressed_by_terrain": tangent_blocked_by_terrain,
                "reject_terrain_count": reject_counts["terrain"],
                "reject_obstacle_count": reject_counts["obstacle"],
                "reject_agent_count": reject_counts["agent"],
                "reject_speed_count": reject_counts["speed"],
                "reject_accel_count": reject_counts["accel"],
                "reject_projection_failed_count": reject_counts["projection_failed"],
                "best_candidate_score_margin": score_margin,
                "terrain_recovery_fallback_active": bool(fallback is not None),
                "terrain_recovery_fallback_speed_limited": fallback_update.get("speed_limited") if fallback is not None else False,
                "terrain_recovery_fallback_accel_limited": fallback_update.get("accel_limited") if fallback is not None else False,
            }
        selected = max(safe_candidates.keys(), key=lambda key: scores.get(key, -1e18))
        return safe_candidates[selected], {
            "candidate_arbitration_active": arbitration_active,
            "selected_candidate_name": selected,
            "candidate_scores": scores,
            "candidate_goal_projections": projections,
            "candidate_rejections": rejected,
            "candidate_terrain_clearances": terrain_clearances,
            "candidate_safety_clearances": safety_clearances,
            "candidate_rollout_predicted_goal_distances": rollout_goal_distances,
            "accepted_candidate_count": accepted_count,
            "candidate_pool_collapsed": accepted_count <= 1,
            "candidate_tangent_suppressed_by_terrain": tangent_blocked_by_terrain,
            "reject_terrain_count": reject_counts["terrain"],
            "reject_obstacle_count": reject_counts["obstacle"],
            "reject_agent_count": reject_counts["agent"],
            "reject_speed_count": reject_counts["speed"],
            "reject_accel_count": reject_counts["accel"],
            "reject_projection_failed_count": reject_counts["projection_failed"],
            "best_candidate_score_margin": score_margin,
        }

    def _candidate_score(
        self,
        candidate: np.ndarray,
        nominal: np.ndarray,
        prev: np.ndarray,
        record: Dict[str, Any],
    ) -> Tuple[float, Optional[float]]:
        goal_dir = self._vec3_or_none(record.get("goal_direction"))
        goal_projection = float(np.dot(candidate, goal_dir)) if goal_dir is not None else 0.0
        filter_delta = float(np.linalg.norm(candidate - nominal))
        smooth_delta = float(np.linalg.norm(candidate - _norm_or_zero(prev)))
        agent_closing = self._agent_agent_closing_penalty(candidate, record)
        formation_increase = self._predicted_formation_error_increase(candidate, record)
        score = (
            float(self.config.w_goal) * goal_projection
            - float(self.config.w_filter_delta) * filter_delta
            - float(self.config.w_smooth) * smooth_delta
            - float(self.config.w_agent_closing) * agent_closing
            - float(self.config.w_formation) * formation_increase
        )
        return float(score), float(goal_projection) if goal_dir is not None else None

    def _agent_agent_closing_penalty(self, candidate: np.ndarray, record: Dict[str, Any]) -> float:
        rel_pos = self._vec3_or_none(record.get("nearest_agent_relative_position"))
        if rel_pos is None:
            return 0.0
        rel_vel_other_minus_self = _norm_or_zero(record.get("nearest_agent_relative_velocity"))
        relative_candidate = candidate - rel_vel_other_minus_self
        closing = float(np.dot(relative_candidate, rel_pos))
        distance = _safe_float(record.get("nearest_agent_distance"), float("inf"))
        if not np.isfinite(distance):
            return max(0.0, closing)
        intensity = max(0.0, float(self.config.agent_agent_constraint_distance) - distance)
        return max(0.0, closing) * (1.0 + intensity)

    def _predicted_formation_error_increase(self, candidate: np.ndarray, record: Dict[str, Any]) -> float:
        current_error = _safe_float(record.get("formation_error_agent"), 0.0)
        pos = _norm_or_zero(record.get("pose"))
        goal = _norm_or_zero(record.get("goal"))
        nearest_rel = record.get("nearest_agent_relative_position")
        nearest_id = record.get("nearest_agent_id")
        if nearest_rel is None or nearest_id is None:
            return 0.0
        other_pos = pos + _norm_or_zero(nearest_rel)
        desired = _safe_float(record.get("nearest_agent_goal_distance"), 0.0)
        predicted_pos = pos + np.asarray(candidate, dtype=np.float64).reshape(-1)[:3] * 0.08
        predicted_error = abs(float(np.linalg.norm(other_pos - predicted_pos)) - desired)
        return max(0.0, predicted_error - current_error)


    def _apply_goal_projection_recovery(
        self,
        filtered: np.ndarray,
        normal: np.ndarray,
        surface_distance: float,
        d_safe: float,
        record: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if bool(record.get("line_to_goal_blocked", False)):
            return filtered, {"goal_projection_recovery_applied": False}
        goal_dir = self._vec3_or_none(record.get("goal_direction"))
        if goal_dir is None:
            return filtered, {"goal_projection_recovery_applied": False}
        before = float(np.dot(filtered, goal_dir))
        nominal_goal_projection = _safe_float(record.get("goal_projection_before_filter"), 0.0)
        target = max(float(self.config.min_goal_projection), float(self.config.recovery_goal_gain) * max(0.0, nominal_goal_projection))
        if before >= target:
            return filtered, {
                "goal_projection_recovery_applied": False,
                "goal_projection_recovery_target": target,
            }
        recovered = filtered + (target - before) * goal_dir
        recovered, projection_update = self._project_halfspace(recovered, normal, surface_distance, d_safe)
        return recovered, {
            "goal_projection_recovery_applied": True,
            "goal_projection_recovery_target": target,
            "goal_projection_before_recovery": before,
            "goal_projection_after_recovery": float(np.dot(recovered, goal_dir)),
            "goal_projection_recovery_projection_delta_norm": projection_update.get("halfspace_projection_delta_norm"),
        }

    def _tangent_direction(
        self,
        state_key: Tuple[int, str],
        normal: np.ndarray,
        cmd: np.ndarray,
        record: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        n = self._vec3_or_none(normal)
        if n is None:
            return None
        goal_dir = self._vec3_or_none(record.get("goal_direction"))
        entry = self._avoidance_state.setdefault(
            state_key,
            {"state": "FREE", "clear_count": 0, "boundary_dwell_steps": 0, "cached_tangent": None},
        )
        cached = self._vec3_or_none(entry.get("cached_tangent"))
        tangent = None
        if goal_dir is not None:
            candidate = goal_dir - float(np.dot(goal_dir, n)) * n
            candidate_norm = float(np.linalg.norm(candidate))
            if candidate_norm > 1e-6:
                tangent = candidate / candidate_norm
                if cached is not None and float(np.dot(tangent, cached)) < -0.25:
                    tangent = -tangent
        if tangent is None and cached is not None:
            tangent = cached
        if tangent is None:
            candidate = np.asarray([-n[1], n[0], 0.0], dtype=np.float64)
            candidate_norm = float(np.linalg.norm(candidate))
            if candidate_norm <= 1e-6:
                candidate = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
                candidate = candidate - float(np.dot(candidate, n)) * n
                candidate_norm = float(np.linalg.norm(candidate))
            if candidate_norm <= 1e-6:
                return None
            tangent = candidate / candidate_norm
            if float(np.dot(cmd, tangent)) < 0.0:
                tangent = -tangent
        entry["cached_tangent"] = tangent.astype(float).tolist()
        return tangent

    def _finalize_record(self, record: Dict[str, Any], cmd: np.ndarray, filtered: np.ndarray) -> None:
        record["final_cmd_vel"] = filtered.astype(float).tolist()
        obstacle_velocity = np.asarray(record.get("obstacle_velocity", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)[:3]
        if obstacle_velocity.size < 3 or not np.all(np.isfinite(obstacle_velocity)):
            obstacle_velocity = np.zeros(3, dtype=np.float64)
        relative_after = filtered - obstacle_velocity
        unit = record.get("unit_to_obstacle")
        tangential_before = self._tangential_speed(cmd, unit)
        tangential_after = self._tangential_speed(filtered, unit)
        cmd_delta = filtered - cmd
        cmd_delta_norm = float(np.linalg.norm(cmd_delta))
        goal_dir = record.get("goal_direction")
        normal = None
        unit_vec = self._vec3_or_none(unit)
        if unit_vec is not None:
            normal = -unit_vec
        outward_added = record.get("outward_velocity_added")
        if outward_added is None:
            outward_added = self._outward_component(cmd_delta, normal)
        filter_invasiveness = cmd_delta_norm / max(float(np.linalg.norm(cmd)), 1e-9)
        if tangential_before > 1e-9:
            tangential_ratio = tangential_after / tangential_before
        else:
            tangential_ratio = 1.0
        record["relative_velocity_after_filter"] = relative_after.astype(float).tolist()
        record["inward_velocity_after_filter"] = self._inward_velocity(filtered, unit)
        record["relative_inward_velocity_after_filter"] = self._inward_velocity(relative_after, unit)
        record["tangential_speed_after_filter"] = tangential_after
        record["tangential_velocity_after_filter"] = tangential_after
        record["tangential_velocity_before_filter"] = tangential_before
        record["tangential_velocity_kept_ratio"] = tangential_ratio
        record["cmd_delta_norm"] = cmd_delta_norm
        record["filter_invasiveness"] = filter_invasiveness
        filter_goal_projection = self._goal_projection(filtered, goal_dir)
        record["goal_projection_after_filter"] = filter_goal_projection
        record["filter_goal_projection"] = filter_goal_projection
        if record.get("apf_goal_projection") is not None and filter_goal_projection is not None:
            record["filter_goal_projection_delta"] = float(filter_goal_projection - record["apf_goal_projection"])
        else:
            record["filter_goal_projection_delta"] = None
        record["outward_velocity_added"] = outward_added
        record.setdefault("outward_speed_applied", outward_added)
        record.setdefault("halfspace_projection_delta_norm", 0.0)
        record.setdefault("arrived_hold_active", False)
        record.setdefault("goal_floor_active", False)
        record.setdefault("goal_floor_safety_conflict", False)
        record.setdefault("candidate_arbitration_active", False)
        record.setdefault("selected_candidate_name", "none")
        record.setdefault("candidate_scores", {})
        record.setdefault("candidate_goal_projections", {})
        record.setdefault("candidate_terrain_clearances", {})
        record.setdefault("candidate_safety_clearances", {})
        record.setdefault("candidate_rollout_predicted_goal_distances", {})
        record.setdefault("accepted_candidate_count", 0)
        record.setdefault("candidate_pool_collapsed", False)
        record.setdefault("candidate_tangent_suppressed_by_terrain", False)
        record.setdefault("reject_terrain_count", 0)
        record.setdefault("reject_obstacle_count", 0)
        record.setdefault("reject_agent_count", 0)
        record.setdefault("reject_speed_count", 0)
        record.setdefault("reject_accel_count", 0)
        record.setdefault("reject_projection_failed_count", 0)
        record.setdefault("best_candidate_score_margin", None)
        record.setdefault("terrain_recovery_fallback_active", False)
        record.setdefault("terrain_recovery_fallback_speed_limited", False)
        record.setdefault("terrain_recovery_fallback_accel_limited", False)
        record.setdefault("single_laggard_finish_active", False)
        record.setdefault("finish_goal_projection_before_safety", None)
        record.setdefault("finish_goal_projection_after_safety", None)
        record.setdefault("finish_safety_conflict", False)
        record.setdefault("finish_line_to_goal_blocked", False)
        record.setdefault("single_laggard_finish_candidate", False)
        record.setdefault("micro_waypoint_fan_active", False)
        record["boundary_dwell_steps"] = self._update_boundary_dwell(record)

    def _update_boundary_dwell(self, record: Dict[str, Any]) -> int:
        obstacle_id = record.get("nearest_obstacle_id")
        if obstacle_id is None:
            return 0
        key = (int(record.get("agent_id", -1)), str(obstacle_id))
        entry = self._avoidance_state.setdefault(
            key,
            {"state": "FREE", "clear_count": 0, "boundary_dwell_steps": 0, "cached_tangent": None},
        )
        surface = _safe_float(record.get("surface_distance"), float("nan"))
        d_enter = _safe_float(record.get("d_enter"), float("nan"))
        goal_after = _safe_float(record.get("goal_projection_after_filter"), float("nan"))
        near_boundary = np.isfinite(surface) and np.isfinite(d_enter) and surface <= d_enter + self.config.boundary_band
        slow_progress = (not np.isfinite(goal_after)) or goal_after <= self.config.boundary_speed_threshold
        state = str(record.get("avoidance_state") or entry.get("state") or "FREE")
        if state != "FREE" and near_boundary and (slow_progress or bool(record.get("filter_active", False))):
            entry["boundary_dwell_steps"] = int(entry.get("boundary_dwell_steps", 0) or 0) + 1
        elif state == "FREE" or not near_boundary:
            entry["boundary_dwell_steps"] = 0
        return int(entry.get("boundary_dwell_steps", 0) or 0)

    @staticmethod
    def _vec3_or_none(values: Any) -> Optional[np.ndarray]:
        try:
            arr = np.asarray(values, dtype=np.float64).reshape(-1)[:3]
            if arr.size >= 3 and np.all(np.isfinite(arr)):
                norm = float(np.linalg.norm(arr))
                if norm > 1e-9:
                    return (arr / norm).astype(np.float64, copy=True)
        except Exception:
            pass
        return None

    @staticmethod
    def _raw_vec3_or_none(values: Any) -> Optional[np.ndarray]:
        try:
            arr = np.asarray(values, dtype=np.float64).reshape(-1)[:3]
            if arr.size >= 3 and np.all(np.isfinite(arr)):
                return arr.astype(np.float64, copy=True)
        except Exception:
            pass
        return None

    @staticmethod
    def _goal_projection(cmd: np.ndarray, goal_dir: Optional[Sequence[float]]) -> Optional[float]:
        direction = GazeboObstacleVelocitySafetyFilter._vec3_or_none(goal_dir)
        if direction is None:
            return None
        try:
            arr = np.asarray(cmd, dtype=np.float64).reshape(-1)[:3]
            if arr.size >= 3 and np.all(np.isfinite(arr)):
                return float(np.dot(arr, direction))
        except Exception:
            pass
        return None

    @staticmethod
    def _outward_component(delta: np.ndarray, normal: Optional[Sequence[float]]) -> float:
        direction = GazeboObstacleVelocitySafetyFilter._vec3_or_none(normal)
        if direction is None:
            return 0.0
        try:
            arr = np.asarray(delta, dtype=np.float64).reshape(-1)[:3]
            if arr.size >= 3 and np.all(np.isfinite(arr)):
                return float(max(0.0, np.dot(arr, direction)))
        except Exception:
            pass
        return 0.0

    def _clamp_speed(self, cmd: np.ndarray, max_speed: float) -> np.ndarray:
        out = np.asarray(cmd, dtype=np.float64).reshape(-1)[:3].copy()
        if not self.config.clamp_to_max_speed:
            return np.where(np.isfinite(out), out, 0.0)
        limit = float(max_speed or 0.0)
        speed = float(np.linalg.norm(out))
        if limit > 0.0 and np.isfinite(speed) and speed > limit:
            out = out / max(speed, 1e-9) * limit
        return np.where(np.isfinite(out), out, 0.0)

    @staticmethod
    def _inward_velocity(cmd: np.ndarray, unit: Optional[Sequence[float]]) -> float:
        if unit is None:
            return 0.0
        try:
            u = np.asarray(unit, dtype=np.float64).reshape(-1)[:3]
            if u.size < 3 or not np.all(np.isfinite(u)):
                return 0.0
            return float(np.dot(np.asarray(cmd, dtype=np.float64).reshape(-1)[:3], u))
        except Exception:
            return 0.0

    @staticmethod
    def _tangential_speed(cmd: np.ndarray, unit: Optional[Sequence[float]]) -> float:
        if unit is None:
            try:
                return float(np.linalg.norm(np.asarray(cmd, dtype=np.float64).reshape(-1)[:3]))
            except Exception:
                return 0.0
        try:
            c = np.asarray(cmd, dtype=np.float64).reshape(-1)[:3]
            u = np.asarray(unit, dtype=np.float64).reshape(-1)[:3]
            inward = float(np.dot(c, u))
            tangent = c - inward * u
            return float(np.linalg.norm(tangent))
        except Exception:
            return 0.0


def refine_progress_failure_labels(records: Sequence[Dict[str, Any]]) -> List[str]:
    flat = [r for r in records or [] if isinstance(r, dict)]
    if not flat:
        return []

    def mean_bool(key: str) -> float:
        return float(np.mean([1.0 if bool(r.get(key, False)) else 0.0 for r in flat]))

    def values(key: str) -> List[float]:
        out: List[float] = []
        for r in flat:
            value = _safe_float(r.get(key), float("nan"))
            if np.isfinite(value):
                out.append(value)
        return out

    def mean_value(key: str, default: float = 0.0) -> float:
        vals = values(key)
        return float(np.mean(vals)) if vals else default

    labels: List[str] = []
    arrived_agents = {
        int(r.get("agent_id", -1))
        for r in flat
        if bool(r.get("agent_arrived", False))
    }
    non_arrived_laggards = [
        r
        for r in flat
        if not bool(r.get("agent_arrived", False)) and _safe_float(r.get("goal_distance"), 0.0) > 1.0
    ]
    if arrived_agents and non_arrived_laggards and mean_value("formation_error_agent") > 5.0:
        labels.append("ARRIVED_AGENT_DRAG")
    if non_arrived_laggards:
        stalled_max = max(values("stalled_steps") or [0.0])
        if stalled_max >= 80.0:
            labels.append("LAGGARD_LIVENESS_FAILURE")
    raw_goal = mean_value("raw_goal_projection")
    apf_goal = mean_value("apf_goal_projection")
    filter_goal = mean_value("filter_goal_projection")
    if apf_goal > 0.2 and filter_goal < max(0.05, apf_goal * 0.5):
        labels.append("FILTER_SUPPRESSED_GOAL_PROGRESS")
    if raw_goal > 0.2 and apf_goal < max(0.05, raw_goal * 0.5):
        labels.append("APF_SUPPRESSED_GOAL_PROGRESS")
    if raw_goal < 0.05 and mean_value("goal_distance") > 5.0:
        labels.append("RAW_POLICY_NO_GOAL_PROGRESS")
    if mean_bool("agent_agent_constraint_active") >= 0.05:
        labels.append("AGENT_AGENT_CONGESTION")
    if mean_value("formation_error_agent") >= 5.0 and filter_goal < 0.8:
        labels.append("FORMATION_OVER_GOAL")
    accepted = [
        _safe_float(r.get("accepted_candidate_count"), float("nan"))
        for r in flat
        if bool(r.get("candidate_arbitration_active", False))
    ]
    accepted = [v for v in accepted if np.isfinite(v)]
    if accepted and float(np.mean(accepted)) <= 1.1:
        labels.append("CANDIDATE_POOL_COLLAPSED")
    return list(dict.fromkeys(labels))


def summarize_velocity_filter_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    flat = [r for r in records or [] if isinstance(r, dict)]

    def vals(key: str) -> List[float]:
        out: List[float] = []
        for r in flat:
            value = _safe_float(r.get(key), float("nan"))
            if np.isfinite(value):
                out.append(float(value))
        return out

    def mean(key: str) -> Optional[float]:
        v = vals(key)
        return float(np.mean(v)) if v else None

    def minv(key: str) -> Optional[float]:
        v = vals(key)
        return float(np.min(v)) if v else None

    def maxv(key: str) -> Optional[float]:
        v = vals(key)
        return float(np.max(v)) if v else None

    active = [r for r in flat if bool(r.get("filter_active", False))]
    steps = []
    for r in flat:
        try:
            if r.get("step") is not None:
                steps.append(int(r.get("step")))
        except Exception:
            pass
    active_steps = []
    for r in active:
        try:
            if r.get("step") is not None:
                active_steps.append(int(r.get("step")))
        except Exception:
            pass
    modes = {}
    for r in flat:
        mode = str(r.get("mode") or "unknown")
        modes[mode] = int(modes.get(mode, 0) + 1)
    candidate_counts = Counter(str(r.get("selected_candidate_name") or "none") for r in flat)
    refined_classes = refine_progress_failure_labels(flat)
    accepted_candidate_values = [
        _safe_float(r.get("accepted_candidate_count"), float("nan"))
        for r in flat
        if bool(r.get("candidate_arbitration_active", False))
    ]
    accepted_candidate_values = [v for v in accepted_candidate_values if np.isfinite(v)]
    return {
        "record_count": int(len(flat)),
        "active_count": int(len(active)),
        "active_rate": float(len(active) / len(flat)) if flat else None,
        "step_count": int(len(set(steps))) if steps else 0,
        "active_step_count": int(len(set(active_steps))) if active_steps else 0,
        "first_active_step": int(min(active_steps)) if active_steps else None,
        "last_active_step": int(max(active_steps)) if active_steps else None,
        "mode_counts": modes,
        "min_surface_distance": minv("surface_distance"),
        "min_clearance": minv("clearance"),
        "mean_inward_velocity_before_filter": mean("inward_velocity_before_filter"),
        "max_inward_velocity_before_filter": maxv("inward_velocity_before_filter"),
        "mean_inward_velocity_after_filter": mean("inward_velocity_after_filter"),
        "max_inward_velocity_after_filter": maxv("inward_velocity_after_filter"),
        "mean_relative_inward_velocity_before_filter": mean("relative_inward_velocity_before_filter"),
        "max_relative_inward_velocity_before_filter": maxv("relative_inward_velocity_before_filter"),
        "mean_relative_inward_velocity_after_filter": mean("relative_inward_velocity_after_filter"),
        "max_relative_inward_velocity_after_filter": maxv("relative_inward_velocity_after_filter"),
        "mean_current_relative_inward_velocity": mean("current_relative_inward_velocity"),
        "max_current_relative_inward_velocity": maxv("current_relative_inward_velocity"),
        "mean_closing_inward_velocity_for_stopping": mean("closing_inward_velocity_for_stopping"),
        "max_closing_inward_velocity_for_stopping": maxv("closing_inward_velocity_for_stopping"),
        "mean_cmd_delta_norm": mean("cmd_delta_norm"),
        "max_cmd_delta_norm": maxv("cmd_delta_norm"),
        "mean_outward_speed_applied": mean("outward_speed_applied"),
        "max_outward_speed_applied": maxv("outward_speed_applied"),
        "mean_goal_projection_before_filter": mean("goal_projection_before_filter"),
        "mean_goal_projection_after_filter": mean("goal_projection_after_filter"),
        "mean_tangential_velocity_before_filter": mean("tangential_velocity_before_filter"),
        "mean_tangential_velocity_after_filter": mean("tangential_velocity_after_filter"),
        "mean_tangential_velocity_kept_ratio": mean("tangential_velocity_kept_ratio"),
        "mean_filter_invasiveness": mean("filter_invasiveness"),
        "max_filter_invasiveness": maxv("filter_invasiveness"),
        "max_boundary_dwell_steps": maxv("boundary_dwell_steps"),
        "mean_boundary_dwell_steps": mean("boundary_dwell_steps"),
        "line_to_goal_blocked_rate": (
            float(np.mean([1.0 if bool(r.get("line_to_goal_blocked", False)) else 0.0 for r in flat]))
            if flat
            else None
        ),
        "line_to_goal_blocked_obstacle_rate": (
            float(np.mean([1.0 if bool(r.get("line_to_goal_blocked_obstacle", False)) else 0.0 for r in flat]))
            if flat
            else None
        ),
        "line_to_goal_blocked_terrain_rate": (
            float(np.mean([1.0 if bool(r.get("line_to_goal_blocked_terrain", False)) else 0.0 for r in flat]))
            if flat
            else None
        ),
        "line_to_goal_blocked_agent_rate": (
            float(np.mean([1.0 if bool(r.get("line_to_goal_blocked_agent", False)) else 0.0 for r in flat]))
            if flat
            else None
        ),
        "line_to_goal_blocked_any_direct_rate": (
            float(np.mean([1.0 if bool(r.get("line_to_goal_blocked_any_direct", False)) else 0.0 for r in flat]))
            if flat
            else None
        ),
        "line_to_goal_min_obstacle_clearance_min": minv("line_to_goal_min_obstacle_clearance"),
        "line_to_goal_min_terrain_clearance_min": minv("line_to_goal_min_terrain_clearance"),
        "line_to_goal_min_agent_clearance_min": minv("line_to_goal_min_agent_clearance"),
        "boundary_dwell_ratio": (
            float(np.mean([1.0 if _safe_float(r.get("boundary_dwell_steps"), 0.0) > 0.0 else 0.0 for r in flat]))
            if flat
            else None
        ),
        "python_clearance_violation_rate": (
            float(
                np.mean(
                    [
                        1.0
                        if (
                            np.isfinite(_safe_float(r.get("clearance"), float("nan")))
                            and _safe_float(r.get("clearance"), float("nan"))
                            < _safe_float(r.get("collision_distance_threshold"), 0.5)
                        )
                        else 0.0
                        for r in flat
                    ]
                )
            )
            if flat
            else None
        ),
        "geometric_penetration_rate": (
            float(
                np.mean(
                    [
                        1.0
                        if (
                            np.isfinite(_safe_float(r.get("clearance"), float("nan")))
                            and _safe_float(r.get("clearance"), float("nan")) < 0.0
                        )
                        else 0.0
                        for r in flat
                    ]
                )
            )
            if flat
            else None
        ),
        "mean_nearest_agent_distance": mean("nearest_agent_distance"),
        "min_nearest_agent_distance": minv("nearest_agent_distance"),
        "mean_formation_error": mean("formation_error"),
        "max_formation_error": maxv("formation_error"),
        "mean_goal_progress_rate": mean("goal_progress_rate"),
        "max_stalled_steps": maxv("stalled_steps"),
        "arrived_hold_active_rate": (
            float(np.mean([1.0 if bool(r.get("arrived_hold_active", False)) else 0.0 for r in flat]))
            if flat
            else None
        ),
        "goal_floor_active_rate": (
            float(np.mean([1.0 if bool(r.get("goal_floor_active", False)) else 0.0 for r in flat]))
            if flat
            else None
        ),
        "goal_floor_safety_conflict_rate": (
            float(np.mean([1.0 if bool(r.get("goal_floor_safety_conflict", False)) else 0.0 for r in flat]))
            if flat
            else None
        ),
        "single_laggard_finish_active_rate": (
            float(np.mean([1.0 if bool(r.get("single_laggard_finish_active", False)) else 0.0 for r in flat]))
            if flat
            else None
        ),
        "finish_safety_conflict_rate": (
            float(np.mean([1.0 if bool(r.get("finish_safety_conflict", False)) else 0.0 for r in flat]))
            if flat
            else None
        ),
        "candidate_pool_collapsed_rate": (
            float(
                np.mean(
                    [
                        1.0 if bool(r.get("candidate_pool_collapsed", False)) else 0.0
                        for r in flat
                        if bool(r.get("candidate_arbitration_active", False))
                    ]
                )
            )
            if any(bool(r.get("candidate_arbitration_active", False)) for r in flat)
            else None
        ),
        "terrain_recovery_fallback_active_rate": (
            float(np.mean([1.0 if bool(r.get("terrain_recovery_fallback_active", False)) else 0.0 for r in flat]))
            if flat
            else None
        ),
        "accepted_candidate_count_mean": (
            float(np.mean(accepted_candidate_values)) if accepted_candidate_values else None
        ),
        "reject_terrain_count_sum": float(np.sum(vals("reject_terrain_count"))) if flat else None,
        "reject_obstacle_count_sum": float(np.sum(vals("reject_obstacle_count"))) if flat else None,
        "reject_agent_count_sum": float(np.sum(vals("reject_agent_count"))) if flat else None,
        "reject_speed_count_sum": float(np.sum(vals("reject_speed_count"))) if flat else None,
        "reject_accel_count_sum": float(np.sum(vals("reject_accel_count"))) if flat else None,
        "reject_projection_failed_count_sum": float(np.sum(vals("reject_projection_failed_count"))) if flat else None,
        "candidate_selected_counts": dict(candidate_counts),
        "refined_failure_labels": refined_classes,
    }


PROGRESS_DIAGNOSTIC_COLUMNS = [
    "episode",
    "step",
    "agent_id",
    "mode",
    "raw_actor_accel",
    "corrected_accel",
    "shadow_raw_cmd_vel",
    "goal_distance",
    "nominal_cmd_vel",
    "filtered_cmd_vel",
    "raw_goal_projection",
    "apf_goal_projection",
    "filter_goal_projection",
    "apf_goal_projection_delta",
    "filter_goal_projection_delta",
    "agent_arrived",
    "agent_laggard_rank",
    "goal_progress_rate",
    "stalled_steps",
    "recent_filter_active_ratio",
    "recent_boundary_dwell_ratio",
    "recent_goal_projection",
    "recent_goal_floor_safety_conflict_ratio",
    "team_centroid",
    "filter_active",
    "nearest_obstacle_id",
    "surface_distance",
    "goal_projection_before_filter",
    "goal_projection_after_filter",
    "inward_velocity_before_filter",
    "inward_velocity_after_filter",
    "tangential_velocity_before_filter",
    "tangential_velocity_after_filter",
    "tangential_velocity_kept_ratio",
    "outward_velocity_added",
    "filter_invasiveness",
    "boundary_dwell_steps",
    "line_to_goal_blocked",
    "line_to_goal_blocked_obstacle",
    "line_to_goal_blocked_terrain",
    "line_to_goal_blocked_agent",
    "line_to_goal_blocked_any_direct",
    "line_to_goal_min_obstacle_clearance",
    "line_to_goal_min_terrain_clearance",
    "line_to_goal_min_agent_clearance",
    "nearest_agent_id",
    "nearest_agent_distance",
    "agent_agent_constraint_active",
    "formation_error",
    "formation_error_agent",
    "terrain_height",
    "terrain_clearance",
    "clearance",
    "collision_distance_threshold",
    "geometric_penetration",
    "python_clearance_violation",
    "avoidance_state",
    "filter_trigger_reason",
    "line_to_goal_clearance",
    "halfspace_lower_bound",
    "normal_velocity_before_projection",
    "normal_velocity_after_projection",
    "halfspace_projection_delta_norm",
    "tangent_recovery_applied",
    "goal_projection_recovery_applied",
    "boundary_escape_tangent_boost_applied",
    "boundary_escape_tangent_target_speed",
    "boundary_escape_dwell_threshold",
    "boundary_escape_goal_projection",
    "arrived_hold_active",
    "arrived_hold_cmd_vel",
    "goal_floor_active",
    "goal_floor_safety_conflict",
    "goal_floor_target",
    "adaptive_goal_floor_active",
    "remaining_time",
    "goal_progress_required_speed",
    "goal_floor_before_projection",
    "goal_floor_after_projection",
    "goal_floor_after_safety_projection",
    "arrived_count",
    "agent_count",
    "single_laggard_finish_candidate",
    "single_laggard_finish_active",
    "finish_goal_projection_before_safety",
    "finish_goal_projection_after_safety",
    "finish_safety_conflict",
    "finish_line_to_goal_blocked",
    "single_laggard_finish_speed",
    "finish_rollout_hard_violation_reason",
    "finish_rollout_min_obstacle_clearance",
    "finish_rollout_min_terrain_clearance",
    "finish_rollout_min_agent_clearance",
    "micro_waypoint_fan_active",
    "micro_waypoint_selected",
    "micro_waypoint_candidates",
    "candidate_arbitration_active",
    "selected_candidate_name",
    "accepted_candidate_count",
    "reject_terrain_count",
    "reject_obstacle_count",
    "reject_agent_count",
    "reject_speed_count",
    "reject_accel_count",
    "reject_projection_failed_count",
    "best_candidate_score_margin",
    "candidate_pool_collapsed",
    "candidate_tangent_suppressed_by_terrain",
    "terrain_recovery_fallback_active",
    "terrain_recovery_fallback_speed_limited",
    "terrain_recovery_fallback_accel_limited",
    "candidate_scores",
    "candidate_goal_projections",
    "candidate_terrain_clearances",
    "candidate_safety_clearances",
    "candidate_rollout_predicted_goal_distances",
    "candidate_rejections",
]


def _csv_value(value: Any) -> Any:
    safe = _json_safe(value)
    if isinstance(safe, (list, dict)):
        return json.dumps(safe, ensure_ascii=False, separators=(",", ":"))
    return safe


def progress_diagnostic_rows(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        clearance = _safe_float(record.get("clearance"), float("nan"))
        threshold = _safe_float(record.get("collision_distance_threshold"), 0.5)
        row = {
            "episode": record.get("episode"),
            "step": record.get("step"),
            "agent_id": record.get("agent_id"),
            "mode": record.get("mode"),
            "raw_actor_accel": record.get("raw_actor_accel"),
            "corrected_accel": record.get("corrected_accel"),
            "shadow_raw_cmd_vel": record.get("shadow_raw_cmd_vel"),
            "goal_distance": record.get("goal_distance"),
            "nominal_cmd_vel": record.get("nominal_cmd_vel"),
            "filtered_cmd_vel": record.get("final_cmd_vel"),
            "raw_goal_projection": record.get("raw_goal_projection"),
            "apf_goal_projection": record.get("apf_goal_projection"),
            "filter_goal_projection": record.get("filter_goal_projection"),
            "apf_goal_projection_delta": record.get("apf_goal_projection_delta"),
            "filter_goal_projection_delta": record.get("filter_goal_projection_delta"),
            "agent_arrived": int(bool(record.get("agent_arrived", False))),
            "agent_laggard_rank": record.get("agent_laggard_rank"),
            "goal_progress_rate": record.get("goal_progress_rate"),
            "stalled_steps": record.get("stalled_steps"),
            "recent_filter_active_ratio": record.get("recent_filter_active_ratio"),
            "recent_boundary_dwell_ratio": record.get("recent_boundary_dwell_ratio"),
            "recent_goal_projection": record.get("recent_goal_projection"),
            "recent_goal_floor_safety_conflict_ratio": record.get("recent_goal_floor_safety_conflict_ratio"),
            "team_centroid": record.get("team_centroid"),
            "filter_active": int(bool(record.get("filter_active", False))),
            "nearest_obstacle_id": record.get("nearest_obstacle_id"),
            "surface_distance": record.get("surface_distance"),
            "goal_projection_before_filter": record.get("goal_projection_before_filter"),
            "goal_projection_after_filter": record.get("goal_projection_after_filter"),
            "inward_velocity_before_filter": record.get("inward_velocity_before_filter"),
            "inward_velocity_after_filter": record.get("inward_velocity_after_filter"),
            "tangential_velocity_before_filter": record.get("tangential_velocity_before_filter"),
            "tangential_velocity_after_filter": record.get("tangential_velocity_after_filter"),
            "tangential_velocity_kept_ratio": record.get("tangential_velocity_kept_ratio"),
            "outward_velocity_added": record.get("outward_velocity_added", record.get("outward_speed_applied")),
            "filter_invasiveness": record.get("filter_invasiveness"),
            "boundary_dwell_steps": record.get("boundary_dwell_steps"),
            "line_to_goal_blocked": int(bool(record.get("line_to_goal_blocked", False))),
            "line_to_goal_blocked_obstacle": int(bool(record.get("line_to_goal_blocked_obstacle", False))),
            "line_to_goal_blocked_terrain": int(bool(record.get("line_to_goal_blocked_terrain", False))),
            "line_to_goal_blocked_agent": int(bool(record.get("line_to_goal_blocked_agent", False))),
            "line_to_goal_blocked_any_direct": int(bool(record.get("line_to_goal_blocked_any_direct", False))),
            "line_to_goal_min_obstacle_clearance": record.get("line_to_goal_min_obstacle_clearance"),
            "line_to_goal_min_terrain_clearance": record.get("line_to_goal_min_terrain_clearance"),
            "line_to_goal_min_agent_clearance": record.get("line_to_goal_min_agent_clearance"),
            "nearest_agent_id": record.get("nearest_agent_id"),
            "nearest_agent_distance": record.get("nearest_agent_distance"),
            "agent_agent_constraint_active": int(bool(record.get("agent_agent_constraint_active", False))),
            "formation_error": record.get("formation_error"),
            "formation_error_agent": record.get("formation_error_agent"),
            "terrain_height": record.get("terrain_height"),
            "terrain_clearance": record.get("terrain_clearance"),
            "clearance": record.get("clearance"),
            "collision_distance_threshold": record.get("collision_distance_threshold"),
            "geometric_penetration": int(np.isfinite(clearance) and clearance < 0.0),
            "python_clearance_violation": int(np.isfinite(clearance) and clearance < threshold),
            "avoidance_state": record.get("avoidance_state"),
            "filter_trigger_reason": record.get("filter_trigger_reason"),
            "line_to_goal_clearance": record.get("line_to_goal_clearance"),
            "halfspace_lower_bound": record.get("halfspace_lower_bound"),
            "normal_velocity_before_projection": record.get("normal_velocity_before_projection"),
            "normal_velocity_after_projection": record.get("normal_velocity_after_projection"),
            "halfspace_projection_delta_norm": record.get("halfspace_projection_delta_norm"),
            "tangent_recovery_applied": int(bool(record.get("tangent_recovery_applied", False))),
            "goal_projection_recovery_applied": int(bool(record.get("goal_projection_recovery_applied", False))),
            "boundary_escape_tangent_boost_applied": int(
                bool(record.get("boundary_escape_tangent_boost_applied", False))
            ),
            "boundary_escape_tangent_target_speed": record.get("boundary_escape_tangent_target_speed"),
            "boundary_escape_dwell_threshold": record.get("boundary_escape_dwell_threshold"),
            "boundary_escape_goal_projection": record.get("boundary_escape_goal_projection"),
            "arrived_hold_active": int(bool(record.get("arrived_hold_active", False))),
            "arrived_hold_cmd_vel": record.get("arrived_hold_cmd_vel"),
            "goal_floor_active": int(bool(record.get("goal_floor_active", False))),
            "goal_floor_safety_conflict": int(bool(record.get("goal_floor_safety_conflict", False))),
            "goal_floor_target": record.get("goal_floor_target"),
            "adaptive_goal_floor_active": int(bool(record.get("adaptive_goal_floor_active", False))),
            "remaining_time": record.get("remaining_time"),
            "goal_progress_required_speed": record.get("goal_progress_required_speed"),
            "goal_floor_before_projection": record.get("goal_floor_before_projection"),
            "goal_floor_after_projection": record.get("goal_floor_after_projection"),
            "goal_floor_after_safety_projection": record.get("goal_floor_after_safety_projection"),
            "arrived_count": record.get("arrived_count"),
            "agent_count": record.get("agent_count"),
            "single_laggard_finish_candidate": int(bool(record.get("single_laggard_finish_candidate", False))),
            "single_laggard_finish_active": int(bool(record.get("single_laggard_finish_active", False))),
            "finish_goal_projection_before_safety": record.get("finish_goal_projection_before_safety"),
            "finish_goal_projection_after_safety": record.get("finish_goal_projection_after_safety"),
            "finish_safety_conflict": int(bool(record.get("finish_safety_conflict", False))),
            "finish_line_to_goal_blocked": int(bool(record.get("finish_line_to_goal_blocked", False))),
            "single_laggard_finish_speed": record.get("single_laggard_finish_speed"),
            "finish_rollout_hard_violation_reason": record.get("finish_rollout_hard_violation_reason"),
            "finish_rollout_min_obstacle_clearance": record.get("finish_rollout_min_obstacle_clearance"),
            "finish_rollout_min_terrain_clearance": record.get("finish_rollout_min_terrain_clearance"),
            "finish_rollout_min_agent_clearance": record.get("finish_rollout_min_agent_clearance"),
            "micro_waypoint_fan_active": int(bool(record.get("micro_waypoint_fan_active", False))),
            "micro_waypoint_selected": record.get("micro_waypoint_selected"),
            "micro_waypoint_candidates": record.get("micro_waypoint_candidates"),
            "candidate_arbitration_active": int(bool(record.get("candidate_arbitration_active", False))),
            "selected_candidate_name": record.get("selected_candidate_name"),
            "accepted_candidate_count": record.get("accepted_candidate_count"),
            "reject_terrain_count": record.get("reject_terrain_count"),
            "reject_obstacle_count": record.get("reject_obstacle_count"),
            "reject_agent_count": record.get("reject_agent_count"),
            "reject_speed_count": record.get("reject_speed_count"),
            "reject_accel_count": record.get("reject_accel_count"),
            "reject_projection_failed_count": record.get("reject_projection_failed_count"),
            "best_candidate_score_margin": record.get("best_candidate_score_margin"),
            "candidate_pool_collapsed": int(bool(record.get("candidate_pool_collapsed", False))),
            "candidate_tangent_suppressed_by_terrain": int(
                bool(record.get("candidate_tangent_suppressed_by_terrain", False))
            ),
            "terrain_recovery_fallback_active": int(bool(record.get("terrain_recovery_fallback_active", False))),
            "terrain_recovery_fallback_speed_limited": int(
                bool(record.get("terrain_recovery_fallback_speed_limited", False))
            ),
            "terrain_recovery_fallback_accel_limited": int(
                bool(record.get("terrain_recovery_fallback_accel_limited", False))
            ),
            "candidate_scores": record.get("candidate_scores"),
            "candidate_goal_projections": record.get("candidate_goal_projections"),
            "candidate_terrain_clearances": record.get("candidate_terrain_clearances"),
            "candidate_safety_clearances": record.get("candidate_safety_clearances"),
            "candidate_rollout_predicted_goal_distances": record.get("candidate_rollout_predicted_goal_distances"),
            "candidate_rejections": record.get("candidate_rejections"),
        }
        rows.append({key: _csv_value(row.get(key)) for key in PROGRESS_DIAGNOSTIC_COLUMNS})
    return rows


def classify_progress_failure(
    records: Sequence[Dict[str, Any]],
    episode_data: Optional[Dict[str, Any]] = None,
    summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    episode_data = episode_data or {}
    summary = dict(summary or summarize_velocity_filter_records(records))
    try:
        steps = int(episode_data.get("steps", 0) or 0)
        episode_length = int(episode_data.get("episode_length", 0) or 0)
    except Exception:
        steps = 0
        episode_length = 0
    done_reason = str(episode_data.get("episode_done_reason") or "").strip().lower()
    success = bool(episode_data.get("team_success", episode_data.get("success", False)))
    hard_contact = bool((episode_data.get("gazebo_contact_count", 0) or 0) > 0 or episode_data.get("first_contact_step") is not None)
    timeout = (done_reason in ("timeout", "time_limit", "max_steps", "episode_length")) or (
        episode_length > 0 and steps >= episode_length and not success and not hard_contact
    )
    classes: List[str] = []
    evidence: Dict[str, Any] = {
        "timeout": bool(timeout),
        "done_reason": done_reason,
        "steps": steps,
        "episode_length": episode_length,
        "success": bool(success),
        "hard_contact": bool(hard_contact),
    }
    if not timeout:
        return {
            "timeout": False,
            "failure_classes": classes,
            "evidence": {**evidence, **summary},
        }

    boundary_ratio = _safe_float(summary.get("boundary_dwell_ratio"), 0.0)
    active_ratio = _safe_float(summary.get("active_rate"), 0.0)
    blocked_rate = _safe_float(summary.get("line_to_goal_blocked_rate"), 0.0)
    kept_ratio = _safe_float(summary.get("mean_tangential_velocity_kept_ratio"), 1.0)
    invasiveness = _safe_float(summary.get("mean_filter_invasiveness"), 0.0)
    goal_before = _safe_float(summary.get("mean_goal_projection_before_filter"), 0.0)
    goal_after = _safe_float(summary.get("mean_goal_projection_after_filter"), 0.0)
    min_agent_distance = _safe_float(summary.get("min_nearest_agent_distance"), float("inf"))
    formation_error = _safe_float(summary.get("mean_formation_error"), 0.0)
    multi_agent_threshold = _safe_float(os.getenv("GAZEBO_PROGRESS_MULTI_AGENT_BLOCK_DISTANCE"), 3.0)
    formation_threshold = _safe_float(os.getenv("GAZEBO_PROGRESS_FORMATION_CONFLICT_ERROR"), 5.0)

    if boundary_ratio >= _safe_float(os.getenv("GAZEBO_PROGRESS_STALL_BOUNDARY_RATIO"), 0.20):
        classes.append("STALL_ON_BOUNDARY")
    if active_ratio >= 0.05 and invasiveness >= _safe_float(os.getenv("GAZEBO_PROGRESS_OVER_FILTERED_INVASIVENESS"), 0.20) and goal_after < max(0.05, goal_before * 0.5):
        classes.append("OVER_FILTERED")
    if blocked_rate >= _safe_float(os.getenv("GAZEBO_PROGRESS_BLOCKED_RATE"), 0.20) and kept_ratio <= _safe_float(os.getenv("GAZEBO_PROGRESS_LOW_TANGENT_RATIO"), 0.55):
        classes.append("NO_TANGENTIAL_ESCAPE")
    if blocked_rate >= _safe_float(os.getenv("GAZEBO_PROGRESS_GOAL_BEHIND_OBSTACLE_RATE"), 0.35):
        classes.append("GOAL_BEHIND_OBSTACLE")
    if min_agent_distance <= multi_agent_threshold:
        classes.append("MULTI_AGENT_BLOCK")
    if formation_error >= formation_threshold:
        classes.append("FORMATION_CONFLICT")
    refined_labels = summary.get("refined_failure_labels")
    if not isinstance(refined_labels, list):
        refined_labels = refine_progress_failure_labels(records)
    for label in refined_labels:
        if label not in classes:
            classes.append(str(label))

    evidence.update(summary)
    evidence.update(
        {
            "active_ratio": active_ratio,
            "boundary_dwell_ratio": boundary_ratio,
            "line_to_goal_blocked_rate": blocked_rate,
            "tangential_velocity_kept_ratio": kept_ratio,
            "filter_invasiveness": invasiveness,
            "goal_projection_before_filter_mean": goal_before,
            "goal_projection_after_filter_mean": goal_after,
            "min_nearest_agent_distance": None if not np.isfinite(min_agent_distance) else min_agent_distance,
            "formation_error_mean": formation_error,
            "refined_failure_labels": refined_labels,
        }
    )
    return {
        "timeout": True,
        "failure_classes": classes,
        "evidence": _json_safe(evidence),
    }


def _aggregate_progress_failure_payload(episodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    class_counts: Dict[str, int] = {}
    for item in episodes:
        for cls in item.get("failure_classes", []) or []:
            class_counts[str(cls)] = int(class_counts.get(str(cls), 0) + 1)

    def mean_evidence(key: str) -> Optional[float]:
        values = []
        for item in episodes:
            evidence = item.get("evidence", {}) if isinstance(item, dict) else {}
            value = _safe_float(evidence.get(key), float("nan"))
            if np.isfinite(value):
                values.append(value)
        return float(np.mean(values)) if values else None

    return {
        "episode_count": int(len(episodes)),
        "timeout_episode_count": int(sum(1 for item in episodes if bool(item.get("timeout", False)))),
        "failure_class_counts": class_counts,
        "mean_boundary_dwell_ratio": mean_evidence("boundary_dwell_ratio"),
        "mean_goal_projection_after_filter": mean_evidence("goal_projection_after_filter_mean"),
        "mean_tangential_velocity_kept_ratio": mean_evidence("tangential_velocity_kept_ratio"),
        "mean_filter_invasiveness": mean_evidence("filter_invasiveness"),
        "mean_python_clearance_violation_rate": mean_evidence("python_clearance_violation_rate"),
        "mean_geometric_penetration_rate": mean_evidence("geometric_penetration_rate"),
        "mean_arrived_hold_active_rate": mean_evidence("arrived_hold_active_rate"),
        "mean_goal_floor_active_rate": mean_evidence("goal_floor_active_rate"),
        "mean_goal_floor_safety_conflict_rate": mean_evidence("goal_floor_safety_conflict_rate"),
    }


def write_progress_diagnostics_artifacts(
    output_dir: Any,
    episode_idx: int,
    records: Sequence[Dict[str, Any]],
    episode_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not output_dir or not records:
        return {}
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "gazebo_progress_diagnostics.csv"
    rows = progress_diagnostic_rows(records)
    if rows:
        write_header = not csv_path.exists() or csv_path.stat().st_size <= 0
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PROGRESS_DIAGNOSTIC_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    summary = summarize_velocity_filter_records(records)
    episode_summary = classify_progress_failure(records, episode_data=episode_data, summary=summary)
    episode_summary.update(
        {
            "episode": int(episode_idx),
            "summary": _json_safe(summary),
        }
    )
    json_path = out_dir / "progress_failure_summary.json"
    payload: Dict[str, Any]
    if json_path.exists() and json_path.stat().st_size > 0:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    else:
        payload = {}
    episodes = [item for item in payload.get("episodes", []) if int(item.get("episode", -1)) != int(episode_idx)] if isinstance(payload.get("episodes"), list) else []
    episodes.append(_json_safe(episode_summary))
    episodes = sorted(episodes, key=lambda item: int(item.get("episode", -1)))
    payload = {
        "episodes": episodes,
        "aggregate": _aggregate_progress_failure_payload(episodes),
    }
    json_path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "gazebo_progress_diagnostics_csv": str(csv_path),
        "progress_failure_summary_json": str(json_path),
        "episode_progress_summary": _json_safe(episode_summary),
    }


def write_velocity_filter_debug_artifacts(
    output_dir: Any,
    episode_idx: int,
    records: Sequence[Dict[str, Any]],
    first_contact_step: Optional[int] = None,
    hard_contact: bool = False,
    agent_id: Optional[int] = None,
    obstacle_name: Optional[str] = None,
    window_start: Optional[int] = None,
    window_end: Optional[int] = None,
) -> Dict[str, Any]:
    filtered = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        try:
            step = int(record.get("step"))
        except Exception:
            continue
        if window_start is not None and step < int(window_start):
            continue
        if window_end is not None and step > int(window_end):
            continue
        if agent_id is not None and int(record.get("agent_id", -1)) != int(agent_id):
            continue
        if obstacle_name and record.get("nearest_obstacle_id") != obstacle_name:
            continue
        filtered.append(record)
    if not output_dir or not filtered:
        return {}
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_obstacle = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(obstacle_name or "obstacle"))
    safe_agent = "all" if agent_id is None else str(agent_id)
    stem = f"velocity_filter_debug_ep{int(episode_idx):03d}_agent{safe_agent}_{safe_obstacle}"
    payload = {
        "episode": int(episode_idx),
        "agent_id": agent_id,
        "obstacle_name": obstacle_name,
        "first_contact_step": int(first_contact_step) if first_contact_step is not None else None,
        "hard_contact": bool(hard_contact),
        "summary": summarize_velocity_filter_records(filtered),
        "records": filtered,
    }
    paths: Dict[str, Any] = {}
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    paths["json"] = str(json_path)

    rows = []
    for r in filtered:
        nominal = r.get("nominal_cmd_vel") or []
        final = r.get("final_cmd_vel") or []
        pose = r.get("pose") or []
        rows.append(
            {
                "step": r.get("step"),
                "agent_id": r.get("agent_id"),
                "mode": r.get("mode"),
                "filter_active": int(bool(r.get("filter_active", False))),
                "filter_trigger_reason": r.get("filter_trigger_reason"),
                "nearest_obstacle_id": r.get("nearest_obstacle_id"),
                "surface_distance": r.get("surface_distance"),
                "clearance": r.get("clearance"),
                "stopping_distance": r.get("stopping_distance"),
                "inward_velocity_before_filter": r.get("inward_velocity_before_filter"),
                "inward_velocity_after_filter": r.get("inward_velocity_after_filter"),
                "relative_inward_velocity_before_filter": r.get("relative_inward_velocity_before_filter"),
                "relative_inward_velocity_after_filter": r.get("relative_inward_velocity_after_filter"),
                "current_relative_inward_velocity": r.get("current_relative_inward_velocity"),
                "closing_inward_velocity_for_stopping": r.get("closing_inward_velocity_for_stopping"),
                "outward_speed_applied": r.get("outward_speed_applied"),
                "cmd_delta_norm": r.get("cmd_delta_norm"),
                "goal_distance": r.get("goal_distance"),
                "goal_projection_before_filter": r.get("goal_projection_before_filter"),
                "goal_projection_after_filter": r.get("goal_projection_after_filter"),
                "tangential_velocity_before_filter": r.get("tangential_velocity_before_filter"),
                "tangential_velocity_after_filter": r.get("tangential_velocity_after_filter"),
                "tangential_velocity_kept_ratio": r.get("tangential_velocity_kept_ratio"),
                "outward_velocity_added": r.get("outward_velocity_added"),
                "filter_invasiveness": r.get("filter_invasiveness"),
                "boundary_dwell_steps": r.get("boundary_dwell_steps"),
                "line_to_goal_blocked": int(bool(r.get("line_to_goal_blocked", False))),
                "nearest_agent_distance": r.get("nearest_agent_distance"),
                "formation_error": r.get("formation_error"),
                "avoidance_state": r.get("avoidance_state"),
                "halfspace_projection_delta_norm": r.get("halfspace_projection_delta_norm"),
                "tangent_recovery_applied": int(bool(r.get("tangent_recovery_applied", False))),
                "goal_projection_recovery_applied": int(bool(r.get("goal_projection_recovery_applied", False))),
                "nominal_cmd_vx": nominal[0] if len(nominal) > 0 else "",
                "nominal_cmd_vy": nominal[1] if len(nominal) > 1 else "",
                "nominal_cmd_vz": nominal[2] if len(nominal) > 2 else "",
                "final_cmd_vx": final[0] if len(final) > 0 else "",
                "final_cmd_vy": final[1] if len(final) > 1 else "",
                "final_cmd_vz": final[2] if len(final) > 2 else "",
                "pose_x": pose[0] if len(pose) > 0 else "",
                "pose_y": pose[1] if len(pose) > 1 else "",
                "pose_z": pose[2] if len(pose) > 2 else "",
            }
        )
    if rows:
        csv_path = out_dir / f"{stem}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        paths["csv"] = str(csv_path)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = np.asarray([float(r["step"]) for r in rows], dtype=np.float64)
        surface = np.asarray([_safe_float(r["surface_distance"], np.nan) for r in rows], dtype=np.float64)
        inward_before = np.asarray([_safe_float(r["inward_velocity_before_filter"], np.nan) for r in rows], dtype=np.float64)
        inward_after = np.asarray([_safe_float(r["inward_velocity_after_filter"], np.nan) for r in rows], dtype=np.float64)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax = axes[0]
        ax.plot(steps, surface, color="tab:red", label="surface_distance")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        if first_contact_step is not None:
            ax.axvline(float(first_contact_step), color="tab:orange", linewidth=1.2, label="first_contact")
        ax2 = ax.twinx()
        ax2.plot(steps, inward_before, color="tab:blue", alpha=0.8, label="inward_before")
        ax2.plot(steps, inward_after, color="tab:green", alpha=0.8, label="inward_after")
        ax.set_xlabel("step")
        ax.set_ylabel("surface distance")
        ax2.set_ylabel("inward velocity")
        ax.grid(True, alpha=0.3)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")

        ax = axes[1]
        poses = np.asarray(
            [
                [r["pose_x"], r["pose_y"]]
                for r in rows
                if r["pose_x"] not in (None, "") and r["pose_y"] not in (None, "")
            ],
            dtype=np.float64,
        )
        if poses.size:
            ax.plot(poses[:, 0], poses[:, 1], color="black", linewidth=1.2, label="gazebo pose")
            ax.scatter([poses[0, 0]], [poses[0, 1]], color="green", s=25, label="window start")
            ax.scatter([poses[-1, 0]], [poses[-1, 1]], color="red", s=25, label="window end")
        nearest = None
        for r in filtered:
            item = r.get("nearest_obstacle")
            if isinstance(item, dict):
                nearest = item
                break
        if isinstance(nearest, dict) and nearest.get("center") is not None:
            center = np.asarray(nearest.get("center"), dtype=np.float64).reshape(-1)
            radius = float(nearest.get("radius", 0.0) or 0.0)
            if center.size >= 2:
                ax.add_patch(plt.Circle((center[0], center[1]), radius, fill=False, color="tab:red", linewidth=1.2))
                ax.scatter([center[0]], [center[1]], color="tab:red", s=20, label=nearest.get("name", "obstacle"))
        stride = max(1, len(rows) // 12)
        for r in rows[::stride]:
            try:
                x = float(r["pose_x"])
                y = float(r["pose_y"])
                nvx = float(r["nominal_cmd_vx"])
                nvy = float(r["nominal_cmd_vy"])
                fvx = float(r["final_cmd_vx"])
                fvy = float(r["final_cmd_vy"])
                ax.arrow(x, y, nvx * 1.2, nvy * 1.2, color="tab:blue", width=0.04, alpha=0.55)
                ax.arrow(x, y, fvx * 1.2, fvy * 1.2, color="tab:green", width=0.04, alpha=0.65)
            except Exception:
                continue
        ax.set_title("blue=nominal cmd, green=filtered cmd")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.suptitle(
            f"Velocity safety filter ep={episode_idx}, agent={agent_id}, obstacle={obstacle_name}, hard_contact={hard_contact}"
        )
        fig.tight_layout()
        plot_path = out_dir / f"{stem}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        paths["plot"] = str(plot_path)
    except Exception:
        pass
    return _json_safe(paths)
