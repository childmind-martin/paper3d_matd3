#!/usr/bin/env python3
"""Python client for streaming live MATD3 agent poses into Gazebo."""

from __future__ import annotations

import os
import re
import json
import select
import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _identity_or_quat(agent) -> np.ndarray:
    state = getattr(agent, "state", None)
    quat = getattr(state, "orientation", None)
    try:
        arr = np.asarray(quat, dtype=np.float64).reshape(-1)
        if arr.size >= 4 and np.all(np.isfinite(arr[:4])):
            norm = float(np.linalg.norm(arr[:4]))
            if norm > 1e-9:
                return arr[:4] / norm
    except Exception:
        pass
    return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _position(agent) -> np.ndarray:
    state = getattr(agent, "state", None)
    pos = getattr(state, "p_pos", None)
    arr = np.asarray(pos, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        raise ValueError(f"agent has invalid position: {getattr(agent, 'name', '<unknown>')}")
    return arr[:3]


def _velocity(agent) -> np.ndarray:
    state = getattr(agent, "state", None)
    vel = getattr(state, "p_vel", None)
    arr = np.asarray(vel, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        raise ValueError(f"agent has invalid velocity: {getattr(agent, 'name', '<unknown>')}")
    return arr[:3]


def _default_bridge_output_dir() -> Path:
    raw = os.getenv("GAZEBO_LIVE_BRIDGE_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parent / "gazebo_live_pose_bridge_runtime"


def ensure_bridge_executable(output_dir: Optional[Path] = None, compile_bridge: bool = True) -> Path:
    output = Path(output_dir).resolve() if output_dir else _default_bridge_output_dir()
    exe = output / "pose_stream_bridge_build" / "pose_stream_bridge"
    force_rebuild = _env_flag("GAZEBO_LIVE_REBUILD_BRIDGE", False)
    if exe.exists() and not force_rebuild:
        try:
            from gazebo_live_pose_bridge_builder import CPP_SOURCE
            src = output / "pose_stream_bridge.cpp"
            expected = CPP_SOURCE.strip() + "\n"
            if src.exists() and src.read_text(encoding="utf-8") == expected:
                return exe
        except Exception:
            return exe
    if not compile_bridge:
        raise FileNotFoundError(f"Gazebo live pose bridge executable not found: {exe}")
    from gazebo_live_pose_bridge_builder import create_live_pose_bridge_project

    meta = create_live_pose_bridge_project(output, compile_bridge=True)
    exe_path = Path(meta.get("executable") or exe).resolve()
    if not exe_path.exists():
        raise FileNotFoundError(f"Gazebo live pose bridge build did not create executable: {exe_path}")
    return exe_path


class GazeboLivePoseClient:
    def __init__(
        self,
        world: str,
        agent_count: int,
        agent_prefix: str = "dynamic_agent_",
        timeout_ms: int = 1000,
        bridge_executable: Optional[Path] = None,
        bridge_output_dir: Optional[Path] = None,
        autobuild: bool = True,
        source_setup: bool = True,
        contact_feedback: bool = True,
        contact_flag_file: Optional[Path] = None,
        step_after_twist: int = 0,
        pre_step_sleep_ms: int = 0,
        post_step_sleep_ms: int = 0,
        wall_time_step_ms: int = 0,
        pause_for_step: bool = False,
        wait_ack: bool = False,
        ack_timeout: float = 2.0,
        state_feedback: bool = False,
        state_file: Optional[Path] = None,
        state_feedback_dt: float = 0.08,
    ) -> None:
        self.world = str(world)
        self.agent_count = int(agent_count)
        self.agent_prefix = str(agent_prefix)
        self.timeout_ms = int(timeout_ms)
        self.bridge_executable = Path(bridge_executable).resolve() if bridge_executable else None
        self.bridge_output_dir = Path(bridge_output_dir).resolve() if bridge_output_dir else None
        self.autobuild = bool(autobuild)
        self.source_setup = bool(source_setup)
        self.contact_feedback = bool(contact_feedback)
        self.step_after_twist = max(0, int(step_after_twist))
        self.pre_step_sleep_ms = max(0, int(pre_step_sleep_ms))
        self.post_step_sleep_ms = max(0, int(post_step_sleep_ms))
        self.wall_time_step_ms = max(0, int(wall_time_step_ms))
        self.pause_for_step = bool(pause_for_step)
        self.wait_ack = bool(wait_ack)
        self.ack_timeout = max(0.001, float(ack_timeout))
        self.state_feedback = bool(state_feedback)
        self.state_file = (
            Path(state_file).resolve()
            if state_file
            else Path(os.getenv("GAZEBO_LIVE_STATE_FILE", "/tmp/matd3_gazebo_live_state.json")).resolve()
        )
        self.state_feedback_dt = max(1e-9, float(state_feedback_dt))
        self.contact_flag_file = (
            Path(contact_flag_file).resolve()
            if contact_flag_file
            else Path(os.getenv("GAZEBO_LIVE_CONTACT_FLAG_FILE", "/tmp/matd3_gazebo_live_contact.flag")).resolve()
        )
        self.proc: Optional[subprocess.Popen] = None
        self.sent_frames = 0
        self.sent_twist_frames = 0
        self.ack_count = 0
        self.ack_timeout_count = 0
        self.last_ack_line: Optional[str] = None
        self.last_ack_state_frame: Optional[int] = None
        self._pending_state_min_frame: Optional[int] = None
        self._last_state_frame: Optional[int] = None
        self._last_state_positions: Optional[np.ndarray] = None
        self._last_state_velocities: Optional[np.ndarray] = None
        mode = os.getenv("GAZEBO_LIVE_FEEDBACK_VELOCITY_MODE", "clamp").strip().lower()
        if mode not in ("preserve", "estimate", "clamp"):
            mode = "clamp"
        self.state_velocity_mode = mode
        acc_mode = os.getenv("GAZEBO_LIVE_FEEDBACK_ACCELERATION_MODE", "estimate").strip().lower()
        if acc_mode not in ("preserve", "estimate", "zero"):
            acc_mode = "estimate"
        self.state_acceleration_mode = acc_mode
        self.state_feedback_max_speed_scale = max(0.01, _env_float("GAZEBO_LIVE_FEEDBACK_MAX_SPEED_SCALE", 1.0))
        self.state_feedback_max_accel_scale = max(0.0, _env_float("GAZEBO_LIVE_FEEDBACK_MAX_ACCEL_SCALE", 0.0))
        self.max_pose_jump = max(0.0, _env_float("GAZEBO_LIVE_MAX_POSE_JUMP", 100.0))
        self.pose_jump_reject_count = 0
        self.max_pose_jump_observed = 0.0
        self.max_feedback_speed_observed = 0.0
        self.max_feedback_accel_observed = 0.0

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def expected_contact_topics(self) -> list[str]:
        return [
            f"/world/{self.world}/model/{self.agent_prefix}{agent_idx}"
            "/link/uav_marker_link/sensor/python_collision_contact/contact"
            for agent_idx in range(self.agent_count)
        ]

    def contact_feedback_armed(self) -> bool:
        return bool(self.contact_feedback and self.is_running())

    def start(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return
        exe = self.bridge_executable or ensure_bridge_executable(
            output_dir=self.bridge_output_dir,
            compile_bridge=self.autobuild,
        )
        cmd = [
            str(exe),
            "--world",
            self.world,
            "--agent-count",
            str(self.agent_count),
            "--agent-prefix",
            self.agent_prefix,
            "--timeout-ms",
            str(self.timeout_ms),
        ]
        if self.step_after_twist > 0:
            cmd.extend(["--step-after-twist", str(self.step_after_twist)])
        if self.pre_step_sleep_ms > 0:
            cmd.extend(["--pre-step-sleep-ms", str(self.pre_step_sleep_ms)])
        if self.post_step_sleep_ms > 0:
            cmd.extend(["--post-step-sleep-ms", str(self.post_step_sleep_ms)])
        if self.wall_time_step_ms > 0:
            cmd.extend(["--wall-time-step-ms", str(self.wall_time_step_ms)])
        if self.pause_for_step:
            cmd.append("--pause-for-step")
        if self.wait_ack:
            cmd.append("--ack")
        if self.state_feedback:
            try:
                if self.state_file.exists():
                    self.state_file.unlink()
            except Exception:
                pass
            cmd.extend(["--state-file", str(self.state_file)])
        if self.contact_feedback:
            try:
                if self.contact_flag_file.exists():
                    self.contact_flag_file.unlink()
            except Exception:
                pass
            cmd.extend(["--contact-flag-file", str(self.contact_flag_file)])
        else:
            cmd.append("--no-contact-watch")
        if self.source_setup:
            shell_cmd = (
                "source /opt/ros/jazzy/setup.bash; exec "
                + shlex.join(cmd)
            )
            popen_cmd = ["bash", "-lc", shell_cmd]
        else:
            popen_cmd = cmd
        self.proc = subprocess.Popen(
            popen_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE if self.wait_ack else subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def contact_detected(self) -> bool:
        try:
            return self.contact_flag_file.exists() and self.contact_flag_file.stat().st_size > 0
        except Exception:
            return False

    def clear_contact_flag(self) -> None:
        try:
            if self.contact_flag_file.exists():
                self.contact_flag_file.unlink()
        except Exception:
            pass

    def contact_agent_indices(self) -> list[int]:
        if not self.contact_detected():
            return []
        try:
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
            if 0 <= idx < self.agent_count:
                indices.add(idx)
        return sorted(indices)

    def contact_pairs(self) -> list[dict]:
        if not self.contact_detected():
            return []
        try:
            lines = self.contact_flag_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return []
        pairs = []
        current = {}
        for line in lines:
            if line.startswith("topic "):
                if current:
                    pairs.append(current)
                current = {"topic": line[len("topic "):].strip()}
            elif line.startswith("contact "):
                if current:
                    pairs.append(current)
                current = {}
                parts = line.split(" ", 3)
                if len(parts) >= 4:
                    current = {
                        "agent": parts[1],
                        "collision1": parts[2],
                        "collision2": parts[3],
                    }
            elif line.startswith("contacts ") and current:
                try:
                    current["contacts"] = int(line[len("contacts "):].strip())
                except Exception:
                    current["contacts"] = line[len("contacts "):].strip()
        if current:
            pairs.append(current)
        return pairs

    def _wait_for_ack(self, expected_mode: Optional[str] = None) -> None:
        if not self.wait_ack:
            return
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("Gazebo live pose bridge is not running")
        if self.proc.stdout is None:
            raise RuntimeError("Gazebo live pose bridge ack stream is not available")
        ready, _, _ = select.select([self.proc.stdout], [], [], self.ack_timeout)
        if not ready:
            self.ack_timeout_count += 1
            raise TimeoutError(f"Gazebo live pose bridge ack timed out after {self.ack_timeout:.3f}s")
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("Gazebo live pose bridge closed before ack")
        line = line.strip()
        self.last_ack_line = line
        parts = line.split()
        if not parts or parts[0] != "ok":
            raise RuntimeError(f"unexpected Gazebo live pose bridge ack: {line}")
        if expected_mode is not None and expected_mode not in parts:
            raise RuntimeError(f"unexpected Gazebo live pose bridge ack mode for {expected_mode}: {line}")
        match = re.search(r"\bstate_frame=(\d+)\b", line)
        if match:
            self.last_ack_state_frame = int(match.group(1))
            self._pending_state_min_frame = int(match.group(1))
        self.ack_count += 1

    def send_arrays(self, positions: Sequence[Sequence[float]], orientations_wxyz: Sequence[Sequence[float]]) -> None:
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("Gazebo live pose bridge is not running")
        values = []
        for agent_idx in range(self.agent_count):
            pos = np.asarray(positions[agent_idx], dtype=np.float64).reshape(-1)[:3]
            quat = np.asarray(orientations_wxyz[agent_idx], dtype=np.float64).reshape(-1)[:4]
            if pos.size < 3 or quat.size < 4:
                raise ValueError(f"invalid live pose for agent {agent_idx}")
            values.extend([pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]])
        line = " ".join(f"{float(v):.9g}" for v in values) + "\n"
        assert self.proc.stdin is not None
        self.proc.stdin.write(line)
        self.proc.stdin.flush()
        self.sent_frames += 1
        self._wait_for_ack("pose")

    def send_twist_arrays(
        self,
        linear_velocities: Sequence[Sequence[float]],
        angular_velocities: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("Gazebo live pose bridge is not running")
        if angular_velocities is None:
            angular_velocities = np.zeros((self.agent_count, 3), dtype=np.float64)
        values = []
        for agent_idx in range(self.agent_count):
            linear = np.asarray(linear_velocities[agent_idx], dtype=np.float64).reshape(-1)[:3]
            angular = np.asarray(angular_velocities[agent_idx], dtype=np.float64).reshape(-1)[:3]
            if linear.size < 3 or angular.size < 3:
                raise ValueError(f"invalid live twist for agent {agent_idx}")
            values.extend([linear[0], linear[1], linear[2], angular[0], angular[1], angular[2]])
        line = "twist " + " ".join(f"{float(v):.9g}" for v in values) + "\n"
        assert self.proc.stdin is not None
        self.proc.stdin.write(line)
        self.proc.stdin.flush()
        self.sent_twist_frames += 1
        self._wait_for_ack("twist")

    def send_agents(self, agents: Sequence[object]) -> None:
        positions = [_position(agent) for agent in list(agents)[: self.agent_count]]
        orientations = [_identity_or_quat(agent) for agent in list(agents)[: self.agent_count]]
        if len(positions) != self.agent_count:
            raise ValueError(f"expected {self.agent_count} agents, got {len(positions)}")
        self.send_arrays(positions, orientations)

    def send_velocity_agents(self, agents: Sequence[object]) -> None:
        velocities = []
        for agent in list(agents)[: self.agent_count]:
            vel = _velocity(agent).astype(np.float64, copy=True)
            try:
                max_speed = float(getattr(agent, "max_speed", 0.0) or 0.0)
            except Exception:
                max_speed = 0.0
            if max_speed > 0.0:
                speed = float(np.linalg.norm(vel))
                if np.isfinite(speed) and speed > max_speed:
                    vel = vel / max(speed, 1e-9) * max_speed
            velocities.append(vel)
        if len(velocities) != self.agent_count:
            raise ValueError(f"expected {self.agent_count} agents, got {len(velocities)}")
        self.send_twist_arrays(velocities)

    def read_state(self, min_frame: Optional[int] = None, timeout: float = 0.0) -> Optional[dict]:
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

    def apply_state_to_agents(self, agents: Sequence[object], timeout: float = 0.0) -> bool:
        if not self.state_feedback:
            return False
        min_frame = (self._last_state_frame + 1) if self._last_state_frame is not None else None
        if self._pending_state_min_frame is not None:
            min_frame = max(min_frame if min_frame is not None else -1, int(self._pending_state_min_frame))
        data = self.read_state(min_frame=min_frame, timeout=timeout)
        if not isinstance(data, dict):
            return False
        frame = int(data.get("frame", -1))
        agent_entries = data.get("agents", [])
        if not isinstance(agent_entries, list) or len(agent_entries) < self.agent_count:
            return False
        positions = []
        for agent_idx in range(self.agent_count):
            entry = agent_entries[agent_idx]
            if not isinstance(entry, dict) or not bool(entry.get("seen", False)):
                return False
            pos = np.asarray(entry.get("position", []), dtype=np.float64).reshape(-1)
            if pos.size < 3 or not np.all(np.isfinite(pos[:3])):
                return False
            positions.append(pos[:3].copy())
        positions_arr = np.asarray(positions, dtype=np.float64)
        velocities = None
        accelerations = None
        reference_positions = self._last_state_positions
        if reference_positions is not None and reference_positions.shape == positions_arr.shape:
            deltas = positions_arr - reference_positions
            jump_norms = np.linalg.norm(deltas, axis=1)
            finite_jumps = jump_norms[np.isfinite(jump_norms)]
            if finite_jumps.size:
                self.max_pose_jump_observed = max(self.max_pose_jump_observed, float(np.max(finite_jumps)))
            if self.max_pose_jump > 0.0 and np.any(jump_norms > self.max_pose_jump):
                self.pose_jump_reject_count += 1
                return False
            velocities = deltas / self.state_feedback_dt
            if (
                self.state_velocity_mode != "preserve"
                and self._last_state_velocities is not None
                and self._last_state_velocities.shape == velocities.shape
            ):
                accelerations = (velocities - self._last_state_velocities) / self.state_feedback_dt
        applied_velocities = None
        for agent_idx, agent in enumerate(list(agents)[: self.agent_count]):
            state = getattr(agent, "state", None)
            if state is None:
                continue
            state.p_pos = positions_arr[agent_idx].astype(np.float64, copy=True)
            if velocities is not None and self.state_velocity_mode != "preserve":
                vel = velocities[agent_idx].astype(np.float64, copy=True)
                speed = float(np.linalg.norm(vel))
                if np.isfinite(speed):
                    self.max_feedback_speed_observed = max(self.max_feedback_speed_observed, speed)
                if self.state_velocity_mode == "clamp":
                    try:
                        limit = float(getattr(agent, "max_speed", 0.0) or 0.0) * self.state_feedback_max_speed_scale
                    except Exception:
                        limit = 0.0
                    if limit > 0.0 and np.isfinite(speed) and speed > limit:
                        vel = vel / max(speed, 1e-9) * limit
                state.p_vel = vel
                if applied_velocities is None:
                    applied_velocities = np.zeros_like(velocities)
                applied_velocities[agent_idx] = vel
            if self.state_acceleration_mode == "zero":
                state.p_acc = np.zeros(3, dtype=np.float64)
            elif self.state_acceleration_mode != "preserve":
                acc = None
                if (
                    velocities is not None
                    and self.state_velocity_mode != "preserve"
                    and self._last_state_velocities is not None
                    and self._last_state_velocities.shape == velocities.shape
                ):
                    current_vel = np.asarray(getattr(state, "p_vel", velocities[agent_idx]), dtype=np.float64).reshape(-1)
                    if current_vel.size >= 3 and np.all(np.isfinite(current_vel[:3])):
                        acc = (current_vel[:3] - self._last_state_velocities[agent_idx]) / self.state_feedback_dt
                elif accelerations is not None:
                    acc = accelerations[agent_idx].astype(np.float64, copy=True)
                if acc is not None:
                    acc_norm = float(np.linalg.norm(acc))
                    if np.isfinite(acc_norm):
                        self.max_feedback_accel_observed = max(self.max_feedback_accel_observed, acc_norm)
                    if self.state_feedback_max_accel_scale > 0.0:
                        try:
                            limit = float(getattr(agent, "accel", 0.0) or 0.0) * self.state_feedback_max_accel_scale
                        except Exception:
                            limit = 0.0
                        if limit > 0.0 and np.isfinite(acc_norm) and acc_norm > limit:
                            acc = acc / max(acc_norm, 1e-9) * limit
                    state.p_acc = acc
            entry = agent_entries[agent_idx]
            quat = np.asarray(entry.get("orientation_wxyz", []), dtype=np.float64).reshape(-1)
            if quat.size >= 4 and np.all(np.isfinite(quat[:4])):
                norm = float(np.linalg.norm(quat[:4]))
                if norm > 1e-9:
                    state.orientation = (quat[:4] / norm).astype(np.float64, copy=True)
        self._last_state_frame = frame
        self._last_state_positions = positions_arr
        if applied_velocities is not None:
            self._last_state_velocities = applied_velocities.copy()
        elif velocities is not None:
            self._last_state_velocities = velocities.copy()
        else:
            try:
                self._last_state_velocities = np.asarray(
                    [_velocity(agent) for agent in list(agents)[: self.agent_count]],
                    dtype=np.float64,
                )
            except Exception:
                self._last_state_velocities = None
        if self._pending_state_min_frame is not None and frame >= int(self._pending_state_min_frame):
            self._pending_state_min_frame = None
        return True

    def close(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write("quit\n")
                proc.stdin.flush()
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def make_live_pose_client_from_env(agent_count: int) -> Optional[GazeboLivePoseClient]:
    if not _env_flag("GAZEBO_LIVE_SYNC", False):
        return None
    exe_raw = os.getenv("GAZEBO_LIVE_POSE_BRIDGE", "").strip()
    bridge_exe = Path(exe_raw).expanduser().resolve() if exe_raw else None
    return GazeboLivePoseClient(
        world=os.getenv("GAZEBO_LIVE_WORLD", "matd3_static_scene"),
        agent_count=int(agent_count),
        agent_prefix=os.getenv("GAZEBO_LIVE_AGENT_PREFIX", "dynamic_agent_"),
        timeout_ms=_env_int("GAZEBO_LIVE_TIMEOUT_MS", 1000),
        bridge_executable=bridge_exe,
        bridge_output_dir=_default_bridge_output_dir(),
        autobuild=_env_flag("GAZEBO_LIVE_AUTOBUILD", True),
        source_setup=_env_flag("GAZEBO_LIVE_SOURCE_SETUP", True),
        contact_feedback=_env_flag("GAZEBO_LIVE_CONTACT_FEEDBACK", True),
        contact_flag_file=Path(os.getenv("GAZEBO_LIVE_CONTACT_FLAG_FILE", "/tmp/matd3_gazebo_live_contact.flag")),
        step_after_twist=_env_int("GAZEBO_LIVE_STEP_ITERATIONS", 0),
        pre_step_sleep_ms=_env_int("GAZEBO_LIVE_PRE_STEP_SLEEP_MS", 0),
        post_step_sleep_ms=_env_int("GAZEBO_LIVE_POST_STEP_SLEEP_MS", 0),
        wall_time_step_ms=_env_int("GAZEBO_LIVE_WALL_TIME_STEP_MS", 0),
        pause_for_step=_env_flag("GAZEBO_LIVE_PAUSE_FOR_STEP", False),
        wait_ack=_env_flag("GAZEBO_LIVE_WAIT_ACK", False),
        ack_timeout=_env_float("GAZEBO_LIVE_ACK_TIMEOUT", 2.0),
        state_feedback=_env_flag("GAZEBO_LIVE_STATE_FEEDBACK", False),
        state_file=Path(os.getenv("GAZEBO_LIVE_STATE_FILE", "/tmp/matd3_gazebo_live_state.json")),
        state_feedback_dt=_env_float("GAZEBO_LIVE_STATE_FEEDBACK_DT", _env_float("SIMULATION_DT", 0.08)),
    )
