#!/usr/bin/env python3
"""Export Gazebo pose-driven dynamic replay assets from MATD3 state snapshots."""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scenario_exporter import _json_safe
except Exception:  # pragma: no cover
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

from gazebo_terrain_exporter import (
    UAV_MARKER_BASE_RADIUS,
    _add_material,
    _box_visual,
    _cylinder_visual,
    _fmt,
    _scaled_pose,
    _scaled_size,
    _sphere_visual,
    _vec,
    _write_xml,
    write_uav_marker_config,
)


DYNAMIC_MODEL_PREFIX = "matd3_dynamic_uav_agent"
DYNAMIC_INCLUDE_PREFIX = "dynamic_agent"


def _normalize_collision_mode(value: Optional[str]) -> str:
    raw = str(value or "").strip().lower()
    if raw in (
        "nonblocking",
        "non_blocking",
        "visual_only",
        "visual-only",
        "soft",
        "python_soft",
        "python-soft",
        "transfer_equivalence",
        "none",
        "disabled",
        "off",
        "0",
        "false",
        "no",
    ):
        return "nonblocking"
    return "hard"


def _sanitize_identifier(value: str, fallback: str) -> str:
    text = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value))
    text = "_".join(part for part in text.split("_") if part)
    if not text:
        text = fallback
    if text[0].isdigit():
        text = f"{fallback}_{text}"
    return text


def _live_namespace(output_dir: Path) -> str:
    raw = os.getenv("GAZEBO_LIVE_NAMESPACE", "").strip()
    if raw:
        return _sanitize_identifier(raw, "matd3_live")
    token = f"{Path(output_dir).name}_{os.getpid()}"
    return _sanitize_identifier(token, "matd3_live")


def _resolve_visual_scale(agent_size: float, uav_marker_scale: Optional[float], default_multiplier: float) -> float:
    if uav_marker_scale is not None:
        return max(0.005, float(uav_marker_scale))
    base_scale = max(0.001, float(agent_size)) / UAV_MARKER_BASE_RADIUS
    raw_multiplier = os.getenv("GAZEBO_UAV_VISUAL_SCALE_MULTIPLIER", "").strip()
    if not raw_multiplier:
        raw_multiplier = os.getenv("GAZEBO_LIVE_UAV_VISUAL_SCALE_MULTIPLIER", str(default_multiplier)).strip()
    try:
        multiplier = float(raw_multiplier)
    except Exception:
        multiplier = float(default_multiplier)
    return max(0.005, base_scale * max(0.01, multiplier))


def _resolve_collision_envelope_radius(agent_size: float, collision_threshold: float) -> Tuple[float, str]:
    """Resolve the physical Gazebo collision radius for the UAV body envelope."""
    override = os.getenv("GAZEBO_LIVE_COLLISION_RADIUS", os.getenv("GAZEBO_DYNAMIC_COLLISION_RADIUS", "")).strip()
    if override:
        try:
            return max(0.001, float(override)), "env_override"
        except Exception:
            pass

    mode = os.getenv(
        "GAZEBO_LIVE_COLLISION_RADIUS_MODE",
        os.getenv("GAZEBO_DYNAMIC_COLLISION_RADIUS_MODE", "agent_size"),
    ).strip().lower()
    if mode in ("threshold", "collision_threshold", "python_threshold"):
        return max(0.001, float(collision_threshold)), "collision_distance_threshold"
    if mode in ("safety", "safety_envelope", "agent_plus_threshold", "python_collision_envelope"):
        return max(0.001, float(agent_size) + max(0.0, float(collision_threshold))), "agent_size_plus_collision_distance_threshold"
    return max(0.001, float(agent_size)), "agent_size"


def _load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _quat_wxyz_to_rpy(quat: Sequence[float]) -> Tuple[float, float, float]:
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    if q.size < 4 or not np.all(np.isfinite(q[:4])):
        return 0.0, 0.0, 0.0
    w, x, y, z = q[:4]
    norm = float(np.linalg.norm(q[:4]))
    if norm <= 1e-9 or not np.isfinite(norm):
        return 0.0, 0.0, 0.0
    w, x, y, z = (q[:4] / norm).tolist()
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return float(roll), float(pitch), float(yaw)


def write_dynamic_uav_sdf(
    model_dir: Path,
    model_name: str,
    accent_rgba: Sequence[float],
    visual_scale: float,
    collision_envelope_radius: float = 0.0,
    enable_velocity_control: bool = True,
    enable_physical_collision: bool = True,
    enable_contact_sensor: bool = True,
    show_collision_envelope: bool = True,
) -> None:
    root = ET.Element("sdf", {"version": "1.10"})
    model = ET.SubElement(root, "model", {"name": model_name})
    ET.SubElement(model, "static").text = "false"
    ET.SubElement(model, "self_collide").text = "false"
    link = ET.SubElement(model, "link", {"name": "uav_marker_link"})
    ET.SubElement(link, "gravity").text = "false"
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass").text = "1.0"
    inertia = ET.SubElement(inertial, "inertia")
    ET.SubElement(inertia, "ixx").text = "0.02"
    ET.SubElement(inertia, "iyy").text = "0.02"
    ET.SubElement(inertia, "izz").text = "0.04"
    ET.SubElement(inertia, "ixy").text = "0"
    ET.SubElement(inertia, "ixz").text = "0"
    ET.SubElement(inertia, "iyz").text = "0"

    visual_scale = max(0.005, float(visual_scale))
    body = (0.08, 0.09, 0.1, 1.0)
    arm = (0.06, 0.06, 0.065, 1.0)
    prop = (0.02, 0.02, 0.025, 0.65)
    envelope = (0.1, 0.55, 1.0, 0.18)
    white = (0.92, 0.92, 0.86, 1.0)
    red = (0.9, 0.05, 0.03, 1.0)
    green = (0.03, 0.65, 0.12, 1.0)
    accent = tuple(float(v) for v in accent_rgba)

    if collision_envelope_radius > 0.0:
        if enable_physical_collision:
            collision = ET.SubElement(link, "collision", {"name": "python_collision_envelope_collision"})
            ET.SubElement(collision, "pose").text = "0 0 0 0 0 0"
            collision_geom = ET.SubElement(collision, "geometry")
            collision_sphere = ET.SubElement(collision_geom, "sphere")
            ET.SubElement(collision_sphere, "radius").text = _fmt(float(collision_envelope_radius))
            if enable_contact_sensor:
                sensor = ET.SubElement(link, "sensor", {"name": "python_collision_contact", "type": "contact"})
                ET.SubElement(sensor, "always_on").text = "true"
                ET.SubElement(sensor, "update_rate").text = "250"
                contact = ET.SubElement(sensor, "contact")
                ET.SubElement(contact, "collision").text = "python_collision_envelope_collision"
        if show_collision_envelope:
            _sphere_visual(link, "python_collision_envelope", (0, 0, 0, 0, 0, 0), float(collision_envelope_radius), envelope)

    _box_visual(link, "body", _scaled_pose((0, 0, 0, 0, 0, 0), visual_scale), _scaled_size((0.95, 0.52, 0.24), visual_scale), body)
    _box_visual(link, "nose", _scaled_pose((0.58, 0, 0.03, 0, 0, 0), visual_scale), _scaled_size((0.42, 0.26, 0.16), visual_scale), accent)
    _box_visual(link, "x_arm", _scaled_pose((0, 0, 0.08, 0, 0, 0), visual_scale), _scaled_size((3.4, 0.12, 0.08), visual_scale), arm)
    _box_visual(link, "y_arm", _scaled_pose((0, 0, 0.08, 0, 0, 0), visual_scale), _scaled_size((0.12, 3.4, 0.08), visual_scale), arm)
    for idx, (x, y) in enumerate(((1.7, 1.7), (-1.7, 1.7), (-1.7, -1.7), (1.7, -1.7))):
        _box_visual(link, f"motor_{idx}", _scaled_pose((x, y, 0.08, 0, 0, 0), visual_scale), _scaled_size((0.25, 0.25, 0.18), visual_scale), body)
        _cylinder_visual(link, f"propeller_disc_{idx}", _scaled_pose((x, y, 0.24, 0, 0, 0), visual_scale), 0.52 * visual_scale, 0.035 * visual_scale, prop)
    _cylinder_visual(link, "front_light", _scaled_pose((0.85, 0.0, 0.19, 0, 1.5707963, 0), visual_scale), 0.07 * visual_scale, 0.04 * visual_scale, white)
    _cylinder_visual(link, "left_nav_light", _scaled_pose((0.0, 0.35, 0.17, 0, 1.5707963, 0), visual_scale), 0.055 * visual_scale, 0.035 * visual_scale, red)
    _cylinder_visual(link, "right_nav_light", _scaled_pose((0.0, -0.35, 0.17, 0, 1.5707963, 0), visual_scale), 0.055 * visual_scale, 0.035 * visual_scale, green)
    if enable_velocity_control:
        ET.SubElement(
            model,
            "plugin",
            {
                "filename": "gz-sim-velocity-control-system",
                "name": "gz::sim::systems::VelocityControl",
            },
        )
    _write_xml(root, model_dir / "model.sdf")


def write_dynamic_uav_model(
    model_parent_dir: Path,
    model_name: str,
    accent_rgba: Sequence[float],
    visual_scale: float,
    collision_envelope_radius: float,
    enable_velocity_control: bool = True,
    enable_physical_collision: bool = True,
    enable_contact_sensor: bool = True,
    show_collision_envelope: bool = True,
) -> Dict[str, Any]:
    model_dir = Path(model_parent_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    write_uav_marker_config(model_dir, model_name)
    write_dynamic_uav_sdf(
        model_dir=model_dir,
        model_name=model_name,
        accent_rgba=accent_rgba,
        visual_scale=visual_scale,
        collision_envelope_radius=collision_envelope_radius,
        enable_velocity_control=enable_velocity_control,
        enable_physical_collision=enable_physical_collision,
        enable_contact_sensor=enable_contact_sensor,
        show_collision_envelope=show_collision_envelope,
    )
    physical_radius = float(collision_envelope_radius) if enable_physical_collision else 0.0
    return {
        "model_name": model_name,
        "model_dir": str(model_dir),
        "uri": f"model://{model_name}",
        "visual_scale": float(visual_scale),
        "collision_envelope_radius": physical_radius,
        "collision_envelope_visual_radius": float(collision_envelope_radius) if show_collision_envelope else 0.0,
        "physical_collision": bool(enable_physical_collision),
        "contact_sensor": bool(enable_physical_collision and enable_contact_sensor),
        "velocity_control": bool(enable_velocity_control),
        "cmd_vel_topic": f"/model/{model_name}/cmd_vel",
    }


def _world_name(root: ET.Element) -> str:
    world = root.find("world")
    if world is None:
        raise ValueError("world.sdf has no <world>")
    return world.attrib.get("name", "matd3_static_scene")


def _ensure_world_plugin(world: ET.Element, name: str, filename: str) -> ET.Element:
    for plugin in world.findall("plugin"):
        if plugin.attrib.get("name") == name or plugin.attrib.get("filename") == filename:
            return plugin
    return ET.SubElement(world, "plugin", {"name": name, "filename": filename})


def _ensure_live_test_world_plugins(world: ET.Element) -> None:
    _ensure_world_plugin(world, "gz::sim::systems::Physics", "gz-sim-physics-system")
    _ensure_world_plugin(world, "gz::sim::systems::UserCommands", "gz-sim-user-commands-system")
    _ensure_world_plugin(world, "gz::sim::systems::SceneBroadcaster", "gz-sim-scene-broadcaster-system")
    _ensure_world_plugin(world, "gz::sim::systems::Contact", "gz-sim-contact-system")
    sensors = _ensure_world_plugin(world, "gz::sim::systems::Sensors", "gz-sim-sensors-system")
    if sensors.find("render_engine") is None:
        ET.SubElement(sensors, "render_engine").text = "ogre2"


def write_dynamic_world_sdf(
    base_world_sdf: Path,
    output_path: Path,
    positions0: np.ndarray,
    orientations0: np.ndarray,
    model_uris: Sequence[str],
    remove_static_start_markers: bool = False,
    world_name_override: Optional[str] = None,
    include_name_prefix: Optional[str] = None,
) -> str:
    root = ET.parse(base_world_sdf).getroot()
    world = root.find("world")
    if world is None:
        raise ValueError(f"world.sdf has no <world>: {base_world_sdf}")
    if world_name_override:
        world.attrib["name"] = str(world_name_override)
    world_name = world.attrib.get("name", "matd3_static_scene")
    include_prefix = str(include_name_prefix) if include_name_prefix else f"{DYNAMIC_INCLUDE_PREFIX}_"
    _ensure_live_test_world_plugins(world)
    for include in list(world.findall("include")):
        name = include.findtext("name") or ""
        if name.startswith(f"{DYNAMIC_INCLUDE_PREFIX}_") or name.startswith(include_prefix):
            world.remove(include)
        elif remove_static_start_markers and name.startswith("start_agent_"):
            world.remove(include)
    for agent_idx, uri in enumerate(model_uris):
        include = ET.SubElement(world, "include")
        ET.SubElement(include, "uri").text = str(uri)
        ET.SubElement(include, "name").text = f"{include_prefix}{agent_idx}"
        pos = positions0[agent_idx, :3]
        rpy = _quat_wxyz_to_rpy(orientations0[agent_idx, :4])
        ET.SubElement(include, "pose").text = _vec([pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2]])
    _write_xml(root, output_path)
    return world_name


def export_gazebo_live_world(
    scenario_json: Path,
    output_dir: Optional[Path] = None,
    base_world_sdf: Optional[Path] = None,
    uav_marker_scale: Optional[float] = None,
    collision_mode: Optional[str] = None,
    reuse_cache: bool = True,
) -> Dict[str, Any]:
    scenario_json = Path(scenario_json).resolve()
    scenario = _load_json(scenario_json)
    if output_dir is None:
        output_dir = scenario_json.parent
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if base_world_sdf is None:
        base_world_sdf = output_dir / "world.sdf"
    base_world_sdf = Path(base_world_sdf).resolve()

    starts = scenario.get("start_positions", []) if isinstance(scenario.get("start_positions"), list) else []
    if not starts:
        raise ValueError(f"scenario has no start_positions: {scenario_json}")
    positions0 = []
    for idx, start in enumerate(starts):
        pos = start.get("position") if isinstance(start, dict) else None
        arr = np.asarray(pos, dtype=np.float64).reshape(-1) if pos is not None else np.zeros(0, dtype=np.float64)
        if arr.size < 3:
            raise ValueError(f"invalid start position for agent {idx}: {pos}")
        positions0.append(arr[:3])
    positions0_arr = np.asarray(positions0, dtype=np.float64)
    orientations0 = np.zeros((positions0_arr.shape[0], 4), dtype=np.float64)
    orientations0[:, 0] = 1.0

    model_parent_dir = output_dir / "models"
    model_parent_dir.mkdir(parents=True, exist_ok=True)
    colors = [
        (0.0, 0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 0.7, 0.2, 1.0),
    ]
    try:
        fallback_agent_size = float(scenario.get("agent_size", 0.5))
    except Exception:
        fallback_agent_size = 0.5
    try:
        collision_threshold = float(scenario.get("collision", {}).get("collision_distance_threshold", 0.5))
    except Exception:
        collision_threshold = 0.5
    collision_mode_norm = _normalize_collision_mode(
        collision_mode if collision_mode is not None else os.getenv("GAZEBO_LIVE_COLLISION_MODE", "hard")
    )
    physical_collision_enabled = collision_mode_norm != "nonblocking"

    dynamic_models = []
    model_uris = []
    live_namespace = _live_namespace(output_dir)
    world_name_override = _sanitize_identifier(os.getenv("GAZEBO_LIVE_WORLD_NAME", "").strip(), "matd3_live")
    if not os.getenv("GAZEBO_LIVE_WORLD_NAME", "").strip():
        world_name_override = f"matd3_live_{live_namespace}"
    agent_prefix_override = os.getenv("GAZEBO_LIVE_AGENT_PREFIX_OVERRIDE", "").strip()
    include_name_prefix = _sanitize_identifier(agent_prefix_override, "dynamic_agent")
    if not agent_prefix_override:
        include_name_prefix = f"dynamic_agent_{live_namespace}_"
    elif not include_name_prefix.endswith("_"):
        include_name_prefix = f"{include_name_prefix}_"
    for agent_idx, start in enumerate(starts):
        try:
            agent_size = float(start.get("agent_size", fallback_agent_size)) if isinstance(start, dict) else fallback_agent_size
        except Exception:
            agent_size = fallback_agent_size
        collision_radius, collision_radius_source = _resolve_collision_envelope_radius(agent_size, collision_threshold)
        visual_scale = _resolve_visual_scale(agent_size, uav_marker_scale, default_multiplier=3.0)
        model_name = f"{DYNAMIC_MODEL_PREFIX}_{agent_idx}"
        model_meta = write_dynamic_uav_model(
            model_parent_dir=model_parent_dir,
            model_name=model_name,
            accent_rgba=colors[agent_idx % len(colors)],
            visual_scale=visual_scale,
            collision_envelope_radius=collision_radius,
            enable_velocity_control=True,
            enable_physical_collision=physical_collision_enabled,
            enable_contact_sensor=physical_collision_enabled,
            show_collision_envelope=True,
        )
        model_meta["agent_size"] = float(agent_size)
        model_meta["collision_distance_threshold"] = float(collision_threshold)
        model_meta["collision_envelope_radius_source"] = collision_radius_source
        model_meta["collision_mode"] = collision_mode_norm
        dynamic_models.append(model_meta)
        model_uris.append(model_meta["uri"])

    world_live_sdf = output_dir / "world_live_weight_test.sdf"
    world_name = write_dynamic_world_sdf(
        base_world_sdf=base_world_sdf,
        output_path=world_live_sdf,
        positions0=positions0_arr,
        orientations0=orientations0,
        model_uris=model_uris,
        remove_static_start_markers=True,
        world_name_override=world_name_override,
        include_name_prefix=include_name_prefix,
    )
    live_meta = {
        "scenario_json": str(scenario_json),
        "base_world_sdf": str(base_world_sdf),
        "world_live_sdf": str(world_live_sdf),
        "world_name": world_name,
        "agent_prefix": include_name_prefix,
        "namespace": live_namespace,
        "model_parent_dir": str(model_parent_dir),
        "dynamic_models": dynamic_models,
        "collision_mode": collision_mode_norm,
        "physical_collision_enabled": bool(physical_collision_enabled),
        "resource_path_command": f"export GZ_SIM_RESOURCE_PATH={model_parent_dir}:${{GZ_SIM_RESOURCE_PATH}}",
        "run_commands": {
            "gazebo_server_paused": f"gz sim -s {world_live_sdf}",
            "gazebo_gui": f"gz sim {world_live_sdf}",
        },
        "test_mode": "live_velocity_cmd_with_state_feedback",
        "reuse_cache": bool(reuse_cache),
    }
    scenario.setdefault("gazebo_live_test", {})
    scenario["gazebo_live_test"].update(live_meta)
    with scenario_json.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(scenario), f, ensure_ascii=False, indent=2)
    return live_meta


def write_player_script(path: Path, dynamic_npz_path: Path, world_name: str, agent_count: int) -> None:
    script = f"""#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


DEFAULT_NPZ = {str(dynamic_npz_path)!r}
DEFAULT_WORLD = {world_name!r}
DEFAULT_AGENT_COUNT = {int(agent_count)}


def _req(name, pos, quat):
    w, x, y, z = [float(v) for v in quat[:4]]
    px, py, pz = [float(v) for v in pos[:3]]
    return (
        'name: "' + str(name) + '" '
        + 'position: {{' + f'x: {{px:.8g}} y: {{py:.8g}} z: {{pz:.8g}}' + '}} '
        + 'orientation: {{' + f'w: {{w:.8g}} x: {{x:.8g}} y: {{y:.8g}} z: {{z:.8g}}' + '}}'
    )


def _set_pose(world, name, pos, quat, timeout):
    cmd = [
        'gz', 'service',
        '-s', f'/world/{{world}}/set_pose',
        '--reqtype', 'gz.msgs.Pose',
        '--reptype', 'gz.msgs.Boolean',
        '--timeout', str(int(timeout)),
        '--req', _req(name, pos, quat),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f'gz service failed with code {{proc.returncode}}')


def _pose_entry(name, pos, quat):
    return 'pose: {{' + _req(name, pos, quat) + '}}'


def _set_pose_vector(world, names, positions, orientations, timeout):
    req = ' '.join(_pose_entry(name, pos, quat) for name, pos, quat in zip(names, positions, orientations))
    cmd = [
        'gz', 'service',
        '-s', f'/world/{{world}}/set_pose_vector',
        '--reqtype', 'gz.msgs.Pose_V',
        '--reptype', 'gz.msgs.Boolean',
        '--timeout', str(int(timeout)),
        '--req', req,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f'gz set_pose_vector failed with code {{proc.returncode}}')


def _median_dt(times):
    try:
        diffs = np.diff(np.asarray(times, dtype=np.float64).reshape(-1))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            return float(np.median(diffs))
    except Exception:
        pass
    return 0.08


def _frame_indices(frame_count, times, stride, speed, auto_stride, min_call_period):
    stride = max(1, int(stride))
    if auto_stride:
        dt = max(1e-6, _median_dt(times))
        effective_stride = int(np.ceil(max(1e-6, float(speed)) * max(0.0, float(min_call_period)) / dt))
        stride = max(stride, effective_stride)
    indices = list(range(0, int(frame_count), stride))
    if not indices or indices[-1] != int(frame_count) - 1:
        indices.append(int(frame_count) - 1)
    return indices, stride


def main():
    parser = argparse.ArgumentParser(description='Play MATD3 dynamic replay by setting Gazebo model poses.')
    parser.add_argument('--npz', default=DEFAULT_NPZ)
    parser.add_argument('--world', default=DEFAULT_WORLD)
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--no-batch', action='store_true', help='Use one set_pose service call per agent instead of one set_pose_vector call per frame.')
    parser.add_argument('--no-auto-stride', action='store_true', help='Do not increase stride automatically for high playback speeds.')
    parser.add_argument('--min-call-period', type=float, default=0.035, help='Minimum target seconds between Gazebo service calls when auto-stride is enabled.')
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--timeout-ms', type=int, default=1000)
    parser.add_argument('--agent-prefix', default='{DYNAMIC_INCLUDE_PREFIX}_')
    args = parser.parse_args()
    if shutil.which('gz') is None:
        print('gz command not found. Source Gazebo/ROS setup before running this player.', file=sys.stderr)
        return 2
    data = np.load(Path(args.npz))
    positions = data['positions']
    orientations = data['orientations_wxyz'] if 'orientations_wxyz' in data.files else np.zeros((positions.shape[0], positions.shape[1], 4), dtype=np.float32)
    if not np.any(orientations):
        orientations[:, :, 0] = 1.0
    times = data['times'] if 'times' in data.files else np.arange(positions.shape[0], dtype=np.float32)
    speed = max(1e-6, float(args.speed))
    agent_count = min(DEFAULT_AGENT_COUNT, positions.shape[1])
    frame_indices, effective_stride = _frame_indices(
        positions.shape[0],
        times,
        args.stride,
        speed,
        not bool(args.no_auto_stride),
        args.min_call_period,
    )
    agent_names = [f'{{args.agent_prefix}}{{agent_idx}}' for agent_idx in range(agent_count)]
    mode = 'set_pose' if args.no_batch else 'set_pose_vector'
    print(
        f'Playing {{len(frame_indices)}}/{{positions.shape[0]}} frames, agents={{agent_count}}, '
        f'world={{args.world}}, speed={{speed}}x, stride={{effective_stride}}, mode={{mode}}'
    )
    while True:
        previous_t = float(times[frame_indices[0]])
        for frame_idx in frame_indices:
            frame_start = time.perf_counter()
            if args.no_batch:
                for agent_idx in range(agent_count):
                    _set_pose(
                        args.world,
                        agent_names[agent_idx],
                        positions[frame_idx, agent_idx, :],
                        orientations[frame_idx, agent_idx, :],
                        args.timeout_ms,
                    )
            else:
                _set_pose_vector(
                    args.world,
                    agent_names,
                    positions[frame_idx, :agent_count, :],
                    orientations[frame_idx, :agent_count, :],
                    args.timeout_ms,
                )
            current_t = float(times[frame_idx])
            target_sleep = max(0.0, (current_t - previous_t) / speed)
            previous_t = current_t
            elapsed = time.perf_counter() - frame_start
            if target_sleep > elapsed:
                time.sleep(target_sleep - elapsed)
        if not args.loop:
            break
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
"""
    path.write_text(script, encoding="utf-8")
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def export_gazebo_dynamic_replay(
    scenario_json: Path,
    trajectory_json: Path,
    dynamic_json: Path,
    output_dir: Optional[Path] = None,
    base_world_sdf: Optional[Path] = None,
    uav_marker_scale: Optional[float] = None,
    reuse_cache: bool = True,
    build_fast_replay: bool = False,
    compile_fast_replay: bool = False,
) -> Dict[str, Any]:
    scenario_json = Path(scenario_json).resolve()
    trajectory_json = Path(trajectory_json).resolve()
    dynamic_json = Path(dynamic_json).resolve()
    scenario = _load_json(scenario_json)
    dynamic = _load_json(dynamic_json)
    if output_dir is None:
        output_dir = dynamic_json.parent
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if base_world_sdf is None:
        base_world_sdf = output_dir / "world_replay.sdf"
    base_world_sdf = Path(base_world_sdf).resolve()
    dynamic_npz_path = Path(dynamic.get("dynamic_npz_path") or dynamic.get("export_paths", {}).get("dynamic_replay_npz", "")).resolve()
    if not dynamic_npz_path.exists():
        raise FileNotFoundError(f"dynamic replay npz not found: {dynamic_npz_path}")

    with np.load(dynamic_npz_path) as data:
        positions = np.asarray(data["positions"], dtype=np.float64)
        orientations = np.asarray(
            data["orientations_wxyz"] if "orientations_wxyz" in data.files else np.zeros((positions.shape[0], positions.shape[1], 4)),
            dtype=np.float64,
        )
    if positions.ndim != 3 or positions.shape[2] < 3:
        raise ValueError("dynamic positions must have shape [frame][agent][xyz]")
    if orientations.shape[:2] != positions.shape[:2] or orientations.shape[2] < 4:
        orientations = np.zeros((positions.shape[0], positions.shape[1], 4), dtype=np.float64)
        orientations[:, :, 0] = 1.0
    agent_count = int(positions.shape[1])

    model_parent_dir = output_dir / "models"
    model_parent_dir.mkdir(parents=True, exist_ok=True)
    colors = [
        (0.0, 0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 0.7, 0.2, 1.0),
    ]
    starts = scenario.get("start_positions", []) if isinstance(scenario.get("start_positions"), list) else []
    try:
        fallback_agent_size = float(dynamic.get("agent_size", 0.5))
    except Exception:
        fallback_agent_size = 0.5
    try:
        collision_threshold = float(dynamic.get("collision_distance_threshold", 0.5))
    except Exception:
        collision_threshold = 0.5

    dynamic_models = []
    model_uris = []
    for agent_idx in range(agent_count):
        start = starts[agent_idx] if agent_idx < len(starts) and isinstance(starts[agent_idx], dict) else {}
        try:
            agent_size = float(start.get("agent_size", fallback_agent_size))
        except Exception:
            agent_size = fallback_agent_size
        collision_radius, collision_radius_source = _resolve_collision_envelope_radius(agent_size, collision_threshold)
        visual_scale = _resolve_visual_scale(agent_size, uav_marker_scale, default_multiplier=3.0)
        model_name = f"{DYNAMIC_MODEL_PREFIX}_{agent_idx}"
        model_meta = write_dynamic_uav_model(
            model_parent_dir=model_parent_dir,
            model_name=model_name,
            accent_rgba=colors[agent_idx % len(colors)],
            visual_scale=visual_scale,
            collision_envelope_radius=collision_radius,
        )
        model_meta["agent_size"] = float(agent_size)
        model_meta["collision_distance_threshold"] = float(collision_threshold)
        model_meta["collision_envelope_radius_source"] = collision_radius_source
        dynamic_models.append(model_meta)
        model_uris.append(model_meta["uri"])

    world_dynamic_sdf = output_dir / "world_dynamic_replay.sdf"
    world_name = write_dynamic_world_sdf(
        base_world_sdf=base_world_sdf,
        output_path=world_dynamic_sdf,
        positions0=positions[0, :, :],
        orientations0=orientations[0, :, :],
        model_uris=model_uris,
    )
    player_script = output_dir / "play_dynamic_replay.py"
    write_player_script(player_script, dynamic_npz_path, world_name, agent_count)

    replay_meta = {
        "dynamic_replay_json": str(dynamic_json),
        "dynamic_replay_npz": str(dynamic_npz_path),
        "trajectory_snapshot_json": str(trajectory_json),
        "scenario_json": str(scenario_json),
        "world_dynamic_replay_sdf": str(world_dynamic_sdf),
        "base_world_sdf": str(base_world_sdf),
        "world_name": world_name,
        "player_script": str(player_script),
        "model_parent_dir": str(model_parent_dir),
        "dynamic_models": dynamic_models,
        "resource_path_command": f"export GZ_SIM_RESOURCE_PATH={model_parent_dir}:${{GZ_SIM_RESOURCE_PATH}}",
        "run_commands": {
            "gazebo": f"gz sim {world_dynamic_sdf}",
            "player": f"python {player_script}",
        },
        "replay_mode": "pose_service_player",
        "collision_note": "The player sets recorded Python poses in Gazebo. Contact response does not feed back into the MATD3 policy.",
        "reuse_cache": bool(reuse_cache),
    }
    if build_fast_replay or compile_fast_replay:
        try:
            from gazebo_fast_replay_builder import create_fast_replay_project

            fast_meta = create_fast_replay_project(
                dynamic_json=dynamic_json,
                output_dir=output_dir,
                world=world_name,
                compile_player=bool(compile_fast_replay),
            )
            replay_meta["fast_replay"] = fast_meta
            replay_meta["run_commands"]["fast_player"] = fast_meta.get("run_command")
            replay_meta["replay_mode"] = "native_set_pose_vector_player" if compile_fast_replay else replay_meta["replay_mode"]
        except Exception as exc:
            replay_meta["fast_replay_error"] = str(exc)
    dynamic.setdefault("gazebo_dynamic_replay", {})
    dynamic["gazebo_dynamic_replay"].update(replay_meta)
    with dynamic_json.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(dynamic), f, ensure_ascii=False, indent=2)
    return replay_meta


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Gazebo dynamic replay world and player from MATD3 snapshots.")
    parser.add_argument("--scenario-json", required=True)
    parser.add_argument("--trajectory-json", required=True)
    parser.add_argument("--dynamic-json", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--base-world-sdf", default=None)
    parser.add_argument("--uav-marker-scale", type=float, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--fast-replay", action="store_true", help="Also generate the native Gazebo Transport replay project.")
    parser.add_argument("--compile-fast-replay", action="store_true", help="Compile the native Gazebo Transport replay executable.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    replay = export_gazebo_dynamic_replay(
        scenario_json=Path(args.scenario_json),
        trajectory_json=Path(args.trajectory_json),
        dynamic_json=Path(args.dynamic_json),
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
        base_world_sdf=Path(args.base_world_sdf).resolve() if args.base_world_sdf else None,
        uav_marker_scale=args.uav_marker_scale,
        reuse_cache=not bool(args.no_cache),
        build_fast_replay=bool(args.fast_replay or args.compile_fast_replay),
        compile_fast_replay=bool(args.compile_fast_replay),
    )
    print(f"[gazebo_dynamic_replay_exporter] world_dynamic_replay_sdf={replay['world_dynamic_replay_sdf']}")
    print(f"[gazebo_dynamic_replay_exporter] player_script={replay['player_script']}")
    print(f"[gazebo_dynamic_replay_exporter] model_parent_dir={replay['model_parent_dir']}")
    print(f"[gazebo_dynamic_replay_exporter] resource_path_command={replay['resource_path_command']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
