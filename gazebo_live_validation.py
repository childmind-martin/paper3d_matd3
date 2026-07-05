#!/usr/bin/env python3
"""Validation helpers for MATD3 Gazebo-live evaluation runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_VALIDATION_DIR = Path("results/gazebo_live_validation")


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return str(value)


def validation_root_from_args(args: Any) -> Path:
    raw = getattr(args, "validation_output_dir", None) or os.getenv("GAZEBO_LIVE_VALIDATION_DIR", "")
    if not raw:
        raw = str(DEFAULT_VALIDATION_DIR)
    return Path(raw).expanduser().resolve()


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(record), ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: Path) -> Optional[str]:
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha256(value: Any) -> Optional[str]:
    try:
        arr = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    except Exception:
        return None
    if arr.size == 0:
        return None
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _as_vec3(value: Any) -> Optional[List[float]]:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return [float(v) for v in arr[:3]]


def _distance(a: Any, b: Any) -> Optional[float]:
    va = _as_vec3(a)
    vb = _as_vec3(b)
    if va is None or vb is None:
        return None
    return float(np.linalg.norm(np.asarray(va, dtype=np.float64) - np.asarray(vb, dtype=np.float64)))


def _max_vec_delta(left: Sequence[Any], right: Sequence[Any]) -> Optional[float]:
    if len(left) != len(right):
        return None
    max_delta = 0.0
    for a, b in zip(left, right):
        da = _distance(a, b)
        if da is None:
            return None
        max_delta = max(max_delta, da)
    return float(max_delta)


def _max_array_delta(left: Sequence[Any], right: Sequence[Any]) -> Optional[float]:
    if len(left) != len(right):
        return None
    max_delta = 0.0
    for a, b in zip(left, right):
        if a is None and b is None:
            continue
        try:
            aa = np.asarray(a, dtype=np.float64).reshape(-1)
            bb = np.asarray(b, dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if aa.shape != bb.shape or not np.all(np.isfinite(aa)) or not np.all(np.isfinite(bb)):
            return None
        if aa.size == 0:
            continue
        max_delta = max(max_delta, float(np.max(np.abs(aa - bb))))
    return float(max_delta)


def _normalize_obstacles(raw_obstacles: Sequence[Any]) -> List[Dict[str, Any]]:
    try:
        from scenario_exporter import normalize_obstacle
    except Exception:
        normalize_obstacle = None
    normalized = []
    for idx, obstacle in enumerate(raw_obstacles or []):
        if normalize_obstacle is not None:
            normalized.append(normalize_obstacle(obstacle, idx))
            continue
        if isinstance(obstacle, Mapping):
            item = dict(obstacle)
            item.setdefault("name", f"obstacle_{idx}")
            item.setdefault("type", "sphere" if "radius" in item else "unknown")
            normalized.append(json_safe(item))
            continue
        state = getattr(obstacle, "state", None)
        center = _as_vec3(getattr(state, "p_pos", None))
        radius = getattr(obstacle, "radius", getattr(obstacle, "size", None))
        normalized.append(
            {
                "name": str(getattr(obstacle, "name", f"obstacle_{idx}")),
                "type": "sphere",
                "type_source": "inferred_for_validation",
                "center": center,
                "radius": float(radius) if radius is not None else None,
            }
        )
    return normalized


def _obstacle_signature(obstacles: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    signature = []
    for idx, obstacle in enumerate(obstacles or []):
        obs_type = str(obstacle.get("type", "unknown"))
        item = {
            "idx": int(idx),
            "name": str(obstacle.get("name", f"obstacle_{idx}")),
            "type": obs_type,
            "center": _as_vec3(obstacle.get("center")),
        }
        if obs_type == "sphere":
            item["radius"] = _finite_float(obstacle.get("radius"))
        elif obs_type == "box":
            item["size"] = _float_list(obstacle.get("size"))
        elif obs_type == "cylinder":
            item["radius"] = _finite_float(obstacle.get("radius"))
            item["length"] = _finite_float(obstacle.get("length"))
        else:
            item["raw"] = json_safe(obstacle)
        signature.append(item)
    return signature


def _finite_float(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except Exception:
        return None
    return float(value) if math.isfinite(value) else None


def _float_list(value: Any) -> Optional[List[float]]:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return [float(v) for v in arr.tolist()]


def _python_start_positions(world: Any, scenario: Any) -> List[Dict[str, Any]]:
    starts = []
    for idx, agent in enumerate(getattr(world, "agents", []) or []):
        pos = _as_vec3(getattr(getattr(agent, "state", None), "p_pos", None))
        starts.append(
            {
                "name": str(getattr(agent, "name", f"agent_{idx}")),
                "position": pos,
                "agent_size": _finite_float(getattr(agent, "size", None)),
                "max_speed": _finite_float(getattr(agent, "max_speed", None)),
                "accel": _finite_float(getattr(agent, "accel", None)),
            }
        )
    return starts


def _python_agent_goals(world: Any, scenario: Any) -> List[Dict[str, Any]]:
    goals = []
    raw_world_goals = getattr(world, "agent_goals", None)
    for idx, agent in enumerate(getattr(world, "agents", []) or []):
        goal = None
        if raw_world_goals is not None and idx < len(raw_world_goals):
            goal = raw_world_goals[idx]
        if goal is None:
            goal_a = getattr(agent, "goal_a", None)
            goal = getattr(getattr(goal_a, "state", None), "p_pos", None)
        if goal is None:
            goal = getattr(scenario, "goal_pos", None)
        goals.append({"name": f"agent_goal_{idx}", "position": _as_vec3(goal)})
    return goals


def _python_agent_motion_states(world: Any) -> List[Dict[str, Any]]:
    states = []
    for idx, agent in enumerate(getattr(world, "agents", []) or []):
        state = getattr(agent, "state", None)
        states.append(
            {
                "name": str(getattr(agent, "name", f"agent_{idx}")),
                "initial_velocity": _as_vec3(getattr(state, "p_vel", None)),
                "initial_orientation_wxyz": _float_list(getattr(state, "orientation", None)),
            }
        )
    return states


def _scene_signature_from_parts(parts: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _canonical_signature_value(parts),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _signature_float_decimals() -> int:
    try:
        return max(0, int(os.getenv("GAZEBO_LIVE_SCENE_SIGNATURE_DECIMALS", "5")))
    except Exception:
        return 5


def _canonical_signature_value(value: Any) -> Any:
    value = json_safe(value)
    decimals = _signature_float_decimals()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(float(value), decimals)
    if isinstance(value, Mapping):
        return {str(k): _canonical_signature_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_canonical_signature_value(v) for v in value]
    return value


def build_python_scene_signature(
    *,
    scenario: Any,
    world: Any,
    terrain_info: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a backend-independent signature of the reset Python scenario."""
    terrain_info = dict(terrain_info or {})
    starts = _python_start_positions(world, scenario)
    goals = _python_agent_goals(world, scenario)
    motion_states = _python_agent_motion_states(world)
    obstacles = _obstacle_signature(_normalize_obstacles(list(getattr(scenario, "obstacles", []) or [])))
    terrain = getattr(scenario, "terrain", None)
    terrain_shape = None
    try:
        terrain_shape = [int(v) for v in np.asarray(terrain).shape]
    except Exception:
        terrain_shape = None
    collision_distance_threshold = _finite_float(getattr(scenario, "collision_distance_threshold", None))
    if collision_distance_threshold is None:
        collision_distance_threshold = _finite_float(os.getenv("COLLISION_DISTANCE_THRESHOLD", "0.5"))
    agent_sizes = []
    for agent in getattr(world, "agents", []) or []:
        size = _finite_float(getattr(agent, "size", None))
        if size is not None:
            agent_sizes.append(size)
    agent_size = agent_sizes[0] if agent_sizes else _finite_float(getattr(scenario, "agent_size", None))
    parts = {
        "terrain_seed": _first_not_none(
            terrain_info.get("terrain_seed"),
            getattr(scenario, "current_terrain_seed", getattr(scenario, "seed", None)),
        ),
        "terrain_variant_seed": _first_not_none(
            terrain_info.get("terrain_variant_seed"),
            getattr(scenario, "current_terrain_variant_seed", getattr(scenario, "terrain_variant_seed", None)),
        ),
        "obstacle_seed": _first_not_none(
            terrain_info.get("obstacle_seed"),
            getattr(scenario, "current_episode_obstacle_seed", None),
        ),
        "map_size": getattr(scenario, "map_size", None),
        "agent_count": len(getattr(world, "agents", []) or []),
        "agent_size": agent_size,
        "agent_sizes": agent_sizes,
        "collision_distance_threshold": collision_distance_threshold,
        "starts": [item.get("position") for item in starts],
        "goals": [item.get("position") for item in goals],
        "initial_velocities": [item.get("initial_velocity") for item in motion_states],
        "initial_orientations_wxyz": [item.get("initial_orientation_wxyz") for item in motion_states],
        "obstacles": obstacles,
        "terrain_shape": terrain_shape,
        "terrain_sha256": array_sha256(terrain),
    }
    return {
        "python_scene_signature": _scene_signature_from_parts(parts),
        "python_scene_signature_parts": parts,
    }


def _xml_tag_name(element: ET.Element) -> str:
    tag = str(element.tag)
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _child_text(element: ET.Element, name: str) -> Optional[str]:
    for child in list(element):
        if _xml_tag_name(child) == name:
            text = child.text
            return text.strip() if isinstance(text, str) else None
    return None


def _inspect_world_sdf(path: Optional[str], agent_prefix: Optional[str]) -> Dict[str, Any]:
    info = {
        "exists": False,
        "world_name": None,
        "agent_names": [],
        "uris": [],
        "plugin_names": [],
        "plugin_filenames": [],
        "parse_error": None,
    }
    if not path:
        return info
    sdf_path = Path(str(path)).expanduser()
    info["path"] = str(sdf_path)
    if not sdf_path.exists():
        return info
    info["exists"] = True
    try:
        root = ET.parse(sdf_path).getroot()
    except Exception as exc:
        info["parse_error"] = str(exc)
        return info
    agent_names = set()
    uris = []
    plugin_names = []
    plugin_filenames = []
    for elem in root.iter():
        tag = _xml_tag_name(elem)
        if tag == "world" and info["world_name"] is None:
            info["world_name"] = elem.attrib.get("name")
        elif tag == "plugin":
            if elem.attrib.get("name"):
                plugin_names.append(elem.attrib.get("name"))
            if elem.attrib.get("filename"):
                plugin_filenames.append(elem.attrib.get("filename"))
        elif tag == "include":
            name = _child_text(elem, "name")
            uri = _child_text(elem, "uri")
            if uri:
                uris.append(uri)
            if name and agent_prefix and str(name).startswith(str(agent_prefix)):
                agent_names.add(name)
        elif tag == "model":
            name = elem.attrib.get("name")
            if name and agent_prefix and str(name).startswith(str(agent_prefix)):
                agent_names.add(name)
        elif tag == "uri" and elem.text:
            uris.append(elem.text.strip())
    info["agent_names"] = sorted(agent_names)
    info["uris"] = sorted(set(uris))
    info["plugin_names"] = sorted(set(plugin_names))
    info["plugin_filenames"] = sorted(set(plugin_filenames))
    return info


def _inspect_model_sdf_for_terrain_uri(model_parent_dir: Optional[str]) -> Dict[str, Any]:
    expected_visual_uri = "model://matd3_terrain/meshes/terrain.obj"
    info = {
        "exists": False,
        "path": None,
        "mesh_uris": [],
        "has_expected_visual_uri": False,
        "parse_error": None,
    }
    if not model_parent_dir:
        return info
    model_sdf = Path(str(model_parent_dir)).expanduser() / "matd3_terrain" / "model.sdf"
    info["path"] = str(model_sdf)
    if not model_sdf.exists():
        return info
    info["exists"] = True
    try:
        root = ET.parse(model_sdf).getroot()
        mesh_uris = [
            elem.text.strip()
            for elem in root.iter()
            if _xml_tag_name(elem) == "uri" and isinstance(elem.text, str) and elem.text.strip()
        ]
        info["mesh_uris"] = mesh_uris
        info["has_expected_visual_uri"] = expected_visual_uri in mesh_uris
    except Exception as exc:
        info["parse_error"] = str(exc)
    return info


def _inspect_dynamic_agent_models(model_parent_dir: Optional[str], agent_count: int) -> Dict[str, Any]:
    info = {
        "agent_count": int(agent_count),
        "models": [],
        "all_exist": False,
        "all_have_velocity_control": False,
        "all_have_contact_sensor": False,
        "all_have_collision_envelope": False,
    }
    if not model_parent_dir:
        return info
    parent = Path(str(model_parent_dir)).expanduser()
    for agent_idx in range(int(agent_count)):
        model_sdf = parent / f"matd3_dynamic_uav_agent_{agent_idx}" / "model.sdf"
        item = {
            "agent_idx": int(agent_idx),
            "path": str(model_sdf),
            "exists": model_sdf.exists(),
            "has_velocity_control": False,
            "has_contact_sensor": False,
            "has_collision_envelope": False,
            "collision_envelope_radius": None,
            "contact_sensor_names": [],
            "parse_error": None,
        }
        if model_sdf.exists():
            try:
                root = ET.parse(model_sdf).getroot()
                for elem in root.iter():
                    tag = _xml_tag_name(elem)
                    if tag == "plugin":
                        if (
                            elem.attrib.get("name") == "gz::sim::systems::VelocityControl"
                            or elem.attrib.get("filename") == "gz-sim-velocity-control-system"
                        ):
                            item["has_velocity_control"] = True
                    elif tag == "sensor" and elem.attrib.get("type") == "contact":
                        item["has_contact_sensor"] = True
                        item["contact_sensor_names"].append(elem.attrib.get("name"))
                    elif tag == "collision" and elem.attrib.get("name") == "python_collision_envelope_collision":
                        item["has_collision_envelope"] = True
                        for child in elem.iter():
                            if _xml_tag_name(child) == "radius":
                                item["collision_envelope_radius"] = _finite_float(child.text)
                                break
            except Exception as exc:
                item["parse_error"] = str(exc)
        info["models"].append(item)
    info["all_exist"] = all(bool(item["exists"]) for item in info["models"])
    info["all_have_velocity_control"] = all(bool(item["has_velocity_control"]) for item in info["models"])
    info["all_have_contact_sensor"] = all(bool(item["has_contact_sensor"]) for item in info["models"])
    info["all_have_collision_envelope"] = all(bool(item["has_collision_envelope"]) for item in info["models"])
    return info


def run_scene_binding_check(
    *,
    scenario: Any,
    world: Any,
    gazebo_live: Optional[Mapping[str, Any]],
    episode_idx: int,
    terrain_info: Optional[Mapping[str, Any]] = None,
    validation_root: Optional[Path] = None,
    tolerance: float = 1e-5,
) -> Dict[str, Any]:
    """Verify that the launched Gazebo-live scene matches the current Python scenario."""
    validation_root = Path(validation_root or DEFAULT_VALIDATION_DIR).resolve()
    terrain_info = dict(terrain_info or {})
    checks: List[Dict[str, Any]] = []

    def add_check(name: str, ok: bool, **extra: Any) -> None:
        checks.append({"name": name, "ok": bool(ok), **json_safe(extra)})

    python_starts = _python_start_positions(world, scenario)
    python_goals = _python_agent_goals(world, scenario)
    python_motion_states = _python_agent_motion_states(world)
    python_obstacles = _normalize_obstacles(list(getattr(scenario, "obstacles", []) or []))

    expected_agent_count = len(getattr(world, "agents", []) or [])
    expected_obstacle_count = len(python_obstacles)
    expected_collision_threshold = _finite_float(getattr(scenario, "collision_distance_threshold", None))
    if expected_collision_threshold is None:
        expected_collision_threshold = _finite_float(os.getenv("COLLISION_DISTANCE_THRESHOLD", "0.5"))
    expected_agent_sizes = []
    for agent in getattr(world, "agents", []) or []:
        size = _finite_float(getattr(agent, "size", None))
        if size is not None:
            expected_agent_sizes.append(size)
    expected_agent_size = expected_agent_sizes[0] if expected_agent_sizes else _finite_float(getattr(scenario, "agent_size", None))

    scenario_json_path = None
    snapshot: Dict[str, Any] = {}
    if gazebo_live:
        scenario_json_path = gazebo_live.get("scenario_json") or gazebo_live.get("gazebo_live_scenario_json")
    if scenario_json_path:
        scenario_json_path = str(Path(scenario_json_path).expanduser().resolve())
        try:
            with Path(scenario_json_path).open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                snapshot = loaded
        except Exception as exc:
            add_check("scenario_json_load", False, path=scenario_json_path, error=str(exc))
    add_check("scenario_json_exists", bool(scenario_json_path and Path(scenario_json_path).exists()), path=scenario_json_path)

    terrain_dense_path = None
    terrain_obj_path = None
    if snapshot:
        terrain_dense_path = (
            snapshot.get("terrain", {}).get("dense_npy")
            or snapshot.get("export_paths", {}).get("terrain_dense_npy")
        )
        terrain_obj_path = snapshot.get("gazebo", {}).get("visual_mesh", {}).get("path")
    if not terrain_obj_path and gazebo_live:
        output_dir = Path(str(gazebo_live.get("output_dir", ""))).expanduser()
        terrain_obj_path = str(output_dir / "models" / "matd3_terrain" / "meshes" / "terrain.obj")
    add_check("terrain_dense_exists", bool(terrain_dense_path and Path(terrain_dense_path).exists()), path=terrain_dense_path)
    add_check("terrain_obj_exists", bool(terrain_obj_path and Path(terrain_obj_path).exists()), path=terrain_obj_path)

    dense_shape = None
    dense_sha = None
    if terrain_dense_path and Path(terrain_dense_path).exists():
        try:
            dense = np.load(terrain_dense_path)
            dense_shape = [int(v) for v in dense.shape]
            dense_sha = file_sha256(Path(terrain_dense_path))
            expected_shape = snapshot.get("terrain", {}).get("dense_shape") if snapshot else None
            add_check("terrain_dense_shape", not expected_shape or list(expected_shape) == dense_shape, actual=dense_shape, expected=expected_shape)
        except Exception as exc:
            add_check("terrain_dense_read", False, path=terrain_dense_path, error=str(exc))

    coordinate_domain = snapshot.get("coordinate_domain", {}) if snapshot else {}
    add_check(
        "coordinate_domain_explicit",
        coordinate_domain.get("domain_source") == "[0, map_size-1]",
        coordinate_domain=coordinate_domain,
    )

    live_meta = snapshot.get("gazebo_live_test", {}) if snapshot else {}
    collision_mode = str(
        live_meta.get("collision_mode")
        or (gazebo_live or {}).get("collision_mode")
        or os.getenv("GAZEBO_LIVE_COLLISION_MODE", "hard")
    ).strip().lower()
    if collision_mode in (
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
        collision_mode = "nonblocking"
    else:
        collision_mode = "hard"
    expected_physical_collision = collision_mode != "nonblocking"
    expected_world_name = gazebo_live.get("world_name") if gazebo_live else None
    expected_agent_prefix = gazebo_live.get("agent_prefix") if gazebo_live else None
    world_sdf_path = gazebo_live.get("world_live_sdf") if gazebo_live else None
    model_parent_dir = gazebo_live.get("model_parent_dir") if gazebo_live else None
    add_check("world_name_bound", bool(expected_world_name and live_meta.get("world_name") == expected_world_name), expected=expected_world_name, actual=live_meta.get("world_name"))
    add_check("agent_prefix_bound", bool(expected_agent_prefix and live_meta.get("agent_prefix") == expected_agent_prefix), expected=expected_agent_prefix, actual=live_meta.get("agent_prefix"))

    world_sdf_info = _inspect_world_sdf(world_sdf_path, expected_agent_prefix)
    terrain_model_info = _inspect_model_sdf_for_terrain_uri(model_parent_dir)
    dynamic_agent_model_info = _inspect_dynamic_agent_models(model_parent_dir, expected_agent_count)
    add_check("world_sdf_exists", bool(world_sdf_info.get("exists")), path=world_sdf_path)
    add_check(
        "world_sdf_world_name_match",
        bool(expected_world_name and world_sdf_info.get("world_name") == expected_world_name),
        expected=expected_world_name,
        actual=world_sdf_info.get("world_name"),
        parse_error=world_sdf_info.get("parse_error"),
    )
    add_check(
        "world_sdf_agent_prefix_count",
        len(world_sdf_info.get("agent_names", [])) == expected_agent_count,
        expected=expected_agent_count,
        actual=len(world_sdf_info.get("agent_names", [])),
        agent_names=world_sdf_info.get("agent_names", []),
    )
    add_check(
        "world_sdf_terrain_include",
        "model://matd3_terrain" in world_sdf_info.get("uris", []),
        uris=world_sdf_info.get("uris", []),
    )
    add_check(
        "world_sdf_contact_system",
        "gz::sim::systems::Contact" in world_sdf_info.get("plugin_names", []),
        plugin_names=world_sdf_info.get("plugin_names", []),
    )
    add_check(
        "terrain_model_sdf_mesh_uri",
        bool(terrain_model_info.get("has_expected_visual_uri")),
        model_sdf=terrain_model_info.get("path"),
        mesh_uris=terrain_model_info.get("mesh_uris", []),
        parse_error=terrain_model_info.get("parse_error"),
    )
    add_check(
        "dynamic_agent_velocity_control",
        bool(dynamic_agent_model_info.get("all_have_velocity_control")),
        models=dynamic_agent_model_info.get("models", []),
    )
    dynamic_models = dynamic_agent_model_info.get("models", [])
    if expected_physical_collision:
        dynamic_contact_ok = bool(
            dynamic_agent_model_info.get("all_have_contact_sensor")
            and dynamic_agent_model_info.get("all_have_collision_envelope")
        )
    else:
        dynamic_contact_ok = all(
            (not bool(item.get("has_contact_sensor"))) and (not bool(item.get("has_collision_envelope")))
            for item in dynamic_models
        )
    add_check(
        "dynamic_agent_contact_sensors",
        dynamic_contact_ok,
        expected_physical_collision=expected_physical_collision,
        collision_mode=collision_mode,
        models=dynamic_models,
    )
    resource_parts = [p for p in os.environ.get("GZ_SIM_RESOURCE_PATH", "").split(":") if p]
    add_check(
        "gz_resource_path_contains_model_parent",
        bool(model_parent_dir and str(Path(model_parent_dir).expanduser().resolve()) in {str(Path(p).expanduser().resolve()) for p in resource_parts}),
        model_parent_dir=model_parent_dir,
        gz_sim_resource_path=os.environ.get("GZ_SIM_RESOURCE_PATH", ""),
    )

    snapshot_starts = snapshot.get("start_positions", []) if isinstance(snapshot.get("start_positions"), list) else []
    snapshot_goals = snapshot.get("agent_goals", []) if isinstance(snapshot.get("agent_goals"), list) else []
    snapshot_obstacles = snapshot.get("obstacles", []) if isinstance(snapshot.get("obstacles"), list) else []
    add_check("agent_count", len(snapshot_starts) == expected_agent_count, actual=len(snapshot_starts), expected=expected_agent_count)
    add_check("obstacle_count", len(snapshot_obstacles) == expected_obstacle_count, actual=len(snapshot_obstacles), expected=expected_obstacle_count)

    start_delta = _max_vec_delta(
        [item.get("position") for item in python_starts],
        [item.get("position") for item in snapshot_starts],
    )
    goal_delta = _max_vec_delta(
        [item.get("position") for item in python_goals],
        [item.get("position") for item in snapshot_goals],
    )
    velocity_delta = _max_vec_delta(
        [item.get("initial_velocity") for item in python_motion_states],
        [item.get("initial_velocity") for item in snapshot_starts],
    )
    orientation_delta = _max_array_delta(
        [item.get("initial_orientation_wxyz") for item in python_motion_states],
        [item.get("initial_orientation_wxyz") for item in snapshot_starts],
    )
    add_check("start_positions_match", start_delta is not None and start_delta <= tolerance, max_delta=start_delta, tolerance=tolerance)
    add_check("agent_goals_match", goal_delta is not None and goal_delta <= tolerance, max_delta=goal_delta, tolerance=tolerance)
    add_check("initial_velocities_match", velocity_delta is not None and velocity_delta <= tolerance, max_delta=velocity_delta, tolerance=tolerance)
    add_check("initial_orientations_match", orientation_delta is not None and orientation_delta <= tolerance, max_delta=orientation_delta, tolerance=tolerance)

    python_obstacle_sig = _obstacle_signature(python_obstacles)
    snapshot_obstacle_sig = _obstacle_signature(snapshot_obstacles)
    add_check("obstacle_geometry_match", python_obstacle_sig == snapshot_obstacle_sig, python=python_obstacle_sig, snapshot=snapshot_obstacle_sig)

    expected_obstacle_seed = _first_not_none(
        terrain_info.get("obstacle_seed"),
        getattr(scenario, "current_episode_obstacle_seed", None),
    )
    snapshot_obstacle_seed = snapshot.get("obstacle_seed") if snapshot else None
    add_check(
        "obstacle_seed_match",
        expected_obstacle_seed == snapshot_obstacle_seed,
        expected=expected_obstacle_seed,
        actual=snapshot_obstacle_seed,
    )

    python_terrain_params = dict(getattr(scenario, "terrain_params", {}) or {})
    snapshot_terrain_params = snapshot.get("terrain_params", {}) if snapshot else {}
    expected_min_distance = python_terrain_params.get("min_distance")
    snapshot_min_distance = snapshot_terrain_params.get("min_distance") if isinstance(snapshot_terrain_params, Mapping) else None
    add_check(
        "terrain_min_distance_match",
        expected_min_distance == snapshot_min_distance,
        expected=expected_min_distance,
        actual=snapshot_min_distance,
    )

    collision_meta = snapshot.get("collision", {}) if snapshot else {}
    exported_collision_threshold = _finite_float(collision_meta.get("collision_distance_threshold"))
    exported_agent_sizes = [
        _finite_float(item.get("agent_size"))
        for item in snapshot_starts
        if isinstance(item, Mapping) and _finite_float(item.get("agent_size")) is not None
    ]
    exported_agent_size = exported_agent_sizes[0] if exported_agent_sizes else _finite_float(collision_meta.get("agent_size"))
    add_check(
        "collision_distance_threshold_match",
        expected_collision_threshold is not None
        and exported_collision_threshold is not None
        and abs(expected_collision_threshold - exported_collision_threshold) <= tolerance,
        expected=expected_collision_threshold,
        actual=exported_collision_threshold,
        tolerance=tolerance,
    )
    agent_size_delta = _max_array_delta(expected_agent_sizes, exported_agent_sizes)
    add_check(
        "agent_size_match",
        agent_size_delta is not None and agent_size_delta <= tolerance,
        expected=expected_agent_sizes,
        actual=exported_agent_sizes,
        max_delta=agent_size_delta,
        tolerance=tolerance,
    )
    model_radii = [
        _finite_float(item.get("collision_envelope_radius"))
        for item in dynamic_agent_model_info.get("models", [])
    ]
    if expected_physical_collision:
        model_radius_delta = _max_array_delta(expected_agent_sizes, model_radii)
        model_radius_ok = model_radius_delta is not None and model_radius_delta <= tolerance
        expected_model_radii = expected_agent_sizes
    else:
        expected_model_radii = [0.0] * len(model_radii)
        model_radius_delta = _max_array_delta(
            expected_model_radii,
            [0.0 if value is None else value for value in model_radii],
        )
        model_radius_ok = model_radius_delta is not None and model_radius_delta <= tolerance
    add_check(
        "dynamic_agent_collision_radius_match",
        model_radius_ok,
        expected=expected_model_radii,
        actual=model_radii,
        max_delta=model_radius_delta,
        tolerance=tolerance,
        expected_physical_collision=expected_physical_collision,
        collision_mode=collision_mode,
    )

    signature_parts = {
        "terrain_seed": _first_not_none(terrain_info.get("terrain_seed"), snapshot.get("terrain_seed") if snapshot else None),
        "terrain_variant_seed": _first_not_none(terrain_info.get("terrain_variant_seed"), snapshot.get("terrain_variant_seed") if snapshot else None),
        "obstacle_seed": _first_not_none(terrain_info.get("obstacle_seed"), snapshot.get("obstacle_seed") if snapshot else None),
        "map_size": snapshot.get("map_size") if snapshot else getattr(scenario, "map_size", None),
        "world_name": expected_world_name,
        "agent_prefix": expected_agent_prefix,
        "agent_count": expected_agent_count,
        "agent_size": expected_agent_size,
        "agent_sizes": expected_agent_sizes,
        "collision_mode": collision_mode,
        "physical_collision_enabled": expected_physical_collision,
        "collision_distance_threshold": expected_collision_threshold,
        "starts": [item.get("position") for item in python_starts],
        "goals": [item.get("position") for item in python_goals],
        "initial_velocities": [item.get("initial_velocity") for item in python_motion_states],
        "initial_orientations_wxyz": [item.get("initial_orientation_wxyz") for item in python_motion_states],
        "obstacles": python_obstacle_sig,
        "terrain_dense_shape": dense_shape,
        "terrain_dense_sha256": dense_sha,
        "terrain_obj_sha256": file_sha256(Path(terrain_obj_path)) if terrain_obj_path else None,
    }
    ok = all(bool(item.get("ok")) for item in checks)
    record = {
        "episode": int(episode_idx),
        "ok": bool(ok),
        "scene_signature": _scene_signature_from_parts(signature_parts),
        "scenario_json": scenario_json_path,
        "terrain_dense_npy": str(terrain_dense_path) if terrain_dense_path else None,
        "terrain_obj": str(terrain_obj_path) if terrain_obj_path else None,
        "world_sdf": str(world_sdf_path) if world_sdf_path else None,
        "model_parent_dir": str(model_parent_dir) if model_parent_dir else None,
        "world_name": expected_world_name,
        "agent_prefix": expected_agent_prefix,
        "world_sdf_info": world_sdf_info,
        "terrain_model_info": terrain_model_info,
        "dynamic_agent_model_info": dynamic_agent_model_info,
        "checks": checks,
        "signature_parts": signature_parts,
    }
    append_jsonl(validation_root / "scene_check_logs.jsonl", record)
    return record


def capture_pose_metrics(
    *,
    scenario: Any,
    positions: Sequence[Any],
    goals: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    terrain_clearances = []
    nearest_obstacle_distances = []
    goal_distances = []
    obstacles = list(getattr(scenario, "obstacles", []) or [])
    normalized_obstacles = _normalize_obstacles(obstacles)
    for idx, pos in enumerate(positions or []):
        pos3 = _as_vec3(pos)
        if pos3 is None:
            terrain_clearances.append(None)
            nearest_obstacle_distances.append(None)
            goal_distances.append(None)
            continue
        terrain_clearance = None
        try:
            terrain_clearance = float(pos3[2] - float(scenario.get_terrain_height(pos3[0], pos3[1])))
        except Exception:
            terrain_clearance = None
        terrain_clearances.append(terrain_clearance)

        nearest = None
        for obstacle in normalized_obstacles:
            center = _as_vec3(obstacle.get("center"))
            if center is None:
                continue
            dist = float(np.linalg.norm(np.asarray(pos3) - np.asarray(center)))
            if obstacle.get("type") == "sphere":
                radius = _finite_float(obstacle.get("radius")) or 0.0
                dist -= radius
            nearest = dist if nearest is None else min(nearest, dist)
        nearest_obstacle_distances.append(nearest)

        goal = goals[idx] if goals is not None and idx < len(goals) else None
        goal_distances.append(_distance(pos3, goal))
    return {
        "terrain_clearance": terrain_clearances,
        "nearest_obstacle_distance": nearest_obstacle_distances,
        "goal_distance": goal_distances,
    }


def build_sync_health_record(
    *,
    episode_data: Mapping[str, Any],
    backend: str,
    validation_root: Optional[Path] = None,
) -> Dict[str, Any]:
    nested_sync = (
        episode_data.get("gazebo_live_sync_health")
        if isinstance(episode_data.get("gazebo_live_sync_health"), Mapping)
        else {}
    )

    def sync_value(top_level_key: str, nested_key: Optional[str] = None, default: Any = None) -> Any:
        value = episode_data.get(top_level_key)
        if value is not None:
            return value
        return nested_sync.get(nested_key or top_level_key, default)

    cmd_count = int(episode_data.get("gazebo_live_cmd_vel_publish_count", 0) or 0)
    state_updates = int(episode_data.get("gazebo_live_state_feedback_updates", 0) or 0)
    auth_updates = int(episode_data.get("gazebo_live_authoritative_feedback_updates", 0) or 0)
    denominator = max(cmd_count, 1)
    backend = str(backend)
    state_feedback_enabled = bool(episode_data.get("gazebo_live_state_feedback", False))
    auth_feedback_enabled = bool(episode_data.get("gazebo_live_authoritative_feedback", False))
    contact_feedback_enabled = bool(episode_data.get("gazebo_live_contact_feedback", False))
    contact_feedback_armed = bool(episode_data.get("gazebo_live_contact_feedback_armed", False))
    bridge_running = bool(episode_data.get("gazebo_live_bridge_running", False))
    bridge_ack_enabled = bool(episode_data.get("gazebo_live_bridge_ack_enabled", False))
    bridge_ack_count = int(episode_data.get("gazebo_live_bridge_ack_count", 0) or 0)
    bridge_ack_timeout_count = int(episode_data.get("gazebo_live_bridge_ack_timeout_count", 0) or 0)
    sync_active = bool(episode_data.get("gazebo_live_sync_active", False))
    if backend != "gazebo_live":
        state_feedback_enabled = False
        auth_feedback_enabled = False
        contact_feedback_enabled = False
        contact_feedback_armed = False
        bridge_running = False
    jump_rejects = int(episode_data.get("gazebo_live_pose_jump_reject_count", 0) or 0)
    failure_reasons = []
    if backend == "gazebo_live":
        if not sync_active:
            failure_reasons.append("sync_inactive")
        if sync_active and not bridge_running:
            failure_reasons.append("bridge_not_running")
        if cmd_count <= 0:
            failure_reasons.append("no_cmd_vel_published")
        if bridge_ack_enabled and bridge_ack_count <= 0:
            failure_reasons.append("no_bridge_ack")
        if bridge_ack_timeout_count > 0:
            failure_reasons.append("bridge_ack_timeout")
        if state_feedback_enabled and state_updates <= 0:
            failure_reasons.append("no_pose_feedback_updates")
        if auth_feedback_enabled and auth_updates <= 0:
            failure_reasons.append("no_authoritative_feedback_updates")
        if contact_feedback_enabled and not contact_feedback_armed:
            failure_reasons.append("contact_feedback_not_armed")
        if jump_rejects > 0:
            failure_reasons.append("pose_jump_rejected")
        if episode_data.get("gazebo_live_sync_error"):
            failure_reasons.append("sync_error")
        if episode_data.get("gazebo_live_launch_error"):
            failure_reasons.append("launch_error")
    sync_ok = len(failure_reasons) == 0
    record = {
        "backend": backend,
        "episode": int(episode_data.get("episode", -1)),
        "gazebo_live_consistency_mode": sync_value("gazebo_live_consistency_mode"),
        "gazebo_live_collision_mode": sync_value("gazebo_live_collision_mode", default="hard"),
        "gazebo_live_physical_collision_enabled": bool(sync_value("gazebo_live_physical_collision_enabled", default=True)),
        "gazebo_live_python_authoritative": bool(sync_value("gazebo_live_python_authoritative", default=False)),
        "gazebo_live_contact_authoritative": bool(sync_value("gazebo_live_contact_authoritative", default=True)),
        "gazebo_live_contact_marks_collision": bool(sync_value("gazebo_live_contact_marks_collision", default=True)),
        "gazebo_live_contact_terminates": bool(sync_value("gazebo_live_contact_terminates", default=False)),
        "gazebo_live_pose_correction": bool(sync_value("gazebo_live_pose_correction", default=False)),
        "gazebo_live_sync_active": sync_active,
        "sync_ok": bool(sync_ok),
        "sync_failure_reasons": failure_reasons,
        "bridge_running": bridge_running,
        "cmd_vel_publish_count": cmd_count,
        "pose_publish_count": int(episode_data.get("gazebo_live_pose_publish_count", 0) or 0),
        "bridge_ack_enabled": bridge_ack_enabled,
        "bridge_ack_count": bridge_ack_count,
        "bridge_ack_timeout_count": bridge_ack_timeout_count,
        "bridge_last_ack_state_frame": episode_data.get("gazebo_live_bridge_last_ack_state_frame"),
        "pre_step_sleep_ms": int(episode_data.get("gazebo_live_pre_step_sleep_ms", 0) or 0),
        "post_step_sleep_ms": int(episode_data.get("gazebo_live_post_step_sleep_ms", 0) or 0),
        "wall_time_step_ms": int(episode_data.get("gazebo_live_wall_time_step_ms", 0) or 0),
        "pause_for_step": bool(episode_data.get("gazebo_live_pause_for_step", False)),
        "state_feedback_updates": state_updates,
        "authoritative_feedback_updates": auth_updates,
        "state_feedback_update_ratio": float(state_updates / denominator),
        "feedback_update_ratio": float(auth_updates / denominator),
        "pose_jump_reject_count": jump_rejects,
        "max_pose_jump_observed": _finite_float(episode_data.get("gazebo_live_max_pose_jump_observed")) or 0.0,
        "world_name": episode_data.get("gazebo_live_world_name"),
        "agent_prefix": episode_data.get("gazebo_live_agent_prefix"),
        "state_feedback_enabled": state_feedback_enabled,
        "authoritative_feedback_enabled": auth_feedback_enabled,
        "contact_feedback_enabled": contact_feedback_enabled,
        "contact_feedback_armed": contact_feedback_armed,
        "contact_topics": episode_data.get("gazebo_live_contact_topics", []),
        "state_file": episode_data.get("gazebo_live_state_file"),
        "contact_flag_file": episode_data.get("gazebo_live_contact_flag_file"),
        "gazebo_contact_count": int(episode_data.get("gazebo_contact_count", 0) or 0),
        "gazebo_live_sync_error": episode_data.get("gazebo_live_sync_error"),
        "gazebo_live_launch_error": episode_data.get("gazebo_live_launch_error"),
    }
    if validation_root is not None:
        append_jsonl(Path(validation_root) / "sync_health_logs.jsonl", record)
    return record


def episode_metric_row(backend: str, episode: Mapping[str, Any]) -> Dict[str, Any]:
    min_distance = episode.get("min_distance") if isinstance(episode.get("min_distance"), Mapping) else {}
    sync = build_sync_health_record(episode_data=episode, backend=backend, validation_root=None)
    apf_metrics = episode.get("gazebo_apf_metrics") if isinstance(episode.get("gazebo_apf_metrics"), Mapping) else {}
    adapter_metrics = episode.get("gazebo_apf_adapter_metrics")
    if not isinstance(adapter_metrics, Mapping):
        adapter_metrics = apf_metrics.get("adapter_metrics") if isinstance(apf_metrics, Mapping) else {}
    if not isinstance(adapter_metrics, Mapping):
        adapter_metrics = {}
    adapter_main = adapter_metrics.get("main_report") if isinstance(adapter_metrics.get("main_report"), Mapping) else {}
    adapter_key = adapter_metrics.get("main_speed_threshold_key")
    adapter_post = {}
    if adapter_key and isinstance(adapter_metrics.get("post_contact"), Mapping):
        adapter_post = adapter_metrics["post_contact"].get(adapter_key, {})
    if not isinstance(adapter_post, Mapping):
        adapter_post = {}
    filter_summary = episode.get("gazebo_obstacle_velocity_filter_summary")
    if not isinstance(filter_summary, Mapping):
        filter_summary = {}
    filter_artifacts = episode.get("gazebo_obstacle_velocity_filter_artifacts")
    if not isinstance(filter_artifacts, Mapping):
        filter_artifacts = {}
    progress_summary = episode.get("gazebo_progress_failure_summary")
    if not isinstance(progress_summary, Mapping):
        progress_summary = {}
    progress_evidence = progress_summary.get("evidence") if isinstance(progress_summary.get("evidence"), Mapping) else {}
    progress_classes = progress_summary.get("failure_classes") if isinstance(progress_summary.get("failure_classes"), Sequence) and not isinstance(progress_summary.get("failure_classes"), (str, bytes)) else []
    first_contact_step = (
        episode.get("first_contact_step")
        if episode.get("first_contact_step") is not None
        else episode.get("gazebo_first_contact_step", episode.get("gazebo_contact_step"))
    )
    steps = int(episode.get("steps", 0) or 0)
    episode_length = episode.get("episode_length")
    timeout = bool(episode.get("episode_done_reason") == "timeout")
    if episode_length is not None:
        try:
            timeout = timeout or steps >= int(episode_length)
        except Exception:
            pass
    row = {
        "backend": backend,
        "episode": int(episode.get("episode", -1)),
        "seed": episode.get("terrain_seed"),
        "terrain_seed": episode.get("terrain_seed"),
        "terrain_variant_seed": episode.get("terrain_variant_seed"),
        "obstacle_seed": episode.get("obstacle_seed"),
        "python_scene_signature": episode.get("python_scene_signature"),
        "scene_signature": episode.get("scene_signature"),
        "success": int(episode.get("team_success", episode.get("success", 0)) or 0),
        "collision": int((episode.get("collision_count", 0) or 0) > 0 or (episode.get("gazebo_contact_count", 0) or 0) > 0),
        "timeout": int(timeout),
        "episode_steps": steps,
        "total_reward": _finite_float(episode.get("reward")),
        "path_length": _finite_float(episode.get("path_length")),
        "final_goal_distance": _finite_float(episode.get("final_goal_distance")),
        "min_obstacle_distance": _finite_float(min_distance.get("min")),
        "min_terrain_clearance": _min_tail_metric(episode, "terrain_clearance"),
        "python_collision_count": int(episode.get("collision_count", 0) or 0),
        "gazebo_contact_count": int(episode.get("gazebo_contact_count", 0) or 0),
        "hard_contact": int((episode.get("gazebo_contact_count", 0) or 0) > 0 or first_contact_step is not None),
        "gazebo_contact_raw_flag_count": int(episode.get("gazebo_contact_raw_flag_count", 0) or 0),
        "gazebo_contact_false_positive_count": int(episode.get("gazebo_contact_false_positive_count", 0) or 0),
        "first_contact_step": first_contact_step,
        "first_contact_agent_id": episode.get("first_contact_agent_id"),
        "first_contact_pair": episode.get("first_contact_pair"),
        "obstacle_velocity_filter_mode": episode.get("gazebo_obstacle_velocity_filter_mode", "off"),
        "obstacle_velocity_filter_enabled": int(bool(episode.get("gazebo_obstacle_velocity_filter_enabled", False))),
        "obstacle_velocity_filter_error": episode.get("gazebo_obstacle_velocity_filter_error"),
        "obstacle_filter_record_count": filter_summary.get("record_count"),
        "obstacle_filter_active_count": filter_summary.get("active_count"),
        "obstacle_filter_active_rate": filter_summary.get("active_rate"),
        "obstacle_filter_active_step_count": filter_summary.get("active_step_count"),
        "obstacle_filter_first_active_step": filter_summary.get("first_active_step"),
        "obstacle_filter_last_active_step": filter_summary.get("last_active_step"),
        "obstacle_filter_min_surface_distance": filter_summary.get("min_surface_distance"),
        "obstacle_filter_min_clearance": filter_summary.get("min_clearance"),
        "inward_velocity_before_filter_mean": filter_summary.get("mean_inward_velocity_before_filter"),
        "inward_velocity_before_filter_max": filter_summary.get("max_inward_velocity_before_filter"),
        "inward_velocity_after_filter_mean": filter_summary.get("mean_inward_velocity_after_filter"),
        "inward_velocity_after_filter_max": filter_summary.get("max_inward_velocity_after_filter"),
        "relative_inward_velocity_before_filter_mean": filter_summary.get("mean_relative_inward_velocity_before_filter"),
        "relative_inward_velocity_before_filter_max": filter_summary.get("max_relative_inward_velocity_before_filter"),
        "relative_inward_velocity_after_filter_mean": filter_summary.get("mean_relative_inward_velocity_after_filter"),
        "relative_inward_velocity_after_filter_max": filter_summary.get("max_relative_inward_velocity_after_filter"),
        "current_relative_inward_velocity_mean": filter_summary.get("mean_current_relative_inward_velocity"),
        "current_relative_inward_velocity_max": filter_summary.get("max_current_relative_inward_velocity"),
        "closing_inward_velocity_for_stopping_mean": filter_summary.get("mean_closing_inward_velocity_for_stopping"),
        "closing_inward_velocity_for_stopping_max": filter_summary.get("max_closing_inward_velocity_for_stopping"),
        "obstacle_filter_mean_cmd_delta_norm": filter_summary.get("mean_cmd_delta_norm"),
        "obstacle_filter_max_cmd_delta_norm": filter_summary.get("max_cmd_delta_norm"),
        "obstacle_filter_mean_outward_speed_applied": filter_summary.get("mean_outward_speed_applied"),
        "obstacle_filter_max_outward_speed_applied": filter_summary.get("max_outward_speed_applied"),
        "obstacle_filter_artifacts": filter_artifacts,
        "gazebo_progress_diagnostics_csv": episode.get("gazebo_progress_diagnostics_csv"),
        "progress_failure_summary_json": episode.get("progress_failure_summary_json"),
        "progress_failure_classes": ";".join(str(v) for v in progress_classes),
        "progress_refined_failure_labels": ";".join(
            str(v) for v in (progress_evidence.get("refined_failure_labels") or [])
        ) if isinstance(progress_evidence.get("refined_failure_labels"), Sequence)
        and not isinstance(progress_evidence.get("refined_failure_labels"), (str, bytes)) else "",
        "boundary_dwell_ratio": progress_evidence.get("boundary_dwell_ratio"),
        "goal_projection_after_filter_mean": progress_evidence.get(
            "goal_projection_after_filter_mean",
            progress_evidence.get("mean_goal_projection_after_filter"),
        ),
        "tangential_velocity_kept_ratio": progress_evidence.get(
            "tangential_velocity_kept_ratio",
            progress_evidence.get("mean_tangential_velocity_kept_ratio"),
        ),
        "filter_invasiveness": progress_evidence.get(
            "filter_invasiveness",
            progress_evidence.get("mean_filter_invasiveness"),
        ),
        "arrived_hold_active_ratio": progress_evidence.get("arrived_hold_active_rate"),
        "goal_floor_active_ratio": progress_evidence.get("goal_floor_active_rate"),
        "goal_floor_safety_conflict_rate": progress_evidence.get("goal_floor_safety_conflict_rate"),
        "single_laggard_finish_active_ratio": progress_evidence.get("single_laggard_finish_active_rate"),
        "finish_safety_conflict_rate": progress_evidence.get("finish_safety_conflict_rate"),
        "candidate_pool_collapsed_rate": progress_evidence.get("candidate_pool_collapsed_rate"),
        "accepted_candidate_count_mean": progress_evidence.get("accepted_candidate_count_mean"),
        "reject_terrain_count_sum": progress_evidence.get("reject_terrain_count_sum"),
        "reject_obstacle_count_sum": progress_evidence.get("reject_obstacle_count_sum"),
        "reject_agent_count_sum": progress_evidence.get("reject_agent_count_sum"),
        "reject_speed_count_sum": progress_evidence.get("reject_speed_count_sum"),
        "reject_accel_count_sum": progress_evidence.get("reject_accel_count_sum"),
        "reject_projection_failed_count_sum": progress_evidence.get("reject_projection_failed_count_sum"),
        "candidate_selected_counts": progress_evidence.get("candidate_selected_counts"),
        "max_stalled_steps": progress_evidence.get("max_stalled_steps"),
        "mean_goal_progress_rate": progress_evidence.get("mean_goal_progress_rate"),
        "python_clearance_violation_rate": progress_evidence.get("python_clearance_violation_rate"),
        "geometric_penetration_rate": progress_evidence.get("geometric_penetration_rate"),
        "line_to_goal_blocked_rate": progress_evidence.get("line_to_goal_blocked_rate"),
        "line_to_goal_blocked_obstacle_rate": progress_evidence.get("line_to_goal_blocked_obstacle_rate"),
        "line_to_goal_blocked_terrain_rate": progress_evidence.get("line_to_goal_blocked_terrain_rate"),
        "line_to_goal_blocked_agent_rate": progress_evidence.get("line_to_goal_blocked_agent_rate"),
        "line_to_goal_blocked_any_direct_rate": progress_evidence.get("line_to_goal_blocked_any_direct_rate"),
        "line_to_goal_min_obstacle_clearance_min": progress_evidence.get("line_to_goal_min_obstacle_clearance_min"),
        "line_to_goal_min_terrain_clearance_min": progress_evidence.get("line_to_goal_min_terrain_clearance_min"),
        "line_to_goal_min_agent_clearance_min": progress_evidence.get("line_to_goal_min_agent_clearance_min"),
        "progress_timeout_failure": int(bool(progress_summary.get("timeout", False))),
        "gazebo_live_consistency_mode": sync["gazebo_live_consistency_mode"],
        "gazebo_live_python_authoritative": int(sync["gazebo_live_python_authoritative"]),
        "gazebo_live_contact_authoritative": int(sync["gazebo_live_contact_authoritative"]),
        "gazebo_live_contact_marks_collision": int(sync["gazebo_live_contact_marks_collision"]),
        "gazebo_live_contact_terminates": int(sync["gazebo_live_contact_terminates"]),
        "gazebo_live_pose_correction": int(sync["gazebo_live_pose_correction"]),
        "cmd_vel_publish_count": sync["cmd_vel_publish_count"],
        "bridge_ack_enabled": int(sync["bridge_ack_enabled"]),
        "bridge_ack_count": sync["bridge_ack_count"],
        "bridge_ack_timeout_count": sync["bridge_ack_timeout_count"],
        "pre_step_sleep_ms": sync["pre_step_sleep_ms"],
        "post_step_sleep_ms": sync["post_step_sleep_ms"],
        "wall_time_step_ms": sync["wall_time_step_ms"],
        "pause_for_step": int(sync["pause_for_step"]),
        "state_feedback_updates": sync["state_feedback_updates"],
        "authoritative_feedback_updates": sync["authoritative_feedback_updates"],
        "contact_feedback_enabled": int(sync["contact_feedback_enabled"]),
        "contact_feedback_armed": int(sync["contact_feedback_armed"]),
        "state_feedback_update_ratio": sync["state_feedback_update_ratio"],
        "feedback_update_ratio": sync["feedback_update_ratio"],
        "pose_jump_reject_count": sync["pose_jump_reject_count"],
        "max_pose_jump_observed": sync["max_pose_jump_observed"],
        "gazebo_live_sync_active": int(sync["gazebo_live_sync_active"]),
        "gazebo_live_sync_ok": int(sync["sync_ok"]),
        "sync_failure_reasons": ";".join(sync["sync_failure_reasons"]),
        "adapter_main_speed_threshold": adapter_metrics.get("main_speed_threshold"),
        "pre_contact_cmd_feedback_sample_count": adapter_main.get("sample_count"),
        "pre_contact_cmd_feedback_error_mean": adapter_main.get("mean_error"),
        "pre_contact_cmd_feedback_error_p95": adapter_main.get("p95_error"),
        "pre_contact_cmd_feedback_error_max": adapter_main.get("max_error"),
        "pre_contact_cmd_feedback_cosine_mean": adapter_main.get("mean_cosine"),
        "pre_contact_cmd_feedback_cosine_min": adapter_main.get("min_cosine"),
        "pre_contact_cmd_feedback_axis_mae": adapter_main.get("axis_mae"),
        "pre_contact_cmd_feedback_axis_sign_match_rate": adapter_main.get("axis_sign_match_rate"),
        "post_contact_cmd_feedback_sample_count": adapter_post.get("sample_count"),
        "post_contact_cmd_feedback_error_mean": adapter_post.get("mean_error"),
        "post_contact_cmd_feedback_error_p95": adapter_post.get("p95_error"),
        "post_contact_cmd_feedback_error_max": adapter_post.get("max_error"),
        "post_contact_cmd_feedback_cosine_mean": adapter_post.get("mean_cosine"),
        "post_contact_cmd_feedback_cosine_min": adapter_post.get("min_cosine"),
        "world_name": sync["world_name"],
        "agent_prefix": sync["agent_prefix"],
    }
    return row


def _min_tail_metric(episode: Mapping[str, Any], name: str) -> Optional[float]:
    values = []
    for frame in episode.get("validation_trace_tail", []) or []:
        metrics = frame.get("metrics", {}) if isinstance(frame, Mapping) else {}
        raw = metrics.get(name)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            values.extend(_finite_float(v) for v in raw)
    values = [v for v in values if v is not None]
    return float(min(values)) if values else None


def _episodes_by_id(result: Mapping[str, Any]) -> Dict[int, Mapping[str, Any]]:
    details = result.get("episode_details", []) if isinstance(result, Mapping) else []
    out = {}
    for ep in details:
        if isinstance(ep, Mapping):
            try:
                out[int(ep.get("episode", len(out)))] = ep
            except Exception:
                continue
    return out


def _episode_success_flag(episode: Mapping[str, Any]) -> int:
    return int(episode.get("team_success", episode.get("success", 0)) or 0)


def _episode_collision_flag(episode: Mapping[str, Any]) -> int:
    collision_count = int(episode.get("collision_count", 0) or 0)
    gazebo_contact_count = int(episode.get("gazebo_contact_count", 0) or 0)
    return int(collision_count > 0 or gazebo_contact_count > 0)


def _episode_python_collision_flag(episode: Mapping[str, Any]) -> int:
    return int(int(episode.get("collision_count", 0) or 0) > 0)


def _episode_gazebo_contact_flag(episode: Mapping[str, Any]) -> int:
    return int(int(episode.get("gazebo_contact_count", 0) or 0) > 0)


def _episode_done_reason(episode: Mapping[str, Any]) -> str:
    return str(episode.get("episode_done_reason") or episode.get("done_reason") or "unknown")


def _episode_timeout_flag(episode: Mapping[str, Any]) -> int:
    reason = _episode_done_reason(episode).strip().lower()
    if reason in ("time_limit", "timeout", "max_steps", "episode_length"):
        return 1
    try:
        steps = int(episode.get("steps", 0) or 0)
        episode_length = int(episode.get("episode_length", 0) or 0)
        if episode_length > 0 and steps >= episode_length:
            return 1
    except Exception:
        pass
    return 0


def _abs_gap(left: Any, right: Any) -> Optional[float]:
    lval = _finite_float(left)
    rval = _finite_float(right)
    if lval is None or rval is None:
        return None
    return float(abs(rval - lval))


def _relative_abs_gap(left: Any, right: Any) -> Optional[float]:
    gap = _abs_gap(left, right)
    lval = _finite_float(left)
    if gap is None or lval is None:
        return None
    return float(gap / max(abs(lval), 1.0))


def _finite_values(values: Iterable[Any]) -> List[float]:
    out = []
    for value in values:
        val = _finite_float(value)
        if val is not None:
            out.append(float(val))
    return out


def write_episode_metrics_csv(results_by_backend: Mapping[str, Mapping[str, Any]], output_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for backend, result in results_by_backend.items():
        for episode in _episodes_by_id(result).values():
            rows.append(episode_metric_row(backend, episode))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backend", "episode", "seed", "terrain_seed", "terrain_variant_seed", "obstacle_seed",
        "python_scene_signature", "scene_signature", "success", "collision", "timeout", "episode_steps", "total_reward",
        "path_length", "final_goal_distance", "min_obstacle_distance", "min_terrain_clearance",
        "python_collision_count", "gazebo_contact_count", "gazebo_contact_raw_flag_count",
        "hard_contact", "gazebo_contact_false_positive_count", "first_contact_step", "first_contact_agent_id",
        "first_contact_pair", "obstacle_velocity_filter_mode", "obstacle_velocity_filter_enabled",
        "obstacle_velocity_filter_error", "obstacle_filter_record_count",
        "obstacle_filter_active_count", "obstacle_filter_active_rate",
        "obstacle_filter_active_step_count", "obstacle_filter_first_active_step",
        "obstacle_filter_last_active_step", "obstacle_filter_min_surface_distance",
        "obstacle_filter_min_clearance", "inward_velocity_before_filter_mean",
        "inward_velocity_before_filter_max", "inward_velocity_after_filter_mean",
        "inward_velocity_after_filter_max", "relative_inward_velocity_before_filter_mean",
        "relative_inward_velocity_before_filter_max", "relative_inward_velocity_after_filter_mean",
        "relative_inward_velocity_after_filter_max", "current_relative_inward_velocity_mean",
        "current_relative_inward_velocity_max", "closing_inward_velocity_for_stopping_mean",
        "closing_inward_velocity_for_stopping_max", "obstacle_filter_mean_cmd_delta_norm",
        "obstacle_filter_max_cmd_delta_norm", "obstacle_filter_mean_outward_speed_applied",
        "obstacle_filter_max_outward_speed_applied", "obstacle_filter_artifacts",
        "gazebo_progress_diagnostics_csv", "progress_failure_summary_json",
        "progress_failure_classes", "progress_timeout_failure",
        "progress_refined_failure_labels",
        "python_clearance_violation_rate", "geometric_penetration_rate",
        "boundary_dwell_ratio", "goal_projection_after_filter_mean",
        "tangential_velocity_kept_ratio", "filter_invasiveness",
        "arrived_hold_active_ratio", "goal_floor_active_ratio",
        "goal_floor_safety_conflict_rate", "single_laggard_finish_active_ratio",
        "finish_safety_conflict_rate", "candidate_pool_collapsed_rate",
        "accepted_candidate_count_mean", "reject_terrain_count_sum",
        "reject_obstacle_count_sum", "reject_agent_count_sum",
        "reject_speed_count_sum", "reject_accel_count_sum",
        "reject_projection_failed_count_sum", "candidate_selected_counts",
        "max_stalled_steps", "mean_goal_progress_rate",
        "line_to_goal_blocked_rate", "line_to_goal_blocked_obstacle_rate",
        "line_to_goal_blocked_terrain_rate", "line_to_goal_blocked_agent_rate",
        "line_to_goal_blocked_any_direct_rate",
        "line_to_goal_min_obstacle_clearance_min", "line_to_goal_min_terrain_clearance_min",
        "line_to_goal_min_agent_clearance_min",
        "gazebo_live_sync_active",
        "gazebo_live_sync_ok", "sync_failure_reasons", "cmd_vel_publish_count",
        "gazebo_live_consistency_mode", "gazebo_live_python_authoritative",
        "gazebo_live_contact_authoritative", "gazebo_live_contact_marks_collision",
        "gazebo_live_contact_terminates", "gazebo_live_pose_correction",
        "bridge_ack_enabled", "bridge_ack_count", "bridge_ack_timeout_count",
        "pre_step_sleep_ms", "post_step_sleep_ms", "wall_time_step_ms", "pause_for_step",
        "state_feedback_updates", "authoritative_feedback_updates", "contact_feedback_enabled",
        "contact_feedback_armed", "state_feedback_update_ratio",
        "feedback_update_ratio", "pose_jump_reject_count", "max_pose_jump_observed",
        "adapter_main_speed_threshold", "pre_contact_cmd_feedback_sample_count",
        "pre_contact_cmd_feedback_error_mean", "pre_contact_cmd_feedback_error_p95",
        "pre_contact_cmd_feedback_error_max", "pre_contact_cmd_feedback_cosine_mean",
        "pre_contact_cmd_feedback_cosine_min", "pre_contact_cmd_feedback_axis_mae",
        "pre_contact_cmd_feedback_axis_sign_match_rate",
        "post_contact_cmd_feedback_sample_count", "post_contact_cmd_feedback_error_mean",
        "post_contact_cmd_feedback_error_p95", "post_contact_cmd_feedback_error_max",
        "post_contact_cmd_feedback_cosine_mean", "post_contact_cmd_feedback_cosine_min",
        "world_name", "agent_prefix",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in fieldnames})
    return rows


def build_validation_summary(rows: Sequence[Mapping[str, Any]], difference_cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_backend: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_backend.setdefault(str(row.get("backend")), []).append(row)

    def rate(backend: str, field: str) -> Optional[float]:
        vals = [int(r.get(field, 0) or 0) for r in by_backend.get(backend, [])]
        return float(np.mean(vals)) if vals else None

    def mean(backend: str, field: str) -> Optional[float]:
        vals = [_finite_float(r.get(field)) for r in by_backend.get(backend, [])]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    py_success = rate("python_only", "success")
    gz_success = rate("gazebo_live", "success")
    py_collision = rate("python_only", "collision")
    gz_collision = rate("gazebo_live", "collision")
    gz_hard_contact = rate("gazebo_live", "hard_contact")
    py_reward = mean("python_only", "total_reward")
    gz_reward = mean("gazebo_live", "total_reward")
    py_path = mean("python_only", "path_length")
    gz_path = mean("gazebo_live", "path_length")
    py_by_ep = {int(r.get("episode", -1)): r for r in by_backend.get("python_only", [])}
    gz_by_ep = {int(r.get("episode", -1)): r for r in by_backend.get("gazebo_live", [])}
    paired_scene_signature_mismatch_count = 0
    paired_scene_signature_missing_count = 0
    for episode_id in sorted(set(py_by_ep) & set(gz_by_ep)):
        py_sig = py_by_ep[episode_id].get("python_scene_signature")
        gz_sig = gz_by_ep[episode_id].get("python_scene_signature")
        if not py_sig or not gz_sig:
            paired_scene_signature_missing_count += 1
        elif py_sig != gz_sig:
            paired_scene_signature_mismatch_count += 1

    gazebo_mode_counts: Dict[str, int] = {}
    for row in by_backend.get("gazebo_live", []):
        mode = str(row.get("gazebo_live_consistency_mode") or "unknown")
        gazebo_mode_counts[mode] = int(gazebo_mode_counts.get(mode, 0) + 1)

    return {
        "backends": {backend: len(items) for backend, items in by_backend.items()},
        "gazebo_live_mode_counts": gazebo_mode_counts,
        "gazebo_hard_contact_rate": gz_hard_contact,
        "gazebo_mean_first_contact_step": mean("gazebo_live", "first_contact_step"),
        "gazebo_mean_min_surface_distance": mean("gazebo_live", "obstacle_filter_min_surface_distance"),
        "gazebo_mean_min_clearance": mean("gazebo_live", "obstacle_filter_min_clearance"),
        "gazebo_mean_inward_velocity_before_filter": mean("gazebo_live", "inward_velocity_before_filter_mean"),
        "gazebo_mean_inward_velocity_after_filter": mean("gazebo_live", "inward_velocity_after_filter_mean"),
        "gazebo_max_inward_velocity_before_filter_mean": mean("gazebo_live", "inward_velocity_before_filter_max"),
        "gazebo_max_inward_velocity_after_filter_mean": mean("gazebo_live", "inward_velocity_after_filter_max"),
        "gazebo_mean_relative_inward_velocity_before_filter": mean("gazebo_live", "relative_inward_velocity_before_filter_mean"),
        "gazebo_mean_relative_inward_velocity_after_filter": mean("gazebo_live", "relative_inward_velocity_after_filter_mean"),
        "gazebo_mean_current_relative_inward_velocity": mean("gazebo_live", "current_relative_inward_velocity_mean"),
        "gazebo_mean_closing_inward_velocity_for_stopping": mean("gazebo_live", "closing_inward_velocity_for_stopping_mean"),
        "gazebo_obstacle_filter_active_rate": mean("gazebo_live", "obstacle_filter_active_rate"),
        "gazebo_obstacle_filter_mean_cmd_delta_norm": mean("gazebo_live", "obstacle_filter_mean_cmd_delta_norm"),
        "gazebo_obstacle_filter_max_cmd_delta_norm_mean": mean("gazebo_live", "obstacle_filter_max_cmd_delta_norm"),
        "gazebo_python_clearance_violation_rate": mean("gazebo_live", "python_clearance_violation_rate"),
        "gazebo_geometric_penetration_rate": mean("gazebo_live", "geometric_penetration_rate"),
        "gazebo_boundary_dwell_ratio": mean("gazebo_live", "boundary_dwell_ratio"),
        "gazebo_goal_projection_after_filter_mean": mean("gazebo_live", "goal_projection_after_filter_mean"),
        "gazebo_tangential_velocity_kept_ratio": mean("gazebo_live", "tangential_velocity_kept_ratio"),
        "gazebo_filter_invasiveness": mean("gazebo_live", "filter_invasiveness"),
        "gazebo_line_to_goal_blocked_rate": mean("gazebo_live", "line_to_goal_blocked_rate"),
        "gazebo_line_to_goal_blocked_obstacle_rate": mean("gazebo_live", "line_to_goal_blocked_obstacle_rate"),
        "gazebo_line_to_goal_blocked_terrain_rate": mean("gazebo_live", "line_to_goal_blocked_terrain_rate"),
        "gazebo_line_to_goal_blocked_agent_rate": mean("gazebo_live", "line_to_goal_blocked_agent_rate"),
        "gazebo_line_to_goal_blocked_any_direct_rate": mean("gazebo_live", "line_to_goal_blocked_any_direct_rate"),
        "gazebo_single_laggard_finish_active_ratio": mean("gazebo_live", "single_laggard_finish_active_ratio"),
        "gazebo_finish_safety_conflict_rate": mean("gazebo_live", "finish_safety_conflict_rate"),
        "gazebo_candidate_pool_collapsed_rate": mean("gazebo_live", "candidate_pool_collapsed_rate"),
        "gazebo_accepted_candidate_count_mean": mean("gazebo_live", "accepted_candidate_count_mean"),
        "success_rate_gap": None if py_success is None or gz_success is None else float(gz_success - py_success),
        "collision_rate_gap": None if py_collision is None or gz_collision is None else float(gz_collision - py_collision),
        "reward_gap": None if py_reward is None or gz_reward is None else float(gz_reward - py_reward),
        "path_length_gap": None if py_path is None or gz_path is None else float(gz_path - py_path),
        "paired_scene_signature_mismatch_count": int(paired_scene_signature_mismatch_count),
        "paired_scene_signature_missing_count": int(paired_scene_signature_missing_count),
        "num_gazebo_only_failures": sum(1 for c in difference_cases if c.get("case_type") == "python_success_gazebo_fail"),
        "num_python_only_failures": sum(1 for c in difference_cases if c.get("case_type") == "python_fail_gazebo_success"),
        "num_sync_failures": sum(1 for c in difference_cases if c.get("case_type") == "gazebo_sync_failure"),
        "num_python_no_collision_gazebo_contact": sum(1 for c in difference_cases if c.get("case_type") == "python_no_collision_gazebo_contact"),
        "num_behavior_outcome_mismatches": sum(1 for c in difference_cases if c.get("case_type") == "behavior_outcome_mismatch"),
        "num_gazebo_early_done_mismatches": sum(1 for c in difference_cases if c.get("case_type") == "gazebo_early_done_mismatch"),
        "difference_case_count": len(difference_cases),
    }


def _trajectory_array(episode: Optional[Mapping[str, Any]]) -> Optional[np.ndarray]:
    if not isinstance(episode, Mapping):
        return None
    raw = episode.get("trajectory")
    if not raw:
        return None
    try:
        arr = np.asarray(raw, dtype=np.float64)
    except Exception:
        return None
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return None
    arr = arr[:, :, :3]
    try:
        expected_frames = int(episode.get("steps", 0) or 0) + 1
    except Exception:
        expected_frames = 0
    if expected_frames > 1 and arr.shape[0] != expected_frames and arr.shape[1] == expected_frames:
        arr = np.transpose(arr, (1, 0, 2))
    elif expected_frames <= 1 and arr.shape[0] <= 32 and arr.shape[1] > arr.shape[0]:
        arr = np.transpose(arr, (1, 0, 2))
    return arr


def _nan_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _nanmean_or_none(values: np.ndarray) -> Optional[float]:
    if values.size == 0 or not np.any(np.isfinite(values)):
        return None
    return _nan_float(np.nanmean(values))


def _nanmax_or_none(values: np.ndarray) -> Optional[float]:
    if values.size == 0 or not np.any(np.isfinite(values)):
        return None
    return _nan_float(np.nanmax(values))


def _trajectory_window_mean(delta: np.ndarray, frame_count: int) -> Optional[float]:
    frames = min(int(frame_count), int(delta.shape[0]))
    if frames <= 0:
        return None
    return _nanmean_or_none(delta[:frames])


def compute_paired_trajectory_agreement(results_by_backend: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    if "python_only" not in results_by_backend or "gazebo_live" not in results_by_backend:
        return {}
    py_eps = _episodes_by_id(results_by_backend.get("python_only", {}))
    gz_eps = _episodes_by_id(results_by_backend.get("gazebo_live", {}))
    episode_records: List[Dict[str, Any]] = []
    all_delta_parts: List[np.ndarray] = []
    final_delta_parts: List[np.ndarray] = []
    missing_count = 0

    for episode_id in sorted(set(py_eps) & set(gz_eps)):
        py_traj = _trajectory_array(py_eps.get(episode_id))
        gz_traj = _trajectory_array(gz_eps.get(episode_id))
        if py_traj is None or gz_traj is None:
            missing_count += 1
            continue
        frame_count = min(int(py_traj.shape[0]), int(gz_traj.shape[0]))
        agent_count = min(int(py_traj.shape[1]), int(gz_traj.shape[1]))
        if frame_count <= 0 or agent_count <= 0:
            missing_count += 1
            continue
        py_aligned = py_traj[:frame_count, :agent_count, :]
        gz_aligned = gz_traj[:frame_count, :agent_count, :]
        delta = np.linalg.norm(py_aligned - gz_aligned, axis=2)
        delta[~np.isfinite(delta)] = np.nan
        if not np.any(np.isfinite(delta)):
            missing_count += 1
            continue
        all_delta_parts.append(delta.reshape(-1))
        final_delta_parts.append(delta[-1].reshape(-1))
        record = {
            "episode": int(episode_id),
            "aligned_frames": int(frame_count),
            "aligned_agents": int(agent_count),
            "python_frames": int(py_traj.shape[0]),
            "gazebo_frames": int(gz_traj.shape[0]),
            "mean_delta": _nanmean_or_none(delta),
            "rmse_delta": _nan_float(np.sqrt(np.nanmean(delta * delta))) if np.any(np.isfinite(delta)) else None,
            "max_delta": _nanmax_or_none(delta),
            "final_mean_delta": _nanmean_or_none(delta[-1]),
            "final_max_delta": _nanmax_or_none(delta[-1]),
            "mean_delta_first_10_frames": _trajectory_window_mean(delta, 10),
            "mean_delta_first_50_frames": _trajectory_window_mean(delta, 50),
            "mean_delta_first_100_frames": _trajectory_window_mean(delta, 100),
            "mean_delta_first_500_frames": _trajectory_window_mean(delta, 500),
            "per_agent_mean_delta": [
                _nanmean_or_none(delta[:, agent_idx]) for agent_idx in range(agent_count)
            ],
            "per_agent_max_delta": [
                _nanmax_or_none(delta[:, agent_idx]) for agent_idx in range(agent_count)
            ],
        }
        episode_records.append(record)

    if all_delta_parts:
        all_delta = np.concatenate(all_delta_parts)
        final_delta = np.concatenate(final_delta_parts) if final_delta_parts else np.asarray([], dtype=np.float64)
    else:
        all_delta = np.asarray([], dtype=np.float64)
        final_delta = np.asarray([], dtype=np.float64)

    return {
        "paired_trajectory_episode_count": int(len(episode_records)),
        "paired_trajectory_missing_count": int(missing_count),
        "paired_trajectory_aligned_frames_total": int(sum(r.get("aligned_frames", 0) or 0 for r in episode_records)),
        "paired_trajectory_mean_delta": _nanmean_or_none(all_delta),
        "paired_trajectory_rmse_delta": (
            _nan_float(np.sqrt(np.nanmean(all_delta * all_delta)))
            if all_delta.size and np.any(np.isfinite(all_delta))
            else None
        ),
        "paired_trajectory_max_delta": _nanmax_or_none(all_delta),
        "paired_trajectory_final_mean_delta": _nanmean_or_none(final_delta),
        "paired_trajectory_final_max_delta": _nanmax_or_none(final_delta),
        "paired_trajectory_episodes": episode_records,
    }


def compute_paired_behavior_agreement(results_by_backend: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    if "python_only" not in results_by_backend or "gazebo_live" not in results_by_backend:
        return {}
    py_eps = _episodes_by_id(results_by_backend.get("python_only", {}))
    gz_eps = _episodes_by_id(results_by_backend.get("gazebo_live", {}))
    episode_records: List[Dict[str, Any]] = []
    missing_count = 0

    for episode_id in sorted(set(py_eps) | set(gz_eps)):
        py = py_eps.get(episode_id)
        gz = gz_eps.get(episode_id)
        if py is None or gz is None:
            missing_count += 1
            continue

        py_success = _episode_success_flag(py)
        gz_success = _episode_success_flag(gz)
        py_collision = _episode_collision_flag(py)
        gz_collision = _episode_collision_flag(gz)
        py_timeout = _episode_timeout_flag(py)
        gz_timeout = _episode_timeout_flag(gz)
        py_done_reason = _episode_done_reason(py)
        gz_done_reason = _episode_done_reason(gz)
        py_python_collision = _episode_python_collision_flag(py)
        gz_python_collision = _episode_python_collision_flag(gz)
        gz_contact = _episode_gazebo_contact_flag(gz)
        sync = build_sync_health_record(episode_data=gz, backend="gazebo_live", validation_root=None)

        success_agreement = int(py_success == gz_success)
        collision_agreement = int(py_collision == gz_collision)
        timeout_agreement = int(py_timeout == gz_timeout)
        done_reason_agreement = int(py_done_reason == gz_done_reason)
        outcome_agreement = int(success_agreement and collision_agreement and timeout_agreement)

        record = {
            "episode": int(episode_id),
            "python_success": py_success,
            "gazebo_success": gz_success,
            "success_agreement": success_agreement,
            "python_collision": py_collision,
            "gazebo_collision": gz_collision,
            "collision_agreement": collision_agreement,
            "python_collision_count": int(py.get("collision_count", 0) or 0),
            "gazebo_collision_count": int(gz.get("collision_count", 0) or 0),
            "gazebo_contact_count": int(gz.get("gazebo_contact_count", 0) or 0),
            "python_collision_vs_gazebo_contact_agreement": int(py_python_collision == gz_contact),
            "gazebo_python_collision_vs_contact_agreement": int(gz_python_collision == gz_contact),
            "python_timeout": py_timeout,
            "gazebo_timeout": gz_timeout,
            "timeout_agreement": timeout_agreement,
            "python_done_reason": py_done_reason,
            "gazebo_done_reason": gz_done_reason,
            "done_reason_agreement": done_reason_agreement,
            "outcome_agreement": outcome_agreement,
            "python_steps": int(py.get("steps", 0) or 0),
            "gazebo_steps": int(gz.get("steps", 0) or 0),
            "steps_abs_gap": _abs_gap(py.get("steps"), gz.get("steps")),
            "python_reward": _finite_float(py.get("reward")),
            "gazebo_reward": _finite_float(gz.get("reward")),
            "reward_abs_gap": _abs_gap(py.get("reward"), gz.get("reward")),
            "python_path_length": _finite_float(py.get("path_length")),
            "gazebo_path_length": _finite_float(gz.get("path_length")),
            "path_length_abs_gap": _abs_gap(py.get("path_length"), gz.get("path_length")),
            "path_length_relative_abs_gap": _relative_abs_gap(py.get("path_length"), gz.get("path_length")),
            "python_final_goal_distance": _finite_float(py.get("final_goal_distance")),
            "gazebo_final_goal_distance": _finite_float(gz.get("final_goal_distance")),
            "final_goal_distance_abs_gap": _abs_gap(py.get("final_goal_distance"), gz.get("final_goal_distance")),
            "final_goal_distance_relative_abs_gap": _relative_abs_gap(py.get("final_goal_distance"), gz.get("final_goal_distance")),
            "python_min_goal_distance": _finite_float(py.get("min_goal_distance")),
            "gazebo_min_goal_distance": _finite_float(gz.get("min_goal_distance")),
            "min_goal_distance_abs_gap": _abs_gap(py.get("min_goal_distance"), gz.get("min_goal_distance")),
            "gazebo_sync_ok": bool(sync.get("sync_ok", False)),
            "gazebo_live_consistency_mode": sync.get("gazebo_live_consistency_mode"),
            "gazebo_live_python_authoritative": bool(sync.get("gazebo_live_python_authoritative", False)),
            "gazebo_live_contact_authoritative": bool(sync.get("gazebo_live_contact_authoritative", True)),
            "gazebo_live_contact_marks_collision": bool(sync.get("gazebo_live_contact_marks_collision", True)),
            "gazebo_live_contact_terminates": bool(sync.get("gazebo_live_contact_terminates", False)),
            "gazebo_live_pose_correction": bool(sync.get("gazebo_live_pose_correction", False)),
        }
        episode_records.append(record)

    paired_count = len(episode_records)

    def bool_rate(field: str) -> Optional[float]:
        vals = [int(record.get(field, 0) or 0) for record in episode_records]
        return float(np.mean(vals)) if vals else None

    def gap_mean(field: str) -> Optional[float]:
        vals = _finite_values(record.get(field) for record in episode_records)
        return float(np.mean(vals)) if vals else None

    def gap_max(field: str) -> Optional[float]:
        vals = _finite_values(record.get(field) for record in episode_records)
        return float(max(vals)) if vals else None

    return {
        "paired_behavior_episode_count": int(paired_count),
        "paired_behavior_missing_count": int(missing_count),
        "paired_success_agreement_rate": bool_rate("success_agreement"),
        "paired_collision_agreement_rate": bool_rate("collision_agreement"),
        "paired_timeout_agreement_rate": bool_rate("timeout_agreement"),
        "paired_done_reason_agreement_rate": bool_rate("done_reason_agreement"),
        "paired_outcome_agreement_rate": bool_rate("outcome_agreement"),
        "paired_python_collision_vs_gazebo_contact_agreement_rate": bool_rate("python_collision_vs_gazebo_contact_agreement"),
        "paired_gazebo_collision_vs_contact_agreement_rate": bool_rate("gazebo_python_collision_vs_contact_agreement"),
        "paired_sync_ok_rate": bool_rate("gazebo_sync_ok"),
        "paired_steps_abs_gap_mean": gap_mean("steps_abs_gap"),
        "paired_steps_abs_gap_max": gap_max("steps_abs_gap"),
        "paired_reward_abs_gap_mean": gap_mean("reward_abs_gap"),
        "paired_reward_abs_gap_max": gap_max("reward_abs_gap"),
        "paired_path_length_abs_gap_mean": gap_mean("path_length_abs_gap"),
        "paired_path_length_abs_gap_max": gap_max("path_length_abs_gap"),
        "paired_path_length_relative_abs_gap_mean": gap_mean("path_length_relative_abs_gap"),
        "paired_final_goal_distance_abs_gap_mean": gap_mean("final_goal_distance_abs_gap"),
        "paired_final_goal_distance_abs_gap_max": gap_max("final_goal_distance_abs_gap"),
        "paired_final_goal_distance_relative_abs_gap_mean": gap_mean("final_goal_distance_relative_abs_gap"),
        "paired_min_goal_distance_abs_gap_mean": gap_mean("min_goal_distance_abs_gap"),
        "paired_behavior_collision_mismatch_count": int(sum(1 for record in episode_records if not record.get("collision_agreement"))),
        "paired_behavior_outcome_mismatch_count": int(sum(1 for record in episode_records if not record.get("outcome_agreement"))),
        "paired_behavior_episodes": episode_records,
    }


def extract_difference_cases(results_by_backend: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if "python_only" not in results_by_backend or "gazebo_live" not in results_by_backend:
        return []
    py_eps = _episodes_by_id(results_by_backend.get("python_only", {}))
    gz_eps = _episodes_by_id(results_by_backend.get("gazebo_live", {}))
    cases: List[Dict[str, Any]] = []
    for episode_id in sorted(set(py_eps) | set(gz_eps)):
        py = py_eps.get(episode_id)
        gz = gz_eps.get(episode_id)
        if py is None or gz is None:
            cases.append(_case_record("gazebo_sync_failure", episode_id, py, gz, reason="missing_backend_episode"))
            continue
        py_success = int(py.get("team_success", py.get("success", 0)) or 0)
        gz_success = int(gz.get("team_success", gz.get("success", 0)) or 0)
        py_collision = int(py.get("collision_count", 0) or 0)
        gz_contact = int(gz.get("gazebo_contact_count", 0) or 0)
        sync_record = gz.get("gazebo_live_sync_health") if isinstance(gz.get("gazebo_live_sync_health"), Mapping) else None
        if sync_record is None:
            sync_record = build_sync_health_record(episode_data=gz, backend="gazebo_live", validation_root=None)
        sync_failure = not bool(sync_record.get("sync_ok", False))
        if py_success == 1 and gz_success == 0:
            cases.append(_case_record("python_success_gazebo_fail", episode_id, py, gz))
        if py_collision == 0 and gz_contact > 0:
            cases.append(_case_record("python_no_collision_gazebo_contact", episode_id, py, gz))
        if py_success == 0 and gz_success == 1:
            cases.append(_case_record("python_fail_gazebo_success", episode_id, py, gz))
        py_timeout = _episode_timeout_flag(py)
        gz_timeout = _episode_timeout_flag(gz)
        py_done_reason = _episode_done_reason(py)
        gz_done_reason = _episode_done_reason(gz)
        if py_done_reason != gz_done_reason or py_timeout != gz_timeout:
            cases.append(
                _case_record(
                    "behavior_outcome_mismatch",
                    episode_id,
                    py,
                    gz,
                    reason=f"python_done={py_done_reason},gazebo_done={gz_done_reason}",
                )
            )
        try:
            py_steps = int(py.get("steps", 0) or 0)
            gz_steps = int(gz.get("steps", 0) or 0)
        except Exception:
            py_steps = gz_steps = 0
        if py_timeout == 1 and gz_timeout == 0 and gz_steps > 0 and py_steps > gz_steps:
            cases.append(
                _case_record(
                    "gazebo_early_done_mismatch",
                    episode_id,
                    py,
                    gz,
                    reason=f"python_steps={py_steps},gazebo_steps={gz_steps}",
                )
            )
        if sync_failure:
            cases.append(_case_record("gazebo_sync_failure", episode_id, py, gz))
    for idx, case in enumerate(cases):
        case["case_id"] = f"case_{idx:04d}_{case['case_type']}"
    return cases


def _case_record(case_type: str, episode_id: int, py: Optional[Mapping[str, Any]], gz: Optional[Mapping[str, Any]], reason: Optional[str] = None) -> Dict[str, Any]:
    source = gz or py or {}
    trace_tail = source.get("validation_trace_tail", []) if isinstance(source, Mapping) else []
    last = trace_tail[-1] if trace_tail else {}
    metrics = last.get("metrics", {}) if isinstance(last, Mapping) else {}
    return {
        "case_type": case_type,
        "episode": int(episode_id),
        "reason": reason,
        "seed": source.get("terrain_seed"),
        "terrain_seed": source.get("terrain_seed"),
        "terrain_variant_seed": source.get("terrain_variant_seed"),
        "obstacle_seed": source.get("obstacle_seed"),
        "python_scene_signature": source.get("python_scene_signature"),
        "agent_id": source.get("gazebo_contact_agent_indices", [None])[0] if source.get("gazebo_contact_agent_indices") else None,
        "failure_step": source.get("gazebo_contact_step") or source.get("steps"),
        "contact_pair": source.get("gazebo_contact_pairs", []),
        "python": _compact_episode(py),
        "gazebo": _compact_episode(gz),
        "last_20_actions": _tail_values(source, "actions"),
        "last_20_cmd_vel": _tail_values(source, "cmd_vel"),
        "last_20_python_pose": _tail_values(source, "python_pose"),
        "last_20_gazebo_pose": _tail_values(source, "gazebo_pose"),
        "goal_distance": metrics.get("goal_distance"),
        "terrain_clearance": metrics.get("terrain_clearance"),
        "nearest_obstacle_distance": metrics.get("nearest_obstacle_distance"),
    }


def _compact_episode(ep: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if ep is None:
        return None
    return {
        "success": int(ep.get("team_success", ep.get("success", 0)) or 0),
        "collision_count": int(ep.get("collision_count", 0) or 0),
        "gazebo_contact_count": int(ep.get("gazebo_contact_count", 0) or 0),
        "steps": int(ep.get("steps", 0) or 0),
        "reward": _finite_float(ep.get("reward")),
        "path_length": _finite_float(ep.get("path_length")),
        "final_goal_distance": _finite_float(ep.get("final_goal_distance")),
        "sync_active": bool(ep.get("gazebo_live_sync_active", False)),
        "sync_error": ep.get("gazebo_live_sync_error"),
    }


def _tail_values(ep: Mapping[str, Any], field: str) -> List[Any]:
    return [frame.get(field) for frame in (ep.get("validation_trace_tail", []) or []) if isinstance(frame, Mapping)]


def write_difference_cases(
    *,
    cases: Sequence[Mapping[str, Any]],
    results_by_backend: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    py_eps = _episodes_by_id(results_by_backend.get("python_only", {}))
    gz_eps = _episodes_by_id(results_by_backend.get("gazebo_live", {}))
    for case in cases:
        case_id = str(case.get("case_id", f"case_{len(written):04d}"))
        case_dir = output_dir / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(case)
        episode_id = int(case.get("episode", -1))
        py = py_eps.get(episode_id)
        gz = gz_eps.get(episode_id)
        payload["artifacts"] = export_case_visuals(case_dir=case_dir, python_episode=py, gazebo_episode=gz)
        case_json = case_dir / "case.json"
        case_json.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        payload["case_json"] = str(case_json)
        written.append(payload)
    (output_dir / "difference_cases.json").write_text(
        json.dumps(json_safe(written), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return written


def _snapshot_namespace_from_episode(ep: Optional[Mapping[str, Any]]) -> Optional[SimpleNamespace]:
    if not ep:
        return None
    scenario_json = ep.get("gazebo_live_scenario_json")
    if not scenario_json:
        return None
    path = Path(str(scenario_json))
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    terrain = None
    dense_path = data.get("terrain", {}).get("dense_npy")
    if dense_path and Path(dense_path).exists():
        try:
            terrain = np.load(dense_path)
        except Exception:
            terrain = None
    return SimpleNamespace(
        terrain=terrain,
        map_size=data.get("map_size"),
        obstacles=data.get("obstacles", []),
        goal_pos=(data.get("goal") or {}).get("position"),
        agent_goals=[g.get("position") for g in data.get("agent_goals", []) if isinstance(g, Mapping)],
    )


def export_case_visuals(
    *,
    case_dir: Path,
    python_episode: Optional[Mapping[str, Any]],
    gazebo_episode: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    artifacts: Dict[str, Any] = {}
    try:
        from visualization.trajectory_visualizer import TrajectoryVisualizer
    except Exception as exc:
        return {"error": f"TrajectoryVisualizer import failed: {exc}"}

    scenario = _snapshot_namespace_from_episode(gazebo_episode) or _snapshot_namespace_from_episode(python_episode)
    goal_positions = None
    if scenario is not None:
        goal_positions = {
            "goal_pos": getattr(scenario, "goal_pos", None),
            "agent_goals": getattr(scenario, "agent_goals", []),
        }
    vis = TrajectoryVisualizer(verbose=False)
    for backend, ep in (("python", python_episode), ("gazebo", gazebo_episode)):
        traj = ep.get("trajectory", []) if isinstance(ep, Mapping) else []
        if not traj:
            continue
        html_path = case_dir / f"{backend}_trajectory_interactive.html"
        try:
            ok = vis.generate_trajectory_interactive(
                trajectories=traj,
                save_path=str(html_path),
                title=f"{backend} trajectory difference case",
                goal_positions=goal_positions,
                scenario=scenario,
                env_instance=None,
            )
            if ok:
                artifacts[f"{backend}_trajectory_html"] = str(html_path)
        except Exception as exc:
            artifacts[f"{backend}_trajectory_html_error"] = str(exc)

    overlay_path = case_dir / "overlay_xy.png"
    try:
        _write_overlay_png(overlay_path, python_episode, gazebo_episode)
        artifacts["overlay_png"] = str(overlay_path)
    except Exception as exc:
        artifacts["overlay_error"] = str(exc)
    return artifacts


def _write_overlay_png(path: Path, python_episode: Optional[Mapping[str, Any]], gazebo_episode: Optional[Mapping[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))
    for label, ep, style in (
        ("python", python_episode, "-"),
        ("gazebo", gazebo_episode, "--"),
    ):
        traj = ep.get("trajectory", []) if isinstance(ep, Mapping) else []
        if not traj:
            continue
        arr = np.asarray(traj, dtype=np.float64)
        if arr.ndim != 3 or arr.shape[-1] < 2:
            continue
        for agent_idx in range(arr.shape[1]):
            ax.plot(arr[:, agent_idx, 0], arr[:, agent_idx, 1], linestyle=style, linewidth=1.8, label=f"{label}_agent{agent_idx}")
            ax.scatter(arr[0, agent_idx, 0], arr[0, agent_idx, 1], s=18)
            ax.scatter(arr[-1, agent_idx, 0], arr[-1, agent_idx, 1], s=28, marker="x")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Python-only vs Gazebo-live XY Overlay")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_validation_outputs(
    *,
    results_by_backend: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = write_episode_metrics_csv(results_by_backend, output_dir / "episode_metrics.csv")
    cases = extract_difference_cases(results_by_backend)
    written_cases = write_difference_cases(cases=cases, results_by_backend=results_by_backend, output_dir=output_dir)
    summary = build_validation_summary(rows, written_cases)
    summary.update(compute_paired_behavior_agreement(results_by_backend))
    summary.update(compute_paired_trajectory_agreement(results_by_backend))
    summary["episode_metrics_csv"] = str(output_dir / "episode_metrics.csv")
    summary["difference_cases_json"] = str(output_dir / "difference_cases.json")
    summary["scene_check_logs_jsonl"] = str(output_dir / "scene_check_logs.jsonl")
    summary["sync_health_logs_jsonl"] = str(output_dir / "sync_health_logs.jsonl")
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "rows": rows,
        "difference_cases": written_cases,
        "summary": summary,
    }


def copy_backend_result(result_path: Path, destination_dir: Path) -> Optional[str]:
    result_path = Path(result_path)
    if not result_path.exists():
        return None
    destination_dir.mkdir(parents=True, exist_ok=True)
    dst = destination_dir / "evaluation_results.json"
    shutil.copyfile(result_path, dst)
    return str(dst)
