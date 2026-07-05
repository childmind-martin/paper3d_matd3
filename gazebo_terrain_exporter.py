#!/usr/bin/env python3
"""Generate Gazebo model/world files from a MATD3 scenario snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


UAV_MARKER_BASE_RADIUS = 2.22


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(value: float) -> str:
    return f"{float(value):.8g}"


def _vec(values: Iterable[float]) -> str:
    return " ".join(_fmt(float(v)) for v in values)


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def _write_xml(root: ET.Element, path: Path) -> None:
    _indent_xml(root)
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


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


def _hash_array_into(h: "hashlib._Hash", name: str, array: np.ndarray) -> None:
    arr = np.ascontiguousarray(array)
    h.update(name.encode("utf-8"))
    h.update(b":")
    h.update(str(arr.shape).encode("utf-8"))
    h.update(b":")
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(b":")
    h.update(arr.tobytes(order="C"))


def _terrain_mesh_cache_key(xs: np.ndarray, ys: np.ndarray, z: np.ndarray, extra: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(b"matd3_terrain_obj_v3_with_mtl")
    _hash_array_into(h, "xs", xs)
    _hash_array_into(h, "ys", ys)
    _hash_array_into(h, "z", z)
    h.update(json.dumps(_json_safe(extra), sort_keys=True, ensure_ascii=False).encode("utf-8"))
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
    with _sidecar_path(path).open("w", encoding="utf-8") as f:
        json.dump(_json_safe(meta), f, ensure_ascii=False, indent=2)


def _write_obj_mtl(path: Path, material_name: str, rgba: Sequence[float]) -> Path:
    rgba_arr = np.asarray(rgba, dtype=np.float64).reshape(-1)
    if rgba_arr.size < 4:
        rgba_arr = np.pad(rgba_arr, (0, 4 - rgba_arr.size), constant_values=1.0)
    rgba_arr = np.clip(rgba_arr[:4], 0.0, 1.0)
    mtl_path = path.with_suffix(".mtl")
    with mtl_path.open("w", encoding="utf-8") as f:
        f.write("# MATD3 Gazebo OBJ material\n")
        f.write(f"newmtl {material_name}\n")
        f.write(f"Ka {_fmt(rgba_arr[0])} {_fmt(rgba_arr[1])} {_fmt(rgba_arr[2])}\n")
        f.write(f"Kd {_fmt(rgba_arr[0])} {_fmt(rgba_arr[1])} {_fmt(rgba_arr[2])}\n")
        f.write("Ks 0.08 0.08 0.08\n")
        f.write("Ns 24\n")
        f.write(f"d {_fmt(rgba_arr[3])}\n")
        f.write("illum 2\n")
    return mtl_path


def _terrain_grid_from_snapshot(snapshot: Dict[str, Any], resolution: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    terrain = snapshot["terrain"]
    dense_path = Path(terrain["dense_npy"])
    z_dense = np.load(dense_path).astype(np.float64)
    dense_x = np.asarray(terrain["dense_coordinates"]["x"], dtype=np.float64)
    dense_y = np.asarray(terrain["dense_coordinates"]["y"], dtype=np.float64)
    if resolution is None or int(resolution) == z_dense.shape[0]:
        return dense_x, dense_y, z_dense

    resolution = int(resolution)
    if resolution < 2:
        raise ValueError("OBJ resolution must be at least 2")
    domain = snapshot["coordinate_domain"]
    xs = np.linspace(float(domain["x_min"]), float(domain["x_max"]), resolution, dtype=np.float64)
    ys = np.linspace(float(domain["y_min"]), float(domain["y_max"]), resolution, dtype=np.float64)
    z = np.zeros((resolution, resolution), dtype=np.float64)
    for yi, y in enumerate(ys):
        z[yi, :] = _bilinear_grid_height(dense_x, dense_y, z_dense, xs, y)
    return xs, ys, z


def _bilinear_grid_height(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    z_grid: np.ndarray,
    xs: np.ndarray,
    y: float,
) -> np.ndarray:
    """Bilinear interpolation over an already dense exported grid."""
    ys = np.full_like(xs, float(y), dtype=np.float64)
    x_idx = np.interp(xs, grid_x, np.arange(grid_x.size, dtype=np.float64))
    y_idx = np.interp(ys, grid_y, np.arange(grid_y.size, dtype=np.float64))
    x0 = np.floor(x_idx).astype(int)
    x1 = np.clip(x0 + 1, 0, grid_x.size - 1)
    y0 = np.floor(y_idx).astype(int)
    y1 = np.clip(y0 + 1, 0, grid_y.size - 1)
    x0 = np.clip(x0, 0, grid_x.size - 1)
    y0 = np.clip(y0, 0, grid_y.size - 1)
    wx = x_idx - x0
    wy = y_idx - y0
    z00 = z_grid[y0, x0]
    z10 = z_grid[y0, x1]
    z01 = z_grid[y1, x0]
    z11 = z_grid[y1, x1]
    return (1.0 - wx) * (1.0 - wy) * z00 + wx * (1.0 - wy) * z10 + (1.0 - wx) * wy * z01 + wx * wy * z11


def write_terrain_obj(
    path: Path,
    xs: np.ndarray,
    ys: np.ndarray,
    z: np.ndarray,
    reuse_cache: bool = True,
    cache_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = z.shape
    if width != xs.size or height != ys.size:
        raise ValueError(f"terrain grid shape mismatch: z={z.shape}, xs={xs.size}, ys={ys.size}")
    cache_key = _terrain_mesh_cache_key(xs, ys, z, cache_extra or {})
    meta = {
        "path": str(path),
        "sidecar_meta": str(_sidecar_path(path)),
        "cache_key": cache_key,
        "material_path": str(path.with_suffix(".mtl")),
        "material_name": "matd3_terrain_material",
        "grid_width": int(width),
        "grid_height": int(height),
        "vertex_count": int(width * height),
        "face_count": int((width - 1) * (height - 1) * 2),
        "x_min": float(xs[0]),
        "x_max": float(xs[-1]),
        "y_min": float(ys[0]),
        "y_max": float(ys[-1]),
        "z_min": float(np.min(z)),
        "z_max": float(np.max(z)),
        "cache_hit": False,
    }
    if reuse_cache and path.exists():
        old_meta = _load_sidecar(path)
        if old_meta and old_meta.get("cache_key") == cache_key:
            cached = dict(old_meta)
            cached["path"] = str(path)
            cached["sidecar_meta"] = str(_sidecar_path(path))
            cached["cache_hit"] = True
            return cached
    dz_dy, dz_dx = np.gradient(z, ys, xs, edge_order=1)
    normals = np.dstack((-dz_dx, -dz_dy, np.ones_like(z, dtype=np.float64)))
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    norm[norm == 0.0] = 1.0
    normals = normals / norm
    material_name = "matd3_terrain_material"
    material_path = _write_obj_mtl(path, material_name, (0.35, 0.58, 0.28, 1.0))
    with path.open("w", encoding="utf-8") as f:
        f.write("# MATD3 terrain mesh generated from exported scenario.get_terrain_height samples\n")
        f.write(f"# grid_width {width}\n")
        f.write(f"# grid_height {height}\n")
        f.write(f"mtllib {material_path.name}\n")
        f.write("o matd3_terrain\n")
        f.write(f"usemtl {material_name}\n")
        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                f.write(f"v {_fmt(x)} {_fmt(y)} {_fmt(z[yi, xi])}\n")
        for yi in range(height):
            for xi in range(width):
                nx, ny, nz = normals[yi, xi]
                f.write(f"vn {_fmt(nx)} {_fmt(ny)} {_fmt(nz)}\n")
        for yi in range(height - 1):
            row = yi * width
            next_row = (yi + 1) * width
            for xi in range(width - 1):
                v00 = row + xi + 1
                v10 = row + xi + 2
                v01 = next_row + xi + 1
                v11 = next_row + xi + 2
                f.write(f"f {v00}//{v00} {v10}//{v10} {v11}//{v11}\n")
                f.write(f"f {v00}//{v00} {v11}//{v11} {v01}//{v01}\n")
    _write_sidecar(path, {k: v for k, v in meta.items() if k != "cache_hit"})
    return meta


def _add_material(parent: ET.Element, rgba: Sequence[float]) -> None:
    mat = ET.SubElement(parent, "material")
    ET.SubElement(mat, "ambient").text = _vec(rgba)
    ET.SubElement(mat, "diffuse").text = _vec(rgba)
    ET.SubElement(mat, "specular").text = "0.15 0.15 0.15 1"


def _mesh_geometry(parent: ET.Element, uri: str) -> None:
    geometry = ET.SubElement(parent, "geometry")
    mesh = ET.SubElement(geometry, "mesh")
    ET.SubElement(mesh, "uri").text = uri


def write_model_config(model_dir: Path) -> None:
    root = ET.Element("model")
    ET.SubElement(root, "name").text = "matd3_terrain"
    ET.SubElement(root, "version").text = "1.0"
    sdf = ET.SubElement(root, "sdf", {"version": "1.10"})
    sdf.text = "model.sdf"
    author = ET.SubElement(root, "author")
    ET.SubElement(author, "name").text = "MATD3 scenario exporter"
    ET.SubElement(author, "email").text = "n/a"
    ET.SubElement(root, "description").text = "MATD3 terrain mesh exported from scenario.get_terrain_height."
    _write_xml(root, model_dir / "model.config")


def write_uav_marker_config(model_dir: Path, model_name: str) -> None:
    root = ET.Element("model")
    ET.SubElement(root, "name").text = model_name
    ET.SubElement(root, "version").text = "1.0"
    sdf = ET.SubElement(root, "sdf", {"version": "1.10"})
    sdf.text = "model.sdf"
    author = ET.SubElement(root, "author")
    ET.SubElement(author, "name").text = "MATD3 scenario exporter"
    ET.SubElement(author, "email").text = "n/a"
    ET.SubElement(root, "description").text = "Static visual-only UAV marker for MATD3 start poses."
    _write_xml(root, model_dir / "model.config")


def write_model_sdf(
    model_dir: Path,
    collision_uri: str = "model://matd3_terrain/meshes/terrain.obj",
    enable_collision: bool = True,
) -> None:
    root = ET.Element("sdf", {"version": "1.10"})
    model = ET.SubElement(root, "model", {"name": "matd3_terrain"})
    ET.SubElement(model, "static").text = "true"
    link = ET.SubElement(model, "link", {"name": "terrain_link"})
    visual = ET.SubElement(link, "visual", {"name": "terrain_visual"})
    _mesh_geometry(visual, "model://matd3_terrain/meshes/terrain.obj")
    _add_material(visual, (0.42, 0.55, 0.34, 1.0))
    if enable_collision:
        collision = ET.SubElement(link, "collision", {"name": "terrain_collision"})
        _mesh_geometry(collision, collision_uri)
        surface = ET.SubElement(collision, "surface")
        friction = ET.SubElement(surface, "friction")
        ET.SubElement(friction, "ode")
    _write_xml(root, model_dir / "model.sdf")


def _box_visual(link: ET.Element, name: str, pose: Sequence[float], size: Sequence[float], rgba: Sequence[float]) -> None:
    visual = ET.SubElement(link, "visual", {"name": name})
    ET.SubElement(visual, "pose").text = _vec(pose)
    geom = ET.SubElement(visual, "geometry")
    box = ET.SubElement(geom, "box")
    ET.SubElement(box, "size").text = _vec(size)
    _add_material(visual, rgba)


def _cylinder_visual(link: ET.Element, name: str, pose: Sequence[float], radius: float, length: float, rgba: Sequence[float]) -> None:
    visual = ET.SubElement(link, "visual", {"name": name})
    ET.SubElement(visual, "pose").text = _vec(pose)
    geom = ET.SubElement(visual, "geometry")
    cylinder = ET.SubElement(geom, "cylinder")
    ET.SubElement(cylinder, "radius").text = _fmt(radius)
    ET.SubElement(cylinder, "length").text = _fmt(length)
    _add_material(visual, rgba)


def _sphere_visual(link: ET.Element, name: str, pose: Sequence[float], radius: float, rgba: Sequence[float]) -> None:
    visual = ET.SubElement(link, "visual", {"name": name})
    ET.SubElement(visual, "pose").text = _vec(pose)
    geom = ET.SubElement(visual, "geometry")
    sphere = ET.SubElement(geom, "sphere")
    ET.SubElement(sphere, "radius").text = _fmt(radius)
    _add_material(visual, rgba)


def _scaled_pose(pose: Sequence[float], scale: float) -> Tuple[float, float, float, float, float, float]:
    vals = list(pose)
    while len(vals) < 6:
        vals.append(0.0)
    return (
        float(vals[0]) * scale,
        float(vals[1]) * scale,
        float(vals[2]) * scale,
        float(vals[3]),
        float(vals[4]),
        float(vals[5]),
    )


def _scaled_size(size: Sequence[float], scale: float) -> Tuple[float, float, float]:
    return tuple(float(v) * scale for v in size[:3])


def write_uav_marker_sdf(
    model_dir: Path,
    model_name: str,
    accent_rgba: Sequence[float],
    visual_scale: float = 1.0,
    collision_envelope_radius: float = 0.0,
) -> None:
    """Write a static, visual-only quadrotor marker model.

    The marker deliberately has no collision elements, so replacing the old
    start spheres changes only Gazebo appearance and not the exported geometry
    used by terrain / obstacle validation.
    """
    root = ET.Element("sdf", {"version": "1.10"})
    model = ET.SubElement(root, "model", {"name": model_name})
    ET.SubElement(model, "static").text = "true"
    link = ET.SubElement(model, "link", {"name": "uav_marker_link"})

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
    _write_xml(root, model_dir / "model.sdf")


def write_uav_marker_model(
    model_parent_dir: Path,
    model_name: str,
    accent_rgba: Sequence[float],
    visual_scale: float,
    visual_radius: float,
    collision_envelope_radius: float,
) -> Dict[str, Any]:
    model_dir = model_parent_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    write_uav_marker_config(model_dir, model_name)
    write_uav_marker_sdf(
        model_dir,
        model_name,
        accent_rgba,
        visual_scale=visual_scale,
        collision_envelope_radius=collision_envelope_radius,
    )
    return {
        "model_name": model_name,
        "model_dir": str(model_dir),
        "uri": f"model://{model_name}",
        "visual_only": True,
        "visual_scale": float(visual_scale),
        "visual_radius": float(visual_radius),
        "visual_radius_source": "start_positions[].agent_size",
        "collision_envelope_radius": float(collision_envelope_radius),
    }


def _add_sphere_model(world: ET.Element, name: str, center: Sequence[float], radius: float, rgba: Sequence[float], collision: bool = True) -> None:
    model = ET.SubElement(world, "model", {"name": name})
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = f"{_vec(center)} 0 0 0"
    link = ET.SubElement(model, "link", {"name": "link"})
    visual = ET.SubElement(link, "visual", {"name": "visual"})
    geom = ET.SubElement(visual, "geometry")
    sphere = ET.SubElement(geom, "sphere")
    ET.SubElement(sphere, "radius").text = _fmt(radius)
    _add_material(visual, rgba)
    if collision:
        col = ET.SubElement(link, "collision", {"name": "collision"})
        geom = ET.SubElement(col, "geometry")
        sphere = ET.SubElement(geom, "sphere")
        ET.SubElement(sphere, "radius").text = _fmt(radius)


def _add_box_model(
    world: ET.Element,
    name: str,
    center: Sequence[float],
    size: Sequence[float],
    rgba: Sequence[float],
    collision: bool = True,
) -> None:
    model = ET.SubElement(world, "model", {"name": name})
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = f"{_vec(center)} 0 0 0"
    link = ET.SubElement(model, "link", {"name": "link"})
    for tag in ("visual", "collision") if collision else ("visual",):
        elem = ET.SubElement(link, tag, {"name": tag})
        geom = ET.SubElement(elem, "geometry")
        box = ET.SubElement(geom, "box")
        ET.SubElement(box, "size").text = _vec(size)
        if tag == "visual":
            _add_material(elem, rgba)


def _add_cylinder_model(
    world: ET.Element,
    name: str,
    center: Sequence[float],
    radius: float,
    length: float,
    rgba: Sequence[float],
    collision: bool = True,
) -> None:
    model = ET.SubElement(world, "model", {"name": name})
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = f"{_vec(center)} 0 0 0"
    link = ET.SubElement(model, "link", {"name": "link"})
    for tag in ("visual", "collision") if collision else ("visual",):
        elem = ET.SubElement(link, tag, {"name": tag})
        geom = ET.SubElement(elem, "geometry")
        cyl = ET.SubElement(geom, "cylinder")
        ET.SubElement(cyl, "radius").text = _fmt(radius)
        ET.SubElement(cyl, "length").text = _fmt(length)
        if tag == "visual":
            _add_material(elem, rgba)


def _yaw_from_start_to_goal(start: Sequence[float], goal: Optional[Sequence[float]]) -> float:
    if goal is None:
        return 0.0
    dx = float(goal[0]) - float(start[0])
    dy = float(goal[1]) - float(start[1])
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    return float(math.atan2(dy, dx))


def _add_uav_marker_include(world: ET.Element, name: str, position: Sequence[float], model_uri: str, yaw: float) -> None:
    include = ET.SubElement(world, "include")
    ET.SubElement(include, "uri").text = model_uri
    ET.SubElement(include, "name").text = name
    ET.SubElement(include, "pose").text = f"{_vec(position)} 0 0 {_fmt(yaw)}"


def _agent_radius_from_start(start: Dict[str, Any], fallback: float = 0.5) -> float:
    try:
        radius = float(start.get("agent_size", fallback))
    except Exception:
        radius = float(fallback)
    return max(0.001, radius)


def write_world_sdf(snapshot: Dict[str, Any], world_path: Path, obstacle_collision: bool = True) -> None:
    root = ET.Element("sdf", {"version": "1.10"})
    world = ET.SubElement(root, "world", {"name": "matd3_static_scene"})

    physics = ET.SubElement(world, "physics", {"type": "ode"})
    ET.SubElement(physics, "max_step_size").text = "0.004"
    ET.SubElement(physics, "real_time_factor").text = "1.0"
    ET.SubElement(physics, "real_time_update_rate").text = "250"
    ET.SubElement(world, "plugin", {"name": "gz::sim::systems::Physics", "filename": "gz-sim-physics-system"})
    ET.SubElement(world, "plugin", {"name": "gz::sim::systems::UserCommands", "filename": "gz-sim-user-commands-system"})
    ET.SubElement(world, "plugin", {"name": "gz::sim::systems::SceneBroadcaster", "filename": "gz-sim-scene-broadcaster-system"})
    ET.SubElement(world, "plugin", {"name": "gz::sim::systems::Contact", "filename": "gz-sim-contact-system"})
    sensors = ET.SubElement(world, "plugin", {"name": "gz::sim::systems::Sensors", "filename": "gz-sim-sensors-system"})
    ET.SubElement(sensors, "render_engine").text = "ogre2"
    ET.SubElement(world, "gravity").text = "0 0 -9.8"
    scene = ET.SubElement(world, "scene")
    ET.SubElement(scene, "grid").text = "false"
    ET.SubElement(scene, "ambient").text = "0.45 0.45 0.45 1"
    ET.SubElement(scene, "background").text = "0.72 0.76 0.8 1"
    ET.SubElement(scene, "shadows").text = "true"

    gui = ET.SubElement(world, "gui", {"fullscreen": "false"})
    view = ET.SubElement(gui, "plugin", {"name": "3D View", "filename": "GzScene3D"})
    ET.SubElement(view, "engine").text = "ogre2"
    ET.SubElement(view, "scene").text = "scene"
    ET.SubElement(view, "ambient_light").text = "0.6 0.6 0.6"
    ET.SubElement(view, "background_color").text = "0.72 0.76 0.8"
    ET.SubElement(view, "camera_pose").text = "250 -260 180 0 0.7 2.35"
    ET.SubElement(gui, "plugin", {"name": "World control", "filename": "WorldControl"})
    ET.SubElement(gui, "plugin", {"name": "World stats", "filename": "WorldStats"})
    ET.SubElement(gui, "plugin", {"name": "Entity tree", "filename": "EntityTree"})

    light = ET.SubElement(world, "light", {"name": "sun", "type": "directional"})
    ET.SubElement(light, "pose").text = "0 0 500 0 0 0"
    ET.SubElement(light, "cast_shadows").text = "true"
    ET.SubElement(light, "direction").text = "-0.35 0.45 -0.82"
    ET.SubElement(light, "diffuse").text = "0.95 0.95 0.9 1"
    ET.SubElement(light, "specular").text = "0.25 0.25 0.25 1"

    include = ET.SubElement(world, "include")
    ET.SubElement(include, "uri").text = "model://matd3_terrain"
    ET.SubElement(include, "name").text = "matd3_terrain"

    for idx, obstacle in enumerate(snapshot.get("obstacles", [])):
        obs_type = obstacle.get("type")
        name = obstacle.get("name", f"obstacle_{idx}")
        if obs_type == "sphere":
            _add_sphere_model(
                world,
                name,
                obstacle["center"],
                float(obstacle["radius"]),
                (1.0, 0.05, 0.02, 0.55),
                collision=obstacle_collision,
            )
        elif obs_type == "box":
            _add_box_model(
                world,
                name,
                obstacle["center"],
                obstacle["size"],
                (1.0, 0.05, 0.02, 0.55),
                collision=obstacle_collision,
            )
        elif obs_type == "cylinder":
            _add_cylinder_model(
                world,
                name,
                obstacle["center"],
                float(obstacle["radius"]),
                float(obstacle["length"]),
                (1.0, 0.05, 0.02, 0.55),
                collision=obstacle_collision,
            )
        else:
            raise ValueError(f"unsupported obstacle type in snapshot: {obs_type}")

    start_colors = [
        (0.0, 0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 0.7, 0.2, 1.0),
    ]
    for idx, start in enumerate(snapshot.get("start_positions", [])):
        agent_goal = None
        if idx < len(snapshot.get("agent_goals", [])):
            agent_goal = snapshot["agent_goals"][idx].get("position")
        marker_models = snapshot.get("gazebo", {}).get("uav_marker_models", [])
        if idx < len(marker_models):
            yaw = _yaw_from_start_to_goal(start["position"], agent_goal)
            _add_uav_marker_include(
                world,
                f"start_agent_{idx}",
                start["position"],
                marker_models[idx]["uri"],
                yaw,
            )
        else:
            try:
                marker_radius = _agent_radius_from_start(
                    start,
                    float(snapshot.get("collision", {}).get("collision_distance_threshold", 0.5)),
                )
            except Exception:
                marker_radius = _agent_radius_from_start(start, 0.5)
            _add_sphere_model(
                world,
                f"start_agent_{idx}",
                start["position"],
                marker_radius,
                start_colors[idx % len(start_colors)],
                collision=False,
            )

    if snapshot.get("goal") and snapshot["goal"].get("position") is not None:
        _add_sphere_model(world, "goal_center", snapshot["goal"]["position"], 2.2, (1.0, 0.92, 0.0, 1.0), collision=False)
    for idx, goal in enumerate(snapshot.get("agent_goals", [])):
        _add_sphere_model(world, f"goal_agent_{idx}", goal["position"], 1.7, start_colors[idx % len(start_colors)], collision=False)

    _write_xml(root, world_path)


def export_gazebo_scene(
    scenario_json: Path,
    output_dir: Optional[Path] = None,
    visual_resolution: Optional[int] = None,
    coarse_collision_resolution: int = 80,
    use_coarse_collision: bool = False,
    uav_marker_scale: Optional[float] = None,
    show_agent_collision_envelope: bool = True,
    terrain_collision: bool = True,
    obstacle_collision: bool = True,
    reuse_cache: bool = True,
) -> Dict[str, Any]:
    scenario_json = scenario_json.resolve()
    snapshot = _load_json(scenario_json)
    base_dir = scenario_json.parent
    if output_dir is None:
        output_dir = base_dir
    output_dir = output_dir.resolve()
    model_dir = output_dir / "models" / "matd3_terrain"
    mesh_dir = model_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    xs, ys, z = _terrain_grid_from_snapshot(snapshot, resolution=visual_resolution)
    visual_meta = write_terrain_obj(
        mesh_dir / "terrain.obj",
        xs,
        ys,
        z,
        reuse_cache=reuse_cache,
        cache_extra={
            "mesh_role": "visual",
            "visual_resolution": int(xs.size),
            "coordinate_domain": snapshot.get("coordinate_domain", {}),
            "terrain_dense_sha256": snapshot.get("terrain", {}).get("dense_sha256"),
        },
    )

    coarse_resolution = int(coarse_collision_resolution)
    coarse_meta = None
    collision_uri = "model://matd3_terrain/meshes/terrain.obj"
    if coarse_resolution >= 2:
        cxs, cys, cz = _terrain_grid_from_snapshot(snapshot, resolution=coarse_resolution)
        coarse_meta = write_terrain_obj(
            mesh_dir / "terrain_collision_coarse.obj",
            cxs,
            cys,
            cz,
            reuse_cache=reuse_cache,
            cache_extra={
                "mesh_role": "coarse_collision",
                "coarse_collision_resolution": int(coarse_resolution),
                "coordinate_domain": snapshot.get("coordinate_domain", {}),
                "terrain_dense_sha256": snapshot.get("terrain", {}).get("dense_sha256"),
            },
        )
        if use_coarse_collision:
            collision_uri = "model://matd3_terrain/meshes/terrain_collision_coarse.obj"

    write_model_config(model_dir)
    write_model_sdf(model_dir, collision_uri=collision_uri, enable_collision=terrain_collision)

    start_colors = [
        (0.0, 0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 0.7, 0.2, 1.0),
    ]
    uav_marker_models = []
    explicit_visual_scale = uav_marker_scale is not None
    for idx, start in enumerate(snapshot.get("start_positions", [])):
        collision_radius = _agent_radius_from_start(start, fallback=0.5)
        if not show_agent_collision_envelope:
            collision_radius = 0.0
        visual_radius = _agent_radius_from_start(start, fallback=max(collision_radius, 0.5))
        visual_scale = float(uav_marker_scale) if explicit_visual_scale else visual_radius / UAV_MARKER_BASE_RADIUS
        uav_marker_models.append(
            write_uav_marker_model(
                output_dir / "models",
                f"matd3_uav_marker_agent_{idx}",
                start_colors[idx % len(start_colors)],
                visual_scale=visual_scale,
                visual_radius=visual_radius,
                collision_envelope_radius=collision_radius,
            )
        )
    snapshot.setdefault("gazebo", {})
    snapshot["gazebo"]["uav_marker_models"] = uav_marker_models
    snapshot["gazebo"]["terrain_collision_enabled"] = bool(terrain_collision)
    snapshot["gazebo"]["obstacle_collision_enabled"] = bool(obstacle_collision)

    world_path = output_dir / "world.sdf"
    write_world_sdf(snapshot, world_path, obstacle_collision=obstacle_collision)

    snapshot["gazebo"].update(
        {
            "world_sdf": str(world_path),
            "model_parent_dir": str(output_dir / "models"),
            "model_name": "matd3_terrain",
            "start_marker_mode": "static_uav_visual",
            "uav_marker_base_radius": UAV_MARKER_BASE_RADIUS,
            "uav_marker_scale": float(uav_marker_scale) if explicit_visual_scale else None,
            "uav_marker_scale_source": "cli_override" if explicit_visual_scale else "start_positions[].agent_size",
            "agent_collision_envelope": {
                "visible": bool(show_agent_collision_envelope),
                "radius": float(collision_radius) if show_agent_collision_envelope else 0.0,
                "source": "start_positions[].agent_size",
                "visual_only": True,
            },
            "visual_mesh": visual_meta,
            "coarse_collision_mesh": coarse_meta,
            "collision_mesh_uri": collision_uri,
            "terrain_collision_enabled": bool(terrain_collision),
            "obstacle_collision_enabled": bool(obstacle_collision),
            "cache": {
                "reuse_cache": bool(reuse_cache),
                "visual_mesh_cache_hit": bool(visual_meta.get("cache_hit", False)),
                "coarse_collision_mesh_cache_hit": bool(coarse_meta.get("cache_hit", False)) if coarse_meta else None,
            },
            "resource_path_command": f"export GZ_SIM_RESOURCE_PATH={output_dir / 'models'}:$GZ_SIM_RESOURCE_PATH",
        }
    )
    with scenario_json.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    return snapshot["gazebo"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Gazebo OBJ/model/world from scenario.json.")
    parser.add_argument("--scenario-json", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--visual-resolution", type=int, default=None, help="Terrain OBJ grid resolution; defaults to dense_npy resolution.")
    parser.add_argument("--coarse-collision-resolution", type=int, default=80)
    parser.add_argument("--use-coarse-collision", action="store_true")
    parser.add_argument("--uav-marker-scale", type=float, default=None, help="Override visual UAV marker scale. By default the marker is scaled to start_positions[].agent_size.")
    parser.add_argument("--hide-agent-collision-envelope", action="store_true", help="Do not draw the transparent Python collision / safety envelope around UAV markers.")
    parser.add_argument("--no-terrain-collision", action="store_true", help="Export the terrain as visual-only geometry.")
    parser.add_argument("--no-obstacle-collision", action="store_true", help="Export scenario obstacles as visual-only geometry.")
    parser.add_argument("--no-cache", action="store_true", help="Rewrite terrain mesh artifacts even if matching cache metadata exists.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    gazebo = export_gazebo_scene(
        scenario_json=Path(args.scenario_json),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        visual_resolution=args.visual_resolution,
        coarse_collision_resolution=args.coarse_collision_resolution,
        use_coarse_collision=args.use_coarse_collision,
        uav_marker_scale=args.uav_marker_scale,
        show_agent_collision_envelope=not args.hide_agent_collision_envelope,
        terrain_collision=not args.no_terrain_collision,
        obstacle_collision=not args.no_obstacle_collision,
        reuse_cache=not args.no_cache,
    )
    print(f"[gazebo_terrain_exporter] world_sdf={gazebo['world_sdf']}")
    print(f"[gazebo_terrain_exporter] model_parent_dir={gazebo['model_parent_dir']}")
    print(f"[gazebo_terrain_exporter] visual_mesh={gazebo['visual_mesh']['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
