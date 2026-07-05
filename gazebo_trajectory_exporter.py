#!/usr/bin/env python3
"""Export visual-only Gazebo trajectory replay models from MATD3 snapshots."""

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


REPLAY_MODEL_NAME = "matd3_trajectory_replay"


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
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def _hash_array_into(h: "hashlib._Hash", name: str, array: np.ndarray) -> None:
    arr = np.ascontiguousarray(array)
    h.update(name.encode("utf-8"))
    h.update(b":")
    h.update(str(arr.shape).encode("utf-8"))
    h.update(b":")
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(b":")
    h.update(arr.tobytes(order="C"))


def _array_sha256(array: np.ndarray) -> str:
    h = hashlib.sha256()
    _hash_array_into(h, "array", array)
    return h.hexdigest()


def _mesh_cache_key(points: np.ndarray, extra: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(b"matd3_trajectory_tube_obj_v3_with_mtl")
    _hash_array_into(h, "points", points)
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
        f.write("# MATD3 Gazebo trajectory material\n")
        f.write(f"newmtl {material_name}\n")
        f.write(f"Ka {_fmt(rgba_arr[0])} {_fmt(rgba_arr[1])} {_fmt(rgba_arr[2])}\n")
        f.write(f"Kd {_fmt(rgba_arr[0])} {_fmt(rgba_arr[1])} {_fmt(rgba_arr[2])}\n")
        f.write("Ks 0.05 0.05 0.05\n")
        f.write("Ns 16\n")
        f.write(f"d {_fmt(rgba_arr[3])}\n")
        f.write("illum 2\n")
    return mtl_path


def _save_npy_cached(path: Path, array: np.ndarray, reuse_cache: bool = True) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(array)
    sha256 = _array_sha256(arr)
    meta = {
        "path": str(path),
        "sidecar_meta": str(_sidecar_path(path)),
        "storage": "npy",
        "sha256": sha256,
        "shape": [int(v) for v in arr.shape],
        "dtype": str(arr.dtype),
        "cache_hit": False,
    }
    if reuse_cache and path.exists():
        old_meta = _load_sidecar(path)
        if (
            old_meta
            and old_meta.get("sha256") == sha256
            and old_meta.get("shape") == meta["shape"]
            and old_meta.get("dtype") == meta["dtype"]
            and old_meta.get("storage") == "npy"
        ):
            cached = dict(old_meta)
            cached["path"] = str(path)
            cached["sidecar_meta"] = str(_sidecar_path(path))
            cached["cache_hit"] = True
            return cached
    np.save(path, arr)
    _write_sidecar(path, {k: v for k, v in meta.items() if k != "cache_hit"})
    return meta


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


def _trajectory_array(snapshot: Dict[str, Any]) -> np.ndarray:
    npz_path = snapshot.get("trajectory_npz_path") or snapshot.get("export_paths", {}).get("trajectory_npz")
    if npz_path and Path(npz_path).exists():
        with np.load(Path(npz_path)) as data:
            key = "trajectory" if "trajectory" in data.files else data.files[0]
            arr = data[key].astype(np.float64)
    else:
        npy_path = snapshot.get("trajectory_npy_path") or snapshot.get("export_paths", {}).get("trajectory_npy")
        if npy_path and Path(npy_path).exists():
            arr = np.load(Path(npy_path)).astype(np.float64)
        else:
            arr = np.asarray(snapshot.get("trajectory", []), dtype=np.float64)
    if arr.ndim != 3 or arr.shape[0] < 1 or arr.shape[1] < 1 or arr.shape[2] < 3:
        raise ValueError("trajectory snapshot must have shape [frame][agent][xyz]")
    arr = arr[:, :, :3]
    if not np.all(np.isfinite(arr)):
        raise ValueError("trajectory snapshot contains non-finite positions")
    return arr


def _agent_colors() -> List[Tuple[float, float, float, float]]:
    return [
        (0.02, 0.02, 0.02, 1.0),
        (0.95, 0.03, 0.02, 1.0),
        (0.02, 0.18, 0.95, 1.0),
        (0.05, 0.65, 0.15, 1.0),
        (0.8, 0.05, 0.8, 1.0),
        (0.0, 0.7, 0.75, 1.0),
    ]


def _add_material(parent: ET.Element, rgba: Sequence[float]) -> None:
    mat = ET.SubElement(parent, "material")
    ET.SubElement(mat, "ambient").text = _vec(rgba)
    ET.SubElement(mat, "diffuse").text = _vec(rgba)
    ET.SubElement(mat, "specular").text = "0.08 0.08 0.08 1"


def _decimate_points(points: np.ndarray, stride: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    stride = max(1, int(stride))
    indices = list(range(0, points.shape[0], stride))
    if indices[-1] != points.shape[0] - 1:
        indices.append(points.shape[0] - 1)
    reduced = points[indices]
    if reduced.shape[0] <= 1:
        return reduced
    kept = [reduced[0]]
    for point in reduced[1:]:
        if np.linalg.norm(point - kept[-1]) > 1e-8:
            kept.append(point)
    return np.asarray(kept, dtype=np.float64)


def _frame_for_tangent(tangent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tangent = np.asarray(tangent, dtype=np.float64)
    norm = np.linalg.norm(tangent)
    if norm <= 1e-12:
        tangent = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        tangent = tangent / norm
    ref = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(tangent, ref))) > 0.92:
        ref = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    normal = np.cross(tangent, ref)
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= 1e-12:
        normal = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        normal = normal / normal_norm
    binormal = np.cross(tangent, normal)
    binormal_norm = np.linalg.norm(binormal)
    if binormal_norm <= 1e-12:
        binormal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        binormal = binormal / binormal_norm
    return normal, binormal


def write_polyline_tube_obj(
    path: Path,
    points: np.ndarray,
    radius: float,
    sides: int = 10,
    material_rgba: Optional[Sequence[float]] = None,
    reuse_cache: bool = True,
    cache_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape [n][xyz]")
    points = points[:, :3]
    radius = max(1e-4, float(radius))
    sides = max(5, int(sides))

    if points.shape[0] == 1:
        points = np.vstack([points[0], points[0] + np.asarray([1e-3, 0.0, 0.0])])

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cache_key = _mesh_cache_key(
        points,
        {
            "radius": radius,
            "sides": sides,
            "material_rgba": _json_safe(list(material_rgba)) if material_rgba is not None else None,
            **(cache_extra or {}),
        },
    )
    material_name = f"{path.stem}_material"
    meta = {
        "path": str(path),
        "sidecar_meta": str(_sidecar_path(path)),
        "cache_key": cache_key,
        "material_path": str(path.with_suffix(".mtl")),
        "material_name": material_name,
        "point_count": int(points.shape[0]),
        "vertex_count": int(points.shape[0] * sides),
        "face_count": int((points.shape[0] - 1) * sides * 2),
        "tube_radius": float(radius),
        "tube_sides": int(sides),
        "length": float(np.sum(segment_lengths)) if segment_lengths.size else 0.0,
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

    rings: List[np.ndarray] = []
    for idx, point in enumerate(points):
        if idx == 0:
            tangent = points[1] - points[0]
        elif idx == points.shape[0] - 1:
            tangent = points[-1] - points[-2]
        else:
            tangent = points[idx + 1] - points[idx - 1]
        normal, binormal = _frame_for_tangent(tangent)
        ring = []
        for side in range(sides):
            theta = 2.0 * math.pi * side / float(sides)
            offset = math.cos(theta) * normal + math.sin(theta) * binormal
            ring.append(point + radius * offset)
        rings.append(np.asarray(ring, dtype=np.float64))

    rgba = material_rgba if material_rgba is not None else (0.1, 0.35, 0.95, 0.85)
    material_path = _write_obj_mtl(path, material_name, rgba)
    with path.open("w", encoding="utf-8") as f:
        f.write("# MATD3 visual-only trajectory tube mesh\n")
        f.write(f"# point_count {points.shape[0]}\n")
        f.write(f"# tube_radius {radius:.8g}\n")
        f.write(f"mtllib {material_path.name}\n")
        f.write("o trajectory_path\n")
        f.write(f"usemtl {material_name}\n")
        for ring in rings:
            for vertex in ring:
                f.write(f"v {_fmt(vertex[0])} {_fmt(vertex[1])} {_fmt(vertex[2])}\n")
        for idx in range(points.shape[0] - 1):
            base = idx * sides
            nxt = (idx + 1) * sides
            for side in range(sides):
                a = base + side + 1
                b = base + ((side + 1) % sides) + 1
                c = nxt + ((side + 1) % sides) + 1
                d = nxt + side + 1
                f.write(f"f {a} {b} {c}\n")
                f.write(f"f {a} {c} {d}\n")
    _write_sidecar(path, {k: v for k, v in meta.items() if k != "cache_hit"})
    return meta


def write_model_config(model_dir: Path) -> None:
    root = ET.Element("model")
    ET.SubElement(root, "name").text = REPLAY_MODEL_NAME
    ET.SubElement(root, "version").text = "1.0"
    sdf = ET.SubElement(root, "sdf", {"version": "1.10"})
    sdf.text = "model.sdf"
    author = ET.SubElement(root, "author")
    ET.SubElement(author, "name").text = "MATD3 Gazebo trajectory exporter"
    ET.SubElement(author, "email").text = "n/a"
    ET.SubElement(root, "description").text = "Visual-only trajectory replay paths exported from MATD3 trajectory snapshots."
    _write_xml(root, model_dir / "model.config")


def _add_mesh_visual(link: ET.Element, name: str, uri: str, rgba: Sequence[float]) -> None:
    visual = ET.SubElement(link, "visual", {"name": name})
    geometry = ET.SubElement(visual, "geometry")
    mesh = ET.SubElement(geometry, "mesh")
    ET.SubElement(mesh, "uri").text = uri
    _add_material(visual, rgba)


def _add_sphere_visual(link: ET.Element, name: str, position: Sequence[float], radius: float, rgba: Sequence[float]) -> None:
    visual = ET.SubElement(link, "visual", {"name": name})
    ET.SubElement(visual, "pose").text = f"{_vec(position)} 0 0 0"
    geometry = ET.SubElement(visual, "geometry")
    sphere = ET.SubElement(geometry, "sphere")
    ET.SubElement(sphere, "radius").text = _fmt(radius)
    _add_material(visual, rgba)


def write_replay_model_sdf(
    model_dir: Path,
    agent_path_meta: Sequence[Dict[str, Any]],
    ghost_points: Sequence[Tuple[int, int, Sequence[float], float, Sequence[float]]],
) -> None:
    root = ET.Element("sdf", {"version": "1.10"})
    model = ET.SubElement(root, "model", {"name": REPLAY_MODEL_NAME})
    ET.SubElement(model, "static").text = "true"
    link = ET.SubElement(model, "link", {"name": "trajectory_replay_link"})
    for meta in agent_path_meta:
        _add_mesh_visual(
            link,
            f"agent_{int(meta['agent_index'])}_path_visual",
            f"model://{REPLAY_MODEL_NAME}/meshes/agent_{int(meta['agent_index'])}_path.obj",
            meta["color_rgba"],
        )
    for agent_idx, frame_idx, position, radius, rgba in ghost_points:
        _add_sphere_visual(
            link,
            f"agent_{agent_idx}_ghost_{frame_idx}",
            position,
            radius,
            rgba,
        )
    _write_xml(root, model_dir / "model.sdf")


def write_world_replay_sdf(base_world_sdf: Path, output_path: Path) -> None:
    root = ET.parse(base_world_sdf).getroot()
    world = root.find("world")
    if world is None:
        raise ValueError(f"world.sdf has no <world>: {base_world_sdf}")
    for include in world.findall("include"):
        if include.findtext("name") == REPLAY_MODEL_NAME or include.findtext("uri") == f"model://{REPLAY_MODEL_NAME}":
            world.remove(include)
    include = ET.SubElement(world, "include")
    ET.SubElement(include, "uri").text = f"model://{REPLAY_MODEL_NAME}"
    ET.SubElement(include, "name").text = REPLAY_MODEL_NAME
    _write_xml(root, output_path)


def export_gazebo_trajectory_replay(
    scenario_json: Path,
    trajectory_json: Path,
    output_dir: Optional[Path] = None,
    base_world_sdf: Optional[Path] = None,
    path_stride: int = 2,
    path_radius: float = 0.18,
    tube_sides: int = 10,
    ghost_stride: int = 80,
    ghost_radius_scale: float = 1.0,
    reuse_cache: bool = True,
) -> Dict[str, Any]:
    scenario_json = Path(scenario_json).resolve()
    trajectory_json = Path(trajectory_json).resolve()
    scenario = _load_json(scenario_json)
    trajectory_snapshot = _load_json(trajectory_json)
    trajectory = _trajectory_array(trajectory_snapshot)

    if output_dir is None:
        output_dir = scenario_json.parent
    output_dir = Path(output_dir).resolve()
    model_parent_dir = output_dir / "models"
    model_dir = model_parent_dir / REPLAY_MODEL_NAME
    mesh_dir = model_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    colors = _agent_colors()
    agent_path_meta: List[Dict[str, Any]] = []
    for agent_idx in range(trajectory.shape[1]):
        points = _decimate_points(trajectory[:, agent_idx, :], path_stride)
        color_rgba = colors[agent_idx % len(colors)]
        centerline_path = mesh_dir / f"agent_{agent_idx}_centerline.npy"
        centerline_meta = _save_npy_cached(centerline_path, points.astype(np.float32), reuse_cache=reuse_cache)
        mesh_meta = write_polyline_tube_obj(
            mesh_dir / f"agent_{agent_idx}_path.obj",
            points,
            radius=path_radius,
            sides=tube_sides,
            material_rgba=color_rgba,
            reuse_cache=reuse_cache,
            cache_extra={
                "agent_index": int(agent_idx),
                "source_frame_count": int(trajectory.shape[0]),
                "path_stride": int(path_stride),
            },
        )
        meta = {
            "agent_index": int(agent_idx),
            "source_frame_count": int(trajectory.shape[0]),
            "path_stride": int(path_stride),
            "centerline_npy": str(centerline_path),
            "centerline": centerline_meta,
            "mesh": mesh_meta,
            "color_rgba": color_rgba,
            "visual_only": True,
        }
        agent_path_meta.append(meta)

    try:
        agent_radius = float(trajectory_snapshot.get("agent_size", 0.5))
    except Exception:
        agent_radius = 0.5
    ghost_radius = max(0.001, agent_radius * max(0.01, float(ghost_radius_scale)))
    ghost_points: List[Tuple[int, int, Sequence[float], float, Sequence[float]]] = []
    if ghost_stride > 0:
        frame_indices = list(range(0, trajectory.shape[0], int(ghost_stride)))
        if frame_indices[-1] != trajectory.shape[0] - 1:
            frame_indices.append(trajectory.shape[0] - 1)
        for agent_idx in range(trajectory.shape[1]):
            color = list(colors[agent_idx % len(colors)])
            color[3] = 0.28
            for frame_idx in frame_indices:
                ghost_points.append((agent_idx, int(frame_idx), trajectory[frame_idx, agent_idx, :].tolist(), ghost_radius, tuple(color)))

    write_model_config(model_dir)
    write_replay_model_sdf(model_dir, agent_path_meta, ghost_points)

    if base_world_sdf is None:
        base_world_sdf = Path(scenario.get("gazebo", {}).get("world_sdf", scenario_json.parent / "world.sdf"))
    base_world_sdf = Path(base_world_sdf).resolve()
    world_replay_sdf = output_dir / "world_replay.sdf"
    write_world_replay_sdf(base_world_sdf, world_replay_sdf)

    replay_meta = {
        "scenario_json": str(scenario_json),
        "trajectory_snapshot_json": str(trajectory_json),
        "world_replay_sdf": str(world_replay_sdf),
        "base_world_sdf": str(base_world_sdf),
        "model_parent_dir": str(model_parent_dir),
        "model_name": REPLAY_MODEL_NAME,
        "model_uri": f"model://{REPLAY_MODEL_NAME}",
        "visual_only": True,
        "path_radius": float(path_radius),
        "path_stride": int(path_stride),
        "tube_sides": int(tube_sides),
        "ghost_stride": int(ghost_stride),
        "ghost_radius": float(ghost_radius),
        "agent_paths": agent_path_meta,
        "cache": {
            "reuse_cache": bool(reuse_cache),
            "centerline_cache_hits": [bool(item.get("centerline", {}).get("cache_hit", False)) for item in agent_path_meta],
            "path_mesh_cache_hits": [bool(item.get("mesh", {}).get("cache_hit", False)) for item in agent_path_meta],
        },
        "resource_path_command": f"export GZ_SIM_RESOURCE_PATH={model_parent_dir}:$GZ_SIM_RESOURCE_PATH",
    }

    trajectory_snapshot["gazebo_replay"] = replay_meta
    with trajectory_json.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(trajectory_snapshot), f, ensure_ascii=False, indent=2)

    return replay_meta


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        return int(default)
    return value if value > 0 else int(default)


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except Exception:
        return float(default)
    return value if np.isfinite(value) else float(default)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a visual-only Gazebo trajectory replay world.")
    parser.add_argument("--scenario-json", required=True)
    parser.add_argument("--trajectory-json", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--base-world-sdf", default=None)
    parser.add_argument("--path-stride", type=int, default=_env_int("GAZEBO_REPLAY_PATH_STRIDE", 2))
    parser.add_argument("--path-radius", type=float, default=_env_float("GAZEBO_REPLAY_PATH_RADIUS", 0.18))
    parser.add_argument("--tube-sides", type=int, default=_env_int("GAZEBO_REPLAY_TUBE_SIDES", 10))
    parser.add_argument("--ghost-stride", type=int, default=_env_int("GAZEBO_REPLAY_GHOST_STRIDE", 80))
    parser.add_argument("--ghost-radius-scale", type=float, default=_env_float("GAZEBO_REPLAY_GHOST_RADIUS_SCALE", 1.0))
    parser.add_argument("--no-cache", action="store_true", help="Rewrite trajectory mesh artifacts even if matching cache metadata exists.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    replay = export_gazebo_trajectory_replay(
        scenario_json=Path(args.scenario_json),
        trajectory_json=Path(args.trajectory_json),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        base_world_sdf=Path(args.base_world_sdf) if args.base_world_sdf else None,
        path_stride=args.path_stride,
        path_radius=args.path_radius,
        tube_sides=args.tube_sides,
        ghost_stride=args.ghost_stride,
        ghost_radius_scale=args.ghost_radius_scale,
        reuse_cache=not args.no_cache,
    )
    print(f"[gazebo_trajectory_exporter] world_replay_sdf={replay['world_replay_sdf']}")
    print(f"[gazebo_trajectory_exporter] model_parent_dir={replay['model_parent_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
