#!/usr/bin/env python3
"""Build and optionally launch a one-trajectory Python HTML + Gazebo replay view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gazebo_trajectory_exporter import export_gazebo_trajectory_replay  # noqa: E402
from visualization.trajectory_visualizer import TrajectoryVisualizer  # noqa: E402


def _load_json(path: Path) -> Dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


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


def _episode(results: Dict[str, Any], episode_index: int) -> Dict[str, Any]:
    episodes = results.get("episode_details") or []
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("evaluation_results.json has no episode_details")
    if episode_index < 0 or episode_index >= len(episodes):
        raise IndexError(f"episode_index={episode_index} outside episode_details length {len(episodes)}")
    ep = episodes[episode_index]
    if not isinstance(ep, dict):
        raise ValueError(f"episode_details[{episode_index}] must be an object")
    return ep


def _trajectory(ep: Dict[str, Any]) -> np.ndarray:
    traj = np.asarray(ep.get("trajectory") or [], dtype=np.float32)
    if traj.ndim != 3 or traj.shape[0] < 1 or traj.shape[1] < 1 or traj.shape[2] < 3:
        raise ValueError("episode trajectory must have shape [frame][agent][xyz]")
    traj = np.ascontiguousarray(traj[:, :, :3], dtype=np.float32)
    if not np.all(np.isfinite(traj)):
        raise ValueError("episode trajectory contains non-finite values")
    return traj


def _trajectory_sha256(traj: np.ndarray) -> str:
    arr = np.ascontiguousarray(traj, dtype=np.float32)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(b"|")
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(b"|")
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _scenario_namespace(scenario: Dict[str, Any], scenario_json: Path) -> SimpleNamespace:
    terrain_info = scenario.get("terrain") if isinstance(scenario.get("terrain"), dict) else {}
    terrain_path_raw = terrain_info.get("dense_npy") or terrain_info.get("sampled_npy")
    if not terrain_path_raw:
        raise ValueError("scenario.json must contain terrain.dense_npy or terrain.sampled_npy")
    terrain_path = Path(terrain_path_raw).expanduser()
    if not terrain_path.is_absolute():
        terrain_path = scenario_json.parent / terrain_path
    terrain = np.load(terrain_path)
    if terrain.ndim != 2:
        raise ValueError(f"terrain array must be 2D: {terrain_path}")
    return SimpleNamespace(
        terrain=terrain,
        map_size=float(scenario.get("map_size", terrain.shape[0]) or terrain.shape[0]),
        obstacles=scenario.get("obstacles", []) or [],
        goal_pos=(scenario.get("goal") or {}).get("position"),
        agent_goals=[g.get("position") for g in scenario.get("agent_goals", []) or []],
        agent_size=(scenario.get("collision") or {}).get("agent_size", 0.5),
    )


def _write_trajectory_snapshot(
    output_dir: Path,
    prefix: str,
    traj: np.ndarray,
    ep: Dict[str, Any],
    results_json: Path,
    scenario_json: Path,
    traj_hash: str,
) -> Path:
    npz_path = output_dir / f"{prefix}_trajectory.npz"
    np.savez_compressed(npz_path, trajectory=traj)
    snapshot_path = output_dir / f"{prefix}_trajectory_snapshot.json"
    snapshot = {
        "source": "same_trajectory_python_gazebo",
        "source_results_json": str(results_json),
        "source_scenario_json": str(scenario_json),
        "trajectory_npz_path": str(npz_path),
        "trajectory_shape": [int(v) for v in traj.shape],
        "trajectory_sha256": traj_hash,
        "agent_size": float(ep.get("agent_size", 0.5) or 0.5),
        "episode": int(ep.get("episode", 0) or 0),
        "steps": int(ep.get("steps", traj.shape[0] - 1) or (traj.shape[0] - 1)),
        "success": int(ep.get("success", 0) or 0),
        "team_success": int(ep.get("team_success", ep.get("success", 0)) or 0),
        "terrain_seed": ep.get("terrain_seed"),
        "terrain_variant_seed": ep.get("terrain_variant_seed"),
        "obstacle_seed": ep.get("obstacle_seed"),
        "final_goal_distance": ep.get("final_goal_distance"),
        "agent_final_goal_distances": ep.get("agent_final_goal_distances"),
        "export_paths": {
            "trajectory_npz": str(npz_path),
            "trajectory_snapshot_json": str(snapshot_path),
            "output_dir": str(output_dir),
        },
    }
    with snapshot_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(snapshot), f, ensure_ascii=False, indent=2)
    return snapshot_path


def _write_visual_only_world(src_world: Path, dst_world: Path) -> None:
    tree = ET.parse(src_world)
    root = tree.getroot()
    world = root.find("world")
    if world is None:
        raise ValueError(f"world.sdf has no <world>: {src_world}")
    gui = world.find("gui")
    if gui is not None:
        world.remove(gui)
    for plugin in list(world.findall("plugin")):
        filename = plugin.get("filename", "")
        name = plugin.get("name", "")
        if "contact-system" in filename or "sensors-system" in filename or "Contact" in name or "Sensors" in name:
            world.remove(plugin)
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag == "collision":
                parent.remove(child)
    ET.indent(tree, space="  ")
    tree.write(dst_world, encoding="utf-8", xml_declaration=True)


def _generate_html(
    output_html: Path,
    title: str,
    traj: np.ndarray,
    ep: Dict[str, Any],
    scenario: Dict[str, Any],
    scenario_json: Path,
) -> None:
    scenario_ns = _scenario_namespace(scenario, scenario_json)
    goal_positions = {
        "goal_pos": scenario_ns.goal_pos,
        "agent_goals": scenario_ns.agent_goals,
    }
    viz = TrajectoryVisualizer(figsize=(12, 10), dpi=200, verbose=False)
    ok = viz.generate_trajectory_interactive(
        trajectories=traj.tolist(),
        save_path=str(output_html),
        title=title,
        goal_positions=goal_positions,
        scenario=scenario_ns,
        env_instance=None,
    )
    if not ok:
        raise RuntimeError("failed to generate interactive trajectory HTML")


def _open_html(path: Path) -> None:
    opener = shutil.which("wslview") or shutil.which("xdg-open")
    if not opener:
        return
    try:
        subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return


def _gazebo_command(world_sdf: Path, model_dirs: Sequence[Path], gui_config: Path, render_engine: str) -> str:
    resource = ":".join(str(Path(p).resolve()) for p in model_dirs) + ":${GZ_SIM_RESOURCE_PATH}"
    return (
        "export GZ_SIM_RESOURCE_PATH="
        + resource
        + "\n"
        + "gz sim -v 4 "
        + f"--gui-config {gui_config} "
        + f"--render-engine-gui {render_engine} "
        + str(world_sdf)
    )


def _launch_gazebo(world_sdf: Path, model_dirs: Sequence[Path], gui_config: Path, render_engine: str) -> int:
    env = os.environ.copy()
    env["XDG_RUNTIME_DIR"] = env.get("XDG_RUNTIME_DIR") or "/mnt/wslg/runtime-dir"
    env["DISPLAY"] = env.get("DISPLAY") or ":0"
    env.pop("WAYLAND_DISPLAY", None)
    env.pop("QT_XCB_GL_INTEGRATION", None)
    env.pop("QT_OPENGL", None)
    env.pop("QT_QUICK_BACKEND", None)
    env.pop("LIBGL_ALWAYS_SOFTWARE", None)
    env.pop("MESA_LOADER_DRIVER_OVERRIDE", None)
    env["QT_QPA_PLATFORM"] = "xcb"
    env["GALLIUM_DRIVER"] = "d3d12"
    env["MESA_D3D12_DEFAULT_ADAPTER_NAME"] = env.get("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")
    env["HOME"] = env.get("HOME") or "/tmp/matd3_gz_home_same_trajectory"
    env["GZ_SIM_RESOURCE_PATH"] = ":".join(str(Path(p).resolve()) for p in model_dirs) + ":" + env.get(
        "GZ_SIM_RESOURCE_PATH", ""
    )
    cmd = [
        "gz",
        "sim",
        "-v",
        "4",
        "--gui-config",
        str(gui_config),
        "--render-engine-gui",
        render_engine,
        str(world_sdf),
    ]
    return subprocess.run(cmd, env=env).returncode


def build_same_trajectory_view(args: argparse.Namespace) -> Dict[str, Any]:
    results_json = Path(args.results_json).expanduser().resolve()
    scenario_json = Path(args.scenario_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = _load_json(results_json)
    scenario = _load_json(scenario_json)
    ep = _episode(results, int(args.episode_index))
    traj = _trajectory(ep)
    traj_hash = _trajectory_sha256(traj)
    prefix = args.prefix or f"episode_{int(args.episode_index) + 1:03d}_{args.label}"

    html_path = output_dir / f"{prefix}_python_interactive.html"
    title = (
        f"Same trajectory source: {args.label} | "
        f"seed={ep.get('terrain_seed')} variant={ep.get('terrain_variant_seed')} "
        f"obstacle={ep.get('obstacle_seed')} | sha256={traj_hash[:12]}"
    )
    _generate_html(html_path, title, traj, ep, scenario, scenario_json)

    snapshot_path = _write_trajectory_snapshot(
        output_dir=output_dir,
        prefix=prefix,
        traj=traj,
        ep=ep,
        results_json=results_json,
        scenario_json=scenario_json,
        traj_hash=traj_hash,
    )
    base_world_sdf = Path(args.base_world_sdf).expanduser().resolve() if args.base_world_sdf else None
    if base_world_sdf is None:
        base_world_sdf = Path((scenario.get("gazebo") or {}).get("world_sdf", scenario_json.parent / "world.sdf")).resolve()
    replay = export_gazebo_trajectory_replay(
        scenario_json=scenario_json,
        trajectory_json=snapshot_path,
        output_dir=output_dir,
        base_world_sdf=base_world_sdf,
        path_stride=int(args.path_stride),
        path_radius=float(args.path_radius),
        tube_sides=int(args.tube_sides),
        ghost_stride=int(args.ghost_stride),
        ghost_radius_scale=float(args.ghost_radius_scale),
        reuse_cache=True,
    )
    world_replay = Path(replay["world_replay_sdf"]).resolve()
    visual_world = output_dir / f"{prefix}_gazebo_visual_only_world.sdf"
    _write_visual_only_world(world_replay, visual_world)

    source_model_dir = Path((scenario.get("gazebo") or {}).get("model_parent_dir", scenario_json.parent / "models")).resolve()
    replay_model_dir = Path(replay["model_parent_dir"]).resolve()
    gui_config = Path(args.gui_config).expanduser().resolve()
    model_dirs = [replay_model_dir, source_model_dir]
    gz_command = _gazebo_command(visual_world, model_dirs, gui_config, args.render_engine)

    manifest_path = output_dir / f"{prefix}_same_trajectory_manifest.json"
    manifest = {
        "label": args.label,
        "source_results_json": str(results_json),
        "source_scenario_json": str(scenario_json),
        "episode_index": int(args.episode_index),
        "trajectory_shape": [int(v) for v in traj.shape],
        "trajectory_sha256": traj_hash,
        "python_html": str(html_path),
        "gazebo_world_sdf": str(visual_world),
        "trajectory_snapshot_json": str(snapshot_path),
        "gazebo_replay_model_dir": str(replay_model_dir),
        "source_model_dir": str(source_model_dir),
        "gui_config": str(gui_config),
        "render_engine": args.render_engine,
        "gazebo_command": gz_command,
        "episode_metrics": {
            "steps": ep.get("steps"),
            "success": ep.get("success"),
            "team_success": ep.get("team_success", ep.get("success")),
            "final_goal_distance": ep.get("final_goal_distance"),
            "agent_final_goal_distances": ep.get("agent_final_goal_distances"),
            "terrain_seed": ep.get("terrain_seed"),
            "terrain_variant_seed": ep.get("terrain_variant_seed"),
            "obstacle_seed": ep.get("obstacle_seed"),
        },
    }
    run_script = output_dir / f"run_{prefix}_gazebo.sh"
    run_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd /home/tang/matd3\n"
        "source /opt/ros/jazzy/setup.bash\n"
        "export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/mnt/wslg/runtime-dir}\n"
        "export DISPLAY=${DISPLAY:-:0}\n"
        "unset WAYLAND_DISPLAY QT_XCB_GL_INTEGRATION QT_OPENGL QT_QUICK_BACKEND LIBGL_ALWAYS_SOFTWARE MESA_LOADER_DRIVER_OVERRIDE\n"
        "export QT_QPA_PLATFORM=xcb\n"
        "export GALLIUM_DRIVER=d3d12\n"
        "export MESA_D3D12_DEFAULT_ADAPTER_NAME=${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}\n"
        "export HOME=${HOME:-/tmp/matd3_gz_home_same_trajectory}\n"
        "mkdir -p \"$HOME\"\n"
        + gz_command
        + "\n",
        encoding="utf-8",
    )
    run_script.chmod(0o755)

    manifest["manifest_path"] = str(manifest_path)
    manifest["run_script"] = str(run_script)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(manifest), f, ensure_ascii=False, indent=2)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--scenario-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-world-sdf", default=None)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--label", default="python_original")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--gui-config", default=str(REPO_ROOT / "tools" / "gazebo_minimal_ogre_gui.config"))
    parser.add_argument("--render-engine", default="ogre", choices=["ogre", "ogre2"])
    parser.add_argument("--path-stride", type=int, default=1)
    parser.add_argument("--path-radius", type=float, default=0.20)
    parser.add_argument("--tube-sides", type=int, default=10)
    parser.add_argument("--ghost-stride", type=int, default=80)
    parser.add_argument("--ghost-radius-scale", type=float, default=1.0)
    parser.add_argument("--open-html", action="store_true")
    parser.add_argument("--launch-gazebo", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = build_same_trajectory_view(args)
    print(f"[same_trajectory] html={manifest['python_html']}")
    print(f"[same_trajectory] gazebo_world={manifest['gazebo_world_sdf']}")
    print(f"[same_trajectory] manifest={manifest['manifest_path']}")
    print(f"[same_trajectory] run_script={manifest['run_script']}")
    print(f"[same_trajectory] trajectory_sha256={manifest['trajectory_sha256']}")
    if args.open_html:
        _open_html(Path(manifest["python_html"]))
    if args.launch_gazebo:
        return _launch_gazebo(
            Path(manifest["gazebo_world_sdf"]),
            [Path(manifest["gazebo_replay_model_dir"]), Path(manifest["source_model_dir"])],
            Path(manifest["gui_config"]),
            str(manifest["render_engine"]),
        )
    print("[same_trajectory] gazebo command:")
    print(manifest["gazebo_command"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
