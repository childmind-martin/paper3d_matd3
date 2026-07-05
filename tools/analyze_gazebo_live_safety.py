#!/usr/bin/env python3
"""Diagnose Gazebo-live contact, velocity adapter, and APF safety regions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _finite_values(values: Iterable[Any]) -> np.ndarray:
    arr = np.asarray([_safe_float(v) for v in values], dtype=np.float64)
    return arr[np.isfinite(arr)]


def _summary(values: Iterable[Any]) -> Dict[str, Any]:
    arr = _finite_values(values)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def _vec3(row: Mapping[str, Any], prefix: str) -> Optional[np.ndarray]:
    keys = (f"{prefix}_vx", f"{prefix}_vy", f"{prefix}_vz")
    try:
        arr = np.asarray([float(row[k]) for k in keys], dtype=np.float64)
        if arr.shape == (3,) and np.all(np.isfinite(arr)):
            return arr
    except Exception:
        return None
    return None


def _pose(row: Mapping[str, Any]) -> Optional[np.ndarray]:
    try:
        arr = np.asarray([float(row["pose_x"]), float(row["pose_y"]), float(row["pose_z"])], dtype=np.float64)
        if arr.shape == (3,) and np.all(np.isfinite(arr)):
            return arr
    except Exception:
        return None
    return None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-9:
        return 1.0 if float(np.linalg.norm(a - b)) <= 1e-9 else math.nan
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def _classify_contact_pair(collision1: str, collision2: str) -> str:
    text = f"{collision1} {collision2}".lower()
    if "terrain" in text or "matd3_terrain" in text:
        return "agent_vs_terrain"
    if "obstacle" in text:
        return "agent_vs_obstacle"
    if text.count("dynamic_agent") >= 2:
        return "agent_vs_agent"
    if "ground" in text or "plane" in text:
        return "agent_vs_ground_plane"
    if collision1 or collision2:
        return "other_collision_pair"
    return "topic_marker_only"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_evaluation_results(live_root: Path) -> Path:
    candidates = [
        live_root / "gazebo_live" / "evaluation_results.json",
        live_root / "evaluation_results.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(live_root.glob("**/evaluation_results.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"evaluation_results.json not found under {live_root}")


def _load_episode(evaluation_results_path: Path, episode_index: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data = _load_json(evaluation_results_path)
    episodes = data.get("episode_details") or []
    if not episodes:
        raise ValueError(f"episode_details missing in {evaluation_results_path}")
    idx = min(max(0, int(episode_index)), len(episodes) - 1)
    return data, episodes[idx]


def _find_scenario_path(ep: Mapping[str, Any], evaluation_results_path: Path) -> Path:
    raw = ep.get("gazebo_live_scenario_json")
    if raw:
        path = Path(str(raw)).expanduser()
        if path.exists():
            return path
    matches = sorted(evaluation_results_path.parent.glob("**/scenario.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError("scenario.json not found from episode metadata or evaluation result directory")


def _find_apf_csv(apf_dir: Optional[Path]) -> Optional[Path]:
    if apf_dir is None:
        return None
    if apf_dir.is_file():
        return apf_dir
    candidate = apf_dir / "gazebo_apf_live_metrics.csv"
    if candidate.exists():
        return candidate
    matches = sorted(apf_dir.glob("**/gazebo_apf_live_metrics.csv"))
    return matches[0] if matches else None


def _read_apf_rows(apf_csv: Optional[Path]) -> List[Dict[str, str]]:
    if apf_csv is None or not apf_csv.exists():
        return []
    with apf_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_terrain_dense(scenario: Mapping[str, Any], scenario_path: Path) -> Optional[np.ndarray]:
    candidates: List[Path] = []
    terrain = scenario.get("terrain") if isinstance(scenario.get("terrain"), Mapping) else {}
    for key in ("dense_npy", "sampled_npy"):
        raw = terrain.get(key)
        if raw:
            candidates.append(Path(str(raw)).expanduser())
    candidates.extend([scenario_path.parent / "terrain_dense.npy", scenario_path.parent / "terrain_sampled.npy"])
    for candidate in candidates:
        if candidate.exists():
            try:
                return np.load(candidate)
            except Exception:
                continue
    return None


def _nearest_grid_height(terrain: Optional[np.ndarray], position: Sequence[float]) -> Optional[float]:
    if terrain is None:
        return None
    arr = np.asarray(terrain)
    if arr.ndim != 2 or arr.size == 0:
        return None
    pos = np.asarray(position, dtype=np.float64)
    x = int(round(float(np.clip(pos[0], 0, arr.shape[0] - 1))))
    y = int(round(float(np.clip(pos[1], 0, arr.shape[1] - 1))))
    return float(arr[x, y])


def _obstacle_surface_distance(position: Sequence[float], obstacle: Mapping[str, Any], agent_radius: float) -> float:
    pos = np.asarray(position, dtype=np.float64)
    center = np.asarray(obstacle.get("center", [math.nan, math.nan, math.nan]), dtype=np.float64)
    radius = _safe_float(obstacle.get("radius"), 0.0)
    return float(np.linalg.norm(pos - center) - radius - agent_radius)


def _contact_diagnostics(
    ep: Mapping[str, Any],
    scenario: Mapping[str, Any],
    terrain_dense: Optional[np.ndarray],
) -> Dict[str, Any]:
    collision = scenario.get("collision") if isinstance(scenario.get("collision"), Mapping) else {}
    agent_radius = _safe_float(collision.get("agent_size"), 0.5)
    python_threshold = _safe_float(collision.get("collision_distance_threshold"), 0.0)
    obstacles = list(scenario.get("obstacles") or [])
    starts = list(scenario.get("start_positions") or [])
    pairs = list(ep.get("gazebo_contact_pairs") or [])

    class_counts: Counter[str] = Counter()
    agent_counts: Counter[str] = Counter()
    unique_pairs: Counter[Tuple[str, str, str, str]] = Counter()
    topic_marker_count = 0
    actual_pair_count = 0
    for item in pairs:
        collision1 = str(item.get("collision1", "") or "")
        collision2 = str(item.get("collision2", "") or "")
        agent = str(item.get("agent", "") or "")
        cls = _classify_contact_pair(collision1, collision2)
        class_counts[cls] += 1
        if cls == "topic_marker_only":
            topic_marker_count += 1
        else:
            actual_pair_count += 1
            agent_counts[agent] += 1
            unique_pairs[(agent, collision1, collision2, cls)] += 1

    start_checks = []
    for idx, start in enumerate(starts):
        position = np.asarray(start.get("position", [math.nan, math.nan, math.nan]), dtype=np.float64)
        terrain_height = _nearest_grid_height(terrain_dense, position)
        if terrain_height is None:
            terrain_height = _safe_float(start.get("terrain_height"))
        terrain_clearance = float(position[2] - agent_radius - terrain_height) if math.isfinite(terrain_height) else math.nan
        nearest = []
        for obstacle in obstacles:
            surface = _obstacle_surface_distance(position, obstacle, agent_radius)
            nearest.append(
                {
                    "name": obstacle.get("name"),
                    "surface_distance": surface,
                    "python_margin": surface - python_threshold,
                    "radius": _safe_float(obstacle.get("radius")),
                    "center": obstacle.get("center"),
                }
            )
        nearest.sort(key=lambda item: item["surface_distance"])
        start_checks.append(
            {
                "agent_id": idx,
                "name": start.get("name", f"agent_{idx}"),
                "position": position.astype(float).tolist(),
                "terrain_height": terrain_height,
                "terrain_clearance_minus_agent_radius": terrain_clearance,
                "height_above_terrain_meta": start.get("height_above_terrain"),
                "nearest_obstacles": nearest[:3],
                "starts_overlapping_terrain": bool(math.isfinite(terrain_clearance) and terrain_clearance < 0.0),
                "starts_inside_python_terrain_collision_margin": bool(
                    math.isfinite(terrain_clearance) and terrain_clearance < python_threshold
                ),
                "starts_overlapping_obstacle": bool(nearest and nearest[0]["surface_distance"] < 0.0),
                "starts_inside_python_obstacle_collision_margin": bool(nearest and nearest[0]["python_margin"] < 0.0),
            }
        )

    first_step = ep.get("gazebo_contact_step")
    trajectory = ep.get("trajectory") or []
    first_step_geometry = []
    if first_step is not None and isinstance(trajectory, list):
        step_idx = _safe_int(first_step)
        for probe_step in (step_idx - 2, step_idx - 1, step_idx, step_idx + 1):
            if probe_step < 0 or probe_step >= len(trajectory):
                continue
            positions = np.asarray(trajectory[probe_step], dtype=np.float64)
            if positions.ndim != 2 or positions.shape[1] < 3:
                continue
            per_agent = []
            for agent_id, pos in enumerate(positions[:, :3]):
                nearest = []
                for obstacle in obstacles:
                    surface = _obstacle_surface_distance(pos, obstacle, agent_radius)
                    nearest.append(
                        {
                            "name": obstacle.get("name"),
                            "surface_distance": surface,
                            "python_margin": surface - python_threshold,
                            "radius": _safe_float(obstacle.get("radius")),
                        }
                    )
                nearest.sort(key=lambda item: item["surface_distance"])
                per_agent.append(
                    {
                        "agent_id": int(agent_id),
                        "position": pos.astype(float).tolist(),
                        "nearest_obstacle": nearest[0] if nearest else None,
                    }
                )
            first_step_geometry.append({"step": int(probe_step), "agents": per_agent})

    actual_classes = {k: v for k, v in class_counts.items() if k != "topic_marker_only"}
    true_collision_evidence = bool(actual_classes)
    only_expected_collision_links = all(
        ("python_collision_envelope_collision" in collision1 and ("obstacle" in collision2 or "terrain" in collision2))
        or ("python_collision_envelope_collision" in collision2 and ("obstacle" in collision1 or "terrain" in collision1))
        for (_, collision1, collision2, _) in unique_pairs
    ) if unique_pairs else False

    contact_count = _safe_int(ep.get("gazebo_contact_count"), 0)
    episode_steps = _safe_int(ep.get("steps", ep.get("episode_steps")), 0)
    persistent_steps = None
    if first_step is not None and episode_steps > 0:
        persistent_steps = max(0, episode_steps - _safe_int(first_step) + 1)

    return {
        "gazebo_contact_detected": bool(ep.get("gazebo_contact_detected", False)),
        "gazebo_contact_step": first_step,
        "gazebo_contact_count": contact_count,
        "episode_steps": episode_steps,
        "contact_steps_after_first_if_no_termination": persistent_steps,
        "gazebo_live_contact_terminates": bool(ep.get("gazebo_live_contact_terminates", False)),
        "gazebo_live_contact_marks_collision": bool(ep.get("gazebo_live_contact_marks_collision", False)),
        "gazebo_live_contact_authoritative": bool(ep.get("gazebo_live_contact_authoritative", False)),
        "gazebo_contact_agent_indices": ep.get("gazebo_contact_agent_indices", []),
        "raw_contact_pair_records": len(pairs),
        "topic_marker_records": int(topic_marker_count),
        "actual_collision_pair_records": int(actual_pair_count),
        "contact_class_counts": dict(class_counts),
        "contact_agent_record_counts": dict(agent_counts),
        "unique_collision_pair_count": int(len(unique_pairs)),
        "unique_collision_pairs": [
            {
                "count": int(count),
                "agent": agent,
                "collision1": collision1,
                "collision2": collision2,
                "class": cls,
            }
            for (agent, collision1, collision2, cls), count in unique_pairs.most_common(30)
        ],
        "agent_collision_radius": agent_radius,
        "python_collision_distance_threshold": python_threshold,
        "start_overlap_checks": start_checks,
        "first_contact_geometry_window": first_step_geometry,
        "true_collision_evidence": true_collision_evidence,
        "only_expected_collision_links": bool(only_expected_collision_links),
        "contact_count_interpretation": (
            "persistent_contact_until_timeout"
            if contact_count > 0 and not bool(ep.get("gazebo_live_contact_terminates", False))
            else "terminated_or_no_contact"
        ),
    }


def _adapter_diagnostics(rows: Sequence[Mapping[str, Any]], thresholds: Sequence[float]) -> Dict[str, Any]:
    feedback_rows = [row for row in rows if row.get("source") == "gazebo_feedback"]
    records = []
    for row in feedback_rows:
        cmd = _vec3(row, "cmd")
        feedback = _vec3(row, "feedback")
        if cmd is None or feedback is None:
            continue
        err_vec = feedback - cmd
        records.append(
            {
                "episode": _safe_int(row.get("episode")),
                "step": _safe_int(row.get("step")),
                "agent_id": _safe_int(row.get("agent_id")),
                "cmd": cmd,
                "feedback": feedback,
                "cmd_norm": float(np.linalg.norm(cmd)),
                "feedback_norm": float(np.linalg.norm(feedback)),
                "error_vec": err_vec,
                "error_norm": float(np.linalg.norm(err_vec)),
                "cosine": _cosine(cmd, feedback),
            }
        )

    threshold_stats: Dict[str, Any] = {}
    for threshold in thresholds:
        subset = [item for item in records if item["cmd_norm"] > threshold]
        key = f"cmd_norm_gt_{threshold:g}"
        if not subset:
            threshold_stats[key] = {"sample_count": 0}
            continue
        errors = np.asarray([item["error_norm"] for item in subset], dtype=np.float64)
        cosines = _finite_values(item["cosine"] for item in subset)
        err_mat = np.asarray([item["error_vec"] for item in subset], dtype=np.float64)
        cmd_mat = np.asarray([item["cmd"] for item in subset], dtype=np.float64)
        fb_mat = np.asarray([item["feedback"] for item in subset], dtype=np.float64)
        sign_match = (
            (np.sign(cmd_mat) == np.sign(fb_mat))
            | (np.abs(cmd_mat) <= 1e-9)
            | (np.abs(fb_mat) <= 1e-9)
        )
        threshold_stats[key] = {
            "sample_count": int(len(subset)),
            "error_norm": _summary(errors),
            "cosine": _summary(cosines),
            "axis_mae_xyz": np.mean(np.abs(err_mat), axis=0).astype(float).tolist(),
            "axis_p95_abs_error_xyz": np.quantile(np.abs(err_mat), 0.95, axis=0).astype(float).tolist(),
            "axis_sign_agreement_xyz": np.mean(sign_match, axis=0).astype(float).tolist(),
        }

    top_spikes = sorted(records, key=lambda item: item["error_norm"], reverse=True)[:20]
    transforms = {
        "identity": lambda arr: arr,
        "xy_swap": lambda arr: arr[:, [1, 0, 2]],
        "z_flip": lambda arr: arr * np.asarray([1.0, 1.0, -1.0]),
        "xy_swap_z_flip": lambda arr: arr[:, [1, 0, 2]] * np.asarray([1.0, 1.0, -1.0]),
        "x_flip": lambda arr: arr * np.asarray([-1.0, 1.0, 1.0]),
        "y_flip": lambda arr: arr * np.asarray([1.0, -1.0, 1.0]),
        "xy_flip": lambda arr: arr * np.asarray([-1.0, -1.0, 1.0]),
    }
    transform_stats = {}
    if records:
        cmd = np.asarray([item["cmd"] for item in records], dtype=np.float64)
        feedback = np.asarray([item["feedback"] for item in records], dtype=np.float64)
        mask = np.linalg.norm(cmd, axis=1) > 0.1
        for name, fn in transforms.items():
            if not np.any(mask):
                continue
            transformed = fn(cmd[mask])
            fb = feedback[mask]
            errors = np.linalg.norm(transformed - fb, axis=1)
            cos = np.sum(transformed * fb, axis=1) / (
                np.linalg.norm(transformed, axis=1) * np.linalg.norm(fb, axis=1) + 1e-9
            )
            transform_stats[name] = {
                "sample_count": int(errors.size),
                "mean_error": float(np.mean(errors)),
                "p95_error": float(np.quantile(errors, 0.95)),
                "mean_cosine": float(np.mean(cos[np.isfinite(cos)])),
            }

    best_transform = None
    if transform_stats:
        best_transform = min(transform_stats.items(), key=lambda item: item[1]["mean_error"])[0]

    return {
        "feedback_row_count": int(len(feedback_rows)),
        "valid_sample_count": int(len(records)),
        "threshold_stats": threshold_stats,
        "top_error_spikes": [
            {
                "episode": item["episode"],
                "step": item["step"],
                "agent_id": item["agent_id"],
                "cmd": item["cmd"].astype(float).tolist(),
                "feedback": item["feedback"].astype(float).tolist(),
                "cmd_norm": item["cmd_norm"],
                "feedback_norm": item["feedback_norm"],
                "error_norm": item["error_norm"],
                "cosine": item["cosine"],
            }
            for item in top_spikes
        ],
        "axis_transform_hypotheses_cmd_norm_gt_0.1": transform_stats,
        "best_axis_transform_by_mean_error": best_transform,
    }


def _region_stats(records: Sequence[Mapping[str, Any]], mismatch_threshold: float, direction_threshold: float) -> Dict[str, Any]:
    corrected = [_safe_float(item.get("corrected_action_error")) for item in records]
    pf = [_safe_float(item.get("pf_force_error")) for item in records]
    cos = [_safe_float(item.get("direction_cosine_similarity")) for item in records]
    mismatches = []
    for item, corr, pf_err, cos_val in zip(records, corrected, pf, cos):
        is_mismatch = (
            (math.isfinite(corr) and corr > mismatch_threshold)
            or (math.isfinite(pf_err) and pf_err > mismatch_threshold)
            or (math.isfinite(cos_val) and cos_val < direction_threshold)
        )
        if is_mismatch:
            mismatches.append((item, corr, pf_err, cos_val))
    top = sorted(
        mismatches,
        key=lambda entry: max(
            entry[1] if math.isfinite(entry[1]) else 0.0,
            entry[2] if math.isfinite(entry[2]) else 0.0,
            max(0.0, direction_threshold - entry[3]) if math.isfinite(entry[3]) else 0.0,
        ),
        reverse=True,
    )[:10]
    return {
        "sample_count": int(len(records)),
        "corrected_action_error": _summary(corrected),
        "pf_force_error": _summary(pf),
        "direction_cosine_similarity": _summary(cos),
        "mismatch_count": int(len(mismatches)),
        "mismatch_rate": float(len(mismatches) / max(len(records), 1)),
        "top_mismatch_cases": [
            {
                "episode": item.get("episode"),
                "step": item.get("step"),
                "agent_id": item.get("agent_id"),
                "nearest_obstacle": item.get("nearest_obstacle"),
                "nearest_obstacle_surface_distance": item.get("nearest_obstacle_surface_distance"),
                "terrain_clearance": item.get("terrain_clearance"),
                "min_agent_agent_clearance": item.get("min_agent_agent_clearance"),
                "corrected_action_error": corr,
                "pf_force_error": pf_err,
                "direction_cosine_similarity": cos_val,
            }
            for item, corr, pf_err, cos_val in top
        ],
    }


def _apf_safety_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    ep: Mapping[str, Any],
    scenario: Mapping[str, Any],
    obstacle_threshold: float,
    terrain_threshold: float,
    agent_close_threshold: float,
    contact_window: int,
    mismatch_threshold: float,
    direction_threshold: float,
) -> Dict[str, Any]:
    feedback_rows = [dict(row) for row in rows if row.get("source") == "gazebo_feedback"]
    collision = scenario.get("collision") if isinstance(scenario.get("collision"), Mapping) else {}
    agent_radius = _safe_float(collision.get("agent_size"), 0.5)

    by_step: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in feedback_rows:
        row["step"] = _safe_int(row.get("step"))
        row["agent_id"] = _safe_int(row.get("agent_id"))
        by_step[row["step"]].append(row)
    for step_rows in by_step.values():
        poses = {row["agent_id"]: _pose(row) for row in step_rows}
        for row in step_rows:
            this_pose = poses.get(row["agent_id"])
            clearances = []
            if this_pose is not None:
                for other_id, other_pose in poses.items():
                    if other_id == row["agent_id"] or other_pose is None:
                        continue
                    clearances.append(float(np.linalg.norm(this_pose - other_pose) - 2.0 * agent_radius))
            row["min_agent_agent_clearance"] = min(clearances) if clearances else math.inf

    contact_step = ep.get("gazebo_contact_step")
    contact_step_int = _safe_int(contact_step, -10**9) if contact_step is not None else None
    regions = {
        "near_obstacle_region": [],
        "near_terrain_region": [],
        "agent_agent_close_region": [],
        "contact_before_region": [],
    }
    for row in feedback_rows:
        obstacle_surface = _safe_float(row.get("nearest_obstacle_surface_distance"), math.inf)
        terrain_clearance = _safe_float(row.get("terrain_clearance"), math.inf)
        agent_clearance = _safe_float(row.get("min_agent_agent_clearance"), math.inf)
        step = _safe_int(row.get("step"))
        if obstacle_surface <= obstacle_threshold:
            regions["near_obstacle_region"].append(row)
        if terrain_clearance <= terrain_threshold:
            regions["near_terrain_region"].append(row)
        if agent_clearance <= agent_close_threshold:
            regions["agent_agent_close_region"].append(row)
        if contact_step_int is not None and contact_step_int - contact_window <= step <= contact_step_int:
            regions["contact_before_region"].append(row)

    bins = {}
    obstacle_bins = [(-math.inf, 0.0), (0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 50.0), (50.0, math.inf)]
    terrain_bins = [(-math.inf, 0.0), (0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, math.inf)]
    for label, key, bin_defs in (
        ("obstacle_surface_distance_bins", "nearest_obstacle_surface_distance", obstacle_bins),
        ("terrain_clearance_bins", "terrain_clearance", terrain_bins),
    ):
        bins[label] = {}
        for lo, hi in bin_defs:
            bucket = [
                row for row in feedback_rows
                if lo <= _safe_float(row.get(key), math.inf) < hi
            ]
            bins[label][f"[{lo:g},{hi:g})"] = _region_stats(bucket, mismatch_threshold, direction_threshold)

    return {
        "feedback_row_count": int(len(feedback_rows)),
        "thresholds": {
            "near_obstacle_surface_distance": float(obstacle_threshold),
            "near_terrain_clearance": float(terrain_threshold),
            "agent_agent_clearance": float(agent_close_threshold),
            "contact_before_window_steps": int(contact_window),
            "mismatch_threshold": float(mismatch_threshold),
            "direction_threshold": float(direction_threshold),
        },
        "regions": {
            name: _region_stats(items, mismatch_threshold, direction_threshold)
            for name, items in regions.items()
        },
        "bins": bins,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-root", required=True, help="Gazebo-live comparison root or gazebo_live result directory")
    parser.add_argument("--apf-dir", default=None, help="Directory containing gazebo_apf_live_metrics.csv, or the CSV path itself")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--out-dir", default="diagnostics/gazebo_live_safety")
    parser.add_argument("--speed-thresholds", default="0,0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--near-obstacle-threshold", type=float, default=5.0)
    parser.add_argument("--near-terrain-threshold", type=float, default=5.0)
    parser.add_argument("--agent-close-threshold", type=float, default=2.0)
    parser.add_argument("--contact-before-window", type=int, default=50)
    parser.add_argument("--mismatch-threshold", type=float, default=0.05)
    parser.add_argument("--direction-threshold", type=float, default=0.995)
    args = parser.parse_args()

    live_root = Path(args.live_root).expanduser().resolve()
    evaluation_results_path = _find_evaluation_results(live_root)
    data, ep = _load_episode(evaluation_results_path, args.episode_index)
    scenario_path = _find_scenario_path(ep, evaluation_results_path)
    scenario = _load_json(scenario_path)
    terrain_dense = _load_terrain_dense(scenario, scenario_path)
    apf_csv = _find_apf_csv(Path(args.apf_dir).expanduser().resolve() if args.apf_dir else None)
    apf_rows = _read_apf_rows(apf_csv)
    thresholds = [
        float(item.strip())
        for item in str(args.speed_thresholds).split(",")
        if item.strip()
    ]

    contact = _contact_diagnostics(ep, scenario, terrain_dense)
    adapter = _adapter_diagnostics(apf_rows, thresholds)
    apf_safety = _apf_safety_diagnostics(
        apf_rows,
        ep,
        scenario,
        obstacle_threshold=float(args.near_obstacle_threshold),
        terrain_threshold=float(args.near_terrain_threshold),
        agent_close_threshold=float(args.agent_close_threshold),
        contact_window=int(args.contact_before_window),
        mismatch_threshold=float(args.mismatch_threshold),
        direction_threshold=float(args.direction_threshold),
    )

    setup = data.get("evaluation_setup", {}) if isinstance(data.get("evaluation_setup"), Mapping) else {}
    run_metadata = {
        "live_root": str(live_root),
        "evaluation_results": str(evaluation_results_path),
        "scenario_json": str(scenario_path),
        "apf_live_metrics_csv": str(apf_csv) if apf_csv else None,
        "simulation_dt": setup.get("simulation_dt"),
        "gazebo_live_wall_time_step_ms": ep.get("gazebo_live_wall_time_step_ms"),
        "gazebo_live_step_iterations": ep.get("gazebo_live_step_iterations"),
        "gazebo_live_feedback_velocity_mode": ep.get("gazebo_live_feedback_velocity_mode"),
        "gazebo_live_feedback_acceleration_mode": ep.get("gazebo_live_feedback_acceleration_mode"),
        "gazebo_live_state_feedback_update_ratio": ep.get("gazebo_live_state_feedback_update_ratio"),
        "gazebo_live_authoritative_feedback_updates": ep.get("gazebo_live_authoritative_feedback_updates"),
    }
    summary = {
        "metadata": run_metadata,
        "contact": {
            "gazebo_contact_step": contact["gazebo_contact_step"],
            "gazebo_contact_count": contact["gazebo_contact_count"],
            "actual_collision_pair_records": contact["actual_collision_pair_records"],
            "contact_class_counts": contact["contact_class_counts"],
            "unique_collision_pairs": contact["unique_collision_pairs"][:5],
            "true_collision_evidence": contact["true_collision_evidence"],
            "only_expected_collision_links": contact["only_expected_collision_links"],
            "contact_count_interpretation": contact["contact_count_interpretation"],
        },
        "adapter": {
            "feedback_row_count": adapter["feedback_row_count"],
            "best_axis_transform_by_mean_error": adapter["best_axis_transform_by_mean_error"],
            "threshold_stats": adapter["threshold_stats"],
        },
        "apf_safety_regions": {
            name: {
                "sample_count": stats["sample_count"],
                "mismatch_count": stats["mismatch_count"],
                "mismatch_rate": stats["mismatch_rate"],
                "mean_corrected_error": stats["corrected_action_error"].get("mean"),
                "max_corrected_error": stats["corrected_action_error"].get("max"),
                "mean_pf_force_error": stats["pf_force_error"].get("mean"),
                "max_pf_force_error": stats["pf_force_error"].get("max"),
                "mean_direction_cosine": stats["direction_cosine_similarity"].get("mean"),
                "min_direction_cosine": stats["direction_cosine_similarity"].get("min"),
            }
            for name, stats in apf_safety["regions"].items()
        },
    }

    out_dir = Path(args.out_dir).expanduser().resolve()
    _write_json(out_dir / "gazebo_contact_diagnostics.json", contact)
    _write_json(out_dir / "adapter_velocity_diagnostics.json", adapter)
    _write_json(out_dir / "apf_safety_region_diagnostics.json", apf_safety)
    _write_json(out_dir / "gazebo_live_safety_summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote diagnostics to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
