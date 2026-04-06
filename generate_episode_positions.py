#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为每个episode生成固定的位置文件
确保所有评估模式（Oracle Same、Oracle Dense、Local）使用相同的初始条件
"""

import json
import numpy as np
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from multiagent.environment import MultiAgentEnv
    from multiagent.scenarios import load
except ImportError:
    # 如果导入失败，尝试添加项目路径
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from multiagent.environment import MultiAgentEnv
    from multiagent.scenarios import load


def _env_float(env_vars, key, default):
    try:
        return float(env_vars.get(key, default))
    except Exception:
        return float(default)


def _env_flag(env_vars, key, default=False):
    try:
        raw = env_vars.get(key, default)
    except Exception:
        raw = default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _extract_terrain_setup(env_vars):
    try:
        terrain_base_seed = int(float(env_vars.get("TERRAIN_BASE_SEED", env_vars.get("SCENARIO_SEED", 67))))
    except Exception:
        terrain_base_seed = 67
    try:
        terrain_variant_seed = int(float(env_vars.get("TERRAIN_VARIANT_SEED", terrain_base_seed)))
    except Exception:
        terrain_variant_seed = terrain_base_seed
    peak_jitter_range = float(_env_float(env_vars, "PEAK_JITTER_RANGE", 15.0))
    return {
        "semi_random_terrain": bool(_env_flag(env_vars, "SEMI_RANDOM_TERRAIN", False)),
        "terrain_base_seed": int(terrain_base_seed),
        "terrain_variant_seed": int(terrain_variant_seed),
        "peak_jitter_range": float(peak_jitter_range),
        "peak_center_jitter_range": float(_env_float(env_vars, "PEAK_CENTER_JITTER_RANGE", min(peak_jitter_range, 3.0))),
        "peak_height_jitter_ratio_min": float(_env_float(env_vars, "PEAK_HEIGHT_JITTER_RATIO_MIN", 0.20)),
        "peak_height_jitter_ratio_max": float(_env_float(env_vars, "PEAK_HEIGHT_JITTER_RATIO_MAX", 0.40)),
        "peak_height_max_scale": float(_env_float(env_vars, "PEAK_HEIGHT_MAX_SCALE", 1.30)),
        "terrain_variant_noise_ratio": float(_env_float(env_vars, "TERRAIN_VARIANT_NOISE_RATIO", 0.15)),
    }


def _episode_positions_filename(episode_idx, terrain_seed=None, terrain_variant_seed=None):
    episode_idx = int(episode_idx)
    terrain_seed = None if terrain_seed is None else int(terrain_seed)
    terrain_variant_seed = None if terrain_variant_seed is None else int(terrain_variant_seed)
    if terrain_variant_seed is not None:
        return f"episode_{episode_idx:03d}_seed_{terrain_seed}_variant_{terrain_variant_seed}.json"
    if terrain_seed is not None:
        return f"episode_{episode_idx:03d}_seed_{terrain_seed}.json"
    return f"episode_{episode_idx:03d}.json"


def _sample_disc_offset(rng, radius):
    radius = max(0.0, float(radius))
    if radius <= 1e-9:
        return np.zeros(2, dtype=np.float32)
    theta = rng.uniform(0.0, 2.0 * np.pi)
    r = radius * np.sqrt(rng.uniform(0.0, 1.0))
    return np.array([np.cos(theta) * r, np.sin(theta) * r], dtype=np.float32)


def _load_reference_positions(reference_path):
    if not reference_path:
        return None
    path = Path(reference_path)
    if not path.is_absolute():
        path = (Path(__file__).parent / path).resolve()
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return None
    agents = data.get("agents")
    goal = data.get("goal")
    if not isinstance(agents, list) or not agents or goal is None:
        return None
    try:
        agents_arr = np.asarray(agents, dtype=np.float32)
        goal_arr = np.asarray(goal, dtype=np.float32)
    except Exception:
        return None
    if agents_arr.ndim != 2 or agents_arr.shape[1] < 2 or goal_arr.size < 2:
        return None
    return {
        "path": str(path),
        "agents": agents_arr,
        "goal": goal_arr,
    }


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def _estimate_local_surface_stats(scenario, xy, radius=4.0, grid_points=5):
    xy = np.asarray(xy, dtype=np.float32).reshape(-1)
    if xy.size < 2:
        return {
            "terrain_height": 0.0,
            "height_span": float("inf"),
            "height_std": float("inf"),
            "slope_mag": float("inf"),
        }

    x = float(xy[0])
    y = float(xy[1])
    sample_count = max(3, int(grid_points))
    offsets = np.linspace(-float(radius), float(radius), sample_count)
    heights = []
    for dx in offsets:
        for dy in offsets:
            heights.append(float(scenario.get_terrain_height(x + dx, y + dy)))
    center_height = float(scenario.get_terrain_height(x, y))
    delta = max(1.0, float(radius) * 0.5)
    try:
        h_x_plus = float(scenario.get_terrain_height(x + delta, y))
        h_x_minus = float(scenario.get_terrain_height(x - delta, y))
        h_y_plus = float(scenario.get_terrain_height(x, y + delta))
        h_y_minus = float(scenario.get_terrain_height(x, y - delta))
        dz_dx = (h_x_plus - h_x_minus) / (2.0 * delta)
        dz_dy = (h_y_plus - h_y_minus) / (2.0 * delta)
        slope_mag = float(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    except Exception:
        slope_mag = float("inf")

    return {
        "terrain_height": center_height,
        "height_span": float(max(heights) - min(heights)) if heights else float("inf"),
        "height_std": float(np.std(heights)) if heights else float("inf"),
        "slope_mag": slope_mag,
    }


def _collect_flat_area_candidates(scenario):
    if not hasattr(scenario, "find_flat_area"):
        return []

    search_plan = (
        (0.0, 8.0, 8),
        (0.0, 12.0, 8),
        (0.0, 20.0, 8),
        (0.0, 28.0, 6),
    )
    collected = []
    seen_centers = set()
    for min_height, max_height, min_area_size in search_plan:
        try:
            areas = scenario.find_flat_area(
                min_height=min_height,
                max_height=max_height,
                min_area_size=min_area_size,
            )
        except Exception:
            areas = []
        for area in areas:
            center = np.asarray(area.get("center", ()), dtype=np.float32).reshape(-1)
            if center.size < 2:
                continue
            center_key = (round(float(center[0]), 3), round(float(center[1]), 3))
            if center_key in seen_centers:
                continue
            stats = _estimate_local_surface_stats(scenario, center[:2], radius=max(3.0, float(min_area_size) / 2.0))
            if not np.isfinite(stats["slope_mag"]) or not np.isfinite(stats["height_span"]):
                continue
            candidate = dict(area)
            candidate["center"] = (float(center[0]), float(center[1]))
            candidate["safety"] = stats
            candidate["score"] = (
                float(candidate.get("height", 0.0)) * 1.5
                + float(candidate.get("variance", 0.0)) * 2.0
                + float(candidate.get("range", 0.0)) * 2.5
                + float(stats["height_span"]) * 3.0
                + float(stats["height_std"]) * 4.0
                + float(stats["slope_mag"]) * 50.0
            )
            collected.append(candidate)
            seen_centers.add(center_key)

    collected.sort(key=lambda item: (float(item.get("score", 0.0)), float(item.get("height", 0.0))))
    return collected


def _select_distinct_flat_candidates(flat_candidates, target_xys):
    if not flat_candidates or not target_xys:
        return None

    remaining_indices = set(range(len(flat_candidates)))
    chosen = []
    for target_xy in target_xys:
        if not remaining_indices:
            return None
        target_xy = np.asarray(target_xy, dtype=np.float32).reshape(-1)[:2]
        ranked = []
        for idx in remaining_indices:
            candidate = flat_candidates[idx]
            center = np.asarray(candidate["center"], dtype=np.float32)
            dist = float(np.linalg.norm(center[:2] - target_xy[:2]))
            ranked.append((dist, float(candidate.get("score", 0.0)), idx))
        ranked.sort(key=lambda item: (item[0], item[1]))
        _, _, best_idx = ranked[0]
        chosen.append(flat_candidates[best_idx])
        remaining_indices.remove(best_idx)
    return chosen


def _choose_start_cluster(
    flat_candidates,
    desired_center_xy,
    num_agents,
    cluster_radius,
    min_pairwise_spacing=0.0,
):
    if not flat_candidates:
        return None

    desired_center_xy = np.asarray(desired_center_xy, dtype=np.float32).reshape(-1)[:2]
    min_pairwise_spacing = max(0.0, float(min_pairwise_spacing))
    best_cluster = None
    best_score = None
    for anchor in flat_candidates:
        anchor_xy = np.asarray(anchor["center"], dtype=np.float32)
        neighbours = []
        for candidate in flat_candidates:
            candidate_xy = np.asarray(candidate["center"], dtype=np.float32)
            dist = float(np.linalg.norm(candidate_xy[:2] - anchor_xy[:2]))
            if dist <= cluster_radius:
                neighbours.append((dist, candidate))
        if len(neighbours) < num_agents:
            continue

        neighbours.sort(key=lambda item: (item[0], float(item[1].get("score", 0.0))))
        selected = [candidate for _, candidate in neighbours[:num_agents]]
        cluster_centers = np.asarray([candidate["center"] for candidate in selected], dtype=np.float32)
        cluster_pairwise_stats = _pairwise_xy_stats(cluster_centers[:, :2])
        cluster_min_spacing = cluster_pairwise_stats["min"]
        if (
            min_pairwise_spacing > 0.0
            and cluster_min_spacing is not None
            and float(cluster_min_spacing) < min_pairwise_spacing
        ):
            continue
        cluster_center = np.mean(cluster_centers[:, :2], axis=0)
        cluster_height = float(np.mean([candidate.get("height", 0.0) for candidate in selected]))
        cluster_score = (
            float(np.linalg.norm(cluster_center - desired_center_xy)) * 1.0
            + cluster_height * 1.8
            + float(np.mean([candidate.get("score", 0.0) for candidate in selected])) * 0.25
            + float(np.mean([np.linalg.norm(np.asarray(candidate["center"], dtype=np.float32)[:2] - desired_center_xy) for candidate in selected])) * 0.15
        )
        if best_score is None or cluster_score < best_score:
            best_score = cluster_score
            best_cluster = selected

    return best_cluster


def _fallback_flat_candidate(flat_candidates, desired_xy, used_centers=None):
    if not flat_candidates:
        return None
    used_centers = used_centers or set()
    desired_xy = np.asarray(desired_xy, dtype=np.float32).reshape(-1)[:2]
    ranked = []
    for candidate in flat_candidates:
        center_key = tuple(np.asarray(candidate["center"], dtype=np.float32)[:2].tolist())
        if center_key in used_centers:
            continue
        center = np.asarray(candidate["center"], dtype=np.float32)
        dist = float(np.linalg.norm(center[:2] - desired_xy[:2]))
        ranked.append((dist, float(candidate.get("score", 0.0)), candidate))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked[0][2]


def _pairwise_xy_stats(points_xy):
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
        return {"min": None, "mean": None, "max": None}
    dists = []
    for i in range(pts.shape[0]):
        for j in range(i + 1, pts.shape[0]):
            dists.append(float(np.linalg.norm(pts[i, :2] - pts[j, :2])))
    if not dists:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": float(min(dists)),
        "mean": float(np.mean(dists)),
        "max": float(max(dists)),
    }


def _rotate_vector_xy(vec_xy, angle_rad):
    vec_xy = np.asarray(vec_xy, dtype=np.float32).reshape(-1)
    if vec_xy.size < 2:
        return np.zeros(2, dtype=np.float32)
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    return np.array(
        [
            vec_xy[0] * cos_a - vec_xy[1] * sin_a,
            vec_xy[0] * sin_a + vec_xy[1] * cos_a,
        ],
        dtype=np.float32,
    )


def _scenario_peak_centers_xy(scenario):
    raw_centers = getattr(scenario, "mountain_centers", None)
    if not raw_centers:
        return np.zeros((0, 2), dtype=np.float32)
    centers = []
    for center in raw_centers:
        try:
            center_arr = np.asarray(center, dtype=np.float32).reshape(-1)
        except Exception:
            continue
        if center_arr.size < 2:
            continue
        centers.append(center_arr[:2])
    if not centers:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(centers, dtype=np.float32)


def _nearest_peak_distance(scenario, xy):
    peak_centers = _scenario_peak_centers_xy(scenario)
    if peak_centers.shape[0] == 0:
        return float("inf")
    xy = np.asarray(xy, dtype=np.float32).reshape(-1)[:2]
    dists = np.linalg.norm(peak_centers[:, :2] - xy[None, :2], axis=1)
    if dists.size == 0:
        return float("inf")
    return float(np.min(dists))


def _evaluate_same_region_goal_candidate(
    scenario,
    candidate_xy,
    start_center_xy,
    desired_goal_xy,
    ref_goal_xy,
    ref_route_unit,
    ref_distance,
    min_goal_distance,
    max_goal_distance,
    min_peak_clearance,
    max_surface_span,
    max_slope_mag,
    min_direction_cos,
    base_score=0.0,
):
    candidate_xy = np.asarray(candidate_xy, dtype=np.float32).reshape(-1)
    if candidate_xy.size < 2:
        return None

    start_center_xy = np.asarray(start_center_xy, dtype=np.float32).reshape(-1)[:2]
    desired_goal_xy = np.asarray(desired_goal_xy, dtype=np.float32).reshape(-1)[:2]
    ref_goal_xy = np.asarray(ref_goal_xy, dtype=np.float32).reshape(-1)[:2]

    route_vec = candidate_xy[:2] - start_center_xy[:2]
    route_distance = float(np.linalg.norm(route_vec))
    if route_distance < max(1e-6, float(min_goal_distance)):
        return None
    if route_distance > float(max_goal_distance):
        return None

    direction_cosine = 1.0
    if ref_route_unit is not None:
        ref_route_unit = np.asarray(ref_route_unit, dtype=np.float32).reshape(-1)[:2]
        if ref_route_unit.size >= 2:
            route_unit = route_vec / max(route_distance, 1e-6)
            direction_cosine = float(np.dot(route_unit[:2], ref_route_unit[:2]))
            if direction_cosine < float(min_direction_cos):
                return None

    surface_stats = _estimate_local_surface_stats(scenario, candidate_xy[:2], radius=5.0)
    terrain_height = float(surface_stats["terrain_height"])
    if terrain_height > 40.0:
        return None
    if float(surface_stats["height_span"]) > float(max_surface_span):
        return None
    if float(surface_stats["slope_mag"]) > float(max_slope_mag):
        return None

    peak_clearance = _nearest_peak_distance(scenario, candidate_xy[:2])
    if peak_clearance < float(min_peak_clearance):
        return None

    distance_to_anchor = float(np.linalg.norm(candidate_xy[:2] - desired_goal_xy[:2]))
    distance_to_reference_goal = float(np.linalg.norm(candidate_xy[:2] - ref_goal_xy[:2]))
    distance_deviation = abs(route_distance - float(ref_distance))
    score = (
        float(base_score)
        + distance_to_anchor * 1.15
        + distance_to_reference_goal * 0.35
        + distance_deviation * 0.75
        + max(0.0, 0.95 - direction_cosine) * 60.0
        + float(surface_stats["height_span"]) * 3.0
        + float(surface_stats["height_std"]) * 4.0
        + float(surface_stats["slope_mag"]) * 55.0
        + terrain_height * 0.45
        - min(peak_clearance, 80.0) * 0.4
    )
    return {
        "xy": np.asarray(candidate_xy[:2], dtype=np.float32),
        "score": float(score),
        "surface_stats": surface_stats,
        "peak_clearance": float(peak_clearance),
        "route_distance": float(route_distance),
        "direction_cosine": float(direction_cosine),
        "distance_to_anchor": float(distance_to_anchor),
        "distance_to_reference_goal": float(distance_to_reference_goal),
    }


def _generate_same_region_goal_samples(
    rng,
    start_center_xy,
    desired_goal_xy,
    ref_route_unit,
    ref_distance,
    min_goal_distance,
    max_goal_distance,
    goal_region_radius,
    map_size,
):
    start_center_xy = np.asarray(start_center_xy, dtype=np.float32).reshape(-1)[:2]
    desired_goal_xy = np.asarray(desired_goal_xy, dtype=np.float32).reshape(-1)[:2]
    samples = [desired_goal_xy.copy()]

    local_radii = [
        0.0,
        goal_region_radius * 0.35,
        goal_region_radius * 0.7,
        goal_region_radius * 1.1,
        max(goal_region_radius * 1.6, 24.0),
    ]
    local_angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    for radius in local_radii:
        if radius <= 1e-6:
            continue
        for angle in local_angles:
            offset = np.array([np.cos(angle) * radius, np.sin(angle) * radius], dtype=np.float32)
            samples.append(desired_goal_xy + offset)

    if ref_route_unit is not None and ref_distance > 1e-6:
        route_low = max(float(min_goal_distance), float(ref_distance) * 0.86)
        route_high = min(float(max_goal_distance), float(ref_distance) * 1.12)
        if route_high > route_low + 1e-6:
            route_distances = np.linspace(route_low, route_high, 6)
        else:
            route_distances = np.asarray([route_low], dtype=np.float32)
        angle_offsets = np.deg2rad(np.asarray([-30.0, -20.0, -12.0, -6.0, 0.0, 6.0, 12.0, 20.0, 30.0], dtype=np.float32))
        for route_distance in route_distances:
            for angle in angle_offsets:
                direction = _rotate_vector_xy(ref_route_unit, float(angle))
                samples.append(start_center_xy + direction[:2] * float(route_distance))
        for _ in range(20):
            angle = float(rng.uniform(np.deg2rad(-24.0), np.deg2rad(24.0)))
            route_distance = float(rng.uniform(route_low, route_high))
            direction = _rotate_vector_xy(ref_route_unit, angle)
            samples.append(start_center_xy + direction[:2] * route_distance)

    deduped = []
    seen = set()
    for sample_xy in samples:
        sample_xy = np.asarray(sample_xy, dtype=np.float32).reshape(-1)[:2]
        sample_xy = np.clip(sample_xy, 8.0, float(map_size) - 8.0)
        key = (round(float(sample_xy[0]), 3), round(float(sample_xy[1]), 3))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sample_xy)
    return deduped


def _select_distinct_flat_candidates_with_spacing(
    flat_candidates,
    target_xys,
    min_pairwise_spacing,
):
    if not flat_candidates or not target_xys:
        return None

    target_arr = np.asarray(target_xys, dtype=np.float32)
    if target_arr.ndim != 2 or target_arr.shape[1] < 2:
        return None

    desired_center = np.mean(target_arr[:, :2], axis=0)
    chosen = [None] * len(target_xys)
    chosen_centers = []
    remaining_indices = set(range(len(flat_candidates)))

    # 先给离队形中心最远的目标分配位置，减少冲突。
    order = sorted(
        range(len(target_xys)),
        key=lambda idx: float(np.linalg.norm(target_arr[idx, :2] - desired_center)),
        reverse=True,
    )

    for target_idx in order:
        target_xy = target_arr[target_idx, :2]
        ranked = []
        for candidate_idx in remaining_indices:
            candidate = flat_candidates[candidate_idx]
            center = np.asarray(candidate["center"], dtype=np.float32)[:2]
            if chosen_centers:
                nearest_chosen = min(float(np.linalg.norm(center - prev_center)) for prev_center in chosen_centers)
                if nearest_chosen < float(min_pairwise_spacing):
                    continue
            else:
                nearest_chosen = float("inf")
            dist_to_target = float(np.linalg.norm(center - target_xy))
            ranked.append(
                (
                    dist_to_target,
                    float(candidate.get("score", 0.0)),
                    -nearest_chosen,
                    candidate_idx,
                )
            )
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        best_idx = ranked[0][3]
        chosen[target_idx] = flat_candidates[best_idx]
        remaining_indices.remove(best_idx)
        chosen_centers.append(np.asarray(flat_candidates[best_idx]["center"], dtype=np.float32)[:2])

    return chosen


def _apply_same_region_positions(scenario, position_seed, positions_data, env_vars):
    reference = _load_reference_positions(env_vars.get("HELDOUT_REFERENCE_POSITIONS_FILE"))
    if reference is None:
        return positions_data, None

    position_mode = str(env_vars.get("HELDOUT_POSITION_MODE", "")).strip().lower()
    if position_mode != "same_region":
        return positions_data, None

    rng = np.random.RandomState(int(position_seed) + 7919)
    map_size = float(env_vars.get("MAP_SIZE", 200))
    start_altitude_offset = _env_float(env_vars, "START_ALTITUDE_OFFSET", 7.0)
    goal_altitude = _env_float(env_vars, "GOAL_ALTITUDE", 12.0)
    start_center_jitter = _env_float(env_vars, "HELDOUT_START_CENTER_JITTER", 12.0)
    agent_local_jitter = _env_float(env_vars, "HELDOUT_AGENT_LOCAL_JITTER", 3.0)
    goal_region_radius = _env_float(env_vars, "HELDOUT_GOAL_REGION_RADIUS", 18.0)

    ref_agents = reference["agents"]
    ref_goal = reference["goal"]
    ref_center_xy = np.mean(ref_agents[:, :2], axis=0)
    ref_offsets_xy = ref_agents[:, :2] - ref_center_xy
    ref_goal_xy = ref_goal[:2]
    ref_goal_z = float(ref_goal[2]) if ref_goal.size >= 3 else None
    flat_candidates = _collect_flat_area_candidates(scenario)

    center_shift = _sample_disc_offset(rng, start_center_jitter)
    desired_center_xy = np.clip(ref_center_xy + center_shift, 6.0, map_size - 6.0)

    agent_positions = []
    num_agents = int(ref_agents.shape[0])
    formation_radius = float(np.max(np.linalg.norm(ref_offsets_xy, axis=1))) if ref_offsets_xy.size else 0.0
    cluster_radius = max(18.0, formation_radius + agent_local_jitter + 10.0)
    target_xys = []
    for idx in range(num_agents):
        local_shift = _sample_disc_offset(rng, agent_local_jitter)
        target_xy = np.clip(desired_center_xy + ref_offsets_xy[idx] + local_shift, 6.0, map_size - 6.0)
        target_xys.append(target_xy)

    preferred_start_candidates = [candidate for candidate in flat_candidates if float(candidate.get("height", 0.0)) <= 18.0]
    if len(preferred_start_candidates) < num_agents:
        preferred_start_candidates = flat_candidates

    ref_pairwise_stats = _pairwise_xy_stats(ref_agents[:, :2])
    reference_min_spacing = ref_pairwise_stats["min"] if ref_pairwise_stats["min"] is not None else 0.0
    # heldout same_region 仍然应该保持与训练参考队形接近的编队间距。
    # 旧逻辑只要求参考最小间距的 50%，会让 17m 级别的训练编队坍缩到 8m 左右，
    # 实际生成时甚至会退化到 1-3m 的抱团起点。
    min_pairwise_spacing = max(10.0, float(reference_min_spacing) * 0.85)

    assigned_candidates = _choose_start_cluster(
        preferred_start_candidates,
        desired_center_xy,
        num_agents,
        cluster_radius=max(16.0, formation_radius + agent_local_jitter + 6.0),
        min_pairwise_spacing=min_pairwise_spacing,
    )
    selection_strategy = "center_preserving_safe_cluster"
    if assigned_candidates is not None:
        reordered_candidates = _select_distinct_flat_candidates(assigned_candidates, target_xys)
        if reordered_candidates is not None:
            assigned_candidates = reordered_candidates
        if assigned_candidates is not None:
            assigned_xy = np.asarray([candidate["center"] for candidate in assigned_candidates], dtype=np.float32)[:, :2]
            assigned_pairwise = _pairwise_xy_stats(assigned_xy)
            assigned_min_spacing = assigned_pairwise["min"]
            if (
                assigned_min_spacing is None
                or float(assigned_min_spacing) < min_pairwise_spacing
            ):
                assigned_candidates = None
    if assigned_candidates is None:
        for spacing_scale in (1.0, 0.85, 0.7):
            assigned_candidates = _select_distinct_flat_candidates_with_spacing(
                preferred_start_candidates,
                target_xys,
                min_pairwise_spacing=min_pairwise_spacing * spacing_scale,
            )
            if assigned_candidates is not None:
                selection_strategy = f"formation_projection_spacing_{spacing_scale:.2f}"
                break
    if assigned_candidates is None:
        assigned_candidates = _select_distinct_flat_candidates(preferred_start_candidates, target_xys)
        if assigned_candidates is not None:
            selection_strategy = "formation_projection_preferred"
    if assigned_candidates is None:
        for spacing_scale in (1.0, 0.85, 0.7):
            assigned_candidates = _select_distinct_flat_candidates_with_spacing(
                flat_candidates,
                target_xys,
                min_pairwise_spacing=min_pairwise_spacing * spacing_scale,
            )
            if assigned_candidates is not None:
                selection_strategy = f"global_projection_spacing_{spacing_scale:.2f}"
                break
    if assigned_candidates is None:
        assigned_candidates = _select_distinct_flat_candidates(flat_candidates, target_xys)
        if assigned_candidates is not None:
            selection_strategy = "global_projection"

    used_centers = set()
    for idx in range(num_agents):
        candidate = assigned_candidates[idx] if assigned_candidates and idx < len(assigned_candidates) else None
        if candidate is None:
            candidate = _fallback_flat_candidate(preferred_start_candidates, target_xys[idx], used_centers=used_centers)
        if candidate is None:
            candidate = _fallback_flat_candidate(flat_candidates, target_xys[idx], used_centers=used_centers)
        if candidate is not None:
            center = np.asarray(candidate["center"], dtype=np.float32)
            agent_xy = np.clip(center[:2], 6.0, map_size - 6.0)
            used_centers.add(tuple(agent_xy.tolist()))
        else:
            agent_xy = target_xys[idx]
        terrain_h = float(scenario.get_terrain_height(agent_xy[0], agent_xy[1]))
        agent_z = terrain_h + max(1.0, start_altitude_offset)
        agent_positions.append([float(agent_xy[0]), float(agent_xy[1]), float(agent_z)])

    assigned_center_xy = np.mean(np.asarray(agent_positions, dtype=np.float32)[:, :2], axis=0)
    assigned_center_drift = float(np.linalg.norm(assigned_center_xy - ref_center_xy))
    desired_center_drift = float(np.linalg.norm(desired_center_xy - ref_center_xy))

    goal_xy = None
    desired_goal_xy = None
    ref_route_xy = ref_goal_xy - ref_center_xy
    ref_distance = float(np.linalg.norm(ref_route_xy))
    ref_route_unit = None
    if ref_distance > 1e-6:
        ref_route_unit = (ref_route_xy / ref_distance).astype(np.float32)

    start_center_now = np.mean(np.asarray(agent_positions, dtype=np.float32)[:, :2], axis=0)
    min_goal_distance = max(55.0, ref_distance * 0.82)
    max_goal_distance = max(min_goal_distance + 12.0, ref_distance * 1.22)
    for _ in range(40):
        candidate_xy = np.clip(ref_goal_xy + _sample_disc_offset(rng, goal_region_radius), 8.0, map_size - 8.0)
        if np.linalg.norm(candidate_xy - start_center_now) >= min_goal_distance:
            desired_goal_xy = candidate_xy
            break
    if desired_goal_xy is None:
        desired_goal_xy = np.clip(ref_goal_xy, 8.0, map_size - 8.0)

    candidate_pool = []
    seen_goal_candidates = set()

    def _push_goal_candidate(xy, source, base_score=0.0):
        xy = np.asarray(xy, dtype=np.float32).reshape(-1)
        if xy.size < 2:
            return
        clipped_xy = np.clip(xy[:2], 8.0, map_size - 8.0)
        key = (round(float(clipped_xy[0]), 3), round(float(clipped_xy[1]), 3))
        if key in seen_goal_candidates:
            return
        seen_goal_candidates.add(key)
        candidate_pool.append(
            {
                "xy": clipped_xy.astype(np.float32),
                "source": str(source),
                "base_score": float(base_score),
            }
        )

    _push_goal_candidate(desired_goal_xy, "desired_goal_anchor", base_score=0.0)
    _push_goal_candidate(ref_goal_xy, "reference_goal", base_score=2.0)
    for candidate in flat_candidates:
        center = np.asarray(candidate.get("center", ()), dtype=np.float32).reshape(-1)
        if center.size < 2:
            continue
        _push_goal_candidate(center[:2], "flat_candidate", base_score=float(candidate.get("score", 0.0)) * 0.15)
    for sample_xy in _generate_same_region_goal_samples(
        rng=rng,
        start_center_xy=start_center_now,
        desired_goal_xy=desired_goal_xy,
        ref_route_unit=ref_route_unit,
        ref_distance=ref_distance,
        min_goal_distance=min_goal_distance,
        max_goal_distance=max_goal_distance,
        goal_region_radius=goal_region_radius,
        map_size=map_size,
    ):
        _push_goal_candidate(sample_xy, "route_sector_sample", base_score=0.0)

    goal_selection_strategy = "reference_goal_fallback"
    goal_selection_tier = "fallback"
    goal_candidate_source = "reference_goal"
    chosen_goal_candidate = None
    base_peak_clearance_target = max(22.0, min(34.0, goal_region_radius + 8.0))
    goal_selection_tiers = (
        {
            "name": "strict_route_safe_goal",
            "min_peak_clearance": base_peak_clearance_target,
            "max_surface_span": 10.0,
            "max_slope_mag": 1.25,
            "min_direction_cos": 0.84,
        },
        {
            "name": "balanced_route_safe_goal",
            "min_peak_clearance": max(18.0, base_peak_clearance_target * 0.85),
            "max_surface_span": 13.0,
            "max_slope_mag": 1.65,
            "min_direction_cos": 0.72,
        },
        {
            "name": "relaxed_route_safe_goal",
            "min_peak_clearance": 16.0,
            "max_surface_span": 16.0,
            "max_slope_mag": 2.0,
            "min_direction_cos": 0.58,
        },
    )

    for tier in goal_selection_tiers:
        ranked_goal_candidates = []
        for candidate in candidate_pool:
            evaluated = _evaluate_same_region_goal_candidate(
                scenario=scenario,
                candidate_xy=candidate["xy"],
                start_center_xy=start_center_now,
                desired_goal_xy=desired_goal_xy,
                ref_goal_xy=ref_goal_xy,
                ref_route_unit=ref_route_unit,
                ref_distance=ref_distance,
                min_goal_distance=min_goal_distance,
                max_goal_distance=max_goal_distance,
                min_peak_clearance=tier["min_peak_clearance"],
                max_surface_span=tier["max_surface_span"],
                max_slope_mag=tier["max_slope_mag"],
                min_direction_cos=tier["min_direction_cos"],
                base_score=candidate["base_score"],
            )
            if evaluated is None:
                continue
            evaluated["source"] = candidate["source"]
            evaluated["tier"] = tier["name"]
            ranked_goal_candidates.append(evaluated)
        if ranked_goal_candidates:
            ranked_goal_candidates.sort(
                key=lambda item: (
                    float(item["score"]),
                    float(item["distance_to_anchor"]),
                    float(item["distance_to_reference_goal"]),
                )
            )
            chosen_goal_candidate = ranked_goal_candidates[0]
            break

    if chosen_goal_candidate is not None:
        goal_xy = np.asarray(chosen_goal_candidate["xy"], dtype=np.float32)
        goal_selection_strategy = "route_safe_goal_candidate"
        goal_selection_tier = str(chosen_goal_candidate.get("tier", "selected"))
        goal_candidate_source = str(chosen_goal_candidate.get("source", "unknown"))
    else:
        goal_xy = np.clip(desired_goal_xy, 8.0, map_size - 8.0)

    goal_terrain_h = float(scenario.get_terrain_height(goal_xy[0], goal_xy[1]))
    goal_z = goal_terrain_h + goal_altitude
    if ref_goal_z is not None:
        goal_z = max(goal_z, ref_goal_z)

    start_surface_stats = []
    for pos in agent_positions:
        start_surface_stats.append(
            _estimate_local_surface_stats(scenario, np.asarray(pos[:2], dtype=np.float32), radius=4.0)
        )
    if chosen_goal_candidate is not None:
        goal_surface_stats = dict(chosen_goal_candidate.get("surface_stats", {}))
        goal_peak_clearance = float(chosen_goal_candidate.get("peak_clearance", _nearest_peak_distance(scenario, goal_xy)))
        goal_route_distance = float(chosen_goal_candidate.get("route_distance", np.linalg.norm(goal_xy - start_center_now[:2])))
        goal_direction_cosine = float(chosen_goal_candidate.get("direction_cosine", 1.0))
    else:
        goal_surface_stats = _estimate_local_surface_stats(scenario, goal_xy, radius=5.0)
        goal_peak_clearance = _nearest_peak_distance(scenario, goal_xy)
        goal_route_distance = float(np.linalg.norm(goal_xy[:2] - start_center_now[:2]))
        if ref_route_unit is not None and goal_route_distance > 1e-6:
            goal_direction_cosine = float(
                np.dot((goal_xy[:2] - start_center_now[:2]) / goal_route_distance, ref_route_unit[:2])
            )
        else:
            goal_direction_cosine = 1.0
    assigned_pairwise_stats = _pairwise_xy_stats(np.asarray(agent_positions, dtype=np.float32)[:, :2])

    metadata = {
        "position_family": "same_region",
        "reference_positions_file": reference["path"],
        "terrain_seed": int(positions_data.get("terrain_seed", position_seed)),
        "terrain_variant_seed": int(position_seed),
        "start_center_jitter": float(start_center_jitter),
        "agent_local_jitter": float(agent_local_jitter),
        "goal_region_radius": float(goal_region_radius),
        "reference_start_center": ref_center_xy.tolist(),
        "desired_start_center": desired_center_xy.tolist(),
        "assigned_start_center": assigned_center_xy.tolist(),
        "desired_center_drift": desired_center_drift,
        "assigned_center_drift": assigned_center_drift,
        "reference_goal_xy": ref_goal_xy.tolist(),
        "desired_goal_xy": desired_goal_xy.tolist() if desired_goal_xy is not None else ref_goal_xy.tolist(),
        "assigned_goal_xy": goal_xy.tolist(),
        "goal_center_distance_from_reference": float(np.linalg.norm(goal_xy - ref_goal_xy)),
        "goal_selection_strategy": goal_selection_strategy,
        "goal_selection_tier": goal_selection_tier,
        "goal_candidate_source": goal_candidate_source,
        "selection_strategy": selection_strategy,
        "start_cluster_radius": float(cluster_radius),
        "flat_candidate_count": int(len(flat_candidates)),
        "goal_candidate_pool_size": int(len(candidate_pool)),
        "reference_goal_distance": float(ref_distance),
        "goal_distance_band": [float(min_goal_distance), float(max_goal_distance)],
        "reference_pairwise_xy": ref_pairwise_stats,
        "assigned_pairwise_xy": assigned_pairwise_stats,
        "min_pairwise_spacing_target": float(min_pairwise_spacing),
        "start_surface_stats": start_surface_stats,
        "goal_surface_stats": goal_surface_stats,
        "goal_peak_clearance": float(goal_peak_clearance),
        "goal_route_distance": float(goal_route_distance),
        "goal_direction_cosine": float(goal_direction_cosine),
    }

    positions_data = dict(positions_data)
    positions_data["agents"] = agent_positions
    positions_data["goal"] = [float(goal_xy[0]), float(goal_xy[1]), float(goal_z)]
    positions_data["position_setup"] = metadata
    positions_data["heldout_metadata"] = metadata
    return positions_data, metadata


def generate_episode_positions(
    terrain_seed,
    episode_idx,
    output_dir,
    base_env_vars=None,
    terrain_variant_seed=None,
):
    """
    为单个episode生成固定的位置文件
    
    Args:
        terrain_seed: 基准地形种子
        episode_idx: episode索引（用于文件名）
        output_dir: 输出目录
        base_env_vars: 基础环境变量字典
        terrain_variant_seed: 同源扰动种子；None 时回退到 terrain_seed
    
    Returns:
        位置文件路径
    """
    if base_env_vars is None:
        base_env_vars = {}
    
    # 设置环境变量
    env_vars = os.environ.copy()
    env_vars.update(base_env_vars)
    
    try:
        terrain_seed = int(terrain_seed)
    except Exception:
        terrain_seed = 67
    if terrain_variant_seed is None:
        terrain_variant_seed = terrain_seed
    try:
        terrain_variant_seed = int(terrain_variant_seed)
    except Exception:
        terrain_variant_seed = terrain_seed

    # 设置地形种子
    env_vars["SCENARIO_SEED"] = str(terrain_seed)
    env_vars["USE_SCENARIO_SEED"] = "1"
    env_vars["RANDOM_TERRAIN"] = "0"
    env_vars["PER_EPISODE_TERRAIN"] = "0"
    env_vars["USE_FIXED_POSITIONS"] = "0"  # 禁用固定位置，让系统生成
    env_vars["DYNAMIC_FIRST_TIME"] = "1"  # 启用动态首次，生成位置
    env_vars["TERRAIN_VARIANT_SEED"] = str(terrain_variant_seed)
    
    # 设置其他必要的环境变量
    env_vars.setdefault("MAP_SIZE", "200")
    env_vars.setdefault("TERRAIN_COMPLEXITY_LEVEL", "3")
    env_vars.setdefault("MOUNTAIN_MIN_DISTANCE", "55")
    
    # 临时设置环境变量
    original_env = {}
    for key, value in env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = str(value)
    
    try:
        # 解析关键配置，确保与评估/训练一致
        try:
            terrain_level = int(env_vars.get("TERRAIN_COMPLEXITY_LEVEL", 3))
        except Exception:
            terrain_level = 3
        try:
            map_size = float(env_vars.get("MAP_SIZE", 200))
        except Exception:
            map_size = 200.0

        # 加载场景（与评估保持一致）
        scenario = load("paper3d_terrain_weighted").Scenario(
            seed=int(terrain_seed),
            use_fixed_positions=False,
            dynamic_first_time=True,
            fixed_positions_file=None,
            random_terrain=False,
            terrain_complexity_level=terrain_level,
            map_size=map_size,
        )
        world = scenario.make_world()
        
        # 重置世界（这会生成初始位置）
        scenario.reset_world(world)
        
        # 🚨 关键修复：验证并调整智能体位置，确保所有智能体都在地形上方
        # 原因：reset_world后，智能体位置应该已经在地形上方，但为了确保，再次验证
        altitude_offset = float(env_vars.get('START_ALTITUDE_OFFSET', '7.0'))
        min_air_gap = max(1.0, altitude_offset)
        
        # 提取智能体和目标位置
        agent_positions = []
        for i, agent in enumerate(world.agents):
            pos = agent.state.p_pos.copy()
            # 验证Z坐标是否在地形上方
            terrain_h = scenario.get_terrain_height(pos[0], pos[1])
            required_height = terrain_h + min_air_gap
            if pos[2] < required_height:
                # 如果Z坐标太低，调整到地形上方
                old_z = pos[2]
                pos[2] = required_height
                print(f"⚠️  [位置生成] Agent{i}: Z坐标从{old_z:.2f}调整到{pos[2]:.2f}（地形高度={terrain_h:.2f}）")
            agent_positions.append(pos.tolist())
        
        # 🚨 关键修复：验证并调整目标位置，确保在地形上方
        goal_pos = scenario.goal_pos.copy() if scenario.goal_pos is not None else np.array([100.0, 100.0, 50.0])
        if scenario.goal_pos is not None:
            goal_terrain_h = scenario.get_terrain_height(goal_pos[0], goal_pos[1])
            goal_altitude = float(env_vars.get('GOAL_ALTITUDE', '12.0'))
            required_goal_height = goal_terrain_h + goal_altitude
            if goal_pos[2] < required_goal_height:
                old_goal_z = goal_pos[2]
                goal_pos[2] = required_goal_height
                print(f"⚠️  [位置生成] 目标: Z坐标从{old_goal_z:.2f}调整到{goal_pos[2]:.2f}（地形高度={goal_terrain_h:.2f}）")
        goal_pos = goal_pos.tolist() if isinstance(goal_pos, np.ndarray) else goal_pos
        
        # 构建位置数据
        positions_data = {
            "terrain_seed": terrain_seed,
            "terrain_variant_seed": terrain_variant_seed,
            "episode_idx": episode_idx,
            "agents": agent_positions,
            "goal": goal_pos
        }

        positions_data, position_setup = _apply_same_region_positions(
            scenario=scenario,
            position_seed=terrain_variant_seed,
            positions_data=positions_data,
            env_vars=env_vars,
        )
        terrain_setup = _extract_terrain_setup(env_vars)
        positions_data["terrain_setup"] = terrain_setup
        if isinstance(position_setup, dict):
            position_setup["terrain_setup"] = dict(terrain_setup)

        # 保存到文件
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        positions_file = output_dir / _episode_positions_filename(
            episode_idx,
            terrain_seed=terrain_seed,
            terrain_variant_seed=terrain_variant_seed,
        )
        canonical_file = output_dir / f"episode_{episode_idx:03d}.json"

        _atomic_write_json(positions_file, positions_data)
        _atomic_write_json(canonical_file, positions_data)
        
        print(f"✅ 生成Episode {episode_idx}位置文件: {positions_file}")
        print(f"   地形种子: base={terrain_seed}, variant={terrain_variant_seed}")
        print(f"   智能体数量: {len(agent_positions)}")
        print(f"   智能体位置: {[f'({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})' for p in positions_data['agents']]}")
        print(f"   目标位置: ({positions_data['goal'][0]:.1f}, {positions_data['goal'][1]:.1f}, {positions_data['goal'][2]:.1f})")
        if position_setup is not None:
            print(
                f"   同区域位置约束: start_jitter={position_setup['start_center_jitter']}, "
                f"agent_jitter={position_setup['agent_local_jitter']}, "
                f"goal_radius={position_setup['goal_region_radius']}"
            )

        return positions_file
        
    except Exception as e:
        print(f"❌ 生成Episode {episode_idx}位置文件失败: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # 恢复原始环境变量
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def generate_all_episode_positions(terrain_seeds, output_dir, base_env_vars=None, terrain_variant_seeds=None):
    """
    为所有episode生成固定的位置文件
    
    Args:
        terrain_seeds: 基准地形种子列表
        output_dir: 输出目录
        base_env_vars: 基础环境变量字典
        terrain_variant_seeds: 同源扰动种子列表；None 时与 terrain_seeds 相同
    
    Returns:
        位置文件路径列表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if terrain_variant_seeds is None:
        terrain_variant_seeds = list(terrain_seeds)
    if len(terrain_variant_seeds) != len(terrain_seeds):
        raise ValueError("terrain_variant_seeds 长度必须与 terrain_seeds 一致")

    positions_files = []
    
    print(f"\n{'='*70}")
    print(f"开始为 {len(terrain_seeds)} 个episode生成固定位置文件")
    print(f"输出目录: {output_dir}")
    print(f"{'='*70}\n")
    
    for episode_idx, (terrain_seed, terrain_variant_seed) in enumerate(zip(terrain_seeds, terrain_variant_seeds)):
        positions_file = generate_episode_positions(
            terrain_seed,
            episode_idx,
            output_dir,
            base_env_vars,
            terrain_variant_seed=terrain_variant_seed,
        )
        if positions_file:
            positions_files.append(positions_file)
    
    print(f"\n{'='*70}")
    print(f"✅ 成功生成 {len(positions_files)}/{len(terrain_seeds)} 个位置文件")
    print(f"{'='*70}\n")
    
    return positions_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="为每个episode生成固定的位置文件")
    parser.add_argument("--terrain-seeds", type=str, required=True,
                        help="地形种子序列（逗号分隔）")
    parser.add_argument("--output-dir", type=str, default="./saved_positions/episode_positions",
                        help="输出目录（默认: ./saved_positions/episode_positions）")
    parser.add_argument("--map-size", type=str, default="200",
                        help="地图大小（默认: 200）")
    parser.add_argument("--terrain-complexity", type=str, default="3",
                        help="地形复杂度（默认: 3）")
    
    args = parser.parse_args()
    
    # 解析地形种子序列
    terrain_seeds = [int(s.strip()) for s in args.terrain_seeds.split(',') if s.strip()]
    
    # 基础环境变量
    base_env_vars = {
        "MAP_SIZE": args.map_size,
        "TERRAIN_COMPLEXITY_LEVEL": args.terrain_complexity,
    }
    
    # 生成所有位置文件
    positions_files = generate_all_episode_positions(
        terrain_seeds, args.output_dir, base_env_vars
    )
    
    print(f"\n生成的位置文件列表:")
    for f in positions_files:
        print(f"  - {f}")
