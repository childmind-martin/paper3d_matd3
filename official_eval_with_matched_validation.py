#!/usr/bin/env python3
"""Run official evaluation with optional matched-validation model selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_MODEL_VARIANT = "best_by_team_sr"
DEFAULT_SELECTION_PROTOCOL = "matched_validation"
DEFAULT_VALIDATION_EPISODES = 10
DEFAULT_VALIDATION_CANDIDATES = (
    "best_by_team_sr",
    "best",
    "checkpoint",
    "final",
    "latest_ep",
)
SELECTION_PROTOCOL_CHOICES = ("fixed", "matched_validation")


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"JSON payload must be an object: {path}")
    return data


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _normalize_candidates(raw_value: str) -> List[str]:
    allowed = {"auto", "final", "best", "best_by_team_sr", "latest_ep", "checkpoint"}
    tokens = [
        token.strip().lower()
        for token in str(raw_value or ",".join(DEFAULT_VALIDATION_CANDIDATES)).replace(";", ",").split(",")
    ]
    ordered: List[str] = []
    for token in tokens:
        if not token or token not in allowed or token in ordered:
            continue
        ordered.append(token)
    if not ordered:
        ordered = list(DEFAULT_VALIDATION_CANDIDATES)
    return ordered


def _derive_validation_seed(official_spec: Dict[str, Any], explicit_seed: Optional[int]) -> int:
    if explicit_seed is not None:
        return int(explicit_seed)
    base_seed = int(official_spec.get("seed", official_spec.get("scenario_seed", 88)))
    return int((base_seed + 104729) % 2147483647)


def _generate_sequence_seeds(seed: int, episodes: int, namespace: str) -> List[int]:
    rng = random.Random(f"{int(seed)}::{namespace}")
    return [int(rng.randint(1000, 99999)) for _ in range(int(episodes))]


def _make_hold_length(seed: int, block_idx: int, min_len: int, max_len: int) -> int:
    min_len = max(1, int(min_len))
    max_len = max(min_len, int(max_len))
    if min_len == max_len:
        return int(min_len)
    payload = f"post_eval_hold|seed={int(seed)}|block={int(block_idx)}|min={min_len}|max={max_len}"
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    span = max_len - min_len + 1
    return int(min_len + (int.from_bytes(digest, "little") % span))


def _generate_terrain_variant_seeds(
    seed: int,
    episodes: int,
    hold_mode: str = "episode",
    hold_episodes: int = 1,
    hold_min_episodes: int = 1,
    hold_max_episodes: int = 1,
) -> List[int]:
    episodes = max(0, int(episodes))
    hold_mode = str(hold_mode or "episode").strip().lower()
    if episodes <= 0:
        return []
    if hold_mode == "episode":
        return _generate_sequence_seeds(seed, episodes, "terrain_variant")

    sequence: List[int] = []
    block_idx = 0
    emitted = 0
    while emitted < episodes:
        if hold_mode == "fixed":
            block_length = max(1, int(hold_episodes or 1))
        else:
            block_length = _make_hold_length(
                seed=seed,
                block_idx=block_idx,
                min_len=max(1, int(hold_min_episodes or 1)),
                max_len=max(1, int(hold_max_episodes or hold_min_episodes or 1)),
            )
        block_seed = _generate_sequence_seeds(seed + block_idx, 1, "terrain_variant_block")[0]
        take = min(block_length, episodes - emitted)
        sequence.extend([int(block_seed)] * take)
        emitted += take
        block_idx += 1
    return sequence


def _build_post_eval_sequence_fields(spec: Dict[str, Any]) -> Dict[str, List[int]]:
    episodes = max(0, int(spec.get("episodes", 0) or 0))
    sequence_seed = int(spec.get("seed", spec.get("scenario_seed", 0)) or 0)
    terrain_seed = int(spec.get("terrain_seed", spec.get("scenario_seed", 0)) or 0)
    terrain_base_seed = int(spec.get("terrain_base_seed", terrain_seed) or terrain_seed)

    if bool(spec.get("semi_random_terrain", False)):
        terrain_seed_sequence = [int(terrain_base_seed)] * episodes
        terrain_variant_seed_sequence = _generate_terrain_variant_seeds(
            seed=sequence_seed,
            episodes=episodes,
            hold_mode=str(spec.get("semi_random_hold_mode", "episode") or "episode").strip().lower(),
            hold_episodes=int(spec.get("semi_random_hold_episodes", 1) or 1),
            hold_min_episodes=int(spec.get("semi_random_hold_min_episodes", 1) or 1),
            hold_max_episodes=int(
                spec.get("semi_random_hold_max_episodes", spec.get("semi_random_hold_min_episodes", 1)) or 1
            ),
        )
    elif bool(spec.get("random_terrain", False)):
        terrain_seed_sequence = _generate_sequence_seeds(sequence_seed, episodes, "terrain")
        terrain_variant_seed_sequence = []
    else:
        terrain_seed_sequence = [int(terrain_seed)] * episodes
        terrain_variant_seed_sequence = []

    obstacle_seed_sequence = (
        _generate_sequence_seeds(sequence_seed, episodes, "obstacle")
        if bool(spec.get("use_dynamic_obstacles", False))
        else [0] * episodes
    )
    return {
        "terrain_seed_sequence": [int(seed) for seed in terrain_seed_sequence],
        "terrain_variant_seed_sequence": [int(seed) for seed in terrain_variant_seed_sequence],
        "obstacle_seed_sequence": [int(seed) for seed in obstacle_seed_sequence],
    }


def _generate_post_eval_testset_tag(spec: Dict[str, Any]) -> str:
    terrain_kind = "semi_random" if bool(spec.get("semi_random_terrain", False)) else (
        "random" if bool(spec.get("random_terrain", False)) else "fixed"
    )
    obstacle_kind = "dynobs" if bool(spec.get("use_dynamic_obstacles", False)) else "staticobs"
    return (
        f"seed_{int(spec['seed'])}_{str(spec.get('mode', 'shared_match_train_env'))}_{terrain_kind}_{obstacle_kind}"
        f"_base_{int(spec.get('terrain_base_seed', spec.get('terrain_seed', spec.get('scenario_seed', 0))))}"
        f"_episodes_{int(spec.get('episodes', 0))}_posv6"
    )


def _ensure_episode_positions(spec: Dict[str, Any], *, force_regenerate: bool = False) -> Dict[str, Any]:
    episode_positions_dir_raw = str(spec.get("episode_positions_dir", "")).strip()
    if not episode_positions_dir_raw:
        return spec
    episode_positions_dir = Path(episode_positions_dir_raw)
    episode_positions_dir.mkdir(parents=True, exist_ok=True)
    reference_positions_file = Path(str(spec.get("reference_positions_file", "")).strip())
    if not reference_positions_file.exists():
        raise RuntimeError(f"共享测试集参考位置文件不存在: {reference_positions_file}")

    candidate_files = [
        episode_positions_dir / f"episode_{idx:03d}.json"
        for idx in range(int(spec.get("episodes", 0)))
    ]
    if force_regenerate:
        for candidate in candidate_files:
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass

    with open(reference_positions_file, "r", encoding="utf-8") as f:
        reference_payload = json.load(f)
    if not isinstance(reference_payload, dict):
        raise RuntimeError(f"共享测试集参考位置文件格式错误: {reference_positions_file}")

    terrain_seed_seq = spec.get("terrain_seed_sequence", [])
    terrain_variant_seed_seq = spec.get("terrain_variant_seed_sequence", [])
    obstacle_seed_seq = spec.get("obstacle_seed_sequence", [])
    for idx, candidate in enumerate(candidate_files):
        if candidate.exists():
            continue
        payload = dict(reference_payload)
        payload["episode"] = int(idx)
        payload["terrain_seed"] = (
            int(terrain_seed_seq[idx]) if idx < len(terrain_seed_seq) else int(spec.get("terrain_seed", spec.get("scenario_seed", 0)))
        )
        payload["terrain_variant_seed"] = (
            int(terrain_variant_seed_seq[idx]) if idx < len(terrain_variant_seed_seq) else None
        )
        payload["obstacle_seed"] = (
            int(obstacle_seed_seq[idx]) if idx < len(obstacle_seed_seq) else None
        )
        payload["shared_testset_mode"] = str(spec.get("mode", "shared_match_train_env"))
        payload["goal_flatness_profile_version"] = payload.get("goal_flatness_profile_version", 2)
        _save_json(candidate, payload)

    for candidate in candidate_files:
        if candidate.exists():
            spec["default_positions_file"] = str(candidate)
            break
    return spec


def _validation_artifact_policy() -> Dict[str, bool]:
    return {
        "light_mode": True,
        "save_interactive_html": False,
        "save_all_episodes": False,
        "save_best_reward_html": False,
        "save_team_success_html": False,
        "save_trajectory_json": False,
        "save_trajectory_png": False,
        "save_actor_sequence": False,
        "save_control_diagnostics": False,
        "enable_overlay": False,
        "disable_gif": True,
    }


def _matched_validation_root(output_dir: Path) -> Path:
    output_dir = Path(output_dir).resolve()
    return output_dir.parent / f"{output_dir.name}_matched_validation"


def _build_validation_spec(official_spec: Dict[str, Any], args) -> Dict[str, Any]:
    spec = dict(official_spec)
    spec["episodes"] = max(1, int(args.validation_episodes))
    spec["seed"] = _derive_validation_seed(official_spec, args.validation_seed)
    spec.update(_build_post_eval_sequence_fields(spec))
    spec["artifact_policy"] = _validation_artifact_policy()
    spec["validation_role"] = "checkpoint_selection"
    spec["force_regenerate_testset"] = True
    validation_testset_root = _matched_validation_root(Path(args.output_dir)) / "testset"
    spec["episode_positions_dir"] = str(
        validation_testset_root / _generate_post_eval_testset_tag(spec) / "episode_positions"
    )
    return _ensure_episode_positions(spec, force_regenerate=True)


def _is_valid_model_dir(dir_path: Path) -> bool:
    try:
        if not dir_path.exists() or not dir_path.is_dir():
            return False
        return any(dir_path.glob("actor_*.weights.h5"))
    except Exception:
        return False


def _ep_dir_sort_key(path: Path) -> Tuple[int, str]:
    name = path.name
    digits = "".join(ch for ch in name if ch.isdigit())
    try:
        numeric = int(digits) if digits else -1
    except Exception:
        numeric = -1
    return numeric, name


def _resolve_model_variant_dir(model_root: Path, model_variant: str) -> Tuple[Optional[Path], Optional[str]]:
    variant = str(model_variant or "").strip().lower()
    if not model_root.exists():
        return None, None

    def _latest_ep_dir() -> Optional[Path]:
        ep_dirs = [
            candidate
            for candidate in model_root.iterdir()
            if candidate.is_dir() and candidate.name.startswith("ep") and _is_valid_model_dir(candidate)
        ]
        if not ep_dirs:
            return None
        ep_dirs.sort(key=_ep_dir_sort_key, reverse=True)
        return ep_dirs[0]

    if variant == "auto":
        for fallback_variant in ("best_by_team_sr", "final", "best", "latest_ep", "checkpoint"):
            resolved_dir, resolved_variant = _resolve_model_variant_dir(model_root, fallback_variant)
            if resolved_dir is not None:
                return resolved_dir, resolved_variant
        return None, None

    if variant == "latest_ep":
        latest_ep = _latest_ep_dir()
        if latest_ep is not None:
            return latest_ep, latest_ep.name
        return None, None

    direct_dir = model_root / variant
    if _is_valid_model_dir(direct_dir):
        return direct_dir, direct_dir.name
    return None, None


def _compute_model_signature(model_dir: Path) -> Optional[str]:
    try:
        weight_files = sorted(model_dir.glob("actor_*.weights.h5"))
        if not weight_files:
            return None
        hasher = hashlib.sha1()
        for weight_path in weight_files:
            hasher.update(weight_path.name.encode("utf-8", errors="ignore"))
            with open(weight_path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


def _score_summary(summary: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    def _metric(key: str, *, fallback: float, invert: bool = False) -> float:
        value = _safe_float(summary.get(key))
        if value is None:
            return fallback
        value = float(value)
        return -value if invert else value

    return (
        _metric("team_success_rate", fallback=-1.0),
        _metric("collision_free_rate", fallback=-1.0),
        _metric("avg_team_final_goal_distance", fallback=-1e12, invert=True),
        _metric("avg_collision_count", fallback=-1e12, invert=True),
        _metric("avg_team_total_path_length", fallback=-1e12, invert=True),
    )


def _build_eval_env(base_env: Dict[str, str], spec: Dict[str, Any], *, quiet_output: str, python_bin: str) -> Dict[str, str]:
    env = dict(base_env)
    artifact_policy = spec.get("artifact_policy", {}) if isinstance(spec.get("artifact_policy"), dict) else {}
    env["EVAL_PYTHON_BIN"] = python_bin
    env["MODEL_VARIANT"] = "auto"
    env["STRICT_EVAL_MATCH"] = "1"
    env["USE_SCENARIO_SEED"] = "1"
    env["SCENARIO_SEED"] = str(spec.get("terrain_seed", spec.get("scenario_seed", 88)))
    env["TERRAIN_COMPLEXITY_LEVEL"] = str(spec.get("terrain_complexity", 3))
    env["MAP_SIZE"] = str(spec.get("map_size", 200))
    env["MOUNTAIN_MIN_DISTANCE"] = str(spec.get("mountain_min_distance", 55))
    env["SEMI_RANDOM_TERRAIN"] = "1" if spec.get("semi_random_terrain") else "0"
    env["RANDOM_TERRAIN"] = "1" if spec.get("random_terrain") else "0"
    env["TERRAIN_BASE_SEED"] = str(spec.get("terrain_base_seed", spec.get("scenario_seed", 88)))
    env["PEAK_JITTER_RANGE"] = str(spec.get("peak_jitter_range", 0.0))
    env["PEAK_CENTER_JITTER_RANGE"] = str(spec.get("peak_center_jitter_range", 0.0))
    env["PEAK_HEIGHT_JITTER_RATIO_MIN"] = str(spec.get("peak_height_jitter_ratio_min", 0.0))
    env["PEAK_HEIGHT_JITTER_RATIO_MAX"] = str(spec.get("peak_height_jitter_ratio_max", 0.0))
    env["PEAK_HEIGHT_MAX_SCALE"] = str(spec.get("peak_height_max_scale", 1.0))
    env["TERRAIN_VARIANT_NOISE_RATIO"] = str(spec.get("terrain_variant_noise_ratio", 0.0))
    env["HELDOUT_POSITION_MODE"] = str(spec.get("position_family", "train_match"))
    env["HELDOUT_REFERENCE_POSITIONS_FILE"] = str(spec.get("reference_positions_file", ""))
    env["HELDOUT_START_CENTER_JITTER"] = str(spec.get("start_center_jitter", 0.0))
    env["HELDOUT_AGENT_LOCAL_JITTER"] = str(spec.get("agent_local_jitter", 0.0))
    env["HELDOUT_GOAL_REGION_RADIUS"] = str(spec.get("goal_region_radius", 0.0))
    env["USE_DYNAMIC_OBSTACLES"] = "1" if spec.get("use_dynamic_obstacles") else "0"
    env["POST_EVAL_MODE"] = str(spec.get("mode", "shared_match_train_env"))
    env["POST_EVAL_TERRAIN_FAMILY"] = str(spec.get("terrain_family", "train_match_shared"))
    env["POST_EVAL_POSITION_FAMILY"] = str(spec.get("position_family", "train_match"))
    env["USE_FIXED_POSITIONS"] = "1"
    env["POSITIONS_FILE"] = str(spec.get("default_positions_file", ""))
    env["EVAL_EPISODE_LENGTH_MULTIPLIER"] = str(spec.get("episode_length_multiplier", 1.0))
    env["NOISE_SCALE"] = "0.0"
    env["RANDOM_ACTION_PROB"] = "0.0"
    env["RANDOM_ACTION_PROB_TRAINING"] = "0.0"
    env["ACTION_FORCE_RATIO_SCHEDULE_PCT"] = ""
    env["QUIET_OUTPUT"] = str(quiet_output)
    env["TQDM_DISABLE"] = "1"
    env["SAVE_INTERACTIVE_TRAJ"] = "1" if _to_bool(artifact_policy.get("save_interactive_html", True)) else "0"
    env["SAVE_EVAL_ALL_EPISODES"] = "1" if _to_bool(artifact_policy.get("save_all_episodes", False)) else "0"
    env["SAVE_BEST_TRAJ"] = "1" if _to_bool(artifact_policy.get("save_best_reward_html", True)) else "0"
    env["SAVE_TEAM_SUCCESS_HTML"] = "1" if _to_bool(artifact_policy.get("save_team_success_html", True)) else "0"
    env["SAVE_EVAL_TRAJECTORY_JSON"] = "1" if _to_bool(artifact_policy.get("save_trajectory_json", False)) else "0"
    env["SAVE_EVAL_TRAJECTORY_PNG"] = "1" if _to_bool(artifact_policy.get("save_trajectory_png", False)) else "0"
    env["SAVE_EVAL_ACTOR_SEQUENCE"] = "1" if _to_bool(artifact_policy.get("save_actor_sequence", True)) else "0"
    env["SAVE_EVAL_CONTROL_DIAGNOSTICS"] = "1" if _to_bool(artifact_policy.get("save_control_diagnostics", False)) else "0"
    env["ENABLE_OVERLAY"] = "1" if _to_bool(artifact_policy.get("enable_overlay", False)) else "0"
    env["DISABLE_GIF"] = "1" if _to_bool(artifact_policy.get("disable_gif", True)) else "0"
    if spec.get("episode_positions_dir"):
        env["EVAL_REQUIRE_EPISODE_POSITIONS"] = "1"
        env["EVAL_RESPECT_INPUT_POSITIONS"] = "1"
        env["EPISODE_POSITIONS_DIR"] = str(spec["episode_positions_dir"])
    else:
        env.pop("EVAL_REQUIRE_EPISODE_POSITIONS", None)
        env.pop("EVAL_RESPECT_INPUT_POSITIONS", None)
        env.pop("EPISODE_POSITIONS_DIR", None)
    for env_key, spec_key in (
        ("OBSTACLE_SEED_SEQUENCE", "obstacle_seed_sequence"),
        ("TERRAIN_SEED_SEQUENCE", "terrain_seed_sequence"),
        ("TERRAIN_VARIANT_SEED_SEQUENCE", "terrain_variant_seed_sequence"),
    ):
        sequence = spec.get(spec_key, [])
        if sequence:
            env[env_key] = ",".join(str(item) for item in sequence)
        else:
            env.pop(env_key, None)
    return env


def _run_command_with_live_output(
    command: Sequence[str],
    *,
    env: Dict[str, str],
    cwd: Path,
    log_path: Path,
    prefix: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_fp:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_fp.write(line)
            log_fp.flush()
            print(f"{prefix}{line}", end="")
        returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"{prefix}命令失败，退出码={returncode} | 日志: {log_path}")


def _execute_eval_run(
    *,
    model_path: Path,
    spec: Dict[str, Any],
    eval_dir: Path,
    quiet_output: str,
    python_bin: str,
    banner_prefix: str,
    force_rerun: bool,
) -> Dict[str, Any]:
    if force_rerun and eval_dir.exists():
        shutil.rmtree(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    spec_path = eval_dir / "post_eval_spec.json"
    _save_json(spec_path, spec)
    results_path = eval_dir / "evaluation_results.json"
    log_path = eval_dir / "post_eval.log"
    if not force_rerun and results_path.exists():
        try:
            results = _load_json(results_path)
            summary = results.get("summary", {}) if isinstance(results.get("summary"), dict) else {}
            print(f"{banner_prefix}复用已有评估结果: {results_path}")
            return {
                "results_path": str(results_path),
                "log_path": str(log_path),
                "spec_path": str(spec_path),
                "summary": summary,
                "results": results,
            }
        except Exception:
            print(f"{banner_prefix}已有评估结果无法读取，将重新评估: {results_path}")
    env = _build_eval_env(os.environ.copy(), spec, quiet_output=quiet_output, python_bin=python_bin)
    cmd = [
        "/bin/bash",
        str((Path(__file__).resolve().parent / "run_evaluation.sh").resolve()),
        str(model_path),
        str(int(spec.get("episodes", 20))),
        str(eval_dir),
        str(spec.get("default_positions_file", "")),
        "1",
        "false",
    ]
    print(f"{banner_prefix}模型路径: {model_path}")
    print(f"{banner_prefix}输出目录: {eval_dir}")
    print(f"{banner_prefix}测试回合数: {spec.get('episodes')}")
    _run_command_with_live_output(
        cmd,
        env=env,
        cwd=Path(__file__).resolve().parent,
        log_path=log_path,
        prefix=f"{banner_prefix}",
    )
    if not results_path.exists():
        raise RuntimeError(f"{banner_prefix}缺少 evaluation_results.json: {results_path}")
    results = _load_json(results_path)
    summary = results.get("summary", {}) if isinstance(results.get("summary"), dict) else {}
    return {
        "results_path": str(results_path),
        "log_path": str(log_path),
        "spec_path": str(spec_path),
        "summary": summary,
        "results": results,
    }


def _select_candidate(args, official_spec: Dict[str, Any], experiment_root: Path, quiet_output: str, python_bin: str) -> Dict[str, Any]:
    requested_variant = str(args.model_variant or DEFAULT_MODEL_VARIANT).strip().lower()
    protocol = str(args.selection_protocol or DEFAULT_SELECTION_PROTOCOL).strip().lower()
    output_dir = Path(args.output_dir).resolve()
    if protocol != "matched_validation":
        selected_path, resolved_variant = _resolve_model_variant_dir(experiment_root, requested_variant)
        if selected_path is None or resolved_variant is None:
            raise RuntimeError(f"未找到请求的 checkpoint 变体: {requested_variant} | root={experiment_root}")
        return {
            "selection_protocol": "fixed",
            "candidate_alias": requested_variant,
            "resolved_variant": resolved_variant,
            "model_path": selected_path,
            "validation_selection_summary_path": "",
        }

    validation_spec = _build_validation_spec(official_spec, args)
    validation_root = _matched_validation_root(output_dir)
    validation_root.mkdir(parents=True, exist_ok=True)

    candidate_aliases = _normalize_candidates(args.validation_candidates)
    if requested_variant != "auto" and requested_variant not in candidate_aliases:
        candidate_aliases = [requested_variant, *candidate_aliases]

    candidates: List[Dict[str, Any]] = []
    seen_paths: set[str] = set()
    for alias in candidate_aliases:
        candidate_path, resolved_variant = _resolve_model_variant_dir(experiment_root, alias)
        if candidate_path is None or resolved_variant is None:
            continue
        path_key = str(candidate_path.resolve())
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        candidates.append(
            {
                "candidate_alias": alias,
                "resolved_variant": resolved_variant,
                "model_path": candidate_path,
            }
        )

    if not candidates:
        fallback_path, fallback_variant = _resolve_model_variant_dir(experiment_root, requested_variant)
        if fallback_path is None or fallback_variant is None:
            raise RuntimeError(f"未找到任何可用 checkpoint | root={experiment_root}")
        candidates = [
            {
                "candidate_alias": requested_variant,
                "resolved_variant": fallback_variant,
                "model_path": fallback_path,
            }
        ]

    scored_candidates: List[Dict[str, Any]] = []
    failed_candidates: List[Dict[str, Any]] = []
    cached_by_signature: Dict[str, Dict[str, Any]] = {}
    for order_idx, candidate in enumerate(candidates):
        signature = _compute_model_signature(Path(candidate["model_path"]))
        if signature and signature in cached_by_signature:
            cached = cached_by_signature[signature]
            scored_candidates.append(
                {
                    **candidate,
                    "order": order_idx,
                    "summary": dict(cached["summary"]),
                    "score": list(cached["score"]),
                    "eval_dir": cached["eval_dir"],
                    "results_path": cached["results_path"],
                    "log_path": cached["log_path"],
                    "model_signature": signature,
                    "reused_validation_from_candidate": cached["candidate_alias"],
                }
            )
            continue

        candidate_spec = dict(validation_spec)
        candidate_spec["requested_model_variant"] = str(candidate["resolved_variant"])
        candidate_spec["candidate_alias"] = str(candidate["candidate_alias"])
        candidate_spec["selection_protocol"] = "matched_validation_candidate"
        candidate_eval_dir = validation_root / str(candidate["candidate_alias"])
        try:
            eval_record = _execute_eval_run(
                model_path=Path(candidate["model_path"]),
                spec=candidate_spec,
                eval_dir=candidate_eval_dir,
                quiet_output=quiet_output,
                python_bin=python_bin,
                banner_prefix=f"[验证选模 {candidate['candidate_alias']}] ",
                force_rerun=args.force_rerun,
            )
        except Exception as exc:
            failed_candidates.append(
                {
                    **candidate,
                    "order": order_idx,
                    "model_signature": signature,
                    "failure_reason": str(exc),
                    "eval_dir": str(candidate_eval_dir),
                }
            )
            continue

        score = _score_summary(eval_record["summary"])
        scored_entry = {
            **candidate,
            "order": order_idx,
            "summary": eval_record["summary"],
            "score": list(score),
            "eval_dir": str(candidate_eval_dir),
            "results_path": eval_record["results_path"],
            "log_path": eval_record["log_path"],
            "model_signature": signature,
        }
        scored_candidates.append(scored_entry)
        if signature:
            cached_by_signature[signature] = {
                "candidate_alias": candidate["candidate_alias"],
                "summary": eval_record["summary"],
                "score": list(score),
                "eval_dir": str(candidate_eval_dir),
                "results_path": eval_record["results_path"],
                "log_path": eval_record["log_path"],
            }

    if not scored_candidates:
        details = " | ".join(f"{item['candidate_alias']}={item['failure_reason']}" for item in failed_candidates)
        raise RuntimeError(f"所有 matched validation 候选均失败: {details or '无候选可用'}")

    best_candidate = max(scored_candidates, key=lambda item: (tuple(item["score"]), -int(item["order"])))
    selection_summary = {
        "selection_protocol": "matched_validation",
        "requested_model_variant": requested_variant,
        "validation_episodes": int(validation_spec["episodes"]),
        "validation_seed": int(validation_spec["seed"]),
        "validation_candidates": list(candidate_aliases),
        "candidates": scored_candidates,
        "failed_candidates": failed_candidates,
        "selected": {
            "candidate_alias": best_candidate["candidate_alias"],
            "resolved_variant": best_candidate["resolved_variant"],
            "model_path": str(best_candidate["model_path"]),
            "model_signature": best_candidate.get("model_signature"),
            "score": best_candidate["score"],
            "summary": best_candidate["summary"],
            "eval_dir": best_candidate["eval_dir"],
            "results_path": best_candidate["results_path"],
            "log_path": best_candidate["log_path"],
            "reused_validation_from_candidate": best_candidate.get("reused_validation_from_candidate"),
        },
    }
    selection_summary_path = validation_root / "selection_summary.json"
    _save_json(selection_summary_path, selection_summary)
    return {
        "selection_protocol": "matched_validation",
        "candidate_alias": best_candidate["candidate_alias"],
        "resolved_variant": best_candidate["resolved_variant"],
        "model_path": Path(best_candidate["model_path"]),
        "validation_selection_summary_path": str(selection_summary_path),
    }


def _inject_selection_metadata(results_path: Path, payload: Dict[str, Any]) -> None:
    data = _load_json(results_path)
    data["model_selection"] = payload
    _save_json(results_path, data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official evaluation with matched-validation model selection")
    parser.add_argument("--experiment-root", required=True, help="Experiment model root, e.g. models/exp_xxx")
    parser.add_argument("--official-spec", required=True, help="Official post-eval shared spec JSON")
    parser.add_argument("--output-dir", required=True, help="Final official evaluation output directory")
    parser.add_argument("--model-variant", default=DEFAULT_MODEL_VARIANT, help="Requested model variant")
    parser.add_argument(
        "--selection-protocol",
        default=DEFAULT_SELECTION_PROTOCOL,
        choices=list(SELECTION_PROTOCOL_CHOICES),
        help="fixed=direct variant; matched_validation=validation-set selection before final official test",
    )
    parser.add_argument("--validation-episodes", type=int, default=DEFAULT_VALIDATION_EPISODES)
    parser.add_argument("--validation-seed", type=int, default=None)
    parser.add_argument(
        "--validation-candidates",
        default=",".join(DEFAULT_VALIDATION_CANDIDATES),
        help="Comma-separated checkpoint candidates",
    )
    parser.add_argument("--python-bin", default=os.getenv("EVAL_PYTHON_BIN") or os.getenv("TRAIN_PYTHON_BIN") or sys.executable)
    parser.add_argument("--quiet-output", default=os.getenv("OFFICIAL_EVAL_QUIET", "1"))
    parser.add_argument("--force-rerun", action="store_true", help="Delete old eval dirs before rerunning")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_root = Path(args.experiment_root).resolve()
    official_spec_path = Path(args.official_spec).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not experiment_root.exists():
        raise RuntimeError(f"实验模型目录不存在: {experiment_root}")
    if not official_spec_path.exists():
        raise RuntimeError(f"官方 spec 不存在: {official_spec_path}")

    official_spec = _load_json(official_spec_path)
    requested_variant = str(args.model_variant or official_spec.get("requested_model_variant") or DEFAULT_MODEL_VARIANT).strip().lower()
    args.model_variant = requested_variant

    print("官方评估配置:")
    print(f"  - 模型根目录: {experiment_root}")
    print(f"  - 官方spec: {official_spec_path}")
    print(f"  - 最终输出目录: {output_dir}")
    print(f"  - 请求模型变体: {requested_variant}")
    print(f"  - 选模协议: {args.selection_protocol}")
    if args.selection_protocol == "matched_validation":
        print(f"  - 验证回合数: {args.validation_episodes}")
        print(f"  - 验证候选: {_normalize_candidates(args.validation_candidates)}")
        print(f"  - 验证seed: {_derive_validation_seed(official_spec, args.validation_seed)}")
    print("")

    selection = _select_candidate(
        args=args,
        official_spec=official_spec,
        experiment_root=experiment_root,
        quiet_output=str(args.quiet_output),
        python_bin=str(args.python_bin),
    )
    selected_model_path = Path(selection["model_path"]).resolve()

    final_spec = dict(official_spec)
    final_spec["requested_model_variant"] = requested_variant
    final_spec["selection_protocol"] = str(selection["selection_protocol"])
    final_spec["selected_model_candidate"] = str(selection["candidate_alias"])
    final_spec["selected_model_variant"] = str(selection["resolved_variant"])
    final_spec["selected_model_path"] = str(selected_model_path)
    if selection.get("validation_selection_summary_path"):
        final_spec["validation_selection_summary_path"] = str(selection["validation_selection_summary_path"])
    if final_spec.get("episode_positions_dir"):
        final_spec = _ensure_episode_positions(final_spec, force_regenerate=False)

    print("最终选模结果:")
    print(f"  - 选中候选: {selection['candidate_alias']}")
    print(f"  - 解析变体: {selection['resolved_variant']}")
    print(f"  - 模型路径: {selected_model_path}")
    if selection.get("validation_selection_summary_path"):
        print(f"  - 选模摘要: {selection['validation_selection_summary_path']}")
    print("")

    final_record = _execute_eval_run(
        model_path=selected_model_path,
        spec=final_spec,
        eval_dir=output_dir,
        quiet_output=str(args.quiet_output),
        python_bin=str(args.python_bin),
        banner_prefix="[官方测试] ",
        force_rerun=args.force_rerun,
    )

    selection_metadata = {
        "selection_protocol": str(selection["selection_protocol"]),
        "requested_model_variant": requested_variant,
        "selected_model_candidate": str(selection["candidate_alias"]),
        "selected_model_variant": str(selection["resolved_variant"]),
        "selected_model_path": str(selected_model_path),
        "validation_selection_summary_path": str(selection.get("validation_selection_summary_path", "")),
    }
    _inject_selection_metadata(Path(final_record["results_path"]), selection_metadata)
    print("")
    print("======================================")
    print("✅ 官方评估完成")
    print(f"  - 结果目录: {output_dir}")
    print(f"  - 评估统计: {final_record['results_path']}")
    print("======================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
