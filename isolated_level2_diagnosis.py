#!/usr/bin/env python3
"""
Run an isolated level-2 diagnosis suite without touching historical experiment data.

Design goals:
1. Only read from existing batch/log/model artifacts.
2. Write all newly generated testsets, evaluation outputs, summaries and plots under
   a brand-new diagnostics directory.
3. Reproduce the key diagnosis axes discussed in analysis:
   - training-distribution replay vs normal post-eval
   - training-tail vs deterministic eval gap
   - checkpoint selection stability
   - episode-length sensitivity
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matd3_mpl_diagnostics")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "diagnostics"
DEFAULT_EXPERIMENTS = ["matd3_separated_gradient"]
DEFAULT_SELECTION_CANDIDATES = ["best_by_team_sr", "best", "final", "checkpoint"]
DEFAULT_SELECTION_VALIDATION_EPISODES = [10, 50, 100]
DEFAULT_EPISODE_LENGTH_MULTIPLIERS = [1.0, 1.1, 1.2]
DEFAULT_TRAIN_DISTRIBUTION_EPISODES = 50
SUITE_NAMES = ("obstacle_shift", "training_gap", "selection", "length")
_STREAM_LOCK = threading.Lock()


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"JSON payload must be an object: {path}")
    return data


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _copy_json(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except Exception:
        return None
    if result != result:
        return None
    return result


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _mean(values: Sequence[float]) -> Optional[float]:
    finite = [float(v) for v in values if _safe_float(v) is not None]
    if not finite:
        return None
    return sum(finite) / float(len(finite))


def _slug(value: str) -> str:
    keep = []
    for ch in str(value):
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "item"


def _parse_csv_strings(raw: Optional[str], *, cast=str) -> Optional[List[Any]]:
    if raw is None:
        return None
    parts = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not parts:
        return []
    return [cast(item) for item in parts]


def _parse_float_list(raw: Optional[str], default: Sequence[float]) -> List[float]:
    parsed = _parse_csv_strings(raw, cast=float)
    return list(default if parsed is None else parsed)


def _parse_int_list(raw: Optional[str], default: Sequence[int]) -> List[int]:
    parsed = _parse_csv_strings(raw, cast=int)
    return list(default if parsed is None else parsed)


def _resolve_parent_batch_dir(raw: str) -> Path:
    candidate = Path(raw).expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Parent batch directory not found: {candidate}")
    if not candidate.is_dir():
        raise RuntimeError(f"Parent batch path is not a directory: {candidate}")
    return candidate


def _generate_post_eval_sequence_seeds(seed: int, episodes: int, namespace: str) -> List[int]:
    rng = random.Random(f"{int(seed)}::{namespace}")
    return [int(rng.randint(1000, 99999)) for _ in range(int(episodes))]


def _make_post_eval_hold_length(seed: int, block_idx: int, min_len: int, max_len: int) -> int:
    min_len = max(1, int(min_len))
    max_len = max(min_len, int(max_len))
    if min_len == max_len:
        return int(min_len)
    payload = f"post_eval_hold|seed={int(seed)}|block={int(block_idx)}|min={min_len}|max={max_len}"
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    span = max_len - min_len + 1
    return int(min_len + (int.from_bytes(digest, "little") % span))


def _generate_post_eval_terrain_variant_seeds(
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
        return _generate_post_eval_sequence_seeds(seed, episodes, "terrain_variant")

    sequence: List[int] = []
    block_idx = 0
    emitted = 0
    while emitted < episodes:
        if hold_mode == "fixed":
            block_length = max(1, int(hold_episodes or 1))
        else:
            block_length = _make_post_eval_hold_length(
                seed=seed,
                block_idx=block_idx,
                min_len=max(1, int(hold_min_episodes or 1)),
                max_len=max(1, int(hold_max_episodes or hold_min_episodes or 1)),
            )
        block_seed = _generate_post_eval_sequence_seeds(seed + block_idx, 1, "terrain_variant_block")[0]
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
        terrain_variant_seed_sequence = _generate_post_eval_terrain_variant_seeds(
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
        terrain_seed_sequence = _generate_post_eval_sequence_seeds(sequence_seed, episodes, "terrain")
        terrain_variant_seed_sequence = []
    else:
        terrain_seed_sequence = [int(terrain_seed)] * episodes
        terrain_variant_seed_sequence = []

    obstacle_seed_sequence = (
        _generate_post_eval_sequence_seeds(sequence_seed, episodes, "obstacle")
        if bool(spec.get("use_dynamic_obstacles", False))
        else [0] * episodes
    )
    return {
        "terrain_seed_sequence": [int(seed) for seed in terrain_seed_sequence],
        "terrain_variant_seed_sequence": [int(seed) for seed in terrain_variant_seed_sequence],
        "obstacle_seed_sequence": [int(seed) for seed in obstacle_seed_sequence],
    }


def _generate_trainlike_obstacle_seeds(base_seed: int, episodes: int) -> List[int]:
    base_seed = int(base_seed)
    return [int(base_seed + 10000 + (idx + 1) * 1000) for idx in range(int(episodes))]


def _ensure_episode_positions(spec: Dict[str, Any]) -> Dict[str, Any]:
    episode_positions_dir = Path(spec["episode_positions_dir"])
    episode_positions_dir.mkdir(parents=True, exist_ok=True)
    reference_positions_file = Path(str(spec.get("reference_positions_file", "")).strip())
    if not reference_positions_file.exists():
        raise RuntimeError(f"Reference positions file not found: {reference_positions_file}")

    with open(reference_positions_file, "r", encoding="utf-8") as f:
        reference_payload = json.load(f)
    if not isinstance(reference_payload, dict):
        raise RuntimeError(f"Invalid reference positions payload: {reference_positions_file}")

    terrain_seed_seq = list(spec.get("terrain_seed_sequence", []) or [])
    terrain_variant_seed_seq = list(spec.get("terrain_variant_seed_sequence", []) or [])
    obstacle_seed_seq = list(spec.get("obstacle_seed_sequence", []) or [])
    episodes = int(spec.get("episodes", 0) or 0)

    for idx in range(episodes):
        candidate = episode_positions_dir / f"episode_{idx:03d}.json"
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

    if episodes > 0:
        spec["default_positions_file"] = str(episode_positions_dir / "episode_000.json")
    return spec


def _score_post_eval_summary(summary: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
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


def _resolve_model_root_from_log_dir(log_dir: Path, exp_name_hint: Optional[str] = None) -> Path:
    log_dir = log_dir.resolve()
    candidate_names: List[str] = []
    hint = str(exp_name_hint or "").strip()
    if hint:
        candidate_names.append(hint)
    parent_name = str(log_dir.parent.name).strip()
    if parent_name and parent_name != "logs":
        candidate_names.append(parent_name)
    log_dir_name = str(log_dir.name).strip()
    if log_dir_name:
        candidate_names.append(log_dir_name)

    seen = set()
    ordered_candidates: List[str] = []
    for name in candidate_names:
        if name not in seen:
            ordered_candidates.append(name)
            seen.add(name)

    checked_paths: List[str] = []
    for exp_name in ordered_candidates:
        model_root = REPO_ROOT / "models" / exp_name
        checked_paths.append(str(model_root))
        if model_root.exists():
            return model_root

    raise FileNotFoundError(
        "Model root inferred from log_dir does not exist. "
        f"log_dir={log_dir}, exp_name_hint={hint or 'N/A'}, checked={checked_paths}"
    )


def _resolve_model_variant_dir(model_root: Path, variant: str) -> Path:
    variant = str(variant).strip()
    direct = model_root / variant
    if direct.exists():
        return direct
    if variant == "latest_ep":
        ep_dirs = sorted(
            [p for p in model_root.iterdir() if p.is_dir() and p.name.startswith("ep")],
            key=lambda p: p.name,
        )
        if ep_dirs:
            return ep_dirs[-1]
    raise FileNotFoundError(f"Model variant not found: {model_root} / {variant}")


def _find_spec_path(root: Path, preferred_variants: Sequence[str]) -> Path:
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Spec root does not exist: {root}")
    for variant in preferred_variants:
        candidate = root / variant / "post_eval_spec.json"
        if candidate.exists():
            return candidate
    matches = sorted(root.glob("*/post_eval_spec.json"))
    if not matches:
        raise FileNotFoundError(f"No post_eval_spec.json found under: {root}")
    return matches[0]


def _derive_extra_validation_seeds(base_seed: int, extra_count: int) -> List[int]:
    if extra_count <= 0:
        return []
    return _generate_post_eval_sequence_seeds(base_seed, extra_count, "diagnostic_validation_seed")


@dataclass
class ExperimentSeedContext:
    label: str
    seed: int
    child_batch_dir: Path
    manifest_path: Path
    log_dir: Path
    model_root: Path
    eval_python_bin: Optional[str]
    training_tail100_success_mean: Optional[float]
    training_tail100_reward_mean: Optional[float]
    existing_post_eval_summary: Dict[str, Any]
    test_spec_path: Path
    validation_spec_path: Path


@dataclass(frozen=True)
class EvalJobSpec:
    suite: str
    label: str
    seed: int
    description: str
    model_dir: Path
    variant: str
    spec: Dict[str, Any]
    save_dir: Path
    eval_python_bin: Optional[str]
    quiet_eval: bool
    dry_run: bool
    stream_output: bool
    metadata: Dict[str, Any]


def _collect_contexts(
    parent_batch_dir: Path,
    experiments: Sequence[str],
    selected_seeds: Optional[Sequence[int]],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[ExperimentSeedContext]]:
    parent_config = _load_json(parent_batch_dir / "config.json")
    parent_summary = _load_json(parent_batch_dir / "plots" / "latest_summary.json")

    child_batch_by_seed: Dict[int, Path] = {}
    for child in parent_summary.get("child_runs", []) or []:
        seed = _safe_int(child.get("seed"))
        batch_dir = child.get("batch_dir")
        if seed is None or not batch_dir:
            continue
        child_batch_by_seed[int(seed)] = Path(str(batch_dir)).resolve()

    summary_by_label: Dict[str, Dict[str, Any]] = {}
    for row in parent_summary.get("aggregated_experiments", []) or []:
        label = str(row.get("label", "")).strip()
        if label:
            summary_by_label[label] = row

    seed_filter = {int(seed) for seed in selected_seeds} if selected_seeds else None
    contexts: List[ExperimentSeedContext] = []
    for label in experiments:
        if label not in summary_by_label:
            raise RuntimeError(f"Experiment label not found in parent summary: {label}")
        row = summary_by_label[label]
        for seed_row in row.get("seed_values", []) or []:
            seed = _safe_int(seed_row.get("seed"))
            log_dir = seed_row.get("log_dir")
            if seed is None or not log_dir:
                continue
            if seed_filter is not None and int(seed) not in seed_filter:
                continue
            child_batch_dir = child_batch_by_seed.get(int(seed))
            if child_batch_dir is None:
                raise RuntimeError(f"Missing child batch directory for seed={seed}")
            manifest_path = child_batch_dir / "manifests" / f"{label}_resolved_manifest.json"
            manifest_payload = _load_json(manifest_path) if manifest_path.exists() else {}
            manifest_meta = manifest_payload.get("meta", {}) if isinstance(manifest_payload.get("meta"), dict) else {}
            exp_name_hint = str(manifest_meta.get("exp_name_with_timestamp", "")).strip() or None
            log_dir_path = Path(str(log_dir)).resolve()
            model_root = _resolve_model_root_from_log_dir(log_dir_path, exp_name_hint=exp_name_hint)
            test_spec_path = _find_spec_path(
                child_batch_dir / "results" / "post_eval" / label,
                preferred_variants=("matched_validation", "best_by_team_sr", "best", "checkpoint", "final"),
            )
            validation_spec_path = _find_spec_path(
                child_batch_dir / "results" / "post_eval_validation" / label,
                preferred_variants=("best_by_team_sr", "best", "checkpoint", "final"),
            )
            contexts.append(
                ExperimentSeedContext(
                    label=label,
                    seed=int(seed),
                    child_batch_dir=child_batch_dir,
                    manifest_path=manifest_path,
                    log_dir=log_dir_path,
                    model_root=model_root,
                    eval_python_bin=str(manifest_payload.get("python_executable", "")).strip() or None,
                    training_tail100_success_mean=_safe_float(seed_row.get("tail100_success_mean")),
                    training_tail100_reward_mean=_safe_float(seed_row.get("tail100_reward_mean")),
                    existing_post_eval_summary=dict(seed_row.get("post_eval_summary", {}) or {}),
                    test_spec_path=test_spec_path,
                    validation_spec_path=validation_spec_path,
                )
            )

    if not contexts:
        raise RuntimeError("No experiment/seed contexts matched the requested filters.")
    return parent_config, parent_summary, contexts


def _materialize_spec(
    base_spec: Dict[str, Any],
    *,
    output_positions_dir: Path,
    episodes: Optional[int] = None,
    seed: Optional[int] = None,
    episode_length_multiplier: Optional[float] = None,
    obstacle_mode: str = "heldout",
) -> Dict[str, Any]:
    spec = json.loads(json.dumps(base_spec))
    if episodes is not None:
        spec["episodes"] = int(episodes)
    if seed is not None:
        spec["seed"] = int(seed)
    if episode_length_multiplier is not None:
        spec["episode_length_multiplier"] = float(episode_length_multiplier)
    spec["episode_positions_dir"] = str(output_positions_dir.resolve())
    seq_fields = _build_post_eval_sequence_fields(spec)
    spec.update(seq_fields)
    if obstacle_mode == "trainlike":
        base_seed = int(spec.get("terrain_seed", spec.get("scenario_seed", 0)) or 0)
        spec["obstacle_seed_sequence"] = _generate_trainlike_obstacle_seeds(base_seed, int(spec.get("episodes", 0) or 0))
    elif obstacle_mode != "heldout":
        raise ValueError(f"Unknown obstacle_mode: {obstacle_mode}")
    spec = _ensure_episode_positions(spec)
    return spec


def _build_eval_env(spec: Dict[str, Any], quiet_eval: bool) -> Dict[str, str]:
    env = os.environ.copy()
    # Clear terrain/post-eval related variables first so we don't inherit stale
    # heldout or semi-random settings from the parent shell.
    clear_keys = (
        "USE_SCENARIO_SEED",
        "SCENARIO_SEED",
        "TERRAIN_BASE_SEED",
        "RANDOM_TERRAIN",
        "PER_EPISODE_TERRAIN",
        "PER_ENV_TERRAIN",
        "SEMI_RANDOM_TERRAIN",
        "PEAK_JITTER_RANGE",
        "PEAK_CENTER_JITTER_RANGE",
        "PEAK_HEIGHT_JITTER_RATIO_MIN",
        "PEAK_HEIGHT_JITTER_RATIO_MAX",
        "PEAK_HEIGHT_MAX_SCALE",
        "TERRAIN_VARIANT_NOISE_RATIO",
        "TERRAIN_COMPLEXITY_LEVEL",
        "MAP_SIZE",
        "MOUNTAIN_MIN_DISTANCE",
        "HELDOUT_POSITION_MODE",
        "HELDOUT_REFERENCE_POSITIONS_FILE",
        "HELDOUT_START_CENTER_JITTER",
        "HELDOUT_AGENT_LOCAL_JITTER",
        "HELDOUT_GOAL_REGION_RADIUS",
        "USE_DYNAMIC_OBSTACLES",
        "USE_FIXED_POSITIONS",
        "POSITIONS_FILE",
        "POST_EVAL_MODE",
        "POST_EVAL_TERRAIN_FAMILY",
        "POST_EVAL_POSITION_FAMILY",
        "TERRAIN_SEED_SEQUENCE",
        "TERRAIN_VARIANT_SEED_SEQUENCE",
        "OBSTACLE_SEED_SEQUENCE",
    )
    for key in clear_keys:
        env.pop(key, None)
    env["SUPPRESS_MA_PROMPT"] = "1"
    env["SUPPRESS_TERRAIN_OUTPUT"] = "1"
    env["USE_SCENARIO_SEED"] = "1"
    env["SCENARIO_SEED"] = str(int(spec.get("terrain_seed", spec.get("scenario_seed", 0)) or 0))
    env["TERRAIN_BASE_SEED"] = str(
        int(spec.get("terrain_base_seed", spec.get("terrain_seed", spec.get("scenario_seed", 0))) or 0)
    )
    env["RANDOM_TERRAIN"] = "1" if bool(spec.get("random_terrain", False)) else "0"
    env["PER_EPISODE_TERRAIN"] = "0"
    env["PER_ENV_TERRAIN"] = "0"
    env["SEMI_RANDOM_TERRAIN"] = "1" if bool(spec.get("semi_random_terrain", False)) else "0"
    env["PEAK_JITTER_RANGE"] = str(float(spec.get("peak_jitter_range", 15.0)))
    env["PEAK_CENTER_JITTER_RANGE"] = str(float(spec.get("peak_center_jitter_range", 3.0)))
    env["PEAK_HEIGHT_JITTER_RATIO_MIN"] = str(float(spec.get("peak_height_jitter_ratio_min", 0.20)))
    env["PEAK_HEIGHT_JITTER_RATIO_MAX"] = str(float(spec.get("peak_height_jitter_ratio_max", 0.40)))
    env["PEAK_HEIGHT_MAX_SCALE"] = str(float(spec.get("peak_height_max_scale", 1.30)))
    env["TERRAIN_VARIANT_NOISE_RATIO"] = str(float(spec.get("terrain_variant_noise_ratio", 0.15)))
    env["TERRAIN_COMPLEXITY_LEVEL"] = str(int(spec.get("terrain_complexity", 0) or 0))
    env["MAP_SIZE"] = str(int(spec.get("map_size", 200) or 200))
    env["MOUNTAIN_MIN_DISTANCE"] = str(int(spec.get("mountain_min_distance", 55) or 55))
    env["HELDOUT_POSITION_MODE"] = str(spec.get("position_family", "train_match"))
    env["HELDOUT_REFERENCE_POSITIONS_FILE"] = str(
        spec.get("reference_positions_file", spec.get("default_positions_file", "")) or ""
    )
    env["HELDOUT_START_CENTER_JITTER"] = str(float(spec.get("start_center_jitter", 12.0)))
    env["HELDOUT_AGENT_LOCAL_JITTER"] = str(float(spec.get("agent_local_jitter", 3.0)))
    env["HELDOUT_GOAL_REGION_RADIUS"] = str(float(spec.get("goal_region_radius", 18.0)))
    env["USE_DYNAMIC_OBSTACLES"] = "1" if bool(spec.get("use_dynamic_obstacles", True)) else "0"
    env["USE_FIXED_POSITIONS"] = "1"
    env["POSITIONS_FILE"] = str(spec.get("default_positions_file", ""))
    env["POST_EVAL_MODE"] = str(spec.get("mode", "shared_match_train_env"))
    env["EVAL_REQUIRE_EPISODE_POSITIONS"] = "1"
    env["EVAL_RESPECT_INPUT_POSITIONS"] = "1"
    env["EPISODE_POSITIONS_DIR"] = str(spec["episode_positions_dir"])
    env["EVAL_EPISODE_LENGTH_MULTIPLIER"] = str(spec.get("episode_length_multiplier", 1.0))
    env["POST_EVAL_TERRAIN_FAMILY"] = str(spec.get("terrain_family", ""))
    env["POST_EVAL_POSITION_FAMILY"] = str(spec.get("position_family", ""))
    if quiet_eval:
        env["QUIET_OUTPUT"] = "1"
    terrain_seed_sequence = list(spec.get("terrain_seed_sequence", []) or [])
    terrain_variant_seed_sequence = list(spec.get("terrain_variant_seed_sequence", []) or [])
    obstacle_seed_sequence = list(spec.get("obstacle_seed_sequence", []) or [])
    if terrain_seed_sequence:
        env["TERRAIN_SEED_SEQUENCE"] = ",".join(str(int(v)) for v in terrain_seed_sequence)
    if terrain_variant_seed_sequence:
        env["TERRAIN_VARIANT_SEED_SEQUENCE"] = ",".join(str(int(v)) for v in terrain_variant_seed_sequence)
    if obstacle_seed_sequence:
        env["OBSTACLE_SEED_SEQUENCE"] = ",".join(str(int(v)) for v in obstacle_seed_sequence)
    return env


def _tail_text_file(path: Path, max_lines: int = 80) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def _run_isolated_eval(
    *,
    model_dir: Path,
    variant: str,
    spec: Dict[str, Any],
    save_dir: Path,
    eval_python_bin: Optional[str],
    quiet_eval: bool,
    dry_run: bool,
    stream_output: bool = False,
    stream_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_model_dir = _resolve_model_variant_dir(model_dir, variant)
    save_dir = save_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    spec_copy = dict(spec)
    spec_copy["resolved_model_dir"] = str(resolved_model_dir)
    spec_copy["requested_variant"] = str(variant)
    _save_json(save_dir / "diagnostic_spec.json", spec_copy)

    command = [
        "bash",
        str(REPO_ROOT / "run_evaluation.sh"),
        str(resolved_model_dir),
        str(int(spec.get("episodes", 0) or 0)),
        str(save_dir),
        str(spec.get("default_positions_file", "")),
        "true",
        "false",
    ]
    log_path = save_dir / "diagnostic_eval.log"
    record: Dict[str, Any] = {
        "command": command,
        "resolved_model_dir": str(resolved_model_dir),
        "log_path": str(log_path),
        "save_dir": str(save_dir),
        "results_path": str(save_dir / "evaluation_results.json"),
        "dry_run": bool(dry_run),
    }
    if dry_run:
        record["summary"] = None
        return record

    env = _build_eval_env(spec, quiet_eval=quiet_eval)
    if eval_python_bin:
        env["EVAL_PYTHON_BIN"] = str(eval_python_bin)
    if stream_output:
        env["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "w", encoding="utf-8") as log_file:
        if not stream_output:
            process = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
            returncode = int(process.returncode)
        else:
            prefix = str(stream_prefix or save_dir.name).strip()
            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                with _STREAM_LOCK:
                    sys.stdout.write(f"[{prefix}] {line}")
                    sys.stdout.flush()
            process.stdout.close()
            returncode = int(process.wait())
    record["returncode"] = returncode
    if returncode != 0:
        tail = _tail_text_file(log_path, max_lines=60)
        raise RuntimeError(
            f"Evaluation failed (rc={returncode}) using "
            f"{eval_python_bin or 'default python3'}: {log_path}\n{tail}"
        )
    results_path = save_dir / "evaluation_results.json"
    if not results_path.exists():
        tail = _tail_text_file(log_path, max_lines=60)
        raise RuntimeError(
            f"Missing evaluation_results.json after evaluation using "
            f"{eval_python_bin or 'default python3'}: {results_path}\n"
            f"log: {log_path}\n{tail}"
        )
    results = _load_json(results_path)
    record["summary"] = dict(results.get("summary", {}) or {})
    record["evaluation_setup"] = dict(results.get("evaluation_setup", {}) or {})
    return record


def _run_eval_job(job: EvalJobSpec) -> Dict[str, Any]:
    record = _run_isolated_eval(
        model_dir=job.model_dir,
        variant=job.variant,
        spec=job.spec,
        save_dir=job.save_dir,
        eval_python_bin=job.eval_python_bin,
        quiet_eval=job.quiet_eval,
        dry_run=job.dry_run,
        stream_output=job.stream_output,
        stream_prefix=job.description,
    )
    return {"job": job, "record": record}


def _execute_eval_jobs(jobs: Sequence[EvalJobSpec], max_parallel: int) -> List[Dict[str, Any]]:
    queued_jobs = list(jobs)
    if not queued_jobs:
        return []

    total = len(queued_jobs)
    workers = max(1, int(max_parallel))
    results: List[Dict[str, Any]] = []

    if workers <= 1 or total <= 1:
        for idx, job in enumerate(queued_jobs, start=1):
            print(f"[eval {idx}/{total}] {job.description}", flush=True)
            try:
                results.append(_run_eval_job(job))
            except Exception as exc:
                raise RuntimeError(f"Evaluation job failed: {job.description}\n{exc}") from exc
        return results

    print(f"Launching {total} isolated evaluation jobs with max_parallel={workers}", flush=True)
    failures: List[Tuple[EvalJobSpec, Exception]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_run_eval_job, job): job for job in queued_jobs}
        for future in as_completed(future_map):
            job = future_map[future]
            completed += 1
            try:
                result = future.result()
            except Exception as exc:
                failures.append((job, exc))
                print(f"[eval {completed}/{total}] FAILED: {job.description}", flush=True)
                continue
            results.append(result)
            print(f"[eval {completed}/{total}] done: {job.description}", flush=True)

    if failures:
        preview = []
        for job, exc in failures[:5]:
            preview.append(f"- {job.description}: {exc}")
        if len(failures) > 5:
            preview.append(f"- ... and {len(failures) - 5} more failures")
        raise RuntimeError("One or more isolated evaluation jobs failed:\n" + "\n".join(preview))

    return results


def _write_rows_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["empty"])
        return
    headers: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                headers.append(key)
                seen.add(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate_mean(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    values = []
    for row in rows:
        value = _safe_float(row.get(key))
        if value is not None:
            values.append(float(value))
    return _mean(values)


def _plot_obstacle_shift(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    if plt is None or not rows:
        return
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        label = str(row.get("label"))
        case = str(row.get("case"))
        grouped.setdefault(label, {}).setdefault(case, []).append(float(row.get("team_success_rate", 0.0) or 0.0))
    labels = sorted(grouped.keys())
    case_order = [
        case
        for case in ("train_distribution", "normal_test", "trainlike", "heldout")
        if any(grouped[label].get(case) for label in labels)
    ]
    if not case_order:
        return
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5.2))
    width = min(0.72 / max(len(case_order), 1), 0.36)
    base_offset = -0.5 * width * (len(case_order) - 1)
    display_labels = {
        "train_distribution": "Training Distribution Replay",
        "normal_test": "Normal Test",
        "trainlike": "Train-like Obstacles",
        "heldout": "Held-out Obstacles",
    }
    for idx, case in enumerate(case_order):
        ys = [_mean(grouped[label].get(case, [])) or 0.0 for label in labels]
        offset = base_offset + idx * width
        ax.bar([i + offset for i in x], ys, width=width, label=display_labels.get(case, case))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Team Success Rate")
    ax.set_title("Training Distribution Replay vs Normal Test")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_length_sensitivity(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    if plt is None or not rows:
        return
    grouped: Dict[str, Dict[float, List[float]]] = {}
    for row in rows:
        label = str(row.get("label"))
        mult = float(row.get("episode_length_multiplier"))
        grouped.setdefault(label, {}).setdefault(mult, []).append(float(row.get("team_success_rate", 0.0) or 0.0))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for label in sorted(grouped.keys()):
        xs = sorted(grouped[label].keys())
        ys = [(_mean(grouped[label][x]) or 0.0) for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=label)
    ax.set_xlabel("Episode Length Multiplier")
    ax.set_ylabel("Team Success Rate")
    ax.set_title("Episode-Length Sensitivity")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_training_gap(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    if plt is None or not rows:
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    for row in rows:
        x = _safe_float(row.get("training_tail100_success_mean"))
        y = _safe_float(row.get("train_distribution_team_success_rate"))
        label = f"{row.get('label')}|seed{row.get('seed')}"
        if x is None or y is None:
            continue
        ax.scatter([x], [y], s=60)
        ax.annotate(label, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Training Tail100 Success Mean")
    ax.set_ylabel("Deterministic Eval Success (Training Distribution Replay)")
    ax.set_title("Training vs Deterministic Eval Gap")
    ax.grid(True, linestyle="--", alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_selection_stability(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    if plt is None or not rows:
        return
    keys = []
    for row in rows:
        key = f"{row.get('validation_episodes')}ep/{row.get('validation_seed')}"
        if key not in keys:
            keys.append(key)
    rows_sorted = sorted(rows, key=lambda item: (str(item.get("label")), int(item.get("seed")), int(item.get("validation_episodes")), int(item.get("validation_seed"))))
    labels = [f"{row.get('label')}\nseed{row.get('seed')}\n{row.get('validation_episodes')}ep" for row in rows_sorted]
    values = [float(row.get("selected_test_team_success_rate", 0.0) or 0.0) for row in rows_sorted]
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.7), 5.5))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Selected Candidate Test SR")
    ax.set_title("Selection Stability")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _build_report(
    *,
    output_dir: Path,
    contexts: Sequence[ExperimentSeedContext],
    obstacle_rows: Sequence[Dict[str, Any]],
    training_gap_rows: Sequence[Dict[str, Any]],
    selection_rows: Sequence[Dict[str, Any]],
    length_rows: Sequence[Dict[str, Any]],
) -> str:
    total_cases = len(obstacle_rows) + len(training_gap_rows) + len(selection_rows) + len(length_rows)
    lines = [
        "# Isolated Level-2 Diagnosis Report",
        "",
        f"- Created at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Output dir: `{output_dir}`",
        f"- Experiment-seed contexts: {len(contexts)}",
        f"- Recorded suite rows: {total_cases}",
        "",
        "## Quick Read",
        "",
    ]

    if not obstacle_rows and not training_gap_rows and not selection_rows and not length_rows:
        lines.append("- No suite rows were recorded.")
    if all(
        (_safe_float(row.get("team_success_rate")) is None)
        and (_safe_float(row.get("selected_test_team_success_rate")) is None)
        for row in [*obstacle_rows, *selection_rows, *length_rows]
    ):
        lines.append("- Dry run only: evaluation metrics have not been executed yet.")

    if obstacle_rows:
        deltas = []
        grouped: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = {}
        for row in obstacle_rows:
            grouped.setdefault((str(row["label"]), int(row["seed"])), {})[str(row["case"])] = row
        for _, pair in grouped.items():
            replay_sr = _safe_float((pair.get("train_distribution") or {}).get("team_success_rate"))
            normal_sr = _safe_float((pair.get("normal_test") or {}).get("team_success_rate"))
            if replay_sr is not None and normal_sr is not None:
                deltas.append(replay_sr - normal_sr)
        mean_delta = _mean(deltas)
        lines.append(
            f"- Train-distribution replay delta (replay SR minus normal-test SR): "
            f"{mean_delta:.4f}" if mean_delta is not None else "- Train-distribution replay delta: N/A"
        )

    if training_gap_rows:
        gaps = []
        for row in training_gap_rows:
            train_tail = _safe_float(row.get("training_tail100_success_mean"))
            replay_sr = _safe_float(row.get("train_distribution_team_success_rate"))
            if train_tail is not None and replay_sr is not None:
                gaps.append(train_tail - replay_sr)
        mean_gap = _mean(gaps)
        lines.append(
            f"- Training optimism mean gap (tail100 SR minus deterministic replay SR): "
            f"{mean_gap:.4f}" if mean_gap is not None else "- Training optimism mean gap: N/A"
        )

    if selection_rows:
        unique_selected = sorted(
            {
                str(row.get("selected_candidate"))
                for row in selection_rows
                if row.get("selected_candidate")
            }
        )
        if unique_selected:
            lines.append(f"- Selection stability observed candidates: {', '.join(unique_selected)}")
        else:
            lines.append("- Selection stability observed candidates: N/A")

    if length_rows:
        grouped_length: Dict[float, List[float]] = {}
        for row in length_rows:
            mult = float(row.get("episode_length_multiplier"))
            value = _safe_float(row.get("team_success_rate"))
            if value is not None:
                grouped_length.setdefault(mult, []).append(float(value))
        for mult in sorted(grouped_length):
            mean_sr = _mean(grouped_length[mult])
            lines.append(f"- Mean SR @ length x{mult:.2f}: {mean_sr:.4f}" if mean_sr is not None else f"- Mean SR @ length x{mult:.2f}: N/A")

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `metadata/`: copied parent config, parent summary and original spec snapshots",
            "- `generated_testsets/`: all newly generated validation/test episode position files",
            "- `runs/`: raw isolated evaluation outputs",
            "- `summaries/`: CSV/JSON tables for each diagnosis suite",
            "- `plots/`: quick-look figures",
            "",
            "## Note",
            "",
            "This suite is read-only with respect to the original batch: it never writes into "
            "`ablation_experiments/...` and only consumes historical artifacts from there.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an isolated diagnosis suite for poor post-eval generalization without touching the original batch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("parent_batch_dir", type=str, help="Existing multi-seed parent batch directory.")
    parser.add_argument("--output-dir", type=str, default=None, help="Fresh diagnostics directory. Defaults to diagnostics/level2_diag_<timestamp>.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Experiment labels to diagnose. Defaults to matd3_separated_gradient only for safety/runtime.",
    )
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seed filter.")
    parser.add_argument(
        "--suites",
        nargs="+",
        default=list(SUITE_NAMES),
        choices=list(SUITE_NAMES),
        help="Diagnosis suites to execute.",
    )
    parser.add_argument(
        "--selection-candidates",
        type=str,
        default=",".join(DEFAULT_SELECTION_CANDIDATES),
        help="Checkpoint candidates used in the selection suite.",
    )
    parser.add_argument(
        "--selection-validation-episodes",
        type=str,
        default=",".join(str(v) for v in DEFAULT_SELECTION_VALIDATION_EPISODES),
        help="Validation episode counts for the selection stability sweep.",
    )
    parser.add_argument(
        "--extra-validation-seeds",
        type=int,
        default=2,
        help="How many additional validation seeds to derive beyond the original validation seed.",
    )
    parser.add_argument(
        "--episode-length-multipliers",
        type=str,
        default=",".join(str(v) for v in DEFAULT_EPISODE_LENGTH_MULTIPLIERS),
        help="Episode length multipliers used in the length sensitivity suite.",
    )
    parser.add_argument(
        "--train-distribution-episodes",
        type=int,
        default=DEFAULT_TRAIN_DISTRIBUTION_EPISODES,
        help="Episode count used by the training-distribution replay diagnostic.",
    )
    parser.add_argument(
        "--fixed-variant",
        type=str,
        default="best_by_team_sr",
        help="Checkpoint variant used by obstacle-shift and length-sensitivity suites.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=2,
        help="Maximum number of isolated evaluation subprocesses to run at once.",
    )
    parser.add_argument(
        "--stream-eval-output",
        action="store_true",
        help="Stream child evaluation stdout/stderr to the terminal while still writing diagnostic_eval.log.",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Do not generate quick-look PNG plots.")
    parser.add_argument("--dry-run", action="store_true", help="Build plans/specs only; do not launch evaluations.")
    parser.add_argument("--quiet-eval", action="store_true", help="Run child evaluations with QUIET_OUTPUT=1.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parent_batch_dir = _resolve_parent_batch_dir(args.parent_batch_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (DEFAULT_OUTPUT_ROOT / f"level2_diag_{timestamp}").resolve()
    )
    if output_dir.exists():
        raise RuntimeError(f"Refusing to reuse an existing diagnostics directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (output_dir / "generated_testsets").mkdir(parents=True, exist_ok=True)
    (output_dir / "runs").mkdir(parents=True, exist_ok=True)
    (output_dir / "summaries").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    experiments = list(args.experiments or DEFAULT_EXPERIMENTS)
    selected_seeds = _parse_csv_strings(args.seeds, cast=int)
    selection_candidates = _parse_csv_strings(args.selection_candidates, cast=str) or list(DEFAULT_SELECTION_CANDIDATES)
    validation_episode_sweep = _parse_int_list(args.selection_validation_episodes, DEFAULT_SELECTION_VALIDATION_EPISODES)
    episode_length_multipliers = _parse_float_list(args.episode_length_multipliers, DEFAULT_EPISODE_LENGTH_MULTIPLIERS)

    parent_config, parent_summary, contexts = _collect_contexts(
        parent_batch_dir=parent_batch_dir,
        experiments=experiments,
        selected_seeds=selected_seeds,
    )

    _copy_json(parent_batch_dir / "config.json", output_dir / "metadata" / "parent_config.json")
    _copy_json(parent_batch_dir / "plots" / "latest_summary.json", output_dir / "metadata" / "parent_latest_summary.json")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "parent_batch_dir": str(parent_batch_dir),
        "output_dir": str(output_dir),
        "experiments": experiments,
        "selected_seeds": selected_seeds or [],
        "suites": list(args.suites),
        "selection_candidates": selection_candidates,
        "selection_validation_episodes": validation_episode_sweep,
        "extra_validation_seeds": int(args.extra_validation_seeds),
        "episode_length_multipliers": episode_length_multipliers,
        "train_distribution_episodes": int(args.train_distribution_episodes),
        "fixed_variant": str(args.fixed_variant),
        "max_parallel": max(1, int(args.max_parallel)),
        "stream_eval_output": bool(args.stream_eval_output),
        "dry_run": bool(args.dry_run),
    }
    _save_json(output_dir / "metadata" / "suite_manifest.json", manifest)

    obstacle_rows: List[Dict[str, Any]] = []
    training_gap_rows: List[Dict[str, Any]] = []
    selection_rows: List[Dict[str, Any]] = []
    selection_candidate_rows: List[Dict[str, Any]] = []
    length_rows: List[Dict[str, Any]] = []
    prepared_specs: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = {}
    max_parallel = max(1, int(args.max_parallel))

    for ctx in contexts:
        label_seed_root = output_dir / "metadata" / ctx.label / f"seed_{ctx.seed}"
        label_seed_root.mkdir(parents=True, exist_ok=True)
        _copy_json(ctx.test_spec_path, label_seed_root / "original_test_spec.json")
        _copy_json(ctx.validation_spec_path, label_seed_root / "original_validation_spec.json")

        base_test_spec = _load_json(ctx.test_spec_path)
        base_validation_spec = _load_json(ctx.validation_spec_path)
        prepared_specs[(ctx.label, ctx.seed)] = {
            "test": base_test_spec,
            "validation": base_validation_spec,
        }

        base_test_positions_dir = Path(str(base_test_spec.get("episode_positions_dir", "")))
        base_validation_positions_dir = Path(str(base_validation_spec.get("episode_positions_dir", "")))
        if base_test_positions_dir.exists():
            (label_seed_root / "original_testset_meta.txt").write_text(
                f"original_episode_positions_dir={base_test_positions_dir}\n",
                encoding="utf-8",
            )
        if base_validation_positions_dir.exists():
            (label_seed_root / "original_validation_testset_meta.txt").write_text(
                f"original_episode_positions_dir={base_validation_positions_dir}\n",
                encoding="utf-8",
            )

    if "obstacle_shift" in args.suites or "training_gap" in args.suites:
        obstacle_jobs: List[EvalJobSpec] = []
        for ctx in contexts:
            base_test_spec = prepared_specs[(ctx.label, ctx.seed)]["test"]
            case_specs = [
                {
                    "case": "train_distribution",
                    "episodes": int(args.train_distribution_episodes),
                    "seed": int(
                        base_test_spec.get("terrain_seed", base_test_spec.get("scenario_seed", base_test_spec.get("seed", 0)))
                        or 0
                    ),
                    "episode_length_multiplier": 1.0,
                    "obstacle_mode": "trainlike",
                },
                {
                    "case": "normal_test",
                    "episodes": int(base_test_spec.get("episodes", 0) or 0),
                    "seed": int(base_test_spec.get("seed", 0) or 0),
                    "episode_length_multiplier": float(base_test_spec.get("episode_length_multiplier", 1.0) or 1.0),
                    "obstacle_mode": "heldout",
                },
            ]
            for case_cfg in case_specs:
                case = str(case_cfg["case"])
                spec = _materialize_spec(
                    base_test_spec,
                    output_positions_dir=output_dir / "generated_testsets" / ctx.label / f"seed_{ctx.seed}" / "obstacle_shift" / case,
                    episodes=int(case_cfg["episodes"]),
                    seed=int(case_cfg["seed"]),
                    episode_length_multiplier=float(case_cfg["episode_length_multiplier"]),
                    obstacle_mode=str(case_cfg["obstacle_mode"]),
                )
                obstacle_jobs.append(
                    EvalJobSpec(
                        suite="obstacle_shift",
                        label=ctx.label,
                        seed=ctx.seed,
                        description=f"{ctx.label}/seed{ctx.seed} obstacle_shift {case}",
                        model_dir=ctx.model_root,
                        variant=str(args.fixed_variant),
                        spec=spec,
                        save_dir=output_dir / "runs" / ctx.label / f"seed_{ctx.seed}" / "obstacle_shift" / case,
                        eval_python_bin=ctx.eval_python_bin,
                        quiet_eval=bool(args.quiet_eval),
                        dry_run=bool(args.dry_run),
                        stream_output=bool(args.stream_eval_output),
                        metadata={"case": case, "obstacle_mode": str(case_cfg["obstacle_mode"])},
                    )
                )

        for result in _execute_eval_jobs(obstacle_jobs, max_parallel=max_parallel):
            job = result["job"]
            record = result["record"]
            summary = record.get("summary") or {}
            obstacle_rows.append(
                {
                    "suite": "obstacle_shift",
                    "label": job.label,
                    "seed": job.seed,
                    "case": str(job.metadata.get("case")),
                    "variant": job.variant,
                    "episodes": int(job.spec.get("episodes", 0) or 0),
                    "episode_length_multiplier": float(job.spec.get("episode_length_multiplier", 1.0) or 1.0),
                    "obstacle_sequence_kind": str(job.metadata.get("case")),
                    "team_success_rate": _safe_float(summary.get("team_success_rate")),
                    "collision_free_rate": _safe_float(summary.get("collision_free_rate")),
                    "avg_collision_count": _safe_float(summary.get("avg_collision_count")),
                    "avg_reward": _safe_float(summary.get("avg_reward")),
                    "avg_agent_final_goal_distance": _safe_float(summary.get("avg_agent_final_goal_distance")),
                    "avg_team_total_path_length": _safe_float(summary.get("avg_team_total_path_length")),
                    "save_dir": record.get("save_dir"),
                    "results_path": record.get("results_path"),
                }
            )

    if "training_gap" in args.suites:
        obstacle_lookup = {
            (str(row["label"]), int(row["seed"]), str(row["case"])): row
            for row in obstacle_rows
        }
        for ctx in contexts:
            replay_row = obstacle_lookup.get((ctx.label, ctx.seed, "train_distribution"))
            normal_test_row = obstacle_lookup.get((ctx.label, ctx.seed, "normal_test"))
            training_gap_rows.append(
                {
                    "suite": "training_gap",
                    "label": ctx.label,
                    "seed": ctx.seed,
                    "training_tail100_success_mean": ctx.training_tail100_success_mean,
                    "training_tail100_reward_mean": ctx.training_tail100_reward_mean,
                    "train_distribution_team_success_rate": _safe_float((replay_row or {}).get("team_success_rate")),
                    "train_distribution_avg_collision_count": _safe_float((replay_row or {}).get("avg_collision_count")),
                    "train_distribution_avg_reward": _safe_float((replay_row or {}).get("avg_reward")),
                    "train_distribution_episodes": _safe_int((replay_row or {}).get("episodes")),
                    "normal_test_team_success_rate": _safe_float((normal_test_row or {}).get("team_success_rate")),
                    "normal_test_avg_collision_count": _safe_float((normal_test_row or {}).get("avg_collision_count")),
                    "normal_test_avg_reward": _safe_float((normal_test_row or {}).get("avg_reward")),
                    "normal_test_episodes": _safe_int((normal_test_row or {}).get("episodes")),
                    "existing_post_eval_team_success_rate": _safe_float(ctx.existing_post_eval_summary.get("team_success_rate")),
                    "existing_post_eval_avg_reward": _safe_float(ctx.existing_post_eval_summary.get("avg_reward")),
                }
            )

    if "selection" in args.suites:
        selection_candidate_jobs: List[EvalJobSpec] = []
        selection_group_context: Dict[Tuple[str, int, int, int], ExperimentSeedContext] = {}
        for ctx in contexts:
            base_validation_spec = prepared_specs[(ctx.label, ctx.seed)]["validation"]
            base_validation_seed = int(base_validation_spec.get("seed", 0) or 0)
            validation_seed_sweep = [
                base_validation_seed,
                *_derive_extra_validation_seeds(base_validation_seed, int(args.extra_validation_seeds)),
            ]
            selection_root = output_dir / "runs" / ctx.label / f"seed_{ctx.seed}" / "selection"
            for validation_seed in validation_seed_sweep:
                for validation_episodes in validation_episode_sweep:
                    group_key = (ctx.label, ctx.seed, int(validation_seed), int(validation_episodes))
                    selection_group_context[group_key] = ctx
                    spec = _materialize_spec(
                        base_validation_spec,
                        output_positions_dir=output_dir / "generated_testsets" / ctx.label / f"seed_{ctx.seed}" / "selection" / f"val_seed_{validation_seed}_ep_{validation_episodes}",
                        episodes=int(validation_episodes),
                        seed=int(validation_seed),
                        episode_length_multiplier=float(base_validation_spec.get("episode_length_multiplier", 1.0) or 1.0),
                        obstacle_mode="heldout",
                    )
                    for variant in selection_candidates:
                        selection_candidate_jobs.append(
                            EvalJobSpec(
                                suite="selection_candidate",
                                label=ctx.label,
                                seed=ctx.seed,
                                description=(
                                    f"{ctx.label}/seed{ctx.seed} selection validation "
                                    f"seed={validation_seed} ep={validation_episodes} variant={variant}"
                                ),
                                model_dir=ctx.model_root,
                                variant=str(variant),
                                spec=spec,
                                save_dir=selection_root / f"validation_seed_{validation_seed}_ep_{validation_episodes}" / variant,
                                eval_python_bin=ctx.eval_python_bin,
                                quiet_eval=bool(args.quiet_eval),
                                dry_run=bool(args.dry_run),
                                stream_output=bool(args.stream_eval_output),
                                metadata={
                                    "validation_seed": int(validation_seed),
                                    "validation_episodes": int(validation_episodes),
                                },
                            )
                        )

        grouped_candidate_records: Dict[Tuple[str, int, int, int], List[Dict[str, Any]]] = {}
        for result in _execute_eval_jobs(selection_candidate_jobs, max_parallel=max_parallel):
            job = result["job"]
            record = result["record"]
            summary = record.get("summary") or {}
            score = _score_post_eval_summary(summary if isinstance(summary, dict) else {})
            validation_seed = int(job.metadata.get("validation_seed"))
            validation_episodes = int(job.metadata.get("validation_episodes"))
            selection_candidate_rows.append(
                {
                    "suite": "selection_candidate",
                    "label": job.label,
                    "seed": job.seed,
                    "validation_seed": validation_seed,
                    "validation_episodes": validation_episodes,
                    "candidate": job.variant,
                    "score": list(score),
                    "team_success_rate": _safe_float(summary.get("team_success_rate")),
                    "collision_free_rate": _safe_float(summary.get("collision_free_rate")),
                    "avg_team_final_goal_distance": _safe_float(summary.get("avg_team_final_goal_distance")),
                    "avg_collision_count": _safe_float(summary.get("avg_collision_count")),
                    "avg_team_total_path_length": _safe_float(summary.get("avg_team_total_path_length")),
                    "save_dir": record.get("save_dir"),
                    "results_path": record.get("results_path"),
                }
            )
            grouped_candidate_records.setdefault(
                (job.label, job.seed, validation_seed, validation_episodes),
                [],
            ).append({"variant": job.variant, "record": record, "score": score})

        selected_group_rows: Dict[Tuple[str, int, int, int], Dict[str, Any]] = {}
        selection_test_jobs: List[EvalJobSpec] = []
        for group_key, candidate_records in sorted(grouped_candidate_records.items()):
            ctx = selection_group_context[group_key]
            validation_seed = int(group_key[2])
            validation_episodes = int(group_key[3])
            if args.dry_run:
                selected_group_rows[group_key] = {
                    "suite": "selection",
                    "label": ctx.label,
                    "seed": ctx.seed,
                    "validation_seed": validation_seed,
                    "validation_episodes": validation_episodes,
                    "selected_candidate": None,
                    "selected_record": {},
                }
                continue

            selected = max(candidate_records, key=lambda item: (tuple(item["score"]), item["variant"]))
            selected_variant = str(selected["variant"])
            selected_record = dict(selected["record"])
            base_test_spec = prepared_specs[(ctx.label, ctx.seed)]["test"]
            test_spec = _materialize_spec(
                base_test_spec,
                output_positions_dir=output_dir / "generated_testsets" / ctx.label / f"seed_{ctx.seed}" / "selection" / f"test_selected_from_{validation_seed}_{validation_episodes}",
                episodes=int(base_test_spec.get("episodes", 0) or 0),
                seed=int(base_test_spec.get("seed", 0) or 0),
                episode_length_multiplier=float(base_test_spec.get("episode_length_multiplier", 1.0) or 1.0),
                obstacle_mode="heldout",
            )
            selection_test_jobs.append(
                EvalJobSpec(
                    suite="selection_test",
                    label=ctx.label,
                    seed=ctx.seed,
                    description=(
                        f"{ctx.label}/seed{ctx.seed} selection test "
                        f"from validation seed={validation_seed} ep={validation_episodes} variant={selected_variant}"
                    ),
                    model_dir=ctx.model_root,
                    variant=selected_variant,
                    spec=test_spec,
                    save_dir=output_dir / "runs" / ctx.label / f"seed_{ctx.seed}" / "selection" / f"selected_test_seed_{validation_seed}_ep_{validation_episodes}" / selected_variant,
                    eval_python_bin=ctx.eval_python_bin,
                    quiet_eval=bool(args.quiet_eval),
                    dry_run=bool(args.dry_run),
                    stream_output=bool(args.stream_eval_output),
                    metadata={
                        "validation_seed": validation_seed,
                        "validation_episodes": validation_episodes,
                    },
                )
            )
            selected_group_rows[group_key] = {
                "suite": "selection",
                "label": ctx.label,
                "seed": ctx.seed,
                "validation_seed": validation_seed,
                "validation_episodes": validation_episodes,
                "selected_candidate": selected_variant,
                "selected_record": selected_record,
            }

        selection_test_results: Dict[Tuple[str, int, int, int], Dict[str, Any]] = {}
        for result in _execute_eval_jobs(selection_test_jobs, max_parallel=max_parallel):
            job = result["job"]
            group_key = (
                job.label,
                job.seed,
                int(job.metadata.get("validation_seed")),
                int(job.metadata.get("validation_episodes")),
            )
            selection_test_results[group_key] = result["record"]

        for group_key, base_row in sorted(selected_group_rows.items()):
            selected_record = dict(base_row.get("selected_record") or {})
            test_record = dict(selection_test_results.get(group_key) or {})
            selection_rows.append(
                {
                    "suite": "selection",
                    "label": base_row["label"],
                    "seed": base_row["seed"],
                    "validation_seed": base_row["validation_seed"],
                    "validation_episodes": base_row["validation_episodes"],
                    "selected_candidate": base_row["selected_candidate"],
                    "selected_validation_team_success_rate": _safe_float((selected_record.get("summary") or {}).get("team_success_rate")),
                    "selected_validation_avg_collision_count": _safe_float((selected_record.get("summary") or {}).get("avg_collision_count")),
                    "selected_test_team_success_rate": _safe_float((test_record.get("summary") or {}).get("team_success_rate")),
                    "selected_test_avg_collision_count": _safe_float((test_record.get("summary") or {}).get("avg_collision_count")),
                    "selected_test_avg_reward": _safe_float((test_record.get("summary") or {}).get("avg_reward")),
                    "selected_test_save_dir": test_record.get("save_dir"),
                    "selected_test_results_path": test_record.get("results_path"),
                }
            )

    if "length" in args.suites:
        length_jobs: List[EvalJobSpec] = []
        for ctx in contexts:
            base_test_spec = prepared_specs[(ctx.label, ctx.seed)]["test"]
            for multiplier in episode_length_multipliers:
                spec = _materialize_spec(
                    base_test_spec,
                    output_positions_dir=output_dir / "generated_testsets" / ctx.label / f"seed_{ctx.seed}" / "length" / f"x_{multiplier:.2f}",
                    episodes=int(base_test_spec.get("episodes", 0) or 0),
                    seed=int(base_test_spec.get("seed", 0) or 0),
                    episode_length_multiplier=float(multiplier),
                    obstacle_mode="heldout",
                )
                length_jobs.append(
                    EvalJobSpec(
                        suite="length",
                        label=ctx.label,
                        seed=ctx.seed,
                        description=f"{ctx.label}/seed{ctx.seed} length x{float(multiplier):.2f}",
                        model_dir=ctx.model_root,
                        variant=str(args.fixed_variant),
                        spec=spec,
                        save_dir=output_dir / "runs" / ctx.label / f"seed_{ctx.seed}" / "length" / f"x_{multiplier:.2f}",
                        eval_python_bin=ctx.eval_python_bin,
                        quiet_eval=bool(args.quiet_eval),
                        dry_run=bool(args.dry_run),
                        stream_output=bool(args.stream_eval_output),
                        metadata={"episode_length_multiplier": float(multiplier)},
                    )
                )

        for result in _execute_eval_jobs(length_jobs, max_parallel=max_parallel):
            job = result["job"]
            record = result["record"]
            summary = record.get("summary") or {}
            length_rows.append(
                {
                    "suite": "length",
                    "label": job.label,
                    "seed": job.seed,
                    "variant": job.variant,
                    "episode_length_multiplier": float(job.metadata.get("episode_length_multiplier")),
                    "team_success_rate": _safe_float(summary.get("team_success_rate")),
                    "avg_collision_count": _safe_float(summary.get("avg_collision_count")),
                    "avg_reward": _safe_float(summary.get("avg_reward")),
                    "avg_first_reach_step": _safe_float(summary.get("avg_first_reach_step")),
                    "avg_agent_final_goal_distance": _safe_float(summary.get("avg_agent_final_goal_distance")),
                    "save_dir": record.get("save_dir"),
                    "results_path": record.get("results_path"),
                }
            )

    obstacle_rows = sorted(obstacle_rows, key=lambda row: (str(row.get("label")), int(row.get("seed", 0)), str(row.get("case"))))
    training_gap_rows = sorted(training_gap_rows, key=lambda row: (str(row.get("label")), int(row.get("seed", 0))))
    selection_candidate_rows = sorted(
        selection_candidate_rows,
        key=lambda row: (
            str(row.get("label")),
            int(row.get("seed", 0)),
            int(row.get("validation_episodes", 0)),
            int(row.get("validation_seed", 0)),
            str(row.get("candidate")),
        ),
    )
    selection_rows = sorted(
        selection_rows,
        key=lambda row: (
            str(row.get("label")),
            int(row.get("seed", 0)),
            int(row.get("validation_episodes", 0)),
            int(row.get("validation_seed", 0)),
        ),
    )
    length_rows = sorted(
        length_rows,
        key=lambda row: (
            str(row.get("label")),
            int(row.get("seed", 0)),
            float(row.get("episode_length_multiplier", 0.0)),
        ),
    )

    _write_rows_csv(output_dir / "summaries" / "obstacle_shift_rows.csv", obstacle_rows)
    _write_rows_csv(output_dir / "summaries" / "training_gap_rows.csv", training_gap_rows)
    _write_rows_csv(output_dir / "summaries" / "selection_rows.csv", selection_rows)
    _write_rows_csv(output_dir / "summaries" / "selection_candidate_rows.csv", selection_candidate_rows)
    _write_rows_csv(output_dir / "summaries" / "length_rows.csv", length_rows)

    summary_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "parent_batch_dir": str(parent_batch_dir),
        "output_dir": str(output_dir),
        "dry_run": bool(args.dry_run),
        "max_parallel": max_parallel,
        "stream_eval_output": bool(args.stream_eval_output),
        "contexts": [
            {
                "label": ctx.label,
                "seed": ctx.seed,
                "child_batch_dir": str(ctx.child_batch_dir),
                "manifest_path": str(ctx.manifest_path),
                "log_dir": str(ctx.log_dir),
                "model_root": str(ctx.model_root),
                "eval_python_bin": ctx.eval_python_bin,
                "training_tail100_success_mean": ctx.training_tail100_success_mean,
                "training_tail100_reward_mean": ctx.training_tail100_reward_mean,
                "test_spec_path": str(ctx.test_spec_path),
                "validation_spec_path": str(ctx.validation_spec_path),
            }
            for ctx in contexts
        ],
        "suite_counts": {
            "obstacle_shift_rows": len(obstacle_rows),
            "training_gap_rows": len(training_gap_rows),
            "selection_rows": len(selection_rows),
            "selection_candidate_rows": len(selection_candidate_rows),
            "length_rows": len(length_rows),
        },
        "high_level": {
            "obstacle_shift_mean_train_distribution_sr": _aggregate_mean(
                [row for row in obstacle_rows if row.get("case") == "train_distribution"],
                "team_success_rate",
            ),
            "obstacle_shift_mean_normal_test_sr": _aggregate_mean(
                [row for row in obstacle_rows if row.get("case") == "normal_test"],
                "team_success_rate",
            ),
            "training_gap_mean_tail100_success": _aggregate_mean(training_gap_rows, "training_tail100_success_mean"),
            "training_gap_mean_train_distribution_eval_success": _aggregate_mean(
                training_gap_rows,
                "train_distribution_team_success_rate",
            ),
            "selection_mean_selected_test_sr": _aggregate_mean(selection_rows, "selected_test_team_success_rate"),
        },
    }
    _save_json(output_dir / "summaries" / "summary.json", summary_payload)

    if (not args.skip_plots) and (not args.dry_run):
        _plot_obstacle_shift(obstacle_rows, output_dir / "plots" / "obstacle_shift_team_sr.png")
        _plot_training_gap(training_gap_rows, output_dir / "plots" / "training_vs_deterministic_gap.png")
        _plot_selection_stability(selection_rows, output_dir / "plots" / "selection_stability.png")
        _plot_length_sensitivity(length_rows, output_dir / "plots" / "episode_length_sensitivity.png")

    report_text = _build_report(
        output_dir=output_dir,
        contexts=contexts,
        obstacle_rows=obstacle_rows,
        training_gap_rows=training_gap_rows,
        selection_rows=selection_rows,
        length_rows=length_rows,
    )
    (output_dir / "summaries" / "report.md").write_text(report_text, encoding="utf-8")

    print(
        textwrap.dedent(
            f"""
            Isolated diagnosis suite prepared successfully.
              parent batch : {parent_batch_dir}
              output dir   : {output_dir}
              dry run      : {args.dry_run}
              contexts     : {len(contexts)}
              obstacle rows: {len(obstacle_rows)}
              training rows: {len(training_gap_rows)}
              selection rows: {len(selection_rows)}
              length rows  : {len(length_rows)}
            """
        ).strip()
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
