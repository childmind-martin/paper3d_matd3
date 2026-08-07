#!/usr/bin/env python3
"""Validate and summarize the formal 4-model x 4-mode noise batch.

The validator intentionally does not trust the launcher's final summary.  It
recomputes identities, per-episode aggregates, cross-cell sequence parity, and
paired deltas directly from the frozen run specs and evaluation results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from tools.build_selector_protocol_batch_spec import validate_batch_spec
from selector_experiment_protocol import SELECTOR_PROTOCOL_CONFIG_BY_LABEL

MODEL_LABELS: Tuple[str, ...] = ()
EXPERIMENT_LABEL_BY_MODEL: Dict[str, str] = {}

MODE_CONFIG: "OrderedDict[str, Tuple[float, float]]" = OrderedDict(
    ()
)

EXPECTED_PROTOCOL_VERSION = 0
EXPECTED_EPISODES_PER_CELL = 30
EXPECTED_TRAIN_SEED = 0
EXPECTED_EVAL_NOISE_SEED = 0
EXPECTED_TRAIN_EPISODES = 0
EXPECTED_TRAINING_EPISODE_LENGTH = 0
EXPECTED_EPISODE_LENGTH_MULTIPLIER = 0.0
EXPECTED_PROCESS_SHARDS = 0
EXPECTED_PROCESS_WORKERS = 0
EXPECTED_SHARD_LAYOUT: Tuple[Tuple[int, int, int], ...] = ()
EXPECTED_EPISODE_LENGTH = 0
BATCH_SPEC: Dict[str, Any] = {}
BOOTSTRAP_SAMPLES = 20_000


class ValidationError(RuntimeError):
    pass


def _configure_protocol(output_root: Path) -> Dict[str, Any]:
    global MODEL_LABELS
    global EXPERIMENT_LABEL_BY_MODEL
    global MODE_CONFIG
    global EXPECTED_PROTOCOL_VERSION
    global EXPECTED_EPISODES_PER_CELL
    global EXPECTED_TRAIN_SEED
    global EXPECTED_EVAL_NOISE_SEED
    global EXPECTED_TRAIN_EPISODES
    global EXPECTED_TRAINING_EPISODE_LENGTH
    global EXPECTED_EPISODE_LENGTH_MULTIPLIER
    global EXPECTED_PROCESS_SHARDS
    global EXPECTED_PROCESS_WORKERS
    global EXPECTED_SHARD_LAYOUT
    global EXPECTED_EPISODE_LENGTH
    global BATCH_SPEC

    batch_spec_path = output_root / "batch_spec.json"
    payload = _load_json(batch_spec_path)
    errors = validate_batch_spec(payload)
    if errors:
        raise ValidationError("invalid batch_spec.json: " + "; ".join(errors))

    MODEL_LABELS = tuple(str(item["id"]) for item in payload["models"])
    EXPERIMENT_LABEL_BY_MODEL = {
        str(item["id"]): str(item["label"])
        for item in payload["models"]
    }
    MODE_CONFIG = OrderedDict(
        (
            str(item["id"]),
            (
                float(item["eval_noise_scale"]),
                float(item["eval_random_action_prob"]),
            ),
        )
        for item in payload["modes"]
    )
    EXPECTED_PROTOCOL_VERSION = int(payload["protocol_version"])
    EXPECTED_EPISODES_PER_CELL = int(payload["episodes"])
    EXPECTED_TRAIN_SEED = int(payload["train_seed"])
    EXPECTED_EVAL_NOISE_SEED = int(payload["eval_noise_seed"])
    EXPECTED_TRAIN_EPISODES = int(payload["train_episodes"])
    EXPECTED_TRAINING_EPISODE_LENGTH = int(
        payload["training_episode_length"]
    )
    EXPECTED_EPISODE_LENGTH_MULTIPLIER = float(
        payload["episode_length_multiplier"]
    )
    EXPECTED_PROCESS_SHARDS = int(payload["eval_process_shards"])
    EXPECTED_PROCESS_WORKERS = int(payload["eval_process_workers"])
    EXPECTED_EPISODE_LENGTH = int(payload["episode_length"])

    base_count, remainder = divmod(
        EXPECTED_EPISODES_PER_CELL,
        EXPECTED_PROCESS_SHARDS,
    )
    shard_layout = []
    start = 0
    for index in range(EXPECTED_PROCESS_SHARDS):
        count = base_count + (1 if index < remainder else 0)
        shard_layout.append((index, start, count))
        start += count
    EXPECTED_SHARD_LAYOUT = tuple(shard_layout)
    BATCH_SPEC = payload
    return payload


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationError(f"JSON root is not an object: {path}")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _actor_signature(model_dir: Path) -> str:
    actors = sorted(model_dir.glob("actor_*.weights.h5"))
    _require(bool(actors), f"model has no actor weights: {model_dir}")
    digest = hashlib.sha1()
    for actor in actors:
        digest.update(actor.name.encode("utf-8"))
        with actor.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _finite_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} is not numeric: {value!r}") from exc
    _require(math.isfinite(number), f"{name} is not finite: {number!r}")
    return number


def _assert_close(
    actual: Any,
    expected: Any,
    name: str,
    *,
    rel_tol: float = 1e-10,
    abs_tol: float = 1e-9,
) -> None:
    actual_number = _finite_float(actual, name)
    expected_number = _finite_float(expected, f"expected {name}")
    _require(
        math.isclose(
            actual_number,
            expected_number,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ),
        f"{name}={actual_number!r}, expected={expected_number!r}",
    )


def _assert_setup_mapping(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    name: str,
) -> None:
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        field_name = f"{name}.{key}"
        if isinstance(expected_value, bool):
            _require(
                isinstance(actual_value, bool)
                and actual_value is expected_value,
                f"{field_name}={actual_value!r}, expected={expected_value!r}",
            )
        elif isinstance(expected_value, (int, float)):
            _assert_close(
                actual_value,
                expected_value,
                field_name,
                rel_tol=1e-7,
                abs_tol=1e-8,
            )
        else:
            _require(
                actual_value == expected_value,
                f"{field_name}={actual_value!r}, expected={expected_value!r}",
            )


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    _require(bool(materialized), "cannot compute mean of empty data")
    return float(statistics.fmean(materialized))


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    _require(bool(sorted_values), "cannot compute percentile of empty data")
    position = (len(sorted_values) - 1) * fraction
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(sorted_values[low])
    weight = position - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


def _paired_bootstrap_ci(values: Sequence[float], key: str) -> Tuple[float, float]:
    _require(bool(values), f"cannot bootstrap empty paired values: {key}")
    seed_bytes = hashlib.sha256(key.encode("utf-8")).digest()[:8]
    rng = random.Random(int.from_bytes(seed_bytes, byteorder="big", signed=False))
    count = len(values)
    samples: List[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        samples.append(sum(values[rng.randrange(count)] for _ in range(count)) / count)
    samples.sort()
    return _percentile(samples, 0.025), _percentile(samples, 0.975)


def _exact_mcnemar_p(success_gains: int, success_losses: int) -> float:
    discordant = success_gains + success_losses
    if discordant == 0:
        return 1.0
    tail = min(success_gains, success_losses)
    probability = sum(math.comb(discordant, i) for i in range(tail + 1)) / (2**discordant)
    return float(min(1.0, 2.0 * probability))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _model_setup(training_result: Mapping[str, Any]) -> Dict[str, Any]:
    args = training_result.get("args")
    _require(isinstance(args, dict), "training results has no args object")
    return {
        "train_seed": int(args.get("seed")),
        "algorithm": str(args.get("algo", "")).strip().lower(),
        "gate_mode": args.get("cross_agent_reference_gate_mode"),
        "target_semantics": args.get("cross_agent_reference_target_semantics"),
        "selector_enabled": bool(args.get("cross_agent_reference_selector_enabled")),
        "selector_mode": args.get("cross_agent_reference_selector_mode"),
        "training_gpu_required": bool(
            (training_result.get("training_device") or {}).get(
                "require_gpu",
                False,
            )
        ),
        "training_physical_gpus": int(
            (training_result.get("training_device") or {}).get(
                "physical_gpus",
                0,
            )
            or 0
        ),
        "training_logical_gpus": int(
            (training_result.get("training_device") or {}).get(
                "logical_gpus",
                0,
            )
            or 0
        ),
        "training_num_envs": int(
            (training_result.get("training_parallelism") or {}).get(
                "num_envs",
                0,
            )
            or 0
        ),
        "training_synchronous_iterations": int(
            (training_result.get("training_parallelism") or {}).get(
                "synchronous_iterations",
                0,
            )
            or 0
        ),
        "training_environment_trajectories": int(
            (training_result.get("training_parallelism") or {}).get(
                "environment_trajectories",
                0,
            )
            or 0
        ),
        "selector_init_logit": _finite_float(
            args.get("cross_agent_reference_selector_init_logit"),
            "selector_init_logit",
        ),
        "advantage_ema_decay": _finite_float(
            args.get("cross_agent_reference_advantage_ema_decay"),
            "advantage_ema_decay",
        ),
        "collision_weight": _finite_float(args.get("collision_weight"), "collision_weight"),
        "collision_penalty_value": _finite_float(
            args.get("collision_penalty_value"), "collision_penalty_value"
        ),
        "unsafe_arrival_penalty": _finite_float(
            args.get("unsafe_arrival_penalty"), "unsafe_arrival_penalty"
        ),
        "clearance_quality_bonus_weight": _finite_float(
            args.get("clearance_quality_bonus_weight"), "clearance_quality_bonus_weight"
        ),
    }


def _optional_finite_values(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> List[float]:
    values: List[float] = []
    for row in rows:
        try:
            value = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _selector_training_diagnostics(
    batch_spec: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    diagnostics: List[Dict[str, Any]] = []
    for model in batch_spec["models"]:
        model_id = str(model["id"])
        label = str(model["label"])
        loss_history_path = Path(str(model["loss_history_path"])).resolve()
        try:
            rows = json.loads(loss_history_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValidationError(
                f"cannot read selector loss history {loss_history_path}: {exc}"
            ) from exc
        _require(
            isinstance(rows, list) and bool(rows),
            f"{model_id}: selector loss history is empty",
        )
        mapping_rows = [row for row in rows if isinstance(row, dict)]
        active_rows = [
            row
            for row in mapping_rows
            if _optional_finite_values([row], "cross_ref_active")
            and _optional_finite_values([row], "cross_ref_active")[0] > 0.5
        ]
        eligible_rows = [
            row
            for row in active_rows
            if _optional_finite_values([row], "cross_ref_valid_ratio")
            and _optional_finite_values([row], "cross_ref_valid_ratio")[0] > 0.0
        ]
        _require(bool(active_rows), f"{model_id}: no active reference updates")
        _require(bool(eligible_rows), f"{model_id}: no eligible reference updates")

        setup = SELECTOR_PROTOCOL_CONFIG_BY_LABEL[label]["env"]
        selector_mode = setup["CROSS_AGENT_REFERENCE_SELECTOR_MODE"]
        adaptive = "twin_advantage_head_tail" in selector_mode
        shared = selector_mode == "shared_twin_advantage_head_tail"

        def mean_value(key: str, source=active_rows):
            values = _optional_finite_values(source, key)
            return _mean(values) if values else None

        state = None
        if adaptive:
            state_path = (
                Path(str(model["model_path"]))
                / "cross_agent_reference_state.json"
            )
            state = _load_json(state_path)

        positive_selector_gradients = [
            value
            for value in _optional_finite_values(
                active_rows,
                "cross_ref_selector_gradient_norm",
            )
            if value > 0.0
        ]
        diagnostics.append(
            {
                "model": model_id,
                "experiment_label": label,
                "selector_mode": selector_mode,
                "logged_rows": len(mapping_rows),
                "active_reference_rows": len(active_rows),
                "eligible_reference_rows": len(eligible_rows),
                "mean_valid_ratio": mean_value(
                    "cross_ref_valid_ratio",
                    eligible_rows,
                ),
                "mean_reference_loss": mean_value(
                    "cross_ref_loss",
                    eligible_rows,
                ),
                "mean_head_twin_agreement": (
                    mean_value("cross_ref_head_twin_agreement_ratio")
                    if adaptive
                    else None
                ),
                "mean_tail_twin_agreement": (
                    mean_value("cross_ref_tail_twin_agreement_ratio")
                    if adaptive
                    else None
                ),
                "mean_head_multiplier": (
                    mean_value("cross_ref_head_multiplier")
                    if adaptive
                    else None
                ),
                "mean_tail_multiplier": (
                    mean_value("cross_ref_tail_multiplier")
                    if adaptive
                    else None
                ),
                "mean_head_suppressed_ratio": (
                    mean_value("cross_ref_head_suppressed_ratio")
                    if adaptive
                    else None
                ),
                "mean_tail_suppressed_ratio": (
                    mean_value("cross_ref_tail_suppressed_ratio")
                    if adaptive
                    else None
                ),
                "mean_selector_loss": (
                    mean_value("cross_ref_selector_loss")
                    if shared
                    else None
                ),
                "positive_selector_gradient_rows": (
                    len(positive_selector_gradients) if shared else None
                ),
                "mean_positive_selector_gradient": (
                    _mean(positive_selector_gradients)
                    if positive_selector_gradients
                    else None
                ),
                "final_head_advantage_ema": (
                    float(state["head_advantage_ema"])
                    if state is not None
                    else None
                ),
                "final_tail_advantage_ema": (
                    float(state["tail_advantage_ema"])
                    if state is not None
                    else None
                ),
                "selector_update_count": (
                    int(state["selector_update_count"])
                    if state is not None
                    else None
                ),
            }
        )
    return diagnostics


def _validate_cell(
    output_root: Path,
    repo_root: Path,
    model: str,
    mode: str,
    expected_noise: float,
    expected_random_prob: float,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    cell_name = f"{model}/{mode}"
    cell_dir = output_root / model / mode
    spec_path = cell_dir / "run_spec.json"
    result_path = cell_dir / "evaluation_results.json"
    _require(spec_path.is_file(), f"{cell_name}: missing {spec_path.name}")
    _require(result_path.is_file(), f"{cell_name}: missing {result_path.name}")
    spec = _load_json(spec_path)
    result = _load_json(result_path)

    _require(spec.get("protocol_version") == EXPECTED_PROTOCOL_VERSION, f"{cell_name}: protocol version")
    _require(spec.get("model_id") == model, f"{cell_name}: run_spec model_id mismatch")
    _require(
        spec.get("experiment_label") == EXPERIMENT_LABEL_BY_MODEL[model],
        f"{cell_name}: run_spec experiment_label mismatch",
    )
    _require(spec.get("mode") == mode, f"{cell_name}: run_spec mode mismatch")
    _require(spec.get("episodes") == EXPECTED_EPISODES_PER_CELL, f"{cell_name}: spec episodes")
    _require(spec.get("episode_length") == EXPECTED_EPISODE_LENGTH, f"{cell_name}: spec episode length")
    _assert_close(
        spec.get("episode_length_multiplier"),
        EXPECTED_EPISODE_LENGTH_MULTIPLIER,
        f"{cell_name}: spec episode length multiplier",
        abs_tol=1e-12,
    )
    _require(
        spec.get("eval_noise_seed") == EXPECTED_EVAL_NOISE_SEED,
        f"{cell_name}: noise seed",
    )
    _require(spec.get("eval_process_shards") == EXPECTED_PROCESS_SHARDS, f"{cell_name}: shard count")
    _require(spec.get("eval_process_workers") == EXPECTED_PROCESS_WORKERS, f"{cell_name}: worker count")
    _assert_close(spec.get("eval_noise_scale"), expected_noise, f"{cell_name}: spec noise", abs_tol=1e-12)
    _assert_close(
        spec.get("eval_random_action_prob"),
        expected_random_prob,
        f"{cell_name}: spec random probability",
        abs_tol=1e-12,
    )

    model_path = Path(str(spec.get("model_path", ""))).resolve()
    _require(model_path.is_dir(), f"{cell_name}: model path missing: {model_path}")
    expected_model_entry = next(
        item for item in BATCH_SPEC["models"] if str(item["id"]) == model
    )
    _require(
        model_path == Path(str(expected_model_entry["model_path"])).resolve(),
        f"{cell_name}: model path differs from batch spec",
    )
    _require(_actor_signature(model_path) == spec.get("model_signature"), f"{cell_name}: actor hash mismatch")
    _require(Path(str(result.get("model_path", ""))).resolve() == model_path, f"{cell_name}: result model path")

    for path_key, signature_key in (
        ("training_results_path", "training_results_signature"),
        ("training_runtime_manifest_path", "training_runtime_manifest_signature"),
    ):
        artifact_path = Path(str(spec.get(path_key, ""))).resolve()
        _require(artifact_path.is_file(), f"{cell_name}: missing {path_key}: {artifact_path}")
        _require(_file_sha256(artifact_path) == spec.get(signature_key), f"{cell_name}: {signature_key} mismatch")

    positions_file = Path(str(spec.get("positions_file", ""))).resolve()
    _require(positions_file.is_file(), f"{cell_name}: positions file missing")
    _require(_file_sha256(positions_file) == spec.get("positions_file_signature"), f"{cell_name}: positions hash")
    _require(
        positions_file == Path(str(BATCH_SPEC["positions_file"])).resolve(),
        f"{cell_name}: positions path differs from batch spec",
    )
    _require(
        spec.get("positions_file_signature")
        == BATCH_SPEC["positions_file_sha256"],
        f"{cell_name}: positions hash differs from batch spec",
    )
    _require(
        spec.get("sequence_source_signature")
        == BATCH_SPEC["sequence_source_sha256"],
        f"{cell_name}: sequence source hash differs from batch spec",
    )
    batch_spec_path = Path(str(spec.get("batch_spec_path", ""))).resolve()
    _require(batch_spec_path.is_file(), f"{cell_name}: batch spec path missing")
    _require(
        _file_sha256(batch_spec_path) == spec.get("batch_spec_signature"),
        f"{cell_name}: batch spec file hash",
    )
    _require(
        spec.get("batch_spec_content_sha256") == BATCH_SPEC["content_sha256"],
        f"{cell_name}: batch spec content identity",
    )

    source_signatures = spec.get("protocol_source_signatures")
    _require(isinstance(source_signatures, dict) and bool(source_signatures), f"{cell_name}: source signatures")
    for relative_path, recorded_sha256 in source_signatures.items():
        source_path = repo_root / relative_path
        _require(source_path.is_file(), f"{cell_name}: protocol source missing: {relative_path}")
        _require(_file_sha256(source_path) == recorded_sha256, f"{cell_name}: protocol source drift: {relative_path}")

    training_result = _load_json(Path(str(spec["training_results_path"])))
    _require(
        training_result.get("training_device")
        == expected_model_entry.get("training_device"),
        f"{cell_name}: training device evidence differs from batch spec",
    )
    _require(
        training_result.get("training_parallelism")
        == expected_model_entry.get("training_parallelism"),
        f"{cell_name}: training parallelism evidence differs from batch spec",
    )
    setup = _model_setup(training_result)
    _require(setup["train_seed"] == EXPECTED_TRAIN_SEED, f"{cell_name}: training seed")
    _require(setup["algorithm"] == "matd3", f"{cell_name}: training algorithm")
    _require(
        setup["training_gpu_required"] is True,
        f"{cell_name}: training GPU was not required",
    )
    _require(
        setup["training_physical_gpus"] >= 1,
        f"{cell_name}: no physical training GPU recorded",
    )
    _require(
        setup["training_logical_gpus"] >= 1,
        f"{cell_name}: no logical training GPU recorded",
    )
    _require(training_result.get("episodes") == EXPECTED_TRAIN_EPISODES, f"{cell_name}: training episodes")
    _require(
        setup["training_num_envs"] == int(BATCH_SPEC["train_num_envs"]),
        f"{cell_name}: training num_envs",
    )
    _require(
        setup["training_synchronous_iterations"]
        == int(BATCH_SPEC["train_episodes"]),
        f"{cell_name}: training synchronous iterations",
    )
    _require(
        setup["training_environment_trajectories"]
        == int(BATCH_SPEC["train_environment_trajectories"]),
        f"{cell_name}: training environment trajectories",
    )
    expected_protocol_env = SELECTOR_PROTOCOL_CONFIG_BY_LABEL[
        EXPERIMENT_LABEL_BY_MODEL[model]
    ]["env"]
    _require(
        setup["gate_mode"]
        == expected_protocol_env["CROSS_AGENT_REFERENCE_GATE_MODE"],
        f"{cell_name}: gate mode differs from selector protocol",
    )
    _require(
        setup["target_semantics"]
        == expected_protocol_env["CROSS_AGENT_REFERENCE_TARGET_SEMANTICS"],
        f"{cell_name}: teacher semantics differs from selector protocol",
    )
    _require(
        setup["selector_mode"]
        == expected_protocol_env["CROSS_AGENT_REFERENCE_SELECTOR_MODE"],
        f"{cell_name}: selector mode differs from selector protocol",
    )
    _require(
        setup["selector_enabled"]
        == (
            expected_protocol_env["CROSS_AGENT_REFERENCE_SELECTOR_ENABLED"]
            == "1"
        ),
        f"{cell_name}: selector enabled flag differs from selector protocol",
    )

    details = result.get("episode_details")
    _require(isinstance(details, list), f"{cell_name}: episode_details is not a list")
    _require(len(details) == EXPECTED_EPISODES_PER_CELL, f"{cell_name}: detail count={len(details)}")
    _require(result.get("episodes") == EXPECTED_EPISODES_PER_CELL, f"{cell_name}: result episodes")
    episode_ids = [item.get("episode") for item in details]
    _require(episode_ids == list(range(EXPECTED_EPISODES_PER_CELL)), f"{cell_name}: episode IDs")

    sequence_map = {
        "terrain_complexity_level_sequence": "terrain_complexity_level",
        "terrain_seed_sequence": "terrain_seed",
        "terrain_variant_seed_sequence": "terrain_variant_seed",
        "obstacle_seed_sequence": "obstacle_seed",
    }
    for sequence_key, detail_key in sequence_map.items():
        expected_sequence = spec.get(sequence_key)
        _require(
            isinstance(expected_sequence, list) and len(expected_sequence) == EXPECTED_EPISODES_PER_CELL,
            f"{cell_name}: invalid {sequence_key}",
        )
        _require(result.get(sequence_key) == expected_sequence, f"{cell_name}: result {sequence_key}")
        _require(
            [item.get(detail_key) for item in details] == expected_sequence,
            f"{cell_name}: detail {detail_key} sequence",
        )

    eval_setup = result.get("evaluation_setup")
    _require(isinstance(eval_setup, dict), f"{cell_name}: missing evaluation_setup")
    _assert_close(eval_setup.get("eval_noise_scale"), expected_noise, f"{cell_name}: result noise", abs_tol=1e-12)
    _assert_close(
        eval_setup.get("eval_random_action_prob"),
        expected_random_prob,
        f"{cell_name}: result random probability",
        abs_tol=1e-12,
    )
    _require(
        eval_setup.get("eval_noise_seed") == EXPECTED_EVAL_NOISE_SEED,
        f"{cell_name}: result noise seed",
    )
    _require(
        eval_setup.get("eval_noise_seed_base") == EXPECTED_EVAL_NOISE_SEED,
        f"{cell_name}: noise seed base",
    )
    _require(eval_setup.get("eval_episode_parallelism_mode") == "process_shards", f"{cell_name}: execution mode")
    _require(eval_setup.get("eval_process_shards") == EXPECTED_PROCESS_SHARDS, f"{cell_name}: result shards")
    _require(eval_setup.get("eval_process_workers") == EXPECTED_PROCESS_WORKERS, f"{cell_name}: result workers")
    _require(eval_setup.get("episode_length") == EXPECTED_EPISODE_LENGTH, f"{cell_name}: result episode length")
    _assert_close(
        eval_setup.get("requested_episode_length_multiplier"),
        EXPECTED_EPISODE_LENGTH_MULTIPLIER,
        f"{cell_name}: result episode length multiplier",
        abs_tol=1e-12,
    )
    _require(
        eval_setup.get("eval_noise_type") == "gaussian",
        f"{cell_name}: noise type",
    )
    _require(
        eval_setup.get("eval_noise_stream_mode")
        == "per_episode_seedsequence_pcg64_v1",
        f"{cell_name}: noise stream mode",
    )
    _require(
        eval_setup.get("terrain_family")
        == BATCH_SPEC["environment"]["post_eval_terrain_family"],
        f"{cell_name}: terrain family",
    )
    _require(
        eval_setup.get("position_family")
        == BATCH_SPEC["environment"]["post_eval_position_family"],
        f"{cell_name}: position family",
    )
    _require(
        eval_setup.get("semi_random_terrain")
        is BATCH_SPEC["environment"]["semi_random_terrain"],
        f"{cell_name}: semi-random terrain mode",
    )
    _require(
        eval_setup.get("use_dynamic_obstacles")
        is BATCH_SPEC["environment"]["use_dynamic_obstacles"],
        f"{cell_name}: dynamic-obstacle mode",
    )
    _require(
        eval_setup.get("use_fixed_positions") is True,
        f"{cell_name}: fixed positions disabled",
    )
    _require(
        Path(str(eval_setup.get("positions_file", ""))).resolve()
        == positions_file,
        f"{cell_name}: result positions path",
    )
    _require(
        Path(
            str(eval_setup.get("training_runtime_manifest_path", ""))
        ).resolve()
        == Path(str(spec["training_runtime_manifest_path"])).resolve(),
        f"{cell_name}: result training manifest path",
    )
    training_runtime_setup = spec.get("training_runtime_setup")
    training_reward_setup = spec.get("training_reward_setup")
    _require(
        isinstance(training_runtime_setup, dict),
        f"{cell_name}: invalid training runtime setup",
    )
    _require(
        isinstance(training_reward_setup, dict),
        f"{cell_name}: invalid training reward setup",
    )
    _assert_setup_mapping(
        eval_setup,
        training_runtime_setup,
        f"{cell_name}: runtime setup",
    )
    _assert_setup_mapping(
        eval_setup,
        training_reward_setup,
        f"{cell_name}: reward setup",
    )
    _require(eval_setup.get("eval_backend") == "python_only", f"{cell_name}: evaluation backend")
    device = eval_setup.get("eval_device")
    _require(isinstance(device, dict), f"{cell_name}: missing GPU device record")
    _require(device.get("require_gpu") is True, f"{cell_name}: GPU was not required")
    _require(int(device.get("physical_gpus", 0)) >= 1, f"{cell_name}: no physical GPU recorded")
    _require(int(device.get("logical_gpus", 0)) >= 1, f"{cell_name}: no logical GPU recorded")
    shard_specs = eval_setup.get("eval_process_shard_specs")
    _require(isinstance(shard_specs, list), f"{cell_name}: missing shard specs")
    actual_layout = tuple(
        (item.get("index"), item.get("start"), item.get("count")) for item in shard_specs
    )
    _require(actual_layout == EXPECTED_SHARD_LAYOUT, f"{cell_name}: shard layout={actual_layout!r}")
    _require(
        all(
            item.get("eval_noise_seed") == EXPECTED_EVAL_NOISE_SEED
            for item in shard_specs
        ),
        f"{cell_name}: shard noise seed",
    )

    rewards = [_finite_float(item.get("reward"), f"{cell_name}: reward") for item in details]
    collisions = [_finite_float(item.get("collision_count"), f"{cell_name}: collisions") for item in details]
    final_distances = [
        _finite_float(item.get("final_goal_distance"), f"{cell_name}: final distance") for item in details
    ]
    team_success = [int(item.get("team_success")) for item in details]
    _require(all(value in (0, 1) for value in team_success), f"{cell_name}: invalid team success flag")
    summary = result.get("summary")
    _require(isinstance(summary, dict), f"{cell_name}: missing summary")
    _assert_close(result.get("avg_reward"), _mean(rewards), f"{cell_name}: top avg_reward")
    _assert_close(summary.get("avg_reward"), _mean(rewards), f"{cell_name}: avg_reward")
    _assert_close(summary.get("team_success_rate"), _mean(team_success), f"{cell_name}: team SR")
    _assert_close(summary.get("avg_collision_count"), _mean(collisions), f"{cell_name}: avg collisions")
    _assert_close(
        summary.get("collision_free_rate"),
        _mean(1.0 if value == 0 else 0.0 for value in collisions),
        f"{cell_name}: collision-free rate",
    )
    _assert_close(
        summary.get("avg_team_final_goal_distance"),
        _mean(final_distances),
        f"{cell_name}: avg final distance",
    )

    return spec, result, training_result, setup


def analyze(output_root: Path, repo_root: Path) -> Dict[str, Any]:
    batch_spec = _configure_protocol(output_root)
    selector_training_diagnostics = _selector_training_diagnostics(batch_spec)
    cells: Dict[Tuple[str, str], Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    model_setups: Dict[str, Dict[str, Any]] = {}
    cell_rows: List[Dict[str, Any]] = []

    for model in MODEL_LABELS:
        for mode, (noise, random_prob) in MODE_CONFIG.items():
            spec, result, _training_result, setup = _validate_cell(
                output_root, repo_root, model, mode, noise, random_prob
            )
            cells[(model, mode)] = (spec, result)
            previous_setup = model_setups.setdefault(model, setup)
            _require(previous_setup == setup, f"{model}: model setup changed across modes")
            summary = result["summary"]
            agent_rates = summary.get("agent_success_rates")
            _require(isinstance(agent_rates, list) and len(agent_rates) == 3, f"{model}/{mode}: agent rates")
            agent_reach_rates = summary.get("agent_reach_rates")
            agent_safe_rates = summary.get("agent_safe_rates")
            _require(
                isinstance(agent_reach_rates, list) and len(agent_reach_rates) == 3,
                f"{model}/{mode}: agent reach rates",
            )
            _require(
                isinstance(agent_safe_rates, list) and len(agent_safe_rates) == 3,
                f"{model}/{mode}: agent safe rates",
            )
            done_reason_counts = summary.get("done_reason_counts")
            _require(isinstance(done_reason_counts, dict), f"{model}/{mode}: done reason counts")
            cell_rows.append(
                {
                    "model": model,
                    "mode": mode,
                    "episodes": result["episodes"],
                    "team_success_count": summary["success_episode_count"],
                    "team_success_rate": summary["team_success_rate"],
                    "agent_success_rate_mean": _mean(agent_rates),
                    "agent_0_success_rate": agent_rates[0],
                    "agent_1_success_rate": agent_rates[1],
                    "agent_2_success_rate": agent_rates[2],
                    "agent_reach_rate_mean": _mean(agent_reach_rates),
                    "agent_safe_rate_mean": _mean(agent_safe_rates),
                    "all_reached_without_safe_team_success_rate": summary[
                        "all_reached_without_safe_team_success_rate"
                    ],
                    "unsafe_reached_agent_slot_rate": summary["unsafe_reached_agent_slot_rate"],
                    "time_limit_rate": float(done_reason_counts.get("time_limit", 0))
                    / EXPECTED_EPISODES_PER_CELL,
                    "collision_free_rate": summary["collision_free_rate"],
                    "avg_collision_count": summary["avg_collision_count"],
                    "avg_terrain_collision_count": summary["avg_terrain_collision_count"],
                    "avg_obstacle_collision_count": summary["avg_obstacle_collision_count"],
                    "avg_reward": summary["avg_reward"],
                    "avg_team_final_goal_distance": summary["avg_team_final_goal_distance"],
                    "avg_team_min_goal_distance": summary["avg_team_min_goal_distance"],
                    "avg_steps": summary["avg_steps"],
                }
            )

    first_spec = cells[(MODEL_LABELS[0], next(iter(MODE_CONFIG)))][0]
    common_keys = (
        "positions_file",
        "positions_file_signature",
        "sequence_source_signature",
        "batch_spec_signature",
        "batch_spec_content_sha256",
        "protocol_source_signatures",
        "terrain_complexity_level_sequence",
        "terrain_seed_sequence",
        "terrain_variant_seed_sequence",
        "obstacle_seed_sequence",
        "episodes",
        "episode_length",
        "episode_length_multiplier",
        "eval_noise_seed",
        "eval_process_shards",
        "eval_process_workers",
    )
    for model in MODEL_LABELS:
        model_reference = cells[(model, "deterministic")][0]
        for mode in MODE_CONFIG:
            spec = cells[(model, mode)][0]
            for key in common_keys:
                _require(spec.get(key) == first_spec.get(key), f"{model}/{mode}: common protocol drift: {key}")
            for key in (
                "model_path",
                "model_signature",
                "training_results_path",
                "training_results_signature",
                "training_runtime_manifest_path",
                "training_runtime_manifest_signature",
                "training_reward_setup",
                "training_runtime_setup",
            ):
                _require(spec.get(key) == model_reference.get(key), f"{model}/{mode}: model identity drift: {key}")

    _require(
        len({cells[(model, "deterministic")][0]["model_path"] for model in MODEL_LABELS}) == len(MODEL_LABELS),
        "the four labels do not resolve to four distinct model paths",
    )
    _require(
        len({cells[(model, "deterministic")][0]["model_signature"] for model in MODEL_LABELS})
        == len(MODEL_LABELS),
        "the four labels do not resolve to four distinct actor signatures",
    )

    paired_rows: List[Dict[str, Any]] = []
    for model in MODEL_LABELS:
        baseline_details = cells[(model, "deterministic")][1]["episode_details"]
        for mode in tuple(MODE_CONFIG.keys())[1:]:
            mode_details = cells[(model, mode)][1]["episode_details"]
            baseline_success = [int(item["team_success"]) for item in baseline_details]
            perturbed_success = [int(item["team_success"]) for item in mode_details]
            success_gains = sum(base == 0 and perturbed == 1 for base, perturbed in zip(baseline_success, perturbed_success))
            success_losses = sum(base == 1 and perturbed == 0 for base, perturbed in zip(baseline_success, perturbed_success))

            collision_deltas = [
                float(perturbed["collision_count"]) - float(base["collision_count"])
                for base, perturbed in zip(baseline_details, mode_details)
            ]
            reward_deltas = [
                float(perturbed["reward"]) - float(base["reward"])
                for base, perturbed in zip(baseline_details, mode_details)
            ]
            final_distance_deltas = [
                float(perturbed["final_goal_distance"]) - float(base["final_goal_distance"])
                for base, perturbed in zip(baseline_details, mode_details)
            ]
            collision_free_deltas = [
                float(perturbed["collision_count"] == 0) - float(base["collision_count"] == 0)
                for base, perturbed in zip(baseline_details, mode_details)
            ]

            collision_ci = _paired_bootstrap_ci(collision_deltas, f"{model}/{mode}/collision")
            reward_ci = _paired_bootstrap_ci(reward_deltas, f"{model}/{mode}/reward")
            distance_ci = _paired_bootstrap_ci(final_distance_deltas, f"{model}/{mode}/distance")
            paired_rows.append(
                {
                    "model": model,
                    "mode": mode,
                    "paired_episodes": EXPECTED_EPISODES_PER_CELL,
                    "baseline_success_count": sum(baseline_success),
                    "mode_success_count": sum(perturbed_success),
                    "success_gains": success_gains,
                    "success_losses": success_losses,
                    "success_mcnemar_exact_p": _exact_mcnemar_p(success_gains, success_losses),
                    "delta_team_success_rate": _mean(
                        perturbed - base for base, perturbed in zip(baseline_success, perturbed_success)
                    ),
                    "delta_collision_free_rate": _mean(collision_free_deltas),
                    "mean_collision_delta": _mean(collision_deltas),
                    "collision_delta_bootstrap_ci_low": collision_ci[0],
                    "collision_delta_bootstrap_ci_high": collision_ci[1],
                    "mean_reward_delta": _mean(reward_deltas),
                    "reward_delta_bootstrap_ci_low": reward_ci[0],
                    "reward_delta_bootstrap_ci_high": reward_ci[1],
                    "mean_final_goal_distance_delta": _mean(final_distance_deltas),
                    "final_goal_distance_delta_bootstrap_ci_low": distance_ci[0],
                    "final_goal_distance_delta_bootstrap_ci_high": distance_ci[1],
                }
            )

    robustness_rows: List[Dict[str, Any]] = []
    for model in MODEL_LABELS:
        rows = [row for row in cell_rows if row["model"] == model]
        perturbed = [row for row in rows if row["mode"] != "deterministic"]
        deterministic = next(row for row in rows if row["mode"] == "deterministic")
        robustness_rows.append(
            {
                "model": model,
                "mean_team_success_rate_all_modes": _mean(row["team_success_rate"] for row in rows),
                "worst_team_success_rate": min(row["team_success_rate"] for row in rows),
                "mean_perturbed_team_success_rate": _mean(row["team_success_rate"] for row in perturbed),
                "mean_perturbed_sr_delta_vs_deterministic": _mean(
                    row["team_success_rate"] - deterministic["team_success_rate"] for row in perturbed
                ),
                "mean_collision_count_all_modes": _mean(row["avg_collision_count"] for row in rows),
                "mean_reward_all_modes": _mean(row["avg_reward"] for row in rows),
                "mean_final_goal_distance_all_modes": _mean(
                    row["avg_team_final_goal_distance"] for row in rows
                ),
            }
        )

    validation = {
        "status": "PASS",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "expected_cells": len(MODEL_LABELS) * len(MODE_CONFIG),
        "validated_cells": len(cells),
        "episodes_per_cell": EXPECTED_EPISODES_PER_CELL,
        "expected_episode_details": len(MODEL_LABELS) * len(MODE_CONFIG) * EXPECTED_EPISODES_PER_CELL,
        "validated_episode_details": sum(
            len(result["episode_details"]) for _spec, result in cells.values()
        ),
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "batch_spec_content_sha256": batch_spec["content_sha256"],
        "train_seed": EXPECTED_TRAIN_SEED,
        "train_episodes_per_model": EXPECTED_TRAIN_EPISODES,
        "eval_noise_seed": EXPECTED_EVAL_NOISE_SEED,
        "episode_length": EXPECTED_EPISODE_LENGTH,
        "training_episode_length": EXPECTED_TRAINING_EPISODE_LENGTH,
        "episode_length_multiplier": (
            EXPECTED_EPISODE_LENGTH_MULTIPLIER
        ),
        "process_shards": EXPECTED_PROCESS_SHARDS,
        "process_workers": EXPECTED_PROCESS_WORKERS,
        "gpu_required_and_recorded_for_all_cells": True,
        "training_gpu_required_and_recorded_for_all_models": True,
        "common_positions_file": first_spec["positions_file"],
        "common_positions_file_signature": first_spec["positions_file_signature"],
        "common_sequence_source_signature": first_spec["sequence_source_signature"],
        "protocol_source_signatures": first_spec["protocol_source_signatures"],
        "model_actor_signatures": {
            model: cells[(model, "deterministic")][0]["model_signature"] for model in MODEL_LABELS
        },
        "checks": [
            f"all {len(MODEL_LABELS) * len(MODE_CONFIG)} model/mode cells present",
            (
                f"all {len(MODEL_LABELS) * len(MODE_CONFIG) * EXPECTED_EPISODES_PER_CELL} "
                "episode IDs complete and ordered"
            ),
            (
                "actor, training-result, manifest, frozen positions/source, "
                "and protocol-source hashes match"
            ),
            "terrain, variant, obstacle, and positions sequences match across all cells",
            (
                "noise type, stream, parameters, and per-shard seeds match "
                "each mode"
            ),
            (
                "GPU execution and "
                f"{EXPECTED_PROCESS_SHARDS}-shard layout "
                f"{EXPECTED_SHARD_LAYOUT!r} recorded for every cell"
            ),
            "physical and logical GPU use recorded for every training model",
            "headline metrics recomputed from episode_details",
        ],
    }
    return {
        "validation": validation,
        "model_setups": model_setups,
        "selector_training_diagnostics": selector_training_diagnostics,
        "cell_metrics": cell_rows,
        "paired_deltas_vs_deterministic": paired_rows,
        "model_robustness_summary": robustness_rows,
    }


def _format_percent(value: Any) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_optional(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _markdown_report(analysis: Mapping[str, Any]) -> str:
    validation = analysis["validation"]
    lines = [
        "# Formal 4-model x 4-mode x 30-episode GPU batch",
        "",
        f"Validation: **{validation['status']}**; "
        f"{validation['validated_cells']}/{validation['expected_cells']} cells; "
        f"{validation['validated_episode_details']}/"
        f"{validation['expected_episode_details']} episode records.",
        "",
        f"Protocol: train seed {validation['train_seed']}; evaluation noise seed "
        f"{validation['eval_noise_seed']}; {validation['episodes_per_cell']} paired "
        f"scenarios per cell; episode length {validation['episode_length']}; "
        f"{validation['process_shards']} process shards; GPU required.",
        "",
        "## Model definitions",
        "",
        "| Model | Experiment label | Teacher semantics | Gate | Selector | Selector mode | Init logit | EMA decay |",
        "|---|---|---|---|---:|---|---:|---:|",
    ]
    for model in MODEL_LABELS:
        setup = analysis["model_setups"][model]
        lines.append(
            f"| {model} | {EXPERIMENT_LABEL_BY_MODEL[model]} | "
            f"{setup['target_semantics']} | {setup['gate_mode']} | "
            f"{str(setup['selector_enabled']).lower()} | "
            f"{setup['selector_mode']} | {setup['selector_init_logit']:.2f} | "
            f"{setup['advantage_ema_decay']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Training-time selector diagnostics",
            "",
            "| Model | Active/eligible log rows | Mean valid ratio | Head/tail agreement | Head/tail multiplier | Positive selector-gradient rows | Final head/tail EMA | Updates |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["selector_training_diagnostics"]:
        lines.append(
            f"| {row['model']} | {row['active_reference_rows']}/"
            f"{row['eligible_reference_rows']} | "
            f"{_format_optional(row['mean_valid_ratio'])} | "
            f"{_format_optional(row['mean_head_twin_agreement'])}/"
            f"{_format_optional(row['mean_tail_twin_agreement'])} | "
            f"{_format_optional(row['mean_head_multiplier'])}/"
            f"{_format_optional(row['mean_tail_multiplier'])} | "
            f"{row['positive_selector_gradient_rows'] if row['positive_selector_gradient_rows'] is not None else 'n/a'} | "
            f"{_format_optional(row['final_head_advantage_ema'])}/"
            f"{_format_optional(row['final_tail_advantage_ema'])} | "
            f"{row['selector_update_count'] if row['selector_update_count'] is not None else 'n/a'} |"
        )
    lines.extend(
        [
            "",
            "## Cell metrics",
            "",
            "| Model | Mode | Team success | Collision-free | Avg collisions | Avg reward | Final goal distance |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["cell_metrics"]:
        lines.append(
            f"| {row['model']} | {row['mode']} | {_format_percent(row['team_success_rate'])} "
            f"({row['team_success_count']}/{validation['episodes_per_cell']}) | "
            f"{_format_percent(row['collision_free_rate'])} | "
            f"{row['avg_collision_count']:.2f} | {row['avg_reward']:.2f} | "
            f"{row['avg_team_final_goal_distance']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Failure diagnostics",
            "",
            "| Model | Mode | Mean agent success | Mean reach | Mean safe | All reached but unsafe | Time limit |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["cell_metrics"]:
        lines.append(
            f"| {row['model']} | {row['mode']} | {_format_percent(row['agent_success_rate_mean'])} | "
            f"{_format_percent(row['agent_reach_rate_mean'])} | {_format_percent(row['agent_safe_rate_mean'])} | "
            f"{_format_percent(row['all_reached_without_safe_team_success_rate'])} | "
            f"{_format_percent(row['time_limit_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Paired deltas versus deterministic",
            "",
            "Positive collision/distance deltas are worse; positive reward/success deltas are better. "
            "CIs are paired episode bootstrap percentile intervals (20,000 resamples).",
            "",
            "| Model | Mode | SR delta | Gains/losses | Exact p | Collision delta [95% CI] | Reward delta [95% CI] | Distance delta [95% CI] |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["paired_deltas_vs_deterministic"]:
        lines.append(
            f"| {row['model']} | {row['mode']} | {_format_percent(row['delta_team_success_rate'])} | "
            f"{row['success_gains']}/{row['success_losses']} | {row['success_mcnemar_exact_p']:.4f} | "
            f"{row['mean_collision_delta']:.2f} "
            f"[{row['collision_delta_bootstrap_ci_low']:.2f}, {row['collision_delta_bootstrap_ci_high']:.2f}] | "
            f"{row['mean_reward_delta']:.2f} "
            f"[{row['reward_delta_bootstrap_ci_low']:.2f}, {row['reward_delta_bootstrap_ci_high']:.2f}] | "
            f"{row['mean_final_goal_distance_delta']:.2f} "
            f"[{row['final_goal_distance_delta_bootstrap_ci_low']:.2f}, "
            f"{row['final_goal_distance_delta_bootstrap_ci_high']:.2f}] |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    repo_root = args.repo_root.resolve()
    _require(output_root.is_dir(), f"output root does not exist: {output_root}")

    analysis = analyze(output_root, repo_root)
    validation_path = output_root / "formal_batch_validation.json"
    metrics_path = output_root / "formal_batch_metrics.csv"
    paired_path = output_root / "formal_batch_paired_deltas.csv"
    training_diagnostics_path = (
        output_root / "formal_batch_training_diagnostics.csv"
    )
    analysis_path = output_root / "formal_batch_analysis.json"
    report_path = output_root / "formal_batch_report.md"
    validation_path.write_text(
        json.dumps(analysis["validation"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(metrics_path, analysis["cell_metrics"])
    _write_csv(paired_path, analysis["paired_deltas_vs_deterministic"])
    _write_csv(
        training_diagnostics_path,
        analysis["selector_training_diagnostics"],
    )
    analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_report(analysis), encoding="utf-8")

    print(
        f"PASS cells={analysis['validation']['validated_cells']} "
        f"episodes={analysis['validation']['validated_episode_details']}"
    )
    for path in (
        validation_path,
        metrics_path,
        paired_path,
        training_diagnostics_path,
        analysis_path,
        report_path,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
