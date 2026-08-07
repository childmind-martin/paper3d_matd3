#!/usr/bin/env python3
"""Build and validate the frozen M0-M3 × four-noise-mode evaluation spec."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_DEFAULT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEFAULT_REPO_ROOT))

from experiment_runtime_config import training_unit_completion_errors
from cross_agent_reference_state import (
    ADVANTAGE_SELECTOR_MODES,
    MODE_SHARED_TWIN_HEAD_TAIL,
    selector_state_errors,
)
from selector_experiment_protocol import (
    SELECTOR_PROTOCOL_CONFIG_BY_LABEL,
    SELECTOR_PROTOCOL_EXPERIMENT_LABELS,
    SELECTOR_PROTOCOL_ID_BY_LABEL,
    SELECTOR_PROTOCOL_SCHEMA_VERSION,
)


BATCH_SPEC_SCHEMA_VERSION = 4
BATCH_PROTOCOL_VERSION = 10
FORMAL_EVAL_EPISODES = 30
PRELIMINARY_POST_EVAL_MODE = "shared_match_train_env"
PRELIMINARY_POST_EVAL_SEED = 10088
PRELIMINARY_POST_EVAL_SELECTION_PROTOCOL = "fixed"
PRELIMINARY_POST_EVAL_MODEL_VARIANT = "final"
PRELIMINARY_POST_EVAL_EPISODE_LENGTH_MULTIPLIER = 1.1
FORMAL_MODES = (
    {
        "id": "deterministic",
        "eval_noise_scale": 0.0,
        "eval_random_action_prob": 0.0,
    },
    {
        "id": "gaussian_noise_0p11",
        "eval_noise_scale": 0.11,
        "eval_random_action_prob": 0.0,
    },
    {
        "id": "random_1pct",
        "eval_noise_scale": 0.0,
        "eval_random_action_prob": 0.01,
    },
    {
        "id": "gaussian_0p11_random_1pct",
        "eval_noise_scale": 0.11,
        "eval_random_action_prob": 0.01,
    },
)
TRAINING_PARALLELISM_CONTRACT = {
    "reward_aggregation": "equal_mean_across_environments",
    "success_aggregation": "equal_mean_across_environments",
    "worker_seed_derivation": "base_seed_plus_env_id_times_100003",
    "episode_audit_snapshot_schema_version": 1,
}

_PROTOCOL_RESULT_BOOL_ENV_KEYS = frozenset(
    {
        "MATD3_USE_DUAL_Q",
        "MATD3_USE_SEPARATED_GRADIENT",
        "MATD3_USE_HYBRID_ACTOR_OBJECTIVE",
        "MATD3_RECONSTRUCT_CORRECTED_TARGET",
        "USE_TF_POTENTIAL_FIELD",
        "USE_FR_FEATURE",
        "USE_PF_FEATURE",
        "CROSS_AGENT_REFERENCE_ENABLED",
        "CROSS_AGENT_REFERENCE_ACTOR_REQUIRE_SUCCESS",
        "CROSS_AGENT_REFERENCE_USE_CLEAN_LABEL",
        "CROSS_AGENT_REFERENCE_EXCLUDE_RANDOM",
        "CROSS_AGENT_REFERENCE_QUALITY_GATE",
        "CROSS_AGENT_REFERENCE_SELECTOR_TRAIN_IN_GRAPH",
        "CROSS_AGENT_REFERENCE_SELECTOR_ENABLED",
    }
)
_PROTOCOL_RESULT_INT_ENV_KEYS = frozenset(
    {
        "CROSS_AGENT_REFERENCE_START_EPISODE",
        "CROSS_AGENT_REFERENCE_ACTOR_START_EPISODE",
        "CROSS_AGENT_REFERENCE_ACTOR_RAMP_EPISODES",
        "CROSS_AGENT_REFERENCE_UPDATE_INTERVAL",
        "CROSS_AGENT_REFERENCE_PAIRS_PER_AGENT",
    }
)
_PROTOCOL_RESULT_STRING_ENV_KEYS = frozenset(
    {
        "ALGORITHM",
        "MATD3_ACTION_SEMANTICS_MODE",
        "CROSS_AGENT_REFERENCE_TARGET_SEMANTICS",
        "CROSS_AGENT_REFERENCE_GATE_MODE",
        "CROSS_AGENT_REFERENCE_SELECTOR_MODE",
        "CROSS_AGENT_REFERENCE_SELECTOR_HIDDEN",
    }
)
_PROTOCOL_MANIFEST_ONLY_ENV_KEYS = frozenset(
    {
        "MATD3_REQUIRE_GPU",
        "SELECTOR_PROTOCOL_LOCK",
    }
)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    content = dict(payload)
    content.pop("content_sha256", None)
    encoded = json.dumps(
        content,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _actor_signature(model_dir: Path) -> str:
    actors = sorted(model_dir.glob("actor_*.weights.h5"))
    if not actors:
        raise FileNotFoundError(f"model has no actor weights: {model_dir}")
    digest = hashlib.sha1()
    for actor in actors:
        digest.update(actor.name.encode("utf-8"))
        with actor.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _valid_sha1(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 40 and all(character in "0123456789abcdef" for character in text)


def _strict_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _cross_reference_activity_rows(
    loss_history: list[Any],
    *,
    start_episode: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Return rows that prove an active and an eligible cross-reference update.

    Current trainers persist ``cross_ref_active`` explicitly.  Pilot artifacts
    produced before that field was added still contain the graph outputs
    ``cross_ref_valid_ratio``, ``cross_ref_loss`` and
    ``cross_ref_actor_weight``.  A positive eligible ratio after the configured
    start episode, together with a positive actor weight, can only be emitted by
    an active cross-reference actor update.  Explicit ``cross_ref_active=0``
    always wins and is never inferred away.
    """
    active_rows: list[Mapping[str, Any]] = []
    positive_valid_rows: list[Mapping[str, Any]] = []
    for row in loss_history:
        if not isinstance(row, Mapping):
            continue
        try:
            valid_ratio = float(row.get("cross_ref_valid_ratio"))
            reference_loss = float(row.get("cross_ref_loss"))
        except (TypeError, ValueError):
            continue
        if not (
            math.isfinite(valid_ratio)
            and math.isfinite(reference_loss)
        ):
            continue

        raw_active = row.get("cross_ref_active")
        if raw_active is not None:
            try:
                active_value = float(raw_active)
            except (TypeError, ValueError):
                continue
            active = math.isfinite(active_value) and active_value > 0.5
        else:
            try:
                episode = int(row.get("episode"))
                actor_weight = float(row.get("cross_ref_actor_weight"))
            except (TypeError, ValueError):
                continue
            active = (
                episode >= int(start_episode)
                and valid_ratio > 0.0
                and math.isfinite(actor_weight)
                and actor_weight > 0.0
            )

        if active:
            active_rows.append(row)
            if valid_ratio > 0.0:
                positive_valid_rows.append(row)
    return active_rows, positive_valid_rows


def _selector_result_arg_errors(
    run_args: Mapping[str, Any],
    expected_env: Mapping[str, Any],
) -> list[str]:
    """Compare the effective trainer arguments with the frozen selector setup.

    A resolved launcher manifest proves what was requested.  ``results.args``
    proves what the trainer parser actually received.  The formal batch needs
    both identities so a wrapper propagation bug cannot silently turn M0-M3
    into a different experiment.
    """

    errors: list[str] = []
    for env_key, expected_raw in expected_env.items():
        if env_key in _PROTOCOL_MANIFEST_ONLY_ENV_KEYS:
            continue
        arg_key = "algo" if env_key == "ALGORITHM" else env_key.lower()
        actual = run_args.get(arg_key)
        if env_key in _PROTOCOL_RESULT_BOOL_ENV_KEYS:
            expected = _strict_bool(expected_raw)
            actual_value = _strict_bool(actual)
            matches = expected is not None and actual_value is expected
        elif env_key in _PROTOCOL_RESULT_INT_ENV_KEYS:
            try:
                expected = int(expected_raw)
                actual_value = int(actual)
                matches = (
                    not isinstance(actual, bool)
                    and float(actual) == float(actual_value)
                    and actual_value == expected
                )
            except (TypeError, ValueError, OverflowError):
                matches = False
        elif env_key in _PROTOCOL_RESULT_STRING_ENV_KEYS:
            expected = str(expected_raw).strip().lower()
            actual_value = str(actual or "").strip().lower()
            matches = actual_value == expected
        else:
            try:
                expected = float(expected_raw)
                actual_value = float(actual)
                matches = (
                    math.isfinite(expected)
                    and math.isfinite(actual_value)
                    and math.isclose(
                        actual_value,
                        expected,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                )
            except (TypeError, ValueError, OverflowError):
                matches = False
        if not matches:
            errors.append(
                f"results.args.{arg_key}={actual!r}, "
                f"expected from {env_key}={expected_raw!r}"
            )
    return errors


def _training_parallelism_errors(
    payload: Any,
    *,
    expected_num_envs: int,
    expected_iterations: int,
) -> list[str]:
    if not isinstance(payload, dict):
        return ["training_parallelism is missing or invalid"]
    errors: list[str] = []
    expected_values = {
        "num_envs": int(expected_num_envs),
        "synchronous_iterations": int(expected_iterations),
        "environment_trajectories": (
            int(expected_num_envs) * int(expected_iterations)
        ),
        **TRAINING_PARALLELISM_CONTRACT,
    }
    for key, expected in expected_values.items():
        actual = payload.get(key)
        if isinstance(expected, int):
            try:
                matches = (
                    not isinstance(actual, bool)
                    and int(actual) == expected
                    and float(actual) == float(int(actual))
                )
            except (TypeError, ValueError, OverflowError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            errors.append(
                f"training_parallelism.{key}={actual!r}, expected={expected!r}"
            )
    return errors


def validate_batch_spec(payload: Any, *, require_paths: bool = True) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["batch spec root is not an object"]
    if payload.get("schema_version") != BATCH_SPEC_SCHEMA_VERSION:
        errors.append(
            f"schema_version={payload.get('schema_version')!r}, "
            f"expected={BATCH_SPEC_SCHEMA_VERSION}"
        )
    if payload.get("protocol_version") != BATCH_PROTOCOL_VERSION:
        errors.append(
            f"protocol_version={payload.get('protocol_version')!r}, "
            f"expected={BATCH_PROTOCOL_VERSION}"
        )
    if payload.get("selector_protocol_schema_version") != (
        SELECTOR_PROTOCOL_SCHEMA_VERSION
    ):
        errors.append("selector protocol schema version mismatch")
    if payload.get("content_sha256") != _canonical_sha256(payload):
        errors.append("content_sha256 mismatch")
    if int(payload.get("episodes", 0) or 0) != FORMAL_EVAL_EPISODES:
        errors.append(
            f"episodes={payload.get('episodes')!r}, "
            f"expected={FORMAL_EVAL_EPISODES}"
        )
    try:
        training_episode_length = int(payload.get("training_episode_length"))
    except (TypeError, ValueError):
        training_episode_length = 0
    if training_episode_length <= 0:
        errors.append("training_episode_length must be positive")
    try:
        episode_length = int(payload.get("episode_length"))
    except (TypeError, ValueError):
        episode_length = 0
    if episode_length <= 0:
        errors.append("episode_length must be positive")
    try:
        episode_length_multiplier = float(
            payload.get("episode_length_multiplier")
        )
    except (TypeError, ValueError):
        episode_length_multiplier = float("nan")
    if (
        not math.isfinite(episode_length_multiplier)
        or abs(episode_length_multiplier - 1.1) > 1e-12
    ):
        errors.append("episode_length_multiplier must be exactly 1.1")
    if training_episode_length > 0 and math.isfinite(episode_length_multiplier):
        expected_episode_length = int(
            training_episode_length * episode_length_multiplier + 0.5
        )
        if episode_length != expected_episode_length:
            errors.append(
                f"episode_length={episode_length!r}, "
                f"expected={expected_episode_length!r} from "
                "training_episode_length and episode_length_multiplier"
            )
    for key in (
        "train_seed",
        "train_episodes",
        "train_num_envs",
        "train_environment_trajectories",
        "eval_noise_seed",
    ):
        try:
            value = int(payload.get(key))
        except (TypeError, ValueError):
            errors.append(f"{key} is not an integer")
            continue
        if key in (
            "train_episodes",
            "train_num_envs",
            "train_environment_trajectories",
        ) and value <= 0:
            errors.append(f"{key} must be positive")
    try:
        expected_train_trajectories = (
            int(payload.get("train_episodes"))
            * int(payload.get("train_num_envs"))
        )
        if int(payload.get("train_environment_trajectories")) != (
            expected_train_trajectories
        ):
            errors.append(
                "train_environment_trajectories must equal "
                "train_episodes * train_num_envs"
            )
    except (TypeError, ValueError):
        pass
    try:
        spec_train_episodes = int(payload.get("train_episodes"))
    except (TypeError, ValueError):
        spec_train_episodes = 0
    try:
        spec_train_num_envs = int(payload.get("train_num_envs"))
    except (TypeError, ValueError):
        spec_train_num_envs = 0
    if payload.get("require_gpu") is not True:
        errors.append("formal protocol requires require_gpu=true")
    for key in (
        "eval_process_shards",
        "eval_process_workers",
        "eval_shard_episode_parallelism",
        "eval_shard_env_step_threads",
    ):
        if int(payload.get(key, 0) or 0) <= 0:
            errors.append(f"{key} must be positive")
    environment = payload.get("environment")
    if not isinstance(environment, dict):
        errors.append("environment is missing or invalid")
    else:
        if environment.get("semi_random_terrain") is not True:
            errors.append("formal protocol requires semi_random_terrain=true")
        if environment.get("use_dynamic_obstacles") is not True:
            errors.append("formal protocol requires use_dynamic_obstacles=true")
        for key in ("scenario_seed", "terrain_base_seed"):
            try:
                int(environment.get(key))
            except (TypeError, ValueError):
                errors.append(f"environment.{key} is not an integer")
        if environment.get("post_eval_mode") != "shared_match_train_env":
            errors.append(
                "environment.post_eval_mode must be shared_match_train_env"
            )
        if environment.get("post_eval_terrain_family") != "train_match":
            errors.append(
                "environment.post_eval_terrain_family must be train_match"
            )
        if environment.get("post_eval_position_family") != "train_match":
            errors.append(
                "environment.post_eval_position_family must be train_match"
            )
        if environment.get("position_protocol") != (
            "single_fixed_positions_file"
        ):
            errors.append(
                "environment.position_protocol must be "
                "single_fixed_positions_file"
            )

    models = payload.get("models")
    expected_labels = list(SELECTOR_PROTOCOL_EXPERIMENT_LABELS)
    actual_labels = (
        [str(item.get("label")) for item in models]
        if isinstance(models, list)
        and all(isinstance(item, dict) for item in models)
        else []
    )
    if actual_labels != expected_labels:
        errors.append(
            f"model labels={actual_labels!r}, expected={expected_labels!r}"
        )
    if isinstance(models, list):
        for item in models:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", ""))
            expected_id = SELECTOR_PROTOCOL_ID_BY_LABEL.get(label)
            if item.get("id") != expected_id:
                errors.append(
                    f"model {label} id={item.get('id')!r}, "
                    f"expected={expected_id!r}"
                )
            if item.get("model_variant") != "final":
                errors.append(
                    f"model {label} must use the fixed-horizon final variant"
                )
            preliminary_post_eval = item.get("preliminary_post_eval")
            if not isinstance(preliminary_post_eval, dict):
                errors.append(
                    f"model {label} preliminary_post_eval is missing"
                )
            else:
                expected_preliminary_values = {
                    "status": "completed",
                    "mode": PRELIMINARY_POST_EVAL_MODE,
                    "episodes": FORMAL_EVAL_EPISODES,
                    "episode_length_multiplier": (
                        PRELIMINARY_POST_EVAL_EPISODE_LENGTH_MULTIPLIER
                    ),
                    "seed": PRELIMINARY_POST_EVAL_SEED,
                    "selection_protocol": (
                        PRELIMINARY_POST_EVAL_SELECTION_PROTOCOL
                    ),
                    "requested_model_variant": (
                        PRELIMINARY_POST_EVAL_MODEL_VARIANT
                    ),
                    "resolved_model_variant": (
                        PRELIMINARY_POST_EVAL_MODEL_VARIANT
                    ),
                }
                for key, expected_value in expected_preliminary_values.items():
                    if preliminary_post_eval.get(key) != expected_value:
                        errors.append(
                            f"model {label} preliminary_post_eval.{key}="
                            f"{preliminary_post_eval.get(key)!r}, "
                            f"expected={expected_value!r}"
                        )
                if preliminary_post_eval.get("gpu_required") is not True:
                    errors.append(
                        f"model {label} preliminary post-eval did not "
                        "require GPU"
                    )
                try:
                    preliminary_physical_gpus = int(
                        preliminary_post_eval.get("physical_gpus", 0)
                    )
                    preliminary_logical_gpus = int(
                        preliminary_post_eval.get("logical_gpus", 0)
                    )
                except (TypeError, ValueError):
                    preliminary_physical_gpus = 0
                    preliminary_logical_gpus = 0
                if (
                    preliminary_physical_gpus < 1
                    or preliminary_logical_gpus < 1
                ):
                    errors.append(
                        f"model {label} preliminary post-eval has no "
                        "recorded GPU"
                    )
                if not _valid_sha1(
                    preliminary_post_eval.get(
                        "selected_model_signature_sha1"
                    )
                ):
                    errors.append(
                        f"model {label} preliminary selected model "
                        "signature is invalid"
                    )
                elif preliminary_post_eval.get(
                    "selected_model_signature_sha1"
                ) != item.get("actor_signature_sha1"):
                    errors.append(
                        f"model {label} preliminary selected model "
                        "signature differs from final actor signature"
                    )
                for signature_key in (
                    "results_sha256",
                    "spec_sha256",
                ):
                    if not _valid_sha256(
                        preliminary_post_eval.get(signature_key)
                    ):
                        errors.append(
                            f"model {label} preliminary_post_eval."
                            f"{signature_key} is not a SHA-256 digest"
                        )
            training_device = item.get("training_device")
            if not isinstance(training_device, dict):
                errors.append(f"model {label} training_device is missing")
            else:
                if training_device.get("require_gpu") is not True:
                    errors.append(
                        f"model {label} was not trained with GPU required"
                    )
                try:
                    physical_gpus = int(
                        training_device.get("physical_gpus", 0)
                    )
                    logical_gpus = int(
                        training_device.get("logical_gpus", 0)
                    )
                except (TypeError, ValueError):
                    physical_gpus = 0
                    logical_gpus = 0
                if physical_gpus < 1 or logical_gpus < 1:
                    errors.append(
                        f"model {label} has no recorded training GPU"
                    )
            for parallelism_error in _training_parallelism_errors(
                item.get("training_parallelism"),
                expected_num_envs=spec_train_num_envs,
                expected_iterations=spec_train_episodes,
            ):
                errors.append(f"model {label} {parallelism_error}")
            if require_paths:
                model_path = Path(str(item.get("model_path", "")))
                model_root = Path(str(item.get("model_root", "")))
                training_results_path = Path(
                    str(item.get("training_results_path", ""))
                )
                artifact_path = Path(str(item.get("artifact_path", "")))
                manifest_path = Path(str(item.get("manifest_path", "")))
                loss_history_path = Path(
                    str(item.get("loss_history_path", ""))
                )
                if not model_path.is_dir():
                    errors.append(f"model path is missing: {model_path}")
                if model_path.resolve() != (model_root / "final").resolve():
                    errors.append(
                        f"model {label} path is not model_root/final"
                    )
                if isinstance(preliminary_post_eval, dict):
                    preliminary_model_path = Path(
                        str(
                            preliminary_post_eval.get(
                                "selected_model_path",
                                "",
                            )
                        )
                    )
                    if (
                        preliminary_model_path.resolve()
                        != model_path.resolve()
                    ):
                        errors.append(
                            f"model {label} preliminary post-eval did not "
                            "load the frozen final model"
                        )
                    for path_key, signature_key in (
                        ("results_path", "results_sha256"),
                        ("spec_path", "spec_sha256"),
                    ):
                        preliminary_path = Path(
                            str(preliminary_post_eval.get(path_key, ""))
                        )
                        if not preliminary_path.is_file():
                            errors.append(
                                f"model {label} preliminary post-eval "
                                f"{path_key} is missing: {preliminary_path}"
                            )
                        elif preliminary_post_eval.get(
                            signature_key
                        ) != _file_sha256(preliminary_path):
                            errors.append(
                                f"model {label} preliminary post-eval "
                                f"{signature_key} mismatch"
                            )
                if not artifact_path.is_file():
                    errors.append(f"artifact path is missing: {artifact_path}")
                if not training_results_path.is_file():
                    errors.append(
                        f"training results path is missing: "
                        f"{training_results_path}"
                    )
                else:
                    try:
                        frozen_training_result = _load_json(
                            training_results_path
                        )
                    except Exception as exc:
                        errors.append(
                            f"training results cannot be read: "
                            f"{training_results_path}: {exc}"
                        )
                    else:
                        if (
                            frozen_training_result.get("training_device")
                            != training_device
                        ):
                            errors.append(
                                f"model {label} training_device differs "
                                "from training results"
                            )
                        if (
                            frozen_training_result.get(
                                "training_parallelism"
                            )
                            != item.get("training_parallelism")
                        ):
                            errors.append(
                                f"model {label} training_parallelism differs "
                                "from training results"
                            )
                if not manifest_path.is_file():
                    errors.append(f"manifest path is missing: {manifest_path}")
                if not loss_history_path.is_file():
                    errors.append(
                        f"loss history path is missing: {loss_history_path}"
                    )
                actor_signature = None
                if model_path.is_dir():
                    try:
                        actor_signature = _actor_signature(model_path)
                    except (FileNotFoundError, OSError) as exc:
                        errors.append(str(exc))
                signature_checks = (
                    (
                        "actor_signature_sha1",
                        actor_signature,
                    ),
                    (
                        "training_results_sha256",
                        _file_sha256(training_results_path)
                        if training_results_path.is_file()
                        else None,
                    ),
                    (
                        "artifact_sha256",
                        _file_sha256(artifact_path)
                        if artifact_path.is_file()
                        else None,
                    ),
                    (
                        "manifest_sha256",
                        _file_sha256(manifest_path)
                        if manifest_path.is_file()
                        else None,
                    ),
                    (
                        "loss_history_sha256",
                        _file_sha256(loss_history_path)
                        if loss_history_path.is_file()
                        else None,
                    ),
                )
                for signature_key, actual_signature in signature_checks:
                    if (
                        actual_signature is not None
                        and item.get(signature_key) != actual_signature
                    ):
                        errors.append(
                            f"model {label} {signature_key} mismatch"
                        )

                expected_config = SELECTOR_PROTOCOL_CONFIG_BY_LABEL.get(label)
                selector_mode = (
                    expected_config["env"][
                        "CROSS_AGENT_REFERENCE_SELECTOR_MODE"
                    ]
                    if expected_config is not None
                    else ""
                )
                adaptive = selector_mode in ADVANTAGE_SELECTOR_MODES
                selector_state_raw = item.get("selector_state_path")
                selector_state_path = (
                    Path(str(selector_state_raw))
                    if selector_state_raw
                    else None
                )
                if adaptive:
                    if (
                        selector_state_path is None
                        or not selector_state_path.is_file()
                    ):
                        errors.append(
                            f"model {label} selector state path is missing"
                        )
                    elif item.get("selector_state_sha256") != _file_sha256(
                        selector_state_path
                    ):
                        errors.append(
                            f"model {label} selector_state_sha256 mismatch"
                        )
                elif (
                    selector_state_raw is not None
                    or item.get("selector_state_sha256") is not None
                ):
                    errors.append(
                        f"model {label} unexpectedly freezes selector state"
                    )

                shared_weight_raw = item.get("shared_selector_weight_path")
                shared_weight_path = (
                    Path(str(shared_weight_raw))
                    if shared_weight_raw
                    else None
                )
                if selector_mode == MODE_SHARED_TWIN_HEAD_TAIL:
                    if (
                        shared_weight_path is None
                        or not shared_weight_path.is_file()
                    ):
                        errors.append(
                            f"model {label} shared selector weight is missing"
                        )
                    elif item.get(
                        "shared_selector_weight_sha256"
                    ) != _file_sha256(shared_weight_path):
                        errors.append(
                            f"model {label} shared selector weight hash mismatch"
                        )
                elif (
                    shared_weight_raw is not None
                    or item.get("shared_selector_weight_sha256") is not None
                ):
                    errors.append(
                        f"model {label} unexpectedly freezes shared selector "
                        "weights"
                    )

        valid_actor_signatures = [
            str(item.get("actor_signature_sha1", "")).strip().lower()
            for item in models
            if isinstance(item, dict)
            and _valid_sha1(item.get("actor_signature_sha1"))
        ]
        if (
            len(valid_actor_signatures) == len(expected_labels)
            and len(set(valid_actor_signatures)) != len(expected_labels)
        ):
            errors.append(
                "the four model entries do not contain four distinct actor "
                "signatures"
            )
        if require_paths:
            resolved_model_paths = [
                str(Path(str(item.get("model_path", ""))).resolve())
                for item in models
                if isinstance(item, dict)
            ]
            if (
                len(resolved_model_paths) == len(expected_labels)
                and len(set(resolved_model_paths)) != len(expected_labels)
            ):
                errors.append(
                    "the four model entries do not contain four distinct "
                    "model paths"
                )

    modes = payload.get("modes")
    if modes != list(FORMAL_MODES):
        errors.append("mode matrix differs from the frozen four-mode protocol")

    sequences = payload.get("sequences")
    if not isinstance(sequences, dict):
        errors.append("sequences is missing or invalid")
    else:
        for key in (
            "terrain_complexity_level",
            "terrain_seed",
            "terrain_variant_seed",
            "obstacle_seed",
        ):
            values = sequences.get(key)
            if not isinstance(values, list) or len(values) != FORMAL_EVAL_EPISODES:
                errors.append(
                    f"sequences.{key} length="
                    f"{len(values) if isinstance(values, list) else 'invalid'}, "
                    f"expected={FORMAL_EVAL_EPISODES}"
                )

    if require_paths:
        for key, signature_key in (
            ("positions_file", "positions_file_sha256"),
            ("sequence_source_json", "sequence_source_sha256"),
        ):
            path = Path(str(payload.get(key, "")))
            if not path.is_file():
                errors.append(f"{key} is missing: {path}")
                continue
            recorded_signature = payload.get(signature_key)
            if not _valid_sha256(recorded_signature):
                errors.append(f"{signature_key} is not a SHA-256 digest")
            elif _file_sha256(path) != recorded_signature:
                errors.append(f"{signature_key} mismatch")
        out_root = Path(str(payload.get("out_root", "")))
        if not out_root.is_absolute():
            errors.append("out_root must be absolute")
    else:
        for signature_key in (
            "positions_file_sha256",
            "sequence_source_sha256",
        ):
            if not _valid_sha256(payload.get(signature_key)):
                errors.append(f"{signature_key} is not a SHA-256 digest")
    return errors


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return payload


def _resolve_source_path(repo_root: Path, seed_batch_dir: Path, value: Any) -> Path:
    raw = Path(str(value or "")).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        candidates = [repo_root / raw, seed_batch_dir / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _validate_preliminary_shared_spec(shared_spec: Mapping[str, Any]) -> None:
    expected_values = {
        "enabled": True,
        "mode": PRELIMINARY_POST_EVAL_MODE,
        "episodes": FORMAL_EVAL_EPISODES,
        "episode_length_multiplier": (
            PRELIMINARY_POST_EVAL_EPISODE_LENGTH_MULTIPLIER
        ),
        "seed": PRELIMINARY_POST_EVAL_SEED,
        "model_variant": PRELIMINARY_POST_EVAL_MODEL_VARIANT,
        "selection_protocol": PRELIMINARY_POST_EVAL_SELECTION_PROTOCOL,
        "requested_model_variant": PRELIMINARY_POST_EVAL_MODEL_VARIANT,
        "position_family": "train_match",
        "semi_random_terrain": True,
        "use_dynamic_obstacles": True,
        "use_fixed_positions": True,
    }
    errors = [
        f"{key}={shared_spec.get(key)!r}, expected={expected_value!r}"
        for key, expected_value in expected_values.items()
        if shared_spec.get(key) != expected_value
    ]
    for key in (
        "terrain_seed_sequence",
        "terrain_variant_seed_sequence",
        "obstacle_seed_sequence",
    ):
        values = shared_spec.get(key)
        if (
            not isinstance(values, list)
            or len(values) != FORMAL_EVAL_EPISODES
        ):
            errors.append(
                f"{key} length="
                f"{len(values) if isinstance(values, list) else 'invalid'}, "
                f"expected={FORMAL_EVAL_EPISODES}"
            )
    if errors:
        raise ValueError(
            "post_eval_shared_spec differs from the frozen preliminary "
            "gate: " + "; ".join(errors)
        )


def _preliminary_episode_detail_errors(
    episode_details: Any,
    embedded_spec: Mapping[str, Any],
) -> list[str]:
    if not isinstance(episode_details, list):
        return ["episode_details is not a list"]
    if len(episode_details) != FORMAL_EVAL_EPISODES:
        return [
            "episode_details length="
            f"{len(episode_details)}, expected={FORMAL_EVAL_EPISODES}"
        ]

    errors: list[str] = []
    episode_ids = [
        item.get("episode") if isinstance(item, Mapping) else None
        for item in episode_details
    ]
    expected_episode_ids = list(range(FORMAL_EVAL_EPISODES))
    if episode_ids != expected_episode_ids:
        errors.append(
            f"episode IDs={episode_ids!r}, expected={expected_episode_ids!r}"
        )

    for spec_key, detail_key in (
        ("terrain_seed_sequence", "terrain_seed"),
        ("terrain_variant_seed_sequence", "terrain_variant_seed"),
        ("obstacle_seed_sequence", "obstacle_seed"),
    ):
        expected_sequence = embedded_spec.get(spec_key)
        actual_sequence = [
            item.get(detail_key) if isinstance(item, Mapping) else None
            for item in episode_details
        ]
        if actual_sequence != expected_sequence:
            errors.append(
                f"episode_details.{detail_key} differs from {spec_key}"
            )
    return errors


def _build_preliminary_post_eval_evidence(
    *,
    artifact: Mapping[str, Any],
    experiment: Mapping[str, Any],
    shared_spec: Mapping[str, Any],
    repo_root: Path,
    seed_batch_dir: Path,
    model_final_dir: Path,
    actor_signature: str,
) -> Dict[str, Any]:
    label = str(experiment.get("label", ""))

    def require_equal(
        mapping: Mapping[str, Any],
        key: str,
        expected: Any,
        context: str,
    ) -> None:
        actual = mapping.get(key)
        if actual != expected:
            raise ValueError(
                f"{label} {context}.{key}={actual!r}, "
                f"expected={expected!r}"
            )

    for key, expected_value in {
        "post_eval_enabled": True,
        "allow_post_eval_without_train_success": True,
        "post_eval_mode": PRELIMINARY_POST_EVAL_MODE,
        "post_eval_episodes": FORMAL_EVAL_EPISODES,
        "post_eval_episode_length_multiplier": (
            PRELIMINARY_POST_EVAL_EPISODE_LENGTH_MULTIPLIER
        ),
        "post_eval_seed": PRELIMINARY_POST_EVAL_SEED,
        "post_eval_selection_protocol": (
            PRELIMINARY_POST_EVAL_SELECTION_PROTOCOL
        ),
        "post_eval_model_variant": PRELIMINARY_POST_EVAL_MODEL_VARIANT,
        "post_eval_requested_model_variant": (
            PRELIMINARY_POST_EVAL_MODEL_VARIANT
        ),
    }.items():
        require_equal(artifact, key, expected_value, "artifact")

    post_eval = experiment.get("post_eval")
    if not isinstance(post_eval, dict):
        raise ValueError(f"{label} artifact has no post_eval object")
    for key, expected_value in {
        "enabled": True,
        "eligible": True,
        "status": "completed",
        "skipped": False,
        "mode": PRELIMINARY_POST_EVAL_MODE,
        "episodes": FORMAL_EVAL_EPISODES,
        "episode_length_multiplier": (
            PRELIMINARY_POST_EVAL_EPISODE_LENGTH_MULTIPLIER
        ),
        "seed": PRELIMINARY_POST_EVAL_SEED,
        "model_variant": PRELIMINARY_POST_EVAL_MODEL_VARIANT,
        "selection_protocol": PRELIMINARY_POST_EVAL_SELECTION_PROTOCOL,
        "selected_model_candidate": PRELIMINARY_POST_EVAL_MODEL_VARIANT,
        "selected_model_variant": PRELIMINARY_POST_EVAL_MODEL_VARIANT,
    }.items():
        require_equal(post_eval, key, expected_value, "post_eval")

    selected_model_path = _resolve_source_path(
        repo_root,
        seed_batch_dir,
        post_eval.get("selected_model_path"),
    )
    if selected_model_path != model_final_dir:
        raise ValueError(
            f"{label} preliminary post-eval selected "
            f"{selected_model_path}, expected final model {model_final_dir}"
        )

    embedded_spec = post_eval.get("spec")
    if not isinstance(embedded_spec, dict):
        raise ValueError(
            f"{label} preliminary post-eval has no embedded spec"
        )
    spec_path = _resolve_source_path(
        repo_root,
        seed_batch_dir,
        post_eval.get("spec_path"),
    )
    on_disk_spec = _load_json(spec_path)
    if on_disk_spec != embedded_spec:
        raise ValueError(
            f"{label} preliminary post-eval spec file differs from artifact"
        )
    for key, expected_value in shared_spec.items():
        if embedded_spec.get(key) != expected_value:
            raise ValueError(
                f"{label} preliminary post-eval spec differs from shared "
                f"spec at {key}"
            )
    for key, expected_value in {
        "selected_model_candidate": PRELIMINARY_POST_EVAL_MODEL_VARIANT,
        "selected_model_variant": PRELIMINARY_POST_EVAL_MODEL_VARIANT,
        "resolved_model_variant": PRELIMINARY_POST_EVAL_MODEL_VARIANT,
        "selected_model_path": str(model_final_dir),
        "selected_model_signature": actor_signature,
    }.items():
        require_equal(
            embedded_spec,
            key,
            expected_value,
            "post_eval.spec",
        )

    results_path = _resolve_source_path(
        repo_root,
        seed_batch_dir,
        post_eval.get("results_path"),
    )
    evaluation_result = _load_json(results_path)
    require_equal(
        evaluation_result,
        "episodes",
        FORMAL_EVAL_EPISODES,
        "post_eval.results",
    )
    result_model_path = _resolve_source_path(
        repo_root,
        seed_batch_dir,
        evaluation_result.get("model_path"),
    )
    if result_model_path != model_final_dir:
        raise ValueError(
            f"{label} preliminary result loaded {result_model_path}, "
            f"expected {model_final_dir}"
        )
    episode_details = evaluation_result.get("episode_details")
    episode_detail_errors = _preliminary_episode_detail_errors(
        episode_details,
        embedded_spec,
    )
    if episode_detail_errors:
        raise ValueError(
            f"{label} preliminary result has invalid episode details: "
            + "; ".join(episode_detail_errors)
        )
    summary = evaluation_result.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(
            f"{label} preliminary result has no summary object"
        )
    artifact_summary = post_eval.get("summary")
    if not isinstance(artifact_summary, dict):
        raise ValueError(
            f"{label} preliminary artifact has no summary object"
        )
    for key in (
        "team_success_rate",
        "avg_collision_count",
        "avg_team_total_path_length",
    ):
        if key not in summary:
            raise ValueError(
                f"{label} preliminary result summary has no {key}"
            )
        if artifact_summary.get(key) != summary.get(key):
            raise ValueError(
                f"{label} preliminary artifact/result summary differs "
                f"at {key}"
            )
    for key in (
        "terrain_seed_sequence",
        "terrain_variant_seed_sequence",
        "obstacle_seed_sequence",
    ):
        if evaluation_result.get(key) != embedded_spec.get(key):
            raise ValueError(
                f"{label} preliminary result differs from spec at {key}"
            )

    evaluation_setup = evaluation_result.get("evaluation_setup")
    if not isinstance(evaluation_setup, dict):
        raise ValueError(
            f"{label} preliminary result has no evaluation_setup"
        )
    for key, expected_value in {
        "terrain_family": embedded_spec.get("terrain_family"),
        "position_family": embedded_spec.get("position_family"),
        "semi_random_terrain": embedded_spec.get("semi_random_terrain"),
        "use_dynamic_obstacles": embedded_spec.get(
            "use_dynamic_obstacles"
        ),
        "use_fixed_positions": True,
        "eval_backend": "python_only",
    }.items():
        if evaluation_setup.get(key) != expected_value:
            raise ValueError(
                f"{label} preliminary evaluation_setup.{key}="
                f"{evaluation_setup.get(key)!r}, expected={expected_value!r}"
            )
    try:
        setup_episode_length_multiplier = float(
            evaluation_setup.get("requested_episode_length_multiplier")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} preliminary evaluation_setup has invalid "
            "requested_episode_length_multiplier"
        ) from exc
    if not math.isclose(
        setup_episode_length_multiplier,
        PRELIMINARY_POST_EVAL_EPISODE_LENGTH_MULTIPLIER,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{label} preliminary evaluation_setup."
            "requested_episode_length_multiplier="
            f"{setup_episode_length_multiplier!r}, expected="
            f"{PRELIMINARY_POST_EVAL_EPISODE_LENGTH_MULTIPLIER!r}"
        )
    setup_positions_file = _resolve_source_path(
        repo_root,
        seed_batch_dir,
        evaluation_setup.get("positions_file"),
    )
    expected_positions_file = _resolve_source_path(
        repo_root,
        seed_batch_dir,
        embedded_spec.get("default_positions_file"),
    )
    if setup_positions_file != expected_positions_file:
        raise ValueError(
            f"{label} preliminary evaluation positions file "
            f"{setup_positions_file} differs from {expected_positions_file}"
        )
    eval_device = evaluation_setup.get("eval_device")
    if not isinstance(eval_device, dict):
        raise ValueError(
            f"{label} preliminary result has no eval_device evidence"
        )
    if eval_device.get("require_gpu") is not True:
        raise ValueError(
            f"{label} preliminary post-eval did not require GPU"
        )
    try:
        physical_gpus = int(eval_device.get("physical_gpus", 0) or 0)
        logical_gpus = int(eval_device.get("logical_gpus", 0) or 0)
    except (TypeError, ValueError):
        physical_gpus = 0
        logical_gpus = 0
    if physical_gpus < 1 or logical_gpus < 1:
        raise ValueError(
            f"{label} preliminary post-eval has no physical/logical GPU"
        )
    try:
        noise_scale = float(evaluation_setup.get("eval_noise_scale"))
        random_action_prob = float(
            evaluation_setup.get("eval_random_action_prob")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} preliminary result has invalid action-noise fields"
        ) from exc
    if noise_scale != 0.0 or random_action_prob != 0.0:
        raise ValueError(
            f"{label} preliminary post-eval is not deterministic"
        )

    return {
        "status": "completed",
        "mode": PRELIMINARY_POST_EVAL_MODE,
        "episodes": FORMAL_EVAL_EPISODES,
        "episode_length_multiplier": (
            PRELIMINARY_POST_EVAL_EPISODE_LENGTH_MULTIPLIER
        ),
        "seed": PRELIMINARY_POST_EVAL_SEED,
        "selection_protocol": PRELIMINARY_POST_EVAL_SELECTION_PROTOCOL,
        "requested_model_variant": PRELIMINARY_POST_EVAL_MODEL_VARIANT,
        "resolved_model_variant": PRELIMINARY_POST_EVAL_MODEL_VARIANT,
        "selected_model_path": str(model_final_dir),
        "selected_model_signature_sha1": actor_signature,
        "gpu_required": True,
        "physical_gpus": physical_gpus,
        "logical_gpus": logical_gpus,
        "results_path": str(results_path),
        "results_sha256": _file_sha256(results_path),
        "spec_path": str(spec_path),
        "spec_sha256": _file_sha256(spec_path),
    }


def build_batch_spec(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    seed_batch_value = str(getattr(args, "seed_batch_dir", "") or "").strip()
    if seed_batch_value:
        seed_batch_dir = Path(seed_batch_value).expanduser().resolve()
    else:
        parent_batch_dir = Path(
            str(getattr(args, "parent_batch_dir", "") or "")
        ).expanduser().resolve()
        seed_batches_root = parent_batch_dir / "seed_batches"
        candidates = sorted(
            path.resolve()
            for path in seed_batches_root.glob(
                f"batch_groupB_seed{int(args.train_seed)}_*"
            )
            if path.is_dir()
        )
        if len(candidates) != 1:
            raise ValueError(
                "parent batch must resolve to exactly one seed batch: "
                f"parent={parent_batch_dir}, train_seed={int(args.train_seed)}, "
                f"candidates={[str(path) for path in candidates]}"
            )
        seed_batch_dir = candidates[0]
    if not seed_batch_dir.is_dir():
        raise FileNotFoundError(seed_batch_dir)
    artifacts_dir = seed_batch_dir / "results" / "experiment_artifacts"
    shared_spec_path = seed_batch_dir / "results" / "post_eval_shared_spec.json"
    shared_spec = _load_json(shared_spec_path)
    _validate_preliminary_shared_spec(shared_spec)

    models = []
    observed_seed = None
    observed_train_episodes = None
    observed_train_num_envs = None
    observed_training_episode_length = None
    common_manifest_env = None
    mechanism_env_keys = {
        "CROSS_AGENT_REFERENCE_TARGET_SEMANTICS",
        "CROSS_AGENT_REFERENCE_SELECTOR_MODE",
        "CROSS_AGENT_REFERENCE_SELECTOR_ENABLED",
    }
    for label in SELECTOR_PROTOCOL_EXPERIMENT_LABELS:
        artifact_path = (artifacts_dir / f"{label}.json").resolve()
        artifact = _load_json(artifact_path)
        experiment = artifact.get("experiment")
        if not isinstance(experiment, dict) or experiment.get("label") != label:
            raise ValueError(f"artifact experiment identity mismatch: {artifact_path}")
        seed = int(artifact.get("seed"))
        if observed_seed is None:
            observed_seed = seed
        elif seed != observed_seed:
            raise ValueError("selector artifacts do not share one training seed")

        log_dir = Path(str(experiment.get("log_dir", ""))).expanduser().resolve()
        experiment_name = log_dir.parent.name
        model_root = (repo_root / "models" / experiment_name).resolve()
        result_path = model_root / "results.json"
        result = _load_json(result_path)
        run_args = result.get("args")
        if not isinstance(run_args, dict):
            raise ValueError(f"results.args missing: {result_path}")
        training_device = result.get("training_device")
        if not isinstance(training_device, dict):
            raise ValueError(
                f"{label} results do not record the training device"
            )
        if training_device.get("require_gpu") is not True:
            raise ValueError(
                f"{label} was not trained with MATD3_REQUIRE_GPU=1"
            )
        if (
            int(training_device.get("physical_gpus", 0) or 0) < 1
            or int(training_device.get("logical_gpus", 0) or 0) < 1
        ):
            raise ValueError(
                f"{label} results do not prove physical and logical GPU use"
            )
        train_episodes = int(run_args.get("train_episodes", 0) or 0)
        if observed_train_episodes is None:
            observed_train_episodes = train_episodes
        elif train_episodes != observed_train_episodes:
            raise ValueError("selector models do not share one training horizon")
        training_episode_length = int(
            run_args.get("episode_length", 0) or 0
        )
        if training_episode_length <= 0:
            raise ValueError(f"{label} has invalid training episode_length")
        if observed_training_episode_length is None:
            observed_training_episode_length = training_episode_length
        elif training_episode_length != observed_training_episode_length:
            raise ValueError(
                "selector models do not share one training episode length"
            )
        if args.train_episodes is not None and train_episodes != int(
            args.train_episodes
        ):
            raise ValueError(
                f"{label} train_episodes={train_episodes}, "
                f"expected={int(args.train_episodes)}"
            )
        train_num_envs = int(run_args.get("num_envs", 0) or 0)
        if train_num_envs <= 0:
            raise ValueError(f"{label} has invalid results.args.num_envs")
        if observed_train_num_envs is None:
            observed_train_num_envs = train_num_envs
        elif train_num_envs != observed_train_num_envs:
            raise ValueError(
                "selector models do not share one training num_envs"
            )
        if args.train_num_envs is not None and train_num_envs != int(
            args.train_num_envs
        ):
            raise ValueError(
                f"{label} num_envs={train_num_envs}, "
                f"expected={int(args.train_num_envs)}"
            )
        training_parallelism = result.get("training_parallelism")
        parallelism_errors = _training_parallelism_errors(
            training_parallelism,
            expected_num_envs=train_num_envs,
            expected_iterations=train_episodes,
        )
        if parallelism_errors:
            raise ValueError(
                f"{label} invalid training parallelism: "
                + "; ".join(parallelism_errors)
            )

        expected_config = SELECTOR_PROTOCOL_CONFIG_BY_LABEL[label]
        result_arg_errors = _selector_result_arg_errors(
            run_args,
            expected_config["env"],
        )
        if result_arg_errors:
            raise ValueError(
                f"{label} effective training arguments differ from the "
                "selector protocol: " + "; ".join(result_arg_errors)
            )
        manifest_path = _resolve_source_path(
            repo_root,
            seed_batch_dir,
            experiment.get("manifest_path"),
        )
        manifest = _load_json(manifest_path)
        manifest_env = manifest.get("exec_env")
        if not isinstance(manifest_env, dict):
            raise ValueError(f"manifest exec_env missing: {manifest_path}")
        env_errors = [
            f"{key}={manifest_env.get(key)!r}, expected={expected_value!r}"
            for key, expected_value in expected_config["env"].items()
            if str(manifest_env.get(key, "")) != str(expected_value)
        ]
        if env_errors:
            raise ValueError(
                f"{label} frozen manifest differs from selector protocol: "
                + "; ".join(env_errors)
            )
        manifest_common_projection = {
            str(key): str(value)
            for key, value in manifest_env.items()
            if str(key) not in mechanism_env_keys
        }
        if common_manifest_env is None:
            common_manifest_env = manifest_common_projection
        elif manifest_common_projection != common_manifest_env:
            changed_keys = sorted(
                key
                for key in (
                    set(common_manifest_env)
                    | set(manifest_common_projection)
                )
                if common_manifest_env.get(key)
                != manifest_common_projection.get(key)
            )
            raise ValueError(
                f"{label} resolved manifest has non-mechanism drift: "
                + ", ".join(changed_keys[:20])
            )

        completion_errors = training_unit_completion_errors(
            model_root,
            train_episodes,
            repo_root=repo_root,
            expected_seed=seed,
            expected_num_envs=train_num_envs,
        )
        if completion_errors:
            raise ValueError(
                f"{label} is not a complete training unit: "
                + "; ".join(completion_errors)
            )
        selector_mode = str(
            run_args.get("cross_agent_reference_selector_mode", "hard")
            or "hard"
        ).strip().lower()
        loss_history_path = (log_dir / "loss_history.json").resolve()
        if not loss_history_path.is_file():
            raise ValueError(f"{label} loss_history.json is missing")
        loss_history = json.loads(
            loss_history_path.read_text(encoding="utf-8")
        )
        if not isinstance(loss_history, list) or not loss_history:
            raise ValueError(f"{label} loss history is empty or invalid")
        active_cross_ref_rows, positive_valid_rows = (
            _cross_reference_activity_rows(
                loss_history,
                start_episode=int(
                    run_args.get(
                        "cross_agent_reference_start_episode",
                        0,
                    )
                    or 0
                ),
            )
        )
        if not active_cross_ref_rows:
            raise ValueError(
                f"{label} never recorded an active cross-reference update"
            )
        if not positive_valid_rows:
            raise ValueError(
                f"{label} never recorded an eligible reference sample"
            )

        if selector_mode in ADVANTAGE_SELECTOR_MODES:
            selector_state_path = (
                model_root
                / "final"
                / "cross_agent_reference_state.json"
            )
            selector_state = _load_json(selector_state_path)
            state_errors = selector_state_errors(
                selector_state,
                expected_mode=selector_mode,
                require_null_input_dim=(
                    selector_mode != MODE_SHARED_TWIN_HEAD_TAIL
                ),
            )
            if state_errors:
                raise ValueError(
                    f"{label} selector state is invalid: "
                    + "; ".join(state_errors)
                )
            if int(selector_state["selector_update_count"]) <= 0:
                raise ValueError(
                    f"{label} selector/EMA update_count is zero"
                )
            if not bool(selector_state["head_ema_initialized"]):
                raise ValueError(f"{label} head advantage EMA was never initialized")
            if not bool(selector_state["tail_ema_initialized"]):
                raise ValueError(f"{label} tail advantage EMA was never initialized")

            adaptive_rows = [
                row
                for row in loss_history
                if isinstance(row, dict)
                and row.get("cross_ref_selector_update_count") is not None
            ]
            if not adaptive_rows:
                raise ValueError(
                    f"{label} loss history has no adaptive selector diagnostics"
                )
            for diagnostic_key in (
                "cross_ref_head_twin_agreement_ratio",
                "cross_ref_tail_twin_agreement_ratio",
                "cross_ref_head_multiplier",
                "cross_ref_tail_multiplier",
                "cross_ref_head_advantage_ema",
                "cross_ref_tail_advantage_ema",
            ):
                finite_values = []
                for row in adaptive_rows:
                    try:
                        value = float(row.get(diagnostic_key))
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(value):
                        finite_values.append(value)
                if not finite_values:
                    raise ValueError(
                        f"{label} has no finite {diagnostic_key} diagnostics"
                    )
            if selector_mode == MODE_SHARED_TWIN_HEAD_TAIL:
                positive_gradient_rows = []
                for row in adaptive_rows:
                    try:
                        gradient_norm = float(
                            row.get("cross_ref_selector_gradient_norm")
                        )
                        selector_loss = float(
                            row.get("cross_ref_selector_loss")
                        )
                    except (TypeError, ValueError):
                        continue
                    if (
                        math.isfinite(gradient_norm)
                        and math.isfinite(selector_loss)
                        and gradient_norm > 0.0
                        and selector_loss > 0.0
                    ):
                        positive_gradient_rows.append(row)
                if not positive_gradient_rows:
                    raise ValueError(
                        f"{label} never recorded a positive shared-selector "
                        "gradient and loss"
                    )
        model_final_dir = (model_root / "final").resolve()
        selector_state_artifact = (
            model_final_dir / "cross_agent_reference_state.json"
            if selector_mode in ADVANTAGE_SELECTOR_MODES
            else None
        )
        shared_selector_weight_artifact = (
            model_final_dir / "reference_selector_shared.weights.h5"
            if selector_mode == MODE_SHARED_TWIN_HEAD_TAIL
            else None
        )
        actor_signature = _actor_signature(model_final_dir)
        preliminary_post_eval = _build_preliminary_post_eval_evidence(
            artifact=artifact,
            experiment=experiment,
            shared_spec=shared_spec,
            repo_root=repo_root,
            seed_batch_dir=seed_batch_dir,
            model_final_dir=model_final_dir,
            actor_signature=actor_signature,
        )
        models.append(
            {
                "id": SELECTOR_PROTOCOL_ID_BY_LABEL[label],
                "label": label,
                "model_variant": "final",
                "model_path": str(model_final_dir),
                "model_root": str(model_root),
                "training_device": training_device,
                "training_parallelism": training_parallelism,
                "actor_signature_sha1": actor_signature,
                "preliminary_post_eval": preliminary_post_eval,
                "training_results_path": str(result_path.resolve()),
                "training_results_sha256": _file_sha256(
                    result_path.resolve()
                ),
                "artifact_path": str(artifact_path),
                "artifact_sha256": _file_sha256(artifact_path),
                "manifest_path": str(manifest_path),
                "manifest_sha256": _file_sha256(manifest_path),
                "loss_history_path": str(loss_history_path),
                "loss_history_sha256": _file_sha256(loss_history_path),
                "selector_state_path": (
                    str(selector_state_artifact)
                    if selector_state_artifact is not None
                    else None
                ),
                "selector_state_sha256": (
                    _file_sha256(selector_state_artifact)
                    if selector_state_artifact is not None
                    else None
                ),
                "shared_selector_weight_path": (
                    str(shared_selector_weight_artifact)
                    if shared_selector_weight_artifact is not None
                    else None
                ),
                "shared_selector_weight_sha256": (
                    _file_sha256(shared_selector_weight_artifact)
                    if shared_selector_weight_artifact is not None
                    else None
                ),
            }
        )

    if observed_seed != int(args.train_seed):
        raise ValueError(
            f"selector artifacts train_seed={observed_seed}, "
            f"expected={int(args.train_seed)}"
        )

    terrain_complexity = int(shared_spec.get("terrain_complexity", 0) or 0)
    sequences = {
        "terrain_complexity_level": [
            terrain_complexity
            for _ in range(FORMAL_EVAL_EPISODES)
        ],
        "terrain_seed": [
            int(value)
            for value in shared_spec.get("terrain_seed_sequence", [])
        ],
        "terrain_variant_seed": [
            int(value)
            for value in shared_spec.get("terrain_variant_seed_sequence", [])
        ],
        "obstacle_seed": [
            int(value)
            for value in shared_spec.get("obstacle_seed_sequence", [])
        ],
    }
    positions_file = _resolve_source_path(
        repo_root,
        seed_batch_dir,
        shared_spec.get("default_positions_file"),
    )
    environment = {
        "semi_random_terrain": bool(
            shared_spec.get("semi_random_terrain", False)
        ),
        "use_dynamic_obstacles": bool(
            shared_spec.get("use_dynamic_obstacles", False)
        ),
        "scenario_seed": int(shared_spec.get("scenario_seed")),
        "terrain_base_seed": int(
            shared_spec.get(
                "terrain_base_seed",
                shared_spec.get("scenario_seed"),
            )
        ),
        "post_eval_mode": "shared_match_train_env",
        "post_eval_terrain_family": "train_match",
        "post_eval_position_family": "train_match",
        # Every model/mode uses the exact same fixed start/goal geometry while
        # terrain variants, dynamic obstacles, and action noise follow the
        # explicit per-episode sequences below.
        "position_protocol": "single_fixed_positions_file",
    }
    payload: Dict[str, Any] = {
        "schema_version": BATCH_SPEC_SCHEMA_VERSION,
        "protocol_version": BATCH_PROTOCOL_VERSION,
        "selector_protocol_schema_version": (
            SELECTOR_PROTOCOL_SCHEMA_VERSION
        ),
        "experiment_family": "selector_m0_m3_noise_dependency",
        "train_seed": int(observed_seed),
        "train_episodes": int(observed_train_episodes),
        "train_num_envs": int(observed_train_num_envs),
        "train_environment_trajectories": int(
            int(observed_train_episodes) * int(observed_train_num_envs)
        ),
        "training_episode_length": int(observed_training_episode_length),
        "episodes": FORMAL_EVAL_EPISODES,
        "require_gpu": True,
        "episode_length_multiplier": 1.1,
        "episode_length": int(
            int(observed_training_episode_length) * 1.1 + 0.5
        ),
        "eval_noise_seed": int(args.eval_noise_seed),
        "eval_process_shards": int(args.eval_process_shards),
        "eval_process_workers": int(args.eval_process_workers),
        "eval_shard_episode_parallelism": int(
            args.eval_shard_episode_parallelism
        ),
        "eval_shard_env_step_threads": int(
            args.eval_shard_env_step_threads
        ),
        "out_root": str(Path(args.out_root).expanduser().resolve()),
        "seed_batch_dir": str(seed_batch_dir),
        "positions_file": str(positions_file),
        "positions_file_sha256": _file_sha256(positions_file),
        "sequence_source_json": str(shared_spec_path.resolve()),
        "sequence_source_sha256": _file_sha256(shared_spec_path.resolve()),
        "environment": environment,
        "models": models,
        "modes": list(FORMAL_MODES),
        "sequences": sequences,
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    errors = validate_batch_spec(payload)
    if errors:
        raise ValueError("invalid generated batch spec: " + "; ".join(errors))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(_DEFAULT_REPO_ROOT))
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument("--seed-batch-dir")
    source_group.add_argument("--parent-batch-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--out-root")
    parser.add_argument("--train-seed", type=int, default=101)
    parser.add_argument("--train-episodes", type=int, default=1000)
    parser.add_argument("--train-num-envs", type=int, default=4)
    parser.add_argument("--eval-noise-seed", type=int, default=101)
    parser.add_argument("--eval-process-shards", type=int, default=3)
    parser.add_argument("--eval-process-workers", type=int, default=3)
    parser.add_argument("--eval-shard-episode-parallelism", type=int, default=4)
    parser.add_argument("--eval-shard-env-step-threads", type=int, default=4)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    if args.validate:
        payload = _load_json(output)
        errors = validate_batch_spec(payload)
        if errors:
            raise SystemExit("[INVALID] " + "; ".join(errors))
        print(
            f"[VALID] {output} "
            f"sha256={payload['content_sha256']} "
            f"matrix={len(payload['models'])}x{len(payload['modes'])}x"
            f"{payload['episodes']}"
        )
        return 0
    if not args.seed_batch_dir and not args.parent_batch_dir:
        raise SystemExit(
            "building a batch spec requires exactly one of "
            "--seed-batch-dir or --parent-batch-dir"
        )
    if not str(args.out_root or "").strip():
        raise SystemExit("building a batch spec requires --out-root")
    payload = build_batch_spec(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        f"[BUILT] {output} "
        f"sha256={payload['content_sha256']} "
        f"matrix={len(payload['models'])}x{len(payload['modes'])}x"
        f"{payload['episodes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
