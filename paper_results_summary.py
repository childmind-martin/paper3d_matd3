#!/usr/bin/env python3
"""Paper-grade multi-seed aggregation and paired hierarchical bootstrap."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from paper_seed_statistics import holm_adjust, paired_seed_sign_flip

from paper_experiment_protocol import (
    FORMAL_EVALUATION_DEVICE_REQUIREMENT,
    load_protocol,
)
from paper_checkpoint_selection import validate_selection_result
from paper_experiment_schema import (
    PAPER_EPISODE_COLLECTION_SCHEMA_VERSION,
    PAPER_RUN_MANIFEST_SCHEMA_VERSION,
    PaperSchemaError,
    atomic_write_json,
    content_sha256,
    fingerprint_artifact,
    validate_episode_record,
)
from paper_task_bank import load_task_bank


PAPER_RESULTS_SUMMARY_SCHEMA_VERSION = "paper_results_summary_v6"
_METRICS = (
    "team_success_rate",
    "collision_rate",
    "agent_spl",
    "minimum_clearance_m",
    "energy",
    "control_effort",
)
_METRIC_BETTER_DIRECTION = {
    "team_success_rate": "higher",
    "collision_rate": "lower",
    "agent_spl": "higher",
    "minimum_clearance_m": "higher",
    "energy": "lower",
    "control_effort": "lower",
}
_MECHANISM_METRICS = (
    "configured_action_force_ratio",
    "mean_raw_action_norm",
    "mean_executed_action_norm",
    "mean_action_intervention_norm",
    "apf_intervention_step_fraction",
    "mean_apf_force_norm",
)


def _adjust_exploratory_seed_tests(
    comparisons: Mapping[str, Any],
    generalization_aggregate: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Adjust one complete diagnostic family before JSON/CSV serialization.

    Include every reported split, baseline and metric, plus the G1 aggregate.
    Even the diagnostic p value on the predeclared primary contrast is in this
    family. Its separate, protocol-frozen CI decision is not changed.
    """
    entries = []
    groups = list(comparisons.items())
    if generalization_aggregate is not None:
        groups.append((
            generalization_aggregate["group_id"],
            generalization_aggregate["paired_comparisons"],
        ))
    for split, baselines in groups:
        for baseline, metric_results in baselines.items():
            for metric, comparison in metric_results.items():
                entries.append((split, baseline, metric, comparison["paired_seed_test"]))
    adjusted = holm_adjust([
        test["p_value_two_sided"] for _, _, _, test in entries
    ])
    family_id = "all_reported_pairwise_training_seed_diagnostics_v1"
    for (_, _, _, test), adjusted_p in zip(entries, adjusted):
        test.update({
            "p_value_two_sided_holm": adjusted_p,
            "multiplicity_method": "Holm",
            "multiplicity_family_id": family_id,
            "multiplicity_family_size": len(entries),
        })
    return {
        "analysis_role": "exploratory_diagnostic_not_confirmatory_decision",
        "test": "paired_training_seed_sign_flip_two_sided",
        "multiplicity_method": "Holm",
        "multiplicity_family_id": family_id,
        "multiplicity_family_size": len(entries),
        "multiplicity_scope": (
            "all_reported_splits_x_baselines_x_metrics_and_G1_aggregate_"
            "including_diagnostic_test_of_primary_contrast"
        ),
        "family_members": [
            {"split": split, "baseline_method": baseline, "metric": metric}
            for split, baseline, metric, _ in entries
        ],
        "independent_unit": "train_seed",
        "inference_scope": "training_seed_variation_conditional_on_evaluated_task_set",
        "bootstrap_probability_interpretation": "descriptive_resample_fraction_not_p_value",
        "confirmatory_decision": "unchanged_protocol_frozen_oriented_CI_rule",
        "secondary_decision": "diagnostics_only_no_binary_superiority_claims",
    }


def _seed_test_csv_fields(comparison: Mapping[str, Any]) -> Dict[str, Any]:
    test = comparison["paired_seed_test"]
    return {
        "seed_test": test["test"],
        "seed_test_p_value_two_sided": test["p_value_two_sided"],
        "seed_test_p_value_two_sided_holm": test["p_value_two_sided_holm"],
        "seed_test_independent_train_seed_count": test["independent_train_seed_count"],
        "seed_test_exact": test["exact"],
        "seed_test_sign_patterns": test["sign_patterns"],
        "seed_test_random_seed": test["random_seed"],
        "seed_test_p_value_computation": test["p_value_computation"],
        "seed_test_minimum_nonzero_reportable_p_value": test["minimum_nonzero_reportable_p_value"],
        "seed_test_null_assumption": test["null_assumption"],
        "seed_test_inference_role": test["inference_role"],
        "seed_test_inference_scope": test["inference_scope"],
        "seed_test_multiplicity_method": test["multiplicity_method"],
        "seed_test_multiplicity_family_id": test["multiplicity_family_id"],
        "seed_test_multiplicity_family_size": test["multiplicity_family_size"],
        "bootstrap_probability_interpretation": "descriptive_resample_fraction_not_p_value",
    }


def _build_confirmatory_result(
    contract: Mapping[str, Any],
    comparisons: Mapping[str, Any],
    *,
    proposed_method: str,
) -> Dict[str, Any]:
    expected_policy = "single_predeclared_confirmatory_contrast"
    if contract.get("multiplicity_policy") != expected_policy:
        raise PaperSchemaError("confirmatory multiplicity policy mismatch")
    if str(contract.get("proposed_method", "")) != str(proposed_method):
        raise PaperSchemaError("confirmatory proposed method mismatch")
    split = str(contract.get("split", ""))
    baseline = str(contract.get("baseline_method", ""))
    metric = str(contract.get("metric", ""))
    if not split or not baseline or metric != "team_success_rate":
        raise PaperSchemaError("confirmatory contrast contract is incomplete")
    try:
        comparison = dict(comparisons[split][baseline][metric])
        interval = dict(
            comparison["oriented_improvement_bootstrap_95ci"]
        )
        lower = float(interval["lower"])
        confidence = float(interval["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PaperSchemaError(
            "confirmatory comparison result is incomplete"
        ) from exc
    if not math.isclose(
        confidence,
        float(contract.get("confidence", -1.0)),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise PaperSchemaError("confirmatory confidence mismatch")
    expected_rule = (
        "oriented_improvement_bootstrap_95ci_lower_bound_gt_zero"
    )
    if contract.get("decision_rule") != expected_rule:
        raise PaperSchemaError("confirmatory decision rule mismatch")
    return {
        "contract": dict(contract),
        "comparison": comparison,
        "criterion_met": bool(lower > 0.0),
    }


def _load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _validate_hashed(payload: Mapping[str, Any], label: str) -> Dict[str, Any]:
    expected = content_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    if payload.get("content_sha256") != expected:
        raise PaperSchemaError(f"{label} content hash mismatch")
    return dict(payload)


def _load_official_run(
    manifest_path: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_artifact: Mapping[str, Any],
    bank: Mapping[str, Any],
    method: str,
    train_seed: int,
    split: str,
) -> Dict[str, Any]:
    manifest = _validate_hashed(_load_json(manifest_path), str(manifest_path))
    if manifest.get("schema_version") != PAPER_RUN_MANIFEST_SCHEMA_VERSION:
        raise PaperSchemaError(f"official run manifest schema mismatch: {manifest_path}")
    if manifest.get("split_role") not in {
        "test_in_distribution",
        "test_generalization",
    }:
        raise PaperSchemaError(f"non-test run entered official summary: {manifest_path}")
    split_cfg = dict((bank.get("splits") or {}).get(split) or {})
    if not split_cfg:
        raise PaperSchemaError(f"official split is absent from task bank: {split}")
    expected_role = str(split_cfg.get("role", ""))
    if manifest.get("phase") != expected_role or manifest.get("split_role") != (
        expected_role
    ):
        raise PaperSchemaError(f"official run role mismatch: {manifest_path}")
    if manifest.get("algorithm") != method:
        raise PaperSchemaError(f"official run method mismatch: {manifest_path}")
    if int(manifest.get("train_seed", -1)) != int(train_seed):
        raise PaperSchemaError(f"official run train seed mismatch: {manifest_path}")
    if manifest.get("split") != split:
        raise PaperSchemaError(f"official run split mismatch: {manifest_path}")
    if int(manifest.get("eval_seed", -1)) != int(
        protocol["official_test"]["eval_seed"]
    ):
        raise PaperSchemaError(f"official run eval seed mismatch: {manifest_path}")

    recorded_protocol = dict(manifest.get("experiment_protocol") or {})
    actual_protocol = fingerprint_artifact(recorded_protocol.get("path"))
    if actual_protocol.get("sha256") != recorded_protocol.get("sha256"):
        raise PaperSchemaError(f"official protocol artifact changed: {manifest_path}")
    if recorded_protocol.get("sha256") != protocol_artifact.get("sha256"):
        raise PaperSchemaError(f"official run uses a different protocol: {manifest_path}")
    if recorded_protocol.get("protocol_content_sha256") != protocol.get(
        "content_sha256"
    ):
        raise PaperSchemaError(f"official protocol content mismatch: {manifest_path}")
    runtime_environment = dict(manifest.get("runtime_environment") or {})
    evaluation_device = dict(
        runtime_environment.get("evaluation_device_provenance") or {}
    )
    expected_eval_environment = {
        "CUDA_VISIBLE_DEVICES": FORMAL_EVALUATION_DEVICE_REQUIREMENT[
            "cuda_visible_devices"
        ],
        "TF_DETERMINISTIC_OPS": FORMAL_EVALUATION_DEVICE_REQUIREMENT[
            "tf_deterministic_ops"
        ],
        "TF_ENABLE_ONEDNN_OPTS": FORMAL_EVALUATION_DEVICE_REQUIREMENT[
            "tf_enable_onednn_opts"
        ],
        "TF_NUM_INTRAOP_THREADS": FORMAL_EVALUATION_DEVICE_REQUIREMENT[
            "tf_num_intraop_threads"
        ],
        "TF_NUM_INTEROP_THREADS": FORMAL_EVALUATION_DEVICE_REQUIREMENT[
            "tf_num_interop_threads"
        ],
        "OMP_NUM_THREADS": FORMAL_EVALUATION_DEVICE_REQUIREMENT[
            "omp_num_threads"
        ],
    }
    if (
        evaluation_device.get("requirement")
        != FORMAL_EVALUATION_DEVICE_REQUIREMENT
        or evaluation_device.get("environment") != expected_eval_environment
        or evaluation_device.get("physical_gpus") != []
        or evaluation_device.get("logical_gpus") != []
    ):
        raise PaperSchemaError(
            f"official evaluation device provenance mismatch: {manifest_path}"
        )

    recorded_bank = dict(manifest.get("task_bank") or {})
    actual_bank = fingerprint_artifact(recorded_bank.get("path"))
    if actual_bank.get("sha256") != recorded_bank.get("sha256"):
        raise PaperSchemaError(f"official task bank changed: {manifest_path}")
    if recorded_bank.get("sha256") != protocol["task_bank"].get("sha256"):
        raise PaperSchemaError(f"official task bank artifact mismatch: {manifest_path}")
    if recorded_bank.get("bank_content_sha256") != bank.get("content_sha256"):
        raise PaperSchemaError(f"official task bank content mismatch: {manifest_path}")
    if recorded_bank.get("split") != split or recorded_bank.get(
        "split_role"
    ) != expected_role:
        raise PaperSchemaError(f"official task bank split mismatch: {manifest_path}")

    checkpoint = dict(manifest.get("checkpoint") or {})
    actual_checkpoint = fingerprint_artifact(checkpoint.get("path"))
    if actual_checkpoint.get("sha256") != checkpoint.get("sha256"):
        raise PaperSchemaError(f"official checkpoint changed: {manifest_path}")
    training_metadata = dict(checkpoint.get("training_checkpoint_metadata") or {})
    if training_metadata.get("schema_version") != "paper_training_checkpoint_v5":
        raise PaperSchemaError(f"official checkpoint metadata mismatch: {manifest_path}")
    try:
        checkpoint_force_ratio = float(
            training_metadata["checkpoint_action_force_ratio"]
        )
        checkpoint_force_ratio_episode = int(
            training_metadata["checkpoint_action_force_ratio_episode"]
        )
        completed_training_iterations = int(
            training_metadata["completed_training_iterations"]
        )
        evaluated_force_ratio = float(
            runtime_environment[
                "action_force_ratio"
            ]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PaperSchemaError(
            f"official checkpoint lacks force-ratio provenance: {manifest_path}"
        ) from exc
    if (
        not math.isfinite(checkpoint_force_ratio)
        or not 0.0 <= checkpoint_force_ratio <= 1.0
        or checkpoint_force_ratio_episode != completed_training_iterations
        or not math.isclose(
            checkpoint_force_ratio,
            evaluated_force_ratio,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or runtime_environment.get(
            "force_action_force_ratio"
        ) is not None
    ):
        raise PaperSchemaError(
            f"official checkpoint/evaluation force ratio mismatch: {manifest_path}"
        )
    if int(training_metadata.get("train_seed", -1)) != int(train_seed):
        raise PaperSchemaError(f"official checkpoint train seed mismatch: {manifest_path}")
    if training_metadata.get("paper_method_id") != method:
        raise PaperSchemaError(f"official checkpoint method mismatch: {manifest_path}")
    if training_metadata.get("experiment_protocol_content_sha256") != protocol.get(
        "content_sha256"
    ):
        raise PaperSchemaError(f"official checkpoint protocol mismatch: {manifest_path}")

    selection = dict(manifest.get("checkpoint_selection") or {})
    if not selection.get("sha256"):
        raise PaperSchemaError(f"official run lacks selection provenance: {manifest_path}")
    actual_selection = fingerprint_artifact(selection.get("path"))
    if actual_selection.get("sha256") != selection.get("sha256"):
        raise PaperSchemaError(f"official selection artifact changed: {manifest_path}")
    selected_payload = validate_selection_result(_load_json(selection["path"]))
    if selected_payload.get("experiment_protocol_content_sha256") != protocol.get(
        "content_sha256"
    ):
        raise PaperSchemaError(f"official selection protocol mismatch: {manifest_path}")
    if bool(selected_payload.get("adequacy_required_for_selection")) != bool(
        protocol["methods"][method]["require_validation_adequacy"]
    ):
        raise PaperSchemaError(f"official selection adequacy policy mismatch: {manifest_path}")
    if int(selected_payload.get("train_seed", -1)) != int(train_seed):
        raise PaperSchemaError(f"official selection train seed mismatch: {manifest_path}")
    if selected_payload["selected_checkpoint"].get("sha256") != checkpoint.get(
        "sha256"
    ):
        raise PaperSchemaError(f"official checkpoint was not selected: {manifest_path}")
    selected_difficulty = str(training_metadata.get("trained_on_difficulty_bin", ""))
    expected_validation_split = f"validation_{selected_difficulty}"
    if selected_payload.get("validation_split") != expected_validation_split:
        raise PaperSchemaError(f"official selection difficulty mismatch: {manifest_path}")
    if expected_role == "test_in_distribution" and split != (
        f"test_{selected_difficulty}"
    ):
        raise PaperSchemaError(f"official ID difficulty mismatch: {manifest_path}")
    if expected_role == "test_generalization" and selected_difficulty != "C3":
        raise PaperSchemaError(f"official G1 checkpoint is not C3: {manifest_path}")

    for artifact in manifest.get("source_artifacts") or []:
        actual = fingerprint_artifact(artifact.get("path"))
        if actual.get("sha256") != artifact.get("sha256"):
            raise PaperSchemaError(f"official evaluation source changed: {manifest_path}")
    for artifact_name, artifact in dict(manifest.get("outputs") or {}).items():
        actual = fingerprint_artifact(artifact.get("path"))
        if actual.get("sha256") != artifact.get("sha256"):
            raise PaperSchemaError(
                f"official output changed: {manifest_path} output={artifact_name}"
            )
    collection_artifact = dict(manifest["outputs"].get("episode_records_json") or {})
    if not collection_artifact.get("path"):
        raise PaperSchemaError(f"official run lacks episode records: {manifest_path}")
    collection = _validate_hashed(
        _load_json(collection_artifact["path"]),
        str(collection_artifact["path"]),
    )
    if collection.get("schema_version") != PAPER_EPISODE_COLLECTION_SCHEMA_VERSION:
        raise PaperSchemaError("official episode collection schema mismatch")
    records = [validate_episode_record(record) for record in collection.get("records") or []]
    if int(collection.get("record_count", -1)) != len(records) or not records:
        raise PaperSchemaError("official episode collection count mismatch")
    expected_task_count = int(manifest["task_bank"].get("split_task_count", -1))
    expected_entries = list(split_cfg.get("tasks") or [])
    expected_task_ids = [str(entry.get("task_id", "")) for entry in expected_entries]
    if expected_task_count != len(expected_entries) or len(records) != expected_task_count:
        raise PaperSchemaError("official run does not contain one record per task")
    actual_task_ids = [record["task_id"] for record in records]
    if actual_task_ids != expected_task_ids:
        raise PaperSchemaError("official run task IDs do not match immutable split")
    if recorded_bank.get("split_task_ids_sha256") != content_sha256(
        expected_task_ids
    ):
        raise PaperSchemaError("official task ID hash mismatch")
    for episode_index, (record, entry) in enumerate(zip(records, expected_entries)):
        if record["run_id"] != manifest["run_id"]:
            raise PaperSchemaError("official episode run ID mismatch")
        if record["algorithm"] != method or record["split"] != split:
            raise PaperSchemaError("official episode method/split mismatch")
        if int(record["train_seed"]) != int(train_seed) or int(
            record["eval_seed"]
        ) != int(protocol["official_test"]["eval_seed"]):
            raise PaperSchemaError("official episode seed mismatch")
        if int(record["episode_index"]) != episode_index:
            raise PaperSchemaError("official episode index/order mismatch")
        if record["difficulty_bin"] != entry.get("difficulty_bin"):
            raise PaperSchemaError("official episode difficulty mismatch")
        deadline = dict(entry.get("deadline") or {})
        if int(record["deadline_steps"]) != int(deadline.get("deadline_steps", -1)):
            raise PaperSchemaError("official episode deadline mismatch")
    return {"manifest": manifest, "records": records}


def _episode_metric(record: Mapping[str, Any], metric: str) -> float:
    if metric == "team_success_rate":
        return float(bool(record["team_success"]))
    if metric == "collision_rate":
        return float(not bool(record["collision_free"]))
    agents = list(record["agents"])
    field = {
        "agent_spl": "spl",
        "minimum_clearance_m": "minimum_clearance_m",
        "energy": "energy",
        "control_effort": "control_effort",
    }[metric]
    return float(np.mean([float(agent[field]) for agent in agents]))


def _episode_mechanism_metric(record: Mapping[str, Any], metric: str) -> float:
    if metric not in _MECHANISM_METRICS:
        raise PaperSchemaError(f"unsupported mechanism metric={metric!r}")
    diagnostics = record.get("apf_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise PaperSchemaError("official episode lacks apf_diagnostics")
    try:
        value = float(diagnostics[metric])
    except (KeyError, TypeError, ValueError) as exc:
        raise PaperSchemaError(
            f"invalid episode mechanism metric={metric!r}"
        ) from exc
    if not np.isfinite(value):
        raise PaperSchemaError(
            f"non-finite episode mechanism metric={metric!r}"
        )
    return value


def _ci(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise PaperSchemaError("bootstrap produced invalid values")
    return {
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
        "confidence": 0.95,
    }


def _hierarchical_bootstrap(
    values: np.ndarray,
    *,
    bootstrap_samples: int,
    random_seed: int,
    better_direction: str = "higher",
) -> Dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise PaperSchemaError(
            f"hierarchical bootstrap requires seed x task matrix, got={values.shape}"
        )
    if not np.isfinite(values).all():
        raise PaperSchemaError("metric matrix contains non-finite values")
    if better_direction not in {"higher", "lower"}:
        raise PaperSchemaError(
            f"invalid metric better direction={better_direction!r}"
        )
    rng = np.random.default_rng(int(random_seed))
    seed_count, task_count = values.shape
    replicates = np.empty(int(bootstrap_samples), dtype=np.float64)
    for index in range(int(bootstrap_samples)):
        # Train seeds and immutable tasks are crossed experimental factors:
        # every seed is evaluated on the same task IDs.  Resample each axis
        # once per replicate so task pairing remains intact across seeds.
        sampled_seed_indices = rng.integers(0, seed_count, size=seed_count)
        sampled_tasks = rng.integers(0, task_count, size=task_count)
        replicates[index] = float(
            np.mean(values[np.ix_(sampled_seed_indices, sampled_tasks)])
        )
    per_seed = np.mean(values, axis=1)
    return {
        "estimate": float(np.mean(values)),
        "train_seed_count": int(seed_count),
        "task_count_per_seed": int(task_count),
        "per_train_seed_mean": [float(value) for value in per_seed],
        "between_seed_sample_std": float(np.std(per_seed, ddof=1)),
        "metric_better_direction": better_direction,
        "bootstrap_design": "crossed_train_seed_by_immutable_task",
        "hierarchical_bootstrap_95ci": _ci(replicates),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_random_seed": int(random_seed),
    }


def _descriptive_hierarchical_bootstrap(
    values: np.ndarray,
    *,
    bootstrap_samples: int,
    random_seed: int,
) -> Dict[str, Any]:
    result = _hierarchical_bootstrap(
        values,
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed,
        better_direction="higher",
    )
    result.pop("metric_better_direction", None)
    result["metric_interpretation"] = "descriptive_not_superiority"
    return result


def _paired_hierarchical_difference(
    proposed: np.ndarray,
    baseline: np.ndarray,
    *,
    bootstrap_samples: int,
    random_seed: int,
    better_direction: str = "higher",
) -> Dict[str, Any]:
    proposed = np.asarray(proposed, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if proposed.shape != baseline.shape or proposed.ndim != 2:
        raise PaperSchemaError(
            f"paired metric matrices differ: {proposed.shape} vs {baseline.shape}"
        )
    if proposed.shape[0] < 2 or proposed.shape[1] < 1:
        raise PaperSchemaError(
            f"paired bootstrap requires at least two seeds and one task, got={proposed.shape}"
        )
    if not np.isfinite(proposed).all() or not np.isfinite(baseline).all():
        raise PaperSchemaError("paired metric matrix contains non-finite values")
    if better_direction not in {"higher", "lower"}:
        raise PaperSchemaError(
            f"invalid metric better direction={better_direction!r}"
        )
    rng = np.random.default_rng(int(random_seed))
    seed_count, task_count = proposed.shape
    replicates = np.empty(int(bootstrap_samples), dtype=np.float64)
    delta = proposed - baseline
    for index in range(int(bootstrap_samples)):
        sampled_seed_indices = rng.integers(0, seed_count, size=seed_count)
        sampled_tasks = rng.integers(0, task_count, size=task_count)
        replicates[index] = float(
            np.mean(delta[np.ix_(sampled_seed_indices, sampled_tasks)])
        )
    raw_interval = _ci(replicates)
    orientation = 1.0 if better_direction == "higher" else -1.0
    oriented_replicates = orientation * replicates
    return {
        "estimate_proposed_minus_baseline": float(np.mean(delta)),
        "paired_seed_test": paired_seed_sign_flip(proposed, baseline, random_seed=random_seed),
        "paired_hierarchical_bootstrap_95ci": raw_interval,
        "bootstrap_probability_difference_gt_zero": float(
            np.mean(replicates > 0.0)
        ),
        "metric_better_direction": better_direction,
        "bootstrap_design": "paired_crossed_train_seed_by_immutable_task",
        "estimate_oriented_improvement": float(
            orientation * np.mean(delta)
        ),
        "oriented_improvement_bootstrap_95ci": _ci(
            oriented_replicates
        ),
        "bootstrap_probability_oriented_improvement_gt_zero": float(
            np.mean(oriented_replicates > 0.0)
        ),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_random_seed": int(random_seed),
    }


def _terrain_crossed_bootstrap_replicates(
    values: np.ndarray,
    *,
    bootstrap_samples: int,
    random_seed: int,
) -> np.ndarray:
    """Bootstrap seed x terrain x task data without breaking task pairing."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 2 or values.shape[2] < 1:
        raise PaperSchemaError(
            "terrain bootstrap requires train_seed x terrain x task data, "
            f"got={values.shape}"
        )
    if not np.isfinite(values).all():
        raise PaperSchemaError("terrain metric tensor contains non-finite values")
    rng = np.random.default_rng(int(random_seed))
    seed_count, terrain_count, task_count = values.shape
    replicates = np.empty(int(bootstrap_samples), dtype=np.float64)
    for replicate_index in range(int(bootstrap_samples)):
        sampled_seeds = rng.integers(0, seed_count, size=seed_count)
        sampled_terrains = rng.integers(0, terrain_count, size=terrain_count)
        terrain_means = []
        for terrain_index in sampled_terrains:
            # Tasks are nested within a terrain but crossed with train seeds.
            # One task draw is therefore shared by all sampled train seeds.
            sampled_tasks = rng.integers(0, task_count, size=task_count)
            terrain_means.append(
                float(
                    np.mean(
                        values[
                            np.ix_(
                                sampled_seeds,
                                np.asarray([terrain_index]),
                                sampled_tasks,
                            )
                        ]
                    )
                )
            )
        replicates[replicate_index] = float(np.mean(terrain_means))
    return replicates


def _terrain_crossed_bootstrap(
    values: np.ndarray,
    *,
    bootstrap_samples: int,
    random_seed: int,
    better_direction: str,
) -> Dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if better_direction not in {"higher", "lower"}:
        raise PaperSchemaError(
            f"invalid metric better direction={better_direction!r}"
        )
    replicates = _terrain_crossed_bootstrap_replicates(
        values,
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed,
    )
    return {
        "estimate": float(np.mean(values)),
        "train_seed_count": int(values.shape[0]),
        "held_out_terrain_count": int(values.shape[1]),
        "task_count_per_terrain": int(values.shape[2]),
        "per_train_seed_mean": [
            float(value) for value in np.mean(values, axis=(1, 2))
        ],
        "per_held_out_terrain_mean": [
            float(value) for value in np.mean(values, axis=(0, 2))
        ],
        "between_train_seed_sample_std": float(
            np.std(np.mean(values, axis=(1, 2)), ddof=1)
        ),
        "between_held_out_terrain_sample_std": float(
            np.std(np.mean(values, axis=(0, 2)), ddof=1)
        ),
        "metric_better_direction": better_direction,
        "bootstrap_design": (
            "train_seed_crossed_with_held_out_terrain_with_tasks_nested_in_terrain"
        ),
        "terrain_crossed_bootstrap_95ci": _ci(replicates),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_random_seed": int(random_seed),
    }


def _descriptive_terrain_crossed_bootstrap(
    values: np.ndarray,
    *,
    bootstrap_samples: int,
    random_seed: int,
) -> Dict[str, Any]:
    result = _terrain_crossed_bootstrap(
        values,
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed,
        better_direction="higher",
    )
    result.pop("metric_better_direction", None)
    result["metric_interpretation"] = "descriptive_not_superiority"
    return result


def _paired_terrain_crossed_difference(
    proposed: np.ndarray,
    baseline: np.ndarray,
    *,
    bootstrap_samples: int,
    random_seed: int,
    better_direction: str,
) -> Dict[str, Any]:
    proposed = np.asarray(proposed, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if proposed.shape != baseline.shape or proposed.ndim != 3:
        raise PaperSchemaError(
            "paired terrain metric tensors differ: "
            f"{proposed.shape} vs {baseline.shape}"
        )
    if not np.isfinite(proposed).all() or not np.isfinite(baseline).all():
        raise PaperSchemaError(
            "paired terrain metric tensor contains non-finite values"
        )
    if better_direction not in {"higher", "lower"}:
        raise PaperSchemaError(
            f"invalid metric better direction={better_direction!r}"
        )
    delta = proposed - baseline
    replicates = _terrain_crossed_bootstrap_replicates(
        delta,
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed,
    )
    orientation = 1.0 if better_direction == "higher" else -1.0
    oriented_replicates = orientation * replicates
    return {
        "estimate_proposed_minus_baseline": float(np.mean(delta)),
        "paired_seed_test": paired_seed_sign_flip(proposed, baseline, random_seed=random_seed),
        "paired_terrain_crossed_bootstrap_95ci": _ci(replicates),
        "bootstrap_probability_difference_gt_zero": float(
            np.mean(replicates > 0.0)
        ),
        "metric_better_direction": better_direction,
        "estimate_oriented_improvement": float(
            orientation * np.mean(delta)
        ),
        "oriented_improvement_bootstrap_95ci": _ci(oriented_replicates),
        "bootstrap_probability_oriented_improvement_gt_zero": float(
            np.mean(oriented_replicates > 0.0)
        ),
        "bootstrap_design": (
            "paired_train_seed_crossed_with_held_out_terrain_with_tasks_nested_in_terrain"
        ),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_random_seed": int(random_seed),
    }


def summarize_results(
    *,
    protocol_path: str | Path,
    experiment_root: str | Path,
    splits: Sequence[str],
    output_dir: str | Path,
    proposed_method: str = "EC-MATD3-APF",
    bootstrap_samples: int = 5000,
    bootstrap_random_seed: int = 424242,
) -> Dict[str, Any]:
    protocol_file = Path(protocol_path).expanduser().resolve()
    protocol = load_protocol(protocol_file)
    protocol_artifact = fingerprint_artifact(protocol_file)
    root = Path(experiment_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise PaperSchemaError(
            f"refusing to overwrite non-empty result summary directory: {destination}"
        )
    if int(bootstrap_samples) < 1000:
        raise PaperSchemaError("paper summary requires at least 1000 bootstrap samples")
    methods = list(protocol["methods"])
    if proposed_method not in methods:
        raise PaperSchemaError("proposed method is absent from protocol")
    train_seeds = [int(seed) for seed in protocol["train_seeds"]]
    bank = load_task_bank(protocol["task_bank"]["manifest_path"])
    requested_splits = [str(split).strip() for split in splits]
    if not requested_splits or len(requested_splits) != len(set(requested_splits)):
        raise PaperSchemaError("summary splits must be unique and non-empty")
    expected_official_splits = [
        f"test_{difficulty}" for difficulty in ("C0", "C1", "C2", "C3")
    ] + list(
        (protocol.get("split_contract") or {}).get(
            "generalization_splits"
        )
        or []
    )
    if requested_splits != expected_official_splits:
        raise PaperSchemaError(
            "paper summary must include every official split in canonical order: "
            f"expected={expected_official_splits}, requested={requested_splits}"
        )
    for split in requested_splits:
        role = str((bank["splits"].get(split) or {}).get("role", ""))
        if role not in {"test_in_distribution", "test_generalization"}:
            raise PaperSchemaError(f"summary split is not an official test split: {split}")
    expected_generalization_splits = list(
        (protocol.get("split_contract") or {}).get("generalization_splits") or []
    )
    requested_generalization_splits = [
        split
        for split in requested_splits
        if str(bank["splits"][split].get("role", "")) == "test_generalization"
    ]
    if requested_generalization_splits and set(
        requested_generalization_splits
    ) != set(expected_generalization_splits):
        raise PaperSchemaError(
            "generalization reporting must include all frozen held-out terrain "
            f"splits: expected={expected_generalization_splits}, "
            f"requested={requested_generalization_splits}"
        )

    run_artifacts = []
    matrices: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    task_ids_by_split: Dict[str, list[str]] = {}
    for split in requested_splits:
        matrices[split] = {}
        for method in methods:
            per_seed_records = []
            for train_seed in train_seeds:
                manifest_path = (
                    root
                    / "official_test"
                    / method.replace("/", "_")
                    / f"seed_{train_seed}"
                    / split
                    / "paper_run_manifest.json"
                )
                if not manifest_path.is_file():
                    raise PaperSchemaError(
                        f"missing official run manifest: {manifest_path}"
                    )
                loaded = _load_official_run(
                    manifest_path,
                    protocol=protocol,
                    protocol_artifact=protocol_artifact,
                    bank=bank,
                    method=method,
                    train_seed=train_seed,
                    split=split,
                )
                manifest = loaded["manifest"]
                per_seed_records.append(loaded["records"])
                run_artifacts.append(fingerprint_artifact(manifest_path))
            reference_ids = [record["task_id"] for record in per_seed_records[0]]
            for records in per_seed_records[1:]:
                if [record["task_id"] for record in records] != reference_ids:
                    raise PaperSchemaError(
                        f"task pairing mismatch across seeds for {method}/{split}"
                    )
            if split in task_ids_by_split and task_ids_by_split[split] != reference_ids:
                raise PaperSchemaError(
                    f"task pairing mismatch across methods for split={split}"
                )
            task_ids_by_split[split] = reference_ids
            matrices[split][method] = {
                metric: np.asarray(
                    [
                        [_episode_metric(record, metric) for record in records]
                        for records in per_seed_records
                    ],
                    dtype=np.float64,
                )
                for metric in _METRICS
            }
            matrices[split][method]["mechanism"] = {
                metric: np.asarray(
                    [
                        [
                            _episode_mechanism_metric(record, metric)
                            for record in records
                        ]
                        for records in per_seed_records
                    ],
                    dtype=np.float64,
                )
                for metric in _MECHANISM_METRICS
            }

    summaries: Dict[str, Any] = {}
    mechanism_summaries: Dict[str, Any] = {}
    comparisons: Dict[str, Any] = {}
    for split_index, split in enumerate(requested_splits):
        summaries[split] = {}
        mechanism_summaries[split] = {}
        for method_index, method in enumerate(methods):
            summaries[split][method] = {
                metric: _hierarchical_bootstrap(
                    matrices[split][method][metric],
                    bootstrap_samples=bootstrap_samples,
                    random_seed=(
                        int(bootstrap_random_seed)
                        + split_index * 10000
                        + method_index * 100
                        + metric_index
                    ),
                    better_direction=_METRIC_BETTER_DIRECTION[metric],
                )
                for metric_index, metric in enumerate(_METRICS)
            }
            mechanism_summaries[split][method] = {
                metric: _descriptive_hierarchical_bootstrap(
                    matrices[split][method]["mechanism"][metric],
                    bootstrap_samples=bootstrap_samples,
                    random_seed=(
                        int(bootstrap_random_seed)
                        + 750000
                        + split_index * 10000
                        + method_index * 100
                        + metric_index
                    ),
                )
                for metric_index, metric in enumerate(_MECHANISM_METRICS)
            }
        comparisons[split] = {}
        for baseline_index, baseline in enumerate(
            method for method in methods if method != proposed_method
        ):
            comparisons[split][baseline] = {
                metric: _paired_hierarchical_difference(
                    matrices[split][proposed_method][metric],
                    matrices[split][baseline][metric],
                    bootstrap_samples=bootstrap_samples,
                    random_seed=(
                        int(bootstrap_random_seed)
                        + 500000
                        + split_index * 10000
                        + baseline_index * 100
                        + metric_index
                    ),
                    better_direction=_METRIC_BETTER_DIRECTION[metric],
                )
                for metric_index, metric in enumerate(_METRICS)
            }

    generalization_aggregate = None
    if requested_generalization_splits:
        ordered_g1_splits = list(expected_generalization_splits)
        terrain_seeds = [
            int(bank["splits"][split]["runtime_config"]["terrain_seed"])
            for split in ordered_g1_splits
        ]
        if len(terrain_seeds) != len(set(terrain_seeds)):
            raise PaperSchemaError(
                "generalization aggregate requires unique held-out terrain seeds"
            )
        generalization_tensors = {
            method: {
                metric: np.stack(
                    [
                        matrices[split][method][metric]
                        for split in ordered_g1_splits
                    ],
                    axis=1,
                )
                for metric in _METRICS
            }
            for method in methods
        }
        generalization_mechanism_tensors = {
            method: {
                metric: np.stack(
                    [
                        matrices[split][method]["mechanism"][metric]
                        for split in ordered_g1_splits
                    ],
                    axis=1,
                )
                for metric in _MECHANISM_METRICS
            }
            for method in methods
        }
        aggregate_summaries = {}
        aggregate_mechanism_summaries = {}
        aggregate_comparisons = {}
        for method_index, method in enumerate(methods):
            aggregate_summaries[method] = {
                metric: _terrain_crossed_bootstrap(
                    generalization_tensors[method][metric],
                    bootstrap_samples=bootstrap_samples,
                    random_seed=(
                        int(bootstrap_random_seed)
                        + 1_000_000
                        + method_index * 100
                        + metric_index
                    ),
                    better_direction=_METRIC_BETTER_DIRECTION[metric],
                )
                for metric_index, metric in enumerate(_METRICS)
            }
            aggregate_mechanism_summaries[method] = {
                metric: _descriptive_terrain_crossed_bootstrap(
                    generalization_mechanism_tensors[method][metric],
                    bootstrap_samples=bootstrap_samples,
                    random_seed=(
                        int(bootstrap_random_seed)
                        + 1_250_000
                        + method_index * 100
                        + metric_index
                    ),
                )
                for metric_index, metric in enumerate(_MECHANISM_METRICS)
            }
        for baseline_index, baseline in enumerate(
            method for method in methods if method != proposed_method
        ):
            aggregate_comparisons[baseline] = {
                metric: _paired_terrain_crossed_difference(
                    generalization_tensors[proposed_method][metric],
                    generalization_tensors[baseline][metric],
                    bootstrap_samples=bootstrap_samples,
                    random_seed=(
                        int(bootstrap_random_seed)
                        + 1_500_000
                        + baseline_index * 100
                        + metric_index
                    ),
                    better_direction=_METRIC_BETTER_DIRECTION[metric],
                )
                for metric_index, metric in enumerate(_METRICS)
            }
        generalization_aggregate = {
            "group_id": "G1_held_out_terrain_aggregate",
            "splits": ordered_g1_splits,
            "held_out_terrain_seeds": terrain_seeds,
            "task_count_per_terrain": int(
                generalization_tensors[methods[0]][_METRICS[0]].shape[2]
            ),
            "statistical_unit_contract": {
                "independent_training_unit": "train_seed",
                "independent_generalization_domain_unit": "held_out_terrain_seed",
                "task_unit": "immutable_task_id_nested_within_terrain",
                "pairing": "same_train_seed_same_terrain_same_task_across_methods",
                "bootstrap": (
                    "resample_train_seeds_and_terrains_then_tasks_within_each_terrain"
                ),
                "confidence": 0.95,
            },
            "summaries": aggregate_summaries,
            "mechanism_summaries": aggregate_mechanism_summaries,
            "paired_comparisons": aggregate_comparisons,
        }

    exploratory_seed_test_analysis = _adjust_exploratory_seed_tests(
        comparisons, generalization_aggregate
    )

    payload: Dict[str, Any] = {
        "schema_version": PAPER_RESULTS_SUMMARY_SCHEMA_VERSION,
        "protocol": protocol_artifact,
        "protocol_content_sha256": protocol["content_sha256"],
        "experiment_root": str(root),
        "methods": methods,
        "proposed_method": proposed_method,
        "train_seeds": train_seeds,
        "splits": requested_splits,
        "metrics": list(_METRICS),
        "mechanism_metrics": list(_MECHANISM_METRICS),
        "metric_better_direction": dict(_METRIC_BETTER_DIRECTION),
        "statistical_unit_contract": {
            "independent_unit": "train_seed",
            "crossed_unit": "immutable_task_id",
            "paired_unit": "same_train_seed_same_task_across_methods",
            "bootstrap": (
                "resample_train_seeds_and_immutable_tasks_as_crossed_factors"
            ),
            "confidence": 0.95,
        },
        "official_run_manifests": run_artifacts,
        "task_ids_sha256_by_split": {
            split: content_sha256(task_ids)
            for split, task_ids in task_ids_by_split.items()
        },
        "summaries": summaries,
        "mechanism_summaries": mechanism_summaries,
        "paired_comparisons": comparisons,
        "generalization_aggregate": generalization_aggregate,
        "exploratory_seed_test_analysis": exploratory_seed_test_analysis,
        "confirmatory_analysis": _build_confirmatory_result(
            dict((protocol.get("reporting") or {}).get(
                "confirmatory_analysis"
            ) or {}),
            comparisons,
            proposed_method=proposed_method,
        ),
        "secondary_analysis_policy": (
            protocol["reporting"]["secondary_analysis_policy"]
        ),
    }
    repo_root = Path(__file__).resolve().parent
    payload["summary_source_artifacts"] = [
        {
            **fingerprint_artifact(path),
            "logical_name": str(path.relative_to(repo_root)),
        }
        for path in (
            Path(__file__).resolve(),
            repo_root / "paper_seed_statistics.py",
            repo_root / "paper_experiment_schema.py",
            repo_root / "paper_experiment_protocol.py",
            repo_root / "paper_checkpoint_selection.py",
            repo_root / "paper_task_bank.py",
        )
    ]
    payload["content_sha256"] = content_sha256(payload)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "paper_results_summary.json"
    csv_path = destination / "paper_results_summary.csv"
    paired_csv_path = destination / "paper_paired_comparisons.csv"
    mechanism_csv_path = destination / "paper_apf_mechanism_diagnostics.csv"
    atomic_write_json(json_path, payload)
    rows = []
    for split in requested_splits:
        for method in methods:
            for metric in _METRICS:
                summary = summaries[split][method][metric]
                interval = summary["hierarchical_bootstrap_95ci"]
                rows.append(
                    {
                        "split": split,
                        "method": method,
                        "metric": metric,
                        "estimate": summary["estimate"],
                        "ci95_lower": interval["lower"],
                        "ci95_upper": interval["upper"],
                        "train_seed_count": summary["train_seed_count"],
                        "task_count_per_seed": summary["task_count_per_seed"],
                        "between_seed_sample_std": summary[
                            "between_seed_sample_std"
                        ],
                    }
                )
    if generalization_aggregate is not None:
        for method in methods:
            for metric in _METRICS:
                summary = generalization_aggregate["summaries"][method][metric]
                interval = summary["terrain_crossed_bootstrap_95ci"]
                rows.append(
                    {
                        "split": generalization_aggregate["group_id"],
                        "method": method,
                        "metric": metric,
                        "estimate": summary["estimate"],
                        "ci95_lower": interval["lower"],
                        "ci95_upper": interval["upper"],
                        "train_seed_count": summary["train_seed_count"],
                        "task_count_per_seed": (
                            summary["held_out_terrain_count"]
                            * summary["task_count_per_terrain"]
                        ),
                        "between_seed_sample_std": summary[
                            "between_train_seed_sample_std"
                        ],
                    }
                )
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mechanism_rows = []
    for split in requested_splits:
        for method in methods:
            for metric in _MECHANISM_METRICS:
                summary = mechanism_summaries[split][method][metric]
                interval = summary["hierarchical_bootstrap_95ci"]
                mechanism_rows.append(
                    {
                        "split": split,
                        "method": method,
                        "metric": metric,
                        "estimate": summary["estimate"],
                        "ci95_lower": interval["lower"],
                        "ci95_upper": interval["upper"],
                        "train_seed_count": summary["train_seed_count"],
                        "task_count_per_seed": summary["task_count_per_seed"],
                        "between_seed_sample_std": summary[
                            "between_seed_sample_std"
                        ],
                        "metric_interpretation": summary[
                            "metric_interpretation"
                        ],
                    }
                )
    if generalization_aggregate is not None:
        for method in methods:
            for metric in _MECHANISM_METRICS:
                summary = generalization_aggregate[
                    "mechanism_summaries"
                ][method][metric]
                interval = summary["terrain_crossed_bootstrap_95ci"]
                mechanism_rows.append(
                    {
                        "split": generalization_aggregate["group_id"],
                        "method": method,
                        "metric": metric,
                        "estimate": summary["estimate"],
                        "ci95_lower": interval["lower"],
                        "ci95_upper": interval["upper"],
                        "train_seed_count": summary["train_seed_count"],
                        "task_count_per_seed": (
                            summary["held_out_terrain_count"]
                            * summary["task_count_per_terrain"]
                        ),
                        "between_seed_sample_std": summary[
                            "between_train_seed_sample_std"
                        ],
                        "metric_interpretation": summary[
                            "metric_interpretation"
                        ],
                    }
                )
    with mechanism_csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(mechanism_rows[0])
        )
        writer.writeheader()
        writer.writerows(mechanism_rows)
    paired_rows = []
    for split in requested_splits:
        for baseline, metric_results in comparisons[split].items():
            for metric in _METRICS:
                comparison = metric_results[metric]
                interval = comparison["paired_hierarchical_bootstrap_95ci"]
                better_direction = comparison["metric_better_direction"]
                raw_delta = comparison["estimate_proposed_minus_baseline"]
                oriented_interval = comparison[
                    "oriented_improvement_bootstrap_95ci"
                ]
                paired_rows.append(
                    {
                        "split": split,
                        "proposed_method": proposed_method,
                        **_seed_test_csv_fields(comparison),
                        "baseline_method": baseline,
                        "metric": metric,
                        "better_direction": better_direction,
                        "proposed_minus_baseline": raw_delta,
                        "oriented_improvement": comparison[
                            "estimate_oriented_improvement"
                        ],
                        "ci95_lower_raw_difference": interval["lower"],
                        "ci95_upper_raw_difference": interval["upper"],
                        "bootstrap_probability_raw_difference_gt_zero": comparison[
                            "bootstrap_probability_difference_gt_zero"
                        ],
                        "ci95_lower_oriented_improvement": oriented_interval[
                            "lower"
                        ],
                        "ci95_upper_oriented_improvement": oriented_interval[
                            "upper"
                        ],
                        "bootstrap_probability_oriented_improvement_gt_zero": comparison[
                            "bootstrap_probability_oriented_improvement_gt_zero"
                        ],
                    }
                )
    if generalization_aggregate is not None:
        for baseline, metric_results in generalization_aggregate[
            "paired_comparisons"
        ].items():
            for metric in _METRICS:
                comparison = metric_results[metric]
                interval = comparison[
                    "paired_terrain_crossed_bootstrap_95ci"
                ]
                oriented_interval = comparison[
                    "oriented_improvement_bootstrap_95ci"
                ]
                paired_rows.append(
                    {
                        "split": generalization_aggregate["group_id"],
                        "proposed_method": proposed_method,
                        **_seed_test_csv_fields(comparison),
                        "baseline_method": baseline,
                        "metric": metric,
                        "better_direction": comparison[
                            "metric_better_direction"
                        ],
                        "proposed_minus_baseline": comparison[
                            "estimate_proposed_minus_baseline"
                        ],
                        "oriented_improvement": comparison[
                            "estimate_oriented_improvement"
                        ],
                        "ci95_lower_raw_difference": interval["lower"],
                        "ci95_upper_raw_difference": interval["upper"],
                        "bootstrap_probability_raw_difference_gt_zero": comparison[
                            "bootstrap_probability_difference_gt_zero"
                        ],
                        "ci95_lower_oriented_improvement": oriented_interval[
                            "lower"
                        ],
                        "ci95_upper_oriented_improvement": oriented_interval[
                            "upper"
                        ],
                        "bootstrap_probability_oriented_improvement_gt_zero": comparison[
                            "bootstrap_probability_oriented_improvement_gt_zero"
                        ],
                    }
                )
    with paired_csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    print(f"paper result summary written: {json_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--proposed-method", default="EC-MATD3-APF")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-random-seed", type=int, default=424242)
    args = parser.parse_args()
    summarize_results(
        protocol_path=args.protocol,
        experiment_root=args.experiment_root,
        splits=[value.strip() for value in args.splits.split(",") if value.strip()],
        output_dir=args.output_dir,
        proposed_method=args.proposed_method,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_random_seed=args.bootstrap_random_seed,
    )


if __name__ == "__main__":
    main()
