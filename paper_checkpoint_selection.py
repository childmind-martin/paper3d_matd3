#!/usr/bin/env python3
"""Validation-only checkpoint selection for paper experiments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from paper_experiment_schema import (
    PAPER_AGGREGATE_SCHEMA_VERSION,
    PAPER_RUN_MANIFEST_SCHEMA_VERSION,
    PaperSchemaError,
    atomic_write_json,
    content_sha256,
    fingerprint_artifact,
    wilson_interval,
)
from paper_experiment_protocol import (
    FORMAL_EVALUATION_DEVICE_REQUIREMENT,
    load_protocol,
)
from paper_curriculum import validation_checkpoint_gate


PAPER_CHECKPOINT_SELECTION_SCHEMA_VERSION = "paper_checkpoint_selection_v2"


def validate_run_manifest(payload: Mapping[str, Any]) -> Dict[str, Any]:
    if payload.get("schema_version") != PAPER_RUN_MANIFEST_SCHEMA_VERSION:
        raise PaperSchemaError("candidate is not a paper run manifest")
    expected = content_sha256({key: value for key, value in payload.items() if key != "content_sha256"})
    if payload.get("content_sha256") != expected:
        raise PaperSchemaError("paper run manifest content_sha256 mismatch")
    if payload.get("phase") != "validation" or payload.get("split_role") != "validation":
        raise PaperSchemaError("checkpoint candidates must come only from a validation split")
    return dict(payload)


def _load_validated_json(path: str | Path) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _candidate_from_manifest(label: str, manifest_path: str | Path) -> Dict[str, Any]:
    manifest = validate_run_manifest(_load_validated_json(manifest_path))
    for artifact_name, artifact in (manifest.get("outputs") or {}).items():
        actual_artifact = fingerprint_artifact(artifact.get("path"))
        if actual_artifact.get("sha256") != artifact.get("sha256"):
            raise PaperSchemaError(
                f"candidate={label} output={artifact_name} fingerprint mismatch"
            )
    for artifact in manifest.get("source_artifacts") or []:
        actual_artifact = fingerprint_artifact(artifact.get("path"))
        if actual_artifact.get("sha256") != artifact.get("sha256"):
            raise PaperSchemaError(
                f"candidate={label} evaluation source fingerprint mismatch"
            )
    protocol_artifact = dict(manifest.get("experiment_protocol") or {})
    actual_protocol_artifact = fingerprint_artifact(protocol_artifact.get("path"))
    if actual_protocol_artifact.get("sha256") != protocol_artifact.get("sha256"):
        raise PaperSchemaError(
            f"candidate={label} experiment protocol fingerprint mismatch"
        )
    protocol_content_sha256 = str(
        protocol_artifact.get("protocol_content_sha256", "")
    ).strip()
    if not protocol_content_sha256:
        raise PaperSchemaError(
            f"candidate={label} experiment protocol content hash is missing"
        )
    protocol = load_protocol(protocol_artifact.get("path"))
    if str(protocol.get("content_sha256", "")) != protocol_content_sha256:
        raise PaperSchemaError(
            f"candidate={label} experiment protocol content hash mismatch"
        )
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
            f"candidate={label} evaluation device provenance mismatch"
        )
    method_id = str(manifest.get("algorithm", "")).strip()
    method_spec = dict((protocol.get("methods") or {}).get(method_id) or {})
    if not method_spec:
        raise PaperSchemaError(
            f"candidate={label} method={method_id!r} is absent from protocol"
        )
    training_algorithm = str(method_spec.get("algorithm", "")).strip().lower()
    if not training_algorithm:
        raise PaperSchemaError(
            f"candidate={label} protocol method lacks a training algorithm"
        )
    aggregate_artifact = dict((manifest.get("outputs") or {}).get("aggregate_json") or {})
    aggregate_path = aggregate_artifact.get("path")
    if not aggregate_path:
        raise PaperSchemaError(f"candidate={label} manifest lacks aggregate_json output")
    actual_aggregate_artifact = fingerprint_artifact(aggregate_path)
    if actual_aggregate_artifact.get("sha256") != aggregate_artifact.get("sha256"):
        raise PaperSchemaError(f"candidate={label} aggregate artifact fingerprint mismatch")
    aggregate = _load_validated_json(aggregate_path)
    if aggregate.get("schema_version") != PAPER_AGGREGATE_SCHEMA_VERSION:
        raise PaperSchemaError(f"candidate={label} aggregate schema mismatch")
    expected_aggregate_hash = content_sha256(
        {key: value for key, value in aggregate.items() if key != "content_sha256"}
    )
    if aggregate.get("content_sha256") != expected_aggregate_hash:
        raise PaperSchemaError(f"candidate={label} aggregate content_sha256 mismatch")
    task_bank = dict(manifest.get("task_bank") or {})
    split_task_count = int(task_bank.get("split_task_count", -1))
    episodes = int(aggregate.get("episodes", -1))
    unique_tasks = int(aggregate.get("unique_tasks", -1))
    if split_task_count <= 0:
        raise PaperSchemaError(f"candidate={label} has invalid validation task count")
    if episodes != split_task_count or unique_tasks != split_task_count:
        raise PaperSchemaError(
            f"candidate={label} must evaluate every validation task exactly once: "
            f"episodes={episodes}, unique_tasks={unique_tasks}, "
            f"split_task_count={split_task_count}"
        )
    if str(task_bank.get("split", "")).strip() != str(manifest["split"]).strip():
        raise PaperSchemaError(f"candidate={label} task-bank split mismatch")
    if str(task_bank.get("split_role", "")).strip() != "validation":
        raise PaperSchemaError(f"candidate={label} task-bank role is not validation")
    if aggregate.get("train_seeds") != [int(manifest["train_seed"])]:
        raise PaperSchemaError(f"candidate={label} aggregate train seed mismatch")
    if aggregate.get("eval_seeds") != [int(manifest["eval_seed"])]:
        raise PaperSchemaError(f"candidate={label} aggregate eval seed mismatch")
    checkpoint = dict(manifest.get("checkpoint") or {})
    if not checkpoint.get("sha256"):
        raise PaperSchemaError(f"candidate={label} lacks checkpoint fingerprint")
    actual_checkpoint = fingerprint_artifact(checkpoint.get("path"))
    if actual_checkpoint.get("sha256") != checkpoint.get("sha256"):
        raise PaperSchemaError(f"candidate={label} checkpoint fingerprint mismatch")
    actual_task_bank = fingerprint_artifact(task_bank.get("path"))
    if actual_task_bank.get("sha256") != task_bank.get("sha256"):
        raise PaperSchemaError(f"candidate={label} task-bank fingerprint mismatch")
    training_metadata = dict(checkpoint.get("training_checkpoint_metadata") or {})
    if training_metadata.get("schema_version") != "paper_training_checkpoint_v5":
        raise PaperSchemaError(f"candidate={label} lacks paper training metadata")
    expected_training_hash = content_sha256(
        {key: value for key, value in training_metadata.items() if key != "content_sha256"}
    )
    if training_metadata.get("content_sha256") != expected_training_hash:
        raise PaperSchemaError(f"candidate={label} training metadata hash mismatch")
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
            f"candidate={label} lacks checkpoint-specific force-ratio provenance"
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
            f"candidate={label} checkpoint/evaluation force ratio mismatch"
        )
    if int(training_metadata.get("train_seed")) != int(manifest["train_seed"]):
        raise PaperSchemaError(f"candidate={label} train seed metadata mismatch")
    if str(training_metadata.get("algorithm", "")).strip().lower() != (
        training_algorithm
    ):
        raise PaperSchemaError(f"candidate={label} training algorithm mismatch")
    if str(training_metadata.get("scenario", "")).strip() != str(
        manifest["scenario"]
    ).strip():
        raise PaperSchemaError(f"candidate={label} training scenario mismatch")
    if str(training_metadata.get("paper_method_id", "")).strip() != str(
        manifest["algorithm"]
    ).strip():
        raise PaperSchemaError(f"candidate={label} training method ID mismatch")
    if str(
        training_metadata.get("experiment_protocol_content_sha256", "")
    ) != protocol_content_sha256:
        raise PaperSchemaError(
            f"candidate={label} training/evaluation experiment protocol mismatch"
        )
    if dict(training_metadata.get("experiment_protocol") or {}).get(
        "sha256"
    ) != protocol_artifact.get("sha256"):
        raise PaperSchemaError(
            f"candidate={label} checkpoint protocol artifact mismatch"
        )
    difficulty_bins = list(aggregate.get("difficulty_bins") or [])
    expected_difficulty = str(
        training_metadata.get("trained_on_difficulty_bin", "")
    ).strip()
    if difficulty_bins != [expected_difficulty]:
        raise PaperSchemaError(
            f"candidate={label} aggregate difficulty mismatch: {difficulty_bins}"
        )
    return {
        "label": str(label),
        "run_manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "run_manifest_content_sha256": manifest["content_sha256"],
        "run_id": manifest["run_id"],
        "algorithm": manifest["algorithm"],
        "training_algorithm": training_algorithm,
        "scenario": manifest["scenario"],
        "split": manifest["split"],
        "train_seed": int(manifest["train_seed"]),
        "eval_seed": int(manifest["eval_seed"]),
        "validation_task_bank_sha256": manifest["task_bank"]["sha256"],
        "validation_task_ids_sha256": manifest["task_bank"][
            "split_task_ids_sha256"
        ],
        "training_difficulty_bin": training_metadata[
            "trained_on_difficulty_bin"
        ],
        "training_run_id": training_metadata["training_run_id"],
        "training_method_id": training_metadata["paper_method_id"],
        "experiment_protocol_content_sha256": training_metadata[
            "experiment_protocol_content_sha256"
        ],
        "training_source_bundle_sha256": training_metadata[
            "training_source_bundle_sha256"
        ],
        "checkpoint_action_force_ratio": checkpoint_force_ratio,
        "checkpoint": checkpoint,
        "episodes": episodes,
        "unique_tasks": unique_tasks,
        "team_success": dict(aggregate["team_success"]),
        "collision_rate": dict(aggregate["terminal_reason_rates"]["collision"]),
        "agent_spl": dict(aggregate["agent_spl"]),
        "control_effort": dict(aggregate["control_effort"]),
        "aggregate_content_sha256": aggregate["content_sha256"],
    }


def _validation_eligibility(
    candidate: Mapping[str, Any], gate_contract: Mapping[str, Any]
) -> Dict[str, Any]:
    success = dict(candidate.get("team_success") or {})
    trials = int(success.get("trials", -1))
    successes = int(success.get("successes", -1))
    estimate = float(success.get("estimate", -1.0))
    lower = float(success.get("lower", -1.0))
    if trials <= 0 or successes < 0 or successes > trials:
        raise PaperSchemaError(
            f"candidate={candidate.get('label')} has invalid success counts"
        )
    expected_estimate = successes / trials
    if abs(estimate - expected_estimate) > 1e-12:
        raise PaperSchemaError(
            f"candidate={candidate.get('label')} success estimate/count mismatch"
        )
    if int(candidate.get("episodes", -1)) != trials:
        raise PaperSchemaError(
            f"candidate={candidate.get('label')} success trials/episodes mismatch"
        )
    expected_interval = wilson_interval(successes, trials)
    for field in ("lower", "upper", "confidence"):
        actual_value = float(success.get(field, -1.0))
        if abs(actual_value - float(expected_interval[field])) > 1e-12:
            raise PaperSchemaError(
                f"candidate={candidate.get('label')} Wilson {field} mismatch"
            )

    required_tasks = int(gate_contract["required_tasks"])
    minimum_successes = int(gate_contract["minimum_successes"])
    minimum_rate = float(gate_contract["minimum_observed_success_rate"])
    minimum_lower = float(gate_contract["wilson_lower_bound"])
    reasons = []
    if trials != required_tasks:
        reasons.append(
            f"validation tasks {trials} != required {required_tasks}"
        )
    if successes < minimum_successes:
        reasons.append(
            f"validation successes {successes} < {minimum_successes}"
        )
    if estimate < minimum_rate:
        reasons.append(
            f"validation success rate {estimate:.6f} < {minimum_rate:.6f}"
        )
    if lower < minimum_lower:
        reasons.append(
            f"Wilson lower {lower:.6f} < {minimum_lower:.6f}"
        )
    return {
        "eligible": not reasons,
        "rejection_reasons": reasons,
        "observed_tasks": trials,
        "observed_successes": successes,
        "observed_success_rate": estimate,
        "observed_wilson_lower_bound": lower,
        "requirements": dict(gate_contract),
    }


def _candidate_rank_key(candidate: Mapping[str, Any]):
    spl_mean = candidate["agent_spl"].get("mean")
    effort_mean = candidate["control_effort"].get("mean")
    return (
        not bool(candidate["validation_eligibility"]["eligible"]),
        -float(candidate["team_success"]["estimate"]),
        float(candidate["collision_rate"]["estimate"]),
        -float(spl_mean if spl_mean is not None else 0.0),
        float(effort_mean if effort_mean is not None else float("inf")),
        str(candidate["checkpoint"]["sha256"]),
    )


def select_checkpoint(
    candidate_manifests: Mapping[str, str | Path],
    *,
    require_adequacy: bool = True,
) -> Dict[str, Any]:
    if len(candidate_manifests) < 2:
        raise PaperSchemaError("paper checkpoint selection requires at least two candidates")
    candidates = [
        _candidate_from_manifest(label, path)
        for label, path in sorted(candidate_manifests.items())
    ]
    reference = candidates[0]
    parity_fields = (
        "algorithm",
        "training_algorithm",
        "scenario",
        "split",
        "train_seed",
        "eval_seed",
        "validation_task_bank_sha256",
        "episodes",
        "unique_tasks",
        "validation_task_ids_sha256",
        "training_difficulty_bin",
        "training_run_id",
        "training_method_id",
        "experiment_protocol_content_sha256",
        "training_source_bundle_sha256",
    )
    for candidate in candidates[1:]:
        mismatches = [
            field for field in parity_fields if candidate[field] != reference[field]
        ]
        if mismatches:
            raise PaperSchemaError(
                f"validation candidate protocol mismatch for {candidate['label']}: {mismatches}"
            )

    checkpoint_hashes = [candidate["checkpoint"]["sha256"] for candidate in candidates]
    if len(set(checkpoint_hashes)) != len(checkpoint_hashes):
        raise PaperSchemaError("checkpoint candidates must have distinct artifact hashes")

    gate_contract = validation_checkpoint_gate(
        reference["training_difficulty_bin"]
    )
    for candidate in candidates:
        candidate["validation_eligibility"] = _validation_eligibility(
            candidate, gate_contract
        )

    eligible = [
        candidate
        for candidate in candidates
        if candidate["validation_eligibility"]["eligible"]
    ]
    if bool(require_adequacy) and not eligible:
        diagnostics = {
            candidate["label"]: candidate["validation_eligibility"]
            for candidate in candidates
        }
        raise PaperSchemaError(
            "no checkpoint passed the frozen validation adequacy gate: "
            + json.dumps(diagnostics, sort_keys=True)
        )

    ranked = sorted(candidates, key=_candidate_rank_key)
    selected = ranked[0]
    payload = {
        "schema_version": PAPER_CHECKPOINT_SELECTION_SCHEMA_VERSION,
        "selection_data_role": "validation",
        "adequacy_required_for_selection": bool(require_adequacy),
        "validation_adequacy_gate": dict(gate_contract),
        "eligible_candidate_count": len(eligible),
        "ranking_rule": [
            "maximize team_success estimate",
            "minimize collision terminal rate",
            "maximize mean agent SPL",
            "minimize mean control effort",
            "lexicographic checkpoint sha256 tie break",
        ],
        "validation_split": reference["split"],
        "validation_task_bank_sha256": reference["validation_task_bank_sha256"],
        "experiment_protocol_content_sha256": reference[
            "experiment_protocol_content_sha256"
        ],
        "train_seed": reference["train_seed"],
        "eval_seed": reference["eval_seed"],
        "candidate_count": len(ranked),
        "candidates_ranked": ranked,
        "selected_label": selected["label"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_validation_run_manifest": selected["run_manifest_path"],
    }
    payload["content_sha256"] = content_sha256(payload)
    return payload


def validate_selection_result(payload: Mapping[str, Any]) -> Dict[str, Any]:
    if payload.get("schema_version") != PAPER_CHECKPOINT_SELECTION_SCHEMA_VERSION:
        raise PaperSchemaError("checkpoint selection schema mismatch")
    expected = content_sha256({key: value for key, value in payload.items() if key != "content_sha256"})
    if payload.get("content_sha256") != expected:
        raise PaperSchemaError("checkpoint selection content_sha256 mismatch")
    if payload.get("selection_data_role") != "validation":
        raise PaperSchemaError("checkpoint selection must use validation data only")
    selected = payload.get("selected_checkpoint")
    if not isinstance(selected, Mapping) or not selected.get("sha256"):
        raise PaperSchemaError("checkpoint selection lacks selected checkpoint fingerprint")
    ranked = payload.get("candidates_ranked")
    if not isinstance(ranked, list) or len(ranked) < 2:
        raise PaperSchemaError("checkpoint selection requires at least two ranked candidates")
    if int(payload.get("candidate_count", -1)) != len(ranked):
        raise PaperSchemaError("checkpoint selection candidate_count mismatch")
    difficulties = {
        str(candidate.get("training_difficulty_bin", "")).strip()
        for candidate in ranked
    }
    if len(difficulties) != 1 or "" in difficulties:
        raise PaperSchemaError(
            "checkpoint selection candidates must share one difficulty"
        )
    protocol_hashes = {
        str(candidate.get("experiment_protocol_content_sha256", "")).strip()
        for candidate in ranked
    }
    if len(protocol_hashes) != 1 or "" in protocol_hashes:
        raise PaperSchemaError(
            "checkpoint selection candidates must share one experiment protocol"
        )
    if str(payload.get("experiment_protocol_content_sha256", "")).strip() != (
        next(iter(protocol_hashes))
    ):
        raise PaperSchemaError("checkpoint selection experiment protocol mismatch")
    expected_gate = validation_checkpoint_gate(next(iter(difficulties)))
    if payload.get("validation_adequacy_gate") != expected_gate:
        raise PaperSchemaError("checkpoint selection adequacy gate mismatch")
    for candidate in ranked:
        recorded = candidate.get("validation_eligibility")
        recalculated = _validation_eligibility(candidate, expected_gate)
        if recorded != recalculated:
            raise PaperSchemaError(
                f"checkpoint selection eligibility mismatch for {candidate.get('label')}"
            )
    expected_ranking = sorted(ranked, key=_candidate_rank_key)
    if [candidate.get("label") for candidate in ranked] != [
        candidate.get("label") for candidate in expected_ranking
    ]:
        raise PaperSchemaError("checkpoint selection ranking order mismatch")
    eligible_candidates = [
        candidate
        for candidate in ranked
        if bool((candidate.get("validation_eligibility") or {}).get("eligible"))
    ]
    if int(payload.get("eligible_candidate_count", -1)) != len(
        eligible_candidates
    ):
        raise PaperSchemaError("checkpoint selection eligible_candidate_count mismatch")
    adequacy_required = payload.get("adequacy_required_for_selection")
    if not isinstance(adequacy_required, bool):
        raise PaperSchemaError(
            "checkpoint selection lacks adequacy-required policy"
        )
    if adequacy_required and not eligible_candidates:
        raise PaperSchemaError("checkpoint selection has no validation-eligible candidate")
    selected_label = str(payload.get("selected_label", "")).strip()
    matching = [candidate for candidate in ranked if candidate.get("label") == selected_label]
    if len(matching) != 1:
        raise PaperSchemaError("selected_label does not identify exactly one candidate")
    if matching[0].get("checkpoint") != selected:
        raise PaperSchemaError("selected_checkpoint does not match selected candidate")
    if adequacy_required and not bool(
        (matching[0].get("validation_eligibility") or {}).get("eligible")
    ):
        raise PaperSchemaError("selected checkpoint did not pass validation adequacy")
    if ranked[0].get("label") != selected_label:
        raise PaperSchemaError("selected candidate must be first in recorded ranking")
    return dict(payload)


def _parse_candidates(values: Iterable[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for value in values:
        label, separator, path = str(value).partition("=")
        if not separator or not label.strip() or not path.strip():
            raise PaperSchemaError("--candidate must use LABEL=/path/to/paper_run_manifest.json")
        if label.strip() in parsed:
            raise PaperSchemaError(f"duplicate candidate label={label.strip()!r}")
        parsed[label.strip()] = path.strip()
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = select_checkpoint(_parse_candidates(args.candidate))
    atomic_write_json(args.output, result)
    print(
        f"selected={result['selected_label']} checkpoint_sha256="
        f"{result['selected_checkpoint']['sha256']} output={Path(args.output).resolve()}"
    )


if __name__ == "__main__":
    main()
