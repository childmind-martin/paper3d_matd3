"""Versioned, fail-loud data contracts for paper training and evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


PAPER_EPISODE_SCHEMA_VERSION = "paper_episode_v2"
PAPER_EPISODE_COLLECTION_SCHEMA_VERSION = "paper_episode_collection_v2"
PAPER_AGGREGATE_SCHEMA_VERSION = "paper_aggregate_v2"
PAPER_RUN_MANIFEST_SCHEMA_VERSION = "paper_run_manifest_v2"

PAPER_METRIC_DEFINITIONS = {
    "team_success": "1 iff every agent reached its assigned goal and remained collision-free for the entire episode",
    "agent_success": "reached_goal AND collision_free",
    "spl": "success * reference_path_length / max(reference_path_length, executed_path_length)",
    "minimum_clearance_m": "minimum body-surface clearance over the complete episode; negative values denote penetration",
    "energy": "integral of (1 + squared executed normalized action head) dt; normalized energy proxy, not joules",
    "control_effort": "integral of squared executed normalized action head dt",
    "completion_fraction": "episode_steps divided by the immutable task deadline_steps",
    "episode_simulation_seconds": "episode_steps multiplied by the frozen simulation time step",
    "deadline": "per-task finite horizon derived only from frozen reference-path length before evaluation",
    "mean_action_intervention_norm": "episode mean of the per-step, per-agent L2 difference between executed and raw normalized action heads",
    "apf_intervention_step_fraction": "fraction of episode steps where at least one agent's executed normalized action head differs from its raw head by more than 1e-6",
    "mean_apf_force_norm": "episode mean of the per-step, per-agent APF force-head L2 norm",
}

PAPER_METRIC_UNITS = {
    "minimum_clearance_m": "m",
    "energy": "normalized_energy_seconds",
    "control_effort": "normalized_action_squared_seconds",
    "episode_steps": "simulation_steps",
    "deadline_steps": "simulation_steps",
    "deadline_seconds": "s",
    "simulation_dt_seconds": "s_per_simulation_step",
    "episode_simulation_seconds": "s",
    "completion_fraction": "ratio",
    "final_goal_distance_m": "m",
    "mean_raw_action_norm": "normalized_action_l2",
    "mean_executed_action_norm": "normalized_action_l2",
    "mean_action_intervention_norm": "normalized_action_l2",
    "mean_apf_force_norm": "normalized_action_l2",
    "apf_intervention_step_fraction": "ratio",
    "configured_action_force_ratio": "ratio",
}

TERMINAL_REASONS = frozenset(
    {
        "team_success",
        "all_reached_unsafe",
        "time_limit",
        "collision",
        "out_of_bounds",
        "terminal_semantics_error",
    }
)

_TERMINAL_REASON_ALIASES = {
    "team_success": "team_success",
    "all_reached_unsafe": "all_reached_unsafe",
    "all_reached_without_safe_team_success": "all_reached_unsafe",
    "time_limit": "time_limit",
    "collision": "collision",
    "collision_or_safety_early_done": "collision",
    "gazebo_hard_contact": "collision",
    "out_of_bounds": "out_of_bounds",
    "terminal_semantics_error": "terminal_semantics_error",
}


class PaperSchemaError(ValueError):
    pass


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_artifact(path: os.PathLike[str] | str) -> Dict[str, Any]:
    """Fingerprint one file or an immutable directory tree for provenance."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise PaperSchemaError(f"artifact does not exist: {resolved}")
    if resolved.is_file():
        return {
            "path": str(resolved),
            "kind": "file",
            "size_bytes": int(resolved.stat().st_size),
            "sha256": sha256_file(resolved),
        }
    files = []
    for candidate in sorted(item for item in resolved.rglob("*") if item.is_file()):
        files.append(
            {
                "relative_path": candidate.relative_to(resolved).as_posix(),
                "size_bytes": int(candidate.stat().st_size),
                "sha256": sha256_file(candidate),
            }
        )
    if not files:
        raise PaperSchemaError(f"artifact directory is empty: {resolved}")
    return {
        "path": str(resolved),
        "kind": "directory",
        "file_count": len(files),
        "files": files,
        "sha256": content_sha256(files),
    }


def build_paper_run_manifest(
    *,
    run_id: str,
    phase: str,
    algorithm: str,
    scenario: str,
    split: str,
    split_role: str,
    train_seed: int,
    eval_seed: int,
    experiment_protocol: Mapping[str, Any],
    task_bank: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
    source_artifacts: Sequence[Mapping[str, Any]],
    outputs: Mapping[str, Mapping[str, Any]],
    checkpoint_selection: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build the immutable provenance record that binds code, model, tasks and data."""
    required_strings = {
        "run_id": run_id,
        "phase": phase,
        "algorithm": algorithm,
        "scenario": scenario,
        "split": split,
        "split_role": split_role,
    }
    for name, value in required_strings.items():
        if not str(value or "").strip():
            raise PaperSchemaError(f"paper run manifest missing {name}")
    if str(phase) not in {"validation", "test_in_distribution", "test_generalization"}:
        raise PaperSchemaError(f"unsupported paper run phase={phase!r}")
    if str(split_role) != str(phase):
        raise PaperSchemaError(
            f"paper run phase/split role mismatch: phase={phase!r}, role={split_role!r}"
        )
    for name, artifact in (
        ("experiment_protocol", experiment_protocol),
        ("task_bank", task_bank),
        ("checkpoint", checkpoint),
    ):
        if not isinstance(artifact, Mapping) or not str(artifact.get("sha256", "")).strip():
            raise PaperSchemaError(f"paper run manifest {name} lacks sha256")
    if not source_artifacts:
        raise PaperSchemaError("paper run manifest requires source artifact fingerprints")
    if not outputs:
        raise PaperSchemaError("paper run manifest requires output fingerprints")
    if str(phase) == "validation" and checkpoint_selection is not None:
        raise PaperSchemaError(
            "validation run manifest must not consume checkpoint selection"
        )
    if str(phase) != "validation" and checkpoint_selection is None:
        raise PaperSchemaError(
            "paper test run manifest requires validation checkpoint selection"
        )
    payload = {
        "schema_version": PAPER_RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": str(run_id),
        "phase": str(phase),
        "algorithm": str(algorithm),
        "scenario": str(scenario),
        "split": str(split),
        "split_role": str(split_role),
        "train_seed": int(train_seed),
        "eval_seed": int(eval_seed),
        "experiment_protocol": dict(experiment_protocol),
        "task_bank": dict(task_bank),
        "checkpoint": dict(checkpoint),
        "runtime_environment": dict(runtime_environment),
        "source_artifacts": [dict(item) for item in source_artifacts],
        "outputs": {str(name): dict(value) for name, value in outputs.items()},
        "metric_definitions": dict(PAPER_METRIC_DEFINITIONS),
        "metric_units": dict(PAPER_METRIC_UNITS),
    }
    if checkpoint_selection is not None:
        if not isinstance(checkpoint_selection, Mapping) or not str(
            checkpoint_selection.get("sha256", "")
        ).strip():
            raise PaperSchemaError(
                "paper run manifest checkpoint_selection lacks sha256"
            )
        payload["checkpoint_selection"] = dict(checkpoint_selection)
    payload["content_sha256"] = content_sha256(payload)
    return payload


def build_episode_collection(
    records: Iterable[Mapping[str, Any]],
    *,
    run_id: str,
    split: str,
) -> Dict[str, Any]:
    normalized = [validate_episode_record(record) for record in records]
    if not normalized:
        raise PaperSchemaError("paper episode collection cannot be empty")
    if any(record["run_id"] != str(run_id) for record in normalized):
        raise PaperSchemaError("episode collection contains a different run_id")
    if any(record["split"] != str(split) for record in normalized):
        raise PaperSchemaError("episode collection contains a different split")
    payload = {
        "schema_version": PAPER_EPISODE_COLLECTION_SCHEMA_VERSION,
        "run_id": str(run_id),
        "split": str(split),
        "record_count": len(normalized),
        "records": normalized,
    }
    payload["content_sha256"] = content_sha256(payload)
    return payload


def atomic_write_json(path: os.PathLike[str] | str, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> Dict[str, float]:
    successes = int(successes)
    trials = int(trials)
    if trials <= 0 or successes < 0 or successes > trials:
        raise PaperSchemaError(
            f"invalid binomial counts: successes={successes}, trials={trials}"
        )
    p = successes / trials
    z2 = float(z) ** 2
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denominator
    radius = (
        float(z)
        * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials))
        / denominator
    )
    return {
        "successes": successes,
        "trials": trials,
        "estimate": float(p),
        "lower": float(max(0.0, center - radius)),
        "upper": float(min(1.0, center + radius)),
        "confidence": 0.95,
    }


def _finite_float(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PaperSchemaError(f"{field} must be numeric, got={value!r}") from exc
    if not math.isfinite(result):
        raise PaperSchemaError(f"{field} must be finite, got={result}")
    if minimum is not None and result < minimum:
        raise PaperSchemaError(f"{field} must be >= {minimum}, got={result}")
    if maximum is not None and result > maximum:
        raise PaperSchemaError(f"{field} must be <= {maximum}, got={result}")
    return result


def _normalize_task_difficulty_summary(
    summary: Mapping[str, Any], *, expected_agents: int
) -> Dict[str, Any]:
    if not isinstance(summary, Mapping):
        raise PaperSchemaError("paper episode requires task_difficulty_summary")
    try:
        agent_count = int(summary.get("agent_count"))
        obstacle_count = int(summary.get("obstacle_count"))
    except (TypeError, ValueError) as exc:
        raise PaperSchemaError(
            "task_difficulty_summary agent/obstacle counts must be integers"
        ) from exc
    if agent_count != int(expected_agents):
        raise PaperSchemaError(
            "task_difficulty_summary.agent_count does not match episode agents"
        )
    if obstacle_count < 0:
        raise PaperSchemaError(
            "task_difficulty_summary.obstacle_count must be non-negative"
        )
    normalized = {
        "agent_count": agent_count,
        "obstacle_count": obstacle_count,
        "minimum_horizontal_distance_m": _finite_float(
            summary.get("minimum_horizontal_distance_m"),
            "task_difficulty_summary.minimum_horizontal_distance_m",
            minimum=0.0,
        ),
        "maximum_horizontal_distance_m": _finite_float(
            summary.get("maximum_horizontal_distance_m"),
            "task_difficulty_summary.maximum_horizontal_distance_m",
            minimum=0.0,
        ),
        "mean_reference_path_length_m": _finite_float(
            summary.get("mean_reference_path_length_m"),
            "task_difficulty_summary.mean_reference_path_length_m",
            minimum=0.0,
        ),
        "maximum_reference_path_length_m": _finite_float(
            summary.get("maximum_reference_path_length_m"),
            "task_difficulty_summary.maximum_reference_path_length_m",
            minimum=0.0,
        ),
        "maximum_detour_ratio": _finite_float(
            summary.get("maximum_detour_ratio"),
            "task_difficulty_summary.maximum_detour_ratio",
            minimum=1.0,
        ),
        "maximum_cumulative_climb_m": _finite_float(
            summary.get("maximum_cumulative_climb_m"),
            "task_difficulty_summary.maximum_cumulative_climb_m",
            minimum=0.0,
        ),
        "minimum_boundary_clearance_m": _finite_float(
            summary.get("minimum_boundary_clearance_m"),
            "task_difficulty_summary.minimum_boundary_clearance_m",
            minimum=0.0,
        ),
    }
    obstacle_clearance = summary.get("minimum_obstacle_clearance_m")
    if obstacle_count == 0:
        if obstacle_clearance is not None:
            raise PaperSchemaError(
                "obstacle-free task must have minimum_obstacle_clearance_m=null"
            )
        normalized["minimum_obstacle_clearance_m"] = None
    else:
        normalized["minimum_obstacle_clearance_m"] = _finite_float(
            obstacle_clearance,
            "task_difficulty_summary.minimum_obstacle_clearance_m",
            minimum=0.0,
        )
    if normalized["minimum_horizontal_distance_m"] > normalized[
        "maximum_horizontal_distance_m"
    ]:
        raise PaperSchemaError("task horizontal distance summary is inconsistent")
    if normalized["mean_reference_path_length_m"] > normalized[
        "maximum_reference_path_length_m"
    ]:
        raise PaperSchemaError("task reference path summary is inconsistent")
    return normalized


def validate_episode_record(record: Mapping[str, Any], *, expected_agents: int | None = None) -> Dict[str, Any]:
    if not isinstance(record, Mapping):
        raise PaperSchemaError("episode record must be an object")
    if record.get("schema_version") != PAPER_EPISODE_SCHEMA_VERSION:
        raise PaperSchemaError(
            f"episode schema mismatch: {record.get('schema_version')!r}"
        )
    for field in ("run_id", "algorithm", "split", "task_id", "difficulty_bin"):
        if not str(record.get(field, "")).strip():
            raise PaperSchemaError(f"episode record missing {field}")
    terminal_reason = str(record.get("terminal_reason", "")).strip()
    if terminal_reason not in TERMINAL_REASONS:
        raise PaperSchemaError(f"unsupported terminal_reason={terminal_reason!r}")
    agents = record.get("agents")
    if not isinstance(agents, list) or not agents:
        raise PaperSchemaError("episode agents must be a non-empty list")
    if expected_agents is not None and len(agents) != int(expected_agents):
        raise PaperSchemaError(
            f"agent count mismatch: expected={expected_agents}, actual={len(agents)}"
        )

    all_agent_success = True
    all_agent_reached = True
    all_agent_safe = True
    normalized_agents: List[Dict[str, Any]] = []
    for index, raw_agent in enumerate(agents):
        if not isinstance(raw_agent, Mapping):
            raise PaperSchemaError(f"agents[{index}] must be an object")
        reached = bool(raw_agent.get("reached_goal", False))
        safe = bool(raw_agent.get("collision_free", False))
        success = bool(raw_agent.get("success", False))
        if success != bool(reached and safe):
            raise PaperSchemaError(
                f"agents[{index}].success must equal reached_goal AND collision_free"
            )
        path_length = _finite_float(
            raw_agent.get("path_length_m"), f"agents[{index}].path_length_m", minimum=0.0
        )
        reference_length = _finite_float(
            raw_agent.get("reference_path_length_m"),
            f"agents[{index}].reference_path_length_m",
            minimum=0.0,
        )
        spl = _finite_float(raw_agent.get("spl"), f"agents[{index}].spl", minimum=0.0)
        expected_spl = (
            min(1.0, reference_length / max(reference_length, path_length, 1e-12))
            if success
            else 0.0
        )
        if abs(spl - expected_spl) > 1e-6:
            raise PaperSchemaError(
                f"agents[{index}].spl={spl} inconsistent with success/path lengths; "
                f"expected={expected_spl}"
            )
        normalized_agent = dict(raw_agent)
        normalized_agent.update(
            {
                "agent_index": int(raw_agent.get("agent_index", index)),
                "reached_goal": reached,
                "collision_free": safe,
                "success": success,
                "path_length_m": path_length,
                "reference_path_length_m": reference_length,
                "spl": spl,
                "minimum_clearance_m": _finite_float(
                    raw_agent.get("minimum_clearance_m"),
                    f"agents[{index}].minimum_clearance_m",
                ),
                "energy": _finite_float(
                    raw_agent.get("energy"), f"agents[{index}].energy", minimum=0.0
                ),
                "control_effort": _finite_float(
                    raw_agent.get("control_effort"),
                    f"agents[{index}].control_effort",
                    minimum=0.0,
                ),
                "final_goal_distance_m": _finite_float(
                    raw_agent.get("final_goal_distance_m"),
                    f"agents[{index}].final_goal_distance_m",
                    minimum=0.0,
                ),
            }
        )
        normalized_agents.append(normalized_agent)
        all_agent_success = all_agent_success and success
        all_agent_reached = all_agent_reached and reached
        all_agent_safe = all_agent_safe and safe

    team_success = bool(record.get("team_success", False))
    if team_success != all_agent_success:
        raise PaperSchemaError("team_success must equal all per-agent success flags")
    if terminal_reason == "team_success" and not team_success:
        raise PaperSchemaError("terminal_reason=team_success requires team_success=true")
    if team_success and terminal_reason != "team_success":
        raise PaperSchemaError("successful episode must terminate with team_success")
    if terminal_reason == "time_limit" and team_success:
        raise PaperSchemaError("time_limit episode cannot be team success")

    normalized = dict(record)
    deadline_steps = int(record.get("deadline_steps", 0))
    deadline_seconds = _finite_float(
        record.get("deadline_seconds"), "deadline_seconds", minimum=0.0
    )
    simulation_dt = _finite_float(
        record.get("simulation_dt_seconds"),
        "simulation_dt_seconds",
        minimum=0.0,
    )
    if deadline_steps <= 0 or deadline_seconds <= 0.0 or simulation_dt <= 0.0:
        raise PaperSchemaError("paper episode requires a positive immutable deadline")
    if not math.isclose(
        deadline_seconds,
        deadline_steps * simulation_dt,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise PaperSchemaError("deadline seconds/steps/simulation_dt are inconsistent")
    task_difficulty = _normalize_task_difficulty_summary(
        record.get("task_difficulty_summary"), expected_agents=len(agents)
    )
    raw_apf_diagnostics = record.get("apf_diagnostics")
    if not isinstance(raw_apf_diagnostics, Mapping):
        raise PaperSchemaError("paper episode is missing apf_diagnostics")
    action_diagnostic_steps = int(
        raw_apf_diagnostics.get("action_diagnostic_steps", 0)
    )
    if action_diagnostic_steps != int(record.get("episode_steps", 0)):
        raise PaperSchemaError(
            "apf_diagnostics.action_diagnostic_steps must equal episode_steps"
        )
    missing_action_diagnostic_steps = int(
        raw_apf_diagnostics.get("missing_action_diagnostic_steps", -1)
    )
    missing_pf_force_steps = int(
        raw_apf_diagnostics.get("missing_pf_force_steps", -1)
    )
    if missing_action_diagnostic_steps != 0:
        raise PaperSchemaError(
            "paper episode requires zero missing action diagnostic steps"
        )
    apf_diagnostics = {
        "action_diagnostic_steps": action_diagnostic_steps,
        "missing_action_diagnostic_steps": missing_action_diagnostic_steps,
        "missing_pf_force_steps": missing_pf_force_steps,
        "configured_action_force_ratio": _finite_float(
            raw_apf_diagnostics.get("configured_action_force_ratio"),
            "apf_diagnostics.configured_action_force_ratio",
            minimum=0.0,
        ),
        "mean_raw_action_norm": _finite_float(
            raw_apf_diagnostics.get("mean_raw_action_norm"),
            "apf_diagnostics.mean_raw_action_norm",
            minimum=0.0,
        ),
        "mean_executed_action_norm": _finite_float(
            raw_apf_diagnostics.get("mean_executed_action_norm"),
            "apf_diagnostics.mean_executed_action_norm",
            minimum=0.0,
        ),
        "mean_action_intervention_norm": _finite_float(
            raw_apf_diagnostics.get("mean_action_intervention_norm"),
            "apf_diagnostics.mean_action_intervention_norm",
            minimum=0.0,
        ),
        "apf_intervention_step_fraction": _finite_float(
            raw_apf_diagnostics.get("apf_intervention_step_fraction"),
            "apf_diagnostics.apf_intervention_step_fraction",
            minimum=0.0,
            maximum=1.0,
        ),
        "mean_apf_force_norm": _finite_float(
            raw_apf_diagnostics.get("mean_apf_force_norm"),
            "apf_diagnostics.mean_apf_force_norm",
            minimum=0.0,
        ),
    }
    if missing_pf_force_steps < 0:
        raise PaperSchemaError(
            "apf_diagnostics.missing_pf_force_steps must be non-negative"
        )
    if (
        apf_diagnostics["configured_action_force_ratio"] > 0.0
        and missing_pf_force_steps != 0
    ):
        raise PaperSchemaError(
            "positive APF force ratio requires zero missing PF force steps"
        )
    normalized.update(
        {
            "episode_index": int(record.get("episode_index", 0)),
            "train_seed": int(record.get("train_seed", 0)),
            "eval_seed": int(record.get("eval_seed", 0)),
            "terminal_reason": terminal_reason,
            "team_success": team_success,
            "all_reached": all_agent_reached,
            "collision_free": all_agent_safe,
            "episode_steps": int(record.get("episode_steps", 0)),
            "deadline_steps": deadline_steps,
            "deadline_seconds": deadline_seconds,
            "simulation_dt_seconds": simulation_dt,
            "episode_simulation_seconds": float(
                int(record.get("episode_steps", 0)) * simulation_dt
            ),
            "completion_fraction": float(
                int(record.get("episode_steps", 0)) / deadline_steps
            ),
            "task_difficulty_summary": task_difficulty,
            "apf_diagnostics": apf_diagnostics,
            "agents": normalized_agents,
        }
    )
    if normalized["episode_steps"] <= 0:
        raise PaperSchemaError("episode_steps must be positive")
    if normalized["episode_steps"] > deadline_steps:
        raise PaperSchemaError("episode_steps cannot exceed task deadline_steps")
    if terminal_reason == "time_limit" and normalized["episode_steps"] != deadline_steps:
        raise PaperSchemaError(
            "time_limit episode_steps must equal immutable task deadline_steps"
        )
    return normalized


def normalize_terminal_reason(reason: Any) -> str:
    normalized = _TERMINAL_REASON_ALIASES.get(str(reason or "").strip())
    if normalized is None:
        raise PaperSchemaError(f"unsupported raw terminal reason={reason!r}")
    return normalized


def episode_record_from_evaluation(
    episode: Mapping[str, Any],
    *,
    run_id: str,
    algorithm: str,
    split: str,
    train_seed: int,
    eval_seed: int,
) -> Dict[str, Any]:
    """Convert evaluator output to the strict long-lived paper schema."""
    layout = episode.get("task_layout")
    if not isinstance(layout, Mapping):
        raise PaperSchemaError("paper evaluation episode is missing task_layout")
    task_id = str(layout.get("task_id", "")).strip()
    difficulty = str(layout.get("difficulty_bin", "")).strip()
    layout_contract = dict(layout.get("contract") or {})
    deadline = dict(layout_contract.get("deadline") or {})
    runtime_deadline = episode.get("task_deadline")
    if not isinstance(runtime_deadline, Mapping):
        raise PaperSchemaError(
            "paper evaluation episode is missing the runtime task_deadline"
        )
    if dict(runtime_deadline) != deadline:
        raise PaperSchemaError(
            "runtime task_deadline does not match immutable layout deadline"
        )
    task_difficulty = dict(
        (layout.get("metrics") or {}).get("difficulty_summary") or {}
    )
    paths = list((layout.get("metrics") or {}).get("reference_paths") or [])
    path_lengths = list(episode.get("agent_path_lengths") or [])
    final_distances = list(episode.get("agent_final_goal_distances") or [])
    reached_flags = list(episode.get("agent_reached_flags") or [])
    safe_flags = list(episode.get("agent_safe_flags") or [])
    success_flags = list(episode.get("agent_success_flags") or [])
    clearances = list(episode.get("agent_min_clearances") or [])
    energies = list(episode.get("agent_energy_consumed") or [])
    efforts = list(episode.get("agent_control_effort") or [])
    diagnostic_metrics = episode.get("diagnostic_metrics")
    if not isinstance(diagnostic_metrics, Mapping):
        raise PaperSchemaError(
            "paper evaluation episode is missing diagnostic_metrics"
        )
    lengths = {
        len(paths), len(path_lengths), len(final_distances), len(reached_flags),
        len(safe_flags), len(success_flags), len(clearances), len(energies), len(efforts),
    }
    if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
        raise PaperSchemaError(
            "paper evaluation per-agent arrays are missing or misaligned: "
            f"lengths={sorted(lengths)}"
        )
    agents = []
    for index in range(len(paths)):
        reference_length = _finite_float(
            paths[index].get("path_length"),
            f"task_layout.metrics.reference_paths[{index}].path_length",
            minimum=0.0,
        )
        path_length = _finite_float(
            path_lengths[index], f"agent_path_lengths[{index}]", minimum=0.0
        )
        reached = bool(reached_flags[index])
        safe = bool(safe_flags[index])
        success = bool(success_flags[index])
        if success != bool(reached and safe):
            raise PaperSchemaError(
                f"evaluation success mismatch for agent[{index}]: "
                f"success={success}, reached={reached}, safe={safe}"
            )
        spl = (
            min(1.0, reference_length / max(reference_length, path_length, 1e-12))
            if success else 0.0
        )
        agents.append(
            {
                "agent_index": index,
                "reached_goal": reached,
                "collision_free": safe,
                "success": success,
                "path_length_m": path_length,
                "reference_path_length_m": reference_length,
                "spl": spl,
                "minimum_clearance_m": clearances[index],
                "energy": energies[index],
                "control_effort": efforts[index],
                "final_goal_distance_m": final_distances[index],
            }
        )
    record = {
        "schema_version": PAPER_EPISODE_SCHEMA_VERSION,
        "run_id": str(run_id),
        "algorithm": str(algorithm),
        "split": str(split),
        "train_seed": int(train_seed),
        "eval_seed": int(eval_seed),
        "episode_index": int(episode.get("episode", 0)),
        "task_id": task_id,
        "difficulty_bin": difficulty,
        "terminal_reason": normalize_terminal_reason(episode.get("episode_done_reason")),
        "team_success": bool(episode.get("team_success", False)),
        "episode_steps": int(episode.get("steps", 0)),
        "deadline_steps": int(deadline.get("deadline_steps", 0)),
        "deadline_seconds": deadline.get("deadline_seconds"),
        "simulation_dt_seconds": deadline.get("simulation_dt_seconds"),
        "task_difficulty_summary": task_difficulty,
        "apf_diagnostics": {
            "action_diagnostic_steps": diagnostic_metrics.get(
                "action_diag_step_count"
            ),
            "missing_action_diagnostic_steps": diagnostic_metrics.get(
                "action_diag_missing_steps"
            ),
            "missing_pf_force_steps": diagnostic_metrics.get(
                "pf_force_diag_missing_steps"
            ),
            "configured_action_force_ratio": diagnostic_metrics.get(
                "mean_force_ratio"
            ),
            "mean_raw_action_norm": diagnostic_metrics.get("avg_action_norm"),
            "mean_executed_action_norm": diagnostic_metrics.get(
                "avg_corr_action_norm"
            ),
            "mean_action_intervention_norm": diagnostic_metrics.get(
                "avg_action_delta_norm"
            ),
            "apf_intervention_step_fraction": diagnostic_metrics.get(
                "action_intervention_step_fraction"
            ),
            "mean_apf_force_norm": diagnostic_metrics.get(
                "mean_pf_force_norm"
            ),
        },
        "agents": agents,
    }
    return validate_episode_record(record, expected_agents=len(agents))


def _metric_summary(values: Sequence[float]) -> Dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": None, "std": None, "median": None, "q25": None, "q75": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def _aggregate_core(normalized: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    team_successes = sum(int(record["team_success"]) for record in normalized)
    all_reached = sum(int(record["all_reached"]) for record in normalized)
    collision_free = sum(int(record["collision_free"]) for record in normalized)
    terminal_counts = {
        reason: sum(int(record["terminal_reason"] == reason) for record in normalized)
        for reason in sorted(TERMINAL_REASONS)
    }
    terminal_rates = {
        reason: wilson_interval(count, len(normalized))
        for reason, count in terminal_counts.items()
    }
    all_agents = [agent for record in normalized for agent in record["agents"]]
    successful_agents = [agent for agent in all_agents if agent["success"]]
    return {
        "episodes": len(normalized),
        "unique_tasks": len({record["task_id"] for record in normalized}),
        "team_success": wilson_interval(team_successes, len(normalized)),
        "all_reached": wilson_interval(all_reached, len(normalized)),
        "collision_free": wilson_interval(collision_free, len(normalized)),
        "terminal_reason_counts": terminal_counts,
        "terminal_reason_rates": terminal_rates,
        "timeout": terminal_rates["time_limit"],
        "episode_steps": _metric_summary([record["episode_steps"] for record in normalized]),
        "episode_simulation_seconds": _metric_summary([
            record["episode_simulation_seconds"] for record in normalized
        ]),
        "deadline_steps": _metric_summary([record["deadline_steps"] for record in normalized]),
        "completion_fraction": _metric_summary([record["completion_fraction"] for record in normalized]),
        "task_max_reference_path_length_m": _metric_summary([
            record["task_difficulty_summary"]["maximum_reference_path_length_m"]
            for record in normalized
        ]),
        "task_max_detour_ratio": _metric_summary([
            record["task_difficulty_summary"]["maximum_detour_ratio"]
            for record in normalized
        ]),
        "task_max_cumulative_climb_m": _metric_summary([
            record["task_difficulty_summary"]["maximum_cumulative_climb_m"]
            for record in normalized
        ]),
        "agent_spl": _metric_summary([agent["spl"] for agent in all_agents]),
        "successful_agent_spl": _metric_summary([agent["spl"] for agent in successful_agents]),
        "minimum_clearance_m": _metric_summary([agent["minimum_clearance_m"] for agent in all_agents]),
        "energy": _metric_summary([agent["energy"] for agent in all_agents]),
        "control_effort": _metric_summary([agent["control_effort"] for agent in all_agents]),
        "final_goal_distance_m": _metric_summary([agent["final_goal_distance_m"] for agent in all_agents]),
        "apf_diagnostics": {
            field: _metric_summary(
                [record["apf_diagnostics"][field] for record in normalized]
            )
            for field in (
                "configured_action_force_ratio",
                "mean_raw_action_norm",
                "mean_executed_action_norm",
                "mean_action_intervention_norm",
                "apf_intervention_step_fraction",
                "mean_apf_force_norm",
            )
        },
    }


def aggregate_episode_records(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    normalized = [validate_episode_record(record) for record in records]
    if not normalized:
        raise PaperSchemaError("cannot aggregate zero episodes")
    task_keys = {(record["task_id"], record["eval_seed"], record["episode_index"]) for record in normalized}
    if len(task_keys) != len(normalized):
        raise PaperSchemaError("duplicate task/eval_seed/episode_index record detected")
    payload: Dict[str, Any] = {
        "schema_version": PAPER_AGGREGATE_SCHEMA_VERSION,
        **_aggregate_core(normalized),
        "train_seeds": sorted({record["train_seed"] for record in normalized}),
        "eval_seeds": sorted({record["eval_seed"] for record in normalized}),
        "difficulty_bins": sorted({record["difficulty_bin"] for record in normalized}),
        "metric_definitions": dict(PAPER_METRIC_DEFINITIONS),
        "metric_units": dict(PAPER_METRIC_UNITS),
        "by_difficulty_bin": {
            difficulty: _aggregate_core(
                [record for record in normalized if record["difficulty_bin"] == difficulty]
            )
            for difficulty in sorted({record["difficulty_bin"] for record in normalized})
        },
        "by_train_seed": {
            str(seed): _aggregate_core(
                [record for record in normalized if record["train_seed"] == seed]
            )
            for seed in sorted({record["train_seed"] for record in normalized})
        },
        "by_eval_seed": {
            str(seed): _aggregate_core(
                [record for record in normalized if record["eval_seed"] == seed]
            )
            for seed in sorted({record["eval_seed"] for record in normalized})
        },
    }
    payload["content_sha256"] = content_sha256(payload)
    return payload


def write_episode_csv(path: os.PathLike[str] | str, records: Iterable[Mapping[str, Any]]) -> None:
    normalized = [validate_episode_record(record) for record in records]
    rows: List[Dict[str, Any]] = []
    for record in normalized:
        for agent in record["agents"]:
            rows.append(
                {
                    "schema_version": record["schema_version"],
                    "run_id": record["run_id"],
                    "algorithm": record["algorithm"],
                    "split": record["split"],
                    "train_seed": record["train_seed"],
                    "eval_seed": record["eval_seed"],
                    "episode_index": record["episode_index"],
                    "task_id": record["task_id"],
                    "difficulty_bin": record["difficulty_bin"],
                    "terminal_reason": record["terminal_reason"],
                    "team_success": int(record["team_success"]),
                    "episode_steps": record["episode_steps"],
                    "deadline_steps": record["deadline_steps"],
                    "deadline_seconds": record["deadline_seconds"],
                    "simulation_dt_seconds": record["simulation_dt_seconds"],
                    "episode_simulation_seconds": record["episode_simulation_seconds"],
                    "completion_fraction": record["completion_fraction"],
                    "task_obstacle_count": record["task_difficulty_summary"]["obstacle_count"],
                    "task_max_reference_path_length_m": record["task_difficulty_summary"]["maximum_reference_path_length_m"],
                    "task_max_detour_ratio": record["task_difficulty_summary"]["maximum_detour_ratio"],
                    "task_max_cumulative_climb_m": record["task_difficulty_summary"]["maximum_cumulative_climb_m"],
                    "task_min_boundary_clearance_m": record["task_difficulty_summary"]["minimum_boundary_clearance_m"],
                    "task_min_obstacle_clearance_m": record["task_difficulty_summary"].get("minimum_obstacle_clearance_m"),
                    **record["apf_diagnostics"],
                    **agent,
                }
            )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
