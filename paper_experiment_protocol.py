#!/usr/bin/env python3
"""Frozen method, seed, task, and data-role protocol for paper experiments."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Iterable, Mapping

from paper_curriculum import (
    build_paper_curriculum,
    build_paper_generalization_suite,
    paper_physics_contract,
    paper_task_semantics_contract,
)
from paper_experiment_schema import (
    PaperSchemaError,
    atomic_write_json,
    content_sha256,
    fingerprint_artifact,
)
from paper_task_bank import load_task_bank, validate_dynamic_feasibility_audit


PAPER_EXPERIMENT_PROTOCOL_SCHEMA_VERSION = "paper_experiment_protocol_v15"
FORMAL_INTERRUPTION_POLICY = "restart_train_seed_from_scratch_v1"
FORMAL_SELECTION_DIFFICULTY = "C3"
FORMAL_CHECKPOINT_POLICY = (
    "one_validation_C3_selected_checkpoint_for_all_splits_v1"
)
FORMAL_EVALUATION_DEVICE_REQUIREMENT = {
    "policy": "tensorflow_cpu_deterministic_v1",
    "cuda_visible_devices": "-1",
    "tf_deterministic_ops": "1",
    "tf_enable_onednn_opts": "0",
    "tf_num_intraop_threads": "1",
    "tf_num_interop_threads": "1",
    "omp_num_threads": "1",
    "maximum_physical_gpus": 0,
    "maximum_logical_gpus": 0,
}

_FORMAL_TASK_BANK_GENERATION_SCHEMA_VERSION = (
    "paper_task_bank_build_cache_v5"
)
_FORMAL_TASK_BANK_SOURCE_LOGICAL_NAMES = {
    "build_paper_task_bank.py",
    "audit_paper_task_feasibility.py",
    "paper_curriculum.py",
    "paper_task_bank.py",
    "paper_task_layout.py",
    "multiagent/core.py",
    "multiagent/environment.py",
    "multiagent/scenarios/paper3d_terrain_energy.py",
}

_FORMAL_EXPERIMENT_SOURCE_LOGICAL_NAMES = (
    "paper_experiment_protocol.py",
    "paper_experiment_pipeline.py",
    "paper3d_train_optimized.py",
    "evaluate_optimized.py",
    "run_optimized.sh",
    "run_evaluation.sh",
    "experiment_runtime_config.py",
    "paper_curriculum.py",
    "paper_task_bank.py",
    "paper_task_layout.py",
    "paper_experiment_schema.py",
    "paper_checkpoint_selection.py",
    "paper_results_summary.py",
    "paper_seed_statistics.py",
    "action_semantics.py",
    "paper_task_bank_report.py",
    "build_paper_task_bank.py",
    "audit_paper_task_feasibility.py",
    "potential_field_corrector.py",
    "cross_agent_reference_state.py",
    "cross_agent_reference_selector.py",
    "agents/noise.py",
    "agents/vectorized_ou_noise.py",
    "utils/observation_processor.py",
    "utils/vectorized_observation_processor.py",
    "utils/vectorized_reward_calculator.py",
    "visualization/trajectory_visualizer.py",
    "multiagent/__init__.py",
    "multiagent/core.py",
    "multiagent/environment.py",
    "multiagent/multi_discrete.py",
    "multiagent/quadrotor_dynamics.py",
    "multiagent/scenario.py",
    "multiagent/scenarios/__init__.py",
    "multiagent/scenarios/paper3d_terrain_energy.py",
    "multiagent/scenarios/obstacle_observation.py",
)

_FORMAL_STAGE_TRANSITION_RUNTIME = {
    "policy_id": "distribution_shift_reset_v1",
    "replay_policy": "clear_all",
    "replay_keep_recent_transitions": 0,
    "target_network_policy": "hard_sync_from_online",
    "optimizer_slot_policy": "reset_adam_moments",
    "learning_rate_policy": "multiply_once_at_first_transition",
    "learning_rate_factor": 0.5,
    "adaptive_plateau_policy": "reset_comparison_window_each_stage",
}

_FORMAL_COMMON_TRAINING_HYPERPARAMETERS = {
    "policy_id": "explicit_common_launcher_and_environment_v2",
    "discount_time_scale": {
        "gamma": 0.99,
        "simulation_dt_seconds": 0.08,
        "effective_horizon_steps": 100,
        "effective_horizon_seconds": 8.0,
    },
    "environment_variables": {
        "PAPER_GAMMA": "0.99",
        "OBSTACLE_OBSERVATION_MODE": "risk_lite_v2",
        "OBSTACLE_RISK_VELOCITY_FORWARD_WEIGHT": "4.0",
        "OBSTACLE_RISK_GOAL_ALONG_WEIGHT": "3.0",
        "ADAPTIVE_PATIENCE": "60",
        "ADAPTIVE_LR_DECAY": "0.98",
        "ADAPTIVE_ACTOR_LR_DECAY": "0.98",
        "ADAPTIVE_CRITIC_LR_DECAY": "1.0",
        "ADAPTIVE_MIN_LR": "8e-5",
        "ADAPTIVE_MIN_ACTOR_LR": "8e-5",
        "ADAPTIVE_MIN_CRITIC_LR": "2e-4",
        "ADAPTIVE_PLATEAU_ENABLED": "1",
        "ADAPTIVE_PLATEAU_WINDOW": "50",
        "ADAPTIVE_PLATEAU_REL_DELTA": "0.03",
        "ADAPTIVE_PLATEAU_SLOPE_EPS": "0.002",
        "ADAPTIVE_NOISE_MAX": "0.12",
        "ADAPTIVE_NOISE_SMOOTH": "0.4",
        "NOISE_RESTART_INTERVAL": "25",
    },
    "launcher_flags": {
        "--episode-length": "2800",
        "--save-interval": "0",
        "--learning-rate-actor": "0.0003",
        "--learning-rate-critic": "0.0005",
        "--lr-decay-enabled": "1",
        "--lr-decay-steps": "30000",
        "--lr-decay-rate": "0.9995",
        "--lr-staircase": "1",
        "--lr-min-actor": "0.00005",
        "--lr-min-critic": "0.00008",
        "--gamma": "0.99",
        "--tau": "0.015",
        "--huber-delta": "1.0",
        "--grad-clip-norm": "10.0",
        "--action-reg-coef": "0.00014",
        "--neg-z-reg-coef": "0.0",
        "--actor-hidden": "256,256,256",
        "--critic-hidden": "256,256,256",
        "--lite-buffer": "true",
        "--buffer-size": "450000",
        "--update-rate": "40",
        "--policy-noise": "0.28",
        "--noise-clip": "0.32",
        "--policy-freq": "2",
        "--noise-scale": "0.30",
        "--noise-decay": "0.9995",
        "--noise-decay-steps": "30000",
        "--noise-staircase": "1",
        "--noise-decay-enabled": "1",
        "--noise-min": "0.003",
        # Legacy epsilon actions bypass APF.  Keep the correction operator
        # identical for every transition/method; OU noise still explores the
        # raw proposal before applying APF.
        "--random-action-prob": "0.0",
        "--per-enabled": "1",
        "--per-replace": "0",
        "--per-uniform-mix": "0.40",
        "--per-td-weight": "0.80",
        "--per-reward-weight": "0.10",
        "--per-age-decay": "0.95",
        "--buffer-dtype": "fp32",
        "--reward-scale": "0.002",
        "--q-clip-value": "2000.0",
        "--critic-q-reg": "0.005",
        "--reward-clip-value": "-1500.0",
        "--max-force-magnitude": "80.0",
        "--goal-attraction": "26.0",
        "--lambda-1-base": "8.5",
        "--terrain-repulsion": "1600.0",
        "--agent-influence-range": "150.0",
        "--delta-k-att": "5.0",
        "--delta-lambda-1": "2.2",
        "--delta-k-rep": "600.0",
        "--delta-radius": "80.0",
        "--success-count-mode": "all",
        "--fairness-reweight-enabled": "0",
        "--fairness-window": "20",
        "--fairness-beta": "1.6",
        "--fairness-min-weight": "0.90",
        "--fairness-max-weight": "1.25",
        "--use-vectorization": "1",
        "--vectorized-rewards": "1",
        "--vectorized-observations": "1",
        "--amp-mode": "off",
        "--jit-compile": "1",
        "--obstacle-observation-mode": "risk_lite_v2",
        "--obstacle-risk-velocity-forward-weight": "4.0",
        "--obstacle-risk-goal-along-weight": "3.0",
    },
}

_REPORTING_CONTRACT = {
    "primary_metric": "team_success_rate",
    "official_split_reporting_policy": (
        "all_C0_C3_and_all_frozen_G1_splits_in_canonical_order_v1"
    ),
    "confirmatory_analysis": {
        "analysis_id": "primary_apf_contribution_on_C3_v1",
        "split": "test_C3",
        "proposed_method": "EC-MATD3-APF",
        "baseline_method": "EC-MATD3",
        "metric": "team_success_rate",
        "effect_orientation": "proposed_minus_baseline_higher_is_better",
        "confidence": 0.95,
        "decision_rule": (
            "oriented_improvement_bootstrap_95ci_lower_bound_gt_zero"
        ),
        "multiplicity_policy": (
            "single_predeclared_confirmatory_contrast"
        ),
    },
    "secondary_analysis_policy": (
        "effect_sizes_and_95ci_descriptive_no_binary_superiority_claims"
    ),
    "secondary_metrics": [
        "agent_spl",
        "collision_rate",
        "minimum_clearance_m",
        "energy",
        "control_effort",
    ],
    "uncertainty": "crossed_factor_paired_bootstrap_95ci",
    "in_distribution_design": {
        "independent_training_unit": "train_seed",
        "crossed_task_unit": "immutable_task_id",
        "pairing": "same_train_seed_same_task_across_methods",
    },
    "generalization_design": {
        "independent_training_unit": "train_seed",
        "independent_domain_unit": "held_out_terrain_seed",
        "task_unit": "immutable_task_id_nested_within_terrain",
        "pairing": "same_train_seed_same_terrain_same_task_across_methods",
        "require_all_frozen_held_out_terrains": True,
    },
    "mechanism_diagnostics": {
        "episode_schema": "paper_episode_v2",
        "scope": "descriptive_not_superiority_metric",
        "require_complete_action_step_coverage": True,
        "actual_intervention_threshold_l2": 1e-6,
        "fields": [
            "configured_action_force_ratio",
            "mean_raw_action_norm",
            "mean_executed_action_norm",
            "mean_action_intervention_norm",
            "apf_intervention_step_fraction",
            "mean_apf_force_norm",
        ],
    },
}

_EXECUTION_ENVIRONMENT_ALLOWLIST = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LC_ALL",
    "TZ",
    "TMPDIR",
    "CUDA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "CONDA_PREFIX",
)


def execution_environment_allowlist() -> list[str]:
    return list(_EXECUTION_ENVIRONMENT_ALLOWLIST)


def _validate_common_training_hyperparameter_contract() -> None:
    contract = _FORMAL_COMMON_TRAINING_HYPERPARAMETERS
    flags = dict(contract.get("launcher_flags") or {})
    environment = dict(contract.get("environment_variables") or {})
    scale = dict(contract.get("discount_time_scale") or {})
    try:
        gamma = float(flags["--gamma"])
        injected_gamma = float(environment["PAPER_GAMMA"])
        declared_gamma = float(scale["gamma"])
        simulation_dt = float(scale["simulation_dt_seconds"])
        horizon_steps = int(scale["effective_horizon_steps"])
        horizon_seconds = float(scale["effective_horizon_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PaperSchemaError(
            "formal discount time-scale contract is incomplete"
        ) from exc
    expected_horizon_steps = int(round(1.0 / (1.0 - gamma)))
    if (
        not 0.0 < gamma < 1.0
        or not math.isclose(
            gamma, injected_gamma, rel_tol=0.0, abs_tol=1e-12
        )
        or not math.isclose(gamma, declared_gamma, rel_tol=0.0, abs_tol=1e-12)
        or horizon_steps != expected_horizon_steps
        or not math.isclose(
            horizon_seconds,
            horizon_steps * simulation_dt,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            simulation_dt,
            float(paper_physics_contract()["simulation_dt"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise PaperSchemaError(
            "formal discount time-scale contract is internally inconsistent"
        )


def _python_runtime_descriptor(executable: str | Path) -> Dict[str, Any]:
    resolved = Path(executable).expanduser().resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise PaperSchemaError(
            f"paper Python executable is missing or not executable: {resolved}"
        )
    probe = (
        "import importlib.metadata as m,json,platform,sys;"
        "names=['numpy','scipy','tensorflow','gym','matplotlib','pandas','h5py'];"
        "versions={n:(m.version(n) if n in {d.metadata.get('Name','').lower() "
        "for d in m.distributions()} else None) for n in names};"
        "print(json.dumps({'python':platform.python_version(),"
        "'implementation':platform.python_implementation(),"
        "'executable':str(__import__('pathlib').Path(sys.executable).resolve()),"
        "'packages':versions},sort_keys=True))"
    )
    try:
        completed = subprocess.run(
            [str(resolved), "-c", probe],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        runtime = json.loads(completed.stdout.strip())
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise PaperSchemaError(
            f"failed to inspect paper Python runtime: {resolved}"
        ) from exc
    if Path(str(runtime.get("executable", ""))).resolve() != resolved:
        raise PaperSchemaError("paper Python runtime resolved to another executable")
    if not str((runtime.get("packages") or {}).get("numpy") or "").strip() or (
        not str((runtime.get("packages") or {}).get("tensorflow") or "").strip()
    ):
        raise PaperSchemaError(
            "paper Python runtime requires installed numpy and tensorflow"
        )
    return {
        "executable_artifact": fingerprint_artifact(resolved),
        "runtime": runtime,
    }


def _experiment_source_bundle_descriptor() -> Dict[str, Any]:
    repository_root = Path(__file__).resolve().parent
    artifacts = []
    for logical_name in _FORMAL_EXPERIMENT_SOURCE_LOGICAL_NAMES:
        source_path = repository_root / logical_name
        if not source_path.is_file():
            raise PaperSchemaError(
                f"formal experiment source is missing: {logical_name}"
            )
        artifact = fingerprint_artifact(source_path)
        artifact["logical_name"] = logical_name
        artifacts.append(artifact)
    descriptor = {
        "policy": "immutable_source_bundle_v1",
        "artifacts": artifacts,
    }
    descriptor["content_sha256"] = content_sha256(descriptor)
    return descriptor


_METHOD_SPECS: Dict[str, Dict[str, Any]] = {
    "EC-MATD3-APF": {
        "paper_role": "proposed",
        "require_validation_adequacy": True,
        "algorithm": "matd3",
        "use_weighted_reward": False,
        "environment": {
            "MATD3_USE_DUAL_Q": "false",
            "MATD3_USE_SEPARATED_GRADIENT": "false",
            "MATD3_USE_HYBRID_ACTOR_OBJECTIVE": "false",
            "MATD3_ACTION_SEMANTICS_MODE": "executed_control",
            "MATD3_RECONSTRUCT_CORRECTED_TARGET": "true",
            "ACTOR_UPDATE_DELAY": "2",
            "ENABLE_ACTION_CORRECTION": "true",
            "USE_TF_POTENTIAL_FIELD": "1",
            "USE_FR_FEATURE": "1",
            "USE_PF_FEATURE": "1",
            "ACTION_FORCE_RATIO": "0.50",
            "ACTION_FORCE_RATIO_SCHEDULE_PCT": (
                "0%:0.50,25%:0.48,50%:0.45,70%:0.40,85%:0.35,100%:0.32"
            ),
        },
    },
    "EC-MATD3": {
        "paper_role": "capacity_matched_apf_ablation",
        "require_validation_adequacy": False,
        "algorithm": "matd3",
        "use_weighted_reward": False,
        "environment": {
            "MATD3_USE_DUAL_Q": "false",
            "MATD3_USE_SEPARATED_GRADIENT": "false",
            "MATD3_USE_HYBRID_ACTOR_OBJECTIVE": "false",
            "MATD3_ACTION_SEMANTICS_MODE": "executed_control",
            "MATD3_RECONSTRUCT_CORRECTED_TARGET": "true",
            "ACTOR_UPDATE_DELAY": "2",
            "ENABLE_ACTION_CORRECTION": "false",
            "USE_TF_POTENTIAL_FIELD": "1",
            # Keep the same actor/critic conditioning branches as the proposed
            # method.  FR=0 makes both channels deterministic zero inputs and
            # disables the APF transform, so the ablation does not also change
            # network capacity.
            "USE_FR_FEATURE": "1",
            "USE_PF_FEATURE": "1",
            "ACTION_FORCE_RATIO": "0.0",
            "ACTION_FORCE_RATIO_SCHEDULE_PCT": "DISABLED",
        },
    },
    "DualQ-MATD3-APF": {
        "paper_role": "critic_semantics_ablation",
        "require_validation_adequacy": False,
        "algorithm": "matd3",
        "use_weighted_reward": False,
        "environment": {
            "MATD3_USE_DUAL_Q": "true",
            "MATD3_USE_SEPARATED_GRADIENT": "true",
            "MATD3_USE_HYBRID_ACTOR_OBJECTIVE": "false",
            "MATD3_ACTION_SEMANTICS_MODE": "dual",
            "MATD3_RECONSTRUCT_CORRECTED_TARGET": "true",
            "ACTOR_UPDATE_DELAY": "2",
            "ENABLE_ACTION_CORRECTION": "true",
            "USE_TF_POTENTIAL_FIELD": "1",
            "USE_FR_FEATURE": "1",
            "USE_PF_FEATURE": "1",
            "ACTION_FORCE_RATIO": "0.50",
            "ACTION_FORCE_RATIO_SCHEDULE_PCT": (
                "0%:0.50,25%:0.48,50%:0.45,70%:0.40,85%:0.35,100%:0.32"
            ),
        },
    },
    "MADDPG-APF": {
        "paper_role": "algorithm_baseline",
        "require_validation_adequacy": False,
        "algorithm": "maddpg",
        "use_weighted_reward": False,
        "environment": {
            "MADDPG_USE_DUAL_Q": "false",
            "MADDPG_USE_SEPARATED_GRADIENT": "false",
            "MADDPG_ACTION_SEMANTICS_MODE": "executed_control",
            "ACTOR_UPDATE_DELAY": "1",
            "ENABLE_ACTION_CORRECTION": "true",
            "USE_TF_POTENTIAL_FIELD": "1",
            "USE_FR_FEATURE": "1",
            "USE_PF_FEATURE": "1",
            "ACTION_FORCE_RATIO": "0.50",
            "ACTION_FORCE_RATIO_SCHEDULE_PCT": (
                "0%:0.50,25%:0.48,50%:0.45,70%:0.40,85%:0.35,100%:0.32"
            ),
        },
    },
}


# Natural raw-action control: the same execution operator and scalar twin
# architecture, with APF absorbed into the transition kernel learned by Q_raw.
_METHOD_SPECS["RawOnly-MATD3-APF"] = deepcopy(_METHOD_SPECS["EC-MATD3-APF"])
_METHOD_SPECS["RawOnly-MATD3-APF"].update(
    paper_role="raw_action_critic_control", require_validation_adequacy=False,
)
_METHOD_SPECS["RawOnly-MATD3-APF"]["environment"].update({
    "MATD3_ACTION_SEMANTICS_MODE": "raw_only",
    "MATD3_RECONSTRUCT_CORRECTED_TARGET": "false",
})


def available_method_ids() -> list[str]:
    return list(_METHOD_SPECS)


def method_spec(method_id: str) -> Dict[str, Any]:
    normalized = str(method_id).strip()
    if normalized not in _METHOD_SPECS:
        raise PaperSchemaError(
            f"unknown paper method={normalized!r}; available={available_method_ids()}"
        )
    return deepcopy(_METHOD_SPECS[normalized])


def runtime_method_environment(
    method: Mapping[str, Any],
) -> Dict[str, str]:
    """Resolve launcher input sentinels to their post-launch runtime values."""
    environment = {
        str(key): str(value)
        for key, value in dict(method.get("environment") or {}).items()
    }
    if environment.get("ACTION_FORCE_RATIO_SCHEDULE_PCT") == "DISABLED":
        environment["ACTION_FORCE_RATIO_SCHEDULE_PCT"] = ""
    return environment


def _normalize_unique_ints(values: Iterable[int], field: str) -> list[int]:
    normalized = [int(value) for value in values]
    if not normalized or len(normalized) != len(set(normalized)):
        raise PaperSchemaError(f"{field} must contain unique integers")
    return normalized


def _build_stage_iteration_budget(training_iterations: int) -> Dict[str, int]:
    total = int(training_iterations)
    if total < 40:
        raise PaperSchemaError(
            "formal curriculum requires at least 40 synchronous iterations"
        )
    budget = {
        "C0": max(1, int(round(total * 0.12))),
        "C1": max(1, int(round(total * 0.16))),
        "C2": max(1, int(round(total * 0.24))),
    }
    budget["C3"] = total - sum(budget.values())
    if budget["C3"] <= 0:
        raise PaperSchemaError("formal curriculum leaves no iterations for C3")
    return budget


def _validate_formal_task_bank(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = dict(manifest.get("metadata") or {})
    generation_spec = dict(metadata.get("generation_spec") or {})
    if generation_spec.get("schema_version") != (
        _FORMAL_TASK_BANK_GENERATION_SCHEMA_VERSION
    ) or generation_spec.get("cache_kind") != "generation_spec":
        raise PaperSchemaError("formal task bank lacks its complete generation spec")
    expected_generation_spec_hash = content_sha256(
        {
            key: value
            for key, value in generation_spec.items()
            if key != "content_sha256"
        }
    )
    if generation_spec.get("content_sha256") != expected_generation_spec_hash:
        raise PaperSchemaError("formal task-bank generation spec hash mismatch")
    if metadata.get("generation_spec_content_sha256") != (
        expected_generation_spec_hash
    ):
        raise PaperSchemaError(
            "formal task-bank metadata/generation spec hash mismatch"
        )
    if str(generation_spec.get("bank_id", "")) != str(
        manifest.get("bank_id", "")
    ):
        raise PaperSchemaError("formal task-bank generation spec bank ID mismatch")
    generation_sources = list(metadata.get("generation_source_artifacts") or [])
    if not generation_sources:
        raise PaperSchemaError("formal task bank lacks generation source provenance")
    if generation_sources != list(generation_spec.get("source_artifacts") or []):
        raise PaperSchemaError(
            "formal task-bank source provenance differs from generation spec"
        )
    logical_names = []
    for artifact in generation_sources:
        if not isinstance(artifact, Mapping):
            raise PaperSchemaError("formal task bank source artifact is invalid")
        logical_name = str(artifact.get("logical_name", "")).strip()
        actual = fingerprint_artifact(artifact.get("path"))
        if not logical_name or actual.get("sha256") != artifact.get("sha256"):
            raise PaperSchemaError(
                f"formal task-bank generation source changed: {logical_name!r}"
            )
        logical_names.append(logical_name)
    if len(logical_names) != len(set(logical_names)):
        raise PaperSchemaError("formal task-bank generation source names repeat")
    if set(logical_names) != _FORMAL_TASK_BANK_SOURCE_LOGICAL_NAMES:
        raise PaperSchemaError(
            "formal task-bank generation source set is incomplete or unexpected"
        )
    feasibility_errors = validate_dynamic_feasibility_audit(manifest)
    if feasibility_errors:
        raise PaperSchemaError(
            "formal task bank failed dynamic feasibility audit: "
            + "; ".join(feasibility_errors)
        )
    splits = dict(manifest.get("splits") or {})
    base_terrain_seed = int(generation_spec.get("base_terrain_seed"))
    held_out_terrain_seeds = [
        int(seed) for seed in generation_spec.get("held_out_terrain_seeds") or []
    ]
    if (
        len(held_out_terrain_seeds) != 4
        or len(set(held_out_terrain_seeds)) != 4
        or base_terrain_seed in held_out_terrain_seeds
    ):
        raise PaperSchemaError(
            "formal task bank requires exactly four unique held-out terrain seeds"
        )
    if int(metadata.get("base_terrain_seed")) != base_terrain_seed or [
        int(seed) for seed in metadata.get("held_out_terrain_seeds") or []
    ] != held_out_terrain_seeds:
        raise PaperSchemaError(
            "formal task-bank terrain seed metadata differs from generation spec"
        )
    expected_curriculum = build_paper_curriculum(base_terrain_seed)
    expected_generalization = build_paper_generalization_suite(
        held_out_terrain_seeds
    )
    if generation_spec.get("curriculum") != expected_curriculum:
        raise PaperSchemaError("formal task-bank curriculum configuration drift")
    if generation_spec.get("generalization") != expected_generalization:
        raise PaperSchemaError("formal task-bank generalization configuration drift")
    expected_task_counts = {
        "train_per_difficulty": 256,
        "validation_per_difficulty": 64,
        "test_per_difficulty": 128,
        "generalization_per_terrain_seed": 64,
    }
    if generation_spec.get("task_counts") != expected_task_counts or (
        metadata.get("task_counts") != expected_task_counts
    ):
        raise PaperSchemaError("formal task-bank generation task counts mismatch")
    expected_counts: Dict[str, int] = {}
    expected_runtime_configs: Dict[str, Dict[str, Any]] = {}
    expected_roles: Dict[str, str] = {}
    for stage in expected_curriculum:
        difficulty = str(stage["difficulty_bin"])
        expected_counts[f"train_{difficulty}"] = 256
        expected_counts[f"validation_{difficulty}"] = 64
        expected_counts[f"test_{difficulty}"] = 128
        for prefix, role in (
            ("train", "train"),
            ("validation", "validation"),
            ("test", "test_in_distribution"),
        ):
            split_name = f"{prefix}_{difficulty}"
            expected_runtime_configs[split_name] = stage
            expected_roles[split_name] = role
    for suite in expected_generalization:
        split_name = f"test_{suite['suite_id']}"
        expected_counts[split_name] = 64
        expected_runtime_configs[split_name] = suite
        expected_roles[split_name] = "test_generalization"
    if set(splits) != set(expected_counts):
        raise PaperSchemaError(
            "formal task-bank split set mismatch: "
            f"expected={sorted(expected_counts)}, actual={sorted(splits)}"
        )
    for split_name, expected_count in expected_counts.items():
        actual_count = int(splits[split_name].get("task_count", -1))
        if actual_count != expected_count:
            raise PaperSchemaError(
                f"formal split={split_name} requires {expected_count} tasks, "
                f"got={actual_count}"
            )
        if splits[split_name].get("role") != expected_roles[split_name]:
            raise PaperSchemaError(
                f"formal split={split_name} role mismatch"
            )
        if splits[split_name].get("runtime_config") != (
            expected_runtime_configs[split_name]
        ):
            raise PaperSchemaError(
                f"formal split={split_name} runtime configuration drift"
            )
        split_generation = dict(
            (metadata.get("generation_acceptance_audit") or {}).get(
                split_name
            )
            or {}
        )
        accepted = int(split_generation.get("accepted", -1))
        examined = int(split_generation.get("candidates_examined", -1))
        rejected = int(split_generation.get("rejected", -1))
        if (
            accepted != expected_count
            or examined < accepted
            or rejected != examined - accepted
        ):
            raise PaperSchemaError(
                f"formal split={split_name} generation acceptance audit mismatch"
            )
        expected_rate = accepted / examined
        if abs(
            float(split_generation.get("acceptance_rate", -1.0))
            - expected_rate
        ) > 1e-12:
            raise PaperSchemaError(
                f"formal split={split_name} generation acceptance rate mismatch"
            )
        reason_incidence = split_generation.get(
            "rejection_reason_incidence"
        )
        if not isinstance(reason_incidence, Mapping) or any(
            int(value) < 0 for value in reason_incidence.values()
        ):
            raise PaperSchemaError(
                f"formal split={split_name} rejection reason audit is invalid"
            )
    generalization_splits = sorted(
        split_name
        for split_name, split in splits.items()
        if split.get("role") == "test_generalization"
    )
    if len(generalization_splits) != 4:
        raise PaperSchemaError(
            "formal task bank requires exactly four held-out terrain splits"
        )
    return {
        "expected_in_distribution_counts": expected_counts,
        "generalization_splits": generalization_splits,
        "generalization_task_count_per_split": 64,
    }


def _maximum_training_deadline_steps(manifest: Mapping[str, Any]) -> int:
    values = []
    for difficulty in ("C0", "C1", "C2", "C3"):
        split = dict(
            (manifest.get("splits") or {}).get(f"train_{difficulty}") or {}
        )
        summary = dict(split.get("deadline_summary") or {})
        value = int(summary.get("maximum_deadline_steps", 0) or 0)
        if value <= 0:
            raise PaperSchemaError(
                f"formal train_{difficulty} lacks a positive deadline summary"
            )
        values.append(value)
    return max(values)


def build_protocol(
    *,
    protocol_id: str,
    task_bank_manifest: str | Path,
    method_ids: Iterable[str],
    train_seeds: Iterable[int],
    training_iterations: int = 800,
    training_num_envs: int = 8,
    batch_size: int = 1024,
    validation_eval_seed: int = 1701,
    official_test_eval_seed: int = 2901,
    training_python_executable: str | Path = sys.executable,
    evaluation_python_executable: str | Path | None = None,
) -> Dict[str, Any]:
    _validate_common_training_hyperparameter_contract()
    protocol_id = str(protocol_id).strip()
    if not protocol_id:
        raise PaperSchemaError("protocol_id cannot be empty")
    methods = [str(value).strip() for value in method_ids]
    if not methods or len(methods) != len(set(methods)):
        raise PaperSchemaError("method_ids must be unique and non-empty")
    if "EC-MATD3-APF" not in methods or len(methods) < 2:
        raise PaperSchemaError(
            "formal paper protocol requires the proposed method and at least one baseline"
        )
    method_payload = {method: method_spec(method) for method in methods}
    seeds = _normalize_unique_ints(train_seeds, "train_seeds")
    if len(seeds) < 5:
        raise PaperSchemaError("formal paper protocol requires at least five train seeds")
    if int(training_iterations) <= 0 or int(training_num_envs) <= 0:
        raise PaperSchemaError("training iterations and num_envs must be positive")
    if int(batch_size) <= 0:
        raise PaperSchemaError("batch_size must be positive")
    stage_iterations = _build_stage_iteration_budget(training_iterations)
    training_python = _python_runtime_descriptor(training_python_executable)
    if evaluation_python_executable is None or Path(
        evaluation_python_executable
    ).expanduser().resolve() == Path(training_python_executable).expanduser().resolve():
        evaluation_python = deepcopy(training_python)
    else:
        evaluation_python = _python_runtime_descriptor(
            evaluation_python_executable
        )

    task_bank_path = Path(task_bank_manifest).expanduser().resolve()
    if task_bank_path.is_dir():
        task_bank_path = task_bank_path / "task_bank_manifest.json"
    task_bank = load_task_bank(task_bank_path)
    split_contract = _validate_formal_task_bank(task_bank)
    maximum_training_deadline_steps = _maximum_training_deadline_steps(
        task_bank
    )
    formal_episode_length = int(
        _FORMAL_COMMON_TRAINING_HYPERPARAMETERS["launcher_flags"][
            "--episode-length"
        ]
    )
    if maximum_training_deadline_steps > formal_episode_length:
        raise PaperSchemaError(
            "formal launcher episode length is shorter than a training task "
            f"deadline: episode_length={formal_episode_length}, "
            f"maximum_deadline_steps={maximum_training_deadline_steps}"
        )
    task_bank_artifact = fingerprint_artifact(task_bank_path.parent)
    task_bank_artifact.update(
        {
            "manifest_path": str(task_bank_path),
            "bank_id": str(task_bank["bank_id"]),
            "bank_content_sha256": str(task_bank["content_sha256"]),
        }
    )
    curriculum = build_paper_curriculum(
        int((task_bank.get("metadata") or {}).get("base_terrain_seed"))
    )
    candidate_counts = {}
    stage_task_cycles = {}
    for stage in curriculum:
        difficulty = str(stage["difficulty_bin"])
        stage_trajectories = int(
            stage_iterations[difficulty] * int(training_num_envs)
        )
        train_split = f"train_{difficulty}"
        train_task_count = int(task_bank["splits"][train_split]["task_count"])
        if stage_trajectories % train_task_count != 0:
            raise PaperSchemaError(
                "formal stage trajectory budget must contain complete immutable "
                f"task-bank cycles: difficulty={difficulty}, "
                f"trajectories={stage_trajectories}, tasks={train_task_count}"
            )
        stage_task_cycles[difficulty] = int(
            stage_trajectories // train_task_count
        )
        interval = int(stage["candidate_checkpoint_interval_trajectories"])
        candidate_counts[difficulty] = int(
            (stage_trajectories + interval - 1) // interval
        )
    if any(count < 2 for count in candidate_counts.values()):
        raise PaperSchemaError(
            "formal training budget must yield at least two predeclared "
            "checkpoint candidates per difficulty"
        )
    payload: Dict[str, Any] = {
        "schema_version": PAPER_EXPERIMENT_PROTOCOL_SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "task_bank": task_bank_artifact,
        "methods": method_payload,
        "train_seeds": seeds,
        "execution_environment": {
            "policy": "clean_allowlist_v2",
            "inherited_variable_names": execution_environment_allowlist(),
            "scientific_overrides_source": "immutable_method_spec_and_pipeline",
            "training_device_requirement": {
                "policy": "tensorflow_gpu_required_v1",
                "minimum_physical_gpus": 1,
                "minimum_logical_gpus": 1,
            },
            "evaluation_device_requirement": deepcopy(
                FORMAL_EVALUATION_DEVICE_REQUIREMENT
            ),
            "python_runtimes": {
                "training": training_python,
                "evaluation": evaluation_python,
            },
            "source_bundle": _experiment_source_bundle_descriptor(),
        },
        "training": {
            "synchronous_iterations": int(training_iterations),
            "parallel_environments": int(training_num_envs),
            "environment_trajectories": int(
                training_iterations * training_num_envs
            ),
            "stage_transition_policy": "fixed_iteration_budget_v1",
            "interruption_policy": FORMAL_INTERRUPTION_POLICY,
            "task_sampling_policy": (
                "deterministic_bank_order_complete_cycles_v1"
            ),
            "checkpoint_candidate_policy": (
                "predeclared_stage_milestones_only_no_training_metric_best_v1"
            ),
            "common_hyperparameter_contract": deepcopy(
                _FORMAL_COMMON_TRAINING_HYPERPARAMETERS
            ),
            "maximum_training_task_deadline_steps": int(
                maximum_training_deadline_steps
            ),
            "stage_transition_runtime": deepcopy(
                _FORMAL_STAGE_TRANSITION_RUNTIME
            ),
            "physics_contract": paper_physics_contract(),
            "task_semantics_contract": paper_task_semantics_contract(),
            "stage_synchronous_iterations": stage_iterations,
            "stage_environment_trajectories": {
                difficulty: int(iterations * training_num_envs)
                for difficulty, iterations in stage_iterations.items()
            },
            "complete_task_bank_cycles_per_stage": stage_task_cycles,
            "batch_size": int(batch_size),
            "curriculum_difficulties": [
                stage["difficulty_bin"] for stage in curriculum
            ],
            "candidate_checkpoint_intervals_trajectories": {
                stage["difficulty_bin"]: int(
                    stage["candidate_checkpoint_interval_trajectories"]
                )
                for stage in curriculum
            },
            "expected_checkpoint_candidates_per_difficulty": candidate_counts,
        },
        "validation": {
            "data_role": "validation_only_for_checkpoint_selection",
            "selection_training_difficulty_bin": FORMAL_SELECTION_DIFFICULTY,
            "selection_split": f"validation_{FORMAL_SELECTION_DIFFICULTY}",
            "selection_task_count": 64,
            "early_stage_candidates_role": "training_diagnostics_only",
            "eval_seed": int(validation_eval_seed),
            "evaluate_each_task_exactly_once": True,
            "minimum_distinct_candidates": 2,
            "action_force_ratio_policy": "checkpoint_specific_no_override",
        },
        "official_test": {
            "data_role": "test_only_after_validation_selection",
            "checkpoint_policy": FORMAL_CHECKPOINT_POLICY,
            "selected_training_difficulty_bin": FORMAL_SELECTION_DIFFICULTY,
            "tasks_per_in_distribution_difficulty": 128,
            "tasks_per_generalization_terrain": 64,
            "eval_seed": int(official_test_eval_seed),
            "evaluate_each_task_exactly_once": True,
            "forbid_checkpoint_reselection": True,
            "action_force_ratio_policy": "checkpoint_specific_no_override",
        },
        "split_contract": split_contract,
        "reporting": deepcopy(_REPORTING_CONTRACT),
    }
    payload["content_sha256"] = content_sha256(payload)
    return payload


def load_protocol(path: str | Path) -> Dict[str, Any]:
    _validate_common_training_hyperparameter_contract()
    protocol_path = Path(path).expanduser().resolve()
    with protocol_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema_version") != PAPER_EXPERIMENT_PROTOCOL_SCHEMA_VERSION:
        raise PaperSchemaError("paper experiment protocol schema mismatch")
    expected_hash = content_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    if payload.get("content_sha256") != expected_hash:
        raise PaperSchemaError("paper experiment protocol content hash mismatch")
    task_bank = dict(payload.get("task_bank") or {})
    actual_task_bank = fingerprint_artifact(task_bank.get("path"))
    if actual_task_bank.get("sha256") != task_bank.get("sha256"):
        raise PaperSchemaError("paper experiment task-bank artifact changed")
    manifest = load_task_bank(task_bank.get("manifest_path"))
    if manifest.get("content_sha256") != task_bank.get("bank_content_sha256"):
        raise PaperSchemaError("paper experiment task-bank manifest changed")
    _validate_formal_task_bank(manifest)
    for method_id, recorded_spec in dict(payload.get("methods") or {}).items():
        if recorded_spec != method_spec(method_id):
            raise PaperSchemaError(f"paper method specification drift: {method_id}")
    seeds = _normalize_unique_ints(payload.get("train_seeds") or [], "train_seeds")
    if len(seeds) < 5:
        raise PaperSchemaError("paper protocol no longer has five train seeds")
    execution_environment = dict(payload.get("execution_environment") or {})
    expected_execution_fields = {
        "policy": "clean_allowlist_v2",
        "inherited_variable_names": execution_environment_allowlist(),
        "scientific_overrides_source": "immutable_method_spec_and_pipeline",
        "training_device_requirement": {
            "policy": "tensorflow_gpu_required_v1",
            "minimum_physical_gpus": 1,
            "minimum_logical_gpus": 1,
        },
        "evaluation_device_requirement": deepcopy(
            FORMAL_EVALUATION_DEVICE_REQUIREMENT
        ),
    }
    if any(
        execution_environment.get(key) != value
        for key, value in expected_execution_fields.items()
    ):
        raise PaperSchemaError("paper execution environment policy mismatch")
    python_runtimes = dict(
        execution_environment.get("python_runtimes") or {}
    )
    if set(python_runtimes) != {"training", "evaluation"}:
        raise PaperSchemaError("paper Python runtime contract is incomplete")
    for role in ("training", "evaluation"):
        recorded_runtime = dict(python_runtimes[role] or {})
        executable = dict(
            recorded_runtime.get("executable_artifact") or {}
        ).get("path")
        actual_runtime = _python_runtime_descriptor(executable)
        if actual_runtime != recorded_runtime:
            raise PaperSchemaError(
                f"paper {role} Python runtime or package versions changed"
            )
    recorded_source_bundle = dict(
        execution_environment.get("source_bundle") or {}
    )
    if recorded_source_bundle != _experiment_source_bundle_descriptor():
        raise PaperSchemaError("paper experiment source bundle changed")
    training = dict(payload.get("training") or {})
    if training.get("stage_transition_policy") != "fixed_iteration_budget_v1":
        raise PaperSchemaError("paper protocol stage transition policy mismatch")
    if training.get("interruption_policy") != FORMAL_INTERRUPTION_POLICY:
        raise PaperSchemaError("paper protocol interruption policy mismatch")
    if training.get("task_sampling_policy") != (
        "deterministic_bank_order_complete_cycles_v1"
    ):
        raise PaperSchemaError("paper protocol task sampling policy mismatch")
    if training.get("checkpoint_candidate_policy") != (
        "predeclared_stage_milestones_only_no_training_metric_best_v1"
    ):
        raise PaperSchemaError(
            "paper checkpoint candidate policy mismatch"
        )
    if training.get("common_hyperparameter_contract") != (
        _FORMAL_COMMON_TRAINING_HYPERPARAMETERS
    ):
        raise PaperSchemaError(
            "paper common training hyperparameter contract mismatch"
        )
    maximum_training_deadline_steps = _maximum_training_deadline_steps(
        manifest
    )
    if int(training.get("maximum_training_task_deadline_steps", -1)) != (
        maximum_training_deadline_steps
    ):
        raise PaperSchemaError(
            "paper maximum training task deadline contract mismatch"
        )
    formal_episode_length = int(
        _FORMAL_COMMON_TRAINING_HYPERPARAMETERS["launcher_flags"][
            "--episode-length"
        ]
    )
    if maximum_training_deadline_steps > formal_episode_length:
        raise PaperSchemaError(
            "paper episode length no longer covers all training deadlines"
        )
    if training.get("stage_transition_runtime") != (
        _FORMAL_STAGE_TRANSITION_RUNTIME
    ):
        raise PaperSchemaError("paper stage transition runtime contract mismatch")
    if training.get("physics_contract") != paper_physics_contract():
        raise PaperSchemaError("paper training physics contract mismatch")
    if training.get("task_semantics_contract") != paper_task_semantics_contract():
        raise PaperSchemaError("paper training task semantics contract mismatch")
    if payload.get("reporting") != _REPORTING_CONTRACT:
        raise PaperSchemaError("paper reporting/statistical contract mismatch")
    stage_iterations = dict(training.get("stage_synchronous_iterations") or {})
    if set(stage_iterations) != {"C0", "C1", "C2", "C3"}:
        raise PaperSchemaError("paper protocol stage iteration budget is incomplete")
    normalized_stage_iterations = {
        key: int(value) for key, value in stage_iterations.items()
    }
    if any(value <= 0 for value in normalized_stage_iterations.values()):
        raise PaperSchemaError("paper protocol stage iterations must be positive")
    if sum(normalized_stage_iterations.values()) != int(
        training.get("synchronous_iterations", -1)
    ):
        raise PaperSchemaError("paper protocol stage iterations do not sum to total")
    expected_stage_trajectories = {
        key: int(value * int(training.get("parallel_environments", 0)))
        for key, value in normalized_stage_iterations.items()
    }
    if training.get("stage_environment_trajectories") != (
        expected_stage_trajectories
    ):
        raise PaperSchemaError("paper protocol stage trajectory budget mismatch")
    expected_task_cycles = {}
    for difficulty, trajectories in expected_stage_trajectories.items():
        task_count = int(
            manifest["splits"][f"train_{difficulty}"]["task_count"]
        )
        if trajectories % task_count != 0:
            raise PaperSchemaError(
                "paper protocol stage does not cover complete task-bank cycles"
            )
        expected_task_cycles[difficulty] = trajectories // task_count
    if training.get("complete_task_bank_cycles_per_stage") != (
        expected_task_cycles
    ):
        raise PaperSchemaError("paper protocol task-cycle budget mismatch")
    current_curriculum = build_paper_curriculum(
        int((manifest.get("metadata") or {}).get("base_terrain_seed"))
    )
    expected_difficulties = [
        str(stage["difficulty_bin"]) for stage in current_curriculum
    ]
    if training.get("curriculum_difficulties") != expected_difficulties:
        raise PaperSchemaError("paper protocol curriculum difficulty order mismatch")
    expected_intervals = {
        str(stage["difficulty_bin"]): int(
            stage["candidate_checkpoint_interval_trajectories"]
        )
        for stage in current_curriculum
    }
    if training.get("candidate_checkpoint_intervals_trajectories") != (
        expected_intervals
    ):
        raise PaperSchemaError("paper protocol candidate checkpoint intervals drift")
    for evaluation_role in ("validation", "official_test"):
        if dict(payload.get(evaluation_role) or {}).get(
            "action_force_ratio_policy"
        ) != "checkpoint_specific_no_override":
            raise PaperSchemaError(
                f"paper {evaluation_role} action force ratio policy mismatch"
            )
    validation = dict(payload.get("validation") or {})
    expected_validation_selection = {
        "selection_training_difficulty_bin": FORMAL_SELECTION_DIFFICULTY,
        "selection_split": f"validation_{FORMAL_SELECTION_DIFFICULTY}",
        "selection_task_count": 64,
        "early_stage_candidates_role": "training_diagnostics_only",
    }
    if any(
        validation.get(key) != value
        for key, value in expected_validation_selection.items()
    ):
        raise PaperSchemaError(
            "paper final-checkpoint validation selection policy mismatch"
        )
    official_test = dict(payload.get("official_test") or {})
    if official_test.get("checkpoint_policy") != (
        FORMAL_CHECKPOINT_POLICY
    ) or official_test.get("selected_training_difficulty_bin") != (
        FORMAL_SELECTION_DIFFICULTY
    ):
        raise PaperSchemaError(
            "paper official-test checkpoint policy mismatch"
        )
    expected_candidate_counts = {
        difficulty: int(
            (expected_stage_trajectories[difficulty] + interval - 1)
            // interval
        )
        for difficulty, interval in expected_intervals.items()
    }
    if any(count < 2 for count in expected_candidate_counts.values()) or (
        training.get("expected_checkpoint_candidates_per_difficulty")
        != expected_candidate_counts
    ):
        raise PaperSchemaError("paper protocol checkpoint candidate counts mismatch")
    return dict(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--task-bank-manifest", required=True)
    parser.add_argument(
        "--methods", default=",".join(available_method_ids())
    )
    parser.add_argument("--train-seeds", default="101,202,303,404,505")
    parser.add_argument("--training-iterations", type=int, default=800)
    parser.add_argument("--training-num-envs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--validation-eval-seed", type=int, default=1701)
    parser.add_argument("--official-test-eval-seed", type=int, default=2901)
    parser.add_argument(
        "--training-python-executable", default=sys.executable
    )
    parser.add_argument(
        "--evaluation-python-executable", default=""
    )
    args = parser.parse_args()
    destination = Path(args.output).expanduser().resolve()
    if destination.exists():
        raise PaperSchemaError(
            f"refusing to overwrite immutable protocol: {destination}"
        )
    payload = build_protocol(
        protocol_id=args.protocol_id,
        task_bank_manifest=args.task_bank_manifest,
        method_ids=[value.strip() for value in args.methods.split(",") if value.strip()],
        train_seeds=[int(value.strip()) for value in args.train_seeds.split(",") if value.strip()],
        training_iterations=args.training_iterations,
        training_num_envs=args.training_num_envs,
        batch_size=args.batch_size,
        validation_eval_seed=args.validation_eval_seed,
        official_test_eval_seed=args.official_test_eval_seed,
        training_python_executable=args.training_python_executable,
        evaluation_python_executable=(
            args.evaluation_python_executable or None
        ),
    )
    atomic_write_json(destination, payload)
    print(
        f"paper protocol written: {destination} hash={payload['content_sha256']}"
    )


if __name__ == "__main__":
    main()
