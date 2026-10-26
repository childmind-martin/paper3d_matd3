#!/usr/bin/env python3
"""Execute the frozen train -> validation -> select -> test paper workflow."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, Iterable, Mapping

from paper_checkpoint_selection import select_checkpoint, validate_selection_result
from paper_experiment_protocol import (
    FORMAL_EVALUATION_DEVICE_REQUIREMENT,
    FORMAL_INTERRUPTION_POLICY,
    execution_environment_allowlist,
    load_protocol,
    runtime_method_environment,
)
from paper_experiment_schema import (
    PaperSchemaError,
    atomic_write_json,
    content_sha256,
    fingerprint_artifact,
)
from paper_task_bank import load_task_bank


TRAINING_RECORD_SCHEMA = "paper_training_run_record_v3"
PREFLIGHT_RECORD_SCHEMA = "paper_training_preflight_record_v1"
VALIDATION_INDEX_SCHEMA = "paper_validation_run_index_v1"
OFFICIAL_TEST_RECORD_SCHEMA = "paper_official_test_run_record_v1"


def _component(value: Any) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    if not normalized:
        raise PaperSchemaError(f"unsafe empty path component from {value!r}")
    return normalized


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise PaperSchemaError(f"refusing to overwrite immutable record: {path}")
    atomic_write_json(path, payload)


def _load_hashed_json(path: str | Path, schema_version: str) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema_version") != schema_version:
        raise PaperSchemaError(
            f"schema mismatch for {resolved}: {payload.get('schema_version')!r}"
        )
    expected = content_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    if payload.get("content_sha256") != expected:
        raise PaperSchemaError(f"content hash mismatch for {resolved}")
    return payload


def _validate_identity(
    protocol: Mapping[str, Any], method_id: str, train_seed: int
) -> Dict[str, Any]:
    method_id = str(method_id).strip()
    train_seed = int(train_seed)
    methods = dict(protocol.get("methods") or {})
    if method_id not in methods:
        raise PaperSchemaError(
            f"method={method_id!r} is absent from immutable protocol"
        )
    if train_seed not in [int(value) for value in protocol.get("train_seeds") or []]:
        raise PaperSchemaError(
            f"train_seed={train_seed} is absent from immutable protocol"
        )
    return dict(methods[method_id])


def _experiment_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _clean_subprocess_environment(
    protocol: Mapping[str, Any], *, random_seed: int
) -> Dict[str, str]:
    execution_environment = dict(protocol.get("execution_environment") or {})
    if execution_environment.get("policy") != "clean_allowlist_v2":
        raise PaperSchemaError("paper execution environment policy mismatch")
    recorded_names = list(
        execution_environment.get("inherited_variable_names") or []
    )
    expected_names = execution_environment_allowlist()
    if recorded_names != expected_names:
        raise PaperSchemaError("paper execution environment allowlist mismatch")
    environment = {
        name: os.environ[name]
        for name in recorded_names
        if name in os.environ and str(os.environ[name])
    }
    environment["PYTHONHASHSEED"] = str(int(random_seed))
    return environment


def _physics_environment(protocol: Mapping[str, Any]) -> Dict[str, str]:
    physics = dict((protocol.get("training") or {}).get("physics_contract") or {})
    environment_names = {
        "simulation_dt": "SIMULATION_DT",
        "agent_size": "AGENT_SIZE",
        "agent_max_speed": "AGENT_MAX_SPEED",
        "agent_accel": "AGENT_ACCEL",
        "gravity": "GRAVITY",
        "control_accel_gain": "CONTROL_ACCEL_GAIN",
        "damping": "DAMPING",
        "action_range_x": "ACTION_RANGE_X",
        "action_range_y": "ACTION_RANGE_Y",
        "action_range_z": "ACTION_RANGE_Z",
        "z_action_bias": "Z_ACTION_BIAS",
        "quadrotor_attitude_response_time": "QUADROTOR_ATTITUDE_RESPONSE_TIME",
        "quadrotor_psi_cmd": "QUADROTOR_PSI_CMD",
    }
    missing = [field for field in environment_names if field not in physics]
    if "use_quadrotor_dynamics" not in physics:
        missing.append("use_quadrotor_dynamics")
    if missing:
        raise PaperSchemaError(
            f"paper physics contract is incomplete: missing={missing}"
        )
    resolved = {
        environment_name: str(physics[field])
        for field, environment_name in environment_names.items()
    }
    resolved["USE_QUADROTOR_DYNAMICS"] = (
        "1" if bool(physics["use_quadrotor_dynamics"]) else "0"
    )
    return resolved


def _task_semantics_environment(protocol: Mapping[str, Any]) -> Dict[str, str]:
    semantics = dict(
        (protocol.get("training") or {}).get("task_semantics_contract") or {}
    )
    required = {
        "contract_id",
        "success_distance_threshold",
        "collision_distance_threshold",
        "reference_surface_clearance",
    }
    missing = sorted(required - set(semantics))
    if missing:
        raise PaperSchemaError(
            f"paper task semantics contract is incomplete: missing={missing}"
        )
    return {
        "SUCCESS_DISTANCE_THRESHOLD": str(
            semantics["success_distance_threshold"]
        ),
        "COLLISION_DISTANCE_THRESHOLD": str(
            semantics["collision_distance_threshold"]
        ),
        "PAPER_TASK_REFERENCE_SURFACE_CLEARANCE": str(
            semantics["reference_surface_clearance"]
        ),
    }


def _training_device_environment(
    protocol: Mapping[str, Any]
) -> Dict[str, str]:
    requirement = dict(
        (protocol.get("execution_environment") or {}).get(
            "training_device_requirement"
        )
        or {}
    )
    expected = {
        "policy": "tensorflow_gpu_required_v1",
        "minimum_physical_gpus": 1,
        "minimum_logical_gpus": 1,
    }
    if requirement != expected:
        raise PaperSchemaError("paper training device requirement mismatch")
    return {"MATD3_REQUIRE_GPU": "1"}


def _evaluation_device_environment(
    protocol: Mapping[str, Any],
) -> Dict[str, str]:
    requirement = dict(
        (protocol.get("execution_environment") or {}).get(
            "evaluation_device_requirement"
        )
        or {}
    )
    if requirement != FORMAL_EVALUATION_DEVICE_REQUIREMENT:
        raise PaperSchemaError("paper evaluation device requirement mismatch")
    return {
        "CUDA_VISIBLE_DEVICES": requirement["cuda_visible_devices"],
        "TF_DETERMINISTIC_OPS": requirement["tf_deterministic_ops"],
        "TF_ENABLE_ONEDNN_OPTS": requirement["tf_enable_onednn_opts"],
        "TF_NUM_INTRAOP_THREADS": requirement["tf_num_intraop_threads"],
        "TF_NUM_INTEROP_THREADS": requirement["tf_num_interop_threads"],
        "OMP_NUM_THREADS": requirement["omp_num_threads"],
    }


def _training_record_path(root: Path, method_id: str, train_seed: int) -> Path:
    return (
        root
        / "records"
        / "training"
        / _component(method_id)
        / f"seed_{int(train_seed)}.json"
    )


def _preflight_record_path(root: Path, method_id: str, train_seed: int) -> Path:
    return (
        root
        / "records"
        / "preflight"
        / _component(method_id)
        / f"seed_{int(train_seed)}.json"
    )


def _training_launch_contract(
    protocol: Mapping[str, Any],
    protocol_file: Path,
    method_id: str,
    train_seed: int,
) -> Dict[str, Any]:
    method = _validate_identity(protocol, method_id, train_seed)
    repo_root = Path(__file__).resolve().parent
    base_name = _component(
        f"{protocol['protocol_id']}__{method_id}__seed{int(train_seed)}"
    )
    training = dict(protocol["training"])
    if training.get("interruption_policy") != FORMAL_INTERRUPTION_POLICY:
        raise PaperSchemaError("paper training interruption policy mismatch")
    environment = _clean_subprocess_environment(
        protocol, random_seed=int(train_seed)
    )
    common_hyperparameters = dict(
        training.get("common_hyperparameter_contract") or {}
    )
    environment.update(
        {
            str(key): str(value)
            for key, value in dict(
                common_hyperparameters.get("environment_variables") or {}
            ).items()
        }
    )
    environment.update(
        {
            str(key): str(value)
            for key, value in dict(method.get("environment") or {}).items()
        }
    )
    environment.update(_physics_environment(protocol))
    environment.update(_task_semantics_environment(protocol))
    environment.update(_training_device_environment(protocol))
    environment.update(
        {
            "PAPER_CURRICULUM_ENABLED": "1",
            "PAPER_TASK_BANK_MANIFEST": str(
                protocol["task_bank"]["manifest_path"]
            ),
            "PAPER_EXPERIMENT_PROTOCOL_PATH": str(protocol_file),
            "PAPER_METHOD_ID": str(method_id),
            "SEED": str(int(train_seed)),
            "NUM_ENVS": str(int(training["parallel_environments"])),
            "AUTO_EVAL": "0",
            "SAVE_TRAINING_RESUME_STATE": "0",
            "SAVE_INTERVAL": "0",
            "SAVE_POSITIONS": "0",
            "ENABLE_ROLE_SHUFFLE": "0",
            "FAST_ARTIFACTS": "1",
            "QUIET_OUTPUT": "1",
            "SUPPRESS_MA_PROMPT": "1",
            "SUPPRESS_TERRAIN_OUTPUT": "1",
            "CURRICULUM_RB_KEEP_RECENT": str(
                int(
                    training["stage_transition_runtime"]
                    ["replay_keep_recent_transitions"]
                )
            ),
            "CURRICULUM_LR_FACTOR": str(
                float(
                    training["stage_transition_runtime"]
                    ["learning_rate_factor"]
                )
            ),
            "CURRICULUM_RESET_OPTIMIZER": (
                "1"
                if training["stage_transition_runtime"]
                ["optimizer_slot_policy"] == "reset_adam_moments"
                else "0"
            ),
            "TRAIN_PYTHON_BIN": str(
                protocol["execution_environment"]["python_runtimes"][
                    "training"
                ]["executable_artifact"]["path"]
            ),
        }
    )
    command = [
        "bash",
        str(repo_root / "run_optimized.sh"),
        str(int(training["synchronous_iterations"])),
        str(int(training["batch_size"])),
        base_name,
        "1" if bool(method["use_weighted_reward"]) else "0",
        str(method["algorithm"]),
    ]
    return {
        "method": method,
        "repo_root": repo_root,
        "base_name": base_name,
        "training": training,
        "environment": environment,
        "command": command,
    }


def _single_argv_value(argv: Iterable[Any], option: str) -> str:
    values = list(argv)
    indices = [index for index, value in enumerate(values) if value == option]
    if len(indices) != 1 or indices[0] + 1 >= len(values):
        raise PaperSchemaError(
            f"resolved launcher must contain exactly one {option} value"
        )
    return str(values[indices[0] + 1])


def _validate_resolved_training_manifest(
    resolved: Mapping[str, Any],
    *,
    launch: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_file: Path,
    method_id: str,
    train_seed: int,
) -> Dict[str, Any]:
    if resolved.get("version") != 1 or resolved.get("resolve_only") is not True:
        raise PaperSchemaError("training launcher did not produce a resolve-only manifest")
    repo_root = Path(launch["repo_root"]).resolve()
    if Path(str(resolved.get("cwd", ""))).resolve() != repo_root:
        raise PaperSchemaError("resolved training launcher cwd mismatch")
    if Path(str(resolved.get("python_script", ""))).resolve() != (
        repo_root / "paper3d_train_optimized.py"
    ):
        raise PaperSchemaError("resolved training Python entry point mismatch")
    expected_python = Path(
        protocol["execution_environment"]["python_runtimes"]["training"][
            "executable_artifact"
        ]["path"]
    ).resolve()
    if Path(str(resolved.get("python_executable", ""))).resolve() != (
        expected_python
    ):
        raise PaperSchemaError("resolved training Python executable mismatch")
    meta = dict(resolved.get("meta") or {})
    training = dict(launch["training"])
    expected_meta = {
        "episodes": int(training["synchronous_iterations"]),
        "batch_size": int(training["batch_size"]),
        "use_weighted_reward": (
            "1" if bool(launch["method"]["use_weighted_reward"]) else "0"
        ),
        "algorithm": str(launch["method"]["algorithm"]),
        "exp_name": str(launch["base_name"]),
        "seed": str(int(train_seed)),
        "num_envs": int(training["parallel_environments"]),
    }
    meta_mismatches = {
        key: {"expected": expected, "actual": meta.get(key)}
        for key, expected in expected_meta.items()
        if meta.get(key) != expected
    }
    if meta_mismatches:
        raise PaperSchemaError(
            "resolved training launcher metadata mismatch: "
            + json.dumps(meta_mismatches, sort_keys=True)
        )

    exec_env = {str(key): str(value) for key, value in dict(
        resolved.get("exec_env") or {}
    ).items()}
    expected_environment = {
        **{
            str(key): str(value)
            for key, value in dict(
                training.get("common_hyperparameter_contract", {}).get(
                    "environment_variables", {}
                )
                or {}
            ).items()
        },
        **runtime_method_environment(launch["method"]),
        **_physics_environment(protocol),
        **_task_semantics_environment(protocol),
        **_training_device_environment(protocol),
        "PAPER_CURRICULUM_ENABLED": "1",
        "PAPER_TASK_BANK_MANIFEST": str(
            protocol["task_bank"]["manifest_path"]
        ),
        "PAPER_EXPERIMENT_PROTOCOL_PATH": str(protocol_file),
        "PAPER_METHOD_ID": str(method_id),
        "SEED": str(int(train_seed)),
        "NUM_ENVS": str(int(training["parallel_environments"])),
        "ENABLE_ROLE_SHUFFLE": "0",
        "TRAIN_PYTHON_BIN": str(expected_python),
        "CURRICULUM_RB_KEEP_RECENT": str(
            int(
                training["stage_transition_runtime"]
                ["replay_keep_recent_transitions"]
            )
        ),
        "CURRICULUM_LR_FACTOR": str(
            float(
                training["stage_transition_runtime"]
                ["learning_rate_factor"]
            )
        ),
        "CURRICULUM_RESET_OPTIMIZER": (
            "1"
            if training["stage_transition_runtime"]
            ["optimizer_slot_policy"] == "reset_adam_moments"
            else "0"
        ),
        "SAVE_TRAINING_RESUME_STATE": "0",
    }
    environment_mismatches = {
        key: {"expected": expected, "actual": exec_env.get(key)}
        for key, expected in expected_environment.items()
        if exec_env.get(key) != expected
    }
    if environment_mismatches:
        raise PaperSchemaError(
            "resolved training launcher environment mismatch: "
            + json.dumps(environment_mismatches, sort_keys=True)
        )

    argv = list(resolved.get("argv") or [])
    required_flags = {
        "--scenario": "paper3d_terrain_energy",
        "--algo": str(launch["method"]["algorithm"]),
        "--train-episodes": str(int(training["synchronous_iterations"])),
        "--batch-size": str(int(training["batch_size"])),
        "--num-envs": str(int(training["parallel_environments"])),
        "--seed": str(int(train_seed)),
        "--enable-role-shuffle": "0",
    }
    common_hyperparameters = dict(
        training.get("common_hyperparameter_contract") or {}
    )
    if common_hyperparameters.get("policy_id") != (
        "explicit_common_launcher_and_environment_v2"
    ):
        raise PaperSchemaError(
            "formal common training hyperparameter policy mismatch"
        )
    common_launcher_flags = dict(
        common_hyperparameters.get("launcher_flags") or {}
    )
    if not common_launcher_flags:
        raise PaperSchemaError(
            "formal common training hyperparameter flags are empty"
        )
    required_flags.update(
        {
            str(option): str(value)
            for option, value in common_launcher_flags.items()
        }
    )
    physics_flag_map = {
        "simulation_dt": "--simulation-dt",
        "agent_size": "--agent-size",
        "agent_max_speed": "--agent-max-speed",
        "agent_accel": "--agent-accel",
        "gravity": "--gravity",
        "control_accel_gain": "--control-accel-gain",
        "damping": "--damping",
        "action_range_x": "--action-range-x",
        "action_range_y": "--action-range-y",
        "action_range_z": "--action-range-z",
        "z_action_bias": "--z-action-bias",
        "quadrotor_attitude_response_time": (
            "--quadrotor-attitude-response-time"
        ),
        "quadrotor_psi_cmd": "--quadrotor-psi-cmd",
    }
    for physics_field, option in physics_flag_map.items():
        required_flags[option] = str(
            training["physics_contract"][physics_field]
        )
    required_flags["--success-distance-threshold"] = str(
        training["task_semantics_contract"]["success_distance_threshold"]
    )
    required_flags["--collision-distance-threshold"] = str(
        training["task_semantics_contract"]["collision_distance_threshold"]
    )
    method_flag_map = {
        "MATD3_USE_DUAL_Q": "--matd3-use-dual-q",
        "MATD3_USE_SEPARATED_GRADIENT": "--matd3-use-separated-gradient",
        "MATD3_USE_HYBRID_ACTOR_OBJECTIVE": (
            "--matd3-use-hybrid-actor-objective"
        ),
        "MATD3_ACTION_SEMANTICS_MODE": "--matd3-action-semantics-mode",
        "MATD3_RECONSTRUCT_CORRECTED_TARGET": (
            "--matd3-reconstruct-corrected-target"
        ),
        "MADDPG_USE_DUAL_Q": "--maddpg-use-dual-q",
        "MADDPG_USE_SEPARATED_GRADIENT": "--maddpg-use-separated-gradient",
        "MADDPG_ACTION_SEMANTICS_MODE": (
            "--maddpg-action-semantics-mode"
        ),
        "ACTOR_UPDATE_DELAY": "--actor-update-delay",
        "ENABLE_ACTION_CORRECTION": "--enable-action-correction",
        "USE_TF_POTENTIAL_FIELD": "--use-tf-potential-field",
        "USE_FR_FEATURE": "--use-fr-feature",
        "USE_PF_FEATURE": "--use-pf-feature",
        "ACTION_FORCE_RATIO": "--action-force-ratio",
    }
    for environment_name, option in method_flag_map.items():
        if environment_name in expected_environment:
            required_flags[option] = expected_environment[environment_name]
    flag_mismatches = {}
    for option, expected in required_flags.items():
        actual = _single_argv_value(argv, option)
        if actual != expected:
            flag_mismatches[option] = {"expected": expected, "actual": actual}
    schedule = expected_environment.get("ACTION_FORCE_RATIO_SCHEDULE_PCT")
    schedule_option = "--action-force-ratio-schedule-pct"
    if schedule == "DISABLED":
        if schedule_option in argv:
            raise PaperSchemaError(
                "resolved launcher unexpectedly enabled the disabled FR schedule"
            )
    elif schedule is not None:
        actual_schedule = _single_argv_value(argv, schedule_option)
        if actual_schedule != schedule:
            flag_mismatches[schedule_option] = {
                "expected": schedule,
                "actual": actual_schedule,
            }
    if "--save-model" not in argv:
        raise PaperSchemaError("resolved training launcher disabled model saving")
    if flag_mismatches:
        raise PaperSchemaError(
            "resolved training launcher argument mismatch: "
            + json.dumps(flag_mismatches, sort_keys=True)
        )
    return dict(resolved)


def _validation_index_path(
    root: Path, method_id: str, train_seed: int, difficulty: str
) -> Path:
    return (
        root
        / "records"
        / "validation"
        / _component(method_id)
        / f"seed_{int(train_seed)}"
        / f"{_component(difficulty)}.json"
    )


def _selection_path(
    root: Path, method_id: str, train_seed: int, difficulty: str
) -> Path:
    return (
        root
        / "selections"
        / _component(method_id)
        / f"seed_{int(train_seed)}"
        / f"{_component(difficulty)}.json"
    )


def _load_candidate_index(path: str | Path) -> Dict[str, Any]:
    payload = _load_hashed_json(path, "paper_training_candidate_index_v2")
    candidates = list(payload.get("candidates") or [])
    if int(payload.get("candidate_count", -1)) != len(candidates):
        raise PaperSchemaError("paper candidate index count mismatch")
    if not candidates:
        raise PaperSchemaError("paper candidate index is empty")
    protocol_artifact = dict(payload.get("experiment_protocol") or {})
    actual_protocol = fingerprint_artifact(protocol_artifact.get("path"))
    if actual_protocol.get("sha256") != protocol_artifact.get("sha256"):
        raise PaperSchemaError("paper candidate index protocol artifact changed")
    if not str(payload.get("experiment_protocol_content_sha256", "")).strip():
        raise PaperSchemaError("paper candidate index lacks protocol content hash")
    seen_paths = set()
    for candidate in candidates:
        checkpoint = dict(candidate.get("checkpoint") or {})
        actual = fingerprint_artifact(checkpoint.get("path"))
        if actual.get("sha256") != checkpoint.get("sha256"):
            raise PaperSchemaError("paper candidate checkpoint changed after training")
        if actual["path"] in seen_paths:
            raise PaperSchemaError("paper candidate index repeats a checkpoint path")
        seen_paths.add(actual["path"])
        metadata_path = Path(
            str(candidate.get("training_metadata_path", ""))
        ).expanduser().resolve()
        if metadata_path.parent != Path(actual["path"]).resolve():
            raise PaperSchemaError(
                "paper candidate metadata is outside its checkpoint"
            )
        metadata = _load_hashed_json(
            metadata_path, "paper_training_checkpoint_v5"
        )
        if metadata.get("content_sha256") != candidate.get(
            "training_metadata_content_sha256"
        ):
            raise PaperSchemaError("paper candidate metadata hash mismatch")
        try:
            metadata_ratio = float(metadata["checkpoint_action_force_ratio"])
            candidate_ratio = float(candidate["checkpoint_action_force_ratio"])
            ratio_episode = int(
                metadata["checkpoint_action_force_ratio_episode"]
            )
            completed_iterations = int(
                metadata["completed_training_iterations"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PaperSchemaError(
                "paper candidate lacks checkpoint-specific force ratio"
            ) from exc
        if (
            not math.isfinite(metadata_ratio)
            or not 0.0 <= metadata_ratio <= 1.0
            or not math.isclose(
                metadata_ratio, candidate_ratio, rel_tol=0.0, abs_tol=1e-12
            )
            or ratio_episode != completed_iterations
        ):
            raise PaperSchemaError(
                "paper candidate checkpoint-specific force ratio mismatch"
            )
    return payload


def run_preflight(
    protocol_path: str | Path,
    experiment_root: str | Path,
    method_id: str,
    train_seed: int,
) -> Dict[str, Any]:
    protocol_file = Path(protocol_path).expanduser().resolve()
    protocol = load_protocol(protocol_file)
    _validate_identity(protocol, method_id, train_seed)
    root = _experiment_root(experiment_root)
    record_path = _preflight_record_path(root, method_id, train_seed)
    raw_manifest_path = (
        root
        / "preflight"
        / _component(method_id)
        / f"seed_{int(train_seed)}"
        / "resolved_launcher.json"
    )
    if record_path.exists() or raw_manifest_path.exists():
        raise PaperSchemaError(
            "paper training preflight output already exists for "
            f"method={method_id}, seed={train_seed}"
        )
    launch = _training_launch_contract(
        protocol, protocol_file, method_id, train_seed
    )
    environment = dict(launch["environment"])
    environment.update(
        {
            "ABLATION_RESOLVE_ONLY": "1",
            "ABLATION_MANIFEST_PATH": str(raw_manifest_path),
        }
    )
    subprocess.run(
        list(launch["command"]),
        cwd=launch["repo_root"],
        env=environment,
        check=True,
    )
    if not raw_manifest_path.is_file():
        raise PaperSchemaError(
            "paper training preflight did not create its resolved launcher manifest"
        )
    with raw_manifest_path.open("r", encoding="utf-8") as stream:
        resolved = json.load(stream)
    _validate_resolved_training_manifest(
        resolved,
        launch=launch,
        protocol=protocol,
        protocol_file=protocol_file,
        method_id=method_id,
        train_seed=train_seed,
    )
    repo_root = Path(launch["repo_root"])
    source_artifacts = []
    for source_path in (
        Path(__file__).resolve(),
        repo_root / "run_optimized.sh",
        repo_root / "paper3d_train_optimized.py",
        repo_root / "experiment_runtime_config.py",
        repo_root / "paper_curriculum.py",
        repo_root / "paper_experiment_protocol.py",
        repo_root / "paper_task_bank.py",
        repo_root / "paper_task_layout.py",
        repo_root / "paper_experiment_schema.py",
        repo_root / "cross_agent_reference_state.py",
        repo_root / "potential_field_corrector.py",
        repo_root / "multiagent" / "core.py",
        repo_root / "multiagent" / "environment.py",
        repo_root / "multiagent" / "scenarios" / "paper3d_terrain_energy.py",
        repo_root / "multiagent" / "scenarios" / "obstacle_observation.py",
    ):
        artifact = fingerprint_artifact(source_path)
        artifact["logical_name"] = str(source_path.relative_to(repo_root))
        source_artifacts.append(artifact)
    payload: Dict[str, Any] = {
        "schema_version": PREFLIGHT_RECORD_SCHEMA,
        "protocol": fingerprint_artifact(protocol_file),
        "protocol_content_sha256": protocol["content_sha256"],
        "method_id": str(method_id),
        "method_spec": launch["method"],
        "train_seed": int(train_seed),
        "base_experiment_name": launch["base_name"],
        "command": list(launch["command"]),
        "resolved_launcher_manifest": fingerprint_artifact(raw_manifest_path),
        "source_artifacts": source_artifacts,
        "checks": {
            "protocol_loaded": True,
            "formal_task_bank_validated": True,
            "clean_environment_policy_validated": True,
            "launcher_metadata_validated": True,
            "launcher_environment_validated": True,
            "launcher_arguments_validated": True,
            "training_started": False,
        },
    }
    payload["content_sha256"] = content_sha256(payload)
    _write_immutable(record_path, payload)
    print(f"training preflight record written: {record_path}")
    return payload


def _load_preflight_record(
    protocol: Mapping[str, Any],
    experiment_root: Path,
    method_id: str,
    train_seed: int,
) -> Dict[str, Any]:
    record_path = _preflight_record_path(
        experiment_root, method_id, train_seed
    )
    record = _load_hashed_json(record_path, PREFLIGHT_RECORD_SCHEMA)
    if record.get("protocol_content_sha256") != protocol.get("content_sha256"):
        raise PaperSchemaError("training preflight protocol mismatch")
    if record.get("method_id") != method_id or int(
        record.get("train_seed", -1)
    ) != int(train_seed):
        raise PaperSchemaError("training preflight identity mismatch")
    if dict(record.get("method_spec") or {}) != dict(
        protocol["methods"][method_id]
    ):
        raise PaperSchemaError("training preflight method specification drift")
    recorded_protocol = dict(record.get("protocol") or {})
    actual_protocol = fingerprint_artifact(recorded_protocol.get("path"))
    if actual_protocol.get("sha256") != recorded_protocol.get("sha256"):
        raise PaperSchemaError("training preflight protocol artifact changed")
    recorded_launcher = dict(record.get("resolved_launcher_manifest") or {})
    actual_launcher = fingerprint_artifact(recorded_launcher.get("path"))
    if actual_launcher.get("sha256") != recorded_launcher.get("sha256"):
        raise PaperSchemaError("training preflight launcher manifest changed")
    for artifact in list(record.get("source_artifacts") or []):
        actual = fingerprint_artifact(artifact.get("path"))
        if actual.get("sha256") != artifact.get("sha256"):
            raise PaperSchemaError(
                "training source changed after preflight: "
                f"{artifact.get('logical_name')!r}"
            )
    required_checks = {
        "protocol_loaded",
        "formal_task_bank_validated",
        "clean_environment_policy_validated",
        "launcher_metadata_validated",
        "launcher_environment_validated",
        "launcher_arguments_validated",
    }
    checks = dict(record.get("checks") or {})
    if any(checks.get(name) is not True for name in required_checks) or checks.get(
        "training_started"
    ) is not False:
        raise PaperSchemaError("training preflight checks are incomplete")
    return record


def run_training(
    protocol_path: str | Path,
    experiment_root: str | Path,
    method_id: str,
    train_seed: int,
    *,
    resume_existing: bool = False,
) -> Dict[str, Any]:
    if resume_existing:
        raise PaperSchemaError(
            "formal paper runs forbid trajectory-changing resume; restart this "
            "method/train-seed from a clean model prefix"
        )
    protocol_file = Path(protocol_path).expanduser().resolve()
    protocol = load_protocol(protocol_file)
    _validate_identity(protocol, method_id, train_seed)
    root = _experiment_root(experiment_root)
    preflight_record = _load_preflight_record(
        protocol, root, method_id, train_seed
    )
    preflight_record_path = _preflight_record_path(
        root, method_id, train_seed
    )
    record_path = _training_record_path(root, method_id, train_seed)
    if record_path.exists():
        raise PaperSchemaError(f"training record already exists: {record_path}")

    launch = _training_launch_contract(
        protocol, protocol_file, method_id, train_seed
    )
    method = dict(launch["method"])
    repo_root = Path(launch["repo_root"])
    base_name = str(launch["base_name"])
    preexisting = sorted((repo_root / "models").glob(f"{base_name}_*"))
    resume_source_snapshot = None
    if resume_existing:
        if len(preexisting) != 1:
            raise PaperSchemaError(
                f"formal resume requires exactly one existing model root, got={preexisting}"
            )
        resume_model_root = preexisting[0].resolve()
        resume_checkpoint = resume_model_root / "checkpoint"
        resume_state = resume_checkpoint / "checkpoint_state.json"
        candidate_index_before = resume_model_root / "paper_candidate_index.json"
        if not resume_state.is_file() or not candidate_index_before.is_file():
            raise PaperSchemaError(
                "formal resume requires checkpoint/checkpoint_state.json and "
                "paper_candidate_index.json"
            )
        resume_source_snapshot = {
            "model_root_path": str(resume_model_root),
            "checkpoint_before_launch": fingerprint_artifact(
                resume_checkpoint
            ),
            "candidate_index_before_launch": fingerprint_artifact(
                candidate_index_before
            ),
        }
    elif preexisting:
        raise PaperSchemaError(
            f"model prefix is not clean for immutable run={base_name}: {preexisting}"
        )

    training = dict(launch["training"])
    environment = dict(launch["environment"])
    command = list(launch["command"])
    if resume_existing:
        environment.update(
            {
                "RESUME_MODEL_ENV": str(resume_model_root),
                "EXP_NAME_WITH_TIMESTAMP_OVERRIDE": resume_model_root.name,
            }
        )
    subprocess.run(command, cwd=repo_root, env=environment, check=True)

    created = sorted((repo_root / "models").glob(f"{base_name}_*"))
    if len(created) != 1:
        raise PaperSchemaError(
            f"training must create exactly one model root, got={created}"
        )
    model_root = created[0].resolve()
    candidate_index_path = model_root / "paper_candidate_index.json"
    candidate_index = _load_candidate_index(candidate_index_path)
    if str(candidate_index.get("paper_method_id", "")) != str(method_id):
        raise PaperSchemaError("training candidate index method mismatch")
    if int(candidate_index.get("train_seed", -1)) != int(train_seed):
        raise PaperSchemaError("training candidate index seed mismatch")
    if str(candidate_index.get("task_bank_content_sha256", "")) != str(
        protocol["task_bank"]["bank_content_sha256"]
    ):
        raise PaperSchemaError("training candidate index task-bank mismatch")
    if str(candidate_index.get("experiment_protocol_content_sha256", "")) != str(
        protocol["content_sha256"]
    ):
        raise PaperSchemaError("training candidate index protocol mismatch")
    candidates_by_difficulty = {
        difficulty: [
            candidate
            for candidate in candidate_index["candidates"]
            if candidate.get("trained_on_difficulty_bin") == difficulty
        ]
        for difficulty in ("C0", "C1", "C2", "C3")
    }
    expected_candidate_counts = {
        str(difficulty): int(count)
        for difficulty, count in dict(
            training["expected_checkpoint_candidates_per_difficulty"]
        ).items()
    }
    actual_candidate_counts = {
        difficulty: len(candidates)
        for difficulty, candidates in candidates_by_difficulty.items()
    }
    candidate_count_mismatches = {
        difficulty: {
            "expected": expected_candidate_counts.get(difficulty),
            "actual": actual_count,
        }
        for difficulty, actual_count in actual_candidate_counts.items()
        if actual_count != expected_candidate_counts.get(difficulty)
    }
    if candidate_count_mismatches:
        raise PaperSchemaError(
            "training checkpoint candidate count differs from protocol: "
            + json.dumps(candidate_count_mismatches, sort_keys=True)
        )

    payload: Dict[str, Any] = {
        "schema_version": TRAINING_RECORD_SCHEMA,
        "protocol": fingerprint_artifact(protocol_file),
        "protocol_content_sha256": protocol["content_sha256"],
        "preflight": fingerprint_artifact(preflight_record_path),
        "preflight_content_sha256": preflight_record["content_sha256"],
        "method_id": str(method_id),
        "method_spec": method,
        "train_seed": int(train_seed),
        "launch_mode": "resume_existing" if resume_existing else "fresh",
        "resume_source_snapshot": resume_source_snapshot,
        "command": command,
        "environment_overrides": {
            key: environment[key]
            for key in sorted(
                set(method.get("environment") or {})
                | set(
                    training.get("common_hyperparameter_contract", {}).get(
                        "environment_variables", {}
                    )
                    or {}
                )
                | set(_physics_environment(protocol))
                | set(_task_semantics_environment(protocol))
                | set(_training_device_environment(protocol))
                | {
                    "PAPER_CURRICULUM_ENABLED",
                    "PAPER_TASK_BANK_MANIFEST",
                    "PAPER_EXPERIMENT_PROTOCOL_PATH",
                    "PAPER_METHOD_ID",
                    "SEED",
                    "NUM_ENVS",
                    "AUTO_EVAL",
                    "SAVE_TRAINING_RESUME_STATE",
                    "SAVE_INTERVAL",
                    "SAVE_POSITIONS",
                    "ENABLE_ROLE_SHUFFLE",
                    "FAST_ARTIFACTS",
                    "QUIET_OUTPUT",
                    "SUPPRESS_MA_PROMPT",
                    "SUPPRESS_TERRAIN_OUTPUT",
                    "TRAIN_PYTHON_BIN",
                    "CURRICULUM_RB_KEEP_RECENT",
                    "CURRICULUM_LR_FACTOR",
                    "CURRICULUM_RESET_OPTIMIZER",
                    "RESUME_MODEL_ENV",
                    "EXP_NAME_WITH_TIMESTAMP_OVERRIDE",
                }
            )
            if key in environment
        },
        "inherited_execution_environment": {
            key: environment[key]
            for key in execution_environment_allowlist()
            if key in environment
        },
        "model_root": fingerprint_artifact(model_root),
        "candidate_index": fingerprint_artifact(candidate_index_path),
        "candidate_index_content_sha256": candidate_index["content_sha256"],
        "candidate_counts_by_difficulty": {
            difficulty: len(candidates)
            for difficulty, candidates in candidates_by_difficulty.items()
        },
    }
    payload["content_sha256"] = content_sha256(payload)
    _write_immutable(record_path, payload)
    print(f"training record written: {record_path}")
    return payload


def _load_training_record(
    protocol: Mapping[str, Any],
    experiment_root: Path,
    method_id: str,
    train_seed: int,
) -> Dict[str, Any]:
    record_path = _training_record_path(
        experiment_root, method_id, train_seed
    )
    record = _load_hashed_json(record_path, TRAINING_RECORD_SCHEMA)
    if record.get("protocol_content_sha256") != protocol.get("content_sha256"):
        raise PaperSchemaError("training record protocol mismatch")
    if record.get("method_id") != method_id or int(record.get("train_seed")) != int(
        train_seed
    ):
        raise PaperSchemaError("training record identity mismatch")
    actual_index = fingerprint_artifact(record["candidate_index"]["path"])
    if actual_index.get("sha256") != record["candidate_index"].get("sha256"):
        raise PaperSchemaError("training candidate index changed after record creation")
    actual_model_root = fingerprint_artifact(record["model_root"]["path"])
    if actual_model_root.get("sha256") != record["model_root"].get("sha256"):
        raise PaperSchemaError("training model root changed after record creation")
    return record


def _run_evaluation(
    *,
    protocol: Mapping[str, Any],
    protocol_path: str | Path,
    method_id: str,
    train_seed: int,
    checkpoint_path: str,
    split: str,
    eval_seed: int,
    run_id: str,
    output_dir: Path,
    selection_path: str | None,
) -> Path:
    if output_dir.exists():
        raise PaperSchemaError(
            f"refusing to overwrite evaluation output: {output_dir}"
        )
    bank = load_task_bank(protocol["task_bank"]["manifest_path"])
    split_cfg = dict((bank.get("splits") or {}).get(split) or {})
    if not split_cfg:
        raise PaperSchemaError(f"task-bank split does not exist: {split}")
    task_count = int(split_cfg["task_count"])
    environment = _clean_subprocess_environment(
        protocol, random_seed=int(eval_seed)
    )
    environment.update(
        {
            str(key): str(value)
            for key, value in dict(
                protocol["training"]["common_hyperparameter_contract"].get(
                    "environment_variables", {}
                )
                or {}
            ).items()
        }
    )
    method_environment = dict(protocol["methods"][method_id].get("environment") or {})
    environment.update(
        {str(key): str(value) for key, value in method_environment.items()}
    )
    environment.update(_physics_environment(protocol))
    environment.update(_task_semantics_environment(protocol))
    environment.update(_evaluation_device_environment(protocol))
    environment.update(
        {
            "PAPER_RUN_ID": str(run_id),
            "PAPER_METHOD_ID": str(method_id),
            "PAPER_SPLIT": str(split),
            "PAPER_TRAIN_SEED": str(int(train_seed)),
            "PAPER_EVAL_SEED": str(int(eval_seed)),
            "PAPER_TASK_BANK_MANIFEST": str(
                protocol["task_bank"]["manifest_path"]
            ),
            "PAPER_EXPERIMENT_PROTOCOL_PATH": str(
                Path(protocol_path).expanduser().resolve()
            ),
            "PAPER_TASK_BANK_SPLIT": str(split),
            "PAPER_EVAL_DISABLE_VISUALIZATION": "1",
            "PAPER_SELECTION_RESULT": str(selection_path or ""),
            "STRICT_EVAL_MATCH": "1",
            "EVAL_EPISODE_PARALLELISM": str(min(8, task_count)),
            "EVAL_ENV_STEP_THREADS": str(min(8, task_count)),
            "EVAL_LIGHT_MODE": "1",
            "SAVE_INTERACTIVE_TRAJ": "0",
            "SAVE_EVAL_ALL_EPISODES": "0",
            "SAVE_BEST_TRAJ": "0",
            "SAVE_EVAL_TRAJECTORY_PNG": "0",
            "SAVE_TEAM_SUCCESS_HTML": "0",
            "SAVE_EVAL_ACTOR_SEQUENCE": "0",
            "SAVE_EVAL_CONTROL_DIAGNOSTICS": "0",
            "SAVE_GAZEBO_REPLAY": "0",
            "SAVE_GAZEBO_DYNAMIC_REPLAY": "0",
            "DISABLE_TRAJECTORY_RECORDING": "0",
            "QUIET_OUTPUT": "1",
            "SUPPRESS_MA_PROMPT": "1",
            "SUPPRESS_TERRAIN_OUTPUT": "1",
            "EVAL_PYTHON_BIN": str(
                protocol["execution_environment"]["python_runtimes"][
                    "evaluation"
                ]["executable_artifact"]["path"]
            ),
        }
    )
    repo_root = Path(__file__).resolve().parent
    command = [
        "bash",
        str(repo_root / "run_evaluation.sh"),
        str(Path(checkpoint_path).resolve()),
        str(task_count),
        str(output_dir),
        "",
        "false",
        "false",
    ]
    subprocess.run(command, cwd=repo_root, env=environment, check=True)
    manifest_path = output_dir / "paper_run_manifest.json"
    if not manifest_path.is_file():
        raise PaperSchemaError(
            f"formal evaluation did not create run manifest: {manifest_path}"
        )
    return manifest_path


def run_validation(
    protocol_path: str | Path,
    experiment_root: str | Path,
    method_id: str,
    train_seed: int,
    difficulty: str,
) -> Dict[str, Any]:
    protocol = load_protocol(protocol_path)
    _validate_identity(protocol, method_id, train_seed)
    difficulty = str(difficulty).strip()
    if difficulty not in ("C0", "C1", "C2", "C3"):
        raise PaperSchemaError(f"invalid validation difficulty={difficulty!r}")
    selection_difficulty = str(
        protocol["validation"]["selection_training_difficulty_bin"]
    )
    if difficulty != selection_difficulty:
        raise PaperSchemaError(
            "formal validation selects one final checkpoint from C3 "
            f"candidates; requested difficulty={difficulty!r}"
        )
    root = _experiment_root(experiment_root)
    output_index_path = _validation_index_path(
        root, method_id, train_seed, difficulty
    )
    if output_index_path.exists():
        raise PaperSchemaError(
            f"validation index already exists: {output_index_path}"
        )
    training_record = _load_training_record(
        protocol, root, method_id, train_seed
    )
    candidate_index = _load_candidate_index(
        training_record["candidate_index"]["path"]
    )
    if str(candidate_index.get("paper_method_id", "")) != method_id:
        raise PaperSchemaError("validation candidate index method mismatch")
    if int(candidate_index.get("train_seed", -1)) != int(train_seed):
        raise PaperSchemaError("validation candidate index train seed mismatch")
    if str(candidate_index.get("task_bank_content_sha256", "")) != str(
        protocol["task_bank"]["bank_content_sha256"]
    ):
        raise PaperSchemaError("validation candidate index task-bank mismatch")
    if str(candidate_index.get("experiment_protocol_content_sha256", "")) != str(
        protocol["content_sha256"]
    ):
        raise PaperSchemaError("validation candidate index protocol mismatch")
    candidates = [
        candidate
        for candidate in candidate_index["candidates"]
        if candidate.get("trained_on_difficulty_bin") == difficulty
    ]
    if len(candidates) < int(protocol["validation"]["minimum_distinct_candidates"]):
        raise PaperSchemaError(
            f"difficulty={difficulty} has too few checkpoint candidates"
        )

    validation_runs = []
    for candidate in candidates:
        checkpoint = dict(candidate["checkpoint"])
        label = Path(checkpoint["path"]).name
        run_id = _component(
            f"{protocol['protocol_id']}__validation__{method_id}__seed{train_seed}__"
            f"{difficulty}__{label}"
        )
        output_dir = (
            root
            / "validation"
            / _component(method_id)
            / f"seed_{int(train_seed)}"
            / difficulty
            / _component(label)
        )
        manifest_path = _run_evaluation(
            protocol=protocol,
            protocol_path=protocol_path,
            method_id=method_id,
            train_seed=train_seed,
            checkpoint_path=checkpoint["path"],
            split=f"validation_{difficulty}",
            eval_seed=int(protocol["validation"]["eval_seed"]),
            run_id=run_id,
            output_dir=output_dir,
            selection_path=None,
        )
        validation_runs.append(
            {
                "label": label,
                "checkpoint": checkpoint,
                "run_manifest": fingerprint_artifact(manifest_path),
            }
        )
    payload: Dict[str, Any] = {
        "schema_version": VALIDATION_INDEX_SCHEMA,
        "protocol_content_sha256": protocol["content_sha256"],
        "method_id": method_id,
        "train_seed": int(train_seed),
        "difficulty_bin": difficulty,
        "split": f"validation_{difficulty}",
        "eval_seed": int(protocol["validation"]["eval_seed"]),
        "candidate_count": len(validation_runs),
        "validation_runs": validation_runs,
    }
    payload["content_sha256"] = content_sha256(payload)
    _write_immutable(output_index_path, payload)
    print(f"validation index written: {output_index_path}")
    return payload


def run_selection(
    protocol_path: str | Path,
    experiment_root: str | Path,
    method_id: str,
    train_seed: int,
    difficulty: str,
) -> Dict[str, Any]:
    protocol = load_protocol(protocol_path)
    _validate_identity(protocol, method_id, train_seed)
    difficulty = str(difficulty).strip()
    if difficulty != str(
        protocol["validation"]["selection_training_difficulty_bin"]
    ):
        raise PaperSchemaError(
            "formal checkpoint selection is restricted to final C3 candidates"
        )
    root = _experiment_root(experiment_root)
    validation_index = _load_hashed_json(
        _validation_index_path(root, method_id, train_seed, difficulty),
        VALIDATION_INDEX_SCHEMA,
    )
    if validation_index.get("protocol_content_sha256") != protocol["content_sha256"]:
        raise PaperSchemaError("validation index protocol mismatch")
    if validation_index.get("method_id") != method_id:
        raise PaperSchemaError("validation index method mismatch")
    if int(validation_index.get("train_seed", -1)) != int(train_seed):
        raise PaperSchemaError("validation index train seed mismatch")
    if validation_index.get("difficulty_bin") != str(difficulty).strip():
        raise PaperSchemaError("validation index difficulty mismatch")
    if validation_index.get("split") != f"validation_{str(difficulty).strip()}":
        raise PaperSchemaError("validation index split mismatch")
    if int(validation_index.get("eval_seed", -1)) != int(
        protocol["validation"]["eval_seed"]
    ):
        raise PaperSchemaError("validation index eval seed mismatch")
    for run in validation_index.get("validation_runs") or []:
        recorded_manifest = dict(run.get("run_manifest") or {})
        actual_manifest = fingerprint_artifact(recorded_manifest.get("path"))
        if actual_manifest.get("sha256") != recorded_manifest.get("sha256"):
            raise PaperSchemaError("validation run manifest changed after indexing")
    candidates = {
        str(item["label"]): str(item["run_manifest"]["path"])
        for item in validation_index["validation_runs"]
    }
    result = select_checkpoint(
        candidates,
        require_adequacy=bool(
            protocol["methods"][method_id]["require_validation_adequacy"]
        ),
    )
    destination = _selection_path(root, method_id, train_seed, difficulty)
    _write_immutable(destination, result)
    print(f"checkpoint selection written: {destination}")
    return result


def run_official_test(
    protocol_path: str | Path,
    experiment_root: str | Path,
    method_id: str,
    train_seed: int,
    difficulty: str,
    split: str,
) -> Dict[str, Any]:
    protocol = load_protocol(protocol_path)
    _validate_identity(protocol, method_id, train_seed)
    difficulty = str(difficulty).strip()
    split = str(split).strip()
    root = _experiment_root(experiment_root)
    _load_training_record(protocol, root, method_id, train_seed)
    selection_file = _selection_path(root, method_id, train_seed, difficulty)
    with selection_file.open("r", encoding="utf-8") as stream:
        selection = validate_selection_result(json.load(stream))
    if str(selection.get("experiment_protocol_content_sha256", "")) != str(
        protocol["content_sha256"]
    ):
        raise PaperSchemaError("checkpoint selection protocol mismatch")
    if int(selection.get("train_seed", -1)) != int(train_seed):
        raise PaperSchemaError("checkpoint selection train seed mismatch")
    if selection.get("validation_split") != f"validation_{difficulty}":
        raise PaperSchemaError("checkpoint selection difficulty/split mismatch")
    if bool(selection.get("adequacy_required_for_selection")) != bool(
        protocol["methods"][method_id]["require_validation_adequacy"]
    ):
        raise PaperSchemaError("checkpoint selection adequacy policy mismatch")
    bank = load_task_bank(protocol["task_bank"]["manifest_path"])
    split_cfg = dict((bank.get("splits") or {}).get(split) or {})
    split_role = str(split_cfg.get("role", ""))
    if split_role not in {"test_in_distribution", "test_generalization"}:
        raise PaperSchemaError(f"official split has a non-test role: {split!r}")
    required_selection_difficulty = str(
        protocol["official_test"]["selected_training_difficulty_bin"]
    )
    if difficulty != required_selection_difficulty:
        raise PaperSchemaError(
            "all official tests require the validation-selected final C3 checkpoint"
        )
    if split_role == "test_in_distribution" and split not in {
        "test_C0",
        "test_C1",
        "test_C2",
        "test_C3",
    }:
        raise PaperSchemaError(
            "official in-distribution split is not one of test_C0..test_C3"
        )
    if split_role == "test_generalization" and difficulty != "C3":
        raise PaperSchemaError(
            "only a validation-selected C3 checkpoint may run generalization tests"
        )
    output_dir = (
        root
        / "official_test"
        / _component(method_id)
        / f"seed_{int(train_seed)}"
        / _component(split)
    )
    run_id = _component(
        f"{protocol['protocol_id']}__test__{method_id}__seed{train_seed}__{split}"
    )
    manifest_path = _run_evaluation(
        protocol=protocol,
        protocol_path=protocol_path,
        method_id=method_id,
        train_seed=train_seed,
        checkpoint_path=selection["selected_checkpoint"]["path"],
        split=split,
        eval_seed=int(protocol["official_test"]["eval_seed"]),
        run_id=run_id,
        output_dir=output_dir,
        selection_path=str(selection_file),
    )
    payload: Dict[str, Any] = {
        "schema_version": OFFICIAL_TEST_RECORD_SCHEMA,
        "protocol": fingerprint_artifact(protocol_path),
        "protocol_content_sha256": protocol["content_sha256"],
        "method_id": method_id,
        "train_seed": int(train_seed),
        "selected_difficulty_bin": difficulty,
        "split": split,
        "selection": fingerprint_artifact(selection_file),
        "run_manifest": fingerprint_artifact(manifest_path),
    }
    payload["content_sha256"] = content_sha256(payload)
    record_path = output_dir / "paper_official_test_record.json"
    _write_immutable(record_path, payload)
    print(f"official test record written: {record_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("preflight", "train", "resume", "validate", "select", "test"),
    )
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--train-seed", required=True, type=int)
    parser.add_argument("--difficulty", default="C3")
    parser.add_argument("--split", default="test_C3")
    args = parser.parse_args()
    if args.command == "preflight":
        run_preflight(
            args.protocol, args.experiment_root, args.method, args.train_seed
        )
    elif args.command == "train":
        run_training(
            args.protocol, args.experiment_root, args.method, args.train_seed
        )
    elif args.command == "resume":
        run_training(
            args.protocol,
            args.experiment_root,
            args.method,
            args.train_seed,
            resume_existing=True,
        )
    elif args.command == "validate":
        run_validation(
            args.protocol,
            args.experiment_root,
            args.method,
            args.train_seed,
            args.difficulty,
        )
    elif args.command == "select":
        run_selection(
            args.protocol,
            args.experiment_root,
            args.method,
            args.train_seed,
            args.difficulty,
        )
    else:
        run_official_test(
            args.protocol,
            args.experiment_root,
            args.method,
            args.train_seed,
            args.difficulty,
            args.split,
        )


if __name__ == "__main__":
    main()
