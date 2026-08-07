"""Canonical scientific runtime parameters shared by MATD3 train and eval.

These values used to be read directly from environment variables inside the
scenario.  That made them invisible in ``results.json`` and allowed evaluation
or checkpoint resume to use a different environment while keeping identical
network dimensions.  This module gives the train/eval entry points one typed
mapping and one propagation path.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from cross_agent_reference_state import (
    ADVANTAGE_SELECTOR_MODES,
    MODE_ADAPTIVE_TWIN_HEAD_TAIL,
    MODE_SHARED_TWIN_HEAD_TAIL,
    selector_state_errors,
)


@dataclass(frozen=True)
class RuntimeEnvironmentField:
    attr: str
    env: str
    kind: str
    default: Any

    @property
    def cli_option(self) -> str:
        return "--" + self.attr.replace("_", "-")


SCIENTIFIC_RUNTIME_ENV_FIELDS = (
    RuntimeEnvironmentField("agent_size", "AGENT_SIZE", "float", 0.5),
    # Scenario.make_world assigns these values before the training worker may
    # override them.  Making the scenario defaults explicit preserves direct
    # Python entry-point behavior while recording the actual dynamics.
    RuntimeEnvironmentField("agent_max_speed", "AGENT_MAX_SPEED", "float", 25.0),
    RuntimeEnvironmentField("agent_accel", "AGENT_ACCEL", "float", 8.5),
    RuntimeEnvironmentField("use_quadrotor_dynamics", "USE_QUADROTOR_DYNAMICS", "bool", False),
    RuntimeEnvironmentField("start_altitude_offset", "START_ALTITUDE_OFFSET", "float", 7.0),
    RuntimeEnvironmentField("goal_altitude", "GOAL_ALTITUDE", "float", 12.0),
    RuntimeEnvironmentField("agent_goal_formation_radius", "AGENT_GOAL_FORMATION_RADIUS", "float", 10.0),
    RuntimeEnvironmentField("enable_role_shuffle", "ENABLE_ROLE_SHUFFLE", "bool", True),
    RuntimeEnvironmentField("init_vel_jitter_max", "INIT_VEL_JITTER_MAX", "float", 0.3),
    # Terrain generation indexes a discrete height grid and parses both values
    # with int(os.getenv(...)); keep them integer end-to-end so exporting never
    # turns a valid "55" into the invalid integer literal "55.0".
    RuntimeEnvironmentField("mountain_min_distance", "MOUNTAIN_MIN_DISTANCE", "int", 55),
    RuntimeEnvironmentField("mountain_margin", "MOUNTAIN_MARGIN", "int", 20),
    RuntimeEnvironmentField("obstacle_min_clearance_start_goal", "OBSTACLE_MIN_CLEARANCE_START_GOAL", "float", 25.0),
    RuntimeEnvironmentField("terrain_collision_eps", "TERRAIN_COLLISION_EPS", "float", 0.3),
    RuntimeEnvironmentField("use_legacy_terrain", "USE_LEGACY_TERRAIN", "bool", False),
    RuntimeEnvironmentField("ring_base_reward", "RING_BASE_REWARD", "float", 80.0),
    RuntimeEnvironmentField("min_start_goal_dist", "MIN_START_GOAL_DIST", "float", 40.0),
    RuntimeEnvironmentField("max_start_goal_dist", "MAX_START_GOAL_DIST", "float", 100.0),
    RuntimeEnvironmentField("start_pos_margin", "START_POS_MARGIN", "float", 5.0),
    RuntimeEnvironmentField("start_pos_max_trials", "START_POS_MAX_TRIALS", "int", 2000),
)

_RESOLVED_MANIFEST_HASH_KEY = "content_sha256"


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _coerce(field: RuntimeEnvironmentField, value: Any) -> Any:
    if field.kind == "bool":
        return _coerce_bool(value)
    if field.kind == "int":
        return int(value)
    if field.kind == "float":
        return float(value)
    return str(value)


def _resolved_manifest_content_sha256(manifest: Mapping[str, Any]) -> str:
    """Compute the immutable-manifest digest used by the ablation launcher."""
    payload = copy.deepcopy(dict(manifest))
    meta = payload.get("meta")
    if isinstance(meta, dict):
        meta.pop(_RESOLVED_MANIFEST_HASH_KEY, None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def infer_training_manifest_identity(exp_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(label, batch_dir_name)`` encoded in an ablation exp name."""
    normalized = str(exp_name or "").strip()
    if not normalized or "__seed" not in normalized or "__batch_" not in normalized:
        return None, None
    label = normalized.split("__seed", 1)[0]
    suffix = normalized.split("__batch_", 1)[1]
    # The final two underscore-delimited tokens are the launch date/time that
    # run_optimized.sh appends to the frozen base experiment name.
    parts = suffix.rsplit("_", 2)
    if len(parts) != 3 or not parts[0]:
        return None, None
    return label, "batch_" + parts[0]


def find_training_runtime_manifest(
    repo_root: os.PathLike | str,
    exp_name: str,
    explicit_path: os.PathLike | str | None = None,
) -> Optional[Path]:
    """Find exactly one frozen manifest for a training experiment.

    An explicit path is preferred.  Legacy results did not record that path,
    so their label and batch identity are recovered from ``exp_name``.  An
    ambiguous match is rejected rather than selecting the first filesystem hit.
    """
    root = Path(repo_root).expanduser().resolve()
    explicit_raw = str(explicit_path or "").strip()
    if explicit_raw:
        explicit = Path(explicit_raw).expanduser()
        explicit_candidates = [explicit]
        if not explicit.is_absolute():
            explicit_candidates.insert(0, root / explicit)
        for candidate in explicit_candidates:
            if candidate.is_file():
                return candidate.resolve()

    label, batch_dir = infer_training_manifest_identity(exp_name)
    if not label or not batch_dir:
        if explicit_raw:
            raise FileNotFoundError(
                f"训练结果记录的 resolved manifest 不存在: {explicit_raw}"
            )
        return None
    pattern = (
        f"ablation_experiments/**/seed_batches/{batch_dir}/manifests/"
        f"{label}_resolved_manifest.json"
    )
    candidates = sorted(path.resolve() for path in root.glob(pattern) if path.is_file())
    if not candidates:
        detail = f"；记录路径={explicit_raw}" if explicit_raw else ""
        raise FileNotFoundError(
            f"无法定位训练 resolved manifest: exp_name={exp_name}{detail}"
        )
    if len(candidates) != 1:
        joined = ", ".join(str(path) for path in candidates)
        raise RuntimeError(
            f"训练 resolved manifest 匹配不唯一: exp_name={exp_name}; candidates=[{joined}]"
        )
    return candidates[0]


def load_training_runtime_manifest(
    manifest_path: os.PathLike | str,
    *,
    exp_name: str | None = None,
    expected_content_sha256: str | None = None,
) -> Dict[str, Any]:
    """Load and verify a frozen training manifest and its experiment identity."""
    path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"训练 resolved manifest 顶层必须是对象: {path}")
    exec_env = payload.get("exec_env")
    if not isinstance(exec_env, dict):
        raise TypeError(f"训练 resolved manifest 缺少 exec_env 对象: {path}")

    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        raise TypeError(f"训练 resolved manifest 的 meta 必须是对象: {path}")
    recorded_sha256 = str(meta.get(_RESOLVED_MANIFEST_HASH_KEY, "") or "").strip()
    if recorded_sha256:
        actual_sha256 = _resolved_manifest_content_sha256(payload)
        if actual_sha256 != recorded_sha256:
            raise RuntimeError(
                "训练 resolved manifest 内容指纹不匹配: "
                f"path={path}, recorded={recorded_sha256}, actual={actual_sha256}"
            )
    expected_sha256 = str(expected_content_sha256 or "").strip()
    if expected_sha256:
        if not recorded_sha256:
            raise RuntimeError(f"训练结果要求 manifest 指纹，但文件未记录指纹: {path}")
        if expected_sha256 != recorded_sha256:
            raise RuntimeError(
                "训练结果与 resolved manifest 指纹不一致: "
                f"path={path}, results={expected_sha256}, manifest={recorded_sha256}"
            )

    normalized_exp_name = str(exp_name or "").strip()
    if normalized_exp_name:
        recorded_names = {
            str(meta.get(key) or "").strip()
            for key in ("exp_name", "exp_name_base", "exp_name_with_timestamp")
            if str(meta.get(key) or "").strip()
        }
        # New manifests record exp_name_with_timestamp exactly.  Older ones
        # record only the timestamp-free base, which must still be a full prefix
        # followed by one launch timestamp (YYYYMMDD_HHMMSS).
        identity_matches = normalized_exp_name in recorded_names
        if not identity_matches:
            for recorded_name in recorded_names:
                suffix = normalized_exp_name[len(recorded_name):]
                if normalized_exp_name.startswith(recorded_name) and len(suffix) == 16:
                    identity_matches = (
                        suffix.startswith("_")
                        and suffix[1:9].isdigit()
                        and suffix[9] == "_"
                        and suffix[10:].isdigit()
                    )
                    if identity_matches:
                        break
        if recorded_names and not identity_matches:
            raise RuntimeError(
                "训练模型与 resolved manifest 身份不一致: "
                f"exp_name={normalized_exp_name}, manifest_names={sorted(recorded_names)}, path={path}"
            )
    return payload


def runtime_environment_from_manifest(
    manifest: Mapping[str, Any],
    *,
    include_code_defaults: bool = True,
) -> Dict[str, Any]:
    """Decode shared runtime fields from a frozen manifest's ``exec_env``.

    A missing environment entry means the scenario used its code default at
    training time.  ``include_code_defaults`` therefore reconstructs those
    values from the same canonical table instead of from evaluation defaults.
    """
    exec_env = manifest.get("exec_env", {}) if isinstance(manifest, Mapping) else {}
    if not isinstance(exec_env, Mapping):
        raise TypeError("训练 resolved manifest 的 exec_env 必须是对象")
    resolved: Dict[str, Any] = {}
    for field in SCIENTIFIC_RUNTIME_ENV_FIELDS:
        raw_value = exec_env.get(field.env)
        if raw_value is None or str(raw_value).strip() == "":
            if not include_code_defaults:
                continue
            raw_value = field.default
        resolved[field.attr] = _coerce(field, raw_value)
    return resolved


def _training_result_candidates(
    model_dir: Path,
    repo_root: Optional[os.PathLike | str],
) -> List[Path]:
    """Return result files that can prove a whole training unit finished.

    MATD3/MADDPG mirror ``results.json`` into the model root.  The external
    MAPPO trainer historically wrote it only under ``logs/<exp_name>``.  A
    checkpoint state is intentionally not a candidate: it proves only that a
    snapshot was written, not that the training process completed its final
    result write and shutdown path.
    """
    candidates = [model_dir / "results.json"]
    if repo_root is not None:
        log_root = Path(repo_root).expanduser().resolve() / "logs" / model_dir.name
        candidates.append(log_root / "results.json")
        if log_root.is_dir():
            candidates.extend(sorted(log_root.glob("*/results.json"), reverse=True))

    unique: List[Path] = []
    seen = set()
    for candidate in candidates:
        try:
            key = str(candidate.resolve())
        except Exception:
            key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _result_completion_errors(
    payload: Mapping[str, Any],
    *,
    model_name: str,
    expected_episodes: int,
    expected_seed: Optional[int],
    expected_num_envs: Optional[int],
    require_gpu: bool,
) -> List[str]:
    errors: List[str] = []
    try:
        recorded_episodes = int(payload.get("episodes", 0) or 0)
    except (TypeError, ValueError):
        recorded_episodes = -1
    if recorded_episodes != int(expected_episodes):
        errors.append(
            f"results.episodes={recorded_episodes}, expected={int(expected_episodes)}"
        )

    rewards = payload.get("rewards")
    if not isinstance(rewards, list) or len(rewards) != int(expected_episodes):
        errors.append(
            "results.rewards length="
            f"{len(rewards) if isinstance(rewards, list) else 'invalid'}, "
            f"expected={int(expected_episodes)}"
        )
    else:
        try:
            rewards_are_finite = all(math.isfinite(float(value)) for value in rewards)
        except (TypeError, ValueError, OverflowError):
            rewards_are_finite = False
        if not rewards_are_finite:
            errors.append("results.rewards contains non-numeric or non-finite values")

    run_args = payload.get("args")
    if not isinstance(run_args, Mapping):
        errors.append("results.args is missing or invalid")
        run_args = {}
    try:
        declared_train_episodes = int(run_args.get("train_episodes", 0) or 0)
    except (TypeError, ValueError):
        declared_train_episodes = -1
    if declared_train_episodes != int(expected_episodes):
        errors.append(
            "results.args.train_episodes="
            f"{declared_train_episodes}, expected={int(expected_episodes)}"
        )

    recorded_exp_name = str(run_args.get("exp_name", "") or "").strip()
    if recorded_exp_name != str(model_name):
        errors.append(
            f"results.args.exp_name={recorded_exp_name!r}, expected={str(model_name)!r}"
        )
    if expected_seed is not None:
        try:
            recorded_seed = int(run_args.get("seed"))
        except (TypeError, ValueError):
            recorded_seed = None
        if recorded_seed != int(expected_seed):
            errors.append(
                f"results.args.seed={recorded_seed!r}, expected={int(expected_seed)}"
            )
    if expected_num_envs is not None:
        try:
            recorded_num_envs = int(run_args.get("num_envs"))
        except (TypeError, ValueError):
            recorded_num_envs = None
        if recorded_num_envs != int(expected_num_envs):
            errors.append(
                "results.args.num_envs="
                f"{recorded_num_envs!r}, expected={int(expected_num_envs)}"
            )
        parallelism = payload.get("training_parallelism")
        if not isinstance(parallelism, Mapping):
            errors.append(
                "results.training_parallelism is missing while num_envs "
                "identity is required"
            )
        else:
            expected_parallelism = {
                "num_envs": int(expected_num_envs),
                "synchronous_iterations": int(expected_episodes),
                "environment_trajectories": (
                    int(expected_num_envs) * int(expected_episodes)
                ),
                "reward_aggregation": "equal_mean_across_environments",
                "success_aggregation": "equal_mean_across_environments",
                "worker_seed_derivation": (
                    "base_seed_plus_env_id_times_100003"
                ),
                "episode_audit_snapshot_schema_version": 1,
            }
            for key, expected_value in expected_parallelism.items():
                actual_value = parallelism.get(key)
                if actual_value != expected_value:
                    errors.append(
                        f"results.training_parallelism.{key}="
                        f"{actual_value!r}, expected={expected_value!r}"
                    )

    training_environment = payload.get("training_environment")
    if not isinstance(training_environment, Mapping) or not training_environment:
        errors.append("results.training_environment is missing or invalid")
    if require_gpu:
        training_device = payload.get("training_device")
        if not isinstance(training_device, Mapping):
            errors.append(
                "results.training_device is missing while GPU is required"
            )
        else:
            if training_device.get("require_gpu") is not True:
                errors.append(
                    "results.training_device.require_gpu is not true"
                )
            try:
                physical_gpus = int(
                    training_device.get("physical_gpus", 0) or 0
                )
                logical_gpus = int(
                    training_device.get("logical_gpus", 0) or 0
                )
            except (TypeError, ValueError):
                physical_gpus = 0
                logical_gpus = 0
            if physical_gpus < 1 or logical_gpus < 1:
                errors.append(
                    "results.training_device does not record a physical and "
                    "logical GPU"
                )
    return errors


def training_unit_completion_errors(
    model_dir: os.PathLike | str,
    expected_episodes: int,
    *,
    repo_root: Optional[os.PathLike | str] = None,
    expected_agents: int = 3,
    expected_seed: Optional[int] = None,
    expected_num_envs: Optional[int] = None,
    require_gpu: bool = False,
) -> List[str]:
    """Validate completion of one whole ``(algorithm, seed)`` training unit.

    Completion requires the final result record and the final model files.  A
    periodic ``epN`` directory or ``checkpoint_state.json`` never makes a unit
    complete, even when its episode counter equals the requested total.
    """
    model_root = Path(model_dir).expanduser().resolve()
    errors: List[str] = []
    if not model_root.is_dir():
        return [f"model directory does not exist: {model_root}"]
    if int(expected_episodes) <= 0:
        return [f"expected_episodes must be positive: {expected_episodes}"]
    if int(expected_agents) <= 0:
        return [f"expected_agents must be positive: {expected_agents}"]
    if expected_num_envs is not None and int(expected_num_envs) <= 0:
        return [f"expected_num_envs must be positive: {expected_num_envs}"]

    valid_result: Optional[Mapping[str, Any]] = None
    result_failures: List[str] = []
    for result_path in _training_result_candidates(model_root, repo_root):
        if not result_path.is_file():
            continue
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as exc:
            result_failures.append(f"{result_path}: unreadable ({exc})")
            continue
        if not isinstance(payload, Mapping):
            result_failures.append(f"{result_path}: root is not an object")
            continue
        candidate_errors = _result_completion_errors(
            payload,
            model_name=model_root.name,
            expected_episodes=int(expected_episodes),
            expected_seed=expected_seed,
            expected_num_envs=expected_num_envs,
            require_gpu=bool(require_gpu),
        )
        if candidate_errors:
            result_failures.append(
                f"{result_path}: " + "; ".join(candidate_errors)
            )
            continue
        valid_result = payload
        break
    if valid_result is None:
        if result_failures:
            errors.append("no valid final results.json: " + " | ".join(result_failures[:4]))
        else:
            errors.append("no final results.json in model/log identity")

    final_dir = model_root / "final"
    if not final_dir.is_dir():
        errors.append(f"final model directory is missing: {final_dir}")
        return errors

    expected_actor_names = {
        f"actor_{index}.weights.h5" for index in range(int(expected_agents))
    }
    actual_actor_names = {
        path.name
        for path in final_dir.glob("actor_*.weights.h5")
        if re.fullmatch(r"actor_\d+\.weights\.h5", path.name)
    }
    if actual_actor_names != expected_actor_names:
        errors.append(
            "final actor inventory mismatch: "
            f"got={sorted(actual_actor_names)}, expected={sorted(expected_actor_names)}"
        )

    def require_files(names: Iterable[str]) -> None:
        for name in names:
            path = final_dir / name
            if not path.is_file() or path.stat().st_size <= 0:
                errors.append(f"final model file missing/empty: {path}")

    require_files(expected_actor_names)
    if valid_result is not None:
        run_args = valid_result.get("args", {})
        algo = str(
            (run_args.get("algo") if isinstance(run_args, Mapping) else None)
            or valid_result.get("algo")
            or valid_result.get("algorithm")
            or ""
        ).strip().lower()
        if algo == "matd3":
            require_files(
                name
                for index in range(int(expected_agents))
                for name in (
                    f"critic1_{index}.weights.h5",
                    f"critic2_{index}.weights.h5",
                )
            )
            selector_mode = str(
                run_args.get("cross_agent_reference_selector_mode", "hard")
                or "hard"
            ).strip().lower()
            if selector_mode in ADVANTAGE_SELECTOR_MODES:
                adaptive_required_args = {
                    "cross_agent_reference_enabled": True,
                    "matd3_use_dual_q": True,
                    "matd3_use_separated_gradient": True,
                    "matd3_use_hybrid_actor_objective": False,
                    "matd3_action_semantics_mode": "dual",
                    "matd3_reconstruct_corrected_target": True,
                    "cross_agent_reference_use_clean_label": False,
                    "cross_agent_reference_target_semantics": (
                        "split_raw_head_corrected_tail"
                    ),
                    "cross_agent_reference_exclude_random": True,
                    "cross_agent_reference_quality_gate": True,
                    "cross_agent_reference_gate_mode": "agent_quality",
                }
                for key, expected_value in adaptive_required_args.items():
                    actual_value = run_args.get(key)
                    if isinstance(expected_value, bool):
                        matches = _coerce_bool(actual_value) == expected_value
                    else:
                        matches = (
                            str(actual_value or "").strip().lower()
                            == str(expected_value).strip().lower()
                        )
                    if not matches:
                        errors.append(
                            f"results.args.{key}={actual_value!r}, "
                            f"expected={expected_value!r}"
                        )

                if selector_mode == MODE_SHARED_TWIN_HEAD_TAIL:
                    if not _coerce_bool(
                        run_args.get(
                            "cross_agent_reference_selector_enabled",
                            False,
                        )
                    ):
                        errors.append(
                            "shared selector result does not declare "
                            "cross_agent_reference_selector_enabled=true"
                        )
                    require_files(("reference_selector_shared.weights.h5",))
                elif selector_mode == MODE_ADAPTIVE_TWIN_HEAD_TAIL:
                    if _coerce_bool(
                        run_args.get(
                            "cross_agent_reference_selector_enabled",
                            False,
                        )
                    ):
                        errors.append(
                            "direct adaptive mode must not declare a trainable "
                            "selector"
                        )

                selector_state_path = (
                    final_dir / "cross_agent_reference_state.json"
                )
                legacy_selector_files = sorted(
                    path.name
                    for path in final_dir.glob(
                        "reference_selector_[0-9]*.weights.h5"
                    )
                )
                if legacy_selector_files:
                    errors.append(
                        "adaptive selector model contains retired per-agent "
                        f"selector weights: {legacy_selector_files}"
                    )
                selector_payload: Any = None
                try:
                    selector_payload = json.loads(
                        selector_state_path.read_text(encoding="utf-8")
                    )
                except Exception as exc:
                    errors.append(
                        f"selector state missing/unreadable: "
                        f"{selector_state_path} ({exc})"
                    )
                if selector_payload is not None:
                    expected_selector_input_dim: Optional[int] = None
                    if selector_mode == MODE_SHARED_TWIN_HEAD_TAIL:
                        obs_shapes = run_args.get("base_obs_shapes")
                        action_dims = run_args.get("base_action_dims")
                        try:
                            uniform_shapes = (
                                isinstance(obs_shapes, list)
                                and bool(obs_shapes)
                                and isinstance(action_dims, list)
                                and bool(action_dims)
                                and len(
                                    set(int(value) for value in obs_shapes)
                                )
                                == 1
                                and len(
                                    set(int(value) for value in action_dims)
                                )
                                == 1
                            )
                        except (TypeError, ValueError):
                            uniform_shapes = False
                        if uniform_shapes:
                            expected_selector_input_dim = (
                                int(obs_shapes[0])
                                + 3 * int(action_dims[0])
                                + 1
                            )
                            if _coerce_bool(
                                run_args.get("use_pf_feature", False)
                            ):
                                expected_selector_input_dim += int(
                                    run_args.get("pf_feature_dim", 0) or 0
                                )
                        else:
                            errors.append(
                                "shared selector results.args lacks uniform "
                                "base_obs_shapes/base_action_dims"
                            )
                    errors.extend(
                        selector_state_errors(
                            selector_payload,
                            expected_mode=selector_mode,
                            expected_input_dim=expected_selector_input_dim,
                            require_null_input_dim=(
                                selector_mode
                                == MODE_ADAPTIVE_TWIN_HEAD_TAIL
                            ),
                        )
                    )
                    state_argument_pairs = (
                        (
                            "ema_decay",
                            "cross_agent_reference_advantage_ema_decay",
                        ),
                        (
                            "epsilon",
                            "cross_agent_reference_advantage_epsilon",
                        ),
                        (
                            "advantage_clip",
                            "cross_agent_reference_selector_adv_clip",
                        ),
                    )
                    for state_key, argument_key in state_argument_pairs:
                        try:
                            state_value = float(selector_payload[state_key])
                            argument_value = float(run_args[argument_key])
                            values_match = math.isclose(
                                state_value,
                                argument_value,
                                rel_tol=1e-12,
                                abs_tol=1e-12,
                            )
                        except (
                            KeyError,
                            TypeError,
                            ValueError,
                            OverflowError,
                        ):
                            values_match = False
                        if not values_match:
                            errors.append(
                                f"selector state {state_key} does not match "
                                f"results.args.{argument_key}"
                            )
        elif algo == "maddpg":
            require_files(
                f"critic_{index}.weights.h5"
                for index in range(int(expected_agents))
            )
        elif algo == "mappo":
            require_files(("value_critic.weights.h5", "actor_log_std.npy", "mappo_meta.json"))
        else:
            errors.append(f"unsupported/unknown algorithm in final results: {algo!r}")
    return errors


def training_unit_is_complete(
    model_dir: os.PathLike | str,
    expected_episodes: int,
    **kwargs: Any,
) -> bool:
    return not training_unit_completion_errors(
        model_dir,
        expected_episodes,
        **kwargs,
    )


def resolve_runtime_environment(args=None) -> Dict[str, Any]:
    """Resolve typed values, preferring explicit args over the environment."""
    resolved: Dict[str, Any] = {}
    for field in SCIENTIFIC_RUNTIME_ENV_FIELDS:
        value = getattr(args, field.attr, None) if args is not None else None
        if value is None:
            value = os.getenv(field.env, field.default)
        resolved[field.attr] = _coerce(field, value)
    return resolved


def apply_runtime_environment(args) -> Dict[str, Any]:
    """Resolve, store on ``args``, and export every scientific runtime field."""
    resolved = resolve_runtime_environment(args)
    fields_by_attr = {field.attr: field for field in SCIENTIFIC_RUNTIME_ENV_FIELDS}
    for attr, value in resolved.items():
        setattr(args, attr, value)
        field = fields_by_attr[attr]
        if field.kind == "bool":
            os.environ[field.env] = "1" if value else "0"
        else:
            os.environ[field.env] = str(value)
    return resolved


def add_runtime_environment_arguments(parser, *, skip_options: Iterable[str] = ()):
    """Expose the shared fields on an argparse parser without duplicate options."""
    skipped = set(skip_options)
    existing = getattr(parser, "_option_string_actions", {})
    for field in SCIENTIFIC_RUNTIME_ENV_FIELDS:
        option = field.cli_option
        if option in skipped or option in existing:
            continue
        raw_default = os.getenv(field.env, field.default)
        default = _coerce(field, raw_default)
        if field.kind == "bool":
            value_type = _coerce_bool
        elif field.kind == "int":
            value_type = int
        elif field.kind == "float":
            value_type = float
        else:
            value_type = str
        parser.add_argument(
            option,
            dest=field.attr,
            type=value_type,
            default=default,
            help=f"科学运行参数 {field.env}（训练、续训和评估必须一致）",
        )
    return parser


__all__ = [
    "SCIENTIFIC_RUNTIME_ENV_FIELDS",
    "RuntimeEnvironmentField",
    "add_runtime_environment_arguments",
    "apply_runtime_environment",
    "find_training_runtime_manifest",
    "infer_training_manifest_identity",
    "load_training_runtime_manifest",
    "resolve_runtime_environment",
    "runtime_environment_from_manifest",
    "training_unit_completion_errors",
    "training_unit_is_complete",
]
