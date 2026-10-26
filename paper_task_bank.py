"""Immutable train/validation/test task-bank manifests for paper experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping

from paper_task_layout import (
    atomic_write_json,
    canonical_sha256,
    validate_task_layout_payload,
    validate_task_difficulty_acceptance,
    validate_task_deadline_contract,
)
from paper_curriculum import paper_physics_contract, paper_task_semantics_contract


PAPER_TASK_BANK_SCHEMA_VERSION = "paper_task_bank_v10"
PAPER_POSITIONS_SCHEMA_VERSION = "paper_positions_v1"
ALLOWED_SPLIT_ROLES = frozenset(
    {"train", "validation", "test_in_distribution", "test_generalization"}
)
_SAFE_MEMBER_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_DYNAMIC_AUDIT_CONTROLLER = {
    "policy": (
        "surface_astar_shared_altitude_layers_sync_then_parallel_"
        "vertical_goal_settle"
    ),
    "terrain_and_obstacle_geometry_visibility": "oracle",
    "kp": 0.12,
    "kd": 0.35,
    "z_kp": 0.20,
    "z_kd": 0.20,
    "surface_clearance_margin_m": 4.0,
    "inter_agent_cruise_vertical_separation_m": 3.0,
    "waypoint_capture_radius_m": 1.5,
    "obstacle_inflation_policy": (
        "max_task_reference_and_dynamic_tracking_margin"
    ),
    "dynamic_obstacle_inflation_m": 2.0,
    "terrain_sampling_spacing_m": 0.5,
    "goal_settle_policy": (
        "all_corridors_then_parallel_vertical_at_exact_goal_xy"
    ),
    "goal_success_detection_policy": (
        "environment_terminal_radius_independent_of_controller_phase"
    ),
    "parallel_goal_settle_requires_disjoint_goal_columns": True,
    "dynamics_mode": "quadrotor",
    "physics_contract": paper_physics_contract(),
    "task_semantics": paper_task_semantics_contract(),
}


class TaskBankError(ValueError):
    pass


def _resolve_bank_member(root: Path, relative_path: Any, field: str) -> Path:
    raw = str(relative_path or "").strip()
    if not raw:
        raise TaskBankError(f"task-bank {field} cannot be empty")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise TaskBankError(
            f"task-bank {field} must be a safe relative path: {raw!r}"
        )
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise TaskBankError(
            f"task-bank {field} escapes the immutable bank root: {raw!r}"
        ) from exc
    return resolved


def summarize_task_deadlines(layouts: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return the exact finite-horizon range represented by task layouts."""
    normalized = list(layouts)
    if not normalized:
        raise TaskBankError("cannot summarize deadlines for an empty task split")
    steps: List[int] = []
    seconds: List[float] = []
    policy_ids = set()
    simulation_dts = set()
    for index, layout in enumerate(normalized):
        deadline = dict((layout.get("contract") or {}).get("deadline") or {})
        try:
            deadline_steps = int(deadline["deadline_steps"])
            deadline_seconds = float(deadline["deadline_seconds"])
            simulation_dt = float(deadline["simulation_dt_seconds"])
            policy_id = str(deadline["policy_id"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise TaskBankError(
                f"task index={index} has an incomplete deadline contract"
            ) from exc
        if deadline_steps <= 0 or deadline_seconds <= 0.0 or simulation_dt <= 0.0:
            raise TaskBankError(
                f"task index={index} has a non-positive deadline contract"
            )
        if abs(deadline_seconds - deadline_steps * simulation_dt) > 1e-9:
            raise TaskBankError(
                f"task index={index} deadline seconds/steps/dt are inconsistent"
            )
        if not policy_id:
            raise TaskBankError(f"task index={index} deadline policy_id is empty")
        steps.append(deadline_steps)
        seconds.append(deadline_seconds)
        policy_ids.add(policy_id)
        simulation_dts.add(simulation_dt)
    if len(policy_ids) != 1 or len(simulation_dts) != 1:
        raise TaskBankError(
            "one task split cannot mix deadline policies or simulation time steps"
        )
    return {
        "task_count": len(normalized),
        "policy_id": next(iter(policy_ids)),
        "simulation_dt_seconds": next(iter(simulation_dts)),
        "minimum_deadline_steps": min(steps),
        "maximum_deadline_steps": max(steps),
        "minimum_deadline_seconds": min(seconds),
        "maximum_deadline_seconds": max(seconds),
    }


def validate_dynamic_feasibility_audit(
    manifest: Mapping[str, Any]
) -> List[str]:
    """Validate the task-level oracle witness embedded by the formal builder."""
    audit = dict((manifest.get("metadata") or {}).get("dynamic_feasibility_audit") or {})
    errors: List[str] = []
    if audit.get("schema_version") != "paper_dynamic_feasibility_audit_v11":
        errors.append("dynamic feasibility audit schema mismatch")
    if not bool(audit.get("performed", False)):
        errors.append("dynamic feasibility audit was not performed")
    if not bool(audit.get("all_tasks_dynamically_feasible", False)):
        errors.append("dynamic feasibility audit did not pass every task")
    if audit.get("audit_role") != (
        "oracle_dynamic_feasibility_witness_not_comparison_baseline"
    ):
        errors.append("dynamic feasibility audit role mismatch")
    controller = audit.get("controller")
    if not isinstance(controller, Mapping) or dict(controller) != (
        _DYNAMIC_AUDIT_CONTROLLER
    ):
        errors.append("dynamic feasibility audit controller contract mismatch")
    if not str(audit.get("staging_task_bank_content_sha256", "")).strip():
        errors.append("dynamic feasibility audit lacks staging task-bank hash")
    split_audits = dict(audit.get("splits") or {})
    manifest_splits = dict(manifest.get("splits") or {})
    if set(split_audits) != set(manifest_splits):
        errors.append("dynamic feasibility audit split set mismatch")
        return errors
    for split_name, split in manifest_splits.items():
        split_audit = dict(split_audits.get(split_name) or {})
        split_runtime = dict(split.get("runtime_config") or {})
        split_physics = dict(
            split_runtime.get(
                "paper_physics_contract"
            )
            or {}
        )
        if split_physics != paper_physics_contract():
            errors.append(
                f"dynamic feasibility physics contract mismatch in split={split_name}"
            )
        split_semantics = dict(
            split_runtime.get(
                "paper_task_semantics_contract"
            )
            or {}
        )
        if split_semantics != paper_task_semantics_contract():
            errors.append(
                f"dynamic feasibility task semantics mismatch in split={split_name}"
            )
        if str(split_audit.get("split_role", "")) != str(split.get("role", "")):
            errors.append(
                f"dynamic feasibility split role mismatch in split={split_name}"
            )
        expected_entries = list(split.get("tasks") or [])
        episodes = list(split_audit.get("episodes") or [])
        expected_ids = [str(entry.get("task_id", "")) for entry in expected_entries]
        actual_ids = [str(episode.get("task_id", "")) for episode in episodes]
        if actual_ids != expected_ids:
            errors.append(
                f"dynamic feasibility audit task order mismatch in split={split_name}"
            )
        if int(split_audit.get("tasks_audited", -1)) != len(expected_entries):
            errors.append(
                f"dynamic feasibility audited count mismatch in split={split_name}"
            )
        if int(split_audit.get("team_successes", -1)) != len(expected_entries):
            errors.append(
                f"dynamic feasibility success count mismatch in split={split_name}"
            )
        if not bool(split_audit.get("all_tasks_dynamically_feasible", False)):
            errors.append(
                f"dynamic feasibility split did not pass: {split_name}"
            )
        for episode, entry in zip(episodes, expected_entries):
            if not bool(episode.get("team_success", False)):
                errors.append(
                    f"dynamic feasibility task failed: {entry.get('task_id')}"
                )
            if str(episode.get("terminal_reason", "")) != "team_success":
                errors.append(
                    f"dynamic feasibility terminal mismatch: {entry.get('task_id')}"
                )
            if str(episode.get("difficulty_bin", "")) != str(
                entry.get("difficulty_bin", "")
            ):
                errors.append(
                    f"dynamic feasibility difficulty mismatch: {entry.get('task_id')}"
                )
            if [int(value) for value in list(
                episode.get("agent_success_flags") or []
            )] != [1, 1, 1]:
                errors.append(
                    f"dynamic feasibility agent success mismatch: {entry.get('task_id')}"
                )
            if not bool(episode.get("all_done", False)):
                errors.append(
                    f"dynamic feasibility all_done mismatch: {entry.get('task_id')}"
                )
            if str(episode.get("dynamics_mode", "")) != "quadrotor":
                errors.append(
                    f"dynamic feasibility dynamics mismatch: {entry.get('task_id')}"
                )
            if list(episode.get("quadrotor_state_initialized") or []) != [
                True, True, True
            ]:
                errors.append(
                    f"dynamic feasibility quadrotor witness incomplete: {entry.get('task_id')}"
                )
            if list(episode.get("controller_path_completed") or []) != [
                True, True, True
            ]:
                errors.append(
                    f"dynamic feasibility path witness incomplete: {entry.get('task_id')}"
                )
            if list(episode.get("controller_goal_success_observed") or []) != [
                True, True, True
            ]:
                errors.append(
                    "dynamic feasibility goal-success witness incomplete: "
                    f"{entry.get('task_id')}"
                )
            reference_lengths = list(
                episode.get("reference_path_lengths_m") or []
            )
            witness_lengths = list(
                episode.get("controller_witness_route_lengths_m") or []
            )
            try:
                valid_route_lengths = bool(
                    len(reference_lengths) == 3
                    and len(witness_lengths) == 3
                    and all(
                        math.isfinite(float(value)) and float(value) > 0.0
                        for value in reference_lengths + witness_lengths
                    )
                )
            except (TypeError, ValueError):
                valid_route_lengths = False
            if not valid_route_lengths:
                errors.append(
                    "dynamic feasibility route-length witness invalid: "
                    f"{entry.get('task_id')}"
                )
            else:
                expected_max_reference = float(
                    (entry.get("deadline") or {}).get(
                        "maximum_reference_path_length_m", float("nan")
                    )
                )
                if not math.isclose(
                    max(float(value) for value in reference_lengths),
                    expected_max_reference,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    errors.append(
                        "dynamic feasibility reference-length/deadline mismatch: "
                        f"{entry.get('task_id')}"
                    )
            try:
                controller_inflation = float(
                    episode.get("controller_obstacle_inflation_m")
                )
                expected_controller_inflation = max(
                    float(split_runtime["paper_task_obstacle_inflation"]),
                    float(_DYNAMIC_AUDIT_CONTROLLER[
                        "dynamic_obstacle_inflation_m"
                    ]),
                )
                inflation_valid = (
                    math.isfinite(controller_inflation)
                    and math.isclose(
                        controller_inflation,
                        expected_controller_inflation,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                )
            except (KeyError, TypeError, ValueError):
                inflation_valid = False
            if not inflation_valid:
                errors.append(
                    "dynamic feasibility obstacle-inflation witness invalid: "
                    f"{entry.get('task_id')}"
                )
            try:
                minimum_clearance = float(
                    episode.get("episode_minimum_clearance_m")
                )
                clearance_valid = (
                    math.isfinite(minimum_clearance)
                    and minimum_clearance
                    > float(split_semantics["collision_distance_threshold"])
                )
            except (KeyError, TypeError, ValueError):
                clearance_valid = False
            if not clearance_valid:
                errors.append(
                    "dynamic feasibility safety-clearance witness invalid: "
                    f"{entry.get('task_id')}"
                )
            waypoint_capture = episode.get(
                "controller_waypoint_capture_completed"
            )
            if (
                not isinstance(waypoint_capture, list)
                or len(waypoint_capture) != 3
                or any(not isinstance(value, bool) for value in waypoint_capture)
            ):
                errors.append(
                    "dynamic feasibility waypoint-capture witness invalid: "
                    f"{entry.get('task_id')}"
                )
            else:
                final_distances = list(
                    episode.get("final_goal_distances_m") or []
                )
                success_threshold = float(
                    split_semantics.get(
                        "success_distance_threshold", float("nan")
                    )
                )
                if len(final_distances) != 3:
                    errors.append(
                        "dynamic feasibility final-distance witness invalid: "
                        f"{entry.get('task_id')}"
                    )
                else:
                    for agent_index, captured in enumerate(waypoint_capture):
                        if captured:
                            continue
                        try:
                            legally_terminal = (
                                int(episode["agent_success_flags"][agent_index]) == 1
                                and float(final_distances[agent_index])
                                <= success_threshold
                            )
                        except (IndexError, TypeError, ValueError):
                            legally_terminal = False
                        if not legally_terminal:
                            errors.append(
                                "dynamic feasibility unfinished corridor lacks "
                                "a legal terminal success: "
                                f"{entry.get('task_id')} agent={agent_index}"
                            )
            try:
                executed_steps = int(episode.get("executed_steps", 0))
                deadline_steps = int(episode.get("deadline_steps", 0))
            except (TypeError, ValueError):
                executed_steps = 0
                deadline_steps = 0
            expected_deadline_steps = int(
                (entry.get("deadline") or {}).get("deadline_steps", 0) or 0
            )
            if deadline_steps != expected_deadline_steps:
                errors.append(
                    f"dynamic feasibility deadline mismatch: {entry.get('task_id')}"
                )
            if not 0 < executed_steps <= deadline_steps:
                errors.append(
                    f"dynamic feasibility step contract invalid: {entry.get('task_id')}"
                )
            try:
                deadline_usage = float(episode.get("deadline_usage_fraction"))
            except (TypeError, ValueError):
                deadline_usage = float("nan")
            if not math.isfinite(deadline_usage) or not math.isclose(
                deadline_usage,
                executed_steps / deadline_steps if deadline_steps > 0 else -1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                errors.append(
                    f"dynamic feasibility deadline usage mismatch: {entry.get('task_id')}"
                )
            completion_steps = list(
                episode.get("controller_corridor_completion_steps") or []
            )
            success_steps = list(
                episode.get("controller_goal_success_steps") or []
            )
            targeting_start_step = episode.get(
                "controller_goal_targeting_start_step"
            )
            valid_completion_steps = len(completion_steps) == 3
            valid_success_steps = len(success_steps) == 3
            if valid_completion_steps:
                for captured, value in zip(
                    list(waypoint_capture or []), completion_steps
                ):
                    if captured:
                        if (
                            isinstance(value, bool)
                            or not isinstance(value, int)
                            or not 0 <= value <= executed_steps
                        ):
                            valid_completion_steps = False
                    elif value is not None:
                        valid_completion_steps = False
            if valid_success_steps:
                for value in success_steps:
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or not 1 <= value <= executed_steps
                    ):
                        valid_success_steps = False
            if not valid_completion_steps:
                errors.append(
                    "dynamic feasibility corridor completion timing invalid: "
                    f"{entry.get('task_id')}"
                )
            if not valid_success_steps:
                errors.append(
                    "dynamic feasibility goal-success timing invalid: "
                    f"{entry.get('task_id')}"
                )
            if valid_completion_steps:
                all_corridors_captured = bool(
                    len(waypoint_capture or []) == 3
                    and all(waypoint_capture)
                    and all(value is not None for value in completion_steps)
                )
                if all_corridors_captured:
                    expected_targeting_start = max(completion_steps)
                    if (
                        isinstance(targeting_start_step, bool)
                        or not isinstance(targeting_start_step, int)
                        or targeting_start_step != expected_targeting_start
                        or not 0 <= targeting_start_step <= executed_steps
                    ):
                        errors.append(
                            "dynamic feasibility goal-targeting phase timing "
                            f"invalid: {entry.get('task_id')}"
                        )
                elif targeting_start_step is not None:
                    errors.append(
                        "dynamic feasibility goal-targeting phase began without "
                        f"all corridors captured: {entry.get('task_id')}"
                    )
    return errors


def _validated_layouts(layouts: Iterable[Mapping[str, Any]], split_name: str) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for index, raw in enumerate(layouts):
        payload = dict(raw)
        errors = validate_task_layout_payload(payload)
        if errors:
            raise TaskBankError(
                f"invalid task in split={split_name} index={index}: " + "; ".join(errors)
            )
        normalized.append(payload)
    if not normalized:
        raise TaskBankError(f"task split={split_name} cannot be empty")
    return normalized


def _validate_layout_runtime_contract(
    layout: Mapping[str, Any], runtime_config: Mapping[str, Any]
) -> List[str]:
    """Bind layout-defining runtime fields to their persisted task values."""
    contract = dict(layout.get("contract") or {})
    errors: List[str] = []
    if dict(runtime_config.get("paper_physics_contract") or {}) != (
        paper_physics_contract()
    ):
        errors.append("runtime_config paper_physics_contract mismatch")
    if dict(runtime_config.get("paper_task_semantics_contract") or {}) != (
        paper_task_semantics_contract()
    ):
        errors.append("runtime_config paper_task_semantics_contract mismatch")
    exact_pairs = (
        ("difficulty_bin", layout.get("difficulty_bin")),
        ("paper_task_num_obstacles", len(list(layout.get("obstacles") or []))),
        ("min_start_goal_dist", contract.get("min_start_goal_dist")),
        ("max_start_goal_dist", contract.get("max_start_goal_dist")),
        (
            "paper_task_require_reference_feasible",
            contract.get("require_reference_feasible"),
        ),
        (
            "paper_task_reference_grid_resolution",
            contract.get("reference_grid_resolution"),
        ),
        (
            "paper_task_obstacle_inflation",
            contract.get("reference_obstacle_inflation"),
        ),
        (
            "paper_task_reference_surface_clearance",
            contract.get("reference_surface_clearance"),
        ),
        (
            "success_distance_threshold",
            contract.get("success_distance_threshold"),
        ),
        (
            "collision_distance_threshold",
            contract.get("collision_distance_threshold"),
        ),
        ("paper_goal_formation_radius", contract.get("goal_formation_radius")),
        ("paper_goal_boundary_clearance", contract.get("goal_boundary_clearance")),
    )
    for runtime_name, persisted_value in exact_pairs:
        if runtime_name not in runtime_config:
            errors.append(f"runtime_config missing {runtime_name}")
            continue
        runtime_value = runtime_config[runtime_name]
        if isinstance(persisted_value, float) or isinstance(runtime_value, float):
            try:
                equal = abs(float(runtime_value) - float(persisted_value)) <= 1e-12
            except (TypeError, ValueError):
                equal = False
        else:
            equal = runtime_value == persisted_value
        if not equal:
            errors.append(
                f"runtime {runtime_name}={runtime_value!r} does not match "
                f"task contract value={persisted_value!r}"
            )
    return errors


def _assert_split_independence(
    split_layouts: Mapping[str, List[Dict[str, Any]]],
    split_roles: Mapping[str, str],
) -> None:
    task_owner: Dict[str, str] = {}
    stochastic_seed_owner: Dict[tuple, str] = {}
    terrain_by_role: Dict[str, set] = {}
    for split_name, layouts in split_layouts.items():
        role = split_roles[split_name]
        terrain_by_role.setdefault(role, set())
        for layout in layouts:
            task_id = str(layout["task_id"])
            previous = task_owner.get(task_id)
            if previous is not None:
                raise TaskBankError(
                    f"physical task_id={task_id} appears in both {previous} and {split_name}"
                )
            task_owner[task_id] = split_name
            seeds = dict(layout.get("seeds") or {})
            terrain_seed = seeds.get("terrain_variant")
            if terrain_seed is None:
                terrain_seed = seeds.get("terrain")
            if terrain_seed is not None:
                terrain_by_role[role].add(int(terrain_seed))
            active_seed_kinds = ["position"]
            if list(layout.get("obstacles") or []):
                active_seed_kinds.append("obstacle")
            for seed_kind in active_seed_kinds:
                seed = seeds.get(seed_kind)
                if seed is None:
                    continue
                key = (seed_kind, int(seed))
                owner = stochastic_seed_owner.get(key)
                if owner is not None and owner != split_name:
                    raise TaskBankError(
                        f"{seed_kind} seed={seed} leaks across splits {owner} and {split_name}"
                    )
                stochastic_seed_owner[key] = split_name

    train_terrain = terrain_by_role.get("train", set())
    generalization_terrain = terrain_by_role.get("test_generalization", set())
    overlap = sorted(train_terrain & generalization_terrain)
    if overlap:
        raise TaskBankError(
            f"generalization terrain seeds overlap training terrain seeds: {overlap}"
        )


def write_task_bank(
    output_dir: str | Path,
    *,
    bank_id: str,
    splits: Mapping[str, Iterable[Mapping[str, Any]]],
    split_roles: Mapping[str, str],
    metadata: Mapping[str, Any] | None = None,
    split_runtime_configs: Mapping[str, Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    bank_id = str(bank_id).strip()
    if not bank_id:
        raise TaskBankError("bank_id cannot be empty")
    if not splits:
        raise TaskBankError("task bank requires at least one split")
    if set(splits) != set(split_roles):
        raise TaskBankError("splits and split_roles must have identical keys")
    if split_runtime_configs is not None and set(split_runtime_configs) != set(splits):
        raise TaskBankError("split_runtime_configs must have the same keys as splits")
    normalized_roles: Dict[str, str] = {}
    normalized_splits: Dict[str, List[Dict[str, Any]]] = {}
    for raw_split_name, raw_role in split_roles.items():
        split_name = str(raw_split_name).strip()
        role = str(raw_role).strip()
        if (
            not split_name
            or split_name != str(raw_split_name)
            or _SAFE_MEMBER_NAME.fullmatch(split_name) is None
            or role not in ALLOWED_SPLIT_ROLES
        ):
            raise TaskBankError(f"invalid split/role pair: {split_name!r}={role!r}")
        normalized_roles[split_name] = role
        normalized_splits[split_name] = _validated_layouts(
            splits[raw_split_name], split_name
        )
        runtime_config = dict((split_runtime_configs or {}).get(split_name) or {})
        acceptance = runtime_config.get("difficulty_acceptance")
        deadline_policy = runtime_config.get("paper_task_deadline_policy")
        if not isinstance(acceptance, Mapping):
            raise TaskBankError(
                f"split={split_name} requires difficulty_acceptance in runtime_config"
            )
        if not isinstance(deadline_policy, Mapping):
            raise TaskBankError(
                f"split={split_name} requires paper_task_deadline_policy in runtime_config"
            )
        for index, layout in enumerate(normalized_splits[split_name]):
            runtime_errors = _validate_layout_runtime_contract(
                layout, runtime_config
            )
            if runtime_errors:
                raise TaskBankError(
                    f"task/runtime contract rejection in split={split_name} "
                    f"index={index}: " + "; ".join(runtime_errors)
                )
            acceptance_errors = validate_task_difficulty_acceptance(layout, acceptance)
            if acceptance_errors:
                raise TaskBankError(
                    f"task difficulty rejection in split={split_name} index={index}: "
                    + "; ".join(acceptance_errors)
                )
            deadline_errors = validate_task_deadline_contract(
                layout, deadline_policy
            )
            if deadline_errors:
                raise TaskBankError(
                    f"task deadline rejection in split={split_name} index={index}: "
                    + "; ".join(deadline_errors)
                )
    _assert_split_independence(normalized_splits, normalized_roles)

    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "task_bank_manifest.json"
    if root.exists() and any(root.iterdir()):
        raise TaskBankError(
            f"refusing to write into non-empty immutable task-bank directory: {root}"
        )

    manifest_splits: Dict[str, Any] = {}
    for split_name in sorted(normalized_splits):
        entries = []
        for layout in normalized_splits[split_name]:
            task_id = str(layout["task_id"])
            if _SAFE_MEMBER_NAME.fullmatch(task_id) is None:
                raise TaskBankError(f"unsafe task_id={task_id!r}")
            layout_rel = Path("tasks") / split_name / f"{task_id}.json"
            positions_rel = Path("positions") / split_name / f"{task_id}.json"
            atomic_write_json(root / layout_rel, layout)
            positions_payload = {
                "schema_version": PAPER_POSITIONS_SCHEMA_VERSION,
                "task_id": task_id,
                "agents": layout["agent_starts"],
                "goal": layout["central_goal"],
                "agent_goals": layout["agent_goals"],
                "task_layout_file": str(layout_rel),
            }
            positions_payload["content_sha256"] = canonical_sha256(positions_payload)
            atomic_write_json(root / positions_rel, positions_payload)
            entries.append(
                {
                    "task_id": task_id,
                    "difficulty_bin": layout.get("difficulty_bin"),
                    "task_layout_file": str(layout_rel),
                    "positions_file": str(positions_rel),
                    "task_content_sha256": layout["content_sha256"],
                    "positions_content_sha256": positions_payload["content_sha256"],
                    "seeds": layout.get("seeds"),
                    "deadline": dict(
                        (layout.get("contract") or {}).get("deadline") or {}
                    ),
                }
            )
        manifest_splits[split_name] = {
            "role": normalized_roles[split_name],
            "task_count": len(entries),
            "runtime_config": dict((split_runtime_configs or {}).get(split_name) or {}),
            "deadline_summary": summarize_task_deadlines(
                normalized_splits[split_name]
            ),
            "tasks": entries,
        }

    manifest: Dict[str, Any] = {
        "schema_version": PAPER_TASK_BANK_SCHEMA_VERSION,
        "bank_id": bank_id,
        "immutable": True,
        "split_independence": {
            "physical_task_ids_disjoint": True,
            "position_seeds_disjoint": True,
            "obstacle_seeds_disjoint": True,
            "generalization_terrain_seeds_disjoint_from_train": True,
        },
        "splits": manifest_splits,
        "metadata": dict(metadata or {}),
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_write_json(manifest_path, manifest)
    return manifest


def load_task_bank(path: str | Path) -> Dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "task_bank_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != PAPER_TASK_BANK_SCHEMA_VERSION:
        raise TaskBankError(f"task-bank schema mismatch: {manifest.get('schema_version')!r}")
    if manifest.get("content_sha256") != canonical_sha256(manifest):
        raise TaskBankError("task-bank manifest content_sha256 mismatch")
    if not str(manifest.get("bank_id", "")).strip():
        raise TaskBankError("task-bank bank_id cannot be empty")
    if manifest.get("immutable") is not True:
        raise TaskBankError("task-bank manifest must declare immutable=true")
    manifest_splits = manifest.get("splits")
    if not isinstance(manifest_splits, Mapping) or not manifest_splits:
        raise TaskBankError("task-bank manifest must contain non-empty splits")
    root = manifest_path.parent
    persisted_layouts: Dict[str, List[Dict[str, Any]]] = {}
    persisted_roles: Dict[str, str] = {}
    member_paths: set[Path] = set()
    for split_name, split in dict(manifest_splits).items():
        if not str(split_name).strip() or not isinstance(split, Mapping):
            raise TaskBankError(f"invalid persisted split name/config={split_name!r}")
        tasks = list(split.get("tasks") or [])
        role = str(split.get("role", "")).strip()
        if role not in ALLOWED_SPLIT_ROLES:
            raise TaskBankError(
                f"persisted split={split_name} has invalid role={role!r}"
            )
        persisted_roles[split_name] = role
        persisted_layouts[split_name] = []
        runtime_config = dict(split.get("runtime_config") or {})
        acceptance = runtime_config.get("difficulty_acceptance")
        deadline_policy = runtime_config.get("paper_task_deadline_policy")
        if not isinstance(acceptance, Mapping):
            raise TaskBankError(
                f"persisted split={split_name} lacks difficulty_acceptance"
            )
        if not isinstance(deadline_policy, Mapping):
            raise TaskBankError(
                f"persisted split={split_name} lacks paper_task_deadline_policy"
            )
        if int(split.get("task_count", -1)) != len(tasks):
            raise TaskBankError(f"task count mismatch in split={split_name}")
        if not tasks:
            raise TaskBankError(f"persisted split={split_name} cannot be empty")
        for entry in tasks:
            if not isinstance(entry, Mapping):
                raise TaskBankError(
                    f"persisted split={split_name} has a non-object task entry"
                )
            layout_path = _resolve_bank_member(
                root, entry.get("task_layout_file"), "task_layout_file"
            )
            positions_path = _resolve_bank_member(
                root, entry.get("positions_file"), "positions_file"
            )
            for member_path in (layout_path, positions_path):
                if member_path in member_paths:
                    raise TaskBankError(
                        f"task-bank member path is reused: {member_path}"
                    )
                member_paths.add(member_path)
            with layout_path.open("r", encoding="utf-8") as handle:
                layout = json.load(handle)
            errors = validate_task_layout_payload(layout)
            if errors:
                raise TaskBankError(
                    f"invalid persisted task={layout_path}: " + "; ".join(errors)
                )
            if layout["task_id"] != entry["task_id"]:
                raise TaskBankError(f"task ID mismatch for {layout_path}")
            if layout["content_sha256"] != entry["task_content_sha256"]:
                raise TaskBankError(f"task content hash mismatch for {layout_path}")
            if layout.get("difficulty_bin") != entry.get("difficulty_bin"):
                raise TaskBankError(
                    f"task difficulty mismatch for {layout_path}"
                )
            if layout.get("seeds") != entry.get("seeds"):
                raise TaskBankError(f"task seeds mismatch for {layout_path}")
            if dict((layout.get("contract") or {}).get("deadline") or {}) != dict(
                entry.get("deadline") or {}
            ):
                raise TaskBankError(f"task deadline entry mismatch for {layout_path}")
            runtime_errors = _validate_layout_runtime_contract(
                layout, runtime_config
            )
            if runtime_errors:
                raise TaskBankError(
                    f"persisted task/runtime contract rejection for {layout_path}: "
                    + "; ".join(runtime_errors)
                )
            with positions_path.open("r", encoding="utf-8") as handle:
                positions = json.load(handle)
            if positions.get("schema_version") != PAPER_POSITIONS_SCHEMA_VERSION:
                raise TaskBankError(
                    f"positions schema mismatch for {positions_path}"
                )
            if positions.get("content_sha256") != canonical_sha256(positions):
                raise TaskBankError(
                    f"positions content hash mismatch for {positions_path}"
                )
            if positions.get("content_sha256") != entry.get(
                "positions_content_sha256"
            ):
                raise TaskBankError(
                    f"manifest positions hash mismatch for {positions_path}"
                )
            expected_positions = {
                "task_id": layout["task_id"],
                "agents": layout["agent_starts"],
                "goal": layout["central_goal"],
                "agent_goals": layout["agent_goals"],
                "task_layout_file": entry["task_layout_file"],
            }
            for key, expected_value in expected_positions.items():
                if positions.get(key) != expected_value:
                    raise TaskBankError(
                        f"positions/task layout mismatch for {positions_path}: {key}"
                    )
            acceptance_errors = validate_task_difficulty_acceptance(layout, acceptance)
            if acceptance_errors:
                raise TaskBankError(
                    f"persisted task difficulty rejection for {layout_path}: "
                    + "; ".join(acceptance_errors)
                )
            deadline_errors = validate_task_deadline_contract(
                layout, deadline_policy
            )
            if deadline_errors:
                raise TaskBankError(
                    f"persisted task deadline rejection for {layout_path}: "
                    + "; ".join(deadline_errors)
                )
            persisted_layouts[split_name].append(layout)
        actual_deadline_summary = summarize_task_deadlines(
            persisted_layouts[split_name]
        )
        if split.get("deadline_summary") != actual_deadline_summary:
            raise TaskBankError(
                f"deadline summary mismatch in split={split_name}"
            )
    _assert_split_independence(persisted_layouts, persisted_roles)
    return manifest


def load_task_bank_split(path: str | Path, split_name: str) -> Dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "task_bank_manifest.json"
    manifest = load_task_bank(manifest_path)
    split_name = str(split_name).strip()
    split = dict((manifest.get("splits") or {}).get(split_name) or {})
    if not split:
        raise TaskBankError(
            f"task-bank split={split_name!r} does not exist; "
            f"available={sorted((manifest.get('splits') or {}).keys())}"
        )
    layouts = []
    for entry in split.get("tasks") or []:
        layout_path = _resolve_bank_member(
            manifest_path.parent,
            entry.get("task_layout_file"),
            "task_layout_file",
        )
        with layout_path.open("r", encoding="utf-8") as handle:
            layouts.append(json.load(handle))
    return {
        "manifest_path": str(manifest_path),
        "bank_id": manifest["bank_id"],
        "bank_content_sha256": manifest["content_sha256"],
        "split_name": split_name,
        "role": split["role"],
        "runtime_config": dict(split.get("runtime_config") or {}),
        "layouts": layouts,
    }


__all__ = [
    "ALLOWED_SPLIT_ROLES",
    "PAPER_POSITIONS_SCHEMA_VERSION",
    "PAPER_TASK_BANK_SCHEMA_VERSION",
    "TaskBankError",
    "load_task_bank",
    "load_task_bank_split",
    "summarize_task_deadlines",
    "validate_dynamic_feasibility_audit",
    "write_task_bank",
]
