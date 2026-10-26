"""Paper curriculum definitions and statistically explicit promotion gates."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping

from paper_experiment_schema import PaperSchemaError, wilson_interval


PAPER_CURRICULUM_SCHEMA_VERSION = "paper_curriculum_v3"
PAPER_REFERENCE_SURFACE_CLEARANCE_M = 4.5

_PAPER_TASK_SEMANTICS_CONTRACT = {
    "contract_id": "paper_task_semantics_v1",
    "success_distance_threshold": 2.0,
    "collision_distance_threshold": 0.5,
    "reference_surface_clearance": PAPER_REFERENCE_SURFACE_CLEARANCE_M,
}


def paper_task_semantics_contract() -> Dict[str, Any]:
    return deepcopy(_PAPER_TASK_SEMANTICS_CONTRACT)


_PAPER_PHYSICS_CONTRACT = {
    "contract_id": "quadrotor_runtime_v1",
    "simulation_dt": 0.08,
    "use_quadrotor_dynamics": True,
    "agent_size": 0.5,
    "agent_max_speed": 42.5,
    "agent_accel": 4.6,
    "gravity": 0.0,
    "control_accel_gain": 1.0,
    "damping": 0.12,
    "action_range_x": 2.5,
    "action_range_y": 2.5,
    "action_range_z": 2.2,
    "z_action_bias": 0.0,
    "quadrotor_attitude_response_time": 0.05,
    "quadrotor_psi_cmd": 0.0,
}


def paper_physics_contract() -> Dict[str, Any]:
    return deepcopy(_PAPER_PHYSICS_CONTRACT)


_DIFFICULTY_ACCEPTANCE = {
    "C0": {
        "summary_bounds": {
            "obstacle_count": {"minimum": 0, "maximum": 0},
            "maximum_reference_path_length_m": {"maximum": 70.0},
            "maximum_detour_ratio": {"maximum": 1.50},
            "maximum_cumulative_climb_m": {"maximum": 12.0},
            "minimum_boundary_clearance_m": {"minimum": 3.5},
        }
    },
    "C1": {
        "summary_bounds": {
            "obstacle_count": {"minimum": 4, "maximum": 4},
            "maximum_reference_path_length_m": {"maximum": 105.0},
            "maximum_detour_ratio": {"minimum": 1.10, "maximum": 1.65},
            "maximum_cumulative_climb_m": {"maximum": 18.0},
            "minimum_boundary_clearance_m": {"minimum": 3.5},
            "minimum_obstacle_clearance_m": {"minimum": 1.0, "maximum": 35.0},
        }
    },
    "C2": {
        "summary_bounds": {
            "obstacle_count": {"minimum": 8, "maximum": 8},
            "maximum_reference_path_length_m": {"maximum": 155.0},
            "maximum_detour_ratio": {"minimum": 1.20, "maximum": 1.90},
            "maximum_cumulative_climb_m": {"maximum": 30.0},
            "minimum_boundary_clearance_m": {"minimum": 3.5},
            "minimum_obstacle_clearance_m": {"minimum": 1.0, "maximum": 8.0},
        }
    },
    "C3": {
        "summary_bounds": {
            "obstacle_count": {"minimum": 8, "maximum": 8},
            "maximum_reference_path_length_m": {"maximum": 200.0},
            "maximum_detour_ratio": {"minimum": 1.25, "maximum": 1.90},
            "maximum_cumulative_climb_m": {"minimum": 30.0, "maximum": 85.0},
            "minimum_boundary_clearance_m": {"minimum": 3.5},
            "minimum_obstacle_clearance_m": {"minimum": 1.0, "maximum": 8.0},
        }
    },
}

_TASK_DEADLINE_POLICY = {
    "policy_id": "reference_path_slack_v1",
    "simulation_dt_seconds": 0.08,
    "nominal_speed_mps": 1.25,
    "path_time_slack": 1.25,
    "fixed_overhead_seconds": 30.0,
    "minimum_seconds": 90.0,
    "maximum_seconds": 224.0,
}

_VALIDATION_CHECKPOINT_GATES = {
    "C0": {
        "required_tasks": 64,
        "minimum_successes": 54,
        "minimum_observed_success_rate": 54.0 / 64.0,
        "wilson_lower_bound": 0.72,
    },
    "C1": {
        "required_tasks": 64,
        "minimum_successes": 51,
        "minimum_observed_success_rate": 51.0 / 64.0,
        "wilson_lower_bound": 0.67,
    },
    "C2": {
        "required_tasks": 64,
        "minimum_successes": 48,
        "minimum_observed_success_rate": 48.0 / 64.0,
        "wilson_lower_bound": 0.62,
    },
    "C3": {
        "required_tasks": 64,
        "minimum_successes": 44,
        "minimum_observed_success_rate": 44.0 / 64.0,
        "wilson_lower_bound": 0.55,
    },
}


def validation_checkpoint_gate(difficulty_bin: str) -> Dict[str, Any]:
    difficulty = str(difficulty_bin).strip()
    gate = _VALIDATION_CHECKPOINT_GATES.get(difficulty)
    if gate is None:
        raise PaperSchemaError(
            f"no validation checkpoint gate for difficulty={difficulty!r}"
        )
    return {
        "protocol_id": "validation_checkpoint_adequacy_v1",
        "difficulty_bin": difficulty,
        **deepcopy(gate),
    }


def build_paper_curriculum(base_terrain_seed: int) -> List[Dict[str, Any]]:
    base = int(base_terrain_seed)
    common = {
        "schema_version": PAPER_CURRICULUM_SCHEMA_VERSION,
        "paper_task_contract_enabled": True,
        "paper_task_contract_strict": True,
        "paper_task_require_reference_feasible": True,
        "paper_task_reference_grid_resolution": 4.0,
        "paper_task_obstacle_inflation": 1.5,
        "paper_task_reference_surface_clearance": (
            PAPER_REFERENCE_SURFACE_CLEARANCE_M
        ),
        "paper_task_semantics_contract": paper_task_semantics_contract(),
        "success_distance_threshold": _PAPER_TASK_SEMANTICS_CONTRACT[
            "success_distance_threshold"
        ],
        "collision_distance_threshold": _PAPER_TASK_SEMANTICS_CONTRACT[
            "collision_distance_threshold"
        ],
        "hard_deadline_terminal": True,
        "paper_terminal_on_collision": True,
        "paper_reward_profile": "paper_v1",
        "paper_progress_reward_scale": 20.0,
        "paper_individual_reach_bonus": 10.0,
        "paper_team_success_bonus": 100.0,
        "paper_collision_penalty": 100.0,
        "paper_out_of_bounds_penalty": 100.0,
        "paper_deadline_penalty": 50.0,
        "paper_step_penalty": 0.01,
        "paper_control_effort_penalty": 0.005,
        "paper_clearance_margin": 2.0,
        "paper_clearance_penalty": 0.2,
        "include_time_to_go_observation": True,
        "use_fixed_positions": False,
        "dynamic_first_time": False,
        "random_terrain": False,
        "per_env_terrain": True,
        "min_agent_start_separation": 8.0,
        "paper_goal_formation_radius": 10.0,
        "paper_goal_boundary_clearance": 5.0,
        "paper_task_deadline_policy": deepcopy(_TASK_DEADLINE_POLICY),
        "paper_physics_contract": paper_physics_contract(),
    }
    stages = [
        {
            **common,
            "stage_id": 1,
            "difficulty_bin": "C0",
            "label": "C0 short-range navigation, fixed simple terrain, no obstacles",
            "reward_profile": "paper_c0",
            "min_start_goal_dist": 30.0,
            "max_start_goal_dist": 50.0,
            "terrain_complexity_level": 1,
            "paper_task_num_obstacles": 0,
            "use_dynamic_obstacles": False,
            "semi_random_terrain": False,
            "per_episode_terrain": False,
            "terrain_seed": base,
            "terrain_base_seed": base,
            "multi_terrain_mode": False,
            "multi_terrain_seeds": [],
            "candidate_checkpoint_interval_trajectories": 128,
            "promotion": {"window": 40, "minimum_episodes": 40, "minimum_successes": 32, "wilson_lower_bound": 0.65},
            "difficulty_acceptance": deepcopy(_DIFFICULTY_ACCEPTANCE["C0"]),
        },
        {
            **common,
            "stage_id": 2,
            "difficulty_bin": "C1",
            "label": "C1 medium-range navigation, fixed simple terrain, reset-randomized sparse obstacle layouts",
            "reward_profile": "paper_c1",
            "min_start_goal_dist": 45.0,
            "max_start_goal_dist": 70.0,
            "terrain_complexity_level": 1,
            "paper_task_num_obstacles": 4,
            "use_dynamic_obstacles": True,
            "semi_random_terrain": False,
            "per_episode_terrain": False,
            "terrain_seed": base,
            "terrain_base_seed": base,
            "multi_terrain_mode": False,
            "multi_terrain_seeds": [],
            "candidate_checkpoint_interval_trajectories": 192,
            "promotion": {"window": 50, "minimum_episodes": 50, "minimum_successes": 35, "wilson_lower_bound": 0.55},
            "difficulty_acceptance": deepcopy(_DIFFICULTY_ACCEPTANCE["C1"]),
        },
        {
            **common,
            "stage_id": 3,
            "difficulty_bin": "C2",
            "label": "C2 long-range navigation, fixed medium terrain, randomized obstacle layouts",
            "reward_profile": "paper_c2",
            "min_start_goal_dist": 60.0,
            "max_start_goal_dist": 90.0,
            "terrain_complexity_level": 2,
            "paper_task_num_obstacles": 8,
            "use_dynamic_obstacles": True,
            "semi_random_terrain": False,
            "per_episode_terrain": False,
            "terrain_seed": base,
            "terrain_base_seed": base,
            "multi_terrain_mode": False,
            "multi_terrain_seeds": [],
            "candidate_checkpoint_interval_trajectories": 320,
            "promotion": {"window": 60, "minimum_episodes": 60, "minimum_successes": 39, "wilson_lower_bound": 0.50},
            "difficulty_acceptance": deepcopy(_DIFFICULTY_ACCEPTANCE["C2"]),
        },
        {
            **common,
            "stage_id": 4,
            "difficulty_bin": "C3",
            "label": "C3 long-range navigation, semi-random medium terrain and obstacle layouts",
            "reward_profile": "paper_c3",
            "min_start_goal_dist": 75.0,
            "max_start_goal_dist": 120.0,
            "terrain_complexity_level": 2,
            "paper_task_num_obstacles": 8,
            "use_dynamic_obstacles": True,
            "semi_random_terrain": True,
            "per_episode_terrain": True,
            "terrain_seed": base,
            "terrain_base_seed": base,
            "multi_terrain_mode": False,
            "multi_terrain_seeds": [],
            "candidate_checkpoint_interval_trajectories": 640,
            "promotion": {"window": 80, "minimum_episodes": 80, "minimum_successes": 48, "wilson_lower_bound": 0.45},
            "difficulty_acceptance": deepcopy(_DIFFICULTY_ACCEPTANCE["C3"]),
        },
    ]
    validate_paper_curriculum(stages)
    return deepcopy(stages)


def build_paper_generalization_suite(
    held_out_terrain_seeds: Iterable[int],
) -> List[Dict[str, Any]]:
    """Build evaluation-only tasks; these seeds must never enter training."""
    seeds: List[int] = []
    for value in held_out_terrain_seeds:
        value = int(value)
        if value not in seeds:
            seeds.append(value)
    if not seeds:
        raise PaperSchemaError("held-out generalization suite requires terrain seeds")
    suites = []
    for index, seed in enumerate(seeds):
        suites.append(
            {
                "schema_version": PAPER_CURRICULUM_SCHEMA_VERSION,
                "suite_id": f"G1_terrain_seed_{seed}",
                "difficulty_bin": "G1",
                "split": "test_generalization",
                "evaluation_only": True,
                "terrain_seed": seed,
                "terrain_base_seed": seed,
                "terrain_complexity_level": 2,
                "min_start_goal_dist": 75.0,
                "max_start_goal_dist": 120.0,
                "paper_task_num_obstacles": 8,
                "use_dynamic_obstacles": True,
                "use_fixed_positions": False,
                "semi_random_terrain": False,
                # One held-out terrain identity per G1 split.  Positions and
                # obstacle layouts vary across the split, while the terrain
                # seed remains the independent generalization-domain unit.
                "per_episode_terrain": False,
                "paper_task_contract_enabled": True,
                "paper_physics_contract": paper_physics_contract(),
                "paper_task_contract_strict": True,
                "paper_task_require_reference_feasible": True,
                "paper_task_reference_grid_resolution": 4.0,
                "paper_task_obstacle_inflation": 1.5,
                "paper_task_reference_surface_clearance": (
                    PAPER_REFERENCE_SURFACE_CLEARANCE_M
                ),
                "paper_task_semantics_contract": paper_task_semantics_contract(),
                "success_distance_threshold": _PAPER_TASK_SEMANTICS_CONTRACT[
                    "success_distance_threshold"
                ],
                "collision_distance_threshold": _PAPER_TASK_SEMANTICS_CONTRACT[
                    "collision_distance_threshold"
                ],
                "hard_deadline_terminal": True,
                "paper_terminal_on_collision": True,
                "paper_reward_profile": "paper_v1",
                "paper_progress_reward_scale": 20.0,
                "paper_individual_reach_bonus": 10.0,
                "paper_team_success_bonus": 100.0,
                "paper_collision_penalty": 100.0,
                "paper_out_of_bounds_penalty": 100.0,
                "paper_deadline_penalty": 50.0,
                "paper_step_penalty": 0.01,
                "paper_control_effort_penalty": 0.005,
                "paper_clearance_margin": 2.0,
                "paper_clearance_penalty": 0.2,
                "include_time_to_go_observation": True,
                "paper_goal_formation_radius": 10.0,
                "paper_goal_boundary_clearance": 5.0,
                "difficulty_acceptance": deepcopy(_DIFFICULTY_ACCEPTANCE["C3"]),
                "paper_task_deadline_policy": deepcopy(_TASK_DEADLINE_POLICY),
                "suite_index": index,
            }
        )
    return suites


def validate_paper_curriculum(stages: Iterable[Mapping[str, Any]]) -> None:
    stages = list(stages)
    if not stages:
        raise PaperSchemaError("paper curriculum cannot be empty")
    expected_stage = 1
    previous_max = None
    seen_bins = set()
    for stage in stages:
        stage_id = int(stage.get("stage_id", -1))
        if stage_id != expected_stage:
            raise PaperSchemaError(
                f"paper curriculum stage sequence error: expected={expected_stage}, got={stage_id}"
            )
        expected_stage += 1
        difficulty = str(stage.get("difficulty_bin", "")).strip()
        if not difficulty or difficulty in seen_bins:
            raise PaperSchemaError(f"invalid or duplicate difficulty_bin={difficulty!r}")
        seen_bins.add(difficulty)
        min_distance = float(stage.get("min_start_goal_dist"))
        max_distance = float(stage.get("max_start_goal_dist"))
        if not 0.0 <= min_distance < max_distance:
            raise PaperSchemaError(
                f"invalid distance band for {difficulty}: [{min_distance}, {max_distance}]"
            )
        if previous_max is not None and max_distance < previous_max:
            raise PaperSchemaError(
                f"curriculum max distance decreased at {difficulty}: {max_distance} < {previous_max}"
            )
        previous_max = max_distance
        obstacle_count = int(stage.get("paper_task_num_obstacles", -1))
        if obstacle_count < 0:
            raise PaperSchemaError(f"paper stage {difficulty} requires explicit obstacle count")
        if int(stage.get("candidate_checkpoint_interval_trajectories", 0)) <= 0:
            raise PaperSchemaError(
                f"paper stage {difficulty} requires a positive candidate interval"
            )
        acceptance = stage.get("difficulty_acceptance")
        if not isinstance(acceptance, Mapping) or not isinstance(
            acceptance.get("summary_bounds"), Mapping
        ):
            raise PaperSchemaError(
                f"paper stage {difficulty} requires difficulty_acceptance.summary_bounds"
            )
        if dict(stage.get("paper_task_deadline_policy") or {}) != _TASK_DEADLINE_POLICY:
            raise PaperSchemaError(
                f"paper stage {difficulty} deadline policy is not the frozen protocol"
            )
        if float(stage.get("paper_task_reference_grid_resolution", 0.0)) != 4.0:
            raise PaperSchemaError(
                f"paper stage {difficulty} requires reference grid resolution=4.0"
            )
        if float(stage.get("paper_task_obstacle_inflation", 0.0)) != 1.5:
            raise PaperSchemaError(
                f"paper stage {difficulty} requires obstacle inflation=1.5"
            )
        if float(
            stage.get("paper_task_reference_surface_clearance", 0.0)
        ) != PAPER_REFERENCE_SURFACE_CLEARANCE_M:
            raise PaperSchemaError(
                f"paper stage {difficulty} requires reference surface "
                f"clearance={PAPER_REFERENCE_SURFACE_CLEARANCE_M}"
            )
        for field in (
            "success_distance_threshold",
            "collision_distance_threshold",
        ):
            if float(stage.get(field, -1.0)) != float(
                _PAPER_TASK_SEMANTICS_CONTRACT[field]
            ):
                raise PaperSchemaError(
                    f"paper stage {difficulty} requires {field}="
                    f"{_PAPER_TASK_SEMANTICS_CONTRACT[field]}"
                )
        if dict(stage.get("paper_task_semantics_contract") or {}) != (
            _PAPER_TASK_SEMANTICS_CONTRACT
        ):
            raise PaperSchemaError(
                f"paper stage {difficulty} task semantics contract drift"
            )
        for required_true in (
            "paper_task_contract_enabled",
            "paper_task_contract_strict",
            "paper_task_require_reference_feasible",
            "hard_deadline_terminal",
            "paper_terminal_on_collision",
            "include_time_to_go_observation",
        ):
            if not bool(stage.get(required_true, False)):
                raise PaperSchemaError(f"paper stage {difficulty} requires {required_true}=true")
        if str(stage.get("paper_reward_profile", "")).strip() != "paper_v1":
            raise PaperSchemaError(
                f"paper stage {difficulty} requires paper_reward_profile='paper_v1'"
            )


def promotion_decision(success_flags: Iterable[bool], promotion: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(promotion, Mapping):
        raise PaperSchemaError("promotion contract must be an object")
    window = int(promotion.get("window", 0))
    minimum_episodes = int(promotion.get("minimum_episodes", 0))
    minimum_successes = int(promotion.get("minimum_successes", 0))
    required_lower = float(promotion.get("wilson_lower_bound", 0.0))
    if window <= 0 or minimum_episodes <= 0 or minimum_successes < 0:
        raise PaperSchemaError(f"invalid promotion contract={dict(promotion)!r}")
    flags = [bool(flag) for flag in success_flags]
    selected = flags[-window:]
    trials = len(selected)
    successes = sum(selected)
    if trials == 0:
        interval = None
        lower = 0.0
    else:
        interval = wilson_interval(successes, trials)
        lower = interval["lower"]
    promoted = bool(
        trials >= minimum_episodes
        and successes >= minimum_successes
        and lower >= required_lower
    )
    return {
        "promoted": promoted,
        "window": window,
        "trials": trials,
        "successes": successes,
        "wilson_interval": interval,
        "requirements": {
            "minimum_episodes": minimum_episodes,
            "minimum_successes": minimum_successes,
            "wilson_lower_bound": required_lower,
        },
    }
