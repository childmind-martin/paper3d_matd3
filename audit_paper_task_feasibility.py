#!/usr/bin/env python3
"""Dynamic feasibility witness for immutable paper task-bank splits.

This is an oracle audit, not a comparison baseline: it resolves a conservative
surface-A* corridor from the same task geometry used to define difficulty and
deadline, under the same frozen quadrotor dynamics and terminal callbacks used
by training.  Its purpose is to reject task banks that are statically connected
but cannot be executed within their frozen per-task deadlines.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from multiagent.environment import MultiAgentEnv
from multiagent.scenarios.paper3d_terrain_energy import Scenario
from paper_curriculum import paper_physics_contract, paper_task_semantics_contract
from paper_experiment_schema import atomic_write_json
from paper_task_bank import load_task_bank_split
from paper_task_layout import reference_path_plan


AUDIT_SCHEMA_VERSION = "paper_dynamic_feasibility_audit_v11"
SURFACE_CLEARANCE_MARGIN_M = 4.0
INTER_AGENT_VERTICAL_SEPARATION_M = 3.0
WAYPOINT_CAPTURE_RADIUS_M = 1.5
TERRAIN_SAMPLING_SPACING_M = 0.5
DYNAMIC_OBSTACLE_INFLATION_M = 2.0


def _configure_quadrotor_physics(contract: Mapping[str, Any]) -> None:
    expected = paper_physics_contract()
    if dict(contract) != expected:
        raise RuntimeError(
            "task-bank physics contract does not match the frozen paper contract"
        )
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
    for field, environment_name in environment_names.items():
        os.environ[environment_name] = str(contract[field])
    os.environ["USE_QUADROTOR_DYNAMICS"] = (
        "1" if bool(contract["use_quadrotor_dynamics"]) else "0"
    )


def _configure_task_semantics(contract: Mapping[str, Any]) -> None:
    expected = paper_task_semantics_contract()
    if dict(contract) != expected:
        raise RuntimeError(
            "task-bank task semantics do not match the frozen paper contract"
        )
    os.environ["SUCCESS_DISTANCE_THRESHOLD"] = str(
        contract["success_distance_threshold"]
    )
    os.environ["COLLISION_DISTANCE_THRESHOLD"] = str(
        contract["collision_distance_threshold"]
    )


def _apply_and_validate_quadrotor_physics(
    scenario: Scenario,
    world,
    contract: Mapping[str, Any],
) -> None:
    world.dt = float(contract["simulation_dt"])
    world.gravity = float(contract["gravity"])
    world.control_accel_gain = float(contract["control_accel_gain"])
    world.damping = float(contract["damping"])
    world.action_range = [
        float(contract["action_range_x"]),
        float(contract["action_range_y"]),
        float(contract["action_range_z"]),
    ]
    world.z_action_bias = float(contract["z_action_bias"])
    world.quadrotor_attitude_response_time = float(
        contract["quadrotor_attitude_response_time"]
    )
    world.quadrotor_psi_cmd = float(contract["quadrotor_psi_cmd"])
    scenario.agent_size = float(contract["agent_size"])
    for agent in world.agents:
        agent.size = float(contract["agent_size"])
        agent.max_speed = float(contract["agent_max_speed"])
        agent.accel = float(contract["agent_accel"])
        agent.use_quadrotor_dynamics = bool(
            contract["use_quadrotor_dynamics"]
        )

    actual = {
        "simulation_dt": float(world.dt),
        "use_quadrotor_dynamics": all(
            bool(agent.use_quadrotor_dynamics) for agent in world.agents
        ),
        "agent_size": float(world.agents[0].size),
        "agent_max_speed": float(world.agents[0].max_speed),
        "agent_accel": float(world.agents[0].accel),
        "gravity": float(world.gravity),
        "control_accel_gain": float(world.control_accel_gain),
        "damping": float(world.damping),
        "action_range_x": float(world.action_range[0]),
        "action_range_y": float(world.action_range[1]),
        "action_range_z": float(world.action_range[2]),
        "z_action_bias": float(world.z_action_bias),
        "quadrotor_attitude_response_time": float(
            world.quadrotor_attitude_response_time
        ),
        "quadrotor_psi_cmd": float(world.quadrotor_psi_cmd),
    }
    expected_without_id = {
        key: value for key, value in contract.items() if key != "contract_id"
    }
    if actual != expected_without_id:
        raise RuntimeError(
            f"applied quadrotor physics mismatch: actual={actual}, "
            f"expected={expected_without_id}"
        )


def _turning_points(waypoints) -> np.ndarray:
    """Keep endpoints and direction changes without changing the A* polyline."""
    points = np.asarray(waypoints, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1:
        raise RuntimeError("reference planner returned invalid waypoints")
    if points.shape[0] <= 2:
        return points
    keep = [0]
    previous_direction = None
    for index in range(1, points.shape[0]):
        delta = points[index, :2] - points[index - 1, :2]
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-9:
            continue
        direction = delta / norm
        if previous_direction is not None and not np.allclose(
            direction, previous_direction, atol=1e-12, rtol=0.0
        ):
            keep.append(index - 1)
        previous_direction = direction
    keep.append(points.shape[0] - 1)
    return points[sorted(set(keep))]


def _segment_maximum_terrain_height(
    scenario: Scenario, start_xy: np.ndarray, end_xy: np.ndarray
) -> float:
    distance = float(np.linalg.norm(end_xy - start_xy))
    sample_count = max(
        2, int(math.ceil(distance / TERRAIN_SAMPLING_SPACING_M)) + 1
    )
    interpolation = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
    points = start_xy[None, :] + interpolation[:, None] * (
        end_xy - start_xy
    )[None, :]
    heights = np.asarray(
        scenario.batch_get_terrain_height(points), dtype=np.float64
    ).reshape(-1)
    if heights.size != sample_count or not np.isfinite(heights).all():
        raise RuntimeError("reference corridor terrain samples are invalid")
    return float(np.max(heights))


def _safe_reference_corridors(
    scenario: Scenario,
    layout: Mapping[str, Any],
    agents,
) -> List[np.ndarray]:
    contract = dict(layout.get("contract") or {})
    starts = list(layout.get("agent_starts") or [])
    goals = list(layout.get("agent_goals") or [])
    reference_obstacle_inflation = float(
        contract["reference_obstacle_inflation"]
    )
    controller_obstacle_inflation = max(
        reference_obstacle_inflation,
        DYNAMIC_OBSTACLE_INFLATION_M,
    )
    plans = [
        reference_path_plan(
            starts[index],
            goals[index],
            map_size=float(layout["map_size"]),
            terrain=np.asarray(scenario.terrain, dtype=np.float64),
            obstacles=list(layout.get("obstacles") or []),
            grid_resolution=float(contract["reference_grid_resolution"]),
            obstacle_inflation=controller_obstacle_inflation,
            boundary_clearance=float(contract["reference_boundary_clearance"]),
            surface_clearance=float(contract["reference_surface_clearance"]),
        )
        for index in range(len(agents))
    ]
    for agent_index, plan in enumerate(plans):
        if not plan.metrics.feasible or not plan.waypoints:
            raise RuntimeError(
                f"reference corridor missing for task={layout.get('task_id')} "
                f"agent={agent_index}"
            )
    path_points = [_turning_points(plan.waypoints) for plan in plans]
    path_surface_ceilings = []
    for points in path_points:
        segment_maxima = [
            _segment_maximum_terrain_height(
                scenario, points[index, :2], points[index + 1, :2]
            )
            for index in range(points.shape[0] - 1)
        ]
        path_surface_ceilings.append(
            max(segment_maxima) if segment_maxima else float(points[0, 2])
        )
    shared_surface_ceiling = float(max(path_surface_ceilings))
    maximum_agent_radius = max(
        float(getattr(agent, "size", 0.5)) for agent in agents
    )
    goal_points = [np.asarray(goal, dtype=np.float64) for goal in goals]
    minimum_safe_goal_center_distance = (
        2.0 * maximum_agent_radius
        + float(contract["collision_distance_threshold"])
    )
    for first in range(len(goal_points)):
        for second in range(first + 1, len(goal_points)):
            goal_distance = float(
                np.linalg.norm(
                    goal_points[first][:2] - goal_points[second][:2]
                )
            )
            if goal_distance <= minimum_safe_goal_center_distance:
                raise RuntimeError(
                    "parallel goal-settle columns are not collision-disjoint: "
                    f"task={layout.get('task_id')} agents=({first},{second}) "
                    f"distance={goal_distance} required>"
                    f"{minimum_safe_goal_center_distance}"
                )
    corridors = []
    for agent_index, points in enumerate(path_points):
        corridor = points.copy()
        corridor[:, 2] = (
            shared_surface_ceiling
            + maximum_agent_radius
            + SURFACE_CLEARANCE_MARGIN_M
            + INTER_AGENT_VERTICAL_SEPARATION_M * int(agent_index)
        )
        if not np.isfinite(corridor).all() or float(np.max(corridor[:, 2])) >= 145.0:
            raise RuntimeError(
                f"reference corridor violates the world z ceiling for "
                f"task={layout.get('task_id')} agent={agent_index}"
            )
        if not np.allclose(
            corridor[-1, :2], goal_points[agent_index][:2], atol=1e-12, rtol=0.0
        ):
            raise RuntimeError(
                "reference corridor does not end at the exact goal XY: "
                f"task={layout.get('task_id')} agent={agent_index}"
            )
        corridors.append(corridor)
    return corridors


def _pd_action(
    position,
    velocity,
    target,
    *,
    kp: float,
    kd: float,
    z_kp: float,
    z_kd: float,
) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    proportional = np.asarray([kp, kp, z_kp], dtype=np.float64)
    derivative = np.asarray([kd, kd, z_kd], dtype=np.float64)
    return np.clip(
        proportional * (target - position) - derivative * velocity,
        -1.0,
        1.0,
    )


def audit_split(
    manifest: str | Path,
    split: str,
    *,
    maximum_tasks: int | None = None,
    kp: float = 0.12,
    kd: float = 0.35,
    z_kp: float = 0.20,
    z_kd: float = 0.20,
) -> Dict[str, Any]:
    bank = load_task_bank_split(manifest, split)
    layouts = list(bank["layouts"])
    if maximum_tasks is not None:
        layouts = layouts[: max(0, int(maximum_tasks))]
    if not layouts:
        raise RuntimeError("dynamic feasibility audit requires at least one task")

    runtime_config = dict(bank.get("runtime_config") or {})
    physics_contract = dict(runtime_config.get("paper_physics_contract") or {})
    task_semantics = dict(
        runtime_config.get("paper_task_semantics_contract") or {}
    )
    _configure_quadrotor_physics(physics_contract)
    _configure_task_semantics(task_semantics)

    scenario = Scenario(
        paper_task_bank_manifest=str(bank["manifest_path"]),
        paper_task_bank_split=str(split),
        paper_task_bank_parallel_envs=1,
    )
    world = scenario.make_world()
    world.env_id = 0
    _apply_and_validate_quadrotor_physics(
        scenario, world, physics_contract
    )
    actual_semantics = {
        "contract_id": task_semantics.get("contract_id"),
        "success_distance_threshold": float(
            scenario.success_distance_threshold
        ),
        "collision_distance_threshold": float(
            scenario.collision_distance_threshold
        ),
        "reference_surface_clearance": float(
            scenario.paper_task_reference_surface_clearance
        ),
    }
    if actual_semantics != task_semantics:
        raise RuntimeError(
            f"applied task semantics mismatch: actual={actual_semantics}, "
            f"expected={task_semantics}"
        )
    env = MultiAgentEnv(
        world,
        scenario.reset_world,
        scenario.reward,
        scenario.observation,
        done_callback=scenario.is_done,
        info_callback=None,
        shared_viewer=False,
    )
    env._disable_trajectory_recording = True

    results: List[Dict[str, Any]] = []
    for episode_index, expected_layout in enumerate(layouts):
        world.episode_index = int(episode_index)
        env.reset()
        layout = world.current_task_layout
        if not isinstance(layout, dict):
            raise RuntimeError("environment reset did not expose current_task_layout")
        if layout["task_id"] != expected_layout["task_id"]:
            raise RuntimeError(
                f"task order mismatch: expected={expected_layout['task_id']}, "
                f"actual={layout['task_id']}"
            )
        deadline_steps = int(layout["contract"]["deadline"]["deadline_steps"])
        if int(world.episode_length) != deadline_steps:
            raise RuntimeError("world/task deadline mismatch during feasibility audit")
        reference_corridors = _safe_reference_corridors(
            scenario, layout, world.agents
        )
        waypoint_indices = [0] * len(world.agents)
        path_completed = [False] * len(world.agents)
        corridor_completion_steps: List[int | None] = [
            None
        ] * len(world.agents)
        goal_success_steps: List[int | None] = [None] * len(world.agents)
        goal_targeting_start_step: int | None = None
        done_flags = [False] * len(world.agents)
        step_count = 0

        for _ in range(deadline_steps):
            positions = []
            velocities = []
            goals = []
            for agent_index, agent in enumerate(world.agents):
                position = np.asarray(agent.state.p_pos, dtype=np.float64)
                velocity = np.asarray(agent.state.p_vel, dtype=np.float64)
                goal = np.asarray(agent.goal_a.state.p_pos, dtype=np.float64)
                positions.append(position)
                velocities.append(velocity)
                goals.append(goal)
                corridor = reference_corridors[agent_index]
                waypoint_index = waypoint_indices[agent_index]
                while waypoint_index < corridor.shape[0] and float(
                    np.linalg.norm(position - corridor[waypoint_index])
                ) <= WAYPOINT_CAPTURE_RADIUS_M:
                    waypoint_index += 1
                waypoint_indices[agent_index] = waypoint_index
                if waypoint_index >= corridor.shape[0]:
                    path_completed[agent_index] = True
                    if corridor_completion_steps[agent_index] is None:
                        corridor_completion_steps[agent_index] = int(step_count)
            all_corridors_complete = all(path_completed)
            if (
                all_corridors_complete
                and goal_targeting_start_step is None
            ):
                goal_targeting_start_step = int(step_count)
            actions = []
            for agent_index in range(len(world.agents)):
                corridor = reference_corridors[agent_index]
                waypoint_index = waypoint_indices[agent_index]
                if not all_corridors_complete:
                    target = (
                        corridor[-1]
                        if path_completed[agent_index]
                        else corridor[waypoint_index]
                    )
                else:
                    target = goals[agent_index]
                actions.append(
                    _pd_action(
                        positions[agent_index],
                        velocities[agent_index],
                        target,
                        kp=kp,
                        kd=kd,
                        z_kp=z_kp,
                        z_kd=z_kd,
                    )
                )
            _, _, done_flags, _ = env.step(actions)
            step_count += 1
            current_success_flags = list(
                getattr(world, "_episode_agent_success_flags", []) or []
            )
            if len(current_success_flags) == len(world.agents):
                for agent_index, succeeded in enumerate(
                    current_success_flags
                ):
                    if succeeded and goal_success_steps[agent_index] is None:
                        goal_success_steps[agent_index] = int(step_count)
            if all(bool(value) for value in done_flags):
                break

        success_flags = list(
            getattr(world, "_episode_agent_success_flags", []) or []
        )
        team_success = bool(getattr(world, "_episode_team_success_flag", 0))
        done_reason = str(getattr(world, "_episode_done_reason", "") or "")
        final_distances = [
            float(np.linalg.norm(agent.state.p_pos - agent.goal_a.state.p_pos))
            for agent in world.agents
        ]
        final_positions = [
            [float(value) for value in np.asarray(agent.state.p_pos).tolist()]
            for agent in world.agents
        ]
        final_clearances = [
            scenario._paper_agent_minimum_clearance(agent, world)
            for agent in world.agents
        ]
        quadrotor_state_initialized = [
            bool(
                getattr(agent, "use_quadrotor_dynamics", False)
                and hasattr(agent, "quadrotor_dynamics")
                and hasattr(agent.state, "orientation")
                and hasattr(agent.state, "angular_vel")
            )
            for agent in world.agents
        ]
        if quadrotor_state_initialized != [True] * len(world.agents):
            raise RuntimeError(
                f"quadrotor dynamics witness missing for task={layout['task_id']}"
            )
        waypoint_capture_completed = [bool(value) for value in path_completed]
        path_witness_completed = [
            bool(waypoint_capture_completed[index])
            or bool(success_flags[index])
            for index in range(len(world.agents))
        ]
        for agent_index, succeeded in enumerate(success_flags):
            if succeeded and goal_success_steps[agent_index] is None:
                goal_success_steps[agent_index] = int(step_count)
        starts = [
            np.asarray(value, dtype=np.float64)
            for value in layout["agent_starts"]
        ]
        goals_array = [
            np.asarray(value, dtype=np.float64)
            for value in layout["agent_goals"]
        ]
        witness_route_lengths = []
        for agent_index, corridor in enumerate(reference_corridors):
            corridor_length = float(
                np.sum(np.linalg.norm(np.diff(corridor, axis=0), axis=1))
            )
            witness_route_lengths.append(
                float(
                    np.linalg.norm(starts[agent_index] - corridor[0])
                    + corridor_length
                    + np.linalg.norm(corridor[-1] - goals_array[agent_index])
                )
            )
        results.append(
            {
                "episode_index": int(episode_index),
                "task_id": layout["task_id"],
                "difficulty_bin": layout.get("difficulty_bin"),
                "deadline_steps": deadline_steps,
                "executed_steps": int(step_count),
                "deadline_usage_fraction": float(step_count / deadline_steps),
                "reference_path_lengths_m": [
                    float(path["path_length"])
                    for path in layout["metrics"]["reference_paths"]
                ],
                "controller_witness_route_lengths_m": witness_route_lengths,
                "controller_obstacle_inflation_m": float(
                    max(
                        float(
                            layout["contract"][
                                "reference_obstacle_inflation"
                            ]
                        ),
                        DYNAMIC_OBSTACLE_INFLATION_M,
                    )
                ),
                "reference_waypoint_counts": [
                    int(corridor.shape[0]) for corridor in reference_corridors
                ],
                "maximum_corridor_altitudes_m": [
                    float(np.max(corridor[:, 2]))
                    for corridor in reference_corridors
                ],
                "terminal_reason": done_reason,
                "team_success": team_success,
                "agent_success_flags": [int(value) for value in success_flags],
                "final_goal_distances_m": final_distances,
                "final_positions_m": final_positions,
                "episode_minimum_clearance_m": float(
                    getattr(world, "_episode_dmin_min", float("nan"))
                ),
                "final_minimum_clearances": [
                    {
                        "clearance_m": float(clearance),
                        "source": str(source),
                    }
                    for clearance, source in final_clearances
                ],
                "controller_waypoint_capture_completed": (
                    waypoint_capture_completed
                ),
                "controller_corridor_completion_steps": (
                    corridor_completion_steps
                ),
                "controller_path_completed": path_witness_completed,
                "controller_goal_success_observed": [
                    bool(value) for value in success_flags
                ],
                "controller_goal_success_steps": goal_success_steps,
                "controller_goal_targeting_start_step": (
                    goal_targeting_start_step
                ),
                "all_done": bool(all(bool(value) for value in done_flags)),
                "dynamics_mode": "quadrotor",
                "quadrotor_state_initialized": quadrotor_state_initialized,
            }
        )

    successes = sum(int(result["team_success"]) for result in results)
    payload: Dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_role": "oracle_dynamic_feasibility_witness_not_comparison_baseline",
        "task_bank_manifest": str(Path(bank["manifest_path"]).resolve()),
        "task_bank_content_sha256": bank["bank_content_sha256"],
        "split": str(split),
        "split_role": bank["role"],
        "controller": {
            "policy": (
                "surface_astar_shared_altitude_layers_sync_then_parallel_"
                "vertical_goal_settle"
            ),
            "terrain_and_obstacle_geometry_visibility": "oracle",
            "kp": float(kp),
            "kd": float(kd),
            "z_kp": float(z_kp),
            "z_kd": float(z_kd),
            "surface_clearance_margin_m": SURFACE_CLEARANCE_MARGIN_M,
            "inter_agent_cruise_vertical_separation_m": (
                INTER_AGENT_VERTICAL_SEPARATION_M
            ),
            "waypoint_capture_radius_m": WAYPOINT_CAPTURE_RADIUS_M,
            "obstacle_inflation_policy": (
                "max_task_reference_and_dynamic_tracking_margin"
            ),
            "dynamic_obstacle_inflation_m": (
                DYNAMIC_OBSTACLE_INFLATION_M
            ),
            "terrain_sampling_spacing_m": TERRAIN_SAMPLING_SPACING_M,
            "goal_settle_policy": (
                "all_corridors_then_parallel_vertical_at_exact_goal_xy"
            ),
            "goal_success_detection_policy": (
                "environment_terminal_radius_independent_of_controller_phase"
            ),
            "parallel_goal_settle_requires_disjoint_goal_columns": True,
            "dynamics_mode": "quadrotor",
            "physics_contract": physics_contract,
            "task_semantics": task_semantics,
        },
        "tasks_audited": len(results),
        "team_successes": successes,
        "all_tasks_dynamically_feasible": successes == len(results),
        "episodes": results,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-bank-manifest", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--maximum-tasks", type=int, default=None)
    parser.add_argument("--kp", type=float, default=0.12)
    parser.add_argument("--kd", type=float, default=0.35)
    parser.add_argument("--z-kp", type=float, default=0.20)
    parser.add_argument("--z-kd", type=float, default=0.20)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if not math.isfinite(args.kp) or args.kp <= 0.0:
        raise ValueError("--kp must be finite and positive")
    if not math.isfinite(args.kd) or args.kd < 0.0:
        raise ValueError("--kd must be finite and non-negative")
    if not math.isfinite(args.z_kp) or args.z_kp <= 0.0:
        raise ValueError("--z-kp must be finite and positive")
    if not math.isfinite(args.z_kd) or args.z_kd < 0.0:
        raise ValueError("--z-kd must be finite and non-negative")
    payload = audit_split(
        args.task_bank_manifest,
        args.split,
        maximum_tasks=args.maximum_tasks,
        kp=args.kp,
        kd=args.kd,
        z_kp=args.z_kp,
        z_kd=args.z_kd,
    )
    if args.output:
        atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["all_tasks_dynamically_feasible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
