import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from paper_task_bank import (
    TaskBankError,
    load_task_bank,
    load_task_bank_split,
    validate_dynamic_feasibility_audit,
    write_task_bank,
)
from paper_task_layout import build_task_layout, canonical_sha256
from paper_curriculum import paper_physics_contract, paper_task_semantics_contract


_DEADLINE_POLICY = {
    "policy_id": "reference_path_slack_v1",
    "simulation_dt_seconds": 0.08,
    "nominal_speed_mps": 1.25,
    "path_time_slack": 1.25,
    "fixed_overhead_seconds": 20.0,
    "minimum_seconds": 90.0,
    "maximum_seconds": 224.0,
}


def _layout(offset, *, position_seed, obstacle_seed, terrain_seed=67, difficulty="C0"):
    starts = np.asarray([[10 + offset, 10, 7], [10 + offset, 20, 7], [10 + offset, 30, 7]], dtype=float)
    goals = np.asarray([[45 + offset, 10, 12], [45 + offset, 20, 12], [45 + offset, 30, 12]], dtype=float)
    return build_task_layout(
        agent_starts=starts,
        agent_goals=goals,
        central_goal=[45 + offset, 20, 12],
        map_size=200,
        min_start_goal_dist=30,
        max_start_goal_dist=50,
        terrain=np.zeros((20, 20), dtype=np.float32),
        obstacles=[],
        terrain_seed=terrain_seed,
        position_seed=position_seed,
        obstacle_seed=obstacle_seed,
        difficulty_bin=difficulty,
        deadline_policy=_DEADLINE_POLICY,
        simulation_dt_seconds=0.08,
    )


def _runtime_config(*split_names):
    acceptance = {
        "summary_bounds": {
            "obstacle_count": {"minimum": 0, "maximum": 0},
            "minimum_boundary_clearance_m": {"minimum": 0.0},
        }
    }
    return {
        split_name: {
            "terrain_complexity_level": 1,
            "difficulty_bin": "C0",
            "paper_task_num_obstacles": 0,
            "min_start_goal_dist": 30.0,
            "max_start_goal_dist": 50.0,
            "paper_task_require_reference_feasible": True,
            "paper_task_reference_grid_resolution": 4.0,
            "paper_task_obstacle_inflation": 1.5,
            "paper_task_reference_surface_clearance": 4.5,
            "success_distance_threshold": 2.0,
            "collision_distance_threshold": 0.5,
            "paper_task_semantics_contract": paper_task_semantics_contract(),
            "paper_goal_formation_radius": None,
            "paper_goal_boundary_clearance": None,
            "difficulty_acceptance": acceptance,
            "paper_task_deadline_policy": _DEADLINE_POLICY,
            "paper_physics_contract": paper_physics_contract(),
        }
        for split_name in split_names
    }


class PaperTaskBankTests(unittest.TestCase):
    def _attach_passing_dynamic_audit(self, manifest):
        split_name = next(iter(manifest["splits"]))
        split = manifest["splits"][split_name]
        entry = split["tasks"][0]
        deadline_steps = entry["deadline"]["deadline_steps"]
        manifest["metadata"]["dynamic_feasibility_audit"] = {
            "schema_version": "paper_dynamic_feasibility_audit_v11",
            "audit_role": "oracle_dynamic_feasibility_witness_not_comparison_baseline",
            "performed": True,
            "all_tasks_dynamically_feasible": True,
            "controller": {
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
            },
            "staging_task_bank_content_sha256": "a" * 64,
            "splits": {
                split_name: {
                    "split_role": split["role"],
                    "tasks_audited": 1,
                    "team_successes": 1,
                    "all_tasks_dynamically_feasible": True,
                    "episodes": [
                        {
                            "task_id": entry["task_id"],
                            "difficulty_bin": entry["difficulty_bin"],
                            "deadline_steps": deadline_steps,
                            "executed_steps": deadline_steps - 1,
                            "deadline_usage_fraction": (
                                (deadline_steps - 1) / deadline_steps
                            ),
                            "reference_path_lengths_m": [
                                entry["deadline"][
                                    "maximum_reference_path_length_m"
                                ]
                            ] * 3,
                            "controller_witness_route_lengths_m": [
                                entry["deadline"][
                                    "maximum_reference_path_length_m"
                                ] + 10.0
                            ] * 3,
                            "controller_obstacle_inflation_m": 2.0,
                            "episode_minimum_clearance_m": 1.0,
                            "terminal_reason": "team_success",
                            "team_success": True,
                            "agent_success_flags": [1, 1, 1],
                            "final_goal_distances_m": [1.0, 1.0, 1.0],
                            "all_done": True,
                            "dynamics_mode": "quadrotor",
                            "quadrotor_state_initialized": [True, True, True],
                            "controller_waypoint_capture_completed": [
                                True, True, True
                            ],
                            "controller_corridor_completion_steps": [
                                10, 11, 12
                            ],
                            "controller_path_completed": [True, True, True],
                            "controller_goal_success_observed": [
                                True, True, True
                            ],
                            "controller_goal_success_steps": [
                                deadline_steps - 3,
                                deadline_steps - 2,
                                deadline_steps - 1,
                            ],
                            "controller_goal_targeting_start_step": 12,
                        }
                    ],
                }
            },
        }
        return manifest

    def test_write_and_reload_immutable_split_bank(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = write_task_bank(
                temporary,
                bank_id="unit_bank",
                splits={
                    "train_c0": [_layout(0, position_seed=101, obstacle_seed=201)],
                    "val_c0": [_layout(5, position_seed=102, obstacle_seed=202)],
                },
                split_roles={"train_c0": "train", "val_c0": "validation"},
                split_runtime_configs=_runtime_config("train_c0", "val_c0"),
            )
            loaded = load_task_bank(temporary)
            self.assertEqual(loaded, manifest)
            self.assertEqual(
                loaded["splits"]["train_c0"]["deadline_summary"][
                    "maximum_deadline_steps"
                ],
                _layout(0, position_seed=101, obstacle_seed=201)["contract"][
                    "deadline"
                ]["deadline_steps"],
            )
            loaded_split = load_task_bank_split(temporary, "train_c0")
            self.assertEqual(loaded_split["runtime_config"]["terrain_complexity_level"], 1)
            self.assertEqual(len(loaded_split["layouts"]), 1)
            self.assertTrue((Path(temporary) / "tasks" / "train_c0").is_dir())
            with self.assertRaises(TaskBankError):
                write_task_bank(
                    temporary,
                    bank_id="overwrite",
                    splits={"train": [_layout(10, position_seed=999, obstacle_seed=998)]},
                    split_roles={"train": "train"},
                    split_runtime_configs=_runtime_config("train"),
                )

    def test_positions_artifact_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = write_task_bank(
                temporary,
                bank_id="tamper",
                splits={
                    "train_c0": [
                        _layout(0, position_seed=101, obstacle_seed=201)
                    ]
                },
                split_roles={"train_c0": "train"},
                split_runtime_configs=_runtime_config("train_c0"),
            )
            entry = manifest["splits"]["train_c0"]["tasks"][0]
            positions_path = Path(temporary) / entry["positions_file"]
            positions = json.loads(positions_path.read_text(encoding="utf-8"))
            positions["agents"][0][0] += 1.0
            positions_path.write_text(json.dumps(positions), encoding="utf-8")
            with self.assertRaisesRegex(TaskBankError, "positions content hash"):
                load_task_bank(temporary)

    def test_duplicate_physical_task_across_splits_is_rejected(self):
        task = _layout(0, position_seed=101, obstacle_seed=201)
        duplicate = copy.deepcopy(task)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(TaskBankError, "physical task_id"):
                write_task_bank(
                    temporary,
                    bank_id="duplicate",
                    splits={"train": [task], "validation": [duplicate]},
                    split_roles={"train": "train", "validation": "validation"},
                    split_runtime_configs=_runtime_config("train", "validation"),
                )

    def test_seed_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(TaskBankError, "position seed"):
                write_task_bank(
                    temporary,
                    bank_id="seed_leak",
                    splits={
                        "train": [_layout(0, position_seed=101, obstacle_seed=201)],
                        "validation": [_layout(5, position_seed=101, obstacle_seed=202)],
                    },
                    split_roles={"train": "train", "validation": "validation"},
                    split_runtime_configs=_runtime_config("train", "validation"),
                )

    def test_generalization_terrain_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(TaskBankError, "terrain seeds overlap"):
                write_task_bank(
                    temporary,
                    bank_id="terrain_leak",
                    splits={
                        "train": [_layout(0, position_seed=101, obstacle_seed=201, terrain_seed=67)],
                        "generalization": [_layout(5, position_seed=102, obstacle_seed=202, terrain_seed=67)],
                    },
                    split_roles={"train": "train", "generalization": "test_generalization"},
                    split_runtime_configs=_runtime_config("train", "generalization"),
                )

    def test_manifest_entry_difficulty_tampering_is_rejected_even_when_rehashed(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_task_bank(
                temporary,
                bank_id="entry_tamper",
                splits={
                    "train_c0": [_layout(0, position_seed=101, obstacle_seed=201)]
                },
                split_roles={"train_c0": "train"},
                split_runtime_configs=_runtime_config("train_c0"),
            )
            manifest_path = Path(temporary) / "task_bank_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["splits"]["train_c0"]["tasks"][0]["difficulty_bin"] = "C3"
            manifest["content_sha256"] = canonical_sha256(manifest)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(TaskBankError, "task difficulty mismatch"):
                load_task_bank(manifest_path)

    def test_manifest_member_path_escape_is_rejected_even_when_rehashed(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_task_bank(
                temporary,
                bank_id="path_tamper",
                splits={
                    "train_c0": [_layout(0, position_seed=101, obstacle_seed=201)]
                },
                split_roles={"train_c0": "train"},
                split_runtime_configs=_runtime_config("train_c0"),
            )
            manifest_path = Path(temporary) / "task_bank_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["splits"]["train_c0"]["tasks"][0][
                "task_layout_file"
            ] = "../outside.json"
            manifest["content_sha256"] = canonical_sha256(manifest)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(TaskBankError, "safe relative path"):
                load_task_bank(manifest_path)

    def test_dynamic_feasibility_audit_is_bound_to_task_deadline_and_agent_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = write_task_bank(
                temporary,
                bank_id="dynamic_audit",
                splits={
                    "train_c0": [_layout(0, position_seed=101, obstacle_seed=201)]
                },
                split_roles={"train_c0": "train"},
                split_runtime_configs=_runtime_config("train_c0"),
                metadata={},
            )
            manifest = self._attach_passing_dynamic_audit(manifest)
            self.assertEqual(validate_dynamic_feasibility_audit(manifest), [])
            broken = copy.deepcopy(manifest)
            episode = broken["metadata"]["dynamic_feasibility_audit"]["splits"][
                "train_c0"
            ]["episodes"][0]
            episode["agent_success_flags"] = [1, 0, 1]
            episode["deadline_steps"] += 1
            errors = validate_dynamic_feasibility_audit(broken)
            self.assertTrue(any("agent success mismatch" in error for error in errors))
            self.assertTrue(any("deadline mismatch" in error for error in errors))

            terminal_before_capture = copy.deepcopy(manifest)
            witness = terminal_before_capture["metadata"][
                "dynamic_feasibility_audit"
            ]["splits"]["train_c0"]["episodes"][0]
            witness["controller_waypoint_capture_completed"][0] = False
            witness["controller_corridor_completion_steps"][0] = None
            witness["controller_goal_targeting_start_step"] = None
            witness["final_goal_distances_m"][0] = 1.9
            self.assertEqual(
                validate_dynamic_feasibility_audit(terminal_before_capture), []
            )
            witness["final_goal_distances_m"][0] = 2.1
            errors = validate_dynamic_feasibility_audit(
                terminal_before_capture
            )
            self.assertTrue(
                any("lacks a legal terminal success" in error for error in errors)
            )

            early_success = copy.deepcopy(manifest)
            witness = early_success["metadata"][
                "dynamic_feasibility_audit"
            ]["splits"]["train_c0"]["episodes"][0]
            witness["controller_goal_success_steps"] = [9, 10, 13]
            self.assertEqual(
                validate_dynamic_feasibility_audit(early_success), []
            )

            invalid_targeting_phase = copy.deepcopy(early_success)
            witness = invalid_targeting_phase["metadata"][
                "dynamic_feasibility_audit"
            ]["splits"]["train_c0"]["episodes"][0]
            witness["controller_goal_targeting_start_step"] = 11
            errors = validate_dynamic_feasibility_audit(
                invalid_targeting_phase
            )
            self.assertTrue(
                any(
                    "goal-targeting phase timing invalid" in error
                    for error in errors
                )
            )

    def test_task_semantics_drift_is_rejected_before_bank_write(self):
        runtime = _runtime_config("train_c0")
        runtime["train_c0"]["collision_distance_threshold"] = 0.4
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                TaskBankError, "collision_distance_threshold"
            ):
                write_task_bank(
                    temporary,
                    bank_id="semantics_drift",
                    splits={
                        "train_c0": [
                            _layout(0, position_seed=101, obstacle_seed=201)
                        ]
                    },
                    split_roles={"train_c0": "train"},
                    split_runtime_configs=runtime,
                )


if __name__ == "__main__":
    unittest.main()
