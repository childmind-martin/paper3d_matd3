import tempfile
import unittest
from pathlib import Path

from paper_experiment_schema import (
    PAPER_EPISODE_SCHEMA_VERSION,
    PaperSchemaError,
    aggregate_episode_records,
    episode_record_from_evaluation,
    validate_episode_record,
    wilson_interval,
    write_episode_csv,
)


class PaperExperimentSchemaTests(unittest.TestCase):
    def _record(self, episode_index=0, success=True):
        agents = []
        for agent_index in range(3):
            path_length = 12.0 + agent_index
            reference_length = 10.0
            agents.append(
                {
                    "agent_index": agent_index,
                    "reached_goal": success,
                    "collision_free": True,
                    "success": success,
                    "path_length_m": path_length,
                    "reference_path_length_m": reference_length,
                    "spl": reference_length / path_length if success else 0.0,
                    "minimum_clearance_m": 3.0 + agent_index,
                    "energy": 20.0 + agent_index,
                    "control_effort": 5.0 + agent_index,
                    "final_goal_distance_m": 0.5 if success else 15.0,
                }
            )
        return {
            "schema_version": PAPER_EPISODE_SCHEMA_VERSION,
            "run_id": "run-1",
            "algorithm": "ec-matd3",
            "split": "test",
            "train_seed": 101,
            "eval_seed": 9001,
            "episode_index": episode_index,
            "task_id": f"task-{episode_index}",
            "difficulty_bin": "C1",
            "terminal_reason": "team_success" if success else "time_limit",
            "team_success": success,
            "episode_steps": 50,
            "deadline_steps": 50,
            "deadline_seconds": 4.0,
            "simulation_dt_seconds": 0.08,
            "task_difficulty_summary": {
                "agent_count": 3,
                "obstacle_count": 4,
                "minimum_horizontal_distance_m": 45.0,
                "maximum_horizontal_distance_m": 60.0,
                "mean_reference_path_length_m": 55.0,
                "maximum_reference_path_length_m": 60.0,
                "maximum_detour_ratio": 1.2,
                "maximum_cumulative_climb_m": 5.0,
                "minimum_boundary_clearance_m": 4.0,
                "minimum_obstacle_clearance_m": 2.0,
            },
            "apf_diagnostics": {
                "action_diagnostic_steps": 50,
                "missing_action_diagnostic_steps": 0,
                "missing_pf_force_steps": 0,
                "configured_action_force_ratio": 0.3,
                "mean_raw_action_norm": 0.8,
                "mean_executed_action_norm": 0.9,
                "mean_action_intervention_norm": 0.2,
                "apf_intervention_step_fraction": 0.6,
                "mean_apf_force_norm": 0.4,
            },
            "agents": agents,
        }

    def test_wilson_interval_is_bounded(self):
        interval = wilson_interval(8, 10)
        self.assertLess(interval["lower"], 0.8)
        self.assertGreater(interval["upper"], 0.8)
        self.assertGreaterEqual(interval["lower"], 0.0)
        self.assertLessEqual(interval["upper"], 1.0)

    def test_success_definition_is_fail_loud(self):
        record = self._record(success=True)
        record["agents"][0]["collision_free"] = False
        with self.assertRaisesRegex(PaperSchemaError, "success must equal"):
            validate_episode_record(record)

    def test_spl_definition_is_fail_loud(self):
        record = self._record(success=True)
        record["agents"][1]["spl"] = 1.0
        with self.assertRaisesRegex(PaperSchemaError, "inconsistent"):
            validate_episode_record(record)

    def test_deadline_units_are_fail_loud(self):
        record = self._record(success=True)
        record["deadline_seconds"] = 5.0
        with self.assertRaisesRegex(PaperSchemaError, "inconsistent"):
            validate_episode_record(record)

    def test_task_difficulty_values_are_fail_loud(self):
        record = self._record(success=True)
        record["task_difficulty_summary"]["maximum_detour_ratio"] = float("nan")
        with self.assertRaisesRegex(PaperSchemaError, "finite"):
            validate_episode_record(record)

    def test_apf_mechanism_data_are_required_and_bounded(self):
        record = self._record(success=True)
        del record["apf_diagnostics"]
        with self.assertRaisesRegex(PaperSchemaError, "apf_diagnostics"):
            validate_episode_record(record)
        record = self._record(success=True)
        record["apf_diagnostics"]["apf_intervention_step_fraction"] = 1.1
        with self.assertRaisesRegex(PaperSchemaError, "<= 1.0"):
            validate_episode_record(record)

    def test_apf_diagnostic_steps_must_cover_complete_episode(self):
        record = self._record(success=True)
        record["apf_diagnostics"]["action_diagnostic_steps"] = 49
        with self.assertRaisesRegex(PaperSchemaError, "must equal episode_steps"):
            validate_episode_record(record)

    def test_apf_diagnostic_missing_steps_fail_loudly(self):
        record = self._record(success=True)
        record["apf_diagnostics"]["missing_action_diagnostic_steps"] = 1
        with self.assertRaisesRegex(PaperSchemaError, "zero missing action"):
            validate_episode_record(record)
        record = self._record(success=True)
        record["apf_diagnostics"]["missing_pf_force_steps"] = 1
        with self.assertRaisesRegex(PaperSchemaError, "zero missing PF"):
            validate_episode_record(record)

    def test_aggregate_reports_binomial_intervals_and_hash(self):
        aggregate = aggregate_episode_records(
            [self._record(0, True), self._record(1, False)]
        )
        self.assertEqual(aggregate["episodes"], 2)
        self.assertEqual(aggregate["team_success"]["estimate"], 0.5)
        self.assertEqual(len(aggregate["content_sha256"]), 64)

    def test_duplicate_task_episode_is_rejected(self):
        record = self._record(0, True)
        with self.assertRaisesRegex(PaperSchemaError, "duplicate"):
            aggregate_episode_records([record, dict(record)])

    def test_episode_csv_is_agent_long_format(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "episodes.csv"
            write_episode_csv(output, [self._record(0, True)])
            lines = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 4)
            self.assertIn("task_id", lines[0])

    def test_evaluator_conversion_uses_reference_path_and_exact_success(self):
        episode = {
            "episode": 3,
            "steps": 50,
            "episode_done_reason": "team_success",
            "team_success": 1,
            "agent_reached_flags": [1, 1, 1],
            "agent_safe_flags": [1, 1, 1],
            "agent_success_flags": [1, 1, 1],
            "agent_path_lengths": [12.0, 13.0, 14.0],
            "agent_final_goal_distances": [0.5, 0.6, 0.7],
            "agent_min_clearances": [2.0, 2.5, 3.0],
            "agent_energy_consumed": [20.0, 21.0, 22.0],
            "agent_control_effort": [5.0, 6.0, 7.0],
            "diagnostic_metrics": {
                "action_diag_step_count": 50,
                "action_diag_missing_steps": 0,
                "pf_force_diag_missing_steps": 0,
                "mean_force_ratio": 0.3,
                "avg_action_norm": 0.8,
                "avg_corr_action_norm": 0.9,
                "avg_action_delta_norm": 0.2,
                "action_intervention_step_fraction": 0.6,
                "mean_pf_force_norm": 0.4,
            },
            "task_deadline": {
                "policy_id": "reference_path_slack_v1",
                "deadline_steps": 50,
                "deadline_seconds": 4.0,
                "simulation_dt_seconds": 0.08,
            },
            "task_layout": {
                "task_id": "physical-task",
                "difficulty_bin": "C1",
                "metrics": {
                    "reference_paths": [
                        {"path_length": 10.0},
                        {"path_length": 10.0},
                        {"path_length": 10.0},
                    ],
                    "difficulty_summary": {
                        "agent_count": 3,
                        "obstacle_count": 4,
                        "minimum_horizontal_distance_m": 45.0,
                        "maximum_horizontal_distance_m": 60.0,
                        "mean_reference_path_length_m": 10.0,
                        "maximum_reference_path_length_m": 10.0,
                        "maximum_detour_ratio": 1.2,
                        "maximum_cumulative_climb_m": 5.0,
                        "minimum_boundary_clearance_m": 4.0,
                        "minimum_obstacle_clearance_m": 2.0,
                    },
                },
                "contract": {
                    "deadline": {
                        "policy_id": "reference_path_slack_v1",
                        "deadline_steps": 50,
                        "deadline_seconds": 4.0,
                        "simulation_dt_seconds": 0.08,
                    }
                },
            },
        }
        record = episode_record_from_evaluation(
            episode,
            run_id="run-1",
            algorithm="matd3",
            split="test_C1",
            train_seed=101,
            eval_seed=9001,
        )
        self.assertEqual(record["task_id"], "physical-task")
        self.assertAlmostEqual(record["agents"][0]["spl"], 10.0 / 12.0)
        self.assertEqual(record["apf_diagnostics"]["action_diagnostic_steps"], 50)
        self.assertAlmostEqual(
            record["apf_diagnostics"]["mean_action_intervention_norm"], 0.2
        )


if __name__ == "__main__":
    unittest.main()
