import unittest

import numpy as np

from paper_experiment_schema import PaperSchemaError
from paper_results_summary import (
    _build_confirmatory_result,
    _descriptive_hierarchical_bootstrap,
    _episode_mechanism_metric,
    _hierarchical_bootstrap,
    _paired_hierarchical_difference,
    _paired_terrain_crossed_difference,
    _terrain_crossed_bootstrap,
)


class PaperResultsSummaryTests(unittest.TestCase):
    def test_confirmatory_result_uses_predeclared_oriented_interval(self):
        contract = {
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
        }
        comparison = {
            "oriented_improvement_bootstrap_95ci": {
                "lower": 0.02,
                "median": 0.05,
                "upper": 0.08,
                "confidence": 0.95,
            }
        }
        result = _build_confirmatory_result(
            {**contract},
            {"test_C3": {"EC-MATD3": {"team_success_rate": comparison}}},
            proposed_method="EC-MATD3-APF",
        )
        self.assertTrue(result["criterion_met"])
        self.assertEqual(result["comparison"], comparison)

    def test_mechanism_metric_is_read_from_strict_episode_field(self):
        record = {
            "apf_diagnostics": {"mean_action_intervention_norm": 0.25}
        }
        self.assertEqual(
            _episode_mechanism_metric(
                record, "mean_action_intervention_norm"
            ),
            0.25,
        )
        with self.assertRaisesRegex(PaperSchemaError, "unsupported"):
            _episode_mechanism_metric(record, "success")

    def test_mechanism_bootstrap_is_explicitly_descriptive(self):
        result = _descriptive_hierarchical_bootstrap(
            np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
            bootstrap_samples=1000,
            random_seed=13,
        )
        self.assertEqual(
            result["metric_interpretation"],
            "descriptive_not_superiority",
        )
        self.assertNotIn("metric_better_direction", result)

    def test_hierarchical_bootstrap_uses_seed_by_task_matrix(self):
        values = np.asarray([[1.0, 1.0], [0.0, 0.0]], dtype=np.float64)
        first = _hierarchical_bootstrap(
            values, bootstrap_samples=2000, random_seed=17
        )
        second = _hierarchical_bootstrap(
            values, bootstrap_samples=2000, random_seed=17
        )
        self.assertEqual(first, second)
        self.assertEqual(first["estimate"], 0.5)
        self.assertEqual(first["train_seed_count"], 2)
        self.assertEqual(first["task_count_per_seed"], 2)
        self.assertEqual(first["per_train_seed_mean"], [1.0, 0.0])
        self.assertEqual(first["metric_better_direction"], "higher")
        self.assertEqual(
            first["bootstrap_design"],
            "crossed_train_seed_by_immutable_task",
        )

    def test_hierarchical_bootstrap_records_lower_is_better_direction(self):
        result = _hierarchical_bootstrap(
            np.asarray([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64),
            bootstrap_samples=2000,
            random_seed=19,
            better_direction="lower",
        )
        self.assertEqual(result["metric_better_direction"], "lower")

    def test_bootstrap_rejects_invalid_metric_direction(self):
        with self.assertRaisesRegex(PaperSchemaError, "direction"):
            _hierarchical_bootstrap(
                np.ones((2, 2)),
                bootstrap_samples=1000,
                random_seed=1,
                better_direction="sideways",
            )

    def test_paired_bootstrap_preserves_seed_and_task_pairing(self):
        baseline = np.asarray(
            [[0.0, 0.25, 0.5], [0.25, 0.5, 0.75]], dtype=np.float64
        )
        proposed = baseline + 0.2
        result = _paired_hierarchical_difference(
            proposed, baseline, bootstrap_samples=2000, random_seed=23
        )
        self.assertAlmostEqual(
            result["estimate_proposed_minus_baseline"], 0.2
        )
        interval = result["paired_hierarchical_bootstrap_95ci"]
        self.assertAlmostEqual(interval["lower"], 0.2)
        self.assertAlmostEqual(interval["upper"], 0.2)
        self.assertEqual(
            result["bootstrap_probability_difference_gt_zero"], 1.0
        )

    def test_lower_is_better_metric_orients_difference_and_interval(self):
        baseline = np.asarray([[3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        proposed = baseline - 0.5
        result = _paired_hierarchical_difference(
            proposed,
            baseline,
            bootstrap_samples=2000,
            random_seed=29,
            better_direction="lower",
        )
        self.assertAlmostEqual(
            result["estimate_proposed_minus_baseline"], -0.5
        )
        self.assertAlmostEqual(
            result["estimate_oriented_improvement"], 0.5
        )
        interval = result["oriented_improvement_bootstrap_95ci"]
        self.assertAlmostEqual(interval["lower"], 0.5)
        self.assertAlmostEqual(interval["upper"], 0.5)
        self.assertEqual(
            result[
                "bootstrap_probability_oriented_improvement_gt_zero"
            ],
            1.0,
        )

    def test_paired_bootstrap_rejects_unpaired_or_nonfinite_data(self):
        with self.assertRaisesRegex(PaperSchemaError, "differ"):
            _paired_hierarchical_difference(
                np.ones((2, 3)),
                np.ones((2, 2)),
                bootstrap_samples=1000,
                random_seed=1,
            )
        with self.assertRaisesRegex(PaperSchemaError, "non-finite"):
            _paired_hierarchical_difference(
                np.asarray([[1.0], [np.nan]]),
                np.asarray([[0.0], [0.0]]),
                bootstrap_samples=1000,
                random_seed=1,
            )

    def test_terrain_crossed_bootstrap_preserves_three_level_contract(self):
        values = np.asarray(
            [
                [[0.0, 1.0], [2.0, 3.0]],
                [[1.0, 2.0], [3.0, 4.0]],
            ],
            dtype=np.float64,
        )
        result = _terrain_crossed_bootstrap(
            values,
            bootstrap_samples=2000,
            random_seed=31,
            better_direction="higher",
        )
        self.assertEqual(result["estimate"], 2.0)
        self.assertEqual(result["train_seed_count"], 2)
        self.assertEqual(result["held_out_terrain_count"], 2)
        self.assertEqual(result["task_count_per_terrain"], 2)
        self.assertEqual(result["per_train_seed_mean"], [1.5, 2.5])
        self.assertEqual(result["per_held_out_terrain_mean"], [1.0, 3.0])

    def test_paired_terrain_difference_orients_lower_is_better(self):
        baseline = np.arange(16, dtype=np.float64).reshape(2, 2, 4)
        proposed = baseline - 0.25
        result = _paired_terrain_crossed_difference(
            proposed,
            baseline,
            bootstrap_samples=2000,
            random_seed=37,
            better_direction="lower",
        )
        self.assertEqual(result["estimate_proposed_minus_baseline"], -0.25)
        self.assertEqual(result["estimate_oriented_improvement"], 0.25)
        interval = result["oriented_improvement_bootstrap_95ci"]
        self.assertEqual(interval["lower"], 0.25)
        self.assertEqual(interval["upper"], 0.25)


if __name__ == "__main__":
    unittest.main()
