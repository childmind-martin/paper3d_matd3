import copy
import unittest

from paper_curriculum import (
    build_paper_curriculum,
    build_paper_generalization_suite,
    promotion_decision,
    validation_checkpoint_gate,
    validate_paper_curriculum,
)
from paper_experiment_schema import PaperSchemaError


class PaperCurriculumTests(unittest.TestCase):
    def test_training_stages_are_monotonic_and_exclude_held_out_suite(self):
        stages = build_paper_curriculum(67)
        self.assertEqual([stage["difficulty_bin"] for stage in stages], ["C0", "C1", "C2", "C3"])
        self.assertEqual(stages[0]["paper_task_num_obstacles"], 0)
        self.assertEqual(
            [stage["max_start_goal_dist"] for stage in stages],
            sorted(stage["max_start_goal_dist"] for stage in stages),
        )
        self.assertTrue(all(stage["paper_task_contract_strict"] for stage in stages))

    def test_generalization_suite_is_evaluation_only(self):
        suites = build_paper_generalization_suite([701, 719, 701])
        self.assertEqual([suite["terrain_seed"] for suite in suites], [701, 719])
        self.assertTrue(all(suite["evaluation_only"] for suite in suites))
        self.assertTrue(all(suite["split"] == "test_generalization" for suite in suites))
        self.assertTrue(all(not suite["semi_random_terrain"] for suite in suites))
        self.assertTrue(all(not suite["per_episode_terrain"] for suite in suites))

    def test_promotion_requires_sample_count_successes_and_wilson_lower_bound(self):
        contract = {
            "window": 40,
            "minimum_episodes": 40,
            "minimum_successes": 32,
            "wilson_lower_bound": 0.65,
        }
        self.assertFalse(promotion_decision([True] * 31 + [False] * 8, contract)["promoted"])
        self.assertTrue(promotion_decision([True] * 38 + [False] * 2, contract)["promoted"])

    def test_invalid_stage_contract_fails_loudly(self):
        stages = build_paper_curriculum(67)
        broken = copy.deepcopy(stages)
        broken[1]["hard_deadline_terminal"] = False
        with self.assertRaises(PaperSchemaError):
            validate_paper_curriculum(broken)

    def test_empty_generalization_seed_list_is_rejected(self):
        with self.assertRaises(PaperSchemaError):
            build_paper_generalization_suite([])

    def test_validation_gates_freeze_exact_sample_size_and_progressive_targets(self):
        gates = [validation_checkpoint_gate(name) for name in ("C0", "C1", "C2", "C3")]
        self.assertTrue(all(gate["required_tasks"] == 64 for gate in gates))
        self.assertEqual(
            [gate["minimum_successes"] for gate in gates],
            [54, 51, 48, 44],
        )
        self.assertTrue(
            all(gate["protocol_id"] == "validation_checkpoint_adequacy_v1" for gate in gates)
        )


if __name__ == "__main__":
    unittest.main()
