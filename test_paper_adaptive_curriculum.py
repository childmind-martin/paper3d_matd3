import unittest
from types import SimpleNamespace

import numpy as np

import paper3d_train_optimized as training


class PaperAdaptiveCurriculumTests(unittest.TestCase):
    @staticmethod
    def _controller():
        return SimpleNamespace(
            adaptive_learning={
                "stage_best_reward": -np.inf,
                "consecutive_no_improve": 0,
                "reward_history": [],
                "last_plateau_stats": {},
            }
        )

    def test_harder_stage_reward_does_not_compare_against_previous_stage(self):
        controller = self._controller()
        self.assertTrue(training._update_adaptive_reward_state(controller, 100.0))
        self.assertFalse(training._update_adaptive_reward_state(controller, 90.0))
        self.assertEqual(controller.adaptive_learning["consecutive_no_improve"], 1)

        training._reset_adaptive_learning_stage_state(controller)

        self.assertTrue(training._update_adaptive_reward_state(controller, -500.0))
        self.assertEqual(controller.adaptive_learning["stage_best_reward"], -500.0)
        self.assertEqual(controller.adaptive_learning["consecutive_no_improve"], 0)
        self.assertEqual(controller.adaptive_learning["reward_history"], [-500.0])

    def test_stage_reset_clears_only_plateau_comparison_state(self):
        controller = self._controller()
        controller.adaptive_learning.update(
            {
                "stage_best_reward": 42.0,
                "consecutive_no_improve": 9,
                "reward_history": [1.0, 2.0],
                "last_plateau_stats": {"plateau": True},
                "patience": 60,
                "actor_lr_decay_factor": 0.98,
            }
        )

        training._reset_adaptive_learning_stage_state(controller)

        self.assertTrue(np.isneginf(controller.adaptive_learning["stage_best_reward"]))
        self.assertEqual(controller.adaptive_learning["consecutive_no_improve"], 0)
        self.assertEqual(controller.adaptive_learning["reward_history"], [])
        self.assertEqual(controller.adaptive_learning["last_plateau_stats"], {})
        self.assertEqual(controller.adaptive_learning["patience"], 60)
        self.assertEqual(controller.adaptive_learning["actor_lr_decay_factor"], 0.98)


if __name__ == "__main__":
    unittest.main()
