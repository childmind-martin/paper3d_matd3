import unittest

import numpy as np

from evaluate_optimized import (
    _accumulate_eval_step_diagnostics,
    _finalize_eval_diagnostics,
    _new_eval_diagnostics_accumulator,
)


class PaperAPFDiagnosticsTests(unittest.TestCase):
    def test_actual_action_intervention_is_measured_over_complete_steps(self):
        accumulator = _new_eval_diagnostics_accumulator()
        raw = np.zeros((2, 7), dtype=np.float32)
        corrected = raw.copy()
        corrected[0, 0] = 0.3
        pf_force = np.zeros((2, 3), dtype=np.float32)
        pf_force[:, 0] = 0.4

        _accumulate_eval_step_diagnostics(
            accumulator,
            object(),
            actions=corrected,
            raw_actions=raw,
            pf_forces=pf_force,
            action_force_ratio=0.5,
        )
        _accumulate_eval_step_diagnostics(
            accumulator,
            object(),
            actions=raw,
            raw_actions=raw,
            pf_forces=np.zeros_like(pf_force),
            action_force_ratio=0.5,
        )

        metrics = _finalize_eval_diagnostics(accumulator)[
            "diagnostic_metrics"
        ]
        self.assertEqual(metrics["action_diag_step_count"], 2)
        self.assertEqual(metrics["action_diag_missing_steps"], 0)
        self.assertEqual(metrics["pf_force_diag_missing_steps"], 0)
        self.assertAlmostEqual(
            metrics["action_intervention_step_fraction"], 0.5
        )
        self.assertAlmostEqual(metrics["avg_action_delta_norm"], 0.075)
        self.assertAlmostEqual(metrics["mean_pf_force_norm"], 0.2)
        self.assertAlmostEqual(metrics["mean_force_ratio"], 0.5)

    def test_missing_raw_or_pf_data_are_counted(self):
        accumulator = _new_eval_diagnostics_accumulator()
        _accumulate_eval_step_diagnostics(
            accumulator,
            object(),
            actions=np.zeros((2, 7), dtype=np.float32),
            raw_actions=None,
            pf_forces=None,
            action_force_ratio=0.5,
        )
        metrics = _finalize_eval_diagnostics(accumulator)[
            "diagnostic_metrics"
        ]
        self.assertEqual(metrics["action_diag_step_count"], 1)
        self.assertEqual(metrics["action_diag_missing_steps"], 1)
        self.assertEqual(metrics["pf_force_diag_missing_steps"], 1)


if __name__ == "__main__":
    unittest.main()
