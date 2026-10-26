import unittest

from paper_experiment_schema import PaperSchemaError
from paper_task_bank_report import _numeric_summary, _optional_numeric_summary


class PaperTaskBankReportTests(unittest.TestCase):
    def test_numeric_summary_uses_paper_table_quantiles(self):
        summary = _numeric_summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["minimum"], 1.0)
        self.assertEqual(summary["median"], 2.5)
        self.assertEqual(summary["maximum"], 4.0)
        self.assertGreater(summary["sample_std"], 0.0)

    def test_optional_summary_omits_unavailable_obstacle_clearance(self):
        self.assertIsNone(_optional_numeric_summary([None, None]))
        self.assertEqual(
            _optional_numeric_summary([None, 1.5, 2.5])["mean"], 2.0
        )

    def test_nonfinite_characterization_fails_loudly(self):
        with self.assertRaises(PaperSchemaError):
            _numeric_summary([1.0, float("nan")])


if __name__ == "__main__":
    unittest.main()
