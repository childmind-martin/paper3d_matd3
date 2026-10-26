"""Independent-seed inference, multiplicity and export regression tests."""

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import permutation_test

from paper_experiment_protocol import _REPORTING_CONTRACT
from paper_experiment_schema import content_sha256
from paper_results_summary import (
    _METRICS,
    _MECHANISM_METRICS,
    _adjust_exploratory_seed_tests,
    _build_confirmatory_result,
    _paired_hierarchical_difference,
    _paired_terrain_crossed_difference,
    _seed_test_csv_fields,
    summarize_results,
)
from paper_seed_statistics import holm_adjust, paired_seed_sign_flip


class PairedSeedSignFlipTests(unittest.TestCase):
    def test_three_seeds_have_exact_two_sided_floor_one_quarter(self):
        result = paired_seed_sign_flip(np.ones((3, 10)), np.zeros((3, 10)))
        self.assertEqual(result['p_value_two_sided'], 0.25)
        self.assertEqual(result['minimum_nonzero_reportable_p_value'], 0.25)
        self.assertEqual(result['independent_train_seed_count'], 3)
        self.assertEqual(result['sign_patterns'], 8)
        self.assertTrue(result['exact'])

    def test_evaluation_replication_does_not_create_training_seeds(self):
        delta = np.array([[1., 2.], [3., -1.], [2., 4.]])
        original = paired_seed_sign_flip(delta, np.zeros_like(delta))
        duplicated = np.tile(delta, (1, 100))
        repeated = paired_seed_sign_flip(duplicated, np.zeros_like(duplicated))
        self.assertEqual(original, repeated)
        self.assertEqual(repeated['independent_train_seed_count'], 3)

    def test_terrain_and_task_axes_are_averaged_before_testing(self):
        delta = np.arange(24.).reshape(3, 2, 4)
        result = paired_seed_sign_flip(delta, np.zeros_like(delta))
        flattened = paired_seed_sign_flip(delta.reshape(3, 8), np.zeros((3, 8)))
        self.assertEqual(result, flattened)
        self.assertEqual(result['per_train_seed_mean_difference'], [3.5, 11.5, 19.5])

    def test_identical_methods_have_p_one(self):
        result = paired_seed_sign_flip(np.zeros((3, 5)), np.zeros((3, 5)))
        self.assertEqual(result['p_value_two_sided'], 1.0)

    def test_zero_mean_difference_has_p_one(self):
        delta = np.array([[-1.], [0.], [1.]])
        self.assertEqual(paired_seed_sign_flip(delta, np.zeros_like(delta))['p_value_two_sided'], 1.0)

    def test_exact_test_matches_scipy_paired_label_permutation(self):
        rng = np.random.default_rng(48)
        for n in (2, 3, 5, 8):
            a, b = rng.normal(size=(2, n, 4))
            with self.subTest(n=n):
                ours = paired_seed_sign_flip(a, b)
                oracle = permutation_test(
                    (a.mean(axis=1), b.mean(axis=1)),
                    lambda x, y: np.mean(x - y),
                    permutation_type='samples', vectorized=False,
                    n_resamples=np.inf, alternative='two-sided',
                )
                self.assertAlmostEqual(ours['p_value_two_sided'], oracle.pvalue)

    def test_p_value_is_invariant_to_metric_units(self):
        # Regression: the old max(1, |delta|) tolerance returned p=1 at 1e-20.
        delta = np.array([[1.], [2.], [3.]])
        for scale in (1e-100, 1e-20, 1., 1e100):
            with self.subTest(scale=scale):
                result = paired_seed_sign_flip(delta * scale, np.zeros_like(delta))
                self.assertEqual(result['p_value_two_sided'], 0.25)

    def test_swapping_methods_or_seed_order_preserves_two_sided_p(self):
        delta = np.array([[3.], [-1.], [2.], [0.5]])
        reference = paired_seed_sign_flip(delta, np.zeros_like(delta))['p_value_two_sided']
        self.assertEqual(paired_seed_sign_flip(np.zeros_like(delta), delta)['p_value_two_sided'], reference)
        self.assertEqual(paired_seed_sign_flip(delta[::-1], np.zeros_like(delta))['p_value_two_sided'], reference)

    def test_invalid_pairing_nonfinite_and_empty_axes_are_rejected(self):
        cases = [
            (np.ones((3, 2)), np.ones((3, 3))),
            (np.ones((1, 2)), np.zeros((1, 2))),
            (np.ones(3), np.zeros(3)),
            (np.ones((3, 0)), np.zeros((3, 0))),
            (np.array([[np.nan], [1.]]), np.zeros((2, 1))),
            (np.ones((2, 1)), np.array([[np.inf], [1.]])),
        ]
        for a, b in cases:
            with self.subTest(shape=a.shape), self.assertRaises(ValueError):
                paired_seed_sign_flip(a, b)

    def test_overflow_cannot_produce_a_spurious_zero_p_value(self):
        with self.assertRaisesRegex(ValueError, 'overflowed'):
            paired_seed_sign_flip(np.full((3, 1), 1e308), np.full((3, 1), -1e308))

    def test_monte_carlo_is_reproducible_and_uses_plus_one(self):
        a, b = np.ones((21, 2)), np.zeros((21, 2))
        first = paired_seed_sign_flip(a, b, random_seed=19, samples=1000)
        self.assertEqual(first, paired_seed_sign_flip(a, b, random_seed=19, samples=1000))
        self.assertFalse(first['exact'])
        self.assertEqual(first['random_seed'], 19)
        self.assertEqual(first['p_value_computation'], 'monte_carlo_plus_one_conservative')
        self.assertGreaterEqual(first['p_value_two_sided'], 1 / 1001)
        self.assertAlmostEqual(first['p_value_two_sided'] * 1001, round(first['p_value_two_sided'] * 1001))

    def test_monte_carlo_sample_count_must_be_sufficient_integer(self):
        for samples in (999, 1000.5, '1000'):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                paired_seed_sign_flip(np.ones((21, 2)), np.zeros((21, 2)), samples=samples)


class HolmAdjustmentTests(unittest.TestCase):
    def test_known_values_preserve_input_order_and_step_down_monotonicity(self):
        np.testing.assert_allclose(holm_adjust([0.03, 0.01, 0.04, 0.002]), [0.06, 0.03, 0.06, 0.008])

    def test_ties_empty_and_boundaries(self):
        self.assertEqual(holm_adjust([]), [])
        self.assertEqual(holm_adjust([1., 0.]), [1., 0.])
        self.assertEqual(holm_adjust([0.1, 0.1]), [0.2, 0.2])
        self.assertEqual(holm_adjust([0.6, 0.7, 1.]), [1., 1., 1.])

    def test_invalid_p_values_are_rejected(self):
        for values in ([np.nan], [np.inf], [-0.1], [1.1], [[0.1]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                holm_adjust(values)


class SeedDiagnosticSummaryTests(unittest.TestCase):
    @staticmethod
    def comparison(p=0.01):
        result = paired_seed_sign_flip(np.ones((3, 2)), np.zeros((3, 2)))
        result['p_value_two_sided'] = p
        return {'paired_seed_test': result}

    def test_one_holm_family_spans_all_splits_baselines_metrics_and_aggregate(self):
        comparisons = {
            split: {baseline: {metric: self.comparison() for metric in ('success', 'collision')}
                    for baseline in ('A', 'B')}
            for split in ('test_C0', 'test_C3')
        }
        aggregate = {'group_id': 'G1', 'paired_comparisons': {
            baseline: {metric: self.comparison() for metric in ('success', 'collision')}
            for baseline in ('A', 'B')
        }}
        analysis = _adjust_exploratory_seed_tests(comparisons, aggregate)
        self.assertEqual(analysis['multiplicity_family_size'], 12)
        self.assertEqual(len(analysis['family_members']), 12)
        for groups in (comparisons['test_C0'], comparisons['test_C3'], aggregate['paired_comparisons']):
            for metrics in groups.values():
                for comparison in metrics.values():
                    test = comparison['paired_seed_test']
                    self.assertAlmostEqual(test['p_value_two_sided_holm'], 0.12)
                    self.assertEqual(test['multiplicity_family_size'], 12)
                    self.assertEqual(test['multiplicity_family_id'], analysis['multiplicity_family_id'])

    def test_family_without_generalization_exports_diagnostic_labels(self):
        comparison = self.comparison(0.25)
        result = _adjust_exploratory_seed_tests({'C3': {'A': {'success': comparison}}}, None)
        self.assertEqual(result['multiplicity_family_size'], 1)
        fields = _seed_test_csv_fields(comparison)
        self.assertEqual(fields['seed_test_p_value_two_sided_holm'], 0.25)
        self.assertEqual(fields['seed_test_independent_train_seed_count'], 3)
        self.assertEqual(fields['bootstrap_probability_interpretation'], 'descriptive_resample_fraction_not_p_value')
        self.assertEqual(fields['seed_test_inference_role'], 'exploratory_diagnostic_not_confirmatory_decision')

    def test_confirmatory_ci_rule_is_unchanged_when_diagnostic_p_is_one(self):
        contract = copy.deepcopy(_REPORTING_CONTRACT['confirmatory_analysis'])
        comparison = self.comparison(1.)
        comparison['oriented_improvement_bootstrap_95ci'] = {'lower': 0.01, 'upper': 0.2, 'confidence': 0.95}
        comparisons = {contract['split']: {contract['baseline_method']: {contract['metric']: comparison}}}
        _adjust_exploratory_seed_tests(comparisons, None)
        result = _build_confirmatory_result(contract, comparisons, proposed_method=contract['proposed_method'])
        self.assertTrue(result['criterion_met'])
        self.assertEqual(result['comparison']['paired_seed_test']['p_value_two_sided_holm'], 1.)

    def test_both_bootstrap_designs_keep_resample_fraction_distinct_from_p(self):
        for shape, function in (
            ((3, 4), _paired_hierarchical_difference),
            ((3, 2, 4), _paired_terrain_crossed_difference),
        ):
            with self.subTest(shape=shape):
                result = function(np.ones(shape), np.zeros(shape), bootstrap_samples=1000,
                                  random_seed=2, better_direction='higher')
                self.assertEqual(result['bootstrap_probability_difference_gt_zero'], 1.)
                self.assertEqual(result['paired_seed_test']['p_value_two_sided'], 0.25)
                self.assertEqual(result['paired_seed_test']['independent_train_seed_count'], 3)

    def test_json_and_csv_export_one_complete_family_and_source_provenance(self):
        # Synthetic fixtures exercise aggregation/export, never count as real results.
        proposed, baseline = 'EC-MATD3-APF', 'EC-MATD3'
        splits = ['test_C0', 'test_C1', 'test_C2', 'test_C3', 'G1_a', 'G1_b']
        reporting = copy.deepcopy(_REPORTING_CONTRACT)
        bank = {'splits': {
            split: {'role': 'test_generalization' if split.startswith('G1') else 'test_in_distribution',
                    'runtime_config': {'terrain_seed': index + 100}}
            for index, split in enumerate(splits)
        }}
        protocol = {
            'methods': [proposed, baseline], 'train_seeds': [11, 22, 33],
            'task_bank': {'manifest_path': 'synthetic-bank.json'},
            'split_contract': {'generalization_splits': splits[-2:]},
            'reporting': reporting, 'content_sha256': 'synthetic-protocol',
        }

        def fake_run(_manifest, **kwargs):
            offset = 0.25 if kwargs['method'] == proposed else 0.
            return {'manifest': {}, 'records': [
                {'task_id': f"{kwargs['split']}-{task}",
                 'metrics': {metric: kwargs['train_seed'] / 100 + task / 10 + offset
                             for metric in _METRICS + _MECHANISM_METRICS}}
                for task in (0, 1)
            ]}

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol_path = root / 'protocol.json'
            protocol_path.write_text('{}')
            for method in protocol['methods']:
                for seed in protocol['train_seeds']:
                    for split in splits:
                        path = root / 'official_test' / method / f'seed_{seed}' / split / 'paper_run_manifest.json'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text('{}')
            with patch('paper_results_summary.load_protocol', return_value=protocol), \
                 patch('paper_results_summary.load_task_bank', return_value=bank), \
                 patch('paper_results_summary._load_official_run', side_effect=fake_run), \
                 patch('paper_results_summary._episode_metric', side_effect=lambda record, metric: record['metrics'][metric]), \
                 patch('paper_results_summary._episode_mechanism_metric', side_effect=lambda record, metric: record['metrics'][metric]):
                payload = summarize_results(protocol_path=protocol_path, experiment_root=root,
                                            splits=splits, output_dir=root / 'summary', bootstrap_samples=1000)
            saved = json.loads((root / 'summary/paper_results_summary.json').read_text())
            self.assertEqual(saved, payload)
            self.assertEqual(saved['schema_version'], 'paper_results_summary_v6')
            self.assertEqual(saved['content_sha256'], content_sha256({key: value for key, value in saved.items() if key != 'content_sha256'}))
            family = saved['exploratory_seed_test_analysis']
            self.assertEqual(family['multiplicity_family_size'], 42)
            self.assertEqual(len(family['family_members']), 42)
            self.assertTrue(saved['confirmatory_analysis']['criterion_met'])
            self.assertIn('paper_seed_statistics.py', [row['logical_name'] for row in saved['summary_source_artifacts']])
            with (root / 'summary/paper_paired_comparisons.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 42)
            self.assertEqual({row['split'] for row in rows}, set(splits + ['G1_held_out_terrain_aggregate']))
            for row in rows:
                self.assertEqual(float(row['seed_test_p_value_two_sided']), 0.25)
                self.assertEqual(float(row['seed_test_p_value_two_sided_holm']), 1.)
                self.assertEqual(int(row['seed_test_independent_train_seed_count']), 3)
                self.assertEqual(int(row['seed_test_multiplicity_family_size']), 42)
                self.assertEqual(row['seed_test_multiplicity_family_id'], family['multiplicity_family_id'])
                self.assertEqual(row['bootstrap_probability_interpretation'], 'descriptive_resample_fraction_not_p_value')


if __name__ == '__main__':
    unittest.main()
