import inspect
import unittest
import numpy as np
from action_semantics import (select_replay_action_channels, actor_critic_apf_enabled,
                              validate_matd3_action_semantics, raw_only_critic_actions)
from paper_seed_statistics import paired_seed_sign_flip, holm_adjust
from paper_experiment_protocol import method_spec
from paper_results_summary import _paired_hierarchical_difference, _paired_terrain_crossed_difference
from paper3d_train_optimized import LiteReplayBuffer, OptimizedMATD3
from multiagent.scenarios.paper3d_terrain_energy import Scenario


class RawOnlyTests(unittest.TestCase):
    def test_replay_stores_proposal_even_when_execution_reverses_direction(self):
        raw = np.full((2, 7), 0.4, np.float32)
        executed = raw.copy()
        executed[:, :3] = -0.8
        for mode, expected in [('raw_only', raw), ('collapsed_replay', executed)]:
            with self.subTest(mode=mode):
                first, second = select_replay_action_channels(mode, raw, executed)
                buffer = LiteReplayBuffer(8, 2, [4, 4], [7, 7], use_per=False, seed=11)
                buffer.add_single_aligned(np.zeros((2, 4)), np.ones((2, 4)), first,
                                          np.zeros(2), np.zeros(2), act_corrected=second)
                batch = buffer._build_replay_batch(np.array([0]))
                np.testing.assert_array_equal(batch.actions_raw[0], expected)
                np.testing.assert_array_equal(batch.actions_corr[0], expected)
        first, second = select_replay_action_channels('dual', raw, executed)
        np.testing.assert_array_equal(first, raw)
        np.testing.assert_array_equal(second, executed)

    def test_missing_replay_alias_stays_raw_and_mixed_alias_is_rejected(self):
        raw = np.full((2, 3, 7), 0.4, dtype=np.float32)
        np.testing.assert_array_equal(raw_only_critic_actions(raw), raw)
        np.testing.assert_array_equal(raw_only_critic_actions(raw, raw.copy()), raw)
        with self.assertRaisesRegex(ValueError, "mixed action semantics"):
            raw_only_critic_actions(raw, -raw)

    def test_raw_only_learning_gate_does_not_disable_execution(self):
        self.assertFalse(actor_critic_apf_enabled('raw_only', True))
        for mode in ('dual', 'executed_control', 'collapsed_replay'):
            self.assertTrue(actor_critic_apf_enabled(mode, True))
            self.assertFalse(actor_critic_apf_enabled(mode, False))
        config = method_spec('RawOnly-MATD3-APF')['environment']
        control = method_spec('EC-MATD3-APF')['environment']
        differences = {k for k in config if config[k] != control[k]}
        self.assertEqual(differences, {'MATD3_ACTION_SEMANTICS_MODE', 'MATD3_RECONSTRUCT_CORRECTED_TARGET'})
        self.assertEqual(config['ENABLE_ACTION_CORRECTION'], 'true')

    def test_inconsistent_raw_only_flags_are_rejected(self):
        valid = dict(use_dual_q=False, separated=False, hybrid=False, reconstruct_corrected_target=False)
        self.assertEqual(validate_matd3_action_semantics('raw_only', **valid), 'raw_only')
        for key in valid:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_matd3_action_semantics('raw_only', **{**valid, key: True})

    def test_raw_only_rejects_apf_bypass_exploration(self):
        with self.assertRaisesRegex(ValueError, "bypass APF"):
            validate_matd3_action_semantics('raw_only', use_dual_q=False,
                separated=False, hybrid=False, reconstruct_corrected_target=False,
                random_action_prob=0.01)

    def test_batched_observation_matches_scalar_bounds_after_large_excursion(self):
        scenario = Scenario(seed=67, terrain_complexity_level=1, use_dynamic_obstacles=False)
        world = scenario.make_world()
        agent = world.agents[0]
        agent.state.p_pos = np.array([3500., -3100., 2500.], dtype=np.float32)
        batch = scenario._compute_observations_batch_uncached(world)[id(agent)]
        scalar = scenario._compute_observation_uncached(agent, world)
        self.assertEqual(batch.shape, (81,))
        self.assertTrue(np.isfinite(batch).all())
        self.assertLessEqual(float(np.abs(batch).max()), 10.0)
        np.testing.assert_allclose(batch[:9], scalar[:9], atol=1e-6)


class PairedSeedTests(unittest.TestCase):
    def test_three_seeds_cannot_be_made_significant_by_duplicating_episodes(self):
        a, b = np.ones((3, 1)), np.zeros((3, 1))
        one = paired_seed_sign_flip(a, b)
        repeated = paired_seed_sign_flip(np.repeat(a, 900, axis=1), np.repeat(b, 900, axis=1))
        self.assertEqual(one['p_value_two_sided'], 0.25)
        self.assertEqual(one, repeated)

    def test_null_and_sign_reversal(self):
        a = np.arange(18).reshape(6, 3)
        b = a + np.arange(6)[:, None] + 1
        self.assertEqual(paired_seed_sign_flip(a, a)['p_value_two_sided'], 1.0)
        self.assertEqual(paired_seed_sign_flip(a, b)['p_value_two_sided'], 0.03125)
        self.assertEqual(paired_seed_sign_flip(a, b)['p_value_two_sided'],
                         paired_seed_sign_flip(b, a)['p_value_two_sided'])

    def test_invalid_pairing_and_values(self):
        for a, b in [(np.zeros((1, 3)), np.zeros((1, 3))),
                     (np.zeros((3, 2)), np.zeros((2, 3))),
                     (np.full((3, 2), np.nan), np.zeros((3, 2)))]:
            with self.assertRaises(ValueError):
                paired_seed_sign_flip(a, b)

    def test_holm_known_example_preserves_order(self):
        np.testing.assert_allclose(holm_adjust([0.04, 0.01, 0.03]), [0.06, 0.03, 0.06])

    def test_statistic_is_in_both_summary_paths(self):
        for shape, function in [((3, 4), _paired_hierarchical_difference),
                                ((3, 2, 4), _paired_terrain_crossed_difference)]:
            result = function(np.ones(shape), np.zeros(shape), bootstrap_samples=1000,
                              random_seed=31, better_direction='higher')
            self.assertEqual(result['paired_seed_test']['p_value_two_sided'], 0.25)


if __name__ == '__main__':
    unittest.main()
