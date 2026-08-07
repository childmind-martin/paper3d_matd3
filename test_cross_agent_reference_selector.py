#!/usr/bin/env python3
"""Focused invariants for the active adaptive cross-agent selector."""

from __future__ import annotations

import math
import unittest

import numpy as np
import tensorflow as tf

from cross_agent_reference_selector import (
    MODE_SHARED_TWIN_HEAD_TAIL,
    build_eligible_mask,
    build_shared_selector_features,
    next_ema_scale,
    reduce_reference_loss_fixed_denominator,
    selector_state_errors,
    selector_state_payload,
    suppress_only_multiplier,
    twin_consensus_target,
)


class CrossAgentReferenceMathTests(unittest.TestCase):
    def test_strict_eligibility_uses_full_trajectory_quality(self):
        mask = build_eligible_mask(
            finite=tf.constant([1, 1, 1, 1, 1], tf.int32),
            label_valid=tf.constant([1, 1, 1, 0, 1], tf.int32),
            safe=tf.constant([1, 1, 0, 1, 1], tf.int32),
            success=tf.constant([1, 0, 1, 1, 0], tf.int32),
            reach=tf.constant([0, 1, 0, 0, 0], tf.int32),
            trajectory_progress=tf.constant([0, 0, 0, 0, 1], tf.int32),
            near_goal=tf.zeros([5], tf.int32),
            random_mask=tf.constant([0, 0, 0, 0, 1], tf.float32),
            exclude_random=True,
        )
        np.testing.assert_array_equal(
            mask.numpy(),
            np.asarray([1, 1, 0, 0, 0], dtype=np.float32),
        )

    def test_twin_target_and_disagreement_neutrality(self):
        target, agreement, consensus = twin_consensus_target(
            tf.constant([-2.0, 2.0, 2.0, 0.0, np.nan]),
            tf.constant([-1.0, 1.0, -1.0, 0.0, 1.0]),
            tf.constant(1.0),
        )
        np.testing.assert_array_equal(
            agreement.numpy(),
            np.asarray([1, 1, 0, 1, 0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            consensus.numpy()[:4],
            np.asarray([-1.5, 1.5, 0.5, 0.0], dtype=np.float32),
            rtol=0.0,
            atol=1e-6,
        )
        multiplier = suppress_only_multiplier(target, agreement)
        self.assertAlmostEqual(float(multiplier.numpy()[2]), 1.0, places=7)
        self.assertAlmostEqual(float(multiplier.numpy()[4]), 1.0, places=7)
        self.assertLess(float(multiplier.numpy()[0]), 1.0)
        self.assertAlmostEqual(float(multiplier.numpy()[1]), 1.0, places=7)

    def test_ema_updates_once_from_masked_consensus(self):
        scale, initialized, count = next_ema_scale(
            tf.constant(1.0),
            tf.constant(False),
            tf.constant([-2.0, 9.0]),
            tf.constant([1.0, 0.0]),
            decay=0.5,
        )
        self.assertAlmostEqual(float(scale.numpy()), 2.0, places=7)
        self.assertTrue(bool(initialized.numpy()))
        self.assertEqual(float(count.numpy()), 1.0)

        scale, initialized, count = next_ema_scale(
            scale,
            initialized,
            tf.constant([4.0]),
            tf.constant([1.0]),
            decay=0.5,
        )
        self.assertAlmostEqual(float(scale.numpy()), 3.0, places=7)
        self.assertTrue(bool(initialized.numpy()))
        self.assertEqual(float(count.numpy()), 1.0)

    def test_fixed_denominator_does_not_cancel_suppression(self):
        neutral_loss, denominator = reduce_reference_loss_fixed_denominator(
            head_loss=tf.constant([1.0, 1.0]),
            tail_loss=tf.constant([0.0, 0.0]),
            eligible_mask=tf.constant([1.0, 1.0]),
            head_multiplier=tf.constant([1.0, 1.0]),
            tail_multiplier=tf.constant([1.0, 1.0]),
            head_weight=1.0,
            tail_weight=0.0,
        )
        suppressed_loss, suppressed_denominator = (
            reduce_reference_loss_fixed_denominator(
                head_loss=tf.constant([1.0, 1.0]),
                tail_loss=tf.constant([0.0, 0.0]),
                eligible_mask=tf.constant([1.0, 1.0]),
                head_multiplier=tf.constant([0.0, 1.0]),
                tail_multiplier=tf.constant([1.0, 1.0]),
                head_weight=1.0,
                tail_weight=0.0,
            )
        )
        self.assertEqual(float(denominator.numpy()), 2.0)
        self.assertEqual(float(suppressed_denominator.numpy()), 2.0)
        self.assertAlmostEqual(float(neutral_loss.numpy()), 1.0, places=7)
        self.assertAlmostEqual(float(suppressed_loss.numpy()), 0.5, places=7)

    def test_shared_features_are_leakage_free_and_detached(self):
        observation = tf.Variable(tf.ones([2, 5], tf.float32))
        teacher = tf.Variable(tf.ones([2, 7], tf.float32))
        learner = tf.Variable(tf.zeros([2, 7], tf.float32))
        force_ratio = tf.Variable(tf.fill([2, 1], 0.5))
        pf_feature = tf.Variable(tf.ones([2, 3], tf.float32))
        with tf.GradientTape() as tape:
            features = build_shared_selector_features(
                reference_observation=observation,
                teacher_action=teacher,
                learner_action=learner,
                force_ratio=force_ratio,
                pf_feature=pf_feature,
            )
            loss = tf.reduce_sum(features)
        gradients = tape.gradient(
            loss,
            [observation, teacher, learner, force_ratio, pf_feature],
        )
        self.assertEqual(tuple(features.shape), (2, 30))
        self.assertTrue(all(gradient is None for gradient in gradients))

    def test_selector_state_schema_rejects_mismatch(self):
        payload = selector_state_payload(
            mode=MODE_SHARED_TWIN_HEAD_TAIL,
            head_scale=2.0,
            tail_scale=3.0,
            head_initialized=True,
            tail_initialized=True,
            update_count=4,
            ema_decay=0.99,
            epsilon=1e-6,
            advantage_clip=5.0,
            input_dim=106,
        )
        self.assertEqual(
            selector_state_errors(
                payload,
                expected_mode=MODE_SHARED_TWIN_HEAD_TAIL,
                expected_input_dim=106,
            ),
            [],
        )
        payload["head_advantage_ema"] = math.inf
        self.assertTrue(selector_state_errors(payload))


class SharedSelectorNetworkTests(unittest.TestCase):
    def test_zero_output_kernel_is_exactly_neutral(self):
        from paper3d_train_optimized import (
            build_cross_agent_reference_selector,
        )

        network = build_cross_agent_reference_selector(
            input_dim=11,
            hidden_units=(8, 4),
            init_logit=0.0,
            output_dim=2,
            zero_output_kernel=True,
        )
        scores = network(
            tf.random.normal([16, 11], seed=7),
            training=False,
        ).numpy()
        np.testing.assert_array_equal(
            scores,
            np.full((16, 2), 0.5, dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
