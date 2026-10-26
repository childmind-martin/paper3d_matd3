import unittest

import numpy as np
import tensorflow as tf

from paper3d_train_optimized import (
    OptimizedMADDPG,
    OptimizedMATD3,
    TD3_TARGET_ACTION_CONTRACT,
)


class ExecutedControlCriticTests(unittest.TestCase):
    def _algorithm_shell(self):
        algorithm = OptimizedMATD3.__new__(OptimizedMATD3)
        algorithm.n_agents = 2
        algorithm.action_dims = [7, 7]
        algorithm.total_action_dim = 14
        algorithm.execution_consistent_critic = True
        return algorithm

    def test_projection_removes_every_non_executed_tail_coordinate(self):
        algorithm = self._algorithm_shell()
        packed = tf.constant(
            [[1.0, 2.0, 3.0, 40.0, 50.0, 60.0, 70.0,
              -1.0, -2.0, -3.0, -40.0, -50.0, -60.0, -70.0]],
            dtype=tf.float32,
        )
        projected = algorithm._project_joint_action_to_executed_control(packed).numpy()
        np.testing.assert_array_equal(
            projected,
            [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0,
              -1.0, -2.0, -3.0, 0.0, 0.0, 0.0, 0.0]],
        )

    def test_target_action_contract_is_target_policy_not_replay_behavior(self):
        self.assertEqual(
            TD3_TARGET_ACTION_CONTRACT,
            "smoothed_target_policy_then_apf_v1",
        )

    def test_maddpg_algorithm_baseline_uses_the_same_executed_control_projection(self):
        algorithm = OptimizedMADDPG.__new__(OptimizedMADDPG)
        algorithm.action_dims = [7, 7]
        algorithm.total_action_dim = 14
        algorithm.execution_consistent_critic = True
        packed = tf.Variable(tf.ones((1, 14), dtype=tf.float32))
        with tf.GradientTape() as tape:
            projected = algorithm._project_joint_action_to_executed_control(
                packed
            )
            objective = tf.reduce_sum(projected)
        np.testing.assert_array_equal(
            projected.numpy(),
            [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
              1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
        )
        np.testing.assert_array_equal(
            tape.gradient(objective, packed).numpy(),
            [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
              1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
        )

    def test_projection_has_no_direct_tail_gradient(self):
        algorithm = self._algorithm_shell()
        packed = tf.Variable(tf.ones((1, 14), dtype=tf.float32))
        with tf.GradientTape() as tape:
            projected = algorithm._project_joint_action_to_executed_control(packed)
            loss = tf.reduce_sum(projected)
        gradient = tape.gradient(loss, packed).numpy()[0]
        np.testing.assert_array_equal(
            gradient,
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
             1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        )

    def test_apf_tail_reaches_executed_head_through_differentiable_transform(self):
        algorithm = self._algorithm_shell()
        algorithm.c_pf_goal_attraction = tf.constant(2.0, tf.float32)
        algorithm.c_pf_lambda_1_base = tf.constant(6.0, tf.float32)
        algorithm.c_pf_terrain_repulsion = tf.constant(1000.0, tf.float32)
        algorithm.c_pf_agent_influence_range = tf.constant(100.0, tf.float32)
        algorithm.c_pf_delta_k_att = tf.constant(1.0, tf.float32)
        algorithm.c_pf_delta_lambda_1 = tf.constant(2.0, tf.float32)
        algorithm.c_pf_delta_k_rep = tf.constant(400.0, tf.float32)
        algorithm.c_pf_delta_radius = tf.constant(20.0, tf.float32)
        algorithm.c_max_force = tf.constant(100.0, tf.float32)

        geometry_context = (
            tf.constant([[1.0, 0.0, 0.0]], tf.float32),  # goal direction
            tf.constant([[10.0]], tf.float32),
            tf.constant([[0.0, 0.0, 1.0]], tf.float32),  # terrain normal
            tf.constant([[0.2]], tf.float32),
            tf.constant([[1.0]], tf.float32),
            tf.constant([[1.0]], tf.float32),
            tf.constant([[1000.0]], tf.float32),
            tf.zeros((1, 3, 1), tf.bool),
            tf.zeros((1, 3, 3), tf.float32),
            tf.ones((1, 3, 1), tf.float32) * 100.0,
            tf.ones((1, 3, 1), tf.float32) * 100.0,
            tf.zeros((1, 2, 3), tf.float32),
            tf.ones((1, 2, 1), tf.float32) * 100.0,
        )
        actor_action = tf.Variable(
            [[0.1, 0.0, 0.0, 0.2, 0.0, 0.3, 0.1]], dtype=tf.float32
        )
        with tf.GradientTape() as tape:
            executed_head, _ = algorithm._apply_potential_field_correction_from_geometry_context_tf(
                actor_action,
                tf.constant([[0.6]], tf.float32),
                tf.constant([[True]]),
                geometry_context,
            )
            objective = executed_head[0, 0] - 0.5 * executed_head[0, 2]
        gradient = tape.gradient(objective, actor_action).numpy()[0]
        self.assertTrue(np.all(np.isfinite(gradient)))
        # F = a + r(PF - a) keeps an explicit (1-r) identity path from
        # the policy head to the executed control.  The PF term is independent
        # of the raw head for valid observations, so this derivative is exact.
        np.testing.assert_allclose(
            gradient[:3],
            [0.4, 0.0, -0.2],
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertGreater(float(np.max(np.abs(gradient[3:7]))), 1e-7)


if __name__ == '__main__':
    unittest.main()
