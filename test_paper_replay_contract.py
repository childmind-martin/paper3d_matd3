import unittest

import numpy as np

from paper3d_train_optimized import LiteReplayBuffer


class PaperReplayContractTests(unittest.TestCase):
    def _buffer(self, capacity=8):
        return LiteReplayBuffer(
            capacity=capacity,
            n_agents=2,
            obs_dims=[4, 4],
            act_dims=[7, 7],
            use_per=False,
            seed=11,
        )

    def _transition(self, value=0.0):
        obs = np.full((2, 4), value, dtype=np.float32)
        next_obs = np.full((2, 4), value + 1.0, dtype=np.float32)
        actions = np.full((2, 7), value, dtype=np.float32)
        rewards = np.asarray([value, value + 0.5], dtype=np.float32)
        return obs, next_obs, actions, rewards

    def test_single_transition_preserves_terminal_and_task_metadata(self):
        buffer = self._buffer()
        obs, next_obs, actions, rewards = self._transition()
        buffer.add_single_aligned(
            obs,
            next_obs,
            actions,
            rewards,
            done_n=np.ones(2, dtype=np.float32),
            terminated_n=np.ones(2, dtype=np.float32),
            truncated_n=np.zeros(2, dtype=np.float32),
            task_id='task-0001',
            task_difficulty_bin='C0',
            curriculum_stage_id=2,
        )
        batch = buffer._build_replay_batch(np.asarray([0], dtype=np.int64))
        np.testing.assert_array_equal(batch.dones, [[1.0, 1.0]])
        np.testing.assert_array_equal(batch.terminated, [[1.0, 1.0]])
        np.testing.assert_array_equal(batch.truncated, [[0.0, 0.0]])
        self.assertEqual(int(batch.curriculum_stage_id[0]), 2)
        self.assertEqual(buffer.task_id_by_hash[int(batch.task_hash_u64[0])], 'task-0001')
        self.assertEqual(
            buffer.task_difficulty_by_hash[int(batch.task_difficulty_hash_u32[0])],
            'C0',
        )

    def test_truncation_is_audit_field_not_bootstrap_terminal(self):
        buffer = self._buffer()
        obs, next_obs, actions, rewards = self._transition()
        buffer.add_single_aligned(
            obs,
            next_obs,
            actions,
            rewards,
            done_n=np.zeros(2, dtype=np.float32),
            terminated_n=np.zeros(2, dtype=np.float32),
            truncated_n=np.ones(2, dtype=np.float32),
            task_id='task-truncated',
            task_difficulty_bin='audit',
        )
        batch = buffer._build_replay_batch(np.asarray([0], dtype=np.int64))
        np.testing.assert_array_equal(batch.dones, [[0.0, 0.0]])
        np.testing.assert_array_equal(batch.terminated, [[0.0, 0.0]])
        np.testing.assert_array_equal(batch.truncated, [[1.0, 1.0]])

    def test_batch_transition_metadata_stays_aligned(self):
        buffer = self._buffer()
        transitions = [self._transition(1.0), self._transition(2.0)]
        positions, _ = buffer.add_batch(
            obs_batch=[item[0] for item in transitions],
            next_obs_batch=[item[1] for item in transitions],
            act_batch=[item[2] for item in transitions],
            rew_batch=[item[3] for item in transitions],
            done_batch=[np.zeros(2), np.ones(2)],
            terminated_batch=[np.zeros(2), np.ones(2)],
            truncated_batch=[np.zeros(2), np.zeros(2)],
            task_ids=['task-a', 'task-b'],
            task_difficulty_bins=['C0', 'C1'],
            curriculum_stage_ids=[1, 3],
        )
        batch = buffer._build_replay_batch(positions)
        np.testing.assert_array_equal(batch.curriculum_stage_id, [1, 3])
        np.testing.assert_array_equal(batch.dones[:, 0], [0.0, 1.0])
        labels = [buffer.task_id_by_hash[int(value)] for value in batch.task_hash_u64]
        self.assertEqual(labels, ['task-a', 'task-b'])

    def test_formal_batch_write_does_not_enable_next_behavior_action_targets(self):
        buffer = self._buffer()
        transitions = [self._transition(1.0), self._transition(2.0)]
        positions, _ = buffer.add_batch(
            obs_batch=[item[0] for item in transitions],
            next_obs_batch=[item[1] for item in transitions],
            act_batch=[item[2] for item in transitions],
            rew_batch=[item[3] for item in transitions],
            done_batch=[np.zeros(2), np.zeros(2)],
            next_act_corrected_batch=None,
        )
        self.assertFalse(buffer.stores_next_act_corrected)
        batch = buffer._build_replay_batch(positions)
        np.testing.assert_array_equal(batch.next_actions_corr_compat, 0.0)


if __name__ == '__main__':
    unittest.main()
