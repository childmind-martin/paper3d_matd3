import os
import tempfile
import unittest
from pathlib import Path

from multiagent.scenarios.paper3d_terrain_energy import Scenario
from multiagent.environment import MultiAgentEnv
from paper_task_layout import atomic_write_json
import numpy as np


class PaperTaskReplayTests(unittest.TestCase):
    def setUp(self):
        self.previous = {
            name: os.environ.get(name)
            for name in ("QUIET_OUTPUT", "SUPPRESS_TERRAIN_OUTPUT", "ENABLE_ROLE_SHUFFLE")
        }
        os.environ["QUIET_OUTPUT"] = "1"
        os.environ["SUPPRESS_TERRAIN_OUTPUT"] = "1"
        os.environ["ENABLE_ROLE_SHUFFLE"] = "0"

    def tearDown(self):
        for name, value in self.previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    @staticmethod
    def _scenario(**extra):
        return Scenario(
            seed=67,
            use_fixed_positions=False,
            dynamic_first_time=False,
            random_terrain=False,
            terrain_complexity_level=1,
            deterministic_train_env_sequence=True,
            training_env_sequence_seed=1001,
            min_start_goal_dist=30.0,
            max_start_goal_dist=50.0,
            min_agent_start_separation=8.0,
            paper_task_contract_enabled=True,
            paper_task_contract_strict=True,
            paper_task_require_reference_feasible=True,
            paper_task_num_obstacles=0,
            task_difficulty_bin="C0",
            hard_deadline_terminal=True,
            include_time_to_go_observation=True,
            **extra,
        )

    def test_task_layout_file_replays_identical_physical_task(self):
        source = self._scenario()
        source_world = source.make_world()
        source_world.episode_index = 3
        source.reset_world(source_world)
        source_layout = source_world.current_task_layout
        self.assertIsNotNone(source_layout)

        with tempfile.TemporaryDirectory() as temporary:
            task_path = Path(temporary) / "task.json"
            atomic_write_json(task_path, source_layout)
            replay = self._scenario(paper_task_layout_file=str(task_path))
            replay_world = replay.make_world()
            replay_world.episode_index = 99
            replay.reset_world(replay_world)

        replayed_layout = replay_world.current_task_layout
        self.assertEqual(replayed_layout["task_id"], source_layout["task_id"])
        self.assertEqual(replayed_layout["agent_starts"], source_layout["agent_starts"])
        self.assertEqual(replayed_layout["agent_goals"], source_layout["agent_goals"])
        self.assertEqual(replayed_layout["obstacles"], source_layout["obstacles"])

    def test_hard_deadline_reason_survives_success_snapshot_sync(self):
        scenario = self._scenario()
        world = scenario.make_world()
        world.scenario = scenario
        world.episode_length = 1
        environment = MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
            done_callback=scenario.is_done,
            discrete_action=False,
        )
        environment.scenario = scenario
        environment.reset()
        _, _, done_n, info = environment.step(
            [np.zeros(3, dtype=np.float32) for _ in world.agents]
        )
        self.assertTrue(all(done_n))
        self.assertTrue(info["episode_terminal"])
        self.assertEqual(info["done_reason"], "time_limit")
        self.assertEqual(world._episode_done_reason, "time_limit")
        self.assertTrue(world._episode_terminal)


if __name__ == "__main__":
    unittest.main()
