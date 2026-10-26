import os
import unittest

import numpy as np

os.environ.setdefault("QUIET_OUTPUT", "1")
os.environ.setdefault("SUPPRESS_TERRAIN_OUTPUT", "1")
os.environ.setdefault("SUPPRESS_REWARD_CONFIG_OUTPUT", "1")
os.environ.setdefault("ENABLE_ROLE_SHUFFLE", "0")
os.environ.setdefault("INIT_VEL_JITTER_MAX", "0")

from multiagent.environment import MultiAgentEnv
from multiagent.scenarios.paper3d_terrain_energy import Scenario
from multiagent.scenarios.paper3d_terrain_vectorized import Scenario as VectorizedScenario
from multiagent.scenarios.paper3d_terrain_weighted import Scenario as WeightedScenario


class PaperRewardContractTests(unittest.TestCase):
    def _environment(self, episode_length=10, scenario_class=Scenario):
        scenario = scenario_class(
            seed=67,
            terrain_complexity_level=1,
            paper_task_num_obstacles=0,
            paper_task_contract_enabled=True,
            paper_task_contract_strict=True,
            paper_task_require_reference_feasible=True,
            min_start_goal_dist=30.0,
            max_start_goal_dist=50.0,
            hard_deadline_terminal=True,
            paper_terminal_on_collision=True,
            paper_reward_profile="paper_v1",
            include_time_to_go_observation=True,
            use_dynamic_obstacles=False,
            enable_arw=False,
        )
        world = scenario.make_world()
        world.episode_length = int(episode_length)
        env = MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
            done_callback=scenario.is_done,
            shared_viewer=False,
        )
        env.reset()
        return scenario, world, env

    @staticmethod
    def _zero_actions(world):
        return [np.zeros(7, dtype=np.float32) for _ in world.policy_agents]

    def test_transition_progress_orders_toward_standing_and_away(self):
        scenario, world, _ = self._environment()
        agent = world.policy_agents[0]
        distance = float(np.linalg.norm(agent.state.p_pos - agent.goal_a.state.p_pos))
        agent._paper_reward_initial_goal_dist = distance

        def reward_with_previous(previous, step):
            world.current_step = step
            scenario._paper_prepared_step_key = None
            world._episode_success = False
            world._paper_any_collision = False
            world._paper_any_out_of_bounds = False
            agent._paper_reward_last_goal_dist = previous
            return float(scenario.reward(agent, world))

        toward = reward_with_previous(distance + 1.0, 1)
        standing = reward_with_previous(distance, 2)
        away = reward_with_previous(distance - 1.0, 3)
        self.assertGreater(toward, standing)
        self.assertGreater(standing, away)

    def test_nonterminal_hover_cannot_farm_positive_state_reward(self):
        scenario, world, _ = self._environment()
        agent = world.policy_agents[0]
        distance = float(np.linalg.norm(agent.state.p_pos - agent.goal_a.state.p_pos))
        agent._paper_reward_initial_goal_dist = distance
        values = []
        for step in (1, 2):
            world.current_step = step
            scenario._paper_prepared_step_key = None
            agent._paper_reward_last_goal_dist = distance
            values.append(float(scenario.reward(agent, world)))
        self.assertLess(values[0], 0.0)
        self.assertLess(values[1], 0.0)
        self.assertAlmostEqual(values[0], values[1], places=9)

    def test_team_success_is_terminal_and_positive(self):
        _, world, env = self._environment()
        for agent in world.policy_agents:
            agent.state.p_pos = agent.goal_a.state.p_pos.copy()
            agent.state.p_vel[:] = 0.0
        _, rewards, dones, info = env.step(self._zero_actions(world))
        self.assertTrue(all(dones))
        self.assertTrue(info["episode_terminal"])
        self.assertEqual(info["done_reason"], "team_success")
        self.assertTrue(all(float(value) > 100.0 for value in rewards))

    def test_collision_terminates_team_and_penalizes_every_agent(self):
        scenario, world, env = self._environment()
        agent = world.policy_agents[0]
        terrain_height = scenario.get_terrain_height(
            agent.state.p_pos[0], agent.state.p_pos[1]
        )
        agent.state.p_pos[2] = float(terrain_height + agent.size)
        _, rewards, dones, info = env.step(self._zero_actions(world))
        self.assertTrue(all(dones))
        self.assertTrue(info["episode_terminal"])
        self.assertEqual(info["done_reason"], "collision")
        self.assertTrue(all(float(value) < -90.0 for value in rewards))
        self.assertTrue(agent._episode_has_collision)

    def test_deadline_is_terminal_failure(self):
        _, world, env = self._environment(episode_length=1)
        _, rewards, dones, info = env.step(self._zero_actions(world))
        self.assertTrue(all(dones))
        self.assertEqual(info["done_reason"], "time_limit")
        self.assertTrue(all(float(value) < -49.0 for value in rewards))

    def test_energy_is_executed_control_integral_and_accumulates_once(self):
        _, world, env = self._environment()
        actions = self._zero_actions(world)
        actions[0][0] = 1.0
        env.step(actions)
        expected = 2.0 * float(world.dt)
        self.assertAlmostEqual(world.policy_agents[0].energy_consumed, expected, places=9)
        self.assertEqual(
            world.policy_agents[0].energy_consumed_unit,
            "normalized_energy_seconds",
        )
        self.assertAlmostEqual(
            world.policy_agents[1].energy_consumed, float(world.dt), places=9
        )

    def test_all_three_scenario_backends_share_paper_terminal_reward(self):
        for scenario_class in (Scenario, WeightedScenario, VectorizedScenario):
            with self.subTest(scenario=scenario_class.__module__):
                _, world, env = self._environment(
                    episode_length=1, scenario_class=scenario_class
                )
                _, rewards, dones, info = env.step(self._zero_actions(world))
                self.assertTrue(all(dones))
                self.assertEqual(info["done_reason"], "time_limit")
                self.assertTrue(all(float(value) < -49.0 for value in rewards))


if __name__ == "__main__":
    unittest.main()
