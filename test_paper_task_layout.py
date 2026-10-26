#!/usr/bin/env python3
"""Contract tests for versioned paper task layouts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from paper_task_layout import (
    TaskLayoutError,
    atomic_write_json,
    build_task_layout,
    reference_path_plan,
    reference_path_metrics,
    validate_distance_bounds,
    validate_task_layout_payload,
)


class PaperTaskLayoutTests(unittest.TestCase):
    def setUp(self):
        self.starts = np.asarray(
            [[20.0, 20.0, 12.0], [25.0, 25.0, 12.0], [30.0, 20.0, 12.0]],
            dtype=np.float32,
        )
        self.goals = np.asarray(
            [[100.0, 20.0, 25.0], [105.0, 25.0, 25.0], [110.0, 20.0, 25.0]],
            dtype=np.float32,
        )

    def test_distance_contract_is_per_agent_and_horizontal(self):
        distances = validate_distance_bounds(
            self.starts,
            self.goals,
            min_distance=75.0,
            max_distance=120.0,
        )
        np.testing.assert_allclose(distances, [80.0, 80.0, 80.0])

    def test_out_of_contract_layout_fails_loudly(self):
        invalid_goals = self.goals.copy()
        invalid_goals[1, 0] = 190.0
        with self.assertRaisesRegex(TaskLayoutError, "agent_1"):
            validate_distance_bounds(
                self.starts,
                invalid_goals,
                min_distance=75.0,
                max_distance=120.0,
            )

    def test_task_payload_is_deterministic_hashed_and_self_validating(self):
        kwargs = dict(
            agent_starts=self.starts,
            agent_goals=self.goals,
            central_goal=[105.0, 20.0, 25.0],
            map_size=200.0,
            min_start_goal_dist=75.0,
            max_start_goal_dist=120.0,
            terrain=np.zeros((50, 50), dtype=np.float32),
            obstacles=[],
            terrain_seed=88,
            terrain_variant_seed=901,
            obstacle_seed=902,
            position_seed=903,
            episode_index=7,
            env_id=2,
        )
        first = build_task_layout(**kwargs)
        second = build_task_layout(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(validate_task_layout_payload(first), [])
        self.assertEqual(len(first["task_id"]), 24)
        self.assertTrue(first["metrics"]["all_reference_feasible"])
        self.assertEqual(len(first["metrics"]["reference_paths"]), 3)

        different_bookkeeping = build_task_layout(
            **{
                **kwargs,
                "episode_index": 99,
                "env_id": 7,
                "source": "held_out_bank",
            }
        )
        self.assertEqual(first["task_id"], different_bookkeeping["task_id"])
        self.assertNotEqual(first["content_sha256"], different_bookkeeping["content_sha256"])

        tampered = json.loads(json.dumps(first))
        tampered["agent_starts"][0][0] += 1.0
        self.assertIn("content_sha256 mismatch", validate_task_layout_payload(tampered))
        self.assertIn("task_id physical identity mismatch", validate_task_layout_payload(tampered))

    def test_reference_planner_rejects_disconnected_layout(self):
        metrics = reference_path_metrics(
            [5.0, 5.0, 10.0],
            [95.0, 95.0, 10.0],
            map_size=100.0,
            terrain=np.zeros((25, 25), dtype=np.float32),
            obstacles=[{"center": [50.0, 50.0, 10.0], "radius": 100.0}],
            grid_resolution=5.0,
            obstacle_inflation=1.0,
        )
        self.assertFalse(metrics.feasible)
        self.assertIsNone(metrics.path_length)

    def test_reference_corridor_uses_exact_endpoints_and_clear_edges(self):
        start = np.asarray([5.5, 5.5, 7.0], dtype=float)
        goal = np.asarray([24.5, 24.5, 12.0], dtype=float)
        obstacle = {"center": [15.0, 15.0, 10.0], "radius": 2.0}
        inflation = 3.0
        plan = reference_path_plan(
            start,
            goal,
            map_size=30.0,
            terrain=np.zeros((30, 30), dtype=np.float32),
            obstacles=[obstacle],
            grid_resolution=5.0,
            obstacle_inflation=inflation,
        )
        self.assertTrue(plan.metrics.feasible)
        points = np.asarray(plan.waypoints, dtype=float)
        np.testing.assert_allclose(points[0, :2], start[:2])
        np.testing.assert_allclose(points[-1, :2], goal[:2])
        center = np.asarray(obstacle["center"][:2], dtype=float)
        required = obstacle["radius"] + inflation
        segment_surface_clearances = []
        for segment_start, segment_end in zip(points[:-1], points[1:]):
            edge = segment_end[:2] - segment_start[:2]
            fraction = np.clip(
                np.dot(center - segment_start[:2], edge)
                / max(float(np.dot(edge, edge)), 1e-12),
                0.0,
                1.0,
            )
            closest = segment_start[:2] + fraction * edge
            self.assertGreater(np.linalg.norm(center - closest), required)
            segment_surface_clearances.append(
                float(np.linalg.norm(center - closest) - obstacle["radius"])
            )
        self.assertAlmostEqual(
            plan.metrics.minimum_obstacle_clearance,
            min(segment_surface_clearances),
            places=12,
        )

    def test_reference_metrics_include_entry_and_exit_vertical_motion(self):
        plan = reference_path_plan(
            [5.0, 5.0, 2.0],
            [25.0, 5.0, 3.0],
            map_size=30.0,
            terrain=np.zeros((30, 30), dtype=np.float32),
            obstacles=[],
            grid_resolution=5.0,
            surface_clearance=10.0,
        )
        points = np.asarray(plan.waypoints, dtype=float)
        np.testing.assert_allclose(points[0], [5.0, 5.0, 2.0])
        np.testing.assert_allclose(points[-1], [25.0, 5.0, 3.0])
        self.assertAlmostEqual(np.max(points[:, 2]), 10.0)
        self.assertAlmostEqual(plan.metrics.maximum_climb, 8.0)
        self.assertAlmostEqual(plan.metrics.path_length, 35.0)

    def test_atomic_writer_round_trips_valid_payload(self):
        payload = build_task_layout(
            agent_starts=self.starts,
            agent_goals=self.goals,
            central_goal=[105.0, 20.0, 25.0],
            map_size=200.0,
            min_start_goal_dist=75.0,
            max_start_goal_dist=120.0,
            terrain=np.zeros((20, 20), dtype=np.float32),
            obstacles=[],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = atomic_write_json(Path(tmp_dir) / "task.json", payload)
            with output.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        self.assertEqual(loaded, payload)
        self.assertEqual(validate_task_layout_payload(loaded), [])


if __name__ == "__main__":
    unittest.main()
