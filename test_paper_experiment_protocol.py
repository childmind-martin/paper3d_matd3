import unittest

from paper_curriculum import build_paper_curriculum
from paper_experiment_protocol import (
    FORMAL_EVALUATION_DEVICE_REQUIREMENT,
    FORMAL_CHECKPOINT_POLICY,
    FORMAL_INTERRUPTION_POLICY,
    FORMAL_SELECTION_DIFFICULTY,
    _FORMAL_COMMON_TRAINING_HYPERPARAMETERS,
    _FORMAL_EXPERIMENT_SOURCE_LOGICAL_NAMES,
    _REPORTING_CONTRACT,
    _build_stage_iteration_budget,
    _experiment_source_bundle_descriptor,
    available_method_ids,
    method_spec,
    runtime_method_environment,
)
from paper_experiment_schema import content_sha256


class PaperExperimentProtocolTests(unittest.TestCase):
    def test_formal_evaluation_device_contract_is_cpu_deterministic(self):
        self.assertEqual(
            FORMAL_EVALUATION_DEVICE_REQUIREMENT["policy"],
            "tensorflow_cpu_deterministic_v1",
        )
        self.assertEqual(
            FORMAL_EVALUATION_DEVICE_REQUIREMENT["maximum_physical_gpus"],
            0,
        )

    def test_formal_interruption_policy_requires_a_fresh_complete_seed(self):
        self.assertEqual(
            "restart_train_seed_from_scratch_v1",
            FORMAL_INTERRUPTION_POLICY,
        )

    def test_method_matrix_contains_proposed_and_factor_isolated_ablations(self):
        self.assertEqual(
            available_method_ids(),
            ["EC-MATD3-APF", "EC-MATD3", "DualQ-MATD3-APF", "MADDPG-APF", "RawOnly-MATD3-APF"],
        )
        proposed = method_spec("EC-MATD3-APF")
        no_apf = method_spec("EC-MATD3")
        legacy_semantics = method_spec("DualQ-MATD3-APF")
        maddpg = method_spec("MADDPG-APF")
        self.assertEqual(
            proposed["environment"]["MATD3_ACTION_SEMANTICS_MODE"],
            "executed_control",
        )
        self.assertEqual(
            no_apf["environment"]["ENABLE_ACTION_CORRECTION"], "false"
        )
        self.assertEqual(no_apf["paper_role"], "capacity_matched_apf_ablation")
        self.assertEqual(
            no_apf["environment"]["USE_FR_FEATURE"],
            proposed["environment"]["USE_FR_FEATURE"],
        )
        self.assertEqual(
            no_apf["environment"]["USE_PF_FEATURE"],
            proposed["environment"]["USE_PF_FEATURE"],
        )
        self.assertEqual(no_apf["environment"]["ACTION_FORCE_RATIO"], "0.0")
        self.assertEqual(
            no_apf["environment"]["ACTION_FORCE_RATIO_SCHEDULE_PCT"],
            "DISABLED",
        )
        self.assertEqual(
            legacy_semantics["environment"]["MATD3_ACTION_SEMANTICS_MODE"],
            "dual",
        )
        self.assertEqual(
            maddpg["environment"]["MADDPG_ACTION_SEMANTICS_MODE"],
            "executed_control",
        )
        self.assertEqual(
            maddpg["environment"]["ACTOR_UPDATE_DELAY"],
            "1",
        )
        self.assertEqual(
            maddpg["paper_role"],
            "algorithm_baseline",
        )
        self.assertEqual(proposed["algorithm"], no_apf["algorithm"])
        self.assertTrue(proposed["require_validation_adequacy"])
        self.assertFalse(no_apf["require_validation_adequacy"])

    def test_disabled_force_ratio_schedule_resolves_to_empty_runtime_value(self):
        no_apf = method_spec("EC-MATD3")
        runtime = runtime_method_environment(no_apf)
        self.assertEqual(
            no_apf["environment"]["ACTION_FORCE_RATIO_SCHEDULE_PCT"],
            "DISABLED",
        )
        self.assertEqual(runtime["ACTION_FORCE_RATIO_SCHEDULE_PCT"], "")

    def test_common_training_hyperparameters_are_explicitly_frozen(self):
        contract = _FORMAL_COMMON_TRAINING_HYPERPARAMETERS
        flags = contract["launcher_flags"]
        self.assertEqual(
            contract["policy_id"],
            "explicit_common_launcher_and_environment_v2",
        )
        self.assertEqual(flags["--learning-rate-actor"], "0.0003")
        self.assertEqual(flags["--learning-rate-critic"], "0.0005")
        self.assertEqual(flags["--update-rate"], "40")
        self.assertEqual(flags["--buffer-size"], "450000")
        self.assertEqual(flags["--actor-hidden"], "256,256,256")
        self.assertEqual(flags["--episode-length"], "2800")
        self.assertEqual(flags["--gamma"], "0.99")
        self.assertEqual(flags["--lr-decay-enabled"], "1")
        self.assertEqual(flags["--lr-decay-steps"], "30000")
        self.assertEqual(
            flags["--obstacle-observation-mode"], "risk_lite_v2"
        )
        self.assertEqual(
            contract["environment_variables"]["ADAPTIVE_PATIENCE"],
            "60",
        )
        self.assertEqual(
            contract["environment_variables"][
                "ADAPTIVE_CRITIC_LR_DECAY"
            ],
            "1.0",
        )
        self.assertEqual(
            contract["discount_time_scale"],
            {
                "gamma": 0.99,
                "simulation_dt_seconds": 0.08,
                "effective_horizon_steps": 100,
                "effective_horizon_seconds": 8.0,
            },
        )

    def test_formal_experiment_source_bundle_is_complete_and_content_hashed(self):
        descriptor = _experiment_source_bundle_descriptor()
        self.assertEqual(descriptor["policy"], "immutable_source_bundle_v1")
        self.assertEqual(
            [item["logical_name"] for item in descriptor["artifacts"]],
            list(_FORMAL_EXPERIMENT_SOURCE_LOGICAL_NAMES),
        )
        self.assertEqual(
            descriptor["content_sha256"],
            content_sha256(
                {
                    key: value
                    for key, value in descriptor.items()
                    if key != "content_sha256"
                }
            ),
        )

    def test_official_results_use_one_final_c3_selected_checkpoint(self):
        self.assertEqual(FORMAL_SELECTION_DIFFICULTY, "C3")
        self.assertEqual(
            FORMAL_CHECKPOINT_POLICY,
            "one_validation_C3_selected_checkpoint_for_all_splits_v1",
        )

    def test_returned_method_spec_is_a_deep_copy(self):
        first = method_spec("EC-MATD3-APF")
        first["environment"]["ACTION_FORCE_RATIO"] = "99"
        self.assertEqual(
            method_spec("EC-MATD3-APF")["environment"]["ACTION_FORCE_RATIO"],
            "0.50",
        )

    def test_fixed_stage_budget_is_complete_and_c3_dominant(self):
        budget = _build_stage_iteration_budget(800)
        self.assertEqual(budget, {"C0": 96, "C1": 128, "C2": 192, "C3": 384})
        self.assertEqual(sum(budget.values()), 800)

    def test_default_budget_predeclares_multiple_candidates_per_stage(self):
        budget = _build_stage_iteration_budget(800)
        intervals = {
            stage["difficulty_bin"]: stage[
                "candidate_checkpoint_interval_trajectories"
            ]
            for stage in build_paper_curriculum(67)
        }
        counts = {
            difficulty: (budget[difficulty] * 8 + interval - 1) // interval
            for difficulty, interval in intervals.items()
        }
        self.assertEqual(counts, {"C0": 6, "C1": 6, "C2": 5, "C3": 5})

    def test_default_budget_traverses_each_256_task_split_in_complete_cycles(self):
        budget = _build_stage_iteration_budget(800)
        trajectories = {
            difficulty: iterations * 8
            for difficulty, iterations in budget.items()
        }
        self.assertEqual(
            {
                difficulty: count // 256
                for difficulty, count in trajectories.items()
            },
            {"C0": 3, "C1": 4, "C2": 6, "C3": 12},
        )
        self.assertTrue(all(count % 256 == 0 for count in trajectories.values()))

    def test_reporting_predeclares_actual_apf_mechanism_diagnostics(self):
        diagnostics = _REPORTING_CONTRACT["mechanism_diagnostics"]
        self.assertEqual(diagnostics["episode_schema"], "paper_episode_v2")
        self.assertTrue(diagnostics["require_complete_action_step_coverage"])
        self.assertIn("mean_action_intervention_norm", diagnostics["fields"])
        self.assertIn("apf_intervention_step_fraction", diagnostics["fields"])

    def test_reporting_has_one_predeclared_confirmatory_contrast(self):
        analysis = _REPORTING_CONTRACT["confirmatory_analysis"]
        self.assertEqual(analysis["split"], "test_C3")
        self.assertEqual(analysis["proposed_method"], "EC-MATD3-APF")
        self.assertEqual(analysis["baseline_method"], "EC-MATD3")
        self.assertEqual(analysis["metric"], "team_success_rate")
        self.assertEqual(
            analysis["multiplicity_policy"],
            "single_predeclared_confirmatory_contrast",
        )

if __name__ == "__main__":
    unittest.main()
