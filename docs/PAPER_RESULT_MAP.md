# Paper Result Map

This document maps the paper's result items to their evidence role and expected artifact locations. It is a human-readable companion to `RESULTS_MANIFEST.json`.

## Summary

| Paper item | Evidence role | Main evidence? | Statistical superiority evidence? |
| --- | --- | --- | --- |
| Table 5 | Semantic-ablation mechanism evidence | Yes | No |
| Table 6 | Level-2 checkpoint-FR deployment evidence | Yes | No |
| Fig. 5 | Diagnostic visualization for Level-2 | No | No |
| Table 7 | Level-1 fixed-map support | No | No |
| Table 8 | Level-3 terrain-family support | No | No |
| Appendix H.3 | APF-fusion sanity/reference evidence | No | No |

The paper reports bounded, mechanism-centered evidence under the stated protocols. None of these items should be read as statistically significant universal superiority over all MARL methods.

Some expected source files retain `official_eval` in historical filenames.
These names refer to the repository's checkpoint-FR evaluation outputs and do
not indicate an external benchmark authority or a stronger evidence hierarchy.

## Table 5: Semantic Ablation Evidence

- **Question answered:** Does preserving raw/corrected action roles after APF-corrected execution matter for the dual-semantic learning mechanism?
- **Evidence status:** Main evidence; core mechanism evidence.
- **Expected source directory:** `level2_dual_semantics_partial_summary_20260427_141546/`
- **Expected source files:**
  - `official_eval_multiseed_all_runs.csv`
  - `official_eval_multiseed_aggregated.csv`
  - `official_eval_multiseed_summary.json`
  - `official_eval_multiseed_dashboard.png`
  - `official_eval_cross_algo_summary.csv`
  - `official_eval_cross_algo_summary.json`
  - `official_eval_cross_algo_summary.png`
- **Interpret as statistical superiority evidence?** No. It is mechanism-centered ablation evidence for the raw/corrected semantic split, not a universal ranking.

## Table 6: Level-2 Checkpoint-FR Evidence

- **Question answered:** How do the reported algorithm families behave under the primary Level-2 checkpoint-FR deployment protocol with stochastic obstacles?
- **Evidence status:** Main evidence; primary deployment evidence.
- **Expected source directory:** `diagnostics/level2_official_eval_multiseed_summary_20260428_163815/`
- **Expected source files:**
  - `official_eval_multiseed_all_runs_model_fr.csv`
  - `official_eval_multiseed_aggregated_model_fr.csv`
  - `official_eval_multiseed_summary_model_fr.json`
  - `official_eval_multiseed_dashboard_model_fr.png`
  - `evaluation_results.json` files referenced by the all-runs CSV
- **Partial-arrival source:** Any-agent and two-agent arrival are computed from `episode_details[].agent_success_flags` in the referenced `evaluation_results.json` files.
- **Interpret as statistical superiority evidence?** No. It supports deployment-side interpretation under the reported protocol and should be read together with Table 5.

## Fig. 5: Level-2 Diagnostic Visualization

- **Question answered:** What visual dashboard accompanies the Level-2 checkpoint-FR numerical summary?
- **Evidence status:** Supporting diagnostic visualization for the primary Level-2 result.
- **Expected source directory:** `diagnostics/level2_official_eval_multiseed_summary_20260428_163815/`
- **Expected source files:**
  - `official_eval_multiseed_dashboard_model_fr.png`
  - `official_eval_multiseed_summary_model_fr.json`
  - `official_eval_multiseed_aggregated_model_fr.csv`
- **Interpret as statistical superiority evidence?** No. The figure is a visual companion; numerical claims should be checked against Table 6 and the CSV/JSON summaries.

## Table 7: Level-1 Fixed-Map Support

- **Question answered:** Does the dual-semantic scaffold behave sensibly in a lower-variability fixed-map setting?
- **Evidence status:** Supporting low-variability validation.
- **Expected source directory:** `ablation_experiments/multi_seed_groupA_20260412_205350/`
- **Expected source files:**
  - `plots/summary_20260414_221440.json`
  - `plots/multi_seed_mean_ablation_comparison_20260414_221440.png`
  - `plots/multi_seed_post_eval_summary_20260414_221440.png`
  - `results/multi_seed_audit_20260414_221436.json`
  - `results/latest_multi_seed_audit.json`
  - `config.json`
- **Interpret as statistical superiority evidence?** No. It is a lower-variability support case and is not used to declare a fixed objective ranking.

## Table 8: Level-3 Terrain-Family Support

- **Question answered:** How does objective-family behavior change under semi-random terrain-family stress?
- **Evidence status:** Supporting terrain-family analysis.
- **Expected source directory:** `ablation_experiments/multi_seed_groupB_20260406_230829/`
- **Expected source files:**
  - `plots/summary_20260414_234950.json`
  - `plots/multi_seed_mean_ablation_comparison_20260414_234950.png`
  - `plots/matd3_trio_train_reward_success_20260414_234950.png`
  - `results/multi_seed_audit_20260414_234945.json`
  - `results/latest_multi_seed_audit.json`
  - `config.json`
- **Interpret as statistical superiority evidence?** No. It supports regime-dependent interpretation and is not a second main benchmark universe.

## Appendix H.3: APF-Fusion Sanity/Reference Evidence

- **Question answered:** In a controlled APF sanity check, do learned motion proposals and APF correction provide complementary signals compared with learned-action-only or APF-only modes?
- **Evidence status:** Supplementary sanity/reference evidence only.
- **Expected source directory:** `ablation_experiments/batch_20260403_132242/`
- **Expected source files:**
  - `config.json`
  - `plots/summary_20260403_221128.json`
  - `plots/latest_summary.json`
  - `plots/reward_comparison_20260403_221128.png`
  - `plots/success_collision_clearance_comparison_20260403_221128.png`
  - `plots/success_rate_and_clearance_comparison_20260403_221128.png`
- **Interpret as statistical superiority evidence?** No. It is not used to establish the necessity of dual-semantic replay, critic construction, target reconstruction, or separated-gradient routing.
