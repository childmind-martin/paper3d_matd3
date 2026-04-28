# GPT Data Index

This repository contains lightweight evaluation artifacts that are suitable for
GitHub browsing or GPT retrieval. The files below are intentionally small CSV,
JSON, and PNG outputs.

Remote repository:

```text
https://github.com/childmind-martin/paper3d_matd3
```

Raw URL pattern after pushing to `main`:

```text
https://raw.githubusercontent.com/childmind-martin/paper3d_matd3/main/<relative-path>
```

## Level 2 Official Multiseed Summary

Directory:

```text
level2_partial_summary_20260427_141546/
```

Artifacts:

| File | Purpose |
| --- | --- |
| `official_eval_multiseed_aggregated.csv` | Cross-seed aggregate metrics by algorithm. Best entry point for comparison. |
| `official_eval_multiseed_all_runs.csv` | Per-seed official evaluation rows. Use this for run-level checks. |
| `official_eval_multiseed_summary.json` | Structured multiseed summary with detailed metrics. |
| `official_eval_cross_algo_summary.csv` | Algorithm comparison table derived from official evaluation runs. |
| `official_eval_cross_algo_summary.json` | Structured algorithm comparison summary. |
| `official_eval_multiseed_dashboard.png` | Dashboard figure for multiseed official evaluation. |
| `official_eval_cross_algo_summary.png` | Cross-algorithm comparison figure. |

Primary raw links:

```text
https://raw.githubusercontent.com/childmind-martin/paper3d_matd3/main/level2_partial_summary_20260427_141546/official_eval_multiseed_aggregated.csv
https://raw.githubusercontent.com/childmind-martin/paper3d_matd3/main/level2_partial_summary_20260427_141546/official_eval_multiseed_summary.json
https://raw.githubusercontent.com/childmind-martin/paper3d_matd3/main/level2_partial_summary_20260427_141546/official_eval_multiseed_dashboard.png
```

## Level 2 Dual-Semantics Ablation Summary

Directory:

```text
level2_dual_semantics_partial_summary_20260427_141546/
```

Artifacts:

| File | Purpose |
| --- | --- |
| `official_eval_multiseed_aggregated.csv` | Cross-seed aggregate metrics for dual-semantics ablations. |
| `official_eval_multiseed_all_runs.csv` | Per-seed dual-semantics ablation evaluation rows. |
| `official_eval_multiseed_summary.json` | Structured multiseed ablation summary. |
| `official_eval_cross_algo_summary.csv` | Cross-ablation comparison table. |
| `official_eval_cross_algo_summary.json` | Structured cross-ablation comparison summary. |
| `official_eval_multiseed_dashboard.png` | Dashboard figure for dual-semantics ablation runs. |
| `official_eval_cross_algo_summary.png` | Cross-ablation comparison figure. |

Primary raw links:

```text
https://raw.githubusercontent.com/childmind-martin/paper3d_matd3/main/level2_dual_semantics_partial_summary_20260427_141546/official_eval_multiseed_aggregated.csv
https://raw.githubusercontent.com/childmind-martin/paper3d_matd3/main/level2_dual_semantics_partial_summary_20260427_141546/official_eval_multiseed_summary.json
https://raw.githubusercontent.com/childmind-martin/paper3d_matd3/main/level2_dual_semantics_partial_summary_20260427_141546/official_eval_multiseed_dashboard.png
```

## Suggested GPT Prompt

```text
Please read the GitHub raw CSV/JSON files listed in GPT_DATA_INDEX.md and
summarize the Level 2 official multiseed evaluation. Compare algorithms by
team_success_rate_mean, avg_reward_mean, avg_collision_count_mean,
collision_free_rate_mean, and avg_team_final_goal_distance_mean. Mention which
files you used.
```

## Notes

- CSV files are the easiest entry point for tabular analysis.
- JSON files preserve structured detail for precise metric lookup.
- PNG files are included for direct visual inspection in GitHub.
- Some rows contain local `results_path` values from the machine that generated
  the artifacts; those paths are provenance hints, not GitHub-accessible files.
