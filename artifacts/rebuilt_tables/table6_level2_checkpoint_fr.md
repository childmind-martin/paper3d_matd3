# Rebuilt Table 6 Level-2 Checkpoint-FR

**Evidence role:** primary deployment evidence.

Source: `diagnostics/level2_official_eval_multiseed_summary_20260428_163815/official_eval_multiseed_aggregated_model_fr.csv`

Any-agent and two-agent arrival are recomputed from `episode_details[].agent_success_flags` in the `evaluation_results.json` files referenced by the all-runs CSV.

This table is rebuilt from processed CSV/JSON artifacts. It does not modify source result files.

Interpret this as a matched-protocol deployment diagnostic, not as a universal MARL ranking.

Partial-arrival all-runs source: `diagnostics/level2_official_eval_multiseed_summary_20260428_163815/official_eval_multiseed_all_runs_model_fr.csv`

| Method | n | Team success | Any-agent arrival | Two-agent arrival | Dense reward | Collision-free | Final distance | Total collisions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MATD3 Single-Q | 3 | 0.000 +/- 0.000 | 8.333 | 0.000 | 38727.06 +/- 15500.73 | 68.333 +/- 4.714 | 329.423 +/- 303.868 | 44.283 +/- 21.254 |
| MATD3 Dual-Q | 3 | 13.333 +/- 15.456 | 76.667 | 31.667 | 110742.19 +/- 19714.18 | 83.333 +/- 12.472 | 23.475 +/- 8.966 | 10.533 +/- 7.978 |
| MATD3 Sep-Grad | 3 | 28.333 +/- 16.997 | 95.000 | 48.333 | 35672.96 +/- 37459.79 | 56.667 +/- 4.714 | 35.966 +/- 42.563 | 126.617 +/- 143.442 |
| MATD3 Hybrid alpha=0.80 | 3 | 0.000 +/- 0.000 | 70.000 | 13.333 | 45972.26 +/- 46801.06 | 20.000 +/- 7.071 | 60.990 +/- 38.763 | 174.567 +/- 96.579 |
| MATD3 Hybrid alpha=0.20 | 3 | 28.333 +/- 36.591 | 71.667 | 43.333 | 31450.74 +/- 26594.47 | 45.000 +/- 33.417 | 56.162 +/- 47.159 | 139.300 +/- 132.465 |
| MADDPG Baseline | 3 | 0.000 +/- 0.000 | 0.000 | 0.000 | -267636.22 +/- 164050.97 | 78.333 +/- 16.499 | 3374.60 +/- 697.452 | 5.650 +/- 4.007 |
| MADDPG Dual-Q | 3 | 0.000 +/- 0.000 | 8.333 | 0.000 | -12424.52 +/- 66459.82 | 73.333 +/- 9.428 | 731.315 +/- 529.252 | 59.233 +/- 45.360 |
| MADDPG Sep-Grad | 3 | 0.000 +/- 0.000 | 0.000 | 0.000 | -142941.64 +/- 104284.52 | 60.000 +/- 21.602 | 1228.30 +/- 548.043 | 111.983 +/- 75.199 |
| MAPPO Baseline | 3 | 0.000 +/- 0.000 | 1.667 | 0.000 | 5228.99 +/- 47072.48 | 6.667 +/- 9.428 | 1407.13 +/- 1077.05 | 88.883 +/- 49.834 |
| MAPPO Fusion-Only | 3 | 0.000 +/- 0.000 | 0.000 | 0.000 | -89536.78 +/- 115972.35 | 40.000 +/- 29.439 | 1261.20 +/- 1134.07 | 39.650 +/- 43.911 |
| MAPPO Sep-Grad | 3 | 1.667 +/- 2.357 | 33.333 | 20.000 | -402385.62 +/- 525202.71 | 68.333 +/- 44.783 | 2756.02 +/- 2018.84 | 22.233 +/- 31.443 |
