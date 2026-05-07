# Rebuilt Table 5 Semantic Ablation

**Evidence role:** core mechanism evidence.

Source: `level2_dual_semantics_partial_summary_20260427_141546/official_eval_multiseed_aggregated.csv`

This table is rebuilt from processed CSV/JSON artifacts. It does not modify source result files.

| Method | n | Team success | Dense reward | Collision-free | Final distance | Total collisions |
| --- | --- | --- | --- | --- | --- | --- |
| Full Dual-Semantic | 2 | 0.250 +/- 0.200 | 30399.34 +/- 4935.48 | 0.425 +/- 0.075 | 12.768 +/- 5.705 | 92.000 +/- 58.350 |
| Collapsed Replay | 2 | 0.000 +/- 0.000 | -142495.84 +/- 18447.77 | 0.000 +/- 0.000 | 27.477 +/- 17.666 | 812.000 +/- 212.300 |
| No Corrected Target Recon | 2 | 0.250 +/- 0.150 | 16578.23 +/- 17896.40 | 0.325 +/- 0.075 | 5.540 +/- 3.243 | 112.525 +/- 28.175 |
