# Appendix H.3 APF Sanity Evidence

**Evidence role:** supplementary sanity/reference evidence.

Source: `ablation_experiments/batch_20260403_132242/plots/summary_20260403_221128.json`

This output summarizes the APF-only/action-only/action-plus-APF sanity evidence when source files are available.

It is not used to establish the necessity of dual-semantic replay, critic construction, target reconstruction, or separated-gradient routing.

It should not be presented as equal to Table 5 or Table 6.

| Method | Design role | Train team success | Avg reward | Final reward | Max reward | Eval team success | Eval collision-free |
| --- | --- | --- | --- | --- | --- | --- | --- |
| APF Learnable | APF action only, parameters learned via Actor network | 0.000 | -169173.27 | -148993.86 | -113507.20 | NA | NA |
| Action+APF Fusion | Network action + learnable APF correction (default config) | 0.267 | 101970.61 | 18053.26 | 888276.15 | NA | NA |
| Action Only | Network action only, no APF correction | 0.000 | -63082.79 | 3544.29 | 6510.74 | NA | NA |
| APF Traditional | APF action only, fixed parameters (base values, no learning) | 0.000 | -172390.88 | -151479.10 | -131437.67 | NA | NA |
