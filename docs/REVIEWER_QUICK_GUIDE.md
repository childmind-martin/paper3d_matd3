# Reviewer Quick Guide

This guide is a short path for checking the artifact without running expensive full retraining.

## 1. Read the Paper Claim

The paper claim is a learning-interface claim, not a universal MARL ranking. APF-corrected execution creates distinct raw/corrected action roles; the proposed framework preserves those roles in replay, critic evaluation, target-side construction, and actor-gradient routing.

The APF-fusion/APF-only results are supplementary sanity/reference evidence only. The main mechanism evidence is the focused semantic ablation in Table 5.

## 2. Inspect the Manifest

Start with:

```bash
python scripts/print_result_manifest.py
```

Then inspect `RESULTS_MANIFEST.json` for the exact table-to-artifact mapping:

- Table 5: focused semantic-ablation mechanism evidence.
- Table 6: Level-2 checkpoint-FR primary deployment evidence.
- Appendix Table H.3: APF-fusion/APF-only supplementary sanity/reference evidence.

## 3. Rebuild Table 5 and Table 6 Summaries

These commands rebuild processed Markdown summaries from existing CSV/JSON artifacts. They are not full retraining commands.

Table 5:

```bash
python scripts/rebuild_table5_semantic_ablation.py
```

Table 6:

```bash
python scripts/rebuild_table6_level2_checkpoint_fr.py
```

The rebuilt Markdown files are written under `artifacts/rebuilt_tables/`. If source logs are included and a deeper check is needed, use the summary-builder hints listed in `RESULTS_MANIFEST.json`.

## 4. Optionally Run a Smoke Test

After setting up the environment from `REPRODUCE_ENVIRONMENT.md`, run:

```bash
bash scripts/smoke_test_artifact.sh
```

This checks the reviewer artifact layout and prints the optional preflight command when available. It does not verify the reported results or run full training.

## 5. Understand the Simulation Boundary

The simulator is a guidance-layer benchmark with a first-order attitude-response lag. It is more restrictive than an instantaneous point-mass update, but it is not a full high-fidelity quadrotor simulator. The paper claim is bounded to this reported benchmark and evidence hierarchy.
