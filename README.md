# MATD3 Dual-Semantic APF Reviewer Artifact

This repository is the reviewer-facing artifact for the paper:

**A MATD3-Centered Dual-Semantic Framework with Artificial-Potential-Field Correction for Cooperative Multi-UAV Navigation**

## Main scientific claim

This repository supports a study of APF-corrected off-policy multi-agent reinforcement learning. The central issue is not whether APF alone or RL alone is better for path planning. Instead, APF-corrected execution creates two distinct action roles:

- the actor's raw motion proposal,
- the APF-corrected command executed by the environment.

The proposed dual-semantic framework preserves these roles through replay storage, critic evaluation, target construction, and actor-gradient routing.

The main mechanism evidence is the focused Level-2 semantic ablation. The Level-2 checkpoint-FR evaluation provides primary deployment evidence. APF-only/APF-fusion comparisons are retained as supplementary sanity/reference evidence only.

Submission code snapshot: `5c6cc625d8d9c572bacab114ef042aeea3f2f684`

## Task

The benchmark studies cooperative terrain-aware navigation with:

- three UAVs,
- complex 3-D terrain,
- stochastic obstacles in the main Level-2 deployment setting,
- local per-agent observations with centralized critics during training,
- strict all-agent coordinated arrival as the team-success criterion.

Partial arrival, final distance, collision-free rate, and collision burden are diagnostic metrics. They do not replace the strict team-success criterion.

## Main Mechanism

APF-corrected execution creates different action roles:

- the actor's raw motion proposal represents policy intention,
- the APF-modulation tail parameterizes the local correction law,
- the APF-corrected command is the action executed by the environment.

The proposed MATD3-centered framework preserves these roles through:

- dual-semantic replay records,
- a critic that evaluates raw-intention and corrected-execution channels,
- optional corrected target reconstruction for Bellman targets,
- separated/unified/hybrid actor-gradient routing.

The APF-fusion/APF-only results are not used as the main mechanism evidence. They are retained as supplementary sanity/reference evidence showing that learned motion proposals and APF correction are complementary. The main mechanism evidence is the focused semantic ablation that tests whether raw/corrected action roles should be preserved after APF-corrected execution.

## Evidence Hierarchy

The result manifest maps the paper tables to evidence roles, processed result files, plotting artifacts, and script hints.

- **Table 5**: core semantic-ablation mechanism evidence. This is the focused test of replay-level raw/corrected semantic preservation and target-side reconstruction.
- **Table 6**: primary Level-2 checkpoint-FR deployment evidence across contextual MATD3-, MADDPG-, and MAPPO-family variants.
- **Level-1 and Level-3**: supporting fixed-map and terrain-family analyses.
- **Appendix Table H.3**: APF-fusion/APF-only sanity/reference evidence only. It is not part of the main mechanism hierarchy.

## Quick Smoke Checks

Artifact-level check:

```bash
bash scripts/smoke_test_artifact.sh
```

Runtime/GPU environment smoke test, after installing dependencies:

```bash
python tools/smoke_test_eval_gpu.py \
  --matrix-size 512 \
  --iterations 1 \
  --batch-size 16 \
  --hold-seconds 0
```

The runtime smoke test only checks the TensorFlow/GPU execution path. It does not reproduce the paper tables.

## Result Manifest

Use `RESULTS_MANIFEST.json` as the authoritative map from paper tables to artifact files. It records:

- processed CSV/JSON summaries,
- result directories,
- evidence roles,
- plotting artifacts,
- script hints,
- notes on how each result should be interpreted.

## Environment

Environment setup and dependency details are in `REPRODUCE_ENVIRONMENT.md`. The default documented environment is Linux with Python 3.10 and TensorFlow 2.12. The project also includes `requirements.txt`, `setup_conda_env.sh`, and environment-check utilities.

## Rebuilding Processed Summaries

These commands rebuild reviewer-facing Markdown summaries from existing CSV/JSON artifacts. They do not perform full multi-seed retraining and do not modify source result files.

Table 5:

```bash
python scripts/rebuild_table5_semantic_ablation.py
```

Table 6:

```bash
python scripts/rebuild_table6_level2_checkpoint_fr.py
```

Appendix H.3 quick check:

```bash
python scripts/rebuild_appendix_h3_apf_sanity.py
```

The original summary-builder scripts listed in `RESULTS_MANIFEST.json` can be used for deeper checks when the corresponding source logs are available.

## Cost Boundary

Complete retraining is intentionally not the default reviewer path. Table 6 corresponds to 11 variants across three seeds with 1000-episode runs, checkpoint selection, and held-out checkpoint-FR evaluation. Table 5 adds a multi-seed, multi-evaluation-seed semantic-ablation sweep. The recommended quick review path is to inspect the manifest and processed summaries first, then rerun targeted summary or evaluation scripts only if the corresponding source logs or checkpoints are included.
