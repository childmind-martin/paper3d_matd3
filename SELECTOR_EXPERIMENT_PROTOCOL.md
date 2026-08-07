# MATD3 shared-selector implementation and experiment protocol

This document is the executable protocol for the active cross-agent reference
work. It deliberately separates mechanism attribution, training completion,
formal robustness evaluation, and training-seed replication.

## 1. The four models

All four models use the same MATD3 twin critics, separated actor gradient,
dual action semantics, corrected target reconstruction, reward, terrain,
dynamic obstacles, fixed start/goal positions, training seed, FR schedule,
network size, optimizer, replay, training horizon, and mandatory GPU
execution. They differ in exactly
three frozen environment fields:

- `CROSS_AGENT_REFERENCE_TARGET_SEMANTICS`
- `CROSS_AGENT_REFERENCE_SELECTOR_MODE`
- `CROSS_AGENT_REFERENCE_SELECTOR_ENABLED`

| ID | Experiment label | Teacher | Adaptive filter | Trainable selector |
|---|---|---|---|---|
| M0 | `matd3_cross_agent_ref_behavior_label_agent_quality_gate` | Legacy executed behavior action for all dimensions | None; hard trajectory-quality gate | No |
| M1 | `matd3_cross_agent_ref_aqual_split_teacher` | Raw behavior action for the 3-D head; corrected executed action for the control/APF tail | None; hard trajectory-quality gate | No |
| M2 | `matd3_cross_agent_ref_adaptive_twin_advantage` | Same split teacher as M1 | Target-twin head/tail advantage consensus with suppress-only multipliers | No |
| M3 | `matd3_cross_agent_ref_shared_twin_advantage_selector` | Same split teacher as M1/M2 | Same target-twin labels as M2 | One shared online head/tail selector |

The attribution chain is therefore:

1. M0 -> M1: teacher semantics.
2. M1 -> M2: adaptive target-twin suppression.
3. M2 -> M3: learned shared-selector generalization.

M3 does not pretrain the selector. The selector and actor reference loss both
start at episode 50. Its output layer starts at an exact score of 0.5, which
maps to multiplier 1.0 and therefore reproduces the M2/M1 reference strength
before learning.

## 2. Implemented contracts

The active implementation is split across these files:

- `cross_agent_reference_selector.py`: eligibility, twin consensus, EMA scale,
  suppress-only multiplier, fixed-denominator loss, BCE, and leakage-free
  features.
- `cross_agent_reference_state.py`: mode names and strict persisted-state
  schema.
- `selector_experiment_protocol.py`: the frozen M0-M3 definitions.
- `paper3d_train_optimized.py`: replay-label backfill, actor/critic/selector
  update, diagnostics, save, and strict reload.
- `ablation_dual_q_separated_gradient.py`: active registry, immutable
  manifests, whole-unit resume, and M0-M3 scheduling.
- `experiment_runtime_config.py`: complete training-unit validation.
- `tools/build_selector_protocol_batch_spec.py`: freezes and validates the
  formal model/test matrix and all training-artifact hashes.
- `tools/smoke_parallel_env_audit.py`: executes real production workers and
  verifies per-environment IPC success/collision/D_min snapshots.
- `run_selector_m0_m3_full_gpu.sh`: one-command, whole-unit-resumable driver
  for verification, pilot, formal training, evaluation, and analysis.
- `run_noise_dependency_check_resume_fast.sh`: runs or resumes the formal GPU
  matrix one complete cell at a time.
- `analyze_noise_dependency_batch.py`: independently revalidates 480 episode
  records, recomputes metrics, paired tests, robustness summaries, and
  selector-training diagnostics.

The active eligibility rule is:

`finite AND full-episode-label-valid AND safe AND
(success OR reach OR useful-progress OR near-goal) AND non-random`.

M2/M3 use the source agent's target twin critics. Head and tail are evaluated
separately. A label is trainable only when the two target critics agree in
sign. Advantage magnitude is normalized by a pooled head/tail EMA, and each
EMA is updated once per actor update after all agents and peers are pooled.

M3 selector inputs contain only current information:

- reference observation;
- split teacher action;
- learner action;
- teacher-minus-learner action;
- current FR;
- current PF feature only when PF is an actor input.

Future observations, rewards, terminal outcomes, success labels, and critic
values are not selector inputs.

Old failed selector/reward heuristics are absent from the active experiment
registry and cannot be started as training modes. Historical result reading
and actor-only evaluation compatibility remain; historical data are not
deleted.

## 3. Phase 0: code and update-path verification

Run this after any code change and before starting an environment training
batch:

```bash
cd /home/tang/matd3
PY=/home/tang/miniconda3/envs/maddpg_env/bin/python

"$PY" -m py_compile \
  cross_agent_reference_state.py \
  cross_agent_reference_selector.py \
  selector_experiment_protocol.py \
  experiment_runtime_config.py \
  paper3d_train_optimized.py \
  ablation_dual_q_separated_gradient.py \
  tools/build_selector_protocol_batch_spec.py \
  tools/preflight_selector_num_env_manifests.py \
  tools/smoke_selector_update.py \
  tools/smoke_parallel_env_audit.py \
  analyze_noise_dependency_batch.py

bash -n run_optimized.sh
bash -n run_noise_dependency_check_resume_fast.sh
bash -n run_selector_m0_m3_full_gpu.sh

"$PY" -m unittest -v \
  test_cross_agent_reference_selector.py \
  test_experiment_integrity.py

"$PY" tools/preflight_selector_num_env_manifests.py
"$PY" tools/smoke_selector_update.py
"$PY" tools/smoke_parallel_env_audit.py
```

The manifest preflight resolves all four real launch paths and requires
`num_envs=4` in the command line, strict execution environment, and frozen
manifest. The selector smoke uses the production parser, networks, replay
buffer, compiled update, optimizer, state save, and strict reload. M3 must
report a finite positive selector loss and selector gradient. The parallel
smoke starts real environment subprocesses, steps them, and verifies one
authoritative audit snapshot per worker while leaving the parent GPU visible.

## 4. Shared shell configuration

Use one shell for the commands below:

```bash
cd /home/tang/matd3
PY=/home/tang/miniconda3/envs/maddpg_env/bin/python

SELECTOR_EXPERIMENTS=(
  matd3_cross_agent_ref_behavior_label_agent_quality_gate
  matd3_cross_agent_ref_aqual_split_teacher
  matd3_cross_agent_ref_adaptive_twin_advantage
  matd3_cross_agent_ref_shared_twin_advantage_selector
)

COMMON_TRAIN_ARGS=(
  --batch-size 1024
  --num-envs 4
  --max-parallel 1
  --experiment-max-parallel 1
  --worker-launch-stagger-seconds 0
  --experiment-group B
  --config-mode strict_ablation
  --env-isolation strict
  --scenario-seed 88
  --use-weighted-reward 1
  --action-force-ratio 0.50
  --action-force-ratio-schedule-pct
  '0%:0.50,25%:0.48,50%:0.45,70%:0.40,85%:0.35,100%:0.32'
  --post-eval-episodes 30
  --post-eval-episode-length-multiplier 1.1
  --post-eval-seed 10088
  --post-eval-mode shared_match_train_env
  --post-eval-selection-protocol fixed
  --post-eval-model-variant final
  --allow-post-eval-without-train-success
  --skip-local-plots
)
```

The training batch performs one fixed-final 30-episode post-evaluation for
each model. This has no validation-based checkpoint selection and creates the
shared ordered terrain/obstacle specification consumed by the later 4x4
formal batch. Treat these four preliminary runs as a pipeline gate; the
separate 480-episode matrix remains the formal robustness result.

`--allow-post-eval-without-train-success` is required here because zero
training successes must be reported as an algorithm outcome, not used to
silently omit a model from the common evaluation protocol.

`--max-parallel 1` is intentional for one 8 GB GPU. It schedules the four
algorithm x seed training units sequentially.

`--num-envs 4` is a different layer of parallelism: one model owns the GPU
while four CPU environment workers collect four trajectories synchronously.
Thus 1000 `train_episodes` means 1000 network-training iterations and 4000
complete environment trajectories per model. Reward, success, and collision
curves use the equal mean across the four environments; exact per-environment
values are retained in the training artifacts.

## 5. Phase 1: optional 100-iteration, four-environment pilot

This is not paper data. Each model performs 100 synchronous iterations over
four environments, or 400 complete environment trajectories. It verifies that
full-episode labels become available, M0-M3 all execute the reference branch,
M2/M3 initialize both EMA scales, and M3 receives real selector gradients.

```bash
"$PY" ablation_dual_q_separated_gradient.py \
  --multi-seed \
  --seeds 9101 \
  --episodes 100 \
  "${COMMON_TRAIN_ARGS[@]}" \
  --experiments "${SELECTOR_EXPERIMENTS[@]}"
```

Copy the exact parent directory printed by the launcher:

```bash
PILOT_PARENT=/home/tang/matd3/ablation_experiments/multi_seed_groupB_YYYYMMDD_HHMMSS
PILOT_SPEC="$PILOT_PARENT/results/selector_pilot_spec.json"

"$PY" tools/build_selector_protocol_batch_spec.py \
  --parent-batch-dir "$PILOT_PARENT" \
  --train-seed 9101 \
  --train-episodes 100 \
  --train-num-envs 4 \
  --output "$PILOT_SPEC" \
  --out-root /home/tang/matd3/evaluation_results_selector_pilot_seed9101

"$PY" tools/build_selector_protocol_batch_spec.py \
  --validate \
  --output "$PILOT_SPEC"
```

Do not start the 1000-episode run if this builder rejects inactive reference
updates, zero eligible samples, uninitialized EMA state, missing state/weights,
or the absence of a positive M3 selector gradient.

## 6. Phase 2: formal four-model training

This creates four fixed-horizon final models from training seed 101:

```bash
"$PY" ablation_dual_q_separated_gradient.py \
  --multi-seed \
  --seeds 101 \
  --episodes 1000 \
  "${COMMON_TRAIN_ARGS[@]}" \
  --experiments "${SELECTOR_EXPERIMENTS[@]}"
```

The batch resume unit is a complete `algorithm x seed`, not an episode:

```bash
TRAIN_PARENT=/home/tang/matd3/ablation_experiments/multi_seed_groupB_YYYYMMDD_HHMMSS

"$PY" ablation_dual_q_separated_gradient.py \
  --resume-parent-batch-dir "$TRAIN_PARENT" \
  --max-parallel 1
```

On resume:

- a unit with valid final `results.json`, all final actor/critic weights, the
  correct seed/horizon, and required adaptive-selector artifacts is reused;
- an incomplete unit has only its exact model/log identity removed and restarts
  from episode 0;
- replay, optimizer, and mid-episode state are neither saved nor resumed;
- the original immutable resolved manifest is reused.

## 7. Phase 3: freeze the formal 4 x 4 x 30 GPU specification

Set `TRAIN_PARENT` to the directory from Phase 2, then create a new output root:

```bash
FORMAL_OUT=/home/tang/matd3/evaluation_results_selector_m0_m3_seed101_gpu_v1
FORMAL_SPEC="$TRAIN_PARENT/results/selector_formal_batch_spec_v10.json"

"$PY" tools/build_selector_protocol_batch_spec.py \
  --parent-batch-dir "$TRAIN_PARENT" \
  --train-seed 101 \
  --train-episodes 1000 \
  --train-num-envs 4 \
  --eval-noise-seed 101 \
  --eval-process-shards 3 \
  --eval-process-workers 3 \
  --eval-shard-episode-parallelism 4 \
  --eval-shard-env-step-threads 4 \
  --out-root "$FORMAL_OUT" \
  --output "$FORMAL_SPEC"

"$PY" tools/build_selector_protocol_batch_spec.py \
  --validate \
  --output "$FORMAL_SPEC"
```

The builder refuses to create the specification unless:

- exactly M0-M3 are present in order;
- all four models are complete 1000-episode seed-101 units;
- all four models prove `num_envs=4`, 1000 synchronous iterations, and 4000
  complete environment trajectories in both effective arguments and results;
- the trainer's recorded effective arguments match every frozen selector
  protocol field, rather than merely matching the launch manifest;
- the four final model paths and Actor signatures are all distinct;
- all four training results prove that physical and logical TensorFlow GPUs
  were present while `MATD3_REQUIRE_GPU=1`;
- all four fixed-final preliminary post-evaluations completed 30 episodes,
  loaded the corresponding `final` actor signature, and recorded physical and
  logical GPUs while GPU use was mandatory;
- every preliminary `episode_details` row has the exact frozen episode id,
  terrain seed, terrain-variant seed, and dynamic-obstacle seed;
- every preliminary run used the Python evaluation backend, the frozen fixed
  positions file, and the requested 1.1 episode-length multiplier;
- all four use fixed-horizon `final` weights;
- all non-mechanism resolved-manifest fields are identical;
- all artifact, result, manifest, loss-history, actor, selector-state, and
  selector-weight hashes match;
- the fixed positions file and ordered-sequence source content hashes match;
- the 1.1 multiplier resolves exactly from the recorded training
  `episode_length` to the frozen evaluation `episode_length`;
- each model logged active and eligible reference updates;
- M2/M3 have initialized adaptive state and M3 has a positive selector
  gradient;
- the common test environment is train-match semi-random terrain with dynamic
  obstacles and one fixed positions file.

## 8. Phase 4: preflight and run the formal GPU matrix

The four frozen modes are:

| Mode | Gaussian action noise | Random action probability |
|---|---:|---:|
| `deterministic` | 0.00 | 0.00 |
| `gaussian_noise_0p11` | 0.11 | 0.00 |
| `random_1pct` | 0.00 | 0.01 |
| `gaussian_0p11_random_1pct` | 0.11 | 0.01 |

Every cell uses the same ordered 30 terrain, terrain-variant, dynamic-obstacle,
and action-noise episode seeds. The total is `4 x 4 x 30 = 480` evaluated
episodes.

Verify that TensorFlow in the training environment sees the GPU:

```bash
nvidia-smi
LD_LIBRARY_PATH="/usr/lib/wsl/lib:/home/tang/miniconda3/envs/maddpg_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
"$PY" -c 'import tensorflow as tf; p=tf.config.list_physical_devices("GPU"); l=tf.config.list_logical_devices("GPU"); assert p and l, "TensorFlow has no physical/logical GPU"; print("physical=", p, "logical=", l)'
```

Resolve every model and cell without starting evaluation:

```bash
BATCH_SPEC_JSON="$FORMAL_SPEC" \
PREFLIGHT_ONLY=1 \
bash run_noise_dependency_check_resume_fast.sh
```

Run the formal batch:

```bash
BATCH_SPEC_JSON="$FORMAL_SPEC" \
bash run_noise_dependency_check_resume_fast.sh
```

If interrupted, rerun the exact same command. A cell that passes model,
artifact, protocol, environment, noise, sequence, shard, GPU, and 30-episode
checks is skipped. Any incomplete or mismatched cell has only its exact
`FORMAL_OUT/Mx/mode` directory removed and restarts from episode 0.

## 9. Phase 5: independent validation and analysis

```bash
"$PY" analyze_noise_dependency_batch.py "$FORMAL_OUT"
```

Acceptance requires:

- `16/16` model/mode cells;
- `480/480` ordered episode records;
- physical and logical GPU recorded in every cell;
- physical and logical GPU recorded for every formal training model;
- identical common positions and terrain/obstacle sequences;
- exact model, result, manifest, batch-spec, and protocol-source identities;
- recomputed headline metrics equal the stored summaries.

The generated outputs are:

- `formal_batch_validation.json`
- `formal_batch_training_diagnostics.csv`
- `formal_batch_metrics.csv`
- `formal_batch_paired_deltas.csv`
- `formal_batch_analysis.json`
- `formal_batch_report.md`

Interpret the results along the M0->M1->M2->M3 chain. Primary outcome is team
success; safety outcomes are collision-free rate and collision count; failure
progress is final goal distance and timeout rate. For each perturbation, the
analysis reports paired success gains/losses with exact McNemar p-values and
paired bootstrap confidence intervals for collision, reward, and distance
deltas.

The 4x4x30 batch measures scenario/noise robustness for one trained seed. It
does not establish training-seed stability. Only after this mechanism batch
passes should the complete training-and-evaluation chain be repeated for
additional training seeds such as 202 and 303.

## 10. One-command complete run

The executable wrapper performs every phase above. It persists separate pilot
and formal parent timestamps under `selector_experiment_runs/RUN_ID`, so
rerunning the exact command preserves completed algorithm x seed units,
restarts an interrupted training unit from episode 0, preserves completed
formal evaluation cells, and restarts an incomplete cell from episode 0:

```bash
cd /home/tang/matd3 && RUN_ID=selector_m0_m3_env4_seed101_v10 bash run_selector_m0_m3_full_gpu.sh
```

The pilot is enabled by default. After a pilot has already passed for this
code state, it can be explicitly skipped with `RUN_PILOT=0`; this does not
change the formal training or 4x4x30 evaluation protocol.
