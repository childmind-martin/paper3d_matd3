#!/usr/bin/env python3
"""Run one real MATD3 actor/critic update for the active adaptive modes.

This is intentionally an integration smoke rather than a replacement for the
pure selector-math unit tests.  It exercises the production argument parser,
network/optimizer construction, LiteReplayBuffer quality-label back-fill, the
compiled MATD3 update graph, diagnostics, and strict model/state reload.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("QUIET_OUTPUT", "1")
os.environ.setdefault("LOSS_SYNC_INTERVAL", "1")
os.environ.setdefault("XLA_GLOBAL", "0")
os.environ.setdefault("JIT_COMPILE", "0")
os.environ.setdefault("SPLIT_UPDATE_GRAPH", "1")

from cross_agent_reference_state import (  # noqa: E402
    MODE_ADAPTIVE_TWIN_HEAD_TAIL,
    MODE_SHARED_TWIN_HEAD_TAIL,
    selector_state_errors,
)
import paper3d_train_optimized as training  # noqa: E402


ACTIVE_SMOKE_MODES = (
    MODE_ADAPTIVE_TWIN_HEAD_TAIL,
    MODE_SHARED_TWIN_HEAD_TAIL,
)


def _training_args(mode: str):
    selector_enabled = mode == MODE_SHARED_TWIN_HEAD_TAIL
    parser_argv = [
        "paper3d_train_optimized.py",
        "--algo",
        "matd3",
        "--actor-hidden",
        "16,16",
        "--critic-hidden",
        "16,16",
        "--lr-decay-enabled",
        "false",
        "--batch-size",
        "4",
        "--buffer-size",
        "16",
        "--buffer-dtype",
        "fp32",
        "--policy-freq",
        "2",
        "--use-tf-potential-field",
        "true",
        "--use-fr-feature",
        "true",
        "--use-pf-feature",
        "true",
        "--pf-feature-dim",
        "3",
        "--action-force-ratio",
        "0.2",
        "--matd3-use-dual-q",
        "true",
        "--matd3-use-separated-gradient",
        "true",
        "--matd3-use-hybrid-actor-objective",
        "false",
        "--matd3-action-semantics-mode",
        "dual",
        "--matd3-reconstruct-corrected-target",
        "true",
        "--cross-agent-reference-enabled",
        "true",
        "--cross-agent-reference-start-episode",
        "50",
        "--cross-agent-reference-actor-start-episode",
        "50",
        "--cross-agent-reference-actor-ramp-episodes",
        "0",
        "--cross-agent-reference-actor-require-success",
        "false",
        "--cross-agent-reference-use-clean-label",
        "false",
        "--cross-agent-reference-target-semantics",
        "split_raw_head_corrected_tail",
        "--cross-agent-reference-exclude-random",
        "true",
        "--cross-agent-reference-quality-gate",
        "true",
        "--cross-agent-reference-gate-mode",
        "agent_quality",
        "--cross-agent-reference-update-interval",
        "1",
        "--cross-agent-reference-pairs-per-agent",
        "0",
        "--cross-agent-reference-selector-enabled",
        "true" if selector_enabled else "false",
        "--cross-agent-reference-selector-train-in-graph",
        "true",
        "--cross-agent-reference-selector-mode",
        mode,
        "--cross-agent-reference-selector-hidden",
        "16,8",
        "--cross-agent-reference-selector-init-logit",
        "0.0",
        "--cross-agent-reference-selector-adv-clip",
        "5.0",
        "--cross-agent-reference-advantage-ema-decay",
        "0.99",
        "--cross-agent-reference-advantage-epsilon",
        "1e-6",
        "--cross-agent-reference-advantage-initial-scale",
        "1.0",
    ]
    saved_argv = sys.argv
    try:
        sys.argv = parser_argv
        return training.parse_args()
    finally:
        sys.argv = saved_argv


def _build_replay_buffer(seed: int = 1701):
    n_agents = 3
    obs_dim = 81
    action_dim = 7
    rng = np.random.RandomState(seed)
    replay = training.LiteReplayBuffer(
        capacity=16,
        n_agents=n_agents,
        obs_dims=[obs_dim] * n_agents,
        act_dims=[action_dim] * n_agents,
        use_per=False,
        storage_dtype=np.float32,
        seed=seed,
    )
    positions = []
    generations = []
    for step in range(8):
        obs = rng.uniform(-0.25, 0.25, (n_agents, obs_dim)).astype(np.float32)
        next_obs = obs.copy()
        next_obs[:, 59] = obs[:, 59] - np.float32(0.01)
        raw_action = rng.uniform(-0.8, 0.8, (n_agents, action_dim)).astype(
            np.float32
        )
        pf_forces = rng.uniform(-0.5, 0.5, (n_agents, 3)).astype(np.float32)
        corrected_action = raw_action.copy()
        corrected_action[:, :3] = np.clip(
            raw_action[:, :3]
            + np.float32(0.2) * (pf_forces - raw_action[:, :3]),
            -1.0,
            1.0,
        )
        position, generation = replay.add(
            obs,
            next_obs,
            raw_action,
            np.full((n_agents,), 0.1 + 0.01 * step, dtype=np.float32),
            np.zeros((n_agents,), dtype=np.float32),
            fr_value=0.2,
            pf_forces=pf_forces,
            act_corrected=corrected_action,
            next_act_corrected=corrected_action,
            pf_features=pf_forces,
            next_pf_features=pf_forces,
            act_policy_clean=raw_action,
            random_action_mask=np.zeros((n_agents,), dtype=np.float32),
        )
        positions.append(position)
        generations.append(generation)
    labeled = replay.set_episode_quality_labels(
        positions,
        generations=generations,
        agent_success=[1.0, 0.0, 1.0],
        agent_reach=[1.0, 1.0, 1.0],
        agent_safe=[1.0, 1.0, 1.0],
        team_success=1.0,
        agent_progress=[0.2, 0.15, 0.1],
        agent_final_goal_distance=[0.0, 0.0, 0.0],
    )
    if labeled != len(positions):
        raise RuntimeError(
            f"quality-label back-fill mismatch: {labeled} != {len(positions)}"
        )
    return replay


def _assert_finite_loss_diagnostics(losses, mode: str) -> dict:
    if not isinstance(losses, list) or len(losses) != 3:
        raise RuntimeError(f"{mode}: update returned invalid losses: {losses!r}")
    required = (
        "critic_loss",
        "actor_loss",
        "cross_ref_loss",
        "cross_ref_valid_ratio",
        "cross_ref_head_twin_agreement_ratio",
        "cross_ref_tail_twin_agreement_ratio",
        "cross_ref_head_advantage_ema",
        "cross_ref_tail_advantage_ema",
        "cross_ref_selector_update_count",
    )
    first = losses[0]
    missing = [key for key in required if key not in first]
    if missing:
        raise RuntimeError(f"{mode}: missing diagnostics: {missing}")
    non_finite = [
        key for key in required if not math.isfinite(float(first[key]))
    ]
    if non_finite:
        raise RuntimeError(f"{mode}: non-finite diagnostics: {non_finite}")
    if float(first.get("cross_ref_active", 0.0)) < 0.5:
        raise RuntimeError(f"{mode}: cross-reference graph was not active")
    if float(first["cross_ref_valid_ratio"]) <= 0.0:
        raise RuntimeError(f"{mode}: no eligible reference samples")
    if mode == MODE_SHARED_TWIN_HEAD_TAIL:
        selector_loss = float(first.get("cross_ref_selector_loss", float("nan")))
        selector_grad = float(
            first.get("cross_ref_selector_gradient_norm", float("nan"))
        )
        if not math.isfinite(selector_loss) or selector_loss <= 0.0:
            raise RuntimeError(f"{mode}: invalid selector loss={selector_loss}")
        if not math.isfinite(selector_grad) or selector_grad <= 0.0:
            raise RuntimeError(f"{mode}: selector gradient did not update")
    return first


def run_mode(mode: str) -> dict:
    args = _training_args(mode)
    learner = training.OptimizedMATD3(
        n_agents=3,
        obs_shapes=[81, 81, 81],
        action_dims=[7, 7, 7],
        args=args,
    )
    # Make the twin critics equal only for this smoke so every finite eligible
    # pair exercises the sign-agreement branch deterministically.
    for agent in learner.agents:
        agent["critic2"].set_weights(agent["critic1"].get_weights())
        agent["target_critic2"].set_weights(
            agent["target_critic1"].get_weights()
        )
    learner._current_episode_idx = 50
    losses = learner.update(_build_replay_buffer(), batch_size=4)
    diagnostics = _assert_finite_loss_diagnostics(losses, mode)

    update_count = int(
        learner.cross_agent_reference_selector_update_count.numpy()
    )
    if update_count != 1:
        raise RuntimeError(f"{mode}: update_count={update_count}, expected=1")
    if not bool(learner.cross_agent_reference_head_ema_initialized.numpy()):
        raise RuntimeError(f"{mode}: head EMA was not initialized")
    if not bool(learner.cross_agent_reference_tail_ema_initialized.numpy()):
        raise RuntimeError(f"{mode}: tail EMA was not initialized")

    with tempfile.TemporaryDirectory(prefix="matd3_selector_smoke_") as tmp_dir:
        learner.save_models(tmp_dir)
        state_path = Path(tmp_dir) / "cross_agent_reference_state.json"
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        state_errors = selector_state_errors(
            payload,
            expected_mode=mode,
            expected_input_dim=(
                learner.cross_agent_reference_selector_input_dim
                if mode == MODE_SHARED_TWIN_HEAD_TAIL
                else None
            ),
            require_null_input_dim=mode == MODE_ADAPTIVE_TWIN_HEAD_TAIL,
        )
        if state_errors:
            raise RuntimeError(f"{mode}: invalid saved state: {state_errors}")
        shared_weight = (
            Path(tmp_dir) / "reference_selector_shared.weights.h5"
        )
        if shared_weight.exists() != (mode == MODE_SHARED_TWIN_HEAD_TAIL):
            raise RuntimeError(
                f"{mode}: shared selector weight artifact mismatch"
            )
        learner.load_models(tmp_dir, strict=True)

    return {
        "mode": mode,
        "cross_ref_loss": float(diagnostics["cross_ref_loss"]),
        "valid_ratio": float(diagnostics["cross_ref_valid_ratio"]),
        "head_agreement": float(
            diagnostics["cross_ref_head_twin_agreement_ratio"]
        ),
        "tail_agreement": float(
            diagnostics["cross_ref_tail_twin_agreement_ratio"]
        ),
        "head_ema": float(diagnostics["cross_ref_head_advantage_ema"]),
        "tail_ema": float(diagnostics["cross_ref_tail_advantage_ema"]),
        "selector_loss": float(
            diagnostics.get("cross_ref_selector_loss", 0.0)
        ),
        "selector_grad_norm": float(
            diagnostics.get("cross_ref_selector_gradient_norm", 0.0)
        ),
        "update_count": update_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=ACTIVE_SMOKE_MODES,
        default=list(ACTIVE_SMOKE_MODES),
    )
    cli_args = parser.parse_args()
    results = [run_mode(mode) for mode in cli_args.modes]
    print(json.dumps({"ok": True, "results": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
