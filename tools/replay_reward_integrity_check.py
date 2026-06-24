#!/usr/bin/env python3
"""Sentinel integrity check for LiteReplayBuffer reward tuple layout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper3d_train_optimized import LiteReplayBuffer, _unpack_replay_sample


def _make_buffer(use_per: bool) -> LiteReplayBuffer:
    return LiteReplayBuffer(
        capacity=8,
        n_agents=3,
        obs_dims=5,
        act_dims=4,
        seed=1234,
        storage_dtype=np.float32,
        use_per=use_per,
    )


def _fill_and_sample(use_per: bool):
    buffer = _make_buffer(use_per)
    obs = np.arange(15, dtype=np.float32).reshape(3, 5)
    next_obs = obs + 100.0
    act = np.arange(12, dtype=np.float32).reshape(3, 4) / 10.0
    rewards = np.asarray([123.456, -789.25, 42.125], dtype=np.float32)
    dones = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    fr = 0.375
    pf_forces = np.asarray(
        [
            [0.11, 0.12, 0.13],
            [0.21, 0.22, 0.23],
            [0.31, 0.32, 0.33],
        ],
        dtype=np.float32,
    )
    pf_features = pf_forces + 10.0
    next_pf_features = pf_forces + 20.0

    buffer.add(
        obs,
        next_obs,
        act,
        rewards,
        dones,
        fr_value=fr,
        pf_forces=pf_forces,
        act_corrected=act + 0.5,
        pf_features=pf_features,
        next_pf_features=next_pf_features,
    )
    batch = _unpack_replay_sample(buffer.sample(1))
    return {
        "batch": batch,
        "rewards": rewards,
        "dones": dones,
        "fr": fr,
        "pf_forces": pf_forces,
        "pf_features": pf_features,
        "next_pf_features": next_pf_features,
    }


def _assert_close(name: str, actual, expected, atol: float = 1e-5) -> str:
    if not np.allclose(np.asarray(actual), np.asarray(expected), atol=atol, rtol=0.0):
        raise AssertionError(f"{name} mismatch: actual={actual!r} expected={expected!r}")
    arr = np.asarray(actual)
    return f"- {name}: PASS shape={arr.shape} dtype={arr.dtype}"


def run_check() -> str:
    lines = [
        "# Replay Reward Integrity Test",
        "",
        "Sentinel transition checks that replay sampling and `_unpack_replay_sample()` preserve reward/done/FR field order.",
        "",
    ]
    for use_per in (False, True):
        result = _fill_and_sample(use_per)
        batch = result["batch"]
        lines.append(f"## use_per={use_per}")
        lines.append(_assert_close("ReplayBatch.rewards[0]", batch.rewards[0], result["rewards"]))
        lines.append(_assert_close("ReplayBatch.dones[0]", batch.dones[0], result["dones"]))
        lines.append(_assert_close("ReplayBatch.fr[0]", batch.fr[0], result["fr"]))
        lines.append(_assert_close("ReplayBatch.pf_forces[0]", batch.pf_forces[0], result["pf_forces"]))
        lines.append(_assert_close("ReplayBatch.pf_features[0]", batch.pf_features[0], result["pf_features"]))
        lines.append(_assert_close("ReplayBatch.next_pf_features[0]", batch.next_pf_features[0], result["next_pf_features"]))
        lines.append("")
    lines.append("Result: PASS. Reward remains the env.step reward sentinel after original and PER sampling/unpack.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="diagnostics/replay_reward_integrity_test.md")
    args = parser.parse_args()
    text = run_check()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
