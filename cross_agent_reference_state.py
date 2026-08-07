#!/usr/bin/env python3
"""TensorFlow-free persistence contract for adaptive reference selectors."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional


SELECTOR_STATE_SCHEMA_VERSION = 1
SELECTOR_FEATURE_SCHEMA_VERSION = 1

MODE_HARD = "hard"
MODE_ADAPTIVE_TWIN_HEAD_TAIL = "adaptive_twin_advantage_head_tail"
MODE_SHARED_TWIN_HEAD_TAIL = "shared_twin_advantage_head_tail"

ACTIVE_SELECTOR_MODES = (
    MODE_HARD,
    MODE_ADAPTIVE_TWIN_HEAD_TAIL,
    MODE_SHARED_TWIN_HEAD_TAIL,
)
TRAINABLE_SELECTOR_MODES = (MODE_SHARED_TWIN_HEAD_TAIL,)
ADVANTAGE_SELECTOR_MODES = (
    MODE_ADAPTIVE_TWIN_HEAD_TAIL,
    MODE_SHARED_TWIN_HEAD_TAIL,
)
HEAD_TAIL_SELECTOR_MODES = ADVANTAGE_SELECTOR_MODES


def selector_state_payload(
    *,
    mode: str,
    head_scale: float,
    tail_scale: float,
    head_initialized: bool,
    tail_initialized: bool,
    update_count: int,
    ema_decay: float,
    epsilon: float,
    advantage_clip: float,
    input_dim: Optional[int],
) -> Dict[str, Any]:
    return {
        "schema_version": int(SELECTOR_STATE_SCHEMA_VERSION),
        "feature_schema_version": int(SELECTOR_FEATURE_SCHEMA_VERSION),
        "mode": str(mode),
        "head_advantage_ema": float(head_scale),
        "tail_advantage_ema": float(tail_scale),
        "head_ema_initialized": bool(head_initialized),
        "tail_ema_initialized": bool(tail_initialized),
        "selector_update_count": int(update_count),
        "ema_decay": float(ema_decay),
        "epsilon": float(epsilon),
        "advantage_clip": float(advantage_clip),
        "input_dim": None if input_dim is None else int(input_dim),
    }


def selector_state_errors(
    payload: Any,
    *,
    expected_mode: Optional[str] = None,
    expected_input_dim: Optional[int] = None,
    require_null_input_dim: bool = False,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["selector state root is not an object"]
    if payload.get("schema_version") != SELECTOR_STATE_SCHEMA_VERSION:
        errors.append(
            "selector state schema_version="
            f"{payload.get('schema_version')!r}, "
            f"expected={SELECTOR_STATE_SCHEMA_VERSION}"
        )
    if payload.get("feature_schema_version") != SELECTOR_FEATURE_SCHEMA_VERSION:
        errors.append(
            "selector feature_schema_version="
            f"{payload.get('feature_schema_version')!r}, "
            f"expected={SELECTOR_FEATURE_SCHEMA_VERSION}"
        )
    if expected_mode is not None and str(payload.get("mode", "")) != str(
        expected_mode
    ):
        errors.append(
            f"selector state mode={payload.get('mode')!r}, "
            f"expected={expected_mode!r}"
        )
    if expected_input_dim is not None:
        try:
            actual_input_dim = int(payload.get("input_dim"))
        except (TypeError, ValueError):
            actual_input_dim = None
        if actual_input_dim != int(expected_input_dim):
            errors.append(
                f"selector state input_dim={actual_input_dim!r}, "
                f"expected={int(expected_input_dim)}"
            )
    if require_null_input_dim and payload.get("input_dim") is not None:
        errors.append("selector state input_dim must be null")

    for key in (
        "head_advantage_ema",
        "tail_advantage_ema",
        "ema_decay",
        "epsilon",
        "advantage_clip",
    ):
        try:
            value = float(payload.get(key))
        except (TypeError, ValueError):
            errors.append(f"selector state {key} is not numeric")
            continue
        if not math.isfinite(value):
            errors.append(f"selector state {key} is not finite")
        if (
            key
            in (
                "head_advantage_ema",
                "tail_advantage_ema",
                "epsilon",
                "advantage_clip",
            )
            and value <= 0.0
        ):
            errors.append(f"selector state {key} must be positive")
        if key == "ema_decay" and not (0.0 <= value < 1.0):
            errors.append("selector state ema_decay must satisfy 0 <= value < 1")

    for key in ("head_ema_initialized", "tail_ema_initialized"):
        if not isinstance(payload.get(key), bool):
            errors.append(f"selector state {key} is not a boolean")

    try:
        raw_update_count = payload.get("selector_update_count")
        update_count = int(raw_update_count)
        if isinstance(raw_update_count, bool) or float(raw_update_count) != float(
            update_count
        ):
            raise ValueError
        if update_count < 0:
            errors.append("selector_update_count must be non-negative")
    except (TypeError, ValueError, OverflowError):
        errors.append("selector_update_count is not an integer")

    raw_input_dim = payload.get("input_dim")
    if raw_input_dim is not None:
        try:
            input_dim = int(raw_input_dim)
            if isinstance(raw_input_dim, bool) or float(raw_input_dim) != float(
                input_dim
            ):
                raise ValueError
            if input_dim <= 0:
                errors.append("selector state input_dim must be positive")
        except (TypeError, ValueError, OverflowError):
            errors.append("selector state input_dim is not an integer or null")
    return errors


__all__ = [
    "ACTIVE_SELECTOR_MODES",
    "ADVANTAGE_SELECTOR_MODES",
    "HEAD_TAIL_SELECTOR_MODES",
    "MODE_ADAPTIVE_TWIN_HEAD_TAIL",
    "MODE_HARD",
    "MODE_SHARED_TWIN_HEAD_TAIL",
    "SELECTOR_FEATURE_SCHEMA_VERSION",
    "SELECTOR_STATE_SCHEMA_VERSION",
    "TRAINABLE_SELECTOR_MODES",
    "selector_state_errors",
    "selector_state_payload",
]
