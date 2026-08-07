"""Shared validation for the fixed-width obstacle observation semantics."""

from __future__ import annotations


VALID_OBSTACLE_OBSERVATION_MODES = ("nearest_surface", "risk_lite_v2")


def normalize_obstacle_observation_mode(value) -> str:
    """Return the canonical mode name or fail before an experiment starts."""
    raw_value = "nearest_surface" if value is None else str(value).strip()
    if not raw_value:
        raw_value = "nearest_surface"
    mode = raw_value.lower().replace("-", "_")
    aliases = {
        "nearest": "nearest_surface",
        "surface": "nearest_surface",
        "nearest3": "nearest_surface",
        "nearest_top3": "nearest_surface",
        "abs3": "nearest_surface",
        "absolute": "nearest_surface",
        "absolute_nearest": "nearest_surface",
        "risk": "risk_lite_v2",
        "risk_lite": "risk_lite_v2",
        "risklite": "risk_lite_v2",
        "velocity_goal": "risk_lite_v2",
        "vel_goal": "risk_lite_v2",
    }
    mode = aliases.get(mode, mode)
    if mode not in VALID_OBSTACLE_OBSERVATION_MODES:
        valid_modes = " or ".join(VALID_OBSTACLE_OBSERVATION_MODES)
        raise ValueError(
            f"Unsupported obstacle_observation_mode={raw_value!r}; expected {valid_modes}"
        )
    return mode
