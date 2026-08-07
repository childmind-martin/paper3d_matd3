#!/usr/bin/env python3
"""Shared, deterministic checkpoint scoring for MATD3 validation selectors."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, TypeVar


SELECTION_RESULT_SCHEMA_VERSION = 3
SELECTION_SCORE_SCHEMA_VERSION = 4
SELECTION_SCORE_COMPARISON_DECIMALS = 12
SELECTION_GUARDED_MIN_COLLISION_FREE = 0.05
SELECTION_GUARDED_COLLISION_COUNT_WEIGHT = 5.0
SELECTION_SCORE_FIELDS = (
    "team_success_rate",
    "partial_success_mean",
    "partial_success_max",
    "partial_success_min",
    "guarded_goal_progress_score",
    "neg_all_reached_without_safe_team_success_rate",
    "collision_free_rate",
    "neg_avg_collision_count",
    "neg_avg_team_final_goal_distance",
    "neg_avg_team_total_path_length",
)


def _finite_float(value: Any, *, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(fallback)
    if not math.isfinite(numeric):
        return float(fallback)
    return numeric


def _metric(summary: Mapping[str, Any], key: str, *, fallback: float) -> float:
    return _finite_float(summary.get(key), fallback=fallback)


def _partial_success_scores(summary: Mapping[str, Any]) -> Tuple[float, float, float]:
    rates = summary.get("agent_success_rates")
    if not isinstance(rates, list):
        return 0.0, 0.0, 0.0
    values = [
        max(0.0, min(1.0, _finite_float(value, fallback=0.0)))
        for value in rates
    ]
    if not values:
        return 0.0, 0.0, 0.0
    return sum(values) / len(values), max(values), min(values)


def _guarded_goal_progress_score(summary: Mapping[str, Any]) -> float:
    distance = max(0.0, _metric(summary, "avg_team_final_goal_distance", fallback=1e12))
    collision_count = max(0.0, _metric(summary, "avg_collision_count", fallback=1e12))
    collision_free = max(0.0, min(1.0, _metric(summary, "collision_free_rate", fallback=0.0)))
    guarded_distance = (
        distance / max(collision_free, SELECTION_GUARDED_MIN_COLLISION_FREE)
        + collision_count * SELECTION_GUARDED_COLLISION_COUNT_WEIGHT
    )
    return -guarded_distance


def selection_summary_errors(summary: Mapping[str, Any]) -> List[str]:
    """Validate every metric that can affect checkpoint ordering."""
    if not isinstance(summary, Mapping):
        return ["summary is not an object"]

    errors: List[str] = []
    bounded_rates = (
        "team_success_rate",
        "all_reached_without_safe_team_success_rate",
        "collision_free_rate",
    )
    for key in bounded_rates:
        value = _finite_float(summary.get(key), fallback=float("nan"))
        if not math.isfinite(value):
            errors.append(f"missing/non-finite {key}")
        elif value < 0.0 or value > 1.0:
            errors.append(f"{key} outside [0, 1]")

    nonnegative_metrics = (
        "avg_collision_count",
        "avg_team_final_goal_distance",
        "avg_team_total_path_length",
    )
    for key in nonnegative_metrics:
        value = _finite_float(summary.get(key), fallback=float("nan"))
        if not math.isfinite(value):
            errors.append(f"missing/non-finite {key}")
        elif value < 0.0:
            errors.append(f"negative {key}")

    rates = summary.get("agent_success_rates")
    if not isinstance(rates, list) or not rates:
        errors.append("missing/empty agent_success_rates")
    else:
        for index, raw_value in enumerate(rates):
            value = _finite_float(raw_value, fallback=float("nan"))
            if not math.isfinite(value):
                errors.append(f"non-finite agent_success_rates[{index}]")
            elif value < 0.0 or value > 1.0:
                errors.append(f"agent_success_rates[{index}] outside [0, 1]")
    return errors


def score_summary(summary: Mapping[str, Any]) -> Tuple[float, ...]:
    """Return the full-precision diagnostic score for one aggregate summary."""
    partial_mean, partial_max, partial_min = _partial_success_scores(summary)
    unsafe_arrival_rate = _metric(
        summary,
        "all_reached_without_safe_team_success_rate",
        fallback=1.0,
    )
    return (
        _metric(summary, "team_success_rate", fallback=-1.0),
        partial_mean,
        partial_max,
        partial_min,
        _guarded_goal_progress_score(summary),
        -unsafe_arrival_rate,
        _metric(summary, "collision_free_rate", fallback=-1.0),
        -_metric(summary, "avg_collision_count", fallback=1e12),
        -_metric(summary, "avg_team_final_goal_distance", fallback=1e12),
        -_metric(summary, "avg_team_total_path_length", fallback=1e12),
    )


def comparison_score(score: Sequence[Any]) -> Tuple[float, ...]:
    """Quantize equivalent empirical rates before lexicographic comparison."""
    if len(score) != len(SELECTION_SCORE_FIELDS):
        raise ValueError(
            f"selection score length mismatch: got={len(score)} expected={len(SELECTION_SCORE_FIELDS)}"
        )
    return tuple(
        round(_finite_float(value, fallback=-1e300), SELECTION_SCORE_COMPARISON_DECIMALS)
        for value in score
    )


CandidateT = TypeVar("CandidateT", bound=Mapping[str, Any])


def select_best_candidate(candidates: Iterable[CandidateT]) -> CandidateT:
    materialized = list(candidates)
    if not materialized:
        raise ValueError("cannot select from an empty candidate list")
    return max(
        materialized,
        key=lambda item: (
            comparison_score(item.get("score", ())),
            -int(item.get("order", 0)),
        ),
    )


def selection_score_schema() -> Dict[str, Any]:
    return {
        "version": int(SELECTION_SCORE_SCHEMA_VERSION),
        "fields": list(SELECTION_SCORE_FIELDS),
        "guarded_goal_progress": {
            "min_collision_free_rate": float(SELECTION_GUARDED_MIN_COLLISION_FREE),
            "collision_count_weight": float(SELECTION_GUARDED_COLLISION_COUNT_WEIGHT),
        },
        "ordering": "lexicographic_desc_on_quantized_score_then_candidate_order",
        "comparison": {
            "method": "round_half_even",
            "decimal_places": int(SELECTION_SCORE_COMPARISON_DECIMALS),
        },
    }
