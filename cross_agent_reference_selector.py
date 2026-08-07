#!/usr/bin/env python3
"""Core math for the active cross-agent reference selector.

This module intentionally contains only the current, evidence-backed mechanism:

* a hard trajectory-quality eligibility gate;
* head/tail target-twin advantage consensus;
* adaptive, dimensionless advantage targets;
* suppress-only actor multipliers; and
* a leakage-free shared-selector feature schema.

Keeping these operations outside the large training host makes the invariants
directly testable without constructing the full environment.
"""

from __future__ import annotations

from typing import Optional, Tuple

import tensorflow as tf
from cross_agent_reference_state import (
    ACTIVE_SELECTOR_MODES,
    ADVANTAGE_SELECTOR_MODES,
    HEAD_TAIL_SELECTOR_MODES,
    MODE_ADAPTIVE_TWIN_HEAD_TAIL,
    MODE_HARD,
    MODE_SHARED_TWIN_HEAD_TAIL,
    SELECTOR_FEATURE_SCHEMA_VERSION,
    SELECTOR_STATE_SCHEMA_VERSION,
    TRAINABLE_SELECTOR_MODES,
    selector_state_errors,
    selector_state_payload,
)


def build_eligible_mask(
    *,
    finite: tf.Tensor,
    label_valid: tf.Tensor,
    safe: tf.Tensor,
    success: tf.Tensor,
    reach: tf.Tensor,
    trajectory_progress: tf.Tensor,
    near_goal: tf.Tensor,
    random_mask: tf.Tensor,
    exclude_random: bool = True,
) -> tf.Tensor:
    """Return the strict per-sample reference eligibility mask as float32."""

    finite = tf.cast(finite, tf.bool)
    label_valid = tf.cast(label_valid, tf.bool)
    safe = tf.cast(safe, tf.bool)
    useful = tf.logical_or(
        tf.cast(success, tf.bool),
        tf.logical_or(
            tf.cast(reach, tf.bool),
            tf.logical_or(
                tf.cast(trajectory_progress, tf.bool),
                tf.cast(near_goal, tf.bool),
            ),
        ),
    )
    eligible = tf.logical_and(
        finite,
        tf.logical_and(label_valid, tf.logical_and(safe, useful)),
    )
    if exclude_random:
        eligible = tf.logical_and(
            eligible,
            tf.less(tf.cast(random_mask, tf.float32), tf.cast(0.5, tf.float32)),
        )
    return tf.cast(eligible, tf.float32)


def twin_consensus_target(
    advantage_1: tf.Tensor,
    advantage_2: tf.Tensor,
    scale: tf.Tensor,
    *,
    clip: float = 5.0,
    epsilon: float = 1e-6,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Build a soft target only where the two target critics agree in sign.

    Returns ``(target, agreement_mask, consensus_advantage)``.  All returned
    tensors are finite float32 tensors and detached from the critic graph.
    """

    advantage_1 = tf.stop_gradient(tf.cast(advantage_1, tf.float32))
    advantage_2 = tf.stop_gradient(tf.cast(advantage_2, tf.float32))
    finite = tf.logical_and(
        tf.math.is_finite(advantage_1),
        tf.math.is_finite(advantage_2),
    )
    safe_advantage_1 = tf.where(finite, advantage_1, tf.zeros_like(advantage_1))
    safe_advantage_2 = tf.where(finite, advantage_2, tf.zeros_like(advantage_2))
    agreement = tf.logical_and(
        finite,
        tf.equal(tf.sign(safe_advantage_1), tf.sign(safe_advantage_2)),
    )
    consensus = tf.stop_gradient(
        tf.cast(0.5, tf.float32) * (safe_advantage_1 + safe_advantage_2)
    )
    safe_scale = tf.maximum(
        tf.abs(tf.cast(scale, tf.float32)),
        tf.cast(epsilon, tf.float32),
    )
    scaled = tf.clip_by_value(
        consensus / safe_scale,
        tf.cast(-abs(float(clip)), tf.float32),
        tf.cast(abs(float(clip)), tf.float32),
    )
    target = tf.stop_gradient(tf.math.sigmoid(scaled))
    return target, tf.cast(agreement, tf.float32), consensus


def next_ema_scale(
    current_scale: tf.Tensor,
    initialized: tf.Tensor,
    consensus_advantage: tf.Tensor,
    valid_mask: tf.Tensor,
    *,
    decay: float = 0.99,
    epsilon: float = 1e-6,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Compute one EMA update from all valid pairs in one actor update."""

    current_scale = tf.cast(current_scale, tf.float32)
    initialized = tf.cast(initialized, tf.bool)
    values = tf.abs(tf.stop_gradient(tf.cast(consensus_advantage, tf.float32)))
    mask = tf.cast(valid_mask, tf.float32)
    finite = tf.cast(tf.math.is_finite(values), tf.float32)
    mask = tf.clip_by_value(mask, 0.0, 1.0) * finite
    count = tf.reduce_sum(mask)
    batch_scale = (
        tf.reduce_sum(tf.where(tf.math.is_finite(values), values, tf.zeros_like(values)) * mask)
        / tf.maximum(count, tf.cast(1.0, tf.float32))
    )
    decay_tensor = tf.clip_by_value(
        tf.cast(decay, tf.float32),
        tf.cast(0.0, tf.float32),
        tf.cast(1.0, tf.float32),
    )
    updated = tf.where(
        initialized,
        decay_tensor * current_scale
        + (tf.cast(1.0, tf.float32) - decay_tensor) * batch_scale,
        batch_scale,
    )
    updated = tf.maximum(updated, tf.cast(epsilon, tf.float32))
    has_valid = tf.greater(count, tf.cast(0.0, tf.float32))
    next_scale = tf.where(has_valid, updated, current_scale)
    next_initialized = tf.logical_or(initialized, has_valid)
    return tf.stop_gradient(next_scale), next_initialized, tf.stop_gradient(count)


def suppress_only_multiplier(
    score_or_target: tf.Tensor,
    agreement_mask: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """Map 0.5 to the exact baseline and never amplify reference imitation."""

    score = tf.clip_by_value(tf.cast(score_or_target, tf.float32), 0.0, 1.0)
    multiplier = tf.minimum(
        tf.cast(1.0, tf.float32),
        tf.cast(2.0, tf.float32) * score,
    )
    if agreement_mask is not None:
        agreement = tf.cast(agreement_mask, tf.float32)
        multiplier = tf.where(
            agreement >= tf.cast(0.5, tf.float32),
            multiplier,
            tf.ones_like(multiplier),
        )
    return tf.stop_gradient(multiplier)


def reduce_reference_loss_fixed_denominator(
    head_loss: tf.Tensor,
    tail_loss: tf.Tensor,
    eligible_mask: tf.Tensor,
    head_multiplier: tf.Tensor,
    tail_multiplier: tf.Tensor,
    *,
    head_weight: tf.Tensor | float = 1.0,
    tail_weight: tf.Tensor | float = 0.3,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Reduce reference loss without normalizing selector suppression away."""

    eligible = tf.clip_by_value(tf.cast(eligible_mask, tf.float32), 0.0, 1.0)
    denominator = tf.maximum(
        tf.reduce_sum(eligible),
        tf.cast(1.0, tf.float32),
    )
    weighted = eligible * (
        tf.cast(head_weight, tf.float32)
        * tf.cast(head_loss, tf.float32)
        * tf.stop_gradient(tf.cast(head_multiplier, tf.float32))
        + tf.cast(tail_weight, tf.float32)
        * tf.cast(tail_loss, tf.float32)
        * tf.stop_gradient(tf.cast(tail_multiplier, tf.float32))
    )
    loss = tf.reduce_sum(weighted) / denominator
    return loss, denominator


def binary_cross_entropy_per_sample(
    target: tf.Tensor,
    score: tf.Tensor,
    *,
    epsilon: float = 1e-5,
) -> tf.Tensor:
    target = tf.stop_gradient(tf.clip_by_value(tf.cast(target, tf.float32), 0.0, 1.0))
    score = tf.clip_by_value(
        tf.cast(score, tf.float32),
        tf.cast(epsilon, tf.float32),
        tf.cast(1.0 - epsilon, tf.float32),
    )
    return -(
        target * tf.math.log(score)
        + (tf.cast(1.0, tf.float32) - target)
        * tf.math.log(tf.cast(1.0, tf.float32) - score)
    )


def build_shared_selector_features(
    *,
    reference_observation: tf.Tensor,
    teacher_action: tf.Tensor,
    learner_action: tf.Tensor,
    force_ratio: tf.Tensor,
    pf_feature: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """Build the leakage-free feature vector used by the shared selector."""

    teacher = tf.cast(teacher_action, tf.float32)
    learner = tf.cast(learner_action, tf.float32)
    parts = [
        tf.cast(reference_observation, tf.float32),
        teacher,
        learner,
        teacher - learner,
        tf.cast(force_ratio, tf.float32),
    ]
    if pf_feature is not None:
        parts.append(tf.cast(pf_feature, tf.float32))
    return tf.stop_gradient(tf.concat(parts, axis=1))
