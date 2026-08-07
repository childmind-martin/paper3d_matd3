#!/usr/bin/env python3
"""Frozen M0-M3 experiment definitions for the active selector protocol."""

from __future__ import annotations

from typing import Any, Dict

from cross_agent_reference_state import (
    MODE_ADAPTIVE_TWIN_HEAD_TAIL,
    MODE_HARD,
    MODE_SHARED_TWIN_HEAD_TAIL,
)


SELECTOR_PROTOCOL_SCHEMA_VERSION = 1

SELECTOR_PROTOCOL_EXPERIMENT_LABELS = (
    "matd3_cross_agent_ref_behavior_label_agent_quality_gate",
    "matd3_cross_agent_ref_aqual_split_teacher",
    "matd3_cross_agent_ref_adaptive_twin_advantage",
    "matd3_cross_agent_ref_shared_twin_advantage_selector",
)

SELECTOR_PROTOCOL_ID_BY_LABEL = {
    SELECTOR_PROTOCOL_EXPERIMENT_LABELS[0]: "M0",
    SELECTOR_PROTOCOL_EXPERIMENT_LABELS[1]: "M1",
    SELECTOR_PROTOCOL_EXPERIMENT_LABELS[2]: "M2",
    SELECTOR_PROTOCOL_EXPERIMENT_LABELS[3]: "M3",
}

SELECTOR_PROTOCOL_COMMON_ENV = {
    "ALGORITHM": "matd3",
    "MATD3_USE_DUAL_Q": "1",
    "MATD3_USE_SEPARATED_GRADIENT": "1",
    "MATD3_USE_HYBRID_ACTOR_OBJECTIVE": "0",
    "MATD3_ACTION_SEMANTICS_MODE": "dual",
    "MATD3_RECONSTRUCT_CORRECTED_TARGET": "1",
    "MATD3_REQUIRE_GPU": "1",
    "USE_TF_POTENTIAL_FIELD": "1",
    "USE_FR_FEATURE": "1",
    "USE_PF_FEATURE": "1",
    "CROSS_AGENT_REFERENCE_ENABLED": "1",
    "CROSS_AGENT_REFERENCE_COEF": "0.03",
    "CROSS_AGENT_REFERENCE_START_EPISODE": "50",
    "CROSS_AGENT_REFERENCE_ACTOR_START_EPISODE": "50",
    "CROSS_AGENT_REFERENCE_ACTOR_RAMP_EPISODES": "0",
    "CROSS_AGENT_REFERENCE_ACTOR_REQUIRE_SUCCESS": "0",
    "CROSS_AGENT_REFERENCE_UPDATE_INTERVAL": "1",
    "CROSS_AGENT_REFERENCE_PAIRS_PER_AGENT": "0",
    "CROSS_AGENT_REFERENCE_PROGRESS_THRESHOLD": "0.0005",
    "CROSS_AGENT_REFERENCE_MARGIN": "0.0",
    "CROSS_AGENT_REFERENCE_HEAD_WEIGHT": "1.0",
    "CROSS_AGENT_REFERENCE_TAIL_WEIGHT": "0.3",
    "CROSS_AGENT_REFERENCE_USE_CLEAN_LABEL": "0",
    "CROSS_AGENT_REFERENCE_EXCLUDE_RANDOM": "1",
    "CROSS_AGENT_REFERENCE_QUALITY_GATE": "1",
    "CROSS_AGENT_REFERENCE_GATE_MODE": "agent_quality",
    "CROSS_AGENT_REFERENCE_SELECTOR_TRAIN_IN_GRAPH": "1",
    "CROSS_AGENT_REFERENCE_SELECTOR_LR": "0.0001",
    "CROSS_AGENT_REFERENCE_SELECTOR_HIDDEN": "128,64",
    "CROSS_AGENT_REFERENCE_SELECTOR_INIT_LOGIT": "0.0",
    "CROSS_AGENT_REFERENCE_SELECTOR_ADV_CLIP": "5.0",
    "CROSS_AGENT_REFERENCE_ADVANTAGE_EMA_DECAY": "0.99",
    "CROSS_AGENT_REFERENCE_ADVANTAGE_EPSILON": "1e-6",
    "CROSS_AGENT_REFERENCE_ADVANTAGE_INITIAL_SCALE": "1.0",
    "SELECTOR_PROTOCOL_LOCK": "1",
}


def _selector_protocol_config(
    *,
    label: str,
    name: str,
    description: str,
    target_semantics: str,
    selector_mode: str,
    selector_enabled: bool,
) -> Dict[str, Any]:
    env = dict(SELECTOR_PROTOCOL_COMMON_ENV)
    env.update(
        {
            "CROSS_AGENT_REFERENCE_TARGET_SEMANTICS": str(target_semantics),
            "CROSS_AGENT_REFERENCE_SELECTOR_MODE": str(selector_mode),
            "CROSS_AGENT_REFERENCE_SELECTOR_ENABLED": (
                "1" if selector_enabled else "0"
            ),
        }
    )
    return {
        "label": label,
        "name": name,
        "name_en": name,
        "description": description,
        "env": env,
    }


SELECTOR_PROTOCOL_EXPERIMENT_CONFIGS = (
    _selector_protocol_config(
        label=SELECTOR_PROTOCOL_EXPERIMENT_LABELS[0],
        name="M0 - Agent-Quality Legacy Behavior Teacher",
        description=(
            "Selector protocol control: hard agent-quality eligibility with the "
            "legacy all-dimensions executed behavior teacher."
        ),
        target_semantics="legacy",
        selector_mode=MODE_HARD,
        selector_enabled=False,
    ),
    _selector_protocol_config(
        label=SELECTOR_PROTOCOL_EXPERIMENT_LABELS[1],
        name="M1 - Agent-Quality Split Teacher",
        description=(
            "Teacher-semantic ablation: raw behavior action supervises the actor "
            "head and corrected executed action supervises the APF/control tail."
        ),
        target_semantics="split_raw_head_corrected_tail",
        selector_mode=MODE_HARD,
        selector_enabled=False,
    ),
    _selector_protocol_config(
        label=SELECTOR_PROTOCOL_EXPERIMENT_LABELS[2],
        name="M2 - Adaptive Target-Twin Advantage",
        description=(
            "Direct adaptive target-twin head/tail advantage suppression without "
            "a trainable selector."
        ),
        target_semantics="split_raw_head_corrected_tail",
        selector_mode=MODE_ADAPTIVE_TWIN_HEAD_TAIL,
        selector_enabled=False,
    ),
    _selector_protocol_config(
        label=SELECTOR_PROTOCOL_EXPERIMENT_LABELS[3],
        name="M3 - Shared Target-Twin Advantage Selector",
        description=(
            "One shared leakage-free head/tail selector, trained online from "
            "target-twin sign-agreeing adaptive advantage targets."
        ),
        target_semantics="split_raw_head_corrected_tail",
        selector_mode=MODE_SHARED_TWIN_HEAD_TAIL,
        selector_enabled=True,
    ),
)

SELECTOR_PROTOCOL_CONFIG_BY_LABEL = {
    str(config["label"]): config
    for config in SELECTOR_PROTOCOL_EXPERIMENT_CONFIGS
}


__all__ = [
    "SELECTOR_PROTOCOL_COMMON_ENV",
    "SELECTOR_PROTOCOL_CONFIG_BY_LABEL",
    "SELECTOR_PROTOCOL_EXPERIMENT_CONFIGS",
    "SELECTOR_PROTOCOL_EXPERIMENT_LABELS",
    "SELECTOR_PROTOCOL_ID_BY_LABEL",
    "SELECTOR_PROTOCOL_SCHEMA_VERSION",
]
