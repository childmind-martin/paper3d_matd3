"""Action coordinates used by MATD3 replay and policy evaluation.

Raw-only treats the fixed APF operator as part of the transition kernel.
It retains all seven proposed coordinates, including the APF parameter tail.
The second replay slot is a compatibility alias, not an executed-action record.
"""

import numpy as np

MATD3_ACTION_SEMANTICS = ("dual", "collapsed_replay", "executed_control", "raw_only")


def validate_matd3_action_semantics(mode, *, use_dual_q, separated, hybrid,
                                    reconstruct_corrected_target, random_action_prob=0.0):
    mode = str(mode).strip().lower()
    if mode not in MATD3_ACTION_SEMANTICS:
        raise ValueError(f"Unknown MATD3 action semantics {mode!r}: {MATD3_ACTION_SEMANTICS}")
    if mode == "raw_only" and (use_dual_q or separated or hybrid or reconstruct_corrected_target):
        raise ValueError(
            "raw_only requires scalar twin critics, joint actor gradients, and "
            "matd3_reconstruct_corrected_target=false; APF remains enabled for execution"
        )
    if mode == "raw_only" and float(random_action_prob) != 0.0:
        raise ValueError(
            "raw_only requires random_action_prob=0: legacy epsilon-random actions "
            "bypass APF and violate the fixed raw-action transition kernel; "
            "OU exploration before APF remains available"
        )
    return mode


def select_replay_action_channels(mode, raw_actions, executed_actions):
    if mode not in MATD3_ACTION_SEMANTICS:
        raise ValueError(f"Unknown MATD3 action semantics {mode!r}")
    if mode == "raw_only":
        raw = np.ascontiguousarray(raw_actions, dtype=np.float32)
        return raw, raw
    if mode == "collapsed_replay":
        executed = np.ascontiguousarray(executed_actions, dtype=np.float32)
        return executed, executed
    return raw_actions, executed_actions


def actor_critic_apf_enabled(mode, execution_apf_enabled):
    """Gate only the critic query during learning, never environment execution."""
    if mode not in MATD3_ACTION_SEMANTICS:
        raise ValueError(f"Unknown MATD3 action semantics {mode!r}")
    return bool(execution_apf_enabled) and mode != "raw_only"


def raw_only_critic_actions(raw_actions, compatibility_alias=None):
    """Never reconstruct execution coordinates for a raw-action critic.

    Older tuple buffers may omit the second field. A populated field must
    follow the raw-only alias contract; mixed replay is rejected explicitly.
    """
    raw = np.asarray(raw_actions, dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("raw_only replay contains non-finite proposals")
    if compatibility_alias is not None:
        alias = np.asarray(compatibility_alias, dtype=np.float32)
        if alias.shape != raw.shape or not np.array_equal(alias, raw):
            raise ValueError("raw_only replay alias differs from raw proposals; mixed action semantics")
    return raw.copy()
