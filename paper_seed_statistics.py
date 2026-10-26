"""Paired inference at the independent training-seed level.

Sign flipping assumes exchangeability of method labels within each seed under
the null (or symmetric independent seed differences). It is not a bootstrap
tail probability. Held-out tasks are averaged before testing; duplicating
episodes therefore cannot increase the number of independent observations.
"""

import operator

import numpy as np


def paired_seed_sign_flip(proposed, baseline, *, random_seed=424242, samples=20000):
    a, b = np.asarray(proposed, dtype=float), np.asarray(baseline, dtype=float)
    if a.shape != b.shape or a.ndim < 2 or a.shape[0] < 2 or any(n == 0 for n in a.shape):
        raise ValueError("paired seed statistics require matching seed x task arrays, at least two seeds")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired seed statistics require finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        delta = (a - b).mean(axis=tuple(range(1, a.ndim)))
    if not np.isfinite(delta).all():
        raise ValueError("paired seed differences overflowed; rescale metric inputs")
    n = len(delta)
    # A fixed absolute tolerance can turn tiny, consistent differences into
    # p=1. Normalize before comparing statistics so changing units cannot
    # change the test, while retaining the original effect sizes in output.
    scale = float(np.abs(delta).max())
    normalized_delta = delta / scale if scale > 0 else delta
    observed = abs(float(normalized_delta.mean()))
    tolerance = np.finfo(float).eps * 16
    extreme = 0
    exact = n <= 20
    if exact:
        draws = 2 ** n
    else:
        try:
            draws = operator.index(samples)
        except TypeError as exc:
            raise ValueError("Monte Carlo samples must be an integer") from exc
        if draws < 1000:
            raise ValueError("Monte Carlo sign flip requires at least 1000 samples")
    rng = np.random.default_rng(random_seed)
    for start in range(0, draws, 4096):
        count = min(4096, draws - start)
        if exact:
            ids = np.arange(start, start + count, dtype=np.uint64)[:, None]
            bits = (ids >> np.arange(n, dtype=np.uint64)) & 1
            signs = bits.astype(float) * 2.0 - 1.0
        else:
            signs = rng.integers(0, 2, size=(count, n)) * 2.0 - 1.0
        statistics = np.abs((signs * normalized_delta).mean(axis=1))
        extreme += int(np.count_nonzero(statistics >= observed - tolerance))
    p = extreme / draws if exact else (extreme + 1) / (draws + 1)
    return {
        "test": "paired_training_seed_sign_flip_two_sided",
        "null_assumption": "independent_seed_differences_symmetric_or_within_seed_label_exchangeable",
        "independent_train_seed_count": n,
        "per_train_seed_mean_difference": delta.tolist(),
        "p_value_two_sided": float(p),
        "exact": exact,
        "sign_patterns": draws,
        "random_seed": None if exact else int(random_seed),
        "p_value_computation": (
            "all_sign_patterns_including_observed"
            if exact else "monte_carlo_plus_one_conservative"
        ),
        "minimum_nonzero_reportable_p_value": (
            2.0 / draws if exact else 1.0 / (draws + 1)
        ),
        "inference_role": "exploratory_diagnostic_not_confirmatory_decision",
        "inference_scope": "training_seed_variation_conditional_on_evaluated_task_set",
    }


def holm_adjust(p_values):
    """Holm family-wise correction; returned values preserve input order."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Holm adjustment requires finite p values in [0, 1]")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, float(p[index]) * (len(p) - rank)))
        adjusted[index] = running
    return adjusted.tolist()
