"""Forecasting metrics: AUROC, AUPRC, lead time, calibration, clustered CIs.

Train, dev, and test membership is a function of the prompt id only. The salt
and the 60/20/20 cut are fixed so the same prompt lands in the same split for
the LFM2 checkout and the Qwen run.
"""

from __future__ import annotations

import hashlib
from typing import Callable, Sequence

import numpy as np
from scipy.stats import scoreatpercentile
from sklearn.metrics import average_precision_score, roc_auc_score

SPLIT_SALT = "jspace-forecast-v1"
TRAIN_FRACTION = 0.60
DEV_FRACTION = 0.20
FAR_BUDGET = 0.05

StatFn = Callable[[np.ndarray], float]


def assign_split(prompt_id: int, *, salt: str = SPLIT_SALT) -> str:
    """Hash-locked split. ``prompt_id`` is the antidoom-mix row id, not a row index."""
    digest = hashlib.sha256(f"{salt}:{int(prompt_id)}".encode("utf-8")).digest()
    u = int.from_bytes(digest[:8], "big") / float(2**64)
    if u < TRAIN_FRACTION:
        return "train"
    if u < TRAIN_FRACTION + DEV_FRACTION:
        return "dev"
    return "test"


def _finite_pair(y, scores) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    if y.shape != s.shape:
        raise ValueError(f"y and scores length mismatch: {y.shape} vs {s.shape}")
    mask = np.isfinite(y) & np.isfinite(s)
    return y[mask], s[mask]


def safe_auroc(y, scores) -> float:
    y, s = _finite_pair(y, scores)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def safe_auprc(y, scores) -> float:
    y, s = _finite_pair(y, scores)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def expected_calibration_error(y, probabilities, *, n_bins: int = 10) -> float:
    """Equal-width ECE. ``probabilities`` must be in [0, 1]; values are clipped."""
    y, p = _finite_pair(y, probabilities)
    if y.size == 0:
        return float("nan")
    p = np.clip(p, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    n = y.size
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        acc = float(y[mask].mean())
        conf = float(p[mask].mean())
        ece += (count / n) * abs(acc - conf)
    return float(ece)


def threshold_for_far(maxima: Sequence[float], budget: float = FAR_BUDGET) -> float:
    """Most sensitive threshold whose false-alarm count is at most ``budget``.

    ``maxima`` is one score per non-loop trace (the max of that trace, or the
    single window score). An alarm is ``score >= threshold``. Ties are not
    allowed to push the rate over the budget: the threshold moves just above
    the largest value that would otherwise add an extra alarm. With fewer than
    ``1/budget`` controls the budget admits zero alarms, and the threshold sits
    just above the largest control score.
    """
    m = np.asarray(list(maxima), dtype=float).ravel()
    m = m[np.isfinite(m)]
    if m.size == 0:
        return float("inf")
    m.sort()
    n = int(m.size)
    allowed = int(np.floor(float(budget) * n + 1e-12))
    if allowed >= n:
        return float(np.nextafter(float(m[0]), -np.inf))
    if allowed <= 0:
        return float(np.nextafter(float(m[-1]), np.inf))
    fence = float(m[n - allowed - 1])
    return float(np.nextafter(fence, np.inf))


def first_crossing(scores, threshold: float) -> int | None:
    """First index with a finite score at or above ``threshold``."""
    s = np.asarray(scores, dtype=float).ravel()
    if not np.isfinite(threshold):
        return None
    hit = np.flatnonzero(np.isfinite(s) & (s >= threshold))
    if hit.size == 0:
        return None
    return int(hit[0])


def lead_time_summary(score_rows: Sequence[np.ndarray], onsets: Sequence[int], threshold: float) -> dict:
    """Lead time relative to onset. Positive lead means the alarm is before onset.

    Traces that never cross the threshold are counted in ``n_never`` and are
    left out of the signed-lead median (they are right-censored). ``n_early``
    counts alarms strictly before the onset token.
    """
    early: list[int] = []
    signed: list[int] = []
    n_early = n_at = n_late = n_never = 0
    n = 0
    for scores, onset in zip(score_rows, onsets):
        if onset is None:
            continue
        n += 1
        alarm = first_crossing(scores, threshold)
        if alarm is None:
            n_never += 1
            continue
        lead = int(onset) - int(alarm)
        signed.append(lead)
        if alarm < int(onset):
            n_early += 1
            early.append(lead)
        elif alarm == int(onset):
            n_at += 1
        else:
            n_late += 1

    def _median(xs: list[int]) -> float:
        if not xs:
            return float("nan")
        return float(np.median(np.asarray(xs, dtype=float)))

    return {
        "n": n,
        "n_early": n_early,
        "n_at_onset": n_at,
        "n_late": n_late,
        "n_never": n_never,
        "early_rate": (n_early / n) if n else float("nan"),
        "median_lead_tokens_early": _median(early),
        "median_signed_lead_tokens": _median(signed),
    }


def _percentile_ci(reps: np.ndarray, ci: float) -> tuple[float, float]:
    alpha = (100.0 - ci) / 2.0
    lo, hi = scoreatpercentile(reps, [alpha, 100.0 - alpha])
    return float(lo), float(hi)


def clustered_bootstrap_ci(
    prompt_ids,
    stat_fn: StatFn,
    *,
    n_boot: int = 1000,
    ci: float = 95.0,
    seed: int = 0,
) -> dict:
    """Bootstrap that resamples prompt ids, keeping every row of a drawn prompt.

    A prompt drawn twice contributes its rows twice. Non-finite replicates
    (for example an AUROC draw with one class) are dropped.
    """
    prompt_ids = np.asarray(prompt_ids)
    empty = {
        "point": float("nan"),
        "lo": float("nan"),
        "hi": float("nan"),
        "n_boot_finite": 0,
        "n_boot": int(n_boot),
        "ci": ci,
    }
    if prompt_ids.size == 0:
        return empty
    uniq, inverse = np.unique(prompt_ids, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(uniq.size)]
    try:
        point = float(stat_fn(np.arange(prompt_ids.size)))
    except Exception:
        point = float("nan")
    rng = np.random.default_rng(seed)
    reps = np.empty(int(n_boot), dtype=float)
    n_u = uniq.size
    draws = rng.integers(0, n_u, size=(int(n_boot), n_u))
    for b in range(int(n_boot)):
        idx = np.concatenate([groups[int(j)] for j in draws[b]])
        try:
            reps[b] = float(stat_fn(idx))
        except Exception:
            reps[b] = float("nan")
    finite = reps[np.isfinite(reps)]
    if finite.size == 0 or not np.isfinite(point):
        lo = hi = float("nan")
    else:
        lo, hi = _percentile_ci(finite, ci)
    return {
        "point": point if np.isfinite(point) else float("nan"),
        "lo": lo,
        "hi": hi,
        "n_boot_finite": int(finite.size),
        "n_boot": int(n_boot),
        "ci": ci,
    }


def two_group_bootstrap(
    group_a_ids,
    group_b_ids,
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_boot: int = 1000,
    ci: float = 95.0,
    seed: int = 0,
) -> dict:
    """Resample two prompt-clustered groups independently (loops and controls)."""

    def _groups(ids):
        ids = np.asarray(ids)
        if ids.size == 0:
            return []
        _uniq, inverse = np.unique(ids, return_inverse=True)
        return [np.flatnonzero(inverse == i) for i in range(_uniq.size)]

    ga = _groups(group_a_ids)
    gb = _groups(group_b_ids)
    try:
        point = float(
            stat_fn(np.arange(len(group_a_ids)), np.arange(len(group_b_ids)))
        )
    except Exception:
        point = float("nan")
    if not ga or not gb:
        return {
            "point": point,
            "lo": float("nan"),
            "hi": float("nan"),
            "n_boot_finite": 0,
            "n_boot": int(n_boot),
            "ci": ci,
        }
    rng = np.random.default_rng(seed)
    reps = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        ia = np.concatenate([ga[int(j)] for j in rng.integers(0, len(ga), size=len(ga))])
        ib = np.concatenate([gb[int(j)] for j in rng.integers(0, len(gb), size=len(gb))])
        try:
            reps[b] = float(stat_fn(ia, ib))
        except Exception:
            reps[b] = float("nan")
    finite = reps[np.isfinite(reps)]
    if finite.size == 0 or not np.isfinite(point):
        lo = hi = float("nan")
    else:
        lo, hi = _percentile_ci(finite, ci)
    return {
        "point": point if np.isfinite(point) else float("nan"),
        "lo": lo,
        "hi": hi,
        "n_boot_finite": int(finite.size),
        "n_boot": int(n_boot),
        "ci": ci,
    }


def classification_bundle(y, scores, *, probabilities: bool, n_bins: int = 10) -> dict:
    y_arr, s_arr = _finite_pair(y, scores)
    return {
        "n": int(y_arr.size),
        "n_pos": int(np.sum(y_arr == 1)) if y_arr.size else 0,
        "n_neg": int(np.sum(y_arr == 0)) if y_arr.size else 0,
        "auroc": safe_auroc(y_arr, s_arr),
        "auprc": safe_auprc(y_arr, s_arr),
        "ece": expected_calibration_error(y_arr, s_arr, n_bins=n_bins) if probabilities else float("nan"),
    }


def rates_at_threshold(y, scores, threshold: float) -> dict:
    y_arr, s_arr = _finite_pair(y, scores)
    pos = y_arr == 1
    neg = y_arr == 0
    alarm = np.isfinite(s_arr) & (s_arr >= threshold)
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    tp = int((alarm & pos).sum())
    fp = int((alarm & neg).sum())
    return {
        "threshold": float(threshold) if np.isfinite(threshold) else None,
        "recall": (tp / n_pos) if n_pos else float("nan"),
        "false_alarm_rate": (fp / n_neg) if n_neg else float("nan"),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }
