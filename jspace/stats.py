"""Shared statistical helpers for the J-lens doom-loop experiments.

Pure numpy + Python stdlib on purpose: these functions must run identically on
the Windows laptop, in Colab, and inside CPU-only unit tests (no torch, no
scipy dependency).

Typical use:
    from jspace.stats import bootstrap_ci, clustered_bootstrap_ci, permutation_test, mcnemar_exact

    point, lo, hi = bootstrap_ci(trigger_alignments, n_boot=2000, seed=0)
    ci = clustered_bootstrap_ci(scenario_ids, auroc_on_indices, n_boot=2000, seed=0)
    p = permutation_test(loop_occupancies, nonloop_occupancies, n_perm=10000)
    res = mcnemar_exact(b=2, c=9)  # discordant pairs of a paired 2x2 table
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Union

import numpy as np

__all__ = [
    "bootstrap_ci",
    "clustered_bootstrap_ci",
    "permutation_test",
    "paired_permutation_test",
    "mcnemar_exact",
    "cohens_h",
    "wilson_ci",
    "mcnemar_power",
    "mcnemar_sample_size",
]

StatSpec = Union[None, str, Callable[[np.ndarray], float]]


def _as_clean_1d(values) -> np.ndarray:
    x = np.asarray(values, dtype=float).ravel()
    return x[~np.isnan(x)]


def _log_binom_pmf_row(d: int, p: float, lgam: np.ndarray) -> np.ndarray:
    """Log-pmf of Binom(d, p) for k = 0..d, via a precomputed lgamma table.

    Log-space keeps this finite for large d where C(d, d/2) overflows float64.
    """
    ks = np.arange(d + 1)
    if p <= 0.0:
        row = np.full(d + 1, -np.inf)
        row[0] = 0.0
        return row
    if p >= 1.0:
        row = np.full(d + 1, -np.inf)
        row[d] = 0.0
        return row
    return (
        lgam[d + 1]
        - lgam[ks + 1]
        - lgam[d - ks + 1]
        + ks * math.log(p)
        + (d - ks) * math.log1p(-p)
    )


def _normalized_pmf(log_row: np.ndarray) -> np.ndarray:
    """Exponentiate a log-pmf row safely and renormalize to sum exactly 1."""
    m = float(np.max(log_row))
    if not np.isfinite(m):
        raise ValueError("log-pmf row has no finite entry")
    w = np.exp(log_row - m)
    return w / w.sum()


def _apply_stat(x: np.ndarray, stat: StatSpec) -> float:
    if stat in (None, "mean", np.mean):
        return float(np.mean(x))
    if stat in ("median", np.median):
        return float(np.median(x))
    return float(stat(x))


def bootstrap_ci(
    values: Sequence[float],
    stat: StatSpec = "mean",
    n_boot: int = 2000,
    ci: float = 95.0,
    seed: int = 0,
):
    """Bootstrap confidence interval for ``stat`` over a 1-D sample.

    Resamples the observations with replacement, so it is the right tool for
    "mean over N trigger tokens" style quantities. Returns
    ``(point_estimate, lower, upper)``; NaN triplet for empty input.
    """
    x = _as_clean_1d(values)
    if x.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = _apply_stat(x, stat)
    rng = np.random.default_rng(seed)
    n = x.size
    if stat in (None, "mean", np.mean, "median", np.median):
        idx = rng.integers(0, n, size=(n_boot, n))
        reps = x[idx]
        reps = reps.mean(axis=1) if stat in (None, "mean", np.mean) else np.median(reps, axis=1)
    else:
        reps = np.array(
            [_apply_stat(x[rng.integers(0, n, size=n)], stat) for _ in range(n_boot)]
        )
    alpha = (100.0 - ci) / 2.0
    lo, hi = np.percentile(reps, [alpha, 100.0 - alpha])
    return (point, float(lo), float(hi))


def clustered_bootstrap_ci(
    cluster_ids: Sequence,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    ci: float = 95.0,
    seed: int = 0,
) -> dict:
    """Bootstrap that resamples clusters and keeps every row of a drawn cluster.

    A cluster drawn twice contributes its rows twice. ``stat_fn`` receives the
    integer positions into the original rows (duplicates included). Non-finite
    replicates are dropped before the percentile interval. Returns a dict with
    ``point``, ``lo``, ``hi``, ``n_boot_finite``, ``n_boot`` and ``ci``.

    LeanScope passes ``scenario_id`` as ``cluster_ids`` so paraphrases and
    option orders of one scenario stay together. This is the same resampling
    scheme as the forecasting branch's prompt-clustered bootstrap (that module
    is not on this branch); the implementation here stays numpy-only, like the
    rest of ``jspace.stats``.
    """
    cluster_ids = np.asarray(cluster_ids)
    empty = {
        "point": float("nan"),
        "lo": float("nan"),
        "hi": float("nan"),
        "n_boot_finite": 0,
        "n_boot": int(n_boot),
        "ci": float(ci),
    }
    if cluster_ids.size == 0:
        return empty
    _uniq, inverse = np.unique(cluster_ids, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(int(_uniq.size))]

    def _call(idx: np.ndarray) -> float:
        try:
            return float(stat_fn(np.asarray(idx, dtype=int)))
        except Exception:
            return float("nan")

    point = _call(np.arange(cluster_ids.size))
    rng = np.random.default_rng(seed)
    n_u = len(groups)
    reps = np.empty(int(n_boot), dtype=float)
    draws = rng.integers(0, n_u, size=(int(n_boot), n_u))
    for b in range(int(n_boot)):
        idx = np.concatenate([groups[int(j)] for j in draws[b]])
        reps[b] = _call(idx)
    finite = reps[np.isfinite(reps)]
    if finite.size == 0 or not np.isfinite(point):
        lo = hi = float("nan")
    else:
        alpha = (100.0 - float(ci)) / 2.0
        lo_v, hi_v = np.percentile(finite, [alpha, 100.0 - alpha])
        lo, hi = float(lo_v), float(hi_v)
    return {
        "point": point if np.isfinite(point) else float("nan"),
        "lo": lo,
        "hi": hi,
        "n_boot_finite": int(finite.size),
        "n_boot": int(n_boot),
        "ci": float(ci),
    }


def permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    stat: Optional[Callable] = None,
    n_perm: int = 10000,
    seed: int = 0,
) -> float:
    """Two-sample permutation p-value for mean(a) - mean(b).

    Pools the samples, shuffles labels ``n_perm`` times and returns the
    two-sided fraction of permuted differences at least as extreme as the
    observed one (with the +1 correction). NaN if either sample is empty.
    """
    a = _as_clean_1d(a)
    b = _as_clean_1d(b)
    if a.size == 0 or b.size == 0:
        return float("nan")
    pooled = np.concatenate([a, b])
    na = a.size
    if stat is None:
        obs = float(np.mean(a) - np.mean(b))
        rng = np.random.default_rng(seed)
        perm_idx = np.argsort(rng.random((n_perm, pooled.size)), axis=1)
        reps = pooled[perm_idx]
        diffs = reps[:, :na].mean(axis=1) - reps[:, na:].mean(axis=1)
        p = (np.sum(np.abs(diffs) >= np.abs(obs) - 1e-12) + 1) / (n_perm + 1)
        return float(p)
    obs = float(stat(a, b))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(pooled)
        count += abs(stat(shuffled[:na], shuffled[na:]) - obs) >= 1e-12
    return float((count + 1) / (n_perm + 1))


def paired_permutation_test(
    x: Sequence[float],
    y: Sequence[float],
    n_perm: int = 10000,
    seed: int = 0,
) -> float:
    """Paired permutation test on d = x - y using sign flips (two-sided)."""
    d = _as_clean_1d(np.asarray(x, dtype=float) - np.asarray(y, dtype=float))
    if d.size == 0:
        return float("nan")
    obs = abs(float(np.mean(d)))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    reps = np.abs((d[None, :] * signs).mean(axis=1))
    p = (np.sum(reps >= obs - 1e-12) + 1) / (n_perm + 1)
    return float(p)


def mcnemar_exact(b: int, c: int) -> dict:
    """Exact McNemar test on the discordant counts of a paired 2x2 table.

    ``b`` = pairs where baseline loops and treatment does not;
    ``c`` = pairs where baseline does not loop and treatment loops.
    Returns counts plus the two-sided exact p-value (binomial).
    """
    b, c = int(b), int(c)
    n = b + c
    out = {"b": b, "c": c, "n_discordant": n}
    if n == 0:
        out["p_exact"] = float("nan")
        out["odds_ratio_b_over_c"] = float("nan")
        return out
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    out["p_exact"] = float(min(1.0, 2.0 * tail))
    if c == 0:
        out["odds_ratio_b_over_c"] = float("inf") if b > 0 else float("nan")
    else:
        out["odds_ratio_b_over_c"] = float(b / c)
    return out


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return float(2.0 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2))))


def wilson_ci(k: int, n: int, z: float = 1.959963984540054):
    """Wilson score interval for k successes out of n (default z = 95%)."""
    k, n = int(k), int(n)
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_power(p10: float, p01: float, n: int, alpha: float = 0.05) -> float:
    """Exact power of the exact McNemar test under a specified alternative.

    ``p10`` / ``p01`` are per-pair probabilities of the two discordant
    outcomes. Conditions on the number of discordant pairs D ~ Binom(n, pd)
    and averages the conditional exact-test power over its distribution.
    """
    if not (0.0 <= p10 <= 1.0 and 0.0 <= p01 <= 1.0):
        raise ValueError("p10/p01 must be probabilities")
    if p10 + p01 > 1.0 + 1e-12:
        raise ValueError("p10 + p01 must be <= 1")
    pd = p10 + p01
    n = int(n)
    if n <= 0:
        return float("nan")
    if pd <= 0.0:
        return 0.0
    q = p10 / pd

    lgam = np.fromiter(
        (0.0 if i == 0 else math.lgamma(i) for i in range(n + 2)),
        dtype=float,
        count=n + 2,
    )
    p_d = _normalized_pmf(_log_binom_pmf_row(n, pd, lgam))

    power = 0.0
    for d in range(n + 1):
        w = p_d[d]
        if w < 1e-14:
            continue
        null = _normalized_pmf(_log_binom_pmf_row(d, 0.5, lgam))
        lower = np.cumsum(null)
        upper = np.cumsum(null[::-1])[::-1]
        pvals = np.minimum(1.0, 2.0 * np.minimum(lower, upper))
        reject = pvals <= alpha
        if not reject.any():
            continue
        alt = _normalized_pmf(_log_binom_pmf_row(d, q, lgam))
        power += w * float(alt[reject].sum())
    return float(power)


def mcnemar_sample_size(
    p10: float,
    p01: float,
    power_target: float = 0.8,
    alpha: float = 0.05,
    n_min: int = 10,
    n_max: int = 2000,
) -> Optional[int]:
    """Smallest n (pairs) reaching ``power_target`` for the exact McNemar test.

    Returns None if even ``n_max`` pairs do not reach the target power.
    """
    if mcnemar_power(p10, p01, n_max, alpha) < power_target:
        return None
    lo, hi = n_min, n_max
    while lo < hi:
        mid = (lo + hi) // 2
        if mcnemar_power(p10, p01, mid, alpha) >= power_target:
            hi = mid
        else:
            lo = mid + 1
    return int(lo)
