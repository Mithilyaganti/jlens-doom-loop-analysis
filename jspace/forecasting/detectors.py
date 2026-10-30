"""Trace-only detectors and the CUSUM harness.

The n-gram rule and the surface-feature model read token ids (and, when the
baseline trace stored them, token strings). They do not need hidden states.

``cusum_statistic`` accepts any causal per-token score. A circular-reasoning
probe or a later J-lens feature is the same shape as the n-gram score: one
finite float per generated token, using only tokens up to that position.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from jspace.detection import DEFAULT_REPETITION_KWARGS, find_inner_repetition

# Local names so the online rule cannot drift from the character detector.
MIN_REPEATS = int(DEFAULT_REPETITION_KWARGS["min_repeats"])
MIN_TOTAL_REPEATED = int(DEFAULT_REPETITION_KWARGS["min_total_repeated"])
MAX_NGRAM_PERIOD = 128


def online_ngram_repeat_scores(
    token_ids: Sequence[int],
    *,
    max_period: int = MAX_NGRAM_PERIOD,
    min_repeats: int = MIN_REPEATS,
    token_char_lens: Sequence[int] | None = None,
    min_total_repeated: int = MIN_TOTAL_REPEATED,
) -> np.ndarray:
    """Causal repetition score in ``[0, 1]``, one value per token.

    At position ``t`` the score looks only at ``token_ids[: t + 1]``. For each
    period ``p`` it counts consecutive copies of the length-``p`` block ending
    at ``t``. The score is ``repeats / min_repeats``, clipped to 1. When
    per-token character lengths are provided it is also capped by
    ``repeated_chars / min_total_repeated``, which is the Antidoom length rule
    (60 characters). A score of 1 is four repeats, and 60 characters when
    lengths are known.

    Periods longer than ``max_period`` are ignored. 128 tokens covers the
    short loops the character detector is aimed at; it is not a second copy of
    the character scanner's 1024-character period cap.
    """
    ids = np.asarray(list(token_ids), dtype=np.int64)
    n = int(ids.size)
    scores = np.zeros(n, dtype=np.float64)
    if n == 0:
        return scores
    if min_repeats < 2:
        raise ValueError("min_repeats must be >= 2")
    run = np.zeros(max_period + 1, dtype=np.int32)
    prefix_chars = None
    if token_char_lens is not None:
        char_lens = np.asarray(list(token_char_lens), dtype=np.int64)
        if char_lens.shape[0] != n:
            raise ValueError("token_char_lens length must match token_ids")
        prefix_chars = np.cumsum(char_lens)
    for t in range(n):
        best = 0.0
        p_hi = min(max_period, t)
        tok = ids[t]
        for p in range(1, p_hi + 1):
            if tok == ids[t - p]:
                run[p] += 1
            else:
                run[p] = 0
            if run[p] < p:
                continue
            reps = int(run[p] // p) + 1
            frac = reps / float(min_repeats)
            if prefix_chars is not None:
                span = reps * p
                if span > t + 1:
                    span = t + 1
                end_c = int(prefix_chars[t])
                start_i = t + 1 - span
                start_c = 0 if start_i <= 0 else int(prefix_chars[start_i - 1])
                chars = end_c - start_c
                frac = min(frac, chars / float(min_total_repeated))
            if frac > best:
                best = frac
        scores[t] = 1.0 if best >= 1.0 else best
    return scores


def antidoom_confirmation_char(text: str) -> int | None:
    """Smallest prefix length on which the character detector fires.

    Detection is monotonic for this scanner: sample positions only grow as the
    prefix grows, and a hit already inside the prefix is still inside the
    longer string. The search is binary. The returned length is the first
    character count at which the rule is satisfied; the alarm is after onset
    whenever the rule needs four copies to fire.
    """
    if not text:
        return None
    found, _hit = find_inner_repetition(text)
    if not found:
        return None
    lo, hi = 1, len(text)
    answer = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        if find_inner_repetition(text[:mid])[0]:
            answer = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return int(answer)


def cusum_statistic(scores: Sequence[float], *, mu: float = 0.0, allowance: float = 0.0) -> np.ndarray:
    """Upper tabular CUSUM of a per-token score.

    ``S_t = max(0, S_{t-1} + x_t - mu - allowance)``, with ``S`` starting at 0.
    ``x_t`` is whatever probe produced the series (n-gram score, a
    circular-reasoning probe, a J-lens coordinate). Non-finite scores add
    nothing. The statistic at ``t`` depends only on ``x[: t + 1]``.
    """
    x = np.asarray(list(scores), dtype=float).ravel()
    out = np.zeros(x.size, dtype=np.float64)
    shift = float(mu) + float(allowance)
    state = 0.0
    for i, value in enumerate(x):
        increment = float(value) if np.isfinite(value) else 0.0
        state = max(0.0, state + increment - shift)
        out[i] = state
    return out


def fit_cusum_in_control(series_list: Sequence[Sequence[float]]) -> dict:
    """In-control reference from non-loop token scores.

    ``mu`` is the mean token score. ``allowance`` is half a standard deviation,
    the usual tabular-CUSUM slack. Both are fit on the series passed in, which
    the caller must restrict to the training non-loop traces.
    """
    chunks = []
    for series in series_list:
        arr = np.asarray(list(series), dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            chunks.append(arr)
    if not chunks:
        return {"mu": 0.0, "allowance": 0.0, "sigma": 0.0, "n_tokens": 0}
    x = np.concatenate(chunks)
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    return {
        "mu": mu,
        "allowance": 0.5 * sigma,
        "sigma": sigma,
        "n_tokens": int(x.size),
    }
