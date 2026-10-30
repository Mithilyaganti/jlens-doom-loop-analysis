"""Inference-time anti-doom sidecar.

Default off (``JLENS_ANTIDOOM_SIDECAR=0``). The scientific A1 baseline must
stay unsteered so its loop rate is comparable to Liquid's greedy measurement.

When enabled, this HuggingFace logits processor watches the generated token
ids. Once a period-p pattern has already been emitted ``min_repeats - 1``
times back-to-back (default 3 copies, so the 4th copy is the one we block —
the same repeat count as Liquid's detector), it subtracts a large bias from
the single token that would extend the pattern. Greedy decoding then steps
off the loop. No training, no J-lens, no weight change.

This is not the Exp3 surgical ablation and it is not a substitute for the
unsteered baseline. A run with the sidecar on is an intervention condition;
``scripts/02_baseline_pass.py`` records that in the summary and in every
trace meta.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

import torch

logger = logging.getLogger(__name__)

try:
    from transformers import LogitsProcessor as _PROCESSOR_BASE
except Exception:  # pragma: no cover
    _PROCESSOR_BASE = object  # type: ignore[assignment,misc]

DEFAULT_MIN_REPEATS = 4
DEFAULT_MAX_PERIOD = 32
DEFAULT_BIAS = -1.0e4


def sidecar_enabled() -> bool:
    """JLENS_ANTIDOOM_SIDECAR. Default off so baseline loop rates stay honest."""
    raw = os.environ.get("JLENS_ANTIDOOM_SIDECAR", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def sidecar_min_repeats() -> int:
    try:
        n = int(os.environ.get("JLENS_SIDECAR_MIN_REPEATS", str(DEFAULT_MIN_REPEATS)))
    except ValueError:
        n = DEFAULT_MIN_REPEATS
    return max(2, n)


def sidecar_max_period() -> int:
    try:
        n = int(os.environ.get("JLENS_SIDECAR_MAX_PERIOD", str(DEFAULT_MAX_PERIOD)))
    except ValueError:
        n = DEFAULT_MAX_PERIOD
    return max(1, n)


def sidecar_bias() -> float:
    try:
        return float(os.environ.get("JLENS_SIDECAR_BIAS", str(DEFAULT_BIAS)))
    except ValueError:
        return DEFAULT_BIAS


def continuation_to_block(
    token_ids: Sequence[int],
    *,
    min_repeats: int = DEFAULT_MIN_REPEATS,
    max_period: int = DEFAULT_MAX_PERIOD,
) -> int | None:
    """Token id that would extend a tail already repeated ``min_repeats - 1`` times.

    Shortest period wins, so a unigram loop is caught before a longer phrase
    that happens to contain it. Returns None when the tail is not periodic
    enough to count as a forming doom loop.

    The window is taken from the end of ``token_ids`` and must be an integer
    number of periods, so the next token (one period back) is ``ids[-period]``.
    """
    if min_repeats < 2:
        raise ValueError("min_repeats must be >= 2")
    if max_period < 1:
        raise ValueError("max_period must be >= 1")
    ids = [int(t) for t in token_ids]
    n = len(ids)
    reps_needed = min_repeats - 1
    if n < reps_needed:
        return None
    limit = min(max_period, n // reps_needed)
    for period in range(1, limit + 1):
        window = period * reps_needed
        tail = ids[-window:]
        if any(tail[i] != tail[i % period] for i in range(window)):
            continue
        return ids[-period]
    return None


class AntiDoomSidecar(_PROCESSOR_BASE):  # type: ignore[misc,valid-type]
    """Block the token that would continue a repeated n-gram.

    HF ``generate`` calls this with the full growing ``input_ids``. Tokens
    before ``prompt_len`` are the prompt and never count as a loop. When
    ``prompt_len`` is None, the first call's sequence length is the prompt
    (HF passes exactly the prompt at step 0).

    ``n_suppressed_steps`` counts (row, step) pairs where a token was biased.
    ``blocked_token_ids`` records those token ids in call order.
    """

    def __init__(
        self,
        *,
        prompt_len: int | None = None,
        min_repeats: int | None = None,
        max_period: int | None = None,
        bias: float | None = None,
    ) -> None:
        self.prompt_len = None if prompt_len is None else int(prompt_len)
        self.min_repeats = sidecar_min_repeats() if min_repeats is None else int(min_repeats)
        self.max_period = sidecar_max_period() if max_period is None else int(max_period)
        self.bias = sidecar_bias() if bias is None else float(bias)
        self._first_len: int | None = None
        self.n_suppressed_steps = 0
        self.blocked_token_ids: list[int] = []

    def _prompt_len(self, seq_len: int) -> int:
        if self.prompt_len is not None:
            return self.prompt_len
        if self._first_len is None:
            self._first_len = seq_len
        return self._first_len

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        squeeze_scores = scores.ndim == 1
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if squeeze_scores:
            scores = scores.unsqueeze(0)
        if scores.shape[0] != input_ids.shape[0]:
            raise ValueError(
                f"sidecar batch mismatch: input_ids {tuple(input_ids.shape)} vs scores {tuple(scores.shape)}"
            )

        plen = self._prompt_len(int(input_ids.shape[-1]))
        blocked: list[tuple[int, int]] = []
        for row in range(input_ids.shape[0]):
            gen = input_ids[row, plen:].detach().cpu().tolist()
            token = continuation_to_block(
                gen, min_repeats=self.min_repeats, max_period=self.max_period
            )
            if token is None:
                continue
            if token < 0 or token >= scores.shape[-1]:
                logger.warning("sidecar token id %s outside vocab %s; skipping", token, scores.shape[-1])
                continue
            blocked.append((row, token))

        if not blocked:
            return scores.squeeze(0) if squeeze_scores else scores

        out = scores.clone()
        for row, token in blocked:
            out[row, token] = out[row, token] + self.bias
            self.n_suppressed_steps += 1
            self.blocked_token_ids.append(token)
        return out.squeeze(0) if squeeze_scores else out


def sidecar_from_env(*, prompt_len: int | None = None) -> AntiDoomSidecar | None:
    """A fresh processor when the sidecar is enabled, else None.

    One instance per prompt: the prompt-length snapshot and the suppression
    counters must not leak across traces.
    """
    if not sidecar_enabled():
        return None
    logger.info(
        "Anti-doom sidecar ON (min_repeats=%d max_period=%d). "
        "This generation is an intervention, not the unsteered baseline.",
        sidecar_min_repeats(),
        sidecar_max_period(),
    )
    return AntiDoomSidecar(prompt_len=prompt_len)
