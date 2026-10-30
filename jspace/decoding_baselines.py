"""HF decoding baselines for Exp3, beside the PR #2 anti-doom sidecar.

PR #2 (draft) adds an opt-in logits processor, ``JLENS_ANTIDOOM_SIDECAR``,
that blocks the token which would extend an n-gram already repeated three
times. This module does not reimplement that processor. It wires two
different Hugging Face ``generate`` controls:

- ``repetition_penalty`` — scalar penalty on tokens that already appeared
- ``no_repeat_ngram`` — ``no_repeat_ngram_size``, which blocks any n-gram
  that has already occurred once

Both are off unless ``JLENS_EXP3_DECODING_BASELINES=1``. The penalty and
n-gram size below are fixed wiring defaults. They were not tuned, and no
generation metric was measured for them.
"""

from __future__ import annotations

import os
from typing import Mapping, Sequence

# Untuned. 1.0 is the Hugging Face no-op for repetition_penalty, so the arm
# has to be something else. 1.2 is only a fixed setting, not a fitted value.
REPETITION_PENALTY: float = 1.2
# Untuned. This is not the sidecar's "already repeated 3 times" rule.
NO_REPEAT_NGRAM_SIZE: int = 4

REPETITION_PENALTY_CONDITION = "repetition_penalty"
NO_REPEAT_NGRAM_CONDITION = "no_repeat_ngram"
DECODING_BASELINE_CONDITIONS: tuple[str, ...] = (
    REPETITION_PENALTY_CONDITION,
    NO_REPEAT_NGRAM_CONDITION,
)

ENV_FLAG = "JLENS_EXP3_DECODING_BASELINES"


def decoding_baselines_enabled(env: Mapping[str, str] | None = None) -> bool:
    """True only when the flag is the string ``1``."""
    source = os.environ if env is None else env
    return source.get(ENV_FLAG, "0") == "1"


def extend_condition_names(
    names: Sequence[str],
    *,
    enabled: bool | None = None,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Append the two arms when enabled. Does not drop or reorder ``names``."""
    out = list(names)
    if enabled is None:
        enabled = decoding_baselines_enabled(env)
    if not enabled:
        return out
    for name in DECODING_BASELINE_CONDITIONS:
        if name not in out:
            out.append(name)
    return out


def hf_overrides_for_condition(condition: str) -> dict[str, float | int] | None:
    """HF generate overrides for a decoding-baseline condition, else None.

    The returned dict is the only place the untuned constants enter a
    generate call. It does not contain a loop rate.
    """
    if condition == REPETITION_PENALTY_CONDITION:
        return {"repetition_penalty": REPETITION_PENALTY}
    if condition == NO_REPEAT_NGRAM_CONDITION:
        return {"no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE}
    return None


def apply_hf_overrides(
    base: Mapping[str, object],
    condition: str,
) -> dict[str, object]:
    """Copy ``base`` and set the override keys for ``condition``."""
    overrides = hf_overrides_for_condition(condition)
    if overrides is None:
        raise KeyError(f"{condition} is not a decoding-baseline arm")
    merged = dict(base)
    merged.update(overrides)
    return merged


def requires_hf_backend(condition: str) -> bool:
    """These arms are HF generate kwargs. The vLLM path must refuse them."""
    return hf_overrides_for_condition(condition) is not None
