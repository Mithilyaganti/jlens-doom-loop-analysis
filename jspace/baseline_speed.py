"""T4 baseline speedups that do not change the unsteered loop label.

Two knobs, both read by ``scripts/02_baseline_pass.py``:

* ``JLENS_BASELINE_BATCH_SIZE`` (default 4) — vLLM continuous-batching chunk.
  A Colab T4 cannot hold 200 sequences of length 4000, and one giant call
  loses every in-flight prompt if the runtime disconnects. ``0`` / ``all``
  keeps the legacy single-call behavior.
* ``JLENS_BASELINE_EARLY_STOP`` (default on) — HuggingFace stopping criterion
  that ends a trace once Liquid's character detector has already confirmed a
  loop. The 4000-token cap is unchanged. Non-looping traces still run to EOS
  or the cap, so a loop label cannot flip from true to false: we only stop
  after the same detector returns a hit. Checks run every
  ``JLENS_EARLY_STOP_EVERY`` new tokens (default 32) because decoding the
  whole string every step costs more than a few extra decode steps.

The anti-doom sidecar is a separate, default-off intervention. Early stop
does not steer the model.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Sequence

import torch

from jspace.detection import find_inner_repetition

logger = logging.getLogger(__name__)

try:  # HF base class when transformers is installed; duck-typed in tiny tests.
    from transformers import StoppingCriteria as _STOP_BASE
except Exception:  # pragma: no cover
    _STOP_BASE = object  # type: ignore[assignment,misc]


def early_stop_enabled() -> bool:
    """Baseline HF early-stop. Default on. Set JLENS_BASELINE_EARLY_STOP=0 to disable."""
    raw = os.environ.get("JLENS_BASELINE_EARLY_STOP", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def early_stop_every() -> int:
    """How many new tokens between character-detector checks."""
    try:
        n = int(os.environ.get("JLENS_EARLY_STOP_EVERY", "32"))
    except ValueError:
        n = 32
    return max(1, n)


def resolve_batch_chunk_size(n_pending: int | None = None) -> int:
    """vLLM prompts per ``generate_batch`` call.

    ``JLENS_BASELINE_BATCH_SIZE=0`` or ``all`` means one call for every
    pending prompt (the pre-chunk behavior). A positive integer is the
    T4-sized chunk. Invalid values fall back to 4.
    """
    raw = os.environ.get("JLENS_BASELINE_BATCH_SIZE", "4").strip().lower()
    if raw in ("0", "all", "none"):
        return int(n_pending or 0)
    try:
        size = int(raw)
    except ValueError:
        logger.warning("Bad JLENS_BASELINE_BATCH_SIZE=%r; using 4", raw)
        size = 4
    if size <= 0:
        return int(n_pending or 0)
    return size


def chunk_pending(items: Sequence[Any], chunk_size: int) -> list[list[Any]]:
    """Split ``items`` into contiguous chunks. Non-positive size = one chunk."""
    seq = list(items)
    if not seq:
        return []
    if chunk_size is None or chunk_size <= 0 or chunk_size >= len(seq):
        return [seq]
    return [seq[i : i + chunk_size] for i in range(0, len(seq), chunk_size)]


class LoopConfirmedStoppingCriteria(_STOP_BASE):  # type: ignore[misc,valid-type]
    """Stop a row once ``find_inner_repetition`` confirms a doom loop.

    ``prompt_len`` is the tokenized prompt length for an unpadded batch
    (the baseline generates one prompt at a time). Checks happen every
    ``check_every`` generated tokens. A row that has stopped stays stopped.
    """

    def __init__(
        self,
        tokenizer: Any,
        prompt_len: int,
        *,
        check_every: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_len)
        self.check_every = int(check_every) if check_every is not None else early_stop_every()
        if self.check_every < 1:
            self.check_every = 1
        self._stopped_rows: set[int] = set()
        self.hit: Any = None
        self.stopped_at: int | None = None

    @property
    def stopped(self) -> bool:
        return bool(self._stopped_rows)

    def _row_should_stop(self, row_ids: torch.Tensor, row_index: int) -> bool:
        if row_index in self._stopped_rows:
            return True
        n_gen = int(row_ids.shape[-1]) - self.prompt_len
        if n_gen <= 0 or n_gen % self.check_every != 0:
            return False
        gen_ids = row_ids[self.prompt_len :].detach().cpu().tolist()
        try:
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        except Exception:
            logger.exception("early-stop decode failed; not stopping")
            return False
        found, hit = find_inner_repetition(text)
        if not found:
            return False
        self._stopped_rows.add(row_index)
        self.hit = hit
        self.stopped_at = n_gen
        return True

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        stops = [
            self._row_should_stop(input_ids[b], b) for b in range(input_ids.shape[0])
        ]
        return torch.tensor(stops, dtype=torch.bool, device=input_ids.device)
