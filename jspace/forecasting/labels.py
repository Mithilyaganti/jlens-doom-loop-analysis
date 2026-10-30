"""Loop-onset labels and length-and-position matched control windows.

Onset is the Antidoom trigger: the first token of the first repeated span.
The character rule is the vendored detector (at least 4 repeats and at least
60 repeated characters). A horizon-H example is the window of ``window_len``
tokens ending at index ``onset - H`` (inclusive). That window does not include
the onset token.

Controls are non-loop traces in the same hash-locked split. The window length
matches. The decision index matches absolutely when the control is long enough,
and otherwise matches the fractional position of the decision index in the
loop trace.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from jspace.detection import decode_token, detect_loop, find_inner_repetition
from jspace.forecasting.detectors import antidoom_confirmation_char
from jspace.forecasting.metrics import SPLIT_SALT, assign_split

HORIZONS = (32, 64, 128, 256)
DEFAULT_WINDOW = 64


@dataclass
class PreparedTrace:
    prompt_id: int
    token_ids: list[int]
    token_strings: list[str] | None
    text: str
    aligned: bool
    meta: dict
    source: str | None
    tokenizer: object | None = field(default=None, repr=False)


@dataclass
class OnsetLabel:
    prompt_id: int
    is_loop: bool
    onset_token_index: int | None
    onset_source: str
    n_tokens: int
    split: str
    period: int | None = None
    repeats: int | None = None
    start_char: int | None = None
    repeat_start_char: int | None = None
    end_char: int | None = None
    text_complete: bool = True
    disagrees_with_stored: bool = False
    stored_onset_token_index: int | None = None
    onset_index_mismatch: bool = False
    dataset_source: str | None = None
    confirmation_char: int | None = None
    confirmation_token_index: int | None = None


@dataclass(frozen=True)
class HorizonExample:
    prompt_id: int
    split: str
    horizon: int
    role: str
    y: int
    decision_index: int
    length: int
    onset_index: int | None
    matched_prompt_id: int | None
    match_kind: str | None
    dataset_source: str | None


class _AlignTokenizer:
    """Map the token ids we already have back to the strings that built ``text``."""

    def __init__(self, ids: list[int], raw_strings: list[str]):
        self._map = {int(i): s for i, s in zip(ids, raw_strings)}

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, (int, np.integer)):
            return self._map[int(ids)]
        return [self._map[int(i)] for i in ids]


def _char_to_token(decoded: list[str], char_pos: int) -> int | None:
    if char_pos < 0:
        return None
    running = 0
    for i, piece in enumerate(decoded):
        running += len(piece)
        if char_pos < running:
            return i
    return None


def _stored_onset(meta: dict, n_tokens: int) -> int | None:
    raw = meta.get("trigger_token_index")
    if raw is None:
        raw = meta.get("loop_trigger_pos")
    if raw is None:
        return None
    try:
        idx = int(raw)
    except (TypeError, ValueError):
        return None
    if n_tokens <= 0 or not (0 <= idx < n_tokens):
        return None
    return idx


def label_trace(trace: PreparedTrace) -> OnsetLabel:
    """Re-apply the character rule, then map the hit onto a token index."""
    meta = trace.meta or {}
    n_tokens = len(trace.token_ids)
    text = trace.text or ""
    found, hit = find_inner_repetition(text) if text else (False, None)
    is_loop = bool(found and hit is not None)
    stored_present = "is_loop" in meta and meta.get("is_loop") is not None
    disagrees = bool(stored_present and bool(meta.get("is_loop")) != is_loop)
    stored_idx = _stored_onset(meta, n_tokens)

    mapped: int | None = None
    if is_loop and trace.token_ids:
        tokenizer = None
        if trace.tokenizer is not None:
            tokenizer = trace.tokenizer
        elif trace.aligned and trace.token_strings is not None:
            tokenizer = _AlignTokenizer(trace.token_ids, trace.token_strings)
        if tokenizer is not None:
            loop = detect_loop(text, token_ids=list(trace.token_ids), tokenizer=tokenizer, prompt="")
            if loop.trigger_token_index is not None and 0 <= int(loop.trigger_token_index) < n_tokens:
                mapped = int(loop.trigger_token_index)

    if is_loop and mapped is not None:
        onset = mapped
        source = "detector_token_map"
    elif is_loop and stored_idx is not None and bool(meta.get("is_loop")):
        onset = stored_idx
        source = "stored_trigger_index"
    else:
        onset = None
        source = "unavailable" if is_loop else "not_a_loop"

    confirmation_char = antidoom_confirmation_char(text) if is_loop else None
    confirmation_token = None
    if confirmation_char is not None and trace.aligned and trace.token_strings is not None:
        decoded = [decode_token(s) for s in trace.token_strings]
        confirmation_token = _char_to_token(decoded, confirmation_char - 1)

    return OnsetLabel(
        prompt_id=int(trace.prompt_id),
        is_loop=is_loop,
        onset_token_index=onset,
        onset_source=source,
        n_tokens=n_tokens,
        split=assign_split(int(trace.prompt_id)),
        period=None if hit is None else int(hit.period),
        repeats=None if hit is None else int(hit.repeats),
        start_char=None if hit is None else int(hit.start),
        repeat_start_char=None if hit is None else int(hit.repeat_start),
        end_char=None if hit is None else int(hit.end),
        text_complete=True,
        disagrees_with_stored=disagrees,
        stored_onset_token_index=stored_idx,
        onset_index_mismatch=bool(mapped is not None and stored_idx is not None and mapped != stored_idx),
        dataset_source=trace.source,
        confirmation_char=confirmation_char,
        confirmation_token_index=confirmation_token,
    )


def _depth_decision(loop_n: int, decision_index: int, control_n: int, window_len: int) -> int:
    depth = decision_index / max(loop_n - 1, 1)
    decision = int(round(depth * (control_n - 1)))
    return min(max(decision, window_len - 1), control_n - 1)


def build_horizon_examples(
    labels: list[OnsetLabel],
    *,
    horizons: tuple[int, ...] = HORIZONS,
    window_len: int = DEFAULT_WINDOW,
    salt: str = SPLIT_SALT,
) -> tuple[list[HorizonExample], dict]:
    """Positive windows before onset, plus one matched non-loop window each.

    Matching stays inside a split so a test trace is never a training control.
    """
    by_id = {lb.prompt_id: lb for lb in labels}
    controls = [lb for lb in labels if (not lb.is_loop) and lb.n_tokens > 0]
    positives: list[HorizonExample] = []
    ineligible = {int(h): 0 for h in horizons}
    for lb in labels:
        if not lb.is_loop or lb.onset_token_index is None:
            continue
        onset = int(lb.onset_token_index)
        for horizon in horizons:
            decision = onset - int(horizon)
            start = decision - window_len + 1
            if decision < 0 or start < 0 or decision >= lb.n_tokens:
                ineligible[int(horizon)] += 1
                continue
            positives.append(
                HorizonExample(
                    prompt_id=lb.prompt_id,
                    split=lb.split,
                    horizon=int(horizon),
                    role="pre_onset",
                    y=1,
                    decision_index=decision,
                    length=window_len,
                    onset_index=onset,
                    matched_prompt_id=None,
                    match_kind=None,
                    dataset_source=lb.dataset_source,
                )
            )

    examples: list[HorizonExample] = list(positives)
    unmatched = 0
    used: set[tuple] = set()
    ordered = sorted(positives, key=lambda w: (w.split, w.horizon, w.prompt_id))
    for window in ordered:
        absolute = [
            c
            for c in controls
            if c.split == window.split and c.prompt_id != window.prompt_id and c.n_tokens > window.decision_index
        ]
        depth_pool = [
            c
            for c in controls
            if c.split == window.split and c.prompt_id != window.prompt_id and c.n_tokens >= window_len
        ]
        if absolute:
            kind = "absolute_position"
            pool = sorted(absolute, key=lambda c: c.prompt_id)
        elif depth_pool:
            kind = "normalized_depth"
            pool = sorted(depth_pool, key=lambda c: c.prompt_id)
        else:
            unmatched += 1
            continue
        digest = hashlib.sha256(f"{salt}:{window.prompt_id}:{window.horizon}:{kind}".encode("utf-8")).digest()
        start_i = int.from_bytes(digest[:8], "big") % len(pool)
        chosen = None
        chosen_decision = None
        reused = False
        for shift in range(len(pool)):
            cand = pool[(start_i + shift) % len(pool)]
            if kind == "absolute_position":
                decision = window.decision_index
            else:
                decision = _depth_decision(by_id[window.prompt_id].n_tokens, window.decision_index, cand.n_tokens, window_len)
            key = (window.split, cand.prompt_id, decision, window.length)
            if key not in used:
                used.add(key)
                chosen = cand
                chosen_decision = decision
                break
        if chosen is None:
            reused = True
            chosen = pool[start_i]
            if kind == "absolute_position":
                chosen_decision = window.decision_index
            else:
                chosen_decision = _depth_decision(
                    by_id[window.prompt_id].n_tokens, window.decision_index, chosen.n_tokens, window_len
                )
        assert chosen is not None and chosen_decision is not None
        match_kind = f"{kind}_reused" if reused else kind
        examples.append(
            HorizonExample(
                prompt_id=chosen.prompt_id,
                split=window.split,
                horizon=window.horizon,
                role="control",
                y=0,
                decision_index=int(chosen_decision),
                length=window.length,
                onset_index=None,
                matched_prompt_id=window.prompt_id,
                match_kind=match_kind,
                dataset_source=chosen.dataset_source,
            )
        )
    info = {"ineligible_by_horizon": {str(k): int(v) for k, v in ineligible.items()}, "unmatched_positives": unmatched}
    return examples, info
