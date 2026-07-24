"""Loop detection wrappers — Liquid AI Antidoom criteria, character-based.

Uses Liquid's exact find_inner_repetition algorithm and TokenState.char_to_token_index
mapping. Vendored from Liquid4All/antidoom to avoid full antidoom install deps (vLLM).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Vendored from antidoom/repetition.py (Apache-compatible research use)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepeatHit:
    start: int
    end: int
    period: int
    repeats: int
    snippet: str

    @property
    def repeat_start(self) -> int:
        return self.start + self.period


def _verify_repetition_at(
    text: str,
    start_pos: int,
    period: int,
    min_repeats: int,
    min_total_repeated: int,
) -> tuple[bool, RepeatHit | None]:
    if period < 1 or start_pos < 0 or start_pos + period > len(text):
        return False, None

    pattern = text[start_pos : start_pos + period]
    reps = 0
    pos = start_pos
    while pos + period <= len(text) and text[pos : pos + period] == pattern:
        reps += 1
        pos += period
    end_pos = pos

    pos = start_pos - period
    while pos >= 0 and text[pos : pos + period] == pattern:
        reps += 1
        start_pos = pos
        pos -= period

    total = reps * period
    if reps >= min_repeats and total >= min_total_repeated:
        snippet = pattern if len(pattern) <= 100 else pattern[:100] + "..."
        return True, RepeatHit(start_pos, end_pos, period, reps, snippet)
    return False, None


def find_inner_repetition(
    text: str,
    *,
    min_repeats: int = 4,
    max_period: int = 1024,
    min_period: int = 1,
    min_total_repeated: int = 60,
    sample_len: int = 16,
    sample_interval: int = 128,
) -> tuple[bool, RepeatHit | None]:
    """Detect character-level doom loops (Liquid Antidoom defaults)."""
    if not text or len(text) < min_total_repeated:
        return False, None

    n = len(text)
    for sample_pos in range(0, n - sample_len, sample_interval):
        fingerprint = text[sample_pos : sample_pos + sample_len]

        other_pos = text.find(fingerprint, sample_pos + sample_len)
        if other_pos != -1:
            candidate_period = other_pos - sample_pos
            if min_period <= candidate_period <= max_period:
                found, hit = _verify_repetition_at(
                    text,
                    sample_pos,
                    candidate_period,
                    min_repeats=min_repeats,
                    min_total_repeated=min_total_repeated,
                )
                if found:
                    return True, hit

        other_pos = text.rfind(fingerprint, 0, sample_pos)
        if other_pos != -1:
            candidate_period = sample_pos - other_pos
            if min_period <= candidate_period <= max_period:
                found, hit = _verify_repetition_at(
                    text,
                    other_pos,
                    candidate_period,
                    min_repeats=min_repeats,
                    min_total_repeated=min_total_repeated,
                )
                if found:
                    return True, hit

    return False, None


# ---------------------------------------------------------------------------
# Vendored from antidoom/tokens.py
# ---------------------------------------------------------------------------


def _build_u2b_full() -> dict[int, int]:
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    u2b = {ord(chr(c)): b for b, c in zip(bs, cs)}
    for b in range(256):
        u2b[0x2500 + b] = b
    return u2b


_U2B = _build_u2b_full()


def fix_mojibake(text: str) -> str:
    buf = bytearray()
    changed = False
    for ch in text:
        cp = ord(ch)
        if cp in _U2B:
            buf.append(_U2B[cp])
            changed = True
        else:
            buf.extend(ch.encode("utf-8"))
    if not changed:
        return text
    try:
        return buf.decode("utf-8")
    except UnicodeDecodeError:
        return text


def decode_token(token: str) -> str:
    if not token:
        return token
    token = fix_mojibake(token)
    return token.replace("Ċ", "\n").replace("Ġ", " ").replace("▁", " ")


@dataclass
class TokenState:
    """Tracks generated token strings and maps char positions → token indices."""

    prompt: str
    token_strings: list[str] = field(default_factory=list)
    decoded_lens: list[int] = field(default_factory=list)
    text: str = ""
    logprobs: dict[int, list[tuple[str, float]]] = field(default_factory=dict)

    def append(
        self,
        token_strings: list[str],
        logprobs: dict[int, list[tuple[str, float]]] | None = None,
    ) -> None:
        start = len(self.token_strings)
        for raw in token_strings:
            decoded = decode_token(raw)
            self.token_strings.append(raw)
            self.decoded_lens.append(len(decoded))
            self.text += decoded
        if logprobs:
            for rel_idx, alts in logprobs.items():
                abs_idx = start + rel_idx
                if abs_idx < len(self.token_strings):
                    self.logprobs[abs_idx] = alts

    def append_ids(self, token_ids: list[int], tokenizer: Any) -> None:
        """Append by converting token IDs via the HF tokenizer."""
        raws = tokenizer.convert_ids_to_tokens(token_ids)
        # convert_ids_to_tokens may return str or list
        if isinstance(raws, str):
            raws = [raws]
        self.append(list(raws))

    def char_to_token_index(self, char_pos: int) -> int | None:
        if char_pos < 0 or char_pos >= len(self.text):
            return None
        running = 0
        for idx, length in enumerate(self.decoded_lens):
            running += length
            if char_pos < running:
                return idx
        return None


# Operational trigger set from antidoom/generate.py (_RESTART_WORDS)
RESTART_WORDS: frozenset[str] = frozenset(
    {
        "actually",
        "after",
        "also",
        "alternatively",
        "because",
        "but",
        "finally",
        "first",
        "given",
        "hmm",
        "however",
        "in",
        "let",
        "looking",
        "maybe",
        "now",
        "okay",
        "perhaps",
        "second",
        "since",
        "so",
        "the",
        "then",
        "therefore",
        "this",
        "thus",
        "wait",
    }
)


DEFAULT_REPETITION_KWARGS = dict(
    min_repeats=4,
    max_period=1024,
    min_period=1,
    min_total_repeated=60,
    sample_len=16,
    sample_interval=128,
)


@dataclass
class LoopResult:
    is_loop: bool
    hit: RepeatHit | None
    trigger_token_index: int | None  # index into generated tokens
    trigger_token_id: int | None
    trigger_token_str: str | None
    trigger_decoded: str | None
    start_char: int | None
    repeat_start_char: int | None
    end_char: int | None


def detect_loop(
    generated_text: str,
    *,
    token_ids: list[int] | None = None,
    tokenizer: Any = None,
    prompt: str = "",
    **rep_kwargs: Any,
) -> LoopResult:
    """Detect a doom loop and map the trigger boundary to a token index.

    The trigger is the first token of the first repeat (Liquid's definition).
    """
    kwargs = {**DEFAULT_REPETITION_KWARGS, **rep_kwargs}
    found, hit = find_inner_repetition(generated_text, **kwargs)
    if not found or hit is None:
        return LoopResult(
            is_loop=False,
            hit=None,
            trigger_token_index=None,
            trigger_token_id=None,
            trigger_token_str=None,
            trigger_decoded=None,
            start_char=None,
            repeat_start_char=None,
            end_char=None,
        )

    trigger_idx: int | None = None
    trigger_id: int | None = None
    trigger_str: str | None = None
    trigger_decoded: str | None = None

    if token_ids is not None and tokenizer is not None:
        state = TokenState(prompt)
        state.append_ids(token_ids, tokenizer)
        # Map first token of first repeat (repeat_start char)
        reject_char = hit.repeat_start
        while reject_char < hit.end and reject_char < len(state.text) and state.text[reject_char].isspace():
            reject_char += 1
        if reject_char < len(state.text):
            trigger_idx = state.char_to_token_index(reject_char)
            if trigger_idx is not None and trigger_idx < len(token_ids):
                # Skip pure-boundary tokens
                while trigger_idx < len(token_ids):
                    raw = tokenizer.convert_ids_to_tokens(token_ids[trigger_idx])
                    if isinstance(raw, list):
                        raw = raw[0]
                    decoded = decode_token(raw)
                    if decoded.strip():
                        trigger_str = raw
                        trigger_decoded = decoded
                        trigger_id = token_ids[trigger_idx]
                        break
                    trigger_idx += 1
                else:
                    trigger_idx = None

    return LoopResult(
        is_loop=True,
        hit=hit,
        trigger_token_index=trigger_idx,
        trigger_token_id=trigger_id,
        trigger_token_str=trigger_str,
        trigger_decoded=trigger_decoded,
        start_char=hit.start,
        repeat_start_char=hit.repeat_start,
        end_char=hit.end,
    )
