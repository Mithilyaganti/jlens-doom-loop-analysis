"""Script-aware doom-loop detectors D0–D4.

All five variants keep the Antidoom repeat count: at least 4 copies.
They differ in the unit and in how the 60-unit span is measured.

* D0 — original sampled rule on Python code points (60 code points).
* D1 — the same numeric rule, searched exactly (no 128-character stride).
* D2 — 60 Unicode grapheme clusters, searched exactly.
* D3 — the 60-character English span converted to tokens with a reference
  characters-per-token rate, searched on the token sequence.
* D4 — the 60-character span scaled by this text's characters-per-token
  relative to the English reference, searched on code points.

D3 and D4 do not look at whether an English item was answered correctly.
The reference rate is a tokenizer statistic. Pass the measured English
rate from ``fertility.py`` when you have it; the default 4.0 is only a
scale anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from loop_lingua.antidoom_rule import (
    ANTIDOOM_MAX_PERIOD,
    ANTIDOOM_MIN_CHARS,
    ANTIDOOM_MIN_PERIOD,
    ANTIDOOM_MIN_REPEATS,
    ANTIDOOM_SAMPLE_INTERVAL,
    ANTIDOOM_SAMPLE_LEN,
    find_inner_repetition,
)
from loop_lingua.graphemes import grapheme_clusters

Variant = Literal["D0", "D1", "D2", "D3", "D4"]
VARIANTS: tuple[Variant, ...] = ("D0", "D1", "D2", "D3", "D4")

# Scale anchor only. Replace with fertility measured on the English prompts.
DEFAULT_REFERENCE_CHARS_PER_TOKEN = 4.0


@dataclass(frozen=True)
class LoopHit:
    start_unit: int
    end_unit: int
    period: int
    repeats: int
    unit: str
    snippet: str
    char_span: tuple[int, int] | None

    @property
    def total_units(self) -> int:
        return self.repeats * self.period


@dataclass(frozen=True)
class Detection:
    variant: Variant
    is_loop: bool
    hit: LoopHit | None
    min_repeats: int
    min_total: int
    total_unit: str


def _snippet(units: Sequence[str], *, unit_name: str, limit: int = 100) -> str:
    joiner = " " if unit_name == "token" else ""
    text = joiner.join(units)
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def find_exact_repetition(
    units: Sequence[str],
    *,
    min_repeats: int = ANTIDOOM_MIN_REPEATS,
    min_total: int = ANTIDOOM_MIN_CHARS,
    max_period: int = ANTIDOOM_MAX_PERIOD,
    min_period: int = ANTIDOOM_MIN_PERIOD,
) -> tuple[int, int, int, int] | None:
    """Earliest exact loop as ``(start, end, period, repeats)``.

    Periods are tried from small to large. Within a period the leftmost
    maximal run that meets both thresholds wins. Partial trailing copies
    are not counted. Indices are into ``units``.
    """
    if min_repeats < 2 or min_period < 1 or min_total < 1:
        raise ValueError("repeat thresholds must be positive")
    n = len(units)
    if n < min_total or n < min_repeats * min_period:
        return None
    max_p = min(max_period, n // min_repeats)
    for period in range(min_period, max_p + 1):
        run = 0
        for index in range(period, n + 1):
            matched = index < n and units[index] == units[index - period]
            if matched:
                run += 1
                continue
            if run == 0:
                continue
            complete = (run // period) * period
            if complete >= period * (min_repeats - 1):
                total = complete + period
                repeats = total // period
                if repeats >= min_repeats and total >= min_total:
                    end = index - (run - complete)
                    start = end - total
                    return start, end, period, repeats
            run = 0
    return None


def _hit_from_units(
    units: Sequence[str],
    found: tuple[int, int, int, int],
    *,
    unit_name: str,
    char_of_unit: Sequence[int] | None,
) -> LoopHit:
    start, end, period, repeats = found
    if char_of_unit is None:
        char_span: tuple[int, int] | None = None
    else:
        char_span = (char_of_unit[start], char_of_unit[end])
    return LoopHit(
        start_unit=start,
        end_unit=end,
        period=period,
        repeats=repeats,
        unit=unit_name,
        snippet=_snippet(units[start : start + period], unit_name=unit_name),
        char_span=char_span,
    )


def _codepoint_units(text: str) -> list[str]:
    return list(text)


def _grapheme_index(clusters: Sequence[str]) -> list[int]:
    """``char_of_unit[i]`` is the code-point index where cluster ``i`` starts.

    The sentinel at the end equals ``len(text)``.
    """
    bounds = [0]
    for cluster in clusters:
        bounds.append(bounds[-1] + len(cluster))
    return bounds


def scaled_char_threshold(
    chars_per_token: float,
    *,
    reference_chars_per_token: float = DEFAULT_REFERENCE_CHARS_PER_TOKEN,
    base_chars: int = ANTIDOOM_MIN_CHARS,
) -> int:
    """D4 span: 60 English-reference characters, rescaled to this fertility.

    A language with fewer characters per token (higher fertility) gets a
    shorter character threshold, so the token budget matches English.
    """
    if chars_per_token <= 0 or reference_chars_per_token <= 0:
        raise ValueError("characters-per-token rates must be positive")
    scaled = round(base_chars * (chars_per_token / reference_chars_per_token))
    return max(1, scaled)


def token_threshold(
    *,
    reference_chars_per_token: float = DEFAULT_REFERENCE_CHARS_PER_TOKEN,
    base_chars: int = ANTIDOOM_MIN_CHARS,
) -> int:
    """D3 span in tokens, from the English character rule."""
    if reference_chars_per_token <= 0:
        raise ValueError("reference characters-per-token must be positive")
    return max(ANTIDOOM_MIN_REPEATS, round(base_chars / reference_chars_per_token))


def detect(
    text: str,
    variant: Variant,
    *,
    token_ids: Sequence[int] | None = None,
    token_strings: Sequence[str] | None = None,
    chars_per_token: float | None = None,
    reference_chars_per_token: float = DEFAULT_REFERENCE_CHARS_PER_TOKEN,
    min_repeats: int = ANTIDOOM_MIN_REPEATS,
    base_min_chars: int = ANTIDOOM_MIN_CHARS,
) -> Detection:
    """Run one detector variant."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")

    if variant == "D0":
        found, hit = find_inner_repetition(
            text,
            min_repeats=min_repeats,
            min_total_repeated=base_min_chars,
            max_period=ANTIDOOM_MAX_PERIOD,
            min_period=ANTIDOOM_MIN_PERIOD,
            sample_len=ANTIDOOM_SAMPLE_LEN,
            sample_interval=ANTIDOOM_SAMPLE_INTERVAL,
        )
        loop_hit = None
        if found and hit is not None:
            loop_hit = LoopHit(
                start_unit=hit.start,
                end_unit=hit.end,
                period=hit.period,
                repeats=hit.repeats,
                unit="codepoint",
                snippet=hit.snippet,
                char_span=(hit.start, hit.end),
            )
        return Detection(
            variant="D0",
            is_loop=bool(found),
            hit=loop_hit,
            min_repeats=min_repeats,
            min_total=base_min_chars,
            total_unit="codepoint",
        )

    if variant == "D1":
        units = _codepoint_units(text)
        found = find_exact_repetition(units, min_repeats=min_repeats, min_total=base_min_chars)
        bounds = list(range(len(text) + 1))
        return Detection(
            variant="D1",
            is_loop=found is not None,
            hit=None if found is None else _hit_from_units(units, found, unit_name="codepoint", char_of_unit=bounds),
            min_repeats=min_repeats,
            min_total=base_min_chars,
            total_unit="codepoint",
        )

    if variant == "D2":
        clusters = grapheme_clusters(text)
        found = find_exact_repetition(clusters, min_repeats=min_repeats, min_total=base_min_chars)
        return Detection(
            variant="D2",
            is_loop=found is not None,
            hit=None
            if found is None
            else _hit_from_units(
                clusters,
                found,
                unit_name="grapheme",
                char_of_unit=_grapheme_index(clusters),
            ),
            min_repeats=min_repeats,
            min_total=base_min_chars,
            total_unit="grapheme",
        )

    if variant == "D3":
        if token_ids is not None:
            units = [str(token_id) for token_id in token_ids]
        elif token_strings is not None:
            units = list(token_strings)
        else:
            raise ValueError("D3 needs token_ids or token_strings")
        minimum = token_threshold(
            reference_chars_per_token=reference_chars_per_token,
            base_chars=base_min_chars,
        )
        found = find_exact_repetition(units, min_repeats=min_repeats, min_total=minimum)
        return Detection(
            variant="D3",
            is_loop=found is not None,
            hit=None if found is None else _hit_from_units(units, found, unit_name="token", char_of_unit=None),
            min_repeats=min_repeats,
            min_total=minimum,
            total_unit="token",
        )

    # D4
    if chars_per_token is None:
        raise ValueError("D4 needs chars_per_token (measure it; do not infer it from an English score)")
    minimum = scaled_char_threshold(
        chars_per_token,
        reference_chars_per_token=reference_chars_per_token,
        base_chars=base_min_chars,
    )
    units = _codepoint_units(text)
    found = find_exact_repetition(units, min_repeats=min_repeats, min_total=minimum)
    bounds = list(range(len(text) + 1))
    return Detection(
        variant="D4",
        is_loop=found is not None,
        hit=None if found is None else _hit_from_units(units, found, unit_name="codepoint", char_of_unit=bounds),
        min_repeats=min_repeats,
        min_total=minimum,
        total_unit="codepoint",
    )


def detect_all(
    text: str,
    *,
    token_ids: Sequence[int] | None = None,
    token_strings: Sequence[str] | None = None,
    chars_per_token: float | None = None,
    reference_chars_per_token: float = DEFAULT_REFERENCE_CHARS_PER_TOKEN,
) -> dict[Variant, Detection]:
    """Run every variant that has the inputs it needs.

    D3 is omitted when no tokens are passed. D4 is omitted when fertility
    is not passed. D0–D2 always run.
    """
    results: dict[Variant, Detection] = {}
    for variant in ("D0", "D1", "D2"):
        results[variant] = detect(text, variant)
    if token_ids is not None or token_strings is not None:
        results["D3"] = detect(
            text,
            "D3",
            token_ids=token_ids,
            token_strings=token_strings,
            reference_chars_per_token=reference_chars_per_token,
        )
    if chars_per_token is not None:
        results["D4"] = detect(
            text,
            "D4",
            chars_per_token=chars_per_token,
            reference_chars_per_token=reference_chars_per_token,
        )
    return results
