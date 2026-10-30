"""D0–D4 on synthetic loops in each study script."""

from __future__ import annotations

import pytest

from jspace.detection import find_inner_repetition as jspace_find
from loop_lingua.antidoom_rule import find_inner_repetition
from loop_lingua.detectors import detect, scaled_char_threshold, token_threshold
from loop_lingua.graphemes import grapheme_clusters


def _repeat_until(unit: str) -> str:
    repeats = 4
    while True:
        text = unit * repeats
        if len(text) >= 60 and len(grapheme_clusters(text)) >= 60:
            return text
        repeats += 1
        if repeats > 30:
            raise AssertionError(f"unit never reached both thresholds: {unit!r}")


EN = _repeat_until("Wait, let me rethink this carefully. ")
HI = _repeat_until("फिर से ध्यान से सोचिए। ")
TE = _repeat_until("మళ్ళీ ఆలోచిద్దాం. ")
ROMAN = _repeat_until("malli alochiddam, jaragandi. ")


@pytest.mark.parametrize("text", [EN, HI, TE, ROMAN])
def test_d0_d1_d2_fire_on_long_script_loops(text: str) -> None:
    for variant in ("D0", "D1", "D2"):
        result = detect(text, variant)  # type: ignore[arg-type]
        assert result.is_loop, variant
        assert result.hit is not None
        assert result.hit.repeats >= 4
        assert result.hit.total_units >= 60


@pytest.mark.parametrize(
    "text",
    [
        "This is a perfectly ordinary English answer with no repeated clause at all.",
        "यह एक सामान्य हिंदी वाक्य है जिसमें कोई दोहराव नहीं है।",
        "ఇది సాధారణ తెలుగు వాక్యం. ఇందులో పునరావృతం లేదు.",
        "idi sadharana vakyanam. ikkada punaravruttam ledu, matram.",
    ],
)
def test_plain_sentences_are_not_loops(text: str) -> None:
    for variant in ("D0", "D1", "D2"):
        assert not detect(text, variant).is_loop  # type: ignore[arg-type]


def test_d0_matches_jspace_on_the_same_strings() -> None:
    samples = [EN, HI, TE, ROMAN, "short", "abcd" * 40, "no repetition here at all, thanks."]
    for text in samples:
        ours, our_hit = find_inner_repetition(text)
        theirs, their_hit = jspace_find(text)
        assert ours == theirs
        if our_hit is None or their_hit is None:
            assert our_hit is None and their_hit is None
        else:
            assert (our_hit.start, our_hit.end, our_hit.period, our_hit.repeats, our_hit.snippet) == (
                their_hit.start,
                their_hit.end,
                their_hit.period,
                their_hit.repeats,
                their_hit.snippet,
            )


def test_d0_sampler_misses_a_loop_that_d1_catches() -> None:
    prefix = "abcdefghijklmnopqrst"
    unit = "XYZ1234567890abc"
    text = prefix + unit * 4
    assert len(text) < 128
    assert not detect(text, "D0").is_loop
    found = detect(text, "D1")
    assert found.is_loop
    assert found.hit is not None
    assert found.hit.start_unit == len(prefix)
    assert found.hit.repeats == 4
    assert found.hit.period == len(unit)


def test_telugu_codepoint_loop_can_miss_the_grapheme_threshold() -> None:
    # కా is two code points and one grapheme. 56 graphemes is under 60;
    # 112 code points is over 60.
    text = "కా" * 56
    assert len(text) == 112
    assert len(grapheme_clusters(text)) == 56
    assert detect(text, "D0").is_loop
    assert detect(text, "D1").is_loop
    assert not detect(text, "D2").is_loop


def test_grapheme_hit_covers_a_conjunct() -> None:
    text = "క్ష" * 60
    assert grapheme_clusters("క్ష") == ["క్ష"]
    result = detect(text, "D2")
    assert result.is_loop
    assert result.hit is not None
    assert result.hit.char_span == (0, len(text))


def test_d3_uses_tokens_not_characters() -> None:
    assert token_threshold(reference_chars_per_token=4.0) == 15
    short_chars = "ab" * 10
    assert not detect(short_chars, "D1").is_loop
    looping = detect(short_chars, "D3", token_ids=[1, 2, 3, 4] * 4)
    assert looping.is_loop
    assert looping.min_total == 15
    assert looping.hit is not None
    assert looping.hit.repeats == 4
    below = detect(short_chars, "D3", token_ids=[1, 2, 3] * 4)
    assert not below.is_loop
    assert below.min_total == 15


def test_d3_requires_tokens() -> None:
    with pytest.raises(ValueError):
        detect("hello", "D3")


def test_d4_scales_the_character_threshold_with_fertility() -> None:
    assert scaled_char_threshold(1.0, reference_chars_per_token=4.0) == 15
    short = "abcde" * 4  # 20 code points
    assert not detect(short, "D0").is_loop
    assert detect(short, "D4", chars_per_token=1.0, reference_chars_per_token=4.0).is_loop
    long = "abcd" * 20  # 80 code points, periodic, sampler sees it
    assert detect(long, "D0").is_loop
    # More characters per token than English raises the bar above 80.
    assert scaled_char_threshold(8.0, reference_chars_per_token=4.0) == 120
    assert not detect(long, "D4", chars_per_token=8.0, reference_chars_per_token=4.0).is_loop


def test_d4_requires_a_measured_rate() -> None:
    with pytest.raises(ValueError):
        detect("abcd" * 40, "D4")
