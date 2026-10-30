"""Offset mapping, including the paper-1 Telugu failure mode."""

from __future__ import annotations

import pytest

from loop_lingua.bugreport import compare_token_strings, jspace_reconstructed_text
from loop_lingua.offsets import OffsetMapError, map_token_ids, map_token_strings


# Qwen3.5-4B pieces observed on MGSM Telugu: space + an incomplete Telugu
# sequence, then the last UTF-8 byte. Together they are U+0020 U+0C21 ( డ ).
SPLIT_TOKENS = ["Ġà°", "¡"]


def test_split_telugu_character_decodes_after_concatenation() -> None:
    mapped = map_token_strings(SPLIT_TOKENS)
    assert mapped.text == " డ"
    assert mapped.char_to_token_index(0) == 0
    assert mapped.char_to_token_index(1) == 0
    assert mapped.token_span(1) == (1, 1)


def test_jspace_per_token_decode_mismaps_the_same_pieces() -> None:
    length_mismatch, offset_mismatch, corrected = compare_token_strings(SPLIT_TOKENS)
    reconstructed = jspace_reconstructed_text(SPLIT_TOKENS)
    assert reconstructed != corrected.text
    assert length_mismatch
    assert offset_mismatch
    assert "డ" not in reconstructed


def test_ascii_token_round_trips_and_agrees_with_jspace() -> None:
    tokens = ["Hello", "Ġworld"]
    mapped = map_token_strings(tokens)
    assert mapped.text == "Hello world"
    assert jspace_reconstructed_text(tokens) == mapped.text
    length_mismatch, offset_mismatch, _ = compare_token_strings(tokens)
    assert not length_mismatch
    assert not offset_mismatch
    assert mapped.char_to_token_index(0) == 0
    assert mapped.char_to_token_index(5) == 1


def test_literal_non_byte_token_is_taken_as_text() -> None:
    mapped = map_token_strings(["డ"])
    assert mapped.text == "డ"
    assert mapped.char_to_token_index(0) == 0


def test_map_token_ids_uses_the_lookup() -> None:
    table = {7: "Ġà°", 8: "¡"}
    mapped = map_token_ids([7, 8], table.get)
    assert mapped.text == " డ"


def test_out_of_range_and_empty() -> None:
    mapped = map_token_strings(["A"])
    assert mapped.char_to_token_index(-1) is None
    assert mapped.char_to_token_index(1) is None
    empty = map_token_strings([])
    assert empty.text == ""
    assert empty.char_to_token_index(0) is None


def test_invalid_utf8_is_refused() -> None:
    # 0xFF is in the byte alphabet (ÿ) and is not a valid UTF-8 start.
    with pytest.raises(OffsetMapError):
        map_token_strings(["ÿ"])


def test_id_lookup_must_return_str() -> None:
    with pytest.raises(OffsetMapError):
        map_token_ids([1], lambda _token_id: None)  # type: ignore[arg-type, return-value]
