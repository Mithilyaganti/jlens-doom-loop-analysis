"""Compare ``jspace.detection`` offsets with the corrected mapper.

Paper-1 is not modified. This module imports it and reports where the
per-token decode disagrees with a full-sequence decode.
"""

from __future__ import annotations

from dataclasses import dataclass

from loop_lingua.offsets import TokenOffsetMap, map_token_strings, token_string_to_bytes


@dataclass(frozen=True)
class TextMismatch:
    """One text whose jspace reconstruction is not the real decode."""

    index: int
    decoded_length: int
    reconstructed_length: int
    n_tokens: int
    n_bad_pieces: int


@dataclass(frozen=True)
class MismatchReport:
    n_texts: int
    n_length_mismatches: int
    n_offset_mismatches: int
    examples: tuple[TextMismatch, ...]

    @property
    def n_ok(self) -> int:
        return self.n_texts - self.n_length_mismatches


def jspace_token_state(token_strings: list[str]) -> object:
    """Run the unmodified paper-1 ``TokenState`` on these token strings."""
    from jspace.detection import TokenState

    class _Lookup:
        def convert_ids_to_tokens(self, token_ids: int | list[int]) -> str | list[str]:
            if isinstance(token_ids, int):
                return token_strings[token_ids]
            return [token_strings[index] for index in token_ids]

    state = TokenState("")
    state.append_ids(list(range(len(token_strings))), _Lookup())
    return state


def jspace_reconstructed_text(token_strings: list[str]) -> str:
    state = jspace_token_state(token_strings)
    text = getattr(state, "text")
    if not isinstance(text, str):
        raise TypeError("TokenState.text is not a str")
    return text


def _incomplete_piece_count(token_strings: list[str]) -> int:
    """Pieces whose own bytes are not a finished UTF-8 string.

    Those are the pieces ``jspace.detection.decode_token`` cannot recover,
    because it UTF-8-decodes one token at a time.
    """
    count = 0
    for token in token_strings:
        try:
            token_string_to_bytes(token).decode("utf-8")
        except UnicodeDecodeError:
            count += 1
    return count


def compare_token_strings(token_strings: list[str]) -> tuple[bool, bool, TokenOffsetMap]:
    """Return ``(length_mismatch, offset_mismatch, corrected_map)``.

    Offset mismatch means some code-point index of the *corrected* text is
    given a different token by ``TokenState.char_to_token_index`` when that
    index is even inside the reconstructed string. A length mismatch alone
    is enough to make character coordinates from ``tokenizer.decode`` unsafe.
    """
    corrected = map_token_strings(token_strings)
    state = jspace_token_state(token_strings)
    reconstructed = getattr(state, "text")
    length_mismatch = reconstructed != corrected.text
    offset_mismatch = False
    char_to_token = getattr(state, "char_to_token_index")
    limit = min(len(reconstructed), len(corrected.text))
    for pos in range(limit):
        predicted = char_to_token(pos)
        truth = corrected.char_to_token_index(pos)
        if predicted != truth:
            offset_mismatch = True
            break
    if length_mismatch:
        # Coordinates no longer refer to the same string.
        offset_mismatch = True
    return length_mismatch, offset_mismatch, corrected


def report_encoded_texts(
    texts: list[str],
    encode_tokens: object,
    *,
    example_limit: int = 5,
) -> MismatchReport:
    """Score texts whose token strings come from ``encode_tokens(text) -> list[str]``.

    ``encode_tokens`` must be the tokenizer's own segmentation, so the
    corrected map is the full-sequence decode of those pieces.
    """
    if not callable(encode_tokens):
        raise TypeError("encode_tokens must be callable")
    length_hits = 0
    offset_hits = 0
    examples: list[TextMismatch] = []
    for index, text in enumerate(texts):
        tokens = encode_tokens(text)
        if not isinstance(tokens, list) or not all(isinstance(tok, str) for tok in tokens):
            raise TypeError("encode_tokens must return list[str]")
        length_mismatch, offset_mismatch, corrected = compare_token_strings(tokens)
        if corrected.text != text:
            raise ValueError(
                "Corrected decode does not match the source text at "
                f"index {index}. The encoder and the byte map disagree."
            )
        if length_mismatch:
            length_hits += 1
        if offset_mismatch:
            offset_hits += 1
        if (length_mismatch or offset_mismatch) and len(examples) < example_limit:
            examples.append(
                TextMismatch(
                    index=index,
                    decoded_length=len(corrected.text),
                    reconstructed_length=len(jspace_reconstructed_text(tokens)),
                    n_tokens=len(tokens),
                    n_bad_pieces=_incomplete_piece_count(tokens),
                )
            )
    return MismatchReport(
        n_texts=len(texts),
        n_length_mismatches=length_hits,
        n_offset_mismatches=offset_hits,
        examples=tuple(examples),
    )
