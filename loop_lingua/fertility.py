"""Tokenizer fertility.

Fertility here is tokens per code point (and per grapheme, and per
whitespace-delimited word). The inverse, characters per token, is what
detector D4 multiplies into the 60-character rule.

The encode callback is injected so this module never loads a model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from loop_lingua.graphemes import grapheme_count

EncodeFn = Callable[[str], Sequence[int]]


@dataclass(frozen=True)
class FertilityReport:
    language: str
    n_texts: int
    n_code_points: int
    n_graphemes: int
    n_tokens: int
    n_words: int
    tokens_per_code_point: float
    code_points_per_token: float
    tokens_per_grapheme: float
    tokens_per_word: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "language": self.language,
            "n_texts": self.n_texts,
            "n_code_points": self.n_code_points,
            "n_graphemes": self.n_graphemes,
            "n_tokens": self.n_tokens,
            "n_words": self.n_words,
            "tokens_per_code_point": self.tokens_per_code_point,
            "code_points_per_token": self.code_points_per_token,
            "tokens_per_grapheme": self.tokens_per_grapheme,
            "tokens_per_word": self.tokens_per_word,
        }


def _words(text: str) -> int:
    return sum(1 for piece in text.split() if piece)


def measure_fertility(texts: Sequence[str], encode: EncodeFn, *, language: str) -> FertilityReport:
    """Aggregate fertility of ``texts`` under ``encode``."""
    if not texts:
        raise ValueError("fertility needs at least one text")
    n_chars = 0
    n_graphs = 0
    n_tokens = 0
    n_words = 0
    for text in texts:
        if not isinstance(text, str):
            raise TypeError("texts must be strings")
        n_chars += len(text)
        n_graphs += grapheme_count(text)
        n_words += _words(text)
        ids = encode(text)
        n_tokens += len(ids)
    if n_chars == 0 or n_tokens == 0:
        raise ValueError("fertility is undefined for empty text or an empty tokenization")
    return FertilityReport(
        language=language,
        n_texts=len(texts),
        n_code_points=n_chars,
        n_graphemes=n_graphs,
        n_tokens=n_tokens,
        n_words=n_words,
        tokens_per_code_point=n_tokens / n_chars,
        code_points_per_token=n_chars / n_tokens,
        tokens_per_grapheme=(n_tokens / n_graphs) if n_graphs else float("inf"),
        tokens_per_word=(n_tokens / n_words) if n_words else float("inf"),
    )
