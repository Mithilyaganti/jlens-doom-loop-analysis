"""Grapheme clusters for Latin, Devanagari, and Telugu.

This is a deliberate subset of UAX #29, not a full grapheme breaker.
It covers the rules that change loop thresholds for the scripts in this
study:

* CR LF stays one cluster
* characters of category Mn or Me (vowel signs, virama, nukta) do not start
  a new cluster
* spacing marks (Mc), including Telugu anusvara and several vowel signs,
  do not start a new cluster
* an Indic consonant does not split off from a preceding virama (GB9c),
  so conjuncts such as Telugu ``క్ష`` and Devanagari ``क्ष`` are one cluster
* ZWJ does not break a cluster

Not implemented: Hangul syllable types, emoji flag pairs, and the full
Prepend class. Those do not appear in the EN / HI / TE prompt set. See
DESIGN_DECISIONS.md.
"""

from __future__ import annotations

import unicodedata

VIRAMAS = frozenset(
    {
        "\u094d",  # Devanagari
        "\u09cd",  # Bengali
        "\u0a4d",  # Gurmukhi
        "\u0acd",  # Gujarati
        "\u0b4d",  # Oriya
        "\u0bcd",  # Tamil
        "\u0c4d",  # Telugu
        "\u0ccd",  # Kannada
        "\u0d4d",  # Malayalam
    }
)

# Consonant ranges used by GB9c for the scripts we actually score.
_CONSONANT_RANGES: tuple[tuple[int, int], ...] = (
    (0x0915, 0x0939),
    (0x0958, 0x095F),
    (0x0978, 0x097A),
    (0x0C15, 0x0C39),
    (0x0C58, 0x0C5A),
)


def _is_consonant(char: str) -> bool:
    code = ord(char)
    if unicodedata.category(char) != "Lo":
        return False
    return any(start <= code <= end for start, end in _CONSONANT_RANGES)


def _virama_links(text: str, index: int) -> bool:
    """True when ``text[index]`` continues a virama-linked consonant cluster."""
    if not _is_consonant(text[index]):
        return False
    cursor = index - 1
    saw_virama = False
    while cursor >= 0:
        char = text[cursor]
        category = unicodedata.category(char)
        if char in VIRAMAS:
            saw_virama = True
            cursor -= 1
            continue
        if category in {"Mn", "Me"} or char == "\u200d":
            cursor -= 1
            continue
        break
    return saw_virama


def _keep_together(text: str, index: int) -> bool:
    prev = text[index - 1]
    cur = text[index]
    if prev == "\r" and cur == "\n":
        return True
    if cur == "\u200d" or prev == "\u200d":
        return True
    category = unicodedata.category(cur)
    if category in {"Mn", "Me", "Mc"}:
        return True
    if _virama_links(text, index):
        return True
    return False


def grapheme_clusters(text: str) -> list[str]:
    """Split ``text`` into extended grapheme clusters (study subset)."""
    if not text:
        return []
    clusters: list[str] = []
    buffer = text[0]
    for index in range(1, len(text)):
        if _keep_together(text, index):
            buffer += text[index]
        else:
            clusters.append(buffer)
            buffer = text[index]
    clusters.append(buffer)
    return clusters


def grapheme_count(text: str) -> int:
    return len(grapheme_clusters(text))
