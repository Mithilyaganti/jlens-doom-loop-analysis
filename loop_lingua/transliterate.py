"""Telugu to Latin transliteration (ISO 15919).

The romanized prompt arm is produced by running this function over the
Telugu prompt, not by a model. Inherent vowel ``a`` is written on each
consonant and removed by virama or replaced by a vowel sign. Unmapped
Telugu-block characters raise ``TransliterationError`` so a silent
leftover cannot pass the "romanized text is Latin" check.
"""

from __future__ import annotations

class TransliterationError(ValueError):
    """A Telugu character has no ISO 15919 mapping."""


# Independent vowels. Values do not include a following consonant.
_VOWELS: dict[str, str] = {
    "అ": "a",
    "ఆ": "ā",
    "ఇ": "i",
    "ఈ": "ī",
    "ఉ": "u",
    "ఊ": "ū",
    "ఋ": "r̥",
    "ౠ": "r̥̄",
    "ఌ": "l̥",
    "ౡ": "l̥̄",
    "ఎ": "e",
    "ఏ": "ē",
    "ఐ": "ai",
    "ఒ": "o",
    "ఓ": "ō",
    "ఔ": "au",
}

# Consonants with the inherent vowel still attached.
_CONSONANTS: dict[str, str] = {
    "క": "ka",
    "ఖ": "kha",
    "గ": "ga",
    "ఘ": "gha",
    "ఙ": "ṅa",
    "చ": "ca",
    "ఛ": "cha",
    "జ": "ja",
    "ఝ": "jha",
    "ఞ": "ña",
    "ట": "ṭa",
    "ఠ": "ṭha",
    "డ": "ḍa",
    "ఢ": "ḍha",
    "ణ": "ṇa",
    "త": "ta",
    "థ": "tha",
    "ద": "da",
    "ధ": "dha",
    "న": "na",
    "ప": "pa",
    "ఫ": "pha",
    "బ": "ba",
    "భ": "bha",
    "మ": "ma",
    "య": "ya",
    "ర": "ra",
    "ఱ": "ṟa",
    "ల": "la",
    "ళ": "ḷa",
    "ఴ": "ḻa",
    "వ": "va",
    "శ": "śa",
    "ష": "ṣa",
    "స": "sa",
    "హ": "ha",
}

# Vowel signs replace the inherent "a".
_VOWEL_SIGNS: dict[str, str] = {
    "ా": "ā",
    "ి": "i",
    "ీ": "ī",
    "ు": "u",
    "ూ": "ū",
    "ృ": "r̥",
    "ౄ": "r̥̄",
    "ె": "e",
    "ే": "ē",
    "ై": "ai",
    "ొ": "o",
    "ో": "ō",
    "ౌ": "au",
    "ౢ": "l̥",
    "ౣ": "l̥̄",
}

_SIGNS: dict[str, str] = {
    "ఁ": "m̐",
    "ం": "ṁ",
    "ః": "ḥ",
}

_DIGITS: dict[str, str] = {chr(0x0C66 + i): str(i) for i in range(10)}

_VIRAMA = "్"
_SKIP = frozenset({"\u200c", "\u200d"})  # ZWNJ, ZWJ

# Length marks and nukta have no independent ISO letter. They are dropped
# when they only modify a vowel we already wrote, which matches how the
# base letter is romanized.
_DROP = frozenset({"\u0c3c", "\u0c55", "\u0c56"})


def _in_telugu_block(char: str) -> bool:
    return 0x0C00 <= ord(char) <= 0x0C7F


def transliterate_telugu(text: str) -> str:
    """Transliterate Telugu script to ISO 15919 Latin. Other scripts pass through."""
    out: list[str] = []
    index = 0
    chars = list(text)
    limit = len(chars)
    while index < limit:
        char = chars[index]
        if char in _SKIP or char in _DROP:
            index += 1
            continue
        if char in _VOWELS:
            out.append(_VOWELS[char])
            index += 1
            continue
        if char in _CONSONANTS:
            stem = _CONSONANTS[char]
            index += 1
            if index < limit and chars[index] == _VIRAMA:
                out.append(stem[:-1])
                index += 1
            elif index < limit and chars[index] in _VOWEL_SIGNS:
                out.append(stem[:-1] + _VOWEL_SIGNS[chars[index]])
                index += 1
            else:
                out.append(stem)
            continue
        if char in _VOWEL_SIGNS:
            out.append(_VOWEL_SIGNS[char])
            index += 1
            continue
        if char == _VIRAMA:
            index += 1
            continue
        if char in _SIGNS:
            out.append(_SIGNS[char])
            index += 1
            continue
        if char in _DIGITS:
            out.append(_DIGITS[char])
            index += 1
            continue
        if _in_telugu_block(char):
            name = f"U+{ord(char):04X}"
            raise TransliterationError(f"no ISO 15919 mapping for Telugu character {name}")
        out.append(char)
        index += 1
    return "".join(out)
