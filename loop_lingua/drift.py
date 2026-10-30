"""Output-language drift from script membership.

The score is the share of letters (and combining marks inside a script
block) that sit in the script the prompt asked for. Digits, punctuation,
and whitespace are ignored, because math prompts are full of them in every
language.

A flag is a convenience threshold on that share. The share itself is the
measurement. Tenglish is mixed on purpose: it drifts when either Latin or
Telugu collapses below a floor, not when both are present.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Literal

ExpectedLanguage = Literal["en", "hi", "te", "te_roman", "tenglish"]

_EXPECTED_SCRIPT: dict[str, str] = {
    "en": "latin",
    "te_roman": "latin",
    "hi": "devanagari",
    "te": "telugu",
}


@dataclass(frozen=True)
class ScriptProfile:
    counts: dict[str, int]
    n_letters: int

    def fraction(self, script: str) -> float:
        if self.n_letters == 0:
            return 0.0
        return self.counts.get(script, 0) / self.n_letters


@dataclass(frozen=True)
class DriftResult:
    expected: ExpectedLanguage
    profile: ScriptProfile
    expected_fraction: float | None
    drifted: bool | None
    low_confidence: bool
    reason: str


def classify_codepoint(code: int) -> str | None:
    """Return a script name, or None if the code point is not a letter."""
    char = chr(code)
    category = unicodedata.category(char)
    if 0x0300 <= code <= 0x036F:
        return "latin"
    if (
        (0x0041 <= code <= 0x005A)
        or (0x0061 <= code <= 0x007A)
        or (0x00C0 <= code <= 0x024F)
        or (0x1E00 <= code <= 0x1EFF)
    ):
        if category.startswith("L"):
            return "latin"
        return None
    if 0x0900 <= code <= 0x097F:
        if category.startswith("L") or category.startswith("M"):
            return "devanagari"
        return None
    if 0x0C00 <= code <= 0x0C7F:
        if category.startswith("L") or category.startswith("M"):
            return "telugu"
        return None
    if category.startswith("L"):
        return "other"
    return None


def script_profile(text: str) -> ScriptProfile:
    counts = {"latin": 0, "devanagari": 0, "telugu": 0, "other": 0}
    for char in text:
        script = classify_codepoint(ord(char))
        if script is None:
            continue
        counts[script] = counts.get(script, 0) + 1
    return ScriptProfile(counts=counts, n_letters=sum(counts.values()))


def measure_drift(
    text: str,
    expected: ExpectedLanguage,
    *,
    threshold: float = 0.5,
    mixed_floor: float = 0.05,
    min_letters: int = 20,
) -> DriftResult:
    """Score how far ``text`` sits from the script ``expected`` asks for.

    ``drifted`` is None when there are fewer than ``min_letters`` classified
    letters. That is low confidence, not evidence of a language switch.
    """
    if expected not in {"en", "hi", "te", "te_roman", "tenglish"}:
        raise ValueError(f"unknown expected language {expected!r}")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    profile = script_profile(text)
    if profile.n_letters < min_letters:
        return DriftResult(
            expected=expected,
            profile=profile,
            expected_fraction=None,
            drifted=None,
            low_confidence=True,
            reason=f"fewer than {min_letters} letters",
        )
    if expected == "tenglish":
        latin = profile.fraction("latin")
        telugu = profile.fraction("telugu")
        drifted = latin < mixed_floor or telugu < mixed_floor
        return DriftResult(
            expected=expected,
            profile=profile,
            expected_fraction=min(latin, telugu),
            drifted=drifted,
            low_confidence=False,
            reason="mixed Latin+Telugu floor" if drifted else "both scripts present",
        )
    script = _EXPECTED_SCRIPT[expected]
    fraction = profile.fraction(script)
    drifted = fraction < threshold
    return DriftResult(
        expected=expected,
        profile=profile,
        expected_fraction=fraction,
        drifted=drifted,
        low_confidence=False,
        reason=f"{script} fraction {fraction:.3f} vs threshold {threshold:.3f}",
    )
