#!/usr/bin/env python
"""CPU-only smoke test for Liquid loop detector."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jspace.detection import find_inner_repetition, TokenState, detect_loop


def main() -> int:
    # Synthetic loop: period ~37 chars, 8 repeats, total >> 60
    # Detector samples every 128 chars, so the text must be long enough.
    unit = "Wait, let me rethink this carefully. "
    text = unit * 8
    found, hit = find_inner_repetition(text)
    assert found, "should detect synthetic loop"
    assert hit is not None and hit.repeats >= 4
    print("OK find_inner_repetition:", hit)

    state = TokenState("")
    # fake tokens as single-char-ish strings
    toks = list("Hello") + list(unit * 5)
    state.append(toks)
    idx = state.char_to_token_index(hit.repeat_start)
    print("OK char_to_token_index:", idx)

    clean = "This is a normal answer with no repetition at all."
    found2, _ = find_inner_repetition(clean)
    assert not found2
    print("OK negative control")
    print("ALL DETECTION TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
