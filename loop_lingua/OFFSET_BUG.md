# Offset bug in `jspace/detection.py`

Paper-1 is not modified. This note is the record of the bug and of the
replacement in `offsets.py`.

## Rule that is not the bug

Loop *detection* in paper-1 runs on decoded text. A loop is a substring that
repeats at least 4 times and covers at least 60 characters
(`find_inner_repetition`). That part does not depend on token offsets.

The bug is the next step: mapping a character index in that decoded text
back to a generated token, which paper-1 does for the trigger-token readout.

## What the code does

`TokenState.append_ids` asks the tokenizer for each piece
(`convert_ids_to_tokens`) and decodes that piece alone
(`decode_token` → `fix_mojibake`). It stores `len(decoded)` and later
`char_to_token_index` walks the sum of those lengths.

Qwen3.5-4B is a byte-level BPE. The decoder concatenates the raw bytes of
the whole sequence and UTF-8-decodes **once**. A Telugu letter is three
UTF-8 bytes. When those bytes are split across pieces, each piece is not
valid UTF-8. `fix_mojibake` then keeps the byte-level characters (after
turning `Ġ` into a space). The reconstructed string is longer than
`tokenizer.decode`, so a character index into the real text lands on the
wrong token, or off the end of the reconstructed string.

English ASCII pieces are already valid UTF-8, so the same function matches.

## Measurement

MGSM questions (250 English, 250 Telugu), tokenizer
`Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,
unmodified `jspace.detection.TokenState`:

| Language | Texts whose per-token reconstruction ≠ full decode |
| --- | ---: |
| English | 0 / 250 |
| Telugu | 149 / 250 |

The same run's corrected map agreed with the tokenizer's character offsets
(first token whose span covers the index) on all 250 + 250 MGSM questions
and on all 588 MMLU-ProX-Lite Hindi test questions. Hindi is not immune:
58 of those 588 questions also mismatch under `TokenState`. The 149/250
figure the study started from is the MGSM Telugu count.

A minimal witness, taken from a real MGSM Telugu split:

```text
pieces:  Ġà°    ¡
bytes:   20 E0 B0   A1
decode:  " డ"     (U+0020, U+0C21)
jspace:  " à°¡"
```

The Telugu letter's first byte is in the first piece, so
`TokenOffsetMap.char_to_token_index` returns that piece. The second piece
holds only a continuation byte and has an empty span.

## What we did not change

`jspace/detection.py` is still the paper-1 implementation. D0 calls a
verbatim copy of `find_inner_repetition` and tests compare it to
`jspace.detection.find_inner_repetition`. The corrected map is
`loop_lingua.offsets.map_token_strings`.

`fix_mojibake` also maps U+2500–U+25FF onto bytes 0–255. That extra map is
not what produced the 149 Telugu mismatches. The corrected mapper uses only
the standard GPT-2 byte-to-unicode table.
