# Process

What was done in this CPU package, what was checked, and what is still open.
Updated as the work proceeds. Times are UTC on 2026-10-30.

## Done

- Read the Antidoom rule in `PROJECT_SPEC.md`, `PROMPT.md`, and `jspace/detection.py`.
  The rule is unchanged: at least 4 repeats and at least 60 characters, fingerprint
  length 16, sample stride 128. Characters are Python code points.
- Did not edit paper-1 files (`jspace/`, `scripts/`, root `README.md`, `PROJECT_SPEC.md`).
- Reproduced the offset bug on MGSM with the Qwen3.5-4B tokenizer
  (`tokenizer.json`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`) and the
  unmodified `jspace.detection.TokenState`:
  - English: 0/250 texts mis-reconstructed
  - Telugu: 149/250 texts mis-reconstructed
  - The failure is per-token UTF-8 decode of byte-level pieces that split a character.
    Example pieces `Ġà°` + `¡` are the character sequence ` డ` only after the bytes
    are concatenated.
- Correct mapper is `loop_lingua.offsets`. On that same run it matched the
  tokenizer's first-covering character offsets for MGSM English (250), MGSM
  Telugu (250), and MMLU-ProX-Lite Hindi questions (588).
- D0 is a verbatim copy of the Antidoom sampler. D1–D4 are exact searches with
  code-point, grapheme, token, and fertility-scaled thresholds.
- Telugu romanization is ISO 15919 in `transliterate.py`.
- Prompt sampler reads only item id, category, and which languages exist.
  MGSM has no Hindi, so it cannot enter the matched 300. Licenses were read
  from the upstream files on 2026-10-30 (see `licenses.py`).
- Power formula and the 96-generation T4 go/no-go planner run on CPU.
  No model was loaded.

## Verified

- Bug counts 149/250 and 0/250, by executing `jspace.detection.TokenState` on
  MGSM questions encoded with the Qwen tokenizer. Not assumed.
- Offset mapper agreement with `tokenizers` offsets on those texts. Not assumed.
- License texts: MMLU-ProX-Lite card says MIT; MGSM `LICENSE` file is CC-BY-4.0
  (not the CC-BY-SA label on a third-party catalog page); MMLU-Pro is Apache-2.0.

## Assumed (not measured)

- Liquid's 22.9% Qwen3.5-4B loop rate is a planning prior from their Antidoom
  mix. It is not a loop rate on these prompts.
- T4 tokens/second. The pilot has not been run. `--assume-tok-s` is a what-if.
- Grapheme breaking is a UAX #29 subset (no Hangul, no emoji flags).
- ISO 15919 was checked on examples, not by a native-speaker pass over all 300.
- Instruction lines were written for this package, not taken from the datasets.
- Greedy decoding is the pilot plan. Liquid used temperature 0.01.
- Constant tokens/second from a short pilot overstates long-context speed.

## Remaining

- CPU unit tests currently passing: 34 (`python -m pytest tests/loop_lingua`). Corpus and notebook tests are not in that count yet.
- Build and commit the 300-prompt corpus, Tenglish placeholder slots, and MGSM supplement.
- Write README, ARCHITECTURE, DESIGN_DECISIONS, and OFFSET_BUG.
- Add the T4 notebook (dry-run only).
- Re-run the full CPU test suite and record the count here.
- Self-review pass.
- GPU still to do, on a T4, not in this package run:
  - 96-generation throughput measurement
  - go/no-go from that measurement
  - full 300×4 generation and loop-rate comparison
  - fertility measured with the live Qwen tokenizer on the committed prompts
  - hand edit of the 100 Tenglish slots
