# Process

What was done, what was checked, and what is still open.
Updated 2026-10-30. This file is the status list; the draft PR should match it.

## Done

- Read the Antidoom rule in `PROJECT_SPEC.md`, `PROMPT.md`, and `jspace/detection.py`.
  At least 4 repeats, at least 60 characters, fingerprint 16, stride 128.
  Those characters are Python code points. Paper-1 files were not edited.
- Reproduced the offset bug with unmodified `jspace.detection.TokenState` and
  `Qwen/Qwen3.5-4B` tokenizer revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
  on MGSM: English 0/250, Telugu 149/250. Recorded in
  `corpus/offset_bug_mgsm.json` (no machine path). The corrected map matched
  tokenizer offsets on those 500 strings and on 588 Hindi MMLU-ProX-Lite
  test questions. 58 of those 588 Hindi questions also mismatch under `TokenState`.
- D0–D4, ISO 15919 romanization, fertility, script drift, power, and the
  96-generation planner.
- Built `corpus/`: 300 matched prompts × 4 languages from MMLU-ProX-Lite test,
  stratified, seed 20261030. MGSM's 250 pairs are a supplement (dropped from
  the 300: no Hindi). 100 Tenglish slots are null. Gold is a sidecar.
  Re-running `select_item_ids` on `pool_index.jsonl` reproduces the 300 ids.
- Docs: `README.md`, `ARCHITECTURE.md`, `DESIGN_DECISIONS.md`, `OFFSET_BUG.md`.
- Notebook `notebooks/t4_pilot.ipynb` dry-run only. The printed 17.07 h figure
  uses an assumed 40 tok/s. It is not a measurement.
- CPU tests: 40 passed (`python -m pytest tests/loop_lingua`).

## Verified by running code

- 149/250 and 0/250, twice (exploratory script, then `python -m loop_lingua.reproduce_bug`).
- Offset agreement with the `tokenizers` library on MGSM EN+TE and Hindi MMLU questions.
- D0 output equals `jspace.detection.find_inner_repetition` on the detector test strings.
- Corpus invariants: 300 shared ids, no Telugu-block characters in the romanized
  prompts, Hindi and Telugu prompts contain their scripts, Tenglish prompts are null,
  selection fields are only id, category, and languages.
- Licenses read from upstream on 2026-10-30: MMLU-ProX-Lite card says MIT;
  MGSM `LICENSE` is CC-BY-4.0; MMLU-Pro is Apache-2.0. A catalog page that said
  CC-BY-SA was not used.

## Assumed, not measured

- 22.9% is Liquid's Antidoom-mix loop rate for this model. It is only a planning prior.
- No tokens/second on a T4. Go/no-go is `pending` until someone measures it.
- The default 4.0 characters/token is a D3 scale anchor, not Qwen fertility.
- Grapheme breaking is a UAX #29 subset (no Hangul, no emoji flags).
- ISO 15919 was checked on examples and by "no Telugu code points left" on all
  300 romanized prompts, not by a native speaker.
- Instruction lines were written for this package.
- Greedy decoding vs Liquid's temperature 0.01 is untested.
- Constant tok/s and a full 2048-token cap both mis-state a real T4 run. The
  planner prints that caveat.
- `transformers_greedy_generate` was not executed. Qwen3.5 is multimodal;
  the right `generate` class was not confirmed.

## Remaining

CPU work left for a reviewer, not blocking the draft:

- Native-speaker pass on the Hindi and Telugu instruction lines and on a sample of romanization.
- Fill `corpus/tenglish_slots.json` by hand (100 prompts). Do not auto-mix.

GPU, on a T4, not done:

- Run the 96 jobs in `corpus/pilot_96.jsonl` with greedy decoding, cap 512.
- Record tokens/second from `run_harness` and call `plan_go_nogo(..., basis="measured")`.
- If the decision is `replan` or `no-go`, change the cap or the budget before the full study.
- Generate the 300×4 matched set. Score D0–D4 and `measure_drift`.
- Measure fertility with the live tokenizer and pass those rates into D3/D4.
  Do not leave the 4.0 anchor in the study.
- Only after the Tenglish slots are hand-edited: add that arm.
- Do not map trigger tokens with `jspace.detection.TokenState` on Telugu or Hindi.
  Use `loop_lingua.offsets`.
