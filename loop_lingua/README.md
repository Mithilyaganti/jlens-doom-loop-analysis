# Loop-Lingua

CPU tools for the question: do doom loops depend on language?

Languages: English, Hindi, Telugu, romanized Telugu (ISO 15919), and a
placeholder for 100 hand-edited code-mixed Tenglish prompts. The model in
the plan is `Qwen/Qwen3.5-4B`. Nothing here loads that model.

Paper-1 (`jspace/`) is not modified. Read `OFFSET_BUG.md` for the token-offset
bug, `ARCHITECTURE.md` for the data flow, `DESIGN_DECISIONS.md` for the
choices, and `PROCESS.md` for what was checked versus what is still open.

## Layout

```text
loop_lingua/
  antidoom_rule.py    D0, the original Antidoom sampler
  offsets.py          corrected token → character map
  detectors.py        D0–D4
  prompts.py          matched-sample builder
  transliterate.py    Telugu → ISO 15919
  fertility.py        tokens per character
  drift.py            output script vs prompt language
  power.py            two-proportion power
  pilot.py            96-generation harness and go/no-go
  pilot_t4.py         CLI, dry-run unless you are on a T4
  corpus/             300 matched prompts and the MGSM supplement
notebooks are under loop_lingua/notebooks/
```

The Antidoom rule, copied and checked against `jspace.detection`: a substring
that repeats at least 4 times and covers at least 60 characters. Detection
is on decoded text. The 60 is a count of Python code points.

| Variant | What the 60 means |
| --- | --- |
| D0 | 60 code points, original 128-character sampler |
| D1 | 60 code points, exact scan |
| D2 | 60 grapheme clusters |
| D3 | token span equal to 60 English-reference characters |
| D4 | 60 characters rescaled by measured characters-per-token |

## Tests

From the repo root, on CPU:

```bash
python -m pytest tests/loop_lingua -q
```

Core tests use the standard library plus `jspace` (imported, not edited).
Re-measuring the 149/250 figure needs the Qwen tokenizer file and the MGSM
TSVs in `corpus/mgsm/`. If the tokenizer is not cached, that one test skips.
The committed count is `corpus/offset_bug_mgsm.json`.

```bash
python -m loop_lingua.reproduce_bug
python -m loop_lingua.power
python -m loop_lingua.pilot_t4
python -m loop_lingua.pilot_t4 --assume-tok-s 40
```

`--assume-tok-s` is a what-if. It is not a T4 measurement. `--run` without
`--i-have-a-t4` refuses to start a GPU job.

## Corpus

`corpus/prompts_matched.jsonl` has 300 items × 4 languages. Selection used
item id, category, and language availability. It did not use gold answers.
`corpus/gold.jsonl` is a sidecar for later scoring. `corpus/tenglish_slots.json`
has 100 null prompts. `corpus/mgsm_supplement.jsonl` is the 250 English–Telugu
MGSM pairs, kept out of the 300 because MGSM has no Hindi. Licenses are in
`corpus/DATA_NOTICES.md`.

Rebuild (downloads the MMLU-ProX-Lite test shards if they are not cached):

```bash
python -m loop_lingua.build_dataset
```

## Not done on a GPU

Throughput, the real go/no-go, loop rates, and fertility under the live
tokenizer. See `notebooks/t4_pilot.ipynb` for the pilot procedure. Do not
treat any number it prints from `--assume-tok-s` as a result.
