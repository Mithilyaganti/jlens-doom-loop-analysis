# Process: loop-detector audit

## Steps

1. Read `jspace/detection.py` on `cursor/lfm2-exp1-jlens-fit` and kept it unchanged.
2. Added an exhaustive period scan and a fixture file of short loops, tail loops, and negatives.
3. Ran the tests and the audit script on CPU. No GPU, no model load.
4. Left `audit/human_loop_audit_sheet.csv` as a header row.

## Verified

Command:

```bash
python -m pytest tests/test_detector_audit.py -q
```

Output:

```
.......                                                                  [100%]
7 passed in 0.14s
```

Command:

```bash
python scripts/audit_loop_detector.py
```

Fixture recall at the published rule (4 repeats, 60 characters), taken from that command's JSON:

- `short`: 4 exhaustive positives, 1 sampler hit, 3 misses, recall 0.25
- `tail`: 2 exhaustive positives, 1 sampler hit, 1 miss, recall 0.5
- `corpus_recall.status`: `not_measured`
- checked-in HTML example (`results/exp2/example_looping_trace.html`): 12681 characters, exhaustive true, sampler true. That is one example, not a corpus recall.

The published sweep cell on the 8 fixtures was `n_exhaustive_positive` 6, `n_sampler_positive` 2, `n_agree` 4, sampler recall 0.333... Other grid cells are in the same JSON. They were not used to pick a new threshold.

## Assumed

- The vendored sampler in `jspace/detection.py` is the labeler the study runs. It was not re-derived from Liquid's GitHub in this change.
- "Short" and "tail" are the `kind` fields on the fixtures, not a claim about the length distribution of the 200-prompt run.

## Not measured

Study-corpus recall. The generation logs do not contain full strings. No loop rate was recomputed.
