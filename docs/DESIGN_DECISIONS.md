# Design decisions: loop-detector audit

## What was compared

The published sampler fingerprints 16 characters every 128 characters, then verifies a candidate period. An exhaustive scan walks every period from 1 up to `min(1024, n // min_repeats)` and accepts the first run whose whole-period span has at least 4 repeats and at least 60 characters. Both use that same numeric rule. The comparison is sampler versus exhaustive, not a third detector with its own threshold.

## Alternatives considered

- Relabeling the checked-in Exp2/Exp3 tables with the exhaustive scan. Rejected. Those tables were produced by the sampler, and the full generation strings are not in the checkout, so a new label would be invented.
- Tuning `min_repeats` or `min_total_repeated` from the sweep. Rejected. The sweep reports agreement. It does not select a replacement for the Antidoom defaults.
- Filling the human-audit sheet from the fixture labels. Rejected. Fixture labels are machine checks. The sheet is for a person reading a trace.

## Why fixtures are the recall denominator

`results/trigger_gen_log.jsonl` stores `generated_preview` cutoffs, not the strings that were labeled. Recall on that file would treat a prefix as a full trace. The denominator is therefore the eight strings in `tests/fixtures/detector_loops.jsonl`, split into `short` and `tail`. The HTML file `results/exp2/example_looping_trace.html` is scored as one example and is not called a recall.

## Residual choice

The exhaustive hit returns the shortest qualifying period. The sampler often reports a longer fingerprint-aligned period. Equality of period is not required for a hit. A hit is the boolean label only.
