# Architecture: loop-detector audit

The study's loop label is `jspace.detection.find_inner_repetition`. That function is unchanged. This change adds a read-only audit beside it.

```
tests/fixtures/detector_loops.jsonl
        │
        ▼
jspace/detector_audit.py
  exhaustive_repetition   period scan, numeric rule only
  sampler_hit             calls find_inner_repetition
  recall_for_kind         hits / exhaustive positives, or not_measured
  threshold_sweep         agreement grid, no chosen winner
  score_checked_in_html   one example trace, not a corpus rate
        │
        ▼
scripts/audit_loop_detector.py    stdout JSON, does not write results/
audit/human_loop_audit_sheet.csv  header only
```

`jspace/detection.py` stays the labeler used by Exp2 and Exp3. The audit does not import generation, torch, or the experiment scripts, so it cannot rewrite a checkpoint.
