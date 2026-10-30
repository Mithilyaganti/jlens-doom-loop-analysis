# Pre-onset doom-loop forecast — lfm2-2.6b

Question: does the J-lens workspace announce a doom loop before it starts, and how far ahead?

Data status: **insufficient**.

No forecasting metrics were computed.

- No forecasting metrics were computed. No usable trace was available: a usable trace needs the full generated text and the token ids from the baseline pass.
- The checkout has no per-trace generations. Baseline traces live under results/exp2_{slug}/{looping,nonlooping}/p*/ (meta.json + tokens.pt) and are gitignored. Onset token indices cannot be recomputed from the summary files.
- The generation log has 3 preview rows (generated_preview is truncated to 500 characters). Those rows were not labelled.
- results/checkpoints/baseline_pass.json lists 3 completed ids, which does not match results/baseline_pass_summary.json (200 completed, 11 loops). The summary counts were not recomputed.
- The Exp2 checkpoint records 200 completed hooked traces and 5 loops, but those trace directories are not in the checkout.
- Hash-locked split of the recorded baseline looping prompt ids (train 8, dev 1, test 2). A test fold that small cannot carry a four-horizon forecast even after the traces are present. These ids were not re-labelled; the onset detector did not see their text.

## Recorded counts (not re-estimated)

| File | Completed | Loops |
| --- | --- | --- |
| `results/baseline_pass_summary.json` | 200 | 11 |
| `results/checkpoints/baseline_pass.json` | 3 | None |
| `results/checkpoints/exp2_gen_lfm2-2.6b.json` | 200 | 5 |

Prompt sample: 200 ids, seed 42. Hash-locked split (`jspace-forecast-v1`, 60/20/20): train 121, dev 39, test 40.
Recorded baseline looping ids in that split: train 8, dev 1, test 2.

## Labelling

Usable traces: 0 of 0 loaded. Loops with an onset token: 0. Non-loops: 0.
Window length: 64 tokens. Horizons: [32, 64, 128, 256].
Character rule: at least 4 repeats and at least 60 characters.

## Detectors

The n-gram rule, the surface logistic / LightGBM models, and the CUSUM harness were not scored. There is no test-set estimate to report.

## Running this on Qwen traces

Leave the generation scripts alone. After `scripts/02_baseline_pass.py` has written `results/exp2_qwen3.5-4b/{looping,nonlooping}/p*/meta.json` and `tokens.pt`, run:

```
python scripts/07_forecast_report.py --slug qwen3.5-4b
```

Hidden-state or J-lens features, once a hooked run writes them, go in `results/forecast/feature_tables/*.npz` with arrays `prompt_ids`, `y`, `X`, `feature_names`, and optional `horizon`. The split is recomputed from the prompt id (`jspace-forecast-v1`, 60/20/20). CUSUM accepts any per-token score through `cusum_statistic`.
