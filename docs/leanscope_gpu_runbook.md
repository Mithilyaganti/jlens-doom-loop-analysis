# LeanScope GPU runbook (Qwen3.5-4B)

This is the model run for paper 2. It was not executed here. CPU tests use a fake model and do not download Qwen. Do not start this run until the anti-doom Qwen bf16 baseline and the workspace-band file exist.

The command list, in order, is `scripts/lean_run_all.sh`. Set `LEANSCOPE_LAYERS` before you run it. The script exits immediately if that variable is empty, so a missing band does not start generation.

## What this run is

Installed and induced leans in an open 4B model. The model is `Qwen/Qwen3.5-4B` in bf16 with its pre-fitted J-lens (`neuronpedia/jacobian-lens`, file `qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens.pt`). No claim about Claude. No human subjects. The alternatives-panel numbers are a Tier A pre-test. Every panel report is labelled `pre-test, not evidence about people`.

DeepSeek V4.1 Flash is not called. The drafter and the judge interfaces refuse a network call. The GPU scripts use the offline fake judge and the offline fake analyst.

## Setup

```bash
export JLENS_MODEL=Qwen/Qwen3.5-4B
export JLENS_HF_QUANTIZE=0
export PYTHONPATH="$PWD"
```

`JLENS_HF_QUANTIZE=0` loads bf16. The laptop default is 4-bit and is the wrong precision for this run.

Confirm the lens file resolves through `jspace.loading.load_stack` and that `scripts/01_workspace_band.py` has written `results/workspace_band_qwen3.5-4b.json`. Read that file and set:

```bash
export LEANSCOPE_LAYERS=18,20,22   # replace with the band's layer list
```

`18,20,22` is only an illustration from the causal script's docstring. It is not a measured Qwen band.

## Steps

Run from the repo root. Each step reads the previous step's file.

### 1. Build items (CPU)

```bash
python scripts/lean_00_build_items.py \
  --update-lock \
  --out data/leanscope/records.jsonl
```

Reads `data/leanscope/seed_scenarios.jsonl`. Expands 5 paraphrases and both option orders. Assigns one coin-flip lean per `scenario_id` (seed 42) and the split in `data/leanscope/split_lock.json`. `--update-lock` appends new scenario ids and does not move ids already in the lock.

Current seed, counted from the file in this branch: **72 scenarios, 24 in each family** (analysis, product, advice). Splits in the lock: train 43, dev 11, test 18. Conditions per surface: 14 (none, trait, and for each of the four strengths an assigned lean, a structure-matched neutral, and a conceal prompt).

```text
72 x 5 paraphrases x 2 orders x 14 conditions = 10080 records
```

Writes:

- `data/leanscope/records.jsonl`
- `data/leanscope/records.manifest.json` with `n_records`, `n_scenarios`, and the by-family / by-kind / by-split counts

Expect `n_records` equal to scenarios x 140. If the seed grew after this document was written, trust the manifest, not the 10080 figure.

Runtime: seconds on CPU. No model.

### 2. Generate replies

```bash
python scripts/lean_01_generate.py \
  --records data/leanscope/records.jsonl \
  --out results/leanscope/records_generated.jsonl \
  --max-new-tokens 300
```

Greedy generation through `jspace.generation`. Resumes from `--out`. This step fills `output_text` and `recommended`. It does not write the P0/P1 residual files that detection and the causal test read. Leave `--activations-dir` unset here so you do not pay for a second activation format.

Optional second pass, not in `lean_run_all.sh`:

```bash
python scripts/lean_01_generate.py \
  --records results/leanscope/records_generated.jsonl \
  --out results/leanscope/records_generated.jsonl \
  --natural-samples 20 \
  --sample-temperature 0.7
```

That draws 20 samples for each `none` and `trait` row only, and writes `records_generated.natural.jsonl` plus `records_generated.natural_report.json` (majority option and agreement). It does not sample the assigned-lean rows.

### 3. Record residuals at P0 and P1

```bash
python scripts/lean_02_record.py \
  --records results/leanscope/records_generated.jsonl \
  --out results/leanscope/records_positions.jsonl \
  --activations-dir results/leanscope/activations \
  --max-new-tokens 64
```

Regenerates each item and stores residuals at P0 (last prompt token) and P1 (token before the first option word). P2, the end of thinking, stays off unless you pass `--record-p2`. One torch file per item: `results/leanscope/activations/<item_id>.pt`, with `residuals["P0"]` and `residuals["P1"]` keyed by layer. The output JSONL is the same records with `output_text`, `recommended`, and `activations_path` filled.

`--all-layers` records every layer instead of the workspace band. The default is the band, which is what D1 averages.

### 4. Detectors

```bash
python scripts/lean_03_detect.py \
  --records results/leanscope/records_positions.jsonl \
  --out results/leanscope/detection.csv \
  --scores-out results/leanscope/d1_scores.json \
  --samples 0 \
  --n-boot 1000
```

Writes:

- `results/leanscope/detection.csv` and `detection.md` (AUROC by detector, position, layer, strength, with scenario-clustered bootstrap intervals)
- `results/leanscope/detection.reliability.json`
- `results/leanscope/d1_scores.json`: D1 at P0, workspace-band mean, as `{item_id: score}` where the score is option A minus option B

D1 is the J-lens concept readout. D2 is the same readout through the unembedding. D3 and D4 are fit on the train split. D5 is the offline text judge. D6 is skipped because `--samples 0`. D7 is the system-prompt oracle and is a ceiling, not a result. `--samples 20` would add twenty generations per item on every row. Do not turn that on for the full 10080-row file. Natural-lean samples belong on step 2.

### 5. Causal test, steered on D1

```bash
python scripts/lean_04_causal.py \
  --records results/leanscope/records_positions.jsonl \
  --direction jlens \
  --split test \
  --layers "$LEANSCOPE_LAYERS" \
  --positions P0,P1 \
  --alpha 1.0 \
  --max-new-tokens 256 \
  --out results/leanscope/causal
```

`--direction jlens` plugs in `leanscope.detectors.make_jlens_direction_fn`: the J-space direction of option A's tokens minus option B's, one vector per item, because option text is not shared across scenarios. `--direction diffmeans` is the D4-style mean residual (lean A minus lean B) on the train split, one shared vector. The random control is a norm-matched direction. The positive control regenerates with the lean prompt removed.

Writes `results/leanscope/causal/stats.json`, `per_item.jsonl`, and `summary.md`. The summary has flip rate, exact McNemar, Wilson interval, and Cohen's h for each edit. `stats.json` field `direction` is `D1_jlens` for this command.

The script runs every test-split row whose condition has a lean, which is the assigned and conceal rows. At 18 test scenarios that is 18 x 10 surfaces x 8 leaned conditions = 1440 items, and each item is generated about ten times (baseline, positive control, steer and ablate of the option direction and of the random direction, at P0 and at P1).

### 6. Alternatives panel

```bash
python scripts/lean_05_panel_sim.py \
  --records results/leanscope/records_positions.jsonl \
  --scores results/leanscope/d1_scores.json \
  --prior-opposite \
  --analyst fake \
  --max-new-tokens 256 \
  --out results/leanscope/panel
```

The C2 threshold is chosen on the dev split from `|D1 score|`. C2 shows the other option's case only when the score crosses that threshold. The case is a rerun of Qwen under the opposite-lean prompt. C0 is the recommendation alone, C1 always shows the other case, C3 is a disclosure line. `--analyst fake` is the offline stand-in. `--analyst deepseek` exits before any request.

`--prior-opposite` sets each simulated user's prior to the option the stored recommendation did not name. That is the disagreement subset, not a measured preference.

Writes `results/leanscope/panel/simulation.json`, `views.jsonl`, and `summary.md`. The first line of the summary is `pre-test, not evidence about people`.

## Runtime estimates

These are arithmetic on an assumed 50 tokens/s of decode on an L4, plus one extra forward where a step records activations. They are not measurements. The plan's 2 to 4 L4-hour envelope was for about 500 items. This seed is larger.

| Step | Work, at 72 scenarios | Rough L4 time |
| --- | --- | --- |
| 1 Build | 10080 records, no model | seconds, CPU |
| 2 Generate | 10080 x 300 new tokens | about 17 hours of decode. Budget 20 to 30 L4-hours with resume and checkpoint overhead |
| 2b Natural samples | 72 x 10 surfaces x 2 no-prompt conditions x 20 samples x 300 tokens = 8640000 tokens | about 48 hours. Leave this off the first pass |
| 3 Record | 10080 x 64 new tokens, plus one residual forward | about 4 to 8 L4-hours |
| 4 Detect | reads the activation files, `--samples 0` | CPU, well under an hour if the files are local |
| 5 Causal | about 1440 test items x 10 generations x 256 tokens | about 20 L4-hours. Cut it by passing a smaller JSONL (one paraphrase, explicit and indirect only) |
| 6 Panel | one opposite-lean generation per eval item that shows a case | a few L4-hours on the test split, less than the causal step |

A first GPU session that can finish in an afternoon: steps 1 to 4 with `lean_01` and `lean_02` `--limit 200`, then step 4, then step 5 on that same 200-row file. Check three things before the full bank: the model follows an explicit lean on most of those rows, at least one detector is above chance on the indirect rows, and D1 and D2 are not identical. Those are the plan's pilot gates, stated there as speculative thresholds of 70% follow and AUROC 0.75 on soft leans.

Scaling: if the seed moves from 72 to N scenarios, multiply the generation and record hours by N/72. The split lock does not move existing ids when you append.

## What the CPU suite already checked

`python -m pytest` runs `tests/` only. It includes `tests/test_leanscope_e2e.py`, which builds a data-layer slice, records it with the fake, scores D1, steers with the J-lens option direction, and opens the panel when that score crosses the dev threshold. The fake's plant is strong for explicit, weak for indirect, and zero for neutral, so the explicit AUROC is higher than the indirect AUROC. That ordering is a property of the fake. It is not a Qwen result.

## Remaining unknowns

Check these off on the GPU machine. None of them were measured in the CPU suite.

- [ ] The workspace-band layer list for Qwen3.5-4B. `results/workspace_band_qwen3.5-4b.json` is not in the repo. `LEANSCOPE_LAYERS` has to come from that file.
- [ ] Whether thinking mode should be on. P2 is off unless `--record-p2` is passed. The plan left this as a pilot decision.
- [ ] Whether the pre-fitted J-lens encodes multi-token option names. If the readout collapses, fall back to single-token labels A and B plus the description. The recorder already records a label fallback when the option name does not encode.
- [ ] G1. Does the model follow an explicit lean on at least about 70% of items? The plan's threshold is speculative.
- [ ] G2. Does any detector reach about AUROC 0.75 on soft leans for held-out scenarios? Explicit detection is expected to be near ceiling and is a sanity check, not a result. Soft and indirect are the open question. On the loyalty-probe paper they were weak. This fake cannot speak to that.
- [ ] G3. Do D1 and D2 differ by a measurable amount, in either direction?
- [ ] Silent leans. How often behaviour follows the assigned lean while the text does not advocate it? If that subset is tiny, the conceal condition is the one to report.
- [ ] Natural-lean label noise. Agreement across the 20 samples, and across a rerun. `none` and `trait` have no prompt. `trait.lean` stays empty until a behavioural majority is written.
- [ ] Causal alpha and layers. `alpha 1.0` is the script default, not a tuned value. A flip rate near the random control means the direction or the scale is wrong.
- [ ] Per-item J-lens direction versus one difference-of-means vector. The runbook uses per-item D1 because option strings differ. If that direction does not flip answers and difference-of-means does, report both.
- [ ] One generation pass versus two. Step 2 writes the text the data layer labels. Step 3 generates again to store P0/P1. Combining them would cut the decode roughly in half. They are separate because the position file is the format D1 and the causal loader read.
- [ ] Exact L4-hours. The table above is arithmetic at 50 tokens/s. Time the first 50 prompts and rescale.
- [ ] Tier A validity. The panel follow-rate drop is a pre-test of the pipeline. It is not evidence about people.
- [ ] A paid judge is still unwired on purpose. Do not point D5 or the analyst at DeepSeek until there is a separate decision to spend money.
- [ ] LFM2 transfer and a human study are out of this run.
