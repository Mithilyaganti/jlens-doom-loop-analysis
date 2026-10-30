# LeanScope GPU runbook (Qwen3.5-4B)

This is the model run for paper 2. It was not executed here. CPU tests use a fake model and do not download Qwen. Do not start this run until the anti-doom Qwen bf16 baseline and the workspace-band file exist.

The command list, in order, is `scripts/lean_run_all.sh`. Set `LEANSCOPE_LAYERS` before you run it. The script exits immediately if that variable is empty. `LEANSCOPE_PRESET` defaults to `main` (`pilot`, `main`, or `full`).

## What this run is

Installed and induced leans in an open 4B model. The model is `Qwen/Qwen3.5-4B` in bf16 with its pre-fitted J-lens (`neuronpedia/jacobian-lens`, file `qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens.pt`). No claim about Claude. No human subjects. The alternatives-panel numbers are a Tier A pre-test. Every panel report is labelled `pre-test, not evidence about people`.

DeepSeek V4.1 Flash is not called. The drafter and the judge interfaces refuse a network call. The GPU scripts use the offline fake judge and the offline fake analyst.

## Setup

```bash
export JLENS_MODEL=Qwen/Qwen3.5-4B
export JLENS_HF_QUANTIZE=0
export PYTHONPATH="$PWD"
export LEANSCOPE_PRESET=main          # pilot for the first session
export LEANSCOPE_LAYERS=18,20,22      # replace with the band file's layers
```

`JLENS_HF_QUANTIZE=0` loads bf16. The laptop default is 4-bit and is the wrong precision for this run.

`18,20,22` is only an illustration from the causal script's docstring. It is not a measured Qwen band. There is no Qwen workspace-band file in the repo (the LFM2 band is a different model). After step 0, read `results/workspace_band_qwen3.5-4b.json` and export its layer list. The slug comes from `jspace.model_config` (`workspace_band_<slug>.json`).

## The three presets

`scripts/lean_00_build_items.py` always writes the full expansion unless you pass `--preset`. `lean_run_all.sh` writes that full file, then `lean_01_generate.py --preset` drops rows. A kept scenario keeps its locked split and its coin-flip lean. Presets do not re-roll seed 42 and do not move the lock.

Seed in this branch: **100 scenarios** (34 analysis, 33 product, 33 advice). Lock: **60 train / 15 dev / 25 test**.

| Preset | Records generated | Natural-lean samples | What is dropped |
| --- | --- | --- | --- |
| `pilot` | 800 | 10 on each kept `none` row (1,600 generations). 40 scenarios, 24/6/10, families kept even, paraphrases p0–p1, both orders, `none` plus explicit and soft assigned and neutral | p2–p4, trait, conceal, indirect, conflicting, and 60 scenarios |
| `main` | 6,000 | 10 on the locked dev `none` and `trait` rows (1,800 generations). All 100 scenarios, p0–p2, both orders, `none`, `trait`, and all four assigned and neutral strengths | p3–p4, conceal, and natural samples outside dev |
| `full` | 14,000 | off. 100 × 5 paraphrases × 2 orders × 14 conditions | nothing |

`lean_00` and `lean_01` print `records=` and `estimated_l4_hours=` before any GPU work. That printed hour **includes** a residual forward on the greedy pass (greedy hours = 1.5 × decode). This command list does not pass `--activations-dir` to `lean_01`. Activations are step 3. So the printed number is longer than step 2 by half of the greedy decode: 0.7 h on pilot (4.7 printed), 5.0 h on main (18.0 printed), 11.7 h on full (35.0 printed). The optional full natural pass (`--natural-samples 20`, 40,000 generations) is what brings the printed full estimate to about 101.7 L4-hours. It is not in `lean_run_all.sh`.

## Steps

Run from the repo root, or run `bash scripts/lean_run_all.sh` after the two exports above. Each step reads the previous step's file. Paths below use `main`. For another preset, replace `main` with `$LEANSCOPE_PRESET`.

### 0. Workspace band

```bash
python scripts/01_workspace_band.py
```

Writes `results/workspace_band_qwen3.5-4b.json`, `results/workspace_band_metrics_qwen3.5-4b.json`, and `results/workspace_band_metrics.png`. Then:

```bash
export LEANSCOPE_LAYERS=<layers from that JSON>
```

One model load and a short forward sweep. Budget under an L4-hour. Not measured.

### 1. Build items (CPU)

```bash
python scripts/lean_00_build_items.py \
  --update-lock \
  --out data/leanscope/records.jsonl
```

Reads `data/leanscope/seed_scenarios.jsonl`. Expands 5 paraphrases and both option orders. Assigns one coin-flip lean per `scenario_id` (seed 42) and the split in `data/leanscope/split_lock.json`. `--update-lock` appends new scenario ids and does not move ids already in the lock. The default preset is `full`, which is what the shell script uses, so the file is the full bank even when generation uses `main`.

```text
100 x 5 paraphrases x 2 orders x 14 conditions = 14000 records
```

Conditions: `none`, `trait`, and for each of explicit, soft, indirect, conflicting an assigned lean, a structure-matched neutral, and a conceal prompt.

Writes:

- `data/leanscope/records.jsonl`
- `data/leanscope/records.manifest.json` with `n_records` 14000, `n_scenarios` 100, and the by-family / by-kind / by-split counts

The script also prints one estimate line per preset (pilot 4.7, main 18.0, full 35.0) plus the optional natural-sample line (about 101.7). Those lines assume activations on the greedy pass. See the note above.

Runtime: seconds on CPU. No model.

### 2. Generate replies

```bash
python scripts/lean_01_generate.py \
  --records data/leanscope/records.jsonl \
  --preset main \
  --out results/leanscope/records_main.jsonl \
  --max-new-tokens 300
```

Greedy generation through `jspace.generation`. Resumes from `--out`. Fills `output_text` and `recommended` on the preset's rows only. Does not write the P0/P1 files that detection and the causal test read.

`main` also draws 10 natural-lean samples on the dev `none` and `trait` rows and appends `results/leanscope/records_main.natural.jsonl` plus `records_main.natural_report.json` (majority option and agreement). `pilot` does the same for every kept `none` row (`records_pilot.natural.jsonl`). `full` writes no natural file unless you pass `--natural-samples`.

Expect `preset=main records=6000` on the first line. The `estimated_l4_hours=18.0` on that line counts an activation forward this step does not do. Decode-only time is in the table below.

### 3. Record residuals at P0 and P1

```bash
python scripts/lean_02_record.py \
  --records results/leanscope/records_main.jsonl \
  --out results/leanscope/records_main_positions.jsonl \
  --activations-dir results/leanscope/activations_main \
  --max-new-tokens 64
```

Regenerates each item and stores residuals at P0 (last prompt token) and P1 (token before the first option word). P2, the end of thinking, stays off unless you pass `--record-p2`. One torch file per item: `results/leanscope/activations_main/<item_id>.pt`, with `residuals["P0"]` and `residuals["P1"]` keyed by layer. The output JSONL is the same records with `output_text`, `recommended`, and `activations_path` filled. For `main`, that is 6,000 files.

`--all-layers` records every layer instead of the workspace band. The default is the band, which is what D1 averages.

### 4. Detectors

```bash
python scripts/lean_03_detect.py \
  --records results/leanscope/records_main_positions.jsonl \
  --out results/leanscope/detection_main.csv \
  --scores-out results/leanscope/d1_scores_main.json \
  --samples 0 \
  --n-boot 1000
```

Writes:

- `results/leanscope/detection_main.csv` and `detection_main.md` (AUROC by detector, position, layer, strength, with scenario-clustered bootstrap intervals)
- `results/leanscope/detection_main.reliability.json`
- `results/leanscope/d1_scores_main.json`: D1 at P0, workspace-band mean, as `{detector, position, layer, scores: {item_id: score}}`. The score is option A minus option B. `lean_05` reads that `scores` map.

D1 is the J-lens concept readout. D2 is the same readout through the unembedding. D3 and D4 are fit on the train split. D5 is the offline text judge. D6 is skipped because `--samples 0`. D7 is the system-prompt oracle and is a ceiling, not a result. `--samples 20` would add twenty generations per item on every row. Do not turn that on for a 6,000-row or 14,000-row file. Natural-lean samples belong on step 2.

This step loads Qwen once to build option directions, then reads the activation files. It is not a decode pass. Budget under an L4-hour for `main` once the files are local.

### 5. Causal test, steered on D1

```bash
python scripts/lean_04_causal.py \
  --records results/leanscope/records_main_positions.jsonl \
  --direction jlens \
  --split test \
  --layers "$LEANSCOPE_LAYERS" \
  --positions P0,P1 \
  --alpha 1.0 \
  --max-new-tokens 256 \
  --out results/leanscope/causal_main
```

`--direction jlens` plugs in `leanscope.detectors.make_jlens_direction_fn`: the J-space direction of option A's tokens minus option B's, one vector per item, because option text is not shared across scenarios. `--direction diffmeans` is the train-split mean residual (lean A minus lean B), one shared vector. The random control is a norm-matched direction. The positive control regenerates with the lean prompt removed.

Writes `results/leanscope/causal_main/stats.json`, `per_item.jsonl`, and `summary.md`. The summary has flip rate, exact McNemar, Wilson interval, and Cohen's h for each edit. `stats.json` field `direction` is `D1_jlens` for this command.

The script generates for every test-split row whose condition lean is A or B. On `main` that is the assigned rows only (`none`, `trait`, and `neutral_matched` have no lean). 25 test scenarios × 3 paraphrases × 2 orders × 4 strengths = **600 items**, and each item is generated 10 times (baseline, positive control, and steer/ablate of the option direction and of the random direction, at P0 and at P1). Editing several layers is one generation, not one per layer.

`pilot` is 10 test scenarios × p0–p1 × 2 orders × explicit and soft = **80 items**. `full` adds conceal and p3–p4, so 25 × 5 × 2 × 8 leaned conditions = **2,000 items**.

### 6. Alternatives panel

```bash
python scripts/lean_05_panel_sim.py \
  --records results/leanscope/records_main_positions.jsonl \
  --scores results/leanscope/d1_scores_main.json \
  --prior-opposite \
  --analyst fake \
  --max-new-tokens 256 \
  --out results/leanscope/panel_main
```

The C2 threshold is chosen on the dev split from `|D1 score|`. C2 shows the other option's case only when the score crosses that threshold. C1 always shows it, so the script generates the opposite-lean case once per test item that has a recommendation. C0 is the recommendation alone. C3 is a disclosure line. `--analyst fake` is the offline stand-in. `--analyst deepseek` exits before any request.

`--prior-opposite` sets each simulated user's prior to the option the stored recommendation did not name. That is the disagreement subset, not a measured preference.

Writes `results/leanscope/panel_main/simulation.json`, `views.jsonl`, and `summary.md`. The first line of the summary is `pre-test, not evidence about people`.

Test rows that can request a case: pilot 200, main 1,500, full 3,500.

## Runtime estimates

Arithmetic at an assumed 50 tokens/s of decode on an L4. Not a measurement. Time the first 50 prompts and rescale. Model-load time (steps 0, 4, 5, and 6 each load the stack) is not in the decode column.

Decode hours = generations × new tokens / 50 / 3600. Step 3 also does one residual forward; that column charges it as half of the 64-token decode, the same 1.5 factor `leanscope/presets.py` uses. A long prompt can make that forward cost more.

| Step | pilot | main (default) | full |
| --- | --- | --- | --- |
| 0 Band | under 1 h, one load | under 1 h | under 1 h |
| 1 Build | seconds, CPU, 14,000-row file | same | same |
| 2 Generate, no activations | 4.0 h (800 × 300 tok + 1,600 natural) | 13.0 h (6,000 × 300 tok + 1,800 natural) | 23.3 h (14,000 × 300 tok, natural off) |
| 3 Record P0/P1 | 0.4 h | 3.2 h | 7.5 h |
| 4 Detect, `--samples 0` | under 1 h, not decode | under 1 h | under 1 h if the 14,000 files are local |
| 5 Causal, D1, test split | 1.1 h (80 × 10 × 256 tok) | 8.5 h (600 × 10 × 256 tok) | 28.4 h (2,000 × 10 × 256 tok) |
| 6 Panel, one case each test row | 0.3 h (200 × 256 tok) | 2.1 h (1,500 × 256 tok) | 5.0 h (3,500 × 256 tok) |
| Decode total, steps 2–3 and 5–6 | about 6 h | about 27 h | about 64 h |

The printed preset hours (4.7 / 18.0 / 35.0) are step 2 **with** activations, and they do not include steps 3, 5, or 6. Adding `--natural-samples 20` on the full file is a separate ~67 h of decode (about 102 h together with the 35 h activated greedy pass). Leave it off until the gates below pass.

First GPU session: `LEANSCOPE_PRESET=pilot`. That is the whole chain at about 6 L4-hours. Before spending the main budget, check three things on that pilot: the model follows an explicit lean on most of those rows, at least one detector is above chance on the indirect rows (indirect is not in `pilot`; use a `main` slice or add indirect to a 200-row `--limit` run), and D1 and D2 are not identical. The plan's pilot gates, stated there as speculative thresholds, are about 70% follow on explicit leans and AUROC 0.75 on soft leans.

A 200-row gate is still available by adding `--limit 200` to `lean_01` and `lean_02` and pointing steps 4–6 at that shorter file. `lean_run_all.sh` does not pass `--limit`.

## What the CPU suite already checked

`python -m pytest` runs `tests/` only (`pytest.ini` sets `testpaths = tests`). It includes `tests/test_leanscope_e2e.py`, which builds a data-layer slice, records it with the fake, scores D1, steers with the J-lens option direction, and opens the panel when that score crosses the dev threshold. The fake's plant is strong for an explicit lean, weak for an indirect lean, and zero for a neutral prompt, so the explicit AUROC is higher than the indirect AUROC. That ordering is a property of the fake. It is not a Qwen result.

## Remaining unknowns

Check these off on the GPU machine. None of them were measured in the CPU suite.

- [ ] The workspace-band layer list for Qwen3.5-4B. `results/workspace_band_qwen3.5-4b.json` is not in the repo. `LEANSCOPE_LAYERS` has to come from that file. Do not treat 18, 20, 22 as measured.
- [ ] Whether thinking mode should be on. P2 is off unless `--record-p2` is passed. The plan left this as a pilot decision.
- [ ] Whether the pre-fitted J-lens encodes multi-token option names. If the readout collapses, fall back to single-token labels A and B plus the description. The recorder already records a label fallback when the option name does not encode.
- [ ] G1. Does the model follow an explicit lean on at least about 70% of items? The plan's threshold is speculative.
- [ ] G2. Does any detector reach about AUROC 0.75 on soft leans for held-out scenarios? Explicit detection is expected to be near ceiling and is a sanity check, not a result. Soft and indirect are the open question. On the loyalty-probe paper they were weak. This fake cannot speak to that. `pilot` has explicit and soft only; indirect and conflicting are on `main`.
- [ ] G3. Do D1 and D2 differ by a measurable amount, in either direction?
- [ ] Silent leans. How often behaviour follows the assigned lean while the text does not advocate it? If that subset is tiny, the conceal condition is the one to report. Conceal is on `full` only.
- [ ] Natural-lean label noise. Agreement across the 10 samples (`main` and `pilot`) or 20 (`full`, if you turn it on), and across a rerun. `none` and `trait` have no prompt. `trait.lean` stays empty until a behavioural majority is written.
- [ ] Causal alpha and layers. `alpha 1.0` is the script default, not a tuned value. A flip rate near the random control means the direction or the scale is wrong.
- [ ] Per-item J-lens direction versus one difference-of-means vector. The runbook uses per-item D1 because option strings differ. If that direction does not flip answers and difference-of-means does, report both. `--direction diffmeans` is the comparison.
- [ ] One generation pass versus two. Step 2 writes the text the data layer labels. Step 3 generates again to store P0/P1. Combining them would cut the greedy decode roughly in half. They are separate because the position file is the format D1 and the causal loader read.
- [ ] Exact L4-hours. The table is arithmetic at 50 tokens/s. Time the first 50 prompts and rescale. The prompt-length residual forward in step 3 may exceed the half-decode allowance.
- [ ] Tier A validity. The panel follow-rate drop is a pre-test of the pipeline. It is not evidence about people.
- [ ] A paid judge is still unwired on purpose. Do not point D5 or the analyst at DeepSeek until there is a separate decision to spend money.
- [ ] LFM2 transfer and a human study are out of this run.
