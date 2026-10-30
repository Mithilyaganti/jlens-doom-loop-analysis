# J-Space Analysis of Doom Loops

Mechanistic interpretability project: apply Anthropic's **Jacobian Lens (J-lens / J-space)** to characterize what happens inside small LMs when they enter **doom loops**, and test whether intervening on J-space directions can prevent them **without training**.

**Current Phase 1 target:** **`LiquidAI/LFM2-2.6B`** (completed run on 200 antidoom-mix prompts). Earlier Qwen3.5-4B work is archived under `artifacts/archive/`.

See `PROMPT.md`, `PROJECT_SPEC.md`, and **`LFM2_26B_FINDINGS_AND_FAQ.md`** for full research design and results FAQ.

## Hardware

- Windows laptop
- **RTX 5050, 8 GB VRAM**, 16 GB system RAM
- Models loaded in **4-bit NF4** (bitsandbytes)
- J-lens tensors loaded in fp16/bf16 (not quantized)

## Setup

```powershell
cd j-lens
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# CUDA torch (pick the index matching your driver; CUDA 12.8 example):
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
# Vendored jlens package path is added at runtime via jspace/loading.py
```

Cloned dependencies (already under `vendor/` if you used the bootstrap):

- `vendor/open-jlens-data` — J-lens fitters / `jlens` library
- `vendor/antidoom` — Liquid loop detection (also vendored into `jspace/detection.py`)

Pre-fitted lens (auto-downloaded by setup):

- `neuronpedia/jacobian-lens` → `lenses/qwen3.5-4b.pt`

## Run Phase 1

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD"
# Optional scale knobs (laptop defaults are conservative):
# $env:JLENS_MAX_PROMPTS = "120"
# $env:JLENS_EXP2_PROMPTS = "80"
# $env:JLENS_EXP3_PROMPTS = "40"
# $env:JLENS_MAX_NEW_TOKENS = "1024"
python scripts/run_phase1.py
```

Or step-by-step:

```powershell
python scripts/00_setup.py
python scripts/01_workspace_band.py
python scripts/02_trigger_tokens.py
python scripts/03_exp1_static_geometry.py
python scripts/04_exp2_dynamic.py
python scripts/05_exp3_causal.py
```

Resume from step N: `python scripts/run_phase1.py 2` (0-based index into the pipeline).

### Checkpoints

Long runs save JSON checkpoints under `results/checkpoints/` (Exp3 resumes automatically).
Create a git safety commit anytime:

```powershell
python scripts/save_checkpoint.py
```

Re-analyze Exp2 traces without re-generating:

```powershell
$env:JLENS_EXP2_ANALYZE_ONLY = "1"
python scripts/04_exp2_dynamic.py
```

Merge empirical trigger tokens from all logs:

```powershell
python scripts/merge_trigger_tables.py
```

## Package layout

```
jspace/
  loading.py       # model + lens load, sanity check
  detection.py     # Liquid find_inner_repetition + TokenState
  geometry.py      # J-lens directions, workspace band metrics
  intervention.py  # residual ablation hooks
  generation.py    # greedy gen + tiered caching
  analysis.py      # plots, stats, status notes
  token_sets.py    # Exp 1 token sets
scripts/           # Phase 1 pipeline
results/           # plots, CSVs, status notes, RESULTS_SUMMARY.md
artifacts/         # Drive exports, Colab fit blobs, archived runs (see artifacts/README.md)
docs/              # handoffs and session notes
vendor/            # open-jlens-data, antidoom
lenses/            # pre-fitted .pt
```

## LeanScope (paper 2 data)

Hidden-lean items for a follow-up study on Qwen3.5-4B. The package is `leanscope/`. Letters are canonical: `A` is `option_a` and `B` is `option_b` under both presentation orders. `order=BA` only prints option B first.

| Module | Role |
| --- | --- |
| `leanscope/schema.py` | `Scenario`, `LeanCondition`, `Record`, JSONL read/write |
| `leanscope/scenarios.py` | Three families, five paraphrases, both orders, DeepSeek V4.1 Flash drafter plus an offline fake |
| `leanscope/conditions.py` | No prompt, assigned lean at four strengths, structure-matched neutral, trait, conceal |
| `leanscope/splits.py` | Hash-locked train/dev/test by `scenario_id`, same write-once rule as `jspace/eval_sets.py` |
| `leanscope/labels.py` | Greedy recommendation, natural-lean majority and agreement, voiced/silent judge, hand-label check |
| `leanscope/presets.py` | `pilot`, `main`, and `full` subsets of the same locked splits and coin flips |
| `scripts/lean_00_build_items.py` | Seed JSONL to records JSONL, with `--preset` |
| `scripts/lean_01_generate.py` | Generation via `jspace.generation` and `jspace.loading`, with `--preset` |

The hand-written seed is `data/leanscope/seed_scenarios.jsonl` (100 scenarios: 34 analysis, 33 product, 33 advice). The frozen split file is `data/leanscope/split_lock.json`, a 60/15/25 rank cut by scenario (60 train, 15 dev, 25 test). Assigned leans are a seeded coin flip on `scenario_id` (seed 42). Expanding the seed gives 14,000 records (100 scenarios x 5 paraphrases x 2 orders x 14 conditions). That full file is unchanged. `--preset pilot` and `--preset main` only drop rows from it. A kept scenario keeps its locked split and its coin-flip lean.

CPU tests, no model download and no API call:

```bash
python -m pytest tests/test_leanscope_data.py -q
```

GPU generation was not run here. The hour column is an estimate, not a measurement: 50 tokens/s of decode on an L4, 300 new tokens (6 seconds per generation). Activations add about half of that again on the greedy pass, so greedy hours are 1.5 times the decode hours. Natural-lean samples are decode only. Both scripts print `records=` and `estimated_l4_hours=` for the preset they are about to run.

| Preset | What it keeps | Records | Natural-lean samples | Estimated L4-hours |
| --- | --- | --- | --- | --- |
| `pilot` | 40 scenarios (24 train / 6 dev / 10 test, families kept even), paraphrases p0–p1, both orders, `none` plus explicit and soft assigned and neutral prompts | 800 | 10 on each kept `none` row (1,600 generations) | 4.7 |
| `main` | all 100 scenarios, paraphrases p0–p2, both orders, `none`, `trait`, and all four assigned and neutral strengths | 6,000 | 10 on the locked dev split's `none` and `trait` rows (1,800 generations) | 18.0 |
| `full` | every paraphrase, order, and condition | 14,000 | off unless you pass `--natural-samples 20` (40,000 generations, about 67 hours) | 35.0 with activations; about 101.7 with that natural-sample pass |

```bash
python scripts/lean_00_build_items.py --preset pilot --out data/leanscope/records_pilot.jsonl
JLENS_MODEL=Qwen/Qwen3.5-4B JLENS_HF_QUANTIZE=0 \
  python scripts/lean_01_generate.py --preset pilot \
    --records data/leanscope/records_pilot.jsonl \
    --out results/leanscope/records_pilot.jsonl \
    --activations-dir results/leanscope/activations_pilot \
    --max-new-tokens 300
```

Pass the same `--preset` to generate if the records file is the full 14,000-row build. The filter is the same subset either way.

## Caching (memory safety)

Experiment 2 uses a **tiered cache** (PROJECT_SPEC §2.1):

| Tier | What | When |
|------|------|------|
| 1 | Top-10 J-lens readouts per layer × token | All prompts |
| 2 | Full residuals at 3–5 workspace-band layers | All prompts, streamed to disk |
| 3 | Full residuals all layers | ~5 hand-picked traces only |

Never cache full residuals for all prompts — that OOMs 8 GB + 16 GB RAM.

## Deliverables

- `results/trigger_tokens_qwen3.5-4b.csv` — empirical Qwen trigger table (publishable)
- `results/workspace_band_qwen3.5-4b.json`
- `results/exp1/`, `exp2/`, `exp3/` plots + CSVs
- `results/status/` notes after each stage
- `results/RESULTS_SUMMARY.md`

## Phase 2 / 3 (not run by default)

After Phase 1 validates: Gemma-4 Edge models + Qwen base (needs cloud lens fit); Liquid LFM2.5 hybrid fitter is stretch.

## Citations (required)

Liquid AI Antidoom (Jul 2026); Anthropic J-lens `arXiv:2607.15495`; Lazaridis et al. `arXiv:2606.13705`; Silent Alarm / JADR `arXiv:2607.12792`; Elie Bakouch open-jlens-data; related loop papers in PROJECT_SPEC §14.

## License notes

- Anthropic `jacobian-lens` (vendored): Apache-2.0
- Liquid `antidoom` detection logic: see their repo license
- This project code: research use; cite sources above
