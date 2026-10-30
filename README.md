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
| `scripts/lean_00_build_items.py` | Seed JSONL to records JSONL |
| `scripts/lean_01_generate.py` | Generation via `jspace.generation` and `jspace.loading` |

The hand-written seed is `data/leanscope/seed_scenarios.jsonl`. The frozen split file is `data/leanscope/split_lock.json`. Assigned leans are a seeded coin flip on `scenario_id` (seed 42). Counts and the GPU command order are in `docs/leanscope_gpu_runbook.md`. The shell list is `scripts/lean_run_all.sh` (do not run it on CPU).

CPU tests, no model download and no API call. `tests/test_leanscope_e2e.py` runs data, recording, D1, the J-lens steer, and the score-triggered panel:

```bash
python -m pytest tests -q
```

The GPU order is `scripts/lean_run_all.sh`. Runtime estimates are in `docs/leanscope_gpu_runbook.md`. They are not measurements.

## LeanScope detection

`leanscope/positions.py` records residuals at P0 (last prompt token), P1 (token before the first option word), and optional P2. `leanscope/detectors.py` is D1–D7. D1 is the J-lens concept readout (option A minus option B). `leanscope/metrics.py` reports lean-direction AUROC with scenario-clustered bootstrap intervals from `jspace.stats`.

```bash
python scripts/lean_02_record.py --fake --out results/leanscope/records.jsonl --activations-dir results/leanscope/activations
python scripts/lean_03_detect.py --fake --records results/leanscope/records.jsonl --out results/leanscope/detection.csv --scores-out results/leanscope/d1_scores.json --samples 3 --n-boot 40
```

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

## LeanScope (paper 2) — causal test and alternatives panel

`leanscope/` is the causal and intervention side of the hidden-lean study. It reads `Record` rows from `leanscope/schema.py` (scenario, lean condition, recommendation, split, optional activation path).

- `causal.py` steers and ablates an option direction at P0 (last prompt token) and P1 (the token just before the first option word). `scripts/lean_04_causal.py --direction jlens` plugs in D1: the J-space direction of option A minus option B, one vector per item. `--direction diffmeans` keeps the train-split difference of means. The control is a norm-matched random direction. The positive control regenerates with the lean prompt removed. Flip rates use exact McNemar, Wilson intervals, and Cohen's h from `jspace.stats`.
- `panel.py` builds the alternatives panel. C2 adds the other option's case only when the D1 score from `lean_03_detect.py --scores-out` crosses a threshold chosen on the dev split. C0 is the recommendation alone, C1 always adds the other case, and C3 is a disclosure line.
- `simulate.py` is the Tier A pre-test. Analyst agents get a prior and see C0–C3. The agent is an `LLMClient`. CPU runs use `OfflineFakeAnalyst`. `DeepSeekFlashClient` refuses instead of placing a request. Every simulation report is labelled `pre-test, not evidence about people`.

CPU tests, with no GPU and no model download:

```bash
python -m pytest tests/test_leanscope_causal.py tests/test_leanscope_panel_sim.py -q
```

Smoke entry points on the same test double:

```bash
python scripts/lean_04_causal.py --smoke-fake --out /tmp/leanscope-causal
python scripts/lean_05_panel_sim.py --smoke-fake --out /tmp/leanscope-panel
```

A Qwen3.5-4B run of those scripts is documented in their docstrings. It is not part of the CPU tests.

## Citations (required)

Liquid AI Antidoom (Jul 2026); Anthropic J-lens `arXiv:2607.15495`; Lazaridis et al. `arXiv:2606.13705`; Silent Alarm / JADR `arXiv:2607.12792`; Elie Bakouch open-jlens-data; related loop papers in PROJECT_SPEC §14.

## License notes

- Anthropic `jacobian-lens` (vendored): Apache-2.0
- Liquid `antidoom` detection logic: see their repo license
- This project code: research use; cite sources above
