# J-Space Analysis of Doom Loops

Mechanistic interpretability project: apply Anthropic's **Jacobian Lens (J-lens / J-space)** to characterize what happens inside small LMs when they enter **doom loops**, and test whether intervening on J-space directions can prevent them **without training**.

Phase 1 focuses on **`Qwen/Qwen3.5-4B`** only (Liquid reported 22.9% doom-loop rate under greedy sampling on this model).

See `PROMPT.md` and `PROJECT_SPEC.md` for full research design.

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
vendor/            # open-jlens-data, antidoom
lenses/            # pre-fitted .pt
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

## Citations (required)

Liquid AI Antidoom (Jul 2026); Anthropic J-lens `arXiv:2607.15495`; Lazaridis et al. `arXiv:2606.13705`; Silent Alarm / JADR `arXiv:2607.12792`; Elie Bakouch open-jlens-data; related loop papers in PROJECT_SPEC §14.

## License notes

- Anthropic `jacobian-lens` (vendored): Apache-2.0
- Liquid `antidoom` detection logic: see their repo license
- This project code: research use; cite sources above
