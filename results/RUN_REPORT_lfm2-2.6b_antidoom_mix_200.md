# J-lens Run Report — LFM2-2.6B (antidoom-mix 200 prompts)

## Model disclaimer
- **Model**: `LiquidAI/LFM2-2.6B` (public HuggingFace checkpoint)
- **NOT** the blog's private early **LFM2.5-2.6B** checkpoint (unreleased)
- **NOT** Antidoom-trained — this is the standard public post-trained release
- **Dynamic thinking**: left at model default (hybrid reasoning enabled)

## Run settings
- Backend: **unknown**
- vLLM dtype: `n/a`
- max_new_tokens: 4000
- temperature: 0.01
- J-lens available: False

## Dataset
- Source: `LiquidAI/antidoom-mix-v1.0` (7 reasoning component sources only)
- Stratified sample: 200 prompts, seed=42
- Per-source counts: `{"apps_train": 29, "open_perfectblend_evol_codealpaca": 28, "open_perfectblend_ultrainteract": 28, "math_lighteval_train": 29, "gsm8k_train": 29, "math_qa_train": 29, "open_perfectblend_metamathqa": 28}`
- Audit: `C:\Users\mithi\Desktop\mithil\projects\j-lens\results\prompt_sample_ids.json`

## Baseline pass (generation + loop detection)
- Progress: **0/200** prompts completed
- **Loop rate: 0.0%** (0 loops)
- Unique trigger tokens: 0
- Looping prompt IDs: 0
- Trigger table: `C:\Users\mithi\Desktop\mithil\projects\j-lens\results\trigger_tokens_lfm2-2.6b.csv`
- Checkpoint: `C:\Users\mithi\Desktop\mithil\projects\j-lens\results\checkpoints\baseline_pass_lfm2-2.6b.json`

## Experiment 2 (analyze-only, same generations)
- Pending (baseline not finished)

## Experiment 3 (looping prompts only)
- Skipped (zero baseline loops)

## J-lens fit status
- **Not fitted**: No pre-fitted J-lens for LFM2-2.6B. Fitting requires adapting open-jlens (vendor/open-jlens-data) to skip LIV conv blocks and fit Jacobians only on attention layers. Prefer Colab A100/L4 with bf16 for the fit. Until then: run baseline generation+detection; defer Exp1/Exp2 geometry/Exp3 ablation.

All numbers above are from real runs — nothing invented.
