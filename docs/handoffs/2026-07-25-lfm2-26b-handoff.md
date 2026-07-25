# Handoff — Switch Phase 1 target to LiquidAI/LFM2-2.6B

**Date:** 2026-07-25  
**From:** side-chat agent (planning + scaffolding)  
**To:** main agent (Composer 2.5) — execute + monitor end-to-end  

---

## Decisions (locked)

| Item | Choice |
|---|---|
| Model | **`LiquidAI/LFM2-2.6B`** (public HF) |
| Not using | Blog’s private early **LFM2.5-2.6B** (not released) |
| Antidoom-fixed? | **No** — public download is not Antidoom-trained |
| Thinking | **Leave default** (dynamic hybrid reasoning). Do **not** force `enable_thinking=False` |
| Base variant? | **None published** for 2.6B — this post-trained ckpt is the one |
| max_new_tokens | **4000** |
| temperature | **0.01** |
| Dataset | Same 200 stratified antidoom-mix reasoning sources, seed=42 (`results/prompt_sample_ids.json`) |
| Backend | Colab: **vLLM** (`fp8`, fallback `bfloat16`). Windows laptop: **HF NF4** (`JLENS_BACKEND=hf`) |
| Qwen runs | **Stopped**. Archive under `results/archive_2048_*` and partial qwen4000 |

## Already done (scaffolding)

- `jspace/model_config.py` — active model registry (default LFM2-2.6B)
- `jspace/vllm_backend.py` — optional vLLM generator with fp8→bf16 fallback
- `jspace/loading.py` — `require_lens=False`; generation works without J-lens
- `jspace/generation.py` — vLLM hook; skip tier cache if no lens
- `scripts/02_baseline_pass.py` — model-agnostic artifacts; gen-only if no band/lens
- `scripts/06_fit_lfm_lens.py` — status stub (lens not fitted yet)
- `scripts/run_lfm2_26b.py` — generation-first runner
- `notebooks/colab_lfm2_26b_baseline.ipynb` — Colab GPU notebook

## Blockers / hard parts

1. **J-lens for hybrid LFM** — no neuronpedia pre-fit. Exp1–3 need fitting attention layers only (skip LIV conv). Do **generation+loop-rate first**; fit lens on Colab second.
2. **vLLM on Windows** — prefer Colab kernel. Local use HF 4-bit.
3. **VRAM** — 2.6B bf16 + 4000 tokens is tight on 8GB; local NF4 OK; Colab T4/L4 use vLLM fp8/bf16.
4. Scripts `01/03/04/05` may still hardcode `qwen3.5-4b` paths — **update to `artifact_paths()`** before Exp1–3.

## Execution order for main agent

1. Confirm GPU free (no leftover Qwen python).
2. Zip/upload repo for Colab OR run local:
   ```powershell
   $env:JLENS_MODEL="LiquidAI/LFM2-2.6B"
   $env:JLENS_BACKEND="hf"
   $env:JLENS_MAX_NEW_TOKENS="4000"
   $env:PYTHONPATH="C:\Users\mithi\Desktop\mithil\projects\j-lens"
   .\.venv\Scripts\python.exe scripts\run_lfm2_26b.py
   ```
3. Prefer user Colab notebook `notebooks/colab_lfm2_26b_baseline.ipynb` with `JLENS_BACKEND=vllm`.
4. After 200/200: write `results/RUN_REPORT_lfm2-2.6b_antidoom_mix_200.md` (honest loop rate).
5. Then attempt J-lens fit → workspace band → Exp1–3 (only if lens succeeds).
6. Never invent loop rates or trigger tables.

## Success criteria

- [ ] 200 prompts completed on `LiquidAI/LFM2-2.6B`, max_new=4000
- [ ] `results/trigger_tokens_lfm2-2.6b.csv` + `baseline_pass_summary.json` real
- [ ] Report file with loop rate (even if low/null)
- [ ] Exp1–3 only after real lens file exists; else document blocker honestly

---

## Copy-paste prompt for main agent (Composer 2.5)

```text
Execute the LFM2-2.6B Phase-1 protocol for this repo (j-lens).

CONTEXT: Read docs/handoffs/2026-07-25-lfm2-26b-handoff.md (or results/HANDOFF_LFM2_26B.md). Side-chat already scaffolded model_config, vllm_backend, baseline_pass for LiquidAI/LFM2-2.6B. Qwen runs were stopped.

GOAL: End-to-end on LiquidAI/LFM2-2.6B only (NOT Qwen). This is the public HF model — NOT Antidoom-trained, NOT the private early LFM2.5-2.6B from the blog. Leave dynamic thinking ON (do not disable). Dataset: same 200 stratified antidoom-mix prompts (seed=42) in results/prompt_sample_ids.json. max_new_tokens=4000, temperature=0.01. Prefer vLLM+fp8 (fallback bf16) on Colab; on Windows local use JLENS_BACKEND=hf with 4-bit NF4.

DO:
1. Kill any leftover Qwen GPU python processes if present.
2. Run scripts/run_lfm2_26b.py (or Colab notebook notebooks/colab_lfm2_26b_baseline.ipynb). Monitor until 200/200 baseline completes. Resume from results/checkpoints/baseline_pass_lfm2-2.6b.json if interrupted.
3. Write results/RUN_REPORT_lfm2-2.6b_antidoom_mix_200.md with honest loop rate, backend, dtype, and note that this is not the blog’s private 2.6B early ckpt.
4. Update scripts 01/03/04/05 to use jspace.model_config.artifact_paths() instead of hard-coded qwen3.5-4b filenames.
5. Attempt J-lens fit for hybrid LFM (attention layers only). If impossible in-session, leave clear status in results/jlens_fit_status_lfm2-2.6b.json and skip Exp1–3 rather than faking.
6. If lens exists: run workspace band → Exp1 → Exp2 analyze → Exp3 on looping prompts only.
7. Never invent results. Single GPU worker only. Checkpoint often.

User may attach Colab GPU via the Colab extension; prefer that for vLLM. Local RTX 5050 8GB is backup with HF NF4.
```
