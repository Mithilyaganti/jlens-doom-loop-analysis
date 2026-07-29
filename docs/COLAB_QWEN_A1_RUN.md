# Colab A1 runbook — Qwen3.5-4B vLLM baseline (T4)

## What you are running

**A1 only:** 200 prompts (seed=42), `max_new_tokens=4000`, `temperature=0.01`, official `Qwen/Qwen3.5-4B`, **vLLM bfloat16** on T4.

Not Exp1/2/3. Not nerfed.

## Notebook

Upload / open from the repo:

`notebooks/colab_qwen35_4b_a1_baseline.ipynb`

## Steps

1. Colab → Runtime → **GPU (T4)**.
2. Open the notebook (File → Upload, or clone repo and open).
3. **Cell 1** — mount Drive (`MyDrive/jlens_qwen_a1_baseline/`).
4. **Cell 2** — clone branch `cursor/lfm2-exp1-jlens-fit` (or the branch you pushed).
5. **Cell 3** — install vLLM + deps (several minutes).
6. **Cell 4** — env (bf16, 4000 tokens, Drive sync every **10** prompts).
7. **Cell 5** — restore from Drive if resuming.
8. **Cell 6** — **SMOKE** (1 prompt). Confirm it finishes and Drive `LAST_SYNC.txt` updates.
9. **Cell 7** — **FULL 200**. Leave it running (overnight OK).
10. **Cell 8** — status + final Drive sync.

## Resume after disconnect

New runtime → Cells **1–5** → **Cell 7** again. Completed prompts are skipped (local + Drive metas/checkpoint).

## Artifacts (canonical)

| Path | Meaning |
|------|---------|
| `results/checkpoints/baseline_pass_qwen3.5-4b.json` | Resume checkpoint |
| `results/trigger_gen_log_qwen3.5-4b.jsonl` | Per-prompt log |
| `results/baseline_pass_summary_qwen3.5-4b.json` | Summary when done/partial |
| `results/trigger_tokens_qwen3.5-4b.csv` | Trigger table |
| `results/exp2_qwen3.5-4b/looping|nonlooping/p*/meta.json` | Per-prompt metas |
| Drive mirror | Same under `MyDrive/jlens_qwen_a1_baseline/results/` |

## If smoke fails

- OOM / dtype: in Cell 4 set `JLENS_VLLM_DTYPE=float16` and re-run Cell 6.
- vLLM install fail: Runtime → Restart session, re-run Cell 3, then continue.
- Do **not** lower `JLENS_MAX_NEW_TOKENS` below 4000 for the real run.

## Time honesty

T4 + 4000 tokens: **4–5 hours may be partial**. Local checkpoint every prompt; Drive sync every **10** prompts (+ start/final). Resume next session until `n_completed=200`.
