# Colab A1 runbook — Qwen3.5-4B vLLM baseline (T4)

## What you are running

**A1 only:** 200 prompts (seed=42), `max_new_tokens=4000`, greedy (`temperature=0`), official `Qwen/Qwen3.5-4B`, **vLLM bfloat16** on T4.

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

T4 + 4000 tokens can still take hours. Two speedups keep the cap at 4000:

- **vLLM** submits chunks of 4 (`JLENS_BASELINE_BATCH_SIZE`, `JLENS_VLLM_MAX_NUM_SEQS`). A disconnect loses at most the in-flight chunk. Set the batch size to `all` only if you want the old single call.
- **HF fallback** (the usual T4 path when vLLM cannot import) stops a trace once the character detector has already confirmed a loop (`JLENS_BASELINE_EARLY_STOP=1`, checked every 32 new tokens). Non-looping traces still run to EOS or 4000, so a loop label cannot flip off. SDPA is requested via `JLENS_HF_ATTN=sdpa`.

`JLENS_ANTIDOOM_SIDECAR` stays **0** in this notebook. Turning it on forces the HF backend and blocks the token that would extend a repeated n-gram. That run is an intervention, not the unsteered baseline, and the summary says so.

Local checkpoint every prompt; Drive sync every **10** prompts (+ start/final). Resume next session until `n_completed=200`.
