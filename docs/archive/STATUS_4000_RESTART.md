# Status — stopped 2048 run, preparing 4000-token restart

## Stopped
- Killed baseline Python worker (PID 17524 and shims) at ~12:08 local time.
- Progress at stop: **124 / 200** prompts, **2 loops** (~1.6%), `max_new_tokens=2048`.

## Saved (do not delete)
- Archive: `results/archive_2048_20260725_120857/`
  - `baseline_pass.json`, `trigger_gen_log.jsonl`, Exp2 looping/nonlooping traces
  - `ARCHIVE_META.json`
- Same sample kept for restart: `results/prompt_sample_ids.json` (seed=42, 7 reasoning sources)

## Code changes for restart
- Default `JLENS_MAX_NEW_TOKENS` / generation defaults → **4000** (Liquid `configs/default.yaml`).
- Cleared live 2048 checkpoint + exp2 dirs so the new run regenerates with 4000 tokens.
- Colab notebook: `notebooks/colab_antidoom_baseline.ipynb`

## Next
- Local: `python scripts/run_antidoom_200.py` with `JLENS_MAX_NEW_TOKENS=4000`
- Or Colab: open the notebook, GPU runtime, upload/clone repo, run cells
