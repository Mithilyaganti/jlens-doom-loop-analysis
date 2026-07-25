# Colab run — LFM2-2.6B baseline (200 prompts)

**This agent cannot attach to Google Colab from Cursor.** You run the notebook; it uses Colab GPU + vLLM.

## Steps (browser Colab or Cursor + Colab kernel)

**Important:** The Colab GPU runs on Google's VM — it cannot read files directly from your laptop. Cell 1 must get the code via **GitHub clone** or **Drive zip**.

### Recommended: GitHub (faster setup than Drive)

1. Push this repo to GitHub (private is fine).
2. In cell 1, `REPO_URL` is set to `https://github.com/Mithilyaganti/jlens-doom-loop-analysis.git`.
3. Connect Colab GPU → Run All.

### Alternative: Google Drive zip

1. Zip the whole `j-lens` folder (include `results/prompt_sample_ids.json`).
2. Upload `j-lens.zip` to Google Drive → `My Drive/j-lens.zip`.
3. In cell 1, set `DRIVE_ZIP = "/content/drive/MyDrive/j-lens.zip"`.
4. Run All (cell 1 mounts Drive and unzips).

GitHub is usually faster: no manual zip/upload, and `git clone` on Colab takes ~seconds.

## What runs

| Step | Script | Notes |
|------|--------|-------|
| Lens status | `06_fit_lfm_lens.py` | Writes `jlens_fit_status_lfm2-2.6b.json` (no pre-fit) |
| Baseline | `02_baseline_pass.py` | 200 prompts, max_new=4000, vLLM fp8→bf16 |
| Report | `write_lfm_report.py` | Honest loop rate + disclaimers |

Exp1–3 run **only** if `lenses/lfm2-2.6b.pt` exists (not expected on first pass).

## Env (set in notebook cell 3)

```
JLENS_MODEL=LiquidAI/LFM2-2.6B
JLENS_BACKEND=vllm
JLENS_VLLM_DTYPE=fp8
JLENS_MAX_NEW_TOKENS=4000
JLENS_TEMPERATURE=0.01
```

## Sync results back to laptop

Copy from Colab Drive (`/content/drive/MyDrive/j-lens-results-lfm2-2.6b`) or download:

- `results/baseline_pass_summary.json`
- `results/trigger_tokens_lfm2-2.6b.csv`
- `results/checkpoints/baseline_pass_lfm2-2.6b.json`
- `results/RUN_REPORT_lfm2-2.6b_antidoom_mix_200.md`
- `results/trigger_gen_log.jsonl`

## Local fallback (RTX 5050, slower)

Only if Colab is unavailable:

```powershell
$env:JLENS_MODEL="LiquidAI/LFM2-2.6B"
$env:JLENS_BACKEND="hf"
$env:JLENS_MAX_NEW_TOKENS="4000"
$env:PYTHONPATH="C:\Users\mithi\Desktop\mithil\projects\j-lens"
.\.venv\Scripts\python.exe scripts\run_lfm2_26b.py
```

## Model disclaimer

- Public `LiquidAI/LFM2-2.6B` — **not** Antidoom-trained, **not** blog private LFM2.5-2.6B.
- Dynamic thinking left **on** (default).
