# Artifacts

Local-only outputs that are too large or redundant for the main `results/` tree. **Binary lens files (`.pt`) are gitignored.**

## Layout

| Path | Contents |
|------|----------|
| `exports/` | Full Google Drive snapshot downloads (e.g. `jlens_results_lfm2_20260725_123653/`) — mirror of results + lens at time of export |
| `jlens_fit/` | Colab J-lens fitting runs (e.g. `lfm2_fp16/` — `lfm2-2.6b.pt`, fit checkpoints, status JSON) |
| `archive/` | Superseded runs (Qwen 2048-token pass, pre-LFM Exp2/Exp3 partials) moved from `results/archive_*` |

## Canonical results

Use **`results/`** for the current LFM2-2.6B pipeline outputs:

- `results/RUN_REPORT_lfm2-2.6b_antidoom_mix_200.md`
- `results/exp1/`, `results/exp2/`, `results/exp3/`
- `results/checkpoints/`

## Drive export

If you downloaded a folder from Google Drive, place it under `artifacts/exports/<timestamp_or_name>/`. The export may duplicate `results/`; keep `results/` as source of truth for the repo and treat exports as backups.
