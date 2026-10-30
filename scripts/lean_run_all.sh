#!/usr/bin/env bash
# LeanScope GPU order for Qwen3.5-4B. This file is the command list.
# It is not part of the CPU tests. Do not run it on a machine without a GPU.
# Steps, expected files, and runtime estimates: docs/leanscope_gpu_runbook.md
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${LEANSCOPE_LAYERS:-}" ]]; then
  echo "Set LEANSCOPE_LAYERS to the workspace-band layer list before this script." >&2
  echo "Read results/workspace_band_qwen3.5-4b.json after scripts/01_workspace_band.py." >&2
  echo "Example shape only: LEANSCOPE_LAYERS=18,20,22" >&2
  exit 2
fi

# pilot ~6 L4-hours, main ~27, full ~64. See the runbook. Default is the main study.
PRESET="${LEANSCOPE_PRESET:-main}"
case "$PRESET" in
  pilot|main|full) ;;
  *)
    echo "LEANSCOPE_PRESET must be pilot, main, or full (got ${PRESET})." >&2
    exit 2
    ;;
esac

export JLENS_MODEL="${JLENS_MODEL:-Qwen/Qwen3.5-4B}"
export JLENS_HF_QUANTIZE="${JLENS_HF_QUANTIZE:-0}"

GEN="results/leanscope/records_${PRESET}.jsonl"
POS="results/leanscope/records_${PRESET}_positions.jsonl"
ACT="results/leanscope/activations_${PRESET}"
SCORES="results/leanscope/d1_scores_${PRESET}.json"

python scripts/01_workspace_band.py

# Full 14,000-row bank. --preset on the next step drops rows. The lock is not rewritten.
python scripts/lean_00_build_items.py \
  --update-lock \
  --out data/leanscope/records.jsonl

# Text and labels only. P0/P1 residuals are the next step, so do not pass --activations-dir.
# The printed estimated_l4_hours assumes activations on this pass; this script does not record them here.
python scripts/lean_01_generate.py \
  --records data/leanscope/records.jsonl \
  --preset "$PRESET" \
  --out "$GEN" \
  --max-new-tokens 300

python scripts/lean_02_record.py \
  --records "$GEN" \
  --out "$POS" \
  --activations-dir "$ACT" \
  --max-new-tokens 64

python scripts/lean_03_detect.py \
  --records "$POS" \
  --out "results/leanscope/detection_${PRESET}.csv" \
  --scores-out "$SCORES" \
  --samples 0 \
  --n-boot 1000

python scripts/lean_04_causal.py \
  --records "$POS" \
  --direction jlens \
  --split test \
  --layers "${LEANSCOPE_LAYERS}" \
  --positions P0,P1 \
  --alpha 1.0 \
  --max-new-tokens 256 \
  --out "results/leanscope/causal_${PRESET}"

python scripts/lean_05_panel_sim.py \
  --records "$POS" \
  --scores "$SCORES" \
  --prior-opposite \
  --analyst fake \
  --max-new-tokens 256 \
  --out "results/leanscope/panel_${PRESET}"
