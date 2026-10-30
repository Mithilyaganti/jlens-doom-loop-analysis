#!/usr/bin/env bash
# LeanScope GPU order for Qwen3.5-4B. This file is the command list.
# It is not part of the CPU tests. Do not run it on a machine without a GPU.
# Steps, expected files, and runtime estimates: docs/leanscope_gpu_runbook.md
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${LEANSCOPE_LAYERS:-}" ]]; then
  echo "Set LEANSCOPE_LAYERS to the workspace-band layer list before this script." >&2
  echo "Example: LEANSCOPE_LAYERS=18,20,22  (replace with the band file's layers)." >&2
  exit 2
fi

export JLENS_MODEL="${JLENS_MODEL:-Qwen/Qwen3.5-4B}"
export JLENS_HF_QUANTIZE="${JLENS_HF_QUANTIZE:-0}"

python scripts/01_workspace_band.py

python scripts/lean_00_build_items.py \
  --update-lock \
  --out data/leanscope/records.jsonl

python scripts/lean_01_generate.py \
  --records data/leanscope/records.jsonl \
  --out results/leanscope/records_generated.jsonl \
  --max-new-tokens 300

python scripts/lean_02_record.py \
  --records results/leanscope/records_generated.jsonl \
  --out results/leanscope/records_positions.jsonl \
  --activations-dir results/leanscope/activations \
  --max-new-tokens 64

python scripts/lean_03_detect.py \
  --records results/leanscope/records_positions.jsonl \
  --out results/leanscope/detection.csv \
  --scores-out results/leanscope/d1_scores.json \
  --samples 0 \
  --n-boot 1000

python scripts/lean_04_causal.py \
  --records results/leanscope/records_positions.jsonl \
  --direction jlens \
  --split test \
  --layers "${LEANSCOPE_LAYERS}" \
  --positions P0,P1 \
  --alpha 1.0 \
  --max-new-tokens 256 \
  --out results/leanscope/causal

python scripts/lean_05_panel_sim.py \
  --records results/leanscope/records_positions.jsonl \
  --scores results/leanscope/d1_scores.json \
  --prior-opposite \
  --analyst fake \
  --max-new-tokens 256 \
  --out results/leanscope/panel
