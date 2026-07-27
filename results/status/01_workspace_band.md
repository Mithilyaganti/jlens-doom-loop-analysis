# Workspace Band — LFM2-2.6B

Empirically identified on **attention-only** J-lens layers (hybrid LFM; conv skipped).

- **Model**: `LiquidAI/LFM2-2.6B` (NF4 load; lens fp16)
- **Lens**: `lenses/lfm2-2.6b.pt` (270 WikiText prompts, attn layers `[2,5,9,13,17,21,24,27]`)
- **Workspace band**: **L21–L21** (single fitted attn layer scored as workspace; L24/L27 treated as motor)
- **Mid / key layer**: 21
- **Sensory (fitted)**: 2, 5, 9, 13, 17
- **Motor (fitted)**: 24, 27
- NTP prompts: 40 wikitext-2 snippets

Artifacts: `results/workspace_band_lfm2-2.6b.json`, `results/workspace_band_metrics_lfm2-2.6b.json`, `results/*_by_layer.png`.

**Caveat:** With only 8 fitted layers, the contiguous “band” collapses to one layer. Use plots before over-interpreting width; Exp2/3 should still target L21 (± neighbors if needed).
