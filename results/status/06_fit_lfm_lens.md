# LFM J-lens fit — DONE

- **Model**: `LiquidAI/LFM2-2.6B` (hybrid; Jacobians on attn layers only)
- **Dtype**: FP16 (Colab Tesla T4)
- **Prompts**: 270 WikiText snippets (`max_seq_len=128`, `dim_batch=16`)
- **Source layers**: `[2, 5, 9, 13, 17, 21, 24, 27]`
- **Artifact**: `lenses/lfm2-2.6b.pt` (~64 MB; 8× 2048×2048 FP16 Jacobians)
- **Fit time**: ~116.5 min
- **Status JSON**: `jlens_fit_status_lfm2-2.6b.json` (`fitted: true`)

Downstream completed with this lens: workspace band (L21) and Exp1 static geometry.
