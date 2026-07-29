# J-lens Run Report — LFM2-2.6B (antidoom-mix 200 prompts)

## Model disclaimer
- **Model**: `LiquidAI/LFM2-2.6B` (public HuggingFace checkpoint)
- **NOT** the blog's private early **LFM2.5-2.6B** checkpoint (unreleased)
- **NOT** Antidoom-trained — this is the standard public post-trained release
- **Dynamic thinking**: left at model default (hybrid reasoning enabled)

## Run settings
- Backend: **hf** (4-bit NF4 for generation; J-lens fitted separately in FP16)
- max_new_tokens: 4000
- temperature: 0.01
- J-lens available: **True** (`lenses/lfm2-2.6b.pt`, 270-prompt FP16 attn-only fit)

## Dataset
- Source: `LiquidAI/antidoom-mix-v1.0` (7 reasoning component sources only)
- Stratified sample: 200 prompts, seed=42
- Sample file: `prompt_sample_ids.json`

## Baseline pass (generation + loop detection) — DONE
- Progress: **200/200** prompts completed
- **Loop rate: 5.5%** (11 / 200)
- Unique trigger tokens: 11 (one each)
- Looping prompt IDs: `86352, 96637, 120702, 123613, 133187, 179962, 199303, 290449, 340846, 346805, 422988`

## J-lens fit — DONE
- Colab FP16, WikiText **270** prompts, attn layers `[2,5,9,13,17,21,24,27]`
- Lens: `lenses/lfm2-2.6b.pt`

## Workspace band — DONE
- Band: **L21–L21** (attn-only hybrid lens; L24/L27 treated as motor)

## Exp1 static geometry — DONE
- Workspace means (L21): pairwise sim trigger **0.0905** vs controls **0.0878**; alignment trigger **0.2477** vs controls **0.1942**
- `success_criterion_met`: **true** (mainly alignment)

## Exp2 dynamic monitoring — DONE
- Hooked re-gen: **200/200** Tier-1/2 at L21 (`max_new_tokens=4000`)
- Regen loop rate: **2.5%** (5/200)
- Alignment −1: loop **0.172** vs non **0.130**; top-1=trigger at −1: **0.60** vs **0.041**
- `success_criterion_met`: **true** (preliminary; n_loop=5)

## Exp3 causal ablation — DONE
- 11 baseline looping prompts × 6 conditions = **66/66** (`max_new_tokens=4000`)
- Loop rates: baseline **0.182**, ablate_trigger **0.182**, random **0.091**, control **0.273**, sensory **0.182**, motor **0.091**
- McNemar p=**0.62**, Cohen's h=**0.0** — primary trigger ablation did **not** reduce loops vs baseline
- Result type: **mixed / null primary** (underpowered; many known loopers did not re-loop)
- Eval accuracy stayed 1.0 on tiny arithmetic set

## Stretch / not done
- Exp2 Tier-3 full-residual HTML visualization

All numbers above are from completed LFM artifacts — nothing invented.
