# What J-Space Sees in Doom Loops (and What It Doesn’t Prove)

**A honest write-up of Phase 1 on public `LiquidAI/LFM2-2.6B`**

*Draft for Option C — observational + negative causal finding. No further experiments required for this draft; numbers are from completed runs only.*

---

## The pitch in one paragraph

Doom loops — models that start repeating and never stop — look like a failure of *control*, not of knowledge. Jacobian lenses (J-lenses) give a verbalizable map of residual-stream directions: at each layer, you can ask which token directions the model’s state is aligned with. We asked whether the tokens that *start* loops occupy a special place in that map, whether the map lights up just before a loop begins, and whether *removing* those directions during generation prevents loops. On public LFM2-2.6B, we found a modest geometric signal and a suggestive pre-onset signature — and a causal ablation that did **not** reduce looping. That combination is still a story: J-space can *describe* loop onset better than it can *stop* it, at least under this intervention.

---

## What we are *not* claiming

- We did **not** test Liquid’s private early LFM2.5-2.6B checkpoint from the Antidoom blog.
- We did **not** use an Antidoom-trained model. Public `LiquidAI/LFM2-2.6B` is the standard release.
- We are **not** claiming that trigger tokens are proven *causes* of loops. Our ablation failed to show that.
- Loop rates here are for **this** model, under **NF4** generation on a laptop / HF path, at temperature ≈ 0.01 and 4000 max new tokens — not a replica of Liquid’s vLLM pipeline numbers.

If you came for “we cured doom loops with a J-lens ablation,” this is not that post. If you came for “what does a careful Phase-1 pipeline actually find,” keep reading.

---

## Setup (short)

| Item | Choice |
|------|--------|
| Model | `LiquidAI/LFM2-2.6B` (hybrid attention + conv) |
| Prompts | 200 stratified from `LiquidAI/antidoom-mix-v1.0` (7 reasoning sources), seed=42 |
| Generation | HF, 4-bit NF4, temp 0.01, max 4000 new tokens |
| Loop detector | Liquid-style inner repetition |
| J-lens | Fitted ourselves (no public pre-fit): FP16, 270 WikiText prompts, **attention layers only** `[2,5,9,13,17,21,24,27]` (conv skipped) |
| Workspace band | Empirically scored → **L21 only** among fitted layers |

Hybrid architecture forced an attention-only lens. That already narrows what “workspace band” can mean: with eight discrete fitted layers, the usual multi-layer band collapsed to a single layer.

---

## Finding 0 — Loops are rare, and labels are fragile

Baseline on 200 prompts: **11 loops (5.5%)**, eleven distinct trigger tokens (one each) — words and fragments like ` given`, `But`, ` try`, ` difference`, not a single “doom token.”

When we re-generated the same 200 prompts with J-lens hooks (Exp 2): **5 loops (2.5%)**.  
When we re-ran the 11 known baseline loopers for causal tests (Exp 3): only **2/11** looped again under the no-intervention condition.

Same model family, same temperature regime, same prompt IDs — different loop counts. We treat that as **quantization / rerun nondeterminism under NF4**, not as “the science changed.” It is also a methodological warning: any causal claim that needs *paired* re-loops on known bad prompts is underpowered until generation is stable.

**Takeaway for practitioners:** if your loop rate is single-digit and you are on 4-bit, measure reproducibility before you interpret interventions.

---

## Finding 1 — Trigger directions sit a bit more “inside” the workspace (Exp 1)

**Question:** Do doom-loop *trigger* tokens look geometrically special in J-space compared to controls?

**Controls:** frequency-matched content words, discourse markers (`But`, `So`, …), and random vocab — not “non-looping tokens.”

**At L21:**

| Metric | Triggers | Controls (mean) |
|--------|----------|-----------------|
| Mean pairwise cosine similarity | 0.0905 | 0.0878 |
| Workspace alignment (variance in top PCA of Jacobian) | **0.2477** | **0.1942** |

Clustering is barely there. **Alignment** is clearer: trigger directions put more of their energy in the subspace the lens treats as workspace-like.

We pre-registered success as “higher alignment *and/or* higher clustering.” On that bar, Exp 1 **passes** — mainly via alignment. We would not bet a paper on the pairwise-sim gap alone.

**Interpretation we endorse:** loop-onset tokens are *somewhat more verbalizable / workspace-aligned* than matched controls on this model.  
**Interpretation we reject:** “triggers form a tight cluster that screams attractor.”

---

## Finding 2 — Just before onset, the map points at the trigger (Exp 2)

**Question:** In the five tokens before loop onset, does J-space state differ between looping and non-looping traces?

On a hooked re-generation (5 loops vs 195 non-loops), at L21:

| Signal (pre-onset) | Looping | Non-looping |
|--------------------|---------|-------------|
| Cosine(h at −1, trigger J-direction) | **0.172** | 0.130 |
| Top-1 J-readout = trigger at −1 | **60%** | **4%** |
| Occupancy proxy (readout entropy) at −2 | 0.249 | 0.577 |

The headline is **premature readout convergence**: in looping traces, the trigger token often becomes the top J-lens readout *one token before* it is emitted as the loop onset. Non-looping traces almost never show that pattern for their next token.

**Caveats we will not bury:** n_loop = 5; first analysis pass had a cache-window bug that dropped mid-sequence triggers (fixed with trigger-centered recache); occupancy is a proxy, not a full subspace count.

**Interpretation we endorse:** something measurable happens in J-space *before* the string detector fires — consistent with “lining up” on the trigger direction.  
**Interpretation we reject:** proof that this lining-up *causes* the loop.

---

## Finding 3 — Ablating that direction did not stop loops (Exp 3)

**Question:** If Exp 1–2 are more than coincidence, projecting the trigger direction out of the residual stream in the workspace should reduce loop rate relative to controls.

**Design (what we ran):** 11 baseline looping prompts × 6 conditions (baseline, ablate trigger @ L21, ablate random, ablate control-token, trigger @ sensory, trigger @ motor). Intervention = remove `h · v` along a unit J-direction during generation.

**Result:**

| Condition | Loop rate |
|-----------|-----------|
| baseline | 2/11 (18%) |
| ablate trigger @ L21 | 2/11 (18%) |
| ablate random | 1/11 |
| ablate control | 3/11 |
| trigger @ sensory | 2/11 |
| trigger @ motor | 1/11 |

McNemar baseline vs trigger ablation: **p ≈ 0.62**, Cohen’s h = **0**. Eval on tiny arithmetic stayed perfect — we did not merely lobotomize the model.

**Why this is a *null primary*, not a gotcha:**

1. Only **2** prompts re-looped even without ablation — almost no signal to reduce.
2. We ablated a **global** top empirical trigger (` given`), not each prompt’s own onset token.
3. We ablated at **all positions**, not only at −1 / onset (the Exp 2 critical window).
4. Workspace was **one layer**.

So: **this intervention, on this underpowered re-loop set, did not help.** That is compatible with (a) geometry is epiphenomenal, (b) we ablated the wrong thing the wrong way, or (c) both. We cannot distinguish (a) from (b) yet. What we *can* say is that a naive “delete the trigger direction everywhere at L21” is not a free lunch for anti-doom.

---

## How to read the three findings together

```
Exp 1:  triggers slightly more workspace-aligned than controls
Exp 2:  state / readout tip toward the trigger just before onset
Exp 3:  removing (a crude proxy of) that direction did not cut loops
```

The coherent short version:

> **J-space tracks loop onset; we have not shown it steers it.**

That is closer to “descriptive geometry of failure” than to “mechanistic kill switch.” It still matters. Detection and monitoring can use Exp 2–style signals even when ablation fails. Training-time or decoding-time fixes (Antidoom’s lane) remain necessary until someone shows a surgical residual intervention that works under fair controls.

---

## Why L21? (for the methods footnote)

We did not pick L21 because it is “the middle of the model.” We fitted Jacobians only on attention layers of the hybrid stack. On those eight layers we scored kurtosis, next-token prediction accuracy through the lens, and effective dimensionality. L21 was the fitted layer that landed in the workspace score region; later fitted layers (24, 27) were peeled as motor-like; earlier ones as sensory. With so few fitted layers, the “band” is a point. Any story that needs a *thick* workspace is limited by the hybrid fit, not by a preference for the number 21.

---

## Limitations (please cite these if you cite us)

- Public LFM2 ≠ blog early checkpoint; loop base rate is modest.
- NF4 generation: loop labels shift across reruns.
- Small looping *n* for Exp 2–3.
- Attention-only lens; conv layers skipped.
- Exp 3 intervention mismatched the Exp 2 critical window and used a global trigger direction.
- No Tier-3 full-layer HTML tour in this write-up.

---

## What we would do next (optional; not required for this post)

1. Stable generation (BF16/FP16 or vLLM) so known loopers re-loop.
2. Onset-timed, **per-trace** trigger ablation (and dose at −1).
3. More / harder prompts if this model still rarely loops.
4. Or move the same protocol to a high-loop model (e.g. Qwen3.5-4B under Liquid’s reported regime) if the goal is a decisive causal test.

Until then, the honest Phase-1 verdict on public LFM2-2.6B is:

**Geometry: mild yes. Dynamics: suggestive yes. Causality: not shown.**

---

## Cite / reproduce

- Prompt sample: `results/prompt_sample_ids.json` (seed=42)
- Run report: `results/RUN_REPORT_lfm2-2.6b_antidoom_mix_200.md`
- FAQ / numbers: `LFM2_26B_FINDINGS_AND_FAQ.md`
- Code: this repo’s `scripts/01`–`05`, lens `lenses/lfm2-2.6b.pt` (fit locally / Colab)

Related work to credit in a published version: Liquid Antidoom; Anthropic J-lens; Lazaridis et al. on knowledge-precision / loops; Silent Alarm / JADR on J-space under quantization; Elie Bakouch open-jlens-data.

---

*All quantitative claims above are from completed LFM2-2.6B artifacts. Nothing invented for narrative convenience.*
