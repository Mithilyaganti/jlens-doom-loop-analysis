# LFM2-2.6B J-Lens Doom-Loop Analysis — Report & FAQ

**Model:** `LiquidAI/LFM2-2.6B` (public HuggingFace checkpoint)  
**Dataset:** 200 prompts stratified across 7 reasoning sources from `LiquidAI/antidoom-mix-v1.0`, `seed=42`  
**Generation:** NF4 (bitsandbytes), `max_new_tokens=4000`, `temperature=0.01`  
**J-lens fit:** FP16 on Colab, 270 WikiText prompts, attn layers `[2,5,9,13,17,21,24,27]` → `lenses/lfm2-2.6b.pt`  
**Workspace band:** L21 only (attn-only hybrid lens)

> **Disclaimers:** This is **not** the blog's private early LFM2.5-2.6B checkpoint. It is **not** Antidoom-trained. Dynamic hybrid reasoning was left at model default.

---

## Table of contents

1. [Run report (summary)](#1-run-report-summary)
2. [Experiment overview](#2-experiment-overview)
3. [FAQ — methodology & metrics](#3-faq--methodology--metrics)
4. [Experiment 3 — detailed explanation](#4-experiment-3--detailed-explanation)
5. [Trigger token tables](#5-trigger-token-tables)
6. [Caveats & next steps](#6-caveats--next-steps)
7. [Repository layout](#7-repository-layout)

---

## 1. Run report (summary)

| Stage | Result |
|-------|--------|
| **Baseline** (200 prompts) | **11 loops (5.5%)**, 11 unique trigger tokens |
| **J-lens fit** | Done — `lenses/lfm2-2.6b.pt` |
| **Workspace band** | L21–L21 |
| **Exp1** static geometry | Success (mainly workspace alignment) |
| **Exp2** dynamic monitoring | Preliminary success (n_loop=5 on regen) |
| **Exp3** causal ablation | Null primary (underpowered) |

### Baseline looping prompt IDs

`86352, 96637, 120702, 123613, 133187, 179962, 199303, 290449, 340846, 346805, 422988`

### Exp1 — workspace means at L21

| Metric | Trigger | Controls (mean) | Interpretation |
|--------|---------|-----------------|----------------|
| Pairwise cosine sim | 0.0905 | 0.0878 | Tiny clustering advantage |
| Workspace alignment | **0.2477** | **0.1942** | Clearer signal |

`success_criterion_met: true` — mainly from alignment, not pairwise clustering.

### Exp2 — pre-onset signals (regen: 5 loops / 195 non-loops)

| Signal | Looping (n=5) | Non-looping (n=195) | Direction |
|--------|---------------|---------------------|-----------|
| Alignment at −1 | 0.172 | 0.130 | Looping higher |
| Top-1 = trigger at −1 | 60% | 4% | Strong convergence |
| Occupancy at −2 | 0.249 | 0.577 | Suggestive collapse |

`success_criterion_met: true` (preliminary; small n).

### Exp3 — loop rates (11 baseline loopers × 6 conditions = 66 pairs)

| Condition | Loop rate | Count |
|-----------|-----------|-------|
| baseline | 0.182 | 2/11 |
| ablate_trigger (L21) | 0.182 | 2/11 |
| ablate_random | 0.091 | 1/11 |
| ablate_control | 0.273 | 3/11 |
| ablate_trigger_sensory | 0.182 | 2/11 |
| ablate_trigger_motor | 0.091 | 1/11 |

McNemar baseline vs trigger ablate: **p=0.62**, Cohen's h=**0.0** — primary trigger ablation did **not** reduce loops vs baseline. Result type: **mixed / null primary**.

---

## 2. Experiment overview

### Research question

Do doom-loop trigger tokens occupy a special region of **J-space** (the Jacobian lens representation), and can intervening on those directions **causally** prevent loops?

### Three experiments

| Exp | Type | Question |
|-----|------|----------|
| **1** | Static / descriptive | Do trigger tokens cluster differently in J-space than control tokens? |
| **2** | Dynamic / descriptive | Does internal state differ in the 5 tokens before loop onset? |
| **3** | **Causal** | If we ablate trigger directions during generation, do loops stop? |

Exp 1–2 show correlation. Exp 3 tests mechanism.

---

## 3. FAQ — methodology & metrics

### Q: Is `seed=42` set in code, and is it for prompts or tokens?

**Yes, it is set in code — for prompt selection only.**

In `jspace/prompts.py`, `sample_reasoning_prompts(..., seed=42)` stratified-samples row indices from antidoom-mix, then saves to `results/prompt_sample_ids.json`. Later runs reload that same list.

It does **not** control which tokens the model generates or which become loop triggers.

**Control token sets in Exp1** use separate logic:
- **Triggers:** empirical baseline CSV + restart words (deterministic)
- **Freq-content:** `build_frequency_matched_content(..., seed=0)` — default seed **0**
- **Random:** `build_random_tokens(..., seed=42)`
- **Discourse:** fixed marker list

---

### Q: What are “control tokens”?

In Exp1, **controls are comparison token sets**, not “tokens that didn’t loop.”

| Set | What it is | Purpose |
|-----|------------|---------|
| **Trigger** | Tokens at loop onset (+ restart-word seeds) | What we study |
| **Freq-content** | Tokens frequency-matched to triggers | “Is it just a common word?” |
| **Discourse** | Markers like `But`, `So`, `However` | “Is it just discourse structure?” |
| **Random** | Random vocab sample | Generic baseline |

We ask: do **loop trigger directions** look more special than these baselines?

---

### Q: What are the Exp1 metrics, and why was success met?

**At workspace layer L21:**

**Pairwise cosine similarity** — mean off-diagonal cosine similarity between J-lens direction vectors **within** each token set. Higher ⇒ more clustering in direction space.

**Workspace alignment** — for each token direction, project onto top-64 right singular vectors of the layer Jacobian; report mean **fraction of direction variance** captured. Higher ⇒ direction lives more in the “workspace” subspace.

**Success rule** (`scripts/03_exp1_static_geometry.py`):

```python
success = (trig_sim > ctrl_sim) or (trig_al > ctrl_al)
```

Pairwise sim barely moved (0.0905 vs 0.0878). **Alignment** moved clearly (0.2477 vs 0.1942), so success was met mainly from alignment.

---

### Q: What does “lower than baseline 5.5% due to NF4 nondeterminism” mean?

Same setup, **different loop counts on re-run**:

| Run | Loops | Rate |
|-----|-------|------|
| Baseline (first 200) | 11 | 5.5% |
| Exp2 hooked regen (200) | 5 | 2.5% |
| Exp3 (11 known loopers) | 2 re-looped | 18% of 11 |

Baseline, Exp2, and Exp3 **generation** were all **NF4 on HF**. FP16 was **only for J-lens fitting** on Colab — not a precision switch between runs.

With NF4 + `temperature=0.01`, re-running the same prompt can still yield different text and different loop labels (quantization, kernels, numerics). With only ~5–11 loops total, small changes look large in percentage terms.

---

### Q: What are the Exp2 metrics?

All measured in the **5 tokens before loop onset** (positions −5…0), anchored at the trigger token, at **L21**.

**Alignment at −1** — cosine similarity between residual `h` at L21 position −1 and the J-lens direction of the **trigger token**. Higher ⇒ state is more “lined up” with that direction right before onset.

**Top-1 = trigger at −1** — fraction of traces where the trigger token is the **#1** J-lens readout at position −1.

**Occupancy at −2** — normalized entropy of top-k J-lens readout scores (`entropy / log(k)`). 1 = spread; 0 = collapsed onto one direction.

**Exp2 success** (PROJECT_SPEC §10.7): looping traces differ pre-onset by **any** of occupancy collapse, premature readout convergence, or elevated alignment. Marked preliminary because n_loop=5.

---

### Q: Causal ablation null — does that mean loops aren’t due to trigger tokens?

**No — it does not prove that.**

Exp3 showed projecting out the trigger J-lens direction at L21 did **not** lower loop rate vs baseline (both 2/11). That means **this specific intervention** showed no causal effect, not that triggers are innocent.

Plausible reasons: wrong layer (only L21), wrong timing (whole-gen ablation vs at −1), wrong direction (global top trigger vs per-trace), tiny n, reruns didn’t re-loop.

---

### Q: NF4 — baseline vs later runs, and FP16

**Correct:** baseline and Exp2/Exp3 generation were all **NF4**. FP16 was **only J-lens fitting**.

**“NF4 nondeterminism”** means: same model, same temp, same prompt → outputs can still differ across runs, so loop labels can flip. The initial 11 loops **did** happen on NF4; the issue is **reruns** often don’t reproduce them.

**FP16/BF16 generation** is usually more stable than NF4 but not guaranteed bit-identical. “Reduce NF4 nondeterminism” as a next step means try FP16/BF16 for generation if stable loop labels across reruns are needed.

---

## 4. Experiment 3 — detailed explanation

### Motive

Experiments 1 and 2 are **observational** — they describe geometry and pre-onset state. Exp 3 asks the **causal** question:

> If we surgically remove trigger-token J-lens directions from the model’s workspace hidden state during generation, do doom loops stop?

That is the step that turns “interesting geometry” into a mechanistic claim. PROJECT_SPEC §11 calls it the experiment that decides blog post vs paper-grade result.

### Hypothesis

**If true:** loops depend on the model using specific verbalizable directions in workspace layers. Removing those directions should reduce looping, specifically — not random directions, not similar non-trigger words, and concentrated in workspace (not sensory/motor).

**If false:** J-space trigger geometry is **descriptive** (reflects loops) but **not causal** (removing the direction doesn’t prevent them).

Both outcomes are publishable.

### What we did

1. **Prompts:** Only the **11 prompts that looped in baseline** (known loop-prone subset).
2. **Conditions:** Same prompt × 6 generation modes = **66 paired runs**.
3. **Intervention:** Forward hooks on residual stream at selected layers:

   ```
   h_new = h - (h · v) * v
   ```

   where `v` is a unit J-lens direction for some token at that layer.

4. **Six conditions:**

   | Condition | What gets removed | Why |
   |-----------|-------------------|-----|
   | baseline | Nothing | Reference |
   | ablate_trigger | Top empirical trigger direction @ L21 | Main test |
   | ablate_random | Random unit vector @ L21 | Any direction removal? |
   | ablate_control | Frequency-matched non-trigger word @ L21 | Any verbalizable word? |
   | ablate_trigger_sensory | Trigger direction @ early layers | Workspace-specific? |
   | ablate_trigger_motor | Trigger direction @ late layers | Different band? |

5. **Outcome:** Loop detector on each generation; McNemar paired test; dose-response (top-1/3/5 directions); tiny eval sanity check.

### Ideal positive vs actual result

**Ideal:** `ablate_trigger` clearly below baseline, below random/control, sensory/motor don’t help as much, eval intact.

**Actual:** trigger ablation **did not** beat baseline (2/11 each). McNemar p=0.62. **Null primary** — no causal evidence with this setup.

### Implementation caveats (important)

1. **Ablate everywhere** — spec says ablate at trigger position (−1/−2); we used `position=None` → ablate at **every token** for the whole generation.
2. **Single layer L21** — not a multi-layer workspace band.
3. **One global trigger** — main condition ablates top empirical trigger (` given`, id 2997), not each prompt’s own trigger.
4. **Tiny n + rerun instability** — only 2/11 re-looped on Exp3 baseline rerun.

### One-sentence summary

Exp 3 asked whether removing trigger J-directions from workspace hidden state stops loops; under this coarse intervention it did not — inconclusive/negative, not a clean refutation of trigger involvement.

---

## 5. Trigger token tables

### Baseline — 11 loops (first pass)

| # | Token ID | Decoded |
|---|----------|---------|
| 1 | 2997 | ` given` |
| 2 | 4837 | `180` |
| 3 | 2011 | ` +` |
| 4 | 3351 | ` try` |
| 5 | 1141 | ` -` |
| 6 | 6654 | ` currently` |
| 7 | 941 | ` The` |
| 8 | 7383 | `^\\` |
| 9 | 5521 | ` difference` |
| 10 | 3498 | ` says` |
| 11 | 4412 | `But` |

### Exp2 hooked regen — 5 loops (“last run”)

| Prompt ID | Trigger |
|-----------|---------|
| 133187 | `.` |
| 157993 | ` {` |
| 207228 | ` But` |
| 346805 | ` but` |
| 422988 | `option` |

Only **3/5** overlap baseline looping prompts (`133187`, `346805`, `422988`). Most baseline loopers did not loop again on regen.

---

## 6. Caveats & next steps

- Public LFM2-2.6B ≠ Liquid blog checkpoint; not Antidoom-trained.
- Small n (11 → 5 → 2 loops across runs).
- NF4 rerun instability — not an FP16 vs NF4 mix-up for generation.
- Exp3 null ≠ “triggers don’t cause loops.”
- Exp2 Tier-3 full-residual HTML visualization not done.

**Possible improvements:** FP16/BF16 generation, per-trace trigger ablation, ablate only at −1, multi-layer band, more prompts / hard packs, larger loop sample.

---

## 7. Repository layout

After cleanup (see `artifacts/README.md`):

```
j-lens/
├── LFM2_26B_FINDINGS_AND_FAQ.md   ← this file
├── PROJECT_SPEC.md
├── PROMPT.md
├── README.md
├── jspace/                         # core library
├── scripts/                        # pipeline scripts
├── results/                        # canonical active results (plots, JSON, CSV)
├── lenses/                         # fitted J-lens weights (.pt, gitignored)
├── artifacts/
│   ├── exports/                    # Google Drive snapshot downloads
│   ├── jlens_fit/                  # Colab fit outputs
│   └── archive/                    # superseded Qwen / partial runs
├── docs/                           # handoffs and session notes
├── notebooks/                      # Colab notebooks
└── vendor/                         # open-jlens-data, antidoom
```

Canonical run report (short): `results/RUN_REPORT_lfm2-2.6b_antidoom_mix_200.md`  
Handoff notes: `results/HANDOFF_LFM2_26B.md` or `docs/handoffs/`

---

*All numbers in this document are from completed LFM2-2.6B artifacts — nothing invented.*
