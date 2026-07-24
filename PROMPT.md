# PROMPT — J-Space Analysis of Doom Loops

You are implementing a mechanistic-interpretability research project: applying Anthropic's Jacobian Lens (J-lens / J-space) to characterize what happens inside small language models when they enter doom loops, and testing whether intervening on J-space directions can prevent them.

This prompt is the operator-level directive. The full project specification lives in `PROJECT_SPEC.md` — read it completely before writing any code.

---

## CRITICAL INSTRUCTIONS

### 1. EXECUTION PHILOSOPHY — ONE MODEL FIRST, THEN EXPAND

This is the most important rule. Do NOT run all models in parallel. Do NOT start cross-model comparisons until Phase 1 is complete.

- **Phase 1 (week 1–3): Qwen/Qwen3.5-4B only.** Run Experiments 1, 2, 3 in full. Validate every step. Produce all deliverables for this one model.
- **Phase 2 (week 4–5, only after Phase 1 succeeds):** Add `Qwen/Qwen3.5-4B-Base`, `google/gemma-4-E2B` (+ `-it` variant), `google/gemma-4-E4B` (+ `-it` variant). Re-run Exp 1 + Exp 3 only (skip the dynamic monitoring — you already know the protocol works).
- **Phase 3 (stretch, only if time and compute allow):** Add Liquid LFM2.5 family — `LiquidAI/LFM2.5-1.2B-Base`, `LiquidAI/LFM2.5-1.2B-Instruct`, `LiquidAI/LFM2.5-1.2B-Thinking` (the reasoning model — most doom-loop-relevant). These require custom J-lens fitting because LFM2.5 is a hybrid architecture (attention + conv + MLP), not a pure transformer. See PROJECT_SPEC.md §5.3.

Rationale: saves compute, catches bugs early on one model before they propagate, gives you a clean baseline to compare cross-model results against.

### 2. MANDATORY READING — read these yourself before writing any code

In priority order:

1. **Liquid AI Antidoom blog** (source of truth for trigger tokens and loop definition):
   https://www.liquid.ai/blog/antidoom (published Jul 7, 2026)

2. **Liquid AI Antidoom code** — read `src/antidoom/repetition.py`, `src/antidoom/generate.py`, `src/antidoom/tokens.py`, `src/antidoom/sampling.py`, `configs/default.yaml`:
   https://github.com/Liquid4All/antidoom

3. **Antidoom prompt mix** (prompts only, no gold answers):
   https://huggingface.co/datasets/LiquidAI/antidoom-mix-v1.0

4. **Anthropic J-lens paper** (also on arXiv as `arXiv:2607.15495`, published Jul 6, 2026):
   https://transformer-circuits.pub/2026/workspace/index.html

5. **Elie Bakouch's open-jlens-data repo — USE THIS CODEBASE.** It vendors Anthropic's `jacobian-lens` (which is explicitly unmaintained per its README: "Reference implementation. Not maintained and not accepting contributions.") and adds MoE fitters, sharding, CKA utilities, and 11 experiment protocols:
   https://github.com/eliebak/open-jlens-data

6. **Elie's CKA explorer** (for understanding cross-model geometry analysis):
   https://eliebak.com/viz/jspace-open

7. **Pre-fitted J-lenses** (38 open models, including Qwen3.5-4B — download, don't refit):
   https://huggingface.co/neuronpedia/jacobian-lens

8. **Antidoom HuggingFace dataset** (already listed as #3 — keep separate in your mind: prompts come from here, loop-detection code comes from the GitHub repo in #2).

Extract the exact list of high-frequency loop-initiating tokens and the precise loop detection criteria (repetition count + length, character-based not token-based) directly from Liquid's materials. Do not hard-code an assumed list.

### 3. TRIGGER TOKENS — DO NOT INVENT THE LIST

This is the most common failure mode. Liquid published the top-5 trigger tokens ONLY for `LFM2.5-2.6B-early-ckpt`, which is an **internal Liquid checkpoint that is NOT public**. The top-5 was:

```
count    share  token
   2277  11.39%  ' the'
    902   4.51%  ' So'
    644   3.22%  'Alternatively'
    511   2.56%  'Wait'
    493   2.46%  ' But'
```

These are **NOT** the trigger tokens for Qwen3.5-4B. For Qwen3.5-4B, **nobody has published the equivalent table**. You MUST generate it yourself by:

1. Running Antidoom's generation pipeline on `LiquidAI/antidoom-mix-v1.0` + `Qwen/Qwen3.5-4B` + greedy sampling (temp=0.01).
2. Inspecting `iter_0_ftpo_pairs.jsonl` to extract the actual top-N first-repeated-token distribution.
3. This generated table is itself a publishable artifact — you will be the first to publicly document Qwen3.5-4B's trigger-token distribution.

The OPERATIONAL trigger set used by Antidoom internally is `_RESTART_WORDS` in `src/antidoom/generate.py` — 26 words:

```
actually, after, also, alternatively, because, but, finally, first, given,
hmm, however, in, let, looking, maybe, now, okay, perhaps, second, since,
so, the, then, therefore, this, thus, wait
```

Use this 26-word list as your initial trigger-token set for Experiment 1, then refine it with the top-N tokens you extract from your own Qwen3.5-4B run.

### 4. LOOP DETECTION — use Liquid's exact criteria (CHARACTER-BASED, not token-based)

From `src/antidoom/repetition.py`, verbatim defaults:

```python
def find_inner_repetition(
    text: str,
    *,
    min_repeats: int = 4,
    max_period: int = 1024,
    min_period: int = 1,
    min_total_repeated: int = 60,   # <-- CHARACTERS, not tokens
    sample_len: int = 16,
    sample_interval: int = 128,
) -> tuple[bool, RepeatHit | None]:
```

- A loop = a substring that repeats **≥4 times** with total span **≥60 characters**.
- The `period` (loop unit length) is between 1 and 1024 characters.
- Detection runs on DECODED TEXT, not tokens. The detector fingerprints 16-character substrings at every 128-character interval.
- When you need to map a loop boundary back to a token position (which you will, for J-lens readouts), use `antidoom/tokens.py: TokenState.char_to_token_index`. Do not write your own char-to-token mapper — use Liquid's.

### 5. J-LENS — DO NOT REFIT IF A PRE-FITTED LENS EXISTS

This saves you 1–2 GPU-hours per model and gets you to Experiment 1 in an afternoon.

- **Qwen3.5-4B**: Download `qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens.pt` (387 MB) from `https://huggingface.co/neuronpedia/jacobian-lens`. Fit by Mateusz Piotrowski (@mntss, Anthropic) on 417 wikitext prompts, converged 2026-06-13. **Caveat**: this lens was fit on `Qwen/Qwen3.5-4B` (the INSTRUCT/post-trained version), NOT on `-Base`. For base-vs-instruct comparison you will need to fit a lens on `Qwen/Qwen3.5-4B-Base` yourself.
- **Gemma-4 Edge models** (`google/gemma-4-E2B`, `google/gemma-4-E4B`, plus their `-it` variants — note the **capital "E"** in the HF slug, this is Google's Edge-model naming convention): pre-fitted lenses also exist on the same `neuronpedia/jacobian-lens` repo. The neuronpedia lens directory uses lowercase (`gemma-4-e2b`, `gemma-4-e4b`) but the HF model slug is `google/gemma-4-E2B` etc. — don't confuse the two.
- **Qwen3.5-4B-Base**: NO pre-fitted lens. You will need to fit it yourself using the recipe in `eliebak/open-jlens-data` (1000 wikitext-103 prompts, max_seq_len=128, skip_first=16, dim_batch=64, target=final layer, stop_at_delta=0.002). Estimated ~1–2 GPU-hours on H200; **much longer on RTX 5050** — consider renting 2 hours of cloud H200 time for this one task if you want to keep your laptop free for the actual analysis.
- **Liquid LFM2.5 models**: NO pre-fitted lens exists for ANY Liquid model. LFM2.5 is a hybrid architecture (attention + conv + MLP), and the standard J-lens fitter assumes pure transformer. You will need to adapt the fitter — see `eliebak/open-jlens-data/code/moe/fit_moe.py` for an example of how Elie adapted it for MoE, and be prepared to write custom hooks for the conv layers. This is why Liquid models are Phase 3, not Phase 1.

Load Qwen3.5-4B in 4-bit (bitsandbytes NF4). Load the J-lens itself in fp16 or bf16 — it's only 387 MB and fits easily alongside the model. Do NOT quantize the lens.

### 6. WORKSPACE BAND — DO NOT ASSUME BOUNDARIES

The mid-depth workspace band is **model-specific**. For Sonnet 4.5 the Anthropic paper says ~L38–L92 (out of ~100+ layers). For Qwen3.5-4B (36 layers), the "first third / last few" heuristic gives ~L12–L33 — but **you MUST compute the actual boundary empirically** using:

- Excess kurtosis of J-lens vectors per layer (near zero in sensory, rises sharply at workspace onset, falls again in motor).
- Next-token-prediction accuracy of `lens(h_ℓ)` per layer.
- Effective dimensionality of the J-space per layer.

Anthropic paper Figure 28 shows the recipe. `eliebak/open-jlens-data` has utilities for this. Do not assume any layer indices until you have plotted these three curves for your specific model.

### 7. HARDWARE — Windows, RTX 5050 8GB VRAM, 16GB RAM

- Load all models in 4-bit quantization (bitsandbytes NF4 is the default; GPTQ is also fine — pick one and be consistent across experiments).
- Load J-lenses in fp16/bf16 — they are small (≤400 MB for 4B-class models).
- Never attempt >4B parameters without 4-bit quant and careful memory management.
- For `LiquidAI/LFM2.5-8B-A1B` (MoE, 8B total / 1B active): use the GGUF quantized version via llama.cpp, OR skip if MoE fitter causes issues. Elie's `open-jlens-data/code/moe/fit_moe.py` has a MoE-capable fitter (program.md experiment E6). MoE J-lens behavior is different from dense — interpret results cautiously.
- For 12B Gemma (if you attempt it): runs in ~16GB VRAM full precision or ~8GB quantized — borderline on your card. Only attempt after E2B/E4B work, and expect possible OOM during generation.

### 7b. CACHING STRATEGY — DO NOT CACHE FULL RESIDUAL STREAMS BY DEFAULT

This is a critical memory-management rule. Caching the full residual stream `h_ℓ` for every layer × every token × many prompts is infeasible on 8GB VRAM + 16GB RAM. The math: 36 layers × 2048 tokens × 2560 d_model × 2 bytes (bf16) ≈ **377 MB per prompt**. For 200 prompts in Experiment 2 = ~75 GB. This will OOM your system on prompt 4–5.

**Default strategy (use this in Experiment 2):**

- **Primary cache**: top-k J-lens readouts (just the top-10 token indices and their probabilities) at every layer × every token. Size: 36 × 2048 × 10 × 4 bytes ≈ **3 MB per prompt**. 200 prompts = 600 MB. Trivial.
- **Secondary cache** (only for workspace-band layers): full residual stream at 3–5 key workspace-band layers (you identified these in §8). Size: 5 × 2048 × 2560 × 2 ≈ **52 MB per prompt**. 200 prompts = 10 GB. Workable if streamed to disk and freed between prompts.
- **Loop boundary metadata**: char position, token position (via `TokenState.char_to_token_index`), trigger token, generated text. Tiny.

**Stretch / optional (only for ~5 hand-picked looping traces, for the HTML visualization in Exp 2):**

- Full residual stream at ALL layers for these 5 traces only. ~1.9 GB total. Manageable.

**FORBIDDEN:**

- Caching full residuals at all layers for all prompts. Will OOM. Don't try.
- Caching residuals in GPU memory across prompts. Always stream to disk and `del` + `torch.cuda.empty_cache()` between prompts.

The J-lens readouts (primary cache) are sufficient for the dynamic monitoring analysis in Experiment 2 — you don't actually need raw residuals for the top-1/top-3 convergence analysis or the workspace occupancy measurement. The residuals are only needed for the cosine-similarity-to-trigger-direction analysis, and that can be done with just the workspace-band layers (secondary cache).


### 8. SKIP THE VL MODEL

`LiquidAI/LFM2.5-VL-1.6B` (verify it exists publicly before referencing) is a vision-language model. The Anthropic J-lens is text-only. Applying it to a VL model requires significant adaptation (handling the vision encoder, multimodal fusion layers, cross-modal attention) — this is its own research project, not a side experiment. **Skip for now.** If you want to extend to multimodal later, the Inkling J-lens fit by PrimeIntellect (also MoE) is the closest precedent and still required custom fitter work.

### 9. PRIOR ART YOU MUST ENGAGE WITH

- **Lazaridis et al. "Can Editing 1 Neuron Fix Repetition Loops in LLMs?"** (`arXiv:2606.13705`, Jun 9 2026) — per-neuron attribution localization of doom loops in Gemma 4, with single-neuron weight edits sufficient to suppress loops in some models. **THIS IS YOUR CLOSEST COMPETITOR.** You must:
  - Cite it.
  - Differentiate explicitly: J-lens gives a vocabulary-grounded geometric handle (steering directions for specific tokens in workspace coordinates) that per-neuron attribution lacks. Per-neuron finds *where*; J-lens finds *what concept is being over-represented*.
  - Engage with their "knowledge-precision problem" framing for long-budget loops — J-lens ablation may not solve loops that are fundamentally about missing facts, only loops that are about over-attracted trigger tokens.

- **Silent Alarm / JADR** (`arXiv:2607.12792`, Jul 14 2026) — applies J-space to jailbreak safety recognition across Qwen3 / Gemma 2 models and BF16/INT8/INT4 quantization. **Cite as independent evidence that J-lens is mature enough for downstream behavioral analysis.** Also useful for the quantization-robustness question — they show J-space signatures survive INT4.

- **Liquid AI Antidoom blog** (Jul 7 2026) — engineering fix (FTPO training), no interpretability. You are the mechanistic complement. Cite as the trigger-token and loop-detection source of truth.

- **Anthropic J-lens paper** (`arXiv:2607.15495`, Jul 6 2026) — establishes J-lens, J-space, mid-depth workspace band, activation patching and ablation methodologies. Cite as methodological foundation.

- **Elie Bakouch's CKA explorer** (Jul 17–19 2026) — replications and cross-model geometry. Cite for the open-model lens inventory and cross-model CKA methodology.

- **Word Salad Chopper** (`arXiv:2511.00536`, Nov 2025) — loop detection via hidden states of `<\n\n>` tokens with a linear classifier. Different mechanism (post-hoc detection, not causal localization). Cite as related work.

- **Wait, Wait, Wait** (`arXiv:2512.12895`, Dec 2025) and **Circular Reasoning** (`arXiv:2601.05693`, Jan 2026) — behavioral theory of why reasoning models loop. Cite for motivation.

### 10. EXECUTION ORDER — strict

1. Read all mandatory materials (§2). Do not skip.
2. Setup: clone `eliebak/open-jlens-data`, clone `Liquid4All/antidoom`, download Qwen3.5-4B J-lens from `neuronpedia/jacobian-lens`, download Qwen3.5-4B in 4-bit.
3. Compute the workspace band for Qwen3.5-4B empirically (PROJECT_SPEC.md §7).
4. Generate Qwen3.5-4B's trigger-token table by running Antidoom's pipeline (PROJECT_SPEC.md §6).
5. Experiment 1 — Static Geometry (PROJECT_SPEC.md §9).
6. Experiment 2 — Dynamic Monitoring (PROJECT_SPEC.md §10).
7. Experiment 3 — Causal Interventions (PROJECT_SPEC.md §11). **This is the experiment that decides blog-post vs. paper.**
8. Status notes after each experiment. Write `RESULTS_SUMMARY.md` incrementally.
9. Only after Phase 1 is complete: proceed to Phase 2 (PROJECT_SPEC.md §5.2) and repeat Exp 1 + Exp 3.
10. Phase 3 (Liquid LFM2.5) only if time permits — requires custom J-lens fitter work.

### 11. DELIVERABLES

- Clean codebase with README.
- `results/` folder with plots (PNG + SVG), CSVs/JSONs, and `RESULTS_SUMMARY.md`.
- `trigger_tokens_qwen3.5-4b.csv` — the table you generate for Qwen3.5-4B (publishable artifact).
- Status notes after each experiment in `results/status/`.
- Everything must be reproducible on the stated hardware (Windows + RTX 5050 8GB + 16GB RAM).

### 12. KEY INNOVATION REMINDER (for your own understanding)

Liquid AI identified bad *output tokens* and changed their preference via FTPO training. We are looking *inside* the model with J-lens to understand the internal workspace state that makes those tokens attractive, and testing whether intervening on the internal directions themselves can affect the failure mode — **without training**. This is mechanistic, not behavioral. If Experiment 3 succeeds (ablating trigger-token J-lens directions in the workspace band reduces loop rate, with controls showing the effect is layer-band-specific and trigger-token-specific), you have an inference-time antidoom that doesn't require LoRA training — useful for production and a clean publishable result.

If you need more information, read the links above. Do NOT guess trigger tokens, model names, or detection criteria. If something is unverified, mark it UNVERIFIED in your output and ask before proceeding.

Now read `PROJECT_SPEC.md` for the full specification.
