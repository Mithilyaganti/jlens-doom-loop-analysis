# Project Specification: J-Space Analysis of Doom Loops

## 0. Execution Philosophy — One Model First, Then Expand

This project uses a **phased execution model** to save compute, catch bugs early, and produce a clean baseline before scaling. Do NOT skip phases. Do NOT run models in parallel until Phase 1 is complete and validated.

| Phase | Duration | Models | Experiments | Gate to next phase |
|---|---|---|---|---|
| **Phase 1** | Week 1–3 | `Qwen/Qwen3.5-4B` only | Full Exp 1 + 2 + 3 | All three experiments produce clean plots + CSVs + status notes; Exp 3 ablation shows a measurable effect (positive OR negative — both are publishable) |
| **Phase 2** | Week 4–5 | + `Qwen/Qwen3.5-4B-Base`, `google/gemma-4-E2B` (+`-it`), `google/gemma-4-E4B` (+`-it`) | Exp 1 + Exp 3 only (skip Exp 2 — protocol already validated) | Cross-model comparison plots produced; consistency (or inconsistency) of trigger-token workspace location documented |
| **Phase 3** (stretch) | Week 6+, if time permits | + Liquid LFM2.5 family | Exp 1 + Exp 3 | Custom J-lens fitter for hybrid architecture built and validated |

**Rationale**: A single-model result with full causal ablation is publishable as a blog post and a workshop paper. Cross-model replication strengthens it to ICML-tier. Liquid model extension is a bonus that requires significant fitter-engineering work and is not on the critical path.

---

## 1. Core Goal

Characterize what is happening in J-space (the subspace spanned by J-lens directions, also called the model's "verbalizable workspace") when small language models enter doom loops. Specifically:

1. **Static**: Do doom-loop trigger tokens (e.g., `'Wait'`, `'So'`, `'Alternatively'`, `'But'`, `'the'`) have distinctive J-lens direction geometry in the mid-depth workspace band, compared to control tokens?
2. **Dynamic**: Does the J-space state of the model change measurably in the tokens immediately before loop onset, in a way that distinguishes looping from non-looping traces?
3. **Causal**: Does intervening on the J-lens directions of trigger tokens (projecting them out of the residual stream in the workspace band) reduce the doom-loop rate, with appropriate controls?

Produce clean experimental results that can later support a blog post and a possible ICML 2027 submission.

---

## 2. Hardware Constraints (strict)

- Windows laptop
- RTX 5050, 8 GB VRAM
- 16 GB system RAM
- Use 4-bit quantization aggressively (bitsandbytes NF4 is the default; GPTQ also acceptable — pick one and stay consistent).
- Only models that realistically fit at 4-bit in 8 GB VRAM.
- J-lens tensors themselves (≤400 MB for 4B-class models) are loaded in fp16/bf16, NOT quantized.
- Never attempt >4B parameters without 4-bit quant and explicit memory management (`torch.cuda.empty_cache()` between runs, batch size 1 for generation, gradient checkpointing off during inference).

### 2.1 Caching Strategy — Do NOT Cache Full Residual Streams by Default

This is a critical memory-management rule. Caching the full residual stream `h_ℓ` for every layer × every token × many prompts is infeasible on 8GB VRAM + 16GB RAM. The math:

- Qwen3.5-4B: 36 layers × 2048 tokens × 2560 d_model × 2 bytes (bf16) ≈ **377 MB per prompt**.
- 200 prompts in Experiment 2 = ~75 GB. Will OOM your system on prompt 4–5.

**Tiered caching strategy** (used in Experiment 2 and any other experiment that needs to record intermediate state across many prompts):

| Tier | What to cache | Size per prompt | When to use |
|---|---|---|---|
| **Tier 1 — always** | Top-k J-lens readouts (top-10 token indices + probabilities) at every layer × every token. Plus loop boundary metadata (char pos, token pos via `TokenState.char_to_token_index`, trigger token, generated text). | ~3 MB | Default. Sufficient for top-1/top-3 convergence analysis, workspace occupancy measurement, and most of Experiment 2's analysis. |
| **Tier 2 — default for workspace analysis** | Full residual stream at 3–5 key workspace-band layers (the layers you identified in §8 as the workspace band). | ~52 MB | Default for the cosine-similarity-to-trigger-direction analysis. Stream to disk, free between prompts. 200 prompts = ~10 GB total disk. |
| **Tier 3 — stretch** | Full residual stream at ALL layers. | ~377 MB | ONLY for ~5 hand-picked looping traces, for the HTML visualization in §10.6. ~1.9 GB total. |

**FORBIDDEN:**

- Caching full residuals (Tier 3) for all prompts. Will OOM. Don't try.
- Caching residuals in GPU memory across prompts. Always stream to disk, then `del tensor` + `torch.cuda.empty_cache()` between prompts.
- Caching J-lens readouts as full probability distributions over the vocab. Cache only the top-k indices + values.

For one-off tasks that exceed laptop capability (specifically: fitting a J-lens on `Qwen/Qwen3.5-4B-Base`), budget 2 hours of rented cloud GPU time (H200 or A100). Do NOT attempt to fit a base-model J-lens on the RTX 5050 — it will be painfully slow and may OOM.

---

## 3. Mandatory Reading (do this first, in this order)

You must read these yourself before writing any code:

1. **Liquid AI Antidoom blog** (source of truth for trigger tokens and loop definition, published Jul 7, 2026):
   https://www.liquid.ai/blog/antidoom

2. **Liquid AI Antidoom code** — read `src/antidoom/repetition.py`, `src/antidoom/generate.py`, `src/antidoom/tokens.py`, `src/antidoom/sampling.py`, `configs/default.yaml`:
   https://github.com/Liquid4All/antidoom

3. **Antidoom prompt mixture** (prompts only, no gold answers):
   https://huggingface.co/datasets/LiquidAI/antidoom-mix-v1.0

4. **Anthropic J-lens paper** (also on arXiv as `arXiv:2607.15495`, published Jul 6, 2026):
   https://transformer-circuits.pub/2026/workspace/index.html

5. **Elie Bakouch's open-jlens-data repo — USE THIS CODEBASE**. It vendors the Anthropic `jacobian-lens` repo (which is explicitly unmaintained per its README) and adds MoE fitters, sharding, CKA utilities, and 11 experiment protocols:
   https://github.com/eliebak/open-jlens-data

6. **Elie's CKA explorer** (for understanding cross-model geometry analysis):
   https://eliebak.com/viz/jspace-open

7. **Pre-fitted J-lenses** (38 open models — download, do not refit when possible):
   https://huggingface.co/neuronpedia/jacobian-lens

8. **Closest competitor paper** — Lazaridis et al., "Can Editing 1 Neuron Fix Repetition Loops in LLMs?", `arXiv:2606.13705` (Jun 9, 2026). Read in full. You must differentiate from this work.

9. **Independent J-lens behavioral prior art** — Silent Alarm / JADR, `arXiv:2607.12792` (Jul 14, 2026). J-space applied to jailbreak safety recognition.

Extract the exact list of high-frequency loop-initiating tokens and the precise loop detection criteria (repetition count + length, character-based not token-based) directly from Liquid's materials. Do not hard-code an assumed list.

---

## 4. Models (precise)

### 4.1 Phase 1 — Primary (highest priority, run first and alone)

- **`Qwen/Qwen3.5-4B`** — verified to exist on HuggingFace (created 2026-02-27, 6.65M downloads). Liquid explicitly reported a 22.9% doom-loop rate on this exact model under greedy sampling, reduced to 1% after their FTPO training. This is the best publicly available model that matches their experimental conditions. A pre-fitted J-lens exists at `neuronpedia/jacobian-lens/qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens.pt` (387 MB).

  **Important caveat**: The pre-fitted lens was fit on `Qwen/Qwen3.5-4B` (the INSTRUCT / post-trained version). The `hf_model_name` field in the lens config confirms this. So this lens sees the post-trained model's workspace, not the base model's. That is fine for Phase 1.

### 4.2 Phase 2 — Replication (only after Phase 1 validates the pipeline)

- **`Qwen/Qwen3.5-4B-Base`** — verified to exist on HuggingFace (created 2026-02-27, 203K downloads). The base (pre-instruct) version. **NO pre-fitted J-lens exists.** You will need to fit it yourself using the same 1000-prompt wikitext-103 recipe documented in `eliebak/open-jlens-data`. Budget 2 hours of rented H200 time. The base-vs-instruct comparison is a strong result: does post-training *install* the doom-loop-trigger dispositions in the workspace?

- **`google/gemma-4-E2B`** and **`google/gemma-4-E2B-it`** (note the **capital "E"** — Google's Edge-model naming convention). ~2.3B effective parameters, built for edge/mobile. Pre-fitted J-lens exists on `neuronpedia/jacobian-lens` under the lowercase directory `gemma-4-e2b` (HF slug and lens directory naming differ — don't confuse them). Easy fit at 4-bit on RTX 5050.

- **`google/gemma-4-E4B`** and **`google/gemma-4-E4B-it`** (same capital-E convention). ~4.5B effective parameters, best fit for your 8 GB card. Pre-fitted J-lens exists on `neuronpedia/jacobian-lens` under `gemma-4-e4b`. This is the most direct comparator to Qwen3.5-4B in Phase 2.

- **12B Unified Gemma 4** (if you attempt it): runs in ~16 GB VRAM full precision or ~8 GB quantized — borderline on your card. Only attempt after E2B/E4B work. **NO pre-fitted J-lens for the 12B variant exists** in the neuronpedia inventory (only `gemma-4-31b`, `gemma-4-e2b`, `gemma-4-e4b` — all lowercase as lens directory names). If you want to include it, you must fit the lens yourself. Treat as stretch.

### 4.3 Phase 3 — Stretch (Liquid LFM2.5 family, only if time permits)

Liquid's own models. The specific "early LFM2.5-2.6B checkpoint" that had 10.2% loops was internal and is **NOT public**. Current public LFM2.5 models are likely already improved. Still test the public ones below and clearly document the observed loop rate. Do not invent model names.

- **`LiquidAI/LFM2.5-1.2B-Base`** — exists publicly. Fits in 8 GB easily at 4-bit (actually fits at fp16).

- **`LiquidAI/LFM2.5-1.2B-Instruct`** — exists publicly (405K downloads, the most downloaded LFM2.5 variant). Fits easily.

- **`LiquidAI/LFM2.5-1.2B-Thinking`** — exists publicly (Jan 16, 2026). The public "thinking" / reasoning model in the LFM2.5 family. **Most doom-loop-relevant Liquid model** — reasoning models loop more, and this is the public reasoning model. If you only do one Liquid model, do this one. (Note: this was not in your original list — add it.)

- **`LiquidAI/LFM2.5-8B-A1B`** — exists publicly (Base + Instruct + GGUF + MLX quantized variants). MoE with 8B total / 1B active params. **Use the GGUF quantized version via llama.cpp**, OR skip if MoE fitter causes issues. Elie's `open-jlens-data/code/moe/fit_moe.py` has a MoE-capable fitter (program.md experiment E6). MoE J-lens behavior is qualitatively different from dense — interpret results cautiously and document.

**Critical caveat for ALL Liquid models**: LFM2.5 is a **hybrid architecture** (attention + conv + MLP), not a pure transformer. The standard J-lens fitter assumes pure transformer. The conv layers will need either (a) to be skipped during J-lens computation (treat as frozen bypass), or (b) to have their own per-layer Jacobian computation. This is real engineering work — budget at least 2–3 days for the fitter adaptation before any analysis. NO pre-fitted J-lens exists for any Liquid model. If the fitter adaptation proves too difficult, document the attempt and drop Liquid from the project — the Qwen + Gemma results alone are sufficient for a publishable blog post.

### 4.4 Explicitly Excluded

- **`LiquidAI/LFM2.5-VL-1.6B`** (if it exists publicly — verify before referencing). Vision-language model. The Anthropic J-lens is text-only; applying it to a VL model requires significant adaptation (vision encoder, multimodal fusion, cross-modal attention). This is its own research project, not a side experiment. **Skip.**

- **`Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-2B`** — too small to reliably produce doom loops at meaningful rates; Liquid's reported rate was specifically for the 4B variant. Skip.

- **Any model >12B parameters.** Will not fit on RTX 5050 at usable precision. Skip.

- **`LFM2.5-2.6B` of any kind.** Does not exist publicly. The blog's "early checkpoint of LFM2.5-2.6B" was an internal Liquid checkpoint. Do not search further; do not invent.

Prefer base or lightly post-trained versions that still produce long reasoning traces, but always include the post-trained version when available — the doom-loop phenomenon is primarily observed in post-trained reasoning models, not in base models.

---

## 5. Foundation: Setup

Before any experiment:

1. `git clone https://github.com/eliebak/open-jlens-data` (primary codebase).
2. `git clone https://github.com/Liquid4All/antidoom` (loop detection + generation pipeline).
3. `pip install` per both repos' requirements. Use `uv` for the antidoom repo per their README.
4. Download `Qwen3.5-4B_jacobian_lens.pt` from `https://huggingface.co/neuronpedia/jacobian-lens` (path: `qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens.pt`, 387 MB). Place in `lenses/qwen3.5-4b.pt`.
5. Download `Qwen/Qwen3.5-4B` in 4-bit (bitsandbytes NF4). Cache locally to avoid re-download.
6. Sanity check: load the model, load the lens, run `lens(h_ℓ)` on a single prompt, verify the top-k readouts at the final layer match the model's actual top-k next-token predictions. If they don't match, the lens is misapplied (wrong layer mapping, wrong norm, wrong unembedding) — fix before proceeding.

---

## 6. Foundation: Trigger Token Extraction Protocol

Liquid published the top-5 trigger tokens ONLY for `LFM2.5-2.6B-early-ckpt` (internal, not public):

```
count    share  token
   2277  11.39%  ' the'
    902   4.51%  ' So'
    644   3.22%  'Alternatively'
    511   2.56%  'Wait'
    493   2.46%  ' But'
```

**For Qwen3.5-4B, you MUST generate the equivalent table yourself.** Nobody has published it.

### 6.1 Operational trigger set (use as initial seed for Exp 1)

From `src/antidoom/generate.py` lines 37–65, the `_RESTART_WORDS` set is the operational trigger list Antidoom uses internally — 26 words:

```python
_RESTART_WORDS = {
    "actually", "after", "also", "alternatively", "because", "but",
    "finally", "first", "given", "hmm", "however", "in", "let",
    "looking", "maybe", "now", "okay", "perhaps", "second", "since",
    "so", "the", "then", "therefore", "this", "thus", "wait",
}
```

Note this is much broader than the blog's published top-5. Use this as the initial trigger set for Experiment 1, then refine with the top-N tokens you extract below.

### 6.2 Qwen3.5-4B trigger token generation procedure

1. Use the Antidoom pipeline:
   ```bash
   uv run antidoom -c configs/default.yaml \
       --temp 0.01 \
       --model-name Qwen/Qwen3.5-4B \
       --quantization bnb-nf4
   ```
   on `LiquidAI/antidoom-mix-v1.0` (prompts-only dataset).

2. Inspect `iter_0_ftpo_pairs.jsonl` (output of the generation pipeline). For each sample where `find_inner_repetition` returned a hit, extract the **first token of the first repeat** — this is the trigger token.

3. Aggregate across all samples. Produce a table:
   ```
   count    share  token    token_id
   ```
   sorted by count descending. Save as `results/trigger_tokens_qwen3.5-4b.csv`.

4. This table is itself a publishable artifact. Include it in the final deliverables.

### 6.3 Loop detection criteria (use Liquid's exact code, verbatim)

From `src/antidoom/repetition.py`:

```python
def find_inner_repetition(
    text: str,
    *,
    min_repeats: int = 4,
    max_period: int = 1024,
    min_period: int = 1,
    min_total_repeated: int = 60,   # CHARACTERS, not tokens
    sample_len: int = 16,
    sample_interval: int = 128,
) -> tuple[bool, RepeatHit | None]:
```

- Detection runs on **decoded text characters**, not tokens.
- A loop = substring repeating ≥4 times with total span ≥60 characters.
- Period (loop unit length) between 1 and 1024 characters.
- Detector fingerprints 16-character substrings at every 128-character interval.

When mapping a detected loop boundary (character position) back to a token position for J-lens readouts, use `antidoom/tokens.py: TokenState.char_to_token_index`. **Do not write your own char-to-token mapper.** Liquid's handles edge cases (multi-byte chars, BPE merges across the boundary, leading space tokens).

---

## 7. Foundation: J-lens Loading Protocol

### 7.1 Loading the pre-fitted Qwen3.5-4B lens

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load model in 4-bit
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-4B",
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")

# Load the pre-fitted J-lens (kept in fp16/bf16 — NOT quantized)
lens = torch.load("lenses/qwen3.5-4b.pt", map_location="cuda", weights_only=True)
# lens is a tensor of shape [n_layers, d_model, d_model]
# Verify shape matches model.config.num_hidden_layers and model.config.hidden_size
assert lens.shape[0] == model.config.num_hidden_layers
assert lens.shape[1] == lens.shape[2] == model.config.hidden_size
```

### 7.2 Applying the lens to a residual-stream activation

```python
# W_U = unembedding matrix [vocab_size, d_model]
W_U = model.lm_head.weight  # for Qwen3.5; verify exact attribute name

# At layer ℓ, given residual activation h_ℓ [batch, seq, d_model]:
def jlens_readout(h_l, layer_idx, top_k=10):
    J_l = lens[layer_idx]                  # [d_model, d_model]
    projected = h_l @ J_l.T                # [batch, seq, d_model]
    # Apply final-norm equivalent (the lens was fit using the model's final norm)
    # Check the lens config.yaml for the exact norm convention used during fitting
    projected = model.model.norm(projected)  # verify this matches the lens fitting recipe
    logits = projected @ W_U.T              # [batch, seq, vocab_size]
    topk = logits.topk(top_k, dim=-1)
    return topk.indices, topk.values
```

**Sanity check**: at the final layer, `J_ℓ → I` (identity), so `jlens_readout(h_final, n_layers-1)` should match the model's actual next-token top-k predictions exactly. If they don't, your norm or unembedding is wrong — fix before any analysis.

### 7.3 Extracting J-lens directions for specific tokens

A J-lens direction for token `t` at layer `ℓ` is the row of `W_U · J_ℓ`:

```python
def jlens_direction(token_id, layer_idx):
    J_l = lens[layer_idx]                  # [d_model, d_model]
    W_U_t = W_U[token_id]                  # [d_model]
    direction = W_U_t @ J_l.T              # [d_model]
    return direction / direction.norm()    # unit vector
```

This is the steering vector for token `t` at layer `ℓ`. Adding `α · jlens_direction(t, ℓ)` to the residual stream at layer `ℓ` pushes the model toward eventually emitting token `t`.

### 7.4 Fitting a new lens (for Qwen3.5-4B-Base or any model without a pre-fit)

Use the recipe in `eliebak/open-jlens-data` (which vendors and improves on `anthropics/jacobian-lens`):

- Corpus: `Salesforce/wikitext`, `wikitext-103-raw-v1`, train split.
- 1000 prompts, `max_chars=2000`, `max_seq_len=128`, `skip_first=16` (attention sinks).
- `dim_batch=64`, dtype `bfloat16`, target = final layer.
- Convergence: `stop_at_delta=0.002` (mean relative change < 0.2% per new prompt).
- Estimated time: ~1–2 GPU-hours on H200. **Rent cloud GPU time for this — do not run on RTX 5050.**

For Liquid LFM2.5 (hybrid architecture), you will need to adapt the fitter. See `eliebak/open-jlens-data/code/moe/fit_moe.py` for an example of fitter adaptation (for MoE). The conv layers in LFM2.5 are a different challenge — they may need to be treated as frozen bypasses or have their own per-layer Jacobian computation. Budget 2–3 days of engineering for this.

---

## 8. Foundation: Workspace Band Identification Protocol

The mid-depth workspace band is **model-specific**. Do NOT assume boundaries. Compute them empirically for Qwen3.5-4B (36 layers, per Qwen3 architecture) using three complementary measurements:

### 8.1 Excess kurtosis of J-lens vectors per layer

For each layer `ℓ`, compute the J-lens vectors `V_ℓ = W_U · J_ℓ` (shape `[vocab_size, d_model]`). Compute excess kurtosis of the singular value distribution or of the entry distribution. Near zero in sensory layers, rises sharply at workspace onset, falls again in motor layers.

### 8.2 Next-token-prediction accuracy of `lens(h_ℓ)` per layer

Run the model on a held-out set of ~100 wikitext prompts. At each layer, apply the lens and compute top-1 / top-5 accuracy against the model's actual next-token predictions. Accuracy is near-chance in sensory layers, rises sharply through the workspace band, saturates near 1.0 in motor layers.

### 8.3 Effective dimensionality of the J-space per layer

Compute the singular values of `V_ℓ = W_U · J_ℓ`. Effective dimensionality = `exp(entropy(singular_value_distribution))` or the participation ratio `(Σλ)² / Σλ²`. Near zero in sensory layers, rises to ~25 (per Anthropic paper) in the workspace band.

### 8.4 Determining the band boundaries

Plot all three curves on the same x-axis (layer index 0 to N-1). The workspace band is the region where:
- Excess kurtosis is elevated (above the sensory baseline).
- Next-token accuracy is rising or plateaued above chance.
- Effective dimensionality is elevated (above ~5, say).

For Qwen3.5-4B (36 layers), expect roughly L10–L33, but **verify empirically**. Document the boundaries in `results/workspace_band_qwen3.5-4b.json` with the actual numbers.

Anthropic paper Figure 28 shows the recipe for these measurements.

---

## 9. Experiment 1 — Static Geometry of Trigger Tokens

### 9.1 Goal

Determine whether doom-loop trigger tokens have distinctive J-lens direction geometry in the workspace band, compared to control token sets.

### 9.2 Token sets

Define four token sets, each with ~30–50 tokens (match sizes across sets):

1. **Trigger set**: top-30 tokens from your Qwen3.5-4B trigger table (§6.2) PLUS the 26 `_RESTART_WORDS` (de-duplicated). Tokenize each carefully — note that `' the'` and `'the'` are different tokens, and `' So'` vs `'So'` are different tokens. Use the leading-space variants as they appear in actual generation.

2. **Frequency-matched content words**: 30 tokens with similar unigram frequency to the trigger set (use a wikitext frequency table), but that are content words (nouns, verbs) rather than discourse markers. Match within ±20% frequency.

3. **Other high-frequency discourse markers** that do NOT appear in the trigger set: 30 tokens like `'and'`, `'or'`, `'when'`, `'if'`, `'I'`, `'you'`, etc. — common but not loop-triggering.

4. **Random tokens**: 30 tokens sampled uniformly at random from the vocabulary, restricted to alphabetic tokens of length ≥3 (to avoid pathological single-char or punctuation tokens).

### 9.3 Procedure

For each token set, for each layer `ℓ ∈ [0, N-1]`:

1. Extract J-lens directions: `directions[t, ℓ] = jlens_direction(t, ℓ)` for each token `t` in the set.
2. Compute the **norm** of each direction (before unit-normalization) — this is the "strength" of the token's steering direction at that layer.
3. Compute the **pairwise cosine similarity matrix** within the set: `S[i, j] = cos(directions[t_i, ℓ], directions[t_j, ℓ])`.
4. Compute the **alignment with the workspace subspace**: project each direction onto the workspace subspace (defined as the top-k principal components of `V_ℓ` for workspace-band layers) and measure the fraction of variance captured.

### 9.4 Comparisons

For each layer, compare the four token sets on:

- Mean norm (do trigger tokens have abnormally strong steering directions?).
- Mean pairwise cosine similarity (do trigger tokens cluster together more than controls?).
- Workspace alignment (do trigger tokens live more inside the workspace subspace than controls?).
- Pairwise similarity to the model's "default" next-token distribution (do trigger tokens align with high-probability tokens under greedy sampling?).

### 9.5 Plots to produce

Save all to `results/exp1/`:

- `norm_by_layer.png` — line plot, x=layer, y=mean norm, four lines (one per token set), workspace band shaded.
- `pairwise_sim_by_layer.png` — same axes, y=mean pairwise cosine similarity.
- `workspace_alignment_by_layer.png` — same axes, y=fraction of variance in workspace subspace.
- `pca_workspace_band.png` — PCA of all token directions in the workspace band (use the middle layer of the band), colored by token set. Trigger tokens should cluster if the hypothesis is correct.
- `trigger_vs_controls_heatmap.png` — for the middle workspace layer, similarity matrix between trigger set and each control set.
- `trigger_token_table.csv` — the actual top-30 trigger tokens you extracted (§6.2).

### 9.6 Success criterion

Trigger tokens show measurably higher workspace alignment and/or higher pairwise clustering than controls, specifically in the workspace band (not in sensory or motor layers). If they don't, document the null result — it is still publishable as a negative finding.

---

## 10. Experiment 2 — Dynamic Monitoring

### 10.1 Goal

Determine whether the J-space state of the model changes measurably in the tokens immediately before loop onset, in a way that distinguishes looping from non-looping traces.

### 10.2 Prompt set

- All prompts from `LiquidAI/antidoom-mix-v1.0` (prompts-only dataset).
- PLUS 50 additional hard math prompts (e.g., GSM8K-hard, MATH-level problems).
- PLUS 50 additional hard coding prompts (e.g., HumanEval-hard, MBPP-level problems).

### 10.3 Generation

For each prompt:

1. Generate at temp=0.01 (effectively greedy) with `Qwen/Qwen3.5-4B` in 4-bit. Max length 2048 tokens.
2. During generation, apply the **tiered caching strategy from §2.1** — do NOT cache the full residual stream for every prompt:
   - **Tier 1 (always, all prompts)**: Hook the residual stream at every layer, compute top-10 J-lens readouts on-the-fly (via `jlens_readout(h_ℓ, ℓ, top_k=10)` from §7.2), discard the residual, keep only the indices + probabilities. Plus the generated token IDs and (later) the decoded text.
   - **Tier 2 (always, all prompts)**: Additionally, for the 3–5 workspace-band layers identified in §8, save the full residual stream `h_ℓ` to disk. This is needed for the cosine-similarity-to-trigger-direction analysis in §10.5.
   - **Tier 3 (stretch, ~5 prompts only)**: For a small handful of hand-picked looping traces (chosen after Tier 1+2 analysis reveals interesting cases), re-run with full residual caching at all layers — solely to produce the HTML visualization in §10.6.
   - Process in batches of 1 prompt at a time. After each prompt: `del` all tensors, `torch.cuda.empty_cache()`, `gc.collect()`.
3. Run `find_inner_repetition` on the decoded text. Record whether a loop was detected, and if so, the loop boundary (character position → token position via `TokenState.char_to_token_index`).
4. Save looping traces to `results/exp2/looping/` and non-looping to `results/exp2/nonlooping/`, with:
   - Tier 1: `readouts.pt` (top-10 indices + probs per layer per token), `meta.json` (loop boundary, trigger token, prompt, generated text).
   - Tier 2: `residuals_workspace_band.pt` (full residuals at the 3–5 workspace-band layers).
   - Tier 3: only for the ~5 selected traces — `residuals_full.pt` (all layers).

### 10.4 Analysis

For each looping trace:

1. Identify the **trigger position**: the token position of the first token of the first repeat (the loop onset).
2. From the Tier 1 cache, extract the top-10 J-lens readouts at the 5 token positions immediately BEFORE the trigger position, at each workspace-band layer.
3. Compute **workspace occupancy** at each of these 5 positions: how many J-lens directions have nonzero projection onto `h_ℓ` (above some threshold)? Use the Tier 2 cached residuals at workspace-band layers for this.
4. From the Tier 2 cache, compute cosine similarity between `h_ℓ` at position −1 and the J-lens direction of the trigger token (is the model "lining up" with the trigger direction?).

For each non-looping trace, match it to a looping trace on the same prompt (if possible) or on a similar prompt (same source, similar length). Snapshot the same 5 positions relative to the matched trigger position.

### 10.5 Comparisons

For looping vs non-looping traces, compare:

- Workspace occupancy at positions −5 to −1 before trigger (does occupancy collapse onto fewer directions in looping traces?). **Source: Tier 1 readouts.**
- Top-1 J-lens readout at positions −5 to −1 (does the readout converge to the trigger token earlier in looping traces?). **Source: Tier 1 readouts.**
- Cosine similarity between `h_ℓ` at position −1 and the J-lens direction of the trigger token (is the model "lining up" with the trigger direction?). **Source: Tier 2 residuals.**

### 10.6 Plots to produce

Save all to `results/exp2/`:

- `workspace_occupancy_pre_onset.png` — line plot, x=position relative to trigger (−5 to 0), y=mean workspace occupancy, two lines (looping vs non-looping), with confidence intervals.
- `top1_readout_convergence.png` — for looping traces, fraction of cases where the trigger token is in the top-1 / top-3 / top-10 J-lens readout at each pre-onset position. Compare to non-looping traces (where "trigger" is the actual next token in the trace).
- `trigger_alignment_pre_onset.png` — cosine similarity between `h_ℓ` at position −1 and trigger-token J-lens direction, looping vs non-looping.
- `example_looping_trace.html` — annotated visualization of one looping trace: token-by-token, with J-lens top-5 readouts at each workspace layer, loop boundary marked. **This is the only plot that needs Tier 3 data** — re-run generation with full residual caching for ~5 hand-picked traces after the Tier 1+2 analysis identifies the most interesting ones.

### 10.7 Success criterion

Looping traces show measurably different J-space state in the 5 tokens before onset compared to non-looping traces — either collapsed occupancy, premature readout convergence, or elevated trigger-direction alignment. If no difference is detectable, document the null result.

---

## 11. Experiment 3 — Causal Interventions

### 11.1 Goal

Test whether ablating the J-lens directions of trigger tokens in the workspace band reduces the doom-loop rate, with appropriate controls. **This is the experiment that decides whether the project is a blog post or an ICML paper.**

### 11.2 Intervention

At the critical position identified in Exp 2 (the trigger position, or 1–2 tokens before — test both), project out the trigger token's J-lens direction from the residual stream at workspace-band layers:

```python
def ablate_trigger_direction(h_l, trigger_token_id, layer_idx):
    v = jlens_direction(trigger_token_id, layer_idx)  # unit vector
    projection = (h_l @ v).unsqueeze(-1) * v         # [batch, seq, d_model]
    return h_l - projection                           # remove the trigger direction
```

Apply this intervention at every workspace-band layer (or test layer-by-layer to find the most effective band).

### 11.3 Conditions

For each prompt in the Exp 2 prompt set (or a subset of 200 prompts if compute is constrained), generate under five conditions:

1. **No intervention** (baseline) — measure baseline loop rate.
2. **Ablate trigger direction** — at the trigger position, project out the trigger token's J-lens direction in workspace-band layers. Measure loop rate.
3. **Ablate random direction (norm-matched)** — project out a random unit vector. Controls for the generic effect of removing any direction.
4. **Ablate control-token direction** — project out the J-lens direction of a frequency-matched control token (not a trigger token). Controls for the effect of removing any *verbalizable* direction.
5. **Ablate trigger direction in sensory band** — project out the trigger direction, but in sensory layers (first third) instead of workspace band. Should do nothing if the hypothesis is correct.
6. **Ablate trigger direction in motor band** — project out the trigger direction in motor layers (last few). Should also do nothing or break the model (different failure mode).

### 11.4 Measurement

For each condition, measure:

- **Loop rate**: fraction of prompts where `find_inner_repetition` detects a loop.
- **Eval quality**: on a small held-out eval set (use MMLU-small or GSM8K-small — 200 examples), measure accuracy. The intervention should not significantly degrade eval quality.
- **Generation length**: mean / median tokens generated before stopping (intervention may make the model stop earlier or later).

### 11.5 Plots to produce

Save all to `results/exp3/`:

- `loop_rate_by_condition.png` — bar chart, x=condition (1–6), y=loop rate, with confidence intervals.
- `loop_rate_by_layer_band.png` — for the trigger-ablation condition, loop rate as a function of which layer band is ablated (sensory / workspace / motor).
- `eval_quality_by_condition.png` — bar chart, x=condition, y=eval accuracy. Should be flat across conditions (intervention is surgical).
- `loop_rate_vs_ablation_strength.png` — for the trigger-ablation condition, vary the number of trigger directions ablated (top-1, top-3, top-5, top-10) and plot loop rate. Is the effect dose-dependent?
- `per_prompt_loop_rate_change.csv` — for each prompt, the loop status under each condition. Allows paired statistical tests.

### 11.6 Success criterion

The success criterion is **NOT** "loop rate goes down." Both positive and negative results are publishable:

- **Positive result (blog + paper)**: condition 2 (ablate trigger) shows significantly lower loop rate than conditions 3, 4, 5, 6. Effect is dose-dependent. Eval quality is preserved. This is a clean causal story: ablating trigger-token J-lens directions in the workspace band prevents doom loops, and the effect is specific to the workspace band and to trigger tokens.

- **Negative result (blog only)**: condition 2 does not differ from conditions 3, 4. This means the J-lens geometry of trigger tokens is *descriptive but not causal* — the workspace reflects the loop-attractor but is not the mechanism. Still publishable as a clean negative finding, especially if it engages seriously with Lazaridis et al.'s "knowledge-precision problem" framing.

- **Mixed result**: condition 2 reduces loop rate but condition 5 (sensory ablation) also does. This means the effect is not workspace-specific — interesting but harder to publish.

### 11.7 Statistical analysis

Use paired statistical tests (McNemar's test for binary loop/no-loop per prompt across conditions). Report effect sizes (Cohen's h for proportions). Pre-register the analysis plan in `results/exp3/analysis_plan.md` before running the experiment.

---

## 12. Cross-Model Replication Protocol (Phase 2)

After Phase 1 succeeds (or fails cleanly), repeat Exp 1 + Exp 3 on:

- `Qwen/Qwen3.5-4B-Base` (requires fitting a new J-lens — see §7.4).
- `google/gemma-4-E2B` + `google/gemma-4-E2B-it` (pre-fitted lens available).
- `google/gemma-4-E4B` + `google/gemma-4-E4B-it` (pre-fitted lens available).

For each model:

1. Compute the workspace band empirically (§8). Boundaries will differ from Qwen3.5-4B.
2. Generate the model's own trigger-token table by running the Antidoom pipeline (§6.2). The table will differ from Qwen3.5-4B's.
3. Run Exp 1 (static geometry) with the model's own trigger tokens.
4. Run Exp 3 (causal ablation) with the model's own trigger tokens.

### 12.1 Cross-model comparison plots

Save to `results/cross_model/`:

- `workspace_band_comparison.png` — for each model, the workspace band boundaries (as % of depth) overlaid on the excess-kurtosis / accuracy / dimensionality curves.
- `trigger_token_workspace_location.png` — for each model, where in the workspace band do trigger tokens cluster? Is the relative depth consistent across models?
- `ablation_effect_comparison.png` — for each model, the loop-rate reduction from trigger-direction ablation. Is the effect size consistent across models?

### 12.2 Cross-model questions

- Do trigger tokens live at the same relative depth across models? (Elie's CKA framework supports this question directly.)
- Does post-training (Qwen3.5-4B-Base vs Qwen3.5-4B) install or amplify trigger-token directions in the workspace?
- Does the ablation effect generalize across model families (Qwen vs Gemma)?

---

## 13. Deliverables

- `README.md` — setup instructions, hardware requirements, how to reproduce.
- `code/` — clean codebase with modules: `loading.py` (model + lens loading), `detection.py` (loop detection wrappers), `geometry.py` (J-lens direction extraction), `intervention.py` (ablation), `analysis.py` (plotting + stats).
- `results/` folder containing:
  - `trigger_tokens_qwen3.5-4b.csv` — the table you generate (publishable artifact).
  - `workspace_band_qwen3.5-4b.json` — empirically computed band boundaries.
  - `exp1/` — all static-geometry plots + CSVs.
  - `exp2/` — all dynamic-monitoring plots + CSVs + cached activations (or links to them).
  - `exp3/` — all causal-intervention plots + CSVs + per-prompt data.
  - `cross_model/` — Phase 2 cross-model plots (after Phase 2).
  - `status/` — status notes after each experiment (1–2 paragraphs each).
  - `RESULTS_SUMMARY.md` — incremental summary, updated after each experiment.
- Everything must be reproducible on the stated hardware (Windows + RTX 5050 8GB + 16GB RAM) for Phase 1. Phase 2 base-model lens fitting may require rented cloud GPU time — document this clearly.

---

## 14. Citation & Prior Art

You MUST cite and engage with:

1. **Liquid AI Antidoom blog** (Jul 7, 2026) — `https://www.liquid.ai/blog/antidoom`. Source of trigger tokens and loop detection.
2. **Anthropic J-lens paper** (`arXiv:2607.15495`, Jul 6, 2026) — J-lens method, J-space, workspace band.
3. **Lazaridis et al.** (`arXiv:2606.13705`, Jun 9, 2026) — per-neuron attribution of doom loops in Gemma 4. **Your closest competitor.** Differentiate explicitly.
4. **Silent Alarm / JADR** (`arXiv:2607.12792`, Jul 14, 2026) — J-space applied to jailbreak safety.
5. **Elie Bakouch's open-jlens-data** (Jul 17–19, 2026) — open-model lens inventory, CKA methodology.
6. **Antislop paper** (`arXiv:2510.15061`, Oct 2025) — Liquid's ref [3], basis for Antidoom.
7. **Wait, Wait, Wait** (`arXiv:2512.12895`, Dec 2025) — behavioral theory of looping.
8. **Circular Reasoning** (`arXiv:2601.05693`, Jan 2026) — V-shaped attention pattern, semantic repetition precedes textual.
9. **Word Salad Chopper** (`arXiv:2511.00536`, Nov 2025) — loop detection via hidden states.

For each, include a 1-paragraph note in `RESULTS_SUMMARY.md` explaining how your work relates (complement, differentiate, build on, etc.).

---

## 15. Key Innovation Reminder (for your own understanding)

Liquid AI identified bad *output tokens* and changed their preference via FTPO training (a DPO variant that operates only on the trailing token's logits).

We are looking *inside* the model with J-lens to understand the internal workspace state that makes those tokens attractive, and testing whether intervening on the internal directions themselves can affect the failure mode — **without training**.

This is mechanistic, not behavioral. The hypothesis is that trigger tokens have over-represented J-lens directions in the workspace band — the model's "verbalizable workspace" is biased toward emitting them, especially under greedy sampling and reinforced prior context. Antidoom changes the bias by training; we show the bias has a specific geometric signature in workspace coordinates, and (if Exp 3 succeeds) that we can remove the bias without training by intervening on the geometry itself.

If Exp 3 works, we have a cheap inference-time antidoom that doesn't require LoRA training — useful for production. If it doesn't work, we have a clean negative result showing the workspace geometry is descriptive but not causal — still publishable as a workshop paper or a blog post, and it would engage seriously with Lazaridis et al.'s "knowledge-precision problem" framing.

---

## 16. ICML 2027 Note (informational, not for execution)

ICML 2027's CFP is not yet posted but historically the submission window is late January. ICML 2026 was Jan 23 (abstract) / Jan 28 (full paper). The January 2027 timeline is realistic for this project IF:

- Phase 1 + Phase 2 complete by end of October 2026.
- November 2026: cross-model analysis, writing.
- December 2026: revision, supplementary experiments.
- January 2027: submission.

Pure descriptive analysis (Exp 1 + Exp 2 only) = workshop paper or blog, not ICML. Exp 3 (causal ablation) is what makes it ICML-grade. The closest competitor (Lazaridis et al.) used per-neuron attribution; you would differentiate by showing J-lens finds something per-neuron attribution cannot — candidates: cross-model workspace-band depth alignment of trigger tokens, base-vs-posttrained comparison, or the verbalizable structure (trigger tokens are exactly the ones the workspace represents as steering directions).

Training a J-space-regularized model to reduce loops would be a stronger ICML paper but is risky on 8 GB VRAM and is a 3–6 month project by itself. **Do not attempt for the initial experiment.** If Exp 3 works and you want to extend toward ICML, LoRA fine-tuning with a J-lens-direction regularization loss is the right design — fits in your VRAM via 4-bit + LoRA rank 128–256 (exactly Liquid's recipe).

For now: **blog post + Twitter first.** The J-lens field is forming right now (Anthropic paper Jul 6, Elie's replications Jul 17–19, Silent Alarm Jul 14). Being visibly early with a clean result buys you the network effects that make the ICML submission stronger.
