# Results Summary — J-Space Analysis of Doom Loops

Phase 1 model: `Qwen/Qwen3.5-4B` (4-bit NF4). Hardware: Windows + RTX 5050 8GB.

This file is updated incrementally after each pipeline stage.

## Relation to prior art

### Liquid AI Antidoom (Jul 2026)
Behavioral / training fix: identify the first token of the first character-level repeat and train FTPO against it. We reuse their loop detector and prompt mix, but ask a different question: what is the **internal J-space geometry** of those trigger tokens, and can we ablate it at inference time without training?

### Anthropic J-lens (arXiv:2607.15495, Jul 2026)
Methodological foundation: Jacobian lens, J-space, mid-depth workspace band, transport + unembed readout. We apply their open-model lens inventory (via neuronpedia / open-jlens-data) to doom loops.

### Lazaridis et al. (arXiv:2606.13705, Jun 2026)
Closest competitor: per-neuron attribution / single-neuron edits for repetition loops. Differentiation: J-lens gives **vocabulary-grounded steering directions** (what concept is over-represented in the verbalizable workspace) rather than only *where* a neuron lives. We also test workspace-band specificity vs sensory/motor controls.

### Silent Alarm / JADR (arXiv:2607.12792, Jul 2026)
Independent evidence that J-space is mature enough for downstream behavioral analysis (jailbreak recognition), including under INT4 — relevant to our 4-bit setup.

### Elie Bakouch open-jlens-data (Jul 2026)
Open-model lens inventory, fit recipes, CKA tools. We vendor `jacobian-lens` from this repo and use pre-fitted Qwen3.5-4B lens rather than refitting.

### Wait, Wait, Wait (arXiv:2512.12895) & Circular Reasoning (arXiv:2601.05693)
Behavioral theory of why reasoning models loop (overtrained discourse markers, V-shaped attention, semantic before textual repetition). Motivates studying discourse-marker trigger tokens in workspace coordinates.

### Word Salad Chopper (arXiv:2511.00536)
Post-hoc loop detection via hidden states / linear classifier — related but not causal localization of trigger directions.

### Antislop (arXiv:2510.15061)
Liquid's precursor framework for repetitive patterns; Antidoom adapts FTPO from this line of work.

## Workspace Band (Qwen3.5-4B)

# Workspace Band — Qwen3.5-4B

Empirically identified workspace band:

- **Start layer**: 23
- **End layer**: 30
- **Mid layer**: 27
- **Key layers (Tier-2 cache)**: [23, 24, 26, 28, 30]
- Sensory layers: 23 layers before band
- Motor layers: 0 layers after band
- NTP prompts used: 40

Metrics and plots: `results/workspace_band_*.png`, `results/workspace_band_qwen3.5-4b.json`.

## Workspace Band (Qwen3.5-4B)

# Workspace Band — Qwen3.5-4B

Empirically identified workspace band:

- **Start layer**: 24
- **End layer**: 28
- **Mid layer**: 26
- **Key layers (Tier-2 cache)**: [24, 25, 26, 27, 28]
- Sensory layers: 24 layers before band
- Motor layers: 2 layers after band
- NTP prompts used: 40

Metrics and plots: `results/workspace_band_*.png`, `results/workspace_band_qwen3.5-4b.json`.

## Trigger Tokens (Qwen3.5-4B)

# Trigger Token Extraction — Qwen3.5-4B

- Prompts attempted: 28
- Loops detected: 0 (0.0%)
- Unique trigger token IDs: 0
- Max new tokens: 2048, temperature: 0.01
- Quantization: bitsandbytes NF4, backend: HF transformers generate
- Table: results/trigger_tokens_qwen3.5-4b.csv (empirical only; empty if zero loops)
- Generation log: results/trigger_gen_log.jsonl

## Honest finding
Under this setup, **zero** generations met Liquid find_inner_repetition criteria
(min_repeats=4, min_total_repeated=60 chars). The empirical trigger CSV is therefore empty.
We do **not** invent frequencies. Exp1 uses Liquid operational _RESTART_WORDS seed as labeled
non-empirical seed where needed.

Liquid blog reported 22.9% loop rate for Qwen3.5-4B under their Antidoom vLLM pipeline
(max_new_tokens up to 4000). Possible drivers of the gap: generation backend, max length,
prompt sampling, chat-template details, model revision. Our measurement is the rate under
the conditions we actually ran.

Top-10 triggers:
`
(empty — no loops observed)
`

## Experiment 1 — Static Geometry

# Experiment 1 — Static Geometry

Compared trigger tokens vs frequency-matched content, discourse controls, and random tokens
across all layers.

**Workspace-band means (L24–L28):**
- Pairwise sim: trigger=0.1586 vs controls=0.1336
- Workspace align: trigger=0.0526 vs controls=0.0545
- Success criterion (higher clustering and/or alignment): **True**

Plots in `results/exp1/`. Mid-layer PCA at L26.

## Experiment 2 — Dynamic Monitoring

# Experiment 2 — Dynamic Monitoring

- Prompts: 0, loops: 1, non-loops: 49, rate: 2.0%
- Workspace layers cached (Tier-2): [24, 25, 26, 27, 28]
- Mean trigger alignment (−1): loop=0.0071596247144043446, non=0.03229696372029733

Plots: `workspace_occupancy_pre_onset.png`, `top1_readout_convergence.png`,
`trigger_alignment_pre_onset.png`, `example_looping_trace.html`.
