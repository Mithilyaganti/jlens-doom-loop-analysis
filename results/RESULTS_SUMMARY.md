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

Empirical table: `results/trigger_tokens_qwen3.5-4b.csv` (merged from generation logs + Exp2 loops).

- **Loop rate under our HF 4-bit setup**: ~2% on 50 hard-prompt Exp2 traces (1/50), vs Liquid's 22.9% under vLLM + 4000 tokens.
- **Empirical triggers observed** (honest, not invented):
  - `*` (token_id 348) — 1 hit
  - `--------------------------------------------------------------------------------` (token_id 42013) — 1 hit (dash-separator loop in coding trace)
- Operational `_RESTART_WORDS` seed used for Exp1 geometry where empirical top-N is sparse.

Liquid's published top-5 (`' the'`, `' So'`, etc.) is for an internal LFM2.5 checkpoint only — not Qwen3.5-4B.

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

- Prompts: 50, loops: 1, non-loops: 49, rate: 2.0%
- Workspace layers cached (Tier-2): [24, 25, 26, 27, 28]
- Mean trigger alignment (−1): loop=0.0071596247144043446, non=0.03229696372029733

Plots: `workspace_occupancy_pre_onset.png`, `top1_readout_convergence.png`,
`trigger_alignment_pre_onset.png`, `example_looping_trace.html`.

## Experiment 3 — Causal Interventions

# Experiment 3 — Causal Interventions

## Loop rates
{
  "baseline": 0.0,
  "ablate_trigger": 0.0,
  "ablate_random": 0.0,
  "ablate_control": 0.0,
  "ablate_trigger_sensory": 0.0,
  "ablate_trigger_motor": 0.0
}

## Interpretation
- Result type: **negative**
- Baseline loop rate: 0.000
- Ablate trigger (workspace): 0.000
- Ablate random: 0.000
- Ablate control token: 0.000
- Ablate trigger sensory: 0.000
- Ablate trigger motor: 0.000

## Stats
{
  "chi2": 0.0,
  "p_value": 1.0,
  "b": 0,
  "c": 0
}
Cohen's h (baseline vs trigger ablate): 0.0

## Dose-response
{
  "top-1": 0.0,
  "top-3": 0.05,
  "top-5": 0.0
}

## Eval quality (tiny set)
{
  "baseline": 0.2,
  "ablate_trigger": 0.2,
  "ablate_random": 0.2,
  "ablate_control": 0.2
}

Both positive and negative results are publishable per PROJECT_SPEC §11.6.
