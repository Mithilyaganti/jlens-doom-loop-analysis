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

**Important honest caveat:** Exp3 causal test used 20 hard-preferred antidoom-mix prompts with 0% baseline loop rate under our HF 4-bit setup (1536 max tokens). With zero loops in any condition, McNemar and Cohen's h are undefined for effect detection — this is a **null/negative** result: we cannot claim workspace-band ablation prevents loops when loops did not occur in this prompt set. Exp2 (50 prompts + hard extras) did observe 1 loop (2%). Dose-response on subset: top-3 ablation showed 5% loop rate (1/20) vs 0% baseline — exploratory only.

## Phase 1 Completion Status

- **Model**: `Qwen/Qwen3.5-4B` (bitsandbytes NF4, fits 8GB VRAM)
- **Lens**: pre-fitted `neuronpedia/jacobian-lens` → `lenses/qwen3.5-4b.pt`
- **Workspace band**: L24–L28 (`results/workspace_band_qwen3.5-4b.json`)
- **Exp 1**: complete — `results/exp1/`
- **Exp 2**: complete — 50 traces, 1 loop, `results/exp2/`
- **Exp 3**: complete — 120/120 checkpoint pairs, all plots + `stats.json`
- **Verification**: `scripts/verify_deliverables.py` — GATE PASSED
- **Phase 2/3**: not run (per spec — only after Phase 1 validates)

## Workspace Band (LFM2-2.6B)

# Workspace Band — LFM2-2.6B

Empirically identified workspace band:

- **Model**: `LiquidAI/LFM2-2.6B`
- **Start layer**: 21
- **End layer**: 21
- **Mid layer**: 21
- **Key layers (Tier-2 cache)**: [21]
- Sensory layers: 5 layers before band
- Motor layers: 2 layers after band
- NTP prompts used: 40

Metrics and plots: `results/workspace_band_*.png`, `C:\Users\mithi\Desktop\mithil\projects\j-lens\results\workspace_band_lfm2-2.6b.json`.

## Experiment 1 — Static Geometry

# Experiment 1 — Static Geometry

Compared trigger tokens vs frequency-matched content, discourse controls, and random tokens
across all layers.

**Workspace-band means (L21–L21):**
- Pairwise sim: trigger=0.0905 vs controls=0.0878
- Workspace align: trigger=0.2477 vs controls=0.1942
- Success criterion (higher clustering and/or alignment): **True**

Plots in `results/exp1/`. Mid-layer PCA at L21.

## Experiment 2 — Dynamic Monitoring

# Experiment 2 — Dynamic Monitoring

- Prompts: 200, loops: 5, non-loops: 195, rate: 2.5%
- Workspace layers cached (Tier-2): [21]
- Mean trigger alignment (−1): loop=0.07942445576190948, non=0.13036531308749452

Plots: `workspace_occupancy_pre_onset.png`, `top1_readout_convergence.png`,
`trigger_alignment_pre_onset.png`, `example_looping_trace.html`.

## Experiment 2 — Dynamic Monitoring

# Experiment 2 — Dynamic Monitoring

- Prompts: 200, loops: 5, non-loops: 195, rate: 2.5%
- Workspace layers cached (Tier-2): [21]
- Mean trigger alignment (−1): loop=0.17168146148324012, non=0.13036531308749452

Plots: `workspace_occupancy_pre_onset.png`, `top1_readout_convergence.png`,
`trigger_alignment_pre_onset.png`, `example_looping_trace.html`.

## Experiment 3 — Causal Interventions

# Experiment 3 — Causal Interventions

## Loop rates
{
  "baseline": 0.18181818181818182,
  "ablate_trigger": 0.18181818181818182,
  "ablate_random": 0.09090909090909091,
  "ablate_control": 0.2727272727272727,
  "ablate_trigger_sensory": 0.18181818181818182,
  "ablate_trigger_motor": 0.09090909090909091
}

## Interpretation
- Result type: **mixed**
- Baseline loop rate: 0.182
- Ablate trigger (workspace): 0.182
- Ablate random: 0.091
- Ablate control token: 0.273
- Ablate trigger sensory: 0.182
- Ablate trigger motor: 0.091

## Stats
{
  "chi2": 0.25,
  "p_value": 0.6170750774519739,
  "b": 2,
  "c": 2
}
Cohen's h (baseline vs trigger ablate): 0.0

## Dose-response
{
  "top-1": 0.2727272727272727,
  "top-3": 0.18181818181818182,
  "top-5": 0.36363636363636365
}

## Eval quality (tiny set)
{
  "baseline": 1.0,
  "ablate_trigger": 1.0,
  "ablate_random": 1.0,
  "ablate_control": 1.0
}

Both positive and negative results are publishable per PROJECT_SPEC §11.6.

## Experiment 2 — Dynamic Monitoring

# Experiment 2 — Dynamic Monitoring

- Prompts: 200, loops: 5, non-loops: 195, rate: 2.5%
- Workspace layers cached (Tier-2): [21]
- Mean trigger alignment (−1): loop=0.17168146148324012, non=0.13036531308749452

**Matched-control headline metrics** (bootstrap 95% CIs + permutation p-values in
`findings.json` → `matched_control_analysis`):
- occupancy_at_m2: loop 0.249 [n=5] vs same-token 0.568 [n=196] (p=0.0153), pos-matched 0.542 [n=890] (p=0.038), legacy 0.577 [n=195] (p=0.0226)
- top1_trigger_at_m1: loop 0.600 [n=5] vs same-token 0.520 [n=196] (p=1), pos-matched 0.228 [n=890] (p=0.08759), legacy 0.041 [n=195] (p=0.0013)
- align_trigger_dir_at_m1: loop 0.172 [n=5] vs same-token 0.176 [n=196] (p=0.8986), pos-matched 0.131 [n=890] (p=0.1637), legacy 0.130 [n=195] (p=0.2142)

Plots: `workspace_occupancy_pre_onset.png`, `top1_readout_convergence.png`,
`trigger_alignment_pre_onset.png`, `matched_controls_headline.png`,
`example_looping_trace.html`.

## Experiment 1 — Static Geometry

# Experiment 1 — Static Geometry

Compared trigger tokens vs frequency-matched content, discourse controls, and random tokens
across all layers.

**Workspace-band means (L21–L21):**
- Pairwise sim: trigger=0.0905 vs controls=0.0878 (min perm p=0.0002)
- Workspace align: trigger=0.2477 vs controls=0.1942 (min perm p=0.0002)
- Success criterion (higher clustering and/or alignment): **True**

**Significance statements**
- Pairwise-sim criterion met: trigger 0.0905 [0.0729, 0.1091] > controls [0.061888810417925315, 0.15700757627685866, 0.04463245685522755] with min permutation p=0.0002 (n=30 trigger-token observations pooled over band layers [21])
- Workspace-alignment criterion met: trigger 0.2477 [0.2107, 0.2876] > controls [0.17548099532723427, 0.25838261023163794, 0.1487494890888532] with min permutation p=0.0002 (n=30 trigger-token observations pooled over band layers [21])

Bootstrap 95% CIs over tokens and trigger-vs-control permutation tests per layer are in
`geometry_by_layer.csv` (`*_ci_lo/hi`, `perm_p_*` columns); pooled band statistics under
`workspace_band_stats` in `findings.json`.

Plots in `results/exp1/`. Mid-layer PCA at L21.

## Experiment 2 — Dynamic Monitoring

# Experiment 2 — Dynamic Monitoring

- Prompts: 200, loops: 5, non-loops: 195, rate: 2.5%
- Workspace layers cached (Tier-2): [21]
- Mean trigger alignment (−1): loop=0.17168146148324012, non=0.13036531308749452

**Matched-control headline metrics** (bootstrap 95% CIs + permutation p-values in
`findings.json` → `matched_control_analysis`):
- occupancy_at_m2: loop 0.249 [n=5] vs same-token 0.568 [n=196] (p=0.0153), pos-matched 0.542 [n=890] (p=0.038), legacy 0.577 [n=195] (p=0.0226)
- top1_trigger_at_m1: loop 0.600 [n=5] vs same-token 0.520 [n=196] (p=1), pos-matched 0.228 [n=890] (p=0.08759), legacy 0.041 [n=195] (p=0.0013)
- align_trigger_dir_at_m1: loop 0.172 [n=5] vs same-token 0.176 [n=196] (p=0.8986), pos-matched 0.131 [n=890] (p=0.1637), legacy 0.130 [n=195] (p=0.2142)

Plots: `workspace_occupancy_pre_onset.png`, `top1_readout_convergence.png`,
`trigger_alignment_pre_onset.png`, `matched_controls_headline.png`,
`example_looping_trace.html`.

## Experiment 2 — Dynamic Monitoring

# Experiment 2 — Dynamic Monitoring

- Prompts: 200, loops: 5, non-loops: 195, rate: 2.5%
- Workspace layers cached (Tier-2): [21]
- Mean trigger alignment (−1): loop=0.17168146148324012, non=0.13036531308749452

**Matched-control headline metrics** (bootstrap 95% CIs + permutation p-values in
`findings.json` → `matched_control_analysis`):
- occupancy_at_m2: loop 0.249 [n=5] vs same-token 0.568 [n=196] (p=0.0153), pos-matched 0.542 [n=890] (p=0.038), legacy 0.577 [n=195] (p=0.0226)
- top1_trigger_at_m1: loop 0.600 [n=5] vs same-token 0.520 [n=196] (p=1), pos-matched 0.228 [n=890] (p=0.08759), legacy 0.041 [n=195] (p=0.0013)
- align_trigger_dir_at_m1: loop 0.172 [n=5] vs same-token 0.176 [n=196] (p=0.8986), pos-matched 0.131 [n=890] (p=0.1637), legacy 0.130 [n=195] (p=0.2142)

Plots: `workspace_occupancy_pre_onset.png`, `top1_readout_convergence.png`,
`trigger_alignment_pre_onset.png`, `matched_controls_headline.png`,
`example_looping_trace.html`.

## Experiment 3 — Causal Interventions (fixed protocol v2)

# Experiment 3 — Causal Interventions (fixed protocol v2)

No looping prompts in baseline pass; Exp3 skipped.

## Experiment 3 — Causal Interventions (fixed protocol v2)

# Experiment 3 — Causal Interventions (fixed protocol v2)

## Design
Two-pass, per-trace onset-timed interventions. 1 looping prompts;
0 re-looped in pass 1, 1 stable_nonloop.
Onset sources: {"global_fallback": 1}

## Loop rates (with Wilson 95% CI)
{
  "baseline_rerun": [
    0.0,
    [
      0.0,
      0.7935
    ]
  ],
  "ablate_trigger_ws": [
    0.0,
    [
      0.0,
      0.7935
    ]
  ],
  "ablate_random": [
    0.0,
    [
      0.0,
      0.7935
    ]
  ],
  "ablate_control_token": [
    0.0,
    [
      0.0,
      0.7935
    ]
  ],
  "ablate_trigger_sensory": [
    0.0,
    [
      0.0,
      0.7935
    ]
  ],
  "ablate_trigger_motor": [
    0.0,
    [
      0.0,
      0.7935
    ]
  ],
  "logit_suppress_trigger": [
    0.0,
    [
      0.0,
      0.7935
    ]
  ]
}

## Contrasts vs baseline_rerun (exact McNemar)
{
  "ablate_trigger_ws": {
    "n_pairs": 1,
    "b_baseline_only": 0,
    "c_condition_only": 0,
    "p_exact": NaN,
    "odds_ratio_b_over_c": NaN,
    "rate_baseline": 0.0,
    "rate_condition": 0.0,
    "wilson_ci_baseline": [
      0.0,
      0.7934506856227626
    ],
    "wilson_ci_condition": [
      0.0,
      0.7934506856227626
    ],
    "cohens_h": 0.0
  },
  "ablate_random": {
    "n_pairs": 1,
    "b_baseline_only": 0,
    "c_condition_only": 0,
    "p_exact": NaN,
    "odds_ratio_b_over_c": NaN,
    "rate_baseline": 0.0,
    "rate_condition": 0.0,
    "wilson_ci_baseline": [
      0.0,
      0.7934506856227626
    ],
    "wilson_ci_condition": [
      0.0,
      0.7934506856227626
    ],
    "cohens_h": 0.0
  },
  "ablate_control_token": {
    "n_pairs": 1,
    "b_baseline_only": 0,
    "c_condition_only": 0,
    "p_exact": NaN,
    "odds_ratio_b_over_c": NaN,
    "rate_baseline": 0.0,
    "rate_condition": 0.0,
    "wilson_ci_baseline": [
      0.0,
      0.7934506856227626
    ],
    "wilson_ci_condition": [
      0.0,
      0.7934506856227626
    ],
    "cohens_h": 0.0
  },
  "ablate_trigger_sensory": {
    "n_pairs": 1,
    "b_baseline_only": 0,
    "c_condition_only": 0,
    "p_exact": NaN,
    "odds_ratio_b_over_c": NaN,
    "rate_baseline": 0.0,
    "rate_condition": 0.0,
    "wilson_ci_baseline": [
      0.0,
      0.7934506856227626
    ],
    "wilson_ci_condition": [
      0.0,
      0.7934506856227626
    ],
    "cohens_h": 0.0
  },
  "ablate_trigger_motor": {
    "n_pairs": 1,
    "b_baseline_only": 0,
    "c_condition_only": 0,
    "p_exact": NaN,
    "odds_ratio_b_over_c": NaN,
    "rate_baseline": 0.0,
    "rate_condition": 0.0,
    "wilson_ci_baseline": [
      0.0,
      0.7934506856227626
    ],
    "wilson_ci_condition": [
      0.0,
      0.7934506856227626
    ],
    "cohens_h": 0.0
  },
  "logit_suppress_trigger": {
    "n_pairs": 1,
    "b_baseline_only": 0,
    "c_condition_only": 0,
    "p_exact": NaN,
    "odds_ratio_b_over_c": NaN,
    "rate_baseline": 0.0,
    "rate_condition": 0.0,
    "wilson_ci_baseline": [
      0.0,
      0.7934506856227626
    ],
    "wilson_ci_condition": [
      0.0,
      0.7934506856227626
    ],
    "cohens_h": 0.0
  }
}

## Mean generation lengths (bootstrap 95% CI)
{
  "baseline_rerun": [
    256.0,
    [
      256.0,
      256.0
    ]
  ],
  "ablate_trigger_ws": [
    256.0,
    [
      256.0,
      256.0
    ]
  ],
  "ablate_random": [
    256.0,
    [
      256.0,
      256.0
    ]
  ],
  "ablate_control_token": [
    256.0,
    [
      256.0,
      256.0
    ]
  ],
  "ablate_trigger_sensory": [
    256.0,
    [
      256.0,
      256.0
    ]
  ],
  "ablate_trigger_motor": [
    256.0,
    [
      256.0,
      256.0
    ]
  ],
  "logit_suppress_trigger": [
    256.0,
    [
      256.0,
      256.0
    ]
  ]
}

## Timing sweep (subset n=0)
{
  "ablate_trigger_ws@-3": null,
  "ablate_trigger_ws@-1": null
}

## Dose-response (subset n=0)
{
  "dose_top-1": null,
  "dose_top-3": null,
  "dose_top-5": null
}

## GSM8K surgical eval
{
  "n_problems": 2,
  "start_pos": 50,
  "trigger_token_id": 4703,
  "max_new_tokens": 512,
  "source": "HF openai/gsm8k test split, seed=42, data/gsm8k_eval.json",
  "accuracy": {
    "baseline_rerun": 0.5,
    "ablate_trigger_ws": 0.5,
    "logit_suppress_trigger": 0.5
  },
  "wilson_ci": {
    "baseline_rerun": [
      0.09453120573423074,
      0.9054687942657693
    ],
    "ablate_trigger_ws": [
      0.09453120573423074,
      0.9054687942657693
    ],
    "logit_suppress_trigger": [
      0.09453120573423074,
      0.9054687942657693
    ]
  },
  "mean_gen_length": {
    "baseline_rerun": 282.0,
    "ablate_trigger_ws": 282.0,
    "logit_suppress_trigger": 282.0
  }
}

## Power note
{
  "target_power": 0.8,
  "smallest_effect": "halve the baseline loop rate",
  "note": "baseline loop rate is 0 (or no pairs) \u2014 power note undefined"
}

## Interpretation
- Result type: **negative**
- Primary contrast (onset-timed per-trace trigger ablation, all workspace layers): loop rate 0.000 vs baseline_rerun 0.000 (random 0.000, control-token 0.000, sensory 0.000, logit-suppression positive control 0.000). Result type: negative. Both positive and negative results are publishable per PROJECT_SPEC §11.6.

Both positive and negative results are publishable per PROJECT_SPEC §11.6.
