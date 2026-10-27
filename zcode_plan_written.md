# ZCode Plan — J-Lens Doom Loops → arXiv Paper

Written 2026-10-26 (updated same day after a full direction review and provenance
verification). This document captures every decision, finding, and next step discussed in
the planning sessions, so the plan survives outside the chat history. **It is written to
be self-contained: a fresh agent with no access to the original conversation should be
able to execute or discuss this plan from this file alone.**

Repo: `C:\Users\mithi\Desktop\mithil\projects\j-lens`
Remote: `github.com/Mithilyaganti/jlens-doom-loop-analysis`
Branch of record: **`cursor/lfm2-exp1-jlens-fit`** (the only branch with current work;
`main` is stale and 8+ commits behind). All work happens here; merge to `main` only at the end.
Key commits: `453bb34` (Phase A code + stats), `e30d5ba` (this plan document).
The Colab notebook clones this branch, so every push flows to Colab automatically.

**Provenance verification (done 2026-10-26):** the Liquid AI Antidoom blog directly states
Qwen3.5-4B's doom-loop rate dropped "from 22.9% to 1% **under greedy sampling**" after FTPO
training (their LFM2.5-2.6B early checkpoint is a different model, 10.2% → 1.4%). This is
the anchor for the B1 expectation, and it has now been verified at the source, not just
copied from `PROJECT_SPEC.md`. It also confirms our Phase A switch to greedy decoding
matches Liquid's own measurement protocol.

---

## 1. Where the project actually stood when we resumed

Two branches exist: `main` (stale) and `cursor/lfm2-exp1-jlens-fit` (everything, pushed).
The `INTERVIEW_*.md` files are unrelated interview notes and are not part of this project.

**LFM2-2.6B Phase 1 — complete but statistically thin:**

| Stage | Result | Problem |
|---|---|---|
| Lens fit | `lenses/lfm2-2.6b.pt`, 270 WikiText prompts, fp16, 116 min on T4 | attention layers only; conv skipped |
| Workspace band | "band" = **L21 only** | only 8 fitted layers, so the band collapses to one layer |
| Baseline | 11 loops / 200 prompts (5.5%) | 11 unique triggers, each seen exactly once |
| Exp1 geometry | pairwise sim 0.0905 vs 0.0878; alignment 0.2477 vs 0.1942 | passed via alignment only, no p-values |
| Exp2 dynamics | 5 loops / 200; align@−1 0.172 vs 0.130; top1=trigger 60% vs 4% | n=5; mid-run cache bug (fixed via 04b recache) |
| Exp3 causal | baseline = ablate_trigger = 2/11; McNemar p=0.62 | **null, n=11**, 4 protocol flaws |

The four self-identified Exp3 flaws: (1) ablation applied at **every** token position
instead of the onset, (2) a **single layer**, (3) **one global trigger** instead of per-trace,
(4) **no seeds** — so loop labels changed between runs (11 → 5 → 2 loopers), destroying the
paired McNemar test.

**Qwen3.5-4B — the planned decisive model, never actually run.** The locked plan
(`docs/PLAN_QWEN_A_B_MIX.md`) pivots to Qwen: real 5-layer band **L24–28**, free pre-fitted
lens from neuronpedia (388 MB), and Liquid AI reported a 22.9% loop rate under vLLM bf16 —
versus the ~2–5.5% seen locally under NF4. The Colab A1 notebook was written and debugged
(5 commits of vLLM/Colab fixes) but produced **no results**. That is exactly where the
project stopped a month ago.

**Root causes of "it works badly" (all now fixed in Phase A):**
low loop yield · unseeded T=0.01 sampling · blunt all-position single-trigger ablation ·
zero error bars anywhere · analysis scripts reloading the model · a Colab fallback bug that
silently loaded Qwen in NF4 when the notebook promised bf16.

---

## 2. Locked decisions (agreed in this session)

| Decision | Choice |
|---|---|
| GPU platform | **Google AI Pro Colab quota** (user has it via the Jio India offer). ~200 compute units/month; estimated need 25–50 CU. The **$20 is contingency only** — hard cap on any paid fallback (RunPod/Vast). |
| Scope | **Qwen3.5-4B decisive** runs + **LFM2-2.6B secondary** (re-analyzed free from existing artifacts). Gemma-4 cross-model is a **stretch**, only if Qwen results are strong and ≥100 CU remain. |
| Eval pool | Locked 200 prompts (seed 42) first. **May extend to ~500** with a second seeded batch, **disclosed** in the paper, if loop yield is 8–15%. Headline loop rates stay per-batch. |
| Subagent models | **GLM-5.3 (full), not GLM-5.3-Flash**, for all code/analysis/writing agents. Rationale: the expensive failure mode is a subtle experiment bug that burns GPU budget, or a weak stats/paper section. Flash is fine only for read-only exploration sweeps. |

**Paper target: arXiv.** Note that `PROJECT_SPEC.md` §16 originally framed a blog-first then
ICML 2027 path; the user's goal here is an arXiv paper, which supersedes that.

---

## 3. Methodology changes made after pressure-testing the science

The three-step ladder (observe geometry → watch pre-onset dynamics → test causality) was
judged sound and kept. Four changes were added because without them results would not be
interpretable:

1. **Positive control in Exp3 (most important).** If ablation does nothing, we cannot
   distinguish "the direction is not causal" from "our intervention was too weak or badly
   timed." Added a **trigger-token logit suppression** condition that blocks the trigger from
   being selected for K steps — loop prevention by construction. If suppression stops loops
   and surgical ablation does not, the finding is clean: *the workspace describes the failure
   but cannot steer it.*
2. **Exp2 matched controls.** The old comparison anchored non-looping traces at an arbitrary
   mid-answer position, and the headline "top-1 = trigger 60% vs 4%" was nearly tautological.
   New control: moments in non-looping traces where the **same trigger token was emitted once
   and the model moved on**. This is what produced the key early result (§4).
3. **Real eval instead of 5 arithmetic questions.** A 50-problem GSM8K sample per key
   condition (`data/gsm8k_eval.json`, seed 42, committed), scored by last-number exact match,
   to preempt "your intervention lobotomized the model."
4. **Onset timing sweep** (−3, −1, 0 steps) so "we ablated too late" is not a possible
   objection.

**Rejected alternatives (and why):** retraining-based fixes (that is Liquid's Antidoom, and
it defeats the "without training" question); SAE/attention-surgery approaches (different
paper); larger models (budget).

**Honest per-experiment expectation** stated up front:
baseline + trigger table ≈ certain deliverable · Exp1 ≈ modest, may be weak ·
Exp2 ≈ most likely strong positive · Exp3 ≈ genuine coin flip, decisively answerable either way.

---

## 4. Phase A — completed (commit `453bb34`, pushed)

All CPU tests pass (4 suites, 60+ tests). Both GPU smoke runs pass on the laptop
(LFM2-2.6B, NF4).

**Code changes**
- `jspace/generation.py` — greedy (T=0) default, per-prompt seeds
  (`seed = base*1000003 + pid mod 2^31`), seeds recorded in `meta.json`; one-pass activation
  recording (`JLENS_RECORD_MODE=onepass`, verified bit-identical to the old 5-pass chunked
  path, kept as fallback); `logits_processor` support; loop trigger metadata
  (`loop_trigger_token_id`, `loop_trigger_text`, `loop_trigger_pos`).
- `jspace/vllm_backend.py` — `generate_batch()` (one continuous-batched call),
  `enable_prefix_caching=True` with TypeError fallback, greedy default, per-prompt seeds;
  fixed the fallback memory leak (release engine + empty_cache before HF load).
- `jspace/stats.py` (new) — bootstrap CIs, two-sample and paired permutation tests,
  exact McNemar, Cohen's h, Wilson interval, exact McNemar power and sample-size
  (log-space binomial to avoid overflow). numpy + stdlib only.
- `jspace/geometry.py` — dequantize the lm_head **once** (was re-dequantizing the whole
  `[vocab, d]` matrix per token); lazy safetensors row provider; per-token metric helpers;
  SVD basis reuse.
- `jspace/intervention.py` — `make_onset_ablation_hook` (positions ≥ start_pos only,
  multi-direction, Gram–Schmidt orthonormalized), `TriggerSuppressLogitsProcessor`
  (positive control), `build_trigger_directions` (per-trace).
- `jspace/eval_sets.py` (new) — GSM8K loader/sampler/last-number scorer.
- `scripts/02_baseline_pass.py` — batched vLLM path, seed/temperature/greedy in meta,
  HF fallback defaults to bf16.
- `scripts/03_exp1_static_geometry.py` — one SVD per layer (was 4×), per-token bootstrap CIs,
  permutation p-values vs each control set, proper success statements,
  `JLENS_EXP1_ANALYZE_ONLY=1` runs fully model-free.
- `scripts/04_exp2_dynamic.py` — ANALYZE_ONLY loads **no model** (lens + tokenizer + cached
  unembedding rows), streams traces instead of loading ~1 GB, **matched-control analysis**
  (same-token anchors + position-matched anchors) with permutation tests.
- `scripts/05_exp3_causal.py` — rewritten two-pass protocol: fresh onset detection per
  looper, 7 conditions (baseline_rerun, ablate_trigger_ws, ablate_random,
  ablate_control_token, ablate_trigger_sensory, ablate_trigger_motor,
  logit_suppress_trigger), timing sweep @−3/@−1, dose top-1/3/5, GSM8K eval, generation-length
  secondary metric, exact McNemar + Wilson CIs + Cohen's h per condition, automatic power note.
- `notebooks/colab_qwen35_4b_a1_baseline.ipynb` — greedy; **`JLENS_HF_QUANTIZE=0` on both
  branches** (silent NF4 fallback bug fixed); backend/dtype audit printed after the smoke cell.
- Tests: `tests/test_stats.py`, `test_generation_logic.py`, `test_analysis_stats.py`,
  `test_intervention.py`. Docs: `docs/PROJECT_EXPLAINER.md` (plain-language guide).

**Early scientific result from re-analyzing existing LFM2 artifacts (free, no GPU):**
- **Survives matched controls:** pre-loop workspace occupancy 0.249 vs 0.568 at same-token
  anchors (permutation **p = 0.015**) and 0.542 at position-matched anchors (p = 0.038).
  This is now the Exp2 headline.
- **Does not survive:** top-1 readout = trigger (0.600 vs same-token 0.520, **p = 1.0**) and
  trigger-direction alignment (0.172 vs 0.176, **p = 0.90**) — these mostly reflect "a token is
  imminent," not loop specificity.
- **Exp1** now has p-values: both criteria met at **p = 0.0002** vs the weakest control, with
  the honest caveat that the discourse-marker control scores higher on pairwise sim.

---

## 5. Phase B — the GPU runbook (user runs cells; I track quota)

**B1 — Qwen3.5-4B A1 baseline** (`notebooks/colab_qwen35_4b_a1_baseline.ipynb`, cells 1→8,
runtime L4 preferred, T4 acceptable): batched vLLM bf16, greedy, 200 prompts, 4000 tokens.
Deliverable: headline loop rate + empirical trigger table.
**Pre-registered gates:** ≥15% → proceed · 8–15% → extend to 500 (disclosed batch 2) ·
<8% → add GSM8K/MATH mine pool for Exp3 power only, headline stays eval-pool.

**B2 — Qwen Exp2 hooked regen** (HF bf16, one-pass recording): pre-onset dynamics with
matched controls and CIs; expect 30–80 loopers.

**B1.5 (added after review — do at the START of the B2 session, same runtime): re-compute the
Qwen workspace band.** No canonical `workspace_band_qwen3.5-4b.json` survives in `results/`
(only two conflicting historical ranges, L23–30 and L24–28, in old exports). B2's recording
layers and B3's band-specific conditions both depend on it. Run
`python scripts/01_workspace_band.py` with `JLENS_MODEL=Qwen/Qwen3.5-4B` — it's cheap
(~40 WikiText NTP prompts, model + pre-fitted lens only) and must be done **before** B2's
recording, on the same runtime session if possible. Note the A1 notebook is baseline-only
by design, so this step was not part of it. The plan is to fold this into the B2 notebook
cells when we set up B2.

**B3 — Qwen Exp3, fixed protocol** (onset-timed, per-trace, positive control, timing sweep,
dose, GSM8K). Pre-registered branches: positive → causal headline; null → "describes but
cannot steer" headline. Both publishable per `PROJECT_SPEC` §11.6.

**B4 — stretch:** Gemma-4 E2B/E4B Exp1+Exp3 only if results are strong and ≥100 CU remain.

**Budget guard:** ~25–50 CU estimated (8–16 GPU-hours). I track burn per step and will stop
for approval before anything that threatens the monthly quota. $20 hard cap for paid fallback.

---

## 6. Phase C — analysis and paper

1. Publication figures (300 dpi, error bars and CIs on every panel) and final tables,
   including the trigger tables.
2. arXiv LaTeX paper (~8–10 pages): doom loops break long-CoT reasoning; Antidoom fixes it by
   retraining; we test whether the J-lens localizes the failure and whether it can be switched
   off without training. Qwen primary; LFM2 secondary with hybrid-architecture caveats;
   side effects analyzed honestly (control-token ablation raised loops in LFM2; motor-band
   ablation halved generation length). Related work: Anthropic J-lens `arXiv:2607.15495`,
   Liquid Antidoom, Lazaridis et al. `arXiv:2606.13705`, Word Salad Chopper, Wait×3,
   Silent Alarm/JADR. Limitations pre-declared.
3. Clean `RESULTS_SUMMARY.md` (currently has duplicated stale sections), repro docs
   (seeds, env, CU/$ spent), merge branch → `main`, tag. **User submits to arXiv**; I produce
   the PDF and source tarball.

---

## 7. Risks and standing mitigations

| Risk | Mitigation |
|---|---|
| Qwen bf16 loop rate ≪ 22.9% | B1 gates above; worst case the paper leads descriptive and is still publishable |
| Colab GPU availability / version friction | notebook keeps HF-bf16 fallback; RunPod contingency inside $20 |
| Exp3 null | positive control makes a null interpretable and publishable; pre-registered narrative |
| Loss of a long run | greedy + seeds make reruns reproducible; Drive checkpoint every 10 prompts |
| NF4/bf16 backend confound | determinism study is now part of the paper (it is what invalidated LFM2 Exp3) |

---

## 8. Full-project retrospective — verified mistakes, costs, and status

A direction review was done on 2026-10-26 covering the project's whole history, not just
this chat. Conclusion: **the scientific direction is correct and was kept; the historical
problems were execution and sequencing, and every one is now fixed or scheduled.** The
mistakes, most fundamental first:

1. **Hardware-first instead of validity-first (founding mistake).** Qwen3.5-4B was run on an
   8 GB laptop in 4-bit NF4. Loop behavior is precision-sensitive, so quantization was a
   confound from run one (~2% loop rate vs Liquid's 22.9%), and sparse triggers / tiny n
   cascaded from it for weeks. *Fixed:* decisive runs on Colab bf16; the NF4-vs-bf16 gap
   itself becomes a determinism-study section of the paper.
2. **Papering over the trigger-extraction failure.** When 28 prompts yielded zero loops,
   Exp1 kept moving on an invented `_RESTART_WORDS` word list instead of empirical triggers
   (documented honestly, but the early geometry claims were partly built on guesses).
   *Fixed:* Qwen's expected high loop yield supplies real triggers; Exp1 p-values now state
   which control beats which metric.
3. **Running the decisive causal test before generation was stable.** Original Exp3 ran at
   n=11 with unseeded T=0.01 sampling, so baseline loop labels changed across runs
   (11 → 5 → 2 loopers), voiding the paired McNemar test regardless of design. *Fixed:*
   greedy + per-prompt seeds came first; the new Exp3 detects onset fresh per trace.
4. **Generation cap chosen before understanding the phenomenon** — a 124/200 baseline at
   2048 tokens was killed and archived (loops need 4000-token horizons to appear). *Fixed:*
   4000 locked; the archived run is kept as honest history.
5. **Exp2 cache window bug** invalidated 4/5 loops in the first analysis pass (tail-window
   dropped mid-sequence triggers); salvaged by the 04b trigger-centered recache. Lesson:
   validate window bookkeeping on synthetic traces before a run — now covered by test suites.
6. **Accepting a degenerate single-layer "band"** to keep moving on LFM2. The pivot to Qwen
   (real 5-layer band) was correct but ~2 weeks later than it should have been.
7. **Engineering around instability instead of fixing it** — a pile of Windows watchdog
   scripts existed to babysit slow nondeterministic runs.
8. **Housekeeping debt:** `RESULTS_SUMMARY.md` accumulated duplicated/contradictory sections
   (two Qwen band ranges; pre- and post-fix Exp2 numbers side by side). *Scheduled:* Phase C.
9. **The month of silence:** the July 30 pivot plan and A1 notebook were correct and fully
   debugged — and then never executed. The project stopped one execution away from its
   decisive data.

Near-misses in the resumption chat, both caught and handled: a smoke run briefly wrote a
slug-scoped summary that shadowed the legacy 11-looper baseline (Exp3 smoke saw "0 looping
prompts" until traced and cleaned — lesson: smoke runs and real state should never share
files; a `JLENS_SMOKE_DIR` isolation knob is worth adding before any further local smoke
runs); and the batched vLLM call trades checkpoint granularity (a Colab disconnect mid-batch
loses in-flight prompts; they rerun on resume — accepted given Drive checkpointing every 10).

**Remaining genuine uncertainties (not mistakes):** the Exp3 causal outcome is a real coin
flip — that is the experiment; gates are pre-registered and both outcomes are publishable,
with the logit-suppression positive control distinguishing "not causal" from "intervened
wrong." Whether the LFM2 secondary section holds up with real error bars is already known
(see §4): the occupancy-collapse headline survives (p=0.015), the top-1-readout and
alignment headlines do not (p=1.0 / p=0.90 against matched controls) and are dropped.

## 9. Handoff notes for a coordinating agent

This section is for an agent that receives this document as its brief and will discuss,
plan, or fan out tasks on the user's behalf. It states what is **locked** (do not relitigate),
what is **open** (genuine decisions for the user), and what to **ask the user for**.

**Locked — treat as settled user decisions:**
- arXiv paper is the goal (this supersedes `PROJECT_SPEC.md` §16's blog-first/ICML-2027
  path; the spec is background, not the plan).
- Branch of record `cursor/lfm2-exp1-jlens-fit`; do not merge to `main` until Phase C ends.
- GPU: Google AI Pro Colab quota; the **$20 is a contingency hard cap** (user's money,
  ask before touching it). Prefer L4; T4 acceptable.
- Qwen3.5-4B decisive + LFM2-2.6B secondary; Gemma-4 only as stretch if strong results
  AND ≥100 CU remain.
- Greedy decoding (T=0) with per-prompt seeds everywhere; near-greedy sampling is
  forbidden in headline runs (it caused the label instability that voided the original Exp3).
- Eval-pool discipline: the locked seed-42 200-prompt batch is the headline set; any
  extension to ~500 or mine-pool prompts must be disclosed per-batch and never silently
  merged into headline numbers.
- The Exp3 protocol in `scripts/05_exp3_causal.py` is pre-registered — positive control,
  matched controls, timing sweep, dose, GSM8K eval, power note. Do not weaken controls to
  chase a positive result.
- Eval-pool gates are pre-registered (§5).
- Subagent model for research/implementation work: **GLM-5.3 full, not Flash** (correctness
  protects a finite GPU budget and the paper). Flash only for read-only exploration sweeps.

**Open — genuine user decisions, do not decide unilaterally:**
- B1 gate outcome: extend to 500 prompts vs add mine pool vs proceed (recommendation rules
  in §5; user should see the loop rate before choosing).
- Whether the B4 Gemma-4 stretch runs at all.
- arXiv submission timing/authorship (user submits; we produce PDF + tarball).
- Any spend from the $20 contingency.

**Ask the user for (things only they can provide):**
- The B1 loop rate and the content of `baseline_pass_summary_qwen3.5-4b.json` after they
  run the Colab notebook (cells 1→8, L4/T4 runtime).
- Colab session realities: which GPU they were actually allocated, whether vLLM installed
  or it fell back to HF bf16 (the notebook's smoke-cell audit prints this).
- Drive checkpoint state if a session disconnected (notebook Cell 5 restores it).

**Execution facts an agent needs (verified):**
- Phase A is complete, tested, committed (`453bb34`), and pushed; all 4 test suites pass
  (60+ tests); both GPU smoke runs passed locally (LFM2-2.6B NF4: baseline + full new Exp3
  protocol including GSM8K scoring and the power note).
- The immediate next action is B1 in Colab — the user runs the cells; **no agent can run
  Colab for them.** After B1, the pipeline steps are runnable by agents locally for
  analysis, but every GPU session needs the user to open Colab.
- Key files: `jspace/stats.py` (shared stats — use it, don't hand-roll), `jspace/eval_sets.py`
  (GSM8K), `docs/PROJECT_EXPLAINER.md` (plain-language guide for the user), the plan you are
  reading, `docs/PLAN_QWEN_A_B_MIX.md` (July 30 locked decisions — historical context),
  `LFM2_26B_FINDINGS_AND_FAQ.md` (honest LFM2 caveats), `PROJECT_SPEC.md` (original design).
- Env knobs of record: `JLENS_MODEL`, `JLENS_BACKEND`, `JLENS_RECORD_MODE=onepass`,
  `JLENS_EXP1_ANALYZE_ONLY=1`, `JLENS_EXP2_ANALYZE_ONLY=1` (default on),
  `JLENS_EXP3_PROMPTS` (smoke cap), `JLENS_GSM8K_N`, `JLENS_EXP3_TIMING_N/DOSE_N`,
  `JLENS_SUPPRESS_K`, `JLENS_BASELINE_BATCH_GEN`, `JLENS_MAX_NEW_TOKENS=4000`.
- Known open items: B1.5 Qwen band recompute (§5); `RESULTS_SUMMARY.md` dedup + Phase-C
  figures; possible `JLENS_SMOKE_DIR` isolation knob before further local smoke runs;
  `data/` is gitignored except `data/gsm8k_eval.json`; `INTERVIEW_*.md` files are unrelated
  interview notes — do not commit or touch them; `vendor/open-jlens-data` is a gitlink pinned
  at `49010193` (its only "diff" is a stray `__pycache__` — nothing to commit).
- Reproducibility facts for the paper: seeds recorded per prompt in every `meta.json`;
  LFM2 loop-label instability across runs (11 → 5 → 2) is itself a documented finding and
  motivates the determinism study section.

## 10. Immediate next action

Run **B1** in Colab (cells 1→8). Report back the loop rate from the 200-prompt run; the gate
in §5 decides whether we proceed straight to B2/B3, extend the pool to 500, or add the mine pool.
