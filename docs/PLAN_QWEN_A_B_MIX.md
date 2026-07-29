# Final Plan — Qwen3.5-4B Phase 1 (A + B mix)

**Status:** LOCKED  
**Date:** 2026-07-30  
**Supersedes:** earlier draft in this file; LFM2 Phase 1 remains archived under `artifacts/` / `results/` as historical only.

Related write-ups: `LFM2_26B_FINDINGS_AND_FAQ.md`, `docs/BLOG_DRAFT_LFM2_JLENS_OPTION_C.md` (LFM observational). This plan is the **Qwen** execution path.

---

## Goal

Run a clean Exp1 → Exp2 → **fixed Exp3** on **`Qwen/Qwen3.5-4B`** using the **pre-fitted Neuronpedia J-lens**, with **stable Colab generation** so loop labels are reproducible enough for causal tests. If Exp3 is interpretable (positive **or** clean negative), replicate on **Gemma-4 Edge** (Phase B).

---

## Locked decisions (A1–A5 + B)

| Option | Decision | Notes |
|--------|----------|-------|
| **A1** Stable generation | **Do now** | Colab **vLLM**; not laptop 16-bit; not GGUF |
| **A2** More / harder data | **Later** — only if n_loops too small after A1+Exp pipeline | Eval vs mine split (below) |
| **A3** Fix Exp3 intervention | **Do on Qwen** | Onset-timed, per-trace triggers, dose, controls |
| **A4** Richer Exp2 / Tier-3 HTML | **Leave for later** | Additive only; does **not** require redoing Exp1/2/3 |
| **A5** Stronger J-lens fit | **Skip** | Pre-fitted lens is enough |
| **B** Cross-model | **After Qwen** | Gemma-4 E2B/E4B if Qwen Exp3 is interpretable |

---

## Model & lens

| Use | Do **not** use |
|-----|----------------|
| **`Qwen/Qwen3.5-4B`** (official HuggingFace) | `unsloth/Qwen3.5-4B-GGUF` or any GGUF |
| Lens: `lenses/qwen3.5-4b.pt` (Neuronpedia `Salesforce-wikitext`) | Refitting 1000 WikiText prompts on free Colab |

**Why not GGUF:** Exp1–3 need Transformers residual hooks + J-lens. llama.cpp/GGUF cannot drive that stack. Q8_0 size (~4.5 GB) is irrelevant.

**Laptop smoke (done):** NF4 + lens OK (~3.2 GB). Full bf16/fp16 on 8 GB only via CPU offload — too slow for 4000-token runs. vLLM not installed on Windows laptop.

---

## Data policy (eval vs mine)

### Now (A1 and first Exp pass)

- **Eval set only:** same **200** stratified `antidoom-mix` prompts, **seed=42** (`results/prompt_sample_ids.json`).
- Comparable to LFM2 run; honest loop-rate reporting.
- **No** MATH/GSM8K expansion, **no** embed mining yet.

### Later (A2) — only if loops are too few for Exp3

Keep two pools; never silently merge:

| Pool | Contents | Used for |
|------|----------|----------|
| **Eval** | seed=42 200 (optionally larger *stratified* draw later) | Reported loop **rates**, main tables |
| **Mine** | Hard **GSM8K / MATH** first; optional **small** coding slice; optional **prompt-neighborhood** retrieval from known loopers | Extra Exp2/3 *n* only — label as **enriched / mine** |

**Coding:** Allowed as a **minor** mine source (harder for 4B; domain-biased triggers). Prefer math packs first; add ~50 coding only if needed.

**Prompt-neighborhood search:** Allowed for **mine only**. It **is** selection bias by design — OK for finding loopers for ablation; **not** OK as the reported natural loop rate. Prefer embedding similarity to **known looping prompts**, not “contains Liquid trigger strings” (`But`, ` the`, …).

**Circularity rule:** Exp1 trigger tables and primary rates come from **eval** loops when possible. Mine loops may feed Exp3 power; disclose enrichment.

---

## Precision & backends

### A1 — Baseline (Colab vLLM)

| Item | Value |
|------|--------|
| Model | `Qwen/Qwen3.5-4B` |
| Backend | **vLLM** |
| Tokens | `max_new_tokens=4000`, `temperature≈0.01` |
| Prompts | seed=42 **200** |

**Dtype smoke order (Colab):**

1. **T4 (free):** smoke **`bfloat16`** first (1–2 prompts, short then 4000). `fp8` often unsupported on T4.  
2. **L4 / A100:** may try **`fp8`** first; code already falls back **fp8 → bfloat16**.  
3. **`float16`:** only if bf16 misbehaves.

Do **not** assume “bf16 fail → fp8 saves T4.”

Checkpoint baseline progress; sync to Drive.

### Exp2 — Hooked monitoring

vLLM does **not** replace Exp2 (need residual / J-lens hooks).

| Preference | Backend |
|------------|---------|
| **1st** | Colab **HF bf16** + hooks + checkpoints |
| **2nd** | Laptop **HF NF4** + lens (backup) |

**Checkpoints:** Resume-safe (`results/checkpoints/` + per-prompt traces under `exp2/looping|nonlooping`). Stop/restart anytime; only unfinished prompts re-run. Sync Drive if using Colab.

**Core metrics only** (alignment −1, top-1 convergence, occupancy). **A4 / Tier-3 HTML later** — additive; no full Exp2 redo required.

Ballpark time for 200×≤4000 hooked: Colab bf16 often ~4–12 h; laptop NF4 often longer. Use checkpoints either way.

### Exp3 — Causal (A3 fixes)

- Prefer **stable re-loop** on known loopers (same precision family as A1 for the no-ablation arm when possible).  
- Ablations need **HF hooks** (Colab bf16 preferred).  
- **Must fix vs LFM Exp3:** ablate at **onset / −1** (not all positions); **per-trace** trigger direction; dose top-1/3/5; random + control + band controls.  
- Skip Exp3 only if ~0 eval loops → run **A2 mine** first, then Exp3 on enriched loopers (disclose).

### Exp1 / workspace band

- Laptop or Colab HF NF4/bf16 + pre-fitted lens.  
- Expect a **multi-layer** workspace band (unlike LFM L21-only).

---

## Execution order (final)

1. Point active model at **`Qwen/Qwen3.5-4B`**; keep LFM artifacts archived (do not mix numbers).  
2. **A1** Colab vLLM smoke (bf16 on T4) → full **200** baseline → trigger CSV + loop rate.  
3. **Workspace band** (`01_…`).  
4. **Exp1** static geometry.  
5. **Exp2** hooked gen (Colab HF bf16 preferred) → core pre-onset metrics; checkpoint throughout.  
6. **A3 Exp3** with fixed intervention (if enough loops; else A2 then Exp3).  
7. **A2** only if needed: MATH/GSM8K-hard → optional small coding → optional embed-neighbors; screen with vLLM; tag **mine**.  
8. **A4** optional later (Tier-3 HTML / richer plots) — no pipeline redo.  
9. **B** Gemma-4 E2B/E4B if Qwen Exp3 is interpretable.

---

## Explicit non-goals

- Refitting Qwen J-lens on free Colab  
- Unsloth / GGUF / llama.cpp path  
- Full FP16/BF16 Qwen as primary runner on 8 GB laptop  
- Doing A4 before core Exp2/Exp3  
- Replacing eval 200 with mined prompts for headline rates  
- Mixing LFM2 and Qwen results into one claim  
- Trigger-string corpus filters as the main A2 strategy  

---

## Success criteria (honest)

- **A1:** Reproducible Qwen loop rate + empirical triggers on eval 200 (vLLM).  
- **Exp1:** Alignment and/or clustering vs controls in workspace band (or documented null).  
- **Exp2:** Pre-onset J-space difference looping vs non-looping (or documented null); n_loop reported honestly.  
- **Exp3:** Interpretable positive **or** clean negative under **fixed** intervention; underpowered null only if n still tiny after A2.  
- **B:** Same protocol on Gemma only after Qwen Exp3 is readable.

---

## Next action

Start **A1**: open `notebooks/colab_qwen35_4b_a1_baseline.ipynb` on Colab **T4**.

- Smoke **bf16** (Cell 6) → full **200** baseline (Cell 7).
- Checkpoints every prompt locally; Drive sync every **10** prompts (`MyDrive/jlens_qwen_a1_baseline/`).
- Defer A2/A4 until this path is green.

See also: `docs/COLAB_QWEN_A1_RUN.md`.
