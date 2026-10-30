# Architecture: Exp3 decoding baselines

The locked Exp3 condition list stays `MAIN_CONDITIONS` in `scripts/05_exp3_causal.py`. `active_main_conditions()` copies that list and, only when `JLENS_EXP3_DECODING_BASELINES=1`, appends two names.

```
jspace/decoding_baselines.py
  extend_condition_names
  hf_overrides_for_condition
        │
        ▼
scripts/05_exp3_causal.py  run_intervention_condition
        │
        ▼
jspace/generation.py  generate_greedy → build_hf_gen_kwargs
        │
        ├─ HF model.generate(..., repetition_penalty=1.2)
        └─ HF model.generate(..., no_repeat_ngram_size=4)
```

If a vLLM generator is passed together with either override, `generate_greedy` raises. The override is not silently dropped.

Draft PR #2's sidecar (`jspace/antidoom_sidecar.py`, `JLENS_ANTIDOOM_SIDECAR`) is a logits processor on the baseline pass. It is not imported here and not copied.
