# Process: Exp3 decoding baselines

## Steps

1. Read `scripts/05_exp3_causal.py` and `jspace/generation.py` on `cursor/lfm2-exp1-jlens-fit`.
2. Read draft PR #2's file list. The sidecar lives on `cursor/t4-baseline-antidoom-sidecar-7f71`, not on this base. This branch does not cherry-pick that PR.
3. Added `jspace/decoding_baselines.py` and threaded two optional kwargs through `generate_greedy`.
4. Exp3 calls those kwargs only when the condition name is one of the two arms, and the arms are appended only if `JLENS_EXP3_DECODING_BASELINES=1`.
5. Did not run `scripts/05_exp3_causal.py`. Did not edit `results/`.

## Verified

`python -m py_compile scripts/05_exp3_causal.py jspace/decoding_baselines.py jspace/generation.py` exited 0 and printed nothing.

Command:

```bash
python -m pytest tests/test_decoding_baselines.py -q
```

Output:

```
........                                                                 [100%]
8 passed in 0.02s
```

The tests check:

- the locked seven names are unchanged when the flag is off
- the flag appends `repetition_penalty` then `no_repeat_ngram`
- the override dicts are exactly `{repetition_penalty: 1.2}` and `{no_repeat_ngram_size: 4}`
- `ANALYSIS_PLAN` in the script does not mention either arm
- `generate_greedy` passes both kwargs into `build_hf_gen_kwargs` and contains a `raise`

## Assumed

- Hugging Face `model.generate` accepts `repetition_penalty` and `no_repeat_ngram_size`. That was not executed here.
- PR #2 remains the sidecar. This PR does not depend on merging it. The arms are neighbors in the experiment design, not a git stack.

## Not measured

Any loop rate, length, or GSM8K accuracy under either arm. No GPU generation was run.
