# Design decisions: Exp3 decoding baselines

## Choice

Add `repetition_penalty` and `no_repeat_ngram` as opt-in Exp3 arms that pass Hugging Face `generate` kwargs. Default off, so the locked v2 condition list and `ANALYSIS_PLAN` text stay as they are on `cursor/lfm2-exp1-jlens-fit`.

The constants are `repetition_penalty=1.2` and `no_repeat_ngram_size=4`. `1.0` would be a no-op for the penalty, so the arm would not be a baseline. `4` is a fixed n-gram width. Neither value was fit to a loop rate.

## Alternatives considered

- Copying the PR #2 sidecar into this branch. Rejected. That processor already has its own draft PR. These arms are a different mechanism: a penalty on tokens that already appeared, and a ban on repeating any n-gram, not a block on the single token that would make a fourth copy of one n-gram.
- Putting the arms inside `MAIN_CONDITIONS` unconditionally. Rejected. That would change the pre-registered condition count the next time Exp3 is launched, including checkpoint `n_total_pairs`, without a measured reason.
- Running the arms on GPU in this change. Rejected. No generation metric is claimed.

## Why HF only

`build_hf_gen_kwargs` is the call site that already builds the unsteered greedy kwargs. vLLM sampling params in this repo do not take these two controls. Raising on the vLLM path is the alternative to recording a run that never applied the arm.
