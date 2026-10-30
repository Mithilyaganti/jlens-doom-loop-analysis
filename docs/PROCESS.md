# Process: Holm and Benjamini-Hochberg

## Steps

1. Added `holm_adjust` and `benjamini_hochberg` to `jspace/stats.py`.
2. Derived the expected arrays below by hand, then asserted them in `tests/test_stats.py`.
3. Ran the existing stats tests plus the new ones. Did not open `results/exp3/stats.json` as an input to these functions.

## Hand derivation

Input `p = [0.01, 0.04, 0.03]`, so `m = 3`. Sorted order is index 0 (`0.01`), index 2 (`0.03`), index 1 (`0.04`).

Holm, multipliers `m, m-1, ..., 1`, then a cumulative maximum, each term clipped at 1:

- `3 * 0.01 = 0.03`
- `max(0.03, 2 * 0.03) = max(0.03, 0.06) = 0.06`
- `max(0.06, 1 * 0.04) = 0.06`

Back in input order: `[0.03, 0.06, 0.06]`.

Benjamini-Hochberg, `raw_i = min(1, (m / i) * p_(i))` with `i` starting at 1, then `a_i = min(a_{i+1}, raw_i)`:

- `(3/1) * 0.01 = 0.03`
- `(3/2) * 0.03 = 0.045`
- `(3/3) * 0.04 = 0.04`
- from the right: `0.04`, `min(0.04, 0.045) = 0.04`, `min(0.04, 0.03) = 0.03`

Back in input order: `[0.03, 0.04, 0.04]`.

NaN check. Finite family `[0.01, 0.04]`, `m = 2`, input `[0.01, NaN, 0.04]`:

- Holm: `2 * 0.01 = 0.02`, then `max(0.02, 0.04) = 0.04` → `[0.02, NaN, 0.04]`
- BH: `(2/1) * 0.01 = 0.02`, `(2/2) * 0.04 = 0.04` → `[0.02, NaN, 0.04]`

One test: multiplier 1, so `0.2` stays `0.2` for both.

Two values of `0.6`: Holm `min(1, 2 * 0.6) = 1`, and the cumulative max stays 1. BH from the right is `0.6`, then `min(0.6, min(1, 2 * 0.6)) = 0.6`.

## Verified

```bash
python tests/test_stats.py
```

```
PASS test_bootstrap_ci_empty_and_nan
PASS test_bootstrap_ci_mean
PASS test_bootstrap_ci_median
PASS test_cohens_h_known_value
PASS test_holm_and_bh_clip_at_one_and_identity_for_one_test
PASS test_holm_and_bh_hand_derived_triple
PASS test_holm_and_bh_nan_is_outside_the_family
PASS test_mcnemar_exact_known_values
PASS test_mcnemar_power_monotone_and_sane
PASS test_mcnemar_sample_size_roundtrip
PASS test_paired_permutation_test
PASS test_permutation_test_null_and_signal
PASS test_wilson_ci_known_value
ALL PASS
```

```bash
python -m pytest tests/test_stats.py -q
```

```
.............                                                            [100%]
13 passed in 0.13s
```

## Assumed

The formulas above are the standard Holm step-down and BH step-up adjustments. No paper table was passed through them.

## Not measured

Adjusted p-values for the checked-in Exp3 contrasts. The script still reports the unadjusted exact McNemar p-value.

## Done

- `holm_adjust` and `benjamini_hochberg` are in `jspace/stats.py`.
- Tests in `tests/test_stats.py` use the hand-derived values above.
- `python tests/test_stats.py` printed `ALL PASS`. `python -m pytest tests/test_stats.py -q` printed `13 passed in 0.13s`.
- `docs/ARCHITECTURE.md`, `docs/DESIGN_DECISIONS.md`, `docs/PROCESS.md`, and a README CPU section are on this branch.
- Exp3 script and `results/` were not modified.

## Remaining

- These functions are not called from `scripts/05_exp3_causal.py`.
- No adjusted p-value was computed for the checked-in Exp3 table.
- Pre-registration file was not started.
- `LICENSE` and `CITATION.cff` were not started.
