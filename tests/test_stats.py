"""CPU-only unit tests for jspace.stats (run: python tests/test_stats.py)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jspace.stats import (  # noqa: E402
    benjamini_hochberg,
    bootstrap_ci,
    cohens_h,
    holm_adjust,
    mcnemar_exact,
    mcnemar_power,
    mcnemar_sample_size,
    paired_permutation_test,
    permutation_test,
    wilson_ci,
)


def test_bootstrap_ci_mean():
    point, lo, hi = bootstrap_ci(list(range(1, 11)), n_boot=2000, seed=1)
    assert abs(point - 5.5) < 1e-12
    assert lo < 5.5 < hi
    assert hi - lo < 4.0
    # Determinism: same seed -> identical interval.
    _, lo2, hi2 = bootstrap_ci(list(range(1, 11)), n_boot=2000, seed=1)
    assert (lo, hi) == (lo2, hi2)


def test_bootstrap_ci_median():
    point, lo, hi = bootstrap_ci([1, 2, 3, 4, 100], stat="median", n_boot=500, seed=2)
    assert point == 3.0
    assert lo <= 3.0 <= hi


def test_bootstrap_ci_empty_and_nan():
    res = bootstrap_ci([])
    assert all(np.isnan(v) for v in res)
    point, lo, hi = bootstrap_ci([1.0, float("nan"), 3.0], seed=3)
    assert abs(point - 2.0) < 1e-12


def test_permutation_test_null_and_signal():
    rng = np.random.default_rng(7)
    a = rng.normal(0.0, 1.0, 60)
    p_null = permutation_test(a, a + 0.0, n_perm=2000, seed=0)
    assert p_null > 0.3
    p_sig = permutation_test([10.0] * 8, [0.0] * 8, n_perm=2000, seed=0)
    assert p_sig < 0.01


def test_paired_permutation_test():
    x = np.arange(10, dtype=float)
    p_same = paired_permutation_test(x, x.copy(), n_perm=2000, seed=0)
    assert p_same > 0.9
    p_shift = paired_permutation_test(x + 1.0, x, n_perm=5000, seed=0)
    assert p_shift < 0.01


def test_mcnemar_exact_known_values():
    # All discordants in one direction: p = 2 * 0.5^10
    res = mcnemar_exact(0, 10)
    assert abs(res["p_exact"] - 2.0 * 0.5 ** 10) < 1e-12
    # Symmetric discordants: exact p = 1
    assert abs(mcnemar_exact(5, 5)["p_exact"] - 1.0) < 1e-12
    # Known value: b=3, c=12 -> p = 2 * sum_{i<=3} C(15,i) * 0.5^15
    expected = 2.0 * (1 + 15 + 105 + 455) * 0.5 ** 15
    assert abs(mcnemar_exact(3, 12)["p_exact"] - expected) < 1e-12
    # No discordants: undefined (NaN)
    assert np.isnan(mcnemar_exact(0, 0)["p_exact"])


def test_wilson_ci_known_value():
    lo, hi = wilson_ci(5, 10)
    assert abs(lo - 0.2366) < 1e-3
    assert abs(hi - 0.7634) < 1e-3


def test_cohens_h_known_value():
    # h(0.5, 0.25) = 2*(pi/4 - asin(0.5)) ~= 0.5236
    assert abs(cohens_h(0.5, 0.25) - 0.5236) < 1e-3


def test_mcnemar_power_monotone_and_sane():
    # One-sided discordance only (q=1): with n=40 pairs, power ~ P(D>=~9) ~ 1.
    assert mcnemar_power(0.5, 0.0, 40) > 0.99
    # Power grows with n.
    p_small = mcnemar_power(0.2, 0.05, 60)
    p_large = mcnemar_power(0.2, 0.05, 240)
    assert p_small < p_large
    # No true effect (symmetric discordance) -> power <= alpha.
    assert mcnemar_power(0.1, 0.1, 200) <= 0.05 + 1e-9
    # Zero effect probability -> power 0.
    assert mcnemar_power(0.0, 0.0, 100) == 0.0


def test_mcnemar_sample_size_roundtrip():
    n = mcnemar_sample_size(0.3, 0.1, power_target=0.8, n_max=1000)
    assert n is not None and 20 <= n <= 600
    assert mcnemar_power(0.3, 0.1, n) >= 0.8
    assert mcnemar_power(0.3, 0.1, n - 1) < 0.8
    # Impossibly weak effect -> None.
    assert mcnemar_sample_size(0.02, 0.018, power_target=0.8, n_max=200) is None


def test_holm_and_bh_hand_derived_triple() -> None:
    # p = [0.01, 0.04, 0.03], m = 3.
    # Sorted: 0.01 (idx 0), 0.03 (idx 2), 0.04 (idx 1).
    # Holm multipliers 3, 2, 1:
    #   3*0.01 = 0.03
    #   2*0.03 = 0.06
    #   1*0.04 = 0.04
    #   cumulative max: 0.03, 0.06, 0.06
    #   back to input order: [0.03, 0.06, 0.06]
    # BH raw (m/i)*p_(i):
    #   (3/1)*0.01 = 0.03
    #   (3/2)*0.03 = 0.045
    #   (3/3)*0.04 = 0.04
    #   enforce from the right: 0.04, min(0.04, 0.045)=0.04, min(0.04, 0.03)=0.03
    #   back to input order: [0.03, 0.04, 0.04]
    p = [0.01, 0.04, 0.03]
    holm = holm_adjust(p)
    bh = benjamini_hochberg(p)
    assert np.allclose(holm, [0.03, 0.06, 0.06])
    assert np.allclose(bh, [0.03, 0.04, 0.04])


def test_holm_and_bh_nan_is_outside_the_family() -> None:
    # Finite family is [0.01, 0.04], m = 2.
    # Holm: 2*0.01 = 0.02; max(0.02, 1*0.04) = 0.04 → [0.02, nan, 0.04]
    # BH: (2/1)*0.01 = 0.02; (2/2)*0.04 = 0.04; from the right min keeps both.
    p = [0.01, float("nan"), 0.04]
    assert np.allclose(holm_adjust(p), [0.02, np.nan, 0.04], equal_nan=True)
    assert np.allclose(benjamini_hochberg(p), [0.02, np.nan, 0.04], equal_nan=True)


def test_holm_and_bh_clip_at_one_and_identity_for_one_test() -> None:
    # One test: multiplier 1, so the adjusted value equals the raw value.
    assert holm_adjust([0.2]).item() == 0.2
    assert benjamini_hochberg([0.2]).item() == 0.2
    # Two equal 0.6. Holm: 2*0.6 = 1.2, clipped to 1; cumulative max stays 1.
    assert np.allclose(holm_adjust([0.6, 0.6]), [1.0, 1.0])
    # BH from the large end: (2/2)*0.6 = 0.6, then min(0.6, min(1, (2/1)*0.6)) = 0.6.
    assert np.allclose(benjamini_hochberg([0.6, 0.6]), [0.6, 0.6])
    assert holm_adjust([]).size == 0
    assert benjamini_hochberg([float("nan")]).shape == (1,)
    assert np.isnan(benjamini_hochberg([float("nan")])[0])


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    print("ALL PASS" if failures == 0 else f"{failures} FAILURES")
    sys.exit(1 if failures else 0)
