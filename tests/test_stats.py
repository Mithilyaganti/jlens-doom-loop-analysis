"""CPU-only unit tests for jspace.stats (run: python tests/test_stats.py)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jspace.stats import (  # noqa: E402
    bootstrap_ci,
    clustered_bootstrap_ci,
    cohens_h,
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


def test_clustered_bootstrap_keeps_rows_of_one_scenario_together():
    ids = np.array(["s0", "s0", "s1", "s1"])

    def stat(idx):
        idx = np.asarray(idx)
        assert int(np.sum(idx == 0)) == int(np.sum(idx == 1))
        assert int(np.sum(idx == 2)) == int(np.sum(idx == 3))
        return float(np.mean(idx))

    first = clustered_bootstrap_ci(ids, stat, n_boot=30, seed=0)
    second = clustered_bootstrap_ci(ids, stat, n_boot=30, seed=0)
    assert first == second
    assert np.isfinite(first["point"])
    assert first["lo"] <= first["point"] <= first["hi"]
    assert first["n_boot_finite"] == 30
    empty = clustered_bootstrap_ci([], stat, n_boot=5, seed=0)
    assert np.isnan(empty["point"])


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
