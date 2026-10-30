"""Power for a difference in doom-loop rates.

The calculation is the usual equal-n two-proportion z formula. It plans a
study. It does not estimate a loop rate: no generations have been run.

The default base rate 0.229 is Liquid's published Qwen3.5-4B figure on
their Antidoom prompt mix, used only as a planning prior. It is not a
measurement on MMLU-ProX or MGSM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Peter J. Acklam's inverse-normal approximation (public-domain coefficients).
_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_P_LOW = 0.02425
_P_HIGH = 1 - _P_LOW

LIQUID_QWEN35_4B_PRIOR = 0.229
PLANNED_N_PER_LANGUAGE = 300
# EN vs HI, EN vs TE, EN vs romanized Telugu. Tenglish has no prompts yet.
PLANNED_COMPARISONS = 3


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("probability must be in (0, 1)")
    if p < _P_LOW:
        q = math.sqrt(-2 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1
        )
    if p > _P_HIGH:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / (
        ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1
    )


def cohens_h(p1: float, p2: float) -> float:
    if not 0.0 < p1 < 1.0 or not 0.0 < p2 < 1.0:
        raise ValueError("proportions must be in (0, 1)")
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def _validate_pair(p_base: float, p_alt: float) -> None:
    if not 0.0 < p_base < 1.0 or not 0.0 < p_alt < 1.0:
        raise ValueError("proportions must be in (0, 1)")
    if p_base == p_alt:
        raise ValueError("proportions are identical")


def n_per_group(
    p_base: float,
    p_alt: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Equal sample size per language for a two-sided two-proportion z test."""
    _validate_pair(p_base, p_alt)
    if not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise ValueError("alpha and power must be in (0, 1)")
    z_alpha = norm_ppf(1 - alpha / 2)
    z_beta = norm_ppf(power)
    variance = p_base * (1 - p_base) + p_alt * (1 - p_alt)
    return (z_alpha + z_beta) ** 2 * variance / (p_base - p_alt) ** 2


def power_at_n(
    p_base: float,
    p_alt: float,
    n: float,
    *,
    alpha: float = 0.05,
) -> float:
    """Two-sided power at ``n`` items per language, matching ``n_per_group``."""
    _validate_pair(p_base, p_alt)
    if n <= 0:
        raise ValueError("n must be positive")
    z_alpha = norm_ppf(1 - alpha / 2)
    standard_error = math.sqrt((p_base * (1 - p_base) + p_alt * (1 - p_alt)) / n)
    z = abs(p_base - p_alt) / standard_error
    # The far tail of a two-sided test is negligible but kept so the formula is complete.
    return norm_cdf(z - z_alpha) + norm_cdf(-z - z_alpha)


def minimum_detectable_delta(
    p_base: float,
    n: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Smallest absolute rate difference with the requested power, at this n."""
    if not 0.0 < p_base < 1.0:
        raise ValueError("base rate must be in (0, 1)")
    low = 1e-4
    high = min(p_base, 1 - p_base) - 1e-4
    if high <= low:
        raise ValueError("base rate is too close to 0 or 1 to search for a delta")
    if power_at_n(p_base, p_base + high, n, alpha=alpha) < power:
        raise ValueError("even the largest delta does not reach the requested power")
    for _ in range(60):
        mid = (low + high) / 2
        if power_at_n(p_base, p_base + mid, n, alpha=alpha) >= power:
            high = mid
        else:
            low = mid
    return high


@dataclass(frozen=True)
class PowerRow:
    p_base: float
    delta: float
    p_alt: float
    alpha: float
    comparisons: int
    n_for_80: float
    power_at_planned_n: float
    mde_at_planned_n: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "p_base": self.p_base,
            "delta": self.delta,
            "p_alt": self.p_alt,
            "alpha": self.alpha,
            "comparisons": self.comparisons,
            "n_for_80": self.n_for_80,
            "power_at_planned_n": self.power_at_planned_n,
            "mde_at_planned_n": self.mde_at_planned_n,
        }


def analysis_table(
    *,
    base_rates: tuple[float, ...] = (0.10, LIQUID_QWEN35_4B_PRIOR, 0.30),
    deltas: tuple[float, ...] = (0.05, 0.10, 0.15),
    planned_n: int = PLANNED_N_PER_LANGUAGE,
    comparisons: int = PLANNED_COMPARISONS,
    alpha: float = 0.05,
) -> list[PowerRow]:
    """Rows for uncorrected alpha and for Bonferroni across ``comparisons``."""
    rows: list[PowerRow] = []
    alphas = [alpha]
    if comparisons > 1:
        alphas.append(alpha / comparisons)
    for p_base in base_rates:
        mde_cache: dict[float, float] = {}
        for used_alpha in alphas:
            mde_cache[used_alpha] = minimum_detectable_delta(p_base, planned_n, alpha=used_alpha, power=0.8)
        for delta in deltas:
            p_alt = p_base + delta
            if not 0.0 < p_alt < 1.0:
                continue
            for used_alpha in alphas:
                rows.append(
                    PowerRow(
                        p_base=p_base,
                        delta=delta,
                        p_alt=p_alt,
                        alpha=used_alpha,
                        comparisons=1 if used_alpha == alpha else comparisons,
                        n_for_80=n_per_group(p_base, p_alt, alpha=used_alpha, power=0.8),
                        power_at_planned_n=power_at_n(p_base, p_alt, planned_n, alpha=used_alpha),
                        mde_at_planned_n=mde_cache[used_alpha],
                    )
                )
    return rows


def format_table(rows: list[PowerRow]) -> str:
    header = (
        "p_base  delta  alpha   comparisons  n_for_80%  power_at_300  mde_at_300"
    )
    lines = [
        header,
        "The 0.229 base rate is Liquid's Antidoom-mix figure for Qwen3.5-4B, a planning prior only.",
    ]
    for row in rows:
        lines.append(
            f"{row.p_base:6.3f}  {row.delta:5.2f}  {row.alpha:7.4f}  "
            f"{row.comparisons:11d}  {row.n_for_80:9.1f}  {row.power_at_planned_n:12.3f}  "
            f"{row.mde_at_planned_n:10.3f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    print(format_table(analysis_table()))


if __name__ == "__main__":
    main()
