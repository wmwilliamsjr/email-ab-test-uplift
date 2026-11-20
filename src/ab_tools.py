"""Classic A/B testing tools: sample size, sample-ratio check, two-proportion test, Holm, bootstrap.

Every function uses only numpy and scipy so you can see the math.
"""
import math

import numpy as np
from scipy.stats import chisquare, norm


def sample_size_per_group(p_control: float, min_effect: float,
                          alpha: float = 0.05, power: float = 0.80) -> int:
    """Customers needed in EACH group to detect an absolute change of `min_effect`.

    n = (z_alpha/2 + z_beta)^2 * (p1(1-p1) + p2(1-p2)) / (p1 - p2)^2
    """
    p1, p2 = p_control, p_control + min_effect
    z_alpha = norm.ppf(1 - alpha / 2)     # two-sided test
    z_beta = norm.ppf(power)
    n = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p1 - p2) ** 2
    return math.ceil(n)


def srm_check(counts: list[int]) -> float:
    """Sample ratio mismatch for any number of groups that should be equal-sized.
    Returns the chi-square p-value. p < 0.01 is a red flag."""
    total = sum(counts)
    expected = [total / len(counts)] * len(counts)
    return float(chisquare(counts, f_exp=expected).pvalue)


def two_proportion_test(success_a: int, n_a: int, success_b: int, n_b: int,
                        alpha: float = 0.05) -> dict:
    """Two-sided z-test for a difference in proportions (B minus A), with a confidence interval."""
    p_a, p_b = success_a / n_a, success_b / n_b
    diff = p_b - p_a
    pooled = (success_a + success_b) / (n_a + n_b)        # under the null, both groups share one rate
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = diff / se_pooled
    p_value = float(2 * (1 - norm.cdf(abs(z))))
    se = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)   # CI uses each group's own rate
    z_crit = norm.ppf(1 - alpha / 2)
    return {"rate_a": p_a, "rate_b": p_b, "diff": diff, "z": z, "p_value": p_value,
            "ci_low": float(diff - z_crit * se), "ci_high": float(diff + z_crit * se)}


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm correction for several tests. Returns adjusted p-values in the original order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted, running_max = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running_max
    return adjusted


def bootstrap_diff(a, b, n_boot: int = 2000, seed: int = 42) -> np.ndarray:
    """Resample each group with replacement; return n_boot differences in means (b - a).
    Useful for skewed outcomes like spend, where most customers spend $0."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.array([rng.choice(b, b.size).mean() - rng.choice(a, a.size).mean() for _ in range(n_boot)])