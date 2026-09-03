"""Valid paired statistical inference with noise floor reporting and power analysis."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


class InsufficientSampleSizeError(ValueError):
    """Raised when the sample size N is below the power-analysis minimum."""


@dataclass(frozen=True)
class PowerAnalysisConfig:
    alpha: float = 0.05
    power: float = 0.80
    min_detectable_effect_d: float = 0.5


def compute_min_sample_size(
    alpha: float = 0.05,
    power: float = 0.80,
    effect_size_d: float = 0.5,
) -> int:
    """Computes minimum sample size for a paired two-sided test under normal approximation.

    Formula: N_min = ceil( ((z_{1 - alpha/2} + z_{power}) / effect_size_d) ^ 2 )
    """
    if effect_size_d <= 0:
        raise ValueError("Effect size Cohen's d must be positive.")

    # Standard normal quantile approximations
    # For alpha=0.05: z ~ 1.95996
    # For power=0.80: z ~ 0.84162; power=0.90: z ~ 1.28155
    z_alpha = _approx_norm_ppf(1.0 - alpha / 2.0)
    z_beta = _approx_norm_ppf(power)

    n_raw = ((z_alpha + z_beta) / effect_size_d) ** 2
    return max(4, math.ceil(n_raw))


def _approx_norm_ppf(p: float) -> float:
    """Rational approximation for inverse standard normal CDF (Acklam's formula)."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("Probability must be in (0, 1)")

    # Coefficients in rational approximations
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )


@dataclass(frozen=True)
class PairedDifferenceInference:
    """Valid statistical inference for paired experimental designs."""

    sample_size: int
    min_sample_size_required: int
    power_adequate: bool
    mean_difference: float
    p95_difference: float
    p99_difference: float
    std_difference: float
    standard_error: float
    t_statistic: float
    ci_95: tuple[float, float]
    noise_floor_estimate_ms: float
    exceeds_noise_floor: bool
    stratified_mean_difference: float | None = None
    stratified_std_error: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "min_sample_size_required": self.min_sample_size_required,
            "power_adequate": self.power_adequate,
            "mean_difference": round(self.mean_difference, 4),
            "p95_difference": round(self.p95_difference, 4),
            "p99_difference": round(self.p99_difference, 4),
            "std_difference": round(self.std_difference, 4),
            "standard_error": round(self.standard_error, 4),
            "t_statistic": round(self.t_statistic, 4),
            "ci_95": [round(self.ci_95[0], 4), round(self.ci_95[1], 4)],
            "noise_floor_estimate_ms": round(self.noise_floor_estimate_ms, 4),
            "exceeds_noise_floor": self.exceeds_noise_floor,
            "stratified_mean_difference": round(self.stratified_mean_difference, 4)
            if self.stratified_mean_difference is not None
            else None,
            "stratified_std_error": round(self.stratified_std_error, 4)
            if self.stratified_std_error is not None
            else None,
        }


def compute_paired_inference(
    baseline: Sequence[float] | np.ndarray,
    candidate: Sequence[float] | np.ndarray,
    clusters: Sequence[str] | None = None,
    alpha: float = 0.05,
    power: float = 0.80,
    min_detectable_effect_d: float = 0.5,
    noise_floor_ms: float = 1.0,
    enforce_min_sample_size: bool = True,
) -> PairedDifferenceInference:
    """Computes paired difference statistics across actual trials.

    Supports:
    - Sample size power check rejecting N < N_min
    - Noise floor reporting alongside p95/p99
    - Stratified sampling clustered by prompt/task type
    """
    b = np.asarray(baseline, dtype=np.float64)
    c = np.asarray(candidate, dtype=np.float64)

    if len(b) != len(c):
        raise ValueError(f"Mismatched paired trial counts: baseline={len(b)} vs candidate={len(c)}")

    n = len(b)
    n_min = compute_min_sample_size(alpha=alpha, power=power, effect_size_d=min_detectable_effect_d)

    if enforce_min_sample_size and n < n_min:
        raise InsufficientSampleSizeError(
            f"Sample size N={n} is below the power-analysis minimum of {n_min} "
            f"(alpha={alpha}, power={power}, effect_size_d={min_detectable_effect_d})."
        )

    power_adequate = n >= n_min

    # Paired differences: baseline - candidate
    diffs = b - c
    mean_diff = float(np.mean(diffs)) if n > 0 else 0.0
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    se_diff = (std_diff / math.sqrt(n)) if n > 0 else 0.0

    t_stat = (mean_diff / se_diff) if se_diff > 0 else 0.0
    ci_95 = (mean_diff - 1.96 * se_diff, mean_diff + 1.96 * se_diff)

    p95_diff = float(np.percentile(b, 95) - np.percentile(c, 95)) if n > 0 else 0.0
    p99_diff = float(np.percentile(b, 99) - np.percentile(c, 99)) if n > 0 else 0.0

    exceeds_noise = abs(mean_diff) > noise_floor_ms

    # Stratified inference if clusters provided
    strat_mean: float | None = None
    strat_se: float | None = None

    if clusters is not None and len(clusters) == n:
        cluster_map: dict[str, list[float]] = defaultdict(list)
        for cl, d in zip(clusters, diffs, strict=True):
            cluster_map[cl].append(d)

        # Neyman / proportional stratified allocation
        cluster_means: list[float] = []
        cluster_vars: list[float] = []
        cluster_weights: list[float] = []

        for _cl, vals in cluster_map.items():
            w_k = len(vals) / n
            mean_k = float(np.mean(vals))
            var_k = float(np.var(vals, ddof=1)) if len(vals) > 1 else 0.0

            cluster_weights.append(w_k)
            cluster_means.append(mean_k)
            cluster_vars.append(var_k / max(1, len(vals)))

        strat_mean = float(sum(w * m for w, m in zip(cluster_weights, cluster_means, strict=True)))
        strat_var = float(sum((w**2) * v for w, v in zip(cluster_weights, cluster_vars, strict=True)))
        strat_se = math.sqrt(strat_var)

    return PairedDifferenceInference(
        sample_size=n,
        min_sample_size_required=n_min,
        power_adequate=power_adequate,
        mean_difference=mean_diff,
        p95_difference=p95_diff,
        p99_difference=p99_diff,
        std_difference=std_diff,
        standard_error=se_diff,
        t_statistic=t_stat,
        ci_95=ci_95,
        noise_floor_estimate_ms=noise_floor_ms,
        exceeds_noise_floor=exceeds_noise,
        stratified_mean_difference=strat_mean,
        stratified_std_error=strat_se,
    )
