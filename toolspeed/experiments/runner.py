"""Statistical summarizer, paired bootstrap confidence intervals, and hypothesis evaluations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from toolspeed.core.types import (
    ArtifactManifest,
    EvidenceLevel,
    LatencyProfile,
    VerdictState,
    sanitize_for_json,
)


class WorkloadFamily(str, Enum):
    """Canonical workload families evaluated across synthetic and empirical benchmarks."""

    W1_FANOUT = "W1: Fan-out Independent Reads"
    W2_CHAINS = "W2: Dependent Sequential Chains"
    W3_BRANCHING = "W3: Branching with Speculative Read"
    W4_REPEATED = "W4: Repeated Workflows (High Plan Locality)"
    W5_LARGE_PAYLOADS = "W5: Large Arguments & High-Volume Responses"
    W6_SANDBOX_COLDSTART = "W6: Cold-Start Sandboxes (Pre-warming vs On-Demand)"
    W7_SIDE_EFFECTS = "W7: Side-Effecting Actions & Approval Gates"


def samples(
    rng_or_profile: Any,
    mean_or_median_or_count: Any = None,
    sigma_or_std: float | None = None,
    shape_or_size: Any = None,
    **kwargs: Any,
) -> np.ndarray:
    """Generate reproducible latency samples supporting all signatures."""
    if isinstance(rng_or_profile, LatencyProfile):
        profile = rng_or_profile
        count = int(mean_or_median_or_count) if mean_or_median_or_count is not None else 100
        seed = int(sigma_or_std) if sigma_or_std is not None else 42
        rng = np.random.default_rng(seed)
        med = getattr(profile, "median_ms", getattr(profile, "tool_ms", 20.0))
        std = getattr(profile, "std_dev_ms", getattr(profile, "sigma", 2.0))
        return np.maximum(1.0, rng.normal(loc=med, scale=std, size=count))

    rng = rng_or_profile if hasattr(rng_or_profile, "normal") else np.random.default_rng(42)
    median_ms = kwargs.get("median_ms", mean_or_median_or_count)
    if median_ms is None:
        median_ms = 20.0
    sigma = kwargs.get("sigma", sigma_or_std)
    if sigma is None:
        sigma = 2.0
    shape = kwargs.get("shape", shape_or_size)
    if shape is None:
        shape = 100

    scale = sigma if sigma > 1.0 else float(median_ms) * float(sigma)
    return np.maximum(1.0, rng.normal(loc=float(median_ms), scale=scale, size=shape))


@dataclass
class HypothesisCheck:
    """Individual hypothesis criteria check evaluated against frozen thresholds."""

    name: str
    target: str
    measured: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FalsificationVerdict:
    """Rigorous scientific hypothesis verdict."""

    experiment_id: str
    hypothesis: str
    passed: bool
    falsified: bool
    summary: str = ""
    state: VerdictState = VerdictState.INCONCLUSIVE
    evidence_level: EvidenceLevel = EvidenceLevel.SYNTHETIC
    checks: list[HypothesisCheck] = field(default_factory=list)
    is_verdict_eligible: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence_log_row: dict[str, Any] = field(default_factory=dict)
    target_claim: str = ""
    p95_speedup: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "hypothesis": self.hypothesis,
            "passed": self.passed,
            "falsified": self.falsified,
            "state": self.state.value if isinstance(self.state, VerdictState) else str(self.state),
            "evidence_level": self.evidence_level.value
            if isinstance(self.evidence_level, EvidenceLevel)
            else str(self.evidence_level),
            "summary": self.summary,
            "checks": [c.to_dict() for c in self.checks],
            "is_verdict_eligible": self.is_verdict_eligible,
            "metadata": self.metadata,
        }


@dataclass
class MetricSummary:
    """Comprehensive paired evaluation statistical summary with honest null defaults."""

    baseline_p50_ms: float | None = None
    candidate_p50_ms: float | None = None
    p50_speedup: float | None = None
    baseline_p90_ms: float | None = None
    candidate_p90_ms: float | None = None
    p90_speedup: float | None = None
    baseline_p95_ms: float | None = None
    candidate_p95_ms: float | None = None
    p95_speedup: float | None = None
    baseline_p99_ms: float | None = None
    candidate_p99_ms: float | None = None
    p99_speedup: float | None = None
    baseline_mean_ms: float | None = None
    candidate_mean_ms: float | None = None
    mean_speedup: float | None = None
    baseline_success_rate: float | None = None
    candidate_success_rate: float | None = None
    success_rate_delta: float | None = None
    paired_success_counts: dict[str, int] = field(default_factory=dict)
    wasted_call_rate: float | None = None
    cost_multiplier: float | None = None
    rate_limit_error_rate: float | None = None
    input_tokens_baseline: float | None = None
    input_tokens_candidate: float | None = None
    token_reduction_pct: float | None = None
    deopt_rate: float | None = None
    tool_start_p50_ms: float | None = None
    tool_start_p95_ms: float | None = None
    tool_start_speedup_p95: float | None = None
    semantic_mutation_rate: float | None = None
    p95_reduction_ci: tuple[float | None, float | None] | None = None
    p50_reduction_ci: tuple[float | None, float | None] | None = None
    p99_reduction_ci: tuple[float | None, float | None] | None = None
    success_delta_ci: tuple[float | None, float | None] | None = None
    cost_delta_ci: tuple[float | None, float | None] | None = None
    unapproved_side_effects: int = 0
    tool_selection_accuracy: float | None = None
    arg_selection_accuracy: float | None = None
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return sanitize_for_json(asdict(self))


@dataclass
class ExperimentResult:
    """Top-level container for an experiment evaluation."""

    experiment_id: str
    title: str
    parameter_name: str
    rows: list[dict[str, Any]]
    verdict: FalsificationVerdict
    evidence_level: EvidenceLevel = EvidenceLevel.SYNTHETIC
    runtime_sec: float = 0.0
    manifest: ArtifactManifest | None = None
    workloads: list[Any] = field(default_factory=list)
    evaluations: list[Any] = field(default_factory=list)
    trials: int = 1000
    seed: int = 42
    profile: LatencyProfile | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "title": self.title,
            "evidence_level": self.evidence_level.value
            if isinstance(self.evidence_level, EvidenceLevel)
            else str(self.evidence_level),
            "parameter_name": self.parameter_name,
            "rows": sanitize_for_json(self.rows),
            "verdict": self.verdict.to_dict(),
            "runtime_sec": self.runtime_sec,
            "manifest": self.manifest.to_dict() if self.manifest else None,
            "workloads": self.workloads,
            "evaluations": self.evaluations,
            "trials": self.trials,
            "seed": self.seed,
            "profile": self.profile.to_dict() if self.profile else None,
        }

    def get_row(self, param_value: Any) -> dict[str, Any] | None:
        for r in self.rows:
            if r.get(self.parameter_name) == param_value:
                return r
        return None


def compute_percentiles(arr: np.ndarray, percentiles: list[float] | None = None) -> dict[str, float]:
    """Compute percentile dictionary from array."""
    pcts = percentiles or [50.0, 90.0, 95.0, 99.0]
    if len(arr) == 0:
        return {f"p{int(p)}": 0.0 for p in pcts}
    return {f"p{int(p)}": float(np.percentile(arr, p)) for p in pcts}


def bootstrap_confidence_interval(
    data: np.ndarray,
    stat_func: Callable[[np.ndarray], float],
    num_samples: int = 2000,
    ci: float = 0.95,
    seed: int | None = 42,
) -> tuple[float, float]:
    """Calculate univariate bootstrap confidence interval."""
    if len(data) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_stats = np.empty(num_samples)
    for i in range(num_samples):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = stat_func(sample)
    lower = float(np.percentile(boot_stats, ((1.0 - ci) / 2.0) * 100.0))
    upper = float(np.percentile(boot_stats, (1.0 - (1.0 - ci) / 2.0) * 100.0))
    return lower, upper


def paired_bootstrap_ci(
    baseline: np.ndarray,
    candidate: np.ndarray,
    stat_func: Callable[[np.ndarray, np.ndarray], float],
    num_samples: int = 2000,
    ci: float = 0.95,
    seed: int | None = 42,
) -> tuple[float | None, float | None]:
    """Generic paired bootstrap confidence interval function."""
    n = min(len(baseline), len(candidate))
    if n < 5:
        return None, None

    rng = np.random.default_rng(seed)
    boot_stats = np.empty(num_samples)

    for i in range(num_samples):
        idx = rng.choice(n, size=n, replace=True)
        b_sample = baseline[idx]
        c_sample = candidate[idx]
        boot_stats[i] = stat_func(b_sample, c_sample)

    lower_pct = ((1.0 - ci) / 2.0) * 100.0
    upper_pct = (1.0 - (1.0 - ci) / 2.0) * 100.0
    return float(np.percentile(boot_stats, lower_pct)), float(np.percentile(boot_stats, upper_pct))


def paired_bootstrap_p95_ci(
    baseline: np.ndarray,
    candidate: np.ndarray,
    num_samples: int = 2000,
    ci: float = 0.95,
    seed: int | None = 42,
) -> tuple[float | None, float | None]:
    """Paired bootstrap confidence interval for P95 reduction percentage."""

    def _calc_p95_red(b: np.ndarray, c: np.ndarray) -> float:
        b_p95 = float(np.percentile(b, 95))
        c_p95 = float(np.percentile(c, 95))
        if b_p95 > 0:
            return ((b_p95 - c_p95) / b_p95) * 100.0
        return 0.0

    return paired_bootstrap_ci(baseline, candidate, _calc_p95_red, num_samples=num_samples, ci=ci, seed=seed)


def compute_summary(
    baseline: np.ndarray,
    candidate: np.ndarray,
    baseline_success: np.ndarray | None = None,
    candidate_success: np.ndarray | None = None,
    wasted_calls: np.ndarray | None = None,
    cost_multipliers: np.ndarray | None = None,
    rate_limit_errors: np.ndarray | None = None,
    input_tokens_base: float | None = None,
    input_tokens_cand: float | None = None,
    deopt_events: np.ndarray | None = None,
    tool_start_base: np.ndarray | None = None,
    tool_start_cand: np.ndarray | None = None,
    semantic_mutations: np.ndarray | None = None,
    unapproved_side_effects: int = 0,
    tool_selection_accuracy: float | None = None,
    arg_selection_accuracy: float | None = None,
    extra: dict[str, Any] | None = None,
) -> MetricSummary:
    """Compute comprehensive MetricSummary from paired trial arrays.

    Strict CCL invariant: Primary paired CCL percentiles and speedups are computed strictly
    over the population where BOTH baseline AND candidate succeeded (both_succeeded pairs).
    Missing metrics default strictly to None.
    """
    b_succ = float(np.mean(baseline_success)) if baseline_success is not None and len(baseline_success) > 0 else None
    c_succ = float(np.mean(candidate_success)) if candidate_success is not None and len(candidate_success) > 0 else None
    succ_delta = float(c_succ - b_succ) if (c_succ is not None and b_succ is not None) else None

    paired_counts: dict[str, int] = {}
    both_succeeded_mask: np.ndarray | None = None

    if (
        baseline_success is not None
        and candidate_success is not None
        and len(baseline_success) == len(candidate_success)
    ):
        bs = np.asarray(baseline_success, dtype=bool)
        cs = np.asarray(candidate_success, dtype=bool)
        both_succeeded_mask = bs & cs
        paired_counts = {
            "both_succeeded": int(np.sum(both_succeeded_mask)),
            "candidate_only": int(np.sum(~bs & cs)),
            "baseline_only": int(np.sum(bs & ~cs)),
            "both_failed": int(np.sum(~bs & ~cs)),
        }

    # Strict CCL filtering: compute paired latencies over trials where both succeeded
    if both_succeeded_mask is not None and np.sum(both_succeeded_mask) > 0:
        valid_baseline = baseline[both_succeeded_mask]
        valid_candidate = candidate[both_succeeded_mask]
    elif baseline_success is None and candidate_success is None:
        valid_baseline = baseline
        valid_candidate = candidate
    else:
        valid_baseline = np.array([], dtype=np.float64)
        valid_candidate = np.array([], dtype=np.float64)

    b50: float | None = None
    b90: float | None = None
    b95: float | None = None
    b99: float | None = None
    b_mean: float | None = None
    if len(valid_baseline) > 0:
        b50 = float(np.percentile(valid_baseline, 50))
        b90 = float(np.percentile(valid_baseline, 90))
        b95 = float(np.percentile(valid_baseline, 95))
        b99 = float(np.percentile(valid_baseline, 99))
        b_mean = float(np.mean(valid_baseline))

    c50: float | None = None
    c90: float | None = None
    c95: float | None = None
    c99: float | None = None
    c_mean: float | None = None
    if len(valid_candidate) > 0:
        c50 = float(np.percentile(valid_candidate, 50))
        c90 = float(np.percentile(valid_candidate, 90))
        c95 = float(np.percentile(valid_candidate, 95))
        c99 = float(np.percentile(valid_candidate, 99))
        c_mean = float(np.mean(valid_candidate))

    p50_speedup = float(b50 / c50) if (b50 is not None and c50 is not None and c50 > 0) else None
    p90_speedup = float(b90 / c90) if (b90 is not None and c90 is not None and c90 > 0) else None
    p95_speedup = float(b95 / c95) if (b95 is not None and c95 is not None and c95 > 0) else None
    p99_speedup = float(b99 / c99) if (b99 is not None and c99 is not None and c99 > 0) else None
    mean_speedup = float(b_mean / c_mean) if (b_mean is not None and c_mean is not None and c_mean > 0) else None

    wasted_rate = float(np.mean(wasted_calls)) if wasted_calls is not None and len(wasted_calls) > 0 else None
    cost_mult = float(np.mean(cost_multipliers)) if cost_multipliers is not None and len(cost_multipliers) > 0 else None
    rl_err_rate = (
        float(np.mean(rate_limit_errors)) if rate_limit_errors is not None and len(rate_limit_errors) > 0 else None
    )

    token_red = (
        float((input_tokens_base - input_tokens_cand) / input_tokens_base * 100.0)
        if (input_tokens_base is not None and input_tokens_cand is not None and input_tokens_base > 0)
        else None
    )
    deopt_rate_val = float(np.mean(deopt_events)) if deopt_events is not None and len(deopt_events) > 0 else None

    t_start_p50 = (
        float(np.percentile(tool_start_cand, 50)) if tool_start_cand is not None and len(tool_start_cand) > 0 else None
    )
    t_start_p95 = (
        float(np.percentile(tool_start_cand, 95)) if tool_start_cand is not None and len(tool_start_cand) > 0 else None
    )
    if (
        tool_start_base is not None
        and tool_start_cand is not None
        and len(tool_start_base) > 0
        and len(tool_start_cand) > 0
    ):
        t_base_p95 = float(np.percentile(tool_start_base, 95))
        t_start_speedup = float(t_base_p95 / t_start_p95) if (t_start_p95 is not None and t_start_p95 > 0) else 1.0
    else:
        t_start_speedup = None

    mutation_rate = (
        float(np.mean(semantic_mutations)) if semantic_mutations is not None and len(semantic_mutations) > 0 else None
    )

    # Paired bootstrap CIs with 2,000 samples over valid population
    if len(valid_baseline) >= 5 and len(valid_candidate) >= 5:
        ci_p95_low, ci_p95_high = paired_bootstrap_p95_ci(valid_baseline, valid_candidate, num_samples=2000)
        p95_ci = (ci_p95_low, ci_p95_high) if (ci_p95_low is not None and ci_p95_high is not None) else None

        def _calc_p50_red(b: np.ndarray, c: np.ndarray) -> float:
            b5 = float(np.percentile(b, 50))
            c5 = float(np.percentile(c, 50))
            return ((b5 - c5) / b5) * 100.0 if b5 > 0 else 0.0

        ci_p50_low, ci_p50_high = paired_bootstrap_ci(valid_baseline, valid_candidate, _calc_p50_red, num_samples=2000)
        p50_ci = (ci_p50_low, ci_p50_high) if (ci_p50_low is not None and ci_p50_high is not None) else None

        def _calc_p99_red(b: np.ndarray, c: np.ndarray) -> float:
            b9 = float(np.percentile(b, 99))
            c9 = float(np.percentile(c, 99))
            return ((b9 - c9) / b9) * 100.0 if b9 > 0 else 0.0

        ci_p99_low, ci_p99_high = paired_bootstrap_ci(valid_baseline, valid_candidate, _calc_p99_red, num_samples=2000)
        p99_ci = (ci_p99_low, ci_p99_high) if (ci_p99_low is not None and ci_p99_high is not None) else None
    else:
        p95_ci = None
        p50_ci = None
        p99_ci = None

    def _calc_succ_delta(b: np.ndarray, c: np.ndarray) -> float:
        return float(np.mean(c) - np.mean(b))

    if baseline_success is not None and candidate_success is not None and len(baseline_success) >= 5:
        ci_succ_low, ci_succ_high = paired_bootstrap_ci(
            baseline_success.astype(float), candidate_success.astype(float), _calc_succ_delta, num_samples=2000
        )
        succ_ci = (ci_succ_low, ci_succ_high) if (ci_succ_low is not None and ci_succ_high is not None) else None
    else:
        succ_ci = None

    return MetricSummary(
        baseline_p50_ms=b50,
        candidate_p50_ms=c50,
        p50_speedup=p50_speedup,
        baseline_p90_ms=b90,
        candidate_p90_ms=c90,
        p90_speedup=p90_speedup,
        baseline_p95_ms=b95,
        candidate_p95_ms=c95,
        p95_speedup=p95_speedup,
        baseline_p99_ms=b99,
        candidate_p99_ms=c99,
        p99_speedup=p99_speedup,
        baseline_mean_ms=b_mean,
        candidate_mean_ms=c_mean,
        mean_speedup=mean_speedup,
        baseline_success_rate=b_succ,
        candidate_success_rate=c_succ,
        success_rate_delta=succ_delta,
        paired_success_counts=paired_counts,
        wasted_call_rate=wasted_rate,
        cost_multiplier=cost_mult,
        rate_limit_error_rate=rl_err_rate,
        input_tokens_baseline=input_tokens_base,
        input_tokens_candidate=input_tokens_cand,
        token_reduction_pct=token_red,
        deopt_rate=deopt_rate_val,
        tool_start_p50_ms=t_start_p50,
        tool_start_p95_ms=t_start_p95,
        tool_start_speedup_p95=t_start_speedup,
        semantic_mutation_rate=mutation_rate,
        p95_reduction_ci=p95_ci,
        p50_reduction_ci=p50_ci,
        p99_reduction_ci=p99_ci,
        success_delta_ci=succ_ci,
        cost_delta_ci=None,
        unapproved_side_effects=unapproved_side_effects,
        tool_selection_accuracy=tool_selection_accuracy,
        arg_selection_accuracy=arg_selection_accuracy,
        extra_metrics=extra or {},
    )
