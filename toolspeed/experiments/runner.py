"""Core experiment framework and statistical computation for ToolSpeed."""

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
    """Seven canonical workload families from research plan."""
    W1_FANOUT = "W1: Independent fan-out reads"
    W2_CHAINS = "W2: Deterministic dependent chains"
    W3_BRANCHING = "W3: Branching workflows"
    W4_REPEATED = "W4: Repeated workflows with plan locality"
    W5_LARGE_PAYLOADS = "W5: Large tool arguments and results"
    W6_SANDBOX_COLDSTART = "W6: Cold-start code/browser sandboxes"
    W7_SIDE_EFFECTS = "W7: Side-effecting actions requiring approval"


@dataclass
class HypothesisCheck:
    """Individual hypothesis criterion check."""
    name: str
    target: str
    measured: Any
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return sanitize_for_json(asdict(self))


@dataclass
class FalsificationVerdict:
    """Verdict of scientific hypothesis testing and falsification."""
    experiment_id: str
    hypothesis: str
    passed: bool
    falsified: bool
    summary: str
    checks: list[HypothesisCheck] = field(default_factory=list)
    state: VerdictState = VerdictState.INCONCLUSIVE
    evidence_level: EvidenceLevel = EvidenceLevel.SYNTHETIC
    evidence_log_row: dict[str, str] = field(default_factory=dict)
    is_verdict_eligible: bool = True

    @property
    def inconclusive(self) -> bool:
        return self.state == VerdictState.INCONCLUSIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "hypothesis": self.hypothesis,
            "passed": self.passed,
            "falsified": self.falsified,
            "state": self.state.value if isinstance(self.state, VerdictState) else str(self.state),
            "evidence_level": self.evidence_level.value if isinstance(self.evidence_level, EvidenceLevel) else str(self.evidence_level),
            "summary": self.summary,
            "checks": [c.to_dict() for c in self.checks],
            "evidence_log_row": self.evidence_log_row,
            "is_verdict_eligible": self.is_verdict_eligible,
        }


@dataclass
class MetricSummary:
    """Rigorous statistical summary of paired baseline vs candidate execution."""
    baseline_p50_ms: float | None
    candidate_p50_ms: float | None
    p50_speedup: float | None
    baseline_p90_ms: float | None
    candidate_p90_ms: float | None
    p90_speedup: float | None
    baseline_p95_ms: float | None
    candidate_p95_ms: float | None
    p95_speedup: float | None
    baseline_p99_ms: float | None
    candidate_p99_ms: float | None
    p99_speedup: float | None
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
    tool_selection_accuracy: float = 1.0
    arg_selection_accuracy: float = 1.0
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return sanitize_for_json(asdict(self))


@dataclass
class ExperimentResult:
    """Complete result container for an experiment."""
    experiment_id: str
    title: str
    workloads: list[str]
    trials: int
    seed: int
    profile: LatencyProfile
    parameter_name: str
    rows: list[dict[str, Any]]
    verdict: FalsificationVerdict
    evidence_level: EvidenceLevel = EvidenceLevel.SYNTHETIC
    runtime_sec: float = 0.0
    manifest: ArtifactManifest | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "title": self.title,
            "workloads": self.workloads,
            "trials": self.trials,
            "seed": self.seed,
            "evidence_level": self.evidence_level.value if isinstance(self.evidence_level, EvidenceLevel) else str(self.evidence_level),
            "profile": self.profile.to_dict(),
            "parameter_name": self.parameter_name,
            "rows": sanitize_for_json(self.rows),
            "verdict": self.verdict.to_dict(),
            "runtime_sec": self.runtime_sec,
            "manifest": self.manifest.to_dict() if self.manifest else None,
        }

    def get_row(self, param_value: Any) -> dict[str, Any] | None:
        for r in self.rows:
            if r.get(self.parameter_name) == param_value:
                return r
        return None


def paired_bootstrap_ci(
    baseline: np.ndarray,
    candidate: np.ndarray,
    stat_func: Callable[[np.ndarray, np.ndarray], float],
    num_samples: int = 1000,
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
    num_samples: int = 1000,
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
    tool_selection_accuracy: float = 1.0,
    arg_selection_accuracy: float = 1.0,
    extra: dict[str, Any] | None = None,
) -> MetricSummary:
    """Compute comprehensive MetricSummary from paired trial arrays.
    
    Strict CCL invariant: CCL percentiles are computed on trials where tasks completed successfully.
    """
    b_succ = float(np.mean(baseline_success)) if baseline_success is not None and len(baseline_success) > 0 else 1.0
    c_succ = float(np.mean(candidate_success)) if candidate_success is not None and len(candidate_success) > 0 else 1.0
    succ_delta = float(c_succ - b_succ)

    paired_counts: dict[str, int] = {}
    if baseline_success is not None and candidate_success is not None and len(baseline_success) == len(candidate_success):
        bs = np.asarray(baseline_success, dtype=bool)
        cs = np.asarray(candidate_success, dtype=bool)
        paired_counts = {
            "both_succeeded": int(np.sum(bs & cs)),
            "candidate_only": int(np.sum(~bs & cs)),
            "baseline_only": int(np.sum(bs & ~cs)),
            "both_failed": int(np.sum(~bs & ~cs)),
        }

    # Filter for CCL (Correct Completion Latency)
    if baseline_success is not None and len(baseline_success) == len(baseline):
        valid_baseline = baseline[baseline_success.astype(bool)]
    else:
        valid_baseline = baseline

    if candidate_success is not None and len(candidate_success) == len(candidate):
        valid_candidate = candidate[candidate_success.astype(bool)]
    else:
        valid_candidate = candidate

    if len(valid_baseline) > 0:
        b50 = float(np.percentile(valid_baseline, 50))
        b90 = float(np.percentile(valid_baseline, 90))
        b95 = float(np.percentile(valid_baseline, 95))
        b99 = float(np.percentile(valid_baseline, 99))
        b_mean = float(np.mean(valid_baseline))
    else:
        b50 = b90 = b95 = b99 = b_mean = None

    if len(valid_candidate) > 0:
        c50 = float(np.percentile(valid_candidate, 50))
        c90 = float(np.percentile(valid_candidate, 90))
        c95 = float(np.percentile(valid_candidate, 95))
        c99 = float(np.percentile(valid_candidate, 99))
        c_mean = float(np.mean(valid_candidate))
    else:
        c50 = c90 = c95 = c99 = c_mean = None

    p50_speedup = float(b50 / c50) if (b50 is not None and c50 is not None and c50 > 0) else None
    p90_speedup = float(b90 / c90) if (b90 is not None and c90 is not None and c90 > 0) else None
    p95_speedup = float(b95 / c95) if (b95 is not None and c95 is not None and c95 > 0) else None
    p99_speedup = float(b99 / c99) if (b99 is not None and c99 is not None and c99 > 0) else None
    mean_speedup = float(b_mean / c_mean) if (b_mean is not None and c_mean is not None and c_mean > 0) else None

    wasted_rate = float(np.mean(wasted_calls)) if wasted_calls is not None and len(wasted_calls) > 0 else 0.0
    cost_mult = float(np.mean(cost_multipliers)) if cost_multipliers is not None and len(cost_multipliers) > 0 else 1.0
    rl_err_rate = float(np.mean(rate_limit_errors)) if rate_limit_errors is not None and len(rate_limit_errors) > 0 else 0.0

    token_red = (
        float((input_tokens_base - input_tokens_cand) / input_tokens_base * 100.0)
        if (input_tokens_base is not None and input_tokens_cand is not None and input_tokens_base > 0)
        else 0.0
    )
    deopt_rate_val = float(np.mean(deopt_events)) if deopt_events is not None and len(deopt_events) > 0 else 0.0

    t_start_p50 = float(np.percentile(tool_start_cand, 50)) if tool_start_cand is not None and len(tool_start_cand) > 0 else None
    t_start_p95 = float(np.percentile(tool_start_cand, 95)) if tool_start_cand is not None and len(tool_start_cand) > 0 else None
    if tool_start_base is not None and tool_start_cand is not None and len(tool_start_base) > 0 and len(tool_start_cand) > 0:
        t_base_p95 = float(np.percentile(tool_start_base, 95))
        t_start_speedup = float(t_base_p95 / t_start_p95) if (t_start_p95 is not None and t_start_p95 > 0) else 1.0
    else:
        t_start_speedup = None

    mutation_rate = float(np.mean(semantic_mutations)) if semantic_mutations is not None and len(semantic_mutations) > 0 else 0.0

    # Paired bootstrap CIs
    ci_p95_low, ci_p95_high = paired_bootstrap_p95_ci(baseline, candidate, num_samples=500)
    p95_ci = (ci_p95_low, ci_p95_high) if (ci_p95_low is not None and ci_p95_high is not None) else None

    def _calc_p50_red(b: np.ndarray, c: np.ndarray) -> float:
        b5 = float(np.percentile(b, 50))
        c5 = float(np.percentile(c, 50))
        return ((b5 - c5) / b5) * 100.0 if b5 > 0 else 0.0

    ci_p50_low, ci_p50_high = paired_bootstrap_ci(baseline, candidate, _calc_p50_red, num_samples=500)
    p50_ci = (ci_p50_low, ci_p50_high) if (ci_p50_low is not None and ci_p50_high is not None) else None

    def _calc_p99_red(b: np.ndarray, c: np.ndarray) -> float:
        b9 = float(np.percentile(b, 99))
        c9 = float(np.percentile(c, 99))
        return ((b9 - c9) / b9) * 100.0 if b9 > 0 else 0.0

    ci_p99_low, ci_p99_high = paired_bootstrap_ci(baseline, candidate, _calc_p99_red, num_samples=500)
    p99_ci = (ci_p99_low, ci_p99_high) if (ci_p99_low is not None and ci_p99_high is not None) else None

    def _calc_succ_delta(b: np.ndarray, c: np.ndarray) -> float:
        return float(np.mean(c) - np.mean(b))

    if baseline_success is not None and candidate_success is not None:
        ci_succ_low, ci_succ_high = paired_bootstrap_ci(baseline_success.astype(float), candidate_success.astype(float), _calc_succ_delta, num_samples=500)
        succ_ci = (ci_succ_low, ci_succ_high) if (ci_succ_low is not None and ci_succ_high is not None) else None
    else:
        succ_ci = (0.0, 0.0)

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
        cost_delta_ci=(0.0, 0.0),
        unapproved_side_effects=unapproved_side_effects,
        tool_selection_accuracy=tool_selection_accuracy,
        arg_selection_accuracy=arg_selection_accuracy,
        extra_metrics=extra or {},
    )


def samples(rng: np.random.Generator, median_ms: float, sigma: float, shape: int | tuple[int, ...]) -> np.ndarray:
    """Generate log-normal latency samples centered around median_ms."""
    return rng.lognormal(np.log(max(1.0, median_ms)), sigma, shape)


def compute_percentiles(arr: np.ndarray, percentiles: tuple[int, ...] = (50, 90, 95, 99)) -> dict[str, float]:
    """Compute percentiles robustly."""
    if len(arr) == 0:
        return {f"p{p}": 0.0 for p in percentiles}
    res = {}
    for p in percentiles:
        res[f"p{p}"] = float(np.percentile(arr, p))
    return res


def bootstrap_confidence_interval(
    data: np.ndarray,
    stat_func: Callable[[np.ndarray], float] = np.mean,
    num_samples: int = 1000,
    ci: float = 0.95,
    seed: int | None = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a 1D array."""
    if len(data) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_stats = np.empty(num_samples)
    for i in range(num_samples):
        boot_sample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = stat_func(boot_sample)
    lower_pct = ((1.0 - ci) / 2.0) * 100.0
    upper_pct = (1.0 - (1.0 - ci) / 2.0) * 100.0
    return float(np.percentile(boot_stats, lower_pct)), float(np.percentile(boot_stats, upper_pct))

