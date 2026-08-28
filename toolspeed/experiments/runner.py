"""Core experiment framework and statistical computation for ToolSpeed."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import time

import numpy as np

from toolspeed.core.types import (
    EvidenceLevel,
    VerdictState,
    ArtifactManifest,
    LatencyProfile,
    sanitize_for_json,
    strict_json_dumps,
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

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_for_json(asdict(self))


@dataclass
class FalsificationVerdict:
    """Verdict of scientific hypothesis testing and falsification."""
    experiment_id: str
    hypothesis: str
    passed: bool
    falsified: bool
    summary: str
    checks: List[HypothesisCheck] = field(default_factory=list)
    state: VerdictState = VerdictState.INCONCLUSIVE
    evidence_level: EvidenceLevel = EvidenceLevel.SYNTHETIC
    evidence_log_row: Dict[str, str] = field(default_factory=dict)

    @property
    def inconclusive(self) -> bool:
        return self.state == VerdictState.INCONCLUSIVE

    def to_dict(self) -> Dict[str, Any]:
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
        }


@dataclass
class MetricSummary:
    """Rigorous statistical summary of paired baseline vs candidate execution."""
    baseline_p50_ms: Optional[float]
    candidate_p50_ms: Optional[float]
    p50_speedup: Optional[float]
    baseline_p90_ms: Optional[float]
    candidate_p90_ms: Optional[float]
    p90_speedup: Optional[float]
    baseline_p95_ms: Optional[float]
    candidate_p95_ms: Optional[float]
    p95_speedup: Optional[float]
    baseline_p99_ms: Optional[float]
    candidate_p99_ms: Optional[float]
    p99_speedup: Optional[float]
    baseline_mean_ms: Optional[float] = None
    candidate_mean_ms: Optional[float] = None
    mean_speedup: Optional[float] = None
    baseline_success_rate: Optional[float] = None
    candidate_success_rate: Optional[float] = None
    success_rate_delta: Optional[float] = None
    paired_success_counts: Dict[str, int] = field(default_factory=dict)
    wasted_call_rate: Optional[float] = None
    cost_multiplier: Optional[float] = None
    rate_limit_error_rate: Optional[float] = None
    input_tokens_baseline: Optional[float] = None
    input_tokens_candidate: Optional[float] = None
    token_reduction_pct: Optional[float] = None
    deopt_rate: Optional[float] = None
    tool_start_p50_ms: Optional[float] = None
    tool_start_p95_ms: Optional[float] = None
    tool_start_speedup_p95: Optional[float] = None
    semantic_mutation_rate: Optional[float] = None
    p95_reduction_ci: Optional[Tuple[Optional[float], Optional[float]]] = None
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_for_json(asdict(self))


@dataclass
class ExperimentResult:
    """Complete result container for an experiment."""
    experiment_id: str
    title: str
    workloads: List[str]
    trials: int
    seed: int
    profile: LatencyProfile
    parameter_name: str
    rows: List[Dict[str, Any]]
    verdict: FalsificationVerdict
    evidence_level: EvidenceLevel = EvidenceLevel.SYNTHETIC
    runtime_sec: float = 0.0
    manifest: Optional[ArtifactManifest] = None

    def to_dict(self) -> Dict[str, Any]:
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

    def get_row(self, param_value: Any) -> Optional[Dict[str, Any]]:
        for r in self.rows:
            if r.get(self.parameter_name) == param_value:
                return r
        return None


def samples(rng: np.random.Generator, median_ms: float, sigma: float, shape: Union[int, Tuple[int, ...]]) -> np.ndarray:
    """Generate log-normal latency samples centered around median_ms."""
    return rng.lognormal(np.log(max(1.0, median_ms)), sigma, shape)


def compute_percentiles(arr: np.ndarray, percentiles: Tuple[int, ...] = (50, 90, 95, 99)) -> Dict[str, float]:
    """Compute percentiles robustly."""
    if len(arr) == 0:
        return {f"p{p}": 0.0 for p in percentiles}
    res = {}
    for p in percentiles:
        res[f"p{p}"] = float(np.percentile(arr, p))
    return res


def paired_bootstrap_p95_ci(
    baseline: np.ndarray,
    candidate: np.ndarray,
    num_samples: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = 42,
) -> Tuple[Optional[float], Optional[float]]:
    """Paired bootstrap confidence interval for P95 speedup / reduction.
    
    Resamples paired indices with replacement to compute distribution of P95 reduction.
    Returns (None, None) if sample count is insufficient (< 10).
    """
    n = min(len(baseline), len(candidate))
    if n < 10:
        return None, None

    rng = np.random.default_rng(seed)
    boot_reductions = np.empty(num_samples)

    for i in range(num_samples):
        idx = rng.choice(n, size=n, replace=True)
        b_sample = baseline[idx]
        c_sample = candidate[idx]
        b_p95 = float(np.percentile(b_sample, 95))
        c_p95 = float(np.percentile(c_sample, 95))
        if b_p95 > 0:
            boot_reductions[i] = ((b_p95 - c_p95) / b_p95) * 100.0
        else:
            boot_reductions[i] = 0.0

    lower_pct = ((1.0 - ci) / 2.0) * 100.0
    upper_pct = (1.0 - (1.0 - ci) / 2.0) * 100.0
    return float(np.percentile(boot_reductions, lower_pct)), float(np.percentile(boot_reductions, upper_pct))


def compute_summary(
    baseline: np.ndarray,
    candidate: np.ndarray,
    baseline_success: Optional[np.ndarray] = None,
    candidate_success: Optional[np.ndarray] = None,
    wasted_calls: Optional[np.ndarray] = None,
    cost_multipliers: Optional[np.ndarray] = None,
    rate_limit_errors: Optional[np.ndarray] = None,
    input_tokens_base: Optional[float] = None,
    input_tokens_cand: Optional[float] = None,
    deopt_events: Optional[np.ndarray] = None,
    tool_start_base: Optional[np.ndarray] = None,
    tool_start_cand: Optional[np.ndarray] = None,
    semantic_mutations: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> MetricSummary:
    """Compute comprehensive MetricSummary from paired trial arrays.
    
    Strict CCL invariant: CCL percentiles are computed on trials where tasks completed successfully.
    Unpaired or missing evidence fields return None (null in JSON).
    """
    b_succ = float(np.mean(baseline_success)) if baseline_success is not None and len(baseline_success) > 0 else None
    c_succ = float(np.mean(candidate_success)) if candidate_success is not None and len(candidate_success) > 0 else None
    succ_delta = float(c_succ - b_succ) if (c_succ is not None and b_succ is not None) else None

    paired_counts = {}
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
    valid_baseline = baseline[baseline_success.astype(bool)] if (baseline_success is not None and len(baseline_success) == len(baseline)) else baseline
    valid_candidate = candidate[candidate_success.astype(bool)] if (candidate_success is not None and len(candidate_success) == len(candidate)) else candidate

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

    wasted_rate = float(np.mean(wasted_calls)) if wasted_calls is not None and len(wasted_calls) > 0 else None
    cost_mult = float(np.mean(cost_multipliers)) if cost_multipliers is not None and len(cost_multipliers) > 0 else None
    rl_err_rate = float(np.mean(rate_limit_errors)) if rate_limit_errors is not None and len(rate_limit_errors) > 0 else None

    token_red = (
        float((input_tokens_base - input_tokens_cand) / input_tokens_base * 100.0)
        if (input_tokens_base is not None and input_tokens_cand is not None and input_tokens_base > 0)
        else None
    )
    deopt_rate_val = float(np.mean(deopt_events)) if deopt_events is not None and len(deopt_events) > 0 else None

    t_start_p50 = float(np.percentile(tool_start_cand, 50)) if tool_start_cand is not None and len(tool_start_cand) > 0 else None
    t_start_p95 = float(np.percentile(tool_start_cand, 95)) if tool_start_cand is not None and len(tool_start_cand) > 0 else None
    if tool_start_base is not None and tool_start_cand is not None and len(tool_start_base) > 0 and len(tool_start_cand) > 0:
        t_base_p95 = float(np.percentile(tool_start_base, 95))
        t_start_speedup = float(t_base_p95 / t_start_p95) if (t_start_p95 is not None and t_start_p95 > 0) else 1.0
    else:
        t_start_speedup = None

    mutation_rate = float(np.mean(semantic_mutations)) if semantic_mutations is not None and len(semantic_mutations) > 0 else None

    # Paired bootstrap CI for P95 reduction
    ci_low, ci_high = paired_bootstrap_p95_ci(baseline, candidate, num_samples=500)
    p95_ci = (ci_low, ci_high) if (ci_low is not None and ci_high is not None) else None

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
        extra_metrics=extra or {},
    )


def bootstrap_confidence_interval(
    data: np.ndarray,
    stat_func: Callable[[np.ndarray], float] = np.mean,
    num_samples: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = 42,
) -> Tuple[float, float]:
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
