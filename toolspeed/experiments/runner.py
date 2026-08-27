"""Core experiment framework and statistical computation for ToolSpeed."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import time

import numpy as np


@dataclass(frozen=True)
class LatencyProfile:
    """Latency parameters and operational characteristics."""
    model_decision_ms: float = 450.0
    model_final_ms: float = 300.0
    tool_ms: float = 600.0
    draft_model_ms: float = 70.0
    program_runtime_overhead_ms: float = 80.0
    cache_lookup_ms: float = 8.0
    token_decode_ms_per_token: float = 12.0
    tokens_per_tool_json: int = 150
    tokens_per_tool_bytecode: int = 25
    rate_limit_capacity: int = 10
    rate_limit_refill_per_sec: float = 20.0
    sigma: float = 0.45

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
        return asdict(self)


@dataclass
class FalsificationVerdict:
    """Verdict of scientific hypothesis testing and falsification."""
    experiment_id: str
    hypothesis: str
    passed: bool
    falsified: bool
    summary: str
    checks: List[HypothesisCheck] = field(default_factory=list)
    inconclusive: bool = False
    evidence_log_row: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "hypothesis": self.hypothesis,
            "passed": self.passed,
            "falsified": self.falsified,
            "inconclusive": self.inconclusive,
            "summary": self.summary,
            "checks": [c.to_dict() for c in self.checks],
            "evidence_log_row": self.evidence_log_row,
        }


@dataclass
class MetricSummary:
    """Rigorous statistical summary of baseline vs candidate execution."""
    baseline_p50_ms: float
    candidate_p50_ms: float
    p50_speedup: float
    baseline_p90_ms: float
    candidate_p90_ms: float
    p90_speedup: float
    baseline_p95_ms: float
    candidate_p95_ms: float
    p95_speedup: float
    baseline_p99_ms: float
    candidate_p99_ms: float
    p99_speedup: float
    baseline_mean_ms: float = 0.0
    candidate_mean_ms: float = 0.0
    mean_speedup: float = 1.0
    baseline_success_rate: float = 1.0
    candidate_success_rate: float = 1.0
    success_rate_delta: float = 0.0
    wasted_call_rate: float = 0.0
    cost_multiplier: float = 1.0
    rate_limit_error_rate: float = 0.0
    input_tokens_baseline: float = 0.0
    input_tokens_candidate: float = 0.0
    token_reduction_pct: float = 0.0
    deopt_rate: float = 0.0
    tool_start_p50_ms: float = 0.0
    tool_start_p95_ms: float = 0.0
    tool_start_speedup_p95: float = 1.0
    semantic_mutation_rate: float = 0.0
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


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
    runtime_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "title": self.title,
            "workloads": self.workloads,
            "trials": self.trials,
            "seed": self.seed,
            "profile": self.profile.to_dict(),
            "parameter_name": self.parameter_name,
            "rows": self.rows,
            "verdict": self.verdict.to_dict(),
            "runtime_sec": self.runtime_sec,
        }

    def get_row(self, param_value: Any) -> Optional[Dict[str, Any]]:
        for r in self.rows:
            if r.get(self.parameter_name) == param_value:
                return r
        return None


def samples(rng: np.random.Generator, median_ms: float, sigma: float, shape: Union[int, Tuple[int, ...]]) -> np.ndarray:
    """Generate log-normal latency samples centered around median_ms."""
    return rng.lognormal(np.log(median_ms), sigma, shape)


def compute_percentiles(arr: np.ndarray, percentiles: Tuple[int, ...] = (50, 90, 95, 99)) -> Dict[str, float]:
    """Compute percentiles robustly."""
    if len(arr) == 0:
        return {f"p{p}": 0.0 for p in percentiles}
    res = {}
    for p in percentiles:
        res[f"p{p}"] = float(np.percentile(arr, p))
    return res


def compute_summary(
    baseline: np.ndarray,
    candidate: np.ndarray,
    baseline_success: Optional[np.ndarray] = None,
    candidate_success: Optional[np.ndarray] = None,
    wasted_calls: Optional[np.ndarray] = None,
    cost_multipliers: Optional[np.ndarray] = None,
    rate_limit_errors: Optional[np.ndarray] = None,
    input_tokens_base: float = 0.0,
    input_tokens_cand: float = 0.0,
    deopt_events: Optional[np.ndarray] = None,
    tool_start_base: Optional[np.ndarray] = None,
    tool_start_cand: Optional[np.ndarray] = None,
    semantic_mutations: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> MetricSummary:
    """Compute comprehensive MetricSummary from trial arrays.
    
    Strict CCL invariant: Only trials with validated task success are included in
    CCL percentiles (P50, P90, P95, P99, Mean). Failed trials are excluded from
    CCL percentiles and strictly penalize success rate metrics.
    """
    b_succ = float(np.mean(baseline_success)) if baseline_success is not None and len(baseline_success) > 0 else 1.0
    c_succ = float(np.mean(candidate_success)) if candidate_success is not None and len(candidate_success) > 0 else 1.0
    succ_delta = float(c_succ - b_succ)

    # Filter by success for Correct Completion Latency (CCL) calculation
    if baseline_success is not None:
        b_mask = np.asarray(baseline_success, dtype=bool)
        valid_baseline = baseline[b_mask] if len(baseline) == len(b_mask) else baseline
    else:
        valid_baseline = baseline

    if candidate_success is not None:
        c_mask = np.asarray(candidate_success, dtype=bool)
        valid_candidate = candidate[c_mask] if len(candidate) == len(c_mask) else candidate
    else:
        valid_candidate = candidate

    if len(valid_baseline) > 0:
        b50 = float(np.percentile(valid_baseline, 50))
        b90 = float(np.percentile(valid_baseline, 90))
        b95 = float(np.percentile(valid_baseline, 95))
        b99 = float(np.percentile(valid_baseline, 99))
        b_mean = float(np.mean(valid_baseline))
    else:
        b50 = b90 = b95 = b99 = b_mean = 0.0

    if len(valid_candidate) > 0:
        c50 = float(np.percentile(valid_candidate, 50))
        c90 = float(np.percentile(valid_candidate, 90))
        c95 = float(np.percentile(valid_candidate, 95))
        c99 = float(np.percentile(valid_candidate, 99))
        c_mean = float(np.mean(valid_candidate))
    else:
        c50 = c90 = c95 = c99 = c_mean = 0.0

    p50_speedup = float(b50 / c50) if c50 > 0 else 1.0
    p90_speedup = float(b90 / c90) if c90 > 0 else 1.0
    p95_speedup = float(b95 / c95) if c95 > 0 else 1.0
    p99_speedup = float(b99 / c99) if c99 > 0 else 1.0
    mean_speedup = float(b_mean / c_mean) if c_mean > 0 else 1.0

    wasted_rate = float(np.mean(wasted_calls)) if wasted_calls is not None and len(wasted_calls) > 0 else 0.0
    cost_mult = float(np.mean(cost_multipliers)) if cost_multipliers is not None and len(cost_multipliers) > 0 else 1.0
    rl_err_rate = float(np.mean(rate_limit_errors)) if rate_limit_errors is not None and len(rate_limit_errors) > 0 else 0.0

    token_red = (
        float((input_tokens_base - input_tokens_cand) / input_tokens_base * 100.0)
        if input_tokens_base > 0
        else 0.0
    )
    deopt_rate_val = float(np.mean(deopt_events)) if deopt_events is not None and len(deopt_events) > 0 else 0.0

    t_start_p50 = float(np.percentile(tool_start_cand, 50)) if tool_start_cand is not None and len(tool_start_cand) > 0 else 0.0
    t_start_p95 = float(np.percentile(tool_start_cand, 95)) if tool_start_cand is not None and len(tool_start_cand) > 0 else 0.0
    if tool_start_base is not None and tool_start_cand is not None and len(tool_start_base) > 0 and len(tool_start_cand) > 0:
        t_base_p95 = float(np.percentile(tool_start_base, 95))
        t_start_speedup = float(t_base_p95 / t_start_p95) if t_start_p95 > 0 else 1.0
    else:
        t_start_speedup = 1.0

    mutation_rate = float(np.mean(semantic_mutations)) if semantic_mutations is not None and len(semantic_mutations) > 0 else 0.0

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
        extra_metrics=extra or {},
    )


def bootstrap_confidence_interval(
    data: np.ndarray,
    stat_func: Callable[[np.ndarray], float] = np.mean,
    num_samples: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = 42,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for a metric."""
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
