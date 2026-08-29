"""Frozen Scientific Hypotheses, Baselines, and Threshold Specification Policy (Version 2.1.0)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from toolspeed.core.protocol import FROZEN_PROTOCOL


class HypothesisType(str, Enum):
    LATENCY_EFFICACY = "latency_efficacy"
    CORRECTNESS_NON_INFERIORITY = "correctness_non_inferiority"
    TAIL_LATENCY_STABILITY = "tail_latency_stability"
    SAFETY_GATE = "safety_gate"
    COST_CONSTRAINT = "cost_constraint"
    NULL_CONTROL = "null_control"
    SENSITIVITY_CONTROL = "sensitivity_control"


@dataclass(frozen=True)
class BenchmarkHypothesisPolicy:
    """Frozen policy governing verdict eligibility, statistical criteria, and baseline comparisons."""

    policy_version: str = "2.1.0"
    policy_date: str = "2026-08-29"

    # Latency: Central hypothesis requires >= 10% P95 CCL reduction (speedup >= 1.11x)
    min_p95_speedup_efficacy: float = 1.11
    min_p95_reduction_pct: float = 10.0

    # Tail Latency: P99 CCL non-regression (allowable tail regression <= 5%, speedup >= 0.95x)
    min_p99_speedup_non_regression: float = 0.95

    # Correctness: Candidate success >= 95% and non-inferior (delta >= -0.5 pp)
    min_candidate_success_rate: float = 0.95
    max_allowable_success_drop: float = -0.005

    # Cost: Candidate monetary/token cost multiplier <= 1.10x
    max_cost_multiplier: float = 1.10

    # Safety: Zero unapproved side-effects and zero duplicate mutations
    max_unapproved_side_effects: int = 0
    max_duplicate_mutations: int = 0

    # Controls: Null control equivalence region [0.95x, 1.05x]
    null_control_lower_bound: float = 0.95
    null_control_upper_bound: float = 1.05
    positive_control_min_speedup: float = 1.50

    # Sample Size Gates
    min_trials_replay: int = 1000
    min_trials_local: int = 200
    bootstrap_samples: int = 2000
    confidence_level: float = 0.95


FROZEN_POLICY = BenchmarkHypothesisPolicy(
    min_trials_replay=FROZEN_PROTOCOL.trials_per_seed_replay,
    min_trials_local=FROZEN_PROTOCOL.trials_per_seed_local,
    bootstrap_samples=FROZEN_PROTOCOL.bootstrap_resamples,
    confidence_level=FROZEN_PROTOCOL.bootstrap_ci,
)

WORKLOAD_BASELINES: dict[str, dict[str, str]] = {
    wl_id: {
        "candidate": m.candidate,
        "primary_baseline": m.primary_attribution_baseline,
        "practical_baseline": m.practical_baseline,
        "mechanism": m.name,
    }
    for wl_id, m in FROZEN_PROTOCOL.mechanisms.items()
}
