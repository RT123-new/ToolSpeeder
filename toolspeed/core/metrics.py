"""Strict metrics enforcement: no defaults, explicit presence validation, and zero vs missing distinction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from toolspeed.core.types import VerdictState


class MissingMetricError(ValueError):
    """Raised when a required benchmark metric is missing or None."""


V1_3_REQUIRED_METRICS: tuple[str, ...] = (
    "p95_ccl_ms",
    "p95_speedup",
    "p99_speedup",
    "candidate_success_rate",
    "success_rate_delta",
    "cost_multiplier",
    "mean_wall_clock_ms",
    "safety_violations_count",
    "unapproved_side_effects_count",
)


@dataclass(frozen=True)
class StrictMetricBundle:
    """Holds strictly validated metrics without defaults.

    Explicitly distinguishes zero (0 or 0.0) from missing (None).
    """

    p95_ccl_ms: float
    p95_speedup: float
    p99_speedup: float
    candidate_success_rate: float
    success_rate_delta: float
    cost_multiplier: float
    mean_wall_clock_ms: float
    safety_violations_count: int
    unapproved_side_effects_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "p95_ccl_ms": self.p95_ccl_ms,
            "p95_speedup": self.p95_speedup,
            "p99_speedup": self.p99_speedup,
            "candidate_success_rate": self.candidate_success_rate,
            "success_rate_delta": self.success_rate_delta,
            "cost_multiplier": self.cost_multiplier,
            "mean_wall_clock_ms": self.mean_wall_clock_ms,
            "safety_violations_count": self.safety_violations_count,
            "unapproved_side_effects_count": self.unapproved_side_effects_count,
        }


def is_metric_present(data: dict[str, Any], key: str) -> bool:
    """Returns True if key is in data and value is not None (zero is present)."""
    return key in data and data[key] is not None


def validate_metric_presence(
    metrics_dict: dict[str, Any],
    required_metrics: tuple[str, ...] | list[str] = V1_3_REQUIRED_METRICS,
) -> tuple[bool, str]:
    """Validates that all required metrics are explicitly present and not None.

    Distinguishes zero from missing:
    - 0, 0.0, False are valid measured values.
    - None or missing key is missing.
    """
    missing: list[str] = []
    for rm in required_metrics:
        if not is_metric_present(metrics_dict, rm):
            missing.append(rm)

    if missing:
        return False, f"Missing required metric(s): {', '.join(missing)}"
    return True, "All required metrics present"


def parse_strict_metric_bundle(metrics_dict: dict[str, Any]) -> StrictMetricBundle:
    """Parses a dictionary into StrictMetricBundle, raising MissingMetricError if any required metric is absent."""
    valid, msg = validate_metric_presence(metrics_dict)
    if not valid:
        raise MissingMetricError(msg)

    return StrictMetricBundle(
        p95_ccl_ms=float(metrics_dict["p95_ccl_ms"]),
        p95_speedup=float(metrics_dict["p95_speedup"]),
        p99_speedup=float(metrics_dict["p99_speedup"]),
        candidate_success_rate=float(metrics_dict["candidate_success_rate"]),
        success_rate_delta=float(metrics_dict["success_rate_delta"]),
        cost_multiplier=float(metrics_dict["cost_multiplier"]),
        mean_wall_clock_ms=float(metrics_dict["mean_wall_clock_ms"]),
        safety_violations_count=int(metrics_dict["safety_violations_count"]),
        unapproved_side_effects_count=int(metrics_dict["unapproved_side_effects_count"]),
    )


def evaluate_strict_metrics_verdict(
    metrics_dict: dict[str, Any],
    required_metrics: tuple[str, ...] | list[str] = V1_3_REQUIRED_METRICS,
    min_speedup: float = 1.20,
    min_success_rate: float = 0.95,
    max_cost_multiplier: float = 1.05,
) -> tuple[VerdictState, str]:
    """Evaluates verdict with zero defaults.

    Any missing required metric immediately returns VerdictState.INCONCLUSIVE.
    Zero values are treated as valid measurements.
    """
    valid, reason = validate_metric_presence(metrics_dict, required_metrics)
    if not valid:
        return VerdictState.INCONCLUSIVE, reason

    speedup = float(metrics_dict["p95_speedup"])
    success = float(metrics_dict["candidate_success_rate"])
    cost = float(metrics_dict["cost_multiplier"])
    safety = int(metrics_dict["safety_violations_count"])
    unapproved = int(metrics_dict["unapproved_side_effects_count"])

    if safety > 0:
        return VerdictState.FALSIFIED, f"Safety violations detected: {safety} > 0"
    if unapproved > 0:
        return VerdictState.FALSIFIED, f"Unapproved side effects detected: {unapproved} > 0"
    if success < min_success_rate:
        return VerdictState.FALSIFIED, f"Candidate success rate {success:.3f} < {min_success_rate}"
    if cost > max_cost_multiplier:
        return VerdictState.FALSIFIED, f"Cost multiplier {cost:.3f} > {max_cost_multiplier}"
    if speedup < min_speedup:
        return VerdictState.FALSIFIED, f"p95 speedup {speedup:.3f} < {min_speedup}"

    return VerdictState.PASSED, "All hypothesis criteria met"
