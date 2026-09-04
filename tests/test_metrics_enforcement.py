"""Tests for Phase 26: Strict metrics enforcement, removal of default fallbacks, and zero vs missing distinction."""

from __future__ import annotations

import unittest
from typing import Any

from toolspeed.core.metrics import (
    V1_3_REQUIRED_METRICS,
    MissingMetricError,
    StrictMetricBundle,
    evaluate_strict_metrics_verdict,
    is_metric_present,
    parse_strict_metric_bundle,
    validate_metric_presence,
)
from toolspeed.core.types import VerdictState


class TestMetricsEnforcement(unittest.TestCase):
    """Verifies strict presence of required metrics, rejection of defaults, and zero vs missing distinction."""

    def setUp(self) -> None:
        self.complete_valid_metrics: dict[str, Any] = {
            "p95_ccl_ms": 120.5,
            "p95_speedup": 1.45,
            "p99_speedup": 1.35,
            "candidate_success_rate": 0.98,
            "success_rate_delta": 0.0,
            "cost_multiplier": 1.01,
            "mean_wall_clock_ms": 115.0,
            "safety_violations_count": 0,
            "unapproved_side_effects_count": 0,
        }

    def test_01_all_required_metrics_present_and_valid(self) -> None:
        """Complete metrics dictionary passes validation and parses into StrictMetricBundle."""
        valid, msg = validate_metric_presence(self.complete_valid_metrics)
        self.assertTrue(valid, msg)

        bundle = parse_strict_metric_bundle(self.complete_valid_metrics)
        self.assertIsInstance(bundle, StrictMetricBundle)
        self.assertEqual(bundle.p95_speedup, 1.45)
        self.assertEqual(bundle.safety_violations_count, 0)
        self.assertEqual(bundle.unapproved_side_effects_count, 0)

        verdict, _reason = evaluate_strict_metrics_verdict(self.complete_valid_metrics)
        self.assertEqual(verdict, VerdictState.PASSED)

    def test_02_missing_metric_triggers_validation_error_and_inconclusive(self) -> None:
        """Any missing required metric immediately causes validation error and INCONCLUSIVE verdict."""
        for metric_name in V1_3_REQUIRED_METRICS:
            tampered = dict(self.complete_valid_metrics)
            del tampered[metric_name]

            valid, msg = validate_metric_presence(tampered)
            self.assertFalse(valid, f"Expected validation failure when '{metric_name}' is missing")
            self.assertIn(metric_name, msg)

            with self.assertRaises(MissingMetricError):
                parse_strict_metric_bundle(tampered)

            verdict, reason = evaluate_strict_metrics_verdict(tampered)
            self.assertEqual(
                verdict,
                VerdictState.INCONCLUSIVE,
                f"Missing '{metric_name}' did not yield INCONCLUSIVE verdict",
            )
            self.assertIn(metric_name, reason)

    def test_03_zero_is_distinguished_from_missing(self) -> None:
        """0, 0.0, and False are valid measurements and must NOT be treated as missing."""
        metrics_with_zeros = dict(self.complete_valid_metrics)
        metrics_with_zeros["safety_violations_count"] = 0
        metrics_with_zeros["unapproved_side_effects_count"] = 0
        metrics_with_zeros["success_rate_delta"] = 0.0

        self.assertTrue(is_metric_present(metrics_with_zeros, "safety_violations_count"))
        self.assertTrue(is_metric_present(metrics_with_zeros, "success_rate_delta"))

        valid, msg = validate_metric_presence(metrics_with_zeros)
        self.assertTrue(valid, msg)

        # None is missing, even if key exists
        metrics_with_none = dict(self.complete_valid_metrics)
        metrics_with_none["safety_violations_count"] = None
        self.assertFalse(is_metric_present(metrics_with_none, "safety_violations_count"))

        valid_none, msg_none = validate_metric_presence(metrics_with_none)
        self.assertFalse(valid_none)
        self.assertIn("safety_violations_count", msg_none)

    def test_04_no_favourable_default_fallbacks(self) -> None:
        """Metrics never default favourably; missing candidate_success_rate yields INCONCLUSIVE, not 1.0."""
        missing_success = dict(self.complete_valid_metrics)
        missing_success["candidate_success_rate"] = None

        verdict, reason = evaluate_strict_metrics_verdict(missing_success)
        self.assertEqual(verdict, VerdictState.INCONCLUSIVE)
        self.assertIn("candidate_success_rate", reason)

        # Missing cost_multiplier does not default to 1.0
        missing_cost = dict(self.complete_valid_metrics)
        del missing_cost["cost_multiplier"]

        verdict_cost, reason_cost = evaluate_strict_metrics_verdict(missing_cost)
        self.assertEqual(verdict_cost, VerdictState.INCONCLUSIVE)
        self.assertIn("cost_multiplier", reason_cost)

    def test_05_falsified_when_thresholds_not_met(self) -> None:
        """When all metrics are present, threshold violations produce FALSIFIED verdict."""
        # Safety violation > 0
        unsafe = dict(self.complete_valid_metrics)
        unsafe["safety_violations_count"] = 1
        v1, r1 = evaluate_strict_metrics_verdict(unsafe)
        self.assertEqual(v1, VerdictState.FALSIFIED)
        self.assertIn("Safety violations detected", r1)

        # Insufficient speedup
        slow = dict(self.complete_valid_metrics)
        slow["p95_speedup"] = 1.05  # below 1.20 threshold
        v2, r2 = evaluate_strict_metrics_verdict(slow, min_speedup=1.20)
        self.assertEqual(v2, VerdictState.FALSIFIED)
        self.assertIn("p95 speedup", r2)


if __name__ == "__main__":
    unittest.main()
