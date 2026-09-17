"""Tests verifying statistical calibration (Brier score, ECE) and CCL accounting."""

from __future__ import annotations

import unittest
from collections.abc import Sequence


def compute_brier_score(predictions: Sequence[float], outcomes: Sequence[int | bool]) -> float:
    """Computes the mean squared error Brier score: (1/N) * sum((f_i - o_i)^2)."""
    if len(predictions) != len(outcomes):
        raise ValueError("Predictions and outcomes must have identical length")
    if not predictions:
        return 0.0
    return sum((float(f) - float(o)) ** 2 for f, o in zip(predictions, outcomes, strict=True)) / len(predictions)


def compute_ece(
    confidences: Sequence[float],
    outcomes: Sequence[int | bool],
    num_bins: int = 5,
) -> float:
    """Computes Expected Calibration Error (ECE) across confidence bins."""
    if len(confidences) != len(outcomes):
        raise ValueError("Confidences and outcomes must have identical length")
    if not confidences:
        return 0.0

    n = len(confidences)
    bin_size = 1.0 / num_bins
    ece = 0.0

    for i in range(num_bins):
        bin_lower = i * bin_size
        bin_upper = (i + 1) * bin_size
        bin_items = [
            (c, o)
            for c, o in zip(confidences, outcomes, strict=True)
            if bin_lower <= c < bin_upper or (i == num_bins - 1 and c == 1.0)
        ]

        if not bin_items:
            continue

        bin_count = len(bin_items)
        avg_confidence = sum(c for c, _ in bin_items) / bin_count
        avg_accuracy = sum(float(o) for _, o in bin_items) / bin_count
        ece += (bin_count / n) * abs(avg_confidence - avg_accuracy)

    return ece


class TestTypeSafeMetricsAndCalibration(unittest.TestCase):
    """Verifies scientific calibration metrics and critical-path latency calculation."""

    def test_perfect_brier_score(self) -> None:
        # Perfectly calibrated confident predictions
        predictions = [1.0, 1.0, 0.0, 0.0]
        outcomes = [1, 1, 0, 0]
        self.assertAlmostEqual(compute_brier_score(predictions, outcomes), 0.0)

    def test_overconfident_uncalibrated_brier_score(self) -> None:
        # 100% confident when wrong
        predictions = [1.0, 1.0, 1.0, 1.0]
        outcomes = [0, 0, 0, 0]
        self.assertAlmostEqual(compute_brier_score(predictions, outcomes), 1.0)

    def test_expected_calibration_error_calculation(self) -> None:
        # 10 samples: half at 0.9 confidence (all correct), half at 0.8 confidence (half correct)
        confidences = [0.9] * 5 + [0.8] * 5
        outcomes = [1] * 5 + [1, 1, 0, 0, 0]  # 5/5 for first group, 2/5 (0.4) for second group

        # Group 1: conf 0.9, acc 1.0 -> diff 0.1
        # Group 2: conf 0.8, acc 0.4 -> diff 0.4
        # Weighted ECE = 0.5 * 0.1 + 0.5 * 0.4 = 0.25
        ece = compute_ece(confidences, outcomes, num_bins=10)
        self.assertAlmostEqual(ece, 0.25, places=3)

    def test_critical_path_completion_latency_accounting(self) -> None:
        """Verifies CCL accounting model under hit vs miss scenarios."""
        llm_latency = 300.0
        provider_latency = 25.0
        tool_latency = 500.0

        # Case 1: No speculation
        ccl_no_spec = llm_latency + tool_latency
        self.assertEqual(ccl_no_spec, 800.0)

        # Case 2: Speculation Hit
        # Provider runs concurrently or before tool; speculative tool begins at max(0, provider_latency)
        # Speculative tool finishes at provider_latency + tool_latency = 525ms
        # LLM finishes at 300ms.
        # Since speculative tool is running, remaining tool latency when LLM finishes is max(0, 525 - 300) = 225ms.
        # Total CCL = 300 + 225 = 525ms.
        # Net saving = 800 - 525 = 275ms!
        tool_finish_time = provider_latency + tool_latency
        ccl_hit = max(llm_latency, tool_finish_time)
        self.assertEqual(ccl_hit, 525.0)
        self.assertLess(ccl_hit, ccl_no_spec)

        # Case 3: Speculation Miss (wasted work)
        # Main model selects a DIFFERENT tool (latency = 500ms) after LLM finishes at 300ms.
        # In isolated mode (no contention), correct tool runs 300ms to 800ms.
        # Total CCL = 800ms (no wall-clock penalty, but 100% wasted tool compute).
        # In shared contention mode with capacity limit 1: correct tool must wait for speculative tool!
        ccl_miss_contended = tool_finish_time + tool_latency
        self.assertEqual(ccl_miss_contended, 1025.0)
        self.assertGreater(ccl_miss_contended, ccl_no_spec)


if __name__ == "__main__":
    unittest.main()
