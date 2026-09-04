"""Tests for Phase 25: Measured, calibrated, non-tautological positive and negative controls."""

from __future__ import annotations

import unittest

from toolspeed.benchmarks.controls import (
    NegativeControlResult,
    PositiveControlResult,
    run_measured_negative_control,
    run_measured_positive_control,
)


class TestControls(unittest.IsolatedAsyncioTestCase):
    """Verifies measured positive sensitivity controls and true identity negative controls."""

    async def test_01_positive_control_measured_delay_slowdown(self) -> None:
        """Executes real delay in candidate arm and asserts candidate is measurably slower by >= expected delay."""
        res = await run_measured_positive_control(
            injected_delay_ms=25.0,
            work_duration_ms=5.0,
            trials=10,
        )

        self.assertIsInstance(res, PositiveControlResult)
        self.assertFalse(res.is_hardcoded_literal)
        self.assertTrue(res.candidate_is_slower)
        self.assertTrue(res.slowdown_meets_expected_delay)
        # Measured difference >= 20ms for a 25ms injected delay (within OS scheduling margin)
        self.assertGreaterEqual(res.measured_difference_ms, 20.0)

    async def test_02_negative_control_identical_code_paths_within_noise_floor(self) -> None:
        """Baseline and candidate execute identical code paths; speedup is within noise floor [0.98, 1.02] with 95% confidence."""
        res = await run_measured_negative_control(
            trials=40,
            work_duration_ms=5.0,
            noise_floor_range=(0.98, 1.02),
        )

        self.assertIsInstance(res, NegativeControlResult)
        self.assertFalse(res.is_hardcoded_literal)
        self.assertTrue(res.is_within_noise_floor)
        self.assertGreaterEqual(res.mean_speedup, 0.97)
        self.assertLessEqual(res.mean_speedup, 1.03)
        # 95% confidence bounds tightly bounded around 1.00
        self.assertLessEqual(res.ci_95_lower, 1.02)
        self.assertGreaterEqual(res.ci_95_upper, 0.98)

    async def test_03_positive_control_detects_small_delays(self) -> None:
        """Verifies sensitivity to small (15ms) injected latency additions."""
        res = await run_measured_positive_control(
            injected_delay_ms=15.0,
            work_duration_ms=5.0,
            trials=10,
        )

        self.assertTrue(res.candidate_is_slower)
        self.assertGreater(res.candidate_duration_ms, res.baseline_duration_ms)
        self.assertGreaterEqual(res.measured_difference_ms, 10.0)


if __name__ == "__main__":
    unittest.main()
