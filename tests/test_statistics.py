"""Tests for Phase 27: Valid paired difference inference, stratified sampling, noise floor, and power analysis."""

from __future__ import annotations

import unittest

import numpy as np

from toolspeed.core.statistics import (
    InsufficientSampleSizeError,
    PairedDifferenceInference,
    compute_min_sample_size,
    compute_paired_inference,
)


class TestStatistics(unittest.TestCase):
    """Verifies paired inference, stratified cluster sampling, noise floor bounds, and power analysis enforcement."""

    def test_01_paired_difference_statistics_across_actual_trials(self) -> None:
        """Computes paired differences, percentiles, standard error, and confidence intervals across real trials."""
        rng = np.random.default_rng(42)
        n = 50
        baseline = rng.normal(loc=100.0, scale=10.0, size=n)
        # Candidate is 15ms faster on average
        candidate = baseline - rng.normal(loc=15.0, scale=3.0, size=n)

        res = compute_paired_inference(
            baseline=baseline,
            candidate=candidate,
            noise_floor_ms=1.5,
            min_detectable_effect_d=0.5,
        )

        self.assertIsInstance(res, PairedDifferenceInference)
        self.assertEqual(res.sample_size, 50)
        self.assertTrue(res.power_adequate)
        # Mean difference should be around 15ms
        self.assertAlmostEqual(res.mean_difference, 15.0, delta=2.0)
        self.assertGreater(res.p95_difference, 0.0)
        self.assertGreater(res.p99_difference, 0.0)
        self.assertGreater(res.t_statistic, 0.0)
        self.assertTrue(res.exceeds_noise_floor)
        self.assertLess(res.ci_95[0], res.mean_difference)
        self.assertGreater(res.ci_95[1], res.mean_difference)

    def test_02_stratified_sampling_by_task_clusters(self) -> None:
        """Calculates stratified mean difference and standard error clustered by task type."""
        rng = np.random.default_rng(101)
        clusters: list[str] = []
        baseline_list: list[float] = []
        candidate_list: list[float] = []

        # Cluster A: lightweight queries (30 items)
        for _ in range(30):
            clusters.append("lightweight_query")
            b = float(rng.normal(20.0, 2.0))
            baseline_list.append(b)
            candidate_list.append(b - float(rng.normal(2.0, 0.3)))

        # Cluster B: heavy aggregation (20 items)
        for _ in range(20):
            clusters.append("heavy_aggregation")
            b = float(rng.normal(200.0, 15.0))
            baseline_list.append(b)
            candidate_list.append(b - float(rng.normal(30.0, 2.0)))

        res = compute_paired_inference(
            baseline=baseline_list,
            candidate=candidate_list,
            clusters=clusters,
            noise_floor_ms=1.0,
        )

        self.assertIsNotNone(res.stratified_mean_difference)
        self.assertIsNotNone(res.stratified_std_error)
        assert res.stratified_mean_difference is not None
        assert res.stratified_std_error is not None
        self.assertGreater(res.stratified_mean_difference, 0.0)
        self.assertGreater(res.stratified_std_error, 0.0)

    def test_03_noise_floor_reporting_and_threshold(self) -> None:
        """Correctly flags whether observed effect size exceeds empirical noise floor."""
        rng = np.random.default_rng(202)
        n = 40
        # Case A: negligible difference within 1.0ms noise floor
        b_noise = rng.normal(50.0, 1.0, size=n)
        c_noise = b_noise + rng.normal(0.1, 0.2, size=n)

        res_noise = compute_paired_inference(
            baseline=b_noise,
            candidate=c_noise,
            noise_floor_ms=1.0,
        )
        self.assertFalse(res_noise.exceeds_noise_floor)

        # Case B: genuine signal well above noise floor
        b_sig = rng.normal(50.0, 1.0, size=n)
        c_sig = b_sig - 10.0

        res_sig = compute_paired_inference(
            baseline=b_sig,
            candidate=c_sig,
            noise_floor_ms=1.0,
        )
        self.assertTrue(res_sig.exceeds_noise_floor)

    def test_04_reject_sample_size_below_power_analysis_minimum(self) -> None:
        """Rejects evaluations where sample size N is strictly less than power-analysis minimum."""
        n_min = compute_min_sample_size(alpha=0.05, power=0.80, effect_size_d=0.5)
        self.assertGreaterEqual(n_min, 30)

        # Small sample size (N=10) below required power minimum
        b_small = [10.0] * 10
        c_small = [8.0] * 10

        with self.assertRaises(InsufficientSampleSizeError) as ctx:
            compute_paired_inference(
                baseline=b_small,
                candidate=c_small,
                min_detectable_effect_d=0.5,
                enforce_min_sample_size=True,
            )
        self.assertIn("below the power-analysis minimum", str(ctx.exception))

        # Disabling enforcement reports power_adequate=False without raising
        res_unenforced = compute_paired_inference(
            baseline=b_small,
            candidate=c_small,
            min_detectable_effect_d=0.5,
            enforce_min_sample_size=False,
        )
        self.assertFalse(res_unenforced.power_adequate)
        self.assertEqual(res_unenforced.sample_size, 10)

    def test_05_power_formula_monotonicity(self) -> None:
        """Smaller effect sizes Cohen's d require strictly larger sample sizes."""
        n_small_effect = compute_min_sample_size(effect_size_d=0.2)
        n_medium_effect = compute_min_sample_size(effect_size_d=0.5)
        n_large_effect = compute_min_sample_size(effect_size_d=0.8)

        self.assertGreater(n_small_effect, n_medium_effect)
        self.assertGreater(n_medium_effect, n_large_effect)


if __name__ == "__main__":
    unittest.main()
