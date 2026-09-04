"""Tests for Workload W1 fanout under concurrency-limit pressure."""

from __future__ import annotations

import unittest

from toolspeed.benchmarks.local_backend import LocalWallClockBackend
from toolspeed.schedulers.b1_sync_react import SyncReActScheduler
from toolspeed.schedulers.b2_native_parallel import NativeParallelScheduler
from toolspeed.workloads.w1_independent import (
    W1ConcurrencySweepReport,
    evaluate_w1_concurrency_pressure,
)


class TestWorkloadW1(unittest.IsolatedAsyncioTestCase):
    """Verifies real parallel execution, concurrency-limit sweeps, and diminishing speedup at limit 1."""

    async def asyncSetUp(self) -> None:
        self.backend = LocalWallClockBackend(seed=42)

    async def asyncTearDown(self) -> None:
        self.backend.cleanup()

    async def test_01_w1_real_parallel_execution(self) -> None:
        """NativeParallelScheduler executes fanout calls concurrently under high limit."""
        report = await evaluate_w1_concurrency_pressure(
            backend=self.backend,
            baseline_cls=SyncReActScheduler,
            candidate_cls=NativeParallelScheduler,
            limits=(1, 16),
            trial_index=0,
        )

        self.assertEqual(len(report.points), 2)
        p1 = report.points[0]
        p16 = report.points[1]

        self.assertEqual(p1.concurrency_limit, 1)
        self.assertEqual(p16.concurrency_limit, 16)
        self.assertGreater(p16.speedup, p1.speedup)

    async def test_02_w1_concurrency_limit_sweep_and_diminishing_speedup(self) -> None:
        """Varying concurrency limit across 1, 2, 4, 8, 16 proves speedup diminishes as limit approaches 1."""
        report = await evaluate_w1_concurrency_pressure(
            backend=self.backend,
            baseline_cls=SyncReActScheduler,
            candidate_cls=NativeParallelScheduler,
            limits=(1, 2, 4, 8, 16),
            trial_index=0,
        )

        self.assertIsInstance(report, W1ConcurrencySweepReport)
        self.assertEqual(len(report.points), 5)

        valid, reason = report.verify_concurrency_pressure_invariants()
        self.assertTrue(valid, f"Concurrency pressure invariants failed: {reason}")

        # Limit 1: candidate queue time is positive and speedup is constrained
        p1 = report.points[0]
        self.assertGreater(p1.candidate_queue_time_ms, 0.0)

        # Limit 16: candidate queue time is minimal
        p16 = report.points[4]
        self.assertLess(p16.candidate_queue_time_ms, p1.candidate_queue_time_ms)


if __name__ == "__main__":
    unittest.main()
