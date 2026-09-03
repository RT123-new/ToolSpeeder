"""Tests for Workload W6 subprocess sandbox warm vs cold start."""

from __future__ import annotations

import unittest

from toolspeed.workloads.w6_cold_start import (
    W6SubprocessSweepReport,
    WarmSubprocessWorker,
    evaluate_w6_subprocess_warm_vs_cold,
    execute_cold_subprocess,
)


class TestWorkloadW6(unittest.IsolatedAsyncioTestCase):
    """Verifies real subprocess spawn overhead, concurrency scaling (1, 2, 4, 8), and pool amortization invariants."""

    async def test_01_subprocess_warm_vs_cold_amortization_sweep(self) -> None:
        """Evaluates cold vs warm across concurrency [1, 2, 4, 8] and verifies speedup > 1.0 ONLY when calls > pool size."""
        report = await evaluate_w6_subprocess_warm_vs_cold(
            concurrencies=(1, 2, 4, 8),
            pool_size=4,
            under_amortized_calls=2,
            over_amortized_calls=12,
        )

        self.assertIsInstance(report, W6SubprocessSweepReport)
        self.assertEqual(len(report.points), 8)

        valid, reason = report.verify_subprocess_amortization_invariants()
        self.assertTrue(valid, f"Subprocess amortization invariants failed: {reason}")

        for p in report.points:
            if p.call_count <= p.pool_size:
                # Under-amortized: pool initialization overhead exceeds cold execution
                self.assertLessEqual(p.speedup, 1.05)
            else:
                # Over-amortized: warm pool amortizes startup across calls > pool size
                self.assertGreater(p.speedup, 1.0)

    async def test_02_warm_worker_lifecycle_and_reuse(self) -> None:
        """Verifies persistent warm worker lifecycle and reuse across multiple expressions."""
        worker = WarmSubprocessWorker()
        await worker.start()

        res1, _dur1_ms = await worker.execute("10 + 20")
        res2, dur2_ms = await worker.execute("sum([1, 2, 3, 4, 5])")
        res3, dur3_ms = await worker.execute("abs(-42)")

        await worker.close()

        self.assertEqual(res1, 30)
        self.assertEqual(res2, 15)
        self.assertEqual(res3, 42)

        # Warm worker reuse executes each in low milliseconds / sub-millisecond range
        self.assertLess(dur2_ms, 50.0)
        self.assertLess(dur3_ms, 50.0)

    async def test_03_cold_subprocess_spawns_fresh_process_each_call(self) -> None:
        """Verifies cold subprocess execution executes accurately with real process overhead."""
        res, dur_ms = await execute_cold_subprocess("100 * 5")
        self.assertEqual(res, 500)
        self.assertGreater(dur_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
