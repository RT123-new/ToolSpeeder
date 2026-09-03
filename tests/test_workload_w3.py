"""Tests for Workload W3 single-slot speculative execution under injected failure."""

from __future__ import annotations

import unittest

from toolspeed.benchmarks.replay_backend import ReplayBackend, ReplayLLMAdapter
from toolspeed.core.clock import WallClock
from toolspeed.core.types import ToolCall
from toolspeed.schedulers.b1_sync_react import SyncReActScheduler
from toolspeed.schedulers.base import SchedulerConfig
from toolspeed.schedulers.e3_speculation import SpeculativeReadScheduler
from toolspeed.workloads.w3_branching import (
    W3SpeculationSweepReport,
    evaluate_w3_speculation_failure_sweep,
)


class TestWorkloadW3(unittest.IsolatedAsyncioTestCase):
    """Verifies single-slot speculative execution under failure: 0%, 25%, 50%, 75%, 100%."""

    def setUp(self) -> None:
        self.backend = ReplayBackend(seed=42, clock=WallClock())

    async def test_01_speculation_failure_sweep_invariants(self) -> None:
        """Evaluates W3 across failure rates [0%, 25%, 50%, 75%, 100%].

        Proves speedup is positive at 0%, near zero at breakeven, and negative at 100%.
        """
        baseline = SyncReActScheduler()
        report = await evaluate_w3_speculation_failure_sweep(
            backend=self.backend,
            baseline_scheduler=baseline,
            failure_rates=(0.0, 0.25, 0.50, 0.75, 1.0),
            seed=42,
            trials_per_point=4,
        )

        self.assertIsInstance(report, W3SpeculationSweepReport)
        self.assertEqual(len(report.points), 5)

        valid, reason = report.verify_speculation_failure_invariants()
        self.assertTrue(valid, f"Speculation invariants failed: {reason}")

        p0 = report.points[0]
        p50 = report.points[2]
        p100 = report.points[4]

        # 0% failure: positive speedup (> 1.0x)
        self.assertGreater(p0.speedup, 1.0)
        # 100% failure: negative speedup (< 1.0x)
        self.assertLess(p100.speedup, 1.0)
        # Breakeven (~50%): between 0% and 100%
        self.assertGreaterEqual(p0.speedup, p50.speedup)
        self.assertGreaterEqual(p50.speedup, p100.speedup)

        # Assert no unhandled cancellation exception leaked
        for p in report.points:
            self.assertFalse(p.cancelled_leaked)

    async def test_02_cancelled_draft_never_leaks_unhandled_exception(self) -> None:
        """Asserts that a cancelled draft under single-slot contention never leaks CancelledError or raises."""
        task = self.backend.generate_task("W3", trial_index=0)
        tools, model = self.backend.create_workload_environment("W3", trial_index=0)

        self.assertIsInstance(model, ReplayLLMAdapter)
        model.draft_prediction = ToolCall(  # type: ignore[attr-defined]
            name="audit_transaction",
            arguments={"customer_id": "cust_divergent_cancel_test"},
            speculation_confidence=0.99,
        )

        sched = SpeculativeReadScheduler(
            SchedulerConfig(
                concurrency_limit=1,
                speculation_enabled=True,
                speculation_contention_mode="single_slot",
            )
        )

        task_model = task.to_model_task() if hasattr(task, "to_model_task") else task
        result = await sched.execute(task_model, model, tools)

        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        self.assertIsNotNone(result.final_answer)


if __name__ == "__main__":
    unittest.main()
