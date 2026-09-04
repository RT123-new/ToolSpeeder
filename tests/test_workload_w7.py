"""Tests for Workload W7 side-effects safety-latency split and exact-state verification."""

from __future__ import annotations

import unittest

from toolspeed.workloads.w7_side_effects import (
    ExternalStateLedger,
    W7aSafetyWorkload,
    W7bLatencyWorkload,
    evaluate_w7a_safety,
    evaluate_w7b_latency,
)


class TestWorkloadW7(unittest.TestCase):
    """Verifies W7 split into W7a safety-critical and W7b pure latency, with exact-state queries."""

    def test_01_w7a_safety_split_and_exact_state_verification(self) -> None:
        """Verifies W7a: exactly 1 mutation on commit, 0 on crash/rollback, and 0 on duplicate replay."""
        report = evaluate_w7a_safety()

        self.assertTrue(report.all_passed, f"W7a safety verification failed: {report.message}")
        self.assertTrue(report.commit_verified)
        self.assertEqual(report.mutation_count_commit, 1)

        self.assertTrue(report.rollback_zero_mutations_verified)
        self.assertEqual(report.mutation_count_rollback, 0)

        self.assertTrue(report.idempotency_dedup_verified)
        self.assertEqual(report.mutation_count_replay, 0)

    def test_02_w7b_pure_latency_benchmark(self) -> None:
        """Verifies W7b: pure latency benchmarking without failure injection."""
        report = evaluate_w7b_latency(iterations=30, sleep_ms=0.1)

        self.assertEqual(report.sample_count, 30)
        self.assertGreater(report.mean_ms, 0.0)
        self.assertGreater(report.p50_ms, 0.0)
        self.assertGreaterEqual(report.p95_ms, report.p50_ms)
        self.assertGreaterEqual(report.max_ms, report.min_ms)

    def test_03_workload_specs_and_family_split(self) -> None:
        """Verifies discrete registration and specs for W7a and W7b."""
        w7a = W7aSafetyWorkload()
        spec_a = w7a.get_spec()
        self.assertEqual(spec_a.family, "w7a_safety")
        self.assertEqual(spec_a.parameters.get("workload_id"), "W7_SAFETY")

        w7b = W7bLatencyWorkload()
        spec_b = w7b.get_spec()
        self.assertEqual(spec_b.family, "w7b_latency")
        self.assertEqual(spec_b.parameters.get("workload_id"), "W7_LATENCY")

    def test_04_external_state_ledger_query_and_immutability(self) -> None:
        """Verifies external state query returns immutable StateSnapshot before and after mutation."""
        ledger = ExternalStateLedger(initial_balance=500.0)
        snap1 = ledger.snapshot()

        # Mutation
        res = ledger.transfer("acc_001", "acc_002", 50.0, "k_snap_01")
        self.assertTrue(res["mutated"])

        snap2 = ledger.snapshot()
        self.assertEqual(snap1.get("acc_001"), 500.0)
        self.assertEqual(snap2.get("acc_001"), 450.0)
        self.assertEqual(snap2.get("acc_002"), 550.0)


if __name__ == "__main__":
    unittest.main()
