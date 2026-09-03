"""Comprehensive tests for immutable arm-isolated replay fixtures."""

from __future__ import annotations

import unittest

from toolspeed.benchmarks.replay_backend import (
    ReplayBackend,
    ReplayCaseFixture,
    ReplayFixtureManager,
)
from toolspeed.core.types import TokenUsage


class TestReplayFixtures(unittest.TestCase):
    """Verifies pre-generated immutable per-case fixtures for seed, workload, and arm with distinct epochs."""

    def setUp(self) -> None:
        self.manager = ReplayFixtureManager()

    def test_01_fixture_generation_structure_and_fields(self) -> None:
        """Fixtures include seed, workload, arm, tool/model latency, responses, trace events, tokens, side effects."""
        fixture = self.manager.get_or_create_fixture(
            workload_id="W7",
            seed=42,
            arm="baseline",
            trial_index=3,
        )

        self.assertIsInstance(fixture, ReplayCaseFixture)
        self.assertEqual(fixture.seed, 42)
        self.assertEqual(fixture.workload_id, "W7")
        self.assertEqual(fixture.arm, "baseline")
        self.assertGreater(fixture.epoch, 0)
        self.assertEqual(fixture.trial_index, 3)

        # Tool and model latencies
        self.assertIn("execute_fund_transfer", fixture.tool_latencies)
        self.assertGreater(fixture.tool_latencies["execute_fund_transfer"], 0.0)
        self.assertGreater(fixture.model_latency_ms, 0.0)

        # Responses, trace events, tokens, side effects
        self.assertIn("execute_fund_transfer", fixture.tool_responses)
        self.assertIn("CASE_START", fixture.trace_events)
        self.assertIn("CASE_END", fixture.trace_events)
        self.assertIsInstance(fixture.tokens, TokenUsage)
        self.assertGreater(fixture.tokens.total_tokens, 0)
        self.assertIn("balance", fixture.side_effects)

    def test_02_immutable_in_place_mutation_prevention(self) -> None:
        """Fixtures cannot be mutated in place; setting attributes raises RuntimeError."""
        fixture = self.manager.get_or_create_fixture(
            workload_id="W1",
            seed=100,
            arm="baseline",
            trial_index=0,
        )

        with self.assertRaises(RuntimeError) as ctx:
            fixture.model_latency_ms = 999.0  # type: ignore[misc]
        self.assertIn("Cannot mutate immutable ReplayCaseFixture", str(ctx.exception))

        with self.assertRaises(RuntimeError) as ctx2:
            fixture.arm = "tampered_arm"  # type: ignore[misc]
        self.assertIn("Cannot mutate immutable ReplayCaseFixture", str(ctx2.exception))

    def test_03_arm_isolation_and_distinct_epochs(self) -> None:
        """Baseline and candidate arms receive distinct epoch numbers, distinct case IDs, and separate instances."""
        fix_base = self.manager.get_or_create_fixture(
            workload_id="W2",
            seed=101,
            arm="baseline",
            trial_index=1,
        )
        fix_cand = self.manager.get_or_create_fixture(
            workload_id="W2",
            seed=101,
            arm="candidate",
            trial_index=1,
        )

        self.assertNotEqual(fix_base.arm, fix_cand.arm)
        self.assertNotEqual(fix_base.epoch, fix_cand.epoch)
        self.assertNotEqual(fix_base.case_id, fix_cand.case_id)
        self.assertIn("_baseline_", fix_base.case_id)
        self.assertIn("_candidate_", fix_cand.case_id)

    def test_04_deep_cloning_and_cache_isolation(self) -> None:
        """Mutating sub-dictionaries in retrieved fixture copy does not mutate subsequent retrieves."""
        fix1 = self.manager.get_or_create_fixture("W3", seed=200, arm="baseline", trial_index=0)
        fix1.expected_output["status"] = "mutated_status"

        fix2 = self.manager.get_or_create_fixture("W3", seed=200, arm="baseline", trial_index=0)
        self.assertEqual(fix2.expected_output["status"], "approved")

    def test_05_replay_backend_creates_arm_isolated_environments(self) -> None:
        """ReplayBackend instantiates tool adapters and model using arm-isolated fixtures."""
        backend = ReplayBackend(seed=300, fixture_manager=self.manager)
        tools_b, model_b = backend.create_workload_environment("W1", trial_index=0, arm="baseline")
        tools_c, model_c = backend.create_workload_environment("W1", trial_index=0, arm="candidate")

        self.assertIsNotNone(tools_b.get("server_metric_0"))
        self.assertIsNotNone(tools_c.get("server_metric_0"))
        self.assertIsNotNone(model_b)
        self.assertIsNotNone(model_c)

        task_b = backend.generate_task("W1", trial_index=0, arm="baseline")
        task_c = backend.generate_task("W1", trial_index=0, arm="candidate")
        self.assertIn("baseline", task_b.task_id)
        self.assertIn("candidate", task_c.task_id)


if __name__ == "__main__":
    unittest.main()
