"""End-to-end scientific integrity regression test suite for ToolSpeed PR #1."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from toolspeed.adapters.base import ToolRegistry
from toolspeed.adapters.mock_tools import MockToolAdapter, MockToolConfig
from toolspeed.benchmarks.harness import BenchmarkConfig, BenchmarkHarness
from toolspeed.cli import cmd_falsify, cmd_validate_bundle
from toolspeed.core.protocol import load_frozen_protocol, validate_protocol_dict
from toolspeed.core.rate_limiter import RateLimiter
from toolspeed.core.types import (
    ApprovalGrant,
    EvidenceLevel,
    ExecutionAuthorityContext,
    Task,
    ToolCall,
    ToolResult,
)
from toolspeed.schedulers.executor import SharedIdempotencyStore, ToolExecutor
from toolspeed.schedulers.phase2_cache import ToolResultCache
from toolspeed.visualization.report import save_benchmark_reports


class TestScientificIntegrity(unittest.IsolatedAsyncioTestCase):
    """Rigorous scientific integrity regressions validating protocol, oracles, runtimes, and artifacts."""

    def test_01_protocol_schema_validation(self) -> None:
        """Verify tool-speed-v1.1.json validates against schema invariants."""
        protocol = load_frozen_protocol("benchmark-plans/tool-speed-v1.1.json")
        self.assertEqual(protocol.plan_id, "tool-speed-v1.1")
        self.assertIn(protocol.plan_version, ("1.1.0", "1.1.1"))
        self.assertIn("W1", protocol.mechanisms)
        self.assertIn("W7_SAFETY", protocol.mechanisms)
        self.assertIn("W7_LATENCY", protocol.mechanisms)
        self.assertIn("E5a", protocol.mechanisms)
        self.assertEqual(protocol.mechanisms["E5b"].status, "UNIMPLEMENTED")

        # Test invalid protocol rejection
        bad_data = json.loads(protocol.raw_json)
        del bad_data["plan_id"]
        errors = validate_protocol_dict(bad_data)
        self.assertTrue(len(errors) > 0)
        self.assertIn("plan_id", errors[0])

    def test_02_oracle_separation_and_negative_test(self) -> None:
        """Verify model receives strictly AgentTask, and Task.validate fails if required tool omitted."""
        grant = ApprovalGrant.create("fund_transfer", {"amount": 500})
        task = Task(
            task_id="t_oracle_test",
            prompt="Transfer 500 dollars",
            expected_output={"balance": 500},
            metadata={
                "approval_grant": grant,
                "secret_oracle_canary": "CANARY_12345",
                "required_tools": ["fund_transfer"],
                "user_id": "u_999",
            },
        )

        agent_task = task.to_agent_task()
        self.assertNotIn("approval_grant", agent_task.metadata)
        self.assertNotIn("secret_oracle_canary", agent_task.metadata)
        self.assertEqual(agent_task.metadata.get("user_id"), "u_999")

        # Negative test: Model outputs exact expected final value, but never called the required tool!
        # Task.validate MUST return False!
        self.assertFalse(task.validate(actual_output={"balance": 500}))

    async def test_03_rate_limiter_double_release_prevention(self) -> None:
        """Verify RateLimiter.lease() does not double release concurrency slot when body raises."""
        limiter = RateLimiter(max_concurrency=2, rate_per_sec=100.0)

        for _ in range(10):
            try:
                async with limiter.lease(tokens=1):
                    raise RuntimeError("Simulated failure inside lease")
            except RuntimeError:
                pass

        self.assertEqual(limiter.concurrency_limiter.active_count, 0)

        # Ensure we can still acquire all slots up to max concurrency
        l1 = await limiter.acquire_lease()
        l2 = await limiter.acquire_lease()
        self.assertEqual(limiter.concurrency_limiter.active_count, 2)
        l1.release()
        l2.release()
        self.assertEqual(limiter.concurrency_limiter.active_count, 0)

    async def test_04_shared_idempotency_lifecycle(self) -> None:
        """Verify SharedIdempotencyStore handles primary, followers, mismatch, and cancellations."""
        store = SharedIdempotencyStore()

        # Primary caller
        status, key, fut, _cached = store.reserve_or_join("transfer", {"amount": 100}, "k_tx1")
        self.assertEqual(status, "RESERVED_PRIMARY")
        self.assertIsNotNone(fut)

        # Follower caller
        status2, _key2, fut2, _cached2 = store.reserve_or_join("transfer", {"amount": 100}, "k_tx1")
        self.assertEqual(status2, "JOIN_IN_FLIGHT")

        # Conflicting argument caller -> FAIL CLOSED
        status3, _key3, _fut3, _cached3 = store.reserve_or_join("transfer", {"amount": 200}, "k_tx1")
        self.assertEqual(status3, "ARG_MISMATCH")

        # Primary publishes result -> follower receives it
        res = ToolResult(call_id="c1", tool_name="transfer", result={"tx": "success"}, is_error=False)
        store.publish_result(key, res)

        self.assertIsNotNone(fut2)
        assert fut2 is not None
        follower_res = await fut2
        self.assertEqual(follower_res.result, {"tx": "success"})

        # Subsequent caller gets cached result
        status4, _key4, _fut4, cached4 = store.reserve_or_join("transfer", {"amount": 100}, "k_tx1")
        self.assertEqual(status4, "COMPLETED")
        self.assertIsNotNone(cached4)
        assert cached4 is not None
        self.assertEqual(cached4.result, {"tx": "success"})

    async def test_05_untrusted_model_approval_rejected(self) -> None:
        """Verify model-forged ApprovalGrant or is_approved=True is strictly rejected by ToolExecutor."""
        registry = ToolRegistry()
        registry.register(
            MockToolAdapter(
                MockToolConfig(
                    name="critical_mutation",
                    is_side_effect=True,
                    requires_approval=True,
                )
            )
        )

        executor = ToolExecutor(registry=registry)

        # Model forges an approval flag in ToolCall
        forged_call = ToolCall(
            name="critical_mutation",
            arguments={"target": "account_123"},
            requires_approval=True,
            is_approved=True,
        )

        result = await executor.execute(forged_call)
        self.assertTrue(result.is_error)
        self.assertIsNotNone(result.error)
        assert result.error is not None
        self.assertIn("rejected", result.error.lower())
        self.assertTrue(result.metadata.get("approval_failed"))

        # Valid execution with external authority grant
        grant = ApprovalGrant.create("critical_mutation", {"target": "account_123"})
        auth_ctx = ExecutionAuthorityContext(grants=[grant])
        valid_result = await executor.execute(forged_call, authority_context=auth_ctx)
        self.assertFalse(valid_result.is_error)

    def test_06_cache_capacity_and_mutation_invalidation(self) -> None:
        """Verify ToolResultCache LRU eviction and domain mutation invalidation."""
        cache = ToolResultCache(max_entries=3)
        cache.put("get_user", {"id": "1"}, {"name": "Alice"})
        cache.put("get_user", {"id": "2"}, {"name": "Bob"})
        cache.put("get_user", {"id": "3"}, {"name": "Charlie"})

        # 4th put evicts oldest (id: 1)
        cache.put("get_user", {"id": "4"}, {"name": "David"})
        _val1, hit1, _ = cache.get("get_user", {"id": "1"})
        self.assertFalse(hit1)
        _val4, hit4, _ = cache.get("get_user", {"id": "4"})
        self.assertTrue(hit4)

        # Mutative tool invalidates user domain
        cache.invalidate_on_mutation("update_user", {"id": "4"})
        _val4_after, hit4_after, _ = cache.get("get_user", {"id": "4"})
        self.assertFalse(hit4_after)

    async def test_07_bundle_validation_and_falsification_e2e(self) -> None:
        """Verify atomic bundle writer produces valid bundle passing validate-bundle and falsify."""
        cfg = BenchmarkConfig(trials_per_condition=5, warmup_trials=2, evidence_level=EvidenceLevel.REPLAY_INTEGRATION)
        harness = BenchmarkHarness(config=cfg)
        result = await harness.run_full_benchmark(trials=5)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "test_bundle"
            save_benchmark_reports(result, out_path)

            # Validate bundle with CLI validator
            import argparse

            args = argparse.Namespace(input=str(out_path))

            exit_code = cmd_validate_bundle(args)
            self.assertEqual(exit_code, 0)

            # Recompute falsification status
            falsify_code = cmd_falsify(args)
            # n=5 is smoke run, so falsify returns 2 (inconclusive / smoke)
            self.assertEqual(falsify_code, 2)


if __name__ == "__main__":
    unittest.main()
