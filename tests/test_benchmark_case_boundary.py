"""Unit tests for BenchmarkCase abstraction and zero-oracle scheduler boundary."""

from __future__ import annotations

import unittest

from toolspeed.adapters.base import BaseLLMAdapter, LLMDecision, ToolRegistry
from toolspeed.core.types import BenchmarkCase
from toolspeed.schedulers.b1_sync_react import SyncReActScheduler
from toolspeed.schedulers.base import SchedulerConfig


class TestBenchmarkCaseBoundary(unittest.IsolatedAsyncioTestCase):
    """Proves that BenchmarkCase enforces a strict zero-oracle boundary for schedulers."""

    def test_01_case_to_model_task_strips_expected_answers_and_validators(self) -> None:
        """to_model_task() must strictly remove expected_output, validator, and oracle metadata."""
        case = BenchmarkCase(
            task_id="case_001",
            prompt="Run query",
            expected_output={"status": "success", "secret": "oracle_leak_payload"},
            validator=None,
            context={"user_id": "u123", "secret_key": "untrusted_internal_key"},
            parameters={"query": "test"},
            metadata={"secret_canary": "backdoor_value", "workload_family": "database"},
        )

        model_task = case.to_model_task()

        self.assertEqual(model_task.task_id, "case_001")
        self.assertEqual(model_task.prompt, "Run query")
        # Assert oracles are stripped
        self.assertIsNone(model_task.expected_output)
        self.assertIsNone(model_task.validator)
        # Assert non-whitelisted metadata is stripped
        self.assertNotIn("secret_canary", model_task.metadata)
        # Verify validation still works through the original BenchmarkCase
        self.assertTrue(case.validate({"status": "success", "secret": "oracle_leak_payload"}))
        self.assertFalse(case.validate({"status": "failed"}))

    def test_02_case_to_agent_task_strips_oracles(self) -> None:
        """to_agent_task() must produce an AgentTask with no access to test oracles."""
        case = BenchmarkCase(
            task_id="case_002",
            prompt="Run query 2",
            expected_output={"result": 42},
            validator=lambda x: True,
            context={"user": "alice"},
            metadata={"oracle_secret": "forbidden", "workload_family": "analytics"},
        )

        agent_task = case.to_agent_task()
        self.assertEqual(agent_task.task_id, "case_002")
        self.assertFalse(hasattr(agent_task, "expected_output"))
        self.assertFalse(hasattr(agent_task, "validator"))
        self.assertNotIn("oracle_secret", agent_task.metadata)

    async def test_03_scheduler_cannot_read_expected_answers(self) -> None:
        """A scheduler attempting to access expected_output during execution must see None."""
        case = BenchmarkCase(
            task_id="case_003",
            prompt="Compute answer",
            expected_output={"answer": 12345},
            validator=lambda x: x == {"answer": 12345},
        )

        # Build a task with stripped expected_output
        model_task = case.to_model_task()
        self.assertIsNone(model_task.expected_output)

        scheduler = SyncReActScheduler(SchedulerConfig())

        class DummyModel(BaseLLMAdapter):
            async def decide(self, *args: object, **kwargs: object) -> LLMDecision:
                return LLMDecision(tool_calls=[], final_answer={"answer": 12345})

        registry = ToolRegistry()
        result = await scheduler.execute(model_task, DummyModel(), registry)

        self.assertEqual(result.final_answer, {"answer": 12345})
        # The scheduler could not inspect expected_output because it was stripped
        self.assertTrue(case.validate(result.final_answer))


if __name__ == "__main__":
    unittest.main()
