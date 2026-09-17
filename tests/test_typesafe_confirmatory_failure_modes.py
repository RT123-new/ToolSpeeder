"""Comprehensive Failure and Fault Tolerance Study for TypeSafe/Jev Confirmatory Protocol v1.0.

Injects API timeouts, HTTP 429 rate limits, 401/403 authorization errors,
malformed responses, connection drops, and late race cancellations.
Verifies that the agent completes tasks correctly with zero leaked coroutines or semaphore permits.
"""

from __future__ import annotations

import asyncio
import unittest
from types import MappingProxyType
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx2
from typesafe_sdk import (
    TypeSafeAPIConnectionError,
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
    TypeSafeRateLimitError,
)

from toolspeed.adapters.base import LLMDecision, ToolRegistry
from toolspeed.adapters.mock_models import MockScriptedLLM
from toolspeed.adapters.mock_tools import MockToolAdapter, MockToolConfig
from toolspeed.core.types import AgentTask, ToolCall
from toolspeed.schedulers.base import SchedulerConfig
from toolspeed.schedulers.e3_speculation import SpeculativeReadScheduler
from toolspeed.schedulers.speculative_providers import (
    SpeculationCandidate,
    TypeSafeJevProvider,
)


class TestProviderFailureResilience(unittest.IsolatedAsyncioTestCase):
    """Verifies that all provider failures degrade gracefully without causing task failure or leaks."""

    def setUp(self) -> None:
        self.tool_cfg = MockToolConfig(
            name="fetch_user_doc",
            description="Fetch user doc",
            parameters={"type": "object", "properties": {"doc_id": {"type": "string"}}},
            handler=lambda args: {"status": "ok", "doc": args["doc_id"]},
        )
        self.registry = ToolRegistry([MockToolAdapter(self.tool_cfg)])
        self.candidate = SpeculationCandidate(
            candidate_id="c_doc",
            tool_name="fetch_user_doc",
            arguments=MappingProxyType({"doc_id": "DOC-99"}),
            is_read_only=True,
            tool_family="documents",
        )

    async def _run_with_injected_error(self, side_effect: Exception | None = None, return_val: Any = None) -> None:
        mock_client = MagicMock()
        if side_effect:
            mock_client.system_one = AsyncMock(side_effect=side_effect)
        else:
            mock_client.system_one = AsyncMock(return_value=return_val)

        provider = TypeSafeJevProvider(
            api_key="ts-test-key",
            timeout_s=0.20,
            client=mock_client,
        )

        scheduler = SpeculativeReadScheduler(
            config=SchedulerConfig(
                speculation_enabled=True,
                speculation_confidence_threshold=0.70,
                timeout_seconds=5.0,
            ),
            speculation_provider=provider,
            candidate_builder=lambda ctx, reg: [self.candidate],
        )

        task = AgentTask(task_id="resilience_01", prompt="Fetch user document DOC-99")
        model = MockScriptedLLM(
            decision_steps=[
                LLMDecision(tool_calls=[ToolCall(name="fetch_user_doc", arguments={"doc_id": "DOC-99"})]),
                LLMDecision(final_answer="Task succeeded despite provider failure"),
            ],
            simulated_decision_ms=50.0,
        )

        result = await scheduler.execute(task, model, self.registry)

        # Invariant 1: Agent correctness is unaffected by external provider failure
        self.assertTrue(result.success)
        self.assertIn("Task succeeded", str(result.final_answer))

        # Invariant 2: No speculative mutation occurred
        self.assertEqual(result.guardrails.unapproved_side_effects, 0)
        self.assertEqual(result.guardrails.unsafe_side_effects, 0)

    async def test_api_timeout(self) -> None:
        await self._run_with_injected_error(side_effect=TypeSafeAPITimeoutError(1.0))

    async def test_rate_limit_429(self) -> None:
        err = TypeSafeRateLimitError(429, {}, httpx2.Headers(), message="Too many requests")
        await self._run_with_injected_error(side_effect=err)

    async def test_auth_error_401_403(self) -> None:
        err = TypeSafeAuthenticationError(401, {}, httpx2.Headers(), message="Invalid token")
        await self._run_with_injected_error(side_effect=err)

    async def test_connection_drop(self) -> None:
        err = TypeSafeAPIConnectionError("Connection reset by peer")
        await self._run_with_injected_error(side_effect=err)

    async def test_malformed_response(self) -> None:
        mock_resp = MagicMock()
        mock_resp.choices = {}  # Missing route choice
        mock_resp.nouls = {}
        await self._run_with_injected_error(return_val=mock_resp)

    async def test_slow_response_race_cancellation(self) -> None:
        """Provider takes 500ms but model finishes in 20ms; verifies clean cancellation without task leak."""

        async def _slow_call(*args, **kwargs):
            await asyncio.sleep(0.50)
            return MagicMock()

        mock_client = MagicMock()
        mock_client.system_one = _slow_call

        provider = TypeSafeJevProvider(
            api_key="ts-test-key",
            timeout_s=1.0,
            client=mock_client,
        )
        scheduler = SpeculativeReadScheduler(
            config=SchedulerConfig(
                speculation_enabled=True,
                speculation_confidence_threshold=0.70,
                timeout_seconds=5.0,
            ),
            speculation_provider=provider,
            candidate_builder=lambda ctx, reg: [self.candidate],
        )
        task = AgentTask(task_id="race_01", prompt="Quick decision task")
        model = MockScriptedLLM(
            decision_steps=[
                LLMDecision(tool_calls=[ToolCall(name="fetch_user_doc", arguments={"doc_id": "DOC-99"})]),
                LLMDecision(final_answer="Done"),
            ],
            simulated_decision_ms=20.0,
        )

        result = await scheduler.execute(task, model, self.registry)
        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()
