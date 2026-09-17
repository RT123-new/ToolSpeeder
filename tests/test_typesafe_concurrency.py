"""Tests verifying concurrency safety, cancellation cleanup, and permit release."""

from __future__ import annotations

import asyncio
import unittest
from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock

from toolspeed.schedulers.base import cancel_and_await
from toolspeed.schedulers.speculative_providers import (
    SpeculationCandidate,
    SpeculationDecision,
    SpeculationState,
    TypeSafeJevProvider,
)


class TestTypeSafeConcurrency(unittest.IsolatedAsyncioTestCase):
    """Verifies that TypeSafe integration is thread-safe, coroutine-safe, and leak-free."""

    def setUp(self) -> None:
        self.state = SpeculationState(
            task_id="task_concurrency_01",
            prompt="Analyze server metrics and query log traces",
            step_index=1,
            history_summary=(),
        )
        self.candidates = [
            SpeculationCandidate(
                candidate_id="c_metrics",
                tool_name="get_server_metrics",
                arguments=MappingProxyType({"server_id": "srv-99"}),
                is_read_only=True,
                tool_family="infra",
            ),
            SpeculationCandidate(
                candidate_id="c_logs",
                tool_name="get_log_traces",
                arguments=MappingProxyType({"server_id": "srv-99"}),
                is_read_only=True,
                tool_family="infra",
            ),
        ]

    async def test_concurrent_decide_invocations(self) -> None:
        """Verifies that multiple concurrent callers to provider.decide() execute safely."""
        mock_choice = MagicMock()
        mock_choice.choice = "c_metrics"
        mock_choice.confidence = 0.95
        mock_choice.probabilities = {"c_metrics": 0.90, "c_logs": 0.05, "no_speculation": 0.05}

        mock_noul = MagicMock()
        mock_noul.noul = 0.92

        mock_response = MagicMock()
        mock_response.choices = {"route": mock_choice}
        mock_response.nouls = {"should_speculate": mock_noul}

        # Simulate a realistic 10ms network delay per inference call
        async def mock_system_one(*args: object, **kwargs: object) -> MagicMock:
            await asyncio.sleep(0.01)
            return mock_response

        mock_client = AsyncMock()
        mock_client.system_one.side_effect = mock_system_one

        provider = TypeSafeJevProvider(api_key="mock_key", client=mock_client)

        # Launch 25 concurrent decide requests
        num_requests = 25
        tasks = [
            asyncio.create_task(provider.decide(self.state, self.candidates, confidence_threshold=0.70))
            for _ in range(num_requests)
        ]

        decisions = await asyncio.gather(*tasks)

        self.assertEqual(len(decisions), num_requests)
        for d in decisions:
            self.assertIsInstance(d, SpeculationDecision)
            self.assertEqual(d.selected_candidate_id, "c_metrics")
            self.assertTrue(d.should_speculate)
            self.assertEqual(d.provider, "typesafe_jev")
            self.assertFalse(d.fallback_used)

    async def test_task_cancellation_during_inference(self) -> None:
        """Verifies that cancelling a decision task in flight terminates cleanly without leaking."""
        started_event = asyncio.Event()
        cancelled_cleanly = False

        async def long_running_call(*args: object, **kwargs: object) -> MagicMock:
            started_event.set()
            try:
                await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                nonlocal cancelled_cleanly
                cancelled_cleanly = True
                raise
            return MagicMock()

        mock_client = AsyncMock()
        mock_client.system_one.side_effect = long_running_call

        provider = TypeSafeJevProvider(api_key="mock_key", client=mock_client)

        task = asyncio.create_task(provider.decide(self.state, self.candidates, confidence_threshold=0.70))

        # Wait until inference starts, then cancel it
        await started_event.wait()
        await cancel_and_await(task)

        self.assertTrue(task.done())
        self.assertTrue(task.cancelled())
        self.assertTrue(cancelled_cleanly, "Underlying inference coroutine was not cancelled cleanly")

    async def test_timeout_cancels_in_flight_underlying_request(self) -> None:
        """Verifies that an external API timeout promptly aborts the pending request."""
        was_aborted = False

        async def hanging_call(*args: object, **kwargs: object) -> MagicMock:
            try:
                await asyncio.sleep(10.0)
            except asyncio.CancelledError:
                nonlocal was_aborted
                was_aborted = True
                raise
            return MagicMock()

        mock_client = AsyncMock()
        mock_client.system_one.side_effect = hanging_call

        provider = TypeSafeJevProvider(api_key="mock_key", timeout_s=0.05, client=mock_client)

        decision = await provider.decide(self.state, self.candidates)

        self.assertFalse(decision.should_speculate)
        self.assertTrue(decision.fallback_used)
        self.assertEqual(decision.error_class, "TIMEOUT")
        self.assertTrue(was_aborted, "In-flight HTTP coroutine was not cancelled after timeout")


if __name__ == "__main__":
    unittest.main()
