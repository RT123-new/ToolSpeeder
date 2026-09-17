"""Tests verifying robust failure handling and graceful degradation in TypeSafe provider."""

from __future__ import annotations

import asyncio
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx2
from typesafe_sdk import (
    TypeSafeAPIConnectionError,
    TypeSafeAuthenticationError,
    TypeSafeRateLimitError,
)

from toolspeed.schedulers.speculative_providers import (
    NoSpeculationProvider,
    SpeculationCandidate,
    SpeculationState,
    TypeSafeJevProvider,
)


class TestFailureHandling(unittest.IsolatedAsyncioTestCase):
    """Verifies that provider errors fall back gracefully without task failure."""

    def setUp(self) -> None:
        self.state = SpeculationState(task_id="t_fail", prompt="fetch record", step_index=1)
        self.candidates = [
            SpeculationCandidate(
                candidate_id="c1",
                tool_name="read_record",
                arguments={"id": "rec_001"},
                is_read_only=True,
            )
        ]

    async def test_timeout_fallback(self) -> None:
        mock_client = AsyncMock()

        async def slow_call(*args: Any, **kwargs: Any) -> None:
            await asyncio.sleep(0.5)

        mock_client.system_one.side_effect = slow_call

        # Provider with 50ms timeout
        provider = TypeSafeJevProvider(api_key="k", timeout_s=0.05, client=mock_client)
        decision = await provider.decide(self.state, self.candidates)

        self.assertFalse(decision.should_speculate)
        self.assertTrue(decision.fallback_used)
        self.assertEqual(decision.error_class, "TIMEOUT")

    async def test_rate_limit_429_fallback(self) -> None:
        mock_client = AsyncMock()
        mock_client.system_one.side_effect = TypeSafeRateLimitError(429, {}, httpx2.Headers(), message="Rate limit exceeded")

        provider = TypeSafeJevProvider(api_key="k", client=mock_client)
        decision = await provider.decide(self.state, self.candidates)

        self.assertFalse(decision.should_speculate)
        self.assertTrue(decision.fallback_used)
        self.assertEqual(decision.error_class, "RATE_LIMIT_429")

    async def test_authentication_401_403_fallback(self) -> None:
        mock_client = AsyncMock()
        mock_client.system_one.side_effect = TypeSafeAuthenticationError(401, {}, httpx2.Headers(), message="Unauthorized")

        provider = TypeSafeJevProvider(api_key="k", client=mock_client)
        decision = await provider.decide(self.state, self.candidates)

        self.assertFalse(decision.should_speculate)
        self.assertTrue(decision.fallback_used)
        self.assertEqual(decision.error_class, "AUTH_FAILURE_401_403")

    async def test_connection_error_fallback(self) -> None:
        mock_client = AsyncMock()
        mock_client.system_one.side_effect = TypeSafeAPIConnectionError("Connection refused")

        provider = TypeSafeJevProvider(api_key="k", client=mock_client)
        decision = await provider.decide(self.state, self.candidates)

        self.assertFalse(decision.should_speculate)
        self.assertTrue(decision.fallback_used)
        self.assertEqual(decision.error_class, "CONNECTION_FAILURE")

    async def test_absent_api_key_graceful_fallback(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            provider = TypeSafeJevProvider(api_key=None)
            decision = await provider.decide(self.state, self.candidates)

            self.assertFalse(decision.should_speculate)
            self.assertTrue(decision.fallback_used)
            self.assertEqual(decision.error_class, "TYPESAFE_API_KEY_MISSING")

    async def test_custom_fallback_provider_delegation(self) -> None:
        mock_client = AsyncMock()
        mock_client.system_one.side_effect = Exception("General network failure")

        fallback_baseline = NoSpeculationProvider()
        provider = TypeSafeJevProvider(
            api_key="k",
            client=mock_client,
            fallback_provider=fallback_baseline,
        )

        decision = await provider.decide(self.state, self.candidates)
        self.assertTrue(decision.fallback_used)
        self.assertEqual(decision.metadata.get("fallback_to"), "no_speculation")


if __name__ == "__main__":
    unittest.main()
