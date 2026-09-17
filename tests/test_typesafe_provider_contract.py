"""Tests for SpeculationDecisionProvider contracts across all implementations."""

from __future__ import annotations

import unittest
from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock

from toolspeed.schedulers.speculative_providers import (
    DeterministicFrequencyProvider,
    NoSpeculationProvider,
    SpeculationCandidate,
    SpeculationDecision,
    SpeculationState,
    TypeSafeJevProvider,
)


class TestProviderContract(unittest.IsolatedAsyncioTestCase):
    """Verifies that all providers strictly satisfy the SpeculationDecisionProvider contract."""

    def setUp(self) -> None:
        self.state = SpeculationState(
            task_id="task_contract_01",
            prompt="Find invoice INV-1001 for customer CUST-500",
            step_index=1,
            history_summary=(),
        )
        self.candidates = [
            SpeculationCandidate(
                candidate_id="c_invoice",
                tool_name="fetch_invoice_document",
                arguments=MappingProxyType({"doc_id": "INV-1001"}),
                is_read_only=True,
                tool_family="finance",
            ),
            SpeculationCandidate(
                candidate_id="c_orders",
                tool_name="query_order_history",
                arguments=MappingProxyType({"customer_id": "CUST-500"}),
                is_read_only=True,
                tool_family="crm",
            ),
        ]

    async def test_no_speculation_provider(self) -> None:
        provider = NoSpeculationProvider()
        decision = await provider.decide(self.state, self.candidates)

        self.assertIsInstance(decision, SpeculationDecision)
        self.assertFalse(decision.should_speculate)
        self.assertIsNone(decision.selected_candidate_id)
        self.assertEqual(decision.provider, "no_speculation")
        self.assertEqual(decision.probability, 0.0)
        self.assertEqual(decision.confidence, 0.0)

    async def test_deterministic_provider_selection(self) -> None:
        provider = DeterministicFrequencyProvider(latency_ms=1.0)
        decision = await provider.decide(self.state, self.candidates, confidence_threshold=0.50)

        self.assertIsInstance(decision, SpeculationDecision)
        self.assertIn(decision.selected_candidate_id, ["c_invoice", "c_orders"])
        self.assertTrue(decision.should_speculate)
        self.assertGreaterEqual(decision.confidence, 0.50)
        self.assertGreaterEqual(decision.latency_ms, 1.0)

    async def test_typesafe_jev_mock_contract(self) -> None:
        # Construct mock response following official SystemOneResponse structure
        mock_choice = MagicMock()
        mock_choice.choice = "c_invoice"
        mock_choice.confidence = 0.92
        mock_choice.probabilities = {"c_invoice": 0.88, "c_orders": 0.08, "no_speculation": 0.04}

        mock_noul = MagicMock()
        mock_noul.noul = 0.95

        mock_response = MagicMock()
        mock_response.choices = {"route": mock_choice}
        mock_response.nouls = {"should_speculate": mock_noul}
        mock_response.model = "jev-latest"

        mock_client = AsyncMock()
        mock_client.system_one.return_value = mock_response

        provider = TypeSafeJevProvider(api_key="test_key", client=mock_client)
        decision = await provider.decide(self.state, self.candidates, confidence_threshold=0.70)

        self.assertIsInstance(decision, SpeculationDecision)
        self.assertEqual(decision.selected_candidate_id, "c_invoice")
        self.assertTrue(decision.should_speculate)
        self.assertEqual(decision.confidence, 0.92)
        self.assertEqual(decision.probability, 0.88)
        self.assertEqual(decision.provider, "typesafe_jev")
        self.assertFalse(decision.fallback_used)
        self.assertIsNone(decision.error_class)

    async def test_typesafe_jev_selects_no_speculation(self) -> None:
        mock_choice = MagicMock()
        mock_choice.choice = "no_speculation"
        mock_choice.confidence = 0.80
        mock_choice.probabilities = {"c_invoice": 0.10, "c_orders": 0.10, "no_speculation": 0.80}

        mock_noul = MagicMock()
        mock_noul.noul = 0.10

        mock_response = MagicMock()
        mock_response.choices = {"route": mock_choice}
        mock_response.nouls = {"should_speculate": mock_noul}

        mock_client = AsyncMock()
        mock_client.system_one.return_value = mock_response

        provider = TypeSafeJevProvider(api_key="test_key", client=mock_client)
        decision = await provider.decide(self.state, self.candidates, confidence_threshold=0.70)

        self.assertFalse(decision.should_speculate)
        self.assertIsNone(decision.selected_candidate_id)

    async def test_candidate_id_mapping_and_confidence_threshold_filtering(self) -> None:
        mock_choice = MagicMock()
        mock_choice.choice = "c_invoice"
        mock_choice.confidence = 0.65  # Below 0.70 threshold
        mock_choice.probabilities = {"c_invoice": 0.65, "c_orders": 0.25, "no_speculation": 0.10}

        mock_noul = MagicMock()
        mock_noul.noul = 0.80

        mock_response = MagicMock()
        mock_response.choices = {"route": mock_choice}
        mock_response.nouls = {"should_speculate": mock_noul}

        mock_client = AsyncMock()
        mock_client.system_one.return_value = mock_response

        provider = TypeSafeJevProvider(api_key="test_key", client=mock_client)
        decision = await provider.decide(self.state, self.candidates, confidence_threshold=0.70)

        # Candidate was chosen, but should_speculate is False due to threshold
        self.assertEqual(decision.selected_candidate_id, "c_invoice")
        self.assertFalse(decision.should_speculate)
        self.assertEqual(decision.confidence, 0.65)


if __name__ == "__main__":
    unittest.main()
