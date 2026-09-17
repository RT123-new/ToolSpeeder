"""Tests verifying strict candidate immutability, argument integrity, and hash stability."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

from toolspeed.schedulers.speculative_providers import (
    SpeculationCandidate,
    SpeculationState,
    TypeSafeJevProvider,
)
from toolspeed.schedulers.speculative_providers.replay import compute_composite_hash


class TestCandidateIntegrity(unittest.IsolatedAsyncioTestCase):
    """Verifies that speculative candidates are tamper-proof and immutable."""

    def test_candidate_arguments_are_immutable(self) -> None:
        cand = SpeculationCandidate(
            candidate_id="cand_1",
            tool_name="read_user_profile",
            arguments={"user_id": 42},
            is_read_only=True,
        )
        # Attempting mutation on arguments mapping must raise TypeError
        with self.assertRaises(TypeError):
            cand.arguments["user_id"] = 999  # type: ignore[index]

        with self.assertRaises(TypeError):
            cand.arguments["injected_arg"] = "malicious"  # type: ignore[index]

    def test_candidate_set_hash_sensitivity(self) -> None:
        state = SpeculationState(task_id="t1", prompt="test", step_index=0)

        cand_a1 = SpeculationCandidate(
            candidate_id="cand_1",
            tool_name="read_table",
            arguments={"table": "users"},
        )
        cand_a2 = SpeculationCandidate(
            candidate_id="cand_2",
            tool_name="read_table",
            arguments={"table": "orders"},
        )

        h1, req1, set1 = compute_composite_hash(state, [cand_a1, cand_a2])

        # If argument changes, hash must differ
        cand_b1 = SpeculationCandidate(
            candidate_id="cand_1",
            tool_name="read_table",
            arguments={"table": "users_modified"},
        )
        h2, req2, set2 = compute_composite_hash(state, [cand_b1, cand_a2])

        self.assertNotEqual(h1, h2)
        self.assertNotEqual(set1, set2)
        self.assertEqual(req1, req2)

    async def test_unknown_candidate_id_rejected(self) -> None:
        """If provider hallucinates or returns a candidate ID not in the set, it must be rejected."""
        mock_choice = MagicMock()
        mock_choice.choice = "cand_hallucinated_unknown"
        mock_choice.confidence = 0.99
        mock_choice.probabilities = {"cand_hallucinated_unknown": 0.99}

        mock_noul = MagicMock()
        mock_noul.noul = 0.99

        mock_response = MagicMock()
        mock_response.choices = {"route": mock_choice}
        mock_response.nouls = {"should_speculate": mock_noul}

        mock_client = AsyncMock()
        mock_client.system_one.return_value = mock_response

        provider = TypeSafeJevProvider(api_key="key", client=mock_client)

        state = SpeculationState(task_id="t1", prompt="lookup data", step_index=1)
        valid_candidates = [
            SpeculationCandidate(
                candidate_id="cand_valid_1",
                tool_name="lookup_data",
                arguments={"id": 1},
            )
        ]

        decision = await provider.decide(state, valid_candidates)
        self.assertFalse(decision.should_speculate)
        self.assertIsNone(decision.selected_candidate_id)


if __name__ == "__main__":
    unittest.main()
