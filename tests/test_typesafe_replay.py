"""Tests verifying deterministic replay provider contract and offline trace simulation."""

from __future__ import annotations

import unittest
from types import MappingProxyType

from toolspeed.schedulers.speculative_providers import (
    ReplaySpeculationProvider,
    SpeculationCandidate,
    SpeculationDecision,
    SpeculationState,
)
from toolspeed.schedulers.speculative_providers.replay import (
    ReplayRecord,
    compute_composite_hash,
)


class TestTypeSafeReplay(unittest.IsolatedAsyncioTestCase):
    """Verifies that ReplaySpeculationProvider operates deterministically and handles offline traces."""

    def setUp(self) -> None:
        self.state = SpeculationState(
            task_id="t_replay_01",
            prompt="Lookup user preferences for user_441",
            step_index=1,
            history_summary=(),
        )
        self.candidates = [
            SpeculationCandidate(
                candidate_id="c_prefs",
                tool_name="get_user_preferences",
                arguments=MappingProxyType({"user_id": "user_441"}),
                is_read_only=True,
                tool_family="crm",
            ),
            SpeculationCandidate(
                candidate_id="c_history",
                tool_name="get_purchase_history",
                arguments=MappingProxyType({"user_id": "user_441"}),
                is_read_only=True,
                tool_family="crm",
            ),
        ]

    async def test_in_memory_trace_deterministic_match(self) -> None:
        """Verifies that an in-memory record matches identical prompts and candidates."""
        comp_hash, req_hash, cand_hash = compute_composite_hash(self.state, self.candidates)

        record = ReplayRecord(
            composite_hash=comp_hash,
            request_hash=req_hash,
            candidate_set_hash=cand_hash,
            provider="typesafe_jev",
            selected_candidate_id="c_prefs",
            probability=0.94,
            confidence=0.94,
            should_speculate=True,
            observed_latency_ms=10.0,
            probabilities={"c_prefs": 0.94, "c_history": 0.04, "no_speculation": 0.02},
            metadata={"model": "jev-latest"},
        )

        provider = ReplaySpeculationProvider(records=[record], simulate_latency=True)
        decision = await provider.decide(self.state, self.candidates, confidence_threshold=0.80)

        self.assertIsInstance(decision, SpeculationDecision)
        self.assertEqual(decision.selected_candidate_id, "c_prefs")
        self.assertTrue(decision.should_speculate)
        self.assertEqual(decision.confidence, 0.94)
        self.assertEqual(decision.probability, 0.94)
        self.assertFalse(decision.fallback_used)
        self.assertGreaterEqual(decision.latency_ms, 10.0)

    async def test_trace_export_and_import(self) -> None:
        """Verifies exporting records to dicts and re-importing."""
        comp_hash, req_hash, cand_hash = compute_composite_hash(self.state, self.candidates)

        record = ReplayRecord(
            composite_hash=comp_hash,
            request_hash=req_hash,
            candidate_set_hash=cand_hash,
            provider="typesafe_jev",
            selected_candidate_id="c_history",
            probability=0.82,
            confidence=0.85,
            should_speculate=True,
            observed_latency_ms=5.0,
            probabilities={"c_history": 0.82},
            metadata={},
        )

        exported = [record.to_dict()]
        provider = ReplaySpeculationProvider(records=exported, simulate_latency=False)
        decision = await provider.decide(self.state, self.candidates)

        self.assertEqual(decision.selected_candidate_id, "c_history")
        self.assertTrue(decision.should_speculate)
        self.assertEqual(decision.confidence, 0.85)

    async def test_unrecorded_state_triggers_graceful_fallback(self) -> None:
        """Verifies that queries not present in the trace fall back safely."""
        provider = ReplaySpeculationProvider(records=[])

        decision = await provider.decide(self.state, self.candidates)
        self.assertFalse(decision.should_speculate)
        self.assertTrue(decision.fallback_used)
        self.assertEqual(decision.error_class, "REPLAY_RECORD_NOT_FOUND")

    async def test_candidate_tampering_invalidates_hash(self) -> None:
        """Verifies that altering candidate arguments changes the SHA-256 hash preventing replay."""
        hash1, _, _ = compute_composite_hash(self.state, self.candidates)

        tampered_candidates = [
            SpeculationCandidate(
                candidate_id="c_prefs",
                tool_name="get_user_preferences",
                arguments=MappingProxyType({"user_id": "ATTACKER_INJECTED_ID"}),
                is_read_only=True,
                tool_family="crm",
            )
        ]
        hash2, _, _ = compute_composite_hash(self.state, tampered_candidates)

        self.assertNotEqual(hash1, hash2, "Tampered candidate args must yield a different state hash")


if __name__ == "__main__":
    unittest.main()
