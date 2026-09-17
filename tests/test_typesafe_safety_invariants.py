"""Adversarial safety tests verifying that Jev can never cause mutative speculation."""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from toolspeed.adapters.base import ToolRegistry
from toolspeed.adapters.mock_tools import MockToolAdapter, MockToolConfig
from toolspeed.core.guardrails import GuardrailMonitor
from toolspeed.core.types import AgentTask, ToolCall
from toolspeed.schedulers.base import ExecutionContext, SchedulerConfig
from toolspeed.schedulers.e3_speculation import SpeculativeReadScheduler
from toolspeed.schedulers.executor import ToolExecutor
from toolspeed.schedulers.speculative_providers import (
    SpeculationCandidate,
    SpeculationState,
    TypeSafeJevProvider,
)


class TestSafetyInvariants(unittest.IsolatedAsyncioTestCase):
    """Verifies that mutative actions are strictly blocked from speculative execution."""

    async def test_provider_rejects_mutative_candidate_even_at_full_confidence(self) -> None:
        """Provider must reject non-read-only candidate even if Jev outputs confidence 1.0."""
        mock_choice = MagicMock()
        mock_choice.choice = "cand_mutative"
        mock_choice.confidence = 1.0
        mock_choice.probabilities = {"cand_mutative": 1.0}

        mock_noul = MagicMock()
        mock_noul.noul = 1.0

        mock_response = MagicMock()
        mock_response.choices = {"route": mock_choice}
        mock_response.nouls = {"should_speculate": mock_noul}

        mock_client = AsyncMock()
        mock_client.system_one.return_value = mock_response

        provider = TypeSafeJevProvider(api_key="key", client=mock_client)

        state = SpeculationState(task_id="t_safe", prompt="execute action", step_index=1)
        mutative_cand = SpeculationCandidate(
            candidate_id="cand_mutative",
            tool_name="delete_database_records",
            arguments={"table": "production_logs"},
            is_read_only=False,  # Mutative!
        )

        decision = await provider.decide(state, [mutative_cand], confidence_threshold=0.50)

        # Must decline speculation despite confidence = 1.0
        self.assertFalse(decision.should_speculate)
        self.assertIsNone(decision.selected_candidate_id)
        self.assertEqual(decision.error_class, "UNSAFE_CANDIDATE_REJECTED")

    async def test_downstream_toolexecutor_gate_blocks_disguised_mutative_speculation(self) -> None:
        """Even if an adversary bypasses the scheduler and marks call.is_speculative=True,

        ToolExecutor must unconditionally reject any tool that has side_effects or requires_approval.
        """
        mutated = False

        def _bad_handler(args: dict[str, Any]) -> dict[str, Any]:
            nonlocal mutated
            mutated = True
            return {"status": "mutated"}

        # Mutative tool requiring approval with side effects
        mutative_tool = MockToolAdapter(
            MockToolConfig(
                name="transfer_funds_mutative",
                description="Transfer money",
                parameters={"type": "object", "properties": {"amount": {"type": "number"}}},
                is_side_effect=True,
                requires_approval=True,
                handler=_bad_handler,
            )
        )
        registry = ToolRegistry([mutative_tool])
        monitor = GuardrailMonitor()
        executor = ToolExecutor(registry=registry, guardrails=monitor)

        malicious_speculative_call = ToolCall(
            name="transfer_funds_mutative",
            tool_name="transfer_funds_mutative",
            arguments={"amount": 10_000.0},
            is_speculative=True,  # Disguised as speculative
        )

        # Execute through ToolExecutor under speculative flag
        res = await executor.execute(malicious_speculative_call, is_speculative=True)

        # Must return error and block underlying execution
        self.assertTrue(res.is_error)
        self.assertIsNotNone(res.error)
        self.assertIn("prohibited", str(res.error).lower())
        self.assertFalse(mutated, "Mutative tool handler must NOT have executed!")

        metrics = monitor.get_metrics()
        self.assertGreater(metrics.blocked_unsafe_attempts, 0)
        self.assertEqual(metrics.unsafe_side_effects, 0)

    async def test_scheduler_never_speculates_approval_requiring_tool(self) -> None:
        """SpeculativeReadScheduler must never launch a tool requiring approval."""
        approval_tool = MockToolAdapter(
            MockToolConfig(
                name="approve_wire_transfer",
                description="Sensitive wire approval",
                parameters={"type": "object"},
                is_side_effect=True,
                requires_approval=True,
            )
        )
        registry = ToolRegistry([approval_tool])

        # Mock a provider that claims approval_wire_transfer should speculate
        mock_provider = AsyncMock()
        mock_provider.provider_name = "adversarial_mock"
        from toolspeed.schedulers.speculative_providers import SpeculationDecision
        mock_provider.decide.return_value = SpeculationDecision(
            selected_candidate_id="cand_wire",
            probability=1.0,
            confidence=1.0,
            should_speculate=True,
            provider="adversarial_mock",
        )

        def bad_candidate_builder(ctx: Any, tools: Any) -> list[SpeculationCandidate]:
            return [
                SpeculationCandidate(
                    candidate_id="cand_wire",
                    tool_name="approve_wire_transfer",
                    arguments={},
                    is_read_only=False,
                )
            ]

        scheduler = SpeculativeReadScheduler(
            speculation_provider=mock_provider,
            candidate_builder=bad_candidate_builder,
        )

        ctx = ExecutionContext(
            task=AgentTask(task_id="t_adv", prompt="approve wire"),
            config=SchedulerConfig(speculation_enabled=True),
            tools=registry,
        )

        pred = await scheduler._predict_candidate(ctx, MagicMock(), registry, threshold=0.70)
        # Prediction must be None because is_read_only is False
        self.assertIsNone(pred)


if __name__ == "__main__":
    unittest.main()
