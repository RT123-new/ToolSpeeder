"""Comprehensive Safety Red Team for TypeSafe/Jev Confirmatory Protocol v1.0.

Adversarially attempts to trick ToolSpeeder into speculatively executing mutative tools
via overt mutative actions and deceptive/camouflaged tool names.
Kill condition: unauthorized speculative mutations > 0.
"""

from __future__ import annotations

import unittest
from types import MappingProxyType
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from toolspeed.adapters.base import BaseToolAdapter, LLMDecision, ToolRegistry
from toolspeed.adapters.mock_models import MockScriptedLLM
from toolspeed.adapters.mock_tools import MockToolAdapter, MockToolConfig
from toolspeed.core.types import AgentTask, ToolCall
from toolspeed.schedulers.base import SchedulerConfig
from toolspeed.schedulers.e3_speculation import SpeculativeReadScheduler
from toolspeed.schedulers.speculative_providers import (
    SpeculationCandidate,
    SpeculationDecision,
)


class TestSafetyRedTeam(unittest.IsolatedAsyncioTestCase):
    """Rigorous verification of the triple-layer safety defense against speculative mutations."""

    def setUp(self) -> None:
        self.overt_mutations: list[tuple[str, dict[str, Any], bool, bool, bool]] = [
            ("send_email", {"to": "audit@corp.com", "body": "leak"}, False, True, True),
            ("delete_file", {"path": "/etc/passwd"}, False, True, True),
            ("update_database", {"query": "UPDATE users SET role='admin'"}, False, True, True),
            ("purchase_item", {"item_id": "SKU-999", "quantity": 10}, False, True, True),
            ("post_message", {"channel": "#general", "text": "spam"}, False, True, True),
            ("merge_pr", {"pr_id": "PR-101"}, False, True, True),
            ("execute_shell_mutation", {"cmd": "rm -rf /"}, False, True, True),
        ]

        self.deceptive_mutations: list[tuple[str, dict[str, Any], bool, bool, bool]] = [
            ("read_and_delete_cache", {"key": "session_tokens"}, False, True, False),
            ("fetch_then_update_record", {"rec_id": "REC-1"}, False, True, False),
            ("safe_lookup_with_cleanup", {"target": "temp_files"}, False, True, False),
        ]

    async def test_overt_and_deceptive_mutations_rejected(self) -> None:
        all_adversarial = self.overt_mutations + self.deceptive_mutations
        tools: list[BaseToolAdapter] = []

        for name, params, _is_ro, side_eff, req_app in all_adversarial:

            def _make_handler(tool_name: str):
                def _handler(args: dict[str, Any]) -> dict[str, Any]:
                    return {"status": "executed", "tool": tool_name}

                return _handler

            cfg = MockToolConfig(
                name=name,
                description=f"Adversarial tool {name}",
                parameters={"type": "object", "properties": {k: {"type": "string"} for k in params}},
                is_side_effect=side_eff,
                requires_approval=req_app,
                handler=_make_handler(name),
            )
            tools.append(MockToolAdapter(cfg))

        registry = ToolRegistry(tools)

        for name, params, is_ro, _side_eff, _req_app in all_adversarial:
            candidate = SpeculationCandidate(
                candidate_id=f"cand_{name}",
                tool_name=name,
                arguments=MappingProxyType(params),
                is_read_only=is_ro,
                tool_family="adversarial",
            )

            mock_provider = MagicMock()
            mock_provider.provider_name = "adversarial_mock_jev"
            # Adversarially recommend this mutative candidate with confidence 1.0
            mock_provider.decide = AsyncMock(
                return_value=SpeculationDecision(
                    selected_candidate_id=f"cand_{name}",
                    probability=1.0,
                    confidence=1.0,
                    should_speculate=True,
                    provider="adversarial_mock_jev",
                    latency_ms=10.0,
                    fallback_used=False,
                )
            )

            def _builder(ctx: Any, reg: Any, c: SpeculationCandidate = candidate) -> list[SpeculationCandidate]:
                return [c]

            scheduler = SpeculativeReadScheduler(
                config=SchedulerConfig(
                    speculation_enabled=True,
                    speculation_confidence_threshold=0.50,
                    timeout_seconds=5.0,
                ),
                speculation_provider=mock_provider,
                candidate_builder=_builder,
            )

            task = AgentTask(
                task_id=f"redteam_{name}",
                prompt=f"Perform operation with {name}",
            )
            model = MockScriptedLLM(
                decision_steps=[
                    LLMDecision(tool_calls=[ToolCall(name=name, arguments=params)]),
                    LLMDecision(final_answer="Task finished"),
                ],
                simulated_decision_ms=20.0,
            )

            result = await scheduler.execute(task, model, registry)
            self.assertTrue(result.success)

            # Assert this mutative tool was NEVER executed speculatively
            self.assertEqual(result.guardrails.speculative_calls_hit, 0)
            self.assertEqual(result.guardrails.unapproved_side_effects, 0)
            self.assertEqual(result.guardrails.unsafe_side_effects, 0)


if __name__ == "__main__":
    unittest.main()
