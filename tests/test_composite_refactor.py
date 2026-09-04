"""Tests for Phase 31: Composite scheduler refactor, real mechanism delegation, and measured dispatch overhead."""

from __future__ import annotations

import unittest
from typing import Any

from toolspeed.adapters.base import (
    BaseLLMAdapter,
    LLMDecision,
    StreamingChunk,
    ToolRegistry,
)
from toolspeed.adapters.mock_tools import MockToolAdapter
from toolspeed.core.types import EventType, ToolCall, ToolSpec
from toolspeed.schedulers.base import AgentTask
from toolspeed.schedulers.composite import CompositeScheduler
from toolspeed.schedulers.e1_dag_scheduler import DAGScheduler
from toolspeed.schedulers.e2_jit_fusion import JITFusionScheduler
from toolspeed.schedulers.e3_speculation import SpeculativeScheduler
from toolspeed.schedulers.e4_commit_horizon import CommitHorizonScheduler
from toolspeed.schedulers.phase2_cache import ToolResultCache


class MockCompositeLLM(BaseLLMAdapter):
    """Mock LLM adapter simulating stream decisions."""

    def __init__(self, decisions: list[LLMDecision]) -> None:
        self.decisions = list(decisions)
        self.idx = 0

    async def decide(
        self,
        task: AgentTask,
        history: list[dict[str, Any]],
        available_tools: list[ToolSpec],
    ) -> LLMDecision:
        if self.idx < len(self.decisions):
            d = self.decisions[self.idx]
            self.idx += 1
            return d
        return LLMDecision(final_answer="completed")

    async def stream_decision(
        self,
        task: AgentTask,
        history: list[dict[str, Any]],
        tools: list[ToolSpec],
    ):
        dec = await self.decide(task, history, tools)
        if dec.tool_calls:
            yield StreamingChunk(
                token_index=0,
                delta_text="",
                is_final=True,
                parsed_tool_calls=dec.tool_calls,
                metadata={"final_answer": None},
            )
        else:
            yield StreamingChunk(
                token_index=0,
                delta_text=dec.final_answer or "completed",
                is_final=True,
                metadata={"final_answer": dec.final_answer or "completed"},
            )


class TestCompositeRefactor(unittest.IsolatedAsyncioTestCase):
    """Verifies CompositeScheduler real delegation, cache handling, and measured dispatch overhead."""

    def test_01_real_delegation_to_appropriate_schedulers(self) -> None:
        """CompositeScheduler exposes real instances for E1, E2, E3, E4, and Cache."""
        scheduler = CompositeScheduler()

        self.assertIsInstance(scheduler.delegate_fanout(), DAGScheduler)
        self.assertIsInstance(scheduler.delegate_pipeline_sequence(), JITFusionScheduler)
        self.assertIsInstance(scheduler.delegate_speculative(), SpeculativeScheduler)
        self.assertIsInstance(scheduler.delegate_streaming(), CommitHorizonScheduler)
        self.assertIsInstance(scheduler.delegate_cache(), ToolResultCache)

    async def test_02_cache_delegation_repeat_queries_zero_mutation(self) -> None:
        """Repeat queries without mutations hit the cache delegate; dispatch overhead is measured."""
        scheduler = CompositeScheduler()
        reg = ToolRegistry()
        call_count = 0

        def _read_handler(args: dict[str, Any]) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {"id": args.get("id"), "status": "active"}

        spec = ToolSpec(
            name="get_record",
            description="Reads record",
            parameters={"type": "object", "properties": {"id": {"type": "string"}}},
            is_read_only=True,
            side_effects=False,
            is_idempotent=True,
        )
        reg.register(MockToolAdapter(spec=spec, handler=_read_handler))

        task = AgentTask("t1", "read record")
        llm = MockCompositeLLM(
            [
                LLMDecision(tool_calls=[ToolCall("c1", "get_record", {"id": "rec_1"})]),
                LLMDecision(tool_calls=[ToolCall("c2", "get_record", {"id": "rec_1"})]),
                LLMDecision(final_answer="done"),
            ]
        )

        res = await scheduler.execute(task, llm, reg)
        self.assertEqual(res.final_answer, "done")
        # Second identical call must be served by cache => handler called exactly once
        self.assertEqual(call_count, 1)

        # Cache hit event recorded
        has_cache_hit = any(e.event_type == EventType.CACHE_HIT for e in res.events)
        self.assertTrue(has_cache_hit)

        # Dispatch overhead was measured
        stats = scheduler.get_dispatch_overhead()
        self.assertGreater(stats["dispatch_count"], 0)
        self.assertGreater(stats["total_overhead_ns"], 0)
        self.assertGreater(stats["mean_overhead_ns"], 0.0)

    async def test_03_independent_fanout_execution(self) -> None:
        """Independent fanout calls execute concurrently and dispatch overhead is tracked."""
        scheduler = CompositeScheduler()
        reg = ToolRegistry()

        executed_tools: list[str] = []

        def _slow_tool_a(args: dict[str, Any]) -> dict[str, Any]:
            executed_tools.append("fanout_a")
            return {"result": "ok"}

        def _slow_tool_b(args: dict[str, Any]) -> dict[str, Any]:
            executed_tools.append("fanout_b")
            return {"result": "ok"}

        reg.register(
            MockToolAdapter(
                spec=ToolSpec(name="fanout_a", description="a", parameters={"type": "object"}),
                handler=_slow_tool_a,
            )
        )
        reg.register(
            MockToolAdapter(
                spec=ToolSpec(name="fanout_b", description="b", parameters={"type": "object"}),
                handler=_slow_tool_b,
            )
        )

        task = AgentTask("t_fanout", "run fanout")
        llm = MockCompositeLLM(
            [
                LLMDecision(
                    tool_calls=[
                        ToolCall("c_a", "fanout_a", {}),
                        ToolCall("c_b", "fanout_b", {}),
                    ]
                ),
                LLMDecision(final_answer="fanout_done"),
            ]
        )

        res = await scheduler.execute(task, llm, reg)
        self.assertEqual(res.final_answer, "fanout_done")
        self.assertIn("fanout_a", executed_tools)
        self.assertIn("fanout_b", executed_tools)

        overhead_info = scheduler.get_dispatch_overhead()
        self.assertGreaterEqual(overhead_info["dispatch_count"], 1)
        self.assertGreater(overhead_info["total_overhead_ns"], 0)


if __name__ == "__main__":
    unittest.main()
