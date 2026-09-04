"""Unit tests for causal event-driven execution engine in ReplayBackend."""

from __future__ import annotations

import asyncio
import unittest
from typing import Any

from toolspeed.adapters.base import AgentTask, LLMDecision, ToolCall, ToolSpec
from toolspeed.benchmarks.replay_backend import ReplayLLMAdapter, ReplayToolAdapter
from toolspeed.core.clock import VirtualClock


class TestCausalReplayEngine(unittest.IsolatedAsyncioTestCase):
    """Verifies causal event-driven execution, state persistence, and virtual-time semantics."""

    async def test_01_causal_ordering_and_parallel_timing(self) -> None:
        """Parallel tool executions on VirtualClock advance time to max duration, not sum."""
        clock = VirtualClock()
        spec_a = ToolSpec(name="fetch_a", description="Fetch A")
        spec_b = ToolSpec(name="fetch_b", description="Fetch B")

        adapter_a = ReplayToolAdapter(name="fetch_a", spec=spec_a, default_latency_ms=25.0, clock=clock)
        adapter_b = ReplayToolAdapter(name="fetch_b", spec=spec_b, default_latency_ms=40.0, clock=clock)

        start_ns = clock.now_ns()

        call_a = ToolCall(call_id="c1", name="fetch_a", arguments={"id": 1})
        call_b = ToolCall(call_id="c2", name="fetch_b", arguments={"id": 2})

        res_a, res_b = await asyncio.gather(adapter_a.execute(call_a), adapter_b.execute(call_b))

        total_elapsed_ms = (clock.now_ns() - start_ns) / 1_000_000.0

        self.assertEqual(res_a.execution_time_ms, 25.0)
        self.assertEqual(res_b.execution_time_ms, 40.0)
        # Because they ran in parallel, virtual elapsed time is max(25.0, 40.0) = 40.0ms
        self.assertAlmostEqual(total_elapsed_ms, 40.0, places=2)

    async def test_02_state_persistence_and_causality(self) -> None:
        """Mutative tool call updates state, observable by subsequent causal read tool calls."""
        clock = VirtualClock()
        shared_state: dict[str, int] = {"balance": 1000}

        def transfer_handler(args: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
            amt = int(args.get("amount", 0))
            cur = int(state.get("balance", 0))
            if cur < amt:
                return {"status": "error", "error": "insufficient_funds"}
            state["balance"] = cur - amt
            return {"status": "success", "new_balance": state["balance"]}

        def balance_handler(args: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
            return {"balance": state.get("balance", 0)}

        spec_tx = ToolSpec(name="transfer", description="Transfer funds", side_effects=True)
        spec_bal = ToolSpec(name="get_balance", description="Get balance", side_effects=False)

        adapter_tx = ReplayToolAdapter(
            name="transfer",
            spec=spec_tx,
            default_latency_ms=30.0,
            clock=clock,
            state=shared_state,
            handler=transfer_handler,
        )
        adapter_bal = ReplayToolAdapter(
            name="get_balance",
            spec=spec_bal,
            default_latency_ms=15.0,
            clock=clock,
            state=shared_state,
            handler=balance_handler,
        )

        # Initial balance check
        res1 = await adapter_bal.execute(ToolCall(call_id="c1", name="get_balance", arguments={}))
        self.assertEqual(res1.output["balance"], 1000)

        # Causal mutation
        res2 = await adapter_tx.execute(
            ToolCall(call_id="c2", name="transfer", arguments={"amount": 400, "idempotency_key": "tx_01"})
        )
        self.assertEqual(res2.output["status"], "success")
        self.assertEqual(res2.output["new_balance"], 600)

        # Subsequent balance check must observe mutation
        res3 = await adapter_bal.execute(ToolCall(call_id="c3", name="get_balance", arguments={}))
        self.assertEqual(res3.output["balance"], 600)

        # Total sequential time: 15ms + 30ms + 15ms = 60ms
        self.assertAlmostEqual(clock.now_ns() / 1_000_000.0, 60.0, places=2)

    async def test_03_replay_idempotency_caching(self) -> None:
        """Duplicate tool calls with the same idempotency key return cached response with fast path."""
        clock = VirtualClock()
        shared_state: dict[str, int] = {"count": 0}

        def increment_handler(args: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
            state["count"] = int(state.get("count", 0)) + 1
            return {"count": state["count"]}

        spec = ToolSpec(name="increment", description="Increment counter", side_effects=True)
        adapter = ReplayToolAdapter(
            name="increment",
            spec=spec,
            default_latency_ms=50.0,
            clock=clock,
            state=shared_state,
            handler=increment_handler,
        )

        call_first = ToolCall(call_id="c1", name="increment", arguments={"idempotency_key": "k_1"})
        res_first = await adapter.execute(call_first)
        self.assertEqual(res_first.output["count"], 1)
        self.assertEqual(res_first.execution_time_ms, 50.0)

        # Duplicate call
        call_dup = ToolCall(call_id="c2", name="increment", arguments={"idempotency_key": "k_1"})
        res_dup = await adapter.execute(call_dup)
        # Count must remain 1 (no duplicate mutation)
        self.assertEqual(res_dup.output["count"], 1)
        # Idempotency hit is fast path (5ms)
        self.assertEqual(res_dup.execution_time_ms, 5.0)
        self.assertEqual(shared_state["count"], 1)

    async def test_04_causal_policy_driven_llm_adapter(self) -> None:
        """ReplayLLMAdapter can causally adapt subsequent decisions based on tool execution history."""
        clock = VirtualClock()

        def adaptive_policy(task: AgentTask, history: list[dict[str, object]]) -> LLMDecision:
            # If no history yet, request tool call
            if not history:
                return LLMDecision(
                    tool_calls=[ToolCall(call_id="c_step1", name="query_status", arguments={"task": task.task_id})]
                )
            # If history has tool output, produce final answer
            last_msg = history[-1]
            content = str(last_msg.get("content", ""))
            return LLMDecision(
                tool_calls=[],
                final_answer={"result": "processed", "evidence": content},
            )

        llm = ReplayLLMAdapter(decision_delay_ms=20.0, clock=clock, policy=adaptive_policy)
        task = AgentTask(task_id="t_dyn", prompt="Check status")

        d1 = await llm.decide(task, history=[], tools=[])
        self.assertEqual(len(d1.tool_calls), 1)
        self.assertEqual(d1.tool_calls[0].name, "query_status")

        mock_history: list[dict[str, object]] = [{"role": "tool", "name": "query_status", "content": "status_ok_ready"}]
        d2 = await llm.decide(task, history=mock_history, tools=[])
        self.assertEqual(len(d2.tool_calls), 0)
        self.assertEqual(d2.final_answer, {"result": "processed", "evidence": "status_ok_ready"})


if __name__ == "__main__":
    unittest.main()
