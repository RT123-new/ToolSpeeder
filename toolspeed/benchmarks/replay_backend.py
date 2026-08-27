"""Deterministic Replay Backend for ToolSpeed Benchmarking."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional
import time

from toolspeed.adapters.base import (
    BaseLLMAdapter,
    BaseToolAdapter,
    LLMDecision,
    StreamingChunk,
    ToolRegistry,
    ToolSchema,
)
from toolspeed.core.types import (
    EvidenceLevel,
    Task,
    ToolCall,
    ToolResult,
    ToolSpec,
)


class ReplayToolAdapter(BaseToolAdapter):
    """Deterministic tool adapter executing pre-configured tool behaviors."""

    def __init__(
        self,
        name: str,
        latency_ms: float = 50.0,
        output: Any = None,
        is_read_only: bool = True,
        side_effects: bool = False,
        requires_approval: bool = False,
    ):
        self._name = name
        self.latency_ms = latency_ms
        self.fixed_output = output
        self._is_read_only = is_read_only
        self._side_effects = side_effects
        self._requires_approval = requires_approval
        self.call_count = 0

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self._name,
            description=f"Replay tool for {self._name}",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            is_read_only=self._is_read_only,
            is_side_effect=self._side_effects,
            requires_approval=self._requires_approval,
            cost_usd=0.0001,
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        start_ns = time.perf_counter_ns()
        self.call_count += 1
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)

        out = self.fixed_output if self.fixed_output is not None else {"status": "ok", "call": call.arguments}
        return ToolResult(
            call_id=call.call_id,
            name=self._name,
            tool_name=self._name,
            result=out,
            output=out,
            error=None,
            is_error=False,
            execution_time_ns=time.perf_counter_ns() - start_ns,
            cost_usd=0.0001,
        )


class ReplayLLMAdapter(BaseLLMAdapter):
    """Deterministic LLM adapter replaying decisions and streaming chunks."""

    def __init__(
        self,
        decisions: Optional[List[LLMDecision]] = None,
        draft_prediction: Optional[ToolCall] = None,
        decision_delay_ms: float = 40.0,
        stream_chunks_count: int = 5,
    ):
        self.decisions = list(decisions or [])
        self.draft_prediction = draft_prediction
        self.decision_delay_ms = decision_delay_ms
        self.stream_chunks_count = stream_chunks_count
        self._turn_index = 0

    async def decide(
        self,
        task: Task,
        history: List[Dict[str, Any]],
        available_tools: List[ToolSpec],
    ) -> LLMDecision:
        if self.decision_delay_ms > 0:
            await asyncio.sleep(self.decision_delay_ms / 1000.0)

        if self._turn_index < len(self.decisions):
            decision = self.decisions[self._turn_index]
            self._turn_index += 1
            return decision

        # Default final decision
        return LLMDecision(
            reasoning="Task complete.",
            tool_calls=[],
            final_answer=task.expected_output or {"status": "done"},
            input_tokens=100,
            output_tokens=20,
        )

    async def predict_draft(
        self,
        task: Task,
        history: List[Dict[str, Any]],
        available_tools: List[ToolSpec],
    ) -> Optional[ToolCall]:
        # Fast draft latency
        await asyncio.sleep(max(0.001, self.decision_delay_ms * 0.2 / 1000.0))
        return self.draft_prediction

    async def stream_decision(
        self,
        task: Task,
        history: List[Dict[str, Any]],
        available_tools: List[ToolSpec],
    ) -> AsyncIterator[StreamingChunk]:
        decision = await self.decide(task, history, available_tools)
        chunk_delay = (self.decision_delay_ms / 1000.0) / max(1, self.stream_chunks_count)

        for i in range(self.stream_chunks_count):
            if chunk_delay > 0:
                await asyncio.sleep(chunk_delay)
            is_final = (i == self.stream_chunks_count - 1)
            ready_calls = list(decision.tool_calls) if i >= 1 else []
            yield StreamingChunk(
                token_index=i,
                delta_text=f"token_{i} ",
                commit_horizon_ready=ready_calls,
                is_final=is_final,
                parsed_tool_calls=decision.tool_calls if is_final else [],
                metadata={"final_answer": decision.final_answer} if is_final else {},
            )


class ReplayBackend:
    """Deterministic trace-replay execution backend for benchmark evaluation."""

    def __init__(self, evidence_level: EvidenceLevel = EvidenceLevel.REPLAY_INTEGRATION):
        self.evidence_level = evidence_level

    def create_workload_environment(
        self,
        workload_id: str,
        tool_delay_ms: float = 30.0,
        model_delay_ms: float = 40.0,
    ) -> tuple[ToolRegistry, ReplayLLMAdapter]:
        registry = ToolRegistry()

        if workload_id == "W1":
            # W1: Fan-out reads (5 independent tools)
            for i in range(5):
                t_name = f"read_shard_{i}"
                registry.register(ReplayToolAdapter(t_name, latency_ms=tool_delay_ms, output={"data": f"shard_{i}_value"}))
            calls = [ToolCall(name=f"read_shard_{i}", arguments={"query": f"key_{i}"}) for i in range(5)]
            decisions = [
                LLMDecision(reasoning="Fanout", tool_calls=calls),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"shards": 5}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=model_delay_ms)

        elif workload_id == "W2":
            # W2: Dependent chains
            registry.register(ReplayToolAdapter("fetch_user", latency_ms=tool_delay_ms, output={"user_id": "u42", "org_id": "org9"}))
            registry.register(ReplayToolAdapter("fetch_orders", latency_ms=tool_delay_ms, output={"user_id": "u42", "orders": [101, 102]}))
            c1 = ToolCall(call_id="c1", name="fetch_user", arguments={"user_id": "u42"})
            c2 = ToolCall(call_id="c2", name="fetch_orders", arguments={"user_id": "$c1.user_id"})
            decisions = [
                LLMDecision(reasoning="Chain", tool_calls=[c1, c2]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"orders": [101, 102]}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=model_delay_ms)

        elif workload_id == "W3":
            # W3: Branching with speculative read
            registry.register(ReplayToolAdapter("search_catalog", latency_ms=tool_delay_ms, output={"item": "prod_1", "price": 99}))
            predicted = ToolCall(name="search_catalog", arguments={"query": "laptop"}, speculation_confidence=0.9)
            decisions = [
                LLMDecision(reasoning="Search", tool_calls=[ToolCall(name="search_catalog", arguments={"query": "laptop"})]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"item": "prod_1"}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, draft_prediction=predicted, decision_delay_ms=model_delay_ms)

        elif workload_id == "W4":
            # W4: Repeated workflows with plan locality (caching)
            registry.register(ReplayToolAdapter("cached_query", latency_ms=tool_delay_ms, output={"result": "cached_payload"}))
            c = ToolCall(name="cached_query", arguments={"key": "lookup_1"})
            decisions = [
                LLMDecision(reasoning="Lookup", tool_calls=[c]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"result": "cached_payload"}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=model_delay_ms)

        elif workload_id == "W5":
            # W5: Large payloads & early commit
            registry.register(ReplayToolAdapter("process_payload", latency_ms=tool_delay_ms, output={"processed": True}))
            c = ToolCall(name="process_payload", arguments={"payload": "x" * 500})
            decisions = [
                LLMDecision(reasoning="Process", tool_calls=[c]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"processed": True}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=model_delay_ms)

        elif workload_id == "W6":
            # W6: Cold-start tool / sandbox
            registry.register(ReplayToolAdapter("sandbox_run", latency_ms=tool_delay_ms * 2.5, output={"exit": 0}))
            c = ToolCall(name="sandbox_run", arguments={"code": "print(1)"})
            decisions = [
                LLMDecision(reasoning="Run", tool_calls=[c]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"exit": 0}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=model_delay_ms)

        elif workload_id == "W7":
            # W7: Side-effecting actions requiring approval
            registry.register(ReplayToolAdapter("execute_payment", latency_ms=tool_delay_ms, output={"tx": "tx99"}, is_read_only=False, side_effects=True, requires_approval=True))
            c = ToolCall(name="execute_payment", arguments={"amount": 50}, is_approved=True)
            decisions = [
                LLMDecision(reasoning="Pay", tool_calls=[c]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"tx": "tx99"}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=model_delay_ms)

        else:
            registry.register(ReplayToolAdapter("generic_tool", latency_ms=tool_delay_ms, output={"ok": True}))
            decisions = [
                LLMDecision(reasoning="Action", tool_calls=[ToolCall(name="generic_tool", arguments={})]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"ok": True}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=model_delay_ms)

        return registry, model
