"""Deterministic Replay Backend for ToolSpeed Benchmarking."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
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
        latency_ms: float = 30.0,
        output: Any = None,
        is_read_only: bool = True,
        side_effects: bool = False,
        requires_approval: bool = False,
        is_idempotent: bool = True,
        cold_start_ms: float = 0.0,
    ):
        self._name = name
        self.latency_ms = latency_ms
        self.cold_start_ms = cold_start_ms
        self.fixed_output = output
        self._is_read_only = is_read_only
        self._side_effects = side_effects
        self._requires_approval = requires_approval
        self._is_idempotent = is_idempotent
        self.call_count = 0
        self.is_warmed = False

    def prewarm(self) -> None:
        self.is_warmed = True

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
        
        delay_ms = self.latency_ms
        if not self.is_warmed and self.cold_start_ms > 0:
            delay_ms += self.cold_start_ms
            self.is_warmed = True

        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)

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
        decision_delay_ms: float = 30.0,
        stream_chunks_count: int = 4,
    ):
        self.decisions = list(decisions or [])
        self.draft_prediction = draft_prediction
        self.decision_delay_ms = decision_delay_ms
        self.stream_chunks_count = stream_chunks_count
        self._turn_index = 0

    def _get_decision_sync(self, task: Task) -> LLMDecision:
        if self._turn_index < len(self.decisions):
            decision = self.decisions[self._turn_index]
            self._turn_index += 1
            return decision
        return LLMDecision(
            reasoning="Task complete.",
            tool_calls=[],
            final_answer=task.expected_output or {"status": "done"},
            input_tokens=100,
            output_tokens=20,
        )

    async def decide(
        self,
        task: Task,
        history: List[Dict[str, Any]],
        available_tools: List[ToolSpec],
    ) -> LLMDecision:
        if self.decision_delay_ms > 0:
            await asyncio.sleep(self.decision_delay_ms / 1000.0)
        return self._get_decision_sync(task)

    async def predict_draft(
        self,
        task: Task,
        history: List[Dict[str, Any]],
        available_tools: List[ToolSpec],
    ) -> Optional[ToolCall]:
        await asyncio.sleep(max(0.001, self.decision_delay_ms * 0.15 / 1000.0))
        return self.draft_prediction

    async def stream_decision(
        self,
        task: Task,
        history: List[Dict[str, Any]],
        available_tools: List[ToolSpec],
    ) -> AsyncIterator[StreamingChunk]:
        decision = self._get_decision_sync(task)
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
        self._prepared = False

    async def prepare_run(self, plan: Any) -> None:
        self._prepared = True

    async def finalize_run(self) -> Dict[str, Any]:
        return {"backend": "ReplayBackend", "evidence_level": self.evidence_level.value}

    async def close(self) -> None:
        pass

    def create_workload_environment(
        self,
        workload_id: str,
        tool_delay_ms: float = 25.0,
        model_delay_ms: float = 30.0,
        trial_index: int = 0,
    ) -> Tuple[ToolRegistry, ReplayLLMAdapter]:
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
            registry.register(ReplayToolAdapter("fetch_user", latency_ms=tool_delay_ms, output={"user_id": "u42", "name": "Alice", "org_id": "org9"}))
            registry.register(ReplayToolAdapter("fetch_orders", latency_ms=tool_delay_ms, output={"user_id": "u42", "orders": [101, 102]}))
            c1 = ToolCall(call_id="c1", name="fetch_user", arguments={"user_id": "u42"})
            c2 = ToolCall(call_id="c2", name="fetch_orders", arguments={"user_id": "$c1.user_id"})
            decisions = [
                LLMDecision(reasoning="Chain", tool_calls=[c1, c2]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"user": {"user_id": "u42", "name": "Alice", "org_id": "org9"}, "orders": {"user_id": "u42", "orders": [101, 102]}, "fused": True}),
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
            # Repeated keys to test caching hit rate
            user_key = f"usr_{trial_index % 3:03d}"
            registry.register(ReplayToolAdapter("lookup_user_profile", latency_ms=tool_delay_ms, output={"user_id": user_key, "tier": "enterprise", "discount_pct": 20}))
            registry.register(ReplayToolAdapter("calculate_final_invoice", latency_ms=10.0, output={"final_price": 80.0, "currency": "USD"}))
            decisions = [
                LLMDecision(reasoning="Lookup", tool_calls=[ToolCall(name="lookup_user_profile", arguments={"user_id": user_key})]),
                LLMDecision(reasoning="Invoice", tool_calls=[ToolCall(name="calculate_final_invoice", arguments={"user_id": user_key, "tier": "enterprise", "base_amount": 100.0, "discount_pct": 20})]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"user_id": user_key, "tier": "enterprise", "final_price": 80.0}),
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
            # W6: Cold-start tool / sandbox (cold_start_ms=80ms vs warm=20ms)
            registry.register(ReplayToolAdapter("sandbox_python_eval", latency_ms=15.0, cold_start_ms=80.0, output={"result": 42}))
            c = ToolCall(name="sandbox_python_eval", arguments={"expression": "6 * 7"})
            decisions = [
                LLMDecision(reasoning="Run", tool_calls=[c]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"result": 42}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=model_delay_ms)

        elif workload_id == "W7":
            # W7: Side-effecting actions requiring approval
            idem_key = f"idem_w7_{trial_index:04d}"
            registry.register(ReplayToolAdapter("execute_fund_transfer", latency_ms=tool_delay_ms, output={"status": "TRANSFERRED", "tx": "tx99"}, is_read_only=False, side_effects=True, requires_approval=True))
            c = ToolCall(name="execute_fund_transfer", arguments={"from_account": "acc_001", "to_account": "acc_002", "amount": 100.0, "idempotency_key": idem_key}, is_approved=True)
            decisions = [
                LLMDecision(reasoning="Transfer", tool_calls=[c]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"status": "TRANSFERRED"}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=model_delay_ms)

        elif workload_id in ("E5", "E5a"):
            # E5a: Action Bytecode Transport Codec
            registry.register(ReplayToolAdapter("process_packet", latency_ms=tool_delay_ms, output={"parsed": True}))
            c = ToolCall(name="process_packet", arguments={"header": "v2", "payload": "data_packet"})
            decisions = [
                LLMDecision(reasoning="Encode", tool_calls=[c]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"parsed": True}),
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
