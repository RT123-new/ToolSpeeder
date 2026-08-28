"""Deterministic Virtual-Time Replay Backend for ToolSpeed Benchmarking."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import json
import time
from typing import Any

from toolspeed.adapters.base import (
    BaseLLMAdapter,
    BaseToolAdapter,
    LLMDecision,
    StreamingChunk,
    ToolRegistry,
    ToolSchema,
)
from toolspeed.core.types import (
    ApprovalGrant,
    EvidenceLevel,
    Task,
    ToolCall,
    ToolResult,
    ToolSpec,
)


class ReplayToolAdapter(BaseToolAdapter):
    """Deterministic virtual-time tool adapter executing pre-configured tool behaviors."""

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
        parameters: dict[str, Any] | None = None,
        handler: Any = None,
    ):
        self._name = name
        self.latency_ms = latency_ms
        self.cold_start_ms = cold_start_ms
        self.fixed_output = output
        self._is_read_only = is_read_only
        self._side_effects = side_effects
        self._requires_approval = requires_approval
        self._is_idempotent = is_idempotent
        self._parameters = parameters or {"type": "object", "properties": {"query": {"type": "string"}}}
        self._handler = handler
        self.call_count = 0
        self.is_warmed = False

    def prewarm(self) -> None:
        self.is_warmed = True

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self._name,
            description=f"Replay tool for {self._name}",
            parameters=self._parameters,
            is_read_only=self._is_read_only,
            is_side_effect=self._side_effects,
            requires_approval=self._requires_approval,
            is_idempotent=self._is_idempotent,
            cost_usd=0.0001,
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        # Yield to event loop to allow concurrency interleaving
        await asyncio.sleep(0)

        self.call_count += 1
        delay_ms = self.latency_ms
        if not self.is_warmed and self.cold_start_ms > 0:
            delay_ms += self.cold_start_ms
            self.is_warmed = True

        start_wall = time.perf_counter()
        dur_ns = int(delay_ms * 1_000_000)

        if self._handler is not None and callable(self._handler):
            out = self._handler(call.arguments)
        elif self.fixed_output is not None:
            out = self.fixed_output
        else:
            out = {"status": "ok", "call": call.arguments}

        return ToolResult(
            call_id=call.call_id,
            name=self._name,
            tool_name=self._name,
            result=out,
            output=out,
            error=None,
            is_error=False,
            execution_time_ns=dur_ns,
            execution_time_ms=delay_ms,
            started_at=start_wall,
            finished_at=start_wall + (delay_ms / 1000.0),
            cost_usd=0.0001,
        )


class ReplayLLMAdapter(BaseLLMAdapter):
    """Deterministic virtual-time LLM adapter replaying decisions and streaming chunks."""

    def __init__(
        self,
        decisions: list[LLMDecision] | None = None,
        draft_prediction: ToolCall | None = None,
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
        history: list[dict[str, Any]],
        available_tools: list[ToolSpec],
    ) -> LLMDecision:
        await asyncio.sleep(0)
        decision = self._get_decision_sync(task)
        decision.duration_ms = self.decision_delay_ms
        return decision

    async def predict_draft(
        self,
        task: Task,
        history: list[dict[str, Any]],
        available_tools: list[ToolSpec],
    ) -> ToolCall | None:
        await asyncio.sleep(0)
        return self.draft_prediction

    async def stream_decision(
        self,
        task: Task,
        history: list[dict[str, Any]],
        available_tools: list[ToolSpec],
    ) -> AsyncIterator[StreamingChunk]:
        decision = self._get_decision_sync(task)
        total_chunks = max(2, self.stream_chunks_count)

        for i in range(total_chunks):
            await asyncio.sleep(0)
            is_final = (i == total_chunks - 1)
            ready_calls = list(decision.tool_calls) if (i >= 1 and decision.tool_calls) else []
            fragment = json.dumps(ready_calls[0].arguments) if ready_calls else (json.dumps(decision.tool_calls[0].arguments) if (is_final and decision.tool_calls) else "")

            yield StreamingChunk(
                token_index=i,
                delta_text=f"token_{i} ",
                commit_horizon_ready=ready_calls,
                raw_json_fragment=fragment,
                is_final=is_final,
                parsed_tool_calls=decision.tool_calls if is_final else [],
                metadata={"final_answer": decision.final_answer} if is_final else {},
            )


class ReplayBackend:
    """Deterministic trace-replay virtual-time execution backend for benchmark evaluation."""

    def __init__(self, evidence_level: EvidenceLevel = EvidenceLevel.REPLAY_INTEGRATION):
        self.evidence_level = evidence_level
        self._prepared = False

    async def prepare_run(self, plan: Any) -> None:
        self._prepared = True

    async def finalize_run(self) -> dict[str, Any]:
        return {"backend": "ReplayBackend", "evidence_level": self.evidence_level.value}

    async def close(self) -> None:
        pass

    def create_workload_environment(
        self,
        workload_id: str,
        tool_delay_ms: float = 25.0,
        model_delay_ms: float = 30.0,
        trial_index: int = 0,
    ) -> tuple[ToolRegistry, ReplayLLMAdapter]:
        registry = ToolRegistry()

        if workload_id == "W1":
            # W1: Fan-out reads (5 independent tools)
            params = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
            for i in range(5):
                t_name = f"read_shard_{i}"
                registry.register(ReplayToolAdapter(t_name, latency_ms=tool_delay_ms, output={"data": f"shard_{i}_value"}, parameters=params))
            calls = [ToolCall(name=f"read_shard_{i}", arguments={"query": f"key_{i}"}) for i in range(5)]
            decisions = [
                LLMDecision(reasoning="Fanout", tool_calls=calls),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"shards": 5}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=model_delay_ms)

        elif workload_id == "W2":
            # W2: Dependent chains
            p_user = {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}
            p_orders = {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}
            registry.register(ReplayToolAdapter("fetch_user", latency_ms=tool_delay_ms, output={"user_id": "u42", "name": "Alice", "org_id": "org9"}, parameters=p_user))
            registry.register(ReplayToolAdapter("fetch_orders", latency_ms=tool_delay_ms, output={"user_id": "u42", "orders": [101, 102]}, parameters=p_orders))
            c1 = ToolCall(call_id="c1", name="fetch_user", arguments={"user_id": "u42"})
            c2 = ToolCall(call_id="c2", name="fetch_orders", arguments={"user_id": "u42"})
            decisions = [
                LLMDecision(reasoning="Chain step 1", tool_calls=[c1]),
                LLMDecision(reasoning="Chain step 2", tool_calls=[c2]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"user": {"user_id": "u42", "name": "Alice", "org_id": "org9"}, "orders": {"user_id": "u42", "orders": [101, 102]}, "fused": True}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=model_delay_ms)

        elif workload_id == "W3":
            # W3: Branching with speculative read
            p_search = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
            registry.register(ReplayToolAdapter("search_catalog", latency_ms=tool_delay_ms, output={"item": "prod_1", "price": 99}, parameters=p_search))
            predicted = ToolCall(name="search_catalog", arguments={"query": "laptop"}, speculation_confidence=0.9)
            decisions = [
                LLMDecision(reasoning="Search", tool_calls=[ToolCall(name="search_catalog", arguments={"query": "laptop"})]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"item": "prod_1"}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, draft_prediction=predicted, decision_delay_ms=model_delay_ms)

        elif workload_id == "W4":
            # W4: Repeated workflows with plan locality (caching)
            user_key = f"usr_{trial_index % 3:03d}"
            p_usr = {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}
            p_calc = {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "tier": {"type": "string"},
                    "base_amount": {"type": "number"},
                    "discount_pct": {"type": "number"},
                },
                "required": ["user_id", "tier", "base_amount"],
            }
            registry.register(ReplayToolAdapter("lookup_user_profile", latency_ms=tool_delay_ms, output={"user_id": user_key, "tier": "enterprise", "discount_pct": 20}, parameters=p_usr))
            registry.register(ReplayToolAdapter("calculate_final_invoice", latency_ms=10.0, output={"final_price": 80.0, "currency": "USD"}, parameters=p_calc))
            decisions = [
                LLMDecision(reasoning="Lookup", tool_calls=[ToolCall(name="lookup_user_profile", arguments={"user_id": user_key})]),
                LLMDecision(reasoning="Invoice", tool_calls=[ToolCall(name="calculate_final_invoice", arguments={"user_id": user_key, "tier": "enterprise", "base_amount": 100.0, "discount_pct": 20})]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"user_id": user_key, "tier": "enterprise", "final_price": 80.0}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=model_delay_ms)

        elif workload_id == "W5":
            # W5: Large payloads & early commit
            p_proc = {"type": "object", "properties": {"payload": {"type": "string"}}, "required": ["payload"]}
            registry.register(ReplayToolAdapter("process_payload", latency_ms=tool_delay_ms, output={"processed": True}, parameters=p_proc))
            c = ToolCall(name="process_payload", arguments={"payload": "x" * 500})
            decisions = [
                LLMDecision(reasoning="Process", tool_calls=[c]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"processed": True}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=model_delay_ms)

        elif workload_id == "W6":
            # W6: Cold-start tool / sandbox (cold_start_ms=80ms vs warm=15ms)
            p_eval = {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
            registry.register(ReplayToolAdapter("sandbox_python_eval", latency_ms=15.0, cold_start_ms=80.0, output={"result": 42}, parameters=p_eval))
            c = ToolCall(name="sandbox_python_eval", arguments={"expression": "6 * 7"})
            decisions = [
                LLMDecision(reasoning="Run", tool_calls=[c]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"result": 42}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=model_delay_ms)

        elif workload_id == "W7":
            # W7: Side-effecting actions requiring approval
            idem_key = f"idem_w7_{trial_index:04d}"
            p_trans = {
                "type": "object",
                "properties": {
                    "from_account": {"type": "string"},
                    "to_account": {"type": "string"},
                    "amount": {"type": "number"},
                    "idempotency_key": {"type": "string"},
                },
                "required": ["from_account", "to_account", "amount", "idempotency_key"],
            }
            transfer_args = {
                "from_account": "acc_001",
                "to_account": "acc_002",
                "amount": 100.0,
                "idempotency_key": idem_key,
            }
            grant = ApprovalGrant.create(
                tool_name="execute_fund_transfer",
                arguments=transfer_args,
                authority="trusted_system",
            )
            registry.register(ReplayToolAdapter("execute_fund_transfer", latency_ms=tool_delay_ms, output={"status": "TRANSFERRED", "tx": "tx99"}, is_read_only=False, side_effects=True, requires_approval=True, parameters=p_trans))
            c = ToolCall(name="execute_fund_transfer", arguments=transfer_args, requires_approval=True, is_approved=True, approval_grant=grant)
            decisions = [
                LLMDecision(reasoning="Transfer", tool_calls=[c]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"status": "TRANSFERRED"}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=model_delay_ms)

        elif workload_id in ("E5", "E5a"):
            # E5a: Action Bytecode Transport Codec
            p_pack = {"type": "object", "properties": {"header": {"type": "string"}, "payload": {"type": "string"}}, "required": ["header", "payload"]}
            registry.register(ReplayToolAdapter("process_packet", latency_ms=tool_delay_ms, output={"parsed": True}, parameters=p_pack))
            c = ToolCall(name="process_packet", arguments={"header": "v2", "payload": "data_packet"})
            decisions = [
                LLMDecision(reasoning="Encode", tool_calls=[c]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"parsed": True}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=model_delay_ms)

        else:
            p_gen = {"type": "object", "properties": {"query": {"type": "string"}}}
            registry.register(ReplayToolAdapter("generic_tool", latency_ms=tool_delay_ms, output={"ok": True}, parameters=p_gen))
            decisions = [
                LLMDecision(reasoning="Action", tool_calls=[ToolCall(name="generic_tool", arguments={"query": "test"})]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"ok": True}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=model_delay_ms)

        return registry, model
