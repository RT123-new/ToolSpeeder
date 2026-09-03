"""Deterministic virtual-clock replay backend with immutable paired fixtures."""

from __future__ import annotations

import copy
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from toolspeed.adapters.base import (
    BaseLLMAdapter,
    BaseToolAdapter,
    LLMDecision,
    StreamingChunk,
    ToolRegistry,
    ToolSchema,
)
from toolspeed.core.clock import Clock, VirtualClock
from toolspeed.core.types import (
    AgentTask,
    ApprovalGrant,
    EvidenceLevel,
    Task,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from toolspeed.workloads.w1_independent import W1IndependentWorkload
from toolspeed.workloads.w2_chains import W2ChainsWorkload
from toolspeed.workloads.w3_branching import W3BranchingWorkload
from toolspeed.workloads.w4_locality import W4LocalityWorkload
from toolspeed.workloads.w5_large_payloads import W5LargePayloadsWorkload
from toolspeed.workloads.w6_cold_start import W6ColdStartWorkload
from toolspeed.workloads.w7_side_effects import W7SideEffectsWorkload


class ReplayToolAdapter(BaseToolAdapter):
    """Deterministic virtual-time tool adapter replaying pre-recorded responses."""

    def __init__(
        self,
        name: str,
        spec: ToolSpec,
        recorded_responses: list[dict[str, Any]] | None = None,
        default_latency_ms: float = 20.0,
        clock: Clock | None = None,
    ):
        super().__init__(spec=spec)
        self._name = name
        self._spec: ToolSpec = spec
        self._recorded_responses = list(recorded_responses or [])
        self._default_latency_ms = default_latency_ms
        self._call_index = 0
        self.clock = clock or VirtualClock()

    @property
    def name(self) -> str:
        return self._name

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self._name,
            description=self._spec.description,
            parameters=self._spec.parameters,
            is_side_effect=self._spec.side_effects,
            cost_usd=0.0001,
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        resp_data: dict[str, Any] = {}
        delay_ms = self._default_latency_ms
        if self._call_index < len(self._recorded_responses):
            recorded = self._recorded_responses[self._call_index]
            self._call_index += 1
            resp_data = recorded.get("output", {})
            delay_ms = float(recorded.get("latency_ms", self._default_latency_ms))
        else:
            resp_data = {"status": "ok", "echo_args": call.arguments}

        start_wall = time.perf_counter()
        dur_ns = int(delay_ms * 1_000_000)

        # Advance virtual or wall time via Clock
        await self.clock.sleep_ms(delay_ms)

        return ToolResult(
            call_id=call.call_id,
            name=self._name,
            tool_name=self._name,
            result=resp_data,
            output=resp_data,
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
        clock: Clock | None = None,
    ):
        self.decisions = list(decisions or [])
        self.draft_prediction = draft_prediction
        self.decision_delay_ms = decision_delay_ms
        self.stream_chunks_count = stream_chunks_count
        self._turn_index = 0
        self.clock = clock or VirtualClock()

    def _get_decision_sync(self, task: AgentTask) -> LLMDecision:
        if self._turn_index < len(self.decisions):
            decision = self.decisions[self._turn_index]
            self._turn_index += 1
            return decision
        return LLMDecision(
            reasoning="Task complete.",
            tool_calls=[],
            final_answer=None,
            input_tokens=100,
            output_tokens=20,
        )

    async def decide(
        self,
        task: AgentTask,
        history: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> LLMDecision:
        await self.clock.sleep_ms(self.decision_delay_ms)
        return self._get_decision_sync(task)

    async def predict_draft(
        self,
        task: AgentTask,
        history: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> ToolCall | None:
        await self.clock.sleep_ms(self.decision_delay_ms * 0.2)
        return self.draft_prediction

    async def stream_decision(
        self,
        task: AgentTask,
        history: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> AsyncIterator[StreamingChunk]:
        decision = self._get_decision_sync(task)
        chunks = max(1, self.stream_chunks_count)
        chunk_delay = self.decision_delay_ms / chunks if chunks > 0 else 0.0

        for i in range(chunks):
            if chunk_delay > 0:
                await self.clock.sleep_ms(chunk_delay)
            is_final = i == chunks - 1
            ready_calls = list(decision.tool_calls) if (i >= 1 and decision.tool_calls) else []
            fragment = (
                json.dumps(ready_calls[0].arguments)
                if ready_calls
                else (json.dumps(decision.tool_calls[0].arguments) if (is_final and decision.tool_calls) else "")
            )

            yield StreamingChunk(
                token_index=i,
                delta_text=f"chunk_{i} ",
                commit_horizon_ready=ready_calls,
                raw_json_fragment=fragment,
                is_final=is_final,
                parsed_tool_calls=decision.tool_calls if is_final else [],
                metadata={"final_answer": decision.final_answer} if is_final else {},
            )


class ReplayBackend:
    """Deterministic virtual-clock replay backend generating immutable paired workload fixtures."""

    def __init__(
        self,
        evidence_level: EvidenceLevel = EvidenceLevel.REPLAY_INTEGRATION,
        clock: Clock | None = None,
        seed: int = 42,
    ):
        self.evidence_level = evidence_level
        self.clock = clock or VirtualClock()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._workloads: dict[str, Any] = {
            "W1": W1IndependentWorkload(),
            "W2": W2ChainsWorkload(),
            "W3": W3BranchingWorkload(),
            "W4": W4LocalityWorkload(),
            "W5": W5LargePayloadsWorkload(),
            "W6": W6ColdStartWorkload(),
            "W7": W7SideEffectsWorkload(),
        }

    def reseed(self, seed: int) -> None:
        """Reseeds the backend RNG and trial parameter generator."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_task(self, workload_id: str, trial_index: int = 0, seed: int | None = None) -> Task:
        """Constructs an immutable Task with seeded parameters and strict validator."""
        eff_seed = seed if seed is not None else self.seed
        if workload_id == "W1":
            return Task(
                task_id=f"w1_replay_s{eff_seed}_t{trial_index:04d}",
                prompt=f"Dispatch fanout metric queries for trial {trial_index} (seed={eff_seed})",
                expected_output={"status": "success", "total_load": 250, "server_count": 5},
                metadata={"workload_id": "W1", "trial_index": trial_index, "seed": eff_seed},
            )
        elif workload_id == "W2":
            return Task(
                task_id=f"w2_replay_s{eff_seed}_t{trial_index:04d}",
                prompt=f"Execute user orders chain for user u_{trial_index} (seed={eff_seed})",
                context={"user_id": f"u_{trial_index}"},
                expected_output={
                    "user": {"user_id": f"u_{trial_index}", "name": "Alice"},
                    "orders": {"orders": [{"id": "ord_101", "total": 99.0}]},
                    "status": "compiled_complete",
                    "fused": True,
                },
                metadata={
                    "workload_id": "W2",
                    "trial_index": trial_index,
                    "workflow_id": "user_orders",
                    "seed": eff_seed,
                },
            )
        elif workload_id == "W3":
            return Task(
                task_id=f"w3_replay_s{eff_seed}_t{trial_index:04d}",
                prompt=f"Execute branching customer check for cust_{trial_index} (seed={eff_seed})",
                expected_output={"status": "approved", "customer_id": f"cust_{trial_index}"},
                metadata={"workload_id": "W3", "trial_index": trial_index, "seed": eff_seed},
            )
        elif workload_id == "W4":
            key_id = f"item_{trial_index % 10}"
            return Task(
                task_id=f"w4_replay_s{eff_seed}_t{trial_index:04d}",
                prompt=f"Lookup price for {key_id} (seed={eff_seed})",
                expected_output={"sku": key_id, "price": 49.99},
                metadata={"workload_id": "W4", "trial_index": trial_index, "seed": eff_seed},
            )
        elif workload_id == "W5":
            return Task(
                task_id=f"w5_replay_s{eff_seed}_t{trial_index:04d}",
                prompt=f"Stream query dataset for trial {trial_index} (seed={eff_seed})",
                expected_output={"status": "success", "count": 100},
                metadata={"workload_id": "W5", "trial_index": trial_index, "seed": eff_seed},
            )
        elif workload_id == "W6":
            return Task(
                task_id=f"w6_replay_s{eff_seed}_t{trial_index:04d}",
                prompt=f"Run sandbox compute task for trial {trial_index} (seed={eff_seed})",
                expected_output={"result": 55},
                metadata={"workload_id": "W6", "trial_index": trial_index, "seed": eff_seed},
            )
        elif workload_id == "W7":
            idemp_key = f"tx_replay_{trial_index:04d}"
            grant = ApprovalGrant.create(
                "execute_fund_transfer", {"recipient": "Alice", "amount": 100.0, "idempotency_key": idemp_key}
            )
            return Task(
                task_id=f"w7_replay_s{eff_seed}_t{trial_index:04d}",
                prompt=f"Execute approved fund transfer tx_{trial_index} (seed={eff_seed})",
                parameters={"recipient": "Alice", "amount": 100.0, "idempotency_key": idemp_key},
                expected_output={"status": "transferred", "idempotency_key": idemp_key},
                metadata={
                    "workload_id": "W7",
                    "trial_index": trial_index,
                    "approval_grant": grant,
                    "seed": eff_seed,
                },
            )
        else:
            return Task(
                task_id=f"e5a_replay_s{eff_seed}_t{trial_index:04d}",
                prompt=f"Serialize action bytecode for trial {trial_index} (seed={eff_seed})",
                expected_output={"status": "done", "trial": trial_index},
                metadata={"workload_id": "E5a", "trial_index": trial_index, "seed": eff_seed},
            )

    def create_workload_environment(
        self, workload_id: str, trial_index: int = 0
    ) -> tuple[ToolRegistry, BaseLLMAdapter]:
        """Creates paired tool registry and replay model adapter for the trial."""
        registry = ToolRegistry()
        clock = self.clock or VirtualClock()
        registry.clock = clock  # type: ignore[attr-defined]

        if workload_id == "W1":
            for i in range(5):
                spec = ToolSpec(name=f"server_metric_{i}", is_read_only=True, is_idempotent=True)
                registry.register(ReplayToolAdapter(f"server_metric_{i}", spec, default_latency_ms=20.0, clock=clock))
            calls = [
                ToolCall(name=f"server_metric_{i}", arguments={"metric": "cpu", "server_id": f"srv_{i}"})
                for i in range(5)
            ]
            decisions = [
                LLMDecision(reasoning="Dispatching 5 fanout metrics", tool_calls=calls),
                LLMDecision(
                    reasoning="Synthesizing metrics",
                    tool_calls=[],
                    final_answer={"status": "success", "total_load": 250, "server_count": 5},
                ),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=25.0, clock=clock)
            return registry, model

        elif workload_id == "W2":
            spec_user = ToolSpec(name="fetch_user", is_read_only=True, is_idempotent=True)
            spec_orders = ToolSpec(name="fetch_orders", is_read_only=True, is_idempotent=True)
            registry.register(
                ReplayToolAdapter(
                    "fetch_user",
                    spec_user,
                    [{"output": {"user_id": f"u_{trial_index}", "name": "Alice"}, "latency_ms": 20.0}],
                    default_latency_ms=20.0,
                    clock=clock,
                )
            )
            registry.register(
                ReplayToolAdapter(
                    "fetch_orders",
                    spec_orders,
                    [{"output": {"orders": [{"id": "ord_101", "total": 99.0}]}, "latency_ms": 20.0}],
                    default_latency_ms=20.0,
                    clock=clock,
                )
            )

            decisions = [
                LLMDecision(
                    reasoning="Fetching user profile",
                    tool_calls=[ToolCall(name="fetch_user", arguments={"user_id": f"u_{trial_index}"})],
                ),
                LLMDecision(
                    reasoning="Fetching orders for user",
                    tool_calls=[ToolCall(name="fetch_orders", arguments={"user_id": f"u_{trial_index}"})],
                ),
                LLMDecision(
                    reasoning="Complete",
                    tool_calls=[],
                    final_answer={
                        "user": {"user_id": f"u_{trial_index}", "name": "Alice"},
                        "orders": {"orders": [{"id": "ord_101", "total": 99.0}]},
                        "status": "compiled_complete",
                        "fused": True,
                    },
                ),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=25.0, clock=clock)
            return registry, model

        elif workload_id == "W3":
            spec_read = ToolSpec(name="read_customer_state", is_read_only=True, is_idempotent=True)
            spec_audit = ToolSpec(name="audit_transaction", is_read_only=True, is_idempotent=True)
            registry.register(ReplayToolAdapter("read_customer_state", spec_read, default_latency_ms=25.0, clock=clock))
            registry.register(ReplayToolAdapter("audit_transaction", spec_audit, default_latency_ms=25.0, clock=clock))

            spec_call = ToolCall(
                name="read_customer_state",
                arguments={"customer_id": f"cust_{trial_index}"},
                speculation_confidence=0.92,
            )
            decisions = [
                LLMDecision(reasoning="Reading customer state", tool_calls=[copy.deepcopy(spec_call)]),
                LLMDecision(
                    reasoning="Evaluating risk",
                    tool_calls=[],
                    final_answer={"status": "approved", "customer_id": f"cust_{trial_index}"},
                ),
            ]
            model = ReplayLLMAdapter(
                decisions=decisions, draft_prediction=spec_call, decision_delay_ms=25.0, clock=clock
            )
            return registry, model

        elif workload_id == "W4":
            spec = ToolSpec(name="pricing_lookup", is_read_only=True, is_idempotent=True)
            registry.register(ReplayToolAdapter("pricing_lookup", spec, default_latency_ms=25.0, clock=clock))
            key_id = f"item_{trial_index % 10}"
            call = ToolCall(name="pricing_lookup", arguments={"sku": key_id})
            decisions = [
                LLMDecision(reasoning="Looking up cached price", tool_calls=[call]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"sku": key_id, "price": 49.99}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=25.0, clock=clock)
            return registry, model

        elif workload_id == "W5":
            spec = ToolSpec(
                name="stream_query_data", is_read_only=True, is_idempotent=True, commit_horizon_args=["query", "limit"]
            )
            registry.register(ReplayToolAdapter("stream_query_data", spec, default_latency_ms=30.0, clock=clock))
            call = ToolCall(name="stream_query_data", arguments={"query": "SELECT * FROM metrics", "limit": 100})
            decisions = [
                LLMDecision(reasoning="Querying streaming dataset", tool_calls=[call]),
                LLMDecision(
                    reasoning="Synthesizing dataset", tool_calls=[], final_answer={"status": "success", "count": 100}
                ),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=35.0, stream_chunks_count=4, clock=clock)
            return registry, model

        elif workload_id == "W6":
            spec = ToolSpec(name="sandbox_compute", is_read_only=True, is_idempotent=True)
            registry.register(ReplayToolAdapter("sandbox_compute", spec, default_latency_ms=25.0, clock=clock))
            call = ToolCall(name="sandbox_compute", arguments={"op": "fib", "n": 10})
            decisions = [
                LLMDecision(reasoning="Computing sandbox output", tool_calls=[call]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"result": 55}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=25.0, clock=clock)
            return registry, model

        elif workload_id == "W7":
            spec = ToolSpec(
                name="execute_fund_transfer",
                is_read_only=False,
                side_effects=True,
                requires_approval=True,
                is_idempotent=True,
            )
            registry.register(ReplayToolAdapter("execute_fund_transfer", spec, default_latency_ms=25.0, clock=clock))
            idemp_key = f"tx_replay_{trial_index:04d}"
            call_args = {"recipient": "Alice", "amount": 100.0, "idempotency_key": idemp_key}
            call = ToolCall(
                name="execute_fund_transfer", arguments=call_args, requires_approval=True, idempotency_key=idemp_key
            )

            decisions = [
                LLMDecision(reasoning="Executing side-effect fund transfer", tool_calls=[call]),
                LLMDecision(
                    reasoning="Completed transfer",
                    tool_calls=[],
                    final_answer={"status": "transferred", "idempotency_key": idemp_key},
                ),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=25.0, clock=clock)
            return registry, model

        else:
            # E5a Bytecode transport
            spec = ToolSpec(name="bytecode_transport_tool", is_read_only=True, is_idempotent=True)
            registry.register(ReplayToolAdapter("bytecode_transport_tool", spec, default_latency_ms=20.0, clock=clock))
            call = ToolCall(
                name="bytecode_transport_tool", arguments={"payload_id": trial_index, "data": f"content_{trial_index}"}
            )
            decisions = [
                LLMDecision(reasoning="Bytecode transport dispatch", tool_calls=[call]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"status": "done", "trial": trial_index}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=25.0, clock=clock)
            return registry, model
