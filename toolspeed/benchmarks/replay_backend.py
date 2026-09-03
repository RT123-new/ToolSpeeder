"""Deterministic virtual-clock replay backend with immutable paired fixtures."""

from __future__ import annotations

import copy
import json
import threading
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from typing_extensions import Self

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
    TokenUsage,
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
    """Deterministic, causal virtual-time tool adapter evaluating arguments and state transitions."""

    def __init__(
        self,
        name: str,
        spec: ToolSpec,
        recorded_responses: list[dict[str, Any]] | None = None,
        default_latency_ms: float = 20.0,
        clock: Clock | None = None,
        state: dict[str, Any] | None = None,
        handler: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
    ):
        super().__init__(spec=spec)
        self._name = name
        self._spec: ToolSpec = spec
        self._recorded_responses = list(recorded_responses or [])
        self._default_latency_ms = default_latency_ms
        self._call_index = 0
        self.clock = clock or VirtualClock()
        self.state: dict[str, Any] = state if state is not None else {}
        self.handler = handler
        self.call_history: list[tuple[dict[str, Any], int]] = []
        self.idempotency_records: dict[str, dict[str, Any]] = {}

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
        start_ns = self.clock.now_ns()
        delay_ms = self._default_latency_ms
        resp_data: dict[str, Any] = {}

        # 1. Check idempotency for side-effect calls
        idemp_key = call.arguments.get("idempotency_key") if call.arguments else None
        if idemp_key and idemp_key in self.idempotency_records:
            resp_data = dict(self.idempotency_records[idemp_key])
            delay_ms = 5.0  # Idempotency hit executes with minimal cache-like latency
        elif self.handler is not None:
            # 2. Causal execution handler evaluates arguments against current state
            resp_data = self.handler(dict(call.arguments), self.state)
        else:
            # 3. Match by arguments if recorded response specifies arguments
            matched = None
            for r in self._recorded_responses:
                if "arguments" in r and r["arguments"] == call.arguments:
                    matched = r
                    break
            if matched is not None:
                resp_data = matched.get("output", {})
                delay_ms = float(matched.get("latency_ms", self._default_latency_ms))
            elif self._call_index < len(self._recorded_responses):
                recorded = self._recorded_responses[self._call_index]
                self._call_index += 1
                resp_data = recorded.get("output", {})
                delay_ms = float(recorded.get("latency_ms", self._default_latency_ms))
            else:
                resp_data = {"status": "ok", "echo_args": call.arguments}

        if idemp_key and idemp_key not in self.idempotency_records:
            self.idempotency_records[idemp_key] = resp_data

        # Advance virtual or wall time via Clock
        await self.clock.sleep_ms(delay_ms)
        end_ns = self.clock.now_ns()
        dur_ns = end_ns - start_ns if end_ns > start_ns else int(delay_ms * 1_000_000)

        self.call_history.append((dict(call.arguments), start_ns))

        return ToolResult(
            call_id=call.call_id,
            name=self._name,
            tool_name=self._name,
            result=resp_data,
            output=resp_data,
            is_error=False,
            execution_time_ns=dur_ns,
            execution_time_ms=delay_ms,
            started_at=start_ns / 1_000_000_000.0,
            finished_at=end_ns / 1_000_000_000.0,
            cost_usd=0.0001,
        )


class ReplayLLMAdapter(BaseLLMAdapter):
    """Deterministic, causal virtual-time LLM adapter evaluating decisions from execution history."""

    def __init__(
        self,
        decisions: list[LLMDecision] | None = None,
        draft_prediction: ToolCall | None = None,
        decision_delay_ms: float = 30.0,
        stream_chunks_count: int = 4,
        clock: Clock | None = None,
        policy: Callable[[AgentTask, list[dict[str, Any]]], LLMDecision] | None = None,
    ):
        self.decisions = list(decisions or [])
        self.draft_prediction = draft_prediction
        self.decision_delay_ms = decision_delay_ms
        self.stream_chunks_count = stream_chunks_count
        self._turn_index = 0
        self.clock = clock or VirtualClock()
        self.policy = policy

    def _get_decision_sync(self, task: AgentTask, history: list[dict[str, Any]] | None = None) -> LLMDecision:
        if self.policy is not None:
            return self.policy(task, history or [])
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
        return self._get_decision_sync(task, history)

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
        decision = self._get_decision_sync(task, history)
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


@dataclass
class ReplayCaseFixture:
    """Pre-generated immutable per-case fixture bound to seed, workload, arm, and epoch."""

    case_id: str
    seed: int
    workload_id: str
    arm: str
    epoch: int
    trial_index: int
    tool_latencies: dict[str, float]
    model_latency_ms: float
    tool_responses: dict[str, Any]
    trace_events: list[str]
    tokens: TokenUsage
    side_effects: dict[str, Any]
    prompt: str
    expected_output: dict[str, Any]
    metadata: dict[str, Any]
    _frozen: bool = False

    def freeze(self) -> Self:
        self._frozen = True
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False) and name != "_frozen":
            raise RuntimeError(f"Cannot mutate immutable ReplayCaseFixture: attempted to set '{name}'")
        super().__setattr__(name, value)

    def clone(self, new_arm: str | None = None, new_epoch: int | None = None) -> ReplayCaseFixture:
        """Create an independent, deep-copied fixture instance with optional new arm/epoch."""
        return ReplayCaseFixture(
            case_id=f"{self.workload_id}_s{self.seed}_{new_arm or self.arm}_ep{new_epoch if new_epoch is not None else self.epoch}_t{self.trial_index:04d}",
            seed=self.seed,
            workload_id=self.workload_id,
            arm=new_arm or self.arm,
            epoch=new_epoch if new_epoch is not None else self.epoch,
            trial_index=self.trial_index,
            tool_latencies=copy.deepcopy(self.tool_latencies),
            model_latency_ms=self.model_latency_ms,
            tool_responses=copy.deepcopy(self.tool_responses),
            trace_events=copy.deepcopy(self.trace_events),
            tokens=copy.deepcopy(self.tokens),
            side_effects=copy.deepcopy(self.side_effects),
            prompt=self.prompt,
            expected_output=copy.deepcopy(self.expected_output),
            metadata=copy.deepcopy(self.metadata),
            _frozen=False,
        ).freeze()


class ReplayFixtureManager:
    """Generates and manages pre-generated immutable fixtures across seeds, workloads, arms, and epochs."""

    def __init__(self) -> None:
        self._fixtures: dict[tuple[int, str, str, int], ReplayCaseFixture] = {}
        self._arm_epochs: dict[str, int] = {}
        self._lock = threading.RLock()

    def get_epoch_for_arm(self, arm: str) -> int:
        with self._lock:
            if arm not in self._arm_epochs:
                self._arm_epochs[arm] = len(self._arm_epochs) + 1
            return self._arm_epochs[arm]

    def advance_arm_epoch(self, arm: str) -> int:
        with self._lock:
            self._arm_epochs[arm] = self._arm_epochs.get(arm, 0) + 1
            return self._arm_epochs[arm]

    def get_or_create_fixture(
        self,
        workload_id: str,
        seed: int,
        arm: str,
        trial_index: int,
    ) -> ReplayCaseFixture:
        key = (seed, workload_id, arm, trial_index)
        with self._lock:
            if key in self._fixtures:
                return self._fixtures[key].clone()

            epoch = self.get_epoch_for_arm(arm)
            fixture = self._build_case_fixture(workload_id, seed, arm, trial_index, epoch)
            self._fixtures[key] = fixture
            return fixture.clone()

    def _build_case_fixture(
        self,
        workload_id: str,
        seed: int,
        arm: str,
        trial_index: int,
        epoch: int,
    ) -> ReplayCaseFixture:
        rng = np.random.default_rng(seed + trial_index * 1000 + epoch * 100)
        base_tool_delay = 20.0 + float(rng.uniform(0.0, 5.0))
        model_delay = 25.0 + float(rng.uniform(0.0, 5.0))

        tool_latencies: dict[str, float] = {}
        tool_responses: dict[str, Any] = {}
        trace_events: list[str] = ["CASE_START", "LLM_DECISION_START", "LLM_DECISION_END"]
        side_effects: dict[str, Any] = {}
        tokens = TokenUsage(
            prompt_tokens=150 + trial_index,
            completion_tokens=50,
            total_tokens=200 + trial_index,
            cost_usd=0.001,
        )

        if workload_id == "W1":
            for i in range(5):
                t_name = f"server_metric_{i}"
                tool_latencies[t_name] = base_tool_delay
                tool_responses[t_name] = {"metric": "cpu", "server_id": f"srv_{i}", "load": 50}
                trace_events.append(f"TOOL_CALL_{t_name}")
            prompt = f"Dispatch fanout metric queries for trial {trial_index} (seed={seed})"
            expected_output = {"status": "success", "total_load": 250, "server_count": 5}
            metadata = {"workload_id": "W1", "trial_index": trial_index, "seed": seed, "arm": arm, "epoch": epoch}

        elif workload_id == "W2":
            tool_latencies["fetch_user"] = base_tool_delay
            tool_responses["fetch_user"] = {"user_id": f"u_{trial_index}", "name": "Alice"}
            tool_latencies["fetch_orders"] = base_tool_delay
            tool_responses["fetch_orders"] = {"orders": [{"id": "ord_101", "total": 99.0}]}
            trace_events.extend(["TOOL_CALL_fetch_user", "TOOL_CALL_fetch_orders"])
            prompt = f"Execute user orders chain for user u_{trial_index} (seed={seed})"
            expected_output = {
                "user": {"user_id": f"u_{trial_index}", "name": "Alice"},
                "orders": {"orders": [{"id": "ord_101", "total": 99.0}]},
                "status": "compiled_complete",
                "fused": True,
            }
            metadata = {
                "workload_id": "W2",
                "trial_index": trial_index,
                "workflow_id": "user_orders",
                "seed": seed,
                "arm": arm,
                "epoch": epoch,
            }

        elif workload_id == "W3":
            tool_latencies["read_customer_state"] = base_tool_delay
            tool_responses["read_customer_state"] = {"customer_id": f"cust_{trial_index}", "risk_score": 10}
            tool_latencies["audit_transaction"] = base_tool_delay
            tool_responses["audit_transaction"] = {"customer_id": f"cust_{trial_index}", "audited": True}
            trace_events.extend(["TOOL_CALL_read_customer_state", "TOOL_CALL_audit_transaction"])
            prompt = f"Execute branching customer check for cust_{trial_index} (seed={seed})"
            expected_output = {"status": "approved", "customer_id": f"cust_{trial_index}"}
            metadata = {"workload_id": "W3", "trial_index": trial_index, "seed": seed, "arm": arm, "epoch": epoch}

        elif workload_id == "W4":
            key_id = f"item_{trial_index % 10}"
            tool_latencies["pricing_lookup"] = base_tool_delay
            tool_responses["pricing_lookup"] = {"sku": key_id, "price": 49.99}
            trace_events.append("TOOL_CALL_pricing_lookup")
            prompt = f"Lookup price for {key_id} (seed={seed})"
            expected_output = {"sku": key_id, "price": 49.99}
            metadata = {"workload_id": "W4", "trial_index": trial_index, "seed": seed, "arm": arm, "epoch": epoch}

        elif workload_id == "W5":
            tool_latencies["stream_query_data"] = base_tool_delay + 10.0
            tool_responses["stream_query_data"] = {"status": "success", "count": 100}
            trace_events.append("TOOL_CALL_stream_query_data")
            prompt = f"Stream query dataset for trial {trial_index} (seed={seed})"
            expected_output = {"status": "success", "count": 100}
            metadata = {"workload_id": "W5", "trial_index": trial_index, "seed": seed, "arm": arm, "epoch": epoch}

        elif workload_id == "W6":
            tool_latencies["sandbox_compute"] = base_tool_delay + 5.0
            tool_responses["sandbox_compute"] = {"result": 55}
            trace_events.append("TOOL_CALL_sandbox_compute")
            prompt = f"Run sandbox compute task for trial {trial_index} (seed={seed})"
            expected_output = {"result": 55}
            metadata = {"workload_id": "W6", "trial_index": trial_index, "seed": seed, "arm": arm, "epoch": epoch}

        elif workload_id == "W7":
            idemp_key = f"tx_replay_{trial_index:04d}"
            tool_latencies["execute_fund_transfer"] = base_tool_delay + 5.0
            tool_responses["execute_fund_transfer"] = {"status": "transferred", "idempotency_key": idemp_key}
            side_effects["balance"] = 1000.0 - (trial_index + 1) * 100.0
            trace_events.append("TOOL_CALL_execute_fund_transfer")
            prompt = f"Execute approved fund transfer tx_{trial_index} (seed={seed})"
            expected_output = {"status": "transferred", "idempotency_key": idemp_key}
            metadata = {
                "workload_id": "W7",
                "trial_index": trial_index,
                "seed": seed,
                "arm": arm,
                "epoch": epoch,
                "idempotency_key": idemp_key,
            }

        else:  # E5a
            tool_latencies["bytecode_transport_tool"] = base_tool_delay
            tool_responses["bytecode_transport_tool"] = {"status": "done", "trial": trial_index}
            trace_events.append("TOOL_CALL_bytecode_transport_tool")
            prompt = f"Serialize action bytecode for trial {trial_index} (seed={seed})"
            expected_output = {"status": "done", "trial": trial_index}
            metadata = {"workload_id": "E5a", "trial_index": trial_index, "seed": seed, "arm": arm, "epoch": epoch}

        trace_events.append("CASE_END")

        case_id = f"{workload_id}_s{seed}_{arm}_ep{epoch}_t{trial_index:04d}"
        fixture = ReplayCaseFixture(
            case_id=case_id,
            seed=seed,
            workload_id=workload_id,
            arm=arm,
            epoch=epoch,
            trial_index=trial_index,
            tool_latencies=tool_latencies,
            model_latency_ms=model_delay,
            tool_responses=tool_responses,
            trace_events=trace_events,
            tokens=tokens,
            side_effects=side_effects,
            prompt=prompt,
            expected_output=expected_output,
            metadata=metadata,
            _frozen=False,
        )
        return fixture.freeze()


class ReplayBackend:
    """Deterministic virtual-clock replay backend generating immutable paired workload fixtures."""

    def __init__(
        self,
        evidence_level: EvidenceLevel = EvidenceLevel.REPLAY_INTEGRATION,
        clock: Clock | None = None,
        seed: int = 42,
        fixture_manager: ReplayFixtureManager | None = None,
    ):
        self.evidence_level = evidence_level
        self.clock = clock or VirtualClock()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.fixture_manager = fixture_manager or ReplayFixtureManager()
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

    def generate_task(
        self, workload_id: str, trial_index: int = 0, seed: int | None = None, arm: str = "baseline"
    ) -> Task:
        """Constructs an immutable Task with seeded parameters from arm-isolated fixtures."""
        eff_seed = seed if seed is not None else self.seed
        fixture = self.fixture_manager.get_or_create_fixture(
            workload_id=workload_id,
            seed=eff_seed,
            arm=arm,
            trial_index=trial_index,
        )
        task_meta = dict(fixture.metadata)
        task_params: dict[str, Any] = {}
        if workload_id == "W7":
            idemp_key = f"tx_replay_{trial_index:04d}"
            task_params = {"recipient": "Alice", "amount": 100.0, "idempotency_key": idemp_key}
            grant = ApprovalGrant.create("execute_fund_transfer", task_params)
            task_meta["approval_grant"] = grant

        return Task(
            task_id=fixture.case_id,
            prompt=fixture.prompt,
            context={"user_id": f"u_{trial_index}"} if workload_id == "W2" else {},
            parameters=task_params,
            expected_output=copy.deepcopy(fixture.expected_output),
            metadata=task_meta,
        )

    def create_workload_environment(
        self, workload_id: str, trial_index: int = 0, arm: str = "baseline"
    ) -> tuple[ToolRegistry, BaseLLMAdapter]:
        """Creates paired tool registry and replay model adapter for the trial."""
        registry = ToolRegistry()
        clock = self.clock or VirtualClock()
        registry.clock = clock  # type: ignore[attr-defined]
        fixture = self.fixture_manager.get_or_create_fixture(
            workload_id=workload_id,
            seed=self.seed,
            arm=arm,
            trial_index=trial_index,
        )

        if workload_id == "W1":
            for i in range(5):
                t_name = f"server_metric_{i}"
                lat = fixture.tool_latencies.get(t_name, 20.0)
                spec = ToolSpec(name=t_name, is_read_only=True, is_idempotent=True)
                registry.register(ReplayToolAdapter(t_name, spec, default_latency_ms=lat, clock=clock))
            calls = [
                ToolCall(name=f"server_metric_{i}", arguments={"metric": "cpu", "server_id": f"srv_{i}"})
                for i in range(5)
            ]
            decisions = [
                LLMDecision(reasoning="Dispatching 5 fanout metrics", tool_calls=calls),
                LLMDecision(
                    reasoning="Synthesizing metrics",
                    tool_calls=[],
                    final_answer=dict(fixture.expected_output),
                ),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=fixture.model_latency_ms, clock=clock)
            return registry, model

        elif workload_id == "W2":
            spec_user = ToolSpec(name="fetch_user", is_read_only=True, is_idempotent=True)
            spec_orders = ToolSpec(name="fetch_orders", is_read_only=True, is_idempotent=True)
            u_lat = fixture.tool_latencies.get("fetch_user", 20.0)
            o_lat = fixture.tool_latencies.get("fetch_orders", 20.0)
            registry.register(
                ReplayToolAdapter(
                    "fetch_user",
                    spec_user,
                    [
                        {
                            "output": fixture.tool_responses.get(
                                "fetch_user", {"user_id": f"u_{trial_index}", "name": "Alice"}
                            ),
                            "latency_ms": u_lat,
                        }
                    ],
                    default_latency_ms=u_lat,
                    clock=clock,
                )
            )
            registry.register(
                ReplayToolAdapter(
                    "fetch_orders",
                    spec_orders,
                    [
                        {
                            "output": fixture.tool_responses.get(
                                "fetch_orders", {"orders": [{"id": "ord_101", "total": 99.0}]}
                            ),
                            "latency_ms": o_lat,
                        }
                    ],
                    default_latency_ms=o_lat,
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
                    final_answer=dict(fixture.expected_output),
                ),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=fixture.model_latency_ms, clock=clock)
            return registry, model

        elif workload_id == "W3":
            spec_read = ToolSpec(name="read_customer_state", is_read_only=True, is_idempotent=True)
            spec_audit = ToolSpec(name="audit_transaction", is_read_only=True, is_idempotent=True)
            r_lat = fixture.tool_latencies.get("read_customer_state", 25.0)
            a_lat = fixture.tool_latencies.get("audit_transaction", 25.0)
            registry.register(
                ReplayToolAdapter("read_customer_state", spec_read, default_latency_ms=r_lat, clock=clock)
            )
            registry.register(ReplayToolAdapter("audit_transaction", spec_audit, default_latency_ms=a_lat, clock=clock))

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
                    final_answer=dict(fixture.expected_output),
                ),
            ]
            model = ReplayLLMAdapter(
                decisions=decisions, draft_prediction=spec_call, decision_delay_ms=fixture.model_latency_ms, clock=clock
            )
            return registry, model

        elif workload_id == "W4":
            p_lat = fixture.tool_latencies.get("pricing_lookup", 25.0)
            spec = ToolSpec(name="pricing_lookup", is_read_only=True, is_idempotent=True)
            registry.register(ReplayToolAdapter("pricing_lookup", spec, default_latency_ms=p_lat, clock=clock))
            key_id = f"item_{trial_index % 10}"
            call = ToolCall(name="pricing_lookup", arguments={"sku": key_id})
            decisions = [
                LLMDecision(reasoning="Looking up cached price", tool_calls=[call]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer=dict(fixture.expected_output)),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=fixture.model_latency_ms, clock=clock)
            return registry, model

        elif workload_id == "W5":
            s_lat = fixture.tool_latencies.get("stream_query_data", 30.0)
            spec = ToolSpec(
                name="stream_query_data", is_read_only=True, is_idempotent=True, commit_horizon_args=["query", "limit"]
            )
            registry.register(ReplayToolAdapter("stream_query_data", spec, default_latency_ms=s_lat, clock=clock))
            call = ToolCall(name="stream_query_data", arguments={"query": "SELECT * FROM metrics", "limit": 100})
            decisions = [
                LLMDecision(reasoning="Querying streaming dataset", tool_calls=[call]),
                LLMDecision(
                    reasoning="Synthesizing dataset", tool_calls=[], final_answer=dict(fixture.expected_output)
                ),
            ]
            model = ReplayLLMAdapter(
                decisions=decisions, decision_delay_ms=fixture.model_latency_ms, stream_chunks_count=4, clock=clock
            )
            return registry, model

        elif workload_id == "W6":
            c_lat = fixture.tool_latencies.get("sandbox_compute", 25.0)
            spec = ToolSpec(name="sandbox_compute", is_read_only=True, is_idempotent=True)
            registry.register(ReplayToolAdapter("sandbox_compute", spec, default_latency_ms=c_lat, clock=clock))
            call = ToolCall(name="sandbox_compute", arguments={"op": "fib", "n": 10})
            decisions = [
                LLMDecision(reasoning="Computing sandbox output", tool_calls=[call]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer=dict(fixture.expected_output)),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=fixture.model_latency_ms, clock=clock)
            return registry, model

        elif workload_id == "W7":
            f_lat = fixture.tool_latencies.get("execute_fund_transfer", 25.0)
            spec = ToolSpec(
                name="execute_fund_transfer",
                is_read_only=False,
                side_effects=True,
                requires_approval=True,
                is_idempotent=True,
            )
            registry.register(ReplayToolAdapter("execute_fund_transfer", spec, default_latency_ms=f_lat, clock=clock))
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
                    final_answer=dict(fixture.expected_output),
                ),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=fixture.model_latency_ms, clock=clock)
            return registry, model

        else:
            # E5a Bytecode transport
            b_lat = fixture.tool_latencies.get("bytecode_transport_tool", 20.0)
            spec = ToolSpec(name="bytecode_transport_tool", is_read_only=True, is_idempotent=True)
            registry.register(ReplayToolAdapter("bytecode_transport_tool", spec, default_latency_ms=b_lat, clock=clock))
            call = ToolCall(
                name="bytecode_transport_tool", arguments={"payload_id": trial_index, "data": f"content_{trial_index}"}
            )
            decisions = [
                LLMDecision(reasoning="Bytecode transport dispatch", tool_calls=[call]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer=dict(fixture.expected_output)),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=fixture.model_latency_ms, clock=clock)
            return registry, model
