"""45 Comprehensive Adversarial Tests for Scientific Integrity and Runtime Safety."""

from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from toolspeed import cli
from toolspeed.adapters.base import (
    BaseLLMAdapter,
    BaseToolAdapter,
    LLMDecision,
    StreamingChunk,
    ToolRegistry,
    ToolSchema,
)
from toolspeed.adapters.live_tools import (
    AsyncSQLiteTool,
    SafeSubprocessSandbox,
)
from toolspeed.benchmarks.harness import (
    BenchmarkConfig,
    BenchmarkHarness,
)
from toolspeed.core.guardrails import GuardrailTracker
from toolspeed.core.rate_limiter import AsyncConcurrencyLimiter, RateLimiter
from toolspeed.core.types import (
    AgentTask,
    ApprovalGrant,
    EventType,
    ExecutionTrace,
    Task,
    TaskInstance,
    ToolCall,
    ToolResult,
    ToolSpec,
    strict_json_dumps,
)
from toolspeed.schedulers.b1_sync_react import SyncReActScheduler
from toolspeed.schedulers.b2_native_parallel import NativeParallelScheduler
from toolspeed.schedulers.base import SchedulerConfig
from toolspeed.schedulers.composite import CompositeScheduler
from toolspeed.schedulers.e1_dag_scheduler import DAGScheduler, ToolDAG
from toolspeed.schedulers.e2_jit_fusion import (
    DeclarativeWorkflow,
    JITFusionScheduler,
    WorkflowNode,
)
from toolspeed.schedulers.e3_speculation import SpeculativeReadScheduler
from toolspeed.schedulers.e4_commit_horizon import CommitHorizonScheduler, IncrementalCommitParser
from toolspeed.schedulers.e5_action_bytecode import ActionBytecodeCodec, ActionBytecodeScheduler
from toolspeed.schedulers.executor import SharedIdempotencyStore, ToolExecutor
from toolspeed.schedulers.phase2_cache import CacheEntry, ToolResultCache


class SimpleMockTool(BaseToolAdapter):
    def __init__(
        self,
        name: str,
        output: Any = "ok",
        is_read_only: bool = True,
        side_effects: bool = False,
        requires_approval: bool = False,
        latency_s: float = 0.001,
        raise_error: str | None = None,
    ):
        self._name = name
        self.output = output
        self._is_read_only = is_read_only
        self._side_effects = side_effects
        self._requires_approval = requires_approval
        self.latency_s = latency_s
        self.raise_error = raise_error
        self.executions: list[ToolCall] = []

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self._name,
            description=f"Mock {self._name}",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "amount": {"type": "number"},
                    "user_id": {"type": "string"},
                },
            },
            is_read_only=self._is_read_only,
            is_side_effect=self._side_effects,
            requires_approval=self._requires_approval,
            cost_usd=0.0001,
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        self.executions.append(call)
        if self.latency_s > 0:
            await asyncio.sleep(self.latency_s)
        if self.raise_error:
            return ToolResult(
                call_id=call.call_id, name=self._name, tool_name=self._name, error=self.raise_error, is_error=True
            )
        return ToolResult(
            call_id=call.call_id,
            name=self._name,
            tool_name=self._name,
            result=self.output,
            output=self.output,
            is_error=False,
        )


class MockLLM(BaseLLMAdapter):
    def __init__(
        self,
        decisions: list[LLMDecision] | None = None,
        draft_prediction: ToolCall | None = None,
        predict_draft_raises: bool = False,
        chunks: list[StreamingChunk] | None = None,
    ):
        self.decisions = list(decisions or [])
        self.draft_prediction = draft_prediction
        self.predict_draft_raises = predict_draft_raises
        self.chunks = list(chunks or [])
        self._turn = 0

    async def decide(
        self, task: AgentTask, history: list[dict[str, Any]], available_tools: list[ToolSpec]
    ) -> LLMDecision:
        if self._turn < len(self.decisions):
            d = self.decisions[self._turn]
            self._turn += 1
            return d
        return LLMDecision(reasoning="Done", tool_calls=[], final_answer="done")

    async def predict_draft(
        self, task: AgentTask, history: list[dict[str, Any]], available_tools: list[ToolSpec]
    ) -> ToolCall | None:
        if self.predict_draft_raises:
            raise RuntimeError("Draft predictor failed")
        return self.draft_prediction

    async def stream_decision(
        self, task: AgentTask, history: list[dict[str, Any]], available_tools: list[ToolSpec]
    ) -> AsyncIterator[StreamingChunk]:
        if self.chunks and self._turn == 0:
            self._turn += 1
            for c in self.chunks:
                yield c
            return

        if self._turn < len(self.decisions):
            d = self.decisions[self._turn]
            self._turn += 1
            if d.final_answer is not None or not d.tool_calls:
                yield StreamingChunk(
                    token_index=0,
                    delta_text=d.reasoning or "done",
                    is_final=True,
                    parsed_tool_calls=[],
                    metadata={"final_answer": d.final_answer},
                )
            else:
                yield StreamingChunk(
                    token_index=0,
                    delta_text=d.reasoning or "calling tools",
                    is_final=True,
                    parsed_tool_calls=d.tool_calls,
                )
            return

        final_ans = "done"
        yield StreamingChunk(
            token_index=0,
            delta_text="done",
            is_final=True,
            metadata={"final_answer": final_ans},
        )


class TestAdversarialIntegrity(unittest.IsolatedAsyncioTestCase):
    """45 Exhaustive Adversarial Tests for Scientific Integrity and Runtime Safety."""

    # 1. DAG: Unknown dependency reference fails closed before dispatch
    async def test_01_unknown_reference_fails_closed(self) -> None:
        dag = ToolDAG()
        call = ToolCall(call_id="node1", name="tool_a", arguments={"input": "$non_existent_node.result"})
        dag.register_calls([call])
        self.assertEqual(dag.nodes["node1"].status, "failed")
        self.assertIn("Unknown dependency reference", dag.nodes["node1"].error or "")

    # 2. DAG: Nested reference resolution
    async def test_02_nested_reference_resolution(self) -> None:
        dag = ToolDAG()
        c1 = ToolCall(call_id="c1", name="parent_tool", arguments={})
        c2 = ToolCall(
            call_id="c2", name="child_tool", arguments={"nested": {"items": ["$c1.user_id", {"key": "$c1.org"}]}}
        )
        dag.register_calls([c1, c2])

        dag.nodes["c1"].result = ToolResult(
            call_id="c1", name="parent_tool", output={"user_id": "u123", "org": "corp_a"}
        )
        dag.nodes["c1"].status = "completed"

        resolved, err = dag.resolve_arguments(dag.nodes["c2"])
        self.assertIsNone(err)
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved["nested"]["items"][0], "u123")
        self.assertEqual(resolved["nested"]["items"][1]["key"], "corp_a")

    # 3. DAG: Direct 2-node cycle detection
    async def test_03_dag_direct_cycle_diagnostic(self) -> None:
        dag = ToolDAG()
        c1 = ToolCall(call_id="a", name="t1", arguments={"ref": "$b.out"})
        c2 = ToolCall(call_id="b", name="t2", arguments={"ref": "$a.out"})
        dag.register_calls([c1, c2])
        cycle = dag.detect_cycles()
        self.assertIsNotNone(cycle)
        assert cycle is not None
        self.assertIn("a", cycle)
        self.assertIn("b", cycle)

    # 4. DAG: Indirect 3-node cycle detection
    async def test_04_dag_indirect_cycle_detection(self) -> None:
        dag = ToolDAG()
        c1 = ToolCall(call_id="x", name="t1", arguments={"ref": "$z.out"})
        c2 = ToolCall(call_id="y", name="t2", arguments={"ref": "$x.out"})
        c3 = ToolCall(call_id="z", name="t3", arguments={"ref": "$y.out"})
        dag.register_calls([c1, c2, c3])
        cycle = dag.detect_cycles()
        self.assertIsNotNone(cycle)

    # 5. DAG: Ambiguous tool reference rejected
    async def test_05_ambiguous_reference_rejected(self) -> None:
        dag = ToolDAG()
        c1 = ToolCall(call_id="call_1", name="lookup", arguments={})
        c2 = ToolCall(call_id="call_2", name="lookup", arguments={})
        c3 = ToolCall(call_id="call_3", name="aggregator", arguments={"data": "$lookup.val"})
        dag.register_calls([c1, c2, c3])
        self.assertEqual(dag.nodes["call_3"].status, "failed")
        self.assertIn("Ambiguous reference", dag.nodes["call_3"].error or "")

    # 6. DAG: Missing parent output field fails closed
    async def test_06_missing_parent_output_field_fails_closed(self) -> None:
        dag = ToolDAG()
        c1 = ToolCall(call_id="c1", name="parent", arguments={})
        c2 = ToolCall(call_id="c2", name="child", arguments={"target": "$c1.non_existent_key"})
        dag.register_calls([c1, c2])
        dag.nodes["c1"].result = ToolResult(call_id="c1", name="parent", output={"other_key": 42})
        dag.nodes["c1"].status = "completed"

        _resolved, err = dag.resolve_arguments(dag.nodes["c2"])
        self.assertIsNotNone(err)
        self.assertIn("Missing output field 'non_existent_key'", err or "")

    # 7. DAG: Parent failure propagates downstream
    async def test_07_dag_parent_failure_propagation(self) -> None:
        dag = ToolDAG()
        c1 = ToolCall(call_id="c1", name="parent", arguments={})
        c2 = ToolCall(call_id="c2", name="child", arguments={"input": "$c1.val"})
        dag.register_calls([c1, c2])
        dag.nodes["c1"].status = "failed"
        dag.nodes["c1"].error = "Parent network timeout"

        ready = dag.get_ready_nodes()
        self.assertEqual(len(ready), 0)
        self.assertEqual(dag.nodes["c2"].status, "failed")
        self.assertIn("Parent dependency 'c1' failed", dag.nodes["c2"].error or "")

    # 8. JIT: Arbitrary untrusted string callable rejected
    async def test_08_jit_arbitrary_callable_rejected(self) -> None:
        sched = JITFusionScheduler()
        tools = ToolRegistry()
        tools.register(SimpleMockTool("t1"))

        task = Task(task_id="t_bad", prompt="eval", metadata={"declarative_workflow": "lambda x: 1/0"})
        model = MockLLM(decisions=[LLMDecision(reasoning="Done", tool_calls=[], final_answer="fallback_done")])
        res = await sched.execute(task, model, tools)
        self.assertEqual(res.final_answer, "fallback_done")

    # 9. JIT: Deopt does not repeat completed side effects
    async def test_09_jit_deopt_does_not_repeat_side_effects(self) -> None:
        sched = JITFusionScheduler()
        tools = ToolRegistry()
        t_write = SimpleMockTool("charge_card", output={"charged": 100}, is_read_only=False, side_effects=True)
        t_fail = SimpleMockTool("send_receipt", raise_error="SMTP server unreachable")
        tools.register(t_write)
        tools.register(t_fail)

        wf = DeclarativeWorkflow(
            workflow_id="order_charge",
            nodes=(
                WorkflowNode(
                    step_id="s1",
                    tool_name="charge_card",
                    args_template={"amount": 100},
                    output_key="charge",
                    is_side_effect=True,
                ),
                WorkflowNode(
                    step_id="s2",
                    tool_name="send_receipt",
                    args_template={"receipt": "$charge.charged"},
                    output_key="receipt",
                ),
            ),
        )
        task = Task(
            task_id="t_deopt",
            prompt="Charge",
            expected_output={"status": "handled_in_fallback"},
            validator=lambda out, trace=None: out == {"status": "handled_in_fallback"},
            metadata={"declarative_workflow": wf},
        )
        model = MockLLM(
            decisions=[
                LLMDecision(reasoning="Fallback handle", tool_calls=[], final_answer={"status": "handled_in_fallback"})
            ]
        )
        res = await sched.execute(task, model, tools)
        self.assertTrue(res.success)
        self.assertEqual(len(t_write.executions), 1)

    # 10. Speculation: Concurrent draft prediction alongside model reasoning
    async def test_10_speculation_concurrent_model_reasoning(self) -> None:
        sched = SpeculativeReadScheduler(
            SchedulerConfig(speculation_enabled=True, speculation_confidence_threshold=0.5)
        )
        tools = ToolRegistry()
        tools.register(SimpleMockTool("search", output={"val": 123}, latency_s=0.005))

        predicted_call = ToolCall(name="search", arguments={"query": "alpha"}, speculation_confidence=0.95)
        model = MockLLM(
            decisions=[
                LLMDecision(reasoning="Deciding", tool_calls=[ToolCall(name="search", arguments={"query": "alpha"})]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"val": 123}),
            ],
            draft_prediction=predicted_call,
        )
        task = Task(task_id="t_spec", prompt="Search", expected_output={"val": 123})
        res = await sched.execute(task, model, tools)
        self.assertTrue(res.success)
        events = [
            e
            for e in res.events
            if e.event_type in (EventType.SPECULATION_HIT, "speculation_hit", EventType.SPECULATION_HIT.value)
        ]
        self.assertGreaterEqual(len(events), 1)

    # 11. Speculation: Multi-call decision matching
    async def test_11_speculation_multi_call_matching(self) -> None:
        sched = SpeculativeReadScheduler(
            SchedulerConfig(speculation_enabled=True, speculation_confidence_threshold=0.5)
        )
        tools = ToolRegistry()
        tools.register(SimpleMockTool("read_a", output={"a": 1}))
        tools.register(SimpleMockTool("read_b", output={"b": 2}))

        predicted = ToolCall(name="read_b", arguments={"id": 2}, speculation_confidence=0.9)
        call_a = ToolCall(call_id="ca", name="read_a", arguments={"id": 1})
        call_b = ToolCall(call_id="cb", name="read_b", arguments={"id": 2})

        model = MockLLM(
            decisions=[
                LLMDecision(reasoning="Two calls", tool_calls=[call_a, call_b]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"ok": True}),
            ],
            draft_prediction=predicted,
        )
        task = Task(task_id="t_multi", prompt="Read two", expected_output={"ok": True})
        res = await sched.execute(task, model, tools)
        self.assertTrue(res.success)
        self.assertEqual(len(res.tool_results), 2)
        b_res = next(r for r in res.tool_results if r.call_id == "cb")
        self.assertEqual(b_res.output, {"b": 2})

    # 12. Speculation: Predictor error fallback
    async def test_12_speculation_predictor_failure_fallback(self) -> None:
        sched = SpeculativeReadScheduler(SchedulerConfig(speculation_enabled=True))
        tools = ToolRegistry()
        tools.register(SimpleMockTool("lookup", output={"val": 42}))

        model = MockLLM(
            decisions=[
                LLMDecision(reasoning="Lookup", tool_calls=[ToolCall(name="lookup", arguments={})]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"val": 42}),
            ],
            predict_draft_raises=True,
        )
        task = Task(task_id="t_pred_fail", prompt="Lookup", expected_output={"val": 42})
        res = await sched.execute(task, model, tools)
        self.assertTrue(res.success)
        self.assertEqual(res.final_answer, {"val": 42})

    # 13. Speculation: Cancelled cleanup on miss
    async def test_13_speculation_cancelled_cleanup(self) -> None:
        sched = SpeculativeReadScheduler(
            SchedulerConfig(speculation_enabled=True, speculation_contention_mode="shared_cancellable")
        )
        tools = ToolRegistry()
        t_slow = SimpleMockTool("slow_read", latency_s=0.2)
        tools.register(t_slow)
        tools.register(SimpleMockTool("actual_read", output="good"))

        predicted = ToolCall(name="slow_read", arguments={}, speculation_confidence=0.9)
        model = MockLLM(
            decisions=[
                LLMDecision(reasoning="Choose actual", tool_calls=[ToolCall(name="actual_read", arguments={})]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer="good"),
            ],
            draft_prediction=predicted,
        )
        task = Task(task_id="t_miss", prompt="Miss test", expected_output="good")
        res = await sched.execute(task, model, tools)
        self.assertTrue(res.success)
        cancels = [
            e
            for e in res.events
            if e.event_type
            in (EventType.SPECULATION_CANCELLED, "speculative_cancel", EventType.SPECULATION_CANCELLED.value)
        ]
        self.assertGreaterEqual(len(cancels), 1)

    # 14. Speculation: Mutative tools strictly prohibited from speculation
    async def test_14_speculation_mutative_tool_prohibited(self) -> None:
        sched = SpeculativeReadScheduler(SchedulerConfig(speculation_enabled=True))
        tools = ToolRegistry()
        t_mutative = SimpleMockTool("delete_db", is_read_only=False, side_effects=True)
        tools.register(t_mutative)

        predicted = ToolCall(name="delete_db", arguments={}, speculation_confidence=0.99)
        model = MockLLM(
            decisions=[LLMDecision(reasoning="Stop", tool_calls=[], final_answer="safe")],
            draft_prediction=predicted,
        )
        task = Task(task_id="t_mut_spec", prompt="Check", expected_output="safe")
        res = await sched.execute(task, model, tools)
        self.assertTrue(res.success)
        self.assertEqual(len(t_mutative.executions), 0)

    # 15. Commit Horizon: Syntax closure required
    def test_15_commit_horizon_syntax_closure(self) -> None:
        self.assertFalse(IncrementalCommitParser.is_syntax_closed('{"query": "open string'))
        self.assertFalse(IncrementalCommitParser.is_syntax_closed('{"count": 12,'))
        self.assertTrue(IncrementalCommitParser.is_syntax_closed('{"query": "closed string", "count": 12}'))

    # 16. Commit Horizon: Mutated arguments reconciled with authoritative final
    async def test_16_commit_horizon_mutated_args_reconciled(self) -> None:
        sched = CommitHorizonScheduler()
        tools = ToolRegistry()
        t_search = SimpleMockTool("search", output={"items": ["res"]})
        tools.register(t_search)

        early_call = ToolCall(call_id="c_early", name="search", arguments={"query": "preliminary"})
        final_call = ToolCall(call_id="c_early", name="search", arguments={"query": "authoritative_final"})

        chunks = [
            StreamingChunk(
                token_index=0,
                delta_text="call search",
                commit_horizon_ready=[early_call],
                raw_json_fragment='{"query": "preliminary"}',
            ),
            StreamingChunk(token_index=1, delta_text="finished", is_final=True, parsed_tool_calls=[final_call]),
        ]
        model = MockLLM(
            chunks=chunks,
            decisions=[LLMDecision(reasoning="Done", tool_calls=[], final_answer="done")],
        )
        task = Task(task_id="t_ch_mut", prompt="Search", expected_output="done")
        res = await sched.execute(task, model, tools)
        self.assertTrue(res.success)
        self.assertEqual(t_search.executions[-1].arguments["query"], "authoritative_final")

    # 17. Commit Horizon: Side-effects cannot early dispatch
    async def test_17_commit_horizon_side_effect_safety(self) -> None:
        sched = CommitHorizonScheduler()
        tools = ToolRegistry()
        t_pay = SimpleMockTool("pay", output={"tx": "tx1"}, is_read_only=False, side_effects=True)
        tools.register(t_pay)

        call_pay = ToolCall(call_id="c_pay", name="pay", arguments={"amount": 100})
        chunks = [
            StreamingChunk(
                token_index=0, delta_text="pay", commit_horizon_ready=[call_pay], raw_json_fragment='{"amount": 100}'
            ),
            StreamingChunk(token_index=1, delta_text="end", is_final=True, parsed_tool_calls=[]),
        ]
        model = MockLLM(chunks=chunks)
        task = Task(task_id="t_ch_side", prompt="Pay", expected_output="done")
        await sched.execute(task, model, tools)
        self.assertEqual(len(t_pay.executions), 0)

    # 18. Action Bytecode: 16-bit opcode with no collision at 256
    def test_18_action_bytecode_16bit_opcode_no_collision(self) -> None:
        codec = ActionBytecodeCodec()
        op255 = codec.register_tool("tool_255", opcode=255)
        op256 = codec.register_tool("tool_256", opcode=256)
        self.assertEqual(op255, 255)
        self.assertEqual(op256, 256)

        call = ToolCall(name="tool_256", arguments={"user_id": 999, "active": True})
        encoded = codec.encode(call)
        decoded = codec.decode(encoded)
        self.assertEqual(decoded.name, "tool_256")
        self.assertEqual(decoded.arguments, {"user_id": 999, "active": True})

    # 19. Action Bytecode: Trailing bytes rejected
    def test_19_action_bytecode_trailing_bytes_rejected(self) -> None:
        codec = ActionBytecodeCodec()
        codec.register_tool("lookup", opcode=1)
        encoded = codec.encode(ToolCall(name="lookup", arguments={"key": "v"}))
        corrupted = encoded + b"EXTRA_CORRUPT_BYTES"
        with self.assertRaises(ValueError) as cm:
            codec.decode(corrupted)
        self.assertIn("unexpected trailing bytes", str(cm.exception))

    # 20. Rate Limiter: Cancellation safe without token loss
    async def test_20_rate_limiter_cancellation_safe(self) -> None:
        limiter = RateLimiter(rate_per_sec=10.0, burst_capacity=5.0, max_concurrency=1)
        await limiter.concurrency_limiter.acquire()

        async def _acquire_timeout() -> None:
            await limiter.acquire(tokens=2, timeout=0.02)

        with self.assertRaises((asyncio.TimeoutError, asyncio.CancelledError)):
            await _acquire_timeout()

        limiter.concurrency_limiter.release()
        self.assertAlmostEqual(limiter.token_bucket.available_tokens, 5.0, delta=0.5)

    # 21. Rate Limiter: Bounded semaphore prevents over-release
    def test_21_rate_limiter_bounded_semaphore(self) -> None:
        conc = AsyncConcurrencyLimiter(max_concurrency=2)
        conc.release()
        conc.release()
        self.assertEqual(conc.active_count, 0)

    # 22. Guardrails: Extra unexpected arguments penalized
    def test_22_guardrails_extra_arguments_penalized(self) -> None:
        tracker = GuardrailTracker()
        task = TaskInstance(
            task_id="t1",
            workload_id="W1",
            expected_tools=["search"],
            expected_args={"search": {"query": "apple"}},
        )
        tracker.register_task(task)

        call = ToolCall(name="search", arguments={"query": "apple", "extra_field": "injected"})
        trace = ExecutionTrace(
            task_id="t1",
            tool_calls=[call],
            success=True,
            start_ns=0,
            end_ns=1000,
        )
        tracker.record_trace(trace)
        metrics = tracker.calculate_metrics()
        self.assertEqual(metrics.argument_accuracy, 0.0)

    # 23. Zero unhandled leaked asyncio tasks across all schedulers
    async def test_23_no_unhandled_asyncio_task_leaks(self) -> None:
        schedulers = [
            SyncReActScheduler(),
            NativeParallelScheduler(),
            DAGScheduler(),
            JITFusionScheduler(),
            SpeculativeReadScheduler(SchedulerConfig(speculation_enabled=True)),
            CommitHorizonScheduler(),
            ActionBytecodeScheduler(),
        ]
        tools = ToolRegistry()
        tools.register(SimpleMockTool("t_ok", output="val"))

        task = Task(task_id="leak_test", prompt="Run", expected_output="val")
        model = MockLLM(
            decisions=[
                LLMDecision(reasoning="Call", tool_calls=[ToolCall(name="t_ok", arguments={})]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer="val"),
            ],
            draft_prediction=ToolCall(name="t_ok", arguments={}, speculation_confidence=0.9),
            chunks=[
                StreamingChunk(token_index=0, is_final=True, parsed_tool_calls=[ToolCall(name="t_ok", arguments={})])
            ],
        )

        for sched in schedulers:
            model = MockLLM(
                decisions=[
                    LLMDecision(reasoning="Call", tool_calls=[ToolCall(name="t_ok", arguments={})]),
                    LLMDecision(reasoning="Done", tool_calls=[], final_answer="val"),
                ],
                draft_prediction=ToolCall(name="t_ok", arguments={}, speculation_confidence=0.9),
                chunks=[
                    StreamingChunk(
                        token_index=0, is_final=True, parsed_tool_calls=[ToolCall(name="t_ok", arguments={})]
                    )
                ],
            )
            res = await sched.execute(task, model, tools)
            self.assertTrue(res.success)

        await asyncio.sleep(0.01)
        current = asyncio.current_task()
        pending = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
        self.assertEqual(len(pending), 0, f"Leaked tasks detected: {pending}")

    # 24. ToolExecutor: Schema validation rejects invalid argument types
    async def test_24_tool_executor_schema_validation_types(self) -> None:
        tools = ToolRegistry()
        tools.register(SimpleMockTool("query_tool"))
        executor = ToolExecutor(tools)
        call = ToolCall(name="query_tool", arguments={"amount": "not_a_number"})
        res = await executor.execute(call)
        self.assertTrue(res.is_error)
        self.assertIn("expected number", res.error or "")

    # 25. ToolExecutor: Rejection of unresolved variable reference strings ($c1.user_id)
    async def test_25_tool_executor_unresolved_ref_rejection(self) -> None:
        tools = ToolRegistry()
        tools.register(SimpleMockTool("query_tool"))
        executor = ToolExecutor(tools)
        call = ToolCall(name="query_tool", arguments={"user_id": "$c1.user_id"})
        res = await executor.execute(call)
        self.assertTrue(res.is_error)
        self.assertIn("Unresolved reference", res.error or "")

    # 26. ToolExecutor: Model-forged is_approved flag rejected without ApprovalGrant
    async def test_26_tool_executor_approval_forged_flag_rejected(self) -> None:
        tools = ToolRegistry()
        tools.register(SimpleMockTool("transfer", is_read_only=False, side_effects=True, requires_approval=True))
        executor = ToolExecutor(tools)
        # Model claims is_approved=True without trusted ApprovalGrant
        call = ToolCall(name="transfer", arguments={"amount": 100}, is_approved=True, approval_grant=None)
        res = await executor.execute(call)
        self.assertTrue(res.is_error)
        self.assertIn("Action rejected: tool requires explicit trusted approval grant", res.error or "")

    # 27. ToolExecutor: Valid trusted ApprovalGrant accepted
    async def test_27_tool_executor_trusted_approval_grant_accepted(self) -> None:
        tools = ToolRegistry()
        tools.register(SimpleMockTool("transfer", is_read_only=False, side_effects=True, requires_approval=True))
        executor = ToolExecutor(tools)
        args = {"amount": 100}
        grant = ApprovalGrant.create(tool_name="transfer", arguments=args, authority="trusted_system")
        call = ToolCall(name="transfer", arguments=args, is_approved=True)
        res = await executor.execute(call, trusted_grant=grant)
        self.assertFalse(res.is_error)

    # 28. SharedIdempotencyStore: Concurrent deduplication (primary + joiners)
    async def test_28_idempotency_store_concurrent_deduplication(self) -> None:
        tools = ToolRegistry()
        t = SimpleMockTool("idempotent_write", latency_s=0.01, is_read_only=False, side_effects=True)
        tools.register(t)
        store = SharedIdempotencyStore()
        executor = ToolExecutor(tools, idempotency_store=store)

        call1 = ToolCall(call_id="c1", name="idempotent_write", arguments={"amount": 50}, idempotency_key="key_123")
        call2 = ToolCall(call_id="c2", name="idempotent_write", arguments={"amount": 50}, idempotency_key="key_123")

        res1, res2 = await asyncio.gather(executor.execute(call1), executor.execute(call2))
        self.assertFalse(res1.is_error)
        self.assertFalse(res2.is_error)
        self.assertEqual(len(t.executions), 1)

    # 29. SharedIdempotencyStore: Argument mismatch for same key fails closed
    async def test_29_idempotency_store_argument_mismatch_fails_closed(self) -> None:
        tools = ToolRegistry()
        t = SimpleMockTool("idempotent_write", latency_s=0.01, is_read_only=False, side_effects=True)
        tools.register(t)
        store = SharedIdempotencyStore()
        executor = ToolExecutor(tools, idempotency_store=store)

        call1 = ToolCall(call_id="c1", name="idempotent_write", arguments={"amount": 50}, idempotency_key="key_123")
        call2 = ToolCall(call_id="c2", name="idempotent_write", arguments={"amount": 999}, idempotency_key="key_123")

        res1, res2 = await asyncio.gather(executor.execute(call1), executor.execute(call2))
        self.assertFalse(res1.is_error)
        self.assertTrue(res2.is_error)
        self.assertIn("Idempotency conflict", res2.error or "")

    # 30. Cache: Invalidation on matching mutative tool
    async def test_30_cache_invalidation_on_mutation(self) -> None:
        cache = ToolResultCache()
        cache.put("get_user", {"user_id": "u1"}, {"name": "Alice"})
        _cached, hit, _ = cache.get("get_user", {"user_id": "u1"})
        self.assertTrue(hit)

        cache.invalidate_on_mutation("update_user")
        _cached2, hit2, _ = cache.get("get_user", {"user_id": "u1"})
        self.assertFalse(hit2)

    # 31. Cache: Strict freshness contract rejects expired TTL
    def test_31_cache_freshness_contract(self) -> None:
        entry = CacheEntry(
            tool_name="get_data",
            arguments={},
            output={"val": 1},
            created_at=time.perf_counter() - 100.0,
            ttl_seconds=60.0,
            freshness_contract="strict",
        )
        self.assertFalse(entry.is_fresh())

    # 32. Negative Control E1: Parallelism disabled produces ~1.0x speedup
    async def test_32_negative_control_e1_null_speedup(self) -> None:
        harness = BenchmarkHarness(BenchmarkConfig(trials_per_condition=10, warmup_trials=1))
        controls = await harness.run_negative_controls(trials=10)
        e1_ctrl = next(c for c in controls if c["control"] == "E1_parallelism_disabled")
        self.assertTrue(e1_ctrl["passed_expected_null"])

    # 33. Negative Control E2: Fusion disabled produces ~1.0x speedup
    async def test_33_negative_control_e2_null_speedup(self) -> None:
        harness = BenchmarkHarness(BenchmarkConfig(trials_per_condition=10, warmup_trials=1))
        controls = await harness.run_negative_controls(trials=10)
        e2_ctrl = next(c for c in controls if c["control"] == "E2_fusion_disabled")
        self.assertTrue(e2_ctrl["passed_expected_null"])

    # 34. Negative Control E3: Speculation disabled produces ~1.0x speedup
    async def test_34_negative_control_e3_null_speedup(self) -> None:
        harness = BenchmarkHarness(BenchmarkConfig(trials_per_condition=10, warmup_trials=1))
        controls = await harness.run_negative_controls(trials=10)
        e3_ctrl = next(c for c in controls if c["control"] == "E3_speculation_disabled")
        self.assertTrue(e3_ctrl["passed_expected_null"])

    # 35. Negative Control E4: Early dispatch disabled produces ~1.0x speedup
    async def test_35_negative_control_e4_null_speedup(self) -> None:
        harness = BenchmarkHarness(BenchmarkConfig(trials_per_condition=10, warmup_trials=1))
        controls = await harness.run_negative_controls(trials=10)
        e4_ctrl = next(c for c in controls if c["control"] == "E4_early_dispatch_disabled")
        self.assertTrue(e4_ctrl["passed_expected_null"])

    # 36. Negative Control Cache: Cache disabled produces ~1.0x speedup
    async def test_36_negative_control_cache_null_speedup(self) -> None:
        harness = BenchmarkHarness(BenchmarkConfig(trials_per_condition=10, warmup_trials=1))
        controls = await harness.run_negative_controls(trials=10)
        c_ctrl = next(c for c in controls if c["control"] == "Cache_disabled")
        self.assertTrue(c_ctrl["passed_expected_null"])

    # 37. Positive Sensitivity Control: Harness confirms detection of latency reductions
    async def test_37_positive_sensitivity_control(self) -> None:
        harness = BenchmarkHarness(BenchmarkConfig(trials_per_condition=10, warmup_trials=1))
        controls = await harness.run_negative_controls(trials=10)
        pos_ctrl = next(c for c in controls if c["control"] == "Positive_sensitivity_injected_50pct_speedup")
        self.assertTrue(pos_ctrl["passed_expected_null"])

    # 38. Task validation fails when final answer is emitted after required tool failure
    def test_38_task_validate_rejects_hallucinated_answer_on_tool_failure(self) -> None:
        task = Task(task_id="t1", prompt="Fetch data", expected_output={"data": 42})
        failed_res = ToolResult(call_id="c1", name="query", error="DB connection refused", is_error=True)
        trace = ExecutionTrace(
            task_id="t1",
            tool_calls=[ToolCall(name="query", arguments={})],
            tool_results=[failed_res],
            success=True,
            final_output={"data": 42},
            start_ns=0,
            end_ns=1000,
        )
        valid = task.validate({"data": 42}, trace)
        self.assertFalse(valid)

    # 39. CLI Falsify: Exit code 2 on synthetic bundle
    def test_39_cli_falsify_exit_code_2_synthetic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "summary_report.json"
            p.write_text('{"evidence_level": "synthetic", "evaluations": []}', encoding="utf-8")
            code = cli.main(["falsify", "--input", str(p)])
            self.assertEqual(code, 2)

    # 40. CLI Falsify: Exit code 2 on smoke run with insufficient sample size
    def test_40_cli_falsify_exit_code_2_smoke_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "benchmark_result.json"
            p.write_text(
                '{"evidence_level": "replay_integration", "manifest": {"is_verdict_eligible": false, "trial_count": 50}, "evaluations": [{"workload_id": "W1", "verdict": {"passed": true}}]}',
                encoding="utf-8",
            )
            code = cli.main(["falsify", "--input", str(p)])
            self.assertEqual(code, 2)

    # 41. CLI Report: Writes report.md and report.html without re-running
    def test_41_cli_report_preserves_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "benchmark_result.json"
            p.write_text(
                '{"evidence_level": "replay_integration", "title": "Test Bundle", "evaluations": []}', encoding="utf-8"
            )
            code = cli.main(["report", "--input", str(p), "--out", tmpdir])
            self.assertEqual(code, 0)
            md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")
            self.assertIn("replay_integration", md)
            html = (Path(tmpdir) / "report.html").read_text(encoding="utf-8")
            self.assertIn("Test Bundle", html)

    # 42. CLI Validate-Bundle: Passes valid manifest and bundle
    def test_42_cli_validate_bundle_passes_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "benchmark_result.json"
            bundle = {
                "evidence_level": "replay_integration",
                "manifest": {
                    "code_git_sha": "abcd1234",
                    "evidence_level": "replay_integration",
                    "trial_count": 1000,
                    "benchmark_config_hash": "hash_cfg",
                    "workload_fixture_hash": "hash_fix",
                    "raw_trace_hash": "hash_raw",
                },
                "evaluations": [
                    {
                        "workload_id": "W1",
                        "summary": {
                            "candidate_p95_ms": 10.0,
                            "baseline_p95_ms": 20.0,
                            "p95_speedup": 2.0,
                            "candidate_success_rate": 1.0,
                            "unapproved_side_effects": 0,
                        },
                        "verdict": {"passed": True},
                    }
                ],
            }
            p.write_text(strict_json_dumps(bundle), encoding="utf-8")
            (Path(tmpdir) / "report.md").write_text("# Report", encoding="utf-8")
            (Path(tmpdir) / "report.html").write_text("<html></html>", encoding="utf-8")
            code = cli.main(["validate-bundle", "--input", str(p)])
            self.assertEqual(code, 0)

    # 43. Local wall-clock SQLite executes queries in threadpool without blocking event loop
    async def test_43_local_sqlite_threadpool_execution(self) -> None:
        sqlite_tool = AsyncSQLiteTool(db_path=":memory:", name="sqlite_test")
        sqlite_tool._sync_execute("CREATE TABLE test_table (id INT, val TEXT)", [])
        call = ToolCall(
            name="sqlite_test", arguments={"query": "INSERT INTO test_table VALUES (?, ?)", "params": [1, "alpha"]}
        )
        res = await sqlite_tool.execute(call)
        self.assertFalse(res.is_error)
        sqlite_tool.close()

    # 44. SafeSubprocessSandbox enforces timeout and terminates stuck child processes
    async def test_44_subprocess_sandbox_timeout(self) -> None:
        sandbox = SafeSubprocessSandbox(default_timeout_s=0.1)
        call = ToolCall(name="subprocess_sandbox", arguments={"command": "sleep 5", "timeout_s": 0.1})
        res = await sandbox.execute(call)
        self.assertTrue(res.is_error)
        self.assertIn("timed out", res.error or "")

    # 45. CompositeScheduler single dispatch ownership per turn
    async def test_45_composite_scheduler_single_dispatch_ownership(self) -> None:
        sched = CompositeScheduler()
        tools = ToolRegistry()
        t = SimpleMockTool("t1", output="ok")
        tools.register(t)

        task = Task(task_id="t_comp", prompt="Run", expected_output="ok")
        model = MockLLM(
            decisions=[
                LLMDecision(reasoning="Call", tool_calls=[ToolCall(name="t1", arguments={})]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer="ok"),
            ]
        )
        res = await sched.execute(task, model, tools)
        self.assertTrue(res.success)
        # Tool call dispatched exactly ONCE
        self.assertEqual(len(t.executions), 1)


if __name__ == "__main__":
    unittest.main()
