"""22 Mandatory Adversarial Tests for Scientific Integrity and Runtime Safety."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional
import unittest
import time

from toolspeed.adapters.base import (
    BaseLLMAdapter,
    BaseToolAdapter,
    LLMDecision,
    StreamingChunk,
    ToolRegistry,
    ToolSchema,
)
from toolspeed.adapters.live_tools import (
    AsyncHTTPClientTool,
    AsyncLocalFileIOTool,
    AsyncSQLiteTool,
    SafeSubprocessSandbox,
)
from toolspeed.core.guardrails import GuardrailTracker
from toolspeed.core.rate_limiter import AsyncConcurrencyLimiter, AsyncTokenBucket, RateLimiter
from toolspeed.core.types import (
    EventType,
    ExecutionTrace,
    Task,
    TaskInstance,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from toolspeed.schedulers.base import ExecutionContext, SchedulerConfig
from toolspeed.schedulers.b1_sync_react import SyncReActScheduler
from toolspeed.schedulers.b2_native_parallel import NativeParallelScheduler
from toolspeed.schedulers.e1_dag_scheduler import DAGScheduler, ToolDAG
from toolspeed.schedulers.e2_jit_fusion import (
    DeclarativeWorkflow,
    FusedKernel,
    JITFusionScheduler,
    WorkflowInvariant,
    WorkflowNode,
)
from toolspeed.schedulers.e3_speculation import SpeculativeReadScheduler
from toolspeed.schedulers.e4_commit_horizon import CommitHorizonScheduler, IncrementalCommitParser
from toolspeed.schedulers.e5_action_bytecode import ActionBytecodeCodec, ActionBytecodeScheduler
import toolspeed.benchmarks.harness
import toolspeed.schedulers.executor
import toolspeed.schedulers.phase2_cache
import toolspeed.visualization.report


class SimpleMockTool(BaseToolAdapter):
    def __init__(
        self,
        name: str,
        output: Any = "ok",
        is_read_only: bool = True,
        side_effects: bool = False,
        requires_approval: bool = False,
        latency_s: float = 0.01,
        raise_error: Optional[str] = None,
    ):
        self._name = name
        self.output = output
        self._is_read_only = is_read_only
        self._side_effects = side_effects
        self._requires_approval = requires_approval
        self.latency_s = latency_s
        self.raise_error = raise_error
        self.executions: List[ToolCall] = []

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self._name,
            description=f"Mock {self._name}",
            parameters={"type": "object", "properties": {"query": {"type": "string"}, "amount": {"type": "number"}}},
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
            return ToolResult(call_id=call.call_id, name=self._name, error=self.raise_error, is_error=True)
        return ToolResult(call_id=call.call_id, name=self._name, result=self.output, output=self.output, is_error=False)


class MockLLM(BaseLLMAdapter):
    def __init__(
        self,
        decisions: Optional[List[LLMDecision]] = None,
        draft_prediction: Optional[ToolCall] = None,
        predict_draft_raises: bool = False,
        chunks: Optional[List[StreamingChunk]] = None,
    ):
        self.decisions = list(decisions or [])
        self.draft_prediction = draft_prediction
        self.predict_draft_raises = predict_draft_raises
        self.chunks = list(chunks or [])
        self._turn = 0

    async def decide(self, task: Task, history: List[Dict[str, Any]], available_tools: List[ToolSpec]) -> LLMDecision:
        if self._turn < len(self.decisions):
            d = self.decisions[self._turn]
            self._turn += 1
            return d
        return LLMDecision(reasoning="Done", tool_calls=[], final_answer=task.expected_output or "done")

    async def predict_draft(self, task: Task, history: List[Dict[str, Any]], available_tools: List[ToolSpec]) -> Optional[ToolCall]:
        if self.predict_draft_raises:
            raise RuntimeError("Draft predictor failed")
        return self.draft_prediction

    async def stream_decision(self, task: Task, history: List[Dict[str, Any]], available_tools: List[ToolSpec]) -> AsyncIterator[StreamingChunk]:
        if self._turn == 0 and self.chunks:
            self._turn += 1
            for c in self.chunks:
                yield c
        else:
            final_ans = task.expected_output or "done"
            yield StreamingChunk(
                token_index=0,
                delta_text="done",
                is_final=True,
                metadata={"final_answer": final_ans},
            )


class TestAdversarialIntegrity(unittest.IsolatedAsyncioTestCase):
    """22 Comprehensive Adversarial and Scientific Integrity Tests."""

    # 1. DAG: Unknown reference fails closed
    async def test_unknown_reference_fails_closed(self):
        dag = ToolDAG()
        call = ToolCall(call_id="node1", name="tool_a", arguments={"input": "$non_existent_node.result"})
        dag.register_calls([call])
        self.assertEqual(dag.nodes["node1"].status, "failed")
        self.assertIn("Unknown dependency reference", dag.nodes["node1"].error or "")

    # 2. DAG: Nested reference resolution
    async def test_nested_reference_resolution(self):
        dag = ToolDAG()
        c1 = ToolCall(call_id="c1", name="parent_tool", arguments={})
        c2 = ToolCall(call_id="c2", name="child_tool", arguments={"nested": {"items": ["$c1.user_id", {"key": "$c1.org"}]}})
        dag.register_calls([c1, c2])

        # Simulate c1 completion
        dag.nodes["c1"].result = ToolResult(call_id="c1", name="parent_tool", output={"user_id": "u123", "org": "corp_a"})
        dag.nodes["c1"].status = "completed"

        resolved, err = dag.resolve_arguments(dag.nodes["c2"])
        self.assertIsNone(err)
        self.assertEqual(resolved["nested"]["items"][0], "u123")
        self.assertEqual(resolved["nested"]["items"][1]["key"], "corp_a")

    # 3. DAG: Direct & indirect cycle diagnostic
    async def test_dag_cycle_diagnostic(self):
        dag = ToolDAG()
        c1 = ToolCall(call_id="a", name="t1", arguments={"ref": "$b.out"})
        c2 = ToolCall(call_id="b", name="t2", arguments={"ref": "$a.out"})
        dag.register_calls([c1, c2])
        cycle = dag.detect_cycles()
        self.assertIsNotNone(cycle)
        self.assertIn("a", cycle)
        self.assertIn("b", cycle)

    # 4. DAG: Ambiguous reference rejected
    async def test_ambiguous_reference_rejected(self):
        dag = ToolDAG()
        c1 = ToolCall(call_id="call_1", name="lookup", arguments={})
        c2 = ToolCall(call_id="call_2", name="lookup", arguments={})
        c3 = ToolCall(call_id="call_3", name="aggregator", arguments={"data": "$lookup.val"})
        dag.register_calls([c1, c2, c3])
        self.assertEqual(dag.nodes["call_3"].status, "failed")
        self.assertIn("Ambiguous reference", dag.nodes["call_3"].error or "")

    # 5. DAG: Missing parent output field fails closed
    async def test_missing_parent_output_field_fails_closed(self):
        dag = ToolDAG()
        c1 = ToolCall(call_id="c1", name="parent", arguments={})
        c2 = ToolCall(call_id="c2", name="child", arguments={"target": "$c1.non_existent_key"})
        dag.register_calls([c1, c2])
        dag.nodes["c1"].result = ToolResult(call_id="c1", name="parent", output={"other_key": 42})
        dag.nodes["c1"].status = "completed"

        resolved, err = dag.resolve_arguments(dag.nodes["c2"])
        self.assertIsNotNone(err)
        self.assertIn("Missing output field 'non_existent_key'", err or "")

    # 6. DAG: Parent failure propagates downstream
    async def test_dag_parent_failure_propagation(self):
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

    # 7. JIT: Arbitrary untrusted callable rejected
    async def test_jit_arbitrary_callable_rejected(self):
        sched = JITFusionScheduler()
        tools = ToolRegistry()
        tools.register(SimpleMockTool("t1"))

        # Task providing untrusted arbitrary lambda
        task = Task(task_id="t_bad", prompt="eval", metadata={"declarative_workflow": "lambda x: 1/0"})
        model = MockLLM(decisions=[LLMDecision(reasoning="Done", tool_calls=[], final_answer="fallback_done")])
        res = await sched.execute(task, model, tools)
        self.assertEqual(res.final_answer, "fallback_done")

    # 8. JIT: Deopt does not repeat completed side effects
    async def test_jit_deopt_does_not_repeat_side_effects(self):
        sched = JITFusionScheduler()
        tools = ToolRegistry()
        t_write = SimpleMockTool("charge_card", output={"charged": 100}, is_read_only=False, side_effects=True)
        t_fail = SimpleMockTool("send_receipt", raise_error="SMTP server unreachable")
        tools.register(t_write)
        tools.register(t_fail)

        wf = DeclarativeWorkflow(
            workflow_id="order_charge",
            nodes=[
                WorkflowNode(step_id="s1", tool_name="charge_card", args_template={"amount": 100}, output_key="charge", is_side_effect=True),
                WorkflowNode(step_id="s2", tool_name="send_receipt", args_template={"receipt": "$charge.charged"}, output_key="receipt"),
            ],
        )
        task = Task(
            task_id="t_deopt",
            prompt="Charge",
            expected_output={"status": "handled_in_fallback"},
            metadata={"declarative_workflow": wf},
        )
        model = MockLLM(decisions=[LLMDecision(reasoning="Fallback handle", tool_calls=[], final_answer={"status": "handled_in_fallback"})])
        res = await sched.execute(task, model, tools)
        self.assertTrue(res.success)
        # Verify mutative charge_card was executed exactly ONCE
        self.assertEqual(len(t_write.executions), 1)

    # 9. Speculation: Concurrent draft prediction and model reasoning
    async def test_speculation_concurrent_model_reasoning(self):
        sched = SpeculativeReadScheduler(SchedulerConfig(speculation_enabled=True, speculation_confidence_threshold=0.5))
        tools = ToolRegistry()
        tools.register(SimpleMockTool("search", output={"val": 123}, latency_s=0.02))

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
        events = [e for e in res.events if e.event_type in (EventType.SPECULATION_HIT, "speculation_hit", EventType.SPECULATION_HIT.value)]
        self.assertGreaterEqual(len(events), 1)

    # 10. Speculation: Multi-call decision matching
    async def test_speculation_multi_call_matching(self):
        sched = SpeculativeReadScheduler(SchedulerConfig(speculation_enabled=True, speculation_confidence_threshold=0.5))
        tools = ToolRegistry()
        tools.register(SimpleMockTool("read_a", output={"a": 1}))
        tools.register(SimpleMockTool("read_b", output={"b": 2}))

        predicted = ToolCall(name="read_b", arguments={"id": 2}, speculation_confidence=0.9)
        # Model emits 2 calls: read_a (first) and read_b (matching prediction)
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
        # Both results must be recorded
        self.assertEqual(len(res.tool_results), 2)
        # cb was matched to speculative hit
        b_res = next(r for r in res.tool_results if r.call_id == "cb")
        self.assertEqual(b_res.output, {"b": 2})

    # 11. Speculation: Predictor error fallback
    async def test_speculation_predictor_failure_fallback(self):
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

    # 12. Speculation: Cancelled cleanup on miss
    async def test_speculation_cancelled_cleanup(self):
        sched = SpeculativeReadScheduler(SchedulerConfig(speculation_enabled=True, speculation_contention_mode="cancellable"))
        tools = ToolRegistry()
        t_slow = SimpleMockTool("slow_read", latency_s=0.5)
        tools.register(t_slow)
        tools.register(SimpleMockTool("actual_read", output="good"))

        predicted = ToolCall(name="slow_read", arguments={}, speculation_confidence=0.9)
        # Model chooses actual_read instead
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
        cancels = [e for e in res.events if e.event_type in (EventType.SPECULATION_CANCELLED, "speculative_cancel", EventType.SPECULATION_CANCELLED.value)]
        self.assertGreaterEqual(len(cancels), 1)

    # 13. Speculation: Mutative tools prohibited from speculation
    async def test_speculation_mutative_tool_prohibited(self):
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

    # 14. Commit Horizon: Syntax closure required
    def test_commit_horizon_syntax_closure(self):
        self.assertFalse(IncrementalCommitParser.is_syntax_closed('{"query": "open string'))
        self.assertFalse(IncrementalCommitParser.is_syntax_closed('{"count": 12,'))
        self.assertTrue(IncrementalCommitParser.is_syntax_closed('{"query": "closed string", "count": 12}'))

    # 15. Commit Horizon: Mutated arguments reconciled
    async def test_commit_horizon_mutated_args_reconciled(self):
        sched = CommitHorizonScheduler()
        tools = ToolRegistry()
        t_search = SimpleMockTool("search", output={"items": ["res"]})
        tools.register(t_search)

        early_call = ToolCall(call_id="c_early", name="search", arguments={"query": "preliminary"})
        final_call = ToolCall(call_id="c_early", name="search", arguments={"query": "authoritative_final"})

        chunks = [
            StreamingChunk(token_index=0, delta_text="call search", commit_horizon_ready=[early_call], raw_json_fragment='{"query": "preliminary"}'),
            StreamingChunk(token_index=1, delta_text="finished", is_final=True, parsed_tool_calls=[final_call]),
        ]
        model = MockLLM(
            chunks=chunks,
            decisions=[LLMDecision(reasoning="Done", tool_calls=[], final_answer="done")],
        )
        task = Task(task_id="t_ch_mut", prompt="Search", expected_output="done")
        res = await sched.execute(task, model, tools)
        self.assertTrue(res.success)
        # The final executed call must have received authoritative args
        self.assertEqual(t_search.executions[-1].arguments["query"], "authoritative_final")

    # 16. Commit Horizon: Side-effects cannot early dispatch
    async def test_commit_horizon_side_effect_safety(self):
        sched = CommitHorizonScheduler()
        tools = ToolRegistry()
        t_pay = SimpleMockTool("pay", output={"tx": "tx1"}, is_read_only=False, side_effects=True)
        tools.register(t_pay)

        call_pay = ToolCall(call_id="c_pay", name="pay", arguments={"amount": 100})
        chunks = [
            StreamingChunk(token_index=0, delta_text="pay", commit_horizon_ready=[call_pay], raw_json_fragment='{"amount": 100}'),
            StreamingChunk(token_index=1, delta_text="end", is_final=True, parsed_tool_calls=[]),
        ]
        model = MockLLM(chunks=chunks)
        task = Task(task_id="t_ch_side", prompt="Pay", expected_output="done")
        res = await sched.execute(task, model, tools)
        # Side effect was not in final decision parsed_tool_calls -> must never have been dispatched early!
        self.assertEqual(len(t_pay.executions), 0)

    # 17. Action Bytecode: 16-bit opcode with no collision at 256
    def test_action_bytecode_16bit_opcode_no_collision(self):
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

    # 18. Action Bytecode: Trailing bytes rejected
    def test_action_bytecode_trailing_bytes_rejected(self):
        codec = ActionBytecodeCodec()
        codec.register_tool("lookup", opcode=1)
        encoded = codec.encode(ToolCall(name="lookup", arguments={"key": "v"}))
        corrupted = encoded + b"EXTRA_CORRUPT_BYTES"
        with self.assertRaises(ValueError) as cm:
            codec.decode(corrupted)
        self.assertIn("unexpected trailing bytes", str(cm.exception))

    # 19. Rate Limiter: Cancellation safe without token loss
    async def test_rate_limiter_cancellation_safe(self):
        limiter = RateLimiter(rate_per_sec=10.0, burst_capacity=5.0, max_concurrency=1)
        # Fill concurrency slot
        await limiter.concurrency_limiter.acquire()

        async def _acquire_timeout():
            await limiter.acquire(tokens=2, timeout=0.02)

        with self.assertRaises((asyncio.TimeoutError, asyncio.CancelledError)):
            await _acquire_timeout()

        # Release the first slot
        limiter.concurrency_limiter.release()

        # Available tokens should still be intact
        self.assertAlmostEqual(limiter.token_bucket.available_tokens, 5.0, delta=0.5)

    # 20. Rate Limiter: Bounded semaphore prevents over-release
    def test_rate_limiter_bounded_semaphore(self):
        conc = AsyncConcurrencyLimiter(max_concurrency=2)
        conc.release()  # Calling release without acquire should not exceed bound
        conc.release()
        self.assertEqual(conc.active_count, 0)

    # 21. Guardrails: Extra unexpected arguments penalized
    def test_guardrails_extra_arguments_penalized(self):
        tracker = GuardrailTracker()
        task = TaskInstance(
            task_id="t1",
            workload_id="W1",
            expected_tools=["search"],
            expected_args={"search": {"query": "apple"}},
        )
        tracker.register_task(task)

        # Call with unexpected extra argument "extra_field"
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
        # Argument accuracy should be 0.0 because arguments did not match exactly
        self.assertEqual(metrics.argument_accuracy, 0.0)

    # 22. Zero unhandled leaked asyncio tasks across all schedulers
    async def test_no_unhandled_asyncio_task_leaks(self):
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
            chunks=[StreamingChunk(token_index=0, is_final=True, parsed_tool_calls=[ToolCall(name="t_ok", arguments={})])],
        )

        for sched in schedulers:
            res = await sched.execute(task, model, tools)
            self.assertTrue(res.success)

        # Allow event loop to tick
        await asyncio.sleep(0.01)
        # Check running tasks (excluding current test task)
        current = asyncio.current_task()
        pending = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
        self.assertEqual(len(pending), 0, f"Leaked tasks detected: {pending}")

    # 23. ToolExecutor: Schema validation rejects invalid argument types
    async def test_tool_executor_schema_validation_types(self):
        tools = ToolRegistry()
        tools.register(SimpleMockTool("query_tool"))
        executor = toolspeed.schedulers.executor.ToolExecutor(tools)
        call = ToolCall(name="query_tool", arguments={"amount": "not_a_number"})
        res = await executor.execute(call)
        self.assertTrue(res.is_error)
        self.assertIn("expected number", res.error)

    # 24. ToolExecutor: Unapproved mutative calls rejected
    async def test_tool_executor_approval_enforced(self):
        tools = ToolRegistry()
        tools.register(SimpleMockTool("transfer", is_read_only=False, side_effects=True, requires_approval=True))
        executor = toolspeed.schedulers.executor.ToolExecutor(tools)
        call = ToolCall(name="transfer", arguments={"amount": 100}, is_approved=False)
        res = await executor.execute(call)
        self.assertTrue(res.is_error)
        self.assertIn("requires explicit approval", res.error)

    # 25. Idempotency: Shared store prevents duplicate execution
    async def test_idempotency_store_replay(self):
        tools = ToolRegistry()
        t = SimpleMockTool("idempotent_write", is_read_only=False, side_effects=True)
        tools.register(t)
        store = toolspeed.schedulers.executor.SharedIdempotencyStore()
        executor = toolspeed.schedulers.executor.ToolExecutor(tools, idempotency_store=store)

        call1 = ToolCall(name="idempotent_write", arguments={"amount": 50}, idempotency_key="key_123")
        res1 = await executor.execute(call1)
        self.assertFalse(res1.is_error)
        self.assertEqual(len(t.executions), 1)

        # Second call with same idempotency key
        call2 = ToolCall(name="idempotent_write", arguments={"amount": 50}, idempotency_key="key_123")
        res2 = await executor.execute(call2)
        self.assertTrue(res2.cached)
        self.assertEqual(len(t.executions), 1)  # NOT executed again!

    # 26. Cache: Invalidation on mutation
    async def test_cache_invalidation_on_mutation(self):
        cache = toolspeed.schedulers.phase2_cache.ToolResultCache()
        cache.put("get_user", {"user_id": "u1"}, {"name": "Alice"})
        cached, hit, _ = cache.get("get_user", {"user_id": "u1"})
        self.assertTrue(hit)

        # Mutate user
        cache.invalidate_on_mutation("update_user")
        cached2, hit2, _ = cache.get("get_user", {"user_id": "u1"})
        self.assertFalse(hit2)

    # 27. Cache: Strict freshness contract rejects expired items
    def test_cache_freshness_contract(self):
        entry = toolspeed.schedulers.phase2_cache.CacheEntry(
            tool_name="get_data",
            arguments={},
            output={"val": 1},
            created_at=time.perf_counter() - 100.0,
            ttl_seconds=60.0,
            freshness_contract="strict",
        )
        self.assertFalse(entry.is_fresh())

    # 28. Negative Control E1: Disabled produces null speedup
    async def test_negative_control_e1_null_speedup(self):
        harness = toolspeed.benchmarks.harness.BenchmarkHarness()
        controls = await harness.run_negative_controls(trials=5)
        e1_ctrl = next(c for c in controls if c["control"] == "E1_disabled")
        self.assertTrue(e1_ctrl["passed_expected_null"])

    # 29. Negative Control E3: Disabled produces null speedup
    async def test_negative_control_e3_null_speedup(self):
        harness = toolspeed.benchmarks.harness.BenchmarkHarness()
        controls = await harness.run_negative_controls(trials=5)
        e3_ctrl = next(c for c in controls if c["control"] == "E3_disabled")
        self.assertTrue(e3_ctrl["passed_expected_null"])

    # 30. Negative Control E4: Disabled produces null speedup
    async def test_negative_control_e4_null_speedup(self):
        harness = toolspeed.benchmarks.harness.BenchmarkHarness()
        controls = await harness.run_negative_controls(trials=5)
        e4_ctrl = next(c for c in controls if c["control"] == "E4_disabled")
        self.assertTrue(e4_ctrl["passed_expected_null"])

    # 31. Benchmark Backend Spy Isolation
    async def test_benchmark_backend_spy_isolation(self):
        h_replay = toolspeed.benchmarks.harness.BenchmarkHarness(
            toolspeed.benchmarks.harness.BenchmarkConfig(evidence_level=toolspeed.core.types.EvidenceLevel.REPLAY_INTEGRATION)
        )
        self.assertEqual(h_replay.backend.evidence_level, toolspeed.core.types.EvidenceLevel.REPLAY_INTEGRATION)

        h_local = toolspeed.benchmarks.harness.BenchmarkHarness(
            toolspeed.benchmarks.harness.BenchmarkConfig(evidence_level=toolspeed.core.types.EvidenceLevel.LOCAL_WALL_CLOCK)
        )
        self.assertEqual(h_local.backend.evidence_level, toolspeed.core.types.EvidenceLevel.LOCAL_WALL_CLOCK)

    # 32. CLI Falsify Exit Codes
    def test_cli_falsify_exit_codes(self):
        import toolspeed.cli as cli
        # Synthetic bundle -> exit code 2 (inconclusive for real-world claims)
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "summary_report.json"
            p.write_text('{"evidence_level": "synthetic", "evaluations": []}', encoding="utf-8")
            code = cli.main(["falsify", "--input", str(p)])
            self.assertEqual(code, 2)

    # 33. CLI Report Preserves Provenance
    def test_cli_report_preserves_provenance(self):
        import toolspeed.cli as cli
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "benchmark_result.json"
            p.write_text('{"evidence_level": "replay_integration", "title": "Test Bundle", "evaluations": []}', encoding="utf-8")
            code = cli.main(["report", "--input", str(p), "--out", tmpdir])
            self.assertEqual(code, 0)
            md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")
            self.assertIn("replay_integration", md)

    # 34. CCL Excludes Failed Tasks
    def test_ccl_excludes_failed_tasks(self):
        import numpy as np
        from toolspeed.experiments.runner import compute_summary
        baseline = np.array([100.0, 100.0, 100.0])
        candidate = np.array([1.0, 1.0, 1.0])
        # Candidate failed all tasks
        summary = compute_summary(
            baseline=baseline,
            candidate=candidate,
            baseline_success=np.array([True, True, True]),
            candidate_success=np.array([False, False, False]),
        )
        self.assertIn(summary.candidate_p50_ms, (None, 0.0))
        self.assertEqual(summary.candidate_success_rate, 0.0)

    # 35. Paired Bootstrap CI Resampling
    def test_paired_bootstrap_ci_coverage(self):
        import numpy as np
        from toolspeed.experiments.runner import paired_bootstrap_p95_ci
        base = np.array([100.0] * 50)
        cand = np.array([50.0] * 50)
        low, high = paired_bootstrap_p95_ci(base, cand, num_samples=100)
        self.assertIsNotNone(low)
        self.assertIsNotNone(high)
        self.assertAlmostEqual(low, 50.0, delta=1.0)
        self.assertAlmostEqual(high, 50.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
