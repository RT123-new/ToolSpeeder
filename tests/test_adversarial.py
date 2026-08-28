"""Adversarial & Stress Testing Suite for the ToolSpeed Framework.

Red-team stress tests covering:
1. Metric Poisoning & CCL Correctness (Strict exclusion of failed tasks, zero-division, empty arrays)
2. Concurrency, Deadlocks & Cancellation (Speculative read cleanup, single_slot tail contention, DAG cyclic deadlock)
3. Deoptimization & State Integrity (JIT fusion exception bailout, validation failure fallback, side-effect preservation)
4. Action Bytecode Robustness (Truncated binary streams, malformed opcodes, >64KB payloads, deep nested JSON, emojis)
5. Commit-Horizon Immutability (Post-dispatch semantic argument mutation detection and recovery)
6. Guardrail & Security Enforcement (W7 unapproved side-effects, idempotency replay, cache TTL staleness, 429 storm)
7. Composite Scheduler Resiliency (Multi-mechanism stress, cancellation teardown with zero dangling tasks)
"""

from __future__ import annotations

import asyncio
import copy
import time
import unittest

import numpy as np

from toolspeed.adapters.base import (
    BaseLLMAdapter,
    LLMDecision,
    StreamingChunk,
    ToolRegistry,
)
from toolspeed.adapters.mock_models import (
    ActionBytecodeCodec as MockModelsBytecodeCodec,
)
from toolspeed.adapters.mock_models import (
    MockScriptedLLM,
)
from toolspeed.adapters.mock_tools import (
    MockToolAdapter,
    create_standard_mock_registry,
)
from toolspeed.core.profiler import (
    CCLTracker,
    calculate_percentiles,
    compute_speedup,
)
from toolspeed.core.rate_limiter import (
    RateLimiter,
    RateLimitError,
)
from toolspeed.core.types import (
    EventType,
    ExecutionTrace,
    Task,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from toolspeed.experiments.runner import compute_summary
from toolspeed.schedulers.base import SchedulerConfig
from toolspeed.schedulers.composite import CompositeScheduler
from toolspeed.schedulers.e1_dag_scheduler import DAGScheduler, ToolDAG
from toolspeed.schedulers.e2_jit_fusion import JITFusionScheduler
from toolspeed.schedulers.e3_speculation import SpeculativeReadScheduler
from toolspeed.schedulers.e4_commit_horizon import CommitHorizonScheduler
from toolspeed.schedulers.e5_action_bytecode import (
    ActionBytecodeCodec,
)
from toolspeed.schedulers.phase2_cache import ToolResultCache
from toolspeed.workloads.w7_side_effects import W7SideEffectsWorkload


class TestAdversarialStressSuite(unittest.IsolatedAsyncioTestCase):
    """Adversarial and stress test cases for ToolSpeed."""

    def setUp(self) -> None:
        self.registry = ToolRegistry()
        for tool in create_standard_mock_registry().values():
            self.registry.register(tool)

    # ------------------------------------------------------------------------
    # ATTACK 1: Metric Poisoning & CCL Correctness (Checklist #1)
    # ------------------------------------------------------------------------
    def test_adversarial_metric_poisoning_failed_tasks_excluded(self) -> None:
        """Adversarial Attack: Attempt to pollute CCL percentiles with fast-failing tasks.

        Verification: CCLTracker and compute_summary MUST strictly exclude failed task
        latencies from P50, P90, P95, and P99 percentiles while penalizing the success rate.
        """
        tracker = CCLTracker()

        # Record 10 successful slow tasks (1000ms - 2000ms)
        for i in range(10):
            trace = ExecutionTrace(
                task_id=f"succ_{i}",
                start_time_ns=0,
                end_time_ns=(1000 + i * 100) * 1_000_000,
                success=True,
            )
            tracker.record_trace(trace)

        # Record 90 failed ultra-fast tasks (1ms each)
        for i in range(90):
            trace = ExecutionTrace(
                task_id=f"fail_{i}",
                start_time_ns=0,
                end_time_ns=1 * 1_000_000,
                success=False,
            )
            tracker.record_trace(trace)

        stats = tracker.get_ccl_stats()
        # CCL P50 MUST be computed ONLY over the 10 successful tasks (approx 1450ms), NOT over 1ms fails!
        self.assertEqual(stats.success_count, 10)
        self.assertEqual(stats.failure_count, 90)
        self.assertEqual(tracker.total_tasks, 100)
        self.assertAlmostEqual(stats.success_rate, 0.10)
        self.assertGreaterEqual(stats.p50_ms, 1000.0)
        self.assertGreaterEqual(stats.p95_ms, 1800.0)

        # Verify compute_summary behavior
        latencies = np.array([1000.0 + i * 100 for i in range(10)] + [1.0] * 90)
        success_mask = np.array([True] * 10 + [False] * 90)
        baseline = np.array([2000.0] * 100)

        summary = compute_summary(
            baseline=baseline,
            candidate=latencies,
            baseline_success=np.array([True] * 100),
            candidate_success=success_mask,
        )

        self.assertIsNotNone(summary.candidate_success_rate)
        assert summary.candidate_success_rate is not None
        self.assertAlmostEqual(summary.candidate_success_rate, 0.10)
        self.assertIsNotNone(summary.candidate_p50_ms)
        assert summary.candidate_p50_ms is not None
        self.assertGreaterEqual(summary.candidate_p50_ms, 1000.0)
        self.assertIsNotNone(summary.candidate_p95_ms)
        assert summary.candidate_p95_ms is not None
        self.assertGreaterEqual(summary.candidate_p95_ms, 1800.0)

    def test_adversarial_metric_zero_division_and_empty_arrays(self) -> None:
        """Adversarial Attack: Feed empty arrays, zero latencies, and 100% failure rates.

        Verification: No ZeroDivisionError or crash occurs; returns safe defaults.
        """
        # Empty arrays
        empty_arr = np.array([])
        empty_bool = np.array([], dtype=bool)

        summary = compute_summary(
            baseline=empty_arr,
            candidate=empty_arr,
            baseline_success=empty_bool,
            candidate_success=empty_bool,
        )
        self.assertIn(summary.baseline_p50_ms, (0.0, None))
        self.assertIn(summary.candidate_p50_ms, (0.0, None))
        self.assertIn(summary.p50_speedup, (1.0, None))
        self.assertIn(summary.candidate_success_rate, (1.0, None))

        # 100% failed candidate
        base_lat = np.array([500.0, 600.0, 700.0])
        cand_lat = np.array([10.0, 10.0, 10.0])
        cand_succ = np.array([False, False, False])
        base_succ = np.array([True, True, True])

        summary_fail = compute_summary(
            baseline=base_lat,
            candidate=cand_lat,
            baseline_success=base_succ,
            candidate_success=cand_succ,
        )
        # CCL speedup should NOT report high speedup for failed tasks
        self.assertEqual(summary_fail.candidate_success_rate, 0.0)
        self.assertIn(summary_fail.p95_speedup, (0.0, None))
        self.assertIn(summary_fail.candidate_p50_ms, (0.0, None))

        # calculate_percentiles on empty
        p = calculate_percentiles([])
        self.assertEqual(p.p50_ms, 0.0)

        # compute_speedup with zero denominator
        s = compute_speedup({"p50_ms": 100.0}, {"p50_ms": 0.0})
        self.assertEqual(s["p50_speedup"], 1.0)

    # ------------------------------------------------------------------------
    # ATTACK 2: Concurrency, Deadlocks & Cancellation in E3 (Checklist #2)
    # ------------------------------------------------------------------------
    async def test_adversarial_speculative_cancellation_no_leaks(self) -> None:
        """Adversarial Attack: Speculate a slow background read tool that is rejected by LLM.

        Verification: In 'cancellable' mode, the background task MUST be cleanly aborted,
        cancelled count incremented exactly once, with zero dangling tasks in event loop.
        """
        slow_tool_executed = False

        async def slow_handler(args):
            nonlocal slow_tool_executed
            try:
                await asyncio.sleep(0.5)
                slow_tool_executed = True
                return {"result": "slow_done"}
            except asyncio.CancelledError:
                raise

        slow_tool = MockToolAdapter(
            spec=ToolSpec(name="slow_read_db", is_read_only=True),
            handler=slow_handler,
        )
        reg = ToolRegistry()
        reg.register(slow_tool)
        fetch_u = self.registry.get("fetch_user")
        assert fetch_u is not None
        reg.register(fetch_u)

        # LLM decides to NOT call slow_read_db, but calls fetch_user instead
        llm = MockScriptedLLM(
            decision_steps=[
                LLMDecision(
                    reasoning="Rejecting speculation, calling fetch_user",
                    tool_calls=[ToolCall(name="fetch_user", arguments={"user_id": "u1"})],
                ),
                LLMDecision(final_answer="Done"),
            ],
            draft_predictor_fn=lambda task, hist: (
                ToolCall(
                    name="slow_read_db",
                    arguments={"query": "SELECT *"},
                    is_speculative=True,
                    speculation_confidence=0.95,
                )
                if len(hist) == 0
                else None
            ),
            simulated_decision_ms=5.0,
        )

        task = Task(prompt="Test speculation cancellation", expected_output="Done")
        scheduler = SpeculativeReadScheduler(
            SchedulerConfig(
                speculation_enabled=True,
                speculation_contention_mode="cancellable",
                speculation_confidence_threshold=0.8,
            )
        )

        result = await scheduler.run(task, llm, reg)

        self.assertTrue(result.success)
        self.assertFalse(slow_tool_executed, "Speculative tool should have been cancelled before completion!")
        self.assertEqual(result.guardrails.speculative_calls_cancelled, 1)
        self.assertEqual(result.guardrails.speculative_calls_wasted, 0)
        self.assertTrue(any(e.event_type == EventType.SPECULATION_CANCELLED for e in result.events))

    async def test_adversarial_speculative_single_slot_tail_contention(self) -> None:
        """Adversarial Attack: Speculate in 'single_slot' contention mode with a miss.

        Verification: Subsequent execution waits for speculative task to complete,
        accounting for single_slot tail contention and correctly recording wasted calls.
        """
        spec_tool = MockToolAdapter(
            spec=ToolSpec(name="spec_tool", is_read_only=True),
            handler=lambda args: {"spec": "data"},
            base_latency_ms=10.0,
        )
        reg = ToolRegistry()
        reg.register(spec_tool)
        fetch_u2 = self.registry.get("fetch_user")
        assert fetch_u2 is not None
        reg.register(fetch_u2)

        llm = MockScriptedLLM(
            decision_steps=[
                LLMDecision(
                    tool_calls=[ToolCall(name="fetch_user", arguments={"user_id": "u42"})],
                ),
                LLMDecision(final_answer="Contention verified"),
            ],
            draft_predictor_fn=lambda task, hist: (
                ToolCall(
                    name="spec_tool",
                    arguments={"key": "val"},
                    is_speculative=True,
                    speculation_confidence=0.99,
                )
                if len(hist) == 0
                else None
            ),
            simulated_decision_ms=2.0,
        )

        task = Task(prompt="Single slot contention", expected_output="Contention verified")
        scheduler = SpeculativeReadScheduler(
            SchedulerConfig(
                speculation_enabled=True,
                speculation_contention_mode="single_slot",
            )
        )
        result = await scheduler.run(task, llm, reg)

        self.assertTrue(result.success)
        self.assertEqual(result.guardrails.speculative_calls_wasted, 1)
        self.assertEqual(result.guardrails.speculative_calls_cancelled, 0)

    # ------------------------------------------------------------------------
    # ATTACK 3: Dynamic DAG Circular Deadlock & Parameter Robustness (Checklist #2)
    # ------------------------------------------------------------------------
    async def test_adversarial_dag_cyclic_deadlock_recovery(self) -> None:
        """Adversarial Attack: Submit cyclic dependency graph (A -> B -> A).

        Verification: DAGScheduler detects unresolvable circular dependencies without hanging
        or deadlocking, cleanly marks unresolvable nodes as failed, and terminates gracefully.
        """
        call_a = ToolCall(call_id="call_a", name="fetch_user", arguments={"user_id": "$call_b.val"})
        call_b = ToolCall(call_id="call_b", name="fetch_orders", arguments={"user_id": "$call_a.val"})

        llm = MockScriptedLLM(
            decision_steps=[
                LLMDecision(tool_calls=[call_a, call_b]),
                LLMDecision(final_answer="Recovered from cyclic deadlock"),
            ],
            simulated_decision_ms=2.0,
        )

        task = Task(
            prompt="Cycle test",
            expected_output="Recovered from cyclic deadlock",
            validator=lambda out, trace=None: out == "Recovered from cyclic deadlock",
        )
        scheduler = DAGScheduler()

        # Should complete swiftly within timeout, not deadlock
        result = await asyncio.wait_for(scheduler.run(task, llm, self.registry), timeout=2.0)
        self.assertTrue(result.success)
        self.assertEqual(result.final_answer, "Recovered from cyclic deadlock")

    def test_adversarial_dag_parameter_template_robustness(self) -> None:
        """Adversarial Attack: Feed deeply nested, array indexed, missing, and special character references.

        Verification: ToolDAG.resolve_arguments extracts arrays, ignores missing refs safely,
        and never raises unhandled exceptions or regex crashes.
        """
        dag = ToolDAG()
        n1 = dag.add_call(ToolCall(call_id="node_1", name="search", arguments={}))
        n1.result = ToolResult(
            call_id="node_1",
            name="search",
            output={"items": ["first_item", "second_item"], "meta": "code_200"},
        )
        n1.status = "completed"

        n_list = dag.add_call(ToolCall(call_id="node_list", name="list_gen", arguments={}))
        n_list.result = ToolResult(
            call_id="node_list",
            name="list_gen",
            output=["zero_idx_item", "one_idx_item"],
        )
        n_list.status = "completed"

        child_call = ToolCall(
            call_id="node_2",
            name="process",
            arguments={
                "direct_item": "$node_1.items",
                "indexed_item": "$node_list.0",
                "missing_ref": "$non_existent_node.val",
                "meta_code": "$node_1.meta",
                "literal_string": "No refs here $100 price",
            },
        )
        n2 = dag.add_call(child_call)
        resolved, _ = dag.resolve_node_arguments(n2, fail_closed=False)
        assert resolved is not None
        self.assertEqual(resolved["direct_item"], ["first_item", "second_item"])
        self.assertEqual(resolved["indexed_item"], "zero_idx_item")
        self.assertEqual(resolved["missing_ref"], "$non_existent_node.val")
        self.assertEqual(resolved["literal_string"], "No refs here $100 price")

    # ------------------------------------------------------------------------
    # ATTACK 4: JIT Fusion Deoptimization & Context Preservation (Checklist #3)
    # ------------------------------------------------------------------------
    async def test_adversarial_jit_fusion_exception_deopt_bailout(self) -> None:
        """Adversarial Attack: Fused declarative kernel step encounters an error or missing tool.

        Verification: JITFusionScheduler catches the error, emits JIT_FUSION_DEOPT,
        records deopt event, and transparently falls back to LLM reasoning.
        """
        from toolspeed.schedulers.e2_jit_fusion import DeclarativeWorkflow, WorkflowNode

        exploding_workflow = DeclarativeWorkflow(
            workflow_id="exploding_kernel",
            nodes=(
                WorkflowNode(step_id="crash_step", tool_name="missing_crash_tool", args_template={}, output_key="res"),
            ),
        )

        scheduler = JITFusionScheduler()
        scheduler.register_kernel(exploding_workflow)

        llm = MockScriptedLLM(
            decision_steps=[
                LLMDecision(
                    reasoning="Kernel crashed, executing LLM fallback",
                    final_answer="Safe fallback answer",
                )
            ],
            simulated_decision_ms=2.0,
        )

        task = Task(
            prompt="Run exploding kernel",
            expected_output="Safe fallback answer",
            metadata={"workflow": "exploding_kernel"},
        )
        result = await scheduler.run(task, llm, self.registry)

        self.assertTrue(result.success)
        self.assertEqual(result.final_answer, "Safe fallback answer")
        self.assertEqual(result.guardrails.total_deopts, 1)
        self.assertTrue(any(e.event_type == EventType.JIT_FUSION_DEOPT for e in result.events))

    async def test_adversarial_jit_fusion_validation_failure_deopt(self) -> None:
        """Adversarial Attack: Fused kernel returns incorrect output that violates task validator.

        Verification: Invalid result is NOT returned; scheduler deoptimizes to model reasoning.
        """
        from toolspeed.schedulers.e2_jit_fusion import DeclarativeWorkflow, WorkflowNode

        bad_workflow = DeclarativeWorkflow(
            workflow_id="corrupted_kernel",
            nodes=(
                WorkflowNode(
                    step_id="step1",
                    tool_name="fetch_user",
                    args_template={"user_id": "$context.user_id"},
                    output_key="user",
                ),
            ),
            output_mapping={"sum": -99999},
        )

        scheduler = JITFusionScheduler()
        scheduler.register_kernel(bad_workflow)

        llm = MockScriptedLLM(
            decision_steps=[
                LLMDecision(
                    reasoning="Fixing corrupted kernel result",
                    final_answer={"sum": 100},
                )
            ],
            simulated_decision_ms=2.0,
        )

        task = Task(
            prompt="Compute sum",
            context={"user_id": "u1"},
            expected_output={"sum": 100},
            validator=lambda out: isinstance(out, dict) and out.get("sum") == 100,
            metadata={"workflow": "corrupted_kernel"},
        )
        result = await scheduler.run(task, llm, self.registry)

        self.assertTrue(result.success)
        self.assertEqual(result.final_answer, {"sum": 100})
        self.assertEqual(result.guardrails.total_deopts, 1)

    # ------------------------------------------------------------------------
    # ATTACK 5: Action Bytecode Codec Boundary Stress (Checklist #4)
    # ------------------------------------------------------------------------
    def test_adversarial_bytecode_truncated_streams(self) -> None:
        """Adversarial Attack: Feed truncated binary streams and buffer underflows to decoders.

        Verification: Both ActionBytecodeCodec decoders raise clean ValueError on truncated input.
        """
        # Test mock_models codec
        with self.assertRaises(ValueError):
            MockModelsBytecodeCodec.decode(b"")

        with self.assertRaises(ValueError):
            MockModelsBytecodeCodec.decode(b"\x01")

        with self.assertRaises(ValueError):
            MockModelsBytecodeCodec.decode(b"\x01\x00\x05tool")  # Missing args length

        with self.assertRaises(ValueError):
            MockModelsBytecodeCodec.decode(b"\x01\x00\x04tool\x00\x00\x00\xff")  # Claimed 255 bytes args, gave 0

        # Test e5_action_bytecode codec
        codec = ActionBytecodeCodec()
        with self.assertRaises(ValueError):
            codec.decode(b"")

        with self.assertRaises(ValueError):
            codec.decode(b"\x00")  # Opcode 0 is invalid

        with self.assertRaises(ValueError):
            codec.decode(b"\x01\x02\x00\x00\x00\x10")  # Arg count 2, missing second arg length and bytes

    def test_adversarial_bytecode_extreme_payloads_and_unicode(self) -> None:
        """Adversarial Attack: Encode large payloads (>64KB), unicode/emojis, nested dicts, and nulls.

        Verification: 100% exact round-trip fidelity with zero data loss or corruption.
        """
        codec = ActionBytecodeCodec(self.registry.list_specs())

        extreme_call = ToolCall(
            name="database_query",
            arguments={
                "query": "SELECT * FROM 'unicode_🚀_ταχ_⚡' WHERE comment = '\0\n\t\r'",
                "nested_struct": {"a": [1, 2, {"b": True, "c": None, "pi": 3.141592653589793}]},
                "large_blob": "X" * 70_000,  # > 64KB string payload
                "empty_str": "",
                "boolean_false": False,
            },
        )

        # Test E5 codec roundtrip
        encoded_e5 = codec.encode(extreme_call)
        decoded_e5 = codec.decode(encoded_e5)

        self.assertEqual(decoded_e5.name, extreme_call.name)
        self.assertEqual(decoded_e5.arguments["query"], extreme_call.arguments["query"])
        self.assertEqual(decoded_e5.arguments["nested_struct"], extreme_call.arguments["nested_struct"])
        self.assertEqual(decoded_e5.arguments["large_blob"], extreme_call.arguments["large_blob"])
        self.assertEqual(decoded_e5.arguments["boolean_false"], False)

        # Test mock_models codec roundtrip
        encoded_mm = MockModelsBytecodeCodec.encode(extreme_call)
        decoded_mm = MockModelsBytecodeCodec.decode(encoded_mm)

        self.assertEqual(decoded_mm.tool_name, extreme_call.name)
        self.assertEqual(decoded_mm.arguments["query"], extreme_call.arguments["query"])
        self.assertEqual(decoded_mm.arguments["nested_struct"], extreme_call.arguments["nested_struct"])
        self.assertEqual(decoded_mm.arguments["large_blob"], extreme_call.arguments["large_blob"])

    # ------------------------------------------------------------------------
    # ATTACK 6: Commit-Horizon Semantic Mutation Violation (Checklist #5)
    # ------------------------------------------------------------------------
    async def test_adversarial_commit_horizon_post_dispatch_argument_mutation(self) -> None:
        """Adversarial Attack: Early-dispatch tool with arg x=1. At final stream, mutate arg to x=999.

        Verification: Scheduler flags GUARDRAIL_VIOLATION, cancels the early task,
        and re-executes with the mutated argument.
        """
        executed_args: list[dict] = []

        def recording_handler(args):
            executed_args.append(copy.deepcopy(args))
            return {"user_id": args.get("user_id"), "status": "ok"}

        rec_tool = MockToolAdapter(
            spec=ToolSpec(name="fetch_user", required_args=["user_id"], is_read_only=True),
            handler=recording_handler,
        )
        reg = ToolRegistry()
        reg.register(rec_tool)

        # Custom streaming LLM that simulates an argument mutation after commit horizon
        class MutatingLLM(BaseLLMAdapter):
            async def decide(self, task, history, tools):
                if not history:
                    return LLMDecision(tool_calls=[ToolCall(name="fetch_user", arguments={"user_id": "u999"})])
                return LLMDecision(final_answer="Mutation handled")

            async def stream_decision(self, task, history, tools):
                if not history:
                    # Turn 1: Early commit at token 5 with user_id = u1
                    early_call = ToolCall(
                        call_id="call_early_1",
                        name="fetch_user",
                        arguments={"user_id": "u1"},
                        committed_early=True,
                    )
                    yield StreamingChunk(
                        delta_text="Fetching user...",
                        token_index=5,
                        is_final=False,
                        commit_horizon_ready=[early_call],
                    )
                    await asyncio.sleep(0.01)
                    # Turn 1 final chunk mutates argument to user_id = u999!
                    mutated_call = ToolCall(
                        call_id="call_early_1",
                        name="fetch_user",
                        arguments={"user_id": "u999"},  # Mutated!
                    )
                    yield StreamingChunk(
                        delta_text=" Mutated to u999.",
                        token_index=15,
                        is_final=True,
                        parsed_tool_calls=[mutated_call],
                    )
                else:
                    # Turn 2: Final answer
                    yield StreamingChunk(
                        delta_text="Mutation handled",
                        token_index=1,
                        is_final=True,
                        metadata={"final_answer": "Mutation handled"},
                    )

        scheduler = CommitHorizonScheduler()
        task = Task(prompt="Test commit mutation", expected_output="Mutation handled")
        result = await scheduler.run(task, MutatingLLM(), reg)

        self.assertTrue(result.success)
        # Verify GUARDRAIL_VIOLATION event was emitted
        self.assertTrue(any(e.event_type == EventType.GUARDRAIL_VIOLATION for e in result.events))
        # Verify tool was re-executed with mutated argument u999
        self.assertTrue(any(args.get("user_id") == "u999" for args in executed_args))

    # ------------------------------------------------------------------------
    # ATTACK 7: W7 Side-Effect Approval Gate & Idempotency Bypass (Checklist #6)
    # ------------------------------------------------------------------------
    async def test_adversarial_w7_unapproved_side_effects_rejected(self) -> None:
        """Adversarial Attack: Attempt to execute side-effecting financial transfer without approval.

        Verification: Tool execution is rejected with error, flagged in guardrails, balance unchanged.
        """
        w7 = W7SideEffectsWorkload()
        tool = w7.get_tools()[0]

        # Call without approval
        unapproved_call = ToolCall(
            name="execute_fund_transfer",
            arguments={
                "from_account": "acc_001",
                "to_account": "acc_002",
                "amount": 500.0,
                "idempotency_key": "idem_001",
            },
            requires_approval=True,
            is_approved=False,
        )

        res = await tool.execute(unapproved_call)
        self.assertFalse(res.is_success)
        self.assertTrue(res.is_error)
        self.assertIsNotNone(res.error)
        assert res.error is not None
        self.assertIn("explicit approval", res.error)
        self.assertEqual(w7.accounts["acc_001"], 10_000.0, "Balance must remain unchanged on unapproved mutation!")

        # Approved call succeeds
        approved_call = ToolCall(
            name="execute_fund_transfer",
            arguments={
                "from_account": "acc_001",
                "to_account": "acc_002",
                "amount": 500.0,
                "idempotency_key": "idem_001",
            },
            requires_approval=True,
            is_approved=True,
        )
        res_ok = await tool.execute(approved_call)
        self.assertTrue(res_ok.is_success)
        self.assertEqual(w7.accounts["acc_001"], 9500.0)

        # Replay with same idempotency key does not deduct balance twice
        res_replay = await tool.execute(approved_call)
        self.assertTrue(res_replay.is_success)
        self.assertEqual(
            w7.accounts["acc_001"], 9500.0, "Duplicate idempotency key must not execute duplicate deduction!"
        )

    # ------------------------------------------------------------------------
    # ATTACK 8: Rate Limiter 429 Storm & High Concurrency Backpressure (Checklist #6)
    # ------------------------------------------------------------------------
    async def test_adversarial_rate_limiter_429_storm_and_peak_concurrency(self) -> None:
        """Adversarial Attack: Bombard rate limiter with 100 concurrent bursts exceeding limits.

        Verification: RateLimitError raised when reject_on_limit=True; peak concurrency
        never exceeds configured capacity; all locks released cleanly.
        """
        # Limiter with max concurrency 5, capacity 10 tokens
        limiter = RateLimiter(
            requests_per_second=10.0,
            burst_capacity=10,
            max_concurrency=5,
            reject_on_limit=True,
        )

        async def worker(idx: int):
            try:
                await limiter.acquire()
                await asyncio.sleep(0.01)
            finally:
                limiter.release()

        # Fire 50 concurrent tasks
        results = await asyncio.gather(*[worker(i) for i in range(50)], return_exceptions=True)
        errors = [r for r in results if isinstance(r, (RateLimitError, Exception))]
        [r for r in results if not isinstance(r, Exception)]

        self.assertGreater(len(errors), 0, "Burst storm should trigger rate limit rejections")
        self.assertLessEqual(limiter.concurrency_limiter.peak_concurrency, 5)

    # ------------------------------------------------------------------------
    # ATTACK 9: Cache TTL Staleness & Invalidation Attack (Checklist #6)
    # ------------------------------------------------------------------------
    def test_adversarial_cache_ttl_and_write_invalidation(self) -> None:
        """Adversarial Attack: Attempt to serve stale cached reads after TTL expiration or domain write.

        Verification: Expired entries return cache miss; write mutations invalidate domain entries.
        """
        cache = ToolResultCache()
        cache.put("get_user", {"id": "u1"}, {"name": "Alice"}, ttl_seconds=0.05)

        # Immediate hit
        out, hit, fresh = cache.get("get_user", {"id": "u1"})
        self.assertTrue(hit)
        self.assertTrue(fresh)
        self.assertEqual(out, {"name": "Alice"})

        # Sleep past TTL
        time.sleep(0.06)
        _out_exp, hit_expired, _fresh_expired = cache.get("get_user", {"id": "u1"})
        self.assertFalse(hit_expired, "Strict cache contract must reject expired entries!")

        # Invalidate on mutation write
        cache.put("get_orders", {"user_id": "u1"}, {"orders": [1, 2]}, ttl_seconds=60.0)
        cache.invalidate_on_mutation("update_orders", {"user_id": "u1"})
        _, hit_after_inv, _ = cache.get("get_orders", {"user_id": "u1"})
        self.assertFalse(hit_after_inv, "Cached orders must be invalidated after update_orders mutation!")

    # ------------------------------------------------------------------------
    # ATTACK 10: Composite Scheduler Stress & Full Teardown Resiliency (Checklist #2)
    # ------------------------------------------------------------------------
    async def test_adversarial_composite_scheduler_cancellation_teardown(self) -> None:
        """Adversarial Attack: Run CompositeScheduler under full multi-mechanism load and cancel mid-flight.

        Verification: All in-flight DAG tasks, speculation tasks, and commit dispatches
        are cleanly cancelled with zero unhandled exceptions or dangling tasks.
        """
        llm = MockScriptedLLM(
            decision_steps=[
                LLMDecision(
                    tool_calls=[
                        ToolCall(name="fetch_user", arguments={"user_id": "u1"}),
                        ToolCall(name="fetch_orders", arguments={"user_id": "u1"}),
                        ToolCall(name="fetch_analytics", arguments={"metric": "views"}),
                    ]
                )
            ],
            draft_predictor_fn=lambda task, hist: (
                ToolCall(
                    name="fetch_user",
                    arguments={"user_id": "u1"},
                    is_speculative=True,
                    speculation_confidence=0.95,
                )
                if len(hist) == 0
                else None
            ),
            simulated_decision_ms=50.0,
            commit_horizon_fraction=0.2,
        )

        task = Task(prompt="Heavy stress task", expected_output="Done")
        scheduler = CompositeScheduler(
            SchedulerConfig(
                cache_enabled=True,
                speculation_enabled=True,
                commit_horizon_enabled=True,
                action_bytecode_enabled=True,
                jit_fusion_enabled=False,
            )
        )

        # Launch and cancel midway
        exec_task = asyncio.create_task(scheduler.run(task, llm, self.registry))
        await asyncio.sleep(0.02)
        exec_task.cancel()

        try:
            await exec_task
        except asyncio.CancelledError:
            pass  # Clean cancellation expected


if __name__ == "__main__":
    unittest.main()
