"""Comprehensive Unit and Integration Tests for all ToolSpeed Schedulers."""

from __future__ import annotations

import asyncio
import time
import unittest

from toolspeed.adapters.base import LLMDecision, StreamingChunk, ToolRegistry
from toolspeed.adapters.mock_models import MockScriptedLLM
from toolspeed.adapters.mock_tools import (
    MockToolAdapter,
    create_standard_mock_registry,
)
from toolspeed.core.types import (
    EventType,
    GuardrailMetrics,
    Task,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from toolspeed.schedulers.base import SchedulerConfig
from toolspeed.schedulers.b1_sync_react import SyncReActScheduler
from toolspeed.schedulers.b2_native_parallel import NativeParallelScheduler
from toolspeed.schedulers.b4_oracle_dag import OracleDAGScheduler
from toolspeed.schedulers.b5_handwritten import HandwrittenWorkflowScheduler
from toolspeed.schedulers.e1_dag_scheduler import DAGScheduler, ToolDAG
from toolspeed.schedulers.e2_jit_fusion import FusedKernel, JITFusionScheduler
from toolspeed.schedulers.e3_speculation import SpeculativeReadScheduler
from toolspeed.schedulers.e4_commit_horizon import CommitHorizonScheduler
from toolspeed.schedulers.e5_action_bytecode import (
    ActionBytecodeCodec,
    ActionBytecodeScheduler,
)
from toolspeed.schedulers.phase2_cache import (
    CacheScheduler,
    ToolResultCache,
)
from toolspeed.schedulers.composite import CompositeScheduler


class TestToolSpeedSchedulers(unittest.TestCase):

    def setUp(self) -> None:
        self.registry = ToolRegistry()
        for tool in create_standard_mock_registry().values():
            self.registry.register(tool)

    # ------------------------------------------------------------------------
    # 1. Baseline 1: SyncReActScheduler Tests
    # ------------------------------------------------------------------------
    def test_b1_sync_react_single_turn(self) -> None:
        """Test B1 single tool call followed by final answer."""
        llm = MockScriptedLLM(
            decision_steps=[
                LLMDecision(
                    reasoning="Querying database for user",
                    tool_calls=[ToolCall(name="database_query", arguments={"query": "SELECT * FROM users"})],
                ),
                LLMDecision(
                    reasoning="User found",
                    final_answer={"status": "done", "user_found": True},
                ),
            ],
            simulated_decision_ms=5.0,
        )

        task = Task(
            prompt="Find user in database",
            expected_output={"status": "done", "user_found": True},
        )

        scheduler = SyncReActScheduler()
        result = scheduler.run_sync(task, llm, self.registry)

        self.assertTrue(result.success)
        self.assertEqual(result.final_answer, task.expected_output)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(len(result.tool_results), 1)
        self.assertFalse(result.tool_results[0].cached)
        self.assertGreater(result.ccl_ms, 0.0)

    def test_b1_sync_react_sequential_ordering(self) -> None:
        """Test B1 executes multiple tool calls sequentially."""
        llm = MockScriptedLLM(
            decision_steps=[
                LLMDecision(
                    tool_calls=[
                        ToolCall(name="fetch_user", arguments={"user_id": "u1"}),
                        ToolCall(name="fetch_orders", arguments={"user_id": "u1"}),
                    ]
                ),
                LLMDecision(final_answer="All user data fetched"),
            ],
            simulated_decision_ms=2.0,
        )

        task = Task(prompt="Fetch user and orders", expected_output="All user data fetched")
        scheduler = SyncReActScheduler()
        result = scheduler.run_sync(task, llm, self.registry)

        self.assertTrue(result.success)
        self.assertEqual(len(result.tool_results), 2)
        # Sequential execution: tool 2 must start after tool 1 completes
        t1 = result.tool_results[0]
        t2 = result.tool_results[1]
        self.assertGreaterEqual(t2.started_at, t1.finished_at - 0.001)

    # ------------------------------------------------------------------------
    # 2. Baseline 2: NativeParallelScheduler Tests
    # ------------------------------------------------------------------------
    def test_b2_native_parallel_concurrent_fanout(self) -> None:
        """Test B2 executes independent tool calls in a single turn concurrently."""
        llm = MockScriptedLLM(
            decision_steps=[
                LLMDecision(
                    tool_calls=[
                        ToolCall(name="fetch_user", arguments={"user_id": "u1"}),
                        ToolCall(name="fetch_orders", arguments={"user_id": "u1"}),
                    ]
                ),
                LLMDecision(final_answer="Fanout completed"),
            ],
            simulated_decision_ms=2.0,
        )

        task = Task(prompt="Fetch user and orders concurrently", expected_output="Fanout completed")
        scheduler = NativeParallelScheduler()
        result = scheduler.run_sync(task, llm, self.registry)

        self.assertTrue(result.success)
        self.assertEqual(len(result.tool_results), 2)
        # Peak concurrency should be at least 2
        self.assertGreaterEqual(result.guardrails.peak_concurrency, 2)

    # ------------------------------------------------------------------------
    # 3. Baseline 4: OracleDAGScheduler Tests
    # ------------------------------------------------------------------------
    def test_b4_oracle_dag_waves(self) -> None:
        """Test B4 with an explicit oracle wave plan."""
        task = Task(
            prompt="Oracle pipeline",
            expected_output={"final": "completed"},
            metadata={
                "oracle_plan": [
                    [ToolCall(name="fetch_user", arguments={"user_id": "u1"})],
                    [ToolCall(name="fetch_orders", arguments={"user_id": "$fetch_user.user_id"})],
                ],
                "oracle_final_answer_fn": lambda outputs: {"final": "completed"},
            },
        )

        llm = MockScriptedLLM()
        scheduler = OracleDAGScheduler()
        result = scheduler.run_sync(task, llm, self.registry)

        self.assertTrue(result.success)
        self.assertEqual(result.final_answer, {"final": "completed"})
        self.assertEqual(len(result.tool_results), 2)

    # ------------------------------------------------------------------------
    # 4. Baseline 5: HandwrittenWorkflowScheduler Tests
    # ------------------------------------------------------------------------
    def test_b5_handwritten_zero_model_latency(self) -> None:
        """Test B5 executes without any LLM model round-trips."""
        task = Task(
            prompt="Run compiled user fetch",
            context={"user_id": "42"},
            expected_output={"user": {"user_id": "42", "name": "User_42", "tier": "gold"}, "orders": {"orders": [{"order_id": "ord_42_1", "total": 99.5}]}, "status": "compiled_complete"},
        )

        llm = MockScriptedLLM()
        scheduler = HandwrittenWorkflowScheduler()
        result = scheduler.run_sync(task, llm, self.registry)

        self.assertTrue(result.success)
        # Model tokens should be 0
        self.assertEqual(result.guardrails.total_model_calls, 0)
        self.assertEqual(result.guardrails.total_model_input_tokens, 0)

    # ------------------------------------------------------------------------
    # 5. Experiment E1: DAGScheduler Tests
    # ------------------------------------------------------------------------
    def test_e1_dag_dynamic_parameter_binding(self) -> None:
        """Test E1 binds output of parent tool to child tool argument dynamically."""
        call1 = ToolCall(call_id="c1", name="fetch_user", arguments={"user_id": "777"})
        call2 = ToolCall(call_id="c2", name="fetch_orders", arguments={"user_id": "$c1.user_id"})

        llm = MockScriptedLLM(
            decision_steps=[
                LLMDecision(tool_calls=[call1, call2]),
                LLMDecision(final_answer="Orders fetched for user 777"),
            ],
            simulated_decision_ms=2.0,
        )

        task = Task(
            prompt="Get user 777 orders",
            expected_output="Orders fetched for user 777",
        )

        scheduler = DAGScheduler()
        result = scheduler.run_sync(task, llm, self.registry)

        self.assertTrue(result.success)
        self.assertEqual(len(result.tool_results), 2)
        # Verify call2 received resolved argument
        self.assertEqual(call2.arguments["user_id"], "777")

    # ------------------------------------------------------------------------
    # 6. Experiment E2: JITFusionScheduler Tests
    # ------------------------------------------------------------------------
    def test_e2_jit_fusion_success(self) -> None:
        """Test E2 executes recognized sub-plan locally with fused kernel."""
        task = Task(
            prompt="Fetch user profile and orders for user 123",
            context={"user_id": "123"},
            expected_output={"user": {"user_id": "123", "name": "User_123", "tier": "gold"}, "orders": {"orders": [{"order_id": "ord_123_1", "total": 99.5}]}, "fused": True},
        )

        llm = MockScriptedLLM()
        scheduler = JITFusionScheduler()
        result = scheduler.run_sync(task, llm, self.registry)

        self.assertTrue(result.success)
        # Fused kernel should bypass initial model turns
        has_fusion_event = any(e.event_type == EventType.JIT_FUSION_SUCCESS for e in result.events)
        self.assertTrue(has_fusion_event)

    def test_e2_jit_fusion_deopt_fallback(self) -> None:
        """Test E2 deoptimizes cleanly to model reasoning when kernel invariant fails."""
        # Create a tool that fails or returns error
        error_tool = MockToolAdapter(
            spec=ToolSpec(name="fetch_user", required_args=["user_id"], is_read_only=True),
            handler=lambda args: {"error": "User not found"},
        )
        custom_registry = ToolRegistry()
        custom_registry.register(error_tool)
        custom_registry.register(self.registry.get("fetch_orders"))

        llm = MockScriptedLLM(
            decision_steps=[
                LLMDecision(
                    reasoning="User not found in fast path, handling missing user edge case",
                    final_answer="Handled missing user fallback",
                )
            ],
            simulated_decision_ms=2.0,
        )

        task = Task(
            prompt="Fetch user profile and orders for user 999",
            context={"user_id": "999"},
            expected_output="Handled missing user fallback",
        )

        scheduler = JITFusionScheduler()
        result = scheduler.run_sync(task, llm, custom_registry)

        self.assertTrue(result.success)
        self.assertEqual(result.final_answer, "Handled missing user fallback")
        self.assertEqual(result.guardrails.total_deopts, 1)

    # ------------------------------------------------------------------------
    # 7. Experiment E3: SpeculativeReadScheduler Tests
    # ------------------------------------------------------------------------
    def test_e3_speculation_hit(self) -> None:
        """Test E3 speculative hit when draft predictor matches model decision."""
        llm = MockScriptedLLM(
            decision_steps=[
                LLMDecision(
                    tool_calls=[ToolCall(name="web_search", arguments={"query": "python performance"})]
                ),
                LLMDecision(final_answer="Search results analyzed"),
            ],
            draft_predictor_fn=lambda task, hist: ToolCall(
                name="web_search",
                arguments={"query": "python performance"},
                is_speculative=True,
                speculation_confidence=0.95,
            ) if len(hist) == 0 else None,
            simulated_decision_ms=30.0,
        )

        task = Task(
            prompt="web_search python performance",
            expected_output="Search results analyzed",
        )

        scheduler = SpeculativeReadScheduler(
            SchedulerConfig(speculation_enabled=True, speculation_confidence_threshold=0.70)
        )
        result = scheduler.run_sync(task, llm, self.registry)

        self.assertTrue(result.success)
        self.assertEqual(result.guardrails.speculative_calls_hit, 1)
        self.assertEqual(result.guardrails.speculative_calls_wasted, 0)

    def test_e3_speculation_cancellation_on_mismatch(self) -> None:
        """Test E3 cancels running speculative call if main model chooses different tool."""
        llm = MockScriptedLLM(
            decision_steps=[
                LLMDecision(
                    tool_calls=[ToolCall(name="fetch_user", arguments={"user_id": "abc"})]
                ),
                LLMDecision(final_answer="Done"),
            ],
            draft_predictor_fn=lambda task, hist: ToolCall(
                name="web_search",
                arguments={"query": "wrong speculative query"},
                is_speculative=True,
                speculation_confidence=0.90,
            ),
            simulated_decision_ms=10.0,
        )

        task = Task(prompt="Fetch user abc", expected_output="Done")
        scheduler = SpeculativeReadScheduler(
            SchedulerConfig(
                speculation_enabled=True,
                speculation_contention_mode="cancellable",
                speculation_confidence_threshold=0.70,
            )
        )
        result = scheduler.run_sync(task, llm, self.registry)

        self.assertTrue(result.success)
        self.assertGreaterEqual(
            result.guardrails.speculative_calls_cancelled + result.guardrails.speculative_calls_wasted,
            1,
        )

    def test_e3_speculation_confidence_gating(self) -> None:
        """Test E3 does not speculate if predictor confidence is below threshold."""
        llm = MockScriptedLLM(
            decision_steps=[
                LLMDecision(tool_calls=[ToolCall(name="web_search", arguments={"query": "test"})]),
                LLMDecision(final_answer="Done"),
            ],
            draft_predictor_fn=lambda task, hist: ToolCall(
                name="web_search",
                arguments={"query": "test"},
                is_speculative=True,
                speculation_confidence=0.40,  # Below 0.70 threshold
            ),
            simulated_decision_ms=10.0,
        )

        task = Task(prompt="Test confidence", expected_output="Done")
        scheduler = SpeculativeReadScheduler(
            SchedulerConfig(speculation_enabled=True, speculation_confidence_threshold=0.70)
        )
        result = scheduler.run_sync(task, llm, self.registry)

        self.assertTrue(result.success)
        # Speculation was gated out
        self.assertEqual(result.guardrails.speculative_calls_launched, 0)

    # ------------------------------------------------------------------------
    # 8. Experiment E4: CommitHorizonScheduler Tests
    # ------------------------------------------------------------------------
    def test_e4_commit_horizon_early_dispatch(self) -> None:
        """Test E4 dispatches tool call at commit horizon before streaming finishes."""
        target_call = ToolCall(name="database_query", arguments={"query": "SELECT count(*) FROM items"})
        llm = MockScriptedLLM(
            decision_steps=[
                LLMDecision(tool_calls=[target_call]),
                LLMDecision(final_answer="Count retrieved"),
            ],
            simulated_decision_ms=40.0,
            commit_horizon_fraction=0.3,
            token_chunk_count=6,
        )

        task = Task(prompt="Count items", expected_output="Count retrieved")
        scheduler = CommitHorizonScheduler()
        result = scheduler.run_sync(task, llm, self.registry)

        self.assertTrue(result.success)
        has_commit_event = any(e.event_type == EventType.COMMIT_HORIZON_REACHED for e in result.events)
        self.assertTrue(has_commit_event)

    # ------------------------------------------------------------------------
    # 9. Experiment E5: ActionBytecodeCodec & Scheduler Tests
    # ------------------------------------------------------------------------
    def test_e5_bytecode_codec_encode_decode(self) -> None:
        """Test Action Bytecode compression and lossless roundtrip decoding."""
        codec = ActionBytecodeCodec(self.registry.list_specs())
        original = ToolCall(name="database_query", arguments={"query": "SELECT * FROM sales WHERE year = 2026"})
        encoded = codec.encode(original)
        decoded = codec.decode(encoded)

        self.assertEqual(decoded.name, original.name)
        self.assertEqual(decoded.arguments["query"], original.arguments["query"])

        json_len, bc_len, ratio = codec.calculate_compression_ratio(original)
        self.assertGreater(ratio, 1.5)  # Significant compression

    def test_e5_action_bytecode_scheduler(self) -> None:
        """Test ActionBytecodeScheduler execution with token compression."""
        llm = MockScriptedLLM(
            decision_steps=[
                LLMDecision(tool_calls=[ToolCall(name="fetch_user", arguments={"user_id": "u99"})]),
                LLMDecision(final_answer="Bytecode executed"),
            ],
            simulated_decision_ms=5.0,
        )

        task = Task(prompt="Fetch user u99 with bytecode", expected_output="Bytecode executed")
        scheduler = ActionBytecodeScheduler()
        result = scheduler.run_sync(task, llm, self.registry)

        self.assertTrue(result.success)
        has_bytecode_event = any(e.event_type == EventType.BYTECODE_ENCODE for e in result.events)
        self.assertTrue(has_bytecode_event)

    # ------------------------------------------------------------------------
    # 10. Phase 2: ToolResultCache & CacheScheduler Tests
    # ------------------------------------------------------------------------
    def test_phase2_cache_exact_and_semantic_hit(self) -> None:
        """Test exact and normalized semantic cache hits."""
        cache = ToolResultCache(default_ttl_seconds=10.0)
        cache.put("web_search", {"query": "apple inc revenue"}, {"revenue": 383e9})

        # Exact match
        out1, hit1, fresh1 = cache.get("web_search", {"query": "apple inc revenue"})
        self.assertTrue(hit1)
        self.assertTrue(fresh1)
        self.assertEqual(out1["revenue"], 383e9)

        # Semantic match (different spacing / case)
        out2, hit2, fresh2 = cache.get("web_search", {"query": "  Apple  INC   Revenue  "})
        self.assertTrue(hit2)
        self.assertTrue(fresh2)

    def test_phase2_cache_mutation_invalidation(self) -> None:
        """Test that executing a mutation tool invalidates related cached reads."""
        cache = ToolResultCache(default_ttl_seconds=60.0)
        cache.put("execute_payment", {"order_id": "ord_1", "amount": 50}, {"status": "paid"})
        
        # Invalidate on write
        invalidated = cache.invalidate_tool("execute_payment")
        self.assertGreaterEqual(invalidated, 1)

        out, hit, _ = cache.get("execute_payment", {"order_id": "ord_1", "amount": 50})
        self.assertFalse(hit)

    def test_phase2_cache_scheduler(self) -> None:
        """Test CacheScheduler serves second identical query from cache."""
        llm = MockScriptedLLM(
            decision_steps=[
                LLMDecision(tool_calls=[ToolCall(name="fetch_user", arguments={"user_id": "cache_me"})]),
                LLMDecision(tool_calls=[ToolCall(name="fetch_user", arguments={"user_id": "cache_me"})]),
                LLMDecision(final_answer="Done with cache"),
            ],
            simulated_decision_ms=2.0,
        )

        task = Task(prompt="Cached fetch", expected_output="Done with cache")
        scheduler = CacheScheduler()
        result = scheduler.run_sync(task, llm, self.registry)

        self.assertTrue(result.success)
        self.assertEqual(result.guardrails.cache_hits, 1)
        self.assertTrue(result.tool_results[1].cached)

    # ------------------------------------------------------------------------
    # 11. Composite Scheduler Test
    # ------------------------------------------------------------------------
    def test_composite_scheduler_end_to_end(self) -> None:
        """Test CompositeScheduler orchestrating caching, speculation, streaming commit, and DAG execution."""
        llm = MockScriptedLLM(
            decision_steps=[
                LLMDecision(
                    tool_calls=[
                        ToolCall(name="fetch_user", arguments={"user_id": "comp_1"}),
                        ToolCall(name="fetch_orders", arguments={"user_id": "comp_1"}),
                    ]
                ),
                LLMDecision(final_answer="Composite test successful"),
            ],
            draft_predictor_fn=lambda task, hist: ToolCall(
                name="fetch_user",
                arguments={"user_id": "comp_1"},
                is_speculative=True,
                speculation_confidence=0.90,
            ) if len(hist) == 0 else None,
            simulated_decision_ms=20.0,
            commit_horizon_fraction=0.3,
        )

        task = Task(
            prompt="Run full composite pipeline for user comp_1",
            expected_output="Composite test successful",
        )

        scheduler = CompositeScheduler(
            SchedulerConfig(
                cache_enabled=True,
                speculation_enabled=True,
                commit_horizon_enabled=True,
                action_bytecode_enabled=True,
                jit_fusion_enabled=False,  # Test dynamic composite path
            )
        )
        result = scheduler.run_sync(task, llm, self.registry)

        self.assertTrue(result.success)
        self.assertEqual(result.final_answer, "Composite test successful")
        self.assertGreaterEqual(result.guardrails.total_tool_calls, 2)


if __name__ == "__main__":
    unittest.main()
