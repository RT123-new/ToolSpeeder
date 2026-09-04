"""Unit tests for ToolSpeed core types, profiler, guardrails, and rate limiter."""

import asyncio
import time
import unittest

from toolspeed.core.guardrails import GuardrailTracker
from toolspeed.core.profiler import (
    CCLTracker,
    NanosecondProfiler,
)
from toolspeed.core.rate_limiter import (
    AsyncConcurrencyLimiter,
    AsyncTokenBucket,
    RateLimiter,
    RateLimitError,
)
from toolspeed.core.types import (
    EventType,
    ExecutionEvent,
    ExecutionTrace,
    TaskInstance,
    TokenUsage,
    ToolCall,
    ToolResult,
    WorkloadSpec,
)


class TestCoreTypes(unittest.TestCase):
    """Test serialization, defaults, and methods of core data structures."""

    def test_token_usage(self):
        t1 = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150, cost_usd=0.001)
        t2 = TokenUsage(prompt_tokens=50, completion_tokens=25, total_tokens=75, cost_usd=0.0005)
        combined = t1.add(t2)
        self.assertEqual(combined.prompt_tokens, 150)
        self.assertEqual(combined.completion_tokens, 75)
        self.assertEqual(combined.total_tokens, 225)
        self.assertAlmostEqual(combined.cost_usd, 0.0015)

        d = t1.to_dict()
        restored = TokenUsage.from_dict(d)
        self.assertEqual(t1, restored)

    def test_tool_call_serialization(self):
        call = ToolCall(
            tool_name="search_db",
            arguments={"query": "SELECT 1", "limit": 10},
            is_speculative=True,
            commit_horizon=0.45,
            idempotency_key="idem_123",
            requires_approval=True,
            is_approved=False,
            bytecode=b"\x01\x02\x03",
            metadata={"priority": 1},
        )
        d = call.to_dict()
        restored = ToolCall.from_dict(d)
        self.assertEqual(call.tool_name, restored.tool_name)
        self.assertEqual(call.arguments, restored.arguments)
        self.assertEqual(call.call_id, restored.call_id)
        self.assertEqual(call.is_speculative, restored.is_speculative)
        self.assertEqual(call.commit_horizon, restored.commit_horizon)
        self.assertEqual(call.idempotency_key, restored.idempotency_key)
        self.assertEqual(call.requires_approval, restored.requires_approval)
        self.assertEqual(call.is_approved, restored.is_approved)

    def test_tool_result_serialization(self):
        res = ToolResult(
            call_id="call_999",
            tool_name="calculate",
            result={"value": 42},
            error=None,
            is_error=False,
            cached=True,
            cache_timestamp=1700000000.0,
            execution_time_ns=500_000,
            cost_usd=0.0001,
            metadata={"hit": True},
        )
        d = res.to_dict()
        restored = ToolResult.from_dict(d)
        self.assertEqual(res.call_id, restored.call_id)
        self.assertEqual(res.result, restored.result)
        self.assertEqual(res.execution_time_ns, restored.execution_time_ns)
        self.assertEqual(res.cost_usd, restored.cost_usd)
        self.assertTrue(restored.cached)

    def test_execution_trace(self):
        trace = ExecutionTrace(
            task_id="task_1",
            start_time_ns=1_000_000_000,
            end_time_ns=1_050_000_000,
            success=True,
            final_output={"answer": "ok"},
        )
        ev1 = ExecutionEvent(event_type=EventType.TASK_START, timestamp_ns=1_000_000_000, task_id="task_1")
        ev2 = ExecutionEvent(event_type=EventType.TASK_END, timestamp_ns=1_050_000_000, task_id="task_1")
        trace.events = [ev1, ev2]

        self.assertEqual(trace.duration_ns, 50_000_000)
        self.assertEqual(trace.duration_ms, 50.0)
        self.assertEqual(len(trace.get_events_by_type(EventType.TASK_START)), 1)

        d = trace.to_dict()
        restored = ExecutionTrace.from_dict(d)
        self.assertEqual(restored.task_id, "task_1")
        self.assertEqual(restored.duration_ms, 50.0)
        self.assertTrue(restored.success)

    def test_task_instance_and_specs(self):
        task = TaskInstance(
            task_id="t_001",
            workload_family="w1_independent",
            prompt="Compute sum",
            expected_tools=["tool_a"],
            expected_output={"sum": 10},
        )
        d = task.to_dict()
        restored = TaskInstance.from_dict(d)
        self.assertEqual(task.task_id, restored.task_id)
        self.assertEqual(task.expected_output, restored.expected_output)

        spec = WorkloadSpec(
            name="TestSpec",
            family="w1_independent",
            description="Testing spec",
            num_tasks=50,
        )
        spec_dict = spec.to_dict()
        restored_spec = WorkloadSpec.from_dict(spec_dict)
        self.assertEqual(spec.name, restored_spec.name)
        self.assertEqual(spec.num_tasks, 50)


class TestProfilerAndCCL(unittest.TestCase):
    """Test nanosecond profiler and CCL correctness calculation."""

    def test_nanosecond_profiler_spans_and_timeline(self):
        profiler = NanosecondProfiler(task_id="test_task")
        profiler.record_event(EventType.TASK_START)

        with profiler.span("model_reasoning", category="llm"):
            time.sleep(0.01)

        profiler.record_event(EventType.TASK_END)

        events = profiler.get_events()
        spans = profiler.get_spans()
        timeline = profiler.get_timeline()

        self.assertEqual(len(events), 2)
        self.assertEqual(len(spans), 1)
        self.assertGreater(spans[0].duration_ms, 5.0)
        self.assertEqual(len(timeline), 4)  # 2 events + 2 span bounds (start, end)
        # Ensure chronological ordering
        for i in range(len(timeline) - 1):
            self.assertLessEqual(timeline[i]["timestamp_ns"], timeline[i + 1]["timestamp_ns"])

    def test_ccl_tracker_excludes_failed_tasks(self):
        """CCL metric MUST only count successful tasks in percentile aggregation."""
        tracker = CCLTracker()

        # 5 successful tasks: durations 100ms, 200ms, 300ms, 400ms, 500ms (in ns)
        for d_ms in [100, 200, 300, 400, 500]:
            trace = ExecutionTrace(
                task_id=f"succ_{d_ms}",
                start_time_ns=0,
                end_time_ns=d_ms * 1_000_000,
                success=True,
            )
            tracker.record_trace(trace)

        # 5 failed tasks: durations 10ms, 10ms, 10ms, 10ms, 10ms (in ns)
        for idx in range(5):
            trace = ExecutionTrace(
                task_id=f"fail_{idx}",
                start_time_ns=0,
                end_time_ns=10 * 1_000_000,
                success=False,
            )
            tracker.record_trace(trace)

        self.assertEqual(tracker.total_tasks, 10)
        self.assertEqual(tracker.successful_tasks, 5)
        self.assertEqual(tracker.failed_tasks, 5)
        self.assertEqual(tracker.success_rate, 0.5)

        stats = tracker.get_ccl_stats()
        self.assertEqual(stats.count, 5)
        self.assertEqual(stats.success_count, 5)
        self.assertEqual(stats.failure_count, 5)
        self.assertEqual(stats.success_rate, 0.5)
        # CCL median should be 300.0 ms (middle of 100..500), NOT skewed by 10ms failures
        self.assertAlmostEqual(stats.p50_ms, 300.0)
        self.assertAlmostEqual(stats.min_ms, 100.0)
        self.assertAlmostEqual(stats.max_ms, 500.0)

    def test_ccl_tracker_zero_successes(self):
        tracker = CCLTracker()
        tracker.record_execution(duration_ns=100_000_000, success=False)
        stats = tracker.get_ccl_stats()
        self.assertEqual(stats.count, 0)
        self.assertEqual(stats.success_count, 0)
        self.assertEqual(stats.failure_count, 1)
        self.assertEqual(stats.success_rate, 0.0)
        self.assertEqual(stats.p50_ms, 0.0)


class TestGuardrails(unittest.TestCase):
    """Test guardrails tracker measuring accuracy, waste, side-effects, and concurrency."""

    def test_guardrails_calculations(self):
        tracker = GuardrailTracker()

        # Task 1: Success, correct tools, no waste
        task1 = TaskInstance(
            task_id="t1",
            workload_family="test",
            prompt="Test",
            expected_tools=["search_tool"],
            expected_args={"search_tool": {"q": "python"}},
        )
        tracker.register_task(task1)

        trace1 = ExecutionTrace(
            task_id="t1",
            start_time_ns=1000,
            end_time_ns=5000,
            success=True,
            tool_calls=[ToolCall(tool_name="search_tool", arguments={"q": "python"})],
            tool_results=[ToolResult(call_id="c1", tool_name="search_tool", result="res", cost_usd=0.002)],
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=20, cost_usd=0.001),
        )
        tracker.record_trace(trace1)

        # Task 2: Failure, unapproved side-effect, duplicated calls, speculative waste
        task2 = TaskInstance(
            task_id="t2",
            workload_family="test",
            prompt="Test 2",
            expected_tools=["write_tool"],
        )
        tracker.register_task(task2)

        call_unapproved = ToolCall(
            tool_name="delete_db",
            arguments={"id": "1"},
            requires_approval=True,
            is_approved=False,
        )
        call_dup1 = ToolCall(tool_name="write_tool", arguments={"text": "hi"})
        call_dup2 = ToolCall(tool_name="write_tool", arguments={"text": "hi"})
        call_spec_wasted = ToolCall(
            tool_name="spec_read",
            arguments={"k": "v"},
            is_speculative=True,
            metadata={"committed": False, "cancelled": False},
        )

        trace2 = ExecutionTrace(
            task_id="t2",
            start_time_ns=1000,
            end_time_ns=6000,
            success=False,
            tool_calls=[call_unapproved, call_dup1, call_dup2, call_spec_wasted],
            events=[
                ExecutionEvent(event_type=EventType.RATE_LIMIT_ERROR, timestamp_ns=2000, task_id="t2"),
                ExecutionEvent(event_type=EventType.SPECULATIVE_CANCEL, timestamp_ns=3000, task_id="t2"),
            ],
            tool_results=[
                ToolResult(call_id="c_stale", tool_name="cache_tool", cached=True, metadata={"is_stale": True})
            ],
            token_usage=TokenUsage(prompt_tokens=200, completion_tokens=50, cost_usd=0.002),
        )
        tracker.record_trace(trace2)

        metrics = tracker.calculate_metrics()
        self.assertEqual(metrics.total_tasks, 2)
        self.assertEqual(metrics.successful_tasks, 1)
        self.assertEqual(metrics.exact_success, 0.5)
        self.assertEqual(metrics.duplicated_calls, 1)
        self.assertEqual(metrics.unsafe_side_effects, 1)
        self.assertEqual(metrics.speculative_wasted, 1)
        self.assertEqual(metrics.speculative_cancelled, 1)
        self.assertEqual(metrics.cache_freshness_violations, 1)
        self.assertEqual(metrics.rate_limit_errors, 1)
        self.assertAlmostEqual(metrics.total_cost_usd, 0.005)


class TestRateLimiter(unittest.IsolatedAsyncioTestCase):
    """Test async token bucket and concurrency limiters."""

    async def test_token_bucket_acquire_and_429(self):
        bucket = AsyncTokenBucket(rate=1000.0, capacity=10.0, reject_on_limit=True)
        # Drain all tokens
        granted = bucket.try_acquire(tokens=10)
        self.assertTrue(granted)

        # Immediate next attempt with reject_on_limit should raise RateLimitError
        with self.assertRaises(RateLimitError):
            await bucket.acquire(tokens=1)

        self.assertGreater(bucket.total_429_errors, 0)

    async def test_concurrency_limiter(self):
        limiter = AsyncConcurrencyLimiter(max_concurrency=2)
        active_track = []

        async def worker():
            async with limiter:
                active_track.append(limiter.active_count)
                await asyncio.sleep(0.01)

        await asyncio.gather(worker(), worker(), worker(), worker())
        self.assertLessEqual(max(active_track), 2)
        self.assertLessEqual(limiter.peak_concurrency, 2)

    async def test_combined_rate_limiter(self):
        rate_limiter = RateLimiter(rate_per_sec=1000.0, burst_capacity=50.0, max_concurrency=5)
        async with rate_limiter:
            metrics = rate_limiter.get_metrics()
            self.assertEqual(metrics["active_concurrency"], 1)
            self.assertEqual(metrics["tokens_granted"], 1)


if __name__ == "__main__":
    unittest.main()
