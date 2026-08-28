"""Guardrail tracking and metrics calculation for ToolSpeed evaluation."""

from __future__ import annotations

import json
import threading
from typing import Any

from toolspeed.core.types import (
    EventType,
    ExecutionTrace,
    GuardrailMetrics,
    TaskInstance,
    ToolCall,
)


class GuardrailTracker:
    """Computes guardrail metrics across execution traces and task instances."""

    def __init__(self) -> None:
        self._traces: list[ExecutionTrace] = []
        self._tasks: dict[str, TaskInstance] = {}
        self._active_concurrency_samples: list[int] = []
        self._lock = threading.Lock()

    def register_task(self, task: TaskInstance) -> None:
        """Register a task specification for ground-truth comparison."""
        with self._lock:
            self._tasks[task.task_id] = task

    def record_trace(self, trace: ExecutionTrace, task: TaskInstance | None = None) -> None:
        """Record an execution trace for guardrail evaluation."""
        with self._lock:
            self._traces.append(trace)
            if task is not None:
                self._tasks[task.task_id] = task

    def record_active_concurrency(self, active_count: int) -> None:
        """Record an observed active concurrency sample."""
        with self._lock:
            self._active_concurrency_samples.append(active_count)

    def calculate_metrics(self) -> GuardrailMetrics:
        """Calculate comprehensive guardrail metrics across all recorded traces."""
        with self._lock:
            traces = list(self._traces)
            tasks = dict(self._tasks)
            concurrency_samples = list(self._active_concurrency_samples)

        total_tasks = len(traces)
        if total_tasks == 0:
            return GuardrailMetrics()

        successful_tasks = 0
        total_tool_calls = 0
        unnecessary_calls = 0
        duplicated_calls = 0
        speculative_cancelled = 0
        speculative_wasted = 0
        speculative_committed = 0
        speculative_launched = 0
        cache_hits = 0
        cache_misses = 0
        cache_freshness_violations = 0
        unapproved_side_effects = 0
        unsafe_side_effects = 0
        blocked_unsafe_attempts = 0
        rate_limit_errors = 0
        rate_limit_failures = 0
        total_deopts = 0
        total_model_calls = 0
        total_model_input_tokens = 0
        total_model_output_tokens = 0
        total_cost_usd = 0.0

        tool_selection_scores: list[float] = []
        argument_accuracy_scores: list[float] = []

        for trace in traces:
            if trace.success:
                successful_tasks += 1

            task = tasks.get(trace.task_id)
            expected_tools = set(task.expected_tools) if task else None
            expected_args = task.expected_args if task else None

            # Track tool selection accuracy
            actual_tools = [call.name or call.tool_name for call in trace.tool_calls]
            actual_tools_set = set(actual_tools)

            if expected_tools is not None:
                if len(expected_tools) == 0:
                    tool_selection_scores.append(1.0 if not actual_tools else 0.0)
                else:
                    intersection = actual_tools_set.intersection(expected_tools)
                    union = actual_tools_set.union(expected_tools)
                    jaccard = len(intersection) / len(union) if union else 1.0
                    tool_selection_scores.append(jaccard)

            # Track argument accuracy if expected_args provided
            if expected_args is not None:
                matched_args = 0
                total_expected_args = len(expected_args)
                for call in trace.tool_calls:
                    call_name = call.name or call.tool_name
                    if call_name in expected_args:
                        target = expected_args[call_name]
                        # Exact schema comparison (no extra keys allowed)
                        if isinstance(target, dict):
                            if call.arguments == target:
                                matched_args += 1
                        elif call.arguments == target:
                            matched_args += 1
                arg_score = (matched_args / total_expected_args) if total_expected_args > 0 else 1.0
                argument_accuracy_scores.append(min(1.0, arg_score))

            # Track unnecessary and duplicated calls
            seen_calls: set[str] = set()
            for call in trace.tool_calls:
                call_name = call.name or call.tool_name
                total_tool_calls += 1
                if expected_tools is not None and expected_tools and call_name not in expected_tools:
                    unnecessary_calls += 1

                call_signature = f"{call_name}:{json.dumps(call.arguments, sort_keys=True)}"
                if call_signature in seen_calls:
                    duplicated_calls += 1
                else:
                    seen_calls.add(call_signature)

                if call.is_speculative:
                    speculative_launched += 1

                # Check unapproved side effects
                if call.requires_approval and not call.is_approved:
                    unapproved_side_effects += 1

            # Track events
            for event in trace.events:
                ev_type = str(event.event_type)
                if ev_type in (EventType.SPECULATION_CANCELLED.value, "speculative_cancel", "speculation_cancelled"):
                    speculative_cancelled += 1
                elif ev_type in (EventType.SPECULATION_HIT.value, "speculative_commit", "speculation_hit"):
                    speculative_committed += 1
                elif ev_type in (EventType.SPECULATION_START.value, "speculation_start"):
                    speculative_launched = max(speculative_launched, speculative_launched + 1)
                elif ev_type in (EventType.CACHE_HIT.value, "cache_hit"):
                    cache_hits += 1
                elif ev_type in (EventType.CACHE_MISS.value, "cache_miss"):
                    cache_misses += 1
                elif ev_type in (EventType.CACHE_FRESHNESS_VIOLATION.value, "cache_freshness_violation"):
                    cache_freshness_violations += 1
                elif ev_type in (EventType.RATE_LIMIT_ERROR.value, "rate_limit_error"):
                    rate_limit_errors += 1
                    rate_limit_failures += 1
                elif ev_type in (EventType.JIT_FUSION_DEOPT.value, "jit_fusion_deopt"):
                    total_deopts += 1
                elif ev_type in (EventType.APPROVAL_REJECTED.value, "approval_rejected", EventType.GUARDRAIL_VIOLATION.value, "guardrail_violation"):
                    blocked_unsafe_attempts += 1

            # Speculative wasted = speculative calls executed but not committed or hit
            for call in trace.tool_calls:
                if call.requires_approval and not call.is_approved:
                    unsafe_side_effects += 1
                if call.is_speculative and not call.metadata.get("committed", False) and not call.metadata.get("hit", False):
                    if not call.metadata.get("cancelled", False):
                        speculative_wasted += 1

            # Check cache freshness violations from tool results
            for res in trace.tool_results:
                if res.cached and res.metadata.get("is_stale", False):
                    cache_freshness_violations += 1
                if res.metadata.get("unsafe_executed", False):
                    unsafe_side_effects += 1
                total_cost_usd += res.cost_usd

            # Add token costs and usage
            total_cost_usd += trace.token_usage.cost_usd
            total_model_input_tokens += trace.token_usage.prompt_tokens
            total_model_output_tokens += trace.token_usage.completion_tokens
            total_model_calls += 1

        # Calculate peak concurrency
        peak_concurrency = self._compute_peak_concurrency_locked(traces, concurrency_samples)

        exact_success = successful_tasks / total_tasks if total_tasks > 0 else 0.0
        avg_tool_selection = (sum(tool_selection_scores) / len(tool_selection_scores)) if tool_selection_scores else 1.0
        avg_argument_acc = (sum(argument_accuracy_scores) / len(argument_accuracy_scores)) if argument_accuracy_scores else 1.0
        cost_per_task = total_cost_usd / total_tasks if total_tasks > 0 else 0.0

        return GuardrailMetrics(
            total_tasks=total_tasks,
            successful_tasks=successful_tasks,
            exact_success=exact_success,
            exact_accuracy=exact_success,
            tool_selection_accuracy=avg_tool_selection,
            argument_accuracy=avg_argument_acc,
            total_tool_calls=total_tool_calls,
            unnecessary_calls=unnecessary_calls,
            duplicated_calls=duplicated_calls,
            speculative_calls_launched=max(speculative_launched, speculative_committed + speculative_wasted + speculative_cancelled),
            speculative_calls_hit=speculative_committed,
            speculative_calls_wasted=speculative_wasted,
            speculative_calls_cancelled=speculative_cancelled,
            speculative_cancelled=speculative_cancelled,
            speculative_wasted=speculative_wasted,
            speculative_committed=speculative_committed,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_freshness_violations=cache_freshness_violations,
            unapproved_side_effects=unapproved_side_effects,
            unsafe_side_effects=unsafe_side_effects,
            blocked_unsafe_attempts=blocked_unsafe_attempts,
            rate_limit_failures=rate_limit_failures,
            rate_limit_errors=rate_limit_errors,
            peak_concurrency=peak_concurrency,
            total_model_input_tokens=total_model_input_tokens,
            total_model_output_tokens=total_model_output_tokens,
            total_model_calls=total_model_calls,
            total_deopts=total_deopts,
            cost_per_task_usd=cost_per_task,
            total_cost_usd=total_cost_usd,
        )

    def _compute_peak_concurrency_locked(self, traces: list[ExecutionTrace], concurrency_samples: list[int]) -> int:
        """Compute peak concurrency across all events and active samples."""
        max_c = max(concurrency_samples, default=1)

        intervals: list[tuple[int, int]] = []  # (timestamp_ns, +1/-1)
        for trace in traces:
            for ev in trace.events:
                ev_type = str(ev.event_type)
                if ev_type in (EventType.TOOL_START.value, EventType.SPECULATION_START.value, "tool_start", "speculation_start"):
                    intervals.append((ev.timestamp_ns, 1))
                elif ev_type in (EventType.TOOL_END.value, EventType.SPECULATION_CANCELLED.value, EventType.TOOL_CANCELLED.value, "tool_end", "speculation_cancelled", "tool_cancelled"):
                    intervals.append((ev.timestamp_ns, -1))

        intervals.sort(key=lambda x: (x[0], -x[1]))
        current = 0
        peak = max_c
        for _, change in intervals:
            current += change
            if current > peak:
                peak = current

        return max(1, peak)

    def reset(self) -> None:
        with self._lock:
            self._traces.clear()
            self._tasks.clear()
            self._active_concurrency_samples.clear()


class GuardrailMonitor:
    """Runtime monitor that tracks metrics during live scheduler execution."""

    def __init__(self) -> None:
        self.metrics = GuardrailMetrics()
        self._lock = threading.Lock()
        self._seen_tool_calls: set[str] = set()
        self._current_concurrency: int = 0

    def record_task_start(self) -> None:
        with self._lock:
            self.metrics.total_tasks += 1

    def record_task_finish(self, task: Any, final_answer: Any, success: bool) -> None:
        with self._lock:
            if success:
                self.metrics.successful_tasks += 1
            self.metrics.compute_derived()

    def record_tool_dispatch(
        self,
        tool_spec: Any,
        tool_call: ToolCall,
        is_speculative: bool = False,
    ) -> None:
        with self._lock:
            self.metrics.total_tool_calls += 1
            call_key = tool_call.key()

            if call_key in self._seen_tool_calls and not is_speculative:
                self.metrics.duplicated_calls += 1
            self._seen_tool_calls.add(call_key)

            if is_speculative:
                self.metrics.speculative_calls_launched += 1
                is_read_only = getattr(tool_spec, "is_read_only", not getattr(tool_spec, "side_effects", False))
                side_effects = getattr(tool_spec, "side_effects", not getattr(tool_spec, "is_read_only", True))
                if not is_read_only or side_effects:
                    self.metrics.unapproved_side_effects += 1
                    self.metrics.unsafe_side_effects += 1

    def record_speculation_resolved(
        self,
        hit: bool,
        cancelled: bool = False,
    ) -> None:
        with self._lock:
            if hit:
                self.metrics.speculative_calls_hit += 1
                self.metrics.speculative_committed += 1
            elif cancelled:
                self.metrics.speculative_calls_cancelled += 1
                self.metrics.speculative_cancelled += 1
            else:
                self.metrics.speculative_calls_wasted += 1
                self.metrics.speculative_wasted += 1

    def record_cache_event(self, hit: bool, is_fresh: bool = True) -> None:
        with self._lock:
            if hit:
                self.metrics.cache_hits += 1
                if not is_fresh:
                    self.metrics.cache_freshness_violations += 1
            else:
                self.metrics.cache_misses += 1

    def record_concurrency_enter(self) -> int:
        with self._lock:
            self._current_concurrency += 1
            if self._current_concurrency > self.metrics.peak_concurrency:
                self.metrics.peak_concurrency = self._current_concurrency
            return self._current_concurrency

    def record_concurrency_exit(self) -> None:
        with self._lock:
            self._current_concurrency = max(0, self._current_concurrency - 1)

    def record_rate_limit_failure(self) -> None:
        with self._lock:
            self.metrics.rate_limit_failures += 1
            self.metrics.rate_limit_errors += 1

    def record_model_usage(
        self, input_tokens: int = 0, output_tokens: int = 0
    ) -> None:
        with self._lock:
            self.metrics.total_model_calls += 1
            self.metrics.total_model_input_tokens += input_tokens
            self.metrics.total_model_output_tokens += output_tokens

    def record_deopt(self) -> None:
        with self._lock:
            self.metrics.total_deopts += 1

    def get_metrics(self) -> GuardrailMetrics:
        with self._lock:
            self.metrics.compute_derived()
            return self.metrics
