"""Centralized ToolExecutor owning validation, safety, rate-limiting, and lifecycle management."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, Set
import time

from toolspeed.adapters.base import BaseToolAdapter, ToolRegistry
from toolspeed.core.guardrails import GuardrailMonitor
from toolspeed.core.profiler import LatencyProfiler
from toolspeed.core.rate_limiter import RateLimitError, RateLimiter
from toolspeed.core.types import (
    EventType,
    ExecutionEvent,
    ToolCall,
    ToolResult,
    ToolSpec,
)


class ToolExecutor:
    """Central execution engine for dispatching tools across all schedulers."""

    def __init__(
        self,
        registry: ToolRegistry,
        rate_limiter: Optional[RateLimiter] = None,
        profiler: Optional[LatencyProfiler] = None,
        guardrails: Optional[GuardrailMonitor] = None,
        default_timeout_s: float = 30.0,
    ):
        self.registry = registry
        self.rate_limiter = rate_limiter or RateLimiter()
        self.profiler = profiler or LatencyProfiler()
        self.guardrails = guardrails or GuardrailMonitor()
        self.default_timeout_s = default_timeout_s
        self._idempotency_cache: Dict[str, ToolResult] = {}
        self._active_calls: Set[str] = set()

    async def execute(
        self,
        call: ToolCall,
        is_speculative: bool = False,
        is_early_dispatched: bool = False,
        timeout_s: Optional[float] = None,
    ) -> ToolResult:
        """Execute a tool call with full lifecycle instrumentation, validation, and safety."""
        start_ns = time.perf_counter_ns()
        start_wall = time.perf_counter()
        call_id = call.call_id
        tool_name = call.name or call.tool_name
        span_prefix = "spec_" if is_speculative else ("early_" if is_early_dispatched else "")
        span_name = f"{span_prefix}tool_{call_id}"

        # 1. Lookup tool in registry
        adapter: Optional[BaseToolAdapter] = self.registry.get(tool_name)
        if adapter is None:
            res = ToolResult(
                call_id=call_id,
                name=tool_name,
                tool_name=tool_name,
                error=f"Tool '{tool_name}' not found in registry",
                is_error=True,
                started_at=start_wall,
                finished_at=time.perf_counter(),
                execution_time_ns=time.perf_counter_ns() - start_ns,
            )
            return res

        spec = adapter.spec

        # 2. Argument & Schema validation
        required_args = spec.required_args or list(spec.parameters.get("required", []))
        missing_args = [arg for arg in required_args if arg not in call.arguments]
        if missing_args:
            res = ToolResult(
                call_id=call_id,
                name=tool_name,
                tool_name=tool_name,
                error=f"Tool '{tool_name}' missing required arguments: {missing_args}",
                is_error=True,
                started_at=start_wall,
                finished_at=time.perf_counter(),
                execution_time_ns=time.perf_counter_ns() - start_ns,
            )
            return res

        # 3. Read-only / Side-effect safety classification
        is_read_only = spec.is_read_only and not spec.side_effects
        is_mutative = not is_read_only or spec.side_effects

        # Speculative & Early-dispatch safety: NEVER execute side-effects speculatively!
        if (is_speculative or is_early_dispatched) and is_mutative:
            self.guardrails.metrics.unapproved_side_effects += 1
            self.guardrails.metrics.unsafe_side_effects += 1
            self.profiler.record_event(
                EventType.GUARDRAIL_VIOLATION,
                details={
                    "error": f"Attempted speculative/early execution of mutative side-effecting tool '{tool_name}'",
                    "call_id": call_id,
                },
            )
            return ToolResult(
                call_id=call_id,
                name=tool_name,
                tool_name=tool_name,
                error=f"Safety violation: mutative tool '{tool_name}' cannot be executed speculatively or early",
                is_error=True,
                started_at=start_wall,
                finished_at=time.perf_counter(),
                execution_time_ns=time.perf_counter_ns() - start_ns,
            )

        # 4. Approval gate enforcement
        requires_approval = spec.requires_approval or (is_mutative and not call.is_approved) or call.requires_approval
        if requires_approval and not call.is_approved:
            self.guardrails.metrics.unapproved_side_effects += 1
            self.guardrails.metrics.unsafe_side_effects += 1
            self.profiler.record_event(
                EventType.APPROVAL_REJECTED,
                details={"tool": tool_name, "call_id": call_id},
            )
            return ToolResult(
                call_id=call_id,
                name=tool_name,
                tool_name=tool_name,
                error="Action rejected: tool requires explicit approval before execution.",
                is_error=True,
                started_at=start_wall,
                finished_at=time.perf_counter(),
                execution_time_ns=time.perf_counter_ns() - start_ns,
                metadata={"approval_required": True, "approval_failed": True},
            )

        # 5. Idempotency Check
        if call.idempotency_key and call.idempotency_key in self._idempotency_cache:
            cached_res = self._idempotency_cache[call.idempotency_key]
            # Return cached result without re-executing
            return ToolResult(
                call_id=call_id,
                name=tool_name,
                tool_name=tool_name,
                result=cached_res.output,
                output=cached_res.output,
                error=cached_res.error,
                is_error=cached_res.is_error,
                cached=True,
                started_at=start_wall,
                finished_at=time.perf_counter(),
                execution_time_ns=time.perf_counter_ns() - start_ns,
                metadata={"idempotent_replay": True, "idempotency_key": call.idempotency_key},
            )

        # 6. Record tool dispatch in guardrails and profiler
        self.guardrails.record_tool_dispatch(spec, call, is_speculative=is_speculative)
        self.profiler.start_span(span_name)
        self._active_calls.add(call_id)

        timeout = timeout_s if timeout_s is not None else self.default_timeout_s
        deadline = time.perf_counter() + timeout

        try:
            # Acquire concurrency slot and rate-limiting tokens
            queue_delay_ms = await self.rate_limiter.acquire(tokens=1, timeout=timeout)
            if queue_delay_ms > 0:
                self.profiler.record_event(
                    EventType.RATE_LIMIT_DELAY,
                    duration_ms=queue_delay_ms,
                    details={"tool": tool_name, "call_id": call_id},
                )
            self.guardrails.record_concurrency_enter()

            try:
                remaining_timeout = max(0.001, deadline - time.perf_counter())
                res = await asyncio.wait_for(adapter.execute(call), timeout=remaining_timeout)
            finally:
                self.rate_limiter.release()
                self.guardrails.record_concurrency_exit()

        except RateLimitError as rle:
            self.guardrails.record_rate_limit_failure()
            self.profiler.record_event(
                EventType.RATE_LIMIT_ERROR,
                details={"tool": tool_name, "error": str(rle)},
            )
            res = ToolResult(
                call_id=call_id,
                name=tool_name,
                tool_name=tool_name,
                error=f"Rate limit exceeded: {str(rle)}",
                is_error=True,
                started_at=start_wall,
                finished_at=time.perf_counter(),
                execution_time_ns=time.perf_counter_ns() - start_ns,
            )

        except asyncio.TimeoutError:
            res = ToolResult(
                call_id=call_id,
                name=tool_name,
                tool_name=tool_name,
                error=f"Tool execution timed out after {timeout:.1f}s",
                is_error=True,
                started_at=start_wall,
                finished_at=time.perf_counter(),
                execution_time_ns=time.perf_counter_ns() - start_ns,
            )

        except asyncio.CancelledError:
            self.profiler.record_event(
                EventType.TOOL_CANCELLED,
                details={"tool": tool_name, "call_id": call_id},
            )
            res = ToolResult(
                call_id=call_id,
                name=tool_name,
                tool_name=tool_name,
                error="Tool execution cancelled.",
                is_error=True,
                cancelled=True,
                started_at=start_wall,
                finished_at=time.perf_counter(),
                execution_time_ns=time.perf_counter_ns() - start_ns,
            )
            raise

        except Exception as ex:
            res = ToolResult(
                call_id=call_id,
                name=tool_name,
                tool_name=tool_name,
                error=f"Tool execution error: {str(ex)}",
                is_error=True,
                started_at=start_wall,
                finished_at=time.perf_counter(),
                execution_time_ns=time.perf_counter_ns() - start_ns,
            )

        finally:
            self._active_calls.discard(call_id)
            self.profiler.end_span(
                span_name,
                EventType.TOOL_END,
                details={"tool": tool_name, "call_id": call_id, "is_speculative": is_speculative},
            )

        # Store in idempotency cache if successful and key present
        if call.idempotency_key and res.is_success:
            self._idempotency_cache[call.idempotency_key] = res

        return res
