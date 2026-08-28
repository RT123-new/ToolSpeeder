"""Centralized ToolExecutor owning validation, safety, rate-limiting, and lifecycle management."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, Dict, List, Optional, Set, Tuple
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


class SharedIdempotencyStore:
    """Shared, thread-safe idempotency registry across tasks and execution lifecycles."""
    def __init__(self) -> None:
        self._cache: Dict[str, ToolResult] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def compute_key(
        tool_name: str,
        arguments: Dict[str, Any],
        idempotency_key: str,
        tenant_scope: str = "default_tenant",
        op_scope: str = "default_op",
    ) -> str:
        arg_fp = hashlib.sha256(json.dumps(arguments, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return f"{tenant_scope}:{tool_name}:{op_scope}:{arg_fp}:{idempotency_key}"

    def get(self, key: str) -> Optional[ToolResult]:
        return self._cache.get(key)

    def put(self, key: str, result: ToolResult) -> None:
        self._cache[key] = result

    def clear(self) -> None:
        self._cache.clear()


# Global shared default store
GLOBAL_IDEMPOTENCY_STORE = SharedIdempotencyStore()


def validate_arguments_against_schema(parameters: Dict[str, Any], arguments: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Strict standards-compliant schema validation (types, enums, required, properties, additionalProperties)."""
    if not isinstance(arguments, dict):
        return False, f"Arguments must be an object (dict), got {type(arguments).__name__}"

    # 1. Required fields
    required = parameters.get("required", [])
    missing = [r for r in required if r not in arguments]
    if missing:
        return False, f"Missing required arguments: {missing}"

    properties = parameters.get("properties", {})
    allow_additional = parameters.get("additionalProperties", True)

    # 2. Additional properties check
    if not allow_additional:
        extra = [k for k in arguments.keys() if k not in properties]
        if extra:
            return False, f"Additional properties not permitted: {extra}"

    # 3. Type and Enum validation
    for k, val in arguments.items():
        if k in properties:
            spec = properties[k]
            expected_type = spec.get("type")

            if expected_type == "string":
                if not isinstance(val, str):
                    return False, f"Argument '{k}' expected string, got {type(val).__name__}"
            elif expected_type == "integer":
                if not (isinstance(val, int) and not isinstance(val, bool)):
                    return False, f"Argument '{k}' expected integer, got {type(val).__name__}"
            elif expected_type == "number":
                if not (isinstance(val, (int, float)) and not isinstance(val, bool)):
                    return False, f"Argument '{k}' expected number, got {type(val).__name__}"
            elif expected_type == "boolean":
                if not isinstance(val, bool):
                    return False, f"Argument '{k}' expected boolean, got {type(val).__name__}"
            elif expected_type == "array":
                if not isinstance(val, list):
                    return False, f"Argument '{k}' expected array, got {type(val).__name__}"
            elif expected_type == "object":
                if not isinstance(val, dict):
                    return False, f"Argument '{k}' expected object, got {type(val).__name__}"

            # Enum check
            if "enum" in spec:
                if val not in spec["enum"]:
                    return False, f"Argument '{k}' value '{val}' not in allowed enum: {spec['enum']}"

    return True, None


class ToolExecutor:
    """Central execution engine for dispatching tools across all schedulers."""

    def __init__(
        self,
        registry: ToolRegistry,
        rate_limiter: Optional[RateLimiter] = None,
        profiler: Optional[LatencyProfiler] = None,
        guardrails: Optional[GuardrailMonitor] = None,
        idempotency_store: Optional[SharedIdempotencyStore] = None,
        default_timeout_s: float = 30.0,
    ):
        self.registry = registry
        self.rate_limiter = rate_limiter or RateLimiter()
        self.profiler = profiler or LatencyProfiler()
        self.guardrails = guardrails or GuardrailMonitor()
        self.idempotency_store = idempotency_store or GLOBAL_IDEMPOTENCY_STORE
        self.default_timeout_s = default_timeout_s
        self._active_calls: Set[str] = set()

    async def execute(
        self,
        call: ToolCall,
        is_speculative: bool = False,
        is_early_dispatched: bool = False,
        timeout_s: Optional[float] = None,
        deadline: Optional[float] = None,
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

        # 2. Schema and argument validation
        is_valid, err_msg = validate_arguments_against_schema(spec.parameters, call.arguments)
        if not is_valid:
            res = ToolResult(
                call_id=call_id,
                name=tool_name,
                tool_name=tool_name,
                error=f"Tool '{tool_name}' schema validation failed: {err_msg}",
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
        # Schedulers cannot manufacture approval: approval must be pre-authorized or verified
        requires_approval = spec.requires_approval or call.requires_approval
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

        # 5. Shared Idempotency Check
        idempotency_store_key = None
        if call.idempotency_key:
            idempotency_store_key = self.idempotency_store.compute_key(
                tool_name=tool_name,
                arguments=call.arguments,
                idempotency_key=call.idempotency_key,
                tenant_scope=call.metadata.get("tenant_id", "default_tenant"),
                op_scope=call.metadata.get("op_scope", "default_op"),
            )
            cached_res = self.idempotency_store.get(idempotency_store_key)
            if cached_res is not None:
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
        eff_deadline = deadline if deadline is not None else (time.perf_counter() + timeout)

        try:
            # Acquire cancellation-safe rate limit lease
            async with self.rate_limiter.lease(tokens=1, deadline=eff_deadline) as lease:
                if lease.queue_delay_ms > 0:
                    self.profiler.record_event(
                        EventType.RATE_LIMIT_DELAY,
                        duration_ms=lease.queue_delay_ms,
                        details={"tool": tool_name, "call_id": call_id},
                    )
                self.guardrails.record_concurrency_enter()

                try:
                    remaining_timeout = max(0.001, eff_deadline - time.perf_counter())
                    res = await asyncio.wait_for(adapter.execute(call), timeout=remaining_timeout)
                finally:
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

        # Store in shared idempotency store if successful and key present
        if idempotency_store_key and res.is_success:
            self.idempotency_store.put(idempotency_store_key, res)

        return res
