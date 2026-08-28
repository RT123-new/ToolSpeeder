"""Centralized ToolExecutor owning validation, safety, rate-limiting, and lifecycle management."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import threading
import time
from typing import Any

from toolspeed.adapters.base import BaseToolAdapter, ToolRegistry
from toolspeed.core.guardrails import GuardrailMonitor
from toolspeed.core.profiler import LatencyProfiler
from toolspeed.core.rate_limiter import RateLimitError, RateLimiter
from toolspeed.core.types import (
    ApprovalGrant,
    EventType,
    ToolCall,
    ToolResult,
)


class IdempotencyEntry:
    """Atomic state entry for shared idempotency store."""
    def __init__(self, key: str, arg_fingerprint: str):
        self.key = key
        self.arg_fingerprint = arg_fingerprint
        self.state: str = "IN_FLIGHT"  # IN_FLIGHT, SUCCEEDED, FAILED
        self.result: ToolResult | None = None
        self.future: asyncio.Future[ToolResult] | None = None
        self.created_at: float = time.perf_counter()


class SharedIdempotencyStore:
    """Shared, atomic, thread-safe idempotency registry across tasks and execution lifecycles.
    
    Implements atomic ABSENT -> IN_FLIGHT -> SUCCEEDED/FAILED lifecycle:
    - Exactly one caller executes the underlying mutation.
    - Concurrent duplicate callers await the same in-flight future.
    - Differing arguments with the same idempotency key fail closed.
    """

    def __init__(self) -> None:
        self._entries: dict[str, IdempotencyEntry] = {}
        self._sync_lock = threading.Lock()

    @staticmethod
    def compute_arg_fingerprint(arguments: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(arguments, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def compute_key(
        tool_name: str,
        idempotency_key: str,
        tenant_scope: str = "default_tenant",
        op_scope: str = "default_op",
    ) -> str:
        return f"{tenant_scope}:{op_scope}:{tool_name}:{idempotency_key}"

    def reserve_or_join(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        idempotency_key: str,
        tenant_scope: str = "default_tenant",
        op_scope: str = "default_op",
    ) -> tuple[str, str, asyncio.Future[ToolResult] | None, ToolResult | None]:
        """Atomically reserve execution slot or join in-flight/completed execution.
        
        Returns:
            (status, store_key, future, cached_result)
            status can be: 'RESERVED_PRIMARY', 'JOIN_IN_FLIGHT', 'COMPLETED', 'ARG_MISMATCH'
        """
        store_key = self.compute_key(tool_name, idempotency_key, tenant_scope, op_scope)
        arg_fp = self.compute_arg_fingerprint(arguments)

        with self._sync_lock:
            entry = self._entries.get(store_key)
            if entry is None:
                # Primary executor: reserve slot
                new_entry = IdempotencyEntry(key=store_key, arg_fingerprint=arg_fp)
                try:
                    loop = asyncio.get_running_loop()
                    new_entry.future = loop.create_future()
                except RuntimeError:
                    new_entry.future = None
                self._entries[store_key] = new_entry
                return "RESERVED_PRIMARY", store_key, new_entry.future, None

            # Existing entry: check argument fingerprint
            if entry.arg_fingerprint != arg_fp:
                return "ARG_MISMATCH", store_key, None, None

            if entry.state == "SUCCEEDED" and entry.result is not None:
                return "COMPLETED", store_key, None, copy.deepcopy(entry.result)
            elif entry.state == "IN_FLIGHT":
                return "JOIN_IN_FLIGHT", store_key, entry.future, None
            else:
                # Previous attempt failed: allow retry
                entry.state = "IN_FLIGHT"
                try:
                    loop = asyncio.get_running_loop()
                    entry.future = loop.create_future()
                except RuntimeError:
                    entry.future = None
                return "RESERVED_PRIMARY", store_key, entry.future, None

    def publish_result(self, store_key: str, result: ToolResult) -> None:
        """Atomically publish result for in-flight followers and future callers."""
        with self._sync_lock:
            entry = self._entries.get(store_key)
            if entry is not None:
                entry.result = copy.deepcopy(result)
                entry.state = "SUCCEEDED" if result.is_success else "FAILED"
                fut = entry.future

        if fut is not None and not fut.done():
            fut.set_result(copy.deepcopy(result))

    def get(self, key: str) -> ToolResult | None:
        with self._sync_lock:
            entry = self._entries.get(key)
            if entry and entry.state == "SUCCEEDED" and entry.result is not None:
                return copy.deepcopy(entry.result)
            return None

    def put(self, key: str, result: ToolResult) -> None:
        self.publish_result(key, result)

    def clear(self) -> None:
        with self._sync_lock:
            self._entries.clear()


# Global shared default store
GLOBAL_IDEMPOTENCY_STORE = SharedIdempotencyStore()


def _check_unresolved_references(val: Any) -> str | None:
    """Recursively checks for unresolved variable references ($... or {{...}})."""
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith("$") and len(s) > 1 and not s.startswith("$$")) or ("{{" in s and "}}" in s):
            return f"Unresolved reference detected in argument: '{val}'"
    elif isinstance(val, dict):
        for k, v in val.items():
            err = _check_unresolved_references(k) or _check_unresolved_references(v)
            if err:
                return err
    elif isinstance(val, (list, tuple)):
        for item in val:
            err = _check_unresolved_references(item)
            if err:
                return err
    return None


def validate_arguments_against_schema(parameters: dict[str, Any], arguments: dict[str, Any]) -> tuple[bool, str | None]:
    """Strict standards-compliant JSON schema validation (types, enums, bounds, required, additionalProperties)."""
    if not isinstance(arguments, dict):
        return False, f"Arguments must be an object (dict), got {type(arguments).__name__}"

    # 0. Check for unresolved template / variable references ($c1.user_id, etc.)
    unresolved_err = _check_unresolved_references(arguments)
    if unresolved_err:
        return False, unresolved_err

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

    # 3. Type, Enum, and Bounds validation
    for k, val in arguments.items():
        if k in properties:
            spec = properties[k]
            expected_type = spec.get("type")

            if expected_type == "string":
                if not isinstance(val, str):
                    return False, f"Argument '{k}' expected string, got {type(val).__name__}"
                if "minLength" in spec and len(val) < spec["minLength"]:
                    return False, f"Argument '{k}' shorter than minLength {spec['minLength']}"
                if "maxLength" in spec and len(val) > spec["maxLength"]:
                    return False, f"Argument '{k}' exceeds maxLength {spec['maxLength']}"

            elif expected_type == "integer":
                if not (isinstance(val, int) and not isinstance(val, bool)):
                    return False, f"Argument '{k}' expected integer, got {type(val).__name__}"
                if "minimum" in spec and val < spec["minimum"]:
                    return False, f"Argument '{k}' value {val} is less than minimum {spec['minimum']}"
                if "maximum" in spec and val > spec["maximum"]:
                    return False, f"Argument '{k}' value {val} exceeds maximum {spec['maximum']}"

            elif expected_type == "number":
                if not (isinstance(val, (int, float)) and not isinstance(val, bool)):
                    return False, f"Argument '{k}' expected number, got {type(val).__name__}"
                if "minimum" in spec and val < spec["minimum"]:
                    return False, f"Argument '{k}' value {val} is less than minimum {spec['minimum']}"
                if "maximum" in spec and val > spec["maximum"]:
                    return False, f"Argument '{k}' value {val} exceeds maximum {spec['maximum']}"

            elif expected_type == "boolean":
                if not isinstance(val, bool):
                    return False, f"Argument '{k}' expected boolean, got {type(val).__name__}"

            elif expected_type == "array":
                if not isinstance(val, list):
                    return False, f"Argument '{k}' expected array, got {type(val).__name__}"
                if "items" in spec and isinstance(spec["items"], dict):
                    item_type = spec["items"].get("type")
                    if item_type == "string" and not all(isinstance(x, str) for x in val):
                        return False, f"All items in array '{k}' must be strings"
                    elif item_type == "integer" and not all(isinstance(x, int) and not isinstance(x, bool) for x in val):
                        return False, f"All items in array '{k}' must be integers"
                    elif item_type == "number" and not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in val):
                        return False, f"All items in array '{k}' must be numbers"

            elif expected_type == "object":
                if not isinstance(val, dict):
                    return False, f"Argument '{k}' expected object, got {type(val).__name__}"
                if "properties" in spec:
                    ok, nested_err = validate_arguments_against_schema(spec, val)
                    if not ok:
                        return False, f"Nested object '{k}' validation error: {nested_err}"

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
        rate_limiter: RateLimiter | None = None,
        profiler: LatencyProfiler | None = None,
        guardrails: GuardrailMonitor | None = None,
        idempotency_store: SharedIdempotencyStore | None = None,
        default_timeout_s: float = 30.0,
    ):
        self.registry = registry
        self.rate_limiter = rate_limiter or RateLimiter()
        self.profiler = profiler or LatencyProfiler()
        self.guardrails = guardrails or GuardrailMonitor()
        self.idempotency_store = idempotency_store or GLOBAL_IDEMPOTENCY_STORE
        self.default_timeout_s = default_timeout_s
        self._active_calls: set[str] = set()

    async def execute(
        self,
        call: ToolCall,
        is_speculative: bool = False,
        is_early_dispatched: bool = False,
        timeout_s: float | None = None,
        deadline: float | None = None,
        trusted_grant: ApprovalGrant | None = None,
    ) -> ToolResult:
        """Execute a tool call with full lifecycle instrumentation, validation, and safety."""
        start_ns = time.perf_counter_ns()
        start_wall = time.perf_counter()
        call_id = call.call_id
        tool_name = call.name or call.tool_name
        span_prefix = "spec_" if is_speculative else ("early_" if is_early_dispatched else "")
        span_name = f"{span_prefix}tool_{call_id}"

        # 1. Lookup tool in registry
        adapter: BaseToolAdapter | None = self.registry.get(tool_name)
        if adapter is None:
            return ToolResult(
                call_id=call_id,
                name=tool_name,
                tool_name=tool_name,
                error=f"Tool '{tool_name}' not found in registry",
                is_error=True,
                started_at=start_wall,
                finished_at=time.perf_counter(),
                execution_time_ns=time.perf_counter_ns() - start_ns,
            )

        spec = adapter.spec

        # 2. Schema and argument validation
        is_valid, err_msg = validate_arguments_against_schema(spec.parameters, call.arguments)
        if not is_valid:
            return ToolResult(
                call_id=call_id,
                name=tool_name,
                tool_name=tool_name,
                error=f"Tool '{tool_name}' schema validation failed: {err_msg}",
                is_error=True,
                started_at=start_wall,
                finished_at=time.perf_counter(),
                execution_time_ns=time.perf_counter_ns() - start_ns,
            )

        # 3. Read-only / Side-effect safety classification
        is_read_only = spec.is_read_only and not spec.side_effects
        is_mutative = not is_read_only or spec.side_effects

        # Speculative & Early-dispatch safety: NEVER execute side-effects speculatively!
        if (is_speculative or is_early_dispatched) and is_mutative:
            self.guardrails.metrics.unapproved_side_effects += 1
            self.guardrails.metrics.unsafe_side_effects += 1
            self.guardrails.metrics.blocked_unsafe_attempts += 1
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

        # 4. Trusted Approval Gate Enforcement
        # Model cannot grant its own approval via call.is_approved!
        requires_approval = spec.requires_approval or call.requires_approval
        if requires_approval:
            grant = trusted_grant or call.approval_grant
            grant_valid = False
            if grant is not None and isinstance(grant, ApprovalGrant):
                grant_valid = grant.matches(tool_name, call.arguments)

            # Strict rejection if approval is missing, invalid, or forged
            if not grant_valid:
                self.guardrails.metrics.unapproved_side_effects += 1
                self.guardrails.metrics.blocked_unsafe_attempts += 1
                self.profiler.record_event(
                    EventType.APPROVAL_REJECTED,
                    details={"tool": tool_name, "call_id": call_id, "forged_model_approval": call.is_approved},
                )
                return ToolResult(
                    call_id=call_id,
                    name=tool_name,
                    tool_name=tool_name,
                    error="Action rejected: tool requires explicit trusted approval grant matching exact arguments.",
                    is_error=True,
                    started_at=start_wall,
                    finished_at=time.perf_counter(),
                    execution_time_ns=time.perf_counter_ns() - start_ns,
                    metadata={"approval_required": True, "approval_failed": True},
                )

        # 5. Atomic Idempotency Check
        idempotency_key = call.idempotency_key or call.arguments.get("idempotency_key")
        idempotency_store_key: str | None = None
        if idempotency_key:
            call.idempotency_key = str(idempotency_key)
            status, store_key, future, cached_res = self.idempotency_store.reserve_or_join(
                tool_name=tool_name,
                arguments=call.arguments,
                idempotency_key=str(idempotency_key),
                tenant_scope=call.metadata.get("tenant_id", "default_tenant"),
                op_scope=call.metadata.get("op_scope", "default_op"),
            )
            idempotency_store_key = store_key

            if status == "ARG_MISMATCH":
                return ToolResult(
                    call_id=call_id,
                    name=tool_name,
                    tool_name=tool_name,
                    error=f"Idempotency conflict: key '{idempotency_key}' already used with different arguments.",
                    is_error=True,
                    started_at=start_wall,
                    finished_at=time.perf_counter(),
                    execution_time_ns=time.perf_counter_ns() - start_ns,
                )
            elif status == "COMPLETED" and cached_res is not None:
                cached_res.call_id = call_id
                cached_res.cached = True
                cached_res.metadata["idempotent_replay"] = True
                cached_res.metadata["idempotency_key"] = str(idempotency_key)
                return cached_res
            elif status == "JOIN_IN_FLIGHT" and future is not None:
                # Await in-flight result from primary caller
                try:
                    awaited_res = await future
                    follower_res = copy.deepcopy(awaited_res)
                    follower_res.call_id = call_id
                    follower_res.cached = True
                    follower_res.metadata["idempotent_replay"] = True
                    follower_res.metadata["idempotency_key"] = str(idempotency_key)
                    return follower_res
                except Exception as ex:
                    return ToolResult(
                        call_id=call_id,
                        name=tool_name,
                        tool_name=tool_name,
                        error=f"Follower idempotency await failed: {ex}",
                        is_error=True,
                        started_at=start_wall,
                        finished_at=time.perf_counter(),
                        execution_time_ns=time.perf_counter_ns() - start_ns,
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
            if idempotency_store_key:
                self.idempotency_store.publish_result(idempotency_store_key, res)
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

        # Publish result into atomic idempotency store
        if idempotency_store_key:
            self.idempotency_store.publish_result(idempotency_store_key, res)

        return res
