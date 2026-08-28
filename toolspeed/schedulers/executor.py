"""Tool execution engine with schema validation, rate-limit leasing, scoped idempotency, and trusted approvals."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import threading
import time
from typing import Any

from toolspeed.adapters.base import ToolRegistry
from toolspeed.core.guardrails import GuardrailMonitor
from toolspeed.core.profiler import LatencyProfiler
from toolspeed.core.rate_limiter import RateLimiter
from toolspeed.core.types import (
    ApprovalGrant,
    EventType,
    ExecutionAuthorityContext,
    ToolCall,
    ToolResult,
    ToolSpec,
)


class IdempotencyState:
    ABSENT = "ABSENT"
    IN_FLIGHT = "IN_FLIGHT"
    COMMITTED = "COMMITTED"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    FAILED_FINAL = "FAILED_FINAL"
    CANCELLED = "CANCELLED"


class IdempotencyEntry:
    """Internal state record for an in-flight or completed idempotent tool execution."""

    def __init__(self, key: str, arg_fingerprint: str, created_at: float = 0.0) -> None:
        self.key: str = key
        self.arg_fingerprint: str = arg_fingerprint
        self.state: str = IdempotencyState.IN_FLIGHT
        self.result: ToolResult | None = None
        self.future: asyncio.Future[ToolResult] | None = None
        self.created_at: float = created_at


class SharedIdempotencyStore:
    """Shared, atomic, thread-safe idempotency registry across tasks and execution lifecycles.

    Implements explicit ABSENT -> IN_FLIGHT -> COMMITTED / FAILED / CANCELLED lifecycle:
    - Exactly one caller executes the underlying mutation.
    - Concurrent duplicate callers await the same in-flight future.
    - Differing arguments with the same idempotency key fail closed.
    - Cancellation, timeout, and failure deterministically resolve all followers.
    """

    def __init__(self, clock: Any = None) -> None:
        self._entries: dict[str, IdempotencyEntry] = {}
        self._sync_lock = threading.Lock()
        self.clock = clock

    def _now_s(self) -> float:
        if self.clock is not None and hasattr(self.clock, "now_s"):
            return self.clock.now_s()
        return time.perf_counter()

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
                new_entry = IdempotencyEntry(key=store_key, arg_fingerprint=arg_fp, created_at=self._now_s())
                try:
                    loop = asyncio.get_running_loop()
                    new_entry.future = loop.create_future()
                except RuntimeError:
                    new_entry.future = None
                self._entries[store_key] = new_entry
                return "RESERVED_PRIMARY", store_key, new_entry.future, None

            if entry.arg_fingerprint != arg_fp:
                return "ARG_MISMATCH", store_key, None, None

            if entry.state == IdempotencyState.COMMITTED and entry.result is not None:
                return "COMPLETED", store_key, None, copy.deepcopy(entry.result)
            elif entry.state == IdempotencyState.IN_FLIGHT:
                # Ensure the future belongs to current loop
                if entry.future is None or entry.future.done():
                    try:
                        loop = asyncio.get_running_loop()
                        entry.future = loop.create_future()
                    except RuntimeError:
                        entry.future = None
                return "JOIN_IN_FLIGHT", store_key, entry.future, None
            else:
                entry.state = IdempotencyState.IN_FLIGHT
                try:
                    loop = asyncio.get_running_loop()
                    new_entry_fut = loop.create_future()
                except RuntimeError:
                    new_entry_fut = None
                entry.future = new_entry_fut
                return "RESERVED_PRIMARY", store_key, new_entry_fut, None

    def publish_result(self, store_key: str, result: ToolResult) -> None:
        """Atomically publish result for in-flight followers and future callers."""
        fut = None
        with self._sync_lock:
            entry = self._entries.get(store_key)
            if entry is not None:
                entry.result = copy.deepcopy(result)
                if result.cancelled:
                    entry.state = IdempotencyState.CANCELLED
                elif result.is_success:
                    entry.state = IdempotencyState.COMMITTED
                else:
                    entry.state = IdempotencyState.FAILED_FINAL
                fut = entry.future

        if fut is not None and not fut.done():
            fut.set_result(copy.deepcopy(result))

    def cancel_in_flight(self, store_key: str, reason: str = "Primary execution cancelled") -> None:
        """Deterministically cancel an in-flight entry and resolve followers with error."""
        fut = None
        with self._sync_lock:
            entry = self._entries.get(store_key)
            if entry is not None and entry.state == IdempotencyState.IN_FLIGHT:
                entry.state = IdempotencyState.CANCELLED
                cancelled_res = ToolResult(
                    call_id="",
                    tool_name="",
                    error=reason,
                    is_error=True,
                    cancelled=True,
                )
                entry.result = cancelled_res
                fut = entry.future

        if fut is not None and not fut.done() and entry is not None and entry.result is not None:
            fut.set_result(copy.deepcopy(entry.result))

    def get(self, key: str) -> ToolResult | None:
        with self._sync_lock:
            entry = self._entries.get(key)
            if entry and entry.state == IdempotencyState.COMMITTED and entry.result is not None:
                return copy.deepcopy(entry.result)
            return None

    def put(self, key: str, result: ToolResult) -> None:
        self.publish_result(key, result)

    def clear(self) -> None:
        with self._sync_lock:
            self._entries.clear()


GLOBAL_IDEMPOTENCY_STORE = SharedIdempotencyStore()


class ToolExecutor:
    """Central tool execution pipeline enforcing schema validation, leases, idempotency, and approval."""

    def __init__(
        self,
        registry: ToolRegistry,
        rate_limiter: RateLimiter | None = None,
        profiler: LatencyProfiler | None = None,
        guardrails: GuardrailMonitor | None = None,
        idempotency_store: SharedIdempotencyStore | None = None,
        default_timeout_s: float = 60.0,
        trusted_grants: dict[str, ApprovalGrant] | None = None,
        authority_context: ExecutionAuthorityContext | None = None,
        clock: Any = None,
    ) -> None:
        self.registry = registry
        self.clock = clock
        self.rate_limiter = rate_limiter or RateLimiter(clock=clock)
        self.profiler = profiler or LatencyProfiler(clock=clock)
        self.guardrails = guardrails or GuardrailMonitor()
        self.idempotency_store = idempotency_store or SharedIdempotencyStore(clock=clock)
        self.default_timeout_s = default_timeout_s
        self.trusted_grants = trusted_grants or {}
        self.authority_context = authority_context or ExecutionAuthorityContext()

    def _now_s(self) -> float:
        if self.clock is not None and hasattr(self.clock, "now_s"):
            return self.clock.now_s()
        return time.perf_counter()

    def _now_ns(self) -> int:
        if self.clock is not None and hasattr(self.clock, "now_ns"):
            return self.clock.now_ns()
        return time.perf_counter_ns()

    def _validate_schema(self, spec: ToolSpec, arguments: dict[str, Any]) -> tuple[bool, str]:
        """Strict JSON-Schema validation against parameter spec supporting nested objects, arrays, bounds, enums."""
        schema_params = spec.parameters or {}
        return self._validate_object(schema_params, arguments, path="")

    def _validate_object(self, schema: dict[str, Any], obj: Any, path: str = "") -> tuple[bool, str]:
        if not isinstance(obj, dict):
            return False, f"Expected object at '{path or 'root'}', got {type(obj).__name__}"

        properties = schema.get("properties", {})
        required = schema.get("required", [])
        additional_props = schema.get("additionalProperties", True)

        # 1. Required fields
        for req in required:
            if req not in obj:
                field_path = f"{path}.{req}" if path else req
                return False, f"Missing required parameter: '{field_path}'"

        # 2. Additional properties check
        if additional_props is False:
            for k in obj:
                if k not in properties:
                    field_path = f"{path}.{k}" if path else k
                    return False, f"Unexpected property '{field_path}' not defined in schema"

        # 3. Property validation
        for k, val in obj.items():
            field_path = f"{path}.{k}" if path else k

            if isinstance(val, str) and val.startswith("$"):
                return False, f"Unresolved reference detected in argument '{field_path}': '{val}'"

            if k not in properties:
                continue

            spec_prop = properties[k]
            expected_type = spec_prop.get("type")

            if expected_type == "string":
                if not isinstance(val, str):
                    return False, f"Argument '{field_path}' expected string, got {type(val).__name__}"
                if "minLength" in spec_prop and len(val) < spec_prop["minLength"]:
                    return False, f"Argument '{field_path}' length {len(val)} < minLength {spec_prop['minLength']}"
                if "maxLength" in spec_prop and len(val) > spec_prop["maxLength"]:
                    return False, f"Argument '{field_path}' length {len(val)} > maxLength {spec_prop['maxLength']}"

            elif expected_type == "integer":
                if not isinstance(val, int) or isinstance(val, bool):
                    return False, f"Argument '{field_path}' expected integer, got {type(val).__name__}"
                if "minimum" in spec_prop and val < spec_prop["minimum"]:
                    return False, f"Argument '{field_path}' value {val} < minimum {spec_prop['minimum']}"
                if "maximum" in spec_prop and val > spec_prop["maximum"]:
                    return False, f"Argument '{field_path}' value {val} > maximum {spec_prop['maximum']}"

            elif expected_type == "number":
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    return False, f"Argument '{field_path}' expected number, got {type(val).__name__}"
                if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                    return False, f"Argument '{field_path}' must be a finite number"
                if "minimum" in spec_prop and val < spec_prop["minimum"]:
                    return False, f"Argument '{field_path}' value {val} < minimum {spec_prop['minimum']}"
                if "maximum" in spec_prop and val > spec_prop["maximum"]:
                    return False, f"Argument '{field_path}' value {val} > maximum {spec_prop['maximum']}"

            elif expected_type == "boolean":
                if not isinstance(val, bool):
                    return False, f"Argument '{field_path}' expected boolean, got {type(val).__name__}"

            elif expected_type == "array":
                if not isinstance(val, list):
                    return False, f"Argument '{field_path}' expected list, got {type(val).__name__}"
                items_schema = spec_prop.get("items")
                if items_schema and isinstance(items_schema, dict):
                    for idx, item in enumerate(val):
                        item_path = f"{field_path}[{idx}]"
                        item_type = items_schema.get("type")
                        if item_type == "object":
                            valid, err = self._validate_object(items_schema, item, path=item_path)
                            if not valid:
                                return False, err
                        elif item_type == "string" and not isinstance(item, str):
                            return False, f"Item at '{item_path}' expected string, got {type(item).__name__}"
                        elif item_type == "number" and (not isinstance(item, (int, float)) or isinstance(item, bool)):
                            return False, f"Item at '{item_path}' expected number, got {type(item).__name__}"

            elif expected_type == "object":
                if not isinstance(val, dict):
                    return False, f"Argument '{field_path}' expected dict, got {type(val).__name__}"
                valid, err = self._validate_object(spec_prop, val, path=field_path)
                if not valid:
                    return False, err

            # Enum check
            if "enum" in spec_prop and val not in spec_prop["enum"]:
                return False, f"Argument '{field_path}' value '{val}' not in allowed enum: {spec_prop['enum']}"

        return True, ""

    async def execute(
        self,
        call: ToolCall,
        is_speculative: bool = False,
        trusted_grant: ApprovalGrant | None = None,
        authority_context: ExecutionAuthorityContext | None = None,
    ) -> ToolResult:
        """Executes a single tool call through the complete validation, lease, and safety lifecycle."""
        tool_name = call.name or call.tool_name
        call_id = call.call_id
        start_wall = self._now_s()
        start_ns = self._now_ns()

        # 1. Registry Lookup
        adapter = self.registry.get(tool_name)
        if adapter is None:
            self.guardrails.record_tool_error(tool_name, "Tool not found in registry")
            return ToolResult(
                call_id=call_id,
                name=tool_name,
                tool_name=tool_name,
                error=f"Tool '{tool_name}' is not registered in ToolRegistry.",
                is_error=True,
                started_at=start_wall,
                finished_at=self._now_s(),
                execution_time_ns=self._now_ns() - start_ns,
            )

        spec = adapter.spec

        # 2. Speculation Read-Only Guardrail
        if is_speculative and (not spec.is_read_only or spec.side_effects or spec.requires_approval):
            self.guardrails.record_guardrail_violation(
                rule_id="SPECULATION_SIDE_EFFECT_ATTEMPT",
                details={"tool": tool_name, "call_id": call_id},
            )
            self.profiler.record_event(
                EventType.GUARDRAIL_VIOLATION,
                details={"tool": tool_name, "reason": "Speculative execution of side-effect tool rejected"},
            )
            return ToolResult(
                call_id=call_id,
                name=tool_name,
                tool_name=tool_name,
                error="Speculative execution of mutative or approval-requiring tools is prohibited.",
                is_error=True,
                speculated=True,
                started_at=start_wall,
                finished_at=self._now_s(),
                execution_time_ns=self._now_ns() - start_ns,
            )

        # 3. Strict Schema Validation
        valid_schema, schema_err = self._validate_schema(spec, call.arguments)
        if not valid_schema:
            self.guardrails.record_tool_error(tool_name, schema_err)
            return ToolResult(
                call_id=call_id,
                name=tool_name,
                tool_name=tool_name,
                error=f"Schema validation error for tool '{tool_name}': {schema_err}",
                is_error=True,
                started_at=start_wall,
                finished_at=self._now_s(),
                execution_time_ns=self._now_ns() - start_ns,
            )

        # 4. Trusted Approval Gate Enforcement
        # Model cannot grant its own approval via call.is_approved or call.approval_grant!
        requires_approval = spec.requires_approval or call.requires_approval
        if requires_approval:
            auth_ctx = authority_context or self.authority_context
            grant_valid = False

            # Check explicit trusted_grant passed from scheduler context
            if trusted_grant is not None and isinstance(trusted_grant, ApprovalGrant):
                grant_valid = trusted_grant.matches(
                    tool_name=tool_name,
                    arguments=call.arguments,
                    subject=auth_ctx.subject,
                    tenant=auth_ctx.tenant,
                    run_id=auth_ctx.run_id,
                    current_time=self._now_s(),
                )
            elif auth_ctx is not None:
                grant_valid = auth_ctx.verify_and_consume_grant(
                    tool_name=tool_name,
                    arguments=call.arguments,
                    current_time=self._now_s(),
                )
            elif tool_name in self.trusted_grants:
                t_grant = self.trusted_grants[tool_name]
                grant_valid = t_grant.matches(
                    tool_name=tool_name,
                    arguments=call.arguments,
                    current_time=self._now_s(),
                )

            # Strict Invariant: NEVER accept call.approval_grant from the untrusted ToolCall object!
            # If the model attached an ApprovalGrant, it is untrusted and ignored.

            if not grant_valid:
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
                    finished_at=self._now_s(),
                    execution_time_ns=self._now_ns() - start_ns,
                    metadata={"approval_required": True, "approval_failed": True},
                )
            else:
                self.profiler.record_event(
                    EventType.APPROVAL_GRANTED,
                    details={"tool": tool_name, "call_id": call_id},
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
                    finished_at=self._now_s(),
                    execution_time_ns=self._now_ns() - start_ns,
                )
            elif status == "COMPLETED" and cached_res is not None:
                cached_res.call_id = call_id
                cached_res.cached = True
                cached_res.metadata["idempotent_replay"] = True
                cached_res.metadata["idempotency_key"] = str(idempotency_key)
                return cached_res
            elif status == "JOIN_IN_FLIGHT" and future is not None:
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
                        finished_at=self._now_s(),
                        execution_time_ns=self._now_ns() - start_ns,
                    )

        # 6. Acquire Rate-Limiter Lease (Tokens first, then Concurrency Slot)
        lease = None
        try:
            async with self.rate_limiter.lease(tokens=1) as acquired_lease:
                lease = acquired_lease
                if lease.queue_delay_ms > 0:
                    self.profiler.record_event(
                        EventType.RATE_LIMIT_DELAY,
                        duration_ms=lease.queue_delay_ms,
                        details={"tool": tool_name},
                    )

                # 7. Execute underlying tool
                self.guardrails.total_tool_calls += 1
                self.guardrails.record_concurrency_enter()
                try:
                    self.profiler.record_event(EventType.TOOL_START, details={"tool": tool_name, "call_id": call_id})
                    exec_start = self._now_s()

                    res = await asyncio.wait_for(
                        adapter.execute(call),
                        timeout=self.default_timeout_s,
                    )
                    exec_duration_ms = (self._now_s() - exec_start) * 1000.0

                    self.profiler.record_event(
                        EventType.TOOL_END,
                        duration_ms=exec_duration_ms,
                        details={"tool": tool_name, "call_id": call_id},
                    )
                finally:
                    self.guardrails.record_concurrency_exit()

                res.call_id = call_id
                res.tool_name = tool_name
                res.name = tool_name
                res.started_at = start_wall
                res.finished_at = self._now_s()
                res.execution_time_ns = self._now_ns() - start_ns
                res.execution_time_ms = exec_duration_ms

                # Publish result to idempotency store if reserved
                if idempotency_store_key:
                    res.metadata["idempotency_key"] = str(idempotency_key)
                    self.idempotency_store.publish_result(idempotency_store_key, res)

                return res

        except asyncio.TimeoutError:
            res = ToolResult(
                call_id=call_id,
                name=tool_name,
                tool_name=tool_name,
                error=f"Tool execution timed out after {self.default_timeout_s}s",
                is_error=True,
                started_at=start_wall,
                finished_at=self._now_s(),
                execution_time_ns=self._now_ns() - start_ns,
            )
            if idempotency_store_key:
                self.idempotency_store.publish_result(idempotency_store_key, res)
            return res

        except asyncio.CancelledError:
            if idempotency_store_key:
                self.idempotency_store.cancel_in_flight(idempotency_store_key, "Primary execution cancelled")
            self.profiler.record_event(EventType.TOOL_CANCELLED, details={"tool": tool_name, "call_id": call_id})
            raise

        except Exception as ex:
            res = ToolResult(
                call_id=call_id,
                name=tool_name,
                tool_name=tool_name,
                error=f"Tool execution unhandled error: {ex}",
                is_error=True,
                started_at=start_wall,
                finished_at=self._now_s(),
                execution_time_ns=self._now_ns() - start_ns,
            )
            if idempotency_store_key:
                self.idempotency_store.publish_result(idempotency_store_key, res)
            return res
