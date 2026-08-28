"""Tool execution engine with schema validation, rate-limit leasing, scoped idempotency, and trusted approvals."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
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
    ToolCall,
    ToolResult,
    ToolSpec,
)


class IdempotencyEntry:
    """Internal state record for an in-flight or completed idempotent tool execution."""

    def __init__(self, key: str, arg_fingerprint: str) -> None:
        self.key: str = key
        self.arg_fingerprint: str = arg_fingerprint
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
                new_entry = IdempotencyEntry(key=store_key, arg_fingerprint=arg_fp)
                try:
                    loop = asyncio.get_running_loop()
                    new_entry.future = loop.create_future()
                except RuntimeError:
                    new_entry.future = None
                self._entries[store_key] = new_entry
                return "RESERVED_PRIMARY", store_key, new_entry.future, None

            if entry.arg_fingerprint != arg_fp:
                return "ARG_MISMATCH", store_key, None, None

            if entry.state == "SUCCEEDED" and entry.result is not None:
                return "COMPLETED", store_key, None, copy.deepcopy(entry.result)
            elif entry.state == "IN_FLIGHT":
                return "JOIN_IN_FLIGHT", store_key, entry.future, None
            else:
                entry.state = "IN_FLIGHT"
                try:
                    loop = asyncio.get_running_loop()
                    new_entry_fut = loop.create_future()
                except RuntimeError:
                    new_entry_fut = None
                entry.future = new_entry_fut
                return "RESERVED_PRIMARY", store_key, new_entry_fut, None

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
    ) -> None:
        self.registry = registry
        self.rate_limiter = rate_limiter or RateLimiter()
        self.profiler = profiler or LatencyProfiler()
        self.guardrails = guardrails or GuardrailMonitor()
        self.idempotency_store = idempotency_store or GLOBAL_IDEMPOTENCY_STORE
        self.default_timeout_s = default_timeout_s
        self.trusted_grants = trusted_grants or {}

    def _validate_schema(
        self, spec: ToolSpec, arguments: dict[str, Any]
    ) -> tuple[bool, str]:
        """Strict JSON-Schema validation against parameter spec."""
        schema_params = spec.parameters or {}
        properties = schema_params.get("properties", {})
        required = schema_params.get("required", []) or spec.required_args

        # 1. Required argument check
        for req in required:
            if req not in arguments:
                return False, f"Missing required parameter: '{req}'"

        # 2. Type, bounds, and reference checks
        for k, val in arguments.items():
            if isinstance(val, str) and val.startswith("$"):
                return False, f"Unresolved reference detected in argument '{k}': '{val}'"

            if k not in properties:
                continue
            spec_prop = properties[k]
            expected_type = spec_prop.get("type")

            if expected_type == "string":
                if not isinstance(val, str):
                    return False, f"Argument '{k}' expected string, got {type(val).__name__}"
            elif expected_type == "integer":
                if not isinstance(val, int) or isinstance(val, bool):
                    return False, f"Argument '{k}' expected integer, got {type(val).__name__}"
            elif expected_type == "number":
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    return False, f"Argument '{k}' expected number, got {type(val).__name__}"
            elif expected_type == "boolean":
                if not isinstance(val, bool):
                    return False, f"Argument '{k}' expected boolean, got {type(val).__name__}"
            elif expected_type == "array" and not isinstance(val, list):
                return False, f"Argument '{k}' expected list, got {type(val).__name__}"
            elif expected_type == "object" and not isinstance(val, dict):
                return False, f"Argument '{k}' expected dict, got {type(val).__name__}"

            # Enum check
            if "enum" in spec_prop and val not in spec_prop["enum"]:
                return False, f"Argument '{k}' value '{val}' not in allowed enum: {spec_prop['enum']}"

        return True, ""

    async def execute(
        self,
        call: ToolCall,
        is_speculative: bool = False,
        trusted_grant: ApprovalGrant | None = None,
    ) -> ToolResult:
        """Executes a single tool call through the complete validation, lease, and safety lifecycle."""
        tool_name = call.name or call.tool_name
        call_id = call.call_id
        start_wall = time.perf_counter()
        start_ns = time.perf_counter_ns()

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
                finished_at=time.perf_counter(),
                execution_time_ns=time.perf_counter_ns() - start_ns,
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
                    finished_at=time.perf_counter(),
                    execution_time_ns=time.perf_counter_ns() - start_ns,
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
                finished_at=time.perf_counter(),
                execution_time_ns=time.perf_counter_ns() - start_ns,
            )

        # 4. Trusted Approval Gate Enforcement
        # Model cannot grant its own approval via call.is_approved or call.approval_grant!
        requires_approval = spec.requires_approval or call.requires_approval
        if requires_approval:
            # Grant MUST come from trusted context or trusted_system grant
            grant = trusted_grant or self.trusted_grants.get(tool_name)
            if grant is None and isinstance(call.approval_grant, ApprovalGrant) and getattr(call.approval_grant, "authority", "") != "model":
                grant = call.approval_grant

            grant_valid = False
            if grant is not None and isinstance(grant, ApprovalGrant):
                grant_valid = grant.matches(tool_name, call.arguments)

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
                    exec_start = time.perf_counter()

                    res = await asyncio.wait_for(
                        adapter.execute(call),
                        timeout=self.default_timeout_s,
                    )
                    exec_duration_ms = (time.perf_counter() - exec_start) * 1000.0

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
                res.finished_at = time.perf_counter()
                res.execution_time_ns = time.perf_counter_ns() - start_ns
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
                finished_at=time.perf_counter(),
                execution_time_ns=time.perf_counter_ns() - start_ns,
            )
            if idempotency_store_key:
                self.idempotency_store.publish_result(idempotency_store_key, res)
            return res

        except asyncio.CancelledError:
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
                finished_at=time.perf_counter(),
                execution_time_ns=time.perf_counter_ns() - start_ns,
            )
            if idempotency_store_key:
                self.idempotency_store.publish_result(idempotency_store_key, res)
            return res
