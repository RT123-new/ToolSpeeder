"""Core data structures and types for ToolSpeed benchmark framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import json
import math
import os
import platform
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union
import uuid


class EvidenceLevel(str, Enum):
    """Scientific evidence classification hierarchy."""
    SYNTHETIC = "synthetic"
    REPLAY_INTEGRATION = "replay_integration"
    LOCAL_WALL_CLOCK = "local_wall_clock"
    LIVE = "live"

    def __str__(self) -> str:
        return self.value


class VerdictState(str, Enum):
    """Scientific hypothesis verdict status."""
    PASSED = "passed"
    FALSIFIED = "falsified"
    INCONCLUSIVE = "inconclusive"

    def __str__(self) -> str:
        return self.value


class EventType(str, Enum):
    """Execution event types recorded in high-precision timeline."""
    TASK_START = "task_start"
    TASK_END = "task_end"
    MODEL_START = "model_start"
    MODEL_END = "model_end"
    MODEL_CHUNK = "model_chunk"
    MODEL_DECISION_START = "model_decision_start"
    MODEL_DECISION_END = "model_decision_end"
    DRAFT_MODEL_START = "draft_model_start"
    DRAFT_MODEL_END = "draft_model_end"
    TOOL_DISPATCH = "tool_dispatch"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    TOOL_CANCELLED = "tool_cancelled"
    SPECULATION_START = "speculation_start"
    SPECULATION_HIT = "speculation_hit"
    SPECULATION_MISS = "speculation_miss"
    SPECULATION_CANCELLED = "speculation_cancelled"
    SPECULATIVE_LAUNCH = "speculation_start"
    SPECULATIVE_CANCEL = "speculation_cancelled"
    SPECULATIVE_COMMIT = "speculation_hit"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    CACHE_FRESHNESS_VIOLATION = "cache_freshness_violation"
    DAG_NODE_READY = "dag_node_ready"
    DAG_NODE_DISPATCH = "dag_node_dispatch"
    JIT_FUSION_START = "jit_fusion_start"
    JIT_FUSION_SUCCESS = "jit_fusion_success"
    JIT_FUSION_DEOPT = "jit_fusion_deopt"
    BYTECODE_ENCODE = "bytecode_encode"
    BYTECODE_DECODE = "bytecode_decode"
    GUARDRAIL_VIOLATION = "guardrail_violation"
    RATE_LIMIT_DELAY = "rate_limit_delay"
    RATE_LIMIT_ERROR = "rate_limit_error"
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_REJECTED = "approval_rejected"
    SANDBOX_INIT_START = "sandbox_init_start"
    SANDBOX_INIT_END = "sandbox_init_end"
    COMMIT_HORIZON_REACHED = "commit_horizon_reached"
    CUSTOM = "custom"

    def __str__(self) -> str:
        return self.value


def sanitize_for_json(obj: Any) -> Any:
    """Recursively converts non-standard float values (NaN, Inf) to None for standards-compliant JSON."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, Enum):
        return obj.value
    elif hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        return sanitize_for_json(obj.to_dict())
    return obj


def strict_json_dumps(obj: Any, indent: Optional[int] = 2) -> str:
    """Strict JSON serializer enforcing standards compliance (no NaN / Infinity)."""
    sanitized = sanitize_for_json(obj)
    return json.dumps(sanitized, indent=indent, allow_nan=False)


@dataclass(frozen=True)
class CommittedCall:
    """Immutable representation of a tool call that has safely crossed the commit horizon."""
    tool_name: str
    arguments: Tuple[Tuple[str, Any], ...]
    call_id: str
    schema_hash: str
    semantic_fingerprint: str

    @classmethod
    def from_call(cls, call: ToolCall, schema_hash: str = "") -> CommittedCall:
        sorted_args = tuple(sorted((k, json.dumps(v, sort_keys=True)) for k, v in call.arguments.items()))
        fp_payload = f"{call.name}:{sorted_args}:{schema_hash}".encode("utf-8")
        fingerprint = hashlib.sha256(fp_payload).hexdigest()
        return cls(
            tool_name=call.name or call.tool_name,
            arguments=sorted_args,
            call_id=call.call_id,
            schema_hash=schema_hash,
            semantic_fingerprint=fingerprint,
        )


@dataclass
class ArtifactManifest:
    """Provenance and execution environment metadata for benchmark artifact bundles."""
    git_sha: str = "unknown"
    git_dirty: bool = False
    command: str = ""
    evidence_level: EvidenceLevel = EvidenceLevel.SYNTHETIC
    timestamp_utc: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    seed: int = 0
    python_version: str = platform.python_version()
    os_platform: str = f"{platform.system()} {platform.release()} ({platform.machine()})"
    hardware_info: Dict[str, Any] = field(default_factory=dict)
    dependency_versions: Dict[str, str] = field(default_factory=dict)
    benchmark_config_hash: str = ""
    workload_fixture_hash: str = ""
    raw_trace_hash: str = ""
    report_generator_version: str = "2.0.0"
    is_simulated: bool = True

    @property
    def commit_sha(self) -> str:
        return self.git_sha

    @property
    def dirty(self) -> bool:
        return self.git_dirty

    def to_dict(self) -> Dict[str, Any]:
        return {
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "command": self.command,
            "evidence_level": self.evidence_level.value if isinstance(self.evidence_level, EvidenceLevel) else str(self.evidence_level),
            "timestamp_utc": self.timestamp_utc,
            "seed": self.seed,
            "python_version": self.python_version,
            "os_platform": self.os_platform,
            "hardware_info": self.hardware_info,
            "dependency_versions": self.dependency_versions,
            "benchmark_config_hash": self.benchmark_config_hash,
            "workload_fixture_hash": self.workload_fixture_hash,
            "raw_trace_hash": self.raw_trace_hash,
            "report_generator_version": self.report_generator_version,
            "is_simulated": self.is_simulated,
        }

    @classmethod
    def create(
        cls,
        evidence_level: EvidenceLevel = EvidenceLevel.SYNTHETIC,
        seed: int = 42,
        command: str = "toolspeed",
        is_simulated: bool = False,
        config_data: Any = None,
        fixture_data: Any = None,
        trace_data: Any = None,
    ) -> ArtifactManifest:
        return cls.create_current(
            command=command,
            evidence_level=evidence_level,
            seed=seed,
            is_simulated=is_simulated,
            config_data=config_data,
            fixture_data=fixture_data,
            trace_data=trace_data,
        )

    @classmethod
    def create_current(
        cls,
        command: str,
        evidence_level: EvidenceLevel,
        seed: int,
        is_simulated: bool,
        config_data: Any = None,
        fixture_data: Any = None,
        trace_data: Any = None,
    ) -> ArtifactManifest:
        sha = "unknown"
        dirty = False
        try:
            import subprocess
            res_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2)
            if res_sha.returncode == 0:
                sha = res_sha.stdout.strip()
            res_status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=2)
            if res_status.returncode == 0:
                dirty = len(res_status.stdout.strip()) > 0
        except Exception:
            pass

        deps: Dict[str, str] = {}
        try:
            import numpy
            deps["numpy"] = numpy.__version__
        except Exception:
            pass

        config_hash = hashlib.sha256(json.dumps(sanitize_for_json(config_data), sort_keys=True).encode("utf-8")).hexdigest() if config_data else ""
        fixture_hash = hashlib.sha256(json.dumps(sanitize_for_json(fixture_data), sort_keys=True).encode("utf-8")).hexdigest() if fixture_data else ""
        trace_hash = hashlib.sha256(json.dumps(sanitize_for_json(trace_data), sort_keys=True).encode("utf-8")).hexdigest() if trace_data else ""

        return cls(
            git_sha=sha,
            git_dirty=dirty,
            command=command,
            evidence_level=evidence_level,
            seed=seed,
            dependency_versions=deps,
            benchmark_config_hash=config_hash,
            workload_fixture_hash=fixture_hash,
            raw_trace_hash=trace_hash,
            is_simulated=is_simulated,
        )


@dataclass
class TokenUsage:
    """Token consumption and monetary cost tracker."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: TokenUsage) -> TokenUsage:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.cost_usd += other.cost_usd
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenUsage:
        return cls(
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            total_tokens=int(data.get("total_tokens", 0)),
            cost_usd=float(data.get("cost_usd", 0.0)),
        )


@dataclass(frozen=True)
class ToolSpec:
    """Tool specification metadata."""
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_args: List[str] = field(default_factory=list)
    commit_horizon_args: List[str] = field(default_factory=list)
    is_read_only: bool = True
    is_idempotent: bool = True
    side_effects: bool = False
    requires_approval: bool = False
    estimated_latency_ms: float = 200.0

    def get_commit_args(self) -> Set[str]:
        if self.commit_horizon_args:
            return set(self.commit_horizon_args)
        if self.required_args:
            return set(self.required_args)
        return set(self.parameters.get("properties", {}).keys())


@dataclass
class ToolCall:
    """Representation of an agent's request to execute a tool."""
    tool_name: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_speculative: bool = False
    speculation_confidence: float = 1.0
    commit_horizon: float = 1.0
    emitted_at_token_index: Optional[int] = None
    committed_early: bool = False
    idempotency_key: Optional[str] = None
    requires_approval: bool = False
    is_approved: bool = False
    bytecode: Optional[Union[str, bytes]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tool_name and self.name:
            self.tool_name = self.name
        elif not self.name and self.tool_name:
            self.name = self.tool_name

    def key(self) -> str:
        return f"{self.tool_name}:{json.dumps(self.arguments, sort_keys=True)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "name": self.tool_name,
            "arguments": self.arguments,
            "call_id": self.call_id,
            "is_speculative": self.is_speculative,
            "speculation_confidence": self.speculation_confidence,
            "commit_horizon": self.commit_horizon,
            "committed_early": self.committed_early,
            "idempotency_key": self.idempotency_key,
            "requires_approval": self.requires_approval,
            "is_approved": self.is_approved,
            "bytecode": self.bytecode.hex() if isinstance(self.bytecode, bytes) else self.bytecode,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        tool_name = data.get("tool_name") or data.get("name", "")
        return cls(
            tool_name=tool_name,
            name=tool_name,
            arguments=dict(data.get("arguments", {})),
            call_id=data.get("call_id", str(uuid.uuid4())),
            is_speculative=bool(data.get("is_speculative", False)),
            speculation_confidence=float(data.get("speculation_confidence", 1.0)),
            commit_horizon=float(data.get("commit_horizon", 1.0)),
            committed_early=bool(data.get("committed_early", False)),
            idempotency_key=data.get("idempotency_key"),
            requires_approval=bool(data.get("requires_approval", False)),
            is_approved=bool(data.get("is_approved", False)),
            bytecode=data.get("bytecode"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ToolResult:
    """Output and performance metadata of a tool execution."""
    call_id: str
    tool_name: str = ""
    name: str = ""
    result: Any = None
    output: Any = None
    error: Optional[str] = None
    is_error: bool = False
    cached: bool = False
    speculated: bool = False
    cancelled: bool = False
    cache_timestamp: Optional[float] = None
    started_at: float = 0.0
    finished_at: float = 0.0
    execution_time_ns: int = 0
    execution_time_ms: float = 0.0
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tool_name and self.name:
            self.tool_name = self.name
        elif not self.name and self.tool_name:
            self.name = self.tool_name

        if self.result is None and self.output is not None:
            self.result = self.output
        elif self.output is None and self.result is not None:
            self.output = self.result

        if self.execution_time_ns == 0 and self.execution_time_ms > 0:
            self.execution_time_ns = int(self.execution_time_ms * 1_000_000)
        elif self.execution_time_ms == 0.0 and self.execution_time_ns > 0:
            self.execution_time_ms = self.execution_time_ns / 1_000_000.0

    @property
    def is_success(self) -> bool:
        return not self.is_error and self.error is None and not self.cancelled

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "name": self.tool_name,
            "result": self.result,
            "output": self.result,
            "error": self.error,
            "is_error": self.is_error,
            "cached": self.cached,
            "speculated": self.speculated,
            "cancelled": self.cancelled,
            "cache_timestamp": self.cache_timestamp,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "execution_time_ns": self.execution_time_ns,
            "execution_time_ms": self.execution_time_ms,
            "cost_usd": self.cost_usd,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolResult:
        tool_name = data.get("tool_name") or data.get("name", "")
        result = data.get("result") if "result" in data else data.get("output")
        return cls(
            call_id=data["call_id"],
            tool_name=tool_name,
            name=tool_name,
            result=result,
            output=result,
            error=data.get("error"),
            is_error=bool(data.get("is_error", False)),
            cached=bool(data.get("cached", False)),
            speculated=bool(data.get("speculated", False)),
            cancelled=bool(data.get("cancelled", False)),
            cache_timestamp=data.get("cache_timestamp"),
            started_at=float(data.get("started_at", 0.0)),
            finished_at=float(data.get("finished_at", 0.0)),
            execution_time_ns=int(data.get("execution_time_ns", 0)),
            execution_time_ms=float(data.get("execution_time_ms", 0.0)),
            cost_usd=float(data.get("cost_usd", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ExecutionEvent:
    """A discrete timestamped event on the execution timeline."""
    event_type: Union[EventType, str]
    timestamp: float = field(default_factory=time.perf_counter)
    timestamp_ns: int = field(default_factory=time.perf_counter_ns)
    task_id: str = "default"
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    call_id: Optional[str] = None
    duration_ms: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.data and self.details:
            self.data = self.details
        elif not self.details and self.data:
            self.details = self.data

    def to_dict(self) -> dict[str, Any]:
        ev_str = self.event_type.value if isinstance(self.event_type, EventType) else str(self.event_type)
        return {
            "event_id": self.event_id,
            "event_type": ev_str,
            "timestamp": self.timestamp,
            "timestamp_ns": self.timestamp_ns,
            "task_id": self.task_id,
            "call_id": self.call_id,
            "duration_ms": self.duration_ms,
            "data": self.data,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionEvent:
        event_type_str = data["event_type"]
        try:
            event_type = EventType(event_type_str)
        except ValueError:
            event_type = event_type_str
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=event_type,
            timestamp=float(data.get("timestamp", time.perf_counter())),
            timestamp_ns=int(data.get("timestamp_ns", 0)),
            task_id=data.get("task_id", "default"),
            call_id=data.get("call_id"),
            duration_ms=float(data.get("duration_ms", 0.0)),
            data=dict(data.get("data", {}) or data.get("details", {})),
            details=dict(data.get("details", {}) or data.get("data", {})),
        )


@dataclass
class LatencyProfile:
    """Synthetic or empirical latency profile parameters (milliseconds)."""
    model_decision_ms: float = 450.0
    model_final_ms: float = 300.0
    tool_ms: float = 600.0
    draft_model_ms: float = 70.0
    program_runtime_overhead_ms: float = 80.0
    cache_lookup_ms: float = 8.0
    token_decode_ms_per_token: float = 12.0
    tokens_per_tool_json: int = 150
    tokens_per_tool_bytecode: int = 25
    rate_limit_capacity: int = 10
    rate_limit_refill_per_sec: float = 20.0
    sigma: float = 0.45

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LatencyProfile:
        return cls(**{k: float(v) for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TaskInstance:
    """An individual workload task item for evaluation."""
    task_id: str
    workload_family: str = "default"
    workload_id: str = ""
    prompt: str = ""
    expected_tools: list[str] = field(default_factory=list)
    expected_output: Any = None
    expected_args: Optional[dict[str, Any]] = None
    parameters: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.workload_family and self.workload_id:
            self.workload_family = self.workload_id
        elif not self.workload_id and self.workload_family:
            self.workload_id = self.workload_family

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "workload_family": self.workload_family,
            "prompt": self.prompt,
            "expected_tools": self.expected_tools,
            "expected_output": self.expected_output,
            "expected_args": self.expected_args,
            "parameters": self.parameters,
            "context": self.context,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskInstance:
        return cls(
            task_id=data["task_id"],
            workload_family=data["workload_family"],
            prompt=data["prompt"],
            expected_tools=list(data.get("expected_tools", [])),
            expected_output=data.get("expected_output"),
            expected_args=data.get("expected_args"),
            parameters=dict(data.get("parameters", {})),
            context=dict(data.get("context", {})),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class Task:
    """Task definition with mandatory correctness validation."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    prompt: str = ""
    expected_output: Any = None
    validator: Optional[Callable[..., bool]] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self, actual_output: Any) -> bool:
        """Strict validation: Task cannot automatically pass if neither validator nor expected_output is provided."""
        if self.validator is not None:
            try:
                try:
                    return bool(self.validator(actual_output))
                except TypeError:
                    return bool(self.validator(self.expected_output, actual_output))
            except Exception:
                return False

        if self.expected_output is None:
            # Rule: A task without expected_output or a validator must not automatically pass!
            return False

        return self.expected_output == actual_output


@dataclass
class ExecutionTrace:
    """Complete chronological audit trace of an agent task execution."""
    task_id: str
    workload_family: str = "default"
    scheduler_name: str = "default"
    success: bool = False
    final_output: Any = None
    start_ns: int = 0
    end_ns: int = 0
    duration_ms: float = 0.0
    ccl_ms: Optional[float] = None
    start_time_ns: int = 0
    end_time_ns: int = 0
    events: list[ExecutionEvent] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    validator_result: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start_ns == 0 and self.start_time_ns > 0:
            self.start_ns = self.start_time_ns
        elif self.start_time_ns == 0 and self.start_ns > 0:
            self.start_time_ns = self.start_ns

        if self.end_ns == 0 and self.end_time_ns > 0:
            self.end_ns = self.end_time_ns
        elif self.end_time_ns == 0 and self.end_ns > 0:
            self.end_time_ns = self.end_ns

        if self.duration_ms == 0.0 and self.duration_ns > 0:
            self.duration_ms = self.duration_ns / 1_000_000.0

        if self.success and self.ccl_ms is None and self.duration_ms > 0:
            self.ccl_ms = self.duration_ms

    @property
    def duration_ns(self) -> int:
        if self.end_ns >= self.start_ns >= 0 and self.end_ns > 0:
            return self.end_ns - self.start_ns
        if self.end_time_ns >= self.start_time_ns >= 0 and self.end_time_ns > 0:
            return self.end_time_ns - self.start_time_ns
        return int(self.duration_ms * 1_000_000)

    def get_events_by_type(self, event_type: Union[EventType, str]) -> list[ExecutionEvent]:
        target = event_type.value if isinstance(event_type, EventType) else str(event_type)
        return [
            e for e in self.events
            if (e.event_type.value if isinstance(e.event_type, EventType) else str(e.event_type)) == target
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "workload_family": self.workload_family,
            "scheduler_name": self.scheduler_name,
            "success": self.success,
            "final_output": self.final_output,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "duration_ms": self.duration_ms,
            "ccl_ms": self.ccl_ms,
            "events": [e.to_dict() for e in self.events],
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "tool_results": [r.to_dict() for r in self.tool_results],
            "token_usage": self.token_usage.to_dict(),
            "validator_result": self.validator_result,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionTrace:
        return cls(
            task_id=data["task_id"],
            workload_family=data.get("workload_family", "default"),
            scheduler_name=data.get("scheduler_name", "default"),
            success=bool(data.get("success", False)),
            final_output=data.get("final_output"),
            start_ns=int(data.get("start_ns", 0) or data.get("start_time_ns", 0)),
            end_ns=int(data.get("end_ns", 0) or data.get("end_time_ns", 0)),
            start_time_ns=int(data.get("start_time_ns", 0) or data.get("start_ns", 0)),
            end_time_ns=int(data.get("end_time_ns", 0) or data.get("end_ns", 0)),
            duration_ms=float(data.get("duration_ms", 0.0)),
            ccl_ms=float(data["ccl_ms"]) if data.get("ccl_ms") is not None else None,
            events=[ExecutionEvent.from_dict(e) for e in data.get("events", [])],
            tool_calls=[ToolCall.from_dict(c) for c in data.get("tool_calls", [])],
            tool_results=[ToolResult.from_dict(r) for r in data.get("tool_results", [])],
            token_usage=TokenUsage.from_dict(data.get("token_usage", {})),
            validator_result=dict(data.get("validator_result", {})),
            metadata=dict(data.get("metadata", {})),
        )


class TaskValidator(ABC):
    """Abstract interface for validating task completion and exact correctness."""

    @abstractmethod
    def validate(
        self,
        task: Any,
        output: Any,
        trace: Optional[ExecutionTrace] = None,
    ) -> Tuple[bool, str, dict[str, Any]]:
        """Validate output and execution trace."""
        ...


class FunctionValidator(TaskValidator):
    """Adapter wrapping a validation callable."""

    def __init__(
        self,
        fn: Callable[[Any, Any, Optional[ExecutionTrace]], Tuple[bool, str, dict[str, Any]]],
    ):
        self._fn = fn

    def validate(
        self,
        task: Any,
        output: Any,
        trace: Optional[ExecutionTrace] = None,
    ) -> Tuple[bool, str, dict[str, Any]]:
        return self._fn(task, output, trace)


@dataclass
class GuardrailMetrics:
    """Comprehensive guardrail statistics measured across evaluation runs."""
    total_tasks: int = 0
    successful_tasks: int = 0
    exact_success: float = 0.0
    exact_accuracy: float = 0.0
    tool_selection_accuracy: Optional[float] = None
    argument_accuracy: Optional[float] = None
    total_tool_calls: int = 0
    unnecessary_calls: int = 0
    duplicated_calls: int = 0
    speculative_calls_launched: int = 0
    speculative_calls_hit: int = 0
    speculative_calls_wasted: int = 0
    speculative_calls_cancelled: int = 0
    speculative_cancelled: int = 0
    speculative_wasted: int = 0
    speculative_committed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_freshness_violations: int = 0
    unapproved_side_effects: int = 0
    unsafe_side_effects: int = 0
    rate_limit_failures: int = 0
    rate_limit_errors: int = 0
    peak_concurrency: int = 0
    total_model_input_tokens: int = 0
    total_model_output_tokens: int = 0
    total_model_calls: int = 0
    total_deopts: int = 0
    tool_cost_multiplier: float = 1.0
    cost_per_task_usd: float = 0.0
    total_cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def compute_derived(self) -> None:
        if self.total_tasks > 0:
            self.exact_accuracy = self.successful_tasks / self.total_tasks
            self.exact_success = self.exact_accuracy
        if self.total_tool_calls > 0:
            wasted = self.speculative_calls_wasted or self.speculative_wasted
            base_calls = self.total_tool_calls - wasted
            if base_calls > 0:
                self.tool_cost_multiplier = self.total_tool_calls / base_calls
            else:
                self.tool_cost_multiplier = 1.0

    def to_dict(self) -> dict[str, Any]:
        self.compute_derived()
        return sanitize_for_json(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GuardrailMetrics:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DependencyNode:
    """DAG dependency node structure."""
    call_id: str
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    output: Any = None
    status: str = "pending"


@dataclass
class WorkloadSpec:
    """Specification of a workload experiment."""
    name: str
    family: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    num_tasks: int = 100
    concurrency: int = 10
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkloadSpec:
        return cls(
            name=data["name"],
            family=data["family"],
            description=data.get("description", ""),
            parameters=dict(data.get("parameters", {})),
            num_tasks=int(data.get("num_tasks", 100)),
            concurrency=int(data.get("concurrency", 10)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class TaskResult:
    task_id: str
    success: bool
    final_answer: Any = None
    ccl_ms: Optional[float] = None
    total_duration_ms: float = 0.0
    events: List[ExecutionEvent] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    guardrails: GuardrailMetrics = field(default_factory=GuardrailMetrics)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "final_answer": self.final_answer,
            "ccl_ms": self.ccl_ms,
            "total_duration_ms": self.total_duration_ms,
            "tool_calls_count": len(self.tool_calls),
            "tool_results_count": len(self.tool_results),
            "guardrails": self.guardrails.to_dict(),
            "error": self.error,
            "metadata": self.metadata,
        }
