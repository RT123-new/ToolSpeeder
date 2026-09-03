"""Core data structures and types for ToolSpeed benchmark framework."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import platform
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


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
    SPECULATIVE_LAUNCH = "speculation_start"
    SPECULATIVE_DISPATCH = "speculation_start"
    SPECULATION_HIT = "speculation_hit"
    SPECULATIVE_HIT = "speculation_hit"
    SPECULATION_MISS = "speculation_miss"
    SPECULATIVE_MISS = "speculation_miss"
    SPECULATION_CANCELLED = "speculation_cancelled"
    SPECULATIVE_CANCEL = "speculation_cancelled"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    CACHE_FRESHNESS_VIOLATION = "cache_freshness_violation"
    DAG_NODE_READY = "dag_node_ready"
    DAG_NODE_DISPATCH = "dag_node_dispatch"
    JIT_FUSION_START = "jit_fusion_start"
    JIT_FUSION_SUCCESS = "jit_fusion_success"
    FUSION_HIT = "jit_fusion_success"
    JIT_FUSION_DEOPT = "jit_fusion_deopt"
    FUSION_DEOPT = "jit_fusion_deopt"
    DEOPTIMIZATION = "jit_fusion_deopt"
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
    EARLY_DISPATCH = "commit_horizon_reached"
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
    elif hasattr(obj, "to_dict") and callable(obj.to_dict):
        return sanitize_for_json(obj.to_dict())
    return obj


def strict_json_dumps(obj: Any, indent: int | None = 2) -> str:
    """Strict JSON serializer enforcing standards compliance (no NaN / Infinity)."""
    sanitized = sanitize_for_json(obj)
    return json.dumps(sanitized, indent=indent, allow_nan=False)


def compute_file_sha256(file_path: Any) -> str:
    """Compute SHA-256 over exact file bytes."""
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Cannot compute hash for missing file: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


MODEL_VISIBLE_METADATA_WHITELIST: set[str] = {
    "workload_id",
    "workload_family",
    "trial_index",
    "fan_out_width",
    "user_id",
    "server_count",
    "customer_id",
    "sku",
    "dataset_id",
    "operation",
    "workflow_id",
}

PROHIBITED_METADATA_SUBSTRINGS: list[str] = [
    "approval",
    "grant",
    "expected",
    "oracle",
    "secret",
    "validator",
    "ground_truth",
    "canary",
]


def filter_model_visible_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Filters metadata strictly: only admitted if key is in MODEL_VISIBLE_METADATA_WHITELIST."""
    if not metadata:
        return {}
    result: dict[str, Any] = {}
    for k, v in metadata.items():
        if k not in MODEL_VISIBLE_METADATA_WHITELIST:
            continue
        k_lower = str(k).lower()
        if any(p in k_lower for p in PROHIBITED_METADATA_SUBSTRINGS):
            continue
        # Never expose ApprovalGrant or ExpectedOutcome objects
        if hasattr(v, "approval_id") or hasattr(v, "expected_final_value") or hasattr(v, "oracle_canary"):
            continue
        if any(p in str(v).lower() for p in PROHIBITED_METADATA_SUBSTRINGS):
            continue
        result[k] = v
    return result


def sanitize_model_visible_data(data: Any) -> Any:
    """Recursively scans and sanitizes context/parameters/metadata, removing canaries and prohibited terms."""
    if isinstance(data, (dict, Mapping)):
        clean_dict: dict[str, Any] = {}
        for k, v in data.items():
            k_str = str(k).lower()
            if any(p in k_str for p in PROHIBITED_METADATA_SUBSTRINGS):
                continue
            if hasattr(v, "approval_id") or hasattr(v, "expected_final_value") or hasattr(v, "oracle_canary"):
                continue
            clean_dict[k] = sanitize_model_visible_data(v)
        return clean_dict
    elif isinstance(data, (list, tuple)):
        clean_list = []
        for item in data:
            if hasattr(item, "approval_id") or hasattr(item, "expected_final_value") or hasattr(item, "oracle_canary"):
                continue
            clean_list.append(sanitize_model_visible_data(item))
        return type(data)(clean_list)
    elif isinstance(data, str):
        lower = data.lower()
        if "oracle_canary" in lower or "canary_" in lower or "secret_canary" in lower:
            return "[REDACTED_ORACLE_CANARY]"
        return data
    return data


@dataclass(frozen=True)
class AgentTask:
    """Task input presented strictly to the agent/model (NO oracle or expected outputs)."""

    task_id: str
    prompt: str
    workload_family: str = "default"
    context: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "workload_family": self.workload_family,
            "context": dict(self.context),
            "parameters": dict(self.parameters),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentTask:
        return cls(
            task_id=data["task_id"],
            prompt=data.get("prompt", ""),
            workload_family=data.get("workload_family", "default"),
            context=dict(data.get("context", {})),
            parameters=dict(data.get("parameters", {})),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class StateSnapshot:
    """Immutable state snapshot for environment initialization and validation."""

    state_id: str = "default_state"
    namespace: str = "default"
    data: Mapping[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        state_id_or_data: str | Mapping[str, Any] | None = None,
        namespace: str = "default",
        data: Mapping[str, Any] | None = None,
        state_id: str | None = None,
    ) -> None:
        if isinstance(state_id_or_data, (dict, Mapping)):
            actual_data = dict(state_id_or_data)
            actual_id = state_id or str(uuid.uuid4())[:8]
        else:
            actual_id = state_id or (str(state_id_or_data) if state_id_or_data is not None else "default_state")
            actual_data = dict(data or {})
        object.__setattr__(self, "state_id", actual_id)
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "data", actual_data)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def clone(self) -> StateSnapshot:
        return StateSnapshot(
            state_id_or_data=str(uuid.uuid4()),
            namespace=self.namespace,
            data=copy.deepcopy(dict(self.data)),
        )

    def compute_hash(self) -> str:
        return hashlib.sha256(json.dumps(dict(self.data), sort_keys=True).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "namespace": self.namespace,
            "data": dict(self.data),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateSnapshot:
        return cls(
            state_id=data.get("state_id", str(uuid.uuid4())),
            namespace=data.get("namespace", "default"),
            data=dict(data.get("data", {})),
        )


@dataclass(frozen=True)
class ExpectedOutcome:
    """Oracle ground-truth specifications for task validation (NEVER given to the model)."""

    expected_final_value: Any = None
    expected_tool_sequence: Sequence[str] = field(default_factory=tuple)
    expected_tool_arguments: Mapping[str, Any] = field(default_factory=dict)
    expected_arguments: Mapping[str, Any] = field(default_factory=dict)
    expected_state_diff: Mapping[str, Any] = field(default_factory=dict)
    expected_final_state: Mapping[str, Any] = field(default_factory=dict)
    required_tools: Sequence[str] = field(default_factory=tuple)
    disallowed_tools: Sequence[str] = field(default_factory=tuple)
    required_mutations: int = 0
    max_allowed_calls: int = 100
    oracle_canary: str = "ORACLE_CANARY_SECRET_789XYZ"

    def to_dict(self) -> dict[str, Any]:
        exp_args = dict(self.expected_tool_arguments)
        if self.expected_arguments:
            exp_args.update(self.expected_arguments)
        return {
            "expected_final_value": self.expected_final_value,
            "expected_tool_sequence": list(self.expected_tool_sequence),
            "expected_tool_arguments": exp_args,
            "expected_arguments": dict(self.expected_arguments),
            "expected_state_diff": dict(self.expected_state_diff),
            "expected_final_state": dict(self.expected_final_state),
            "required_tools": list(self.required_tools),
            "disallowed_tools": list(self.disallowed_tools),
            "required_mutations": self.required_mutations,
            "max_allowed_calls": self.max_allowed_calls,
            "oracle_canary": self.oracle_canary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpectedOutcome:
        return cls(
            expected_final_value=data.get("expected_final_value"),
            expected_tool_sequence=tuple(data.get("expected_tool_sequence", ())),
            expected_tool_arguments=dict(data.get("expected_tool_arguments", {})),
            expected_arguments=dict(data.get("expected_arguments", {})),
            expected_state_diff=dict(data.get("expected_state_diff", {})),
            expected_final_state=dict(data.get("expected_final_state", {})),
            required_tools=tuple(data.get("required_tools", ())),
            disallowed_tools=tuple(data.get("disallowed_tools", ())),
            required_mutations=int(data.get("required_mutations", 0)),
            max_allowed_calls=int(data.get("max_allowed_calls", 100)),
            oracle_canary=data.get("oracle_canary", "ORACLE_CANARY_SECRET_789XYZ"),
        )


DEFAULT_ISSUER_SECRET: bytes = b"toolspeed_trusted_authority_hmac_secret_key_32b_fixed"


@dataclass(frozen=True)
class ApprovalGrant:
    """Cryptographically verifiable authorization grant for side-effect execution."""

    approval_id: str
    subject: str
    tool_name: str
    argument_fingerprint: str
    expires_at: float
    authority: str
    nonce: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    signature: str = ""
    tenant: str = "default_tenant"
    run_id: str = "default_run"
    single_use: bool = True

    @staticmethod
    def compute_fingerprint(tool_name: str, arguments: Mapping[str, Any]) -> str:
        fp_payload = f"{tool_name}:{json.dumps(dict(arguments), sort_keys=True)}".encode()
        return hashlib.sha256(fp_payload).hexdigest()

    @staticmethod
    def compute_signature(
        secret: bytes,
        approval_id: str,
        tool_name: str,
        argument_fingerprint: str,
        expires_at: float,
        authority: str,
        nonce: str,
        tenant: str,
        run_id: str,
    ) -> str:
        msg = f"{approval_id}:{tool_name}:{argument_fingerprint}:{expires_at:.6f}:{authority}:{nonce}:{tenant}:{run_id}".encode()
        return hmac.new(secret, msg, hashlib.sha256).hexdigest()

    @classmethod
    def create(
        cls,
        tool_name: str,
        arguments: Mapping[str, Any],
        authority: str = "trusted_system",
        ttl_seconds: float = 300.0,
        subject: str = "default_subject",
        tenant: str = "default_tenant",
        run_id: str = "default_run",
        single_use: bool = True,
        current_time: float | None = None,
        issuer_secret: bytes = DEFAULT_ISSUER_SECRET,
    ) -> ApprovalGrant:
        fp = cls.compute_fingerprint(tool_name, arguments)
        now = current_time if current_time is not None else time.time()
        aid = str(uuid.uuid4())
        nonce = str(uuid.uuid4())[:16]
        exp = now + ttl_seconds
        sig = cls.compute_signature(issuer_secret, aid, tool_name, fp, exp, authority, nonce, tenant, run_id)
        return cls(
            approval_id=aid,
            subject=subject,
            tool_name=tool_name,
            argument_fingerprint=fp,
            expires_at=exp,
            authority=authority,
            nonce=nonce,
            signature=sig,
            tenant=tenant,
            run_id=run_id,
            single_use=single_use,
        )

    def matches(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        subject: str | None = None,
        tenant: str | None = None,
        run_id: str | None = None,
        current_time: float | None = None,
        allowed_authorities: Sequence[str] = ("trusted_system", "human_supervisor"),
        issuer_secret: bytes = DEFAULT_ISSUER_SECRET,
    ) -> bool:
        if self.authority not in allowed_authorities:
            return False
        if self.tool_name != tool_name:
            return False
        if subject is not None and self.subject != subject:
            return False
        if tenant is not None and self.tenant != tenant:
            return False
        if run_id is not None and self.run_id != run_id:
            return False
        now = current_time if current_time is not None else time.time()
        if now > self.expires_at:
            return False
        fp = self.compute_fingerprint(tool_name, arguments)
        if self.argument_fingerprint != fp and self.argument_fingerprint != fp[:16]:
            return False
        if self.signature:
            expected_sig = self.compute_signature(
                issuer_secret,
                self.approval_id,
                self.tool_name,
                self.argument_fingerprint,
                self.expires_at,
                self.authority,
                self.nonce,
                self.tenant,
                self.run_id,
            )
            if not hmac.compare_digest(self.signature, expected_sig):
                return False
        return True


class ApprovalIssuer:
    """Trusted capability issuer generating signed approval grants."""

    def __init__(self, secret: bytes = DEFAULT_ISSUER_SECRET, authority: str = "trusted_system"):
        self.secret = secret
        self.authority = authority

    def issue(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        ttl_seconds: float = 300.0,
        subject: str = "default_subject",
        tenant: str = "default_tenant",
        run_id: str = "default_run",
        single_use: bool = True,
        current_time: float | None = None,
    ) -> ApprovalGrant:
        return ApprovalGrant.create(
            tool_name=tool_name,
            arguments=arguments,
            authority=self.authority,
            ttl_seconds=ttl_seconds,
            subject=subject,
            tenant=tenant,
            run_id=run_id,
            single_use=single_use,
            current_time=current_time,
            issuer_secret=self.secret,
        )


@dataclass
class ExecutionAuthorityContext:
    """Opaque trusted capability store attached to the execution context (NEVER passed to the model)."""

    tenant: str = "default_tenant"
    run_id: str = "default_run"
    subject: str = "default_subject"
    allowed_authorities: list[str] = field(default_factory=lambda: ["trusted_system", "human_supervisor"])
    grants: list[ApprovalGrant] = field(default_factory=list)
    consumed_grant_ids: set[str] = field(default_factory=set)
    issuer_secret: bytes = DEFAULT_ISSUER_SECRET

    def add_grant(self, grant: ApprovalGrant) -> None:
        self.grants.append(grant)

    def verify_and_consume_grant(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        current_time: float | None = None,
    ) -> bool:
        for grant in self.grants:
            if grant.approval_id in self.consumed_grant_ids:
                continue
            if grant.matches(
                tool_name=tool_name,
                arguments=arguments,
                subject=self.subject,
                tenant=self.tenant,
                run_id=self.run_id,
                current_time=current_time,
                allowed_authorities=self.allowed_authorities,
            ):
                if grant.single_use:
                    self.consumed_grant_ids.add(grant.approval_id)
                return True
        return False


@dataclass(frozen=True)
class BenchmarkCase:
    """Immutable paired benchmark case separating model inputs from oracle verification."""

    case_id: str
    workload_id: str
    agent_task: AgentTask
    expected_outcome: ExpectedOutcome
    authority_context: ExecutionAuthorityContext = field(default_factory=ExecutionAuthorityContext)
    initial_state: StateSnapshot = field(default_factory=lambda: StateSnapshot(state_id="init_state"))
    fixture: Mapping[str, Any] = field(default_factory=dict)
    seed: int = 0
    trial_index: int = 0
    parameters: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        case_id: str | None = None,
        workload_id: str = "default",
        agent_task: AgentTask | None = None,
        expected_outcome: ExpectedOutcome | None = None,
        authority_context: ExecutionAuthorityContext | None = None,
        initial_state: StateSnapshot | None = None,
        fixture: Mapping[str, Any] | None = None,
        seed: int = 0,
        trial_index: int = 0,
        parameters: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        task: AgentTask | None = None,
        task_id: str | None = None,
        prompt: str = "",
        expected_output: Any = None,
        validator: Any = None,
        context: Mapping[str, Any] | None = None,
        expected_tools: Sequence[str] | None = None,
        expected_args: Mapping[str, Any] | None = None,
    ) -> None:
        cid = case_id or task_id or str(uuid.uuid4())[:8]
        p_dict = dict(parameters or {})
        c_dict = dict(context or {})
        m_dict = dict(metadata or {})
        actual_task = (
            agent_task
            or task
            or AgentTask(
                task_id=task_id or cid,
                prompt=prompt,
                workload_family=workload_id,
                context=c_dict,
                parameters=p_dict,
                metadata=filter_model_visible_metadata(m_dict),
            )
        )
        if expected_outcome is None:
            expected_outcome = ExpectedOutcome(
                expected_final_value=expected_output,
                required_tools=tuple(expected_tools or ()),
                expected_arguments=dict(expected_args or {}),
            )
        object.__setattr__(self, "case_id", cid)
        object.__setattr__(self, "workload_id", workload_id)
        object.__setattr__(self, "agent_task", actual_task)
        object.__setattr__(self, "expected_outcome", expected_outcome)
        object.__setattr__(self, "authority_context", authority_context or ExecutionAuthorityContext())
        object.__setattr__(self, "initial_state", initial_state or StateSnapshot(state_id="init_state"))
        object.__setattr__(self, "fixture", dict(fixture or {}))
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "trial_index", trial_index)
        object.__setattr__(self, "parameters", p_dict)
        object.__setattr__(self, "metadata", m_dict)

    @property
    def task(self) -> AgentTask:
        return self.agent_task

    @property
    def task_id(self) -> str:
        return self.agent_task.task_id if self.agent_task else self.case_id

    @property
    def prompt(self) -> str:
        return self.agent_task.prompt if self.agent_task else ""

    @property
    def expected_output(self) -> Any:
        return self.expected_outcome.expected_final_value if self.expected_outcome else None

    def to_model_task(self) -> Task:
        """Constructs an isolated Task with zero access to expected output or validators."""
        c = self.agent_task.context if self.agent_task else {}
        p = self.agent_task.parameters if self.agent_task else self.parameters
        m = self.agent_task.metadata if self.agent_task else self.metadata
        return Task(
            task_id=self.task_id,
            prompt=self.prompt,
            context=sanitize_model_visible_data(copy.deepcopy(dict(c))),
            parameters=sanitize_model_visible_data(copy.deepcopy(dict(p))),
            metadata=filter_model_visible_metadata(dict(m)),
            expected_output=None,
            validator=None,
        )

    def to_agent_task(self) -> AgentTask:
        """Constructs a model-visible AgentTask strictly stripped of test oracles."""
        return self.agent_task

    def validate(
        self,
        actual_output: Any,
        trace: Any = None,
        initial_state: StateSnapshot | None = None,
        final_state: StateSnapshot | None = None,
    ) -> bool:
        """Strict validation executed outside scheduler visibility."""
        passed, _, _ = self.validate_execution(actual_output, trace=trace, final_state=final_state)
        return passed

    @classmethod
    def from_task(cls, task: Task) -> BenchmarkCase:
        return cls(
            case_id=task.task_id,
            workload_id=task.metadata.get("workload_id", "default"),
            agent_task=task.to_agent_task(),
            expected_outcome=ExpectedOutcome(
                expected_final_value=task.expected_output,
                required_tools=tuple(task.metadata.get("required_tools", task.metadata.get("expected_tools", ()))),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "workload_id": self.workload_id,
            "agent_task": self.agent_task.to_dict(),
            "expected_outcome": self.expected_outcome.to_dict(),
            "initial_state": self.initial_state.to_dict() if self.initial_state else {},
            "fixture": dict(self.fixture),
            "seed": self.seed,
            "trial_index": self.trial_index,
            "parameters": dict(self.parameters),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkCase:
        return cls(
            case_id=data["case_id"],
            workload_id=data.get("workload_id", ""),
            agent_task=AgentTask.from_dict(data["agent_task"]),
            expected_outcome=ExpectedOutcome.from_dict(data.get("expected_outcome", {})),
            authority_context=ExecutionAuthorityContext(),
            initial_state=StateSnapshot.from_dict(data.get("initial_state", {})),
            fixture=dict(data.get("fixture", {})),
            seed=int(data.get("seed", 0)),
            trial_index=int(data.get("trial_index", 0)),
            parameters=dict(data.get("parameters", {})),
            metadata=dict(data.get("metadata", {})),
        )

    def validate_execution(
        self,
        final_output: Any,
        trace: Any = None,
        final_state: StateSnapshot | None = None,
    ) -> tuple[bool, str, dict[str, Any]]:
        """Oracle validation: checks final output, required tool calls, dependencies, prohibited calls, side-effects."""
        details: dict[str, Any] = {
            "output_matched": False,
            "required_tools_present": True,
            "no_disallowed_tools": True,
            "tool_calls_count": 0,
            "errors": [],
        }

        # 1. Check required tools executed in trace
        if self.expected_outcome.required_tools:
            if trace is None or not getattr(trace, "tool_calls", None):
                details["required_tools_present"] = False
                details["errors"].append("Trace missing tool calls for required tools")
                return False, "Required tools were not executed", details

            executed_tool_names = [getattr(c, "name", None) or getattr(c, "tool_name", "") for c in trace.tool_calls]
            for req in self.expected_outcome.required_tools:
                if req not in executed_tool_names:
                    details["required_tools_present"] = False
                    details["errors"].append(f"Required tool '{req}' was not executed")
                    return False, f"Required tool '{req}' was omitted", details

        # 1b. Check expected tool sequence if declared
        if self.expected_outcome.expected_tool_sequence:
            if trace is None or not getattr(trace, "tool_calls", None):
                details["errors"].append("Trace missing tool calls for expected sequence")
                return False, "Expected tool sequence not executed", details
            executed_tool_names = [getattr(c, "name", None) or getattr(c, "tool_name", "") for c in trace.tool_calls]
            exp_seq = list(self.expected_outcome.expected_tool_sequence)
            if list(executed_tool_names[: len(exp_seq)]) != exp_seq:
                details["errors"].append(f"Expected tool sequence {exp_seq}, got {executed_tool_names}")
                return False, "Executed tool sequence did not match expected sequence", details

        # 1c. Check expected arguments if declared
        exp_args_map = dict(self.expected_outcome.expected_tool_arguments)
        if self.expected_outcome.expected_arguments:
            exp_args_map.update(self.expected_outcome.expected_arguments)
        if exp_args_map and trace is not None and getattr(trace, "tool_calls", None):
            for call in trace.tool_calls:
                t_name = getattr(call, "name", None) or getattr(call, "tool_name", "")
                if t_name in exp_args_map:
                    exp_args = exp_args_map[t_name]
                    actual_args = getattr(call, "arguments", {})
                    if actual_args != exp_args:
                        details["errors"].append(
                            f"Arguments mismatch on {t_name}: expected {exp_args}, got {actual_args}"
                        )
                        return False, f"Tool arguments for '{t_name}' did not match expected", details

        # 1d. Check required mutations
        if self.expected_outcome.required_mutations > 0 and trace is not None and getattr(trace, "tool_calls", None):
            mut_calls = [c for c in trace.tool_calls if not getattr(c, "is_read_only", True)]
            if len(mut_calls) != self.expected_outcome.required_mutations:
                details["errors"].append(
                    f"Expected {self.expected_outcome.required_mutations} mutations, found {len(mut_calls)}"
                )
                return (
                    False,
                    f"Mutation count {len(mut_calls)} did not match expected {self.expected_outcome.required_mutations}",
                    details,
                )

        # 1e. Check expected state diff or final state
        exp_state = dict(self.expected_outcome.expected_final_state)
        if self.expected_outcome.expected_state_diff:
            exp_state.update(self.expected_outcome.expected_state_diff)
        if exp_state and final_state is not None:
            state_dict = (
                final_state.data
                if hasattr(final_state, "data")
                else (final_state if isinstance(final_state, dict) else {})
            )
            for k, v in exp_state.items():
                if state_dict.get(k) != v:
                    details["errors"].append(f"State mismatch on '{k}': expected {v}, got {state_dict.get(k)}")
                    return False, f"Final state did not match expected state for '{k}'", details

        # 2. Check disallowed tools not executed
        if self.expected_outcome.disallowed_tools and trace is not None and getattr(trace, "tool_calls", None):
            executed_tool_names = [getattr(c, "name", None) or getattr(c, "tool_name", "") for c in trace.tool_calls]
            for dis in self.expected_outcome.disallowed_tools:
                if dis in executed_tool_names:
                    details["no_disallowed_tools"] = False
                    details["errors"].append(f"Disallowed tool '{dis}' was executed")
                    return False, f"Disallowed tool '{dis}' was executed", details

        # 3. Check tool result errors
        if trace is not None and getattr(trace, "tool_results", None):
            details["tool_calls_count"] = len(trace.tool_calls)
            for r in trace.tool_results:
                is_err = getattr(r, "is_error", False)
                err_msg = getattr(r, "error", None)
                is_canc = getattr(r, "cancelled", False)
                is_spec = getattr(r, "speculated", False)
                if (is_err or err_msg is not None) and not is_canc and not is_spec:
                    t_name = getattr(r, "tool_name", "") or getattr(r, "name", "")
                    details["errors"].append(f"Tool execution error on {t_name}: {err_msg}")
                    return False, f"Tool execution failed: {err_msg}", details

        # 4. Check max allowed calls
        if (
            trace is not None
            and getattr(trace, "tool_calls", None)
            and len(trace.tool_calls) > self.expected_outcome.max_allowed_calls
        ):
            details["errors"].append(
                f"Tool call count {len(trace.tool_calls)} exceeded max {self.expected_outcome.max_allowed_calls}"
            )
            return False, "Exceeded maximum allowed tool calls", details

        # 5. Check expected final value (exact equality enforced; partial supersets rejected)
        if (
            self.expected_outcome.expected_final_value is not None
            and final_output != self.expected_outcome.expected_final_value
        ):
            details["errors"].append(
                f"Output mismatch: expected {self.expected_outcome.expected_final_value}, got {final_output}"
            )
            return False, "Final output did not match expected outcome", details

        details["output_matched"] = True
        return True, "Validation successful", details


@dataclass(frozen=True)
class CommittedCall:
    """Immutable representation of a tool call that has safely crossed the commit horizon."""

    tool_name: str
    arguments: Mapping[str, Any]
    call_id: str
    schema_hash: str
    semantic_fingerprint: str
    token_index: int = 0
    byte_offset: int = 0

    @classmethod
    def from_call(
        cls,
        call: ToolCall,
        schema_hash: str = "",
        token_index: int = 0,
        byte_offset: int = 0,
    ) -> CommittedCall:
        t_name = call.name or call.tool_name
        args_copy = copy.deepcopy(call.arguments)
        fp_payload = f"{t_name}:{json.dumps(args_copy, sort_keys=True)}:{schema_hash}".encode()
        fingerprint = hashlib.sha256(fp_payload).hexdigest()
        return cls(
            tool_name=t_name,
            arguments=args_copy,
            call_id=call.call_id,
            schema_hash=schema_hash,
            semantic_fingerprint=fingerprint,
            token_index=token_index,
            byte_offset=byte_offset,
        )


@dataclass
class ArtifactManifest:
    """Provenance and execution environment metadata for benchmark artifact bundles."""

    git_sha: str = "unknown"
    git_tree_sha: str = "unknown"
    git_dirty: bool = False
    workflow_run_id: str = ""
    workflow_job_id: str = ""
    command: str = ""
    evidence_level: EvidenceLevel = EvidenceLevel.SYNTHETIC
    timestamp_utc: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    seed: int = 0
    python_version: str = platform.python_version()
    os_platform: str = f"{platform.system()} {platform.release()} ({platform.machine()})"
    hardware_info: dict[str, Any] = field(default_factory=dict)
    dependency_versions: dict[str, str] = field(default_factory=dict)
    benchmark_plan_hash: str = ""
    benchmark_config_hash: str = ""
    fixture_manifest_hash: str = ""
    workload_fixture_hash: str = ""
    cases_hash: str = ""
    baseline_trace_hash: str = ""
    candidate_trace_hash: str = ""
    raw_trace_hash: str = ""
    controls_trace_hash: str = ""
    result_hash: str = ""
    falsification_hash: str = ""
    file_hashes: dict[str, str] = field(default_factory=dict)
    report_generator_version: str = "2.0.0"
    is_simulated: bool = False
    is_verdict_eligible: bool = False
    trial_count: int = 0
    warmup_count: int = 0
    resource_topology: dict[str, Any] = field(default_factory=dict)
    required_metric_policy_version: str = "2.0.0"

    @property
    def commit_sha(self) -> str:
        return self.git_sha

    @property
    def dirty(self) -> bool:
        return self.git_dirty

    def to_dict(self) -> dict[str, Any]:
        return {
            "git_sha": self.git_sha,
            "code_git_sha": self.git_sha,
            "git_tree_sha": self.git_tree_sha,
            "git_dirty": self.git_dirty,
            "workflow_run_id": self.workflow_run_id,
            "workflow_job_id": self.workflow_job_id,
            "command": self.command,
            "evidence_level": self.evidence_level.value
            if isinstance(self.evidence_level, EvidenceLevel)
            else str(self.evidence_level),
            "timestamp_utc": self.timestamp_utc,
            "seed": self.seed,
            "python_version": self.python_version,
            "os_platform": self.os_platform,
            "hardware_info": self.hardware_info,
            "dependency_versions": self.dependency_versions,
            "benchmark_plan_hash": self.benchmark_plan_hash,
            "benchmark_config_hash": self.benchmark_plan_hash,
            "fixture_manifest_hash": self.fixture_manifest_hash or self.cases_hash,
            "workload_fixture_hash": self.workload_fixture_hash or self.fixture_manifest_hash or self.cases_hash,
            "cases_hash": self.cases_hash,
            "baseline_trace_hash": self.baseline_trace_hash,
            "candidate_trace_hash": self.candidate_trace_hash,
            "raw_trace_hash": self.candidate_trace_hash or self.baseline_trace_hash,
            "controls_trace_hash": self.controls_trace_hash,
            "result_hash": self.result_hash,
            "falsification_hash": self.falsification_hash,
            "file_hashes": self.file_hashes,
            "report_generator_version": self.report_generator_version,
            "is_simulated": self.is_simulated,
            "is_verdict_eligible": self.is_verdict_eligible,
            "trial_count": self.trial_count,
            "warmup_count": self.warmup_count,
            "resource_topology": self.resource_topology,
            "required_metric_policy_version": self.required_metric_policy_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactManifest:
        ev_level_raw = data.get("evidence_level", "synthetic")
        try:
            ev_level = EvidenceLevel(ev_level_raw)
        except ValueError:
            ev_level = EvidenceLevel.SYNTHETIC
        return cls(
            git_sha=data.get("git_sha") or data.get("code_git_sha", "unknown"),
            git_tree_sha=data.get("git_tree_sha", "unknown"),
            git_dirty=bool(data.get("git_dirty", False)),
            workflow_run_id=data.get("workflow_run_id", ""),
            workflow_job_id=data.get("workflow_job_id", ""),
            command=data.get("command", ""),
            evidence_level=ev_level,
            timestamp_utc=data.get("timestamp_utc", ""),
            seed=int(data.get("seed", 0)),
            python_version=data.get("python_version", platform.python_version()),
            os_platform=data.get("os_platform", ""),
            hardware_info=dict(data.get("hardware_info", {})),
            dependency_versions=dict(data.get("dependency_versions", {})),
            benchmark_plan_hash=data.get("benchmark_plan_hash", ""),
            fixture_manifest_hash=data.get("fixture_manifest_hash", ""),
            cases_hash=data.get("cases_hash", ""),
            baseline_trace_hash=data.get("baseline_trace_hash", ""),
            candidate_trace_hash=data.get("candidate_trace_hash", ""),
            controls_trace_hash=data.get("controls_trace_hash", ""),
            result_hash=data.get("result_hash", ""),
            falsification_hash=data.get("falsification_hash", ""),
            file_hashes=dict(data.get("file_hashes", {})),
            report_generator_version=data.get("report_generator_version", "2.0.0"),
            is_simulated=bool(data.get("is_simulated", False)),
            is_verdict_eligible=bool(data.get("is_verdict_eligible", False)),
            trial_count=int(data.get("trial_count", 0)),
            warmup_count=int(data.get("warmup_count", 0)),
            resource_topology=dict(data.get("resource_topology", {})),
            required_metric_policy_version=data.get("required_metric_policy_version", "2.0.0"),
        )

    @classmethod
    def create(
        cls,
        evidence_level: EvidenceLevel = EvidenceLevel.SYNTHETIC,
        seed: int = 42,
        command: str = "toolspeed",
        is_simulated: bool = False,
        is_verdict_eligible: bool = False,
        trial_count: int = 0,
        warmup_count: int = 0,
        resource_topology: dict[str, Any] | None = None,
        file_hashes: dict[str, str] | None = None,
        workflow_run_id: str = "",
        workflow_job_id: str = "",
    ) -> ArtifactManifest:
        sha = "unknown"
        tree_sha = "unknown"
        dirty = False
        try:
            import subprocess

            res_sha = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2, check=False
            )
            if res_sha.returncode == 0:
                sha = res_sha.stdout.strip()
            res_tree = subprocess.run(
                ["git", "rev-parse", "HEAD^{tree}"], capture_output=True, text=True, timeout=2, check=False
            )
            if res_tree.returncode == 0:
                tree_sha = res_tree.stdout.strip()
            res_status = subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=2, check=False
            )
            if res_status.returncode == 0:
                dirty = len(res_status.stdout.strip()) > 0
        except Exception:
            pass

        deps: dict[str, str] = {}
        try:
            import numpy

            deps["numpy"] = numpy.__version__
        except Exception:
            pass

        hashes = file_hashes or {}

        return cls(
            git_sha=sha,
            git_tree_sha=tree_sha,
            git_dirty=dirty,
            workflow_run_id=workflow_run_id,
            workflow_job_id=workflow_job_id,
            command=command,
            evidence_level=evidence_level,
            seed=seed,
            dependency_versions=deps,
            benchmark_plan_hash=hashes.get("benchmark_plan.json", ""),
            fixture_manifest_hash=hashes.get("fixture_manifest.json", ""),
            cases_hash=hashes.get("cases.jsonl", ""),
            baseline_trace_hash=hashes.get("baseline_traces.jsonl", ""),
            candidate_trace_hash=hashes.get("candidate_traces.jsonl", ""),
            controls_trace_hash=hashes.get("controls_traces.jsonl", ""),
            result_hash=hashes.get("result.json", ""),
            falsification_hash=hashes.get("falsification.json", ""),
            file_hashes=hashes,
            is_simulated=is_simulated,
            is_verdict_eligible=is_verdict_eligible,
            trial_count=trial_count,
            warmup_count=warmup_count,
            resource_topology=resource_topology or {},
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
    parameters: dict[str, Any] = field(default_factory=dict)
    required_args: list[str] = field(default_factory=list)
    commit_horizon_args: list[str] = field(default_factory=list)
    is_read_only: bool = True
    is_idempotent: bool = True
    side_effects: bool = False
    requires_approval: bool = False
    estimated_latency_ms: float = 200.0

    def get_commit_args(self) -> set[str]:
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
    emitted_at_token_index: int | None = None
    committed_early: bool = False
    idempotency_key: str | None = None
    requires_approval: bool = False
    is_approved: bool = False
    bytecode: str | bytes | None = None
    approval_grant: ApprovalGrant | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tool_name and self.name:
            self.tool_name = self.name
        elif not self.name and self.tool_name:
            self.name = self.tool_name

    def key(self) -> str:
        t_name = self.tool_name or self.name
        return f"{t_name}:{json.dumps(self.arguments, sort_keys=True)}"

    def to_dict(self) -> dict[str, Any]:
        t_name = self.tool_name or self.name
        return {
            "tool_name": t_name,
            "name": t_name,
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
    error: str | None = None
    is_error: bool = False
    cached: bool = False
    speculated: bool = False
    cancelled: bool = False
    cache_timestamp: float | None = None
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
        t_name = self.tool_name or self.name
        return {
            "call_id": self.call_id,
            "tool_name": t_name,
            "name": t_name,
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

    event_type: EventType | str
    timestamp: float = field(default_factory=time.perf_counter)
    timestamp_ns: int = field(default_factory=time.perf_counter_ns)
    task_id: str = "default"
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    call_id: str | None = None
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
            event_type: EventType | str = EventType(event_type_str)
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
        kwargs: dict[str, Any] = {}
        for k, v in data.items():
            if k == "rate_limit_capacity":
                kwargs[k] = int(v)
            elif k in cls.__dataclass_fields__:
                kwargs[k] = float(v)
        return cls(**kwargs)


@dataclass
class TaskInstance:
    """An individual workload task item for evaluation."""

    task_id: str
    workload_family: str = "default"
    workload_id: str = ""
    prompt: str = ""
    expected_tools: list[str] = field(default_factory=list)
    expected_output: Any = None
    expected_args: dict[str, Any] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.workload_family and self.workload_id:
            self.workload_family = self.workload_id
        elif not self.workload_id and self.workload_family:
            self.workload_id = self.workload_family

    def to_agent_task(self) -> AgentTask:
        filtered_meta = filter_model_visible_metadata(self.metadata)
        return AgentTask(
            task_id=self.task_id,
            prompt=self.prompt,
            workload_family=self.workload_family,
            context=sanitize_model_visible_data(copy.deepcopy(dict(self.context))),
            parameters=sanitize_model_visible_data(copy.deepcopy(dict(self.parameters))),
            metadata=filtered_meta,
        )

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
            workload_family=data.get("workload_family", "default"),
            prompt=data.get("prompt", ""),
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
    validator: Callable[..., Any] | None = None
    context: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_agent_task(self) -> AgentTask:
        filtered_meta = filter_model_visible_metadata(self.metadata)
        return AgentTask(
            task_id=self.task_id,
            prompt=self.prompt,
            workload_family=self.metadata.get("workload_family", "default"),
            context=sanitize_model_visible_data(copy.deepcopy(dict(self.context))),
            parameters=sanitize_model_visible_data(
                copy.deepcopy(dict(self.metadata.get("parameters", self.parameters)))
            ),
            metadata=filtered_meta,
        )

    def validate(
        self,
        actual_output: Any,
        trace: ExecutionTrace | None = None,
        initial_state: StateSnapshot | None = None,
        final_state: StateSnapshot | None = None,
    ) -> bool:
        """Strict validation: task cannot automatically pass if neither validator nor expected_output is provided."""
        if self.validator is not None:
            try:
                if hasattr(self.validator, "validate"):
                    res = self.validator.validate(
                        task=self,
                        output=actual_output,
                        trace=trace,
                        initial_state=initial_state,
                        final_state=final_state,
                    )
                    if isinstance(res, tuple):
                        return bool(res[0])
                    return bool(res)
                try:
                    return bool(self.validator(actual_output, trace=trace))
                except TypeError:
                    try:
                        return bool(self.validator(actual_output))
                    except TypeError:
                        return bool(self.validator(self.expected_output, actual_output))
            except Exception:
                return False

        # If required tools are specified, they MUST have been executed in trace
        req_tools = self.metadata.get("required_tools") or self.metadata.get("expected_tools")
        if req_tools:
            if trace is None or not trace.tool_calls:
                return False
            executed = {c.name or c.tool_name for c in trace.tool_calls}
            for req in req_tools:
                if req not in executed:
                    return False

        if trace is not None:
            for r in trace.tool_results:
                if (r.is_error or r.error is not None) and not r.cancelled and not r.speculated:
                    return False

        if self.expected_output is None:
            return False

        if isinstance(self.expected_output, dict) and isinstance(actual_output, dict):
            return all(k in actual_output and actual_output[k] == v for k, v in self.expected_output.items())

        return bool(self.expected_output == actual_output)

    def to_model_task(self) -> Task:
        """Returns a Task strictly stripped of expected_output, validator, and oracle metadata."""
        return Task(
            task_id=self.task_id,
            prompt=self.prompt,
            context=sanitize_model_visible_data(copy.deepcopy(dict(self.context))),
            parameters=sanitize_model_visible_data(
                copy.deepcopy(dict(self.metadata.get("parameters", self.parameters)))
            ),
            metadata=filter_model_visible_metadata(self.metadata),
            expected_output=None,
            validator=None,
        )


@dataclass
class ExecutionTrace:
    """Complete chronological audit trace of an agent task execution."""

    task_id: str
    pair_id: str = ""
    arm: str = "candidate"  # "baseline" or "candidate"
    workload_family: str = "default"
    workload_id: str = "default"
    mechanism: str = "default"
    scheduler_name: str = "default"
    case_id: str = ""
    fixture_hash: str = ""
    initial_state_hash: str = ""
    final_state: dict[str, Any] = field(default_factory=dict)
    model_decisions: list[dict[str, Any]] = field(default_factory=list)
    success: bool = False
    final_output: Any = None
    start_ns: int = 0
    end_ns: int = 0
    duration_ms: float = 0.0
    ccl_ms: float | None = None
    start_time_ns: int = 0
    end_time_ns: int = 0
    events: list[ExecutionEvent] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    validator_result: dict[str, Any] = field(default_factory=dict)
    approval_decisions: list[dict[str, Any]] = field(default_factory=list)
    side_effects_recorded: int = 0
    cost_usd: float = 0.0
    seed: int = 0
    timing_source: str = "virtual_clock"
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

        if not self.workload_id and self.workload_family:
            self.workload_id = self.workload_family
        elif not self.workload_family and self.workload_id:
            self.workload_family = self.workload_id

    @property
    def duration_ns(self) -> int:
        if self.end_ns >= self.start_ns >= 0 and self.end_ns > 0:
            return self.end_ns - self.start_ns
        if self.end_time_ns >= self.start_time_ns >= 0 and self.end_time_ns > 0:
            return self.end_time_ns - self.start_time_ns
        return int(self.duration_ms * 1_000_000)

    def get_events_by_type(self, event_type: EventType | str) -> list[ExecutionEvent]:
        target = event_type.value if isinstance(event_type, EventType) else str(event_type)
        return [
            e
            for e in self.events
            if (e.event_type.value if isinstance(e.event_type, EventType) else str(e.event_type)) == target
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "pair_id": self.pair_id,
            "arm": self.arm,
            "workload_family": self.workload_family,
            "workload_id": self.workload_id,
            "mechanism": self.mechanism,
            "scheduler_name": self.scheduler_name,
            "case_id": self.case_id,
            "fixture_hash": self.fixture_hash,
            "initial_state_hash": self.initial_state_hash,
            "final_state": self.final_state,
            "model_decisions": self.model_decisions,
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
            "approval_decisions": self.approval_decisions,
            "side_effects_recorded": self.side_effects_recorded,
            "cost_usd": self.cost_usd,
            "seed": self.seed,
            "timing_source": self.timing_source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionTrace:
        return cls(
            task_id=data["task_id"],
            pair_id=data.get("pair_id", ""),
            arm=data.get("arm", "candidate"),
            workload_family=data.get("workload_family", "default"),
            workload_id=data.get("workload_id", data.get("workload_family", "default")),
            mechanism=data.get("mechanism", "default"),
            scheduler_name=data.get("scheduler_name", "default"),
            case_id=data.get("case_id", ""),
            fixture_hash=data.get("fixture_hash", ""),
            initial_state_hash=data.get("initial_state_hash", ""),
            final_state=dict(data.get("final_state", {})),
            model_decisions=list(data.get("model_decisions", [])),
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
            approval_decisions=list(data.get("approval_decisions", [])),
            side_effects_recorded=int(data.get("side_effects_recorded", 0)),
            cost_usd=float(data.get("cost_usd", 0.0)),
            seed=int(data.get("seed", 0)),
            timing_source=data.get("timing_source", "virtual_clock"),
            metadata=dict(data.get("metadata", {})),
        )


class TaskValidator(ABC):
    """Abstract interface for validating task completion and exact correctness."""

    @abstractmethod
    def validate(
        self,
        task: Any,
        output: Any,
        trace: ExecutionTrace | None = None,
        initial_state: StateSnapshot | None = None,
        final_state: StateSnapshot | None = None,
    ) -> tuple[bool, str, dict[str, Any]]:
        """Validate output, trace, and state transitions."""
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> bool:
        if len(args) == 2 and isinstance(args[0], Task):
            res = self.validate(
                args[0],
                args[1],
                kwargs.get("trace"),
                kwargs.get("initial_state"),
                kwargs.get("final_state"),
            )
        elif len(args) == 1:
            res = self.validate(
                None,
                args[0],
                kwargs.get("trace"),
                kwargs.get("initial_state"),
                kwargs.get("final_state"),
            )
        else:
            res = self.validate(*args, **kwargs)  # type: ignore[misc]
        if isinstance(res, tuple):
            return bool(res[0])
        return bool(res)


class FunctionValidator(TaskValidator):
    """Adapter wrapping a validation callable."""

    def __init__(
        self,
        fn: Callable[..., tuple[bool, str, dict[str, Any]] | bool],
    ):
        self._fn = fn

    def validate(
        self,
        task: Any,
        output: Any,
        trace: ExecutionTrace | None = None,
        initial_state: StateSnapshot | None = None,
        final_state: StateSnapshot | None = None,
    ) -> tuple[bool, str, dict[str, Any]]:
        try:
            res = self._fn(
                task=task,
                output=output,
                trace=trace,
                initial_state=initial_state,
                final_state=final_state,
            )
        except TypeError:
            try:
                res = self._fn(task, output, trace)
            except TypeError:
                res = self._fn(output)

        if isinstance(res, tuple):
            valid = bool(res[0])
            msg = str(res[1]) if len(res) > 1 else ("Passed" if valid else "Failed")
            details = dict(res[2]) if len(res) > 2 and isinstance(res[2], dict) else {}
            return valid, msg, details
        return bool(res), "Passed" if res else "Failed", {}


@dataclass
class GuardrailMetrics:
    """Comprehensive guardrail statistics measured across evaluation runs."""

    total_tasks: int = 0
    successful_tasks: int = 0
    exact_success: float | None = None
    exact_accuracy: float | None = None
    tool_selection_accuracy: float | None = None
    argument_accuracy: float | None = None
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
    blocked_unsafe_attempts: int = 0
    rate_limit_failures: int = 0
    rate_limit_errors: int = 0
    peak_concurrency: int = 0
    total_model_input_tokens: int = 0
    total_model_output_tokens: int = 0
    total_model_calls: int = 0
    total_deopts: int = 0
    tool_cost_multiplier: float | None = None
    cost_per_task_usd: float | None = None
    total_cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def compute_derived(self) -> None:
        if self.total_tasks > 0:
            self.exact_accuracy = self.successful_tasks / self.total_tasks
            self.exact_success = self.exact_accuracy
            self.cost_per_task_usd = self.total_cost_usd / self.total_tasks
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
    arguments: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
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
    """Summary and raw audit information for an executed task trial."""

    task_id: str
    success: bool
    final_answer: Any = None
    ccl_ms: float | None = None
    total_duration_ms: float = 0.0
    pair_id: str = ""
    arm: str = "candidate"
    workload_id: str = "default"
    scheduler_name: str = "default"
    events: list[ExecutionEvent] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    initial_state: dict[str, Any] = field(default_factory=dict)
    final_state: dict[str, Any] = field(default_factory=dict)
    validator_result: dict[str, Any] = field(default_factory=dict)
    guardrails: GuardrailMetrics = field(default_factory=GuardrailMetrics)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "pair_id": self.pair_id,
            "arm": self.arm,
            "workload_id": self.workload_id,
            "scheduler_name": self.scheduler_name,
            "success": self.success,
            "final_answer": self.final_answer,
            "ccl_ms": self.ccl_ms,
            "total_duration_ms": self.total_duration_ms,
            "events": [e.to_dict() for e in self.events],
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "tool_results": [r.to_dict() for r in self.tool_results],
            "initial_state": self.initial_state,
            "final_state": self.final_state,
            "validator_result": self.validator_result,
            "guardrails": self.guardrails.to_dict(),
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskResult:
        return cls(
            task_id=data["task_id"],
            pair_id=data.get("pair_id", ""),
            arm=data.get("arm", "candidate"),
            workload_id=data.get("workload_id", "default"),
            scheduler_name=data.get("scheduler_name", "default"),
            success=bool(data.get("success", False)),
            final_answer=data.get("final_answer"),
            ccl_ms=float(data["ccl_ms"]) if data.get("ccl_ms") is not None else None,
            total_duration_ms=float(data.get("total_duration_ms", 0.0)),
            events=[ExecutionEvent.from_dict(e) for e in data.get("events", [])],
            tool_calls=[ToolCall.from_dict(c) for c in data.get("tool_calls", [])],
            tool_results=[ToolResult.from_dict(r) for r in data.get("tool_results", [])],
            initial_state=dict(data.get("initial_state", {})),
            final_state=dict(data.get("final_state", {})),
            validator_result=dict(data.get("validator_result", {})),
            guardrails=GuardrailMetrics.from_dict(data.get("guardrails", {})),
            error=data.get("error"),
            metadata=dict(data.get("metadata", {})),
        )
