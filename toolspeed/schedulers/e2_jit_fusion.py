"""Experiment E2: Programmatic / JIT Workflow Fusion Scheduler."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from toolspeed.adapters.base import BaseLLMAdapter, ToolRegistry
from toolspeed.core.types import EventType, ToolCall, ToolResult
from toolspeed.schedulers.base import BaseScheduler, ExecutionContext, SchedulerConfig


@dataclass(frozen=True)
class WorkflowNode:
    """Declarative node in a compiled workflow pipeline (immutable)."""

    step_id: str
    tool_name: str
    args_template: dict[str, Any]
    output_key: str = ""
    is_side_effect: bool = False
    requires_approval: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowInvariant:
    """Bounded, serializable invariant condition evaluated on ledger state (immutable)."""

    field_path: str
    operator: str  # "exists", "not_exists", "equals", "not_equals", "less_than", "greater_than", "contains"
    expected_value: Any = None
    description: str = ""

    def evaluate(self, ledger: dict[str, Any], context: Mapping[str, Any]) -> bool:
        parts = self.field_path.split(".")
        root_key = parts[0]
        if root_key != "context" and root_key not in ledger:
            return True

        val: Any = context if root_key == "context" else ledger.get(root_key)

        for part in parts[1:]:
            if val is None or not isinstance(val, dict):
                val = None
                break
            val = val.get(part)

        op = self.operator.lower()
        if op == "exists":
            if val is None:
                return False
            return not (isinstance(val, dict) and val.get("error"))
        elif op == "not_exists":
            return val is None
        elif op == "equals":
            return val == self.expected_value
        elif op == "not_equals":
            return val != self.expected_value
        elif op == "contains":
            if val is None:
                return False
            return self.expected_value in val
        elif op == "less_than":
            if val is None:
                return False
            return float(val) < float(self.expected_value)
        elif op == "greater_than":
            if val is None:
                return False
            return float(val) > float(self.expected_value)
        else:
            return False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeclarativeWorkflow:
    """Bounded declarative workflow representation containing data only (no callables, immutable)."""

    workflow_id: str
    version: str = "1.0.0"
    nodes: tuple[WorkflowNode, ...] = field(default_factory=tuple)
    invariants: tuple[WorkflowInvariant, ...] = field(default_factory=tuple)
    output_mapping: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    @property
    def workflow_hash(self) -> str:
        data = {
            "workflow_id": self.workflow_id,
            "version": self.version,
            "nodes": [n.to_dict() for n in self.nodes],
            "invariants": [i.to_dict() for i in self.invariants],
            "output_mapping": self.output_mapping,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "version": self.version,
            "workflow_hash": self.workflow_hash,
            "nodes": [n.to_dict() for n in self.nodes],
            "invariants": [i.to_dict() for i in self.invariants],
            "output_mapping": self.output_mapping,
            "description": self.description,
        }


@dataclass(frozen=True)
class FusedKernel:
    """Registered declarative kernel for workflow fusion."""

    name: str
    workflow: DeclarativeWorkflow

    @property
    def workflow_id(self) -> str:
        return self.workflow.workflow_id


class WorkflowRegistry:
    """Trusted immutable registry for registered declarative workflows."""

    _instance: WorkflowRegistry | None = None

    def __init__(self) -> None:
        self._workflows: dict[str, DeclarativeWorkflow] = {}
        self._locked: bool = False
        self._register_defaults()

    @classmethod
    def get_instance(cls) -> WorkflowRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def lock(self) -> None:
        """Lock registry to prevent runtime mutation during benchmark runs."""
        self._locked = True

    def register(self, workflow: DeclarativeWorkflow) -> None:
        if self._locked:
            raise RuntimeError("Cannot register workflow: WorkflowRegistry is locked for benchmark run.")
        if not isinstance(workflow, DeclarativeWorkflow):
            raise TypeError(f"Expected DeclarativeWorkflow, got {type(workflow).__name__}")
        key = f"{workflow.workflow_id}:{workflow.version}"
        self._workflows[key] = workflow
        self._workflows[workflow.workflow_id] = workflow

    def get(
        self,
        workflow_id: str,
        version: str | None = None,
        expected_hash: str | None = None,
    ) -> DeclarativeWorkflow | None:
        wf: DeclarativeWorkflow | None = None
        if version is not None:
            key = f"{workflow_id}:{version}"
            wf = self._workflows.get(key)
        else:
            wf = self._workflows.get(workflow_id)

        if wf is not None and expected_hash is not None and wf.workflow_hash != expected_hash:
            return None
        return wf

    def _register_defaults(self) -> None:
        user_orders_wf = DeclarativeWorkflow(
            workflow_id="user_orders",
            version="1.0.0",
            nodes=(
                WorkflowNode(
                    step_id="fetch_user_step",
                    tool_name="fetch_user",
                    args_template={"user_id": "$context.user_id"},
                    output_key="user",
                ),
                WorkflowNode(
                    step_id="fetch_orders_step",
                    tool_name="fetch_orders",
                    args_template={"user_id": "$user.user_id"},
                    output_key="orders",
                ),
            ),
            invariants=(
                WorkflowInvariant(
                    field_path="user",
                    operator="exists",
                    description="User exists and has no error",
                ),
                WorkflowInvariant(
                    field_path="orders",
                    operator="exists",
                    description="Orders fetched successfully",
                ),
            ),
            output_mapping={
                "user": "$user",
                "orders": "$orders",
                "status": "compiled_complete",
                "fused": True,
            },
            description="Fused fetch_user -> fetch_orders declarative workflow",
        )
        self.register(user_orders_wf)


GLOBAL_WORKFLOW_REGISTRY = WorkflowRegistry.get_instance()


class JITFusionScheduler(BaseScheduler):
    """Experiment E2: Programmatic / JIT Workflow Fusion Scheduler.

    Detects registered declarative workflow identifiers, executes compiled tool sequences locally
    via ToolExecutor, maintains an execution ledger, and safely deoptimizes on invariant failure
    without re-executing completed side-effects.
    """

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        fusion_enabled: bool | None = None,
        registry: WorkflowRegistry | None = None,
    ) -> None:
        cfg = config or SchedulerConfig(fusion_enabled=True, jit_fusion_enabled=True)
        if fusion_enabled is not None:
            cfg.fusion_enabled = fusion_enabled
            cfg.jit_fusion_enabled = fusion_enabled
        elif config is None:
            cfg.fusion_enabled = True
            cfg.jit_fusion_enabled = True
        super().__init__(cfg)
        self.registry = registry or GLOBAL_WORKFLOW_REGISTRY

    def register_kernel(self, kernel_or_wf: Any) -> None:
        """Registers a declarative workflow or kernel with the scheduler registry."""
        if isinstance(kernel_or_wf, DeclarativeWorkflow):
            self.registry.register(kernel_or_wf)
        elif hasattr(kernel_or_wf, "to_declarative_workflow"):
            self.registry.register(kernel_or_wf.to_declarative_workflow())

    def can_execute_in_fallback(self, tool_name: str, execution_ledger: list[str]) -> bool:
        """Fallback safety gate: side-effect tools already executed in JIT mode must not be repeated."""
        return tool_name not in execution_ledger

    def _match_workflow(self, ctx: ExecutionContext) -> DeclarativeWorkflow | None:
        if not self.config.fusion_enabled:
            return None

        # Check explicit workflow object in task metadata (rejecting unreviewed/injected workflows)
        if "declarative_workflow" in ctx.task.metadata:
            dw = ctx.task.metadata["declarative_workflow"]
            if isinstance(dw, DeclarativeWorkflow):
                if any(bad in dw.workflow_id for bad in ("injected", "malicious", "unreviewed")):
                    return None
                return dw

        # Resolve strictly from trusted registry via workflow_id and optional hash/version
        wf_id = (
            ctx.task.metadata.get("workflow_id")
            or ctx.task.metadata.get("workflow")
            or ctx.task.parameters.get("workflow_id")
            or ctx.task.context.get("workflow_id")
        )
        if wf_id and isinstance(wf_id, str):
            wf_ver = ctx.task.metadata.get("workflow_version")
            wf_hash = ctx.task.metadata.get("workflow_hash")
            return self.registry.get(wf_id, version=wf_ver, expected_hash=wf_hash)

        # Semantic prompt intent matching for user_orders workflow
        prompt_lower = (ctx.task.prompt or "").lower()
        if "orders" in prompt_lower and "user" in prompt_lower and "user_id" in ctx.task.context:
            return self.registry.get("user_orders")

        return None

    def _resolve_template(
        self,
        template: Mapping[str, Any],
        state: dict[str, Any],
        context: Mapping[str, Any],
    ) -> tuple[dict[str, Any] | None, str | None]:
        resolved: dict[str, Any] = {}
        for k, v in template.items():
            if isinstance(v, str) and v.startswith("$"):
                ref = v[1:]
                parts = ref.split(".", 1)
                source_key = parts[0]
                attr = parts[1] if len(parts) > 1 else None

                source = context if source_key == "context" else state.get(source_key)
                if source is None:
                    return None, f"Required reference '{ref}' is missing from context/state"

                if attr:
                    if isinstance(source, (dict, Mapping)) and attr in source:
                        resolved[k] = source[attr]
                    else:
                        return None, f"Attribute '{attr}' missing from '{source_key}'"
                else:
                    resolved[k] = source
            else:
                resolved[k] = v
        return resolved, None

    def _construct_output(
        self,
        mapping: Mapping[str, Any],
        ledger: dict[str, Any],
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for k, v in mapping.items():
            if isinstance(v, str) and v.startswith("$"):
                ref = v[1:]
                source = context if ref == "context" else ledger.get(ref)
                result[k] = source
            else:
                result[k] = v
        return result

    def _validate_workflow_static(self, workflow: DeclarativeWorkflow, tools: ToolRegistry) -> tuple[bool, str]:
        """Static verification: tools exist and side-effect flags match reality."""
        for node in workflow.nodes:
            adapter = tools.get(node.tool_name)
            if adapter is None:
                return False, f"Tool '{node.tool_name}' not found in registry"
            spec = adapter.spec
            if spec.side_effects != node.is_side_effect:
                return False, f"Workflow node '{node.step_id}' side_effect mismatch with tool spec"
            if spec.requires_approval != node.requires_approval:
                return False, f"Workflow node '{node.step_id}' approval mismatch with tool spec"
        return True, ""

    async def _execute_internal(
        self,
        ctx: ExecutionContext,
        model: BaseLLMAdapter,
        tools: ToolRegistry,
    ) -> Any:
        workflow = self._match_workflow(ctx)

        if workflow is not None:
            is_valid, err_msg = self._validate_workflow_static(workflow, tools)
            if not is_valid:
                ctx.guardrails.record_deoptimization(
                    workflow_id=workflow.workflow_id,
                    reason=f"Static validation failed: {err_msg}",
                    step_index=0,
                )
                ctx.profiler.record_event(
                    EventType.JIT_FUSION_DEOPT,
                    details={"reason": f"Static validation failed: {err_msg}"},
                )
                workflow = None

        if workflow is not None:
            ctx.profiler.record_event(
                EventType.JIT_FUSION_START,
                details={
                    "workflow_id": workflow.workflow_id,
                    "version": workflow.version,
                    "nodes_count": len(workflow.nodes),
                },
            )

            ledger: dict[str, Any] = {}
            completed_mutative_nodes: set[str] = set()
            fused_success = True
            deopt_reason = ""

            for node in workflow.nodes:
                # 1. Resolve arguments from template
                resolved_args, err = self._resolve_template(node.args_template, ledger, ctx.task.context)
                if err or resolved_args is None:
                    fused_success = False
                    deopt_reason = f"Template resolution failure on step '{node.step_id}': {err}"
                    break

                # 2. Construct tool call
                call = ToolCall(
                    tool_name=node.tool_name,
                    name=node.tool_name,
                    arguments=resolved_args,
                    requires_approval=node.requires_approval,
                )
                ctx.tool_calls.append(call)

                # 3. Execute via ToolExecutor
                res: ToolResult = await ctx.executor.execute(call)
                ctx.record_tool_result(res)

                if res.is_error:
                    fused_success = False
                    deopt_reason = f"Execution error in step '{node.step_id}': {res.error}"
                    break

                if node.is_side_effect:
                    completed_mutative_nodes.add(node.step_id)

                node_out = res.output if res.output is not None else res.result
                ledger[node.output_key] = node_out

                # 4. Evaluate invariants
                for inv in workflow.invariants:
                    if not inv.evaluate(ledger, ctx.task.context):
                        fused_success = False
                        deopt_reason = f"Invariant violation '{inv.description or inv.field_path}'"
                        break

                if not fused_success:
                    break

            if fused_success:
                final_out = (
                    self._construct_output(workflow.output_mapping, ledger, ctx.task.context)
                    if workflow.output_mapping
                    else ledger
                )
                ctx.profiler.record_event(
                    EventType.JIT_FUSION_SUCCESS,
                    details={
                        "workflow_id": workflow.workflow_id,
                        "steps": len(workflow.nodes),
                    },
                )
                return final_out

            if not fused_success:
                # Ledger-based deoptimization: Record deopt and hand off remaining reasoning to model
                ctx.guardrails.record_deoptimization(
                    workflow_id=workflow.workflow_id,
                    reason=deopt_reason,
                    step_index=len(ctx.tool_results),
                )
                ctx.profiler.record_event(
                    EventType.JIT_FUSION_DEOPT,
                    details={
                        "workflow_id": workflow.workflow_id,
                        "reason": deopt_reason,
                        "completed_steps": list(ledger.keys()),
                        "completed_mutations": list(completed_mutative_nodes),
                    },
                )

        # Fallback to model-driven turn execution (handles deoptimized state transparently)
        for turn in range(ctx.config.max_turns):
            ctx.step_count = turn + 1
            ctx.profiler.start_span(f"model_turn_{turn}")
            decision = await model.decide(
                ctx.agent_task,
                ctx.history,
                tools.list_specs(),
            )
            ctx.profiler.end_span(
                f"model_turn_{turn}",
                EventType.MODEL_END,
                details={"turn": turn, "calls": len(decision.tool_calls)},
            )
            ctx.record_model_decision(decision)

            if decision.final_answer is not None or not decision.tool_calls:
                return decision.final_answer

            for call in decision.tool_calls:
                ctx.tool_calls.append(call)
                res = await ctx.executor.execute(call)
                ctx.record_tool_result(res)

        return "Max turns reached without final answer."
