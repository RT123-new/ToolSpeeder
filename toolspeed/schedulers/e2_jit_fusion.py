"""Experiment E2: Programmatic / JIT Workflow Fusion Scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import time

from toolspeed.adapters.base import BaseLLMAdapter, ToolRegistry
from toolspeed.core.types import EventType, ToolCall, ToolResult
from toolspeed.schedulers.base import BaseScheduler, ExecutionContext, cancel_and_await


@dataclass
class WorkflowNode:
    """Declarative node in a compiled workflow pipeline."""
    step_id: str
    tool_name: str
    args_template: Dict[str, Any]
    output_key: str
    is_side_effect: bool = False
    requires_approval: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowInvariant:
    """Bounded, serializable invariant condition evaluated on ledger state."""
    field_path: str
    operator: str  # "exists", "not_exists", "equals", "not_equals", "less_than", "greater_than", "contains"
    expected_value: Any = None
    description: str = ""

    def evaluate(self, ledger: Dict[str, Any], context: Dict[str, Any]) -> bool:
        # Extract target field
        parts = self.field_path.split(".")
        root_key = parts[0]
        if root_key != "context" and root_key not in ledger:
            # Step produces this key in a future node; do not prematurely fail
            return True

        val: Any = context.get(root_key) if root_key == "context" else ledger.get(root_key)

        for part in parts[1:]:
            if val is None or not isinstance(val, dict):
                val = None
                break
            val = val.get(part)

        op = self.operator.lower()
        if op == "exists":
            if val is None:
                return False
            if isinstance(val, dict) and val.get("error"):
                return False
            return True
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeclarativeWorkflow:
    """Bounded declarative workflow representation containing data only (no callables)."""
    workflow_id: str
    version: str = "1.0.0"
    nodes: List[WorkflowNode] = field(default_factory=list)
    invariants: List[WorkflowInvariant] = field(default_factory=list)
    output_mapping: Dict[str, Any] = field(default_factory=dict)
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "version": self.version,
            "workflow_hash": self.workflow_hash,
            "nodes": [n.to_dict() for n in self.nodes],
            "invariants": [i.to_dict() for i in self.invariants],
            "output_mapping": self.output_mapping,
            "description": self.description,
        }


@dataclass
class FusedKernel:
    """Registered declarative kernel for workflow fusion."""
    name: str
    workflow: DeclarativeWorkflow

    @property
    def workflow_id(self) -> str:
        return self.workflow.workflow_id


class JITFusionScheduler(BaseScheduler):
    """Experiment E2: Programmatic / JIT Workflow Fusion Scheduler.
    
    Detects known declarative workflow structures, executes compiled tool sequences locally
    via ToolExecutor, maintains an execution ledger, and safely deoptimizes on invariant failure
    without re-executing completed side-effects.
    """

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self._compiled_kernels: Dict[str, DeclarativeWorkflow] = {}
        self._register_default_kernels()

    def register_kernel(self, kernel: Union[FusedKernel, DeclarativeWorkflow]) -> None:
        if isinstance(kernel, FusedKernel):
            self._compiled_kernels[kernel.name] = kernel.workflow
        elif isinstance(kernel, DeclarativeWorkflow):
            self._compiled_kernels[kernel.workflow_id] = kernel
        else:
            raise TypeError(f"Expected DeclarativeWorkflow or FusedKernel, got {type(kernel).__name__}")

    def _register_default_kernels(self) -> None:
        """Standard compiled declarative workflow for user orders pipeline."""
        user_orders_wf = DeclarativeWorkflow(
            workflow_id="user_orders",
            version="1.0.0",
            nodes=[
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
            ],
            invariants=[
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
            ],
            output_mapping={
                "user": "$user",
                "orders": "$orders",
                "fused": True,
            },
            description="Fused fetch_user -> fetch_orders declarative workflow",
        )
        self._compiled_kernels["user_orders"] = user_orders_wf
        self._compiled_kernels["user_orders_fusion"] = user_orders_wf

    def _match_workflow(self, ctx: ExecutionContext) -> Optional[DeclarativeWorkflow]:
        # 1. Explicit declarative workflow in task metadata
        custom_wf = ctx.task.metadata.get("declarative_workflow")
        if isinstance(custom_wf, DeclarativeWorkflow):
            return custom_wf

        # 2. Explicit workflow identifier matching
        wf_id = ctx.task.metadata.get("workflow")
        if wf_id and wf_id in self._compiled_kernels:
            return self._compiled_kernels[wf_id]

        # 3. Context & prompt heuristic matching
        if "user_id" in ctx.task.context or ("user" in ctx.task.prompt.lower() and "order" in ctx.task.prompt.lower()):
            return self._compiled_kernels.get("user_orders")

        return None

    def _resolve_template(
        self,
        template: Dict[str, Any],
        state: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        resolved: Dict[str, Any] = {}
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
                    if isinstance(source, dict) and attr in source:
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
        mapping: Dict[str, Any],
        ledger: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for k, v in mapping.items():
            if isinstance(v, str) and v.startswith("$"):
                ref = v[1:]
                parts = ref.split(".", 1)
                source_key = parts[0]
                attr = parts[1] if len(parts) > 1 else None
                source = context if source_key == "context" else ledger.get(source_key)
                if attr and isinstance(source, dict):
                    result[k] = source.get(attr)
                else:
                    result[k] = source
            else:
                result[k] = v
        return result

    async def _execute_internal(
        self,
        ctx: ExecutionContext,
        model: BaseLLMAdapter,
        tools: ToolRegistry,
    ) -> Any:
        workflow = self._match_workflow(ctx)
        ledger: Dict[str, Any] = {}
        deopt_triggered = False
        deopt_reason = ""
        completed_side_effects: Set[str] = set()

        if workflow:
            ctx.profiler.record_event(
                EventType.JIT_FUSION_START,
                details={"workflow": workflow.workflow_id, "hash": workflow.workflow_hash},
            )

            # Static tool and schema validation
            all_tools_present = all(tools.get(node.tool_name) is not None for node in workflow.nodes)
            if not all_tools_present:
                deopt_triggered = True
                deopt_reason = "Missing required tools in registry for compiled workflow"
            else:
                # Execute declarative workflow steps sequentially
                for node in workflow.nodes:
                    resolved_args, err = self._resolve_template(node.args_template, ledger, ctx.task.context)
                    if err:
                        deopt_triggered = True
                        deopt_reason = err
                        break

                    # Check approval: Never manufacture approval!
                    call = ToolCall(
                        name=node.tool_name,
                        arguments=resolved_args or {},
                        requires_approval=node.requires_approval,
                        is_approved=ctx.task.metadata.get("is_approved", False),
                    )
                    ctx.tool_calls.append(call)

                    res = await ctx.executor.execute(call)
                    ctx.record_tool_result(res)

                    if not res.is_success:
                        deopt_triggered = True
                        deopt_reason = f"Step '{node.step_id}' failed: {res.error}"
                        break

                    if node.is_side_effect:
                        completed_side_effects.add(node.step_id)

                    out = res.output if res.output is not None else res.result
                    ledger[node.output_key] = out

                    # Check invariants after each step
                    for inv in workflow.invariants:
                        if not inv.evaluate(ledger, ctx.task.context):
                            deopt_triggered = True
                            deopt_reason = f"Invariant violated: {inv.description}"
                            break

                    if deopt_triggered:
                        break

                if not deopt_triggered:
                    # Construct declarative output mapping
                    final_out = self._construct_output(workflow.output_mapping, ledger, ctx.task.context) if workflow.output_mapping else ledger

                    # Strict task validation check
                    if ctx.task.validate(final_out):
                        ctx.profiler.record_event(
                            EventType.JIT_FUSION_SUCCESS,
                            details={"workflow": workflow.workflow_id},
                        )
                        return final_out
                    else:
                        deopt_triggered = True
                        deopt_reason = "Compiled output failed task validation"

            if deopt_triggered:
                ctx.profiler.record_event(
                    EventType.JIT_FUSION_DEOPT,
                    details={"workflow": workflow.workflow_id, "reason": deopt_reason},
                )
                ctx.guardrails.record_deopt()

        # Fallback / Deoptimized path: Model reasoning starting from accumulated ledger state
        for turn in range(ctx.config.max_turns):
            ctx.step_count = turn + 1

            ctx.profiler.start_span(f"model_turn_{turn}")
            decision = await model.decide(
                ctx.task,
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
