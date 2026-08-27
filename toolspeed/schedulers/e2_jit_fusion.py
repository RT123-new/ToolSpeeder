"""Experiment E2: Programmatic / JIT Workflow Fusion Scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import asyncio

from toolspeed.adapters.base import BaseLLMAdapter, ToolRegistry
from toolspeed.core.types import EventType, ToolCall, ToolResult
from toolspeed.schedulers.base import BaseScheduler, ExecutionContext


@dataclass
class WorkflowNode:
    """Declarative node in a compiled workflow pipeline."""
    step_id: str
    tool_name: str
    args_template: Dict[str, Any]
    output_key: str
    is_side_effect: bool = False
    requires_approval: bool = False


@dataclass
class WorkflowInvariant:
    """Invariant condition required for fast-path compiled execution."""
    check_fn: Callable[[Dict[str, Any]], bool]
    description: str


@dataclass
class DeclarativeWorkflow:
    """Bounded declarative workflow representation replacing arbitrary callables."""
    workflow_id: str
    nodes: List[WorkflowNode]
    invariants: List[WorkflowInvariant] = field(default_factory=list)
    output_constructor: Optional[Callable[[Dict[str, Any]], Any]] = None
    description: str = ""


@dataclass
class FusedKernel:
    """Registered compiled kernel for workflow fusion."""
    name: str
    workflow: Optional[DeclarativeWorkflow] = None
    matcher: Optional[Callable[[ExecutionContext], bool]] = None
    tool_sequence: List[str] = field(default_factory=list)
    execute_fn: Optional[Callable[..., Any]] = None
    match_fn: Optional[Callable[..., bool]] = None

    def __post_init__(self) -> None:
        if self.matcher is None:
            if self.match_fn is not None:
                self.matcher = self.match_fn
            elif self.workflow is not None:
                wf_id = self.workflow.workflow_id
                self.matcher = lambda ctx: ctx.task.metadata.get("workflow") == wf_id
            else:
                self.matcher = lambda ctx: False
        if self.match_fn is None:
            self.match_fn = self.matcher


class JITFusionScheduler(BaseScheduler):
    """Experiment E2: Programmatic / JIT Workflow Fusion Scheduler.
    
    Detects known declarative workflow structures, executes compiled tool sequences locally
    via ToolExecutor, maintains an execution ledger, and safely deoptimizes on invariant failure
    without re-executing completed side-effects.
    """

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self._compiled_kernels: Dict[str, FusedKernel] = {}
        self._register_default_kernels()

    def register_kernel(self, kernel: Union[FusedKernel, DeclarativeWorkflow]) -> None:
        if isinstance(kernel, DeclarativeWorkflow):
            fused = FusedKernel(
                name=kernel.workflow_id,
                workflow=kernel,
                matcher=lambda ctx: ctx.task.metadata.get("workflow") == kernel.workflow_id,
            )
            self._compiled_kernels[fused.name] = fused
        else:
            self._compiled_kernels[kernel.name] = kernel

    def _register_default_kernels(self) -> None:
        """Standard compiled declarative workflow for user orders pipeline."""
        user_orders_wf = DeclarativeWorkflow(
            workflow_id="user_orders_fusion",
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
                    check_fn=lambda state: state.get("user") is None or (bool(state.get("user")) and not (isinstance(state.get("user"), dict) and "error" in state.get("user"))),
                    description="User exists and has no error",
                ),
                WorkflowInvariant(
                    check_fn=lambda state: state.get("orders") is None or (bool(state.get("orders")) and not (isinstance(state.get("orders"), dict) and "error" in state.get("orders"))),
                    description="Orders fetched successfully",
                ),
            ],
            output_constructor=lambda state: {
                "user": state.get("user"),
                "orders": state.get("orders"),
                "fused": True,
            },
            description="Fused fetch_user -> fetch_orders declarative workflow",
        )

        def match_user_orders(ctx: ExecutionContext) -> bool:
            return (
                ctx.task.metadata.get("workflow") in ("user_orders", "user_orders_fusion")
                or ("user" in ctx.task.prompt.lower() and "order" in ctx.task.prompt.lower() and "user_id" in ctx.task.context)
            )

        self._compiled_kernels["user_orders_fusion"] = FusedKernel(
            name="user_orders_fusion",
            workflow=user_orders_wf,
            matcher=match_user_orders,
        )

    def _match_workflow(self, ctx: ExecutionContext) -> Optional[DeclarativeWorkflow]:
        # 1. Explicit declarative workflow in task metadata
        custom_wf = ctx.task.metadata.get("declarative_workflow")
        if isinstance(custom_wf, DeclarativeWorkflow):
            return custom_wf

        # 2. Check registered kernels with explicit matchers
        for kernel in self._compiled_kernels.values():
            if kernel.matcher(ctx):
                return kernel.workflow

        return None

    def _resolve_template(self, template: Dict[str, Any], state: Dict[str, Any], context: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        resolved = {}
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

    async def _execute_internal(
        self,
        ctx: ExecutionContext,
        model: BaseLLMAdapter,
        tools: ToolRegistry,
    ) -> Any:
        # Check custom registered kernels with execute_fn
        for kernel in self._compiled_kernels.values():
            if kernel.execute_fn and kernel.matcher(ctx):
                ctx.profiler.record_event(EventType.JIT_FUSION_START, details={"kernel": kernel.name})
                import inspect
                try:
                    if inspect.iscoroutinefunction(kernel.execute_fn):
                        res, is_ok = await kernel.execute_fn(ctx, tools)
                    else:
                        res, is_ok = kernel.execute_fn(ctx, tools)
                except Exception as ex:
                    res, is_ok = None, False

                if is_ok and (ctx.task.validator is None and ctx.task.expected_output is None or ctx.task.validate(res)):
                    ctx.profiler.record_event(EventType.JIT_FUSION_SUCCESS, details={"kernel": kernel.name})
                    return res
                else:
                    ctx.profiler.record_event(EventType.JIT_FUSION_DEOPT, details={"kernel": kernel.name, "reason": "Kernel exception or validation failure"})
                    ctx.guardrails.record_deopt()
                    break

        workflow = self._match_workflow(ctx)
        ledger: Dict[str, Any] = {}
        deopt_triggered = False
        deopt_reason = ""

        if workflow:
            ctx.profiler.record_event(
                EventType.JIT_FUSION_START,
                details={"workflow": workflow.workflow_id},
            )

            # Validate tools are available before attempting compiled execution
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

                    call = ToolCall(name=node.tool_name, arguments=resolved_args or {}, is_approved=True)
                    ctx.tool_calls.append(call)

                    res = await ctx.executor.execute(call)
                    ctx.record_tool_result(res)

                    if not res.is_success:
                        deopt_triggered = True
                        deopt_reason = f"Step '{node.step_id}' failed: {res.error}"
                        break

                    out = res.output if res.output is not None else res.result
                    ledger[node.output_key] = out

                    # Check invariants after each step
                    for inv in workflow.invariants:
                        try:
                            if not inv.check_fn(ledger):
                                deopt_triggered = True
                                deopt_reason = f"Invariant violated: {inv.description}"
                                break
                        except Exception as ex:
                            deopt_triggered = True
                            deopt_reason = f"Invariant evaluation error: {str(ex)}"
                            break

                    if deopt_triggered:
                        break

                if not deopt_triggered:
                    # Construct output
                    if workflow.output_constructor:
                        final_out = workflow.output_constructor(ledger)
                    else:
                        final_out = ledger

                    # Check task validator
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
