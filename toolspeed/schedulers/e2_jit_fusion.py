"""Experiment E2: Programmatic / JIT Workflow Fusion Scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import asyncio

from toolspeed.adapters.base import BaseLLMAdapter, ToolRegistry
from toolspeed.core.types import EventType, ToolCall, ToolResult
from toolspeed.schedulers.base import BaseScheduler, ExecutionContext


@dataclass
class FusedKernel:
    """Compiled Python execution kernel replacing repeated LLM reasoning chains."""

    name: str
    tool_sequence: List[str]
    execute_fn: Callable[[ExecutionContext, ToolRegistry], Any]
    match_fn: Callable[[ExecutionContext], bool]
    description: str = ""


class JITFusionScheduler(BaseScheduler):
    """Experiment E2: Programmatic / JIT Workflow Fusion Scheduler.
    
    Detects repeated multi-step chains, executes them as fused Python kernels, and automatically
    deoptimizes back to LLM reasoning when assumptions fail.
    """

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self._compiled_kernels: Dict[str, FusedKernel] = {}
        self._register_default_kernels()

    def register_kernel(self, kernel: FusedKernel) -> None:
        self._compiled_kernels[kernel.name] = kernel

    def _register_default_kernels(self) -> None:
        """Standard compiled kernels for common repeated sub-plans."""
        # Example kernel: User Profile -> User Orders Lookup
        async def user_orders_pipeline(ctx: ExecutionContext, tools: ToolRegistry) -> Tuple[Any, bool]:
            user_id = ctx.task.context.get("user_id") or "123"
            
            # Step 1: Fetch user
            call1 = ToolCall(name="fetch_user", arguments={"user_id": user_id})
            ctx.tool_calls.append(call1)
            adapter1 = tools.get("fetch_user")
            if not adapter1:
                return None, False
            
            ctx.guardrails.record_tool_dispatch(adapter1.spec, call1)
            ctx.profiler.start_span(f"fused_step_1_{call1.call_id}")
            res1 = await adapter1.execute(call1)
            ctx.profiler.end_span(f"fused_step_1_{call1.call_id}", EventType.TOOL_END)
            ctx.record_tool_result(res1)

            # Invariant check: user must exist and have no error
            if not res1.is_success or not res1.output or res1.output.get("error"):
                return None, False  # Triggers deopt

            # Step 2: Fetch orders
            call2 = ToolCall(name="fetch_orders", arguments={"user_id": user_id})
            ctx.tool_calls.append(call2)
            adapter2 = tools.get("fetch_orders")
            if not adapter2:
                return None, False

            ctx.guardrails.record_tool_dispatch(adapter2.spec, call2)
            ctx.profiler.start_span(f"fused_step_2_{call2.call_id}")
            res2 = await adapter2.execute(call2)
            ctx.profiler.end_span(f"fused_step_2_{call2.call_id}", EventType.TOOL_END)
            ctx.record_tool_result(res2)

            if not res2.is_success or not res2.output:
                return None, False  # Triggers deopt

            final_data = {
                "user": res1.output,
                "orders": res2.output,
                "fused": True,
            }
            return final_data, True

        def match_user_orders(ctx: ExecutionContext) -> bool:
            p = ctx.task.prompt.lower()
            return ("user" in p and "order" in p) or ctx.task.metadata.get("workflow") == "user_orders"

        self.register_kernel(
            FusedKernel(
                name="user_orders_fusion",
                tool_sequence=["fetch_user", "fetch_orders"],
                execute_fn=user_orders_pipeline,
                match_fn=match_user_orders,
                description="Fused fetch_user -> fetch_orders pipeline",
            )
        )

    async def _execute_internal(
        self,
        ctx: ExecutionContext,
        model: BaseLLMAdapter,
        tools: ToolRegistry,
    ) -> Any:
        # Check if task metadata contains custom kernel
        custom_kernel = ctx.task.metadata.get("fused_kernel")
        if custom_kernel and isinstance(custom_kernel, FusedKernel):
            matched_kernel = custom_kernel
        else:
            # Check pattern match against registered kernels
            matched_kernel = None
            for kernel in self._compiled_kernels.values():
                if kernel.match_fn(ctx):
                    matched_kernel = kernel
                    break

        if matched_kernel:
            ctx.profiler.record_event(
                EventType.JIT_FUSION_START,
                details={"kernel": matched_kernel.name},
            )

            deopt_reason: Optional[str] = None
            try:
                result, is_success = await matched_kernel.execute_fn(ctx, tools)
            except Exception as ex:
                result, is_success = None, False
                deopt_reason = f"Kernel exception: {str(ex)}"

            if is_success:
                # Check task validation if task has validation constraints
                if ctx.task.validator is not None or ctx.task.expected_output is not None:
                    if ctx.task.validate(result):
                        ctx.profiler.record_event(
                            EventType.JIT_FUSION_SUCCESS,
                            details={"kernel": matched_kernel.name},
                        )
                        return result
                    else:
                        is_success = False
                        deopt_reason = "Kernel output failed task validation"
                else:
                    ctx.profiler.record_event(
                        EventType.JIT_FUSION_SUCCESS,
                        details={"kernel": matched_kernel.name},
                    )
                    return result

            # Deoptimization: Invariant violated or exception -> deoptimize back to model reasoning
            ctx.profiler.record_event(
                EventType.JIT_FUSION_DEOPT,
                details={"kernel": matched_kernel.name, "reason": deopt_reason or "Invariant violated"},
            )
            ctx.guardrails.record_deopt()

        # Fallback / Deoptimized path: Run standard LLM reasoning with accumulated partial state
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
                adapter = tools.get(call.name)
                if not adapter:
                    continue

                ctx.guardrails.record_tool_dispatch(adapter.spec, call)
                ctx.profiler.start_span(f"tool_{call.call_id}")
                res = await adapter.execute(call)
                ctx.profiler.end_span(f"tool_{call.call_id}", EventType.TOOL_END)
                ctx.record_tool_result(res)

        return "Max turns reached without final answer."
