"""Baseline 4: Oracle DAG Lower Bound Scheduler."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Set, Tuple
import asyncio

from toolspeed.adapters.base import BaseLLMAdapter, ToolRegistry
from toolspeed.core.types import EventType, ToolCall, ToolResult
from toolspeed.schedulers.base import BaseScheduler, ExecutionContext


class OracleDAGScheduler(BaseScheduler):
    """Baseline 4: Oracle DAG lower bound scheduler.
    
    Computes theoretical maximum parallelism / minimal critical path by executing the optimal
    dependency DAG in concurrent topological waves with zero intermediate model reasoning.
    """

    async def _execute_single_call(
        self,
        ctx: ExecutionContext,
        call: ToolCall,
        tools: ToolRegistry,
    ) -> ToolResult:
        adapter = tools.get(call.name)
        if not adapter:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                error=f"Tool '{call.name}' not found",
            )

        ctx.guardrails.record_tool_dispatch(adapter.spec, call, is_speculative=False)
        ctx.profiler.start_span(f"oracle_tool_{call.call_id}")
        ctx.guardrails.record_concurrency_enter()

        await ctx.rate_limiter.acquire()
        try:
            result = await adapter.execute(call)
        finally:
            ctx.rate_limiter.release()
            ctx.guardrails.record_concurrency_exit()

        ctx.profiler.end_span(
            f"oracle_tool_{call.call_id}",
            EventType.TOOL_END,
            details={"tool": call.name, "oracle": True},
        )
        return result

    async def _execute_internal(
        self,
        ctx: ExecutionContext,
        model: BaseLLMAdapter,
        tools: ToolRegistry,
    ) -> Any:
        # Check if oracle plan is provided in task metadata or context
        oracle_plan = ctx.task.metadata.get("oracle_plan") or ctx.task.context.get("oracle_plan")

        if oracle_plan and isinstance(oracle_plan, list):
            # oracle_plan is a list of waves (List[List[ToolCall]])
            # Each wave runs fully concurrently; subsequent waves can use previous wave outputs
            accumulated_outputs: Dict[str, Any] = {}

            for wave_idx, wave_calls in enumerate(oracle_plan):
                ctx.profiler.record_event(
                    EventType.DAG_NODE_DISPATCH,
                    details={"wave": wave_idx, "call_count": len(wave_calls)},
                )

                # Resolve dynamic argument references from prior outputs if any
                resolved_calls: List[ToolCall] = []
                for item in wave_calls:
                    if isinstance(item, ToolCall):
                        call = item
                    elif isinstance(item, dict):
                        call = ToolCall(name=item["name"], arguments=dict(item.get("arguments", {})))
                    else:
                        continue

                    # Substitute arguments if needed (e.g. $wave0.output)
                    resolved_args = dict(call.arguments)
                    for k, v in list(resolved_args.items()):
                        if isinstance(v, str) and v.startswith("$"):
                            ref_key = v[1:]
                            if ref_key in accumulated_outputs:
                                resolved_args[k] = accumulated_outputs[ref_key]
                    call.arguments = resolved_args
                    resolved_calls.append(call)
                    ctx.tool_calls.append(call)

                # Execute all calls in current wave in parallel
                results = await asyncio.gather(
                    *[self._execute_single_call(ctx, call, tools) for call in resolved_calls]
                )

                for res in results:
                    ctx.record_tool_result(res)
                    accumulated_outputs[res.name] = res.output
                    accumulated_outputs[res.call_id] = res.output

            # Oracle construct final answer
            answer_fn = ctx.task.metadata.get("oracle_final_answer_fn")
            if callable(answer_fn):
                return answer_fn(accumulated_outputs)

            if ctx.task.expected_output is not None:
                return ctx.task.expected_output

            return accumulated_outputs

        # Fallback: if no explicit oracle plan, request initial 1-shot model plan and execute in parallel
        ctx.profiler.start_span("oracle_initial_plan")
        decision = await model.decide(ctx.task, ctx.history, tools.list_specs())
        ctx.profiler.end_span("oracle_initial_plan", EventType.MODEL_END)
        ctx.record_model_decision(decision)

        if decision.final_answer is not None or not decision.tool_calls:
            return decision.final_answer

        for call in decision.tool_calls:
            ctx.tool_calls.append(call)

        results = await asyncio.gather(
            *[self._execute_single_call(ctx, call, tools) for call in decision.tool_calls]
        )
        for res in results:
            ctx.record_tool_result(res)

        # Single final synthesis turn
        ctx.profiler.start_span("oracle_synthesis")
        final_decision = await model.decide(ctx.task, ctx.history, tools.list_specs())
        ctx.profiler.end_span("oracle_synthesis", EventType.MODEL_END)
        ctx.record_model_decision(final_decision)

        return final_decision.final_answer or ctx.task.expected_output
