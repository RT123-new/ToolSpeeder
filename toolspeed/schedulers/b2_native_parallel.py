"""Baseline 2: Native Parallel Tool Calling."""

from __future__ import annotations

from typing import Any, List
import asyncio

from toolspeed.adapters.base import BaseLLMAdapter, ToolRegistry
from toolspeed.core.types import EventType, ToolCall, ToolResult
from toolspeed.schedulers.base import BaseScheduler, ExecutionContext


class NativeParallelScheduler(BaseScheduler):
    """Baseline 2: Native parallel tool calling.
    
    When a single model turn produces multiple independent tool calls, executes them concurrently
    using asyncio.gather while respecting concurrency limits.
    """

    async def _execute_tool_call(
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
                error=f"Tool '{call.name}' not found in registry",
            )

        ctx.guardrails.record_tool_dispatch(adapter.spec, call, is_speculative=False)
        ctx.profiler.start_span(f"tool_{call.call_id}")
        ctx.guardrails.record_concurrency_enter()

        await ctx.rate_limiter.acquire()
        try:
            result = await adapter.execute(call)
        finally:
            ctx.rate_limiter.release()
            ctx.guardrails.record_concurrency_exit()

        ctx.profiler.end_span(
            f"tool_{call.call_id}",
            EventType.TOOL_END,
            details={"tool": call.name, "call_id": call.call_id},
        )
        return result

    async def _execute_internal(
        self,
        ctx: ExecutionContext,
        model: BaseLLMAdapter,
        tools: ToolRegistry,
    ) -> Any:
        for turn in range(ctx.config.max_turns):
            ctx.step_count = turn + 1

            # 1. Model Turn
            ctx.profiler.start_span(f"model_turn_{turn}")
            decision = await model.decide(
                ctx.task,
                ctx.history,
                tools.list_specs(),
            )
            ctx.profiler.end_span(
                f"model_turn_{turn}",
                EventType.MODEL_END,
                details={"turn": turn, "tool_calls": len(decision.tool_calls)},
            )
            ctx.record_model_decision(decision)

            if decision.final_answer is not None or not decision.tool_calls:
                return decision.final_answer

            # 2. Parallel Tool Execution for all calls in this turn
            for call in decision.tool_calls:
                ctx.tool_calls.append(call)

            results: List[ToolResult] = await asyncio.gather(
                *[self._execute_tool_call(ctx, call, tools) for call in decision.tool_calls]
            )

            for res in results:
                ctx.record_tool_result(res)

        return "Max turns reached without final answer."
