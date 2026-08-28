"""Baseline 2: Native Parallel Tool Execution."""

from __future__ import annotations

import asyncio
from typing import Any

from toolspeed.adapters.base import BaseLLMAdapter, ToolRegistry
from toolspeed.core.types import EventType, ToolResult
from toolspeed.schedulers.base import BaseScheduler, ExecutionContext


class NativeParallelScheduler(BaseScheduler):
    """Baseline 2: Native Parallel Tool Execution.

    Executes multiple tool calls emitted in a single model decision turn concurrently using asyncio.gather.
    """

    async def _execute_internal(
        self,
        ctx: ExecutionContext,
        model: BaseLLMAdapter,
        tools: ToolRegistry,
    ) -> Any:
        for turn in range(ctx.config.max_turns):
            ctx.step_count = turn + 1

            # 1. Model Decision Step
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

            # 2. Parallel Tool Execution Step via ToolExecutor
            for call in decision.tool_calls:
                ctx.tool_calls.append(call)

            async def _run_call(call):
                return await ctx.executor.execute(call)

            results: list[ToolResult] = await asyncio.gather(*[_run_call(c) for c in decision.tool_calls])

            for res in results:
                ctx.record_tool_result(res)

        return "Max turns reached without final answer."
