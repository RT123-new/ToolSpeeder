"""Baseline 1: Synchronous Serial ReAct Loop."""

from __future__ import annotations

from typing import Any

from toolspeed.adapters.base import BaseLLMAdapter, ToolRegistry
from toolspeed.core.types import EventType
from toolspeed.schedulers.base import BaseScheduler, ExecutionContext


class SyncReActScheduler(BaseScheduler):
    """Baseline 1: Standard synchronous serial ReAct loop.

    Model decisions and tool calls are executed strictly sequentially on the critical path.
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

            # If model produced final answer or no tools to call, finish
            if decision.final_answer is not None or not decision.tool_calls:
                return decision.final_answer

            # 2. Sequential Tool Execution Step (strictly serial via ToolExecutor)
            for call in decision.tool_calls:
                ctx.tool_calls.append(call)
                result = await ctx.executor.execute(call)
                ctx.record_tool_result(result)

        return "Max turns reached without final answer."
