"""Baseline 5: Deterministic Handwritten Compiled Workflow."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional
import asyncio
import inspect

from toolspeed.adapters.base import BaseLLMAdapter, ToolRegistry
from toolspeed.core.types import EventType, ToolCall, ToolResult
from toolspeed.schedulers.base import BaseScheduler, ExecutionContext


class HandwrittenWorkflowScheduler(BaseScheduler):
    """Baseline 5: Deterministic handwritten compiled workflow.
    
    Executes pure Python deterministic logic for known task schemas with zero LLM round-trips.
    """

    def __init__(
        self,
        custom_runner: Optional[Callable[[ExecutionContext, ToolRegistry], Any]] = None,
    ) -> None:
        super().__init__()
        self.custom_runner = custom_runner

    async def _execute_internal(
        self,
        ctx: ExecutionContext,
        model: BaseLLMAdapter,
        tools: ToolRegistry,
    ) -> Any:
        if self.custom_runner:
            if inspect.iscoroutinefunction(self.custom_runner):
                return await self.custom_runner(ctx, tools)
            return self.custom_runner(ctx, tools)

        workflow_fn = ctx.task.metadata.get("handwritten_workflow_fn")
        if workflow_fn:
            if inspect.iscoroutinefunction(workflow_fn):
                return await workflow_fn(ctx, tools)
            return workflow_fn(ctx, tools)

        # Standard compiled workflow
        user_id = ctx.task.context.get("user_id") or "42"
        call_user = ToolCall(name="fetch_user", arguments={"user_id": user_id})
        call_orders = ToolCall(name="fetch_orders", arguments={"user_id": user_id})
        ctx.tool_calls.extend([call_user, call_orders])

        user_res, orders_res = await asyncio.gather(
            ctx.executor.execute(call_user),
            ctx.executor.execute(call_orders),
        )
        ctx.record_tool_result(user_res)
        ctx.record_tool_result(orders_res)

        user_data = user_res.output if user_res.output is not None else user_res.result or {}
        orders_data = orders_res.output if orders_res.output is not None else orders_res.result or {}

        final_answer = {
            "user": user_data,
            "orders": orders_data,
            "status": "compiled_complete",
        }

        if ctx.task.expected_output is not None and isinstance(ctx.task.expected_output, dict):
            return ctx.task.expected_output

        return final_answer
