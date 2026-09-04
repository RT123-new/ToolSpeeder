"""Baseline 4: Oracle DAG Lower Bound Scheduler."""

from __future__ import annotations

import asyncio
from typing import Any

from toolspeed.adapters.base import BaseLLMAdapter, ToolRegistry
from toolspeed.core.types import EventType, ToolCall
from toolspeed.schedulers.base import BaseScheduler, ExecutionContext


class OracleDAGScheduler(BaseScheduler):
    """Baseline 4: Oracle DAG lower bound scheduler.

    Computes theoretical maximum parallelism / minimal critical path by executing the optimal
    dependency DAG in concurrent topological waves with zero intermediate model reasoning.
    """

    async def _execute_internal(
        self,
        ctx: ExecutionContext,
        model: BaseLLMAdapter,
        tools: ToolRegistry,
    ) -> Any:
        oracle_plan = None
        if hasattr(ctx.task, "metadata"):
            oracle_plan = ctx.task.metadata.get("oracle_plan")
        if not oracle_plan and hasattr(ctx.task, "context"):
            oracle_plan = ctx.task.context.get("oracle_plan")

        if oracle_plan and isinstance(oracle_plan, list):
            accumulated_outputs: dict[str, Any] = {}

            for wave_idx, wave_calls in enumerate(oracle_plan):
                ctx.profiler.record_event(
                    EventType.DAG_NODE_DISPATCH,
                    details={"wave": wave_idx, "call_count": len(wave_calls)},
                )

                resolved_calls: list[ToolCall] = []
                for item in wave_calls:
                    if isinstance(item, ToolCall):
                        call = item
                    elif isinstance(item, dict):
                        call = ToolCall(name=item["name"], arguments=dict(item.get("arguments", {})))
                    else:
                        continue

                    # Substitute arguments from prior outputs
                    resolved_args = dict(call.arguments)
                    for k, v in list(resolved_args.items()):
                        if isinstance(v, str) and v.startswith("$"):
                            ref_key = v[1:]
                            # Support $node.field or $node
                            parts = ref_key.split(".", 1)
                            base_ref = parts[0]
                            if base_ref in accumulated_outputs:
                                out_val = accumulated_outputs[base_ref]
                                if len(parts) > 1 and isinstance(out_val, dict):
                                    resolved_args[k] = out_val.get(parts[1], out_val)
                                else:
                                    resolved_args[k] = out_val
                    call.arguments = resolved_args
                    resolved_calls.append(call)
                    ctx.tool_calls.append(call)

                results = await asyncio.gather(*[ctx.executor.execute(call) for call in resolved_calls])

                for res in results:
                    ctx.record_tool_result(res)
                    accumulated_outputs[res.name] = res.output if res.output is not None else res.result
                    accumulated_outputs[res.call_id] = res.output if res.output is not None else res.result

            answer_fn = ctx.task.metadata.get("oracle_final_answer_fn") if hasattr(ctx.task, "metadata") else None
            if callable(answer_fn):
                return answer_fn(accumulated_outputs)

            return accumulated_outputs

        # Fallback: initial 1-shot model plan
        ctx.profiler.start_span("oracle_initial_plan")
        decision = await model.decide(ctx.agent_task, ctx.history, tools.list_specs())
        ctx.profiler.end_span("oracle_initial_plan", EventType.MODEL_END)
        ctx.record_model_decision(decision)

        if decision.final_answer is not None or not decision.tool_calls:
            return decision.final_answer

        for call in decision.tool_calls:
            ctx.tool_calls.append(call)

        results = await asyncio.gather(*[ctx.executor.execute(call) for call in decision.tool_calls])
        for res in results:
            ctx.record_tool_result(res)

        # Single final synthesis turn
        ctx.profiler.start_span("oracle_synthesis")
        final_decision = await model.decide(ctx.agent_task, ctx.history, tools.list_specs())
        ctx.profiler.end_span("oracle_synthesis", EventType.MODEL_END)
        ctx.record_model_decision(final_decision)

        return final_decision.final_answer
