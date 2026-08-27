"""Experiment E4: Commit-Horizon Streaming Dispatch Scheduler."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
import asyncio
import copy
import time

from toolspeed.adapters.base import BaseLLMAdapter, LLMDecision, StreamingChunk, ToolRegistry
from toolspeed.core.types import EventType, ToolCall, ToolResult, ToolSpec
from toolspeed.schedulers.base import BaseScheduler, ExecutionContext


class CommitHorizonScheduler(BaseScheduler):
    """Experiment E4: Commit-Horizon Streaming Dispatch Scheduler.
    
    Dispatches tool execution as soon as required semantics-changing arguments are streamed,
    overlapping remainder token generation with tool runtime.
    """

    async def _execute_tool(
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
                is_error=True,
            )

        ctx.guardrails.record_tool_dispatch(adapter.spec, call, is_speculative=False)
        ctx.profiler.start_span(f"commit_tool_{call.call_id}")
        ctx.guardrails.record_concurrency_enter()

        try:
            await ctx.rate_limiter.acquire()
            try:
                res = await adapter.execute(call)
            finally:
                ctx.rate_limiter.release()
                ctx.guardrails.record_concurrency_exit()
        except asyncio.CancelledError:
            ctx.profiler.record_event(
                EventType.TOOL_CANCELLED,
                details={"call_id": call.call_id, "tool": call.name},
            )
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                cancelled=True,
                error="Cancelled",
                is_error=True,
                execution_time_ms=0.0,
            )
        except Exception as e:
            res = ToolResult(
                call_id=call.call_id,
                name=call.name,
                error=str(e),
                is_error=True,
            )

        ctx.profiler.end_span(
            f"commit_tool_{call.call_id}",
            EventType.TOOL_END,
            details={"tool": call.name, "early_dispatched": call.committed_early},
        )
        return res

    async def _execute_internal(
        self,
        ctx: ExecutionContext,
        model: BaseLLMAdapter,
        tools: ToolRegistry,
    ) -> Any:
        in_flight_dispatches: Dict[str, Tuple[asyncio.Task[ToolResult], ToolCall, Dict[str, Any]]] = {}

        try:
            for turn in range(ctx.config.max_turns):
                ctx.step_count = turn + 1

                ctx.profiler.start_span(f"stream_model_turn_{turn}")
                stream_start = time.perf_counter()

                in_flight_dispatches.clear()
                collected_chunks: List[StreamingChunk] = []
                final_tool_calls: List[ToolCall] = []
                final_reasoning: List[str] = []

                # Consume streaming chunks
                async for chunk in model.stream_decision(ctx.task, ctx.history, tools.list_specs()):
                    collected_chunks.append(chunk)
                    if chunk.delta_text:
                        final_reasoning.append(chunk.delta_text)

                    # Check if commit horizon is reached for any tool call
                    for early_call in chunk.commit_horizon_ready:
                        if early_call.call_id not in in_flight_dispatches:
                            early_call.committed_early = True
                            early_snapshot = copy.deepcopy(early_call.arguments)

                            ctx.profiler.record_event(
                                EventType.COMMIT_HORIZON_REACHED,
                                details={
                                    "tool": early_call.name,
                                    "call_id": early_call.call_id,
                                    "token_index": chunk.token_index,
                                    "lead_time_ms": (time.perf_counter() - stream_start) * 1000.0,
                                },
                            )

                            # Dispatch immediately in background
                            task = asyncio.create_task(
                                self._execute_tool(ctx, early_call, tools)
                            )
                            in_flight_dispatches[early_call.call_id] = (task, early_call, early_snapshot)

                    if chunk.is_final and chunk.parsed_tool_calls:
                        final_tool_calls = chunk.parsed_tool_calls

                ctx.profiler.end_span(
                    f"stream_model_turn_{turn}",
                    EventType.MODEL_END,
                    details={"turn": turn, "early_dispatches": len(in_flight_dispatches)},
                )

                # Build full decision object
                last_meta = collected_chunks[-1].metadata if collected_chunks else {}
                final_ans = last_meta.get("final_answer")
                if final_ans is None and not (final_tool_calls or in_flight_dispatches):
                    final_ans = "".join(final_reasoning).strip()

                decision = LLMDecision(
                    reasoning="".join(final_reasoning),
                    tool_calls=final_tool_calls or [c for _, c, _ in in_flight_dispatches.values()],
                    final_answer=final_ans if not (final_tool_calls or in_flight_dispatches) else None,
                    output_tokens=len(collected_chunks),
                )
                ctx.record_model_decision(decision)

                if decision.is_final:
                    # No tools called -> return final answer
                    return decision.final_answer or "Task completed."

                # Await all early-dispatched tools and check argument immutability
                for call in decision.tool_calls:
                    ctx.tool_calls.append(call)

                    if call.call_id in in_flight_dispatches:
                        task, dispatched_call, snapshot = in_flight_dispatches[call.call_id]
                        
                        # Immutability Check: verify post-dispatch arguments didn't mutate
                        if call.arguments != snapshot:
                            if not task.done():
                                task.cancel()
                                try:
                                    await task
                                except (asyncio.CancelledError, Exception):
                                    pass

                            ctx.profiler.record_event(
                                EventType.GUARDRAIL_VIOLATION,
                                details={
                                    "error": "Semantic mismatch: arguments mutated after commit horizon dispatch!",
                                    "dispatched": snapshot,
                                    "final": call.arguments,
                                },
                            )
                            # Re-execute with mutated arguments
                            res = await self._execute_tool(ctx, call, tools)
                        else:
                            res = await task

                        ctx.record_tool_result(res)

                    else:
                        # Tool call not early-dispatched: run now
                        res = await self._execute_tool(ctx, call, tools)
                        ctx.record_tool_result(res)

            return "Max turns reached without final answer."
        finally:
            if in_flight_dispatches:
                for t, _, _ in in_flight_dispatches.values():
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*[t for t, _, _ in in_flight_dispatches.values()], return_exceptions=True)
