"""Experiment E3: Confidence-Gated Speculative Read Scheduler."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import asyncio
import time

from toolspeed.adapters.base import BaseLLMAdapter, LLMDecision, ToolRegistry
from toolspeed.core.types import EventType, ToolCall, ToolResult, ToolSpec
from toolspeed.schedulers.base import BaseScheduler, ExecutionContext


class SpeculativeReadScheduler(BaseScheduler):
    """Experiment E3: Confidence-Gated Speculative Read Scheduler.
    
    Predicts likely read-only tool calls and launches them in parallel while the primary model reasons.
    Supports 'no_contention', 'cancellable', and 'single_slot' contention modes.
    """

    async def _execute_tool_call(
        self,
        ctx: ExecutionContext,
        call: ToolCall,
        tools: ToolRegistry,
        is_speculative: bool = False,
    ) -> ToolResult:
        adapter = tools.get(call.name)
        if not adapter:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                error=f"Tool '{call.name}' not found",
                is_error=True,
            )

        ctx.guardrails.record_tool_dispatch(adapter.spec, call, is_speculative=is_speculative)
        span_name = f"{'spec_' if is_speculative else ''}tool_{call.call_id}"
        ctx.profiler.start_span(span_name)
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
            span_name,
            EventType.TOOL_END,
            details={"tool": call.name, "speculative": is_speculative},
        )
        return res

    def _matches_call(self, a: ToolCall, b: ToolCall) -> bool:
        """Determines if two tool calls are semantically identical."""
        if a.name != b.name:
            return False
        return a.arguments == b.arguments

    async def _execute_internal(
        self,
        ctx: ExecutionContext,
        model: BaseLLMAdapter,
        tools: ToolRegistry,
    ) -> Any:
        contention_mode = ctx.config.speculation_contention_mode
        threshold = ctx.config.speculation_confidence_threshold
        spec_task: Optional[asyncio.Task[ToolResult]] = None
        speculative_call: Optional[ToolCall] = None

        try:
            for turn in range(ctx.config.max_turns):
                ctx.step_count = turn + 1

                # 1. Check for draft speculative prediction
                spec_start_time: float = 0.0

                try:
                    predicted = await model.predict_draft(ctx.task, ctx.history, tools.list_specs())
                    if (
                        predicted
                        and predicted.speculation_confidence >= threshold
                    ):
                        tool_adapter = tools.get(predicted.name)
                        # Enforce guardrail safety: only read-only tools without side effects
                        if tool_adapter and tool_adapter.spec.is_read_only and not tool_adapter.spec.side_effects:
                            speculative_call = predicted
                            speculative_call.is_speculative = True
                            ctx.profiler.record_event(
                                EventType.SPECULATION_START,
                                details={
                                    "tool": predicted.name,
                                    "confidence": predicted.speculation_confidence,
                                    "mode": contention_mode,
                                },
                            )
                            spec_start_time = time.perf_counter()
                            spec_task = asyncio.create_task(
                                self._execute_tool_call(ctx, predicted, tools, is_speculative=True)
                            )
                except Exception:
                    spec_task = None
                    speculative_call = None

                # 2. Main Model Reasoning (runs concurrently with speculative task)
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

                # 3. Speculation Resolution
                if decision.final_answer is not None or not decision.tool_calls:
                    # Model finished or produced final answer
                    if spec_task and not spec_task.done():
                        if contention_mode == "cancellable":
                            spec_task.cancel()
                            try:
                                await spec_task
                            except (asyncio.CancelledError, Exception):
                                pass
                            ctx.guardrails.record_speculation_resolved(hit=False, cancelled=True)
                            ctx.profiler.record_event(EventType.SPECULATION_CANCELLED)
                        else:
                            # Wasted speculation
                            ctx.guardrails.record_speculation_resolved(hit=False, cancelled=False)
                            ctx.profiler.record_event(EventType.SPECULATION_MISS)
                    elif spec_task and spec_task.done():
                        ctx.guardrails.record_speculation_resolved(hit=False, cancelled=False)
                        ctx.profiler.record_event(EventType.SPECULATION_MISS)
                    spec_task = None
                    speculative_call = None
                    return decision.final_answer

                # Check if any model tool call matches the speculative call
                matched_spec_call = False
                for call in decision.tool_calls:
                    ctx.tool_calls.append(call)

                    if (
                        spec_task
                        and speculative_call
                        and not matched_spec_call
                        and self._matches_call(speculative_call, call)
                    ):
                        # Speculation Hit!
                        matched_spec_call = True
                        hit_res: ToolResult
                        if spec_task.done():
                            hit_res = spec_task.result()
                            saved_ms = max(0.0, hit_res.execution_time_ms)
                        else:
                            hit_res = await spec_task
                            saved_ms = max(0.0, (time.perf_counter() - spec_start_time) * 1000.0)

                        ctx.profiler.record_event(
                            EventType.SPECULATION_HIT,
                            duration_ms=saved_ms,
                            details={"tool": call.name, "saved_ms": saved_ms},
                        )
                        ctx.guardrails.record_speculation_resolved(hit=True)
                        ctx.record_tool_result(hit_res)
                        spec_task = None
                        speculative_call = None

                    else:
                        # Non-matching or subsequent call: execute normally
                        if spec_task and not matched_spec_call:
                            if not spec_task.done():
                                if contention_mode == "cancellable":
                                    spec_task.cancel()
                                    try:
                                        await spec_task
                                    except (asyncio.CancelledError, Exception):
                                        pass
                                    ctx.guardrails.record_speculation_resolved(hit=False, cancelled=True)
                                    ctx.profiler.record_event(EventType.SPECULATION_CANCELLED)
                                elif contention_mode == "single_slot":
                                    await spec_task
                                    ctx.guardrails.record_speculation_resolved(hit=False, cancelled=False)
                                    ctx.profiler.record_event(EventType.SPECULATION_MISS)
                                else:  # no_contention
                                    ctx.guardrails.record_speculation_resolved(hit=False, cancelled=False)
                                    ctx.profiler.record_event(EventType.SPECULATION_MISS)
                            else:
                                ctx.guardrails.record_speculation_resolved(hit=False, cancelled=False)
                                ctx.profiler.record_event(EventType.SPECULATION_MISS)
                            spec_task = None
                            speculative_call = None

                        actual_res = await self._execute_tool_call(ctx, call, tools, is_speculative=False)
                        ctx.record_tool_result(actual_res)

            return "Max turns reached without final answer."
        finally:
            if spec_task and not spec_task.done():
                spec_task.cancel()
                try:
                    await spec_task
                except (asyncio.CancelledError, Exception):
                    pass
