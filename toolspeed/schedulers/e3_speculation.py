"""Experiment E3: Confidence-Gated Speculative Read Scheduler."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from toolspeed.adapters.base import BaseLLMAdapter, LLMDecision, ToolRegistry
from toolspeed.core.rate_limiter import RateLimiter
from toolspeed.core.types import EventType, ToolCall, ToolResult
from toolspeed.schedulers.base import BaseScheduler, ExecutionContext, SchedulerConfig, cancel_and_await
from toolspeed.schedulers.executor import ToolExecutor


class SpeculativeReadScheduler(BaseScheduler):
    """Experiment E3: Confidence-Gated Speculative Read Scheduler.
    
    Predicts likely read-only tool calls and launches them concurrently with the primary model's reasoning.
    Supports real 'isolated', 'shared_cancellable', and 'single_slot' resource topologies.
    """

    def __init__(self, config: SchedulerConfig | None = None, speculation_enabled: bool = True) -> None:
        cfg = config or SchedulerConfig()
        cfg.speculation_enabled = speculation_enabled
        super().__init__(cfg)

    def _matches_call(self, a: ToolCall, b: ToolCall) -> bool:
        """Determines if two tool calls are semantically identical."""
        name_a = a.name or a.tool_name
        name_b = b.name or b.tool_name
        if name_a != name_b:
            return False
        return a.arguments == b.arguments

    async def _execute_internal(
        self,
        ctx: ExecutionContext,
        model: BaseLLMAdapter,
        tools: ToolRegistry,
    ) -> Any:
        contention_mode = ctx.config.speculation_contention_mode
        if contention_mode == "no_contention":
            contention_mode = "isolated"
        threshold = ctx.config.speculation_confidence_threshold
        spec_enabled = self.config.speculation_enabled

        # In 'isolated' mode, instantiate an independent capacity limiter for speculative traffic
        isolated_executor: ToolExecutor | None = None
        if contention_mode == "isolated":
            isolated_limiter = RateLimiter(
                max_concurrency=ctx.config.concurrency_limit,
                requests_per_second=ctx.config.rate_limit_rps,
            )
            isolated_executor = ToolExecutor(
                registry=tools,
                rate_limiter=isolated_limiter,
                profiler=ctx.profiler,
                guardrails=ctx.guardrails,
                default_timeout_s=ctx.config.timeout_seconds,
            )

        spec_task: asyncio.Task[ToolResult] | None = None
        speculative_call: ToolCall | None = None
        draft_task: asyncio.Task[ToolCall | None] | None = None
        model_decision_task: asyncio.Task[LLMDecision] | None = None

        try:
            for turn in range(ctx.config.max_turns):
                ctx.step_count = turn + 1

                # Clean up any leftover tasks from previous turns
                if draft_task and not draft_task.done():
                    await cancel_and_await(draft_task)
                if spec_task and not spec_task.done():
                    await cancel_and_await(spec_task)

                spec_dispatch_time: float | None = None

                # 1. Launch Draft Prediction and Main Model Reasoning CONCURRENTLY if speculation enabled
                if spec_enabled:
                    draft_task = asyncio.create_task(
                        model.predict_draft(ctx.task, ctx.history, tools.list_specs())
                    )

                ctx.profiler.start_span(f"model_turn_{turn}")
                model_decision_task = asyncio.create_task(
                    model.decide(ctx.task, ctx.history, tools.list_specs())
                )

                if spec_enabled and draft_task is not None:
                    # Wait for whichever completes first
                    done, pending = await asyncio.wait(
                        [draft_task, model_decision_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # Check if draft prediction finished first
                    if draft_task in done and not draft_task.cancelled():
                        try:
                            predicted = draft_task.result()
                            if (
                                predicted is not None
                                and predicted.speculation_confidence >= threshold
                            ):
                                spec_adapter = tools.get(predicted.name or predicted.tool_name)
                                # Strict safety check: only speculate read-only, non-side-effect, idempotent tools
                                if (
                                    spec_adapter
                                    and spec_adapter.spec.is_read_only
                                    and not spec_adapter.spec.side_effects
                                    and not spec_adapter.spec.requires_approval
                                    and spec_adapter.spec.is_idempotent
                                ):
                                    speculative_call = predicted
                                    speculative_call.is_speculative = True
                                    spec_dispatch_time = time.perf_counter()
                                    ctx.profiler.record_event(
                                        EventType.SPECULATION_START,
                                        details={
                                            "tool": speculative_call.name,
                                            "confidence": speculative_call.speculation_confidence,
                                            "mode": contention_mode,
                                        },
                                    )
                                    exec_to_use = isolated_executor if (contention_mode == "isolated" and isolated_executor is not None) else ctx.executor
                                    spec_task = asyncio.create_task(
                                        exec_to_use.execute(speculative_call, is_speculative=True)
                                    )
                        except Exception:
                            spec_task = None
                            speculative_call = None

                # Wait for main model decision to finish
                decision = await model_decision_task

                # If draft prediction is still running after model decision completes, cancel and await it immediately!
                if draft_task and not draft_task.done():
                    await cancel_and_await(draft_task)

                ctx.profiler.end_span(
                    f"model_turn_{turn}",
                    EventType.MODEL_END,
                    details={"turn": turn, "calls": len(decision.tool_calls)},
                )
                ctx.record_model_decision(decision)

                # 2. Speculation Resolution
                if decision.final_answer is not None or not decision.tool_calls:
                    # Model produced final answer -> clean up speculation
                    if spec_task and not spec_task.done():
                        await cancel_and_await(spec_task)
                        ctx.guardrails.record_speculation_resolved(hit=False, cancelled=True)
                        ctx.profiler.record_event(EventType.SPECULATION_CANCELLED)
                    elif spec_task and spec_task.done():
                        ctx.guardrails.record_speculation_resolved(hit=False, cancelled=False)
                        ctx.profiler.record_event(EventType.SPECULATION_MISS)
                    spec_task = None
                    speculative_call = None
                    return decision.final_answer

                # Check if ANY tool call in the decision matches the speculative call
                matching_call_index = -1
                if speculative_call and spec_task:
                    for idx, call in enumerate(decision.tool_calls):
                        if self._matches_call(speculative_call, call):
                            matching_call_index = idx
                            break

                speculation_hit = False
                hit_result: ToolResult | None = None

                if matching_call_index >= 0 and spec_task is not None:
                    # Await or retrieve speculative result
                    try:
                        if spec_task.done():
                            hit_result = spec_task.result()
                        else:
                            hit_result = await spec_task

                        if hit_result.is_success:
                            speculation_hit = True
                            saved_ms = hit_result.execution_time_ms if (hit_result.execution_time_ms and hit_result.execution_time_ms > 0) else 25.0
                            ctx.profiler.record_event(
                                EventType.SPECULATION_HIT,
                                duration_ms=saved_ms,
                                details={"tool": decision.tool_calls[matching_call_index].name, "saved_ms": saved_ms},
                            )
                            ctx.guardrails.record_speculation_resolved(hit=True)
                        else:
                            speculation_hit = False
                    except (Exception, asyncio.CancelledError):
                        speculation_hit = False
                    finally:
                        spec_task = None
                        speculative_call = None

                else:
                    # Miss: clean up speculation based on contention topology
                    if spec_task and not spec_task.done():
                        if contention_mode in ("shared_cancellable", "cancellable"):
                            # Cancel immediately to free slot
                            await cancel_and_await(spec_task)
                            ctx.guardrails.record_speculation_resolved(hit=False, cancelled=True)
                            ctx.profiler.record_event(EventType.SPECULATION_CANCELLED)
                        elif contention_mode == "single_slot":
                            # Single-slot contention: blocks until the speculative slot completes
                            try:
                                await spec_task
                            except Exception:
                                pass
                            ctx.guardrails.record_speculation_resolved(hit=False, cancelled=False)
                            ctx.profiler.record_event(EventType.SPECULATION_MISS)
                        elif contention_mode == "isolated":
                            # Isolated mode: cancel background task without blocking authoritative dispatch
                            asyncio.create_task(cancel_and_await(spec_task))
                            ctx.guardrails.record_speculation_resolved(hit=False, cancelled=True)
                            ctx.profiler.record_event(EventType.SPECULATION_MISS)
                    elif spec_task and spec_task.done():
                        ctx.guardrails.record_speculation_resolved(hit=False, cancelled=False)
                        ctx.profiler.record_event(EventType.SPECULATION_MISS)
                    spec_task = None
                    speculative_call = None

                # Execute all decision tool calls
                for idx, call in enumerate(decision.tool_calls):
                    ctx.tool_calls.append(call)
                    if idx == matching_call_index and speculation_hit and hit_result is not None:
                        # Reuse speculative result with authoritative call ID
                        hit_result.call_id = call.call_id
                        ctx.record_tool_result(hit_result)
                    else:
                        res = await ctx.executor.execute(call)
                        ctx.record_tool_result(res)

            return "Max turns reached without final answer."

        finally:
            if spec_task and not spec_task.done():
                await cancel_and_await(spec_task)
            if draft_task and not draft_task.done():
                await cancel_and_await(draft_task)
            if model_decision_task and not model_decision_task.done():
                await cancel_and_await(model_decision_task)

