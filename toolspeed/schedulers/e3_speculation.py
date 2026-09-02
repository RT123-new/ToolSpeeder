"""Experiment E3: Speculative Read Execution Scheduler."""

from __future__ import annotations

import asyncio
from typing import Any

from toolspeed.adapters.base import BaseLLMAdapter, LLMDecision, ToolRegistry
from toolspeed.core.rate_limiter import RateLimiter
from toolspeed.core.types import EventType, ToolCall, ToolResult
from toolspeed.schedulers.base import BaseScheduler, ExecutionContext, SchedulerConfig, cancel_and_await
from toolspeed.schedulers.executor import ToolExecutor


class SpeculativeReadScheduler(BaseScheduler):
    """Experiment E3: Speculative Read Execution Scheduler.

    Speculatively dispatches high-confidence read-only tools concurrently with model reasoning,
    cancelling on decision divergence and reconciling results on speculation hits.
    """

    def __init__(self, config: SchedulerConfig | None = None, speculation_enabled: bool | None = None) -> None:
        cfg = config or SchedulerConfig(speculation_enabled=True)
        if speculation_enabled is not None:
            cfg.speculation_enabled = speculation_enabled
        elif config is None:
            cfg.speculation_enabled = True
        super().__init__(cfg)

    def supports_concurrent_adapter(self, adapter: Any) -> bool:
        """Verifies whether an adapter is explicitly concurrency-safe for overlapped speculative execution."""
        return bool(getattr(adapter, "is_concurrency_safe", True))

    async def _safe_cancel_speculation(self, coro_or_task: Any) -> Exception | None:
        """Safely cancels speculative coroutine/task without leaking CancelledError."""
        task = coro_or_task if isinstance(coro_or_task, asyncio.Task) else asyncio.create_task(coro_or_task)
        task.cancel()
        try:
            await task
            return None
        except asyncio.CancelledError:
            return None
        except Exception as e:
            return e

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
                clock=ctx.clock,
            )
            isolated_executor = ToolExecutor(
                registry=tools,
                rate_limiter=isolated_limiter,
                profiler=ctx.profiler,
                guardrails=ctx.guardrails,
                default_timeout_s=ctx.config.timeout_seconds,
                authority_context=ctx.authority_context,
                clock=ctx.clock,
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

                # 1. Launch Draft Prediction and Main Model Reasoning CONCURRENTLY if speculation enabled
                if spec_enabled:
                    draft_task = asyncio.create_task(
                        model.predict_draft(ctx.agent_task, ctx.history, tools.list_specs())
                    )

                ctx.profiler.start_span(f"model_turn_{turn}")
                model_decision_task = asyncio.create_task(model.decide(ctx.agent_task, ctx.history, tools.list_specs()))

                if spec_enabled and draft_task is not None:
                    # Wait for whichever completes first
                    done, _pending = await asyncio.wait(
                        [draft_task, model_decision_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # Check if draft prediction finished first
                    if draft_task in done and not draft_task.cancelled():
                        try:
                            predicted = draft_task.result()
                            if predicted is not None and predicted.speculation_confidence >= threshold:
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
                                    ctx.profiler.record_event(
                                        EventType.SPECULATION_START,
                                        details={
                                            "tool": speculative_call.name,
                                            "confidence": speculative_call.speculation_confidence,
                                            "mode": contention_mode,
                                        },
                                    )
                                    exec_to_use = (
                                        isolated_executor
                                        if (contention_mode == "isolated" and isolated_executor is not None)
                                        else ctx.executor
                                    )
                                    spec_task = asyncio.create_task(
                                        exec_to_use.execute(speculative_call, is_speculative=True)
                                    )
                        except Exception:
                            spec_task = None
                            speculative_call = None

                # 2. Await Main Model Decision
                decision = await model_decision_task
                ctx.profiler.end_span(
                    f"model_turn_{turn}",
                    EventType.MODEL_END,
                    details={"turn": turn, "calls": len(decision.tool_calls)},
                )
                ctx.record_model_decision(decision)

                # Clean up draft task if still running
                if draft_task and not draft_task.done():
                    await cancel_and_await(draft_task)

                if decision.final_answer is not None or not decision.tool_calls:
                    # Cancel in-flight speculative task on final answer
                    if spec_task and not spec_task.done():
                        await cancel_and_await(spec_task)
                        ctx.guardrails.record_speculation_resolved(hit=False, cancelled=True)
                        ctx.profiler.record_event(EventType.SPECULATION_CANCELLED)
                    elif spec_task and spec_task.done():
                        ctx.guardrails.record_speculation_resolved(hit=False, cancelled=False)
                        ctx.profiler.record_event(EventType.SPECULATION_MISS)
                    return decision.final_answer

                # 3. Check for Speculation Hit vs Miss
                speculation_hit = False
                hit_result: ToolResult | None = None
                matching_call_index = -1

                if speculative_call is not None and spec_task is not None:
                    # Compare predicted call against authoritative decision calls
                    for idx, call in enumerate(decision.tool_calls):
                        pred_name = speculative_call.name or speculative_call.tool_name
                        act_name = call.name or call.tool_name
                        if pred_name == act_name and speculative_call.arguments == call.arguments:
                            speculation_hit = True
                            matching_call_index = idx
                            break

                if speculation_hit and spec_task is not None:
                    # Hit: await the speculative result
                    try:
                        hit_result = await spec_task
                        if hit_result.is_success:
                            ctx.guardrails.record_speculation_resolved(hit=True)
                            ctx.profiler.record_event(
                                EventType.SPECULATION_HIT,
                                details={
                                    "tool": speculative_call.name if speculative_call else "unknown",
                                    "confidence": speculative_call.speculation_confidence if speculative_call else 1.0,
                                },
                            )
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
                            await cancel_and_await(spec_task)
                            ctx.guardrails.record_speculation_resolved(hit=False, cancelled=True)
                            ctx.profiler.record_event(EventType.SPECULATION_CANCELLED)
                        elif contention_mode == "single_slot":
                            try:
                                await spec_task
                            except Exception:
                                pass
                            ctx.guardrails.record_speculation_resolved(hit=False, cancelled=False)
                            ctx.profiler.record_event(EventType.SPECULATION_MISS)
                        elif contention_mode == "isolated":
                            await cancel_and_await(spec_task)
                            ctx.guardrails.record_speculation_resolved(hit=False, cancelled=True)
                            ctx.profiler.record_event(EventType.SPECULATION_MISS)
                    elif spec_task and spec_task.done():
                        ctx.guardrails.record_speculation_resolved(hit=False, cancelled=False)
                        ctx.profiler.record_event(EventType.SPECULATION_MISS)
                    spec_task = None
                    speculative_call = None

                # 4. Execute all decision tool calls
                for idx, call in enumerate(decision.tool_calls):
                    ctx.tool_calls.append(call)
                    if idx == matching_call_index and speculation_hit and hit_result is not None:
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
