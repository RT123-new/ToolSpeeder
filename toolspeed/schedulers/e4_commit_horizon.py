"""Experiment E4: Commit-Horizon Streaming Early Dispatch Scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import asyncio
import copy
import hashlib
import json
import time

from toolspeed.adapters.base import BaseLLMAdapter, LLMDecision, StreamingChunk, ToolRegistry
from toolspeed.core.types import CommittedCall, EventType, ToolCall, ToolResult, ToolSpec
from toolspeed.schedulers.base import BaseScheduler, ExecutionContext


class IncrementalCommitParser:
    """Incremental streaming parser and schema-aware commit validator."""

    @staticmethod
    def is_syntax_closed(json_fragment: str) -> bool:
        """Verifies whether a JSON fragment is syntactically closed and complete."""
        s = json_fragment.strip()
        if not s:
            return False
        try:
            json.loads(s)
            return True
        except Exception:
            return False

    @classmethod
    def try_commit_call(
        cls,
        tool_spec: ToolSpec,
        raw_call: ToolCall,
        raw_fragment: str = "",
    ) -> Optional[CommittedCall]:
        """Proves a tool call has crossed the commit horizon:
        1. Tool identity is fixed.
        2. All required semantics-changing arguments are syntactically closed.
        3. Tool is read-only and idempotent.
        """
        # Strict safety check: Mutative side-effects CANNOT be early dispatched!
        if not tool_spec.is_read_only or tool_spec.side_effects:
            return None

        # Check required commit arguments are present
        commit_args = tool_spec.get_commit_args()
        for req_arg in commit_args:
            if req_arg not in raw_call.arguments:
                return None

        # If raw JSON fragment is provided, ensure syntax closure
        if raw_fragment and not cls.is_syntax_closed(raw_fragment):
            return None

        # Generate committed immutable call
        schema_hash = hashlib.sha256(json.dumps(tool_spec.parameters, sort_keys=True).encode("utf-8")).hexdigest()
        return CommittedCall.from_call(raw_call, schema_hash=schema_hash)


class CommitHorizonScheduler(BaseScheduler):
    """Experiment E4: Commit-Horizon Streaming Early Dispatch Scheduler.
    
    Validates streaming argument immutability, early-dispatches eligible read-only tool calls,
    and reconciles final stream decisions against in-flight executions.
    """

    async def _execute_internal(
        self,
        ctx: ExecutionContext,
        model: BaseLLMAdapter,
        tools: ToolRegistry,
    ) -> Any:
        in_flight_tasks: Dict[str, Tuple[asyncio.Task[ToolResult], ToolCall, Dict[str, Any]]] = {}

        try:
            for turn in range(ctx.config.max_turns):
                ctx.step_count = turn + 1

                ctx.profiler.start_span(f"stream_model_turn_{turn}")
                stream_start = time.perf_counter()

                collected_chunks: List[StreamingChunk] = []
                final_tool_calls: List[ToolCall] = []
                final_reasoning: List[str] = []

                # 1. Stream tokens from model
                async for chunk in model.stream_decision(ctx.task, ctx.history, tools.list_specs()):
                    collected_chunks.append(chunk)
                    if chunk.delta_text:
                        final_reasoning.append(chunk.delta_text)

                    # Check commit horizon candidates
                    for candidate_call in chunk.commit_horizon_ready:
                        call_id = candidate_call.call_id
                        if call_id not in in_flight_tasks:
                            tool_name = candidate_call.name or candidate_call.tool_name
                            adapter = tools.get(tool_name)
                            if not adapter:
                                continue

                            # Validate immutability and eligibility
                            committed = IncrementalCommitParser.try_commit_call(
                                adapter.spec,
                                candidate_call,
                                chunk.raw_json_fragment,
                            )
                            if committed is not None:
                                candidate_call.committed_early = True
                                snapshot_args = copy.deepcopy(candidate_call.arguments)

                                ctx.profiler.record_event(
                                    EventType.COMMIT_HORIZON_REACHED,
                                    details={
                                        "tool": tool_name,
                                        "call_id": call_id,
                                        "token_index": chunk.token_index,
                                        "lead_time_ms": (time.perf_counter() - stream_start) * 1000.0,
                                        "fingerprint": committed.semantic_fingerprint,
                                    },
                                )

                                # Dispatch early via ToolExecutor
                                early_task = asyncio.create_task(
                                    ctx.executor.execute(candidate_call, is_early_dispatched=True)
                                )
                                in_flight_tasks[call_id] = (early_task, candidate_call, snapshot_args)

                    if chunk.is_final and chunk.parsed_tool_calls:
                        final_tool_calls = chunk.parsed_tool_calls

                ctx.profiler.end_span(
                    f"stream_model_turn_{turn}",
                    EventType.MODEL_END,
                    details={"turn": turn, "early_dispatches": len(in_flight_tasks)},
                )

                # Assemble complete decision
                last_meta = collected_chunks[-1].metadata if collected_chunks else {}
                final_ans = last_meta.get("final_answer")
                if final_ans is None and not (final_tool_calls or in_flight_tasks):
                    final_ans = "".join(final_reasoning).strip() or "Task completed."

                decision = LLMDecision(
                    reasoning="".join(final_reasoning),
                    tool_calls=final_tool_calls or [c for _, c, _ in in_flight_tasks.values()],
                    final_answer=final_ans if not (final_tool_calls or in_flight_tasks) else None,
                    output_tokens=len(collected_chunks),
                )
                ctx.record_model_decision(decision)

                if decision.is_final and (decision.final_answer is not None or not decision.tool_calls):
                    # Clean up all in-flight early tasks
                    for task, _, _ in in_flight_tasks.values():
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*[t for t, _, _ in in_flight_tasks.values()], return_exceptions=True)
                    in_flight_tasks.clear()
                    return decision.final_answer

                # 2. Reconcile final streamed calls against in-flight early tasks
                final_call_map = {c.call_id: c for c in decision.tool_calls}

                # Cancel and discard any omitted / retracted early calls
                for early_id, (task, early_call, _) in list(in_flight_tasks.items()):
                    if early_id not in final_call_map:
                        if not task.done():
                            task.cancel()
                        try:
                            await task
                        except Exception:
                            pass
                        in_flight_tasks.pop(early_id, None)

                # Execute / reconcile all authoritative final calls
                for call in decision.tool_calls:
                    ctx.tool_calls.append(call)

                    if call.call_id in in_flight_tasks:
                        task, early_call, snapshot_args = in_flight_tasks.pop(call.call_id)
                        early_tool_name = early_call.name or early_call.tool_name
                        final_tool_name = call.name or call.tool_name

                        # Check for semantic mutation (arguments mutated or tool renamed)
                        if call.arguments != snapshot_args or early_tool_name != final_tool_name:
                            if not task.done():
                                task.cancel()
                            try:
                                await task
                            except Exception:
                                pass

                            ctx.profiler.record_event(
                                EventType.GUARDRAIL_VIOLATION,
                                details={
                                    "error": "Semantic mismatch: arguments/tool mutated after commit horizon dispatch!",
                                    "dispatched": snapshot_args,
                                    "final": call.arguments,
                                },
                            )
                            # Re-execute with mutated arguments
                            res = await ctx.executor.execute(call)
                        else:
                            # Re-use early executed result
                            try:
                                res = await task
                            except Exception as ex:
                                res = await ctx.executor.execute(call)

                        ctx.record_tool_result(res)

                    else:
                        # Call was not early-dispatched: execute normally
                        res = await ctx.executor.execute(call)
                        ctx.record_tool_result(res)

            return "Max turns reached without final answer."

        finally:
            if in_flight_tasks:
                for task, _, _ in in_flight_tasks.values():
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*[t for t, _, _ in in_flight_tasks.values()], return_exceptions=True)
                in_flight_tasks.clear()
