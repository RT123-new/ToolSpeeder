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
from toolspeed.schedulers.base import BaseScheduler, ExecutionContext, cancel_and_await


def _reject_duplicate_keys_hook(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """Strict JSON object hook rejecting duplicate keys."""
    res: Dict[str, Any] = {}
    for k, v in pairs:
        if k in res:
            raise ValueError(f"Duplicate JSON key: '{k}'")
        res[k] = v
    return res


class IncrementalCommitParser:
    """Incremental streaming parser and schema-aware commit validator.
    
    Verifies that a tool call has crossed the commit horizon:
    1. Tool identity is fixed, read-only, and idempotent.
    2. All required semantics-changing arguments are syntactically closed.
    3. JSON fragments are valid, well-formed, reject duplicate keys, and match schemas.
    """

    @staticmethod
    def is_syntax_closed(json_fragment: str) -> bool:
        """Verifies whether a JSON fragment is syntactically closed and complete."""
        s = json_fragment.strip()
        if not s:
            return False
        # Fast reject unclosed string quotes, unclosed brackets, dangling escapes
        if s.endswith("\\"):
            return False
        try:
            json.loads(s, object_pairs_hook=_reject_duplicate_keys_hook)
            return True
        except Exception:
            return False

    @staticmethod
    def validate_schema_types(parameters: Dict[str, Any], arguments: Dict[str, Any]) -> bool:
        """Validates that provided arguments match parameter types declared in schema."""
        properties = parameters.get("properties", {})
        for k, v in arguments.items():
            if k in properties:
                expected_type = properties[k].get("type")
                if expected_type == "string" and not isinstance(v, str):
                    return False
                elif expected_type == "integer" and not (isinstance(v, int) and not isinstance(v, bool)):
                    return False
                elif expected_type == "number" and not (isinstance(v, (int, float)) and not isinstance(v, bool)):
                    return False
                elif expected_type == "boolean" and not isinstance(v, bool):
                    return False
                elif expected_type == "array" and not isinstance(v, list):
                    return False
                elif expected_type == "object" and not isinstance(v, dict):
                    return False
        return True

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
        3. Tool is read-only and idempotent (no unapproved mutative actions).
        """
        # Strict safety check: Mutative side-effects or approval-requiring tools CANNOT be early dispatched!
        if not tool_spec.is_read_only or tool_spec.side_effects or tool_spec.requires_approval or not tool_spec.is_idempotent:
            return None

        # Check required commit arguments are present
        commit_args = tool_spec.get_commit_args()
        for req_arg in commit_args:
            if req_arg not in raw_call.arguments:
                return None

        # If raw JSON fragment is provided, ensure syntax closure and duplicate key rejection
        if raw_fragment:
            if not cls.is_syntax_closed(raw_fragment):
                return None
            try:
                parsed = json.loads(raw_fragment, object_pairs_hook=_reject_duplicate_keys_hook)
                if not isinstance(parsed, dict):
                    return None
            except Exception:
                return None

        # Validate argument types against schema
        if not cls.validate_schema_types(tool_spec.parameters, raw_call.arguments):
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
        in_flight_tasks: Dict[str, Tuple[asyncio.Task[ToolResult], ToolCall, Dict[str, Any], CommittedCall]] = {}

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

                            # Validate immutability, closure, and eligibility
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
                                in_flight_tasks[call_id] = (early_task, candidate_call, snapshot_args, committed)

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
                    tool_calls=final_tool_calls or [c for _, c, _, _ in in_flight_tasks.values()],
                    final_answer=final_ans if not (final_tool_calls or in_flight_tasks) else None,
                    output_tokens=len(collected_chunks),
                )
                ctx.record_model_decision(decision)

                if decision.is_final and (decision.final_answer is not None or not decision.tool_calls):
                    # Clean up all in-flight early tasks cleanly
                    for task, _, _, _ in in_flight_tasks.values():
                        await cancel_and_await(task)
                    in_flight_tasks.clear()
                    return decision.final_answer

                # 2. Reconcile final streamed calls against in-flight early tasks
                final_call_map = {c.call_id: c for c in decision.tool_calls}

                # Cancel and discard any omitted / retracted early calls
                for early_id, (task, early_call, _, _) in list(in_flight_tasks.items()):
                    if early_id not in final_call_map:
                        # Retracted call: cancel child task without escaping to parent
                        await cancel_and_await(task)
                        in_flight_tasks.pop(early_id, None)

                # Execute / reconcile all authoritative final calls
                for call in decision.tool_calls:
                    ctx.tool_calls.append(call)

                    if call.call_id in in_flight_tasks:
                        task, early_call, snapshot_args, committed = in_flight_tasks.pop(call.call_id)
                        early_tool_name = early_call.name or early_call.tool_name
                        final_tool_name = call.name or call.tool_name

                        # Check for semantic mutation (arguments mutated or tool renamed)
                        if call.arguments != snapshot_args or early_tool_name != final_tool_name:
                            # Cancel early task cleanly using cross-version safe helper
                            await cancel_and_await(task)

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
                            # Re-use early executed result safely
                            try:
                                res = await task
                            except asyncio.CancelledError:
                                # Early task was cancelled -> re-execute
                                res = await ctx.executor.execute(call)
                            except Exception:
                                res = await ctx.executor.execute(call)

                        ctx.record_tool_result(res)

                    else:
                        # Call was not early-dispatched: execute normally
                        res = await ctx.executor.execute(call)
                        ctx.record_tool_result(res)

            return "Max turns reached without final answer."

        finally:
            if in_flight_tasks:
                for task, _, _, _ in in_flight_tasks.values():
                    await cancel_and_await(task)
                in_flight_tasks.clear()
