"""Experiment E4: Commit-Horizon Streaming Early Dispatch Scheduler."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from toolspeed.adapters.base import BaseLLMAdapter, LLMDecision, ToolRegistry
from toolspeed.core.types import CommittedCall, EventType, ToolCall, ToolResult, ToolSpec
from toolspeed.schedulers.base import BaseScheduler, ExecutionContext, SchedulerConfig, cancel_and_await


def _reject_duplicate_keys_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Strict JSON object hook rejecting duplicate keys."""
    res: dict[str, Any] = {}
    for k, v in pairs:
        if k in res:
            raise ValueError(f"Duplicate JSON key: '{k}'")
        res[k] = v
    return res


@dataclass
class ReconciledCall:
    call_id: str
    original_call_id: str
    tool_name: str
    arguments: dict[str, Any]


class IncrementalCommitParser:
    """Incremental streaming parser and schema-aware commit validator.

    Maintains stateful parsing buffer across streaming token deltas.
    Verifies that a tool call has crossed the commit horizon:
    1. Tool identity is fixed, read-only, and idempotent.
    2. All required semantics-changing arguments are syntactically closed.
    3. JSON fragments are non-empty, valid, well-formed, reject duplicate keys, and match schemas.
    """

    def __init__(self) -> None:
        self._buffer: str = ""
        self._token_index: int = 0
        self._byte_offset: int = 0
        self._commits: dict[str, dict[str, Any]] = {}
        self._semantic_index: dict[str, str] = {}

    def register_commit(self, call_id: str, tool_name: str, arguments: dict[str, Any]) -> None:
        self._commits[call_id] = {"tool_name": tool_name, "arguments": copy.deepcopy(arguments)}
        fp = f"{tool_name}:{json.dumps(dict(arguments), sort_keys=True)}"
        self._semantic_index[fp] = call_id

    def can_reuse(self, call_id: str, tool_name: str, arguments: dict[str, Any]) -> bool:
        if call_id not in self._commits:
            return False
        committed = self._commits[call_id]
        if committed["tool_name"] != tool_name:
            return False
        return committed["arguments"] == arguments

    def reconcile_call(self, call_id: str, tool_name: str, arguments: dict[str, Any]) -> ReconciledCall:
        fp = f"{tool_name}:{json.dumps(dict(arguments), sort_keys=True)}"
        orig_id = self._semantic_index.get(fp, call_id)
        return ReconciledCall(
            call_id=call_id,
            original_call_id=orig_id,
            tool_name=tool_name,
            arguments=dict(arguments),
        )

    def feed(self, delta_text: str) -> None:
        """Feed incremental character / token delta to parser buffer."""
        self._buffer += delta_text
        self._byte_offset += len(delta_text.encode("utf-8"))
        self._token_index += 1

    @property
    def buffer(self) -> str:
        return self._buffer

    @property
    def byte_offset(self) -> int:
        return self._byte_offset

    @property
    def token_index(self) -> int:
        return self._token_index

    def reset(self) -> None:
        self._buffer = ""
        self._token_index = 0
        self._byte_offset = 0

    @staticmethod
    def is_syntax_closed(json_fragment: str) -> bool:
        """Verifies whether a JSON fragment is syntactically closed and complete."""
        s = json_fragment.strip()
        if not s:
            return False
        if s.endswith("\\"):
            return False
        # Basic bracket balance quick check
        if s.startswith("{") and not s.endswith("}"):
            return False
        if s.startswith("[") and not s.endswith("]"):
            return False
        try:
            json.loads(s, object_pairs_hook=_reject_duplicate_keys_hook)
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    @staticmethod
    def validate_schema_types(parameters: dict[str, Any], arguments: dict[str, Any]) -> bool:
        """Validates that provided arguments match parameter types declared in schema."""
        properties = parameters.get("properties", {})
        for k, v in arguments.items():
            if k in properties:
                expected_type = properties[k].get("type")
                if expected_type == "string" and not isinstance(v, str):
                    return False
                if expected_type == "integer" and (not isinstance(v, int) or isinstance(v, bool)):
                    return False
                if expected_type == "number" and (not isinstance(v, (int, float)) or isinstance(v, bool)):
                    return False
                if expected_type == "boolean" and not isinstance(v, bool):
                    return False
                if expected_type == "array" and not isinstance(v, list):
                    return False
                if expected_type == "object" and not isinstance(v, dict):
                    return False
        return True

    @classmethod
    def try_commit_call(
        cls,
        tool_spec: ToolSpec,
        raw_call: ToolCall,
        raw_fragment: str = "",
        token_index: int = 0,
        byte_offset: int = 0,
    ) -> CommittedCall | None:
        """Proves a tool call has crossed the commit horizon:
        1. Tool identity is fixed.
        2. All required semantics-changing arguments are syntactically closed.
        3. Tool is read-only and idempotent (no unapproved mutative actions).
        4. Raw JSON fragment is present, non-empty, syntactically closed, and matches schema.
        """
        # Strict safety check: Mutative side-effects or approval-requiring tools CANNOT be early dispatched!
        if (
            not tool_spec.is_read_only
            or tool_spec.side_effects
            or tool_spec.requires_approval
            or not tool_spec.is_idempotent
        ):
            return None

        # Ensure raw_fragment is non-empty, syntactically closed, and matches raw_call.arguments
        if not raw_fragment or not raw_fragment.strip():
            return None
        if not cls.is_syntax_closed(raw_fragment):
            return None
        try:
            parsed = json.loads(raw_fragment, object_pairs_hook=_reject_duplicate_keys_hook)
            if not isinstance(parsed, dict):
                return None
            if parsed != raw_call.arguments:
                return None
        except (json.JSONDecodeError, ValueError):
            return None

        # Check required commit arguments are present and resolved
        commit_args = tool_spec.get_commit_args()
        for req_arg in commit_args:
            if req_arg not in raw_call.arguments:
                return None

        # Arguments with unresolved references ($parent.field) cannot be early dispatched
        for val in raw_call.arguments.values():
            if isinstance(val, str) and val.startswith("$"):
                return None

        # Validate argument types against schema
        if not cls.validate_schema_types(tool_spec.parameters, raw_call.arguments):
            return None

        # Generate committed immutable call
        schema_hash = hashlib.sha256(json.dumps(tool_spec.parameters, sort_keys=True).encode("utf-8")).hexdigest()
        return CommittedCall.from_call(
            raw_call, schema_hash=schema_hash, token_index=token_index, byte_offset=byte_offset
        )


class CommitHorizonScheduler(BaseScheduler):
    """Experiment E4: Commit-Horizon Streaming Early Dispatch Scheduler.

    Validates streaming argument immutability, early-dispatches eligible read-only tool calls,
    and reconciles final stream decisions against in-flight executions by semantic identity.
    """

    def __init__(self, config: SchedulerConfig | None = None, early_dispatch_enabled: bool | None = None) -> None:
        cfg = config or SchedulerConfig(commit_horizon_enabled=True, early_dispatch_enabled=True)
        if early_dispatch_enabled is not None:
            cfg.early_dispatch_enabled = early_dispatch_enabled
        elif config is None:
            cfg.early_dispatch_enabled = True
        super().__init__(cfg)

    async def _execute_internal(
        self,
        ctx: ExecutionContext,
        model: BaseLLMAdapter,
        tools: ToolRegistry,
    ) -> Any:
        in_flight_tasks: dict[str, tuple[asyncio.Task[ToolResult], ToolCall, dict[str, Any], CommittedCall]] = {}
        early_enabled = self.config.early_dispatch_enabled

        try:
            for turn in range(ctx.config.max_turns):
                ctx.step_count = turn + 1
                ctx.profiler.start_span(f"stream_model_turn_{turn}")

                final_decision: LLMDecision | None = None
                parser = IncrementalCommitParser()

                # 1. Stream tokens and evaluate commit horizons incrementally
                async for chunk in model.stream_decision(ctx.agent_task, ctx.history, tools.list_specs()):
                    parser.feed(chunk.delta_text)

                    # Check for early-dispatchable tool calls
                    if early_enabled and chunk.commit_horizon_ready:
                        for early_call in chunk.commit_horizon_ready:
                            call_name = early_call.name or early_call.tool_name
                            adapter = tools.get(call_name)
                            if adapter is None:
                                continue

                            frag = chunk.raw_json_fragment
                            if not frag and early_call.committed_early:
                                frag = json.dumps(early_call.arguments)

                            committed = IncrementalCommitParser.try_commit_call(
                                tool_spec=adapter.spec,
                                raw_call=early_call,
                                raw_fragment=frag,
                                token_index=chunk.token_index,
                                byte_offset=parser.byte_offset,
                            )

                            if committed is not None:
                                key = early_call.call_id or committed.semantic_fingerprint
                                if key not in in_flight_tasks:
                                    ctx.profiler.record_event(
                                        EventType.COMMIT_HORIZON_REACHED,
                                        details={
                                            "tool": call_name,
                                            "token_index": chunk.token_index,
                                            "byte_offset": parser.byte_offset,
                                            "fingerprint": committed.semantic_fingerprint,
                                        },
                                    )
                                    early_task = asyncio.create_task(ctx.executor.execute(early_call))
                                    in_flight_tasks[key] = (
                                        early_task,
                                        early_call,
                                        copy.deepcopy(early_call.arguments),
                                        committed,
                                    )

                    if chunk.is_final:
                        final_decision = LLMDecision(
                            reasoning=chunk.text or "",
                            tool_calls=list(chunk.parsed_tool_calls),
                            final_answer=chunk.metadata.get("final_answer"),
                            input_tokens=100,
                            output_tokens=max(1, chunk.token_index),
                        )

                ctx.profiler.end_span(
                    f"stream_model_turn_{turn}",
                    EventType.MODEL_END,
                    details={
                        "turn": turn,
                        "calls": len(final_decision.tool_calls) if final_decision else 0,
                    },
                )

                if final_decision is None:
                    final_decision = await model.decide(ctx.agent_task, ctx.history, tools.list_specs())

                ctx.record_model_decision(final_decision)

                if final_decision.final_answer is not None or not final_decision.tool_calls:
                    return final_decision.final_answer

                # 2. Reconcile Final Tool Calls with In-Flight Early Tasks
                for call in final_decision.tool_calls:
                    call_name = call.name or call.tool_name
                    ctx.tool_calls.append(call)

                    matched_key = None
                    for k, (_t, orig_c, _orig_args, c_obj) in in_flight_tasks.items():
                        # Strict reconciliation by call_id or semantic fingerprint (never tool name alone!)
                        if (call.call_id and orig_c.call_id == call.call_id) or (k == c_obj.semantic_fingerprint):
                            matched_key = k
                            break

                    if matched_key is not None:
                        early_task, orig_c, orig_args, _c_obj = in_flight_tasks.pop(matched_key)
                        if orig_args == call.arguments:
                            try:
                                res = await early_task
                                res.call_id = call.call_id
                            except (asyncio.CancelledError, Exception):
                                res = await ctx.executor.execute(call)
                            ctx.record_tool_result(res)
                        else:
                            # Post-dispatch argument mutation detected!
                            ctx.guardrails.record_semantic_mutation(
                                original_call=orig_args,
                                mutated_call=call.arguments,
                            )
                            ctx.profiler.record_event(
                                EventType.GUARDRAIL_VIOLATION,
                                details={"tool": call_name, "reason": "Post-dispatch argument mutation detected"},
                            )
                            await cancel_and_await(early_task)
                            res = await ctx.executor.execute(call)
                            ctx.record_tool_result(res)
                    else:
                        # Call was not early-dispatched
                        res = await ctx.executor.execute(call)
                        ctx.record_tool_result(res)

                # Clean up any unreferenced early-dispatched tasks from this turn
                for _fp, (t, _c, _a, _co) in list(in_flight_tasks.items()):
                    await cancel_and_await(t)
                in_flight_tasks.clear()

            return "Max turns reached without final answer."

        finally:
            for _fp, (t, _c, _a, _co) in list(in_flight_tasks.items()):
                await cancel_and_await(t)
            in_flight_tasks.clear()
