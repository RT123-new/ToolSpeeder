"""Composite Latency Optimizer: Unifies E1 + E2 + E3 + E4 + E5 + Caching + Prewarming."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import asyncio
import copy
import time

from toolspeed.adapters.base import BaseLLMAdapter, LLMDecision, StreamingChunk, ToolRegistry
from toolspeed.core.types import EventType, ToolCall, ToolResult, ToolSpec
from toolspeed.schedulers.base import BaseScheduler, ExecutionContext, SchedulerConfig, cancel_and_await
from toolspeed.schedulers.e1_dag_scheduler import DAGNode, ToolDAG
from toolspeed.schedulers.e2_jit_fusion import DeclarativeWorkflow, FusedKernel, JITFusionScheduler
from toolspeed.schedulers.e4_commit_horizon import IncrementalCommitParser
from toolspeed.schedulers.e5_action_bytecode import ActionBytecodeCodec
from toolspeed.schedulers.phase2_cache import ToolResultCache


class CompositeScheduler(BaseScheduler):
    """Unified Composite Optimizer combining all latency reduction mechanisms:
    
    1. Predictive Prewarming for cold-start tools / sandboxes
    2. Exact & Semantic Result Caching with Freshness Contracts
    3. JIT Workflow Fusion for recognized sub-plans with Deopt Fallback
    4. Confidence-Gated Speculative Reads during Model Reasoning
    5. Commit-Horizon Streaming Dispatch for Overlapped Generation
    6. Action ByteCode (ABC) for Compact Token Generation
    7. Dynamic DAG Scheduling for Out-of-Order Concurrent Execution
    """

    def __init__(
        self,
        config: Optional[SchedulerConfig] = None,
        shared_cache: Optional[ToolResultCache] = None,
    ) -> None:
        cfg = config or SchedulerConfig(
            cache_enabled=True,
            speculation_enabled=True,
            speculation_confidence_threshold=0.70,
            speculation_contention_mode="cancellable",
            commit_horizon_enabled=True,
            jit_fusion_enabled=True,
            action_bytecode_enabled=True,
        )
        super().__init__(cfg)
        self.cache = shared_cache or ToolResultCache(default_ttl_seconds=cfg.cache_ttl_seconds)
        self.jit_scheduler = JITFusionScheduler(cfg)
        self.bytecode_codec = ActionBytecodeCodec()

    def prewarm_tools(self, tools: ToolRegistry) -> None:
        """Prewarms all cold-start sandbox adapters."""
        for spec in tools.list_specs():
            adapter = tools.get(spec.name)
            if adapter and hasattr(adapter, "prewarm"):
                adapter.prewarm()

    async def _execute_internal(
        self,
        ctx: ExecutionContext,
        model: BaseLLMAdapter,
        tools: ToolRegistry,
    ) -> Any:
        # 1. Prewarming
        self.prewarm_tools(tools)

        # 2. Register tools with Bytecode Codec
        for spec in tools.list_specs():
            self.bytecode_codec.register_tool(spec.name, spec.required_args or list(spec.parameters.get("properties", {}).keys()))

        # 3. JIT Fusion Check
        if self.config.jit_fusion_enabled:
            wf = self.jit_scheduler._match_workflow(ctx)
            if wf is not None:
                ctx.profiler.record_event(EventType.JIT_FUSION_START, details={"workflow": wf.workflow_id})
                res = await self.jit_scheduler._execute_internal(ctx, model, tools)
                if ctx.task.validate(res):
                    return res

        spec_task: Optional[asyncio.Task[ToolResult]] = None
        draft_task: Optional[asyncio.Task[Optional[ToolCall]]] = None
        in_flight_commit: Dict[str, Tuple[asyncio.Task[ToolResult], ToolCall, Dict[str, Any]]] = {}
        active_dag_tasks: Set[asyncio.Task[ToolResult]] = set()

        try:
            for turn in range(ctx.config.max_turns):
                ctx.step_count = turn + 1

                # Speculative Prediction concurrently with Streaming Generation
                spec_call: Optional[ToolCall] = None
                if self.config.speculation_enabled:
                    draft_task = asyncio.create_task(
                        model.predict_draft(ctx.task, ctx.history, tools.list_specs())
                    )

                ctx.profiler.start_span(f"composite_turn_{turn}")
                in_flight_commit.clear()
                collected_chunks: List[StreamingChunk] = []
                final_calls: List[ToolCall] = []
                reasoning_parts: List[str] = []

                async for chunk in model.stream_decision(ctx.task, ctx.history, tools.list_specs()):
                    collected_chunks.append(chunk)
                    if chunk.delta_text:
                        reasoning_parts.append(chunk.delta_text)

                    # Check if draft prediction is ready
                    if draft_task and draft_task.done() and not draft_task.cancelled() and spec_task is None:
                        try:
                            predicted = draft_task.result()
                            if (
                                predicted is not None
                                and predicted.speculation_confidence >= self.config.speculation_confidence_threshold
                            ):
                                spec_adapter = tools.get(predicted.name or predicted.tool_name)
                                if (
                                    spec_adapter
                                    and spec_adapter.spec.is_read_only
                                    and not spec_adapter.spec.side_effects
                                    and not spec_adapter.spec.requires_approval
                                    and spec_adapter.spec.is_idempotent
                                ):
                                    spec_call = predicted
                                    spec_call.is_speculative = True
                                    ctx.profiler.record_event(EventType.SPECULATION_START, details={"tool": predicted.name})
                                    spec_task = asyncio.create_task(
                                        ctx.executor.execute(predicted, is_speculative=True)
                                    )
                        except Exception:
                            spec_task = None
                            spec_call = None

                    # Early commit horizon dispatch
                    if self.config.commit_horizon_enabled:
                        for early_call in chunk.commit_horizon_ready:
                            if early_call.call_id not in in_flight_commit:
                                adapter = tools.get(early_call.name or early_call.tool_name)
                                if adapter:
                                    committed = IncrementalCommitParser.try_commit_call(adapter.spec, early_call, chunk.raw_json_fragment)
                                    if committed is not None:
                                        early_call.committed_early = True
                                        snapshot = copy.deepcopy(early_call.arguments)
                                        ctx.profiler.record_event(EventType.COMMIT_HORIZON_REACHED, details={"tool": early_call.name})
                                        t = asyncio.create_task(ctx.executor.execute(early_call, is_early_dispatched=True))
                                        in_flight_commit[early_call.call_id] = (t, early_call, snapshot)

                    if chunk.is_final and chunk.parsed_tool_calls:
                        final_calls = chunk.parsed_tool_calls

                ctx.profiler.end_span(f"composite_turn_{turn}", EventType.MODEL_END)

                if draft_task and not draft_task.done():
                    await cancel_and_await(draft_task)

                last_meta = collected_chunks[-1].metadata if collected_chunks else {}
                final_ans = last_meta.get("final_answer")
                if final_ans is None and not (final_calls or in_flight_commit):
                    final_ans = "".join(reasoning_parts).strip()

                decision = LLMDecision(
                    reasoning="".join(reasoning_parts),
                    tool_calls=final_calls or [c for _, c, _ in in_flight_commit.values()],
                    final_answer=final_ans if not (final_calls or in_flight_commit) else None,
                    output_tokens=len(collected_chunks),
                )
                ctx.record_model_decision(decision)

                if decision.is_final and (decision.final_answer is not None or not decision.tool_calls):
                    if spec_task and not spec_task.done():
                        await cancel_and_await(spec_task)
                        ctx.guardrails.record_speculation_resolved(hit=False, cancelled=True)
                    for t, _, _ in in_flight_commit.values():
                        await cancel_and_await(t)
                    return decision.final_answer or "Composite execution complete."

                # Action Bytecode tracking
                if self.config.action_bytecode_enabled:
                    for c in decision.tool_calls:
                        try:
                            encoded = self.bytecode_codec.encode(c)
                            ctx.profiler.record_event(EventType.BYTECODE_ENCODE, details={"bytes": len(encoded)})
                        except Exception:
                            pass

                # Resolve Speculation
                matching_spec_idx = -1
                if spec_call and spec_task:
                    for idx, c in enumerate(decision.tool_calls):
                        if (c.name or c.tool_name) == (spec_call.name or spec_call.tool_name) and c.arguments == spec_call.arguments:
                            matching_spec_idx = idx
                            break

                spec_hit = False
                spec_hit_res: Optional[ToolResult] = None
                if matching_spec_idx >= 0 and spec_task is not None:
                    try:
                        if spec_task.done():
                            spec_hit_res = spec_task.result()
                        else:
                            spec_hit_res = await spec_task
                        if spec_hit_res.is_success:
                            spec_hit = True
                            ctx.guardrails.record_speculation_resolved(hit=True)
                    except Exception:
                        spec_hit = False
                    finally:
                        spec_task = None
                elif spec_task is not None:
                    if not spec_task.done():
                        await cancel_and_await(spec_task)
                        ctx.guardrails.record_speculation_resolved(hit=False, cancelled=True)
                    else:
                        ctx.guardrails.record_speculation_resolved(hit=False, cancelled=False)
                    spec_task = None

                # Build DAG for turn execution
                dag = ToolDAG()
                dag.register_calls(decision.tool_calls)

                # Attach in-flight commit horizon tasks or speculative hit
                active_dag_tasks.clear()
                for c in decision.tool_calls:
                    ctx.tool_calls.append(c)
                    node = dag.nodes.get(c.call_id)
                    if not node:
                        continue

                    if c.call_id in in_flight_commit:
                        task, early_call, snapshot = in_flight_commit.pop(c.call_id)
                        if c.arguments != snapshot or (c.name or c.tool_name) != (early_call.name or early_call.tool_name):
                            await cancel_and_await(task)
                            ctx.profiler.record_event(
                                EventType.GUARDRAIL_VIOLATION,
                                details={"error": "Semantic mutation after commit horizon", "tool": c.name},
                            )
                        else:
                            node.status = "running"
                            async def _wrap_early(t=task, n=node):
                                try:
                                    res = await t
                                except Exception as ex:
                                    res = await ctx.executor.execute(n.call)
                                n.result = res
                                n.status = "completed" if res.is_success else "failed"
                                ctx.record_tool_result(res)
                                return res
                            active_dag_tasks.add(asyncio.create_task(_wrap_early()))

                # Dynamic DAG execution
                while not dag.is_complete():
                    ready_nodes = dag.get_ready_nodes()

                    # Record tool results for any newly failed nodes whose parent failed
                    for node in dag.nodes.values():
                        if node.status == "failed" and node.result is None:
                            res_f = ToolResult(call_id=node.node_id, name=node.call.name, error=node.error or "Parent failed", is_error=True)
                            node.result = res_f
                            ctx.record_tool_result(res_f)

                    for node in ready_nodes:
                        resolved, err = dag.resolve_node_arguments(node)
                        if err:
                            res = ToolResult(call_id=node.call.call_id, name=node.call.name, error=err, is_error=True)
                            node.result = res
                            node.status = "failed"
                            ctx.record_tool_result(res)
                            continue

                        node.call.arguments = resolved or {}
                        node.status = "running"

                        async def _exec_node(n=node):
                            res = await ctx.executor.execute(n.call)
                            n.result = res
                            n.status = "completed" if res.is_success else "failed"
                            ctx.record_tool_result(res)
                            return res

                        active_dag_tasks.add(asyncio.create_task(_exec_node()))

                    if not active_dag_tasks:
                        for n in dag.nodes.values():
                            if n.status == "pending":
                                n.status = "failed"
                        break

                    done, pending = await asyncio.wait(active_dag_tasks, return_when=asyncio.FIRST_COMPLETED)
                    active_dag_tasks = pending

            return "Max turns reached without final answer."

        finally:
            if spec_task and not spec_task.done():
                await cancel_and_await(spec_task)
            if draft_task and not draft_task.done():
                await cancel_and_await(draft_task)
            for t, _, _ in in_flight_commit.values():
                await cancel_and_await(t)
            for t in list(active_dag_tasks):
                await cancel_and_await(t)
            active_dag_tasks.clear()
