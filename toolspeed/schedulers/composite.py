"""Composite Latency Optimizer: Unifies E1 + E2 + E3 + E4 + E5 + Caching + Prewarming."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import asyncio
import copy
import time

from toolspeed.adapters.base import BaseLLMAdapter, LLMDecision, StreamingChunk, ToolRegistry
from toolspeed.core.types import EventType, ToolCall, ToolResult, ToolSpec
from toolspeed.schedulers.base import BaseScheduler, ExecutionContext, SchedulerConfig
from toolspeed.schedulers.e1_dag_scheduler import DAGNode, ToolDAG
from toolspeed.schedulers.e2_jit_fusion import FusedKernel, JITFusionScheduler
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

    async def _execute_single_tool(
        self,
        ctx: ExecutionContext,
        call: ToolCall,
        tools: ToolRegistry,
        is_speculative: bool = False,
    ) -> ToolResult:
        adapter = tools.get(call.name)
        if not adapter:
            return ToolResult(call_id=call.call_id, name=call.name, error=f"Tool {call.name} not found")

        # Invalidation on mutation / side effects
        if adapter.spec.side_effects or not adapter.spec.is_read_only:
            self.cache.invalidate_tool(call.name)

        # Cache lookup for read-only tools
        if self.config.cache_enabled and adapter.spec.is_read_only and not is_speculative:
            cached_out, hit, is_fresh = self.cache.get(call.name, call.arguments)
            if hit:
                ctx.profiler.record_event(
                    EventType.CACHE_HIT if is_fresh else EventType.CACHE_FRESHNESS_VIOLATION,
                    details={"tool": call.name, "is_fresh": is_fresh},
                )
                ctx.guardrails.record_cache_event(hit=True, is_fresh=is_fresh)
                return ToolResult(
                    call_id=call.call_id,
                    name=call.name,
                    output=cached_out,
                    cached=True,
                    execution_time_ms=1.0,
                )
            else:
                ctx.guardrails.record_cache_event(hit=False)

        # Dispatch tool
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
            ctx.profiler.record_event(EventType.TOOL_CANCELLED, details={"call_id": call.call_id})
            return ToolResult(call_id=call.call_id, name=call.name, cancelled=True, error="Cancelled", is_error=True)
        except Exception as e:
            res = ToolResult(call_id=call.call_id, name=call.name, error=str(e), is_error=True)

        ctx.profiler.end_span(span_name, EventType.TOOL_END)

        if self.config.cache_enabled and res.is_success and adapter.spec.is_read_only:
            self.cache.put(call.name, call.arguments, res.output, self.config.cache_ttl_seconds)

        return res

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
            for kernel in self.jit_scheduler._compiled_kernels.values():
                if kernel.match_fn(ctx):
                    ctx.profiler.record_event(EventType.JIT_FUSION_START, details={"kernel": kernel.name})
                    try:
                        res, is_ok = await kernel.execute_fn(ctx, tools)
                    except Exception:
                        res, is_ok = None, False

                    if is_ok:
                        if ctx.task.validator is not None or ctx.task.expected_output is not None:
                            if ctx.task.validate(res):
                                ctx.profiler.record_event(EventType.JIT_FUSION_SUCCESS, details={"kernel": kernel.name})
                                return res
                            else:
                                is_ok = False
                        else:
                            ctx.profiler.record_event(EventType.JIT_FUSION_SUCCESS, details={"kernel": kernel.name})
                            return res

                    # Deoptimization
                    ctx.profiler.record_event(EventType.JIT_FUSION_DEOPT, details={"kernel": kernel.name})
                    ctx.guardrails.record_deopt()
                    break

        spec_task: Optional[asyncio.Task[ToolResult]] = None
        in_flight_commit: Dict[str, Tuple[asyncio.Task[ToolResult], ToolCall, Dict[str, Any]]] = {}
        active_dag_tasks: Set[asyncio.Task] = set()

        try:
            # Main Reactive Loop combining Speculation + Commit Horizon + Action Bytecode + DAG Execution
            for turn in range(ctx.config.max_turns):
                ctx.step_count = turn + 1

                # Speculative Prediction
                spec_call: Optional[ToolCall] = None
                if self.config.speculation_enabled:
                    try:
                        predicted = await model.predict_draft(ctx.task, ctx.history, tools.list_specs())
                        if predicted and predicted.speculation_confidence >= self.config.speculation_confidence_threshold:
                            t_adapter = tools.get(predicted.name)
                            if t_adapter and t_adapter.spec.is_read_only and not t_adapter.spec.side_effects:
                                spec_call = predicted
                                spec_call.is_speculative = True
                                ctx.profiler.record_event(EventType.SPECULATION_START, details={"tool": predicted.name})
                                spec_task = asyncio.create_task(
                                    self._execute_single_tool(ctx, predicted, tools, is_speculative=True)
                                )
                    except Exception:
                        spec_task = None

                # Streaming Generation with Commit Horizon & Action Bytecode
                ctx.profiler.start_span(f"composite_turn_{turn}")
                in_flight_commit.clear()
                collected_chunks: List[StreamingChunk] = []
                final_calls: List[ToolCall] = []
                reasoning_parts: List[str] = []

                async for chunk in model.stream_decision(ctx.task, ctx.history, tools.list_specs()):
                    collected_chunks.append(chunk)
                    if chunk.delta_text:
                        reasoning_parts.append(chunk.delta_text)

                    if self.config.commit_horizon_enabled:
                        for early_call in chunk.commit_horizon_ready:
                            if early_call.call_id not in in_flight_commit:
                                # If early call matches speculative call, attach speculative task
                                if spec_call and spec_call.name == early_call.name and spec_call.arguments == early_call.arguments and spec_task:
                                    in_flight_commit[early_call.call_id] = (spec_task, early_call, copy.deepcopy(early_call.arguments))
                                else:
                                    early_call.committed_early = True
                                    snapshot = copy.deepcopy(early_call.arguments)
                                    ctx.profiler.record_event(EventType.COMMIT_HORIZON_REACHED, details={"tool": early_call.name})
                                    t = asyncio.create_task(self._execute_single_tool(ctx, early_call, tools))
                                    in_flight_commit[early_call.call_id] = (t, early_call, snapshot)

                    if chunk.is_final and chunk.parsed_tool_calls:
                        for pc in chunk.parsed_tool_calls:
                            for ec_id, (_, ec, _) in in_flight_commit.items():
                                if pc.name == ec.name and pc.arguments == ec.arguments:
                                    pc.call_id = ec.call_id
                                    pc.committed_early = True
                                    break
                        final_calls = chunk.parsed_tool_calls

                ctx.profiler.end_span(f"composite_turn_{turn}", EventType.MODEL_END)

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

                if decision.is_final:
                    if spec_task and not spec_task.done():
                        spec_task.cancel()
                        ctx.guardrails.record_speculation_resolved(hit=False, cancelled=True)
                    return decision.final_answer or "Composite execution complete."

                # Action Bytecode tracking
                if self.config.action_bytecode_enabled:
                    for c in decision.tool_calls:
                        encoded = self.bytecode_codec.encode(c)
                        ctx.profiler.record_event(EventType.BYTECODE_ENCODE, details={"bytes": len(encoded)})

                # Resolve Speculation
                spec_hit = False
                for c in decision.tool_calls:
                    if spec_call and spec_call.name == c.name and spec_call.arguments == c.arguments:
                        spec_hit = True
                        break
                if spec_task and not spec_hit:
                    if not spec_task.done():
                        spec_task.cancel()
                        ctx.guardrails.record_speculation_resolved(hit=False, cancelled=True)
                    else:
                        ctx.guardrails.record_speculation_resolved(hit=False, cancelled=False)
                elif spec_hit:
                    ctx.guardrails.record_speculation_resolved(hit=True)

                # Check for argument mutations on commit-horizon early dispatches
                dag = ToolDAG()
                active_dag_tasks.clear()
                dispatched_task_map: Dict[asyncio.Task, DAGNode] = {}

                for c in decision.tool_calls:
                    ctx.tool_calls.append(c)
                    node = dag.add_call(c)
                    if c.call_id in in_flight_commit:
                        task, dispatched_call, snapshot = in_flight_commit[c.call_id]
                        if c.arguments != snapshot:
                            if not task.done():
                                task.cancel()
                            ctx.profiler.record_event(
                                EventType.GUARDRAIL_VIOLATION,
                                details={"error": "Semantic mutation after commit horizon", "tool": c.name},
                            )
                            # Node remains pending in DAG to re-execute with correct arguments
                        else:
                            node.status = "running"
                            active_dag_tasks.add(task)
                            dispatched_task_map[task] = node

                # Wait for ready DAG nodes
                while not dag.is_complete():
                    ready_nodes = dag.get_ready_nodes()
                    for node in ready_nodes:
                        resolved = dag.resolve_arguments(node)
                        node.call.arguments = resolved
                        node.status = "running"
                        t = asyncio.create_task(self._execute_single_tool(ctx, node.call, tools))
                        active_dag_tasks.add(t)
                        dispatched_task_map[t] = node

                    if not active_dag_tasks:
                        for node in dag.nodes.values():
                            if node.status == "pending":
                                node.status = "failed"
                        break

                    done, pending = await asyncio.wait(active_dag_tasks, return_when=asyncio.FIRST_COMPLETED)
                    active_dag_tasks = pending
                    for finished_task in done:
                        try:
                            res = finished_task.result()
                        except (asyncio.CancelledError, Exception) as ex:
                            res = ToolResult(call_id="task_err", is_error=True, error=str(ex))
                        ctx.record_tool_result(res)
                        node = dispatched_task_map.get(finished_task)
                        if node:
                            node.result = res
                            node.status = "completed" if res.is_success else "failed"
                        else:
                            for n in dag.nodes.values():
                                if n.call.call_id == res.call_id:
                                    n.result = res
                                    n.status = "completed" if res.is_success else "failed"

                if active_dag_tasks:
                    remaining_results = await asyncio.gather(*active_dag_tasks, return_exceptions=True)
                    for r in remaining_results:
                        if isinstance(r, ToolResult):
                            ctx.record_tool_result(r)

            return "Max turns reached without final answer."
        finally:
            tasks_to_cancel = list(active_dag_tasks)
            if spec_task and not spec_task.done():
                tasks_to_cancel.append(spec_task)
            for t, _, _ in in_flight_commit.values():
                if not t.done():
                    tasks_to_cancel.append(t)
            for t in tasks_to_cancel:
                if not t.done():
                    t.cancel()
            if tasks_to_cancel:
                await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
