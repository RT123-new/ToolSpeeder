"""Experiment E1: Dynamic DAG Parallelism Scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import asyncio
import re

from toolspeed.adapters.base import BaseLLMAdapter, ToolRegistry
from toolspeed.core.types import EventType, ToolCall, ToolResult
from toolspeed.schedulers.base import BaseScheduler, ExecutionContext


@dataclass
class DAGNode:
    node_id: str
    call: ToolCall
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    status: str = "pending"  # "pending", "ready", "running", "completed", "failed"
    result: Optional[ToolResult] = None


class ToolDAG:
    """Manages dynamic tool dependency graphs and data-flow parameter binding."""

    def __init__(self) -> None:
        self.nodes: Dict[str, DAGNode] = {}
        self.name_to_node_ids: Dict[str, List[str]] = {}

    def add_call(
        self,
        call: ToolCall,
        explicit_deps: Optional[Set[str]] = None,
    ) -> DAGNode:
        node_id = call.call_id
        deps = set(explicit_deps or [])

        # Infer dependencies from parameter references (e.g. $node_id.key or $tool_name.key)
        for arg_val in call.arguments.values():
            if isinstance(arg_val, str) and "$" in arg_val:
                refs = re.findall(r"\$([a-zA-Z0-9_\-]+)", arg_val)
                for ref in refs:
                    if ref in self.nodes:
                        deps.add(ref)
                    elif ref in self.name_to_node_ids and self.name_to_node_ids[ref]:
                        deps.add(self.name_to_node_ids[ref][-1])

        node = DAGNode(node_id=node_id, call=call, dependencies=deps)
        self.nodes[node_id] = node

        if call.name not in self.name_to_node_ids:
            self.name_to_node_ids[call.name] = []
        self.name_to_node_ids[call.name].append(node_id)

        for dep_id in deps:
            if dep_id in self.nodes:
                self.nodes[dep_id].dependents.add(node_id)

        return node

    def get_ready_nodes(self) -> List[DAGNode]:
        ready = []
        for node in self.nodes.values():
            if node.status == "pending":
                # Check if all dependencies are completed
                all_deps_done = True
                for dep_id in node.dependencies:
                    dep_node = self.nodes.get(dep_id)
                    if not dep_node or dep_node.status != "completed":
                        all_deps_done = False
                        break
                if all_deps_done:
                    node.status = "ready"
                    ready.append(node)
        return ready

    def resolve_arguments(self, node: DAGNode) -> Dict[str, Any]:
        """Binds intermediate results from parent nodes into arguments."""
        resolved: Dict[str, Any] = {}
        for k, v in node.call.arguments.items():
            if isinstance(v, str) and "$" in v:
                new_val = v
                refs = re.findall(r"\$([a-zA-Z0-9_\-]+)(?:\.([a-zA-Z0-9_]+))?", v)
                for ref_id, attr in refs:
                    target_node = self.nodes.get(ref_id)
                    if not target_node and ref_id in self.name_to_node_ids:
                        target_ids = self.name_to_node_ids[ref_id]
                        if target_ids:
                            target_node = self.nodes.get(target_ids[-1])

                    if target_node and target_node.result is not None:
                        output = target_node.result.output
                        val_to_insert = output
                        if attr:
                            if isinstance(output, dict):
                                val_to_insert = output.get(attr, output)
                            elif isinstance(output, (list, tuple)) and attr.isdigit():
                                idx = int(attr)
                                val_to_insert = output[idx] if 0 <= idx < len(output) else output

                        full_ref = f"${ref_id}.{attr}" if attr else f"${ref_id}"
                        if v.strip() == full_ref:
                            new_val = val_to_insert
                            break
                        else:
                            new_val = new_val.replace(full_ref, str(val_to_insert))
                resolved[k] = new_val
            else:
                resolved[k] = v
        return resolved

    def is_complete(self) -> bool:
        return all(node.status in ("completed", "failed") for node in self.nodes.values())


class DAGScheduler(BaseScheduler):
    """Experiment E1: Dynamic DAG Parallelism Scheduler.
    
    Analyzes argument data dependencies, constructs dynamic DAGs, and dispatches ready tool
    calls asynchronously with optimal concurrency and zero false-waiting.
    """

    async def _execute_dag_node(
        self,
        ctx: ExecutionContext,
        node: DAGNode,
        dag: ToolDAG,
        tools: ToolRegistry,
    ) -> ToolResult:
        node.status = "running"
        call = node.call
        resolved_args = dag.resolve_arguments(node)
        call.arguments = resolved_args

        ctx.tool_calls.append(call)
        adapter = tools.get(call.name)
        if not adapter:
            res = ToolResult(
                call_id=call.call_id,
                name=call.name,
                error=f"Tool '{call.name}' not found",
                is_error=True,
            )
            node.result = res
            node.status = "failed"
            ctx.record_tool_result(res)
            return res

        ctx.guardrails.record_tool_dispatch(adapter.spec, call, is_speculative=False)
        ctx.profiler.start_span(f"dag_tool_{call.call_id}")
        ctx.guardrails.record_concurrency_enter()

        try:
            await ctx.rate_limiter.acquire()
            try:
                res = await adapter.execute(call)
            finally:
                ctx.rate_limiter.release()
                ctx.guardrails.record_concurrency_exit()
        except asyncio.CancelledError:
            node.status = "failed"
            raise
        except Exception as e:
            res = ToolResult(
                call_id=call.call_id,
                name=call.name,
                error=str(e),
                is_error=True,
            )

        ctx.profiler.end_span(
            f"dag_tool_{call.call_id}",
            EventType.TOOL_END,
            details={"tool": call.name, "node_id": node.node_id},
        )
        node.result = res
        node.status = "completed" if res.is_success else "failed"
        ctx.record_tool_result(res)
        return res

    async def _execute_internal(
        self,
        ctx: ExecutionContext,
        model: BaseLLMAdapter,
        tools: ToolRegistry,
    ) -> Any:
        dag = ToolDAG()

        for turn in range(ctx.config.max_turns):
            ctx.step_count = turn + 1

            # 1. Model Decision Step
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

            if decision.final_answer is not None or not decision.tool_calls:
                return decision.final_answer

            # 2. Build / extend DAG with newly planned calls
            for call in decision.tool_calls:
                dag.add_call(call)

            # 3. Dynamic async queue execution: execute ready nodes as soon as dependencies clear
            active_tasks: Set[asyncio.Task] = set()

            try:
                while not dag.is_complete():
                    ready_nodes = dag.get_ready_nodes()
                    for node in ready_nodes:
                        ctx.profiler.record_event(
                            EventType.DAG_NODE_READY,
                            details={"node_id": node.node_id, "tool": node.call.name},
                        )
                        t = asyncio.create_task(
                            self._execute_dag_node(ctx, node, dag, tools)
                        )
                        active_tasks.add(t)

                    if not active_tasks:
                        # Mark unresolvable pending nodes as failed to prevent deadlock
                        for node in dag.nodes.values():
                            if node.status == "pending":
                                node.status = "failed"
                        break

                    done, pending = await asyncio.wait(
                        active_tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    active_tasks = pending
            finally:
                if active_tasks:
                    for t in active_tasks:
                        if not t.done():
                            t.cancel()
                    await asyncio.gather(*active_tasks, return_exceptions=True)

        return "Max turns reached without final answer."
