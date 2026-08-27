"""Experiment E1: Dynamic DAG Parallelism Scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
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
    error: Optional[str] = None


class ResolvedArguments(dict):
    """Dual-mode dictionary and 2-tuple (dict, error) container for DAG parameter resolution."""
    def __init__(self, data: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        super().__init__(data or {})
        self.error = error

    def __iter__(self):
        yield dict(self) if not self.error else None
        yield self.error


class ToolDAG:
    """Manages dynamic tool dependency graphs, cycle detection, and data-flow parameter binding."""

    def __init__(self) -> None:
        self.nodes: Dict[str, DAGNode] = {}
        self.name_to_node_ids: Dict[str, List[str]] = {}

    def _extract_references(self, value: Any) -> List[Tuple[str, Optional[str]]]:
        """Recursively discover all references in nested dicts, lists, tuples, and strings."""
        refs: List[Tuple[str, Optional[str]]] = []
        if isinstance(value, str):
            if "$" in value:
                matches = re.findall(r"\$([a-zA-Z0-9_\-]+)(?:\.([a-zA-Z0-9_]+))?", value)
                refs.extend(matches)
        elif isinstance(value, dict):
            for v in value.values():
                refs.extend(self._extract_references(v))
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                refs.extend(self._extract_references(item))
        return refs

    def add_call(self, call: ToolCall) -> DAGNode:
        """Register a single ToolCall and return its DAGNode."""
        self.register_calls([call])
        return self.nodes[call.call_id]

    def register_calls(self, calls: List[ToolCall]) -> None:
        """Two-pass graph construction:
        Pass 1: Register all nodes first.
        Pass 2: Resolve and validate dependencies.
        """
        # Pass 1: Register all nodes
        for call in calls:
            node_id = call.call_id
            if node_id in self.nodes:
                # Handle duplicate IDs gracefully with unique suffix
                node_id = f"{call.call_id}_{len(self.nodes)}"
                call.call_id = node_id

            node = DAGNode(node_id=node_id, call=call)
            self.nodes[node_id] = node

            call_name = call.name or call.tool_name
            self.name_to_node_ids.setdefault(call_name, []).append(node_id)

        # Pass 2: Resolve dependencies
        for node in self.nodes.values():
            refs = self._extract_references(node.call.arguments)
            for ref_id, _ in refs:
                # 1. Direct node_id match
                if ref_id in self.nodes:
                    if ref_id != node.node_id:
                        node.dependencies.add(ref_id)
                        self.nodes[ref_id].dependents.add(node.node_id)
                # 2. Tool name match
                elif ref_id in self.name_to_node_ids:
                    matching_ids = self.name_to_node_ids[ref_id]
                    if len(matching_ids) == 1:
                        target_id = matching_ids[0]
                        if target_id != node.node_id:
                            node.dependencies.add(target_id)
                            self.nodes[target_id].dependents.add(node.node_id)
                    elif len(matching_ids) > 1:
                        # Ambiguous reference
                        node.status = "failed"
                        node.error = f"Ambiguous reference '${ref_id}': multiple tool calls exist with name '{ref_id}'"
                else:
                    # Unknown reference -> fail closed
                    node.status = "failed"
                    node.error = f"Unknown dependency reference '${ref_id}' in tool call '{node.call.name}'"

    def detect_cycles(self) -> Optional[List[str]]:
        """Detect circular dependencies using DFS and return cycle path if found."""
        visited: Dict[str, int] = {}  # 0=unvisited, 1=visiting, 2=visited
        parent_map: Dict[str, str] = {}

        def _dfs(u: str) -> Optional[List[str]]:
            visited[u] = 1
            for v in self.nodes[u].dependencies:
                if visited.get(v, 0) == 1:
                    # Cycle found
                    cycle = [v, u]
                    curr = u
                    while curr in parent_map and parent_map[curr] != v:
                        curr = parent_map[curr]
                        cycle.append(curr)
                    cycle.append(v)
                    cycle.reverse()
                    return cycle
                elif visited.get(v, 0) == 0:
                    parent_map[v] = u
                    c = _dfs(v)
                    if c:
                        return c
            visited[u] = 2
            return None

        for n_id in list(self.nodes.keys()):
            if visited.get(n_id, 0) == 0:
                cycle_found = _dfs(n_id)
                if cycle_found:
                    return cycle_found
        return None

    def get_ready_nodes(self) -> List[DAGNode]:
        """Returns nodes whose dependencies have all completed successfully."""
        ready = []
        for node in self.nodes.values():
            if node.status == "pending":
                # Check for failed parent
                parent_failed = False
                for dep_id in node.dependencies:
                    dep_node = self.nodes.get(dep_id)
                    if dep_node and dep_node.status == "failed":
                        parent_failed = True
                        node.status = "failed"
                        node.error = f"Parent dependency '{dep_id}' failed"
                        break

                if parent_failed:
                    continue

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

    def resolve_arguments(self, node: DAGNode) -> ResolvedArguments:
        """Resolves template references in node arguments, supporting dict and tuple access."""
        resolved, err = self.resolve_node_arguments(node, fail_closed=True)
        if err:
            loose_resolved, _ = self.resolve_node_arguments(node, fail_closed=False)
            return ResolvedArguments(data=loose_resolved or {}, error=err)
        return ResolvedArguments(data=resolved or {}, error=None)

    def resolve_node_arguments(self, node: DAGNode, fail_closed: bool = True) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Binds intermediate results from parent nodes recursively.
        Returns (resolved_args, error_message).
        """
        def _resolve_val(val: Any) -> Tuple[Any, Optional[str]]:
            if isinstance(val, str) and "$" in val:
                refs = re.findall(r"\$([a-zA-Z0-9_\-]+)(?:\.([a-zA-Z0-9_]+))?", val)
                if not refs:
                    return val, None

                new_val = val
                for ref_id, attr in refs:
                    target_node = self.nodes.get(ref_id)
                    if not target_node and ref_id in self.name_to_node_ids:
                        target_ids = self.name_to_node_ids[ref_id]
                        if len(target_ids) == 1:
                            target_node = self.nodes.get(target_ids[0])
                        elif len(target_ids) > 1:
                            if fail_closed:
                                return None, f"Ambiguous reference '${ref_id}'"
                            continue

                    if not target_node or target_node.result is None:
                        if fail_closed:
                            return None, f"Unresolved dependency reference '${ref_id}'"
                        continue

                    output = target_node.result.output if target_node.result.output is not None else target_node.result.result
                    val_to_insert = output

                    if attr:
                        if isinstance(output, dict):
                            if attr not in output:
                                if fail_closed:
                                    return None, f"Missing output field '{attr}' in parent '{ref_id}'"
                                continue
                            val_to_insert = output[attr]
                        elif isinstance(output, (list, tuple)) and attr.isdigit():
                            idx = int(attr)
                            if 0 <= idx < len(output):
                                val_to_insert = output[idx]
                            else:
                                if fail_closed:
                                    return None, f"Index '{attr}' out of bounds for parent '{ref_id}' output"
                                continue
                        else:
                            if fail_closed:
                                return None, f"Cannot extract field '{attr}' from parent '{ref_id}' output of type {type(output).__name__}"
                            continue

                    full_ref = f"${ref_id}.{attr}" if attr else f"${ref_id}"
                    if val.strip() == full_ref:
                        return val_to_insert, None
                    else:
                        new_val = new_val.replace(full_ref, str(val_to_insert))
                return new_val, None

            elif isinstance(val, dict):
                res_dict = {}
                for k, v in val.items():
                    sub_v, err = _resolve_val(v)
                    if err:
                        return None, err
                    res_dict[k] = sub_v
                return res_dict, None

            elif isinstance(val, list):
                res_list = []
                for item in val:
                    sub_item, err = _resolve_val(item)
                    if err:
                        return None, err
                    res_list.append(sub_item)
                return res_list, None

            return val, None

        resolved_args = {}
        for k, v in node.call.arguments.items():
            res_v, err = _resolve_val(v)
            if err:
                return None, err
            resolved_args[k] = res_v

        return resolved_args, None

    def is_complete(self) -> bool:
        return all(node.status in ("completed", "failed") for node in self.nodes.values())


class DAGScheduler(BaseScheduler):
    """Experiment E1: Dynamic DAG Parallelism Scheduler.
    
    Constructs dependency DAGs with two-pass reference discovery, resolves parameters
    recursively, validates graph topology, and dispatches ready tool waves with optimal concurrency.
    """

    async def _execute_dag_node(
        self,
        ctx: ExecutionContext,
        node: DAGNode,
        dag: ToolDAG,
    ) -> ToolResult:
        node.status = "running"
        call = node.call

        resolved_args, err = dag.resolve_node_arguments(node, fail_closed=True)
        if err:
            res = ToolResult(
                call_id=call.call_id,
                name=call.name,
                tool_name=call.name,
                error=err,
                is_error=True,
            )
            node.result = res
            node.status = "failed"
            ctx.record_tool_result(res)
            return res

        call.arguments = resolved_args or {}
        ctx.tool_calls.append(call)

        res = await ctx.executor.execute(call)
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

            # 2. Build DAG with newly planned calls
            dag = ToolDAG()
            dag.register_calls(decision.tool_calls)

            # 3. Check for cycles
            cycle = dag.detect_cycles()
            if cycle:
                ctx.profiler.record_event(
                    EventType.GUARDRAIL_VIOLATION,
                    details={"error": f"Cyclic dependency detected: {' -> '.join(cycle)}", "cycle": cycle},
                )
                for node_id in cycle:
                    if node_id in dag.nodes:
                        dag.nodes[node_id].status = "failed"
                        res_err = ToolResult(
                            call_id=node_id,
                            name=dag.nodes[node_id].call.name,
                            tool_name=dag.nodes[node_id].call.name,
                            error=f"Cyclic dependency detected: {' -> '.join(cycle)}",
                            is_error=True,
                        )
                        dag.nodes[node_id].result = res_err
                        ctx.record_tool_result(res_err)

            # 4. Handle pre-failed nodes (e.g. unknown reference)
            for node in dag.nodes.values():
                if node.status == "failed" and node.result is None:
                    res_err = ToolResult(
                        call_id=node.node_id,
                        name=node.call.name,
                        tool_name=node.call.name,
                        error=node.error or "Invalid DAG dependency configuration",
                        is_error=True,
                    )
                    node.result = res_err
                    ctx.record_tool_result(res_err)

            # 5. Dynamic async wave execution: execute ready nodes as soon as dependencies clear
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
                            self._execute_dag_node(ctx, node, dag)
                        )
                        active_tasks.add(t)

                    if not active_tasks:
                        # Deadlock prevention: mark remaining pending nodes as failed with explicit dependency errors
                        for node in dag.nodes.values():
                            if node.status == "pending":
                                node.status = "failed"
                                res_deadlock = ToolResult(
                                    call_id=node.node_id,
                                    name=node.call.name,
                                    tool_name=node.call.name,
                                    error="Unresolvable dependency or deadlock",
                                    is_error=True,
                                )
                                node.result = res_deadlock
                                ctx.record_tool_result(res_deadlock)
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
