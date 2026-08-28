"""Phase 2 Candidate: Semantic and Exact Tool-Result Cache with Freshness Contracts."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from typing import Any

from toolspeed.adapters.base import BaseLLMAdapter, ToolRegistry
from toolspeed.core.types import EventType, ToolCall, ToolResult
from toolspeed.schedulers.base import BaseScheduler, ExecutionContext, SchedulerConfig


@dataclass
class CacheEntry:
    tool_name: str
    arguments: dict[str, Any]
    output: Any
    created_at: float = field(default_factory=time.perf_counter)
    ttl_seconds: float = 300.0
    freshness_contract: str = "strict"  # "strict", "relaxed"
    hit_count: int = 0

    def is_fresh(self, current_time: float | None = None) -> bool:
        now = current_time if current_time is not None else time.perf_counter()
        return (now - self.created_at) <= self.ttl_seconds


class ToolResultCache:
    """Multi-tiered Exact and Semantic tool result cache with automatic mutation invalidation."""

    def __init__(
        self,
        default_ttl_seconds: float = 300.0,
        max_entries: int = 1000,
        ttl_seconds: float | None = None,
        clock: Any = None,
    ) -> None:
        self.default_ttl_seconds = ttl_seconds if ttl_seconds is not None else default_ttl_seconds
        self.max_entries = max_entries
        self.clock = clock
        self._exact_store: dict[str, CacheEntry] = {}
        self._semantic_store: dict[str, CacheEntry] = {}

    def _now_s(self) -> float:
        if self.clock is not None and hasattr(self.clock, "now_s"):
            return self.clock.now_s()
        return time.perf_counter()

    def _exact_key(self, tool_name: str, arguments: dict[str, Any]) -> str:
        return f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"

    def _semantic_key(self, tool_name: str, arguments: dict[str, Any]) -> str:
        norm_args = {}
        for k, v in sorted(arguments.items()):
            if isinstance(v, str):
                norm_args[k] = " ".join(v.lower().strip().split())
            else:
                norm_args[k] = v
        return f"{tool_name}:{json.dumps(norm_args, sort_keys=True)}"

    def get(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        allow_semantic: bool = True,
    ) -> tuple[Any | None, bool, bool]:
        """Returns (cached_output, is_hit, is_fresh)."""
        now = self._now_s()
        exact_k = self._exact_key(tool_name, arguments)

        if exact_k in self._exact_store:
            entry = self._exact_store[exact_k]
            is_fresh = entry.is_fresh(now)
            if is_fresh or entry.freshness_contract == "relaxed":
                entry.hit_count += 1
                return copy.deepcopy(entry.output), True, is_fresh
            else:
                # Expired under strict contract -> treat as miss
                return None, False, False

        if allow_semantic:
            sem_k = self._semantic_key(tool_name, arguments)
            if sem_k in self._semantic_store:
                entry = self._semantic_store[sem_k]
                is_fresh = entry.is_fresh(now)
                if is_fresh or entry.freshness_contract == "relaxed":
                    entry.hit_count += 1
                    return copy.deepcopy(entry.output), True, is_fresh

        return None, False, False

    def put(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        output: Any,
        ttl_seconds: float | None = None,
        freshness_contract: str = "strict",
    ) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        entry = CacheEntry(
            tool_name=tool_name,
            arguments=copy.deepcopy(arguments),
            output=copy.deepcopy(output),
            created_at=self._now_s(),
            ttl_seconds=ttl,
            freshness_contract=freshness_contract,
        )
        self._exact_store[self._exact_key(tool_name, arguments)] = entry
        self._semantic_store[self._semantic_key(tool_name, arguments)] = entry

    def invalidate_tool(self, tool_name: str) -> int:
        """Invalidates all cache entries associated with a tool name."""
        to_del_exact = [k for k, v in self._exact_store.items() if v.tool_name == tool_name]
        for k in to_del_exact:
            del self._exact_store[k]

        to_del_sem = [k for k, v in self._semantic_store.items() if v.tool_name == tool_name]
        for k in to_del_sem:
            del self._semantic_store[k]

        return len(to_del_exact)

    def invalidate_on_mutation(self, mutation_tool: str, arguments: dict[str, Any] | None = None) -> int:
        """Invalidates related cache entries when a mutation tool executes."""
        domain = (
            mutation_tool.replace("update_", "")
            .replace("create_", "")
            .replace("delete_", "")
            .replace("set_", "")
            .replace("modify_", "")
        )
        to_del_exact = [
            k for k, v in self._exact_store.items() if domain in v.tool_name or v.tool_name in mutation_tool
        ]
        for k in to_del_exact:
            self._exact_store.pop(k, None)
        to_del_sem = [
            k for k, v in self._semantic_store.items() if domain in v.tool_name or v.tool_name in mutation_tool
        ]
        for k in to_del_sem:
            self._semantic_store.pop(k, None)
        return len(to_del_exact)

    def clear(self) -> None:
        self._exact_store.clear()
        self._semantic_store.clear()


class CacheScheduler(BaseScheduler):
    """Phase 2: Result Cache Scheduler.

    Caches tool outputs with exact & semantic matching, respects freshness contracts, and
    automatically invalidates cached reads when side-effecting write tools execute.
    """

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        shared_cache: ToolResultCache | None = None,
        cache_enabled: bool | None = None,
    ) -> None:
        cfg = config or SchedulerConfig(cache_enabled=True)
        if cache_enabled is not None:
            cfg.cache_enabled = cache_enabled
        elif config is None:
            cfg.cache_enabled = True
        super().__init__(cfg)
        self.cache = shared_cache or ToolResultCache(default_ttl_seconds=self.config.cache_ttl_seconds)

    async def _execute_tool_with_cache(
        self,
        ctx: ExecutionContext,
        call: ToolCall,
        tools: ToolRegistry,
    ) -> ToolResult:
        if not self.config.cache_enabled:
            return await ctx.executor.execute(call)

        adapter = tools.get(call.name)
        if not adapter:
            return await ctx.executor.execute(call)

        # 1. Invalidation on mutation / side effects
        if adapter.spec.side_effects or not adapter.spec.is_read_only:
            self.cache.invalidate_on_mutation(call.name, call.arguments)
            self.cache.invalidate_tool(call.name)

        # 2. Check Cache for Read-Only tools
        if adapter.spec.is_read_only and not adapter.spec.side_effects:
            lookup_start = time.perf_counter()
            cached_output, hit, is_fresh = self.cache.get(call.name, call.arguments)
            lookup_ms = (time.perf_counter() - lookup_start) * 1000.0

            if hit:
                ctx.profiler.record_event(
                    EventType.CACHE_HIT if is_fresh else EventType.CACHE_FRESHNESS_VIOLATION,
                    duration_ms=lookup_ms,
                    details={"tool": call.name, "is_fresh": is_fresh},
                )
                ctx.guardrails.record_cache_event(hit=True, is_fresh=is_fresh)
                return ToolResult(
                    call_id=call.call_id,
                    name=call.name,
                    tool_name=call.name,
                    result=cached_output,
                    output=cached_output,
                    cached=True,
                    execution_time_ms=lookup_ms,
                )
            else:
                ctx.guardrails.record_cache_event(hit=False)
                ctx.profiler.record_event(EventType.CACHE_MISS, duration_ms=lookup_ms)

        # 3. Cache Miss: Execute Tool via ToolExecutor
        res = await ctx.executor.execute(call)

        # 4. Populate Cache if successful and read-only
        if res.is_success and adapter.spec.is_read_only and not adapter.spec.side_effects:
            self.cache.put(
                tool_name=call.name,
                arguments=call.arguments,
                output=res.output if res.output is not None else res.result,
                ttl_seconds=ctx.config.cache_ttl_seconds,
            )

        return res

    async def _execute_internal(
        self,
        ctx: ExecutionContext,
        model: BaseLLMAdapter,
        tools: ToolRegistry,
    ) -> Any:
        for turn in range(ctx.config.max_turns):
            ctx.step_count = turn + 1

            ctx.profiler.start_span(f"model_turn_{turn}")
            decision = await model.decide(ctx.agent_task, ctx.history, tools.list_specs())
            ctx.profiler.end_span(f"model_turn_{turn}", EventType.MODEL_END)
            ctx.record_model_decision(decision)

            if decision.final_answer is not None or not decision.tool_calls:
                return decision.final_answer

            for call in decision.tool_calls:
                ctx.tool_calls.append(call)
                res = await self._execute_tool_with_cache(ctx, call, tools)
                ctx.record_tool_result(res)

        return "Max turns reached without final answer."
