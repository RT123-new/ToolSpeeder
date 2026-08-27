"""Async mock tool execution engine with lognormal latency, jitter, cold starts, and errors."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import inspect
import json
import numpy as np
import random
import time
from typing import Any, Callable, Dict, List, Optional, Union

from toolspeed.adapters.base import BaseToolAdapter, ToolSchema
from toolspeed.core.types import ToolCall, ToolResult, ToolSpec


@dataclass
class MockToolConfig:
    """Configuration for a mock tool's simulation characteristics."""
    name: str
    description: str = "Mock simulated tool"
    parameters: dict[str, Any] = field(default_factory=dict)
    median_ms: float = 600.0
    sigma: float = 0.45
    jitter_ms: float = 5.0
    cold_start_ms: float = 0.0
    error_rate: float = 0.0
    cost_usd: float = 0.001
    cache_ttl_s: Optional[float] = None
    is_side_effect: bool = False
    requires_approval: bool = False
    handler: Optional[Callable[[dict[str, Any]], Any]] = None


class MockToolAdapter(BaseToolAdapter):
    """Async mock tool adapter simulating real-world execution profiles."""

    def __init__(
        self,
        config: Optional[MockToolConfig] = None,
        spec: Optional[ToolSpec] = None,
        handler: Optional[Callable[[dict[str, Any]], Any]] = None,
        seed: Optional[int] = None,
        base_latency_ms: Optional[float] = None,
        latency_jitter_ms: float = 0.0,
        error_rate: float = 0.0,
        cold_start_ms: float = 0.0,
    ):
        if config is None:
            tool_name = spec.name if spec else "mock_tool"
            tool_desc = spec.description if spec else "Mock simulated tool"
            params = spec.parameters if spec else {}
            is_side = (spec.side_effects or not spec.is_read_only) if spec else False
            req_appr = (not spec.is_read_only) if spec else False
            med_ms = base_latency_ms if base_latency_ms is not None else (spec.estimated_latency_ms if spec else 600.0)
            config = MockToolConfig(
                name=tool_name,
                description=tool_desc,
                parameters=params,
                median_ms=med_ms,
                jitter_ms=latency_jitter_ms,
                cold_start_ms=cold_start_ms,
                error_rate=error_rate,
                is_side_effect=is_side,
                requires_approval=req_appr,
                handler=handler,
            )
        super().__init__(spec or ToolSpec(
            name=config.name,
            description=config.description,
            parameters=config.parameters,
            is_read_only=not config.is_side_effect,
            side_effects=config.is_side_effect,
        ))
        self.config = config
        self._rng = np.random.default_rng(seed)
        self._is_warm: bool = config.cold_start_ms <= 0
        self._cache: dict[str, tuple[Any, float]] = {}
        self._in_flight_tasks: dict[str, asyncio.Task[Any]] = {}

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.config.name,
            description=self.config.description,
            parameters=self.config.parameters,
            is_side_effect=self.config.is_side_effect,
            requires_approval=self.config.requires_approval,
            cost_usd=self.config.cost_usd,
            cache_ttl_s=self.config.cache_ttl_s,
        )

    def warm_up(self) -> None:
        """Pre-warm the tool container/sandbox, eliminating cold start delay."""
        self._is_warm = True

    def prewarm(self) -> None:
        self.warm_up()

    def cool_down(self) -> None:
        """Reset tool state to cold."""
        if self.config.cold_start_ms > 0:
            self._is_warm = False

    def clear_cache(self) -> None:
        """Clear cached tool results."""
        self._cache.clear()

    def sample_latency_ms(self) -> float:
        """Sample latency from lognormal distribution with additive jitter and cold start."""
        base_ms = float(self._rng.lognormal(np.log(max(1.0, self.config.median_ms)), self.config.sigma))
        jitter = float(self._rng.uniform(0.0, max(0.0, self.config.jitter_ms)))
        cold_start = 0.0
        if not self._is_warm:
            cold_start = self.config.cold_start_ms
            self._is_warm = True
        return base_ms + jitter + cold_start

    async def execute(self, call: ToolCall) -> ToolResult:
        """Execute mock tool with simulated latency, errors, side effects, and caching."""
        start_ns = time.perf_counter_ns()
        start_wall = time.perf_counter()
        call_id = call.call_id
        tool_name = self.config.name

        # Side-effect approval gate check
        if self.config.requires_approval and not call.is_approved:
            duration_ns = time.perf_counter_ns() - start_ns
            return ToolResult(
                call_id=call_id,
                name=tool_name,
                tool_name=tool_name,
                result=None,
                output=None,
                error="Action rejected: tool requires explicit approval before execution.",
                is_error=True,
                cached=False,
                execution_time_ns=duration_ns,
                execution_time_ms=duration_ns / 1_000_000.0,
                cost_usd=0.0,
                started_at=start_wall,
                finished_at=time.perf_counter(),
                metadata={"approval_required": True, "approval_failed": True},
            )

        # Cache check for read-only tools
        cache_key = f"{tool_name}:{json.dumps(call.arguments, sort_keys=True)}"
        if self.config.cache_ttl_s is not None and not self.config.is_side_effect:
            cached_entry = self._cache.get(cache_key)
            if cached_entry is not None:
                cached_res, cached_time = cached_entry
                age_s = time.time() - cached_time
                is_stale = age_s > self.config.cache_ttl_s
                if not is_stale:
                    duration_ns = time.perf_counter_ns() - start_ns
                    return ToolResult(
                        call_id=call_id,
                        name=tool_name,
                        tool_name=tool_name,
                        result=cached_res,
                        output=cached_res,
                        error=None,
                        is_error=False,
                        cached=True,
                        cache_timestamp=cached_time,
                        execution_time_ns=duration_ns,
                        execution_time_ms=duration_ns / 1_000_000.0,
                        cost_usd=0.0,
                        started_at=start_wall,
                        finished_at=time.perf_counter(),
                        metadata={"cache_hit": True, "is_stale": False, "age_s": age_s},
                    )

        # Check injected error probability
        will_error = self.config.error_rate > 0 and (self._rng.random() < self.config.error_rate)

        # Simulate execution latency
        latency_ms = self.sample_latency_ms()
        sleep_s = max(0.0001, latency_ms / 1000.0)

        # Track task for cancellation support
        current_task = asyncio.current_task()
        if current_task:
            self._in_flight_tasks[call_id] = current_task

        try:
            await asyncio.sleep(sleep_s)
        except asyncio.CancelledError:
            self._in_flight_tasks.pop(call_id, None)
            duration_ns = time.perf_counter_ns() - start_ns
            return ToolResult(
                call_id=call_id,
                name=tool_name,
                tool_name=tool_name,
                result=None,
                output=None,
                error="Tool execution cancelled.",
                is_error=True,
                cached=False,
                cancelled=True,
                execution_time_ns=duration_ns,
                execution_time_ms=duration_ns / 1_000_000.0,
                cost_usd=0.0,
                started_at=start_wall,
                finished_at=time.perf_counter(),
                metadata={"cancelled": True},
            )
        finally:
            self._in_flight_tasks.pop(call_id, None)

        duration_ns = time.perf_counter_ns() - start_ns
        finish_wall = time.perf_counter()

        if will_error:
            return ToolResult(
                call_id=call_id,
                name=tool_name,
                tool_name=tool_name,
                result=None,
                output=None,
                error=f"Injected execution error in tool '{tool_name}'",
                is_error=True,
                cached=False,
                execution_time_ns=duration_ns,
                execution_time_ms=duration_ns / 1_000_000.0,
                cost_usd=self.config.cost_usd,
                started_at=start_wall,
                finished_at=finish_wall,
                metadata={"injected_error": True},
            )

        # Calculate result via handler or default
        if self.config.handler is not None:
            try:
                if inspect.iscoroutinefunction(self.config.handler):
                    res_value = await self.config.handler(call.arguments)
                else:
                    res_value = self.config.handler(call.arguments)
            except Exception as ex:
                return ToolResult(
                    call_id=call_id,
                    name=tool_name,
                    tool_name=tool_name,
                    result=None,
                    output=None,
                    error=str(ex),
                    is_error=True,
                    cached=False,
                    execution_time_ns=duration_ns,
                    execution_time_ms=duration_ns / 1_000_000.0,
                    cost_usd=self.config.cost_usd,
                    started_at=start_wall,
                    finished_at=finish_wall,
                )
        else:
            res_value = {"status": "success", "tool": tool_name, "echo_args": call.arguments}

        # Store in cache if configured
        if self.config.cache_ttl_s is not None and not self.config.is_side_effect:
            self._cache[cache_key] = (res_value, time.time())

        return ToolResult(
            call_id=call_id,
            name=tool_name,
            tool_name=tool_name,
            result=res_value,
            output=res_value,
            error=None,
            is_error=False,
            cached=False,
            execution_time_ns=duration_ns,
            execution_time_ms=duration_ns / 1_000_000.0,
            cost_usd=self.config.cost_usd,
            started_at=start_wall,
            finished_at=finish_wall,
            metadata={"simulated_latency_ms": latency_ms},
        )

    async def cancel(self, call_id: str) -> bool:
        """Cancel in-flight mock execution."""
        task = self._in_flight_tasks.get(call_id)
        if task and not task.done():
            task.cancel()
            return True
        return False


class MockToolEngine:
    """Registry and execution orchestrator for multiple mock tools."""

    def __init__(self, seed: Optional[int] = None):
        self._tools: dict[str, MockToolAdapter] = {}
        self._seed = seed

    def register_tool(self, tool_or_config: Union[MockToolAdapter, MockToolConfig]) -> MockToolAdapter:
        if isinstance(tool_or_config, MockToolConfig):
            adapter = MockToolAdapter(tool_or_config, seed=self._seed)
        else:
            adapter = tool_or_config
        self._tools[adapter.config.name] = adapter
        return adapter

    def get_tool(self, name: str) -> Optional[MockToolAdapter]:
        return self._tools.get(name)

    def list_schemas(self) -> list[ToolSchema]:
        return [tool.get_schema() for tool in self._tools.values()]

    async def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.tool_name)
        if tool is None:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                result=None,
                error=f"Tool '{call.tool_name}' not found in MockToolEngine.",
                is_error=True,
                execution_time_ns=0,
                cost_usd=0.0,
            )
        return await tool.execute(call)

    async def execute_parallel(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls in parallel using asyncio.gather."""
        tasks = [self.execute(c) for c in calls]
        return list(await asyncio.gather(*tasks))

    def warm_up_all(self) -> None:
        for tool in self._tools.values():
            tool.warm_up()

    def cool_down_all(self) -> None:
        for tool in self._tools.values():
            tool.cool_down()

    def clear_all_caches(self) -> None:
        for tool in self._tools.values():
            tool.clear_cache()


def create_standard_mock_registry() -> Dict[str, MockToolAdapter]:
    """Helper to build a suite of standard mock tools for testing."""
    tools = [
        MockToolAdapter(
            spec=ToolSpec(
                name="database_query",
                description="Query relational database records",
                required_args=["query"],
                commit_horizon_args=["query"],
                is_read_only=True,
                estimated_latency_ms=10.0,
            ),
            handler=lambda args: {"rows": [{"id": 1, "value": f"result for {args.get('query')}"}]},
        ),
        MockToolAdapter(
            spec=ToolSpec(
                name="web_search",
                description="Search the web for query terms",
                required_args=["query"],
                commit_horizon_args=["query"],
                is_read_only=True,
                estimated_latency_ms=15.0,
            ),
            handler=lambda args: {"snippets": [f"Information about {args.get('query')}"]},
        ),
        MockToolAdapter(
            spec=ToolSpec(
                name="fetch_user",
                description="Fetch user profile by user_id",
                required_args=["user_id"],
                commit_horizon_args=["user_id"],
                is_read_only=True,
                estimated_latency_ms=10.0,
            ),
            handler=lambda args: {"user_id": args.get("user_id"), "name": f"User_{args.get('user_id')}", "tier": "gold"},
        ),
        MockToolAdapter(
            spec=ToolSpec(
                name="fetch_orders",
                description="Fetch orders for a user_id",
                required_args=["user_id"],
                commit_horizon_args=["user_id"],
                is_read_only=True,
                estimated_latency_ms=10.0,
            ),
            handler=lambda args: {"orders": [{"order_id": f"ord_{args.get('user_id')}_1", "total": 99.5}]},
        ),
        MockToolAdapter(
            spec=ToolSpec(
                name="execute_payment",
                description="Execute payment transaction (side-effecting)",
                required_args=["order_id", "amount"],
                commit_horizon_args=["order_id", "amount"],
                is_read_only=False,
                is_idempotent=False,
                side_effects=True,
                estimated_latency_ms=20.0,
            ),
            handler=lambda args: {"status": "success", "tx_id": "tx_1234"},
        ),
    ]
    return {t.name: t for t in tools}
