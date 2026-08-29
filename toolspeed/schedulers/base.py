"""BaseScheduler interface, ExecutionContext, and SchedulerConfig."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from toolspeed.adapters.base import BaseLLMAdapter, LLMDecision, ToolRegistry
from toolspeed.core.guardrails import GuardrailMonitor
from toolspeed.core.profiler import LatencyProfiler
from toolspeed.core.rate_limiter import RateLimiter
from toolspeed.core.types import (
    AgentTask,
    ApprovalGrant,
    ExecutionAuthorityContext,
    ExecutionTrace,
    StateSnapshot,
    Task,
    TaskResult,
    ToolCall,
    ToolResult,
)
from toolspeed.schedulers.executor import ToolExecutor


async def cancel_and_await(task: asyncio.Task[Any] | None) -> None:
    """Safely cancel an internally managed task and await its termination, consuming internal cancellations."""
    if task is None:
        return
    if not task.done():
        task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


class TaskTracker:
    """Tracks child asyncio tasks and ensures safe cleanup on normal exit, error, or cancellation."""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()

    def track(self, task: asyncio.Task[Any]) -> asyncio.Task[Any]:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def create_task(self, coro: Any, name: str | None = None) -> asyncio.Task[Any]:
        t = asyncio.create_task(coro, name=name) if name else asyncio.create_task(coro)
        return self.track(t)

    async def cancel_and_await(self, task: asyncio.Task[Any] | None) -> None:
        if task is None:
            return
        self._tasks.discard(task)
        await cancel_and_await(task)

    async def cancel_all(self) -> None:
        tasks = list(self._tasks)
        self._tasks.clear()
        for t in tasks:
            if not t.done():
                t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @property
    def active_count(self) -> int:
        return len(self._tasks)


@dataclass
class SchedulerConfig:
    """Universal configuration for all execution schedulers."""

    max_turns: int = 20
    concurrency_limit: int = 16
    rate_limit_rps: float | None = None
    timeout_seconds: float = 60.0
    parallelism_enabled: bool = True
    cache_enabled: bool = False
    cache_ttl_seconds: float = 300.0
    speculation_enabled: bool = False
    speculation_confidence_threshold: float = 0.70
    speculation_contention_mode: str = "cancellable"  # "isolated", "cancellable", "single_slot"
    commit_horizon_enabled: bool = False
    early_dispatch_enabled: bool = True
    jit_fusion_enabled: bool = False
    fusion_enabled: bool = True
    action_bytecode_enabled: bool = False
    prewarmed: bool = True
    atomic_idempotency_enabled: bool = True
    shared_rate_limiter: RateLimiter | None = None
    custom_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    """Execution state tracked across a task run."""

    task: Task | AgentTask
    config: SchedulerConfig
    tools: ToolRegistry
    clock: Any = None
    authority_context: ExecutionAuthorityContext | None = None
    initial_state: StateSnapshot | None = None
    agent_task: AgentTask = field(init=False)
    profiler: LatencyProfiler = field(init=False)
    guardrails: GuardrailMonitor = field(default_factory=GuardrailMonitor)
    rate_limiter: RateLimiter = field(default_factory=RateLimiter)
    executor: ToolExecutor = field(init=False)
    history: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    step_count: int = 0
    cancellation_requested: bool = False
    trusted_grants: dict[str, ApprovalGrant] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.task, Task):
            self.agent_task = self.task.to_agent_task()
            t_id = self.task.task_id
        else:
            self.agent_task = self.task
            t_id = self.task.task_id

        self.profiler = LatencyProfiler(task_id=t_id, clock=self.clock)

        if self.config.shared_rate_limiter is not None:
            self.rate_limiter = self.config.shared_rate_limiter
        else:
            self.rate_limiter = RateLimiter(
                max_concurrency=self.config.concurrency_limit,
                requests_per_second=self.config.rate_limit_rps,
                clock=self.clock,
            )

        auth_ctx = self.authority_context or ExecutionAuthorityContext()
        self.executor = ToolExecutor(
            registry=self.tools,
            rate_limiter=self.rate_limiter,
            profiler=self.profiler,
            guardrails=self.guardrails,
            default_timeout_s=self.config.timeout_seconds,
            trusted_grants=self.trusted_grants,
            authority_context=auth_ctx,
            clock=self.clock,
        )

    def record_model_decision(self, decision: LLMDecision) -> None:
        self.guardrails.record_model_usage(
            input_tokens=decision.input_tokens,
            output_tokens=decision.output_tokens,
        )
        self.history.append(
            {
                "role": "assistant",
                "reasoning": decision.reasoning,
                "tool_calls": [c.to_dict() for c in decision.tool_calls],
                "final_answer": decision.final_answer,
            }
        )

    def record_tool_result(self, result: ToolResult) -> None:
        self.tool_results.append(result)
        self.history.append(
            {
                "role": "tool",
                "name": result.name or result.tool_name,
                "call_id": result.call_id,
                "output": result.output if result.output is not None else result.result,
                "error": result.error,
            }
        )


class BaseScheduler(ABC):
    """Abstract base class for all execution schedulers."""

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.config = config or SchedulerConfig()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    async def execute(
        self,
        task: Task | AgentTask,
        model: BaseLLMAdapter,
        tools: ToolRegistry,
        authority_context: ExecutionAuthorityContext | None = None,
        initial_state: StateSnapshot | None = None,
    ) -> TaskResult:
        """Executes a task under this scheduler policy with full lifecycle instrumentation."""
        clock = getattr(model, "clock", None) or getattr(tools, "clock", None)
        if authority_context is None:
            if isinstance(task, Task) and "approval_grant" in task.metadata:
                grant = task.metadata["approval_grant"]
                if isinstance(grant, ApprovalGrant):
                    authority_context = ExecutionAuthorityContext(grants=[grant])
            elif hasattr(task, "authority_context") and getattr(task, "authority_context"):
                authority_context = getattr(task, "authority_context")

        ctx = ExecutionContext(
            task=task,
            config=self.config,
            tools=tools,
            clock=clock,
            authority_context=authority_context,
            initial_state=initial_state,
        )
        ctx.guardrails.record_task_start()
        ctx.profiler.start()

        final_answer: Any = None
        error: str | None = None
        success: bool = False

        try:
            if self.config.timeout_seconds > 0:
                final_answer = await asyncio.wait_for(
                    self._execute_internal(ctx, model, tools),
                    timeout=self.config.timeout_seconds,
                )
            else:
                final_answer = await self._execute_internal(ctx, model, tools)

            temp_trace = ExecutionTrace(
                task_id=task.task_id,
                success=True,
                final_output=final_answer,
                tool_calls=list(ctx.tool_calls),
                tool_results=list(ctx.tool_results),
                events=list(ctx.profiler.events),
            )

            if isinstance(task, Task):
                success = task.validate(final_answer, trace=temp_trace, initial_state=initial_state)
            else:
                success = True

        except asyncio.TimeoutError:
            error = f"Execution timed out after {self.config.timeout_seconds}s"
            success = False
        except asyncio.CancelledError:
            error = "Execution was cancelled"
            success = False
            raise
        except Exception as e:
            error = f"Scheduler execution error: {e!s}"
            success = False
        finally:
            total_ms = ctx.profiler.finish(ctx)
            if isinstance(task, Task):
                ctx.guardrails.record_task_finish(task, final_answer, success)

        ccl_ms: float | None = total_ms if success else None

        return TaskResult(
            task_id=task.task_id,
            success=success,
            final_answer=final_answer,
            ccl_ms=ccl_ms,
            total_duration_ms=total_ms,
            events=list(ctx.profiler.events),
            tool_calls=list(ctx.tool_calls),
            tool_results=list(ctx.tool_results),
            initial_state=initial_state.to_dict() if initial_state else {},
            final_state={},
            validator_result={"success": success},
            guardrails=ctx.guardrails.get_metrics(),
            error=error,
            metadata={"scheduler": self.name, **ctx.metadata},
        )

    async def run(
        self,
        task: Task | AgentTask,
        model: BaseLLMAdapter,
        tools: ToolRegistry,
    ) -> TaskResult:
        """Asynchronous execution wrapper."""
        return await self.execute(task, model, tools)

    def run_sync(
        self,
        task: Task | AgentTask,
        model: BaseLLMAdapter,
        tools: ToolRegistry,
    ) -> TaskResult:
        """Synchronous convenience wrapper."""
        return asyncio.run(self.execute(task, model, tools))

    @abstractmethod
    async def _execute_internal(
        self,
        ctx: ExecutionContext,
        model: BaseLLMAdapter,
        tools: ToolRegistry,
    ) -> Any:
        """Scheduler-specific execution strategy returning the final task answer."""
