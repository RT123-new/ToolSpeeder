"""BaseScheduler interface, ExecutionContext, and SchedulerConfig."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import asyncio
import time

from toolspeed.adapters.base import BaseLLMAdapter, LLMDecision, ToolRegistry
from toolspeed.core.guardrails import GuardrailMonitor
from toolspeed.core.profiler import LatencyProfiler
from toolspeed.core.rate_limiter import RateLimiter
from toolspeed.core.types import (
    EventType,
    ExecutionEvent,
    Task,
    TaskResult,
    ToolCall,
    ToolResult,
    sanitize_for_json,
)
from toolspeed.schedulers.executor import ToolExecutor


@dataclass
class SchedulerConfig:
    """Universal configuration for all execution schedulers."""

    max_turns: int = 20
    concurrency_limit: int = 16
    rate_limit_rps: Optional[float] = None
    timeout_seconds: float = 60.0
    cache_enabled: bool = False
    cache_ttl_seconds: float = 300.0
    speculation_enabled: bool = False
    speculation_confidence_threshold: float = 0.70
    speculation_contention_mode: str = "cancellable"  # "no_contention", "cancellable", "single_slot"
    commit_horizon_enabled: bool = False
    jit_fusion_enabled: bool = False
    action_bytecode_enabled: bool = False
    shared_rate_limiter: Optional[RateLimiter] = None
    custom_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    """Execution state tracked across a task run."""

    task: Task
    config: SchedulerConfig
    tools: ToolRegistry
    profiler: LatencyProfiler = field(default_factory=LatencyProfiler)
    guardrails: GuardrailMonitor = field(default_factory=GuardrailMonitor)
    rate_limiter: RateLimiter = field(default_factory=RateLimiter)
    executor: ToolExecutor = field(init=False)
    history: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    step_count: int = 0
    cancellation_requested: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.config.shared_rate_limiter is not None:
            self.rate_limiter = self.config.shared_rate_limiter
        else:
            self.rate_limiter = RateLimiter(
                max_concurrency=self.config.concurrency_limit,
                requests_per_second=self.config.rate_limit_rps,
            )
        self.executor = ToolExecutor(
            registry=self.tools,
            rate_limiter=self.rate_limiter,
            profiler=self.profiler,
            guardrails=self.guardrails,
            default_timeout_s=self.config.timeout_seconds,
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

    def __init__(self, config: Optional[SchedulerConfig] = None) -> None:
        self.config = config or SchedulerConfig()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    async def execute(
        self,
        task: Task,
        model: BaseLLMAdapter,
        tools: ToolRegistry,
    ) -> TaskResult:
        """Executes a task under this scheduler policy with full lifecycle instrumentation."""
        ctx = ExecutionContext(task=task, config=self.config, tools=tools)
        ctx.guardrails.record_task_start()
        ctx.profiler.start()

        final_answer: Any = None
        error: Optional[str] = None
        success: bool = False

        try:
            if self.config.timeout_seconds > 0:
                final_answer = await asyncio.wait_for(
                    self._execute_internal(ctx, model, tools),
                    timeout=self.config.timeout_seconds,
                )
            else:
                final_answer = await self._execute_internal(ctx, model, tools)

            # Strict correctness validation
            success = task.validate(final_answer)

        except asyncio.TimeoutError:
            error = f"Execution timed out after {self.config.timeout_seconds}s"
            success = False
        except asyncio.CancelledError:
            error = "Execution was cancelled"
            success = False
            raise
        except Exception as e:
            error = f"Scheduler execution error: {str(e)}"
            success = False
        finally:
            total_ms = ctx.profiler.finish()
            ctx.guardrails.record_task_finish(task, final_answer, success)

        # Correct Completion Latency (CCL): only valid when task passes validation
        ccl_ms: Optional[float] = total_ms if success else None

        return TaskResult(
            task_id=task.task_id,
            success=success,
            final_answer=final_answer,
            ccl_ms=ccl_ms,
            total_duration_ms=total_ms,
            events=list(ctx.profiler.events),
            tool_calls=list(ctx.tool_calls),
            tool_results=list(ctx.tool_results),
            guardrails=ctx.guardrails.get_metrics(),
            error=error,
            metadata={"scheduler": self.name, **ctx.metadata},
        )

    async def run(
        self,
        task: Task,
        model: BaseLLMAdapter,
        tools: ToolRegistry,
    ) -> TaskResult:
        """Asynchronous execution wrapper."""
        return await self.execute(task, model, tools)

    def run_sync(
        self,
        task: Task,
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
        pass
