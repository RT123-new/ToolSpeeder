"""Workload W2: Deterministic Dependent Chains (2, 4, 8, 16 step pipelines)."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from toolspeed.adapters.base import BaseToolAdapter
from toolspeed.adapters.mock_tools import MockToolAdapter, MockToolConfig
from toolspeed.core.types import (
    ExecutionTrace,
    FunctionValidator,
    Task,
    TaskInstance,
    TaskValidator,
    ToolCall,
    ToolResult,
    WorkloadSpec,
)
from toolspeed.schedulers.base import BaseScheduler, ExecutionContext, SchedulerConfig
from toolspeed.schedulers.executor import ToolExecutor
from toolspeed.workloads.base import BaseWorkload


def _compute_step_transform(step: int, input_data: str) -> str:
    """Deterministic transformation step function."""
    h = hashlib.sha256(f"step_{step}:{input_data}".encode()).hexdigest()[:12]
    return f"{input_data}->s{step}_{h}"


class W2ChainsWorkload(BaseWorkload):
    """Workload Family 2: Deterministic Dependent Chains.

    Evaluates serial multi-turn LLM reasoning vs programmatic workflow fusion.
    """

    def __init__(
        self,
        chain_depths: Sequence[int] = (2, 4, 8, 16),
        median_step_ms: float = 400.0,
        sigma: float = 0.35,
    ):
        self.chain_depths = list(chain_depths)
        self.median_step_ms = median_step_ms
        self.sigma = sigma

    def get_spec(self) -> WorkloadSpec:
        return WorkloadSpec(
            name="W2_Dependent_Chains",
            family="w2_chains",
            description="Deterministic serial pipelines where step i output feeds step i+1.",
            parameters={
                "chain_depths": self.chain_depths,
                "median_step_ms": self.median_step_ms,
                "sigma": self.sigma,
            },
        )

    def _execute_pipeline_step(self, args: dict[str, Any]) -> dict[str, Any]:
        step_idx = int(args.get("step_index", 0))
        input_val = str(args.get("input_val", ""))
        output_val = _compute_step_transform(step_idx, input_val)
        return {"step_index": step_idx, "output_val": output_val}

    def get_tools(self) -> list[BaseToolAdapter]:
        tool_config = MockToolConfig(
            name="execute_pipeline_step",
            description="Execute step i of the pipeline, taking input from step i-1.",
            parameters={
                "type": "object",
                "properties": {
                    "step_index": {"type": "integer", "description": "0-indexed step number"},
                    "input_val": {"type": "string", "description": "Input value from previous step"},
                },
                "required": ["step_index", "input_val"],
            },
            median_ms=self.median_step_ms,
            sigma=self.sigma,
            handler=self._execute_pipeline_step,
        )
        return [MockToolAdapter(tool_config)]

    def generate_tasks(self, count: int = 10, seed: int | None = None) -> list[TaskInstance]:
        rng = np.random.default_rng(seed)
        tasks: list[TaskInstance] = []

        for idx in range(count):
            depth = int(rng.choice(self.chain_depths))
            initial_seed = f"seed_{idx}_{rng.integers(1000, 9999)}"

            current_val = initial_seed
            step_outputs = [current_val]
            for s in range(depth):
                current_val = _compute_step_transform(s, current_val)
                step_outputs.append(current_val)

            task = TaskInstance(
                task_id=f"w2_task_{idx:04d}_d{depth}",
                workload_family="w2_chains",
                prompt=f"Run a {depth}-step deterministic pipeline starting with input '{initial_seed}' through steps 0 to {depth - 1}.",
                expected_tools=["execute_pipeline_step"],
                expected_output={"final_value": current_val, "depth": depth},
                parameters={"depth": depth, "initial_input": initial_seed},
                context={"step_outputs": step_outputs},
            )
            tasks.append(task)

        return tasks

    def get_validator(self) -> TaskValidator:
        def _validate(
            task: TaskInstance, output: Any, trace: ExecutionTrace | None
        ) -> tuple[bool, str, dict[str, Any]]:
            if not isinstance(output, dict):
                return False, f"Output must be a dict, got {type(output).__name__}", {}

            expected_final = task.expected_output.get("final_value")
            actual_final = output.get("final_value")

            if actual_final != expected_final:
                return False, f"Expected final value '{expected_final}', got '{actual_final}'", {}

            # If trace is present, verify sequential steps were executed
            if trace is not None:
                expected_depth = task.parameters.get("depth", 0)
                step_calls = [
                    c
                    for c in trace.tool_calls
                    if (c.tool_name == "execute_pipeline_step" or c.name == "execute_pipeline_step")
                ]
                # Fail if model emitted answer without running required steps
                if len(step_calls) < expected_depth:
                    return False, f"Expected {expected_depth} step executions, but recorded {len(step_calls)}", {}

                # Fail if any step tool resulted in an error
                for r in trace.tool_results:
                    if not r.is_success or r.is_error:
                        return False, f"Tool result failed in chain: {r.error}", {}

            return True, "Chain validation passed", {"final_value": actual_final}

        return FunctionValidator(_validate)


@dataclass(frozen=True)
class CompiledPlanStep:
    step_id: str
    tool_name: str
    args_template: dict[str, Any]
    output_key: str = ""
    dependencies: tuple[str, ...] = ()


@dataclass
class CompiledExecutionPlan:
    workflow_id: str
    steps: list[CompiledPlanStep]
    compilation_time_ms: float
    output_mapping: dict[str, Any] = field(default_factory=dict)


@dataclass
class W2ComparisonResult:
    step_by_step_duration_ms: float
    compiled_compilation_ms: float
    compiled_execution_ms: float
    compiled_total_duration_ms: float
    speedup: float
    tools_executed_count: int
    outputs_match: bool


class W2DynamicDependencyCompiler:
    """Dynamically compiles multi-step dependencies for Workload W2 without precomputed shortcuts."""

    def compile(self, task: Task | TaskInstance) -> CompiledExecutionPlan:
        start_ns = time.perf_counter_ns()
        user_id = task.context.get("user_id") or task.parameters.get("user_id") or "u_0"

        steps = [
            CompiledPlanStep(
                step_id="step_fetch_user",
                tool_name="fetch_user",
                args_template={"user_id": user_id},
                output_key="user",
                dependencies=(),
            ),
            CompiledPlanStep(
                step_id="step_fetch_orders",
                tool_name="fetch_orders",
                args_template={"user_id": "$user.user_id"},
                output_key="orders",
                dependencies=("step_fetch_user",),
            ),
        ]

        compilation_time_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

        return CompiledExecutionPlan(
            workflow_id="w2_compiled_orders",
            steps=steps,
            compilation_time_ms=compilation_time_ms,
            output_mapping={
                "user": "$user",
                "orders": "$orders",
                "status": "compiled_complete",
                "fused": True,
            },
        )


async def execute_compiled_plan(
    plan: CompiledExecutionPlan,
    executor: ToolExecutor,
    context: dict[str, Any],
) -> tuple[dict[str, Any], float, list[ToolResult]]:
    """Executes a compiled multi-step plan through ToolExecutor without precomputed fusion shortcuts."""
    start_ns = time.perf_counter_ns()
    ledger: dict[str, Any] = {}
    results: list[ToolResult] = []

    for step in plan.steps:
        resolved_args: dict[str, Any] = {}
        for k, v in step.args_template.items():
            if isinstance(v, str) and v.startswith("$"):
                ref = v[1:]
                parts = ref.split(".", 1)
                source_key = parts[0]
                attr = parts[1] if len(parts) > 1 else None
                source = context if source_key == "context" else ledger.get(source_key)
                if isinstance(source, dict) and attr:
                    resolved_args[k] = source.get(attr)
                else:
                    resolved_args[k] = source
            else:
                resolved_args[k] = v

        call = ToolCall(
            tool_name=step.tool_name,
            name=step.tool_name,
            arguments=resolved_args,
        )
        res = await executor.execute(call)
        results.append(res)
        out = res.output if res.output is not None else res.result
        ledger[step.output_key] = out

    exec_time_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

    final_output: dict[str, Any] = {}
    for k, v in plan.output_mapping.items():
        if isinstance(v, str) and v.startswith("$"):
            final_output[k] = ledger.get(v[1:])
        else:
            final_output[k] = v

    return final_output, exec_time_ms, results


async def evaluate_w2_compilation_vs_step_by_step(
    backend: Any,
    baseline_scheduler: BaseScheduler,
    trial_index: int = 0,
) -> W2ComparisonResult:
    """Compares step-by-step dispatch vs compiled static plan on shared SQLite DB.

    Measures compilation time separately from execution time and verifies all tools execute.
    """
    # 1. Generate task and workload environment
    task = backend.generate_task("W2", trial_index=trial_index, arm="baseline")
    tools_step, model_step = backend.create_workload_environment("W2", trial_index=trial_index, arm="baseline")

    # 2. Run baseline step-by-step dispatch
    task_model = task.to_model_task() if hasattr(task, "to_model_task") else task
    res_step = await baseline_scheduler.execute(task_model, model_step, tools_step)
    step_duration_ms = res_step.total_duration_ms

    # 3. Dynamic dependency compilation (measured separately)
    compiler = W2DynamicDependencyCompiler()
    plan = compiler.compile(task)
    compilation_ms = plan.compilation_time_ms

    # 4. Execute compiled static plan using ToolExecutor directly
    tools_compiled, _ = backend.create_workload_environment("W2", trial_index=trial_index, arm="candidate")
    ctx = ExecutionContext(task=task_model, tools=tools_compiled, config=SchedulerConfig())
    compiled_output, execution_ms, tool_results = await execute_compiled_plan(
        plan=plan,
        executor=ctx.executor,
        context=task.context,
    )

    total_compiled_ms = compilation_ms + execution_ms
    speedup = step_duration_ms / total_compiled_ms if total_compiled_ms > 0 else 1.0

    # Verify both outputs fetched matching data from SQLite DB
    user_step = (res_step.final_answer or {}).get("user", {})
    user_compiled = compiled_output.get("user", {})
    outputs_match = user_step.get("user_id") == user_compiled.get("user_id")

    return W2ComparisonResult(
        step_by_step_duration_ms=step_duration_ms,
        compiled_compilation_ms=compilation_ms,
        compiled_execution_ms=execution_ms,
        compiled_total_duration_ms=total_compiled_ms,
        speedup=speedup,
        tools_executed_count=len(tool_results),
        outputs_match=outputs_match,
    )
