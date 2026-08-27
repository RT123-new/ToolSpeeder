"""Workload W2: Deterministic Dependent Chains (2, 4, 8, 16 step pipelines)."""

from __future__ import annotations

import hashlib
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple

from toolspeed.adapters.base import BaseToolAdapter
from toolspeed.adapters.mock_tools import MockToolAdapter, MockToolConfig
from toolspeed.core.types import (
    ExecutionTrace,
    FunctionValidator,
    TaskInstance,
    TaskValidator,
    WorkloadSpec,
)
from toolspeed.workloads.base import BaseWorkload


def _compute_step_transform(step: int, input_data: str) -> str:
    """Deterministic transformation step function."""
    h = hashlib.sha256(f"step_{step}:{input_data}".encode("utf-8")).hexdigest()[:12]
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

    def generate_tasks(self, count: int = 10, seed: Optional[int] = None) -> list[TaskInstance]:
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
                prompt=f"Run a {depth}-step deterministic pipeline starting with input '{initial_seed}' through steps 0 to {depth-1}.",
                expected_tools=["execute_pipeline_step"],
                expected_output={"final_value": current_val, "depth": depth},
                parameters={"depth": depth, "initial_input": initial_seed},
                context={"step_outputs": step_outputs},
            )
            tasks.append(task)

        return tasks

    def get_validator(self) -> TaskValidator:
        def _validate(task: TaskInstance, output: Any, trace: Optional[ExecutionTrace]) -> Tuple[bool, str, dict[str, Any]]:
            if not isinstance(output, dict):
                return False, f"Output must be a dict, got {type(output).__name__}", {}

            expected_final = task.expected_output.get("final_value")
            actual_final = output.get("final_value")

            if actual_final != expected_final:
                return False, f"Expected final value '{expected_final}', got '{actual_final}'", {}

            # If trace is present, verify sequential steps were executed
            if trace is not None:
                expected_depth = task.parameters.get("depth", 0)
                step_calls = [c for c in trace.tool_calls if c.tool_name == "execute_pipeline_step"]
                if len(step_calls) < expected_depth:
                    return False, f"Expected {expected_depth} step executions, but recorded {len(step_calls)}", {}

            return True, "Chain validation passed", {"final_value": actual_final}

        return FunctionValidator(_validate)
