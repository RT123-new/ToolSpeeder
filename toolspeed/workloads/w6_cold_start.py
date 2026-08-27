"""Workload W6: Cold-Start Sandbox and Container Initialization."""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

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


class W6ColdStartWorkload(BaseWorkload):
    """Workload Family 6: Cold-Start Sandboxes.
    
    Evaluates latency impact of cold-start container initialization vs predictive
    pre-warming and container pooling.
    """

    def __init__(
        self,
        cold_start_ms: float = 1200.0,
        warm_execution_ms: float = 80.0,
        sigma: float = 0.2,
    ):
        self.cold_start_ms = cold_start_ms
        self.warm_execution_ms = warm_execution_ms
        self.sigma = sigma

    def get_spec(self) -> WorkloadSpec:
        return WorkloadSpec(
            name="W6_Cold_Start_Sandbox",
            family="w6_cold_start",
            description="Container and sandbox code execution with cold-start initialization delays.",
            parameters={
                "cold_start_ms": self.cold_start_ms,
                "warm_execution_ms": self.warm_execution_ms,
            },
        )

    def _execute_code_handler(self, args: dict[str, Any]) -> dict[str, Any]:
        expr = str(args.get("expression", "2 + 2"))
        # Safe deterministic evaluation of basic arithmetic
        try:
            # Basic safe math evaluation
            allowed = {"__builtins__": None, "sum": sum, "max": max, "min": min, "len": len, "abs": abs}
            res = eval(expr, allowed, {})
            return {"status": "success", "result": res, "expression": expr}
        except Exception as ex:
            return {"status": "error", "error": str(ex), "expression": expr}

    def get_tools(self) -> list[BaseToolAdapter]:
        sandbox_tool = MockToolAdapter(
            MockToolConfig(
                name="sandbox_python_eval",
                description="Execute isolated Python math and logic expressions in a sandboxed runtime.",
                parameters={
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
                median_ms=self.warm_execution_ms,
                cold_start_ms=self.cold_start_ms,
                sigma=self.sigma,
                cost_usd=0.002,
                handler=self._execute_code_handler,
            )
        )
        return [sandbox_tool]

    def generate_tasks(self, count: int = 10, seed: Optional[int] = None) -> list[TaskInstance]:
        rng = np.random.default_rng(seed)
        tasks: list[TaskInstance] = []

        for idx in range(count):
            a = int(rng.integers(10, 500))
            b = int(rng.integers(10, 500))
            op = rng.choice(["+", "*", "-"])
            expr = f"{a} {op} {b}"
            expected_res = eval(expr, {"__builtins__": None})

            task = TaskInstance(
                task_id=f"w6_task_{idx:04d}",
                workload_family="w6_cold_start",
                prompt=f"Execute expression '{expr}' in isolated sandbox and return the evaluated result.",
                expected_tools=["sandbox_python_eval"],
                expected_output={"result": expected_res, "expression": expr},
                parameters={"expression": expr, "cold_start_sensitive": idx == 0},
                context={"expected_res": expected_res},
            )
            tasks.append(task)

        return tasks

    def get_validator(self) -> TaskValidator:
        def _validate(task: TaskInstance, output: Any, trace: Optional[ExecutionTrace]) -> Tuple[bool, str, dict[str, Any]]:
            if not isinstance(output, dict):
                return False, f"Output must be a dict, got {type(output).__name__}", {}

            expected_res = task.expected_output.get("result")
            actual_res = output.get("result")
            if actual_res != expected_res:
                return False, f"Expected evaluated result {expected_res}, got {actual_res}", {}

            return True, "Sandbox execution validation passed", {"result": actual_res}

        return FunctionValidator(_validate)
