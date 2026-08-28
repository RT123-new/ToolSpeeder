"""Workload W5: Large Tool Arguments and Heavy Results (Decode and Serialization Stress)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

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


class W5LargePayloadsWorkload(BaseWorkload):
    """Workload Family 5: Large Tool Arguments & Heavy Results.

    Stresses JSON decode throughput, serialization overhead, and evaluates
    compact action bytecode acceleration.
    """

    def __init__(
        self,
        payload_sizes_kb: Sequence[int] = (10, 50, 200, 500),
        median_tool_ms: float = 400.0,
        sigma: float = 0.35,
    ):
        self.payload_sizes_kb = list(payload_sizes_kb)
        self.median_tool_ms = median_tool_ms
        self.sigma = sigma

    def get_spec(self) -> WorkloadSpec:
        return WorkloadSpec(
            name="W5_Large_Payloads",
            family="w5_large_payloads",
            description="Heavy tool arguments and large result payloads stressing decode bandwidth.",
            parameters={
                "payload_sizes_kb": self.payload_sizes_kb,
                "median_tool_ms": self.median_tool_ms,
            },
        )

    def _generate_dataset_handler(self, args: dict[str, Any]) -> dict[str, Any]:
        num_rows = int(args.get("num_rows", 100))
        seed = int(args.get("seed", 42))
        rng = np.random.default_rng(seed)
        rows = [
            {
                "row_id": f"r_{i:05d}",
                "val_a": float(rng.uniform(0.0, 100.0)),
                "val_b": int(rng.integers(100, 1000)),
                "label": f"category_{rng.integers(0, 10)}",
                "padding": "x" * 64,  # Padding to bulk up payload
            }
            for i in range(num_rows)
        ]
        return {"rows": rows, "count": len(rows), "checksum": sum(r["val_b"] for r in rows)}

    def _aggregate_dataset_handler(self, args: dict[str, Any]) -> dict[str, Any]:
        rows = args.get("rows", [])
        total_val_b = sum(r.get("val_b", 0) for r in rows)
        mean_val_a = float(np.mean([r.get("val_a", 0.0) for r in rows])) if rows else 0.0
        return {
            "processed_rows": len(rows),
            "total_val_b": total_val_b,
            "mean_val_a": round(mean_val_a, 4),
        }

    def get_tools(self) -> list[BaseToolAdapter]:
        gen_tool = MockToolAdapter(
            MockToolConfig(
                name="generate_heavy_dataset",
                description="Generate heavy tabular dataset with specified row count and seed.",
                parameters={
                    "type": "object",
                    "properties": {
                        "num_rows": {"type": "integer"},
                        "seed": {"type": "integer"},
                    },
                    "required": ["num_rows"],
                },
                median_ms=self.median_tool_ms,
                sigma=self.sigma,
                handler=self._generate_dataset_handler,
            )
        )
        agg_tool = MockToolAdapter(
            MockToolConfig(
                name="aggregate_heavy_dataset",
                description="Aggregate heavy tabular dataset rows.",
                parameters={
                    "type": "object",
                    "properties": {"rows": {"type": "array"}},
                    "required": ["rows"],
                },
                median_ms=self.median_tool_ms,
                sigma=self.sigma,
                handler=self._aggregate_dataset_handler,
            )
        )
        return [gen_tool, agg_tool]

    def generate_tasks(self, count: int = 10, seed: int | None = None) -> list[TaskInstance]:
        rng = np.random.default_rng(seed)
        tasks: list[TaskInstance] = []

        for idx in range(count):
            size_kb = int(rng.choice(self.payload_sizes_kb))
            num_rows = max(10, size_kb * 10)
            task_seed = int(rng.integers(1000, 99999))

            dataset = self._generate_dataset_handler({"num_rows": num_rows, "seed": task_seed})
            agg = self._aggregate_dataset_handler({"rows": dataset["rows"]})

            task = TaskInstance(
                task_id=f"w5_task_{idx:04d}_{size_kb}kb",
                workload_family="w5_large_payloads",
                prompt=f"Generate dataset of {num_rows} rows (seed {task_seed}) and compute aggregate statistics.",
                expected_tools=["generate_heavy_dataset", "aggregate_heavy_dataset"],
                expected_output={
                    "processed_rows": num_rows,
                    "total_val_b": agg["total_val_b"],
                    "mean_val_a": agg["mean_val_a"],
                },
                parameters={"num_rows": num_rows, "seed": task_seed, "target_size_kb": size_kb},
                context={"expected_checksum": dataset["checksum"]},
            )
            tasks.append(task)

        return tasks

    def get_validator(self) -> TaskValidator:
        def _validate(
            task: TaskInstance, output: Any, trace: ExecutionTrace | None
        ) -> tuple[bool, str, dict[str, Any]]:
            if not isinstance(output, dict):
                return False, f"Output must be a dict, got {type(output).__name__}", {}

            expected_total = task.expected_output.get("total_val_b")
            actual_total = output.get("total_val_b")
            if actual_total != expected_total:
                return False, f"Expected total_val_b {expected_total}, got {actual_total}", {}

            expected_rows = task.expected_output.get("processed_rows")
            actual_rows = output.get("processed_rows")
            if actual_rows != expected_rows:
                return False, f"Expected processed_rows {expected_rows}, got {actual_rows}", {}

            return True, "Large payload aggregation validation passed", {"total_val_b": actual_total}

        return FunctionValidator(_validate)
