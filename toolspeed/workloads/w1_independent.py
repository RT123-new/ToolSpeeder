"""Workload W1: Independent Fan-out Reads (2, 4, 8, 16 parallel queries)."""

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


class W1IndependentWorkload(BaseWorkload):
    """Workload Family 1: Independent Fan-out Reads.
    
    Tests speedup from parallel tool dispatch across N independent read queries.
    """

    def __init__(
        self,
        fan_out_widths: Sequence[int] = (2, 4, 8, 16),
        median_tool_ms: float = 600.0,
        sigma: float = 0.45,
    ):
        self.fan_out_widths = list(fan_out_widths)
        self.median_tool_ms = median_tool_ms
        self.sigma = sigma
        self._server_data: dict[str, int] = {
            f"srv-{i:03d}": 100 + (i * 17) % 500 for i in range(256)
        }

    def get_spec(self) -> WorkloadSpec:
        return WorkloadSpec(
            name="W1_Independent_Fanout",
            family="w1_independent",
            description="Parallel independent read queries across N nodes/servers.",
            parameters={
                "fan_out_widths": self.fan_out_widths,
                "median_tool_ms": self.median_tool_ms,
                "sigma": self.sigma,
            },
        )

    def _query_metric_handler(self, args: dict[str, Any]) -> dict[str, Any]:
        server_id = args.get("server_id", "")
        val = self._server_data.get(server_id, 42)
        return {"server_id": server_id, "load_pct": val}

    def get_tools(self) -> list[BaseToolAdapter]:
        tool_config = MockToolConfig(
            name="query_server_load",
            description="Query CPU and memory load percentage for a given server ID.",
            parameters={
                "type": "object",
                "properties": {
                    "server_id": {"type": "string", "description": "Server identifier"},
                },
                "required": ["server_id"],
            },
            median_ms=self.median_tool_ms,
            sigma=self.sigma,
            handler=self._query_metric_handler,
        )
        return [MockToolAdapter(tool_config)]

    def generate_tasks(self, count: int = 10, seed: Optional[int] = None) -> list[TaskInstance]:
        rng = np.random.default_rng(seed)
        tasks: list[TaskInstance] = []

        all_servers = list(self._server_data.keys())

        for idx in range(count):
            width = int(rng.choice(self.fan_out_widths))
            chosen_servers = [str(s) for s in rng.choice(all_servers, size=width, replace=False)]

            total_load = sum(self._server_data[s] for s in chosen_servers)
            expected_output = {
                "server_count": width,
                "total_load": total_load,
                "servers": chosen_servers,
            }

            task = TaskInstance(
                task_id=f"w1_task_{idx:04d}_w{width}",
                workload_family="w1_independent",
                prompt=f"Fetch server load for the following {width} servers: {', '.join(chosen_servers)} and calculate total load.",
                expected_tools=["query_server_load"],
                expected_output=expected_output,
                parameters={"fan_out_width": width, "servers": chosen_servers},
                context={"servers": chosen_servers, "server_data": {s: self._server_data[s] for s in chosen_servers}},
            )
            tasks.append(task)

        return tasks

    def get_validator(self) -> TaskValidator:
        def _validate(task: TaskInstance, output: Any, trace: Optional[ExecutionTrace]) -> Tuple[bool, str, dict[str, Any]]:
            if not isinstance(output, dict):
                return False, f"Output must be a dict, got {type(output).__name__}", {}

            expected = task.expected_output
            if output.get("total_load") != expected.get("total_load"):
                return False, f"Expected total_load {expected.get('total_load')}, got {output.get('total_load')}", {}

            if output.get("server_count") != expected.get("server_count"):
                return False, f"Expected server_count {expected.get('server_count')}, got {output.get('server_count')}", {}

            # Trace verification if trace is provided
            if trace is not None:
                queried_servers = {
                    c.arguments.get("server_id")
                    for c in trace.tool_calls
                    if c.tool_name == "query_server_load"
                }
                expected_servers = set(task.parameters.get("servers", []))
                if not expected_servers.issubset(queried_servers):
                    missing = expected_servers - queried_servers
                    return False, f"Missing required server queries: {missing}", {}

            return True, "Validation passed", {"total_load": output.get("total_load")}

        return FunctionValidator(_validate)
