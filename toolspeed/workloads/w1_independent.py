"""Workload W1: Independent Fan-out Reads (2, 4, 8, 16 parallel queries)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from toolspeed.adapters.base import BaseToolAdapter
from toolspeed.adapters.mock_tools import MockToolAdapter, MockToolConfig
from toolspeed.core.types import (
    EventType,
    ExecutionTrace,
    FunctionValidator,
    TaskInstance,
    TaskValidator,
    WorkloadSpec,
)
from toolspeed.schedulers.base import BaseScheduler, SchedulerConfig
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
        self._server_data: dict[str, int] = {f"srv-{i:03d}": 100 + (i * 17) % 500 for i in range(256)}

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

    def generate_tasks(self, count: int = 10, seed: int | None = None) -> list[TaskInstance]:
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
        def _validate(
            task: TaskInstance, output: Any, trace: ExecutionTrace | None
        ) -> tuple[bool, str, dict[str, Any]]:
            if not isinstance(output, dict):
                return False, f"Output must be a dict, got {type(output).__name__}", {}

            expected = task.expected_output
            if output.get("total_load") != expected.get("total_load"):
                return False, f"Expected total_load {expected.get('total_load')}, got {output.get('total_load')}", {}

            if output.get("server_count") != expected.get("server_count"):
                return (
                    False,
                    f"Expected server_count {expected.get('server_count')}, got {output.get('server_count')}",
                    {},
                )

            # Trace verification if trace is provided
            if trace is not None:
                queried_servers = {
                    c.arguments.get("server_id")
                    for c in trace.tool_calls
                    if (c.tool_name == "query_server_load" or c.name == "query_server_load")
                }
                expected_servers = set(task.parameters.get("servers", []))
                if not expected_servers.issubset(queried_servers):
                    missing = expected_servers - queried_servers
                    return False, f"Missing required server queries: {missing}", {}

            return True, "Validation passed", {"total_load": output.get("total_load")}

        return FunctionValidator(_validate)


@dataclass(frozen=True)
class W1ConcurrencyPressurePoint:
    concurrency_limit: int
    baseline_duration_ms: float
    candidate_duration_ms: float
    speedup: float
    candidate_queue_time_ms: float


@dataclass
class W1ConcurrencySweepReport:
    points: list[W1ConcurrencyPressurePoint]

    def verify_concurrency_pressure_invariants(self) -> tuple[bool, str]:
        """Proves speedup diminishes as limit approaches 1, and queue time decreases with higher limits."""
        if not self.points:
            return False, "No concurrency points evaluated"

        sorted_points = sorted(self.points, key=lambda p: p.concurrency_limit)
        limits = [p.concurrency_limit for p in sorted_points]
        speedups = [p.speedup for p in sorted_points]
        queue_times = [p.candidate_queue_time_ms for p in sorted_points]

        # 1. Speedup at limit=1 must be strictly lower than at highest limit
        if speedups[-1] <= speedups[0]:
            return (
                False,
                f"Speedup at limit={limits[-1]} ({speedups[-1]:.2f}x) does not exceed limit={limits[0]} ({speedups[0]:.2f}x)",
            )

        # 2. Speedup at limit=1 diminishes towards 1.0x (<= 1.8x)
        p1 = next((p for p in sorted_points if p.concurrency_limit == 1), None)
        if p1 is not None and p1.speedup > 1.8:
            return False, f"Speedup at limit=1 did not diminish ({p1.speedup:.2f}x; expected ~1.0x)"

        # 3. Queue time at limit=1 must be greater than at highest limit
        if queue_times[0] <= queue_times[-1]:
            return (
                False,
                f"Queue time at limit=1 ({queue_times[0]:.2f}ms) is not greater than at limit={limits[-1]} ({queue_times[-1]:.2f}ms)",
            )

        return True, "All W1 concurrency pressure invariants hold."


async def evaluate_w1_concurrency_pressure(
    backend: Any,
    baseline_cls: type[BaseScheduler],
    candidate_cls: type[BaseScheduler],
    limits: Sequence[int] = (1, 2, 4, 8, 16),
    trial_index: int = 0,
) -> W1ConcurrencySweepReport:
    """Evaluates W1 under concurrency-limit pressure across limits [1, 2, 4, 8, 16]."""
    points: list[W1ConcurrencyPressurePoint] = []

    for limit in limits:
        task_b = backend.generate_task("W1", trial_index=trial_index, arm="baseline")
        task_c = backend.generate_task("W1", trial_index=trial_index, arm="candidate")

        tools_b, model_b = backend.create_workload_environment("W1", trial_index=trial_index, arm="baseline")
        tools_c, model_c = backend.create_workload_environment("W1", trial_index=trial_index, arm="candidate")

        cfg_b = SchedulerConfig(concurrency_limit=limit)
        cfg_c = SchedulerConfig(concurrency_limit=limit)

        sched_b = baseline_cls(cfg_b)
        sched_c = candidate_cls(cfg_c)

        task_b_model = task_b.to_model_task() if hasattr(task_b, "to_model_task") else task_b
        task_c_model = task_c.to_model_task() if hasattr(task_c, "to_model_task") else task_c

        res_b = await sched_b.execute(task_b_model, model_b, tools_b)
        res_c = await sched_c.execute(task_c_model, model_c, tools_c)

        dur_b = res_b.total_duration_ms
        dur_c = res_c.total_duration_ms
        speedup = dur_b / dur_c if dur_c > 0 else 1.0

        q_time_c = sum(
            e.duration_ms for e in res_c.events if e.event_type in (EventType.RATE_LIMIT_DELAY, "rate_limit_delay")
        )
        if q_time_c == 0.0:
            q_time_c = sum(r.metadata.get("queue_delay_ms", 0.0) for r in res_c.tool_results)

        points.append(
            W1ConcurrencyPressurePoint(
                concurrency_limit=limit,
                baseline_duration_ms=dur_b,
                candidate_duration_ms=dur_c,
                speedup=speedup,
                candidate_queue_time_ms=q_time_c,
            )
        )

    return W1ConcurrencySweepReport(points=points)
