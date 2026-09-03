"""Workload W6: Cold-Start Sandbox and Container Initialization."""

from __future__ import annotations

import time
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
        try:
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

    def generate_tasks(self, count: int = 10, seed: int | None = None) -> list[TaskInstance]:
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
        def _validate(
            task: TaskInstance, output: Any, trace: ExecutionTrace | None
        ) -> tuple[bool, str, dict[str, Any]]:
            if not isinstance(output, dict):
                return False, f"Output must be a dict, got {type(output).__name__}", {}

            expected_res = task.expected_output.get("result")
            actual_res = output.get("result")
            if actual_res != expected_res:
                return False, f"Expected evaluated result {expected_res}, got {actual_res}", {}

            return True, "Sandbox execution validation passed", {"result": actual_res}

        return FunctionValidator(_validate)


class PoolSlot:
    """Represents an isolated sandbox / worker execution slot."""

    def __init__(self, slot_id: str, is_warm: bool = False, init_cost_ms: float = 35.0) -> None:
        self.slot_id = slot_id
        self.is_warm = is_warm
        self.init_cost_ms = init_cost_ms
        self.is_acquired = False
        self.acquired_at: float | None = None
        self.released_at: float | None = None


class BaseContainerPool:
    """Base class for container/sandbox pools with slot lifecycle tracking."""

    def __init__(
        self,
        capacity: int = 10,
        cold_start_delay_ms: float = 35.0,
        warm_start_delay_ms: float = 2.0,
        prewarmed: bool = False,
    ) -> None:
        import asyncio

        self._asyncio = asyncio
        self.capacity = capacity
        self.cold_start_delay_ms = cold_start_delay_ms
        self.warm_start_delay_ms = warm_start_delay_ms
        self.is_prewarmed_pool = prewarmed
        self._slots: list[PoolSlot] = [
            PoolSlot(f"slot_{i}", is_warm=prewarmed, init_cost_ms=cold_start_delay_ms) for i in range(capacity)
        ]
        self._lock: asyncio.Lock | None = None
        self.total_acquisitions = 0
        self.total_prewarm_cost_ms = (capacity * cold_start_delay_ms) if prewarmed else 0.0

    def _get_lock(self) -> Any:
        if self._lock is None:
            self._lock = self._asyncio.Lock()
        return self._lock

    async def acquire_slot(self) -> tuple[PoolSlot, float]:
        """Acquires a slot and returns (slot, latency_cost_ms)."""
        lock = self._get_lock()
        async with lock:
            for slot in self._slots:
                if not slot.is_acquired:
                    slot.is_acquired = True
                    slot.acquired_at = time.perf_counter()
                    self.total_acquisitions += 1
                    if slot.is_warm:
                        return slot, self.warm_start_delay_ms
                    else:
                        slot.is_warm = True
                        return slot, self.cold_start_delay_ms
            raise RuntimeError("Container pool exhausted: no free slots available")

    async def release_slot(self, slot: PoolSlot) -> None:
        """Releases an acquired slot back to the pool."""
        lock = self._get_lock()
        async with lock:
            slot.is_acquired = False
            slot.released_at = time.perf_counter()

    async def acquire_time_ms(self) -> float:
        """Acquires and immediately releases a slot to measure acquisition latency."""
        slot, latency = await self.acquire_slot()
        await self.release_slot(slot)
        return latency


class PersistentColdPool(BaseContainerPool):
    """Cold container pool: slots require cold-start initialization on first acquisition."""

    def __init__(self, capacity: int = 10, init_latency_ms: float = 35.0) -> None:
        super().__init__(
            capacity=capacity,
            cold_start_delay_ms=init_latency_ms,
            warm_start_delay_ms=2.0,
            prewarmed=False,
        )


class PersistentPrewarmedPool(BaseContainerPool):
    """Prewarmed container pool: slots are initialized ahead of time for low latency."""

    def __init__(self, capacity: int = 10, warm_latency_ms: float = 2.0) -> None:
        super().__init__(
            capacity=capacity,
            cold_start_delay_ms=35.0,
            warm_start_delay_ms=warm_latency_ms,
            prewarmed=True,
        )
