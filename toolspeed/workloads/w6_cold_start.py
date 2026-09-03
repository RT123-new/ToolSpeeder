"""Workload W6: Cold-Start Sandbox and Container Initialization."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
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


async def execute_cold_subprocess(expression: str = "2 + 2") -> tuple[Any, float]:
    """Spawns a fresh Python subprocess per call, incurring real process startup overhead."""
    start_ns = time.perf_counter_ns()
    script = f"import json; res = eval({expression!r}, {{'__builtins__': None, 'sum': sum, 'abs': abs}}); print(json.dumps({{'status': 'success', 'result': res}}))"
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    out = json.loads(stdout.decode("utf-8"))
    return out["result"], duration_ms


class WarmSubprocessWorker:
    """Persistent worker process communicating over pipes/stdin/stdout."""

    def __init__(self) -> None:
        self.proc: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        script = (
            "import sys, json\n"
            'sys.stdout.write(json.dumps({"status": "ready"}) + "\\n")\n'
            "sys.stdout.flush()\n"
            "for line in sys.stdin:\n"
            "    if not line.strip(): continue\n"
            "    req = json.loads(line)\n"
            '    if req.get("action") == "shutdown": break\n'
            '    expr = req.get("expression", "2 + 2")\n'
            '    res = eval(expr, {"__builtins__": None, "sum": sum, "abs": abs})\n'
            '    sys.stdout.write(json.dumps({"status": "success", "result": res}) + "\\n")\n'
            "    sys.stdout.flush()\n"
        )
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            script,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if self.proc.stdout is not None:
            ready_line = await self.proc.stdout.readline()
            out = json.loads(ready_line.decode("utf-8"))
            if out.get("status") != "ready":
                raise RuntimeError(f"Unexpected worker handshake: {ready_line!r}")

    async def execute(self, expression: str = "2 + 2") -> tuple[Any, float]:
        if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("Warm worker process not started")
        start_ns = time.perf_counter_ns()
        payload = json.dumps({"expression": expression}) + "\n"
        self.proc.stdin.write(payload.encode("utf-8"))
        await self.proc.stdin.drain()
        line = await self.proc.stdout.readline()
        duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        out = json.loads(line.decode("utf-8"))
        return out["result"], duration_ms

    async def close(self) -> None:
        if self.proc is not None:
            if self.proc.stdin and not self.proc.stdin.is_closing():
                try:
                    self.proc.stdin.write(b'{"action":"shutdown"}\n')
                    await self.proc.stdin.drain()
                    self.proc.stdin.close()
                except Exception:
                    pass
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=1.0)
            except Exception:
                self.proc.kill()
                await self.proc.wait()


class WarmSubprocessPool:
    """Pool of reusable persistent worker processes."""

    def __init__(self, pool_size: int = 4) -> None:
        self.pool_size = pool_size
        self.workers: list[WarmSubprocessWorker] = []
        self.queue: asyncio.Queue[WarmSubprocessWorker] = asyncio.Queue()
        self.init_duration_ms: float = 0.0

    async def start(self, concurrency: int = 4) -> None:
        start_ns = time.perf_counter_ns()
        self.workers = [WarmSubprocessWorker() for _ in range(self.pool_size)]
        sem = asyncio.Semaphore(concurrency)

        async def _start_worker(w: WarmSubprocessWorker) -> None:
            async with sem:
                await w.start()
                await self.queue.put(w)

        await asyncio.gather(*[_start_worker(w) for w in self.workers])
        self.init_duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

    async def execute(self, expression: str = "2 + 2") -> tuple[Any, float]:
        worker = await self.queue.get()
        try:
            return await worker.execute(expression)
        finally:
            await self.queue.put(worker)

    async def close(self) -> None:
        for worker in self.workers:
            await worker.close()
        self.workers.clear()


@dataclass(frozen=True)
class W6AmortizationPoint:
    concurrency: int
    call_count: int
    pool_size: int
    cold_duration_ms: float
    warm_duration_ms: float
    speedup: float
    is_positive: bool


@dataclass
class W6SubprocessSweepReport:
    points: list[W6AmortizationPoint]

    def verify_subprocess_amortization_invariants(self) -> tuple[bool, str]:
        """Verifies:

        - Speedup is positive (> 1.0x) only when call_count > pool_size.
        - Speedup is <= 1.0x when call_count <= pool_size.
        """
        if not self.points:
            return False, "No amortization points evaluated"

        for p in self.points:
            if p.call_count <= p.pool_size and p.speedup > 1.05:
                return (
                    False,
                    f"Speedup was positive ({p.speedup:.2f}x) when call_count ({p.call_count}) <= pool_size ({p.pool_size}) at concurrency {p.concurrency}",
                )
            if p.call_count > p.pool_size and p.speedup <= 1.0:
                return (
                    False,
                    f"Speedup was not positive ({p.speedup:.2f}x) when call_count ({p.call_count}) > pool_size ({p.pool_size}) at concurrency {p.concurrency}",
                )

        return True, "All W6 subprocess warm vs cold amortization invariants hold."


async def evaluate_w6_subprocess_warm_vs_cold(
    concurrencies: Sequence[int] = (1, 2, 4, 8),
    pool_size: int = 4,
    under_amortized_calls: int = 2,
    over_amortized_calls: int = 12,
) -> W6SubprocessSweepReport:
    """Measures real process spawn overhead comparing fresh cold processes vs pooled warm processes."""
    points: list[W6AmortizationPoint] = []

    for conc in concurrencies:
        for count in (under_amortized_calls, over_amortized_calls):
            # Cold execution: fresh process per call under concurrency limit
            sem = asyncio.Semaphore(conc)

            async def _run_cold(idx: int, s: asyncio.Semaphore = sem) -> float:
                async with s:
                    _, dur = await execute_cold_subprocess(f"{idx} + {idx}")
                    return dur

            start_cold = time.perf_counter_ns()
            await asyncio.gather(*[_run_cold(i) for i in range(count)])
            cold_total_ms = (time.perf_counter_ns() - start_cold) / 1_000_000.0

            # Warm execution: prewarmed pool of size pool_size, then run count calls
            pool = WarmSubprocessPool(pool_size=pool_size)
            start_warm = time.perf_counter_ns()
            await pool.start(concurrency=conc)

            async def _run_warm(idx: int, s: asyncio.Semaphore = sem, p: WarmSubprocessPool = pool) -> float:
                async with s:
                    _, dur = await p.execute(f"{idx} + {idx}")
                    return dur

            await asyncio.gather(*[_run_warm(i) for i in range(count)])
            warm_total_ms = (time.perf_counter_ns() - start_warm) / 1_000_000.0
            await pool.close()

            speedup = cold_total_ms / warm_total_ms if warm_total_ms > 0 else 1.0
            points.append(
                W6AmortizationPoint(
                    concurrency=conc,
                    call_count=count,
                    pool_size=pool_size,
                    cold_duration_ms=cold_total_ms,
                    warm_duration_ms=warm_total_ms,
                    speedup=speedup,
                    is_positive=speedup > 1.0,
                )
            )

    return W6SubprocessSweepReport(points=points)
