"""Clock protocols and implementations for wall-clock and deterministic discrete-event virtual time."""

from __future__ import annotations

import asyncio
import heapq
import time
from typing import Protocol


class Clock(Protocol):
    """Timing abstraction separating wall-clock from virtual-clock discrete-event simulations."""

    def now_ns(self) -> int:
        """Current timestamp in nanoseconds."""
        ...

    def now_s(self) -> float:
        """Current timestamp in seconds."""
        ...

    async def sleep_ms(self, delay_ms: float) -> None:
        """Advance time or pause execution for delay_ms milliseconds."""
        ...


class WallClock:
    """Monotonic system wall clock for real OS primitives."""

    def now_ns(self) -> int:
        return time.perf_counter_ns()

    def now_s(self) -> float:
        return time.perf_counter()

    async def sleep_ms(self, delay_ms: float) -> None:
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)


class VirtualClock:
    """Deterministic discrete-event virtual clock.

    Maintains an ordered event queue for concurrent waiting operations.
    Advances simulation time to the next scheduled completion timestamp based on
    actual dispatch and dependency ordering.
    """

    def __init__(self, start_ns: int = 0) -> None:
        self._current_ns: int = start_ns
        # (finish_ns, event_id, asyncio.Event, cancelled_ref)
        self._queue: list[tuple[int, int, asyncio.Event, list[bool]]] = []
        self._event_counter = 0
        self._lock = asyncio.Lock()

    def now_ns(self) -> int:
        return self._current_ns

    def now_s(self) -> float:
        return self._current_ns / 1_000_000_000.0

    def _step_queue_locked(self) -> None:
        while self._queue and self._queue[0][3][0]:
            heapq.heappop(self._queue)
        if not self._queue:
            return

        earliest_ns = self._queue[0][0]
        if earliest_ns > self._current_ns:
            self._current_ns = earliest_ns

        while self._queue and self._queue[0][0] <= self._current_ns:
            _, _, next_evt, cancelled_ref = heapq.heappop(self._queue)
            if not cancelled_ref[0] and not next_evt.is_set():
                next_evt.set()

    async def sleep_ms(self, delay_ms: float) -> None:
        if delay_ms <= 0:
            await asyncio.sleep(0)
            return

        delay_ns = int(delay_ms * 1_000_000)
        target_ns = self._current_ns + delay_ns
        evt = asyncio.Event()
        cancelled_ref = [False]

        async with self._lock:
            self._event_counter += 1
            entry = (target_ns, self._event_counter, evt, cancelled_ref)
            heapq.heappush(self._queue, entry)

        # Allow other concurrent coroutines in the same turn to enqueue their sleep requests
        await asyncio.sleep(0)

        async with self._lock:
            self._step_queue_locked()

        try:
            await evt.wait()
        except asyncio.CancelledError:
            cancelled_ref[0] = True
            async with self._lock:
                self._step_queue_locked()
            raise
