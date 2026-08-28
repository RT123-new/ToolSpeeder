"""Async token bucket and concurrency rate limiters with lease semantics, backpressure, and safety."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional
import time


class RateLimitError(Exception):
    """Raised when rate limit is exceeded (HTTP 429 simulation)."""
    def __init__(self, message: str = "Rate limit exceeded (HTTP 429)", retry_after_s: float = 1.0):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class AsyncTokenBucket:
    """Async token bucket rate limiter with backpressure and queuing metrics."""

    def __init__(
        self,
        rate: float = 100.0,
        capacity: float = 100.0,
        reject_on_limit: bool = False,
    ):
        if rate <= 0:
            raise ValueError(f"Rate must be positive, got {rate}")
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        self.rate = float(rate)
        self.capacity = float(capacity)
        self.reject_on_limit = reject_on_limit
        self._tokens = float(capacity)
        self._last_refill_ns = time.perf_counter_ns()
        self._lock = asyncio.Lock()

        # Metrics
        self.total_tokens_requested: int = 0
        self.total_tokens_granted: int = 0
        self.total_429_errors: int = 0
        self.total_queue_delay_ns: int = 0
        self.max_queue_delay_ns: int = 0

    def _refill(self) -> None:
        now_ns = time.perf_counter_ns()
        elapsed_s = (now_ns - self._last_refill_ns) / 1_000_000_000.0
        if elapsed_s > 0:
            added = elapsed_s * self.rate
            self._tokens = min(self.capacity, self._tokens + added)
            self._last_refill_ns = now_ns

    async def acquire(self, tokens: int = 1, timeout: Optional[float] = None, deadline: Optional[float] = None) -> float:
        """Acquire tokens, asynchronously waiting if needed without holding concurrency slots."""
        if tokens <= 0:
            raise ValueError(f"Requested tokens must be positive, got {tokens}")
        if tokens > self.capacity:
            raise ValueError(f"Requested tokens ({tokens}) exceeds maximum bucket capacity ({self.capacity})")

        start_ns = time.perf_counter_ns()
        eff_deadline = deadline if deadline is not None else ((time.perf_counter() + timeout) if timeout is not None else None)

        async with self._lock:
            self.total_tokens_requested += tokens

        while True:
            if eff_deadline is not None and time.perf_counter() > eff_deadline:
                raise asyncio.TimeoutError(f"Timed out acquiring {tokens} rate limit tokens")

            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self.total_tokens_granted += tokens
                    break
                elif self.reject_on_limit:
                    self.total_429_errors += 1
                    missing = tokens - self._tokens
                    wait_s = missing / self.rate
                    raise RateLimitError(
                        f"Rate limit exceeded: needed {tokens}, available {self._tokens:.2f}",
                        retry_after_s=wait_s,
                    )
                else:
                    missing = tokens - self._tokens
                    wait_s = missing / self.rate

            # Sleep outside lock to allow token accumulation
            sleep_time = max(0.001, wait_s)
            if eff_deadline is not None:
                remaining = eff_deadline - time.perf_counter()
                if remaining <= 0:
                    raise asyncio.TimeoutError(f"Timed out acquiring {tokens} rate limit tokens")
                sleep_time = min(sleep_time, remaining)

            await asyncio.sleep(sleep_time)

        end_ns = time.perf_counter_ns()
        delay_ns = end_ns - start_ns
        async with self._lock:
            self.total_queue_delay_ns += delay_ns
            if delay_ns > self.max_queue_delay_ns:
                self.max_queue_delay_ns = delay_ns
        return delay_ns / 1_000_000.0

    def refund(self, tokens: int = 1) -> None:
        """Refund tokens safely if an operation was cancelled before execution."""
        if tokens <= 0:
            return
        self._refill()
        self._tokens = min(self.capacity, self._tokens + tokens)
        if self.total_tokens_granted >= tokens:
            self.total_tokens_granted -= tokens

    def try_acquire(self, tokens: int = 1) -> bool:
        """Non-blocking attempt to acquire tokens. Returns True if granted."""
        if tokens <= 0:
            raise ValueError(f"Requested tokens must be positive, got {tokens}")
        if tokens > self.capacity:
            return False

        self._refill()
        self.total_tokens_requested += tokens
        if self._tokens >= tokens:
            self._tokens -= tokens
            self.total_tokens_granted += tokens
            return True
        else:
            self.total_429_errors += 1
            return False

    @property
    def available_tokens(self) -> float:
        self._refill()
        return self._tokens

    def reset_metrics(self) -> None:
        self.total_tokens_requested = 0
        self.total_tokens_granted = 0
        self.total_429_errors = 0
        self.total_queue_delay_ns = 0
        self.max_queue_delay_ns = 0


class AsyncConcurrencyLimiter:
    """Async concurrency limiter tracking in-flight executions, queue delays, and peak concurrency."""

    def __init__(self, max_concurrency: int = 10):
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        self.max_concurrency = max_concurrency
        self._semaphore = asyncio.BoundedSemaphore(self.max_concurrency)
        self._active_count = 0
        self._peak_concurrency = 0
        self._lock = asyncio.Lock()

        # Metrics
        self.total_acquisitions = 0
        self.total_queue_delay_ns = 0
        self.max_queue_delay_ns = 0

    @property
    def active_count(self) -> int:
        return self._active_count

    @property
    def peak_concurrency(self) -> int:
        return self._peak_concurrency

    async def acquire(self, timeout: Optional[float] = None, deadline: Optional[float] = None) -> float:
        """Acquire concurrency slot. Returns queue delay in ms."""
        start_ns = time.perf_counter_ns()
        eff_timeout = timeout
        if deadline is not None:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise asyncio.TimeoutError("Deadline expired before acquiring concurrency slot")
            eff_timeout = min(remaining, timeout) if timeout is not None else remaining

        if eff_timeout is not None:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=eff_timeout)
        else:
            await self._semaphore.acquire()

        end_ns = time.perf_counter_ns()
        delay_ns = end_ns - start_ns

        async with self._lock:
            self._active_count += 1
            self.total_acquisitions += 1
            if self._active_count > self._peak_concurrency:
                self._peak_concurrency = self._active_count
            self.total_queue_delay_ns += delay_ns
            if delay_ns > self.max_queue_delay_ns:
                self.max_queue_delay_ns = delay_ns

        return delay_ns / 1_000_000.0

    def release(self) -> None:
        """Release concurrency slot safely, guarding against over-release."""
        if self._active_count <= 0:
            return
        self._active_count -= 1
        try:
            self._semaphore.release()
        except ValueError:
            pass

    async def __aenter__(self) -> AsyncConcurrencyLimiter:
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def reset_metrics(self) -> None:
        self.total_acquisitions = 0
        self.total_queue_delay_ns = 0
        self.max_queue_delay_ns = 0
        self._peak_concurrency = self._active_count


@dataclass
class RateLimitLease:
    """Exclusive cancellation-safe lease holding rate limit tokens and concurrency slots."""
    tokens_acquired: int
    concurrency_acquired: bool
    limiter: RateLimiter
    queue_delay_ms: float
    _released: bool = False

    def release(self) -> None:
        """Release concurrency slot and mark lease as released safely."""
        if self._released:
            return
        self._released = True
        if self.concurrency_acquired:
            self.limiter.concurrency_limiter.release()

    async def __aenter__(self) -> RateLimitLease:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class RateLimiter:
    """Unified rate limiter combining token bucket throughput and concurrency limits with cancellation-safe leases."""

    def __init__(
        self,
        rate_per_sec: float = 50.0,
        burst_capacity: float = 50.0,
        max_concurrency: int = 10,
        reject_on_limit: bool = False,
        requests_per_second: Optional[float] = None,
        concurrency_limit: Optional[int] = None,
    ):
        eff_rate = requests_per_second if requests_per_second is not None else rate_per_sec
        eff_conc = concurrency_limit if concurrency_limit is not None else max_concurrency
        self.token_bucket = AsyncTokenBucket(
            rate=max(0.001, eff_rate),
            capacity=max(1.0, burst_capacity),
            reject_on_limit=reject_on_limit,
        )
        self.concurrency_limiter = AsyncConcurrencyLimiter(max_concurrency=max(1, eff_conc))

    @asynccontextmanager
    async def lease(
        self,
        tokens: int = 1,
        timeout: Optional[float] = None,
        deadline: Optional[float] = None,
    ) -> AsyncIterator[RateLimitLease]:
        """Cancellation-safe context manager acquiring rate tokens first, then concurrency slot."""
        start_ns = time.perf_counter_ns()
        eff_deadline = deadline if deadline is not None else ((time.perf_counter() + timeout) if timeout is not None else None)

        # 1. Acquire rate limit tokens first (WITHOUT holding concurrency slot!)
        tokens_delay = await self.token_bucket.acquire(tokens=tokens, deadline=eff_deadline)
        concurrency_acquired = False

        try:
            # 2. Acquire concurrency slot
            conc_delay = await self.concurrency_limiter.acquire(deadline=eff_deadline)
            concurrency_acquired = True
            total_delay = (time.perf_counter_ns() - start_ns) / 1_000_000.0

            lease_obj = RateLimitLease(
                tokens_acquired=tokens,
                concurrency_acquired=True,
                limiter=self,
                queue_delay_ms=total_delay,
            )
            try:
                yield lease_obj
            finally:
                lease_obj.release()

        except (Exception, asyncio.CancelledError):
            if concurrency_acquired:
                self.concurrency_limiter.release()
            else:
                # If concurrency slot acquisition failed or was cancelled, refund the tokens
                self.token_bucket.refund(tokens=tokens)
            raise

    async def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> float:
        """Acquires tokens and concurrency slot."""
        deadline = (time.perf_counter() + timeout) if timeout is not None else None
        start_ns = time.perf_counter_ns()

        # Tokens first
        await self.token_bucket.acquire(tokens=tokens, deadline=deadline)

        try:
            await self.concurrency_limiter.acquire(deadline=deadline)
        except (Exception, asyncio.CancelledError):
            self.token_bucket.refund(tokens=tokens)
            raise

        return (time.perf_counter_ns() - start_ns) / 1_000_000.0

    def release(self) -> None:
        """Release concurrency slot."""
        self.concurrency_limiter.release()

    async def __aenter__(self) -> RateLimiter:
        await self.acquire(tokens=1)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "tokens_requested": self.token_bucket.total_tokens_requested,
            "tokens_granted": self.token_bucket.total_tokens_granted,
            "rate_limit_429_errors": self.token_bucket.total_429_errors,
            "token_queue_delay_ms": self.token_bucket.total_queue_delay_ns / 1_000_000.0,
            "token_max_queue_delay_ms": self.token_bucket.max_queue_delay_ns / 1_000_000.0,
            "concurrency_acquisitions": self.concurrency_limiter.total_acquisitions,
            "active_concurrency": self.concurrency_limiter.active_count,
            "peak_concurrency": self.concurrency_limiter.peak_concurrency,
            "concurrency_queue_delay_ms": self.concurrency_limiter.total_queue_delay_ns / 1_000_000.0,
        }
