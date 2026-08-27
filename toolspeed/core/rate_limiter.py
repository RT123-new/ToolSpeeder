"""Async token bucket and concurrency rate limiters with backpressure, safety, and 429 simulation."""

from __future__ import annotations

import asyncio
from typing import Optional, Dict, Any
import time


class RateLimitError(Exception):
    """Raised when rate limit is exceeded (HTTP 429 simulation)."""
    def __init__(self, message: str = "Rate limit exceeded (HTTP 429)", retry_after_s: float = 1.0):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class AsyncTokenBucket:
    """Async token bucket rate limiter with backpressure and queuing metrics.
    
    Attributes:
        rate: Refill rate in tokens per second.
        capacity: Maximum burst capacity in tokens.
        reject_on_limit: If True, raises RateLimitError when bucket is empty instead of waiting.
    """

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

    async def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> float:
        """Acquire tokens, asynchronously waiting (backpressure) if needed.
        
        Returns:
            Queuing delay in milliseconds.
            
        Raises:
            ValueError: If tokens <= 0 or tokens > capacity.
            RateLimitError: If reject_on_limit is True and insufficient tokens.
            asyncio.TimeoutError: If timeout expires before tokens are acquired.
        """
        if tokens <= 0:
            raise ValueError(f"Requested tokens must be positive, got {tokens}")
        if tokens > self.capacity:
            raise ValueError(f"Requested tokens ({tokens}) exceeds maximum bucket capacity ({self.capacity})")

        start_ns = time.perf_counter_ns()
        deadline = (time.perf_counter() + timeout) if timeout is not None else None

        async with self._lock:
            self.total_tokens_requested += tokens

        while True:
            if deadline is not None and time.perf_counter() > deadline:
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
            if deadline is not None:
                remaining = deadline - time.perf_counter()
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
        """Refund tokens if an operation was cancelled before execution."""
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

    async def acquire(self, timeout: Optional[float] = None) -> float:
        """Acquire concurrency slot. Returns queue delay in ms."""
        start_ns = time.perf_counter_ns()

        if timeout is not None:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
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
            # BoundedSemaphore raised ValueError because release exceeded initial value
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


class RateLimiter:
    """Unified rate limiter combining token bucket throughput and concurrency limits with cancellation safety."""

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

    async def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> float:
        """Acquire both concurrency slot and rate limit tokens with single deadline and rollback safety."""
        start_ns = time.perf_counter_ns()
        deadline = (time.perf_counter() + timeout) if timeout is not None else None

        # 1. Acquire concurrency slot first
        conc_timeout = (deadline - time.perf_counter()) if deadline is not None else None
        if conc_timeout is not None and conc_timeout <= 0:
            raise asyncio.TimeoutError("Timeout expired before acquiring concurrency slot")
        
        await self.concurrency_limiter.acquire(timeout=conc_timeout)

        # 2. Acquire rate limit tokens. If token acquire fails or is cancelled, release concurrency slot!
        try:
            token_timeout = (deadline - time.perf_counter()) if deadline is not None else None
            if token_timeout is not None and token_timeout <= 0:
                raise asyncio.TimeoutError("Timeout expired before acquiring rate limit tokens")
            await self.token_bucket.acquire(tokens=tokens, timeout=token_timeout)
        except (Exception, asyncio.CancelledError):
            self.concurrency_limiter.release()
            raise

        total_delay_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        return total_delay_ms

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
