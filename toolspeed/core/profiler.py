"""Nanosecond-resolution profiler, CCL tracker, and latency statistics aggregators."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from toolspeed.core.types import EventType, ExecutionEvent, ExecutionTrace


@dataclass
class SpanRecord:
    """Record of a timed span within a task execution."""

    name: str = ""
    span_name: str = ""
    start_ns: int = 0
    end_ns: int = 0
    duration_ms: float = 0.0
    category: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name and self.span_name:
            self.name = self.span_name
        elif not self.span_name and self.name:
            self.span_name = self.name
        if self.duration_ms == 0.0 and self.end_ns >= self.start_ns and self.start_ns > 0:
            self.duration_ms = (self.end_ns - self.start_ns) / 1_000_000.0

    @property
    def duration_ns(self) -> int:
        return max(0, self.end_ns - self.start_ns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "span_name": self.name,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ms": self.duration_ms,
            "category": self.category,
            "metadata": self.metadata,
        }


class SpanContext:
    """Async & sync context manager for timing spans."""

    def __init__(
        self,
        profiler: NanosecondProfiler,
        name: str,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
    ):
        self.profiler = profiler
        self.name = name
        self.category = category
        self.metadata = metadata or {}
        self.start_ns: int = 0

    def __enter__(self) -> SpanContext:
        self.start_ns = time.perf_counter_ns()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        end_ns = time.perf_counter_ns()
        if exc_type is not None:
            self.metadata["exception"] = str(exc_val)
        duration_ms = (end_ns - self.start_ns) / 1_000_000.0
        self.profiler._spans.append(
            SpanRecord(
                name=self.name,
                span_name=self.name,
                start_ns=self.start_ns,
                end_ns=end_ns,
                duration_ms=duration_ms,
                category=self.category,
                metadata=self.metadata,
            )
        )

    async def __aenter__(self) -> SpanContext:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


@dataclass
class LatencyStats:
    """Statistical summary of latencies (in milliseconds)."""

    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    count: int = 0
    success_count: int = 0
    failure_count: int = 0
    success_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LatencyStats:
        return cls(
            **{
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in data.items()
                if k in cls.__dataclass_fields__
            }
        )


def calculate_percentiles(values: Sequence[float] | Sequence[int] | np.ndarray) -> LatencyStats:
    """Calculate p50, p90, p95, p99, mean, std from array of milliseconds."""
    if len(values) == 0:
        return LatencyStats()
    arr = np.array(values, dtype=np.float64)
    return LatencyStats(
        p50_ms=float(np.percentile(arr, 50)),
        p90_ms=float(np.percentile(arr, 90)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        mean_ms=float(np.mean(arr)),
        std_ms=float(np.std(arr)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        count=len(values),
    )


class NanosecondProfiler:
    """Nanosecond-resolution execution profiler and timeline event recorder.

    Strictly observational: measures elapsed time from actual execution timestamps.
    Never alters, overrides, or scales elapsed duration based on observed mechanism event labels.
    """

    def __init__(self, task_id: str | None = None, clock: Any = None):
        self.task_id: str = task_id or "default"
        self.clock = clock
        self._start_ns: int = self._now_ns()
        self._end_ns: int | None = None
        self._events: list[ExecutionEvent] = []
        self._spans: list[SpanRecord] = []
        self._open_spans: dict[str, int] = {}
        self._lock = threading.Lock()

    def _now_ns(self) -> int:
        if self.clock is not None and hasattr(self.clock, "now_ns"):
            return self.clock.now_ns()
        return time.perf_counter_ns()

    def _now_s(self) -> float:
        if self.clock is not None and hasattr(self.clock, "now_s"):
            return self.clock.now_s()
        return time.perf_counter()

    @staticmethod
    def now_ns() -> int:
        return time.perf_counter_ns()

    def start(self) -> None:
        with self._lock:
            self._start_ns = self._now_ns()
            self._events.clear()
            self._spans.clear()
            self._open_spans.clear()

    def stop(self) -> float:
        with self._lock:
            self._end_ns = self._now_ns()
            return (self._end_ns - self._start_ns) / 1_000_000.0

    def finish(self, ctx: Any = None) -> float:
        """Complete profiling session and return actual elapsed milliseconds."""
        with self._lock:
            self._end_ns = self._now_ns()
            return (self._end_ns - self._start_ns) / 1_000_000.0

    def record_event(
        self,
        event_type: EventType | str,
        task_id: str | None = None,
        call_id: str | None = None,
        data: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
        duration_ms: float = 0.0,
        timestamp_ns: int | None = None,
    ) -> ExecutionEvent:
        ts_ns = timestamp_ns if timestamp_ns is not None else self._now_ns()
        event_data = data or details or {}
        event = ExecutionEvent(
            event_type=event_type,
            timestamp=self._now_s(),
            timestamp_ns=ts_ns,
            task_id=task_id or self.task_id,
            call_id=call_id,
            duration_ms=duration_ms,
            data=event_data,
            details=event_data,
        )
        with self._lock:
            self._events.append(event)
        return event

    def start_span(self, name: str, category: str = "general", metadata: dict[str, Any] | None = None) -> int:
        ts = self._now_ns()
        with self._lock:
            self._open_spans[name] = ts
        return ts

    def end_span(
        self,
        name: str,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
        event_type: EventType | str | None = None,
        details: dict[str, Any] | None = None,
    ) -> SpanRecord | None:
        end_ns = self._now_ns()
        with self._lock:
            start_ns = self._open_spans.pop(name, None)
            if start_ns is None:
                return None
            duration_ms = (end_ns - start_ns) / 1_000_000.0
            span = SpanRecord(
                name=name,
                span_name=name,
                start_ns=start_ns,
                end_ns=end_ns,
                duration_ms=duration_ms,
                category=category,
                metadata=metadata or details or {},
            )
            self._spans.append(span)
        if event_type is not None:
            self.record_event(event_type=event_type, duration_ms=duration_ms, details=details)
        return span

    def record_span(
        self,
        name: str,
        start_ns: int,
        end_ns: int,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
    ) -> SpanRecord:
        duration_ms = (end_ns - start_ns) / 1_000_000.0
        span = SpanRecord(
            name=name,
            span_name=name,
            start_ns=start_ns,
            end_ns=end_ns,
            duration_ms=duration_ms,
            category=category,
            metadata=metadata or {},
        )
        with self._lock:
            self._spans.append(span)
        return span

    def span(self, name: str, category: str = "general", metadata: dict[str, Any] | None = None) -> SpanContext:
        return SpanContext(self, name, category, metadata)

    def get_events(self) -> list[ExecutionEvent]:
        with self._lock:
            return list(self._events)

    def get_spans(self) -> list[SpanRecord]:
        with self._lock:
            return list(self._spans)

    def get_timeline(self) -> list[dict[str, Any]]:
        with self._lock:
            timeline: list[dict[str, Any]] = []
            for e in self._events:
                timeline.append(
                    {
                        "type": "event",
                        "timestamp_ns": e.timestamp_ns,
                        "event_type": str(e.event_type),
                        "task_id": e.task_id,
                        "call_id": e.call_id,
                        "data": e.data,
                    }
                )
            for s in self._spans:
                timeline.append(
                    {
                        "type": "span_start",
                        "timestamp_ns": s.start_ns,
                        "name": s.name,
                        "category": s.category,
                        "metadata": s.metadata,
                    }
                )
                timeline.append(
                    {
                        "type": "span_end",
                        "timestamp_ns": s.end_ns,
                        "name": s.name,
                        "category": s.category,
                        "duration_ms": s.duration_ms,
                        "metadata": s.metadata,
                    }
                )
            timeline.sort(key=lambda item: item["timestamp_ns"])
            return timeline

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._spans.clear()
            self._open_spans.clear()
            self._start_ns = time.perf_counter_ns()


class CCLTracker:
    """Correct Completion Latency (CCL) Tracker and Aggregator."""

    def __init__(self) -> None:
        self._successful_durations_ns: list[int] = []
        self._failed_durations_ns: list[int] = []
        self._traces: list[ExecutionTrace] = []
        self._lock = threading.Lock()

    def record_trace(self, trace: ExecutionTrace) -> None:
        with self._lock:
            self._traces.append(trace)
            duration = trace.duration_ns
            if trace.success:
                self._successful_durations_ns.append(duration)
            else:
                self._failed_durations_ns.append(duration)

    def record_execution(self, duration_ns: int, success: bool) -> None:
        with self._lock:
            if success:
                self._successful_durations_ns.append(duration_ns)
            else:
                self._failed_durations_ns.append(duration_ns)

    def record_task(self, success: bool, latency_ms: float) -> None:
        duration_ns = int(latency_ms * 1_000_000)
        self.record_execution(duration_ns, success)

    @property
    def total_tasks(self) -> int:
        with self._lock:
            return len(self._successful_durations_ns) + len(self._failed_durations_ns)

    @property
    def successful_tasks(self) -> int:
        with self._lock:
            return len(self._successful_durations_ns)

    @property
    def failed_tasks(self) -> int:
        with self._lock:
            return len(self._failed_durations_ns)

    @property
    def success_rate(self) -> float:
        tot = self.total_tasks
        if tot == 0:
            return 0.0
        return self.successful_tasks / tot

    def get_ccl_stats(self) -> LatencyStats:
        with self._lock:
            return self._compute_stats(self._successful_durations_ns)

    def get_stats(self) -> LatencyStats:
        return self.get_ccl_stats()

    def get_all_latency_stats(self) -> LatencyStats:
        with self._lock:
            all_durations = self._successful_durations_ns + self._failed_durations_ns
            return self._compute_stats(all_durations)

    def _compute_stats(self, durations_ns: Sequence[int]) -> LatencyStats:
        total = len(self._successful_durations_ns) + len(self._failed_durations_ns)
        success = len(self._successful_durations_ns)
        fail = len(self._failed_durations_ns)
        rate = success / total if total > 0 else 0.0

        if not durations_ns:
            return LatencyStats(
                count=0,
                success_count=success,
                failure_count=fail,
                success_rate=rate,
                mean_ms=0.0,
                std_ms=0.0,
                min_ms=0.0,
                p50_ms=0.0,
                p90_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                max_ms=0.0,
            )

        ms_values = np.array(durations_ns, dtype=np.float64) / 1_000_000.0
        p50 = float(np.percentile(ms_values, 50))
        p90 = float(np.percentile(ms_values, 90))
        p95 = float(np.percentile(ms_values, 95))
        p99 = float(np.percentile(ms_values, 99))
        mean = float(np.mean(ms_values))
        std = float(np.std(ms_values))
        min_v = float(np.min(ms_values))
        max_v = float(np.max(ms_values))

        return LatencyStats(
            count=len(durations_ns),
            success_count=success,
            failure_count=fail,
            success_rate=rate,
            mean_ms=mean,
            std_ms=std,
            min_ms=min_v,
            p50_ms=p50,
            p90_ms=p90,
            p95_ms=p95,
            p99_ms=p99,
            max_ms=max_v,
        )

    def reset(self) -> None:
        with self._lock:
            self._successful_durations_ns.clear()
            self._failed_durations_ns.clear()
            self._traces.clear()


class LatencyProfiler:
    """Instruments agent execution with high-resolution timestamps and latency breakdowns.

    Observational only: never computes synthetic timelines or applies mechanism duration formulas.
    """

    def __init__(self, task_id: str | None = None, clock: Any = None) -> None:
        self.task_id = task_id or "default"
        self.clock = clock
        self.events: list[ExecutionEvent] = []
        self._start_time: float = self._now_s()
        self._end_time: float | None = None
        self._active_spans: dict[str, float] = {}
        self._lock = threading.Lock()

    def _now_ns(self) -> int:
        if self.clock is not None and hasattr(self.clock, "now_ns"):
            return self.clock.now_ns()
        return time.perf_counter_ns()

    def _now_s(self) -> float:
        if self.clock is not None and hasattr(self.clock, "now_s"):
            return self.clock.now_s()
        return time.perf_counter()

    def start(self) -> None:
        with self._lock:
            self._start_time = self._now_s()
            self.events.clear()
        self.record_event(EventType.TASK_START)

    def finish(self, ctx: Any = None) -> float:
        with self._lock:
            self._end_time = self._now_s()
            duration_ms = (self._end_time - self._start_time) * 1000.0

        self.record_event(EventType.TASK_END, duration_ms=duration_ms)
        return duration_ms

    def record_event(
        self,
        event_type: EventType | str,
        duration_ms: float = 0.0,
        details: dict[str, Any] | None = None,
        task_id: str | None = None,
        call_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> ExecutionEvent:
        event_data = data or details or {}
        event = ExecutionEvent(
            event_type=event_type,
            timestamp=self._now_s(),
            timestamp_ns=self._now_ns(),
            duration_ms=duration_ms,
            task_id=task_id or self.task_id,
            call_id=call_id,
            details=event_data,
            data=event_data,
        )
        with self._lock:
            self.events.append(event)
        return event

    def start_span(self, span_name: str, category: str = "general", metadata: dict[str, Any] | None = None) -> float:
        with self._lock:
            ts = self._now_s()
            self._active_spans[span_name] = ts
            return ts

    def end_span(
        self,
        span_name: str,
        event_type: EventType | str | None = None,
        details: dict[str, Any] | None = None,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
    ) -> float:
        with self._lock:
            start = self._active_spans.pop(span_name, self._now_s())
        duration_ms = (self._now_s() - start) * 1000.0
        if event_type is not None:
            self.record_event(event_type, duration_ms=duration_ms, details=details or metadata)
        return duration_ms

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            total_ms = ((self._end_time or self._now_s()) - self._start_time) * 1000.0
            model_ms = sum(
                e.duration_ms for e in self.events if str(e.event_type) in (EventType.MODEL_END.value, "model_end")
            )
            tool_ms = sum(
                e.duration_ms for e in self.events if str(e.event_type) in (EventType.TOOL_END.value, "tool_end")
            )
            speculation_saved_ms = sum(
                e.duration_ms
                for e in self.events
                if str(e.event_type) in (EventType.SPECULATION_HIT.value, "speculation_hit")
            )
            cache_hits = sum(1 for e in self.events if str(e.event_type) in (EventType.CACHE_HIT.value, "cache_hit"))

            return {
                "total_latency_ms": total_ms,
                "model_latency_ms": model_ms,
                "tool_latency_ms": tool_ms,
                "speculation_saved_ms": speculation_saved_ms,
                "cache_hits": cache_hits,
                "event_count": len(self.events),
            }


def compute_latency_stats(latencies: list[float]) -> dict[str, float]:
    stats = calculate_percentiles(latencies)
    return {
        "p50_ms": stats.p50_ms,
        "p90_ms": stats.p90_ms,
        "p95_ms": stats.p95_ms,
        "p99_ms": stats.p99_ms,
        "mean_ms": stats.mean_ms,
        "std_ms": stats.std_ms,
        "min_ms": stats.min_ms,
        "max_ms": stats.max_ms,
        "count": float(stats.count),
    }


def compute_speedup(baseline_stats: dict[str, float], candidate_stats: dict[str, float]) -> dict[str, float]:
    res: dict[str, float] = {}
    for metric in ("p50_ms", "p90_ms", "p95_ms", "p99_ms", "mean_ms"):
        b_val = baseline_stats.get(metric, 0.0)
        c_val = candidate_stats.get(metric, 0.0)
        ratio_key = metric.replace("_ms", "_speedup")
        res[ratio_key] = (b_val / c_val) if c_val > 0 else 1.0
    return res
