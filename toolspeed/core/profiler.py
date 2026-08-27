"""Nanosecond-resolution profiler, CCL tracker, and latency statistics aggregators."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import time

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
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.profiler = profiler
        self.name = name
        self.category = category
        self.metadata = metadata or {}
        self.start_ns: int = 0

    def __enter__(self) -> SpanContext:
        self.start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
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

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
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
        return cls(**{k: float(v) if isinstance(v, (int, float)) else v for k, v in data.items() if k in cls.__dataclass_fields__})


def calculate_percentiles(values: Union[Sequence[float], Sequence[int], np.ndarray]) -> LatencyStats:
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
    """Nanosecond-resolution execution profiler and timeline event recorder."""

    def __init__(self, task_id: Optional[str] = None):
        self.task_id: str = task_id or "default"
        self._start_ns: int = time.perf_counter_ns()
        self._end_ns: Optional[int] = None
        self._events: list[ExecutionEvent] = []
        self._spans: list[SpanRecord] = []
        self._open_spans: dict[str, int] = {}

    @staticmethod
    def now_ns() -> int:
        return time.perf_counter_ns()

    def start(self) -> None:
        self._start_ns = time.perf_counter_ns()
        self._events.clear()
        self._spans.clear()
        self._open_spans.clear()

    def stop(self) -> float:
        self._end_ns = time.perf_counter_ns()
        return (self._end_ns - self._start_ns) / 1_000_000.0

    def record_event(
        self,
        event_type: Union[EventType, str],
        task_id: Optional[str] = None,
        call_id: Optional[str] = None,
        data: Optional[dict[str, Any]] = None,
        details: Optional[dict[str, Any]] = None,
        duration_ms: float = 0.0,
        timestamp_ns: Optional[int] = None,
    ) -> ExecutionEvent:
        ts_ns = timestamp_ns if timestamp_ns is not None else time.perf_counter_ns()
        event_data = data or details or {}
        event = ExecutionEvent(
            event_type=event_type,
            timestamp=time.perf_counter(),
            timestamp_ns=ts_ns,
            task_id=task_id or self.task_id,
            call_id=call_id,
            duration_ms=duration_ms,
            data=event_data,
            details=event_data,
        )
        self._events.append(event)
        return event

    def start_span(self, name: str, category: str = "general", metadata: Optional[dict[str, Any]] = None) -> int:
        ts = time.perf_counter_ns()
        self._open_spans[name] = ts
        return ts

    def end_span(
        self,
        name: str,
        category: str = "general",
        metadata: Optional[dict[str, Any]] = None,
        event_type: Optional[Union[EventType, str]] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> Optional[SpanRecord]:
        end_ns = time.perf_counter_ns()
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
        metadata: Optional[dict[str, Any]] = None,
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
        self._spans.append(span)
        return span

    def span(self, name: str, category: str = "general", metadata: Optional[dict[str, Any]] = None) -> SpanContext:
        return SpanContext(self, name, category, metadata)

    def get_events(self) -> list[ExecutionEvent]:
        return list(self._events)

    def get_spans(self) -> list[SpanRecord]:
        return list(self._spans)

    def get_timeline(self) -> list[dict[str, Any]]:
        timeline: list[dict[str, Any]] = []
        for e in self._events:
            timeline.append({
                "type": "event",
                "timestamp_ns": e.timestamp_ns,
                "event_type": str(e.event_type),
                "task_id": e.task_id,
                "call_id": e.call_id,
                "data": e.data,
            })
        for s in self._spans:
            timeline.append({
                "type": "span_start",
                "timestamp_ns": s.start_ns,
                "name": s.name,
                "category": s.category,
                "metadata": s.metadata,
            })
            timeline.append({
                "type": "span_end",
                "timestamp_ns": s.end_ns,
                "name": s.name,
                "category": s.category,
                "duration_ms": s.duration_ms,
                "metadata": s.metadata,
            })
        timeline.sort(key=lambda item: item["timestamp_ns"])
        return timeline

    def reset(self) -> None:
        self._events.clear()
        self._spans.clear()
        self._open_spans.clear()
        self._start_ns = time.perf_counter_ns()


class CCLTracker:
    """Correct Completion Latency (CCL) Tracker and Aggregator."""

    def __init__(self):
        self._successful_durations_ns: list[int] = []
        self._failed_durations_ns: list[int] = []
        self._traces: list[ExecutionTrace] = []

    def record_trace(self, trace: ExecutionTrace) -> None:
        self._traces.append(trace)
        duration = trace.duration_ns
        if trace.success:
            self._successful_durations_ns.append(duration)
        else:
            self._failed_durations_ns.append(duration)

    def record_execution(self, duration_ns: int, success: bool) -> None:
        if success:
            self._successful_durations_ns.append(duration_ns)
        else:
            self._failed_durations_ns.append(duration_ns)

    def record_task(self, success: bool, latency_ms: float) -> None:
        duration_ns = int(latency_ms * 1_000_000)
        self.record_execution(duration_ns, success)

    @property
    def total_tasks(self) -> int:
        return len(self._successful_durations_ns) + len(self._failed_durations_ns)

    @property
    def successful_tasks(self) -> int:
        return len(self._successful_durations_ns)

    @property
    def failed_tasks(self) -> int:
        return len(self._failed_durations_ns)

    @property
    def success_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks

    def get_ccl_stats(self) -> LatencyStats:
        return self._compute_stats(self._successful_durations_ns)

    def get_stats(self) -> LatencyStats:
        return self.get_ccl_stats()

    def get_all_latency_stats(self) -> LatencyStats:
        all_durations = self._successful_durations_ns + self._failed_durations_ns
        return self._compute_stats(all_durations)

    def _compute_stats(self, durations_ns: Sequence[int]) -> LatencyStats:
        total = self.total_tasks
        success = self.successful_tasks
        fail = self.failed_tasks
        rate = self.success_rate

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
        self._successful_durations_ns.clear()
        self._failed_durations_ns.clear()
        self._traces.clear()


class LatencyProfiler:
    """Instruments agent execution with high-resolution timestamps and latency breakdowns."""

    def __init__(self) -> None:
        self.events: List[ExecutionEvent] = []
        self._start_time: float = time.perf_counter()
        self._end_time: Optional[float] = None
        self._active_spans: Dict[str, float] = {}

    def start(self) -> None:
        self._start_time = time.perf_counter()
        self.events.clear()
        self.record_event(EventType.TASK_START)

    def finish(self) -> float:
        self._end_time = time.perf_counter()
        duration_ms = (self._end_time - self._start_time) * 1000.0
        self.record_event(EventType.TASK_END, duration_ms=duration_ms)
        return duration_ms

    def record_event(
        self,
        event_type: Union[EventType, str],
        duration_ms: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
    ) -> ExecutionEvent:
        event = ExecutionEvent(
            event_type=event_type,
            timestamp=time.perf_counter(),
            timestamp_ns=time.perf_counter_ns(),
            duration_ms=duration_ms,
            details=details or {},
            data=details or {},
        )
        self.events.append(event)
        return event

    def start_span(self, span_name: str) -> None:
        self._active_spans[span_name] = time.perf_counter()

    def end_span(
        self, span_name: str, event_type: Union[EventType, str], details: Optional[Dict[str, Any]] = None
    ) -> float:
        start = self._active_spans.pop(span_name, time.perf_counter())
        duration_ms = (time.perf_counter() - start) * 1000.0
        self.record_event(event_type, duration_ms=duration_ms, details=details)
        return duration_ms

    def get_summary(self) -> Dict[str, Any]:
        total_ms = (
            ((self._end_time or time.perf_counter()) - self._start_time) * 1000.0
        )
        model_ms = sum(
            e.duration_ms for e in self.events if str(e.event_type) in (EventType.MODEL_END.value, "model_end")
        )
        tool_ms = sum(
            e.duration_ms for e in self.events if str(e.event_type) in (EventType.TOOL_END.value, "tool_end")
        )
        speculation_saved_ms = sum(
            e.duration_ms for e in self.events if str(e.event_type) in (EventType.SPECULATION_HIT.value, "speculation_hit")
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


def compute_latency_stats(latencies: List[float]) -> Dict[str, float]:
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


def compute_speedup(
    baseline_stats: Dict[str, float], candidate_stats: Dict[str, float]
) -> Dict[str, float]:
    res = {}
    for metric in ("p50_ms", "p90_ms", "p95_ms", "p99_ms", "mean_ms"):
        b_val = baseline_stats.get(metric, 0.0)
        c_val = candidate_stats.get(metric, 0.0)
        ratio_key = metric.replace("_ms", "_speedup")
        res[ratio_key] = (b_val / c_val) if c_val > 0 else 1.0
    return res
