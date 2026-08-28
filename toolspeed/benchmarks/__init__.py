"""ToolSpeed Real Benchmarking Framework: Harness, Backends, and Paired Evaluation."""

from __future__ import annotations

from toolspeed.benchmarks.harness import (
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkRunResult,
    PairedWorkloadEvaluation,
)
from toolspeed.benchmarks.local_backend import LocalWallClockBackend
from toolspeed.benchmarks.replay_backend import ReplayBackend

__all__ = [
    "BenchmarkConfig",
    "BenchmarkHarness",
    "BenchmarkRunResult",
    "LocalWallClockBackend",
    "PairedWorkloadEvaluation",
    "ReplayBackend",
]
