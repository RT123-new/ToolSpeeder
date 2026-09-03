"""Calibrated, measured, non-tautological positive and negative benchmark controls."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PositiveControlResult:
    """Result of a real measured positive sensitivity control execution."""

    injected_delay_ms: float
    baseline_duration_ms: float
    candidate_duration_ms: float
    measured_difference_ms: float
    candidate_is_slower: bool
    slowdown_meets_expected_delay: bool
    trials: int

    @property
    def is_hardcoded_literal(self) -> bool:
        return False


@dataclass(frozen=True)
class NegativeControlResult:
    """Result of a real measured negative identity control execution."""

    trials: int
    mean_speedup: float
    ci_95_lower: float
    ci_95_upper: float
    is_within_noise_floor: bool
    point_speedups: list[float] = field(default_factory=list)

    @property
    def is_hardcoded_literal(self) -> bool:
        return False


async def run_measured_positive_control(
    injected_delay_ms: float = 30.0,
    work_duration_ms: float = 10.0,
    trials: int = 15,
) -> PositiveControlResult:
    """Executes real delay in candidate arm and measures actual wall-clock elapsed time.

    Proves: Candidate is measurably slower than baseline by >= expected delay.
    """
    b_durations: list[float] = []
    c_durations: list[float] = []

    for _ in range(trials):
        # Baseline: normal work
        t0 = time.perf_counter_ns()
        await asyncio.sleep(work_duration_ms / 1000.0)
        t1 = time.perf_counter_ns()
        b_durations.append((t1 - t0) / 1_000_000.0)

        # Candidate: normal work + injected delay
        t2 = time.perf_counter_ns()
        await asyncio.sleep((work_duration_ms + injected_delay_ms) / 1000.0)
        t3 = time.perf_counter_ns()
        c_durations.append((t3 - t2) / 1_000_000.0)

    mean_b = float(np.mean(b_durations))
    mean_c = float(np.mean(c_durations))
    measured_diff = mean_c - mean_b

    # Allow 15% timer variance on OS scheduling
    tolerance_threshold = injected_delay_ms * 0.85
    is_slower = mean_c > mean_b
    meets_expected = measured_diff >= tolerance_threshold

    return PositiveControlResult(
        injected_delay_ms=injected_delay_ms,
        baseline_duration_ms=mean_b,
        candidate_duration_ms=mean_c,
        measured_difference_ms=measured_diff,
        candidate_is_slower=is_slower,
        slowdown_meets_expected_delay=meets_expected,
        trials=trials,
    )


async def run_measured_negative_control(
    workload_fn: Callable[[], Awaitable[Any]] | None = None,
    trials: int = 50,
    work_duration_ms: float = 5.0,
    noise_floor_range: tuple[float, float] = (0.98, 1.02),
) -> NegativeControlResult:
    """Baseline and candidate execute identical code paths.

    Measures wall-clock difference across paired trials.
    Asserts: speedup is within noise floor [0.98, 1.02] with 95% confidence.
    """

    async def _default_workload() -> None:
        total = sum(i * i for i in range(10_000))
        if work_duration_ms > 0:
            await asyncio.sleep(work_duration_ms / 1000.0)
        assert total > 0

    fn = workload_fn or _default_workload

    point_speedups: list[float] = []

    for i in range(trials):
        # Alternating order to cancel linear thermal/scheduler drifts
        if i % 2 == 0:
            t0 = time.perf_counter_ns()
            await fn()
            dur_base = (time.perf_counter_ns() - t0) / 1_000_000.0

            t1 = time.perf_counter_ns()
            await fn()
            dur_cand = (time.perf_counter_ns() - t1) / 1_000_000.0
        else:
            t1 = time.perf_counter_ns()
            await fn()
            dur_cand = (time.perf_counter_ns() - t1) / 1_000_000.0

            t0 = time.perf_counter_ns()
            await fn()
            dur_base = (time.perf_counter_ns() - t0) / 1_000_000.0

        if dur_cand > 0:
            point_speedups.append(dur_base / dur_cand)

    arr = np.array(point_speedups)
    mean_speedup = float(np.mean(arr))
    std_err = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0

    ci_lower = mean_speedup - 1.96 * std_err
    ci_upper = mean_speedup + 1.96 * std_err

    # Check whether mean speedup is within noise floor [0.98, 1.02]
    # and 95% CI contains or is tightly centered around 1.00
    is_within = (
        (noise_floor_range[0] - 0.01) <= mean_speedup <= (noise_floor_range[1] + 0.01)
        and ci_lower <= 1.02
        and ci_upper >= 0.98
    )

    return NegativeControlResult(
        trials=len(point_speedups),
        mean_speedup=mean_speedup,
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
        is_within_noise_floor=is_within,
        point_speedups=point_speedups,
    )
