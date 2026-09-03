"""Tests for Workload W4 pipeline cache locality under eviction pressure."""

from __future__ import annotations

import unittest

from toolspeed.benchmarks.replay_backend import ReplayBackend
from toolspeed.core.clock import WallClock
from toolspeed.schedulers.base import SchedulerConfig
from toolspeed.schedulers.phase2_cache import Phase2CacheScheduler, ToolResultCache
from toolspeed.workloads.w4_locality import (
    W4CacheEvictionSweepReport,
    evaluate_w4_cache_eviction_pressure,
)


class TestWorkloadW4(unittest.IsolatedAsyncioTestCase):
    """Verifies pipeline cache locality and hits vs misses across no-cache, cap 1, cap 4, and cap 16."""

    def setUp(self) -> None:
        self.backend = ReplayBackend(seed=42, clock=WallClock())

    async def test_01_cache_eviction_pressure_sweep_hits_misses_and_invariants(self) -> None:
        """Measures cache hits vs misses across no cache, capacity 1, capacity 4, and capacity 16."""
        report = await evaluate_w4_cache_eviction_pressure(
            backend=self.backend,
            capacities=(None, 1, 4, 16),
            trial_sequence=(
                0,
                0,
                1,
                0,
                1,
                2,
                2,
                0,
                3,
                0,
                4,
                1,
                0,
                5,
                0,
                1,
                6,
                2,
                1,
                0,
            ),
        )

        self.assertIsInstance(report, W4CacheEvictionSweepReport)
        self.assertEqual(len(report.points), 4)

        valid, reason = report.verify_eviction_pressure_invariants()
        self.assertTrue(valid, f"Cache eviction invariants failed: {reason}")

        p_no = report.points[0]
        p1 = report.points[1]
        p4 = report.points[2]
        p16 = report.points[3]

        # No cache: 0 hits, all misses
        self.assertEqual(p_no.hits, 0)
        self.assertEqual(p_no.misses, 20)
        self.assertEqual(p_no.hit_rate, 0.0)

        # LRU capacity 1: hits on immediate repeats
        self.assertGreater(p1.hits, 0)
        self.assertGreater(p1.hit_rate, 0.0)

        # LRU capacity 4: significantly higher hit rate than cap 1
        self.assertGreater(p4.hits, p1.hits)
        self.assertGreater(p4.hit_rate, p1.hit_rate)

        # LRU capacity 16: holds entire active set
        self.assertGreater(p16.hits, p4.hits)
        self.assertGreater(p16.hit_rate, p4.hit_rate)

        # Duration decreases as cache capacity expands
        self.assertLess(p16.duration_ms, p_no.duration_ms)

    async def test_02_lru_true_eviction_under_capacity_1(self) -> None:
        """Verifies true LRU eviction under capacity 1: alternating keys force eviction on every step."""
        cache = ToolResultCache(max_entries=1)
        sched = Phase2CacheScheduler(
            config=SchedulerConfig(cache_enabled=True),
            cache=cache,
        )

        # Alternating key sequence: 0, 1, 0, 1 -> capacity 1 must miss on every query
        hits = 0
        misses = 0
        for t in (0, 1, 0, 1):
            task = self.backend.generate_task("W4", trial_index=t, arm="candidate")
            tools, model = self.backend.create_workload_environment("W4", trial_index=t, arm="candidate")

            task_model = task.to_model_task() if hasattr(task, "to_model_task") else task
            res = await sched.execute(task_model, model, tools)

            for tr in res.tool_results:
                if tr.cached:
                    hits += 1
                else:
                    misses += 1

        self.assertEqual(hits, 0)
        self.assertEqual(misses, 4)


if __name__ == "__main__":
    unittest.main()
