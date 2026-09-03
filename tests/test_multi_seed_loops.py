"""Unit tests for outer multi-seed benchmark loops and cross-seed variance integrity."""

from __future__ import annotations

import unittest

from toolspeed.benchmarks.harness import BenchmarkConfig, BenchmarkHarness
from toolspeed.core.types import EvidenceLevel, Task


class TestMultiSeedLoops(unittest.IsolatedAsyncioTestCase):
    """Proves genuine outer loops over multiple seeds produce distinct fixtures and traces."""

    async def test_01_multi_seed_produces_distinct_fixtures(self) -> None:
        """Harness executing multiple seeds must produce distinct case digests across seeds."""
        seeds = [101, 103]
        cfg = BenchmarkConfig(
            trials_per_condition=2,
            seeds=seeds,
            evidence_level=EvidenceLevel.REPLAY_INTEGRATION,
        )
        harness = BenchmarkHarness(config=cfg)
        results = await harness.run_multi_seed_benchmark(seeds=seeds, trials=2)

        self.assertEqual(len(results), 2)
        # Check that seed 101 and seed 103 produced distinct manifests and task IDs
        self.assertEqual(results[0].manifest.seed, 101)  # type: ignore[union-attr]
        self.assertEqual(results[1].manifest.seed, 103)  # type: ignore[union-attr]

        task_ids_seed0 = [r_c.task_id for ev in results[0].evaluations for r_c in ev.candidate_results]
        task_ids_seed1 = [r_c.task_id for ev in results[1].evaluations for r_c in ev.candidate_results]

        # Ensure task IDs are distinct across seeds
        self.assertNotEqual(task_ids_seed0, task_ids_seed1)
        self.assertTrue(all("s101" in tid for tid in task_ids_seed0))
        self.assertTrue(all("s103" in tid for tid in task_ids_seed1))

    async def test_02_identical_cases_across_seeds_fails_closed(self) -> None:
        """Harness must raise ValueError if runs claim multi-seed execution but produce identical case digests."""
        seeds = [101, 103]
        cfg = BenchmarkConfig(
            trials_per_condition=2,
            seeds=seeds,
            evidence_level=EvidenceLevel.REPLAY_INTEGRATION,
        )
        harness = BenchmarkHarness(config=cfg)

        # Deliberately sabotage backend to return static unseeded tasks (simulating the Sep 2 bug)
        def static_task(workload_id: str, trial_index: int = 0, seed: int | None = None, arm: str = "baseline") -> Task:
            return Task(
                task_id=f"static_task_{trial_index}",
                prompt="static prompt",
                expected_output={"status": "success"},
            )

        harness.backend.generate_task = static_task  # type: ignore[method-assign,assignment]

        with self.assertRaises(ValueError) as ctx:
            await harness.run_multi_seed_benchmark(seeds=seeds, trials=2)

        self.assertIn("Multi-seed integrity failure", str(ctx.exception))
        self.assertIn("produced identical task cases", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
