"""Tests for local paired execution state isolation and noise floor calibration."""

from __future__ import annotations

import os
import sqlite3
import unittest

from toolspeed.benchmarks.local_backend import (
    LocalNoiseFloorCalibrator,
    LocalWallClockBackend,
    NoiseFloorReport,
)
from toolspeed.schedulers.b1_sync_react import SyncReActScheduler
from toolspeed.schedulers.b2_native_parallel import NativeParallelScheduler


class TestLocalPairedState(unittest.IsolatedAsyncioTestCase):
    """Verifies complete state isolation (ports, dirs, dbs) and empirical noise floor calibration."""

    async def asyncSetUp(self) -> None:
        self.backend = LocalWallClockBackend(seed=42)

    async def asyncTearDown(self) -> None:
        self.backend.cleanup()

    async def test_01_isolated_ports_across_arms_and_trials(self) -> None:
        """Each arm and trial receives an isolated HTTP server with its own distinct port."""
        srv_b0 = self.backend._get_isolated_server("baseline", 0)
        srv_c0 = self.backend._get_isolated_server("candidate", 0)
        srv_b1 = self.backend._get_isolated_server("baseline", 1)

        self.assertNotEqual(srv_b0.port, srv_c0.port)
        self.assertNotEqual(srv_b0.port, srv_b1.port)
        self.assertNotEqual(srv_c0.port, srv_b1.port)

    async def test_02_isolated_databases_and_zero_cross_trial_accumulation(self) -> None:
        """W2 SQLite databases are isolated per arm and trial; mutations do not cross-contaminate."""
        st_b0 = await self.backend.create_w2_state(trial_idx=0, arm="baseline")
        st_c0 = await self.backend.create_w2_state(trial_idx=0, arm="candidate")
        st_b1 = await self.backend.create_w2_state(trial_idx=1, arm="baseline")

        self.assertNotEqual(st_b0.db_path, st_c0.db_path)
        self.assertNotEqual(st_b0.db_path, st_b1.db_path)

        # Mutate baseline trial 0 DB
        conn = sqlite3.connect(st_b0.db_path)
        conn.execute("INSERT INTO orders VALUES (9999, 'injected', 'mutated', 999.0)")
        conn.commit()
        conn.close()

        # Check row counts
        cnt_b0 = await self.backend.get_w2_row_count(trial_idx=0, arm="baseline", db_path=st_b0.db_path)
        cnt_c0 = await self.backend.get_w2_row_count(trial_idx=0, arm="candidate", db_path=st_c0.db_path)
        cnt_b1 = await self.backend.get_w2_row_count(trial_idx=1, arm="baseline", db_path=st_b1.db_path)

        self.assertEqual(cnt_b0, 101)
        self.assertEqual(cnt_c0, 100)  # Untouched
        self.assertEqual(cnt_b1, 100)  # Untouched

    async def test_03_isolated_sandbox_directories(self) -> None:
        """W6 subprocess sandbox directories are completely isolated across arms and trials."""
        sb_b0 = self.backend._get_isolated_sandbox_dir("baseline", 0)
        sb_c0 = self.backend._get_isolated_sandbox_dir("candidate", 0)
        sb_b1 = self.backend._get_isolated_sandbox_dir("baseline", 1)

        self.assertNotEqual(sb_b0, sb_c0)
        self.assertNotEqual(sb_b0, sb_b1)
        self.assertTrue(os.path.isdir(sb_b0))
        self.assertTrue(os.path.isdir(sb_c0))
        self.assertTrue(os.path.isdir(sb_b1))

    async def test_04_noise_floor_null_hypothesis_calibration(self) -> None:
        """Calibrator executes null trials to calculate empirical noise floor and 3x MDE."""
        calibrator = LocalNoiseFloorCalibrator(backend=self.backend)
        report = await calibrator.calibrate(
            baseline_cls=SyncReActScheduler,
            candidate_cls=NativeParallelScheduler,
            workload_id="W1",
            trials=2,
        )

        self.assertIsInstance(report, NoiseFloorReport)
        self.assertGreater(report.noise_floor_ms, 0.0)
        self.assertAlmostEqual(report.mde_ms, report.noise_floor_ms * 3.0, places=5)
        self.assertEqual(report.trials_sampled, 2)
        self.assertTrue(report.is_statistically_sound)

    def test_05_rejection_of_claims_below_3x_noise_floor(self) -> None:
        """Claims where measured effect is less than 3x noise floor (MDE) must be rejected."""
        report = NoiseFloorReport(
            null_baseline_p50_ms=1.0,
            null_baseline_p95_ms=2.0,
            null_candidate_p50_ms=1.2,
            null_candidate_p95_ms=2.2,
            noise_floor_ms=1.5,
            mde_ms=4.5,  # 3x noise floor
            trials_sampled=10,
            is_statistically_sound=True,
        )

        # Effect < 3x noise floor -> REJECT
        accepted_sub, reason_sub = report.evaluate_claim(measured_effect_ms=2.5)
        self.assertFalse(accepted_sub)
        self.assertIn("below the minimum detectable effect", reason_sub)

        # Effect >= 3x noise floor -> ACCEPT
        accepted_sup, reason_sup = report.evaluate_claim(measured_effect_ms=5.0)
        self.assertTrue(accepted_sup)
        self.assertIn("exceeds MDE", reason_sup)


if __name__ == "__main__":
    unittest.main()
