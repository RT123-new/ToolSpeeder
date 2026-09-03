"""Tests for Workload W5 streaming incremental dispatch vs buffering."""

from __future__ import annotations

import unittest

from toolspeed.workloads.w5_large_payloads import (
    W5StreamingSweepReport,
    consume_with_buffered_parser,
    consume_with_stream_parser,
    evaluate_w5_streaming_vs_buffering,
)


class TestWorkloadW5(unittest.IsolatedAsyncioTestCase):
    """Verifies streaming incremental dispatch vs buffered parsing across 10KB, 100KB, 1MB, 5MB payloads."""

    async def test_01_streaming_vs_buffering_sweep_and_invariants(self) -> None:
        """Compares stream parser vs buffered parser across [10KB, 100KB, 1MB, 5MB] chunks."""
        report = await evaluate_w5_streaming_vs_buffering(
            payload_sizes_kb=(10, 100, 1000, 5000),
            chunk_size_bytes=8192,
            chunk_delay_s=0.0001,
        )

        self.assertIsInstance(report, W5StreamingSweepReport)
        self.assertEqual(len(report.points), 4)

        valid, reason = report.verify_streaming_advantage_invariants()
        self.assertTrue(valid, f"Streaming advantage invariants failed: {reason}")

        p10k = report.points[0]
        p100k = report.points[1]
        p1m = report.points[2]
        p5m = report.points[3]

        # TTFA speedup > 1.0x across all sizes
        self.assertGreater(p10k.ttfa_speedup, 1.0)
        self.assertGreater(p100k.ttfa_speedup, 1.0)
        self.assertGreater(p1m.ttfa_speedup, 1.0)
        self.assertGreater(p5m.ttfa_speedup, 1.0)

        # Time savings (buffered_ttfa - stream_ttfa) scales with payload size
        time_sav_10k = p10k.buffered_ttfa_ms - p10k.stream_ttfa_ms
        time_sav_100k = p100k.buffered_ttfa_ms - p100k.stream_ttfa_ms
        time_sav_1m = p1m.buffered_ttfa_ms - p1m.stream_ttfa_ms
        time_sav_5m = p5m.buffered_ttfa_ms - p5m.stream_ttfa_ms

        self.assertLess(time_sav_10k, time_sav_100k)
        self.assertLess(time_sav_100k, time_sav_1m)
        self.assertLess(time_sav_1m, time_sav_5m)

        # Memory savings scales with payload size
        self.assertLess(p10k.memory_savings_bytes, p100k.memory_savings_bytes)
        self.assertLess(p100k.memory_savings_bytes, p1m.memory_savings_bytes)
        self.assertLess(p1m.memory_savings_bytes, p5m.memory_savings_bytes)

    async def test_02_stream_parser_incremental_early_action(self) -> None:
        """Asserts stream parser emits early action significantly faster than full buffered parse on 1MB."""
        payload_bytes = 1024 * 1024
        s_ttfa, s_peak, s_count = await consume_with_stream_parser(payload_bytes, chunk_delay_s=0.0001)
        b_ttfa, b_peak, b_count = await consume_with_buffered_parser(payload_bytes, chunk_delay_s=0.0001)

        self.assertEqual(s_count, b_count)
        self.assertGreater(s_count, 0)
        # Stream TTFA is a fraction of full buffered parse time
        self.assertLess(s_ttfa, b_ttfa)
        # Stream peak memory is drastically lower than buffered peak memory
        self.assertLess(s_peak, b_peak)


if __name__ == "__main__":
    unittest.main()
