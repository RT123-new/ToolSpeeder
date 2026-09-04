"""Tests for Phase 29: Independent metric recomputation from raw traces and discrepancy detection."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from toolspeed.benchmarks.bundle import compute_file_sha256, create_bundle
from toolspeed.benchmarks.recompute import (
    TraceRecomputationError,
    compare_metrics_discrepancy,
    generate_trace_recomputed_reports,
    recompute_bundle_metrics,
)
from toolspeed.cli import cmd_report


class TestEvidenceReport(unittest.TestCase):
    """Verifies that evidence report recomputes metrics from raw traces, detects discrepancies > 1e-6, and requires valid bundle seals."""

    def setUp(self) -> None:
        self.raw_candidate_traces = [
            {"workload_id": "W1", "trial": i, "ccl_ms": 50.0 + (i % 5), "success": True, "scheduler": "Candidate"}
            for i in range(20)
        ]
        self.raw_baseline_traces = [
            {"workload_id": "W1", "trial": i, "ccl_ms": 100.0 + (i % 5), "success": True, "scheduler": "Baseline"}
            for i in range(20)
        ]
        self.stored_summary = {
            "title": "ToolSpeed Replay Run",
            "evidence_level": "replay_integration",
            "evaluations": [
                {
                    "workload_id": "W1",
                    "candidate_name": "Candidate",
                    "baseline_name": "Baseline",
                    "summary": {
                        "baseline_p95_ms": 104.0,
                        "candidate_p95_ms": 54.0,
                        "p95_speedup": 104.0 / 54.0,
                        "candidate_success_rate": 1.0,
                    },
                    "verdict": {"passed": True},
                }
            ],
            "manifest": {"is_verdict_eligible": True, "trial_count": 20},
        }

    def _create_valid_test_bundle(self, bundle_dir: Path) -> Path:
        create_bundle(
            result_data=self.stored_summary,
            out_dir=bundle_dir,
            raw_traces=self.raw_candidate_traces,
        )
        # Write baseline traces
        b_file = bundle_dir / "baseline-traces.jsonl"
        with open(b_file, "w", encoding="utf-8") as f:
            for r in self.raw_baseline_traces:
                f.write(json.dumps(r) + "\n")

        # Update manifest with baseline-traces.jsonl hash and resign manifest.sig
        m_file = bundle_dir / "manifest.json"
        m_data = json.loads(m_file.read_text(encoding="utf-8"))
        m_data["file_hashes"]["baseline-traces.jsonl"] = compute_file_sha256(b_file)
        m_file.write_text(json.dumps(m_data, indent=2, sort_keys=True), encoding="utf-8")

        sig_file = bundle_dir / "manifest.sig"
        sig_file.write_text(f"{compute_file_sha256(m_file)}\n", encoding="utf-8")
        return bundle_dir

    def test_01_report_recomputes_from_raw_traces_not_result_json(self) -> None:
        """Report derives speedup and latency from raw traces, ignoring any forged numbers in result.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "recompute_bundle"
            self._create_valid_test_bundle(bundle_dir)

            recomputed = recompute_bundle_metrics(bundle_dir)
            self.assertTrue(recomputed["recomputed_from_traces"])
            evals = recomputed["evaluations"]
            self.assertEqual(len(evals), 1)
            w1_summ = evals[0]["summary"]
            # Candidate ~50-54ms, baseline ~100-104ms => speedup around ~1.9x
            self.assertAlmostEqual(w1_summ["p95_speedup"], 104.0 / 54.0, delta=0.1)
            self.assertEqual(w1_summ["candidate_success_rate"], 1.0)

    def test_02_flags_discrepancy_greater_than_1e6(self) -> None:
        """Flags discrepancies between recomputed and stored metrics exceeding 1e-6 tolerance."""
        recomputed = {
            "evaluations": [
                {
                    "workload_id": "W1",
                    "summary": {
                        "p95_speedup": 1.92592593,
                        "candidate_success_rate": 1.0,
                    },
                }
            ]
        }
        # Case A: forged stored result differs by 0.05 > 1e-6
        forged_stored = {
            "evaluations": [
                {
                    "workload_id": "W1",
                    "summary": {
                        "p95_speedup": 1.99000000,
                        "candidate_success_rate": 1.0,
                    },
                }
            ]
        }
        discrepancies = compare_metrics_discrepancy(recomputed, forged_stored, tolerance=1e-6)
        self.assertEqual(len(discrepancies), 1)
        self.assertIn("p95_speedup", discrepancies[0])
        self.assertIn("discrepancy", discrepancies[0])

        # Case B: identical metrics within 1e-7 => no discrepancy
        matching_stored = {
            "evaluations": [
                {
                    "workload_id": "W1",
                    "summary": {
                        "p95_speedup": 1.92592593,
                        "candidate_success_rate": 1.0,
                    },
                }
            ]
        }
        clean = compare_metrics_discrepancy(recomputed, matching_stored, tolerance=1e-6)
        self.assertEqual(len(clean), 0)

    def test_03_report_rejects_unsealed_or_tampered_bundle(self) -> None:
        """Trace recompute rejects bundle when manifest.sig is missing or tampered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "tampered_sig_bundle"
            self._create_valid_test_bundle(bundle_dir)

            # Corrupt manifest.sig
            (bundle_dir / "manifest.sig").write_text("corrupted_invalid_signature\n")

            with self.assertRaises(TraceRecomputationError) as ctx:
                generate_trace_recomputed_reports(bundle_dir)
            self.assertIn("manifest.sig mismatch", str(ctx.exception))

    def test_04_cli_cmd_report_integration(self) -> None:
        """`toolspeed report` command generates reports independently from raw traces."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "cli_bundle"
            self._create_valid_test_bundle(bundle_dir)
            out_dir = Path(tmpdir) / "output_reports"

            args = argparse.Namespace(input=str(bundle_dir), out=str(out_dir))
            code = cmd_report(args)

            self.assertEqual(code, 0)
            self.assertTrue((out_dir / "report.md").exists())
            self.assertTrue((out_dir / "report.html").exists())
            md_text = (out_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("W1", md_text)


if __name__ == "__main__":
    unittest.main()
