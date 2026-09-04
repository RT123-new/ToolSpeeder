"""Tests for Phase 30: Independent falsification evaluation, trace recomputation, and fail-closed status codes."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from toolspeed.benchmarks.bundle import compute_file_sha256, create_bundle
from toolspeed.benchmarks.falsify import evaluate_falsification_independent


class TestFalsifyIndependent(unittest.TestCase):
    """Verifies independent recomputation from raw traces, fail-closed handling, and threshold enforcement."""

    def _create_sealed_bundle(
        self,
        bundle_dir: Path,
        raw_traces: list[dict[str, float | str | bool]],
        manifest_extra: dict[str, str | int | bool] | None = None,
    ) -> Path:
        m_extra: dict[str, str | int | bool] = {
            "evidence_level": "replay_integration",
            "trial_count": len(raw_traces),
            "is_verdict_eligible": True,
            "mode": "confirmatory",
        }
        if manifest_extra:
            m_extra.update(manifest_extra)

        result_data: dict[str, Any] = {
            "title": "Falsification Test Bundle",
            "evidence_level": m_extra["evidence_level"],
            "evaluations": [],
            "manifest": m_extra,
        }

        # Create self-contained bundle
        create_bundle(
            result_data=result_data,
            out_dir=bundle_dir,
            raw_traces=raw_traces,
        )

        # Also write baseline-traces.jsonl (with 100ms standard latency)
        b_file = bundle_dir / "baseline-traces.jsonl"
        with open(b_file, "w", encoding="utf-8") as f:
            for idx in range(len(raw_traces)):
                f.write(
                    json.dumps(
                        {"workload_id": "W1", "trial": idx, "ccl_ms": 100.0, "success": True, "scheduler": "Baseline"}
                    )
                    + "\n"
                )

        # Update manifest file_hashes and resign manifest.sig
        m_file = bundle_dir / "manifest.json"
        m_data = json.loads(m_file.read_text(encoding="utf-8"))
        m_data["file_hashes"]["baseline-traces.jsonl"] = compute_file_sha256(b_file)
        m_file.write_text(json.dumps(m_data, indent=2, sort_keys=True), encoding="utf-8")

        sig_file = bundle_dir / "manifest.sig"
        sig_file.write_text(f"{compute_file_sha256(m_file)}\n", encoding="utf-8")

        return bundle_dir

    def test_01_recomputes_from_raw_traces_never_reads_result_json(self) -> None:
        """Recomputes directly from raw traces; forged passed verdict in result.json is ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "forged_bundle"
            # Raw traces show candidate took 200ms vs baseline 100ms (0.5x speedup => falsified)
            failing_traces: list[dict[str, float | str | bool]] = [
                {"workload_id": "W1", "trial": i, "ccl_ms": 200.0, "success": True, "scheduler": "Candidate"}
                for i in range(1000)
            ]
            self._create_sealed_bundle(bundle_dir, failing_traces)

            # Forge result.json to claim 2.5x speedup and passed
            res_file = bundle_dir / "result.json"
            res_data = json.loads(res_file.read_text(encoding="utf-8"))
            res_data["evaluations"] = [
                {
                    "workload_id": "W1",
                    "summary": {"p95_speedup": 2.5, "candidate_success_rate": 1.0},
                    "verdict": {"passed": True, "falsified": False},
                }
            ]
            res_file.write_text(json.dumps(res_data), encoding="utf-8")

            # Resign so bundle hash structure is intact
            m_file = bundle_dir / "manifest.json"
            m_data = json.loads(m_file.read_text(encoding="utf-8"))
            m_data["file_hashes"]["result.json"] = compute_file_sha256(res_file)
            m_file.write_text(json.dumps(m_data, indent=2, sort_keys=True), encoding="utf-8")
            (bundle_dir / "manifest.sig").write_text(f"{compute_file_sha256(m_file)}\n", encoding="utf-8")

            code, msg, _ = evaluate_falsification_independent(bundle_dir)
            self.assertEqual(code, 1, f"Expected exit code 1 (falsified), got {code}: {msg}")
            self.assertIn("p95 speedup", msg)

    def test_02_missing_or_invalid_raw_traces_exits_code_3(self) -> None:
        """If raw-traces.jsonl is missing or invalid JSON, returns exit code 3 (malformed)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "corrupt_bundle"
            valid_traces: list[dict[str, float | str | bool]] = [
                {"workload_id": "W1", "trial": 0, "ccl_ms": 50.0, "success": True}
            ]
            self._create_sealed_bundle(bundle_dir, valid_traces)

            # Delete raw-traces.jsonl
            (bundle_dir / "raw-traces.jsonl").unlink()
            code_missing, _, _ = evaluate_falsification_independent(bundle_dir)
            self.assertEqual(code_missing, 3)

            # Corrupt raw-traces.jsonl
            (bundle_dir / "raw-traces.jsonl").write_text("not_valid_json_line\n", encoding="utf-8")
            code_corrupt, _, _ = evaluate_falsification_independent(bundle_dir)
            self.assertEqual(code_corrupt, 3)

    def test_03_recomputed_metrics_fail_threshold_exits_code_1(self) -> None:
        """Failing success rate or speedup threshold returns exit code 1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "fail_threshold_bundle"
            # 80% success rate (< 95% threshold)
            low_succ_traces: list[dict[str, float | str | bool]] = [
                {"workload_id": "W1", "trial": i, "ccl_ms": 40.0, "success": (i % 5 != 0), "scheduler": "Candidate"}
                for i in range(1000)
            ]
            self._create_sealed_bundle(bundle_dir, low_succ_traces)

            code, msg, _ = evaluate_falsification_independent(bundle_dir)
            self.assertEqual(code, 1)
            self.assertIn("candidate success rate", msg)

    def test_04_confirmatory_pass_exits_code_0(self) -> None:
        """Confirmatory bundle with >=1000 trials passing all thresholds exits code 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "confirmatory_pass_bundle"
            fast_traces: list[dict[str, float | str | bool]] = [
                {"workload_id": "W1", "trial": i, "ccl_ms": 40.0, "success": True, "scheduler": "Candidate"}
                for i in range(1000)
            ]
            self._create_sealed_bundle(bundle_dir, fast_traces, {"mode": "confirmatory"})

            code, msg, _ = evaluate_falsification_independent(bundle_dir)
            self.assertEqual(code, 0, f"Expected exit code 0 (passed), got {code}: {msg}")

    def test_05_exploratory_mode_yields_inconclusive_code_2(self) -> None:
        """Passing run marked as exploratory mode exits code 2 (inconclusive for confirmatory claims)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "exploratory_bundle"
            fast_traces: list[dict[str, float | str | bool]] = [
                {"workload_id": "W1", "trial": i, "ccl_ms": 40.0, "success": True, "scheduler": "Candidate"}
                for i in range(1000)
            ]
            self._create_sealed_bundle(bundle_dir, fast_traces, {"mode": "exploratory"})

            code, msg, _ = evaluate_falsification_independent(bundle_dir)
            self.assertEqual(code, 2, f"Expected exit code 2 (inconclusive), got {code}: {msg}")
            self.assertIn("Exploratory mode", msg)


if __name__ == "__main__":
    unittest.main()
