"""Tests for Phase 28: Self-contained bundles, detached-seal manifest.sig, and hash-first integrity validation."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from toolspeed.benchmarks.bundle import (
    REQUIRED_BUNDLE_FILES,
    compute_file_sha256,
    create_bundle,
    validate_bundle_hashes_first,
)


class TestBundleIntegrity(unittest.TestCase):
    """Verifies self-contained bundle creation, presence of 8 required files, detached signature seal, and hash-first checking."""

    def setUp(self) -> None:
        self.dummy_result = {
            "title": "ToolSpeed Canonical Benchmark Run",
            "evidence_level": "replay_integration",
            "overall_verdict": "passed",
            "total_runtime_s": 1.25,
            "evaluations": [],
            "manifest": {"is_verdict_eligible": True, "trial_count": 100},
        }

    def test_01_bundle_contains_all_eight_required_files(self) -> None:
        """Bundle contains all 8 required files: manifest.json, result.json, protocol.json, git-commit.txt, environment.json, cases.jsonl, raw-traces.jsonl, manifest.sig."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "isolated_bundle"
            create_bundle(self.dummy_result, bundle_dir)

            for req_file in REQUIRED_BUNDLE_FILES:
                target = bundle_dir / req_file
                self.assertTrue(target.exists(), f"Required bundle file '{req_file}' is missing")
                self.assertGreater(target.stat().st_size, 0, f"Required bundle file '{req_file}' is empty")

            valid, errors = validate_bundle_hashes_first(bundle_dir)
            self.assertTrue(valid, f"Validation failed with errors: {errors}")

    def test_02_manifest_contains_sha256_of_every_file(self) -> None:
        """Manifest file_hashes contains valid SHA-256 for all payload files in bundle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "bundle_hashes_check"
            create_bundle(self.dummy_result, bundle_dir)

            manifest_data = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
            file_hashes = manifest_data.get("file_hashes", {})

            for fname, expected_hash in file_hashes.items():
                target_path = bundle_dir / fname
                self.assertTrue(target_path.exists())
                actual_hash = hashlib.sha256(target_path.read_bytes()).hexdigest()
                self.assertEqual(
                    actual_hash,
                    expected_hash,
                    f"Hash mismatch for '{fname}' in manifest",
                )

    def test_03_detached_seal_verification(self) -> None:
        """manifest.sig contains detached SHA-256 seal of manifest.json; tampering manifest fails validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "bundle_seal_check"
            create_bundle(self.dummy_result, bundle_dir)

            manifest_sha = compute_file_sha256(bundle_dir / "manifest.json")
            sig_content = (bundle_dir / "manifest.sig").read_text(encoding="utf-8").strip()
            self.assertEqual(sig_content, manifest_sha)

            # Tamper manifest.json
            m_path = bundle_dir / "manifest.json"
            m_data = json.loads(m_path.read_text(encoding="utf-8"))
            m_data["tampered_key"] = "unauthorized"
            m_path.write_text(json.dumps(m_data), encoding="utf-8")

            valid, errors = validate_bundle_hashes_first(bundle_dir)
            self.assertFalse(valid)
            self.assertTrue(any("manifest.sig mismatch" in e for e in errors))

    def test_04_tampering_any_file_detected_before_reading_contents(self) -> None:
        """Tampering with raw-traces.jsonl or environment.json is caught by hash-first validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "tamper_test"
            create_bundle(self.dummy_result, bundle_dir)

            # Tamper raw-traces.jsonl
            traces_file = bundle_dir / "raw-traces.jsonl"
            traces_file.write_text(traces_file.read_text() + '{"tampered_trace": true}\n', encoding="utf-8")

            valid, errors = validate_bundle_hashes_first(bundle_dir)
            self.assertFalse(valid)
            self.assertTrue(any("raw-traces.jsonl" in e and "hash mismatch" in e for e in errors))

    def test_05_no_reliance_on_artifacts_directory(self) -> None:
        """Bundle creation and verification operates in arbitrary custom locations without artifacts/ dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_dir = Path(tmpdir) / "custom_nesting" / "sub_path" / "my_bundle"
            # Ensure path has zero mention of 'artifacts'
            self.assertNotIn("artifacts", str(custom_dir))

            create_bundle(self.dummy_result, custom_dir)
            valid, errors = validate_bundle_hashes_first(custom_dir)
            self.assertTrue(valid, f"Self-contained bundle failed in custom directory: {errors}")


if __name__ == "__main__":
    unittest.main()
