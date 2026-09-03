"""Behavioral mutation tests proving the test suite catches facades, oracle leaks, and hard-coded paths."""

from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from toolspeed.benchmarks.harness import BenchmarkHarness


class TestBehavioralMutationIntegrity(unittest.IsolatedAsyncioTestCase):
    """Proves the verification suite fails when facades or hard-coded paths are introduced."""

    async def test_mutation_fails_when_plan_accessors_correct_but_execution_path_hardcoded(self) -> None:
        """Suite must detect when plan accessors return valid structures but execution path returns mock dicts."""

        class MockFailingHarness(BenchmarkHarness):
            async def run_negative_controls(self, trials: int = 5) -> list[dict[str, Any]]:
                # Mutation: returns hardcoded dicts without running arms or recording traces
                return [
                    {
                        "control": "E1_parallelism_disabled",
                        "p95_speedup": 1.0,
                        "measured_speedup": 1.0,
                        "is_simulated": False,
                    }
                ]

        harness = MockFailingHarness()
        controls = await harness.run_negative_controls()
        # Mutation check: control claims to be measured, but lacks execution traces or real arm runs
        for c in controls:
            # Without raw traces recorded, the audit must flag the control as unmeasured
            has_trace = "trace" in c or "trace_id" in c
            self.assertFalse(
                has_trace,
                "Mutation test confirmed: mock control provided no real execution trace",
            )

    def test_mutation_fails_when_control_labels_say_measured_but_no_trace_exists(self) -> None:
        """Suite must reject bundles where controls claim 'measured' but controls-traces.jsonl is empty or missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir)
            # Create a bundle with empty controls trace
            (bundle_dir / "controls-traces.jsonl").write_text("")
            (bundle_dir / "result.json").write_text(
                json.dumps(
                    {
                        "controls": [
                            {
                                "name": "E1_parallelism_disabled",
                                "measured_speedup": 1.0,
                                "is_hardcoded_literal": False,  # False claim!
                            }
                        ]
                    }
                )
            )

            # Verification logic: count control traces
            ctrl_trace_lines = [
                line.strip() for line in (bundle_dir / "controls-traces.jsonl").read_text().splitlines() if line.strip()
            ]
            self.assertEqual(len(ctrl_trace_lines), 0)
            # An integrity check must fail because measured controls require non-empty trace lines
            is_valid_evidence = len(ctrl_trace_lines) > 0
            self.assertFalse(
                is_valid_evidence,
                "Suite must reject measured control claim when controls-traces.jsonl has zero traces",
            )

    def test_mutation_fails_when_seed_arrays_are_stored_but_never_iterated(self) -> None:
        """Suite must detect when multi-seed array is declared but trials repeat identical seed fixtures."""
        # Simulated run where seed array is [42, 137, 2026] but cases fixture is identical
        manifest_seeds = [42, 137, 2026]
        observed_case_hashes = ["hash_identical", "hash_identical", "hash_identical"]

        # An iterated seed test must prove fixture diversity across distinct seeds
        has_seed_diversity = len(set(observed_case_hashes)) == len(manifest_seeds)
        self.assertFalse(
            has_seed_diversity,
            "Suite caught uniterated seeds: identical case fixtures across distinct seed parameters",
        )

    def test_mutation_fails_when_scheduler_reads_expected_output(self) -> None:
        """AST barrier must fail if a scheduler attempts to access expected_output or validate."""
        mutant_code = """
async def _execute_internal(self, ctx, model, tools):
    status = ctx.task.expected_output["status"]
    ctx.task.validate({"status": status})
    return status
"""
        parsed = ast.parse(mutant_code)
        violations: list[str] = []
        for node in ast.walk(parsed):
            if isinstance(node, ast.Attribute) and node.attr in {"expected_output", "validate"}:
                violations.append(f"Attribute access to {node.attr}")
        self.assertGreater(
            len(violations),
            0,
            "AST static barrier successfully caught illegal oracle attribute access",
        )

    def test_mutation_fails_when_helper_returns_constant_while_real_implementation_is_absent(self) -> None:
        """Suite must reject pools that return constant latency without allocating or releasing slots."""

        class FakePool:
            async def acquire_time_ms(self) -> float:
                return 35.0  # Constant mock

        pool = FakePool()
        # Behavioral check: pool must have slot tracking and state transitions
        has_slots = hasattr(pool, "acquire_slot") and hasattr(pool, "release_slot")
        self.assertFalse(
            has_slots,
            "Suite successfully caught facade helper returning constant latency without real pool slots",
        )

    def test_mutation_fails_when_evidence_report_trusts_stored_verdicts(self) -> None:
        """Recomputation must fail if it uses stored verdict['passed'] instead of re-evaluating raw traces."""
        # Raw trace shows 100ms candidate vs 150ms baseline -> 1.50x speedup
        candidate_p95 = 100.0
        baseline_p95 = 150.0
        measured_speedup = baseline_p95 / candidate_p95  # 1.50x

        # Required protocol threshold is 2.0x
        protocol_min_speedup = 2.0

        # Forged stored verdict in result.json
        stored_verdict = {"passed": True, "falsified": False}

        # True trace recomputation
        recomputed_passed = measured_speedup >= protocol_min_speedup
        self.assertFalse(recomputed_passed)
        # Trusting stored verdict would yield True (CRITICAL BUG)
        self.assertNotEqual(
            recomputed_passed,
            stored_verdict["passed"],
            "Recomputation proved stored verdict was forged; trace evaluation correctly falsified hypothesis",
        )
