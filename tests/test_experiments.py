"""Comprehensive Unit and Integration Tests for ToolSpeed."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from toolspeed.experiments.runner import (
    LatencyProfile,
    MetricSummary,
    ExperimentResult,
    FalsificationVerdict,
    HypothesisCheck,
    WorkloadFamily,
    compute_summary,
    samples,
    compute_percentiles,
    bootstrap_confidence_interval,
)
from toolspeed.experiments.e1_dag_runner import E1DAGExperiment, run_e1_experiment
from toolspeed.experiments.e2_fusion_runner import E2FusionExperiment, run_e2_experiment
from toolspeed.experiments.e3_spec_runner import E3SpeculationExperiment, run_e3_experiment
from toolspeed.experiments.e4_commit_runner import E4CommitHorizonExperiment, run_e4_experiment
from toolspeed.experiments.e5_bytecode_runner import E5BytecodeExperiment, run_e5_experiment
from toolspeed.experiments.full_suite import SuiteRunner, SuiteResult, run_full_suite
from toolspeed.visualization.charts import (
    generate_speedup_line_chart,
    generate_cdf_chart,
    generate_workload_bar_chart,
    ascii_sparkline,
    ascii_bar_chart,
    ascii_table,
)
from toolspeed.visualization.report import (
    generate_markdown_evidence_log,
    generate_html_dashboard,
    generate_json_summary,
    save_all_reports,
)
import toolspeed.cli as cli


class TestLatencyProfileAndRunner(unittest.TestCase):
    """Test core runner utilities and statistical computations."""

    def test_latency_profile_defaults(self):
        profile = LatencyProfile()
        self.assertEqual(profile.model_decision_ms, 450.0)
        self.assertEqual(profile.tool_ms, 600.0)
        self.assertEqual(profile.draft_model_ms, 70.0)
        d = profile.to_dict()
        self.assertIn("model_decision_ms", d)

    def test_samples_generation(self):
        rng = np.random.default_rng(42)
        arr = samples(rng, median_ms=500.0, sigma=0.4, shape=(1000,))
        self.assertEqual(arr.shape, (1000,))
        self.assertTrue(np.all(arr > 0))
        # Median should be reasonably close to 500.0
        med = float(np.median(arr))
        self.assertGreater(med, 400.0)
        self.assertLess(med, 600.0)

    def test_percentiles(self):
        data = np.arange(1, 101, dtype=float)
        p = compute_percentiles(data, (50, 90, 95, 99))
        self.assertAlmostEqual(p["p50"], 50.5, delta=1.0)
        self.assertAlmostEqual(p["p90"], 90.1, delta=1.0)
        self.assertAlmostEqual(p["p95"], 95.05, delta=1.0)
        self.assertAlmostEqual(p["p99"], 99.01, delta=1.0)

    def test_compute_summary_metrics(self):
        baseline = np.full(100, 1000.0)
        candidate = np.full(100, 500.0)
        summary = compute_summary(
            baseline=baseline,
            candidate=candidate,
            baseline_success=np.ones(100, dtype=bool),
            candidate_success=np.ones(100, dtype=bool),
            input_tokens_base=1000.0,
            input_tokens_cand=500.0,
        )
        self.assertEqual(summary.baseline_p50_ms, 1000.0)
        self.assertEqual(summary.candidate_p50_ms, 500.0)
        self.assertEqual(summary.p50_speedup, 2.0)
        self.assertEqual(summary.p95_speedup, 2.0)
        self.assertEqual(summary.token_reduction_pct, 50.0)
        self.assertEqual(summary.baseline_success_rate, 1.0)
        self.assertEqual(summary.candidate_success_rate, 1.0)

    def test_bootstrap_confidence_interval(self):
        data = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
        low, high = bootstrap_confidence_interval(data, np.mean, num_samples=200, ci=0.95, seed=42)
        self.assertLess(low, np.mean(data))
        self.assertGreater(high, np.mean(data))


class TestE1DAGExperiment(unittest.TestCase):
    """Test E1 DAG Parallelism runner and hypothesis evaluation."""

    def test_e1_run(self):
        exp = E1DAGExperiment(trials=500, seed=123)
        res = exp.run(fanouts=(2, 4, 8), concurrency_limit=4, test_false_independence=True)

        self.assertEqual(res.experiment_id, "E1_DAG")
        self.assertGreater(len(res.rows), 0)
        self.assertTrue(res.verdict.passed)
        self.assertFalse(res.verdict.falsified)

        # Check that speedup increases with fan-out
        row_2 = res.get_row(2)
        row_4 = res.get_row(4)
        self.assertIsNotNone(row_2)
        self.assertIsNotNone(row_4)
        self.assertGreater(row_4["p95_speedup"], row_2["p95_speedup"])
        self.assertGreaterEqual(row_4["p95_reduction_pct"], 20.0)

        # Check evidence log row
        self.assertIn("experiment", res.verdict.evidence_log_row)
        self.assertEqual(res.verdict.evidence_log_row["tested"], "Yes")

    def test_e1_convenience_function(self):
        res = run_e1_experiment(trials=200, seed=42)
        self.assertEqual(res.experiment_id, "E1_DAG")
        self.assertGreater(res.runtime_sec, 0.0)


class TestE2FusionExperiment(unittest.TestCase):
    """Test E2 Programmatic Workflow Fusion runner."""

    def test_e2_run(self):
        exp = E2FusionExperiment(trials=500, seed=123)
        res = exp.run(step_counts=(2, 4, 8), deopt_rates=(0.0, 0.05, 0.15))

        self.assertEqual(res.experiment_id, "E2_FUSION")
        self.assertTrue(res.verdict.passed)

        row_4 = res.get_row(4)
        self.assertIsNotNone(row_4)
        self.assertGreaterEqual(row_4["p95_reduction_pct"], 25.0)
        self.assertGreaterEqual(row_4["input_token_reduction_pct"], 20.0)
        self.assertLessEqual(row_4["deopt_rate"], 0.15)

    def test_e2_convenience_function(self):
        res = run_e2_experiment(trials=200, seed=42)
        self.assertEqual(res.experiment_id, "E2_FUSION")


class TestE3SpeculationExperiment(unittest.TestCase):
    """Test E3 Speculative Reads runner and contention modes."""

    def test_e3_run(self):
        exp = E3SpeculationExperiment(trials=500, seed=123)
        res = exp.run(
            accuracies=np.array([0.0, 0.5, 0.85, 1.0]),
            contention_modes=("no_contention", "cancellable", "single_slot"),
        )

        self.assertEqual(res.experiment_id, "E3_SPECULATION")
        self.assertTrue(res.verdict.passed)

        # Verify gated row exists
        gated_row = next((r for r in res.rows if r.get("contention_mode") == "confidence_gated"), None)
        self.assertIsNotNone(gated_row)
        self.assertLess(gated_row["wasted_call_rate"], 0.20)
        self.assertLess(gated_row["mean_tool_cost_multiplier"], 1.05)

    def test_e3_convenience_function(self):
        res = run_e3_experiment(trials=200, seed=42)
        self.assertEqual(res.experiment_id, "E3_SPECULATION")


class TestE4CommitHorizonExperiment(unittest.TestCase):
    """Test E4 Commit-Horizon Early Dispatch runner."""

    def test_e4_run(self):
        exp = E4CommitHorizonExperiment(trials=500, seed=123)
        res = exp.run(commit_fractions=(0.2, 0.4, 0.6, 0.8, 1.0), target_fraction=0.4)

        self.assertEqual(res.experiment_id, "E4_COMMIT_HORIZON")
        self.assertTrue(res.verdict.passed)

        row_04 = next((r for r in res.rows if abs(r["commit_fraction"] - 0.4) < 0.05), None)
        self.assertIsNotNone(row_04)
        self.assertGreaterEqual(row_04["tool_start_p95_reduction_pct"], 10.0)
        self.assertEqual(row_04["semantic_mutations"], 0)

    def test_e4_convenience_function(self):
        res = run_e4_experiment(trials=200, seed=42)
        self.assertEqual(res.experiment_id, "E4_COMMIT_HORIZON")


class TestE5BytecodeExperiment(unittest.TestCase):
    """Test E5 Action Bytecode runner."""

    def test_e5_run(self):
        exp = E5BytecodeExperiment(trials=500, seed=123)
        res = exp.run(
            decode_shares=(0.25, 0.50, 0.80),
            acceleration_factors=(2.0, 4.0),
            expansion_overhead_ms=3.0,
        )

        self.assertEqual(res.experiment_id, "E5_BYTECODE")
        self.assertTrue(res.verdict.passed)

        # Check decode heavy row
        row_heavy = next(
            (r for r in res.rows if r["tool_call_decode_share"] == 0.80 and r["decode_acceleration_factor"] == 4.0),
            None,
        )
        self.assertIsNotNone(row_heavy)
        self.assertGreater(row_heavy["p95_speedup"], 1.15)

    def test_e5_convenience_function(self):
        res = run_e5_experiment(trials=200, seed=42)
        self.assertEqual(res.experiment_id, "E5_BYTECODE")


class TestSuiteRunnerAndWorkloads(unittest.TestCase):
    """Test Full Suite Runner across W1-W7."""

    def test_suite_run(self):
        runner = SuiteRunner(trials=300, seed=42)
        suite = runner.run()

        self.assertEqual(len(suite.experiments), 5)
        self.assertIn("E1", suite.experiments)
        self.assertIn("E2", suite.experiments)
        self.assertIn("E3", suite.experiments)
        self.assertIn("E4", suite.experiments)
        self.assertIn("E5", suite.experiments)

        self.assertEqual(len(suite.workloads), 7)
        for i in range(1, 8):
            self.assertIn(f"W{i}", suite.workloads)
            w = suite.workloads[f"W{i}"]
            self.assertGreaterEqual(w.p95_speedup, 1.0)
            self.assertTrue(w.central_hypothesis_passed)

        self.assertTrue(suite.central_hypothesis_passed)
        self.assertEqual(len(suite.evidence_log), 6)  # Phase 0 + E1..E5

    def test_suite_save_and_export(self):
        runner = SuiteRunner(trials=200, seed=42)
        suite = runner.run()

        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "test_out.json"
            suite.save_json(json_file)
            self.assertTrue(json_file.exists())

            loaded = json.loads(json_file.read_text(encoding="utf-8"))
            self.assertIn("experiments", loaded)
            self.assertIn("workloads", loaded)
            self.assertIn("evidence_log", loaded)

            csvs = suite.save_csvs(tmpdir)
            self.assertGreater(len(csvs), 0)
            for csv in csvs:
                self.assertTrue(csv.exists())
                self.assertGreater(csv.stat().st_size, 0)


class TestVisualization(unittest.TestCase):
    """Test SVG and ASCII chart generators and report formats."""

    def test_speedup_line_chart_svg(self):
        series = {
            "Mode A": [(1.0, 1.2), (2.0, 1.8), (3.0, 2.4)],
            "Mode B": [(1.0, 1.0), (2.0, 1.3), (3.0, 1.5)],
        }
        svg = generate_speedup_line_chart("Test Chart", "X Axis", "Y Axis", series)
        self.assertIn("<svg", svg)
        self.assertIn("</svg>", svg)
        self.assertIn("Test Chart", svg)
        self.assertIn("Mode A", svg)

    def test_cdf_chart_svg(self):
        datasets = {
            "Baseline": [400, 500, 600, 700, 800],
            "Candidate": [200, 250, 300, 350, 400],
        }
        svg = generate_cdf_chart("Latency CDF", datasets)
        self.assertIn("<svg", svg)
        self.assertIn("Latency CDF", svg)
        self.assertIn("Baseline", svg)

    def test_workload_bar_chart_svg(self):
        workloads = [
            {"workload_id": "W1", "name": "Fanout", "p95_speedup": 2.5, "p95_reduction_pct": 60.0},
            {"workload_id": "W2", "name": "Chains", "p95_speedup": 1.8, "p95_reduction_pct": 44.0},
        ]
        svg = generate_workload_bar_chart(workloads)
        self.assertIn("<svg", svg)
        self.assertIn("W1", svg)
        self.assertIn("2.50x", svg)

    def test_ascii_sparkline(self):
        spark = ascii_sparkline([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 2.0, 1.0])
        self.assertEqual(len(spark), 8)
        self.assertIn("█", spark)
        self.assertIn(" ", spark)

    def test_ascii_bar_chart(self):
        chart = ascii_bar_chart({"W1": 2.5, "W2": 1.5})
        self.assertIn("W1", chart)
        self.assertIn("2.50", chart)
        self.assertIn("█", chart)

    def test_ascii_table(self):
        table = ascii_table(["Col A", "Col B"], [["1", "2"], ["3", "4"]])
        self.assertIn("Col A", table)
        self.assertIn("Col B", table)
        self.assertIn("│", table)
        self.assertIn("┌", table)

    def test_markdown_and_html_reports(self):
        runner = SuiteRunner(trials=100, seed=42)
        suite = runner.run()

        md = generate_markdown_evidence_log(suite)
        self.assertIn("# ToolSpeed — Evidence Log & Experiment Report", md)
        self.assertIn("| Experiment | Tested | Succeeded | Failed | Still unproven | Next action |", md)
        self.assertIn("E1 — DAG Parallelism", md)

        html = generate_html_dashboard(suite)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("ToolSpeed Latency & Falsification Dashboard", html)
        self.assertIn("<svg", html)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = save_all_reports(suite, tmpdir)
            self.assertIn("markdown", artifacts)
            self.assertIn("html", artifacts)
            self.assertIn("json", artifacts)
            self.assertTrue(artifacts["markdown"].exists())
            self.assertTrue(artifacts["html"].exists())
            self.assertTrue(artifacts["json"].exists())


class TestCLI(unittest.TestCase):
    """Test CLI commands and argument parser."""

    def test_cli_help(self):
        ret = cli.main(["--help"])
        self.assertEqual(ret, 0)

    def test_cli_run_e1(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ret = cli.main(["simulate", "--experiment", "e1", "--trials", "50", "--out", tmpdir])
            self.assertEqual(ret, 0)

    def test_cli_falsify(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ret_bm = cli.main(["benchmark", "--backend", "replay", "--trials", "2", "--out", tmpdir])
            self.assertEqual(ret_bm, 0)
            ret = cli.main(["falsify", "--input", tmpdir])
            self.assertIn(ret, (0, 2))

    def test_cli_benchmark(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ret = cli.main(["benchmark", "--backend", "replay", "--trials", "2", "--out", tmpdir])
            self.assertEqual(ret, 0)
            self.assertTrue((Path(tmpdir) / "benchmark_result.json").exists())
            self.assertTrue((Path(tmpdir) / "report.md").exists())
            self.assertTrue((Path(tmpdir) / "report.html").exists())

    def test_cli_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ret_bm = cli.main(["benchmark", "--backend", "replay", "--trials", "2", "--out", tmpdir])
            self.assertEqual(ret_bm, 0)
            ret = cli.main(["report", "--input", tmpdir, "--out", tmpdir])
            self.assertEqual(ret, 0)
            self.assertTrue((Path(tmpdir) / "report.html").exists())


if __name__ == "__main__":
    unittest.main()
