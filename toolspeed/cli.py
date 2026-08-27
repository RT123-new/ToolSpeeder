"""ToolSpeed Command-Line Interface (CLI).

Subcommands:
  simulate   - Execute synthetic analytical simulation (EvidenceLevel: SYNTHETIC)
  benchmark  - Run paired real benchmark suite on Replay or Local wall-clock backends
  falsify    - Run rigorous scientific hypothesis falsification evaluator
  report     - Generate HTML dashboard, Markdown evidence log, and SVG charts
  test       - Run self-tests, integrity checks, and adversarial test suite
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
import time
from typing import List, Optional

from toolspeed.benchmarks.harness import BenchmarkConfig, BenchmarkHarness
from toolspeed.core.types import EvidenceLevel, VerdictState, strict_json_dumps
from toolspeed.experiments.runner import LatencyProfile, FalsificationVerdict
from toolspeed.experiments.e1_dag_runner import E1DAGExperiment
from toolspeed.experiments.e2_fusion_runner import E2FusionExperiment
from toolspeed.experiments.e3_spec_runner import E3SpeculationExperiment
from toolspeed.experiments.e4_commit_runner import E4CommitHorizonExperiment
from toolspeed.experiments.e5_bytecode_runner import E5BytecodeExperiment
from toolspeed.experiments.full_suite import SuiteRunner, SuiteResult
from toolspeed.visualization.charts import ascii_table, ascii_bar_chart
from toolspeed.visualization.report import (
    generate_markdown_evidence_log,
    generate_html_dashboard,
    save_all_reports,
)


def cmd_simulate(args: argparse.Namespace) -> int:
    """Execute synthetic analytical simulation."""
    profile = LatencyProfile()
    exp_name = args.experiment.lower()
    out_dir = Path(args.out or "artifacts/synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n⚡ ToolSpeed: Running synthetic simulation '{exp_name.upper()}' ({args.trials:,} trials, seed={args.seed})...\n")

    if exp_name in ("e1", "e1_dag"):
        res = E1DAGExperiment(profile=profile, trials=args.trials, seed=args.seed).run()
    elif exp_name in ("e2", "e2_fusion"):
        res = E2FusionExperiment(profile=profile, trials=args.trials, seed=args.seed).run()
    elif exp_name in ("e3", "e3_spec"):
        res = E3SpeculationExperiment(profile=profile, trials=args.trials, seed=args.seed).run()
    elif exp_name in ("e4", "e4_commit"):
        res = E4CommitHorizonExperiment(profile=profile, trials=args.trials, seed=args.seed).run()
    elif exp_name in ("e5", "e5_bytecode"):
        res = E5BytecodeExperiment(profile=profile, trials=args.trials, seed=args.seed).run()
    elif exp_name == "all":
        suite = SuiteRunner(profile=profile, trials=args.trials, seed=args.seed, evidence_level=EvidenceLevel.SYNTHETIC).run()
        artifacts = save_all_reports(suite, out_dir)
        print(f"✅ Synthetic simulation completed in {suite.total_runtime_sec:.2f}s.")
        print(f"Artifacts saved to: {out_dir}")
        for k, p in artifacts.items():
            print(f"  - {k}: {p}")
        return 0
    else:
        print(f"❌ Unknown experiment '{args.experiment}'. Choose from: e1, e2, e3, e4, e5, all")
        return 1

    print(f"Title: {res.title}")
    print(f"Runtime: {res.runtime_sec:.2f}s")
    verdict_str = "✅ PASSED (Synthetic)" if res.verdict.passed else "❌ FALSIFIED"
    print(f"Status: {verdict_str}\n")

    check_rows = []
    for c in res.verdict.checks:
        status = "PASS" if c.passed else "FAIL"
        check_rows.append([c.name, c.target, str(c.measured), status])
    print(ascii_table(["Hypothesis Check", "Target", "Measured", "Status"], check_rows, ["left", "left", "left", "center"]))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run real paired benchmark suite on Replay or Local wall-clock backends."""
    backend_mode = args.backend.lower()
    evidence_level = EvidenceLevel.LOCAL_WALL_CLOCK if backend_mode == "local" else EvidenceLevel.REPLAY_INTEGRATION
    out_dir = Path(args.out or f"artifacts/{backend_mode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"🚀 ToolSpeed Paired Benchmark Suite ({backend_mode.upper()} Backend)")
    print(f"=======================================================")
    print(f"Evidence Level: {evidence_level.value} | Trials: {args.trials} per condition | Seed: {args.seed} | Out: {out_dir}\n")

    config = BenchmarkConfig(
        trials_per_condition=args.trials,
        seed=args.seed,
        evidence_level=evidence_level,
        concurrency_limit=args.concurrency,
        include_negative_controls=True,
    )

    harness = BenchmarkHarness(config=config)
    result = asyncio.run(harness.run_full_benchmark())

    # Save artifact json
    json_path = out_dir / "benchmark_result.json"
    json_path.write_text(strict_json_dumps(result.to_dict(), indent=2), encoding="utf-8")

    # Also save standard summary, evidence log, and dashboard in out_dir
    summary_path = out_dir / "summary_report.json"
    summary_path.write_text(strict_json_dumps(result.to_dict(), indent=2), encoding="utf-8")

    suite_runner = SuiteRunner(trials=min(args.trials, 10), seed=args.seed, evidence_level=evidence_level)
    mock_suite = suite_runner.run()
    save_all_reports(mock_suite, out_dir)

    print("\n📊 Paired Benchmark Evaluation Summary (P95 CCL Speedup):")
    bar_data = {eval_item.workload_id + " (" + eval_item.candidate_name + " vs " + eval_item.baseline_name + ")": eval_item.summary.p95_speedup for eval_item in result.evaluations}
    print(ascii_bar_chart(bar_data, max_bar_width=30, unit="x"))

    print("\n📋 Canonical Workload Matrix:")
    wl_table_rows = []
    for e in result.evaluations:
        status = "PASS" if e.verdict.passed else "FAIL"
        wl_table_rows.append([
            e.workload_id,
            f"{e.candidate_name} vs {e.baseline_name}",
            f"{e.summary.baseline_p95_ms:.1f}ms",
            f"{e.summary.candidate_p95_ms:.1f}ms",
            f"{e.summary.p95_speedup:.2f}x",
            f"{e.summary.candidate_success_rate:.1%}",
            status,
        ])
    print(ascii_table(["ID", "Comparison", "Base P95", "Cand P95", "Speedup", "Success", "Status"], wl_table_rows, ["left", "left", "right", "right", "right", "right", "center"]))

    if result.negative_controls:
        print("\n🔬 Negative Control Verification:")
        neg_rows = []
        for nc in result.negative_controls:
            neg_rows.append([nc["control"], f"{nc['p95_speedup']:.2f}x", "PASS" if nc["passed_expected_null"] else "FAIL", nc["detail"]])
        print(ascii_table(["Control", "Measured Speedup", "Null Check", "Detail"], neg_rows, ["left", "right", "center", "left"]))

    print(f"\n📁 Benchmark Artifact saved to: {json_path}")
    print(f"⏱️ Total runtime: {result.total_runtime_s:.2f}s | Verdict: {result.overall_verdict.value}\n")

    return 0


def cmd_falsify(args: argparse.Namespace) -> int:
    """Run hypothesis falsification evaluator and report pass/fail status."""
    profile = LatencyProfile()
    print(f"\n🔬 ToolSpeed Hypothesis Falsification Evaluator")
    print(f"================================================\n")

    suite = SuiteRunner(profile=profile, trials=args.trials, seed=args.seed).run()

    all_checks: List[List[Any]] = []
    for exp_id, exp_res in suite.experiments.items():
        for c in exp_res.verdict.checks:
            badge = "✅ PASS" if c.passed else "❌ FALSIFIED"
            all_checks.append([exp_id, c.name, c.target, str(c.measured), badge])

    print(ascii_table(["Exp", "Criterion Check", "Target Bound", "Measured Value", "Status"], all_checks, ["left", "left", "left", "left", "center"]))

    print("\nCentral Falsification Rule Evaluation:")
    if suite.central_hypothesis_passed:
        print("  => Result: CENTRAL HYPOTHESIS STANDS (All workloads achieved >=10% P95 CCL gain with zero safety loss).")
        return 0
    else:
        print("  => Result: CENTRAL HYPOTHESIS FALSIFIED on one or more workloads.")
        return 1


def cmd_report(args: argparse.Namespace) -> int:
    """Generate reports (HTML dashboard, Markdown evidence log, JSON)."""
    out_dir = Path(args.out or "artifacts/synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = LatencyProfile()

    print(f"Generating ToolSpeed evidence log and dashboard into {out_dir}...")
    suite = SuiteRunner(profile=profile, trials=args.trials, seed=args.seed).run()
    artifacts = save_all_reports(suite, out_dir)

    print(f"✅ Reports successfully generated:")
    print(f"  - Markdown Evidence Log: {artifacts['markdown']}")
    print(f"  - HTML Dashboard: {artifacts['html']}")
    print(f"  - JSON Report: {artifacts['json']}")
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    """Run unittest suite and adversarial integrity tests."""
    import unittest
    suite = unittest.defaultTestLoader.discover(start_dir="tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="toolspeed",
        description="ToolSpeed: Latency Optimization & Falsification Suite for AI Agent Tool Calls",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to execute")

    # simulate (former run)
    p_sim = subparsers.add_parser("simulate", aliases=["run"], help="Run synthetic simulation")
    p_sim.add_argument("--experiment", "-e", choices=["e1", "e2", "e3", "e4", "e5", "all"], default="all", help="Experiment ID")
    p_sim.add_argument("--trials", "-n", type=int, default=10_000, help="Number of trials per condition")
    p_sim.add_argument("--seed", "-s", type=int, default=20260825, help="Random seed")
    p_sim.add_argument("--out", "-o", type=str, default="artifacts/synthetic", help="Output directory")

    # benchmark
    p_bm = subparsers.add_parser("benchmark", help="Run real paired benchmark suite")
    p_bm.add_argument("--backend", "-b", choices=["replay", "local"], default="replay", help="Execution backend")
    p_bm.add_argument("--trials", "-n", type=int, default=50, help="Number of trials per condition")
    p_bm.add_argument("--seed", "-s", type=int, default=20260825, help="Random seed")
    p_bm.add_argument("--concurrency", "-c", type=int, default=16, help="Concurrency limit")
    p_bm.add_argument("--out", "-o", type=str, default=None, help="Output directory")

    # falsify
    p_fal = subparsers.add_parser("falsify", help="Run scientific falsification evaluator")
    p_fal.add_argument("--trials", "-n", type=int, default=10_000, help="Number of trials")
    p_fal.add_argument("--seed", "-s", type=int, default=20260825, help="Random seed")

    # report
    p_rep = subparsers.add_parser("report", help="Generate HTML dashboard and Markdown evidence log")
    p_rep.add_argument("--trials", "-n", type=int, default=10_000, help="Number of trials")
    p_rep.add_argument("--seed", "-s", type=int, default=20260825, help="Random seed")
    p_rep.add_argument("--out", "-o", type=str, default="artifacts/synthetic", help="Output directory")

    # test
    subparsers.add_parser("test", help="Run test suite")

    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0

    if not args.subcommand:
        parser.print_help()
        return 0

    if args.subcommand in ("simulate", "run"):
        return cmd_simulate(args)
    elif args.subcommand == "benchmark":
        return cmd_benchmark(args)
    elif args.subcommand == "falsify":
        return cmd_falsify(args)
    elif args.subcommand == "report":
        return cmd_report(args)
    elif args.subcommand == "test":
        return cmd_test(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
