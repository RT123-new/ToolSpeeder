"""ToolSpeed Command-Line Interface (CLI).

Subcommands:
  run        - Execute an individual experiment or all experiments
  benchmark  - Run comprehensive W1-W7 benchmark suite and generate full artifacts
  falsify    - Run rigorous scientific hypothesis falsification evaluator
  report     - Generate HTML dashboard, Markdown evidence log, and SVG charts
  test       - Run self-tests and diagnostic checks
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import List, Optional

from toolspeed.experiments.runner import LatencyProfile, FalsificationVerdict
from toolspeed.experiments.e1_dag_runner import E1DAGExperiment
from toolspeed.experiments.e2_fusion_runner import E2FusionExperiment
from toolspeed.experiments.e3_spec_runner import E3SpeculationExperiment
from toolspeed.experiments.e4_commit_runner import E4CommitHorizonExperiment
from toolspeed.experiments.e5_bytecode_runner import E5BytecodeExperiment
from toolspeed.experiments.full_suite import SuiteRunner, SuiteResult
from toolspeed.visualization.charts import ascii_table, ascii_bar_chart, ascii_sparkline
from toolspeed.visualization.report import (
    generate_markdown_evidence_log,
    generate_html_dashboard,
    save_all_reports,
)


def cmd_run(args: argparse.Namespace) -> int:
    """Execute individual experiment or suite."""
    profile = LatencyProfile()
    exp_name = args.experiment.lower()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n⚡ ToolSpeed: Running experiment '{exp_name.upper()}' ({args.trials:,} trials, seed={args.seed})...\n")

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
        suite = SuiteRunner(profile=profile, trials=args.trials, seed=args.seed).run()
        artifacts = save_all_reports(suite, out_dir)
        print(f"✅ All experiments completed in {suite.total_runtime_sec:.2f}s.")
        print(f"Artifacts saved to: {out_dir}")
        for k, p in artifacts.items():
            print(f"  - {k}: {p}")
        return 0
    else:
        print(f"❌ Unknown experiment '{args.experiment}'. Choose from: e1, e2, e3, e4, e5, all")
        return 1

    # Display results
    print(f"Title: {res.title}")
    print(f"Runtime: {res.runtime_sec:.2f}s")
    verdict_str = "✅ PASSED" if res.verdict.passed else "❌ FALSIFIED / FAILED"
    print(f"Status: {verdict_str}\n")

    # Table of checks
    check_rows = []
    for c in res.verdict.checks:
        status = "PASS" if c.passed else "FAIL"
        check_rows.append([c.name, c.target, str(c.measured), status])
    print(ascii_table(["Hypothesis Check", "Target", "Measured", "Status"], check_rows, ["left", "left", "left", "center"]))

    # Table of sample parameter rows
    if res.rows:
        print(f"\nParameter Sweep ({res.parameter_name}):")
        param_rows = []
        for r in res.rows[:8]:
            p_val = r.get(res.parameter_name)
            p50_spd = f"{r.get('p50_speedup', 1.0):.2f}x"
            p95_spd = f"{r.get('p95_speedup', 1.0):.2f}x"
            p95_ms = f"{r.get('candidate_p95_ms', 0.0):.1f}ms"
            param_rows.append([str(p_val), p50_spd, p95_spd, p95_ms])
        print(ascii_table([res.parameter_name, "P50 Speedup", "P95 Speedup", "Candidate P95"], param_rows, ["left", "right", "right", "right"]))

    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run full benchmark across workloads W1-W7 and experiments E1-E5."""
    profile = LatencyProfile()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"🚀 ToolSpeed Full Benchmark Suite (W1-W7 x E1-E5)")
    print(f"=======================================================")
    print(f"Trials: {args.trials:,} per condition | Seed: {args.seed} | Out: {out_dir}\n")

    runner = SuiteRunner(profile=profile, trials=args.trials, seed=args.seed)
    suite = runner.run()

    artifacts = save_all_reports(suite, out_dir)

    print("📊 Workload Latency Benchmark Summary (P95 CCL Speedup):")
    bar_data = {w.workload_id + " (" + w.name.split(":")[1].strip() + ")": w.p95_speedup for w in suite.workloads.values()}
    print(ascii_bar_chart(bar_data, max_bar_width=30, unit="x"))

    print("\n📋 Canonical Workload Matrix:")
    wl_table_rows = []
    for w in suite.workloads.values():
        status = "PASS" if w.central_hypothesis_passed else "FAIL"
        wl_table_rows.append([
            w.workload_id,
            w.name.split(":")[1].strip(),
            f"{w.baseline_p95_ms:.1f}ms",
            f"{w.candidate_p95_ms:.1f}ms",
            f"{w.p95_speedup:.2f}x",
            f"{w.p95_reduction_pct:.1f}%",
            status,
        ])
    print(ascii_table(["ID", "Workload Family", "Base P95", "Cand P95", "Speedup", "CCL Gain", "Status"], wl_table_rows, ["left", "left", "right", "right", "right", "right", "center"]))

    print(f"\n📁 Generated Artifacts:")
    for k, p in artifacts.items():
        print(f"  • {k}: {p}")

    print(f"\n⏱️ Completed in {suite.total_runtime_sec:.2f}s.\n")
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
    print("  'The central hypothesis is wrong for a workload when no tested mechanism improves P95 CCL by at least 10%.'")
    if suite.central_hypothesis_passed:
        print("  => Result: CENTRAL HYPOTHESIS STANDS (All workloads achieved >=10% P95 CCL gain with zero safety loss).")
        return 0
    else:
        print("  => Result: CENTRAL HYPOTHESIS FALSIFIED on one or more workloads.")
        return 1


def cmd_report(args: argparse.Namespace) -> int:
    """Generate reports (HTML dashboard, Markdown evidence log, JSON)."""
    out_dir = Path(args.out)
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
    """Run internal test runner."""
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

    # run
    p_run = subparsers.add_parser("run", help="Run an experiment")
    p_run.add_argument("--experiment", "-e", choices=["e1", "e2", "e3", "e4", "e5", "all"], default="all", help="Experiment ID")
    p_run.add_argument("--trials", "-n", type=int, default=10_000, help="Number of trials per condition")
    p_run.add_argument("--seed", "-s", type=int, default=20260825, help="Random seed")
    p_run.add_argument("--out", "-o", type=str, default="results", help="Output directory")

    # benchmark
    p_bm = subparsers.add_parser("benchmark", help="Run comprehensive W1-W7 benchmark suite")
    p_bm.add_argument("--trials", "-n", type=int, default=10_000, help="Number of trials")
    p_bm.add_argument("--seed", "-s", type=int, default=20260825, help="Random seed")
    p_bm.add_argument("--out", "-o", type=str, default="results", help="Output directory")

    # falsify
    p_fal = subparsers.add_parser("falsify", help="Run scientific falsification evaluator")
    p_fal.add_argument("--trials", "-n", type=int, default=10_000, help="Number of trials")
    p_fal.add_argument("--seed", "-s", type=int, default=20260825, help="Random seed")

    # report
    p_rep = subparsers.add_parser("report", help="Generate HTML dashboard and Markdown evidence log")
    p_rep.add_argument("--trials", "-n", type=int, default=10_000, help="Number of trials")
    p_rep.add_argument("--seed", "-s", type=int, default=20260825, help="Random seed")
    p_rep.add_argument("--out", "-o", type=str, default="results", help="Output directory")

    # test
    subparsers.add_parser("test", help="Run test suite")

    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0

    if not args.subcommand:
        parser.print_help()
        return 0

    if args.subcommand == "run":
        return cmd_run(args)
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
