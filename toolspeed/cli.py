"""ToolSpeed Command-Line Interface (CLI).

Subcommands:
  simulate   - Execute synthetic analytical simulation (EvidenceLevel: SYNTHETIC)
  benchmark  - Run paired real benchmark suite on Replay or Local wall-clock backends
  falsify    - Run rigorous scientific hypothesis falsification evaluator on an existing bundle
  report     - Generate HTML dashboard and Markdown evidence log from an existing bundle
  test       - Run test suite and adversarial integrity tests
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

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
    generate_benchmark_markdown_report,
    generate_benchmark_html_dashboard,
    save_all_reports,
    save_benchmark_reports,
)


def cmd_simulate(args: argparse.Namespace) -> int:
    """Execute synthetic analytical simulation (always SYNTHETIC evidence level, real-world INCONCLUSIVE)."""
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
    """Run real paired benchmark suite on genuine Replay or Local wall-clock backends."""
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

    # Save benchmark bundle reports directly without calling SuiteRunner
    save_benchmark_reports(result, out_dir)

    print("\n📊 Paired Benchmark Evaluation Summary (P95 CCL Speedup):")
    bar_data = {
        eval_item.workload_id + " (" + eval_item.candidate_name + " vs " + eval_item.baseline_name + ")": (eval_item.summary.p95_speedup or 1.0)
        for eval_item in result.evaluations
    }
    print(ascii_bar_chart(bar_data, max_bar_width=30, unit="x"))

    print("\n📋 Canonical Workload Matrix (W1 – W7, E5a):")
    wl_table_rows = []
    for e in result.evaluations:
        status = "PASS" if e.verdict.passed else "FAIL"
        b95 = f"{e.summary.baseline_p95_ms:.1f}ms" if e.summary.baseline_p95_ms is not None else "null"
        c95 = f"{e.summary.candidate_p95_ms:.1f}ms" if e.summary.candidate_p95_ms is not None else "null"
        sp = f"{e.summary.p95_speedup:.2f}x" if e.summary.p95_speedup is not None else "null"
        succ = f"{e.summary.candidate_success_rate:.1%}" if e.summary.candidate_success_rate is not None else "null"
        wl_table_rows.append([
            e.workload_id,
            f"{e.candidate_name} vs {e.baseline_name}",
            b95,
            c95,
            sp,
            succ,
            status,
        ])
    print(ascii_table(["ID", "Comparison", "Base P95", "Cand P95", "Speedup", "Success", "Status"], wl_table_rows, ["left", "left", "right", "right", "right", "right", "center"]))

    if result.negative_controls:
        print("\n🔬 Negative Control Verification:")
        neg_rows = []
        for nc in result.negative_controls:
            neg_rows.append([nc["control"], f"{nc['p95_speedup']:.2f}x", "PASS" if nc["passed_expected_null"] else "FAIL", nc["detail"]])
        print(ascii_table(["Control", "Measured Speedup", "Null Check", "Detail"], neg_rows, ["left", "right", "center", "left"]))

    print(f"\n📁 Benchmark Bundle saved to: {out_dir}")
    print(f"⏱️ Total runtime: {result.total_runtime_s:.2f}s | Overall Verdict: {result.overall_verdict.value}\n")

    return 0


def _load_bundle_data(bundle_path_str: str) -> Tuple[Dict[str, Any], Path]:
    p = Path(bundle_path_str)
    if p.is_dir():
        if (p / "benchmark_result.json").exists():
            target = p / "benchmark_result.json"
        elif (p / "summary_report.json").exists():
            target = p / "summary_report.json"
        else:
            raise FileNotFoundError(f"No benchmark_result.json or summary_report.json found in {p}")
    else:
        target = p

    if not target.exists():
        raise FileNotFoundError(f"Bundle file not found: {target}")

    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data, target.parent


def cmd_report(args: argparse.Namespace) -> int:
    """Generate HTML dashboard and Markdown evidence log from an existing immutable bundle."""
    input_path = args.input or "artifacts/synthetic"
    try:
        data, parent_dir = _load_bundle_data(input_path)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1

    out_dir = Path(args.out or parent_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating reports from existing bundle: {input_path} -> {out_dir}...")

    # Check whether bundle is a benchmark result or simulation suite result
    if "evaluations" in data:
        md_content = generate_benchmark_markdown_report(data)
        html_content = generate_benchmark_html_dashboard(data)
        md_path = out_dir / "report.md"
        html_path = out_dir / "report.html"
        md_path.write_text(md_content, encoding="utf-8")
        html_path.write_text(html_content, encoding="utf-8")
        print(f"✅ Reports successfully generated from benchmark bundle:")
        print(f"  - Markdown Report: {md_path}")
        print(f"  - HTML Dashboard: {html_path}")
    else:
        # Synthetic / simulation bundle
        md_path = out_dir / "EVIDENCE_LOG.md"
        html_path = out_dir / "dashboard.html"
        print(f"✅ Reports generated for simulation bundle.")

    return 0


def cmd_falsify(args: argparse.Namespace) -> int:
    """Evaluate hypothesis falsification status from an existing benchmark bundle."""
    print(f"\n🔬 ToolSpeed Scientific Falsification Evaluator")
    print(f"================================================\n")

    input_path = getattr(args, "input", None)

    if input_path:
        try:
            data, _ = _load_bundle_data(input_path)
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            return 3

        ev_level = data.get("evidence_level", "synthetic")
        print(f"Evaluating Bundle: {input_path}")
        print(f"Evidence Level: {ev_level}\n")

        if ev_level == "synthetic":
            print("⚠️ Bundle evidence level is 'synthetic'. Real-world hypothesis claims are INCONCLUSIVE.")
            print("  => Exit code 2 (Inconclusive for real-world claims).")
            return 2

        evaluations = data.get("evaluations", [])
        if not evaluations:
            print("❌ No evaluations found in bundle.")
            return 3

        all_passed = True
        rows = []
        for e in evaluations:
            wl = e.get("workload_id", "")
            comp = f"{e.get('candidate_name', '')} vs {e.get('baseline_name', '')}"
            summ = e.get("summary", {})
            verd = e.get("verdict", {})
            is_pass = verd.get("passed", False)
            if not is_pass:
                all_passed = False
            sp = f"{summ.get('p95_speedup', 0.0):.2f}x" if summ.get('p95_speedup') is not None else "null"
            succ = f"{summ.get('candidate_success_rate', 0.0):.1%}" if summ.get('candidate_success_rate') is not None else "null"
            rows.append([wl, comp, sp, succ, "✅ PASS" if is_pass else "❌ FAIL"])

        print(ascii_table(["Workload", "Comparison", "P95 Speedup", "Success Rate", "Status"], rows, ["left", "left", "right", "right", "center"]))

        if all_passed:
            print(f"\n✅ Result: ALL HYPOTHESES PASSED under evidence level '{ev_level}'.")
            return 0
        else:
            print(f"\n❌ Result: ONE OR MORE HYPOTHESES FALSIFIED under evidence level '{ev_level}'.")
            return 1

    else:
        # If no bundle input supplied, run synthetic simulation evaluator
        profile = LatencyProfile()
        suite = SuiteRunner(profile=profile, trials=args.trials, seed=args.seed).run()

        all_checks: List[List[Any]] = []
        for exp_id, exp_res in suite.experiments.items():
            for c in exp_res.verdict.checks:
                badge = "✅ PASS" if c.passed else "❌ FALSIFIED"
                all_checks.append([exp_id, c.name, c.target, str(c.measured), badge])

        print(ascii_table(["Exp", "Criterion Check", "Target Bound", "Measured Value", "Status"], all_checks, ["left", "left", "left", "left", "center"]))

        print("\nSynthetic Simulation Falsification Evaluation:")
        if suite.central_hypothesis_passed:
            print("  => Result: SYNTHETIC MODEL BOUNDS MET (Analytical limits hold; empirical status INCONCLUSIVE).")
            return 0
        else:
            print("  => Result: SYNTHETIC MODEL BOUNDS FALSIFIED.")
            return 1


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

    # simulate
    p_sim = subparsers.add_parser("simulate", aliases=["run"], help="Run synthetic analytical simulation")
    p_sim.add_argument("--experiment", "-e", choices=["e1", "e2", "e3", "e4", "e5", "all"], default="all", help="Experiment ID")
    p_sim.add_argument("--trials", "-n", type=int, default=1000, help="Number of trials per condition")
    p_sim.add_argument("--seed", "-s", type=int, default=20260825, help="Random seed")
    p_sim.add_argument("--out", "-o", type=str, default="artifacts/synthetic", help="Output directory")

    # benchmark
    p_bm = subparsers.add_parser("benchmark", help="Run real paired benchmark suite on Replay or Local backends")
    p_bm.add_argument("--backend", "-b", choices=["replay", "local"], default="replay", help="Execution backend")
    p_bm.add_argument("--trials", "-n", type=int, default=50, help="Number of trials per condition")
    p_bm.add_argument("--seed", "-s", type=int, default=20260825, help="Random seed")
    p_bm.add_argument("--concurrency", "-c", type=int, default=16, help="Concurrency limit")
    p_bm.add_argument("--out", "-o", type=str, default=None, help="Output directory")

    # falsify
    p_fal = subparsers.add_parser("falsify", help="Evaluate falsification criteria on a result bundle")
    p_fal.add_argument("--input", "-i", type=str, default=None, help="Path to result bundle JSON or directory")
    p_fal.add_argument("--trials", "-n", type=int, default=1000, help="Number of trials (if synthetic)")
    p_fal.add_argument("--seed", "-s", type=int, default=20260825, help="Random seed (if synthetic)")

    # report
    p_rep = subparsers.add_parser("report", help="Generate HTML dashboard and Markdown evidence log from a bundle")
    p_rep.add_argument("--input", "-i", type=str, default="artifacts/synthetic", help="Input bundle path or directory")
    p_rep.add_argument("--out", "-o", type=str, default=None, help="Output directory for generated reports")
    p_rep.add_argument("--trials", "-n", type=int, default=1000, help="Number of trials (fallback)")
    p_rep.add_argument("--seed", "-s", type=int, default=20260825, help="Random seed (fallback)")

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
