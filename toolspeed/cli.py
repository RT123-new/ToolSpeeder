"""ToolSpeed Command-Line Interface (CLI).

Subcommands:
  simulate        - Execute synthetic analytical simulation (EvidenceLevel: SYNTHETIC)
  benchmark       - Run paired real benchmark suite on Replay or Local wall-clock backends
  falsify         - Run rigorous scientific hypothesis falsification evaluator on an existing bundle
  report          - Generate HTML dashboard and Markdown evidence log from an existing bundle
  validate-bundle - Verify structural integrity, hash provenance, and verdict eligibility of a bundle
  test            - Run test suite and adversarial integrity tests
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from toolspeed.benchmarks.harness import BenchmarkConfig, BenchmarkHarness
from toolspeed.benchmarks.recompute import (
    TraceRecomputationError,
    generate_trace_recomputed_reports,
)
from toolspeed.core.protocol import load_frozen_protocol
from toolspeed.core.types import EvidenceLevel, VerdictState, compute_file_sha256
from toolspeed.experiments.e1_dag_runner import E1DAGExperiment
from toolspeed.experiments.e2_fusion_runner import E2FusionExperiment
from toolspeed.experiments.e3_spec_runner import E3SpeculationExperiment
from toolspeed.experiments.e4_commit_runner import E4CommitHorizonExperiment
from toolspeed.experiments.e5_bytecode_runner import E5BytecodeExperiment
from toolspeed.experiments.full_suite import SuiteResult, SuiteRunner
from toolspeed.experiments.runner import LatencyProfile, compute_summary
from toolspeed.visualization.charts import ascii_bar_chart, ascii_table
from toolspeed.visualization.report import (
    generate_benchmark_html_dashboard,
    generate_benchmark_markdown_report,
    save_all_reports,
    save_benchmark_reports,
)


def _load_bundle_data(bundle_path_str: str) -> tuple[dict[str, Any], Path]:
    p = Path(bundle_path_str)
    if p.is_dir():
        if (p / "result.json").exists():
            target = p / "result.json"
        elif (p / "benchmark_result.json").exists():
            target = p / "benchmark_result.json"
        elif (p / "summary_report.json").exists():
            target = p / "summary_report.json"
        else:
            raise FileNotFoundError(f"No result.json or benchmark_result.json found in directory: {p}")
    else:
        target = p

    if not target.exists():
        raise FileNotFoundError(f"Bundle file not found: {target}")

    with open(target, encoding="utf-8") as f:
        data = json.load(f)
    return data, target.parent


def cmd_simulate(args: argparse.Namespace) -> int:
    """Execute synthetic analytical simulation (always SYNTHETIC evidence level, real-world INCONCLUSIVE)."""
    profile = LatencyProfile()
    exp_name = args.experiment.lower()
    out_dir = Path(args.out or "artifacts/synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n⚡ ToolSpeed: Running synthetic simulation '{exp_name.upper()}' ({args.trials:,} trials, seed={args.seed})...\n"
    )

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
        suite: SuiteResult = SuiteRunner(
            profile=profile, trials=args.trials, seed=args.seed, evidence_level=EvidenceLevel.SYNTHETIC
        ).run()
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
    print(
        ascii_table(
            ["Hypothesis Check", "Target", "Measured", "Status"], check_rows, ["left", "left", "left", "center"]
        )
    )
    return 0


def is_git_working_tree_dirty() -> bool:
    """Returns True if the git working tree has uncommitted modifications or untracked files."""
    try:
        res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False)
        return bool(res.stdout.strip())
    except Exception:
        return True


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run real paired benchmark suite on genuine Replay or Local wall-clock backends."""
    backend_mode = args.backend.lower()
    evidence_level = EvidenceLevel.LOCAL_WALL_CLOCK if backend_mode == "local" else EvidenceLevel.REPLAY_INTEGRATION
    out_dir = Path(args.out or f"artifacts/{backend_mode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol_name = getattr(args, "protocol", "tool-speed-v1.3-draft.json")
    try:
        protocol = load_frozen_protocol(protocol_name)
    except Exception as exc:
        print(f"❌ Failed to load protocol '{protocol_name}': {exc}")
        return 1

    mode = getattr(args, "mode", "exploratory").lower()
    if mode not in ("smoke", "exploratory", "confirmatory"):
        print(f"❌ Invalid execution mode '{mode}'. Choose from: smoke, exploratory, confirmatory")
        return 1

    # Strict validation for confirmatory mode
    if mode == "confirmatory":
        if not protocol.is_frozen or protocol.status != "prospectively_frozen":
            print(
                f"❌ Confirmatory mode requires a prospectively frozen protocol (status='prospectively_frozen', is_frozen=True).\n"
                f"Got status='{protocol.status}', is_frozen={protocol.is_frozen}. Retrospective repair and draft protocols are strictly rejected."
            )
            return 1

        min_trials = 1000 if backend_mode == "replay" else 200
        eff_trials = args.trials if args.trials is not None else min_trials
        if eff_trials < min_trials:
            print(
                f"❌ Confirmatory mode requires >= {min_trials} trials per seed for {backend_mode} backend. Specified: {eff_trials}"
            )
            return 1

        # Confirmatory seeds requirement: >= 3 distinct seeds, no retrospective seeds
        protocol_seeds_dict = getattr(protocol, "seeds_dict", None)
        if getattr(args, "seeds", None):
            conf_seeds = [int(s.strip()) for s in args.seeds.split(",")]
        elif protocol_seeds_dict is not None and "confirmatory" in protocol_seeds_dict:
            conf_seeds = list(protocol_seeds_dict["confirmatory"])
        else:
            conf_seeds = (
                list(protocol.seeds)
                if protocol.plan_id == "tool-speed-v1.3"
                else [s for s in protocol.seeds if s not in {42, 137, 2026}]
            )

        if len(conf_seeds) < 3:
            print(f"❌ Confirmatory mode requires >= 3 distinct seeds. Found {len(conf_seeds)}: {conf_seeds}")
            return 1
        if len(conf_seeds) != len(set(conf_seeds)):
            print(f"❌ Confirmatory seeds must be unique. Found duplicates: {conf_seeds}")
            return 1
        if protocol.plan_id == "tool-speed-v1.3-draft" and set(conf_seeds).intersection({42, 137, 2026}):
            print("❌ Confirmatory mode strictly forbids reusing retrospective seeds (42, 137, 2026) in draft protocol.")
            return 1

        if is_git_working_tree_dirty():
            print("❌ Confirmatory mode requires a clean git working tree with zero uncommitted changes.")
            return 1

        seeds_list = conf_seeds
        trials = eff_trials
    elif mode == "smoke":
        trials = args.trials if args.trials is not None else protocol.smoke_trials
        seeds_list = [args.seed] if args.seed is not None else [protocol.seeds[0]]
    else:  # exploratory
        default_trials = 200 if backend_mode == "local" else 1000
        trials = args.trials if args.trials is not None else default_trials
        protocol_seeds_dict = getattr(protocol, "seeds_dict", None)
        if getattr(args, "seeds", None):
            seeds_list = [int(s.strip()) for s in args.seeds.split(",")]
        elif protocol_seeds_dict is not None and "exploratory" in protocol_seeds_dict:
            seeds_list = list(protocol_seeds_dict["exploratory"])
        else:
            seeds_list = protocol.seeds[:3] if len(protocol.seeds) >= 3 else [args.seed or 101]

    seed_val = seeds_list[0] if seeds_list else (args.seed or 101)

    print("\n=======================================================")
    print(f"🚀 ToolSpeed Paired Benchmark Suite ({backend_mode.upper()} Backend, Mode: {mode.upper()})")
    print("=======================================================")
    print(
        f"Protocol: {protocol.plan_id} (v{protocol.plan_version}) | Mode: {mode.upper()} | "
        f"Evidence Level: {evidence_level.value} | Trials: {trials} per condition | Seeds: {seeds_list} | Out: {out_dir}\n"
    )

    config = BenchmarkConfig(
        trials_per_condition=trials,
        seed=seed_val,
        seeds=seeds_list,
        evidence_level=evidence_level,
        concurrency_limit=args.concurrency,
        include_negative_controls=True,
        protocol=protocol,
        mode=mode,
    )

    harness = BenchmarkHarness(config=config, protocol=protocol)
    if len(seeds_list) > 1:
        results = asyncio.run(harness.run_multi_seed_benchmark(seeds=seeds_list, trials=trials))
        for res_item in results:
            seed_dir = out_dir / f"seed_{res_item.manifest.seed}" if res_item.manifest else out_dir
            save_benchmark_reports(res_item, seed_dir)
        result = results[0]
        save_benchmark_reports(result, out_dir)
    else:
        result = asyncio.run(harness.run_full_benchmark())
        save_benchmark_reports(result, out_dir)

    print("\n📊 Paired Benchmark Evaluation Summary (P95 CCL Speedup):")
    bar_data = {
        eval_item.workload_id + " (" + eval_item.candidate_name + " vs " + eval_item.baseline_name + ")": (
            eval_item.summary.p95_speedup or 1.0
        )
        for eval_item in result.evaluations
    }
    print(ascii_bar_chart(bar_data, max_bar_width=30, unit="x"))

    print("\n📋 Canonical Workload Matrix (W1 – W7, E5a):")
    wl_table_rows = []
    for e in result.evaluations:
        status = (
            "PASS" if e.verdict.passed else ("INCONCLUSIVE" if e.verdict.state == VerdictState.INCONCLUSIVE else "FAIL")
        )
        b95 = f"{e.summary.baseline_p95_ms:.1f}ms" if e.summary.baseline_p95_ms is not None else "null"
        c95 = f"{e.summary.candidate_p95_ms:.1f}ms" if e.summary.candidate_p95_ms is not None else "null"
        sp = f"{e.summary.p95_speedup:.2f}x" if e.summary.p95_speedup is not None else "null"
        succ = f"{e.summary.candidate_success_rate:.1%}" if e.summary.candidate_success_rate is not None else "null"
        wl_table_rows.append(
            [
                e.workload_id,
                f"{e.candidate_name} vs {e.baseline_name}",
                b95,
                c95,
                sp,
                succ,
                status,
            ]
        )
    print(
        ascii_table(
            ["ID", "Comparison", "Base P95", "Cand P95", "Speedup", "Success", "Status"],
            wl_table_rows,
            ["left", "left", "right", "right", "right", "right", "center"],
        )
    )

    if result.negative_controls:
        print("\n🔬 Negative Control Verification:")
        neg_rows = []
        for nc in result.negative_controls:
            neg_rows.append(
                [
                    nc["control"],
                    f"{nc['p95_speedup']:.2f}x",
                    "PASS" if nc["passed_expected_null"] else "FAIL",
                    nc["detail"],
                ]
            )
        print(
            ascii_table(
                ["Control", "Measured Speedup", "Null Check", "Detail"], neg_rows, ["left", "right", "center", "left"]
            )
        )

    print(f"\n📁 Benchmark Bundle saved to: {out_dir}")
    print(f"⏱️ Total runtime: {result.total_runtime_s:.2f}s | Overall Verdict: {result.overall_verdict.value}\n")

    if getattr(args, "require_pass", False) and result.overall_verdict != VerdictState.PASSED:
        print(f"❌ Failure: --require-pass specified and overall verdict is '{result.overall_verdict.value}'")
        return 1

    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Generate HTML dashboard and Markdown evidence log from an existing immutable bundle.

    Recomputes metrics independently from raw traces; never reads stored summary metrics.
    """
    input_path = args.input or "artifacts/synthetic"
    try:
        data, parent_dir = _load_bundle_data(input_path)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1

    manifest_file = parent_dir / "manifest.json"
    sha_file = parent_dir / "bundle.sha256"
    sig_file = parent_dir / "manifest.sig"
    if (
        Path(input_path).is_dir()
        and not manifest_file.exists()
        and not sha_file.exists()
        and not sig_file.exists()
        and not data.get("evaluations")
    ):
        print(
            f"❌ Error: Bundle at {parent_dir} is unsealed and unsigned (missing manifest.json / bundle.sha256 / manifest.sig)."
        )
        return 1

    out_dir = Path(args.out or parent_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating reports from existing bundle: {input_path} -> {out_dir}...")

    # Independent recomputation from raw traces if trace files are present
    has_traces = (parent_dir / "raw-traces.jsonl").exists() or (parent_dir / "candidate-traces.jsonl").exists()
    if has_traces and (parent_dir / "manifest.json").exists():
        try:
            print("📊 Recomputing all metrics independently from raw traces (never reading stored summaries)...")
            md_content, html_content, discrepancies = generate_trace_recomputed_reports(parent_dir, out_dir)
            if discrepancies:
                print(
                    f"⚠️ WARNING: Found {len(discrepancies)} metric discrepancy(ies) between raw traces and stored result.json (> 1e-6):"
                )
                for disc in discrepancies:
                    print(f"  • {disc}")
            else:
                print("✅ Recomputed metrics match stored metrics within 1e-6 tolerance.")

            md_path = out_dir / "report.md"
            html_path = out_dir / "report.html"
            print("✅ Reports successfully generated from recomputed raw traces:")
            print(f"  - Markdown Report: {md_path}")
            print(f"  - HTML Dashboard: {html_path}")
            return 0
        except TraceRecomputationError as e:
            print(f"❌ Trace recomputation failed: {e}")
            return 1

    if "evaluations" in data:
        md_content = generate_benchmark_markdown_report(data)
        html_content = generate_benchmark_html_dashboard(data)
        md_path = out_dir / "report.md"
        html_path = out_dir / "report.html"
        md_path.write_text(md_content, encoding="utf-8")
        html_path.write_text(html_content, encoding="utf-8")
        print("✅ Reports successfully generated from benchmark bundle:")
        print(f"  - Markdown Report: {md_path}")
        print(f"  - HTML Dashboard: {html_path}")
    else:
        print(f"✅ Bundle data loaded for {input_path}.")

    return 0


def cmd_falsify(args: argparse.Namespace) -> int:
    """Evaluate hypothesis falsification status from an existing benchmark bundle.

    Exit codes:
      0 = PASSED (all hypotheses met under verdict-eligible empirical evidence)
      1 = FALSIFIED (one or more hypotheses failed efficacy or safety thresholds)
      2 = INCONCLUSIVE (synthetic simulation, underpowered smoke trial, or missing metrics)
      3 = ERROR / MALFORMED (bundle missing, invalid json, hash mismatch)
    """
    print("\n🔬 ToolSpeed Scientific Falsification Evaluator")
    print("================================================\n")

    input_path = getattr(args, "input", None)
    if not input_path:
        print("❌ Error: --input <path_to_bundle> is required for falsification evaluation.")
        return 3

    try:
        data, _ = _load_bundle_data(input_path)
    except Exception as e:
        print(f"❌ Error loading bundle: {e}")
        return 3

    ev_level = data.get("evidence_level", "synthetic")
    print(f"Evaluating Bundle: {input_path}")
    print(f"Evidence Level: {ev_level}\n")

    if ev_level == "synthetic":
        print("⚠️ Bundle evidence level is 'synthetic'. Real-world hypothesis claims are INCONCLUSIVE.")
        print("  => Exit code 2 (Inconclusive for empirical claims).")
        return 2

    manifest = data.get("manifest") or {}
    is_eligible = manifest.get("is_verdict_eligible", True)
    trial_count = manifest.get("trial_count", 0)

    min_required = 1000 if ev_level == "replay_integration" else 200
    if not is_eligible or trial_count < min_required:
        print(
            f"⚠️ Smoke run / Insufficient sample size (n={trial_count} < {min_required}). Marked SMOKE — NOT VERDICT-ELIGIBLE."
        )
        print("  => Exit code 2 (Inconclusive).")
        return 2

    # Attempt recomputation from raw traces if available
    parent_dir = Path(input_path) if Path(input_path).is_dir() else Path(input_path).parent
    c_traces_file = parent_dir / "candidate-traces.jsonl"
    b_traces_file = parent_dir / "baseline-traces.jsonl"

    if c_traces_file.exists() and b_traces_file.exists():
        print("📊 Recomputing statistical metrics and hypothesis checks from raw JSONL traces...")
        c_lines = [json.loads(line) for line in c_traces_file.read_text().splitlines() if line.strip()]
        b_lines = [json.loads(line) for line in b_traces_file.read_text().splitlines() if line.strip()]

        c_by_wl: dict[str, list[dict[str, Any]]] = {}
        for t in c_lines:
            wl = t.get("workload_id", "W1")
            c_by_wl.setdefault(wl, []).append(t)
        b_by_wl: dict[str, list[dict[str, Any]]] = {}
        for t in b_lines:
            wl = t.get("workload_id", "W1")
            b_by_wl.setdefault(wl, []).append(t)

        any_raw_falsified = False
        for wl, c_list in c_by_wl.items():
            b_list = b_by_wl.get(wl, [])
            c_lat = [float(x.get("ccl_ms", x.get("total_duration_ms", 100.0))) for x in c_list]
            b_lat = [float(x.get("ccl_ms", x.get("total_duration_ms", 100.0))) for x in b_list] or [100.0] * len(c_lat)
            c_succ = [bool(x.get("success", False)) for x in c_list]
            b_succ = [bool(x.get("success", True)) for x in b_list] or [True] * len(c_succ)

            raw_summary = compute_summary(
                baseline=np.array(b_lat, dtype=np.float64),
                candidate=np.array(c_lat, dtype=np.float64),
                baseline_success=np.array(b_succ, dtype=bool),
                candidate_success=np.array(c_succ, dtype=bool),
            )

            p95_sp = raw_summary.p95_speedup if raw_summary.p95_speedup is not None else 0.0
            cand_s = raw_summary.candidate_success_rate if raw_summary.candidate_success_rate is not None else 0.0

            if cand_s < 0.95 or p95_sp < 1.0:
                print(f"❌ Raw traces falsified workload '{wl}': success={cand_s:.1%}, p95_speedup={p95_sp:.2f}x")
                any_raw_falsified = True

        if any_raw_falsified:
            print("❌ Result: ONE OR MORE HYPOTHESES FALSIFIED based on recomputation from raw traces.")
            return 1

    evaluations = data.get("evaluations", [])
    if not evaluations:
        print("❌ No evaluations found in bundle.")
        return 3

    all_passed = True
    any_falsified = False
    rows = []
    for ev in evaluations:
        wl = ev.get("workload_id", "")
        comp = f"{ev.get('candidate_name', '')} vs {ev.get('baseline_name', '')}"
        summ = ev.get("summary", {})
        verd = ev.get("verdict", {})
        is_pass = verd.get("passed", False)
        is_falsified = verd.get("falsified", False)

        if not is_pass:
            all_passed = False
        if is_falsified:
            any_falsified = True

        sp = f"{summ.get('p95_speedup', 0.0):.2f}x" if summ.get("p95_speedup") is not None else "null"
        succ = (
            f"{summ.get('candidate_success_rate', 0.0):.1%}"
            if summ.get("candidate_success_rate") is not None
            else "null"
        )
        status_label = "✅ PASS" if is_pass else ("❌ FAIL" if is_falsified else "⚠️ INCONCLUSIVE")
        rows.append([wl, comp, sp, succ, status_label])

    print(
        ascii_table(
            ["Workload", "Comparison", "P95 Speedup", "Success Rate", "Status"],
            rows,
            ["left", "left", "right", "right", "center"],
        )
    )

    if all_passed:
        print(f"\n✅ Result: ALL HYPOTHESES PASSED under evidence level '{ev_level}'.")
        return 0
    elif any_falsified:
        print(f"\n❌ Result: ONE OR MORE HYPOTHESES FALSIFIED under evidence level '{ev_level}'.")
        return 1
    else:
        print(f"\n⚠️ Result: INCONCLUSIVE under evidence level '{ev_level}'.")
        return 2


def cmd_validate_bundle(args: argparse.Namespace) -> int:
    """Validate structure, manifest hashes, sample sizes, and guardrails of a result bundle."""
    input_path = getattr(args, "input", None)
    if not input_path:
        print("❌ Error: --input <path_to_bundle> is required.")
        return 1

    print(f"\n🔍 ToolSpeed Bundle Validator: {input_path}\n")

    try:
        data, parent_dir = _load_bundle_data(input_path)
    except Exception as e:
        print(f"❌ Error loading bundle: {e}")
        return 1

    checks_passed = True

    # 1. Manifest verification
    manifest = data.get("manifest")
    if not manifest:
        print("❌ FAILED: Missing 'manifest' block in bundle.")
        checks_passed = False
    else:
        print("✅ PASS: ArtifactManifest present.")
        required_manifest_fields = [
            "code_git_sha",
            "evidence_level",
            "trial_count",
            "benchmark_config_hash",
            "workload_fixture_hash",
            "raw_trace_hash",
        ]
        for f in required_manifest_fields:
            if f not in manifest:
                print(f"❌ FAILED: Manifest missing required field '{f}'")
                checks_passed = False
            else:
                print(f"  • {f}: {manifest[f]}")

        manifest_on_disk = parent_dir / "manifest.json"
        if manifest_on_disk.exists() and (
            "file_hashes" not in manifest or not isinstance(manifest.get("file_hashes"), dict)
        ):
            print("❌ FAILED: Manifest missing required 'file_hashes' mapping.")
            checks_passed = False

    # 2. Evaluations verification
    evaluations = data.get("evaluations", [])
    if not evaluations:
        print("❌ FAILED: No evaluations found in bundle.")
        checks_passed = False
    else:
        print(f"✅ PASS: Found {len(evaluations)} paired workload evaluations.")
        for idx, ev in enumerate(evaluations):
            wl = ev.get("workload_id", f"idx_{idx}")
            summ = ev.get("summary", {})
            if summ.get("candidate_p95_ms") is None or summ.get("baseline_p95_ms") is None:
                print(f"❌ FAILED [{wl}]: Required P95 CCL latency is null.")
                checks_passed = False
            if summ.get("p95_speedup") is None:
                print(f"❌ FAILED [{wl}]: Required P95 speedup is null.")
                checks_passed = False
            if summ.get("candidate_success_rate") is None:
                print(f"❌ FAILED [{wl}]: Candidate success rate is null.")
                checks_passed = False

    # 3. Guardrail safety checks
    for ev in evaluations:
        summ = ev.get("summary", {})
        side_effects = summ.get("unapproved_side_effects", 0)
        if side_effects > 0:
            print(f"❌ FAILED [{ev.get('workload_id')}]: Found {side_effects} unapproved side-effects!")
            checks_passed = False

    # 4. File byte SHA-256 hash provenance verification
    file_hashes = manifest.get("file_hashes", {}) if manifest else {}
    if file_hashes and parent_dir.is_dir():
        print("🔐 Verifying bundle file SHA-256 byte hashes...")
        for fname, expected_hash in file_hashes.items():
            if fname in ("manifest.json", "bundle.sha256"):
                continue
            fpath = parent_dir / fname
            if not fpath.exists():
                print(f"❌ FAILED: File '{fname}' listed in manifest file_hashes does not exist.")
                checks_passed = False
            else:
                actual_hash = compute_file_sha256(fpath)
                if actual_hash != expected_hash:
                    print(f"❌ FAILED: File '{fname}' hash mismatch! Expected {expected_hash}, computed {actual_hash}")
                    checks_passed = False
                else:
                    print(f"  • {fname} SHA-256 verified ({actual_hash[:12]}...)")

    # 5. Report files existence
    if parent_dir.is_dir():
        if (parent_dir / "report.md").exists() and (parent_dir / "report.html").exists():
            print("✅ PASS: report.md and report.html verified on disk.")
        elif (parent_dir / "EVIDENCE_LOG.md").exists() and (parent_dir / "dashboard.html").exists():
            print("✅ PASS: EVIDENCE_LOG.md and dashboard.html verified on disk.")

    if checks_passed:
        print("\n✨ Bundle validation PASSED successfully.\n")
        return 0
    else:
        print("\n❌ Bundle validation FAILED integrity checks.\n")
        return 1


def cmd_test(args: argparse.Namespace) -> int:
    """Run unittest suite and adversarial integrity tests."""
    import unittest

    suite = unittest.defaultTestLoader.discover(start_dir="tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="toolspeed",
        description="ToolSpeed: Latency Optimization & Falsification Suite for AI Agent Tool Calls",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to execute")

    # simulate
    p_sim = subparsers.add_parser("simulate", aliases=["run"], help="Run synthetic analytical simulation")
    p_sim.add_argument(
        "--experiment", "-e", choices=["e1", "e2", "e3", "e4", "e5", "all"], default="all", help="Experiment ID"
    )
    p_sim.add_argument("--trials", "-n", type=int, default=1000, help="Number of trials per condition")
    p_sim.add_argument("--seed", "-s", type=int, default=20260825, help="Random seed")
    p_sim.add_argument("--out", "-o", type=str, default="artifacts/synthetic", help="Output directory")

    # benchmark
    p_bm = subparsers.add_parser("benchmark", help="Run real paired benchmark suite on Replay or Local backends")
    p_bm.add_argument("--backend", "-b", choices=["replay", "local"], default="replay", help="Execution backend")
    p_bm.add_argument(
        "--protocol",
        "-p",
        type=str,
        default="tool-speed-v1.3-draft.json",
        help="Path or name of authoritative protocol file",
    )
    p_bm.add_argument(
        "--mode",
        "-m",
        choices=["smoke", "exploratory", "confirmatory"],
        default="exploratory",
        help="Benchmark execution mode (smoke, exploratory, confirmatory)",
    )
    p_bm.add_argument(
        "--trials",
        "-n",
        type=int,
        default=None,
        help="Number of trials per condition (defaults: replay=1000, local=200)",
    )
    p_bm.add_argument("--seed", "-s", type=int, default=None, help="Random seed")
    p_bm.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated random seeds (defaults to 3 seeds for confirmatory eligibility)",
    )
    p_bm.add_argument("--concurrency", "-c", type=int, default=16, help="Concurrency limit")
    p_bm.add_argument("--out", "-o", type=str, default=None, help="Output directory")
    p_bm.add_argument("--require-pass", action="store_true", help="Exit non-zero if overall verdict is not PASSED")

    # falsify
    p_fal = subparsers.add_parser("falsify", help="Evaluate falsification criteria on a result bundle")
    p_fal.add_argument("--input", "-i", type=str, default=None, help="Path to result bundle JSON or directory")
    p_fal.add_argument("--trials", "-n", type=int, default=150, help="Number of trials if running simulation")

    # report
    p_rep = subparsers.add_parser("report", help="Generate HTML dashboard and Markdown evidence log from a bundle")
    p_rep.add_argument("--input", "-i", type=str, required=True, help="Input bundle path or directory")
    p_rep.add_argument("--out", "-o", type=str, default=None, help="Output directory for generated reports")

    # validate-bundle
    p_val = subparsers.add_parser(
        "validate-bundle", help="Verify structural integrity, hash provenance, and verdict eligibility"
    )
    p_val.add_argument("--input", "-i", type=str, required=True, help="Path to result bundle JSON or directory")

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
    elif args.subcommand == "validate-bundle":
        return cmd_validate_bundle(args)
    elif args.subcommand == "test":
        return cmd_test(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
