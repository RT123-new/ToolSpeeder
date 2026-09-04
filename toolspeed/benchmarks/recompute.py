"""Independent metric recomputation from raw traces with discrepancy checking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from toolspeed.benchmarks.bundle import validate_bundle_hashes_first
from toolspeed.experiments.runner import compute_summary
from toolspeed.visualization.report import (
    generate_benchmark_html_dashboard,
    generate_benchmark_markdown_report,
)


class TraceRecomputationError(ValueError):
    """Raised when trace recomputation or verification fails."""


def recompute_bundle_metrics(bundle_path: str | Path) -> dict[str, Any]:
    """Recomputes all benchmark metrics strictly from raw trace JSONL files.

    Never reads or trusts summary metrics from result.json.
    Requires a valid bundle seal and matching file hashes before processing.
    """
    b_dir = Path(bundle_path).resolve()
    valid, errors = validate_bundle_hashes_first(b_dir)
    if not valid:
        raise TraceRecomputationError(f"Cannot recompute from unsealed or tampered bundle: {'; '.join(errors)}")

    # Locate trace files: prefer raw-traces + baseline-traces or candidate + baseline
    c_file = b_dir / "candidate-traces.jsonl"
    if not c_file.exists():
        c_file = b_dir / "raw-traces.jsonl"

    b_file = b_dir / "baseline-traces.jsonl"

    if not c_file.exists():
        raise TraceRecomputationError(f"Missing candidate trace file in bundle: {c_file}")

    # Read and parse traces line by line
    c_records: list[dict[str, Any]] = []
    with open(c_file, encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                c_records.append(json.loads(line_str))

    b_records: list[dict[str, Any]] = []
    if b_file.exists():
        with open(b_file, encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    b_records.append(json.loads(line_str))

    # Cluster by workload_id
    c_by_wl: dict[str, list[dict[str, Any]]] = {}
    for r in c_records:
        wl = r.get("workload_id") or r.get("workload") or "default"
        c_by_wl.setdefault(wl, []).append(r)

    b_by_wl: dict[str, list[dict[str, Any]]] = {}
    for r in b_records:
        wl = r.get("workload_id") or r.get("workload") or "default"
        b_by_wl.setdefault(wl, []).append(r)

    recomputed_evaluations: list[dict[str, Any]] = []

    for wl, c_items in sorted(c_by_wl.items()):
        b_items = b_by_wl.get(wl, [])

        c_lats = [float(x.get("ccl_ms") or x.get("total_duration_ms") or x.get("latency_ms") or 0.0) for x in c_items]
        b_lats = (
            [float(x.get("ccl_ms") or x.get("total_duration_ms") or x.get("latency_ms") or 0.0) for x in b_items]
            if b_items
            else list(c_lats)
        )

        c_succ = [bool(x.get("success", True)) for x in c_items]
        b_succ = [bool(x.get("success", True)) for x in b_items] if b_items else [True] * len(c_succ)

        summary = compute_summary(
            baseline=np.array(b_lats, dtype=np.float64),
            candidate=np.array(c_lats, dtype=np.float64),
            baseline_success=np.array(b_succ, dtype=bool),
            candidate_success=np.array(c_succ, dtype=bool),
        )

        recomputed_evaluations.append(
            {
                "workload_id": wl,
                "candidate_name": c_items[0].get("scheduler", "Candidate"),
                "baseline_name": b_items[0].get("scheduler", "Baseline") if b_items else "Baseline",
                "summary": summary.to_dict(),
                "verdict": {
                    "passed": bool(
                        summary.p95_speedup is not None
                        and summary.p95_speedup >= 1.0
                        and (summary.candidate_success_rate is None or summary.candidate_success_rate >= 0.95)
                    ),
                    "falsified": bool(summary.p95_speedup is not None and summary.p95_speedup < 1.0),
                },
            }
        )

    # Read protocol if available
    proto_file = b_dir / "protocol.json"
    proto_data = json.loads(proto_file.read_text(encoding="utf-8")) if proto_file.exists() else {}

    return {
        "title": "ToolSpeed Recomputed Evidence Report",
        "evidence_level": proto_data.get("evidence_level", "replay_integration"),
        "recomputed_from_traces": True,
        "evaluations": recomputed_evaluations,
    }


def compare_metrics_discrepancy(
    recomputed: dict[str, Any],
    stored: dict[str, Any],
    tolerance: float = 1e-6,
) -> list[str]:
    """Compares recomputed metrics against stored metrics from result.json.

    Flags any numeric discrepancy exceeding tolerance (1e-6).
    """
    discrepancies: list[str] = []

    stored_evals = {e.get("workload_id"): e for e in stored.get("evaluations", [])}
    recomputed_evals = {e.get("workload_id"): e for e in recomputed.get("evaluations", [])}

    numeric_keys = (
        "p95_speedup",
        "p50_speedup",
        "p99_speedup",
        "candidate_p95_ms",
        "baseline_p95_ms",
        "candidate_success_rate",
        "baseline_success_rate",
        "success_rate_delta",
    )

    for wl, r_ev in recomputed_evals.items():
        if wl not in stored_evals:
            continue
        s_ev = stored_evals[wl]
        r_summ = r_ev.get("summary", {})
        s_summ = s_ev.get("summary", {})

        for k in numeric_keys:
            r_val = r_summ.get(k)
            s_val = s_summ.get(k)
            if r_val is not None and s_val is not None:
                diff = abs(float(r_val) - float(s_val))
                if diff > tolerance:
                    discrepancies.append(
                        f"Workload '{wl}' metric '{k}' discrepancy: stored={s_val}, recomputed={r_val}, diff={diff:.8f} > {tolerance}"
                    )

    return discrepancies


def generate_trace_recomputed_reports(
    bundle_path: str | Path,
    out_dir: str | Path | None = None,
) -> tuple[str, str, list[str]]:
    """Recomputes metrics from raw traces, flags discrepancies, and outputs markdown/html reports.

    Returns (markdown_report, html_dashboard, list_of_discrepancies).
    """
    b_dir = Path(bundle_path).resolve()
    recomputed_data = recompute_bundle_metrics(b_dir)

    # If result.json exists, check for discrepancies
    discrepancies: list[str] = []
    res_file = b_dir / "result.json"
    if res_file.exists():
        stored_data = json.loads(res_file.read_text(encoding="utf-8"))
        discrepancies = compare_metrics_discrepancy(recomputed_data, stored_data)

    md_report = generate_benchmark_markdown_report(recomputed_data)
    html_dashboard = generate_benchmark_html_dashboard(recomputed_data)

    if out_dir:
        o_dir = Path(out_dir).resolve()
        o_dir.mkdir(parents=True, exist_ok=True)
        (o_dir / "report.md").write_text(md_report, encoding="utf-8")
        (o_dir / "report.html").write_text(html_dashboard, encoding="utf-8")

    return md_report, html_dashboard, discrepancies
