"""Report and Artifact Generators for ToolSpeed."""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from toolspeed.core.types import EvidenceLevel, compute_file_sha256, strict_json_dumps
from toolspeed.experiments.full_suite import SuiteResult
from toolspeed.visualization.charts import (
    generate_speedup_line_chart,
    generate_workload_bar_chart,
)

if TYPE_CHECKING:
    from toolspeed.benchmarks.harness import BenchmarkRunResult


def generate_markdown_evidence_log(suite_result: SuiteResult) -> str:
    """Generate comprehensive Markdown Evidence Log matching research plan template."""
    ev_level = (
        suite_result.evidence_level.value
        if isinstance(suite_result.evidence_level, EvidenceLevel)
        else str(suite_result.evidence_level)
    )
    lines: list[str] = [
        "# ToolSpeed — Evidence Log & Experiment Report",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  ",
        f"**Evidence Level:** `{ev_level}`  ",
        f"**Trials per condition:** {suite_result.trials:,}  ",
        f"**Random Seed:** {suite_result.seed}  ",
        f"**Total Suite Runtime:** {suite_result.total_runtime_sec:.2f}s  ",
    ]

    if suite_result.manifest:
        m = suite_result.manifest
        lines.extend(
            [
                f"**Git Commit:** `{m.commit_sha}` ({'dirty' if m.dirty else 'clean'})  ",
                f"**Hardware / OS:** `{m.os_platform}` | Python `{m.python_version}`  ",
            ]
        )

    lines.extend(
        [
            "",
            "## Executive Summary",
            "",
            f"- **Central Falsification Hypothesis:** {'**CONFIRMED / PASSED (Under Declared Evidence Level)**' if suite_result.central_hypothesis_passed else '**FALSIFIED**'}",
            f"- **Evidence Level:** `{ev_level}`",
            "- **Tested Mechanisms:** DAG Parallelism (E1), JIT Fusion (E2), Speculative Reads (E3), Commit Horizon (E4), Action Bytecode (E5).",
            "- **Primary Metric:** Correct Completion Latency (CCL) at P50, P90, P95, and P99.",
            "",
            "> [!NOTE]",
            f"> Evidence level `{ev_level}` results represent rigorous validation within this test environment. Synthetic simulations must not be conflated with empirical live network validation.",
            "",
            "## Canonical Evidence Log",
            "",
            "| Experiment | Tested | Succeeded | Failed | Still unproven | Next action |",
            "|---|---|---|---|---|---|",
        ]
    )

    for row in suite_result.evidence_log:
        exp = row.get("experiment", "")
        tested = row.get("tested", "Yes")
        succ = row.get("succeeded", "")
        fail = row.get("failed", "")
        unproven = row.get("still_unproven", "")
        action = row.get("next_action", "")
        lines.append(f"| {exp} | {tested} | {succ} | {fail} | {unproven} | {action} |")

    lines.extend(
        [
            "",
            "## Workload Performance Matrix (W1 – W7)",
            "",
            "| Workload | Name | Baseline P95 | Candidate P95 | P95 Speedup | CCL Reduction | Status | Level |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )

    for w in suite_result.workloads.values():
        status_badge = "✅ PASS" if w.central_hypothesis_passed else "❌ FAIL"
        w_level = w.evidence_level.value if isinstance(w.evidence_level, EvidenceLevel) else str(w.evidence_level)
        lines.append(
            f"| {w.workload_id} | {w.name} | {w.baseline_p95_ms:.1f}ms | {w.candidate_p95_ms:.1f}ms | {w.p95_speedup:.2f}x | {w.p95_reduction_pct:.1f}% | {status_badge} | `{w_level}` |"
        )

    lines.extend(
        [
            "",
            "## Detailed Experiment Results & Hypothesis Checks",
            "",
        ]
    )

    for exp_res in suite_result.experiments.values():
        verdict_badge = "PASSED" if exp_res.verdict.passed else "FALSIFIED"
        lines.extend(
            [
                f"### {exp_res.title} [{verdict_badge}]",
                "",
                f"**Hypothesis:** {exp_res.verdict.hypothesis}  ",
                f"**Summary:** {exp_res.verdict.summary}  ",
                "",
                "#### Hypothesis Evaluation Checks:",
                "",
                "| Check Name | Target | Measured | Status | Detail |",
                "|---|---|---|---|---|",
            ]
        )

        for c in exp_res.verdict.checks:
            c_badge = "✅ PASS" if c.passed else "❌ FAIL"
            lines.append(f"| `{c.name}` | {c.target} | {c.measured} | {c_badge} | {c.detail} |")

        lines.extend(
            [
                "",
                "#### Parameter Sweep Summary:",
                "",
            ]
        )

        if exp_res.rows:
            sample_keys = [
                exp_res.parameter_name,
                "baseline_p50_ms",
                "candidate_p50_ms",
                "p50_speedup",
                "baseline_p95_ms",
                "candidate_p95_ms",
                "p95_speedup",
                "wasted_call_rate",
                "candidate_success_rate",
            ]
            valid_keys = [k for k in sample_keys if k in exp_res.rows[0]]
            header_str = " | ".join(valid_keys)
            sep_str = " | ".join(["---"] * len(valid_keys))
            lines.append(f"| {header_str} |")
            lines.append(f"| {sep_str} |")

            for r in exp_res.rows[:10]:
                vals = []
                for k in valid_keys:
                    val = r.get(k, "")
                    if isinstance(val, float):
                        vals.append(f"{val:.2f}")
                    else:
                        vals.append(str(val))
                lines.append(f"| {' | '.join(vals)} |")

            if len(exp_res.rows) > 10:
                lines.append(f"\n*... and {len(exp_res.rows) - 10} more parameter configurations in CSV/JSON.*")

        lines.append("")

    return "\n".join(lines)


def generate_html_dashboard(suite_result: SuiteResult) -> str:
    """Generate standalone interactive HTML dashboard with embedded SVGs, styling, and provenance."""
    ev_level = (
        suite_result.evidence_level.value
        if isinstance(suite_result.evidence_level, EvidenceLevel)
        else str(suite_result.evidence_level)
    )

    workload_items = [w.to_dict() for w in suite_result.workloads.values()]
    workload_svg = generate_workload_bar_chart(workload_items, width=800, height=360)

    e1 = suite_result.experiments.get("E1")
    if e1 and e1.rows:
        e1_series = {
            "P50 Speedup": [
                (r["independent_calls"], r["p50_speedup"])
                for r in e1.rows
                if isinstance(r["independent_calls"], (int, float))
            ],
            "P95 Speedup": [
                (r["independent_calls"], r["p95_speedup"])
                for r in e1.rows
                if isinstance(r["independent_calls"], (int, float))
            ],
        }
        e1_svg = generate_speedup_line_chart(
            "E1: DAG Parallel Fan-out Speedup vs Call Count",
            "Independent Tool Calls",
            "Latency Speedup Factor",
            e1_series,
            width=760,
            height=360,
        )
    else:
        e1_svg = ""

    e2 = suite_result.experiments.get("E2")
    if e2 and e2.rows:
        e2_series = {
            "P50 Speedup": [
                (r["dependent_steps"], r["p50_speedup"])
                for r in e2.rows
                if isinstance(r["dependent_steps"], (int, float))
            ],
            "P95 Speedup": [
                (r["dependent_steps"], r["p95_speedup"])
                for r in e2.rows
                if isinstance(r["dependent_steps"], (int, float))
            ],
        }
        e2_svg = generate_speedup_line_chart(
            "E2: Workflow Fusion Speedup vs Dependent Steps",
            "Dependent Chain Steps",
            "Latency Speedup Factor",
            e2_series,
            width=760,
            height=360,
        )
    else:
        e2_svg = ""

    e3 = suite_result.experiments.get("E3")
    if e3 and e3.rows:
        e3_modes: dict[str, list[tuple[float, float]]] = {}
        for r in e3.rows:
            mode = r.get("contention_mode")
            acc = r.get("prediction_accuracy")
            spd = r.get("p50_speedup")
            if mode and acc is not None and spd is not None and mode != "confidence_gated":
                e3_modes.setdefault(mode, []).append((float(acc), float(spd)))
        e3_svg = generate_speedup_line_chart(
            "E3: Speculative Execution Latency Speedup by Contention Mode",
            "Prediction Accuracy",
            "P50 Latency Speedup",
            e3_modes,
            width=760,
            height=360,
        )
    else:
        e3_svg = ""

    e4 = suite_result.experiments.get("E4")
    if e4 and e4.rows:
        e4_series = {
            "P95 CCL Speedup": [(r["commit_fraction"], r["p95_speedup"]) for r in e4.rows],
            "P95 Tool-Start Speedup": [(r["commit_fraction"], r["tool_start_speedup_p95"]) for r in e4.rows],
        }
        e4_svg = generate_speedup_line_chart(
            "E4: Commit-Horizon Dispatch Speedup vs Commit Fraction",
            "Commit Fraction (Horizon)",
            "Speedup Factor",
            e4_series,
            width=760,
            height=360,
        )
    else:
        e4_svg = ""

    all_speedups = [w.p95_speedup for w in suite_result.workloads.values() if w.p95_speedup is not None]
    best_speedup = max(all_speedups) if all_speedups else 1.0
    valid_reds = [w.p95_reduction_pct for w in suite_result.workloads.values() if w.p95_reduction_pct is not None]
    avg_red = sum(valid_reds) / max(1, len(valid_reds))
    passed_exps = sum(1 for exp in suite_result.experiments.values() if exp.verdict.passed)
    total_exps = len(suite_result.experiments)

    manifest_info = ""
    if suite_result.manifest:
        m = suite_result.manifest
        manifest_info = f"<span>Git: <code>{m.commit_sha[:8]}</code> ({'dirty' if m.dirty else 'clean'})</span> | <span>OS: {m.os_platform}</span>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ToolSpeed — Benchmark & Falsification Dashboard</title>
  <style>
    :root {{
      --bg-main: #090d16;
      --bg-card: #0f172a;
      --bg-card-hover: #1e293b;
      --border: #334155;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #3b82f6;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg-main);
      color: var(--text-main);
      line-height: 1.5;
      padding: 24px;
    }}
    .container {{ max-width: 1280px; margin: 0 auto; }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 24px;
    }}
    h1 {{ font-size: 24px; font-weight: 700; color: #fff; }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 600;
    }}
    .badge-success {{ background: rgba(16, 185, 129, 0.2); color: var(--success); border: 1px solid var(--success); }}
    .badge-danger {{ background: rgba(239, 68, 68, 0.2); color: var(--danger); border: 1px solid var(--danger); }}
    .badge-primary {{ background: rgba(59, 130, 246, 0.2); color: var(--accent); border: 1px solid var(--accent); }}
    .badge-level {{ background: rgba(245, 158, 11, 0.2); color: var(--warning); border: 1px solid var(--warning); }}

    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }}
    .kpi-card {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
    }}
    .kpi-title {{ font-size: 13px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }}
    .kpi-value {{ font-size: 28px; font-weight: 700; margin: 6px 0; color: #fff; }}
    .kpi-subtitle {{ font-size: 12px; color: var(--text-muted); }}

    .section-title {{ font-size: 18px; font-weight: 600; margin: 24px 0 14px 0; color: #fff; }}
    .card {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 24px;
    }}
    .chart-container {{ margin: 12px 0; overflow-x: auto; }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      text-align: left;
    }}
    th {{
      background: #1e293b;
      color: var(--text-muted);
      font-weight: 600;
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
    }}
    td {{
      padding: 10px 14px;
      border-bottom: 1px solid #1e293b;
      color: var(--text-main);
    }}
    tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}

    .grid-2 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
      gap: 20px;
    }}
    footer {{
      margin-top: 40px;
      padding-top: 20px;
      border-top: 1px solid var(--border);
      text-align: center;
      color: var(--text-muted);
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>⚡ ToolSpeed Latency & Falsification Dashboard</h1>
        <p style="color: var(--text-muted); font-size: 13px; margin-top: 4px;">
          Evidence Level: <span class="badge badge-level">{ev_level}</span> | {manifest_info}
        </p>
      </div>
      <div>
        <span class="badge {"badge-success" if suite_result.central_hypothesis_passed else "badge-danger"}">
          {"CENTRAL HYPOTHESIS: PASS" if suite_result.central_hypothesis_passed else "CENTRAL HYPOTHESIS: FALSIFIED"}
        </span>
      </div>
    </header>

    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-title">Max Workload Speedup</div>
        <div class="kpi-value" style="color: var(--success);">{best_speedup:.2f}x</div>
        <div class="kpi-subtitle">Highest P95 CCL latency gain</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Avg P95 CCL Reduction</div>
        <div class="kpi-value" style="color: var(--accent);">{avg_red:.1f}%</div>
        <div class="kpi-subtitle">Across canonical W1-W7 workloads</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Hypotheses Validated</div>
        <div class="kpi-value">{passed_exps} / {total_exps}</div>
        <div class="kpi-subtitle">E1-E5 Falsification checks passed</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Trials Evaluated</div>
        <div class="kpi-value">{suite_result.trials:,}</div>
        <div class="kpi-subtitle">Evidence Level: {ev_level}</div>
      </div>
    </div>

    <div class="card">
      <div class="section-title" style="margin-top:0;">Workload Performance Overview (W1 - W7)</div>
      <div class="chart-container">{workload_svg}</div>
      <table>
        <thead>
          <tr>
            <th>Workload ID</th>
            <th>Name</th>
            <th>Primary Mechanism</th>
            <th>Baseline P95</th>
            <th>Candidate P95</th>
            <th>P95 Speedup</th>
            <th>CCL Reduction</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {
        "".join(
            f'''
          <tr>
            <td><strong>{w.workload_id}</strong></td>
            <td>{w.name}</td>
            <td>{", ".join(w.primary_mechanisms)}</td>
            <td>{w.baseline_p95_ms:.1f} ms</td>
            <td>{w.candidate_p95_ms:.1f} ms</td>
            <td><strong>{w.p95_speedup:.2f}x</strong></td>
            <td style="color: var(--success);">{w.p95_reduction_pct:.1f}%</td>
            <td><span class="badge {'badge-success' if w.central_hypothesis_passed else 'badge-danger'}">{'PASS' if w.central_hypothesis_passed else 'FAIL'}</span></td>
          </tr>
          '''
            for w in suite_result.workloads.values()
        )
    }
        </tbody>
      </table>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="section-title" style="margin-top:0;">E1: DAG Parallelism</div>
        <div class="chart-container">{e1_svg}</div>
      </div>
      <div class="card">
        <div class="section-title" style="margin-top:0;">E2: JIT Workflow Fusion</div>
        <div class="chart-container">{e2_svg}</div>
      </div>
      <div class="card">
        <div class="section-title" style="margin-top:0;">E3: Speculative Reads</div>
        <div class="chart-container">{e3_svg}</div>
      </div>
      <div class="card">
        <div class="section-title" style="margin-top:0;">E4: Commit-Horizon Dispatch</div>
        <div class="chart-container">{e4_svg}</div>
      </div>
    </div>

    <div class="card">
      <div class="section-title" style="margin-top:0;">Canonical Research Evidence Log</div>
      <table>
        <thead>
          <tr>
            <th>Experiment</th>
            <th>Tested</th>
            <th>Succeeded</th>
            <th>Failed</th>
            <th>Still Unproven</th>
            <th>Next Action</th>
          </tr>
        </thead>
        <tbody>
          {
        "".join(
            f'''
          <tr>
            <td><strong>{r.get("experiment", "")}</strong></td>
            <td><span class="badge badge-primary">{r.get("tested", "Yes")}</span></td>
            <td style="color: var(--success);">{r.get("succeeded", "")}</td>
            <td style="color: var(--danger);">{r.get("failed", "None")}</td>
            <td style="color: var(--text-muted);">{r.get("still_unproven", "")}</td>
            <td>{r.get("next_action", "")}</td>
          </tr>
          '''
            for r in suite_result.evidence_log
        )
    }
        </tbody>
      </table>
    </div>

    <footer>
      <p>ToolSpeed — Benchmark & Optimization Framework | Evidence Level: {ev_level} | Runtime: {
        suite_result.total_runtime_sec:.2f}s</p>
    </footer>
  </div>
</body>
</html>
"""
    return html


def generate_json_summary(suite_result: SuiteResult) -> dict[str, Any]:
    """Generate structured JSON representation of suite results."""
    return suite_result.to_dict()


def generate_benchmark_markdown_report(result: dict[str, Any] | Any) -> str:
    """Generate Markdown report for paired benchmark run results."""
    data = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    title = data.get("title", "ToolSpeed Paired Benchmark Suite")
    ev_level = data.get("evidence_level", "replay_integration")
    runtime = data.get("total_runtime_s", 0.0)
    verdict = data.get("overall_verdict", "inconclusive")
    manifest = data.get("manifest") or {}

    lines = [
        f"# {title}",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  ",
        f"**Evidence Level:** `{ev_level}`  ",
        f"**Overall Verdict:** `{'PASSED' if verdict == 'passed' else ('FALSIFIED' if verdict == 'falsified' else 'INCONCLUSIVE')}`  ",
        f"**Total Benchmark Runtime:** {runtime:.2f}s  ",
    ]

    if manifest:
        lines.extend(
            [
                f"**Git Commit:** `{manifest.get('git_sha', 'unknown')}` ({'dirty' if manifest.get('git_dirty') else 'clean'})  ",
                f"**Platform:** `{manifest.get('os_platform', '')}` | Python `{manifest.get('python_version', '')}`  ",
            ]
        )

    lines.extend(
        [
            "",
            "## Paired Workload Evaluations (W1 – W7, E5a)",
            "",
            "| Workload | Comparison | Baseline P95 | Candidate P95 | P95 Speedup | Candidate Success | Status | 95% Bootstrap CI |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )

    for e in data.get("evaluations", []):
        wl = e.get("workload_id", "")
        comp = f"{e.get('candidate_name', '')} vs {e.get('baseline_name', '')}"
        summ = e.get("summary", {})
        verd = e.get("verdict", {})
        status = "✅ PASS" if verd.get("passed") else "❌ FAIL"
        b95 = f"{summ.get('baseline_p95_ms', 0.0):.1f}ms" if summ.get("baseline_p95_ms") is not None else "null"
        c95 = f"{summ.get('candidate_p95_ms', 0.0):.1f}ms" if summ.get("candidate_p95_ms") is not None else "null"
        speedup = f"{summ.get('p95_speedup', 1.0):.2f}x" if summ.get("p95_speedup") is not None else "null"
        succ = (
            f"{summ.get('candidate_success_rate', 1.0):.1%}"
            if summ.get("candidate_success_rate") is not None
            else "null"
        )
        ci = summ.get("p95_reduction_ci")
        ci_str = f"[{ci[0]:.1f}%, {ci[1]:.1f}%]" if ci and ci[0] is not None else "null"
        lines.append(f"| {wl} | {comp} | {b95} | {c95} | {speedup} | {succ} | {status} | {ci_str} |")

    neg = data.get("negative_controls", [])
    if neg:
        lines.extend(
            [
                "",
                "## Negative Control Verification",
                "",
                "| Control | Measured Speedup | Null Check (~1.0x) | Detail |",
                "|---|---|---|---|",
            ]
        )
        for nc in neg:
            c_name = nc.get("control", "")
            sp = f"{nc.get('p95_speedup', 1.0):.2f}x"
            null_pass = "✅ PASS" if nc.get("passed_expected_null") else "❌ FAIL"
            detail = nc.get("detail", "")
            lines.append(f"| {c_name} | {sp} | {null_pass} | {detail} |")

    return "\n".join(lines)


def generate_benchmark_html_dashboard(result: dict[str, Any] | Any) -> str:
    """Generate HTML dashboard for paired benchmark run results."""
    data = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    title = data.get("title", "ToolSpeed Paired Benchmark Suite")
    ev_level = data.get("evidence_level", "replay_integration")
    runtime = data.get("total_runtime_s", 0.0)
    verdict = data.get("overall_verdict", "inconclusive")
    evaluations = data.get("evaluations", [])

    rows_html = ""
    for e in evaluations:
        summ = e.get("summary", {})
        verd = e.get("verdict", {})
        is_pass = verd.get("passed", False)
        b95 = f"{summ.get('baseline_p95_ms', 0.0):.1f} ms" if summ.get("baseline_p95_ms") is not None else "null"
        c95 = f"{summ.get('candidate_p95_ms', 0.0):.1f} ms" if summ.get("candidate_p95_ms") is not None else "null"
        sp = f"{summ.get('p95_speedup', 1.0):.2f}x" if summ.get("p95_speedup") is not None else "null"
        succ = (
            f"{summ.get('candidate_success_rate', 1.0):.1%}"
            if summ.get("candidate_success_rate") is not None
            else "null"
        )
        rows_html += f"""
        <tr>
          <td><strong>{e.get("workload_id", "")}</strong></td>
          <td>{e.get("candidate_name", "")} vs {e.get("baseline_name", "")}</td>
          <td>{b95}</td>
          <td>{c95}</td>
          <td><strong>{sp}</strong></td>
          <td>{succ}</td>
          <td><span class="badge {"badge-success" if is_pass else "badge-danger"}">{"PASS" if is_pass else "FAIL"}</span></td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #0d1117; --card-bg: #161b22; --border: #30363d;
      --text: #c9d1d9; --text-muted: #8b949e; --accent: #58a6ff;
      --success: #3fb950; --danger: #f85149;
    }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); padding: 2rem; margin: 0; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 0.75rem 1rem; border-bottom: 1px solid var(--border); text-align: left; }}
    th {{ color: var(--text-muted); }}
    .badge {{ padding: 0.25rem 0.6rem; border-radius: 12px; font-weight: bold; font-size: 0.85rem; }}
    .badge-success {{ background: rgba(63, 185, 80, 0.2); color: var(--success); }}
    .badge-danger {{ background: rgba(248, 81, 73, 0.2); color: var(--danger); }}
  </style>
</head>
<body>
  <div class="container">
    <h1>{title}</h1>
    <p>Evidence Level: <code>{ev_level}</code> | Runtime: {runtime:.2f}s | Verdict: <strong>{verdict.upper()}</strong></p>
    <div class="card">
      <h2>Workload Performance Matrix</h2>
      <table>
        <thead>
          <tr>
            <th>Workload</th><th>Comparison</th><th>Baseline P95</th><th>Candidate P95</th><th>Speedup</th><th>Success</th><th>Status</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>"""
    return html


def save_benchmark_reports(
    result: BenchmarkRunResult | dict[str, Any] | Any,
    out_dir: str | Path,
) -> dict[str, Path]:
    """Atomically generate, hash, and persist canonical benchmark suite bundle with single manifest."""
    final_out = Path(out_dir).resolve()
    final_out.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = final_out.parent / f"{final_out.name}.staging_{uuid.uuid4().hex[:8]}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        data = result.to_dict() if hasattr(result, "to_dict") else dict(result)

        # 1. Traces JSONL files & Cases
        baseline_traces_path = staging_dir / "baseline-traces.jsonl"
        candidate_traces_path = staging_dir / "candidate-traces.jsonl"
        controls_traces_path = staging_dir / "controls-traces.jsonl"
        cases_path = staging_dir / "cases.jsonl"

        with (
            open(baseline_traces_path, "w", encoding="utf-8") as f_b,
            open(candidate_traces_path, "w", encoding="utf-8") as f_c,
            open(cases_path, "w", encoding="utf-8") as f_cases,
        ):
            if hasattr(result, "evaluations") and result.evaluations:
                for ev in result.evaluations:
                    for idx, r_c in enumerate(ev.candidate_results):
                        f_c.write(json.dumps(r_c.to_dict()) + "\n")
                        f_cases.write(
                            json.dumps({"workload_id": ev.workload_id, "trial_index": idx, "task_id": r_c.task_id})
                            + "\n"
                        )
                    for r_b in ev.baseline_results:
                        f_b.write(json.dumps(r_b.to_dict()) + "\n")
            else:
                f_b.write(json.dumps({"info": "baseline_traces"}) + "\n")
                f_c.write(json.dumps({"info": "candidate_traces"}) + "\n")
                f_cases.write(json.dumps({"info": "cases"}) + "\n")

        with open(controls_traces_path, "w", encoding="utf-8") as f_ctrl:
            if hasattr(result, "negative_controls") and result.negative_controls:
                for nc in result.negative_controls:
                    f_ctrl.write(json.dumps(nc) + "\n")
            else:
                f_ctrl.write(json.dumps({"info": "controls"}) + "\n")

        # 2. Frozen protocol copies
        protocol_path = staging_dir / "protocol.json"
        bm_plan_path = staging_dir / "benchmark-plan.json"
        repo_protocol = Path("benchmark-plans/tool-speed-v1.1.json")
        if not repo_protocol.exists():
            repo_protocol = Path("benchmark-plans/tool-speed-v1.json")
        if repo_protocol.exists():
            proto_content = repo_protocol.read_text(encoding="utf-8")
        else:
            proto_content = json.dumps({"plan_id": "tool-speed-v1.1", "version": "1.1.0"}, indent=2)
        protocol_path.write_text(proto_content, encoding="utf-8")
        bm_plan_path.write_text(proto_content, encoding="utf-8")

        # 3. Compute early payload hashes
        proto_hash = compute_file_sha256(protocol_path)
        cases_hash = compute_file_sha256(cases_path)
        c_trace_hash = compute_file_sha256(candidate_traces_path)
        b_trace_hash = compute_file_sha256(baseline_traces_path)
        ctrl_trace_hash = compute_file_sha256(controls_traces_path)

        # 4. Reports (MD and HTML)
        md_path = staging_dir / "report.md"
        md_content = generate_benchmark_markdown_report(result)
        md_path.write_text(md_content, encoding="utf-8")

        html_path = staging_dir / "report.html"
        html_content = generate_benchmark_html_dashboard(result)
        html_path.write_text(html_content, encoding="utf-8")

        # 5. Falsification Verdict JSON
        falsification_path = staging_dir / "falsification.json"
        falsification_data = {
            "title": data.get("title"),
            "evidence_level": data.get("evidence_level"),
            "overall_verdict": data.get("overall_verdict"),
            "total_runtime_s": data.get("total_runtime_s"),
            "verdicts": [
                {
                    "workload_id": e.get("workload_id"),
                    "comparison": f"{e.get('candidate_name')} vs {e.get('baseline_name')}",
                    "verdict": e.get("verdict"),
                    "summary": e.get("summary"),
                }
                for e in data.get("evaluations", [])
            ],
        }
        falsification_path.write_text(strict_json_dumps(falsification_data, indent=2), encoding="utf-8")

        # 6. Update manifest with payload hashes
        manifest = data.get("manifest") or {}
        manifest["benchmark_plan_hash"] = proto_hash
        manifest["benchmark_config_hash"] = proto_hash
        manifest["workload_fixture_hash"] = cases_hash
        manifest["cases_hash"] = cases_hash
        manifest["fixture_manifest_hash"] = cases_hash
        manifest["candidate_trace_hash"] = c_trace_hash
        manifest["baseline_trace_hash"] = b_trace_hash
        manifest["raw_trace_hash"] = c_trace_hash
        manifest["controls_trace_hash"] = ctrl_trace_hash
        manifest["code_git_sha"] = manifest.get("code_git_sha") or manifest.get("git_sha", "unknown")
        manifest["trial_count"] = manifest.get(
            "trial_count",
            len(data.get("evaluations", [{}])[0].get("candidate_results", [])) if data.get("evaluations") else 0,
        )

        # 7. Write result.json
        result_path = staging_dir / "result.json"
        result_path.write_text(strict_json_dumps(data, indent=2), encoding="utf-8")
        res_hash = compute_file_sha256(result_path)
        manifest["result_hash"] = res_hash

        data["manifest"] = manifest
        result_path.write_text(strict_json_dumps(data, indent=2), encoding="utf-8")
        bm_result_path = staging_dir / "benchmark_result.json"
        bm_result_path.write_text(strict_json_dumps(data, indent=2), encoding="utf-8")

        # 8. File hashes for all bundle artifacts (excluding manifest.json and bundle.sha256)
        file_hashes: dict[str, str] = {
            "report.md": compute_file_sha256(md_path),
            "report.html": compute_file_sha256(html_path),
            "baseline-traces.jsonl": b_trace_hash,
            "candidate-traces.jsonl": c_trace_hash,
            "controls-traces.jsonl": ctrl_trace_hash,
            "cases.jsonl": cases_hash,
            "protocol.json": proto_hash,
            "benchmark-plan.json": proto_hash,
            "falsification.json": compute_file_sha256(falsification_path),
            "result.json": compute_file_sha256(result_path),
            "benchmark_result.json": compute_file_sha256(bm_result_path),
        }
        manifest["file_hashes"] = file_hashes

        # 9. Manifest JSON
        manifest_path = staging_dir / "manifest.json"
        manifest_path.write_text(strict_json_dumps(manifest, indent=2), encoding="utf-8")

        # 10. Checksum bundle.sha256 file
        all_checksums = dict(file_hashes)
        all_checksums["manifest.json"] = compute_file_sha256(manifest_path)
        bundle_sha_path = staging_dir / "bundle.sha256"
        checksum_lines = [f"{hash_val}  {fname}" for fname, hash_val in sorted(all_checksums.items())]
        bundle_sha_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

        # 11. Atomic swap / move staging dir to final destination
        if final_out.exists():
            shutil.rmtree(final_out)
        staging_dir.rename(final_out)

        return {
            "json": final_out / "result.json",
            "markdown": final_out / "report.md",
            "html": final_out / "report.html",
            "manifest": final_out / "manifest.json",
            "bundle_sha": final_out / "bundle.sha256",
        }
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def save_all_reports(
    suite_result: SuiteResult,
    out_dir: str | Path,
) -> dict[str, Path]:
    """Generate and save all simulation report artifacts (MD, HTML, JSON, CSVs, SVGs)."""
    p_out = Path(out_dir)
    p_out.mkdir(parents=True, exist_ok=True)

    json_path = p_out / "summary_report.json"
    suite_result.save_json(json_path)

    md_path = p_out / "EVIDENCE_LOG.md"
    md_content = generate_markdown_evidence_log(suite_result)
    md_path.write_text(md_content, encoding="utf-8")

    html_path = p_out / "dashboard.html"
    html_content = generate_html_dashboard(suite_result)
    html_path.write_text(html_content, encoding="utf-8")

    saved_csvs = suite_result.save_csvs(p_out)

    charts_dir = p_out / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    wl_svg_path = charts_dir / "workload_speedup.svg"
    wl_svg_path.write_text(
        generate_workload_bar_chart([w.to_dict() for w in suite_result.workloads.values()]),
        encoding="utf-8",
    )

    artifacts = {
        "json": json_path,
        "markdown": md_path,
        "html": html_path,
        "workload_svg": wl_svg_path,
    }
    for csv in saved_csvs:
        artifacts[csv.name] = csv

    return artifacts
