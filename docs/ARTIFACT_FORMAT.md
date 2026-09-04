# ToolSpeed Artifact Bundle Format

This document specifies the format, file schema, and validation rules for ToolSpeed benchmark and simulation artifact bundles.

## Directory Structure

A complete benchmark artifact bundle contains the following files:

```
artifacts/<backend>/
├── benchmark_result.json  # Complete structured JSON bundle with evaluations, summaries, and manifest
├── report.md              # Human-readable Markdown report
├── report.html            # Standalone visual HTML dashboard
```

## `benchmark_result.json` Schema

```json
{
  "title": "ToolSpeed Paired Benchmark Suite (Replay Backend)",
  "evidence_level": "replay_integration",
  "overall_verdict": "passed",
  "total_runtime_s": 25.42,
  "manifest": {
    "manifest_version": "1.0.0",
    "git_sha": "1d3b3a61afefcbeb64c3015579ae1d66107e8450",
    "git_dirty": false,
    "os_platform": "Darwin-arm64",
    "python_version": "3.10.16",
    "command": "toolspeed benchmark --backend replay",
    "evidence_level": "replay_integration",
    "timestamp_utc": "2026-08-28T08:00:00Z",
    "seed": 20260825
  },
  "evaluations": [
    {
      "workload_id": "W1",
      "baseline_name": "SyncReActScheduler",
      "candidate_name": "DAGScheduler",
      "evidence_level": "replay_integration",
      "trials": 50,
      "summary": {
        "baseline_p95_ms": 205.1,
        "candidate_p95_ms": 94.0,
        "p95_speedup": 2.18,
        "candidate_success_rate": 1.0,
        "p95_reduction_ci": [51.2, 56.8]
      },
      "verdict": {
        "experiment_id": "W1",
        "passed": true,
        "falsified": false,
        "state": "passed",
        "summary": "P95 speedup: 2.18x, Candidate Success: 100.0%"
      }
    }
  ],
  "negative_controls": [
    {
      "control": "E1_disabled",
      "p95_speedup": 0.99,
      "passed_expected_null": true,
      "detail": "Proves disabled E1 produces ~1.0x speedup as expected"
    }
  ]
}
```
