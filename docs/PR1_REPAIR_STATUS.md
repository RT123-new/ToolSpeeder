# ToolSpeeder PR #1: Scientific Integrity Repair Status

**Document Metadata:**
- **Repository:** `RT123-new/ToolSpeeder`
- **Pull Request:** `#1` (Draft, Unmerged)
- **Base Branch:** `main` (`1d3b3a61afefcbeb64c3015579ae1d66107e8450`)
- **Working Branch:** `repair/benchmark-integrity-runtime-safety`
- **Current Status:** Integrity repair in progress.
- **Current CI Status:** Compatibility and smoke checks pass on `e3c974be61d58df0c2870a098099f1371a161c10`.
- **Current Confirmatory Evidence:** Not yet collected under a valid frozen protocol.
- **Replay/Local Smoke Outputs:** Operational tests only, not verdict-eligible.
- **Live Evidence:** Absent.
- **Authoritative Scientific Verdict:** `INTEGRITY REPAIR INCOMPLETE`

---

## 1. Quality Gates Status at Exact Head

- **Ruff Linter & Formatter:** Clean.
- **Mypy Type Checker:** 0 errors across 63 files under permissive config.
- **Pytest Suite:** 175 passing tests in baseline suite.
- **Regression Suite:** 41 failing red tests in `tests/test_review_findings.py` reproducing findings A–O.
- **Statement Coverage:** 86% across package statements.
- **Packaging Build:** Wheel and sdist build successfully; standalone wheel import outside repo requires resource packaging fix.

---

## 2. Findings Ledger & Implementation Work Breakdown

| Finding | Area | Status | Description |
| :--- | :--- | :--- | :--- |
| **A** | Protocol Harness | OPEN | Harness hard-codes comparison against SyncReAct; unsplit W7; positive control hardcoded. |
| **B** | CLI Falsify | OPEN | `falsify` does not parse raw traces; reads stored `verdict` from `result.json`. |
| **C** | Bundle Hashing | OPEN | Embedded manifest lacks `file_hashes`; `raw_trace_hash` omits baseline/controls; destination deleted before move. |
| **D** | Oracle Separation | OPEN | Value-based whitelist fallback leaks non-prohibited keys; `validate_execution` ignores sequences/args/state. |
| **E** | Authority Context | OPEN | `ApprovalGrant` is self-asserted and imported from `task.metadata`; truncated 16-char digest. |
| **F** | Required Metrics | OPEN | Missing cost defaults to 1.0; pass/fail does not fail closed on missing required metrics. |
| **G** | Local Workloads | OPEN | SQLite database state can accumulate between trials; local delays susceptible to OS jitter. |
| **H** | E4 Parser | OPEN | Bypasses syntax closure on empty fragments; dispatches mutable calls; lacks argument match proof. |
| **I** | E2 Fusion | OPEN | Auto-matches on context `user_id`; accepts task-supplied `DeclarativeWorkflow` objects. |
| **J** | E3 Speculation | OPEN | Fixed/uncalibrated predictor confidence; lacks adapter concurrency contract. |
| **K** | E5a Codec | OPEN | Action bytecode packet does not bind schema identity hash; lacks JSON direct benchmark. |
| **L** | Cache Eviction | OPEN | Eviction is creation-time FIFO, not LRU; permits relaxed stale hits in strict evidence. |
| **M** | Composite | OPEN | Overlapping dispatch ownership; bypasses cache lookup in execution path. |
| **N** | Local Tools | OPEN | Subprocess uses `shell=True` without process-group kill; file tool has prefix-confusion escape. |
| **O** | Packaging | OPEN | `load_frozen_protocol` fails on wheel install outside repo root; schema validation incomplete. |

---

## 3. Claim Audit Summary

All claims of `INTEGRITY REPAIR COMPLETED — EVIDENCE READY`, full raw-trace recomputation, true LRU cache eviction, and isolated security sandboxing have been retracted and qualified across all documentation and PR surfaces.
