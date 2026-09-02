# ToolSpeeder PR #1: Scientific Integrity Repair Status

**Document Metadata:**
- **Repository:** `RT123-new/ToolSpeeder`
- **Pull Request:** `#1` (Draft, Unmerged)
- **Base Branch:** `main` (`1d3b3a61afefcbeb64c3015579ae1d66107e8450`)
- **Working Branch:** `repair/benchmark-integrity-runtime-safety`
- **Current Status:** Software integrity repair and confirmatory benchmarking complete.
- **Current CI Status:** Clean (Ruff, Format, Pytest 220/220 passing).
- **Current Confirmatory Evidence:** Executed on Replay Backend across 3 seeds (42, 137, 2026), $n=1000$ trials each.
- **Replay/Local Confirmatory Outputs:** Structurally sealed bundles with atomic SHA-256 byte manifests.
- **Authoritative Scientific Verdict:** `INTEGRITY REPAIR COMPLETED — CONFIRMATORY FALSIFICATION REPORTED`

---

## 1. Quality Gates Status at Exact Head

- **Ruff Linter & Formatter:** Clean across entire codebase (`All checks passed!`, 84 files formatted).
- **Mypy Type Checker:** 0 errors across 63 files.
- **Pytest Suite:** 220 passing tests, 0 failing across all 11 test modules.
- **Regression Suite:** 45 passing regression tests in `tests/test_review_findings.py` verifying findings A–O.
- **Statement Coverage:** >86% across package statements.
- **Packaging Build:** Wheel and sdist build cleanly; `importlib.resources` packages protocol JSON fixtures into wheel.

---

## 2. Findings Ledger & Implementation Work Breakdown

| Finding | Area | Status | Resolution Detail |
| :--- | :--- | :--- | :--- |
| **A** | Protocol Harness | RESOLVED | Harness protocol-driven; split W7 into W7_SAFETY (single mutation invariant) and W7_LATENCY; positive sensitivity control measures actual execution delay. |
| **B** | CLI Falsify | RESOLVED | `cmd_falsify` recomputes metrics and hypothesis thresholds directly from raw JSONL traces; rejects forged summaries; gates on sample size. |
| **C** | Bundle Hashing | RESOLVED | Embedded manifest includes `file_hashes`; atomic two-phase write of `result.json` ensures manifest digest equals on-disk file digest; `bundle.sha256` signed. |
| **D** | Oracle Separation | RESOLVED | `BenchmarkCase` strictly separates model-visible task from validation oracle; enforces tool sequence, exact arguments, and state transitions. |
| **E** | Authority Context | RESOLVED | `ApprovalIssuer` generates HMAC-SHA256 capability tokens out-of-band; prevents self-authenticating grant injection from `task.metadata`. |
| **F** | Required Metrics | RESOLVED | `MetricSummary` defaults missing metrics to honest `None`; fails closed when required CCL or speedup metrics are null. |
| **G** | Local Workloads | RESOLVED | `LocalWallClockBackend` creates isolated SQLite database files per arm and trial; prevents shared state accumulation. |
| **H** | E4 Parser | RESOLVED | `IncrementalCommitParser` enforces syntax closure on non-empty fragments, schema type verification, and semantic reconciliation against final tool calls. |
| **I** | E2 Fusion | RESOLVED | Rejects untrusted/injected workflow objects from task metadata; matches workflows based on explicit ID or combined prompt intent and context. |
| **J** | E3 Speculation | RESOLVED | Verifies `is_concurrency_safe` contract on adapters; wraps speculation cancellation to safely swallow `asyncio.CancelledError`. |
| **K** | E5a Codec | RESOLVED | Embeds schema identity hash in bytecode header; creates symmetric `CodecConfig` for fair JSON comparison. |
| **L** | Cache Eviction | RESOLVED | `ToolResultCache` implements genuine LRU eviction using `last_accessed_at`; scopes entries by `(tenant, authority)`; rejects expired entries under strict mode. |
| **M** | Composite | RESOLVED | Dispatches read-only tools through cache lookup; provides `has_cache_lookup_in_dispatch_path()`. |
| **N** | Local Tools | RESOLVED | `SafeSubprocessSandbox` uses `start_new_session=True` and kills process tree with `os.killpg`; `AsyncLocalFileIOTool` enforces strict directory containment. |
| **O** | Packaging | RESOLVED | Packaged protocol definitions in `toolspeed/resources/`; loaded via `importlib.resources.files`; schema validated via `jsonschema`. |

---

## 3. Confirmatory Benchmarking Results (Replay Backend, n=1,000, 3 Seeds)

Three independent confirmatory runs were executed at seeds 42, 137, and 2026 with $n=1000$ paired trials each:

| Workload | Comparison | Baseline P95 | Candidate P95 | Measured Speedup | Success Rate | Hypothesis Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **W1** | DAGScheduler vs SyncReActScheduler | 150.0ms | 70.0ms | **2.14x** | 100.0% | **PASS** |
| **W2** | JITFusionScheduler vs SyncReActScheduler | 115.0ms | 40.0ms | **2.87x** | 100.0% | **PASS** |
| **W3** | SpeculativeReadScheduler vs SyncReActScheduler | 75.0ms | 75.0ms | **1.00x** | 100.0% | **FALSIFIED** (<1.25x target) |
| **W4** | CacheScheduler vs SyncReActScheduler | 75.0ms | 75.0ms | **1.00x** | 100.0% | **FALSIFIED** (<1.30x target) |
| **W5** | CommitHorizonScheduler vs SyncReActScheduler | 100.0ms | 82.5ms | **1.21x** | 100.0% | **PASS** |
| **W6** | CompositeScheduler vs SyncReActScheduler | 75.0ms | 75.0ms | **1.00x** | 100.0% | **FALSIFIED** (<1.25x target) |
| **W7_SAFETY** | CompositeScheduler vs CompositeScheduler | 75.0ms | 75.0ms | **1.00x** | 100.0% | **PASS** (Single mutation invariant) |
| **W7_LATENCY** | CompositeScheduler vs SyncReActScheduler | 75.0ms | 75.0ms | **1.00x** | 100.0% | **PASS** (Overhead bounded) |
| **E5a** | ActionBytecodeScheduler vs SyncReActScheduler | 70.0ms | 70.0ms | **1.00x** | 100.0% | **FALSIFIED** (Scheduler vs Codec) |

### Negative & Positive Control Results:
- **E1 Parallelism Disabled:** 1.00x speedup -> **PASS** (Null effect verified)
- **E2 Fusion Disabled:** 1.00x speedup -> **PASS** (Null effect verified)
- **E3 Speculation Disabled:** 1.00x speedup -> **PASS** (Null effect verified)
- **E4 Early Dispatch Disabled:** 1.00x speedup -> **PASS** (Null effect verified)
- **Cache Disabled:** 1.00x speedup -> **PASS** (Null effect verified)
- **Positive Sensitivity Control:** 2.00x speedup -> **PASS** (Derived from execution, not hard-coded)
