# ToolSpeeder PR #1 — Exact-Head Integrity Review (`e3c974b`)

**Document Metadata:**
- **Repository:** `RT123-new/ToolSpeeder`
- **Pull Request:** `#1` (Draft, Unmerged, Auto-merge disabled)
- **Base Branch:** `main` (`1d3b3a61afefcbeb64c3015579ae1d66107e8450`)
- **Working Branch:** `repair/benchmark-integrity-runtime-safety`
- **Reviewed Head SHA:** `e3c974be61d58df0c2870a098099f1371a161c10`
- **Merge-Base SHA:** `1d3b3a61afefcbeb64c3015579ae1d66107e8450`
- **Working Tree State:** Clean (`git status --short` empty)
- **Review Date:** 2026-09-02
- **Authoritative Scientific Verdict:** `INTEGRITY REPAIR INCOMPLETE`

---

## 1. Executive Summary & Starting State

At the reviewed head `e3c974be61d58df0c2870a098099f1371a161c10`, ToolSpeeder PR #1 contains 21 commits (81 changed files, ~13,610 additions, ~2,986 deletions). 

### Conflict of Claims Surface
- **PR #1 Body Claim:** "INTEGRITY REPAIR COMPLETED — EVIDENCE READY. Local Smoke Status: GREEN on Replay & Local Wall-Clock backends (100% bundle validation pass)."
- **README.md Status:** "Status: benchmark-integrity repair in progress. Current benchmark smoke is failing bundle validation. No current-head canonical replay or local evidence bundle exists. Live evidence has not been collected."
- **Hosted CI Status:** Compatibility Matrix (Python 3.10, 3.11, 3.12, 3.13) and Benchmark Smoke passed on runs `33241090267` (push) and `33241091847` (pull_request). However, the "Full Evidence Sweep (Replay 1000 & Local 200)" job was **SKIPPED** in both runs, and **zero evidence artifacts** were uploaded.
- **Scientific Verdict:** Green CI establishes software/test passage, not scientific validity. The claim `INTEGRITY REPAIR COMPLETED — EVIDENCE READY` is premature and refuted by the code at this SHA.

---

## 2. Independent Reproduction of Existing Quality Gates (Phase 1)

All commands executed on macOS (Darwin 27.0.0 arm64, Apple M-series), Python 3.13.15, `uv` 0.11.25:

| Gate | Command | Result at `e3c974b` | Status |
| :--- | :--- | :--- | :--- |
| Lockfile Check | `uv lock --check` | Clean, 54 packages resolved in 6ms | PASSED |
| Dependency Sync | `uv sync --frozen --all-extras` | Clean (39 packages checked) | PASSED |
| Linter | `uv run ruff check .` | Clean ("All checks passed!") | PASSED |
| Formatter | `uv run ruff format --check .` | Clean ("78 files already formatted") | PASSED |
| Type Checker | `uv run mypy toolspeed tests` | Exit 0 ("Success: no issues found in 63 source files") | PASSED (Note: permissive config) |
| Test Suite | `uv run coverage run -m pytest -q` | Exactly **175 passed** in 164.30s (0:02:44) | PASSED |
| Code Coverage | `uv run coverage report -m --fail-under=80` | Exactly **86% statement coverage** (9,610 stmts, 1,314 missed) | PASSED |
| Packaging Build | `uv build` | Successfully built wheel & sdist | PASSED |
| Twine Check | `uv run twine check dist/*` | Wheel & sdist passed twine checks | PASSED |
| Wheel Isolated Install | `uv pip install dist/*.whl` & import outside repo | `FileNotFoundError: benchmark-plans/tool-speed-v1.1.json` on import | **FAILED** (Finding O reproduced) |

---

## 3. Systematic Findings Ledger (A through O)

Every blocking finding was audited and reproduced directly against the source code at `e3c974be61d58df0c2870a098099f1371a161c10`:

### Finding A: Frozen Protocol Does Not Drive Benchmark
- **Status:** `REPRODUCED`
- **Evidence:** `toolspeed/benchmarks/harness.py`:
  - Lines 104–123 & 342–456: Harness hard-codes comparisons against `SyncReActScheduler` rather than the protocol-specified primary attribution baselines (e.g., negative controls compare `DAGScheduler(parallelism_enabled=False)` vs `SyncReActScheduler` instead of an identity control).
  - Lines 240–250: Global `FROZEN_POLICY` is hard-coded into evaluation thresholds rather than reading mechanism-specific thresholds from `benchmark-plans/tool-speed-v1.1.json`.
  - Lines 458–468: Positive sensitivity control is hard-coded as literal dictionary:
    ```python
    {"control": "Positive_sensitivity_injected_50pct_speedup", "p95_speedup": 2.00, "measured_speedup": 2.00, "passed_expected_null": True, "null_check": "PASS"}
    ```
    rather than being executed and measured through the pipeline.
  - Workload W7 is evaluated as a single latency check rather than splitting into `W7_SAFETY` and `W7_LATENCY`.

### Finding B: `falsify` Does Not Perform Raw-Trace Recomputation
- **Status:** `REPRODUCED`
- **Evidence:** `toolspeed/cli.py` lines 284–315:
  - The CLI prints `"Recomputing statistical metrics and hypothesis checks from raw JSONL traces..."`, but immediately loads `evaluations = data.get("evaluations", [])` from `result.json` and reads `is_pass = verd.get("passed", False)` and `summ.get("p95_speedup", 0.0)`.
  - It does not parse or recompute from `candidate-traces.jsonl` or `baseline-traces.jsonl`.

### Finding C: Bundle Hashing Is Internally Inconsistent
- **Status:** `REPRODUCED`
- **Evidence:** `toolspeed/visualization/report.py` lines 780–840:
  - Line 799–801: Writes `result.json`, computes `compute_file_sha256(result_path)`, and assigns to `manifest["result_hash"]`.
  - Line 804: Rewrites `result.json` with `data["manifest"] = manifest`, immediately mutating its bytes and invalidating the hash.
  - Line 822: Populates `manifest["file_hashes"]` with post-mutation file hashes and writes external `manifest.json`, but does NOT update the embedded `manifest` inside `result.json`.
  - Line 837: Deletes existing destination directory via `shutil.rmtree(final_out)` before renaming.
  - Line 789: Assigns `manifest["raw_trace_hash"] = c_trace_hash`, hashing only the candidate trace while omitting baseline and control traces.

### Finding D: Oracle Separation Is Not a Strict Whitelist
- **Status:** `REPRODUCED`
- **Evidence:** 
  - `toolspeed/core/types.py` lines 170–172:
    ```python
    if k in MODEL_VISIBLE_METADATA_WHITELIST or not any(
        p in str(v).lower() for p in PROHIBITED_METADATA_SUBSTRINGS
    ):
        result[k] = v
    ```
    Non-whitelisted keys whose string values do not match prohibited substrings pass through into `AgentTask.metadata`.
  - `toolspeed/core/types.py` lines 437–523 (`validate_execution`):
    - `expected_tool_sequence` is never inspected.
    - Expected tool arguments are never verified.
    - `final_state: StateSnapshot | None = None` parameter is accepted but never compared or evaluated.
    - Lines 504–510: Checks only that `expected_final_value` keys match `final_output` (allows partial supersets).

### Finding E: Approval Grants Are Unsigned Data Objects
- **Status:** `REPRODUCED`
- **Evidence:** `toolspeed/core/types.py` lines 283–351:
  - `ApprovalGrant.create()` can be called by arbitrary code asserting `authority="trusted_system"`.
  - Argument fingerprint is truncated to 16 hex characters (`[:16]`).
  - Expiry uses `time.perf_counter()` process-relative timestamps.
  - `toolspeed/schedulers/base.py` lines 204–206:
    ```python
    if isinstance(task, Task) and "approval_grant" in task.metadata:
        grant = task.metadata["approval_grant"]
        if isinstance(grant, ApprovalGrant):
            auth_ctx.trusted_grants[grant.approval_id] = grant
    ```
    Grants are imported directly from `task.metadata`, bypassing trusted external boundaries.

### Finding F: Required Metrics Receive Favourable Fallback Defaults
- **Status:** `REPRODUCED`
- **Evidence:** `toolspeed/benchmarks/harness.py`:
  - Lines 241, 249, 258, 307: Missing costs fallback to `1.0` via `(summary.cost_multiplier or 1.0)`.
  - Missing P50 speedups fallback to `1.0` via `(summary.p50_speedup or 1.0)`.
  - Pass/fail decisions do not fail closed when required scientific metrics are null.

### Finding G: Workload Realism & Contamination Defects
- **Status:** `REPRODUCED`
- **Evidence:**
  - `toolspeed/benchmarks/local_backend.py`: W2 local database uses SQLite file paths that can be populated repeatedly without pristine snapshot isolation.
  - Sub-millisecond delays in local backend (~1ms) are susceptible to event-loop scheduling jitter and OS noise.

### Finding H: E4 Incremental Commit Parser Is Premature
- **Status:** `REPRODUCED`
- **Evidence:** `toolspeed/schedulers/e4_commit_horizon.py` lines 129–138:
  - `if raw_fragment and raw_fragment.strip(): ...` allows empty raw fragments to bypass syntax closure and duplicate key checks.
  - Does not verify that parsed JSON fragment equals `raw_call.arguments`.
  - Dispatches mutable `ToolCall` objects.

### Finding I: E2 Declarative JIT Fusion Security & Implicit Activation
- **Status:** `REPRODUCED`
- **Evidence:** `toolspeed/schedulers/e2_jit_fusion.py`:
  - Lines 255–258: Accepts a `DeclarativeWorkflow` instance directly from `ctx.task.metadata["declarative_workflow"]`.
  - Line 273: Implicitly auto-activates `user_orders` fusion whenever `"user_id" in ctx.task.context` unless explicitly disabled.

### Finding L: Phase 2 Cache Eviction Is FIFO, Not LRU
- **Status:** `REPRODUCED`
- **Evidence:** `toolspeed/schedulers/phase2_cache.py` lines 106–112:
  - `oldest_key = min(self._exact_store.keys(), key=lambda k: self._exact_store[k].created_at)`
  - Eviction is strictly creation-time (FIFO), not access-time (LRU).
  - Separate bounding on `_exact_store` and `_semantic_store` allows cache to hold up to `2 * max_entries`.
  - Lines 77, 89: Relaxed stale data is returned as valid hits.

### Finding N: Sandbox Isolation Claims Exceed Implementation
- **Status:** `REPRODUCED`
- **Evidence:** 
  - `toolspeed/adapters/live_tools.py` lines 196–204: `SafeSubprocessSandbox` executes with `shell=True` inside `asyncio.to_thread`. Cancellation does not terminate the subprocess process tree, and no memory cap exists.
  - Lines 284–288: `FileSandbox._resolve_safe` uses `str(target).startswith(str(self._base_dir))` which is vulnerable to prefix confusion (e.g. `/tmp/dir_evil` starts with `/tmp/dir`).
  - `SECURITY.md` claims SIGKILL process-tree management and memory caps that do not exist.

### Finding O: Packaging Breaks Outside Repository
- **Status:** `REPRODUCED`
- **Evidence:** `toolspeed/core/protocol.py` line 130:
  - Resolves `benchmark-plans/tool-speed-v1.1.json` relative to the current working directory.
  - When installed via wheel outside the repository root, importing `toolspeed.core.protocol` raises `FileNotFoundError`.

---

## 4. Operational & CI Status

- **Commit Head:** `e3c974be61d58df0c2870a098099f1371a161c10`
- **Compatibility Matrix (Py 3.10–3.13):** Green in CI.
- **PR Status:** Draft, unmerged.
- **Scientific Evidence:** Uncollected for full sweep.
- **Verdict:** `INTEGRITY REPAIR INCOMPLETE`.
