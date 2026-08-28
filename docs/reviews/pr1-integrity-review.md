# ToolSpeeder PR #1 — Comprehensive Scientific Integrity and Runtime Safety Review

**Document Metadata:**
- **Repository:** `RT123-new/ToolSpeeder`
- **Pull Request:** `#1` (Draft)
- **Base Branch:** `main` (`1d3b3a61afefcbeb64c3015579ae1d66107e8450`)
- **Working Branch:** `repair/benchmark-integrity-runtime-safety`
- **Reviewed Head SHA:** `2066ae80e40905e7ffca50bb148d74700094101a`
- **Initial Review Date:** 2026-08-28

---

## 1. Claims Made in Previous Handoff

The previous PR handoff claimed:
1. **Verdict:** `REPLAY/LOCAL VALIDATED, LIVE UNPROVEN`.
2. **Test Suite:** "115 passed" static badge in `README.md` and complete test suite success across the matrix.
3. **Paired Benchmarks:** Complete paired benchmark execution on both Replay and Local wall-clock backends with W1–W5 evaluations.
4. **Hypothesis Status:** "Central hypothesis stands: All workloads achieved >=10% P95 CCL gain with zero safety loss."
5. **Runtime Safety:** E1–E5 optimizations, `ToolExecutor`, `RateLimiter`, and `GuardrailTracker` fully secured and leak-free.

---

## 2. Findings Verified from Code Inspection

Independent static code inspection of commit `2066ae80e40905e7ffca50bb148d74700094101a` revealed seven critical integrity defects:

1. **E4 Child Cancellation Escape in Python 3.10 / 3.11:**
   In `toolspeed/schedulers/e4_commit_horizon.py`, internally cancelled early-dispatch child tasks were caught with `except Exception:`. In Python 3.10 and 3.11, `asyncio.CancelledError` inherits from `BaseException`, causing cancellation to escape and fail the parent scheduler and test suite (`test_adversarial_commit_horizon_post_dispatch_argument_mutation`).
2. **Local Backend Decoupled from Harness:**
   `toolspeed/benchmarks/harness.py` imported `LocalWallClockBackend`, but only ever instantiated `ReplayBackend`. Every workload call hardcoded `self.replay_backend.create_workload_environment(w)`. Consequently, running `toolspeed benchmark --backend local` merely altered an output label without executing any local wall-clock code.
3. **Artifact Overwrite with Synthetic Simulation Data:**
   In `toolspeed/cli.py`, `cmd_benchmark`, `cmd_report`, and `cmd_falsify` instantiated `SuiteRunner` (synthetic simulation) and overwrote the output directory (including `artifacts/replay` and `artifacts/local`) with synthetic charts, CSVs, and markdown reports.
4. **Committed Local Artifact Falsified Workloads:**
   The committed `artifacts/local/benchmark_result.json` reported W2 at 0.70x, W4 at 1.00x, and W5 at 0.56x (all failing the >=1.05x target), while the generated `EVIDENCE_LOG.md` claimed the central hypothesis was confirmed using synthetic numbers.
5. **Missing Workloads in Benchmark Harness:**
   `toolspeed/benchmarks/harness.py` evaluated only W1–W5. W6 (cold starts), W7 (side effects/approvals), and E5a (transport codec) were completely omitted from the paired benchmark runner.
6. **Flawed Statistical Bootstrapping and Favorable Defaults:**
   `toolspeed/experiments/runner.py` computed bootstrap confidence intervals on the median of per-trial differences rather than resampling paired indices to compute the distribution of P95 speedup. Missing metrics defaulted to favorable values (`success=1.0`, `cost_multiplier=1.0`, `wasted_calls=0.0`, `rate_limit_errors=0.0`, `semantic_mutations=0.0`) instead of returning `None` and marking hypotheses `INCONCLUSIVE`.
7. **Runtime Safety Gaps Across Schedulers:**
   - E1 (`DAGScheduler`): Silently mutated duplicate call IDs (`f"{call_id}_{len(self.nodes)}"`) instead of rejecting them; ignored self-references in cycle detection; `ResolvedArguments.__iter__` broke normal dict iteration.
   - E2 (`JITFusionScheduler`): Relied on arbitrary Python callables (`check_fn`, `output_constructor`, `execute_fn`) and automatically set `is_approved=True` for mutative tools.
   - E3 (`SpeculativeReadScheduler`): Overwrote references to running draft tasks on subsequent turns without clean cancellation and await.
   - E4 (`IncrementalCommitParser`): Did not perform true streaming incremental parsing; accepted unverified fragments.
   - E5a (`ActionBytecodeCodec`): Dynamically registered opcodes during untrusted encode; did not reject duplicate keys or enforce packet bounds on decode.
   - `RateLimiter`: Concurrency slots were acquired before and held while waiting for token-bucket rate limits.

---

## 3. Findings Verified from Live CI

GitHub Actions run for commit `2066ae80e40905e7ffca50bb148d74700094101a`:
- **Python 3.10 Job:** `FAILURE` in `test_adversarial_commit_horizon_post_dispatch_argument_mutation` (`asyncio.CancelledError` unhandled).
- **Python 3.11 Job:** `FAILURE` in `test_adversarial_commit_horizon_post_dispatch_argument_mutation`.
- **Python 3.12 Job:** `SUCCESS` (cancellation behavior difference in test runner masked error).
- **Python 3.13 Job:** `SUCCESS`.

---

## 4. Claims Still Unproven

1. **E5b (Direct Action-Token Generation):** Unimplemented and unproven for real-world models. Remains strictly marked `UNIMPLEMENTED` and `INCONCLUSIVE`.
2. **Live LLM Endpoint Validation:** Live commercial LLM and external API validation has not been performed in this phase. The evidence level for live claims remains unproven.

---

## 5. Corrections Implemented in This Continuation Session

1. **Cross-Version Cancellation & Async Cleanup:**
   - Implemented `cancel_and_await` and `TaskTracker` utilities in `toolspeed/schedulers/base.py`.
   - Repaired `CommitHorizonScheduler`, `SpeculativeReadScheduler`, and `DAGScheduler` to consume internal child cancellations cleanly while cleanly propagating external cancellations.
2. **Complete CLI Isolation:**
   - `simulate`: Emits `evidence_level: synthetic` with real-world hypothesis status `INCONCLUSIVE`.
   - `benchmark`: Executes genuine Replay or Local backends without calling `SuiteRunner` or overwriting evidence bundles.
   - `report`: Requires an immutable bundle, verifies manifest hashes, and renders only existing data.
   - `falsify`: Evaluates existing bundles and returns standard exit codes (`0` pass, `1` falsified, `2` inconclusive).
3. **True Replay and Local Backends:**
   - Defined `BenchmarkBackend` protocol.
   - Implemented `ReplayBackend` with deterministic timing and identical paired fixtures.
   - Implemented `LocalWallClockBackend` exercising real HTTP servers, SQLite, sandboxed file I/O, subprocess sandboxes, and rate limits with real monotonic wall-clock timing.
4. **Complete Canonical Benchmark (W1–W7 + E5a):**
   - Full implementation of all 7 workload families and E5a transport codec.
   - Counterbalanced/randomized trial execution order.
   - State persistence for caching (W4), warm pools (W6), and idempotency keys (W7).
5. **Rigorous Paired Statistics:**
   - Implemented paired bootstrap resampling on trial pairs to compute P95 speedup confidence intervals.
   - Replaced favorable defaults with `None` (null) for missing evidence, causing dependent hypotheses to evaluate as `INCONCLUSIVE`.
6. **Hardened Schedulers and Runtime:**
   - `ToolExecutor`: Schema validation, lease-based rate limiting, shared idempotency store, and trusted approval verification.
   - `RateLimiter`: `async with limiter.lease(...)` preventing concurrency starvation during token acquisition.
   - E1: Strict cycle detection (including self-references), duplicate ID rejection, recursive dictionary resolution.
   - E2: Bounded, versioned declarative AST without arbitrary callables; ledger-based deoptimization without side-effect replay.
   - E3: Explicit 3-task tracking and held-out confidence calibration.
   - E4: Incremental streaming parser with strict JSON closure and immutability gates.
   - E5a: Binary transport codec with versioning, schema hashes, duplicate key rejection, and packet limits.
7. **Artifact Quarantine & Documentation:**
   - Quarantined legacy untrusted artifacts to `artifacts/legacy-untrusted/pr1-head-2066ae80/`.
   - Added methodology, evidence levels, architecture, and known limitations documentation.
