# ToolSpeeder PR #1 — Independent Evidence-Integrity and Runtime Safety Review

**Document Metadata:**
- **Repository:** `RT123-new/ToolSpeeder`
- **Pull Request:** `#1` (Draft, Unmerged)
- **Base Branch:** `main` (`1d3b3a61afefcbeb64c3015579ae1d66107e8450`)
- **Working Branch:** `repair/benchmark-integrity-runtime-safety`
- **Reviewed Head SHA:** `050176b98a40273491df4ad55526016a5d11c4ae`
- **Review Date:** 2026-08-28
- **Current Authoritative Verdict:** `INTEGRITY REPAIR INCOMPLETE`

---

## 1. Prior Handoff Claims

The handoff at commit `050176b` claimed:
1. Formula-derived timing was removed from the profiler.
2. Virtual- and wall-clock abstractions were added.
3. Runtime safety was improved.
4. Replay and local benchmark bundles in `artifacts/replay` and `artifacts/local` were described as immutable, validated empirical evidence.
5. 114+ tests were reported passing.
6. Empirical benchmark speedups were reported as verified.

---

## 2. Independently Verified Progress

The following improvements are genuine and must be preserved:
1. **Formula-driven latency discounts** such as `compute_virtual_timeline_ms` have been removed from the profiler.
2. `LatencyProfiler` and `NanosecondProfiler` are strictly observational and support an injected clock.
3. `Clock`, `WallClock`, and `VirtualClock` abstractions exist.
4. Typed `CommittedCall` arguments are deep-copied rather than converted into JSON-encoded strings.
5. The PR remains draft and unmerged.
6. Hosted CI run `33204762557` completed successfully for compatibility matrix (Python 3.10, 3.11, 3.12, 3.13), Ruff linting, test discovery, package build, and non-verdict-eligible smoke benchmarks.
7. The hosted Python 3.12 job ran exactly 138 tests (not merely "114+").
8. Local wall-clock timing honestly demonstrates that sub-millisecond local execution overhead exceeds latency savings.

---

## 3. Invalidated Claims

1. **No hosted verdict-eligible evidence exists at head `050176b`:**
   - The workflow's "Full Evidence Sweep (Replay 1000 & Local 200)" job was skipped on PR run `33204762557`.
   - The workflow run contains zero uploaded artifacts.
2. **Checked-in replay and local bundles are not canonical evidence:**
   - Manifests declare an older code SHA (`14c19751c218b8498712060e2aee81f62283390c`).
   - Manifests declare `git_dirty: true`.
   - Raw traces omit baseline arms and serialize only counts rather than full paired execution traces.
3. **Bundle validation did not prove scientific correctness:**
   - `validate-bundle` accepted placeholder hashes (`abcd1234`, `hash_cfg`, `hash_fix`, `hash_raw`), did not verify Git tree SHA, did not recompute statistics or correctness from raw traces, and did not recompute verdicts against a versioned plan.
4. **`falsify` trusted stored verdicts:**
   - Evaluator reported stored `verdict["passed"]` without independent recomputation from raw traces and pre-registered plan.
5. **Oracle separation was unenforced:**
   - Model adapters received the full `Task` object containing `expected_output` and `validator`, enabling scripted adapters to fall back to `task.expected_output`.
6. **Primary benchmark comparisons were unfair:**
   - Schedulers were compared against `SyncReActScheduler` rather than fair primary baselines (e.g., E1 against native parallel, E2 against handwritten pipeline, E3 against speculation disabled, Cache against cache disabled, E4 against streaming without early dispatch, E5a against JSON transport).
7. **W7 conflated safety with latency:**
   - A 1.00x speedup was passed under a latency check title rather than separating `W7-SAFETY` and `W7-LATENCY`.
8. **Positive sensitivity control was hard-coded:**
   - 2.00x was hard-coded as a dictionary literal rather than measured through the execution pipeline.

---

## 4. Still-Open Gates

1. **Oracle Boundary:** All model interfaces (`decide`, `predict_draft`, `stream_decision`) must strictly accept `AgentTask` only.
2. **Authority Boundary:** Trusted `ExecutionAuthorityContext` must be isolated outside model-visible data and model-generated tool calls.
3. **Clock Injection:** Injected clock must be used across all timed runtime components (profiler, rate limiter, cache TTL, idempotency, retry, approvals).
4. **Pre-registered Benchmark Plan:** Versioned `benchmark-plans/tool-speed-v1.json` with hashable plan and fair baselines.
5. **Workload Correctness:** State-isolated paired executions (W1-W7, E5a) with actual tool results deriving final answers.
6. **E4 Incremental Streaming Parser:** Character/byte delta parsing with syntax closure and immutability invariants.
7. **E5a Packet Binding:** Action bytecode packets bound to protocol version, registry version, schema hash, and opcode-table hash.
8. **Paired Statistics & Missing Data:** Bootstrap over paired-both-success trials (n=2,000); missing evidence yields `None` / `INCONCLUSIVE`.
9. **Measured Controls:** Empirical negative controls and measured positive sensitivity control.
10. **Recomputable Evidence Bundle:** Full bundle layout with paired traces, byte hashes, Git tree SHA, strict validation, recomputed falsification, and report generation post-validation.
11. **Runtime Safety Invariants:** Lease-based rate limiter (over-release raises error), run-scoped idempotency with deterministic follower resolution on cancel/timeout, complete schema validation.
12. **Hosted CI Quality Gates:** Format check, mypy, coverage threshold, twine check, and full evidence artifact upload workflow.

---

## 5. Evidence That Must Be Regenerated After Repair

- Legacy bundles (`artifacts/replay`, `artifacts/local`) are quarantined to `artifacts/legacy-untrusted/pr1-head-050176b/`.
- All canonical benchmark evidence must be regenerated at a clean, verified commit SHA via the dedicated full-evidence workflow and uploaded as GitHub Actions artifacts.
