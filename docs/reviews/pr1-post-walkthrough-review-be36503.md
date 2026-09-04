# PR #1 Starting State Review & Findings Audit

- **Date:** 2026-09-03
- **Reviewer:** Gemini Flash (Primary Orchestrator / Verification Context)
- **Repository:** `RT123-new/ToolSpeeder`
- **PR:** #1 (`repair/benchmark-integrity-runtime-safety` -> `main`)
- **Base SHA:** `1d3b3a61afefcbeb64c3015579ae1d66107e8450`
- **Starting Head SHA:** `be3650362f391c204ba08480a17e011b05e88832`
- **Merge Base:** `1d3b3a61afefcbeb64c3015579ae1d66107e8450`
- **Working Tree Cleanliness:** Clean (`git status --short` empty)
- **Current PR State:** Open, Draft, Unmerged, Auto-merge Disabled
- **GitHub Actions CI Status:** Red (Run `33682319407` / `33682332733` failed on all 4 Python compatibility matrix jobs during mypy)
- **Artifact Availability:** Local directories `artifacts/confirmatory_seed*` exist locally but are excluded by `.gitignore` and absent from GitHub Actions artifacts.

---

## Authoritative Starting Verdict

```text
INTEGRITY REPAIR INCOMPLETE
```

The prior walkthrough declaration (`INTEGRITY REPAIR COMPLETED — CONFIRMATORY FALSIFICATION REPORTED`) is unsupported by the repository state and is hereby retracted.

---

## Detailed Findings Ledger & Reproduction Audit

| Finding | Area | Status | Exact Code Location | Notes & Disposition |
| :--- | :--- | :--- | :--- | :--- |
| **1. Exact-head CI Red** | CI / Typing | **REPRODUCED** | `toolspeed/schedulers/e2_jit_fusion.py:429,430,433`<br>`tests/test_review_findings.py:208,499,524` | 6 mypy errors reproduced locally. Caused by oracle access in E2 scheduler and test typing defects. |
| **2. Replay runs not valid 3-seed** | Benchmark / CLI | **REPRODUCED** | `toolspeed/cli.py:132-135`<br>`toolspeed/benchmarks/harness.py:112` | When `--seeds` absent, CLI synthesizes `[seed, seed+1, seed+2]`. Harness checks length >= 3 without looping over seeds. |
| **3. Claimed protocol mismatch** | Protocol Lineage | **REPRODUCED** | `toolspeed/resources/protocols/tool-speed-v1.1.json`<br>`tool-speed-v1.2-draft.json` | v1.1 is marked `retrospective_repair`. v1.2 is unfrozen draft. No prospective confirmatory protocol was frozen. |
| **4. Benchmark comparisons hard-coded** | Harness | **REPRODUCED** | `toolspeed/benchmarks/harness.py:598-670` | `run_full_benchmark()` hard-codes SyncReAct baselines, bypassing registered protocol attribution plans (W1, W2, W4, W5, W6, W7, E5a). |
| **5. Hard-coded controls** | Harness | **REPRODUCED** | `toolspeed/benchmarks/harness.py:513-585` | `run_negative_controls()` returns hard-coded 1.0 dicts; positive control returns literal 2.0 with fake flag. |
| **6. Incomplete recomputation** | CLI / Evidence | **REPRODUCED** | `toolspeed/cli.py:316-347` | `cmd_falsify` uses fallback IDs/latencies, defaults missing baseline success to True, tests against 1.0x, relies on stored verdicts. |
| **7. Inconsistent bundle sealing** | Artifacts | **REPRODUCED** | `toolspeed/visualization/report.py:789-838` | `raw_trace_hash` only hashes candidate trace; `result.json` written before `result_hash` added to external manifest; `shutil.rmtree` on existing output; gitignored directories. |
| **8. E2 reads oracle directly** | E2 Scheduler | **REPRODUCED** | `toolspeed/schedulers/e2_jit_fusion.py:428-436` | Reads `ctx.task.expected_output["status"]` and calls `ctx.task.validate()`. Direct oracle leak. |
| **9. Canonical boundary uses Task** | Core Types | **REPRODUCED** | `toolspeed/core/types.py:535-612`<br>`toolspeed/schedulers/base.py` | Mutable `Task` with oracle data fed to schedulers; `BenchmarkCase` shallowly frozen with mutable dicts. |
| **10. Scripted replay outputs** | Replay Backend | **REPRODUCED** | `toolspeed/benchmarks/replay_backend.py:80-160` | Replay returns canned static answers; wall-clock `time.perf_counter()` mixed with virtual time. |
| **11. E4 trusts preconstructed readiness** | E4 Scheduler | **REPRODUCED** | `toolspeed/schedulers/e4_commit_horizon.py:235-255` | Consumes `chunk.commit_horizon_ready` containing pre-built `ToolCall`s; dispatches mutable calls; matches by ID alone. |
| **12. E3 speculation uncalibrated** | E3 Scheduler | **REPRODUCED** | `toolspeed/schedulers/e3_speculation.py:30-32` | `supports_concurrent_adapter` defaults to True; no held-out calibration or real resource topologies. |
| **13. E5a not a codec benchmark** | Codec Bench | **REPRODUCED** | `toolspeed/benchmarks/codec_bench.py:1-21`<br>`toolspeed/benchmarks/harness.py:670` | `codec_bench.py` is a 21-line stub; canonical harness runs ActionBytecodeScheduler vs SyncReActScheduler instead of codec vs codec. |
| **14. W6 pool stubs** | W6 Workload | **REPRODUCED** | `toolspeed/workloads/w6_cold_start.py:118-136` | Pools return constant numbers (`35.0`, `2.0`); no real container/resource lifecycle. |
| **15. W7 lacks state mutation ledger** | W7 Workload | **REPRODUCED** | `toolspeed/benchmarks/replay_backend.py:411-430`<br>`toolspeed/benchmarks/local_backend.py` | Returns mock dicts without verifying account balances, single mutation invariant, or idempotent replay. |
| **16. Local backend test stubs** | Local Backend | **REPRODUCED** | `toolspeed/benchmarks/local_backend.py:126-146` | `get_w2_row_count` returns literal 100; `execute_w2_step` is empty pass; constant table hash. |
| **17. Forgeable authority boundary** | Authority | **REPRODUCED** | `toolspeed/core/types.py:347,429,447` | Hardcoded `DEFAULT_ISSUER_SECRET`; unsigned grants accepted when signature empty; truncated fingerprints accepted. |
| **18. Cache not bounded scoped LRU** | Cache | **REPRODUCED** | `toolspeed/schedulers/phase2_cache.py:45-48,128-135` | Dual independent stores double capacity; no concurrency locks; no stampede coalescing; relaxed stale results permitted. |
| **19. Composite lacks clean attribution** | Composite | **REPRODUCED** | `toolspeed/schedulers/composite.py:52-54,87-93` | Calls `jit_scheduler._execute_internal()`; facade helper `has_cache_lookup_in_dispatch_path()`. |
| **20. Subprocess sandbox shell execution** | Local Security | **REPRODUCED** | `toolspeed/adapters/live_tools.py:221-228` | Uses `asyncio.create_subprocess_shell` with raw command string; no executable allowlist or container isolation. |
