# ToolSpeeder PR #1: Scientific Integrity Repair Status

**Document Coordinates:**
- **Repository:** `RT123-new/ToolSpeeder`
- **Pull Request:** `#1` (Draft, Unmerged, Auto-merge Disabled)
- **Base Branch:** `main` (`1d3b3a61afefcbeb64c3015579ae1d66107e8450`)
- **Working Branch:** `repair/benchmark-integrity-runtime-safety`
- **Reviewed Head SHA:** `5cfef2e`
- **Current Status:** INTEGRITY REPAIR INCOMPLETE

---

## Authoritative Status Declaration

```text
INTEGRITY REPAIR INCOMPLETE

The September 2 replay runs are retained as noncanonical exploratory
diagnostics. They do not satisfy the seed, comparison, control, oracle,
artifact, or prospective-protocol requirements for confirmatory evidence.

The current exact-head CI status is in progress (Mypy typechecking restored, 
E2 oracle access removed, multi-python matrix running).

No canonical confirmatory replay evidence exists.

No canonical confirmatory local evidence exists.

Live evidence is absent.
```

---

## Retracted Claims & Audit Ledger

| Claim | Location | Prior wording | Verified state | Corrected wording | Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Integrity Repair Complete** | PR #1 Body, `PR1_REPAIR_STATUS.md` | `INTEGRITY REPAIR COMPLETED — CONFIRMATORY FALSIFICATION REPORTED` | **RETRACTED** | `INTEGRITY REPAIR INCOMPLETE` | Head `be36503` had red CI; E2 accessed oracle; controls were hard-coded; protocol was not prospective. |
| **Quality Gates Passing** | PR #1 Body | `100% quality gates passing across all gates` | **RETRACTED** | CI failed on all 4 Python versions at `be36503`. Restored locally and queued on CI at `5cfef2e`. | GitHub Actions run `33682319407` failed with 6 mypy errors. |
| **Confirmatory 3-Seed Runs** | PR #1 Body, `PR1_REPAIR_STATUS.md` | `Executed on Replay Backend across 3 seeds (42, 137, 2026), n=1,000 trials each.` | **RETRACTED** | Noncanonical exploratory diagnostics. Seeds were not looped by harness; CLI manufactured `[seed, seed+1, seed+2]`. | `toolspeed/cli.py:135`, `toolspeed/benchmarks/harness.py:112`. |
| **Raw Trace Recomputation** | PR #1 Body, `cli.py` | `Recomputed directly from raw JSONL traces (2,000 lines/seed)` | **RETRACTED** | Falsify fell back to stored summaries, used default latencies/IDs, defaulted missing baseline success to True, tested only against 1.0x. | `toolspeed/cli.py:316-368`. |
| **Sealed Canonical Evidence** | PR #1 Body | `Structurally sealed bundles with atomic SHA-256 byte manifests` | **RETRACTED** | `raw_trace_hash` covered only candidate; `result.json` written before `result_hash` added to external manifest; confirmatory directories gitignored. | `toolspeed/visualization/report.py:789-838`, `.gitignore`. |
| **All Findings A–O Resolved** | PR #1 Body, `PR1_REPAIR_STATUS.md` | `All findings A–O RESOLVED` | **RETRACTED** | Findings were partially fixed via test-facing facades (`create_w2_state`, `PersistentColdPool.acquire_time_ms`, etc.) rather than genuine execution. | `tests/test_review_findings.py`, `toolspeed/workloads/w6_cold_start.py`. |
| **True Execution Controls** | PR #1 Body, `harness.py` | `Derived from execution, not hard-coded` | **RETRACTED** | `run_negative_controls()` returned literal 1.0; positive control returned literal 2.0 with a fake boolean flag. | `toolspeed/benchmarks/harness.py:513-585`. |
| **W7 Safety Verified** | PR #1 Body | `W7_SAFETY ... PASS (Single mutation invariant)` | **RETRACTED** | Tools returned mock dictionaries without tracking account balances, verifying idempotency ledgers, or checking state transitions. | `toolspeed/benchmarks/replay_backend.py:411-430`. |
| **E5a Codec Falsified** | PR #1 Body | `E5a ... FALSIFIED (Scheduler vs Codec)` | **RETRACTED** | `codec_bench.py` was an unexecuted 21-line stub; canonical harness ran ActionBytecodeScheduler vs SyncReActScheduler instead of codec vs codec. | `toolspeed/benchmarks/codec_bench.py:1-21`. |
| **E2 Fusion Integrity** | PR #1 Body | `Rejects untrusted/injected workflow objects` | **RETRACTED** | JIT fusion scheduler read `ctx.task.expected_output["status"]` directly and called `ctx.task.validate()`. | `toolspeed/schedulers/e2_jit_fusion.py:428-436`. |

---

## Current Findings Disposition (Findings 1–20)

| Finding | Area | Status | Code Location | Disposition / Next Step |
| :--- | :--- | :--- | :--- | :--- |
| **1. CI Red** | CI / Typing | **FIXED** | `e2_jit_fusion.py`, `test_review_findings.py` | 6 mypy errors fixed at `5cfef2e` without preserving oracle leak. Static barrier test added. |
| **2. Invalid 3-Seed Runs** | CLI / Harness | **REPRODUCED** | `cli.py`, `harness.py` | Need real per-seed case matrices and harness loops. |
| **3. Protocol Mismatch** | Protocol Lineage | **REPRODUCED** | `resources/protocols/` | Retain v1.1 as retrospective repair, v1.2 as draft. Introduce v1.3 draft prospectively. |
| **4. Hard-Coded Comparisons** | Harness | **REPRODUCED** | `harness.py:598-670` | Drive execution from protocol-defined comparison plans with fair baselines. |
| **5. Hard-Coded Controls** | Harness | **REPRODUCED** | `harness.py:513-585` | Execute real paired arms for identity controls and measured delay injection for positive control. |
| **6. Partial Falsify** | CLI / Recompute | **REPRODUCED** | `cli.py:316-370` | Build independent `recompute.py` verifying full trace, metrics, controls, and protocol thresholds. |
| **7. Bundle Sealing Defects** | Artifacts | **REPRODUCED** | `report.py:789-838` | Build atomic immutable bundles with detached `seal.json`, ordered trace hashes, no overwrite. |
| **8. E2 Oracle Leak** | Scheduler | **FIXED** | `e2_jit_fusion.py:425-447` | Removed `expected_output` and `validate` access from scheduler. Output built strictly from ledger. |
| **9. Mutable Task Boundary** | Core Types | **REPRODUCED** | `types.py:535-612` | Make `BenchmarkCase` recursively immutable; schedulers receive only `AgentTask`. |
| **10. Scripted Replay** | Replay Backend | **REPRODUCED** | `replay_backend.py` | Derive final answers causally from observed tool results; isolate virtual clocks per case. |
| **11. E4 Readiness Trust** | E4 Scheduler | **REPRODUCED** | `e4_commit_horizon.py` | Parse character-level deltas; reconcile by tool, schema, canonical args; dispatch immutable calls. |
| **12. E3 Speculation Gating** | E3 Scheduler | **REPRODUCED** | `e3_speculation.py` | Gate speculation on explicit concurrency safety contract; build held-out calibration dataset. |
| **13. E5a Codec Benchmark** | Benchmarks | **REPRODUCED** | `codec_bench.py` | Implement executable comparison of `ActionBytecodeCodec` vs `CanonicalJSONCodec`. |
| **14. W6 Pool Stubs** | W6 Workload | **REPRODUCED** | `w6_cold_start.py` | Implement real container/process pool lifecycle, prewarming, expiration, and acquisition tracking. |
| **15. W7 Mutation Ledger** | W7 Workload | **REPRODUCED** | `w7_side_effects.py` | Build independent transactional ledger verifying exactly-once state mutation and zero unapproved calls. |
| **16. Local Backend Stubs** | Local Backend | **REPRODUCED** | `local_backend.py` | Replace stubs with genuine isolated SQLite database clones and real execution. |
| **17. Forgeable Authority** | Authority | **REPRODUCED** | `types.py:347-460` | Out-of-band `AuthorityProvider` with fresh run secrets; reject unsigned and truncated grants. |
| **18. Cache Architecture** | Cache | **REPRODUCED** | `phase2_cache.py` | Single bounded logical capacity; access-order LRU; stampede coalescing; strict freshness. |
| **19. Composite Coupling** | Composite | **REPRODUCED** | `composite.py` | Refactor into single dispatch state machine; remove private method calls and facade helpers. |
| **20. Subprocess Security** | Security | **REPRODUCED** | `live_tools.py` | Replace shell execution with argument lists; enforce executable allowlists and cwd isolation. |
