# ToolSpeeder PR #1: Scientific Integrity Repair Status

## 1. Summary of Completed Integrity Repairs

1. **Retraction of Unsupported Tables**: All historical empirical benchmark tables in `README.md` have been retracted and quarantined as noncanonical legacy outputs in `artifacts/legacy-untrusted/README.md`.
2. **Authoritative Protocol v1.1 (`benchmark-plans/tool-speed-v1.1.json`)**: Frozen protocol establishing explicit primary attribution baselines, practical baselines, W7 safety/latency separation, true identity negative controls, injected delay sensitivity control, and strict schema validation.
3. **Oracle Separation Boundary**: Implemented `BenchmarkCase` and `filter_model_visible_metadata()`. Models receive strictly `AgentTask` without oracle ground-truth, expectations, or approval grants.
4. **Runtime Concurrency Safety**: Repaired the double-release bug in `RateLimiter.lease()`, preventing over-release of concurrency slots.
5. **Shared Idempotency Lifecycle**: Standardized cross-task `SharedIdempotencyStore` with thread/loop safety, `ARG_MISMATCH` fail-closed semantics, and follower unblocking on cancellation or errors.
6. **Trusted Authority Grants**: Authority grants are evaluated strictly from `ExecutionAuthorityContext` and untrusted model-forged `is_approved` flags are ignored.
7. **Atomic Staged Bundle Writer**: Bundles are staged in isolated temporary directories, verified against required manifest fields (`benchmark_config_hash`, `workload_fixture_hash`, `raw_trace_hash`, `result_hash`), and atomically moved to destination.
8. **Direct Recomputed Falsification**: `toolspeed falsify` recomputes statistical metrics and bootstrap intervals directly from raw JSONL traces.
9. **Real Local & Deterministic Replay Backends**: Concurrency fixes in local HTTP shards, snapshot isolation in SQLite chains, and deterministic timeline advancement in replay.
10. **CI Green Smoke & Evidence Preservation**: Smoke tests pass bundle validation cleanly on Python 3.12, and full-evidence workflows preserve bundles with `if: always()`.

---

## 2. Commit Order Ledger

| Step | Conventional Commit | Focus Area |
|---|---|---|
| 1 | `docs(claims): retract unsupported benchmark tables and update PR status` | Retract tables, clean legacy artifacts, update PR #1 |
| 2 | `fix(protocol): establish one frozen benchmark protocol and schema` | Frozen `tool-speed-v1.1.json` and JSON schema validator |
| 3 | `fix(oracle): separate model task authority and benchmark oracle data` | `BenchmarkCase`, metadata whitelisting, negative validator |
| 4 | `fix(runtime): repair lease ownership approvals and shared idempotency` | Lease double-release fix, shared store lifecycle |
| 5 | `fix(artifacts): build one atomic recomputable bundle format` | Atomic bundle staging, manifest hashing, checksums |
| 6 | `fix(cli): recompute validation falsification and reports from evidence` | Raw-trace recomputation in `falsify` and `validate-bundle` |
| 7 | `fix(benchmarks): implement protocol-driven paired cases and baselines` | Symmetrical warmup, authority context routing, W1–W7 |
| 8 | `fix(backends): make replay deterministic and local workloads genuine` | Threading HTTP server, SQLite isolation, durable W7 |
| 9 | `fix(schedulers): complete E1-E5 and cache integrity repairs` | DAG reference resolution, AST invariants, LRU cache |
| 10 | `test(integrity): replace synthetic dictionary tests with end-to-end regressions` | E2E test suite `test_scientific_integrity.py` |
| 11 | `ci(evidence): make smoke green and preserve falsified full evidence` | CI smoke green, `if: always()` evidence preservation |
| 12 | `docs(methodology): align all documentation with the frozen protocol` | Update methodology and repair status docs |
