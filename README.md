# ⚡ ToolSpeed (ToolSpeeder)

**Scientific Benchmark Suite & Runtime Optimization Schedulers for AI Agent Tool Calling**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

ToolSpeed evaluates and optimizes the serial critical path latency (Correct Completion Latency, CCL) between model reasoning and tool execution in autonomous AI agent systems.

---

## 📋 Current Status & Integrity State

```text
40-PHASE MANDATE COMPLETE — DEFENSIBLE EVIDENCE PRODUCED & INDEPENDENT AUDITS COMPILED

1. Legacy claims (v1.0/v1.1) remain formally RETRACTED due to synthetic simulation reliance.
2. The integrity architecture is fully implemented: AST static barriers, capability grants,
   two-phase rate limiting, scoped idempotency, and self-contained bundles with detached seals.
3. Canonical Confirmatory Evidence (artifacts/confirmatory/) passed under frozen protocol v1.3.
4. Canonical Local Evidence (artifacts/local/) established real speedup on W1 (3.83x) and W5 (1.22x),
   while strictly failing closed (verdict FALSIFIED) on unconfigured host sockets.
5. 9 Independent Gemini Flash Reviews were executed and compiled (docs/reviews/9_independent_flash_reviews.md).
6. Zero-dependency replication package is provided via ./scripts/reproduce_benchmarks.sh.
7. PR #1 remains in DRAFT, UNMERGED, with auto-merge disabled.
```

### Claim Audit & Retraction Table

| Claim | Prior Location | Verified State | Corrected Wording | Evidence |
| :--- | :--- | :--- | :--- | :--- |
| "INTEGRITY REPAIR COMPLETED — CONFIRMATORY FALSIFICATION REPORTED" | PR #1 Body, Status | **RETRACTED** | `INTEGRITY REPAIR INCOMPLETE` | CI was red at `be36503`; E2 oracle leak; controls hard-coded; protocol unverified. |
| "100% quality gates passing" | PR #1 Body | **RETRACTED** | CI failed across all Python versions at `be36503`. Restored locally at `5cfef2e`. | GitHub Actions run `33682319407`. |
| "Confirmatory empirical benchmark (3 seeds)" | PR #1 Body, Status | **RETRACTED** | Noncanonical exploratory diagnostics. | Harness did not loop seeds; CLI synthesized artificial seed arrays. |
| "Recomputed directly from raw traces" | PR #1 Body, `cli.py` | **RETRACTED** | Evaluated from stored summaries with weak fallbacks. | `cli.py:316-368` uses fallback latencies, default success, tests only 1.0x. |
| "Cryptographically sealed canonical evidence" | PR #1 Body, `report.py` | **RETRACTED** | Unsealed/inconsistent manifests; excluded by `.gitignore`. | `raw_trace_hash` covers candidate only; `result_hash` missing from embedded manifest. |
| "All findings A–O resolved" | PR #1 Body | **RETRACTED** | Partially patched via test-facing facades. | Methods like `create_w2_state` and `PersistentColdPool.acquire_time_ms` were stubs. |
| "True execution controls" | PR #1 Body, `harness.py` | **RETRACTED** | Hard-coded literal return values. | `run_negative_controls()` returned literal 1.0 and 2.0 dictionaries without running arms. |
| "W7 safety verified" | PR #1 Body | **RETRACTED** | Unverified mock tool responses without durable ledger. | Tools returned static dicts; no account balance or state transition ledger existed. |
| "E5a codec falsified" | PR #1 Body | **RETRACTED** | Unmeasured; scheduler compared against scheduler. | `codec_bench.py` was a 21-line stub; no codec vs JSON round-trip measured. |

---

## 🔬 Evidence Taxonomy & Scientific Hierarchy

ToolSpeed strictly classifies experimental data and claims under four discrete evidence levels:

1. **`SYNTHETIC`**: Mathematical simulation models evaluating theoretical limits and hypothesis boundaries. Real-world claims are marked **`INCONCLUSIVE`**.
2. **`REPLAY_INTEGRATION`**: Real scheduler code executing deterministic virtual-delay adapters on canonical workload traces (W1–W7, E5a).
3. **`LOCAL_WALL_CLOCK`**: Real scheduler code executing local controlled tools (SQLite databases, mock HTTP servers, local file I/O, and local subprocess primitives).
4. **`LIVE_PRODUCTION`**: Real schedulers connected to live cloud LLM APIs and third-party remote endpoints (scoped as future work).

---

## 🎯 Optimization Schedulers (E1 – E5)

1. **E1 — Dynamic DAG Scheduler (`DAGScheduler`)**: Dependency discovery and cycle detection; executes ready tool waves concurrently with dependency data binding.
2. **E2 — Declarative JIT Fusion (`JITFusionScheduler`)**: Declarative AST (`DeclarativeWorkflow`, `WorkflowNode`, `WorkflowInvariant`) executed locally with side-effect tracking and fallback deoptimization.
3. **E3 — Speculative Reads (`SpeculativeReadScheduler`)**: Concurrent draft prediction during model reasoning, multi-call matching across decision steps, and cancellation-safe task lifecycles.
4. **E4 — Commit-Horizon Streaming (`CommitHorizonScheduler`)**: Incremental streaming parser (`IncrementalCommitParser`) early-dispatching read-only tools upon argument closure.
5. **E5a — Action Bytecode Codec (`ActionBytecodeScheduler`)**: Binary transport codec (`ActionBytecodeCodec`) for efficient tool payload serialization.
6. **Phase 2 Caching (`CacheScheduler`)**: TTL-aware exact and normalized parameter caching with domain-level invalidation upon mutative actions.
7. **Composite Pipeline (`CompositeScheduler`)**: Unified execution coordinating DAG scheduling, caching, speculation, and streaming commit horizons.

---

## 🚀 CLI Commands & Workflows

### 1. Run Benchmark Suite
Executes schedulers on backends (`replay` or `local`), produces bundles with provenance manifests, and computes paired bootstrap confidence intervals.
```bash
# Trace Replay Backend (>= 1,000 trials per seed required for verdict eligibility)
toolspeed benchmark --protocol benchmark-plans/tool-speed-v1.1.json --backend replay --mode smoke --out artifacts/replay

# Local Wall-Clock Backend (>= 200 trials per seed required for verdict eligibility)
toolspeed benchmark --protocol benchmark-plans/tool-speed-v1.1.json --backend local --mode smoke --out artifacts/local
```

### 2. Validate Benchmark Bundle
Verifies structural schema, code git SHA, SHA-256 integrity hashes, trial counts, and paired evaluations:
```bash
toolspeed validate-bundle --input artifacts/replay
```

### 3. Evaluate Hypothesis Falsification
Evaluates an existing benchmark bundle against statistical falsification criteria:
```bash
# Returns exit code 0 (passed), 1 (falsified), or 2 (inconclusive)
toolspeed falsify --input artifacts/replay
```

### 4. Generate Reports from Bundles
Renders Markdown and interactive HTML dashboards directly from existing bundles without rerunning simulations:
```bash
toolspeed report --input artifacts/replay --out artifacts/replay-render
```

### 5. Run Synthetic Analytical Simulation
```bash
toolspeed simulate --experiment all --trials 1000 --out artifacts/synthetic
```

### 6. Run Test Suite
```bash
uv run coverage erase && uv run coverage run -m pytest -q
```

---

## 🛡️ Runtime Safety & Operational Boundaries

- **Execution Routing**: Schedulers route tool calls through `ToolExecutor`.
- **Approval Gating**: Mutative tools require explicit approval. Schedulers cannot self-authorize.
- **Shared Idempotency Store**: Deduplicates execution of side-effecting operations across task lifecycles.
- **Controlled Local Execution**: Local subprocess and file operations execute in temporary workspaces with timeout limits (controlled benchmark tools, not hardened multi-tenant sandboxes).
- **Cancellation Safety**: Cancelled child tasks are caught and handled cleanly across Python 3.10–3.13.
