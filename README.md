# ⚡ ToolSpeed (ToolSpeeder)

**Scientific Benchmark Suite & Runtime Optimization Schedulers for AI Agent Tool Calling**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

ToolSpeed evaluates and optimizes the serial critical path latency (Correct Completion Latency, CCL) between model reasoning and tool execution in autonomous AI agent systems.

---

## 📋 Current Status & Integrity State

> **Current status:** integrity repair in progress.  
> **Current CI status:** compatibility and smoke checks pass on commit `e3c974be61d58df0c2870a098099f1371a161c10`.  
> **Current confirmatory evidence:** not yet collected under a valid frozen protocol.  
> **Replay/local smoke outputs:** operational tests only, not verdict-eligible.  
> **Live evidence:** absent.  
> **Historical numerical outputs:** noncanonical legacy data and must not be used for scientific claims.

### Claim Audit Table

| Claim | Previous Location | Verified? | Exact Evidence | Corrected Wording |
| :--- | :--- | :--- | :--- | :--- |
| "INTEGRITY REPAIR COMPLETED — EVIDENCE READY" | PR #1 Body | **REFUTED** | Skipped CI full-evidence sweep; unaddressed architectural findings A–O. | Current status: integrity repair in progress. |
| "Recomputed directly from raw traces" | PR #1 Body, `cli.py` | **REFUTED** | `cli.py:289-307` reads stored `evaluations` from `result.json`. | Evaluated from stored summaries; raw-trace recomputation pending implementation. |
| "Full bundle validation pass" | PR #1 Body | **QUALIFIED** | Validated only on smoke runs; no canonical full-sweep bundle exists. | Compatibility and smoke checks pass; canonical evidence uncollected. |
| "LRU capacity enforcement" | `phase2_cache.py`, README | **REFUTED** | `phase2_cache.py:108` evicts by creation timestamp (FIFO). | Bounded FIFO eviction with separate sub-store limits. |
| "Safe Subprocess Sandbox with memory caps and SIGKILL" | `SECURITY.md`, README | **REFUTED** | `live_tools.py:198` uses `shell=True`, no memory cap, no process-tree kill. | Controlled local execution tool; not an isolated security sandbox. |
| "Prospectively frozen protocol v1.1" | `tool-speed-v1.1.json` | **QUALIFIED** | Authored retrospectively after initial implementation exploration. | Retrospective repair protocol v1.1; draft v1.2 required. |

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
