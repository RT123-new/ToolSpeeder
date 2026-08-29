# ⚡ ToolSpeed (ToolSpeeder)

**Scientific Benchmark Suite & Runtime Optimization Schedulers for AI Agent Tool Calling**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

ToolSpeed evaluates and optimizes the serial critical path latency (Correct Completion Latency, CCL) between model reasoning and tool execution in autonomous AI agent systems.

---

## 🔬 Evidence Taxonomy & Scientific Hierarchy

ToolSpeed strictly classifies experimental data and claims under four discrete evidence levels:

1. **`SYNTHETIC`**: Mathematical simulation models evaluating theoretical limits and hypothesis boundaries. Real-world claims are marked **`INCONCLUSIVE`**.
2. **`REPLAY_INTEGRATION`**: Real scheduler code executing deterministic virtual-delay adapters on canonical workload traces (W1–W7, E5a).
3. **`LOCAL_WALL_CLOCK`**: Real scheduler code executing real local tools (SQLite databases, mock HTTP servers, sandboxed file I/O, and subprocess sandboxes).
4. **`LIVE_PRODUCTION`**: Real schedulers connected to live cloud LLM APIs and third-party remote endpoints (scoped as future work).

---

## 🎯 Optimization Schedulers (E1 – E5)

1. **E1 — Dynamic DAG Scheduler (`DAGScheduler`)**: Two-pass dependency discovery and DFS cycle detection; executes ready tool waves concurrently with dependency data binding.
2. **E2 — Declarative JIT Fusion (`JITFusionScheduler`)**: Safe declarative AST (`DeclarativeWorkflow`, `WorkflowNode`, `WorkflowInvariant`) executed locally with side-effect tracking and safe fallback deoptimization.
3. **E3 — Speculative Reads (`SpeculativeReadScheduler`)**: Concurrent draft prediction during model reasoning, multi-call matching across decision steps, and cancellation-safe task lifecycles.
4. **E4 — Commit-Horizon Streaming (`CommitHorizonScheduler`)**: Incremental streaming parser (`IncrementalCommitParser`) early-dispatching read-only tools upon argument immutability closure.
5. **E5a — Action Bytecode Codec (`ActionBytecodeScheduler`)**: Compact binary transport codec (`ActionBytecodeCodec`) with strict length and duplicate key validation.
6. **Phase 2 Caching (`CacheScheduler`)**: TTL-aware exact and normalized parameter caching with domain-level invalidation upon mutative actions.
7. **Composite Pipeline (`CompositeScheduler`)**: Unified adaptive execution coordinating DAG scheduling, caching, speculation, and streaming commit horizons.

---

## 📋 Current Status

> **Status:** benchmark-integrity repair in progress.  
> Compatibility tests pass at the referenced head.  
> Current benchmark smoke is failing bundle validation.  
> No current-head canonical replay or local evidence bundle exists.  
> Live evidence has not been collected.  
> Historical numerical outputs are noncanonical and must not be used for claims.

---

## 🚀 CLI Commands & Workflows

### 1. Run Real Paired Benchmark Suite
Executes real schedulers on genuine backends (`replay` or `local`), produces immutable bundles with full provenance manifests, and computes paired bootstrap confidence intervals.
```bash
# Trace Replay Backend (>= 1,000 trials required for verdict eligibility)
toolspeed benchmark --backend replay --trials 1000 --out artifacts/replay

# Local Wall-Clock Backend (>= 200 trials required for verdict eligibility)
toolspeed benchmark --backend local --trials 200 --out artifacts/local
```

### 2. Validate Benchmark Bundle
Verifies structural schema, code git SHA, SHA256 integrity hashes, trial counts, and paired evaluations:
```bash
toolspeed validate-bundle --input artifacts/replay
```

### 3. Evaluate Hypothesis Falsification
Evaluates an existing benchmark bundle against strict statistical falsification criteria:
```bash
# Returns exit code 0 (passed), 1 (falsified), or 2 (inconclusive)
toolspeed falsify --input artifacts/replay
```

### 4. Generate Reports from Immutable Bundles
Renders Markdown and interactive HTML dashboards directly from existing bundles without rerunning simulations:
```bash
toolspeed report --input artifacts/replay --out artifacts/replay
```

### 5. Run Synthetic Analytical Simulation
```bash
toolspeed simulate --experiment all --trials 1000 --out artifacts/synthetic
```

### 6. Run Test Suite
```bash
# Run 114+ unit, scheduler, and adversarial scientific integrity tests
python3 -m unittest discover -s tests -p "test_*.py"
```

---

## 🛡️ Runtime Safety & Security Boundaries

- **Centralized Execution Authority**: All schedulers route tool calls through `ToolExecutor`.
- **Approval Gating**: Mutative tools (`is_read_only=False` or `side_effects=True`) require explicit approval (`is_approved=True`). Schedulers cannot manufacture approval.
- **Shared Idempotency Store**: Prevents duplicate execution of side-effecting operations across task lifecycles.
- **Resource Sandboxing**: Subprocess sandboxing enforces working directory containment, timeout enforcement, and SIGKILL process tree termination.
- **Cancellation Safety**: All cancelled child tasks are cleanly awaited via `cancel_and_await` to prevent unhandled background coroutine exceptions across Python 3.10–3.13.
