# ⚡ ToolSpeed

**Scientific Benchmark Suite & Runtime Optimization Schedulers for AI Agent Tool Calling**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-115%20passed-brightgreen.svg)]()
[![Evidence Level](https://img.shields.io/badge/evidence-Replay%20%26%20Local%20Validated-informational.svg)]()

ToolSpeed evaluates and optimizes the serial critical path latency (Correct Completion Latency, CCL) between model reasoning and tool execution in autonomous AI agents.

---

## 🔬 Evidence Taxonomy

To maintain rigorous scientific integrity, ToolSpeed categorizes all benchmark evidence into four strictly separated levels:

1. **`SYNTHETIC`**: Mathematical simulation models evaluating theoretical limits and hypothesis boundaries.
2. **`REPLAY_INTEGRATION`**: Real scheduler code executing deterministic virtual-delay adapters on canonical workload traces.
3. **`LOCAL_WALL_CLOCK`**: Real scheduler code executing real local tools (SQLite, mock HTTP servers, sandboxed file I/O, and subprocesses).
4. **`LIVE`**: Real schedulers connected to live external LLM endpoints and third-party APIs.

---

## 🎯 Optimization Schedulers (E1 – E5)

1. **E1 — Dynamic DAG Scheduler (`DAGScheduler`)**: Two-pass dependency discovery and DFS cycle detection; executes ready tool waves with concurrency backpressure.
2. **E2 — Declarative JIT Fusion (`JITFusionScheduler`)**: Safe declarative AST (`DeclarativeWorkflow`, `WorkflowNode`, `WorkflowInvariant`) executed locally with side-effect tracking and safe fallback deoptimization.
3. **E3 — Speculative Reads (`SpeculativeReadScheduler`)**: Concurrent draft prediction during model reasoning, multi-call matching across decision steps, and leak-free task cancellation.
4. **E4 — Commit-Horizon Streaming (`CommitHorizonScheduler`)**: Incremental streaming parser (`IncrementalCommitParser`) early-dispatching read-only tools upon argument immutability closure.
5. **E5 — Action Bytecode Codec (`ActionBytecodeScheduler`)**: Compact 16-bit big-endian binary transport codec (`ActionBytecodeCodec`) supporting up to 65,535 tools with strict length and boundary checks.
6. **Phase 2 Caching (`CacheScheduler`)**: TTL-aware exact and normalized parameter caching with domain-level invalidation upon mutative actions.
7. **Composite Pipeline (`CompositeScheduler`)**: Unified adaptive execution coordinating DAG scheduling, caching, speculation, and streaming.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/RT123-new/ToolSpeeder.git
cd ToolSpeeder
pip install -e ".[dev]"
```

### CLI Commands

```bash
# 1. Run Real Paired Benchmark Suite (Replay Backend)
toolspeed benchmark --backend replay --trials 50 --out artifacts/replay

# 2. Run Real Local Wall-Clock Benchmark Suite
toolspeed benchmark --backend local --trials 10 --out artifacts/local

# 3. Run Hypothesis Falsification Evaluator (Synthetic)
toolspeed falsify --trials 150

# 4. Generate Markdown Evidence Log and Interactive HTML Dashboard
toolspeed report --trials 150 --out results

# 5. Run 22 Adversarial Scientific Integrity Unit Tests
python3 -m unittest tests/test_adversarial_integrity.py

# 6. Run Full Unit Test Discovery (115 tests)
python3 -m unittest discover -s tests -p "test_*.py"
```

---

## 🛡️ Runtime Safety & Security

- **Centralized Execution Authority**: All schedulers route tool calls through `ToolExecutor`.
- **Approval Gating**: Mutative tools (`is_read_only=False` or `side_effects=True`) require explicit approval (`is_approved=True`).
- **Resource Sandboxing**: Subprocess sandboxing enforces working directory containment, timeout enforcement, and SIGKILL process tree termination.
- **Leak-Free Async Execution**: Speculative cancellations await task cleanup to prevent unhandled background coroutine exceptions.
