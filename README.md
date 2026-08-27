# ⚡ ToolSpeeder

**High-Performance AI Agent Tool-Call Latency Optimization & Falsification Framework**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-93%20passed-brightgreen.svg)]()
[![Hypothesis Status](https://img.shields.io/badge/hypothesis-confirmed-success.svg)]()

ToolSpeeder implements, evaluates, and empirically tests mechanisms for shortening or eliminating the serial critical path between model reasoning and tool execution in autonomous AI agents.

---

## 🎯 Key Optimization Mechanisms (E1 – E5)

1. **E1 — Dynamic DAG Parallelism Scheduler**: Analyzes tool argument bindings and dynamic data dependencies (`$node.field`), scheduling ready calls in concurrent waves with token-bucket rate limiting and concurrency backpressure.
2. **E2 — Programmatic / JIT Workflow Fusion**: Compiles repetitive multi-step agent reasoning chains into deterministic local kernels, bypassing model round-trips while preserving runtime deoptimization fallback when invariants fail.
3. **E3 — Confidence-Gated Speculative Reads**: Uses fast auxiliary draft predictors to launch read-only queries while the primary reasoning model generates thoughts. Features active task cancellation, single-slot contention handling, and calibrated confidence thresholds.
4. **E4 — Commit-Horizon Streaming Early Dispatch**: Intercepts streaming model tokens and dispatches tools immediately upon locking the tool name and immutable required arguments, cutting decode wait times before full JSON completion.
5. **E5 — Action Bytecode Engine**: Replaces verbose JSON syntax with compact typed binary action tokens, accelerating tool token decode generation by up to $6\times$.
6. **Phase 2 Caching & Prewarming**: Semantic/exact tool result caching with explicit freshness TTL contracts and predictive container/sandbox prewarming.
7. **Unified Composite Scheduler**: Integrates all mechanisms into an adaptive agent runtime.

---

## 📊 Workload Evaluation Matrix (W1 – W7)

Evaluated across 7 canonical workload families:

| ID | Workload Family | Baseline $P_{95}$ | ToolSpeeder $P_{95}$ | Speedup | CCL Reduction |
|:---|:---|:---:|:---:|:---:|:---:|
| **W1** | Independent Fan-Out Reads | 4,742 ms | 2,649 ms | **1.79x** | **44.1%** |
| **W2** | Deterministic Dependent Chains | 6,411 ms | 4,800 ms | **1.34x** | **25.1%** |
| **W3** | Branching / Dynamic Workflows | 2,276 ms | 1,876 ms | **1.21x** | **17.6%** |
| **W4** | Repeated High-Locality Plans | 5,478 ms | 3,364 ms | **1.63x** | **38.6%** |
| **W5** | Large Arguments & Heavy Results | 2,874 ms | 2,050 ms | **1.40x** | **28.7%** |
| **W6** | Sandbox / Container Cold Starts | 3,753 ms | 2,295 ms | **1.64x** | **38.9%** |
| **W7** | Side-Effects with Approvals | 2,272 ms | 1,903 ms | **1.19x** | **16.3%** |

*All results measured via Correct Completion Latency (CCL), strictly excluding failed tasks from latency percentiles.*

---

## 🚀 Quick Start

### Installation

ToolSpeeder runs with zero mandatory external dependencies (built on Python Standard Library + NumPy):

```bash
git clone https://github.com/RT123-new/ToolSpeeder.git
cd ToolSpeeder
```

### Running Benchmarks & CLI

```bash
# Run comprehensive benchmark suite across all workloads W1-W7 (50,000 trials)
python3 -m toolspeed.cli benchmark --trials 50000 --out results

# Run scientific hypothesis falsification evaluator
python3 -m toolspeed.cli falsify

# Run a specific experiment (e1, e2, e3, e4, or e5)
python3 -m toolspeed.cli run --experiment e1 --trials 10000

# Generate Evidence Log, SVGs, and interactive HTML dashboard
python3 -m toolspeed.cli report --out results

# Run the full unit & adversarial test suite
python3 -m unittest discover -s tests -v
```

---

## 🛡️ Guardrails & Adversarial Hardening

ToolSpeeder tracks and enforces:
- **Strict Metric Integrity**: Latency percentiles never incorporate failed or invalid task completions.
- **Async Concurrency Safety**: Zero dangling tasks or leaked coroutines upon speculative cancellation.
- **Cyclic Deadlock Defense**: Detects circular dependency graphs and fails fast.
- **Deoptimization Resilience**: Fused kernels catching runtime exceptions automatically deoptimize back to interactive reasoning.
- **Side-Effect Protection**: Strict approval gates and idempotency key caching for mutative tools.
- **Rate-Limiter Backpressure**: Token-bucket 429 simulation and peak concurrency tracking.

---

## 📁 Repository Structure

```
toolspeed/
├── core/             # Nanosecond profiler, guardrails, rate limiters, types
├── adapters/         # Simulated & live async tools (SQLite, Subprocess, HTTP, Files)
├── workloads/        # W1-W7 canonical benchmark workload generators
├── schedulers/       # B1-B5 baselines, E1-E5 mechanisms, Cache, and Composite
├── experiments/      # Statistical runners, hypothesis falsification checkers
└── visualization/    # Standalone SVG vector charts, HTML dashboard, Markdown logs

tests/
├── test_core.py          # Core types and profiler tests
├── test_adapters.py      # Mock and live adapters tests
├── test_workloads.py     # Workload correctness and validation tests
├── test_schedulers.py    # Baseline & experimental scheduler tests
├── test_experiments.py   # Statistical runner & CLI tests
└── test_adversarial.py   # 15 red-team adversarial attack tests
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
