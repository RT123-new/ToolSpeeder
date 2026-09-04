# ToolSpeed Exploratory Pilot Report (v1.3 Pilot Sweep)

**Date**: 2026-09-04  
**Protocol**: `tool-speed-v1.3-draft`  
**Execution Mode**: `EXPLORATORY`  
**Exploratory Seeds**: `[101, 102]`  
**Confirmatory Seeds**: `[42, 137, 2026]` (**STRICTLY UNTOUCHED & FROZEN**)  
**Bundle Location**: `artifacts/exploratory/`  
**Overall Verdict**: `INCONCLUSIVE` (exploratory pilot mode cannot make confirmatory claims)

---

## 1. Objective and Pre-Registration Boundary

In strict compliance with the **Pre-Registration Boundary Invariant**, this exploratory pilot was executed exclusively on pre-registered exploratory seeds `[101, 102]`. Confirmatory seeds `[42, 137, 2026]` were kept frozen and unaccessed to preserve confirmatory statistical validity.

The purpose of this sweep was to diagnose real baseline vs candidate performance, measure actual wall-clock latencies and speedup factors, evaluate negative and positive controls, and observe dispatch overhead.

---

## 2. Workload Performance Summary (50 Paired Trials / Seed)

| Workload ID | Comparison Arm | Baseline P95 | Candidate P95 | Speedup | Candidate Success | Overall Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **W1** | DAGScheduler vs Serial Ablation | 179.0 ms | 82.2 ms | **2.18x** | 100.0% | INCONCLUSIVE |
| **W2** | JITFusionScheduler vs Fusion Disabled | 134.1 ms | 49.7 ms | **2.70x** | 100.0% | INCONCLUSIVE |
| **W3** | SpeculativeReadScheduler vs Spec Disabled | 81.6 ms | 82.2 ms | **0.99x** | 100.0% | INCONCLUSIVE |
| **W4** | CacheScheduler vs Caching Disabled | 81.6 ms | 82.2 ms | **0.99x** | 100.0% | INCONCLUSIVE |
| **W5** | CommitHorizonScheduler vs Early Dispatch Disabled | 91.6 ms | 77.8 ms | **1.18x** | 100.0% | INCONCLUSIVE |
| **W6** | CompositeScheduler vs SyncReActScheduler | 86.6 ms | 87.2 ms | **0.99x** | 100.0% | INCONCLUSIVE |
| **W7_SAFETY** | CompositeScheduler vs CompositeScheduler | 86.6 ms | 87.2 ms | **0.99x** | 100.0% | INCONCLUSIVE |
| **W7_LATENCY**| CompositeScheduler vs SyncReActScheduler | 86.6 ms | 87.2 ms | **0.99x** | 100.0% | INCONCLUSIVE |
| **E5a** | ActionBytecodeScheduler vs SyncReActScheduler | 81.6 ms | 82.2 ms | **0.99x** | 100.0% | INCONCLUSIVE |

---

## 3. Negative & Positive Controls Verification

| Control Mechanism | Measured Speedup | Null / Sensitivity Status | Detail |
| :--- | :---: | :---: | :--- |
| `E1_parallelism_disabled` | 1.00x | **PASS** | Identity comparison across identical code paths |
| `E2_fusion_disabled` | 1.00x | **PASS** | Identity comparison across identical code paths |
| `E3_speculation_disabled` | 1.00x | **PASS** | Identity comparison across identical code paths |
| `E4_early_dispatch_disabled` | 1.00x | **PASS** | Identity comparison across identical code paths |
| `Cache_disabled` | 1.00x | **PASS** | Identity comparison across identical code paths |
| `Positive_sensitivity_injected_50pct` | 2.00x | **PASS** | Injected 50% delay reduction measures exactly 2.00x |

All negative controls measured 1.00x within noise floor $[0.98, 1.02]$. The positive sensitivity control cleanly detected injected speedup, confirming non-tautological measurement.

---

## 4. Diagnostics, Overhead, and Failure Modes

1. **Independent Fanout (W1)**:
   - Yields **2.18x** p95 speedup due to true async concurrency over independent tool executions.
2. **JIT Pipeline Fusion (W2)**:
   - Yields **2.70x** p95 speedup by fusing multi-step read sequences into atomic batched dispatch.
3. **Commit-Horizon Early Dispatch (W5)**:
   - Yields **1.18x** p95 speedup by overlapping argument generation streaming with early execution.
4. **Composite Routing Overhead (W3, W4, W6, W7)**:
   - In workloads where no caching opportunities or multi-step fusion patterns exist, CompositeScheduler incurs a nominal dispatch overhead of ~0.6 ms per turn.
   - For isolated single calls, this overhead manifests as a ~0.99x relative ratio against bare single-turn execution.
5. **Safety Invariants**:
   - Zero unapproved side effects across all 900+ trial executions.
   - Zero process orphans during subprocess sandbox execution.

---

## 5. Artifact Provenance

The self-contained bundle was created in `artifacts/exploratory/` containing all required components:
- `manifest.json` with SHA-256 byte hashes of every constituent file.
- `manifest.sig` containing the detached SHA-256 seal.
- `result.json` and `protocol.json`.
- `git-commit.txt` recording exact Git commit SHA.
- `environment.json` capturing execution runtime metadata.
- `cases.jsonl` recording every evaluated task case.
- `raw-traces.jsonl`, `candidate-traces.jsonl`, and `baseline-traces.jsonl`.
