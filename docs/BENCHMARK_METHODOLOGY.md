# ToolSpeed Benchmark Methodology

This document defines the scientific benchmarking methodology, statistical criteria, and paired evaluation protocol used across ToolSpeed.

## Core Principles

1. **Paired Comparisons**: Every candidate scheduler is evaluated on the exact same task instance, tool fixture, and model response as its baseline scheduler.
2. **Correct Completion Latency (CCL)**: Latency metrics (P50, P90, P95, P99) are computed strictly on trials where the task completed successfully and passed semantic validation. Failed trials are accounted for in the Success Rate metric and must not artificially depress measured latency.
3. **Execution Order Counterbalancing**: On real wall-clock backends, trial execution order alternates (Trial 0: Baseline then Candidate; Trial 1: Candidate then Baseline) to eliminate thermal throttling and cache sequence bias.
4. **Paired Bootstrap Resampling**: All speedup confidence intervals are computed via paired bootstrap resampling ($B = 1000$ iterations) over paired trial indices.
5. **Negative Control Verification**: The benchmark suite includes negative controls (e.g., candidate with optimization disabled) to prove that the measurement harness produces ~1.00x speedup when no optimization is active.

## Workload Families (W1 – W7)

| ID | Family | Description | Baseline | Candidate |
|---|---|---|---|---|
| **W1** | Independent Fanout Reads | 5 independent shard reads dispatched concurrently | `SyncReActScheduler` | `DAGScheduler` (E1) |
| **W2** | Deterministic Dependent Chains | Two-step user $\to$ orders query pipeline | `SyncReActScheduler` | `JITFusionScheduler` (E2) |
| **W3** | Branching with Speculative Read | Speculative read-only query during model reasoning | `SyncReActScheduler` | `SpeculativeReadScheduler` (E3) |
| **W4** | Repeated Workflows (Locality) | Repeated read queries across sequence with Zipfian locality | `SyncReActScheduler` | `CacheScheduler` |
| **W5** | Large Payloads & Early Commit | Incremental streaming with early commit horizon dispatch | `SyncReActScheduler` | `CommitHorizonScheduler` (E4) |
| **W6** | Sandbox Cold-Start | Sandbox execution with cold-start initialization | `SyncReActScheduler` (Cold) | `CompositeScheduler` (Prewarmed) |
| **W7** | Side-Effects & Idempotency | Mutative fund transfer requiring approval and idempotency | `SyncReActScheduler` | `CompositeScheduler` |
| **E5a** | Action Bytecode Codec | Binary transport packet serialization vs JSON | `SyncReActScheduler` | `ActionBytecodeScheduler` |

## Statistical Metrics & Falsification Criteria

- **P95 Speedup**: $\text{Speedup}_{P95} = \frac{\text{Baseline } P95}{\text{Candidate } P95}$
- **Target Threshold**: $\text{Speedup}_{P95} \ge 1.05\times$ and $\text{Candidate Success Rate} \ge 95.0\%$.
- **Null Check**: Negative controls must satisfy $|\text{Speedup}_{P95} - 1.00| \le 0.25$.
