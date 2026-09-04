# ToolSpeed Benchmark Methodology (Protocol `tool-speed-v1.1`)

This document defines the authoritative scientific benchmarking methodology, statistical criteria, paired evaluation protocol, and hypothesis falsification rules used across ToolSpeed under frozen protocol [`tool-speed-v1.1.json`](file:///Users/regtroka/Downloads/ToolSpeed/benchmark-plans/tool-speed-v1.1.json).

---

## 1. Core Principles & Evaluation Architecture

1. **Prospectively Frozen Protocol**: All comparisons, thresholds, sample sizes, seeds, and aggregation rules are defined in `benchmark-plans/tool-speed-v1.1.json` prior to evidence collection.
2. **Dual-Baseline Architecture**:
   - **Primary Attribution Baseline**: Isolates mechanism efficacy by evaluating an identical scheduler architecture with only the specific optimization disabled (ablation baseline).
   - **Practical Baseline**: Evaluates performance gains against standard sequential ReAct (`SyncReActScheduler`) or handwritten workflows.
3. **Correct Completion Latency (CCL)**: Latency metrics ($P_{50}, P_{90}, P_{95}, P_{99}$) are calculated strictly on trials where the task completed successfully and satisfied all oracle correctness and state invariants. Failed trials are accounted for in the Success Rate metric and do not artificially deflate measured latency.
4. **Oracle Separation Boundary**: The model/agent receives strictly an immutable `AgentTask` containing whitelisted model-visible metadata. Ground-truth expectations, approval grants, and validation functions reside exclusively in `BenchmarkCase` on the evaluation side.
5. **Execution Order Counterbalancing**: On real wall-clock backends, trial execution order alternates (Trial 0: Baseline then Candidate; Trial 1: Candidate then Baseline) to eliminate thermal throttling, cache warmth, and sequence bias.
6. **Paired Bootstrap Resampling**: All speedup confidence intervals are computed via paired bootstrap resampling ($B = 2000$ iterations, 95% CI) over paired trial indices.
7. **Negative & Sensitivity Controls**:
   - **Negative Controls**: Paired identical arms evaluating disabled optimizations to confirm the null equivalence region $[0.95\times, 1.05\times]$.
   - **Positive Sensitivity Control**: Real execution with an injected 50% tool execution delay to verify that the measurement harness accurately detects positive speedups within $[1.80\times, 2.20\times]$.
8. **Missing Data Policy**: Any missing or null metric in a required workload yields an immediate `null_inconclusive` status.

---

## 2. Frozen Workload Matrix & Pairings

| Workload ID | Name / Mechanism | Candidate | Primary Attribution Baseline | Practical Baseline | Efficacy Threshold |
|---|---|---|---|---|---|
| **W1** | Dynamic DAG Scheduling (E1) | `DAGScheduler` | `DAGScheduler_serial_ablation` | `NativeParallelScheduler` | $P_{95} \ge 1.20\times$, Success $\ge 95\%$ |
| **W2** | Declarative JIT Fusion (E2) | `JITFusionScheduler` | `JITFusionScheduler_fusion_disabled` | `HandwrittenWorkflowScheduler` | $P_{95} \ge 1.20\times$, Success $\ge 95\%$ |
| **W3** | Speculative Reads (E3) | `SpeculativeReadScheduler` | `SpeculativeReadScheduler_spec_disabled` | `SyncReActScheduler` | $P_{95} \ge 1.11\times$, Success $\ge 95\%$ |
| **W4** | Locality & Domain Caching | `CacheScheduler` | `CacheScheduler_cache_disabled` | `SyncReActScheduler` | $P_{95} \ge 1.11\times$, Success $\ge 95\%$ |
| **W5** | Streaming Commit Horizon (E4) | `CommitHorizonScheduler` | `CommitHorizonScheduler_early_dispatch_disabled` | `SyncReActScheduler` | $P_{95} \ge 1.10\times$, Success $\ge 95\%$ |
| **W6** | Persistent Pool Prewarming | `PersistentPrewarmedPool` | `PersistentColdPool` | `SyncReActScheduler` | $P_{95} \ge 1.10\times$, Success $\ge 95\%$ |
| **W7_SAFETY** | Side-Effect Safety & Idempotency Gate | `CompositeScheduler` | `IdenticalAuthorizedExecutionPath` | `SyncReActScheduler` | Unapproved $= 0$, Duplicates $= 0$, Success $= 100\%$ |
| **W7_LATENCY** | Side-Effect Latency Overhead | `CompositeScheduler` | `IdenticalAuthorizedExecutionPath` | `SyncReActScheduler` | $P_{95} \ge 1.00\times$, $P_{99} \ge 0.95\times$ |
| **E5a** | Action Bytecode Transport Codec | `ActionBytecodeCodec` | `JSONCodec` | `JSONCodec` | $P_{95} \ge 1.05\times$, Roundtrip Loss $= 0$ |
| **E5b** | Direct Action-Token Generation | *Unimplemented* | — | — | `INCONCLUSIVE` |

---

## 3. Evidence Levels & Verdict Eligibility

- **`replay_integration`**: Minimum 1,000 trials per condition across pre-registered seeds `[20260825, 20260826, 20260827]`. Deterministic replay timing.
- **`local_wall_clock`**: Minimum 200 trials per condition across pre-registered seeds. Real local OS loopback and subprocess primitives.
- **Smoke Runs** ($n < 1000$ replay or $n < 200$ local): Marked `SMOKE — NOT VERDICT-ELIGIBLE` (Falsify exit code `2`).
- **Hypothesis Status**:
  - `PASSED`: All primary mechanisms meet or exceed pre-registered speedup targets and safety gates.
  - `FALSIFIED`: One or more primary mechanisms fail speedup thresholds or violate safety invariants (Falsify exit code `1`).
  - `INCONCLUSIVE`: Insufficient sample size or missing data (Falsify exit code `2`).

---

## 4. Atomic Bundle Format & Hashing

Canonical evidence bundles are atomically staged and sealed with standard provenance manifests:
```
<bundle_dir>/
├── manifest.json              # Provenance metadata and payload SHA-256 byte hashes
├── protocol.json              # Copy of frozen protocol specification (v1.1)
├── cases.jsonl                # BenchmarkCase items with model-isolated tasks and oracle targets
├── baseline-traces.jsonl      # Raw baseline execution traces
├── candidate-traces.jsonl     # Raw candidate execution traces
├── controls-traces.jsonl      # Raw negative and positive control execution traces
├── falsification.json         # Recomputed falsification verdicts and summaries
├── result.json                # Canonical result payload
├── report.md                  # Markdown evaluation report
├── report.html                # Interactive HTML dashboard
└── bundle.sha256              # SHA-256 checksums of all bundle artifacts
```
Manifest verification is performed via `toolspeed validate-bundle --input <bundle_dir>`, verifying exact byte hashes, schema fields, and non-null metric policies.
