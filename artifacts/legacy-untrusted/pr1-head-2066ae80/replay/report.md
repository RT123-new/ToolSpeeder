# ToolSpeed Paired Benchmark Suite (Replay Backend)

**Generated:** 2026-08-28 07:07:56 UTC  
**Evidence Level:** `replay_integration`  
**Overall Verdict:** `FALSIFIED`  
**Total Benchmark Runtime:** 25.80s  
**Git Commit:** `2066ae80e40905e7ffca50bb148d74700094101a` (dirty)  
**Platform:** `Darwin 27.0.0 (arm64)` | Python `3.10.20`  

## Paired Workload Evaluations (W1 – W7, E5a)

| Workload | Comparison | Baseline P95 | Candidate P95 | P95 Speedup | Candidate Success | Status | 95% Bootstrap CI |
|---|---|---|---|---|---|---|---|
| W1 | DAGScheduler vs SyncReActScheduler | 205.1ms | 94.0ms | 2.18x | 100.0% | ✅ PASS | [53.5%, 54.7%] |
| W2 | JITFusionScheduler vs SyncReActScheduler | 120.6ms | 147.9ms | 0.82x | 100.0% | ❌ FAIL | [-24.0%, -20.8%] |
| W3 | SpeculativeReadScheduler vs SyncReActScheduler | 92.5ms | 66.2ms | 1.40x | 100.0% | ✅ PASS | [27.8%, 29.0%] |
| W4 | CacheScheduler vs SyncReActScheduler | 137.5ms | 134.9ms | 1.02x | 100.0% | ❌ FAIL | [1.1%, 29.4%] |
| W5 | CommitHorizonScheduler vs SyncReActScheduler | 92.7ms | 103.7ms | 0.89x | 100.0% | ❌ FAIL | [-13.3%, -10.1%] |
| W6 | CompositeScheduler vs SyncReActScheduler | 163.0ms | 92.4ms | 1.76x | 100.0% | ✅ PASS | [42.9%, 43.8%] |
| W7 | CompositeScheduler vs SyncReActScheduler | 91.9ms | 103.2ms | 0.89x | 100.0% | ❌ FAIL | [-13.0%, -11.7%] |
| E5a | ActionBytecodeScheduler vs SyncReActScheduler | 93.0ms | 92.7ms | 1.00x | 100.0% | ❌ FAIL | [-0.3%, 0.5%] |

## Negative Control Verification

| Control | Measured Speedup | Null Check (~1.0x) | Detail |
|---|---|---|---|
| E1_disabled | 0.99x | ✅ PASS | Proves disabled E1 produces ~1.0x speedup as expected |
| E3_disabled | 1.00x | ✅ PASS | Proves disabled E3 produces ~1.0x speedup as expected |
| E4_disabled | 1.00x | ✅ PASS | Proves disabled E4 produces ~1.0x speedup as expected |