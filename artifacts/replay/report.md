# ToolSpeed Paired Benchmark Suite (Replay Backend)

**Generated:** 2026-08-28 07:38:27 UTC  
**Evidence Level:** `replay_integration`  
**Overall Verdict:** `FALSIFIED`  
**Total Benchmark Runtime:** 93.05s  
**Git Commit:** `2066ae80e40905e7ffca50bb148d74700094101a` (dirty)  
**Platform:** `Darwin 27.0.0 (arm64)` | Python `3.10.20`  

## Paired Workload Evaluations (W1 – W7, E5a)

| Workload | Comparison | Baseline P95 | Candidate P95 | P95 Speedup | Candidate Success | Status | 95% Bootstrap CI |
|---|---|---|---|---|---|---|---|
| W1 | DAGScheduler vs SyncReActScheduler | 206.3ms | 96.5ms | 2.14x | 100.0% | ✅ PASS | [52.8%, 54.1%] |
| W2 | JITFusionScheduler vs SyncReActScheduler | 122.5ms | 58.1ms | 2.11x | 100.0% | ✅ PASS | [52.2%, 53.3%] |
| W3 | SpeculativeReadScheduler vs SyncReActScheduler | 93.6ms | 67.6ms | 1.38x | 100.0% | ✅ PASS | [27.1%, 28.5%] |
| W4 | CacheScheduler vs SyncReActScheduler | 139.8ms | 120.8ms | 1.16x | 100.0% | ✅ PASS | [2.0%, 28.5%] |
| W5 | CommitHorizonScheduler vs SyncReActScheduler | 95.2ms | 107.3ms | 0.89x | 100.0% | ❌ FAIL | [-14.2%, -5.8%] |
| W6 | CompositeScheduler vs SyncReActScheduler | 166.9ms | 96.6ms | 1.73x | 100.0% | ✅ PASS | [41.4%, 42.9%] |
| W7 | CompositeScheduler vs SyncReActScheduler | 94.9ms | 107.2ms | 0.89x | 100.0% | ❌ FAIL | [-14.2%, -11.7%] |
| E5a | ActionBytecodeScheduler vs SyncReActScheduler | 94.5ms | 95.0ms | 0.99x | 100.0% | ❌ FAIL | [-2.4%, 2.5%] |

## Negative Control Verification

| Control | Measured Speedup | Null Check (~1.0x) | Detail |
|---|---|---|---|
| E1_disabled | 1.00x | ✅ PASS | Proves disabled E1 produces ~1.0x speedup as expected |
| E3_disabled | 0.99x | ✅ PASS | Proves disabled E3 produces ~1.0x speedup as expected |
| E4_disabled | 0.97x | ✅ PASS | Proves disabled E4 produces ~1.0x speedup as expected |