# ToolSpeed Paired Benchmark Suite (Local Wall-Clock Backend)

**Generated:** 2026-08-28 07:39:09 UTC  
**Evidence Level:** `local_wall_clock`  
**Overall Verdict:** `FALSIFIED`  
**Total Benchmark Runtime:** 9.26s  
**Git Commit:** `2066ae80e40905e7ffca50bb148d74700094101a` (dirty)  
**Platform:** `Darwin 27.0.0 (arm64)` | Python `3.10.20`  

## Paired Workload Evaluations (W1 – W7, E5a)

| Workload | Comparison | Baseline P95 | Candidate P95 | P95 Speedup | Candidate Success | Status | 95% Bootstrap CI |
|---|---|---|---|---|---|---|---|
| W1 | DAGScheduler vs SyncReActScheduler | 75.7ms | 74.5ms | 1.02x | 100.0% | ❌ FAIL | [-2.3%, 8.3%] |
| W2 | JITFusionScheduler vs SyncReActScheduler | 38.3ms | 19.8ms | 1.94x | 100.0% | ✅ PASS | [47.6%, 49.7%] |
| W3 | SpeculativeReadScheduler vs SyncReActScheduler | 25.6ms | 26.4ms | 0.97x | 100.0% | ❌ FAIL | [-8.1%, 3.4%] |
| W4 | CacheScheduler vs SyncReActScheduler | 40.6ms | 38.2ms | 1.06x | 100.0% | ✅ PASS | [-1.1%, 11.9%] |
| W5 | CommitHorizonScheduler vs SyncReActScheduler | 26.0ms | 31.6ms | 0.82x | 100.0% | ❌ FAIL | [-27.1%, -13.8%] |
| W6 | CompositeScheduler vs SyncReActScheduler | 79.5ms | 32.2ms | 2.47x | 100.0% | ✅ PASS | [58.5%, 64.2%] |
| W7 | CompositeScheduler vs SyncReActScheduler | 27.4ms | 32.1ms | 0.85x | 100.0% | ❌ FAIL | [-24.9%, -2.5%] |
| E5a | ActionBytecodeScheduler vs SyncReActScheduler | 25.9ms | 25.5ms | 1.02x | 100.0% | ❌ FAIL | [-3.0%, 7.1%] |

## Negative Control Verification

| Control | Measured Speedup | Null Check (~1.0x) | Detail |
|---|---|---|---|
| E1_disabled | 0.97x | ✅ PASS | Proves disabled E1 produces ~1.0x speedup as expected |
| E3_disabled | 0.95x | ✅ PASS | Proves disabled E3 produces ~1.0x speedup as expected |
| E4_disabled | 0.98x | ✅ PASS | Proves disabled E4 produces ~1.0x speedup as expected |