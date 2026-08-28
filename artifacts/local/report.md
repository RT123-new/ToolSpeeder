# ToolSpeed Paired Benchmark Suite (LOCAL_WALL_CLOCK)

**Generated:** 2026-08-28 19:36:11 UTC  
**Evidence Level:** `local_wall_clock`  
**Overall Verdict:** `FALSIFIED`  
**Total Benchmark Runtime:** 18.92s  
**Git Commit:** `14c19751c218b8498712060e2aee81f62283390c` (dirty)  
**Platform:** `Darwin 27.0.0 (arm64)` | Python `3.13.15`  

## Paired Workload Evaluations (W1 – W7, E5a)

| Workload | Comparison | Baseline P95 | Candidate P95 | P95 Speedup | Candidate Success | Status | 95% Bootstrap CI |
|---|---|---|---|---|---|---|---|
| W1 | DAGScheduler vs SyncReActScheduler | 5.8ms | 5.4ms | 1.07x | 100.0% | ❌ FAIL | [-4.4%, 15.7%] |
| W2 | JITFusionScheduler vs SyncReActScheduler | 6.2ms | 7.5ms | 0.83x | 100.0% | ❌ FAIL | [-33.1%, -11.1%] |
| W3 | SpeculativeReadScheduler vs SyncReActScheduler | 5.6ms | 5.9ms | 0.95x | 100.0% | ❌ FAIL | [-12.7%, 1.0%] |
| W4 | CacheScheduler vs SyncReActScheduler | 5.9ms | 5.8ms | 1.02x | 100.0% | ❌ FAIL | [-11.4%, 9.1%] |
| W5 | CommitHorizonScheduler vs SyncReActScheduler | 5.5ms | 5.5ms | 0.99x | 100.0% | ❌ FAIL | [-7.9%, 6.8%] |
| W6 | CompositeScheduler vs SyncReActScheduler | 6.9ms | 7.1ms | 0.97x | 100.0% | ❌ FAIL | [-7.0%, 1.8%] |
| W7 | CompositeScheduler vs SyncReActScheduler | 5.3ms | 6.5ms | 0.81x | 100.0% | ❌ FAIL | [-36.6%, -11.3%] |
| E5a | ActionBytecodeScheduler vs SyncReActScheduler | 4.3ms | 4.6ms | 0.94x | 100.0% | ❌ FAIL | [-16.6%, 4.6%] |

## Negative Control Verification

| Control | Measured Speedup | Null Check (~1.0x) | Detail |
|---|---|---|---|
| E1_parallelism_disabled | 1.03x | ✅ PASS | Proves disabled E1 parallelism produces ~1.0x speedup as expected |
| E2_fusion_disabled | 1.07x | ❌ FAIL | Proves disabled E2 fusion produces ~1.0x speedup as expected |
| E3_speculation_disabled | 0.97x | ✅ PASS | Proves disabled E3 produces ~1.0x speedup as expected |
| E4_early_dispatch_disabled | 0.92x | ❌ FAIL | Proves disabled E4 produces ~1.0x speedup as expected |
| Cache_disabled | 0.95x | ❌ FAIL | Proves disabled Cache produces ~1.0x speedup as expected |
| Positive_sensitivity_injected_50pct_speedup | 2.00x | ✅ PASS | Proves harness detects and confirms positive latency reductions |