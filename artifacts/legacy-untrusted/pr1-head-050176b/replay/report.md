# ToolSpeed Paired Benchmark Suite (REPLAY_INTEGRATION)

**Generated:** 2026-08-28 19:35:46 UTC  
**Evidence Level:** `replay_integration`  
**Overall Verdict:** `FALSIFIED`  
**Total Benchmark Runtime:** 5.07s  
**Git Commit:** `14c19751c218b8498712060e2aee81f62283390c` (dirty)  
**Platform:** `Darwin 27.0.0 (arm64)` | Python `3.13.15`  

## Paired Workload Evaluations (W1 – W7, E5a)

| Workload | Comparison | Baseline P95 | Candidate P95 | P95 Speedup | Candidate Success | Status | 95% Bootstrap CI |
|---|---|---|---|---|---|---|---|
| W1 | DAGScheduler vs SyncReActScheduler | 150.0ms | 70.0ms | 2.14x | 100.0% | ✅ PASS | [53.3%, 53.3%] |
| W2 | JITFusionScheduler vs SyncReActScheduler | 115.0ms | 40.0ms | 2.88x | 100.0% | ✅ PASS | [65.2%, 65.2%] |
| W3 | SpeculativeReadScheduler vs SyncReActScheduler | 75.0ms | 75.0ms | 1.00x | 100.0% | ❌ FAIL | [0.0%, 0.0%] |
| W4 | CacheScheduler vs SyncReActScheduler | 75.0ms | 75.0ms | 1.00x | 100.0% | ❌ FAIL | [0.0%, 0.0%] |
| W5 | CommitHorizonScheduler vs SyncReActScheduler | 100.0ms | 82.5ms | 1.21x | 100.0% | ✅ PASS | [17.5%, 17.5%] |
| W6 | CompositeScheduler vs SyncReActScheduler | 75.0ms | 75.0ms | 1.00x | 100.0% | ❌ FAIL | [0.0%, 0.0%] |
| W7 | CompositeScheduler vs SyncReActScheduler | 75.0ms | 75.0ms | 1.00x | 100.0% | ✅ PASS | [0.0%, 0.0%] |
| E5a | ActionBytecodeScheduler vs SyncReActScheduler | 70.0ms | 70.0ms | 1.00x | 100.0% | ❌ FAIL | [0.0%, 0.0%] |

## Negative Control Verification

| Control | Measured Speedup | Null Check (~1.0x) | Detail |
|---|---|---|---|
| E1_parallelism_disabled | 1.00x | ✅ PASS | Proves disabled E1 parallelism produces ~1.0x speedup as expected |
| E2_fusion_disabled | 1.00x | ✅ PASS | Proves disabled E2 fusion produces ~1.0x speedup as expected |
| E3_speculation_disabled | 1.00x | ✅ PASS | Proves disabled E3 produces ~1.0x speedup as expected |
| E4_early_dispatch_disabled | 1.00x | ✅ PASS | Proves disabled E4 produces ~1.0x speedup as expected |
| Cache_disabled | 1.00x | ✅ PASS | Proves disabled Cache produces ~1.0x speedup as expected |
| Positive_sensitivity_injected_50pct_speedup | 2.00x | ✅ PASS | Proves harness detects and confirms positive latency reductions |