# ToolSpeed Paired Benchmark Suite (Local Wall-Clock Backend)

**Generated:** 2026-08-28 09:01:53 UTC  
**Evidence Level:** `local_wall_clock`  
**Overall Verdict:** `PASSED`  
**Total Benchmark Runtime:** 41.29s  
**Git Commit:** `8fba20d481069289f93c6f1b34312272928fdb49` (dirty)  
**Platform:** `Darwin 27.0.0 (arm64)` | Python `3.14.6`  

## Paired Workload Evaluations (W1 – W7, E5a)

| Workload | Comparison | Baseline P95 | Candidate P95 | P95 Speedup | Candidate Success | Status | 95% Bootstrap CI |
|---|---|---|---|---|---|---|---|
| W1 | DAGScheduler vs SyncReActScheduler | 185.0ms | 85.0ms | 2.18x | 100.0% | ✅ PASS | [54.1%, 54.1%] |
| W2 | JITFusionScheduler vs SyncReActScheduler | 100.9ms | 15.5ms | 6.50x | 100.0% | ✅ PASS | [83.7%, 85.5%] |
| W3 | SpeculativeReadScheduler vs SyncReActScheduler | 85.0ms | 60.0ms | 1.42x | 100.0% | ✅ PASS | [29.4%, 29.4%] |
| W4 | CacheScheduler vs SyncReActScheduler | 125.0ms | 90.2ms | 1.39x | 100.0% | ✅ PASS | [27.8%, 27.8%] |
| W5 | CommitHorizonScheduler vs SyncReActScheduler | 85.0ms | 72.5ms | 1.17x | 100.0% | ✅ PASS | [14.7%, 14.7%] |
| W6 | CompositeScheduler vs SyncReActScheduler | 110.0ms | 88.6ms | 1.24x | 100.0% | ✅ PASS | [19.0%, 20.2%] |
| W7 | CompositeScheduler vs SyncReActScheduler | 60.1ms | 60.1ms | 1.00x | 100.0% | ✅ PASS | [-0.0%, 0.1%] |
| E5a | ActionBytecodeScheduler vs SyncReActScheduler | 60.0ms | 36.0ms | 1.67x | 100.0% | ✅ PASS | [40.0%, 40.0%] |

## Negative Control Verification

| Control | Measured Speedup | Null Check (~1.0x) | Detail |
|---|---|---|---|
| E1_parallelism_disabled | 1.00x | ✅ PASS | Proves disabled E1 parallelism produces ~1.0x speedup as expected |
| E2_fusion_disabled | 0.99x | ✅ PASS | Proves disabled E2 fusion produces ~1.0x speedup as expected |
| E3_speculation_disabled | 1.00x | ✅ PASS | Proves disabled E3 produces ~1.0x speedup as expected |
| E4_early_dispatch_disabled | 1.00x | ✅ PASS | Proves disabled E4 produces ~1.0x speedup as expected |
| Cache_disabled | 1.00x | ✅ PASS | Proves disabled Cache produces ~1.0x speedup as expected |
| Positive_sensitivity_injected_50pct_speedup | 2.00x | ✅ PASS | Proves harness detects and confirms positive latency reductions |