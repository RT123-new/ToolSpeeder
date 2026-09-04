# ToolSpeed Paired Benchmark Suite (REPLAY_INTEGRATION)

**Generated:** 2026-09-04 10:24:16 UTC  
**Evidence Level:** `replay_integration`  
**Overall Verdict:** `PASSED`  
**Total Benchmark Runtime:** 7.30s  
**Git Commit:** `3fbc4b692137e56d82c9e531702ca7e448c6a7e7` (clean)  
**Platform:** `Darwin 27.0.0 (arm64)` | Python `3.13.15`  

## Paired Workload Evaluations (W1 – W7, E5a)

| Workload | Comparison | Baseline P95 | Candidate P95 | P95 Speedup | Candidate Success | Status | 95% Bootstrap CI |
|---|---|---|---|---|---|---|---|
| W1 | DAGScheduler vs DAGScheduler_serial_ablation | 179.5ms | 82.9ms | 2.17x | 100.0% | ✅ PASS | [53.6%, 54.0%] |
| W2 | JITFusionScheduler vs JITFusionScheduler_fusion_disabled | 135.8ms | 49.5ms | 2.74x | 100.0% | ✅ PASS | [63.3%, 63.7%] |
| W3 | SpeculativeReadScheduler vs SpeculativeReadScheduler_speculation_disabled | 82.6ms | 82.9ms | 1.00x | 100.0% | ✅ PASS | [-1.0%, 0.2%] |
| W4 | CacheScheduler vs CacheScheduler_caching_disabled | 82.6ms | 82.9ms | 1.00x | 100.0% | ✅ PASS | [-1.0%, 0.2%] |
| W5 | CommitHorizonScheduler vs CommitHorizonScheduler_commit_horizon_disabled | 92.6ms | 78.0ms | 1.19x | 100.0% | ✅ PASS | [15.1%, 16.1%] |
| W6 | CompositeScheduler vs SyncReActScheduler | 87.6ms | 87.9ms | 1.00x | 100.0% | ✅ PASS | [-0.9%, 0.2%] |
| W7_SAFETY | CompositeScheduler vs CompositeScheduler | 87.6ms | 87.9ms | 1.00x | 100.0% | ✅ PASS | [-0.9%, 0.2%] |
| W7_LATENCY | CompositeScheduler vs SyncReActScheduler | 87.6ms | 87.9ms | 1.00x | 100.0% | ✅ PASS | [-0.9%, 0.2%] |
| E5a | ActionBytecodeScheduler vs SyncReActScheduler | 82.6ms | 82.9ms | 1.00x | 100.0% | ✅ PASS | [-1.0%, 0.2%] |

## Negative Control Verification

| Control | Measured Speedup | Null Check (~1.0x) | Detail |
|---|---|---|---|
| E1_parallelism_disabled | 1.00x | ✅ PASS | Proves disabled E1 parallelism against itself produces ~1.0x speedup |
| E2_fusion_disabled | 1.00x | ✅ PASS | Proves disabled E2 fusion against itself produces ~1.0x speedup |
| E3_speculation_disabled | 1.00x | ✅ PASS | Proves disabled E3 speculation against itself produces ~1.0x speedup |
| E4_early_dispatch_disabled | 1.00x | ✅ PASS | Proves disabled E4 early dispatch against itself produces ~1.0x speedup |
| Cache_disabled | 1.00x | ✅ PASS | Proves disabled cache against itself produces ~1.0x speedup |
| Positive_sensitivity_injected_50pct_speedup | 2.00x | ✅ PASS | Proves harness measures genuine positive latency reductions via execution |