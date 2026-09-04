# ToolSpeed Paired Benchmark Suite (LOCAL_WALL_CLOCK)

**Generated:** 2026-09-04 10:28:03 UTC  
**Evidence Level:** `local_wall_clock`  
**Overall Verdict:** `FALSIFIED`  
**Total Benchmark Runtime:** 62.68s  
**Git Commit:** `1a167ebade8278e35cc14d3ace5a7f9c5486f68d` (clean)  
**Platform:** `Darwin 27.0.0 (arm64)` | Python `3.13.15`  

## Paired Workload Evaluations (W1 – W7, E5a)

| Workload | Comparison | Baseline P95 | Candidate P95 | P95 Speedup | Candidate Success | Status | 95% Bootstrap CI |
|---|---|---|---|---|---|---|---|
| W1 | DAGScheduler vs DAGScheduler_serial_ablation | 162.8ms | 42.5ms | 3.83x | 100.0% | ✅ PASS | [73.4%, 75.0%] |
| W2 | JITFusionScheduler vs JITFusionScheduler_fusion_disabled | null | null | null | 0.0% | ❌ FAIL | null |
| W3 | SpeculativeReadScheduler vs SpeculativeReadScheduler_speculation_disabled | null | null | null | 0.0% | ❌ FAIL | null |
| W4 | CacheScheduler vs CacheScheduler_caching_disabled | null | null | null | 0.0% | ❌ FAIL | null |
| W5 | CommitHorizonScheduler vs CommitHorizonScheduler_commit_horizon_disabled | 5.5ms | 4.5ms | 1.22x | 100.0% | ✅ PASS | [9.6%, 24.6%] |
| W6 | CompositeScheduler vs SyncReActScheduler | 6.5ms | 6.9ms | 0.94x | 100.0% | ❌ FAIL | [-10.6%, -2.6%] |
| W7_SAFETY | CompositeScheduler vs CompositeScheduler | null | null | null | 0.0% | ❌ FAIL | null |
| W7_LATENCY | CompositeScheduler vs SyncReActScheduler | null | null | null | 0.0% | ❌ FAIL | null |
| E5a | ActionBytecodeScheduler vs SyncReActScheduler | 3.2ms | 3.4ms | 0.92x | 100.0% | ❌ FAIL | [-18.0%, -2.0%] |

## Negative Control Verification

| Control | Measured Speedup | Null Check (~1.0x) | Detail |
|---|---|---|---|
| E1_parallelism_disabled | 1.00x | ✅ PASS | Proves disabled E1 parallelism against itself produces ~1.0x speedup |
| E2_fusion_disabled | 1.00x | ✅ PASS | Proves disabled E2 fusion against itself produces ~1.0x speedup |
| E3_speculation_disabled | 1.00x | ✅ PASS | Proves disabled E3 speculation against itself produces ~1.0x speedup |
| E4_early_dispatch_disabled | 1.00x | ✅ PASS | Proves disabled E4 early dispatch against itself produces ~1.0x speedup |
| Cache_disabled | 1.00x | ✅ PASS | Proves disabled cache against itself produces ~1.0x speedup |
| Positive_sensitivity_injected_50pct_speedup | 2.00x | ✅ PASS | Proves harness measures genuine positive latency reductions via execution |