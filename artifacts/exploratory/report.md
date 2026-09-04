# ToolSpeed Paired Benchmark Suite (REPLAY_INTEGRATION)

**Generated:** 2026-09-04 10:20:10 UTC  
**Evidence Level:** `replay_integration`  
**Overall Verdict:** `INCONCLUSIVE`  
**Total Benchmark Runtime:** 1.82s  
**Git Commit:** `b8edbdfcfdfb64cd502021cd3f73932700fc61fd` (clean)  
**Platform:** `Darwin 27.0.0 (arm64)` | Python `3.13.15`  

## Paired Workload Evaluations (W1 – W7, E5a)

| Workload | Comparison | Baseline P95 | Candidate P95 | P95 Speedup | Candidate Success | Status | 95% Bootstrap CI |
|---|---|---|---|---|---|---|---|
| W1 | DAGScheduler vs DAGScheduler_serial_ablation | 179.0ms | 82.2ms | 2.18x | 100.0% | ❌ FAIL | [53.2%, 54.4%] |
| W2 | JITFusionScheduler vs JITFusionScheduler_fusion_disabled | 134.1ms | 49.7ms | 2.70x | 100.0% | ❌ FAIL | [62.6%, 63.7%] |
| W3 | SpeculativeReadScheduler vs SpeculativeReadScheduler_speculation_disabled | 81.6ms | 82.2ms | 0.99x | 100.0% | ❌ FAIL | [-2.5%, 0.1%] |
| W4 | CacheScheduler vs CacheScheduler_caching_disabled | 81.6ms | 82.2ms | 0.99x | 100.0% | ❌ FAIL | [-2.5%, 0.1%] |
| W5 | CommitHorizonScheduler vs CommitHorizonScheduler_commit_horizon_disabled | 91.6ms | 77.8ms | 1.18x | 100.0% | ❌ FAIL | [13.8%, 15.9%] |
| W6 | CompositeScheduler vs SyncReActScheduler | 86.6ms | 87.2ms | 0.99x | 100.0% | ❌ FAIL | [-2.4%, 0.1%] |
| W7_SAFETY | CompositeScheduler vs CompositeScheduler | 86.6ms | 87.2ms | 0.99x | 100.0% | ❌ FAIL | [-2.4%, 0.1%] |
| W7_LATENCY | CompositeScheduler vs SyncReActScheduler | 86.6ms | 87.2ms | 0.99x | 100.0% | ❌ FAIL | [-2.4%, 0.1%] |
| E5a | ActionBytecodeScheduler vs SyncReActScheduler | 81.6ms | 82.2ms | 0.99x | 100.0% | ❌ FAIL | [-2.5%, 0.1%] |

## Negative Control Verification

| Control | Measured Speedup | Null Check (~1.0x) | Detail |
|---|---|---|---|
| E1_parallelism_disabled | 1.00x | ✅ PASS | Proves disabled E1 parallelism against itself produces ~1.0x speedup |
| E2_fusion_disabled | 1.00x | ✅ PASS | Proves disabled E2 fusion against itself produces ~1.0x speedup |
| E3_speculation_disabled | 1.00x | ✅ PASS | Proves disabled E3 speculation against itself produces ~1.0x speedup |
| E4_early_dispatch_disabled | 1.00x | ✅ PASS | Proves disabled E4 early dispatch against itself produces ~1.0x speedup |
| Cache_disabled | 1.00x | ✅ PASS | Proves disabled cache against itself produces ~1.0x speedup |
| Positive_sensitivity_injected_50pct_speedup | 2.00x | ✅ PASS | Proves harness measures genuine positive latency reductions via execution |