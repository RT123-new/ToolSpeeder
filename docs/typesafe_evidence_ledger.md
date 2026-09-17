# TypeSafe/Jev Scientific Evidence Ledger

**Protocol:** `benchmarks/protocols/typesafe-speculation-v1.0.json`  
**Status:** Frozen Confirmatory Audit  
**Date:** September 17, 2026  

---

## 1. Audited Evidence Ledger

| Assertion | Historical Status | Reconciled Audit Ground Truth | Evidence Source |
|---|---|---|---|
| **Live API Connection** | Blocked in initial PR #2 | Live connection confirmed with model `jev-1.13.0` over WAN | Raw traces in brain `0fa44afe`, commit `0ac7d81` |
| **Pilot Accuracy** | 5/5 (100%) | 5/5 tasks correct on pilot workload | `scratch/eval_w8_live.py` (steps 658, 660) |
| **Pilot WAN Latency** | ~706 ms | 629 ms to 785 ms, mean 706.2 ms | Raw request timings from `api.typesafe.ai` |
| **Noul Gating in Pilot** | Stated as >= 0.50 | In pilot, Noul was 0.34–0.46; blocked speculation. Changed to 0.30 in commit `0ac7d81` | `typesafe_jev.py` diff in commit `0ac7d81` |
| **3/3 Speculation Hits** | Claimed in report | Turn 1 speculative hits occurred with delay=1500ms; Turn 2 final answer cancelled draft | `scratch/live_reasoning_spec.py` output (step 686) |
| **Launched Calls Counter** | Displayed as 0 | `record_tool_dispatch` was uncalled in codebase; fixed in confirmatory branch | `toolspeed/core/guardrails.py:330` |
| **Report Break-Even Table** | Appeared as live | Measured via deterministic replay (45ms simulated latency), NOT live Jev WAN | `toolspeed/experiments/typesafe_runner.py:318` |
| **Held-Out Accuracy (N=120)** | Unproven | B2 (Det): 95.0% vs B3 (Jev): 95.8% | `confirmatory_evaluation_results_v1.0.json` |
| **Zero Speculative Mutations**| Verified | 0 unauthorized mutations across all red team tests (Kill condition satisfied) | `test_typesafe_confirmatory_safety_redteam.py` |
| **Egress Security Boundary** | Verified | Zero secrets or oracle keys cross egress boundary (AST barrier 0 leaks) | `test_typesafe_confirmatory_egress_redteam.py` |

---

## 2. Threshold Discrepancy Resolution Table

```
Run                      Commit   Provider      Choice Thresh  Noul Thresh  Reasoning Delay  Tool Latency  Selected Cand  Spec Launched?
----------------------------------------------------------------------------------------------------------------------------------------
Initial Harness Run      5c873fc  typesafe_jev  0.70           0.50 (hard)  150 ms           250 ms        None           No (delay < RTT)
Live Pilot (5 tasks)     5c873fc  typesafe_jev  0.70           0.50 (hard)  N/A (routing)    1000 ms       Correct (5/5)  No (Noul < 0.50)
Threshold Patch          0ac7d81  typesafe_jev  0.70           0.30 (arg)   N/A              N/A           N/A            N/A
Live Reasoning Test      0ac7d81  typesafe_jev  0.70           0.30         1500 ms          1000 ms       Correct (3/3)  Yes (Hits=3, Canc=1)
Confirmatory Study G0    0ac7d81+ typesafe_jev  N/A (choice)   None         Sweep (150-2500) Sweep         115/120        Yes (Launched=101)
Confirmatory Study G1    0ac7d81+ typesafe_jev  0.70           None         Sweep            Sweep         115/120        Yes (Launched=44)
Confirmatory Study G2    0ac7d81+ typesafe_jev  None           0.50         Sweep            Sweep         115/120        No (Blocked: 1/120)
Confirmatory Study G3    0ac7d81+ typesafe_jev  0.70           0.50         Sweep            Sweep         115/120        No (Blocked: 1/120)
```
