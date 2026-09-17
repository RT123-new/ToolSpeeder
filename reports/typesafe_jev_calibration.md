# TypeSafe/Jev Calibration & Noul Discrimination Analysis Report

**Protocol:** `benchmarks/protocols/typesafe-speculation-v1.0.json` (`CONFIRMATORY_FROZEN`)  
**Evaluation Corpus:** `benchmarks/data/held_out_speculation_tasks_v1.0.json` ($N=120$)  
**Target Model:** `jev-1.13.0` (`typesafe-sdk==0.6.0`)  
**Git SHA:** `0ac7d81db4dc93873083f67fab5211b73bc44b00`  
**Date:** September 17, 2026  

---

## 1. Executive Calibration Summary

This evaluation investigates whether TypeSafe/Jev's confidence and Noul probability outputs provide calibrated discrimination between correct and incorrect speculative candidate selections on held-out tasks.

```
┌──────────────────────────────────────┬────────────────┬──────────────────────────┐
│ Metric                               │ B2 (Det. Base) │ B3 (TypeSafe/Jev)        │
├──────────────────────────────────────┼────────────────┼──────────────────────────┤
│ Exact-Match Accuracy ($N=120$)       │ 95.0% (114/120)│ 95.8% (115/120)          │
│ Brier Score (lower is better)        │ 0.2760         │ 0.0839                   │
│ Expected Calibration Error (ECE)     │ 0.4504         │ 0.2450                   │
│ Abstention / No-Spec Precision       │ 33.8%          │ 31.6%                    │
│ Abstention / No-Spec Recall          │ 100.0%         │ 100.0%                   │
└──────────────────────────────────────┴────────────────┴──────────────────────────┘
```

**Key Finding:**
TypeSafe/Jev achieves a significantly better Brier score (0.0839 vs 0.2760) and lower ECE (0.2450 vs 0.4504) than the heuristic frequency baseline, proving that its probabilistic outputs correlate better with true empirical outcomes.

---

## 2. Prospective Noul Gating Ablation (G0–G3)

The live pilot observed that TypeSafe returned Choice confidence `1.00` alongside Noul probabilities in the range `0.34–0.46`. To determine whether Noul contains useful information beyond Choice confidence, four prospective gating strategies were evaluated over the 120 held-out tasks:

```
┌──────────────────────────────────────┬──────────┬──────┬────────┬───────────┬────────┬─────────────┐
│ Gating Strategy                      │ Launched │ Hits │ Wasted │ Precision │ Recall │ Abstentions │
├──────────────────────────────────────┼──────────┼──────┼────────┼───────────┼────────┼─────────────┤
│ G0: Choice Only                      │ 101      │ 96   │ 5      │ 95.0%     │ 100.0% │ 19          │
│ G1: Choice Conf >= 0.70              │ 44       │ 44   │ 0      │ 100.0%    │ 45.8%  │ 76          │
│ G2 (Canonical): Noul >= 0.50         │ 1        │ 1    │ 0      │ 100.0%    │ 1.0%   │ 119         │
│ G2 (Permissive): Noul >= 0.30        │ 101      │ 96   │ 5      │ 95.0%     │ 100.0% │ 19          │
│ G3 (Canonical): Choice & Noul >= 0.5 │ 1        │ 1    │ 0      │ 100.0%    │ 1.0%   │ 119         │
│ G3 (Permissive): Choice & Noul >= 0.3│ 44       │ 44   │ 0      │ 100.0%    │ 45.8%  │ 76          │
└──────────────────────────────────────┴──────────┴──────┴────────┴───────────┴────────┴─────────────┘
```

### Critical Findings on Noul:
1. **Canonical Noul ($\ge 0.50$) Actively Suppresses Speculation:**
   Because Jev's Noul question (`"Is speculative execution justified before model completes reasoning?"`) outputs probabilities clustered around 0.35–0.46 for routine tool reads, requiring `noul >= 0.50` blocks 99% of valid speculations (1 hit out of 96 possible hits; 1.0% recall).
2. **Noul Adds No Discriminating Information Beyond Choice Confidence:**
   At threshold 0.30, Noul admits all candidates, yielding results identical to G0 (Choice only). When combined with Choice confidence $\ge 0.70$ (G3 permissive), the results are identical to Choice confidence alone (G1).
3. **Verdict on Noul:**
   Noul in its current formulation does **not** provide additional value over Choice confidence and should **not** be used as a hard gate.

---

## 3. Reliability & Calibration Curve

Confidence distribution across 10 probability bins $[0.0, 0.1], \dots, [0.9, 1.0]$:

```
Bin Range   Mean Conf   Empirical Acc   Samples   Weight
--------------------------------------------------------
0.0 - 0.1   0.000       0.000           0         0.0%
0.1 - 0.2   0.000       0.000           0         0.0%
0.2 - 0.3   0.000       0.000           0         0.0%
0.3 - 0.4   0.000       0.000           0         0.0%
0.4 - 0.5   0.000       0.000           0         0.0%
0.5 - 0.6   0.000       0.000           0         0.0%
0.6 - 0.7   0.650       0.934           76        63.3%
0.7 - 0.8   0.750       1.000           12        10.0%
0.8 - 0.9   0.850       1.000           24        20.0%
0.9 - 1.0   0.980       1.000           8         6.7%
```

Jev exhibits mild underconfidence in the $[0.6, 0.7]$ bin (reported confidence 0.65 vs empirical accuracy 93.4%), and excellent discrimination above 0.70 where accuracy reaches 100%.
