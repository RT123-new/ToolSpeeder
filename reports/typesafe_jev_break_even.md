# TypeSafe/Jev End-to-End Break-Even Surface Report

**Protocol:** `benchmarks/protocols/typesafe-speculation-v1.0.json` (`CONFIRMATORY_FROZEN`)  
**Evaluation Corpus:** `benchmarks/data/held_out_speculation_tasks_v1.0.json`  
**Provider Latency Accounted:** Full WAN RTT + Inference + Scheduling + Cancellation ($P_{50} = 699.9\text{ ms}$)  
**Date:** September 17, 2026  

---

## 1. The 2D Break-Even Surface

A speculative route only benefits ToolSpeeder if it reduces end-to-end Correct Completion Latency (CCL) without reducing task correctness. Because live TypeSafe/Jev incurs ~700ms WAN latency, the speculative opportunity window depends strictly on two dimensions:
1. **Primary Model Reasoning Delay:** Time before the authoritative model produces its decision.
2. **Downstream Tool Read Latency:** Duration of the requested read-only tool execution.

### Net CCL Savings Surface (ms):
$$\Delta \text{CCL} = \text{CCL}_{\text{sequential}} - \text{CCL}_{\text{speculative}}$$
*(Positive numbers indicate latency saved; negative numbers indicate overhead penalty)*

```
Reasoning Delay    Tool: 100ms   250ms   500ms   1000ms   2000ms   3000ms
-------------------------------------------------------------------------
150 ms (fast)      -6 ms (?)     -6 ms   -6 ms   -6 ms    -6 ms    -6 ms
300 ms (fast)      -6 ms (?)     -6 ms   -6 ms   -6 ms    -6 ms    -6 ms
500 ms (medium)    -6 ms (?)     -6 ms   -6 ms   -6 ms    -6 ms    -6 ms
750 ms (boundary)  +12 ms (?)    +24 ms  +24 ms  +24 ms   +24 ms   +24 ms
1000 ms (slow)     +70 ms (+)    +185 ms +224 ms +224 ms  +224 ms  +224 ms
1500 ms (thinking) +70 ms (+)    +185 ms +377 ms +607 ms  +607 ms  +607 ms
2500 ms (deep)     +70 ms (+)    +185 ms +377 ms +760 ms  +1374 ms +1374 ms
```

---

## 2. Regime Characterization

### Regime 1: Fast Primary Models ($\le 500\text{ ms}$ reasoning) — NEGATIVE / NEUTRAL
- The primary model completes its tool decision before Jev's WAN response returns (~700ms).
- ToolSpeeder's race logic aborts the draft request cleanly.
- Net wall-clock savings: **0 ms**.
- Contention/scheduling overhead: **$-6\text{ ms}$**.

### Regime 2: Boundary Region ($500\text{--}800\text{ ms}$ reasoning) — NEUTRAL
- Model reasoning and Jev response finish nearly simultaneously.
- Tool overlap window is narrow ($< 100\text{ ms}$).
- Net wall-clock savings are negligible ($+10\text{ to }+25\text{ ms}$).

### Regime 3: Thinking/Reasoning Models ($\ge 1000\text{ ms}$ reasoning) & Slow Tools ($\ge 500\text{ ms}$) — STRONGLY POSITIVE
- Jev returns early in the reasoning window (at ~700ms).
- Long-running downstream tools start concurrently and execute while the main model continues thinking.
- Significant net wall-clock savings: **$+185\text{ ms}$ to $+1374\text{ ms}$** per task.

---

## 3. Comparison with Deterministic Baseline (B2)

While Jev achieves positive CCL savings on reasoning models, **Deterministic Baseline (B2)** achieves **95.0% routing accuracy locally in 0.5 ms**.

```
┌───────────────────────────┬──────────────┬──────────────┬──────────────────────┐
│ Dimension                 │ B2 (Local)   │ B3 (Jev WAN) │ Winner               │
├───────────────────────────┼──────────────┼──────────────┼──────────────────────┤
│ Accuracy on Held-Out      │ 95.0%        │ 95.8%        │ B3 (+0.8% marginal)  │
│ Decision Latency          │ 0.5 ms       │ 699.9 ms     │ B2 (1400x faster)    │
│ Viable on Fast Models     │ Yes (0.5ms)  │ No (>700ms)  │ B2                   │
│ API Dependency & Cost     │ None         │ Remote API   │ B2                   │
│ Network Failure Risk      │ Zero         │ Possible     │ B2                   │
└───────────────────────────┴──────────────┴──────────────┴──────────────────────┘
```

**Conclusion:**
Because simple deterministic routing achieves comparable accuracy on this workload, Jev's 700ms latency makes it uncompetitive as a standalone primary router. It is only valuable when cascaded or reserved for ambiguous tasks where B2 exhibits low confidence.
