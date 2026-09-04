# Independent Adversarial Audit Report: Dimension 3 — Statistical Inference & Sample Size Power
**Target:** ToolSpeeder PR #1 (`repair/benchmark-integrity-runtime-safety`)  
**Auditor:** Independent Gemini Flash Subagent  
**Date:** 2026-09-04  
**Verdict:** ❌ **REJECTED**

---

## 1. Audit Scope & Files Examined

- `toolspeed/experiments/runner.py`: Percentiles, paired bootstrap CI (2,000 resamples), `compute_summary`.
- `toolspeed/benchmarks/recompute.py`: Metric recomputation from raw JSONL traces.
- `toolspeed/benchmarks/controls.py`: Measured positive and negative control routines.
- `toolspeed/benchmarks/harness.py`: Benchmark harness execution loops, control execution.
- `toolspeed/benchmarks/falsify.py`: Hypothesis evaluation and exit codes.
- `toolspeed/core/statistics.py`: Sample size formulas and Neyman-allocated cluster inference.

---

## 2. Statistical Methodology Verification

1. **Mutual Success Conditioning (Strict CCL Invariant):** Latency percentiles and speedup ratios are calculated strictly over trials where both baseline and candidate succeeded, preventing failure-induced latency artifacts.
2. **Paired Bootstrap Confidence Intervals:** 2,000 paired resamples deterministic under `default_rng(seed=42)`.
3. **Power Analysis:** High sample size regimes ($N \ge 1000$ replay, $N \ge 200$ local) provide $>99.9\%$ statistical power for medium and small effects.

---

## 3. Potential Biases & Critical Defects

1. **Hardcoded Negative & Positive Controls in `harness.py`:**
   - In `toolspeed/benchmarks/harness.py:622-698`, `run_negative_controls()` returns hardcoded static dictionaries (`1.0x` and `2.0x`) with `"is_hardcoded_literal": False`. It never calls `toolspeed/benchmarks/controls.py`!
2. **Point Estimate vs Lower Bound Breach:**
   - Protocol v1.3 specifies that the one-sided 95% bootstrap lower bound must exceed the efficacy threshold. In `harness.py:527`, the harness evaluates `p95_speedup >= threshold` (the point estimate), allowing noisy candidates with overlapping confidence intervals to pass.
3. **Incomplete Recomputation in `falsify.py`:**
   - `falsify.py` and `recompute.py` fail to check P99 tail latency stability, cost multiplier limits, or success rate non-inferiority margins.
4. **Orphaned Statistics Module:**
   - `toolspeed/core/statistics.py` is not used anywhere in the active execution pipeline.

---

## 4. Final Recommendation & Verdict

### **VERDICT: REJECTED**

**Remediation Requirements:**
1. Wire `BenchmarkHarness.run_negative_controls` to execute `run_measured_negative_control` and `run_measured_positive_control` from `controls.py`.
2. Evaluate the 95% bootstrap confidence lower bound in `harness.py:527` rather than the point estimate.
3. Update `falsify.py` to check P99 non-regression and success delta margins from raw traces.
