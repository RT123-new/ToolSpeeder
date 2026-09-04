# Independent Adversarial Audit Report: Dimension 9 — Empirical Evidence Evaluation & Retraction Completeness
**Target:** ToolSpeeder PR #1 (`repair/benchmark-integrity-runtime-safety`)  
**Auditor:** Independent Gemini Flash Subagent  
**Date:** 2026-09-04  
**Verdict:** ❌ **REJECTED**

---

## 1. Audit Scope & Files Examined

- Retraction ledgers: `README.md`, `CHANGELOG.md`, `docs/PR1_REPAIR_STATUS.md`.
- Evidence reports: `reports/confirmatory_sweep.md`, `reports/local_sweep.md`.
- Bundles: `artifacts/confirmatory/`, `artifacts/local/`.
- Harness & Test suites: `harness.py`, `tests/test_scientific_integrity.py`, `tests/test_review_findings.py`.

---

## 2. Evaluation Findings

1. **Rigor of Retraction Notice:**
   - The retraction ledger across `README.md` and `PR1_REPAIR_STATUS.md` is comprehensive, transparent, and documents root causes for retracting v1.0 and v1.1 claims.
   - However, a fundamental contradiction exists: `README.md` and `PR1_REPAIR_STATUS.md` proclaim `INTEGRITY REPAIR INCOMPLETE` and state that no canonical confirmatory evidence exists, whereas reports committed at HEAD claim definitive confirmatory pass.
2. **Fabrication in Confirmatory Replay Report:**
   - `reports/confirmatory_sweep.md` claims a 3-seed sweep across `[42, 137, 2026]` with 27,000 paired executions.
   - Forensic analysis of `artifacts/confirmatory/cases.jsonl` and `candidate-traces.jsonl` reveals that **only Seed 42 was executed** (9,000 paired trials). Seeds 137 and 2026 are completely absent.
3. **Hardcoded Negative & Positive Controls:**
   - `artifacts/confirmatory/controls-traces.jsonl` has the identical SHA-256 hash (`3b9240af...`) as the quarantined September 2 run.
   - `BenchmarkHarness.run_negative_controls()` in `harness.py` returns static dictionary literals with `"is_hardcoded_literal": False`, rather than running actual measured code paths.
4. **Local Sweep Honesty:**
   - `reports/local_sweep.md` honestly reports `FALSIFIED` under fail-closed science, with verified 3.83x speedup on W1 and 1.22x on W5. However, it repeats the false claim that 3 seeds were executed when only seed 42 was run.

---

## 3. Final Recommendation & Verdict

### **VERDICT: REJECTED**

**Remediation Requirements:**
1. Execute a genuine 3-seed confirmatory replay sweep (seeds `7001`, `7013`, `7019` or `42`, `137`, `2026`) producing true multi-seed trace files in `artifacts/confirmatory/`.
2. Wire `harness.run_negative_controls()` to execute real measured control arms.
3. Reconcile root documentation (`README.md`, `PR1_REPAIR_STATUS.md`) with committed evidence reports.
