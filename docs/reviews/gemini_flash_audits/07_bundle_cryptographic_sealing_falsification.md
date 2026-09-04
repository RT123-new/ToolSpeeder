# Independent Adversarial Audit Report: Dimension 7 — Bundle Cryptographic Sealing & Falsification
**Target:** ToolSpeeder PR #1 (`repair/benchmark-integrity-runtime-safety`)  
**Auditor:** Independent Gemini Flash Subagent  
**Date:** 2026-09-04  
**Verdict:** ❌ **REJECTED**

---

## 1. Audit Scope & Files Examined

- `toolspeed/benchmarks/bundle.py`: Atomic bundle creation, `validate_bundle_hashes_first`.
- `toolspeed/benchmarks/recompute.py`: Zero-trust trace recomputation, discrepancy checking.
- `toolspeed/benchmarks/falsify.py`: Hypothesis falsification, exit codes 0/1/2/3.
- `toolspeed/cli.py`: `cmd_falsify`, `cmd_report`, `cmd_validate_bundle`.

---

## 2. Cryptographic Sealing Architecture & Anti-Tampering

1. **Path Traversal / Arbitrary Hash Oracle in `validate_bundle_hashes_first`:**
   - In `bundle.py:204`, `b_dir / fname` does not sanitize `fname`. Supplying `/etc/passwd` or `../../file` causes the validator to hash arbitrary system files outside the bundle directory.
2. **Trace Shadowing via Unhashed File Injection:**
   - `validate_bundle_hashes_first` checks files listed in the manifest, but does not enforce that all files present in the bundle directory are tracked. Dropping an unhashed `candidate-traces.jsonl` into the folder causes `recompute.py` to prioritize it over `raw-traces.jsonl`.
3. **Silent Baseline Imputation:**
   - In `recompute.py:77-85`, if baseline traces are missing, `recompute.py` silently copies candidate latencies to baseline (`b_lats = list(c_lats)`), fabricating a 1.00x speedup.
4. **Non-Fail-Closed Null Metric Pass:**
   - In `falsify.py:113-137`, if `p95_speedup` or `candidate_success_rate` is `None`, the checks are skipped and the evaluator returns Exit Code 0 (PASSED).
5. **CLI Disconnect:**
   - `toolspeed falsify` executes `cmd_falsify` in `cli.py`, which reads stored metrics from `result.json` instead of delegating to `evaluate_falsification_independent`.

---

## 3. Final Recommendation & Verdict

### **VERDICT: REJECTED**

**Remediation Requirements:**
1. Enforce strict path safety in `validate_bundle_hashes_first` (`is_relative_to(b_dir)`).
2. Enforce closed directory verification: any untracked file in the bundle directory invalidates the seal.
3. Fail closed if baseline traces are missing rather than copying candidate latencies.
4. Treat `None` metrics as immediate falsifications in `falsify.py`.
5. Wire `cmd_falsify` to call `evaluate_falsification_independent`.
