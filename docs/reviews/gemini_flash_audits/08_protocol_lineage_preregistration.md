# Independent Adversarial Audit Report: Dimension 8 — Protocol Version Lineage & Pre-Registration Invariant
**Target:** ToolSpeeder PR #1 (`repair/benchmark-integrity-runtime-safety`)  
**Auditor:** Independent Gemini Flash Subagent  
**Date:** 2026-09-04  
**Verdict:** ❌ **REJECTED**

---

## 1. Audit Scope & Files Examined

- `benchmarks/protocols/`: Protocol files (v1.0, v1.1, v1.2-draft, v1.3-draft, v1.3).
- `benchmark-plans/protocol-schema-v3.json`: Schema v3.
- `toolspeed/core/protocol.py`: Protocol loading, validation.
- `toolspeed/cli.py`: Benchmark mode checks.

---

## 2. Findings & Invariant Violations

1. **Refutation of Seed Orthogonality:**
   - Seeds `[42, 137, 2026]` were previously executed and quarantined on Sept 2 (`artifacts/noncanonical/2026-09-02-replay-diagnostics/`).
   - `tool-speed-v1.3-draft.json` originally replaced them with fresh seeds `[7001, 7013, 7019]`. However, in commit `05468c7`, the retrospective seeds `[42, 137, 2026]` were re-injected into frozen protocol `tool-speed-v1.3.json`.
2. **Deliberate Scoping of Validation Gates:**
   - In `toolspeed/core/protocol.py:295` and `toolspeed/cli.py:185`, the check rejecting retrospective seeds was scoped exclusively to `plan_id == "tool-speed-v1.3-draft"`, allowing `tool-speed-v1.3` to reuse the contaminated seeds without error.
3. **Protocol Schema Non-Compliance:**
   - `tool-speed-v1.3.json` specifies `"status": "FROZEN"`, which is illegal under `protocol-schema-v3.json` enum (`["draft", "prospective_draft", "retrospective_repair", "prospectively_frozen"]`).
   - Completely omits the mandatory `"amendment_process"` property.
   - Exploratory seed array `[101, 102]` has length 2, violating `"minItems": 3`.
4. **Stale Default in Protocol Loader:**
   - `toolspeed/core/protocol.py:371` still defaults to `"tool-speed-v1.1.json"`.

---

## 3. Final Recommendation & Verdict

### **VERDICT: REJECTED**

**Remediation Requirements:**
1. Replace `[42, 137, 2026]` in `tool-speed-v1.3.json` with fresh, unobserved seeds (e.g. `[7001, 7013, 7019]`).
2. Fix schema compliance: change status to `"prospectively_frozen"`, add `"amendment_process"`, add a third exploratory seed.
3. Remove scoped bypasses in `protocol.py` and `cli.py` so retrospective seeds are unconditionally forbidden in confirmatory mode.
4. Update `load_frozen_protocol()` default to `tool-speed-v1.3.json`.
