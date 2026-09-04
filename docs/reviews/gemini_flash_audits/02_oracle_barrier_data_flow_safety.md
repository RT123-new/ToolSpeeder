# Independent Adversarial Audit Report: Dimension 2 — Oracle Barrier & Data Flow Safety
**Target:** ToolSpeeder PR #1 (`repair/benchmark-integrity-runtime-safety`)  
**Auditor:** Independent Gemini Flash Subagent  
**Date:** 2026-09-04  
**Verdict:** ❌ **REJECTED**

---

## 1. Audit Scope & Files Examined

- `toolspeed/benchmarks/oracle_static_barrier.py`: **MISSING AS PACKAGE MODULE**. (Only exists in `tests/test_oracle_static_barrier.py`).
- `toolspeed/schedulers/executor.py`: Approval gate enforcement, untrusted model flags.
- `toolspeed/core/types.py`: `ApprovalGrant`, `RuntimeAuthorityProvider`, `ExecutionAuthorityContext`, `BenchmarkCase`, `AgentTask`.
- `tests/test_oracle_static_barrier.py`: AST inspection logic, forbidden identifiers.
- `tests/test_authority_provider.py`: Capability grants, HMAC-SHA256 signature verification.

---

## 2. Strengths & Information-Flow Barriers

1. **Strict Rejection of Model-Forged Approvals:** When a tool requires approval, `ToolExecutor` completely ignores `call.is_approved` and `call.approval_grant`. Approval is validated exclusively through out-of-band authority contexts.
2. **Decoupled Benchmark Architecture:** `BenchmarkCase` cleanly isolates model-visible data (`AgentTask`) from validation oracles (`expected_outcome`).
3. **Cryptographic Integrity & Replay Prevention:** `ApprovalGrant` uses canonical 64-char SHA-256 fingerprints, HMAC-SHA256 signatures, and atomic single-use consumption.
4. **Speculative Execution Guardrail:** Any speculative call to mutative tools is immediately blocked before dispatch.

---

## 3. Vulnerabilities & Threat Models Tested

1. **Missing Barrier Module:** `toolspeed/benchmarks/oracle_static_barrier.py` does not exist as an importable module; it lives solely in `tests/test_oracle_static_barrier.py`.
2. **AST Blind Spot on LLM Adapters:** The AST scanner only checks `toolspeed/schedulers/*.py` and omits `toolspeed/adapters/` completely.
3. **Absence of Import Boundary Validation:** The AST visitor checks attribute names and string constants, but does not inspect `ast.Import` or `ast.ImportFrom`. Schedulers and adapters can freely import test fixtures, oracle data, or secret keys.
4. **Secret Key Exposure & Fallback:** `DEFAULT_ISSUER_SECRET` is publicly exported at module level in `types.py`. In `executor.py:512`, validation of `trusted_grant` falls back to `_RUNTIME_ISSUER_SECRET` rather than `auth_ctx.issuer_secret`.

---

## 4. Final Recommendation & Verdict

### **VERDICT: REJECTED**

**Remediation Requirements:**
1. Export `toolspeed/benchmarks/oracle_static_barrier.py` as an importable production module.
2. Expand AST scan to cover `toolspeed/adapters/`.
3. Add AST checks for `ast.Import` and `ast.ImportFrom` to prevent imports of test fixtures and secrets.
4. Protect module-level secrets and enforce `auth_ctx.issuer_secret` matching in `executor.py`.
