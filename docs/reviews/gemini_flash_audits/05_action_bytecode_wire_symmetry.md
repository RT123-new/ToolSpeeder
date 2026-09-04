# Independent Adversarial Audit Report: Dimension 5 — Action Bytecode & Wire Serialization Symmetry
**Target:** ToolSpeeder PR #1 (`repair/benchmark-integrity-runtime-safety`)  
**Auditor:** Independent Gemini Flash Subagent  
**Date:** 2026-09-04  
**Verdict:** ❌ **REJECTED**

---

## 1. Audit Scope & Files Examined

- `toolspeed/core/bytecode.py`: **MISSING AS MODULE**.
- `toolspeed/schedulers/e5_action_bytecode.py`: Examined `ActionBytecodeCodec`.
- `toolspeed/adapters/mock_models.py`: Identified divergent duplicate codec.
- `toolspeed/core/types.py`: `ToolCall` serialization and deserialization.

---

## 2. Codec Specification & Encoding Symmetry

1. **"Pseudobytecode" Reality:** `ActionBytecodeCodec` wraps JSON text inside a binary length-prefixed packet. It does not implement typed binary serialization; every argument value is serialized via `json.dumps().encode("utf-8")`. This produces wire inflation over plain JSON.
2. **Object Roundtrip Failure:** `ActionBytecodeCodec.encode` drops `call_id`, speculative flags, approval state, and metadata. Upon `decode()`, a brand-new random UUID is generated for `call_id`.
3. **Key Ordering Asymmetry:** Keys are iterated in arbitrary Python dict insertion order without canonical sorting, violating symmetry with `CanonicalJSONCodec`.
4. **Unhandled Unicode Crashes:** In `e5_action_bytecode.py`, lines 152 and 190 call `.decode("utf-8")` without `try...except UnicodeDecodeError`. A corrupted byte immediately crashes the process with an unhandled exception rather than raising a clean `ValueError`.
5. **Type Asymmetry in `ToolCall.from_dict()`:** `to_dict()` converts bytecode bytes to hex string, but `from_dict()` leaves it as `str` instead of calling `bytes.fromhex()`.

---

## 3. Final Recommendation & Verdict

### **VERDICT: REJECTED**

**Remediation Requirements:**
1. Move codec logic to `toolspeed/core/bytecode.py` and eliminate the duplicate in `mock_models.py`.
2. Fix `ToolCall.from_dict()` to restore bytes from hex string.
3. Add canonical key sorting and enforce `allow_nan=False`.
4. Wrap all `.decode("utf-8")` calls in `try...except UnicodeDecodeError` to raise clean `ValueError`s.
