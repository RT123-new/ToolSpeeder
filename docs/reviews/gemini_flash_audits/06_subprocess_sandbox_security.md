# Independent Adversarial Audit Report: Dimension 6 — Subprocess Sandbox & Security Isolation
**Target:** ToolSpeeder PR #1 (`repair/benchmark-integrity-runtime-safety`)  
**Auditor:** Independent Gemini Flash Subagent  
**Date:** 2026-09-04  
**Verdict:** ❌ **REJECTED**

---

## 1. Audit Scope & Files Examined

- `toolspeed/adapters/subprocess_sandbox.py`: **MISSING AS MODULE** (lives in `live_tools.py:153-332`).
- `toolspeed/adapters/live_tools.py`: `SafeSubprocessSandbox`.
- `tests/test_subprocess_security.py`: Process group termination, escalation to SIGKILL.

---

## 2. Sandbox Security Architecture & Deficiencies

1. **Unvalidated Shell Strings:** `SafeSubprocessSandbox` uses `asyncio.create_subprocess_shell` without argument vectorization or executable whitelisting, permitting shell metacharacters and arbitrary binaries.
2. **Missing Module:** `toolspeed/adapters/subprocess_sandbox.py` does not exist as an independent module.
3. **Unbounded Output Buffer DOS:** `proc.communicate()` reads unbounded stdout/stderr into Python heap memory before slicing, permitting child processes to trigger parent OOM.
4. **The "Detached Happy Path" Orphan Leak:** `_terminate_process_group` is only called inside timeout/cancellation exception blocks. When a command succeeds and exits 0, detached background processes (`sleep 300 &`) are never terminated.
5. **Tautological `is_process_tree_terminated`:** The check only inspects `proc.returncode` of the top-level shell leader; it does not inspect the OS process table for surviving child PIDs.

---

## 3. Final Recommendation & Verdict

### **VERDICT: REJECTED**

**Remediation Requirements:**
1. Move `SafeSubprocessSandbox` to `toolspeed/adapters/subprocess_sandbox.py`.
2. Replace `create_subprocess_shell` with `create_subprocess_exec` using argument vectors and an explicit executable allowlist.
3. Add chunked streaming readers with strict max-byte limits.
4. Always terminate the process group upon completion, even on exit code 0, to clean up detached jobs.
