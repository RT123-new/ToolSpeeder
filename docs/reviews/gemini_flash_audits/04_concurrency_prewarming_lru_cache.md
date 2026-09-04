# Independent Adversarial Audit Report: Dimension 4 — Concurrency, Prewarming & LRU Cache Correctness
**Target:** ToolSpeeder PR #1 (`repair/benchmark-integrity-runtime-safety`)  
**Auditor:** Independent Gemini Flash Subagent  
**Date:** 2026-09-04  
**Verdict:** ❌ **REJECTED**

---

## 1. Audit Scope & Files Examined

- `toolspeed/core/rate_limiter.py`: `AsyncTokenBucket`, `AsyncConcurrencyLimiter`, `RateLimitLease`, `RateLimiter`.
- `toolspeed/core/prewarming.py`: **MISSING AS MODULE**. Examined `w6_cold_start.py` and `composite.py`.
- `toolspeed/core/cache.py`: **MISSING AS MODULE**. Examined `phase2_cache.py`.

---

## 2. Concurrency & Synchronization Correctness Analysis

1. **Deadlock on Task Cancellation in `AsyncConcurrencyLimiter`:**
   - If a task is cancelled after `semaphore.acquire()` completes but before `_active_count` increments, `release()` raises `RuntimeError` on over-release and never returns the permit to the semaphore, causing permanent permit leakage and deadlock under cancellation pressure.
2. **Cross-Task Data Corruption on Worker Cancellation:**
   - In `WarmSubprocessPool` (`w6_cold_start.py`), if an execution is cancelled while waiting on stdout, the worker process is returned to the pool queue while still executing. The next task checking out the worker receives the stale output of the cancelled task.
3. **Missing Modules:**
   - Neither `toolspeed/core/prewarming.py` nor `toolspeed/core/cache.py` exists in the core package.
4. **Thread-Safety Deficits in `ToolResultCache`:**
   - `ToolResultCache` has zero lock synchronization. Iterating over keys during concurrent `put()` or `invalidate()` causes `RuntimeError: dictionary changed size during iteration`.
5. **Cross-Tenant Invalidation Bleed:**
   - `invalidate_on_mutation()` fails to filter by tenant, allowing mutations by Tenant A to purge cached entries of Tenant B.

---

## 3. Final Recommendation & Verdict

### **VERDICT: REJECTED**

**Remediation Requirements:**
1. Create `toolspeed/core/prewarming.py` and `toolspeed/core/cache.py`.
2. Fix `AsyncConcurrencyLimiter` so that semaphore acquisition and active count updates are strictly cancellation-atomic.
3. Protect `ToolResultCache` with thread-safe locks and enforce tenant scoping during invalidation.
4. Prevent cancelled subprocess pool workers from returning contaminated stdout to the queue.
