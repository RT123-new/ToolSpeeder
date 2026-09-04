# Independent Adversarial Audit Report: Dimension 1 — Architecture & Design Integrity
**Target:** ToolSpeeder PR #1 (`repair/benchmark-integrity-runtime-safety`)  
**Auditor:** Independent Gemini Flash Subagent  
**Date:** 2026-09-04  
**Verdict:** ❌ **REJECTED**

---

## 1. Audit Scope & Files Examined

This audit conducted an independent, adversarial code-level inspection of **Dimension 1: Architecture & Design Integrity** for ToolSpeeder PR #1. The scope encompassed scheduler composition, execution lifecycle enforcement, concurrency and lease management, authority and idempotency guarantees, and test verification integrity.

### Files Examined
- **Schedulers & Execution Engine:**
  - `toolspeed/schedulers/composite.py`: Composite Latency Optimizer implementation and delegation methods.
  - `toolspeed/schedulers/executor.py`: `ToolExecutor`, `SharedIdempotencyStore`, `IdempotencyEntry`, error hierarchy.
  - `toolspeed/schedulers/base.py`: `BaseScheduler`, `ExecutionContext`, `SchedulerConfig`, `TaskTracker`.
  - `toolspeed/schedulers/e1_dag_scheduler.py`: `DAGScheduler`, `ToolDAG`, `DAGNode`, cycle detection, dependency resolution.
  - `toolspeed/schedulers/e2_jit_fusion.py`: `JITFusionScheduler`, `DeclarativeWorkflow`, `WorkflowRegistry`, deoptimization fallback.
  - `toolspeed/schedulers/e3_speculation.py`: `SpeculativeReadScheduler`, concurrency-safety checks, contention modes.
  - `toolspeed/schedulers/e4_commit_horizon.py`: `CommitHorizonScheduler`, `IncrementalCommitParser`, `CommittedCall`.
  - `toolspeed/schedulers/phase2_cache.py`: `ToolResultCache`, `CacheScheduler`, LRU eviction, tenant scoping.
- **Core Infrastructure & Types:**
  - `toolspeed/core/types.py`: `ApprovalGrant`, `ExecutionAuthorityContext`, `AuthorityProvider`, `RuntimeAuthorityProvider`, `ToolSpec`, `ToolCall`, `ToolResult`.
  - `toolspeed/core/rate_limiter.py`: `RateLimiter`, `RateLimitLease`, `AsyncTokenBucket`, `AsyncConcurrencyLimiter`.
  - `toolspeed/core/guardrails.py`: `GuardrailMonitor`, violation logging.
- **Test Suites:**
  - `tests/test_composite_refactor.py`: Phase 31 refactor and delegation assertions.
  - `tests/test_schedulers.py`: Scheduler unit and integration tests.
  - `tests/test_runtime_hardening.py`: Runtime leases, scoped idempotency, and executor safety.
  - `tests/test_adversarial_integrity.py`: Adversarial runtime edge cases (deduplication, approval forgery).
  - `tests/test_review_findings.py`: Regression ledger tests for findings A through O.

---

## 2. Strengths & Architectural Guarantees

1. **Strict Multi-Phase Schema Validation (`ToolExecutor._validate_schema`):**
   - Rigorously validates incoming tool arguments against JSON Schema definitions.
   - Enforces types (`string`, `integer`, `number`, `boolean`, `array`, `object`), boundary constraints (`minimum`, `maximum`, `minLength`, `maxLength`), floating-point checks (`math.isnan`, `math.isinf` rejection), enum sets, and recursively validates nested arrays and objects.
   - Fails closed with `SchemaValidationError` upon detecting unresolved template references (e.g. `$c1.output`).

2. **Cryptographic Approval Grant Architecture (`ApprovalGrant` & `ExecutionAuthorityContext`):**
   - Out-of-band HMAC-SHA256 signing using process-isolated `_RUNTIME_ISSUER_SECRET`.
   - Binds tool name, full canonical SHA-256 argument fingerprint, expiration timestamp, authority, nonce, tenant ID, and run ID.
   - Explicitly rejects model-forged authority flags (`call.is_approved = True` or model-supplied `call.approval_grant`).
   - Supports atomic consumption of `single_use` grants via `verify_and_consume_grant()` under thread-safe synchronization (`_lock`).

3. **Two-Phase Deadlock-Free Rate Limiting (`RateLimiter.lease`):**
   - Implements a cancellation-safe two-phase acquisition protocol: acquires rate-limiting tokens *first* before attempting to acquire the concurrency slot.
   - Prevents the classic concurrency deadlock where all concurrency slots are held by tasks blocked waiting on rate-limiting token replenishment.
   - If concurrency slot acquisition is cancelled or raises an error, acquired tokens are atomically refunded (`token_bucket.refund`).

4. **Thread-Safe Scoped Idempotency Ledger (`SharedIdempotencyStore`):**
   - Synchronized via internal `threading.Lock` across concurrent asynchronous tasks.
   - Scopes keys across 5 dimensions: `tenant:run:provider:op:tool:idempotency_key`.
   - Detects canonical argument collisions (`ARG_MISMATCH`) and fails closed when the same key is reused with conflicting arguments.
   - Supports the `RESERVED_PRIMARY` / `JOIN_IN_FLIGHT` pattern, where concurrent duplicate dispatches attach to the primary execution's `asyncio.Future` and receive deep-copied results without duplicating the underlying operation.

5. **Strict Read-Only Gating on Predictive Dispatch:**
   - Both speculative execution (`e3_speculation.py`) and commit-horizon early streaming (`e4_commit_horizon.py`) strictly require `is_read_only=True`, `side_effects=False`, `requires_approval=False`, and `is_idempotent=True`.
   - `ToolExecutor` provides a secondary defence-in-depth gate at runtime: if `is_speculative=True` is passed for a tool violating read-only constraints, it immediately rejects the call with `SPECULATION_SIDE_EFFECT_ATTEMPT`.

---

## 3. Vulnerabilities & Architectural Edge Cases Identified

### A. Architectural Dishonesty: Facade Delegation in `CompositeScheduler`
- PR #1 claims `CompositeScheduler` unifies and delegates to the specialized schedulers (`DAGScheduler`, `JITFusionScheduler`, `SpeculativeScheduler`, `CommitHorizonScheduler`, `CacheScheduler`). Methods named `delegate_fanout()`, `delegate_pipeline_sequence()`, `delegate_speculative()`, `delegate_streaming()`, and `delegate_cache()` were introduced.
- **Reality:** Zero delegation occurs during execution. Inlined loops run instead, bypassing delegate schedulers.
- In `tests/test_composite_refactor.py`, `test_01_real_delegation_to_appropriate_schedulers` merely checks `self.assertIsInstance(scheduler.delegate_fanout(), DAGScheduler)`.

### B. Security & Isolation Bypasses in `CompositeScheduler` Inlined Logic
1. **Concurrency Safety Ignored:** `SpeculativeReadScheduler` explicitly checks `supports_concurrent_adapter(model)`. `CompositeScheduler` never checks this.
2. **Resource Starvation via Lack of Isolated Capacity:** In `SpeculativeReadScheduler`, contention mode `"isolated"` spins up an independent capacity limiter and executor. `CompositeScheduler` ignores this and runs speculative calls directly against the production `ctx.executor`.
3. **Cross-Tenant / Cross-Authority Cache Leak:** In `CacheScheduler`, cache keys are scoped by `tenant` and `authority`. In `CompositeScheduler._exec_node`, calls to `self.cache.get()` and `self.cache.put()` omit `tenant` and `authority`, defaulting to `"default_tenant"` and `"default_authority"`.
4. **Negative Control Invalidation:** `DAGScheduler` respects `config.parallelism_enabled = False` for scientific ablation controls. `CompositeScheduler` ignores this flag entirely.

### C. Invariant Violation: Duplicate Side-Effects on JIT Fusion Deoptimization Fallback
- In `JITFusionScheduler` (`toolspeed/schedulers/e2_jit_fusion.py`), when a declarative pipeline executes a mutative node and subsequently encounters an invariant failure or step error, it deoptimizes and falls back to model-driven turn execution.
- Method `can_execute_in_fallback(tool_name, execution_ledger)` was added at line 251 to satisfy Finding 28 in `tests/test_review_findings.py`.
- **However, `can_execute_in_fallback` is NEVER CALLED anywhere in `_execute_internal()`!**
- This allows duplicate execution of mutative side-effects during fallback.

### D. Trivial Security Bypass in JIT Workflow Ingestion
- In `JITFusionScheduler._match_workflow`, a naive blacklist checks whether the workflow ID contains `"injected"`, `"malicious"`, or `"unreviewed"`. Any external workflow named `production_sync` bypasses registry validation completely.

---

## 4. Final Recommendation & Verdict

### **VERDICT: REJECTED**

**Remediation Requirements:**
1. Either refactor `CompositeScheduler` to legitimately delegate dispatch sub-phases to specialized scheduler instances or remove misleading facade methods and document unified behavior.
2. Enforce fallback side-effect ledger (`can_execute_in_fallback`) during JIT deoptimization.
3. Remove naive string blacklists and restrict workflow ingestion to `WorkflowRegistry`.
4. Enforce tenant and authority scoping in composite scheduler caching calls.
