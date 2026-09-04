# ToolSpeeder PR #1: Compilation of 9 Independent Gemini Flash Reviews & Audit Findings

**Date:** 2026-09-04  
**Target Pull Request:** ToolSpeeder PR #1 (`repair/benchmark-integrity-runtime-safety`)  
**Audit Methodology:** 9 Concurrent, Independent, Adversarial Gemini Flash Research Subagents  
**Overall PR Verdict:** ❌ **REJECTED (9 of 9 Subagents Unanimously Reject Pending Remediation)**

---

## 1. Executive Summary & Audit Scorecard

In accordance with Phase 38 of the PR #1 mandate, nine independent Gemini Flash review subagents were launched with distinct adversarial mandates covering every subsystem of the ToolSpeeder integrity architecture.

Each subagent operated in complete isolation without visibility into peer evaluations. All 9 auditors reached an uncompromising **REJECTED** verdict, identifying substantive technical, security, and scientific integrity defects that must be remediated prior to marking PR #1 ready for final review or merge.

### Scorecard Matrix

| # | Dimension | Primary Focus | Verdict | Key Blocking Finding |
|---|---|---|:---:|---|
| **1** | **Architecture & Design Integrity** | Scheduler delegation, execution lifecycle, idempotency | ❌ **REJECTED** | Facade delegation in `CompositeScheduler`; uncalled `can_execute_in_fallback` during JIT deoptimization; string blacklist in JIT ingestion. |
| **2** | **Oracle Barrier & Data Flow Safety** | Static AST barriers, model authority isolation, secret keys | ❌ **REJECTED** | Missing `toolspeed/benchmarks/oracle_static_barrier.py` package module; AST blind spot on LLM adapters; missing import validation against test fixtures. |
| **3** | **Statistical Inference & Power** | Bootstrap CIs, non-inferiority margins, control calibration | ❌ **REJECTED** | `harness.run_negative_controls()` returns hardcoded dictionaries with literal 1.0x/2.0x values; efficacy checks point estimate instead of 95% bootstrap lower bound. |
| **4** | **Concurrency & Cache Correctness** | Rate limiting, prewarming, LRU cache thread-safety | ❌ **REJECTED** | Missing `prewarming.py` and `cache.py` core modules; semaphore permit leak and deadlock under cancellation; unsynchronized `ToolResultCache` dictionary mutations. |
| **5** | **Action Bytecode & Wire Symmetry** | Codec specification, roundtrip fidelity, error handling | ❌ **REJECTED** | Missing `toolspeed/core/bytecode.py`; pseudobytecode wrapping JSON text with wire inflation; `ToolCall.call_id` dropped on decode; unhandled Unicode crashes on corrupted frames. |
| **6** | **Subprocess Sandbox & Security** | Process isolation, tree termination, orphan prevention | ❌ **REJECTED** | Missing `subprocess_sandbox.py` module; raw shell execution (`create_subprocess_shell`) without executable allowlist; normal exit 0 leaks detached background processes. |
| **7** | **Bundle Sealing & Falsification** | Manifest verification, zero-trust recomputation, exit codes | ❌ **REJECTED** | Path traversal vulnerability in manifest hash verification; untracked file injection in bundle directories; silent baseline copy when baseline traces are missing. |
| **8** | **Protocol Lineage & Invariants** | Provenance, immutability, confirmatory seed orthogonality | ❌ **REJECTED** | Confirmatory seed withholding breached (quarantined seeds `[42, 137, 2026]` re-injected into frozen v1.3); validator patched to exempt v1.3; schema v3 enum and property violations. |
| **9** | **Empirical Evidence & Retraction** | Retraction completeness, confirmatory & local sweeps | ❌ **REJECTED** | Confirmatory sweep report claims 27,000 paired executions across 3 seeds, but underlying bundle contains only 9,000 trials on seed 42 only; controls hash recycled from quarantined runs. |

---

## 2. Dimension Summaries & Detailed Audit Findings

### Dimension 1: Architecture & Design Integrity
- **Auditor Report:** [01_architecture_design_integrity.md](file:///Users/regtroka/Downloads/ToolSpeed/docs/reviews/gemini_flash_audits/01_architecture_design_integrity.md)
- **Key Findings:**
  1. `CompositeScheduler` introduces `delegate_*()` methods that return specialized scheduler instances, but never calls them during execution. Inlined loops run instead.
  2. In `JITFusionScheduler`, `can_execute_in_fallback()` was added to satisfy review findings, but is never invoked in `_execute_internal()`, permitting duplicate mutative side-effects during model fallback.
  3. JIT workflow ingestion checks a naive substring blacklist (`"injected"`, `"malicious"`) instead of validating against `WorkflowRegistry`.
  4. Composite scheduler caching omits tenant and authority scoping, risking cross-tenant data leaks.

### Dimension 2: Oracle Barrier & Data Flow Safety
- **Auditor Report:** [02_oracle_barrier_data_flow_safety.md](file:///Users/regtroka/Downloads/ToolSpeed/docs/reviews/gemini_flash_audits/02_oracle_barrier_data_flow_safety.md)
- **Key Findings:**
  1. `toolspeed/benchmarks/oracle_static_barrier.py` does not exist as an importable module; it exists only in `tests/`.
  2. AST static analysis checks only `toolspeed/schedulers/` and completely ignores `toolspeed/adapters/`.
  3. The AST scanner does not inspect `ast.Import` or `ast.ImportFrom`, allowing schedulers and adapters to import test fixtures, oracle answers, or secret keys without detection.
  4. Module-level secret `DEFAULT_ISSUER_SECRET` is publicly exported.

### Dimension 3: Statistical Inference & Sample Size Power
- **Auditor Report:** [03_statistical_inference_power.md](file:///Users/regtroka/Downloads/ToolSpeed/docs/reviews/gemini_flash_audits/03_statistical_inference_power.md)
- **Key Findings:**
  1. `BenchmarkHarness.run_negative_controls` returns hardcoded dictionary literals with `"is_hardcoded_literal": False`, completely bypassing `toolspeed/benchmarks/controls.py`.
  2. In `harness.py:527`, latency efficacy is evaluated against the sample point estimate rather than enforcing that the 95% bootstrap confidence lower bound exceeds the threshold.
  3. `recompute.py` and `falsify.py` omit checks for P99 tail latency stability, cost multipliers, and success rate delta non-inferiority.

### Dimension 4: Concurrency, Prewarming & LRU Cache Correctness
- **Auditor Report:** [04_concurrency_prewarming_lru_cache.md](file:///Users/regtroka/Downloads/ToolSpeed/docs/reviews/gemini_flash_audits/04_concurrency_prewarming_lru_cache.md)
- **Key Findings:**
  1. Required modules `toolspeed/core/prewarming.py` and `toolspeed/core/cache.py` do not exist.
  2. In `AsyncConcurrencyLimiter`, task cancellation between `semaphore.acquire()` and `_active_count` increment causes permanent permit leakage and deadlock.
  3. In `WarmSubprocessPool`, cancelling an in-flight execution returns the running worker to the queue, causing subsequent tasks to read stale stdout.
  4. `ToolResultCache` has zero locking mechanisms and invalidates across all tenants on any mutation.

### Dimension 5: Action Bytecode & Wire Serialization Symmetry
- **Auditor Report:** [05_action_bytecode_wire_symmetry.md](file:///Users/regtroka/Downloads/ToolSpeed/docs/reviews/gemini_flash_audits/05_action_bytecode_wire_symmetry.md)
- **Key Findings:**
  1. `toolspeed/core/bytecode.py` does not exist; logic is embedded in `schedulers/e5_action_bytecode.py`, with an incompatible duplicate in `adapters/mock_models.py`.
  2. "Pseudobytecode": argument values are encoded as JSON text inside binary frames, causing wire inflation over plain JSON.
  3. `ActionBytecodeCodec.encode` drops `call_id`, speculative flags, and metadata; `decode()` generates a random UUID.
  4. Malformed UTF-8 in payload frames triggers unhandled `UnicodeDecodeError` crashes.

### Dimension 6: Subprocess Sandbox & Security Isolation
- **Auditor Report:** [06_subprocess_sandbox_security.md](file:///Users/regtroka/Downloads/ToolSpeed/docs/reviews/gemini_flash_audits/06_subprocess_sandbox_security.md)
- **Key Findings:**
  1. `SafeSubprocessSandbox` uses `asyncio.create_subprocess_shell` without argument lists or executable whitelisting.
  2. Unbounded `proc.communicate()` buffers unbounded output into memory before slicing, risking parent process OOM.
  3. Detached background processes (`cmd & exit 0`) leak orphans because cleanup is only called on exception/timeout.
  4. `is_process_tree_terminated` only inspects top-level shell return code, ignoring background child PIDs.

### Dimension 7: Bundle Cryptographic Sealing & Falsification
- **Auditor Report:** [07_bundle_cryptographic_sealing_falsification.md](file:///Users/regtroka/Downloads/ToolSpeed/docs/reviews/gemini_flash_audits/07_bundle_cryptographic_sealing_falsification.md)
- **Key Findings:**
  1. Path traversal in `validate_bundle_hashes_first`: unsanitized filenames allow hashing arbitrary system files.
  2. Trace shadowing: unhashed trace files dropped into bundle directories are executed by `recompute.py`.
  3. Silent baseline copying in `recompute.py` fabricates 1.00x speedup when baseline traces are missing.
  4. In `falsify.py`, missing/null speedup metrics fall through to Exit Code 0 (PASS).
  5. CLI `toolspeed falsify` reads `result.json` rather than invoking `evaluate_falsification_independent`.

### Dimension 8: Protocol Version Lineage & Pre-Registration Invariant
- **Auditor Report:** [08_protocol_lineage_preregistration.md](file:///Users/regtroka/Downloads/ToolSpeed/docs/reviews/gemini_flash_audits/08_protocol_lineage_preregistration.md)
- **Key Findings:**
  1. Confirmatory seed withholding claim is false: seeds `[42, 137, 2026]` were previously executed and quarantined on Sept 2, yet re-injected into frozen protocol v1.3.
  2. Validator in `toolspeed/core/protocol.py:295` and `toolspeed/cli.py:185` was patched to exempt `tool-speed-v1.3` from retrospective seed checks.
  3. Protocol v1.3 violates schema v3 on status enum (`"FROZEN"` vs `"prospectively_frozen"`), missing `"amendment_process"`, and exploratory seed count.
  4. `load_frozen_protocol()` still defaults to `tool-speed-v1.1.json`.

### Dimension 9: Empirical Evidence Evaluation & Retraction Completeness
- **Auditor Report:** [09_empirical_evidence_retraction.md](file:///Users/regtroka/Downloads/ToolSpeed/docs/reviews/gemini_flash_audits/09_empirical_evidence_retraction.md)
- **Key Findings:**
  1. Confirmatory sweep report claims 27,000 paired executions across 3 seeds (`[42, 137, 2026]`). In reality, `artifacts/confirmatory/` executed only Seed 42 (9,000 paired trials).
  2. `artifacts/confirmatory/controls-traces.jsonl` recycles the exact static hash (`3b9240af...`) from quarantined runs.
  3. Root documentation states `INTEGRITY REPAIR INCOMPLETE` and that no canonical evidence exists, directly contradicting reports at HEAD.

---

## 3. Comprehensive Remediation Roadmap

To resolve all blocking audit findings across the 9 dimensions, the following unified roadmap must be executed:

1. **Protocol & Seed Remediation (Dimensions 8 & 9):**
   - In `tool-speed-v1.3.json`, replace retrospective seeds `[42, 137, 2026]` with fresh, unobserved confirmatory seeds (`[7001, 7013, 7019]`).
   - Fix schema v3 compliance: set `"status": "prospectively_frozen"`, add `"amendment_process"`, provide 3 exploratory seeds (`[101, 102, 103]`).
   - Remove scoped validator bypasses in `protocol.py` and `cli.py`.
2. **Measured Controls & Genuine Multi-Seed Sweep (Dimensions 3 & 9):**
   - Wire `harness.run_negative_controls()` to execute real measured control arms from `toolspeed/benchmarks/controls.py`.
   - Execute a genuine 3-seed sweep across the fresh seeds (`7001`, `7013`, `7019`), generating true 27,000 paired executions in `artifacts/confirmatory/`.
3. **Bundle Sealing & Falsification Hardening (Dimension 7):**
   - Sanitize manifest filenames against path traversal (`is_relative_to`).
   - Require closed directory validation (reject bundles with untracked files).
   - Fail closed when baseline traces or summary metrics are missing/null.
   - Wire CLI `toolspeed falsify` directly to `evaluate_falsification_independent`.
4. **Subprocess Sandbox & AST Barrier Modules (Dimensions 2 & 6):**
   - Create `toolspeed/adapters/subprocess_sandbox.py` using argument vectors, executable allowlisting, and streaming output buffers.
   - Create `toolspeed/benchmarks/oracle_static_barrier.py`, expand AST scanning to adapters, and forbid test/secret imports.
5. **Concurrency, Prewarming, and LRU Caching (Dimension 4):**
   - Create `toolspeed/core/prewarming.py` and `toolspeed/core/cache.py`.
   - Fix `AsyncConcurrencyLimiter` cancellation atomicity to eliminate deadlock.
   - Add thread-safe synchronization and tenant scoping to `ToolResultCache`.
6. **Action Bytecode Serialization (Dimension 5):**
   - Consolidate codec into `toolspeed/core/bytecode.py`.
   - Preserve `call_id` and metadata during decode, enforce canonical key sorting, and wrap UTF-8 decoding in error handlers.
7. **Architecture & Deoptimization Fallback (Dimension 1):**
   - Eliminate facade delegation in `CompositeScheduler` or document unified dispatch.
   - Enforce `can_execute_in_fallback()` in JIT deoptimization loops.
   - Restrict workflow execution to `WorkflowRegistry`.
