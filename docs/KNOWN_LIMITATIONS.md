# ToolSpeed Known Limitations & Scope Boundaries

This document details current scientific and engineering limitations of the ToolSpeed framework.

## 1. Live LLM / Production Spend Scope
- **Current Evidence Levels**: In PR #1, benchmarks are evaluated under `replay_integration` (trace replay) and `local_wall_clock` (local HTTP/SQLite/File/Subprocess primitives).
- **Live Production Level**: `live_production` involves paid cloud LLM inference and live third-party SaaS endpoints. This remains absent and scoped as future work to prevent unmetered external spend during testing and CI.

## 2. Evidence Eligibility Limitations
- **Operational Smoke Runs**: Smoke runs (`n=10–50` trials) are operational smoke tests only and are strictly `INCONCLUSIVE` (`is_verdict_eligible: false`).
- **September 2 Replay Runs**: Retained as noncanonical exploratory diagnostics. Executed under retrospective v1.1 protocol rules with single-process loops, synthetic seed arrays, and hardcoded control dicts. They do not satisfy prospective confirmatory requirements.
- **Confirmatory Evidence**: Confirmatory evidence requires 3 preregistered seeds ($\ge 1,000$ trials/seed for Replay, $\ge 200$ trials/seed for Local Wall-Clock) executed under a prospectively frozen protocol from a clean code tree. Such evidence has not yet been collected.

## 3. Protocol Lineage & Prospective Governance
- **Protocol v1.1**: Retrospective repair protocol. Not prospective.
- **Protocol v1.2-draft**: Unfrozen exploratory draft. Seeds and thresholds visible during exploratory runs; cannot be frozen retroactively.
- **Protocol v1.3-draft**: Prospective protocol defining explicit execution modes, machine-evaluable statistical inference, and strict Draft 2020-12 validation. Must be frozen prior to confirmatory execution.

## 4. E5b Direct Token Generation vs E5a Transport Codec
- **E5a (Implemented)**: Action Bytecode transport serialization codec. Provides binary packet compression and wire serialization efficiency. It does NOT accelerate upstream model reasoning or token generation.
- **E5b (Unimplemented)**: Direct model action-token generation without JSON grammar decoding. Marked strictly `UNIMPLEMENTED` and `INCONCLUSIVE`.

## 5. Local Execution Safety vs Multi-Tenant Sandboxing
- Local execution tools (`SafeSubprocessSandbox`, `AsyncLocalFileIOTool`) provide working directory containment and execution timeouts for controlled local benchmarking. They do NOT provide hard OS-level memory caps, process-tree SIGKILL guarantees, or multi-tenant virtualization.

## 6. Cache Eviction Semantics
- Current caching implementation evicts entries based on creation timestamp (FIFO) rather than true least-recently-used access order (LRU).
