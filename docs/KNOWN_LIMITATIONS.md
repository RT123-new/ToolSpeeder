# ToolSpeed Known Limitations & Scope Boundaries

This document details current scientific and engineering limitations of the ToolSpeed framework.

## 1. Live LLM / Production Spend Scope
- **Current Evidence Levels**: In PR #1, benchmarks are evaluated under `replay_integration` (trace replay) and `local_wall_clock` (local HTTP/SQLite/File/Subprocess primitives).
- **Live Production Level**: `live_production` involves paid cloud LLM inference and live third-party SaaS endpoints. This remains absent and scoped as future work to prevent unmetered external spend during testing and CI.

## 2. Evidence Eligibility Limitations
- **Operational Smoke Runs**: Smoke runs (`n=10–50` trials) are operational smoke tests only and are strictly `INCONCLUSIVE` (`is_verdict_eligible: false`).
- **Confirmatory Evidence**: Confirmatory evidence requires 3 preregistered seeds ($\ge 1,000$ trials/seed for Replay, $\ge 200$ trials/seed for Local Wall-Clock) executed under a prospectively frozen protocol. Such evidence has not yet been collected at the current head.

## 3. Protocol Status
- **Protocol v1.1**: Authored retrospectively after initial implementation exploration; serves as a repair protocol. A draft successor (`tool-speed-v1.2-draft.json`) is required to formally specify machine-evaluable hypothesis rules.

## 4. E5b Direct Token Generation vs E5a Transport Codec
- **E5a (Implemented)**: Action Bytecode transport serialization codec. Provides binary packet compression and wire serialization efficiency. It does NOT accelerate upstream model reasoning or token generation.
- **E5b (Unimplemented)**: Direct model action-token generation without JSON grammar decoding. Marked strictly `UNIMPLEMENTED` and `INCONCLUSIVE`.

## 5. Local Execution Safety vs Multi-Tenant Sandboxing
- Local execution tools (`SafeSubprocessSandbox`, `AsyncLocalFileIOTool`) provide working directory containment and execution timeouts for controlled local benchmarking. They do NOT provide hard OS-level memory caps, process-tree SIGKILL guarantees, or multi-tenant virtualization.

## 6. Cache Eviction Semantics
- Current caching implementation evicts entries based on creation timestamp (FIFO) rather than true least-recently-used access order (LRU).
