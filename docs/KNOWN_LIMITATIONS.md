# ToolSpeed Known Limitations & Scope Boundaries

This document details current scientific and engineering limitations of the ToolSpeed framework.

## 1. Live LLM / Production Spend Scope
- **Current Evidence Levels**: In PR #1, benchmarks are validated under `replay_integration` (trace replay) and `local_wall_clock` (local HTTP/SQLite/File/Subprocess primitives).
- **Live Production Level**: `live_production` involves paid cloud LLM inference and live third-party SaaS endpoints. This remains scoped as future work to prevent unmetered external spend during testing and CI.

## 2. E5b Direct Token Generation vs E5a Transport Codec
- **E5a (Implemented)**: Action Bytecode transport serialization codec. Provides binary packet compression and wire serialization efficiency.
- **E5b (Unimplemented)**: Direct model action-token generation without JSON grammar decoding. Requires specialized model fine-tuning or token-level vocabulary patching, which is currently unproven in live models.

## 3. Speculative Reads (E3) Network Contention
- Speculative read-only tool calls provide substantial latency improvements when server capacity is available, but under single-slot head-of-line blocking (`single_slot` contention mode), mispredicted draft calls can introduce queueing delays.

## 4. JIT Fusion (E2) Scope
- E2 utilizes a bounded, declarative AST. It intentionally disallows executing arbitrary Python callables from untrusted inputs to prevent remote code execution vulnerabilities. Workflows requiring arbitrary computational loops must rely on the model reasoning loop or safe sandboxed tools.
