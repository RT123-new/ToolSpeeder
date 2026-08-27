# Changelog

All notable changes to ToolSpeed are documented in this file.

## [0.2.0] - 2026-08-27

### Added
- **Formal Evidence Taxonomy**: Introduced explicit `EvidenceLevel` tags (`SYNTHETIC`, `REPLAY_INTEGRATION`, `LOCAL_WALL_CLOCK`, `LIVE`) across all reports, CLI outputs, and JSON manifests.
- **Real Paired Benchmark Suite (`toolspeed/benchmarks/`)**:
  - `ReplayBackend`: Deterministic virtual-delay replay harness for canonical workloads W1–W7.
  - `LocalWallClockBackend`: Real local wall-clock harness running SQLite, mock HTTP server, sandboxed file I/O, and subprocesses.
  - `BenchmarkHarness`: Automated paired evaluations, P50/P95/P99 critical path latency (CCL), paired bootstrap 95% confidence intervals, and negative controls.
- **Centralized `ToolExecutor`**: Centralized gate for lookup, schema validation, rate-limiting, approval gating, idempotency, timeouts, and metrics.
- **22 Adversarial Scientific Integrity Unit Tests**: Comprehensive tests covering cycle detection, DAG reference discovery, side-effect safety, token bucket concurrency safety, and leak-free task cleanup.
- **PEP 517/621 Packaging & CI**: `pyproject.toml` with NumPy dependency and GitHub Actions CI matrix for Python 3.10–3.13.

### Fixed & Hardened
- **Rate Limiter Concurrency Bug**: Fixed double-release vulnerability in `AsyncConcurrencyLimiter` and hardened token refunds upon cancellation.
- **E1 DAG Scheduler**: Replaced single-pass regex with two-pass dependency resolution and DFS cycle detection failing closed on unresolved references.
- **E2 JIT Fusion**: Replaced arbitrary executable lambdas with declarative AST (`DeclarativeWorkflow`, `WorkflowNode`, `WorkflowInvariant`) and safe deoptimization ledger.
- **E3 Speculation**: Replaced serial polling with concurrent draft prediction and model reasoning, multi-call matching, and leak-free background task cancellation.
- **E4 Commit Horizon**: Introduced `IncrementalCommitParser` and restricted early dispatch to read-only tools.
- **E5 Action Bytecode**: Replaced 8-bit opcode binary protocol with 16-bit big-endian transport codec (`>H`) supporting up to 65,535 tools with strict payload validation.
- **Report Generator**: Added prominent evidence badges, git commit SHA, OS platform, and benchmark artifact hashes.
