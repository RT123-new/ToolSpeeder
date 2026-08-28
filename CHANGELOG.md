# Changelog
 
 All notable changes to ToolSpeed are documented in this file.
 
## [0.2.0] - 2026-08-28
 
 ### Added
 - **Formal Evidence Taxonomy**: Introduced explicit `EvidenceLevel` tags (`SYNTHETIC`, `REPLAY_INTEGRATION`, `LOCAL_WALL_CLOCK`, `LIVE_PRODUCTION`) across all reports, CLI outputs, and JSON manifests.
 - **Statistical Sample Size Enforcement**: Required minimum sample sizes ($\ge 1,000$ for Replay, $\ge 200$ for Local Wall-Clock) before verdict eligibility. Smoke runs are explicitly labeled `SMOKE — NOT VERDICT-ELIGIBLE` producing `INCONCLUSIVE`.
 - **Deterministic Virtual-Time Discrete-Event Replay Backend**: High-precision discrete event simulation computing non-blocking virtual timelines in seconds without wall-clock sleep overhead.
 - **Real Local Wall-Clock Backend**: Run-level service lifecycle management (zero-leak HTTP server, SQLite threadpool execution, sandboxed file I/O, subprocess sandboxes) with paired state isolation and alternating execution order counterbalancing.
 - **Ablation & Sensitivity Controls**: Wired 5 ablation flags (`parallelism_enabled`, `fusion_enabled`, `speculation_enabled`, `early_dispatch_enabled`, `cache_enabled`) and 6 negative/sensitivity controls into the benchmark harness.
 - **CLI `validate-bundle` Subcommand**: Validates artifact manifest, SHA-256 provenance hashes (`benchmark_config_hash`, `workload_fixture_hash`, `raw_trace_hash`), trial sample sizes, and disk artifacts.
 - **114+ Adversarial and Unit Tests**: 100% green test suite verifying topological cycle recovery, side-effect safety, rate-limiter token refunds, bytecode corrupt transport recovery, and cache invalidation.
 
 ### Fixed & Hardened
 - **Task Correctness Independence**: Fully decoupled `AgentTask`, `ExpectedOutcome`, `StateSnapshot`, and `BenchmarkCase` from model fixtures.
 - **Rate Limiter Concurrency Bug**: Fixed double-release vulnerability in `AsyncConcurrencyLimiter` and hardened token refunds upon cancellation.
 - **E1 DAG Scheduler**: Replaced single-pass regex with two-pass dependency resolution and DFS cycle detection failing closed on unresolved references.
 - **E2 JIT Fusion**: Replaced arbitrary executable lambdas with declarative AST (`DeclarativeWorkflow`, `WorkflowNode`, `WorkflowInvariant`) and safe deoptimization ledger.
 - **E3 Speculation**: Concurrent draft prediction during model reasoning, multi-call matching, and leak-free background task cancellation.
 - **E4 Commit Horizon**: Introduced `IncrementalCommitParser`, syntax closure gating, unresolved variable reference checks, and restricted early dispatch to read-only tools.
 - **E5 Action Bytecode**: Replaced 8-bit opcode binary protocol with 16-bit big-endian transport codec (`>H`) supporting up to 65,535 tools with strict payload validation.
 - **Report Generator**: Added prominent evidence badges, git commit SHA, OS platform, and benchmark artifact hashes.
