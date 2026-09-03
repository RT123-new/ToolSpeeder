# Changelog

All notable changes to ToolSpeed are documented in this file.

## [Unreleased] - Integrity Repair Phase
 
### Retracted Claims & Interim Status
- **Authoritative Status**: Set to `INTEGRITY REPAIR INCOMPLETE`.
- **Confirmatory Replay Claim Retraction**: Retracted `INTEGRITY REPAIR COMPLETED — CONFIRMATORY FALSIFICATION REPORTED`. Reclassified September 2 runs as noncanonical exploratory diagnostics.
- **CI Quality Gates**: Retracted `100% quality gates passing` for head `be36503` where CI was red due to 6 mypy errors. Restored strict typing and removed E2 oracle leak at `5cfef2e`.
- **Seed Matrix Retraction**: Retracted claims of 3 independent confirmatory seeds; harness did not iterate seed arrays and CLI synthesized synthetic seed lists.
- **Controls Retraction**: Retracted claims of verified execution controls; negative and positive controls were hard-coded dictionaries.
- **Facade Retraction**: Retracted resolution of findings A–O that relied on test-facing stubs (`create_w2_state`, `PersistentColdPool.acquire_time_ms`, etc.).
- **Oracle Barrier**: Removed `expected_output` and `validate` access from JIT fusion scheduler; added AST regression test `tests/test_oracle_static_barrier.py`.

## [0.2.0] - 2026-08-28

### Added
- **Formal Evidence Taxonomy**: Introduced explicit `EvidenceLevel` tags (`SYNTHETIC`, `REPLAY_INTEGRATION`, `LOCAL_WALL_CLOCK`, `LIVE_PRODUCTION`).
- **Statistical Sample Size Rules**: Required minimum sample sizes ($\ge 1,000$ for Replay, $\ge 200$ for Local Wall-Clock) before verdict eligibility.
- **Ablation & Control Framework**: Wired ablation flags and negative/sensitivity controls into the benchmark harness.
- **CLI Subcommands**: Added `benchmark`, `validate-bundle`, `falsify`, and `report` subcommands.

### Fixed & Hardened
- **Rate Limiter**: Fixed double-release in `AsyncConcurrencyLimiter` and hardened token refunds upon cancellation.
- **E1 DAG Scheduler**: Two-pass dependency resolution and DFS cycle detection failing closed on unresolved references.
- **E2 JIT Fusion**: Declarative AST (`DeclarativeWorkflow`, `WorkflowNode`, `WorkflowInvariant`) with deoptimization ledger.
- **E3 Speculation**: Concurrent draft prediction during model reasoning with cancellation handling.
- **E4 Commit Horizon**: Syntax closure gating and early dispatch restricted to read-only tools.
- **E5 Action Bytecode**: 16-bit big-endian transport codec (`>H`) with payload validation.
