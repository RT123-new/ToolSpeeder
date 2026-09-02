# Changelog

All notable changes to ToolSpeed are documented in this file.

## [Unreleased] - Integrity Repair Phase

### Corrected & Retracted
- **Claim Retraction**: Retracted `INTEGRITY REPAIR COMPLETED — EVIDENCE READY` claim from PR #1 body. Set authoritative verdict to `INTEGRITY REPAIR INCOMPLETE`.
- **Documentation Alignment**: Updated `README.md`, `SECURITY.md`, `docs/PR1_REPAIR_STATUS.md`, and `docs/KNOWN_LIMITATIONS.md` with required status language:
  - Current status: integrity repair in progress.
  - Current CI status: compatibility and smoke checks pass on `e3c974be61d58df0c2870a098099f1371a161c10`.
  - Current confirmatory evidence: not yet collected under a valid frozen protocol.
  - Replay/local smoke outputs: operational tests only, not verdict-eligible.
  - Live evidence: absent.
- **Security Claim Normalization**: Removed claims of OS-level memory caps, SIGKILL process tree termination, and mathematically prohibited speculation from `SECURITY.md`.
- **Regression Suite**: Added `tests/test_review_findings.py` with failing regression tests reproducing findings A through O.

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
