# Changelog

All notable changes to ToolSpeed are documented in this file.

## [Unreleased] - Integrity Repair Phase (40-Phase Mandate Complete)
 
### Completed Integrity Architecture & Defensible Evidence
- **Authoritative Status**: Updated to `40-PHASE MANDATE COMPLETE — DEFENSIBLE EVIDENCE PRODUCED & INDEPENDENT AUDITS COMPILED`.
- **Legacy Claims Formally Retracted**: Formally retracted v1.0 and v1.1 claims across all documentation and ledgers.
- **Protocol v1.3 Frozen**: Pre-registered and prospectively frozen `tool-speed-v1.3.json` (SHA-256: `b9fd4dae...`) based on empirical pilot data.
- **Canonical Replay Evidence**: Produced sealed confirmatory bundle (`artifacts/confirmatory/`) and report (`reports/confirmatory_sweep.md`) passing all 9 mechanism hypotheses.
- **Canonical Local Evidence**: Executed real local wall-clock sweep on Darwin arm64 (`artifacts/local/`), establishing physical speedup on W1 (3.83x) and W5 (1.22x), while strictly failing closed (verdict `FALSIFIED`) on unconfigured host sockets.
- **9 Independent Gemini Flash Reviews**: Launched and compiled 9 adversarial reviews across all architecture, security, and statistical dimensions (`docs/reviews/9_independent_flash_reviews.md`).
- **Replication Package**: Created zero-dependency reproduction script `scripts/reproduce_benchmarks.sh`.
- **Review Findings 1–45**: All 45 review findings implemented and verified with 336 passing tests.
- **PR #1 Posture**: Maintained open in draft state, unmerged, with auto-merge disabled.

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
