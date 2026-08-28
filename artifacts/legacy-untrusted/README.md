# Quarantined Untrusted Artifacts (PR #1 Head `2066ae80`)

> [!WARNING]
> The artifacts archived in this directory were generated prior to the integrity audit of PR #1 and are **NOT authoritative evidence**.

## Background & Reason for Quarantine

During the audit of PR #1 head commit `2066ae80e40905e7ffca50bb148d74700094101a`, the following critical issues were identified:

1. **Synthetic Simulation Relabelling**: The CLI `benchmark` command previously invoked `SuiteRunner.run()`, which executed synthetic Monte Carlo simulation math instead of actual live tool schedulers, and saved the synthetic results under `artifacts/replay/` and `artifacts/local/` with false `replay_integration` and `local_wall_clock` labels.
2. **Asyncio Cancellation Bugs**: On Python 3.10 and 3.11, child tasks cancelled via `task.cancel()` raised uncaught `asyncio.CancelledError` because `CancelledError` inherits from `BaseException`. This caused silent task failures or unhandled exception leaks.
3. **Incomplete Benchmark Backends**: Schedulers were not executed against paired deterministic fixtures or genuine local OS primitives (HTTP, SQLite, File I/O, Subprocesses).
4. **Arbitrary Python Invariants**: E2 JIT fusion executed untrusted arbitrary Python callables rather than a bounded, declarative AST.

## Quarantine Policy

- Files in `pr1-head-2066ae80/` are retained exclusively for historical audit and regression testing.
- Do not cite these numbers as verified speedup figures.
- Canonical evidence must only be generated via the audited `toolspeed benchmark --backend replay` and `toolspeed benchmark --backend local` commands.
