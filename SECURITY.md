# Security Policy for ToolSpeed

## Overview
ToolSpeed executes and orchestrates automated AI agent tool calling pipelines. Because schedulers may interact with real local wall-clock services, sandboxed environments, and mutative actions, strict runtime safety boundaries are enforced.

## Core Runtime Safety Guarantees

1. **Centralized Tool Execution Authority (`ToolExecutor`)**:
   All scheduler policies (E1–E5, baselines B1–B5, and Phase 2 Cache) MUST route tool execution through `ToolExecutor`. Direct unmetered execution is prohibited.
2. **Approval Gating on Mutative Actions**:
   Any tool with `side_effects=True` or `is_read_only=False` requires explicit approval (`is_approved=True`). Schedulers cannot execute unapproved mutative actions.
3. **Speculation & Horizon Safety**:
   Speculative reads and early commit horizons are strictly restricted to read-only, non-mutative tools. Speculative mutation is mathematically prohibited.
4. **Sandboxed Subprocess Execution**:
   `SafeSubprocessSandbox` enforces working directory containment, memory caps, timeout termination, and process tree SIGKILL cancellation.
5. **Rate Limiting & Anti-Storm Defense**:
   `RateLimiter` enforces double-release safety, strict capacity bounds, unified deadline timeouts, and non-blocking token refunds on cancellation.
6. **Bytecode Transport Security**:
   `ActionBytecodeCodec` enforces strict length bounds, 16-bit opcode bounds, payload size limits (16 MB), and strict rejection of trailing corrupt bytes.

## Reporting Vulnerabilities
Please report security vulnerabilities by filing a confidential security advisory on GitHub or emailing security@toolspeed.org.
