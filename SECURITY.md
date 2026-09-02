# Security Policy for ToolSpeed

## Overview
ToolSpeed executes and orchestrates automated AI agent tool calling pipelines. Because schedulers interact with local system services, filesystem paths, and mutative actions, runtime safety boundaries are enforced to prevent unintended execution.

## Core Runtime Safety Guarantees & Operational Boundaries

1. **Centralized Tool Execution Routing (`ToolExecutor`)**:
   All scheduler policies (E1–E5, baselines B1–B5, and Phase 2 Cache) route tool execution through `ToolExecutor`. Direct unmetered execution is bypassed.
2. **Approval Gating on Mutative Actions**:
   Any tool with `side_effects=True` or `is_read_only=False` requires an active, valid approval grant. Schedulers must not manufacture authorization grants.
3. **Speculation & Horizon Safety**:
   Speculative reads and early commit horizons are restricted by contract to tools declared read-only and idempotent. Mutative tools are rejected from speculative execution.
4. **Controlled Local Tool Execution**:
   Local tools (`AsyncLocalFileIOTool`, `SafeSubprocessSandbox`) operate within designated temporary working directories with configured execution timeouts. They are controlled benchmark execution tools and do not provide hardened OS-level multi-tenant container isolation or hard memory caps.
5. **Rate Limiting**:
   `RateLimiter` enforces capacity bounds, unified deadline timeouts, and non-blocking token refunds on cancellation.
6. **Bytecode Transport Codec**:
   `ActionBytecodeCodec` validates 16-bit opcode bounds, payload size limits (16 MB), and rejects trailing corrupt bytes.

## Reporting Vulnerabilities
Please report security vulnerabilities by filing a confidential security advisory through the repository's GitHub Security Advisory tab.
