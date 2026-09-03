"""Comprehensive unit and adversarial tests for runtime hardening (leases, idempotency, executor)."""

from __future__ import annotations

import unittest

from toolspeed.adapters.base import ToolRegistry, ToolSchema
from toolspeed.adapters.mock_tools import MockToolAdapter
from toolspeed.core.rate_limiter import AsyncConcurrencyLimiter, RateLimiter
from toolspeed.core.types import ApprovalGrant, ExecutionAuthorityContext, ToolCall, ToolSpec
from toolspeed.schedulers.executor import (
    AuthorizationError,
    IdempotencyConflictError,
    RateLimitExceededError,
    SchemaValidationError,
    SharedIdempotencyStore,
    ToolExecutionError,
    ToolExecutor,
    ToolNotFoundError,
)


class TestRuntimeHardening(unittest.IsolatedAsyncioTestCase):
    """Verifies single lease ownership, scoped idempotency, single dispatch authority, and structured errors."""

    async def test_01_rate_limit_lease_fields_and_release_state(self) -> None:
        """Lease tracks tokens, concurrency, absolute deadline, and release state."""
        limiter = RateLimiter(max_concurrency=3, rate_per_sec=50.0)

        # Acquire with explicit deadline
        lease = await limiter.acquire_lease(tokens=2, timeout=1.0)
        self.assertEqual(lease.tokens_acquired, 2)
        self.assertTrue(lease.concurrency_acquired)
        self.assertIsNotNone(lease.absolute_deadline)
        self.assertEqual(lease.release_state, "active")
        self.assertEqual(limiter.concurrency_limiter.active_count, 1)

        # Release lease
        lease.release()
        self.assertEqual(lease.release_state, "released")
        self.assertEqual(limiter.concurrency_limiter.active_count, 0)

        # Multiple releases on lease are idempotent and safe
        lease.release()
        self.assertEqual(lease.release_state, "released")
        self.assertEqual(limiter.concurrency_limiter.active_count, 0)

    def test_02_concurrency_limiter_over_release_invariant(self) -> None:
        """Direct over-release on AsyncConcurrencyLimiter raises RuntimeError."""
        conc = AsyncConcurrencyLimiter(max_concurrency=2)
        self.assertEqual(conc.active_count, 0)
        with self.assertRaises(RuntimeError) as ctx:
            conc.release()
        self.assertIn("Over-release invariant violation", str(ctx.exception))

    async def test_03_scoped_idempotency_store_lifecycle(self) -> None:
        """SharedIdempotencyStore enforces tenant, run, provider, and operation scopes with canonical hashing."""
        store = SharedIdempotencyStore(default_ttl_s=100.0)

        args = {"amount": 100, "currency": "USD"}
        status, key, _fut, _cached = store.reserve_or_join(
            tool_name="transfer",
            arguments=args,
            idempotency_key="idemp_100",
            tenant_scope="tenant_A",
            run_scope="run_01",
            provider_scope="stripe",
            op_scope="charge",
        )
        self.assertEqual(status, "RESERVED_PRIMARY")
        self.assertIn("tenant_A:run_01:stripe:charge:transfer:idemp_100", key)

        # Same key under different tenant must NOT collide
        status_diff_tenant, key_diff, _, _ = store.reserve_or_join(
            tool_name="transfer",
            arguments=args,
            idempotency_key="idemp_100",
            tenant_scope="tenant_B",
            run_scope="run_01",
            provider_scope="stripe",
            op_scope="charge",
        )
        self.assertEqual(status_diff_tenant, "RESERVED_PRIMARY")
        self.assertNotEqual(key, key_diff)

        # Conflicting arguments under same scope fail closed
        status_conflict, _, _, _ = store.reserve_or_join(
            tool_name="transfer",
            arguments={"amount": 200, "currency": "USD"},
            idempotency_key="idemp_100",
            tenant_scope="tenant_A",
            run_scope="run_01",
            provider_scope="stripe",
            op_scope="charge",
        )
        self.assertEqual(status_conflict, "ARG_MISMATCH")

    async def test_04_tool_executor_dispatch_authority_and_safety(self) -> None:
        """ToolExecutor strictly validates schema, checks authority, and returns structured errors."""
        registry = ToolRegistry()
        spec = ToolSpec(
            name="guarded_transfer",
            description="Transfer",
            parameters={
                "type": "object",
                "properties": {"recipient": {"type": "string"}, "amount": {"type": "number", "minimum": 1}},
                "required": ["recipient", "amount"],
            },
            side_effects=True,
            requires_approval=True,
        )

        class EchoTool(MockToolAdapter):
            def get_schema(self) -> ToolSchema:
                return ToolSchema(
                    name=spec.name,
                    description=spec.description,
                    parameters=spec.parameters,
                    is_side_effect=True,
                )

        registry.register(EchoTool(spec=spec))
        executor = ToolExecutor(registry=registry)

        # 1. Unregistered tool error
        res_missing = await executor.execute(ToolCall(name="unknown_tool", arguments={}))
        self.assertTrue(res_missing.is_error)
        self.assertIn("not registered", res_missing.error or "")

        # 2. Schema validation error (missing required property)
        res_bad_schema = await executor.execute(ToolCall(name="guarded_transfer", arguments={"recipient": "Bob"}))
        self.assertTrue(res_bad_schema.is_error)
        self.assertIn("Schema validation error", res_bad_schema.error or "")

        # 3. Authorization rejection without valid ApprovalGrant
        res_unauthorized = await executor.execute(
            ToolCall(
                name="guarded_transfer",
                arguments={"recipient": "Bob", "amount": 50},
                is_approved=True,  # Untrusted model flag must be rejected
            )
        )
        self.assertTrue(res_unauthorized.is_error)
        self.assertIn("Action rejected", res_unauthorized.error or "")

        # 4. Success with valid trusted out-of-band grant
        auth_ctx = ExecutionAuthorityContext()
        valid_grant = ApprovalGrant.create(
            tool_name="guarded_transfer",
            arguments={"recipient": "Bob", "amount": 50},
            authority="trusted_system",
            issuer_secret=auth_ctx.issuer_secret,
        )
        auth_ctx.add_grant(valid_grant)

        res_ok = await executor.execute(
            ToolCall(name="guarded_transfer", arguments={"recipient": "Bob", "amount": 50}),
            authority_context=auth_ctx,
        )
        self.assertFalse(res_ok.is_error)
        self.assertEqual(res_ok.output.get("echo_args", {}).get("recipient"), "Bob")

    def test_05_structured_error_hierarchy(self) -> None:
        """Structured error classes inherit properly from ToolExecutionError."""
        err_schema = SchemaValidationError("invalid args", details={"field": "amount"})
        self.assertIsInstance(err_schema, ToolExecutionError)
        self.assertEqual(err_schema.error_code, "SCHEMA_VALIDATION_ERROR")
        self.assertEqual(err_schema.details["field"], "amount")

        err_auth = AuthorizationError("grant missing")
        self.assertIsInstance(err_auth, ToolExecutionError)
        self.assertEqual(err_auth.error_code, "AUTHORIZATION_ERROR")

        err_idemp = IdempotencyConflictError("key reuse")
        self.assertIsInstance(err_idemp, ToolExecutionError)
        self.assertEqual(err_idemp.error_code, "IDEMPOTENCY_CONFLICT")

        err_rate = RateLimitExceededError("rate limit exceeded", retry_after_s=2.5)
        self.assertIsInstance(err_rate, ToolExecutionError)
        self.assertEqual(err_rate.error_code, "RATE_LIMIT_EXCEEDED")
        self.assertEqual(err_rate.retry_after_s, 2.5)

        err_notfound = ToolNotFoundError("tool missing")
        self.assertIsInstance(err_notfound, ToolExecutionError)
        self.assertEqual(err_notfound.error_code, "TOOL_NOT_FOUND")


if __name__ == "__main__":
    unittest.main()
