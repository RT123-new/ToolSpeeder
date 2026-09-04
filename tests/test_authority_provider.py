"""Comprehensive unit and concurrency tests for AuthorityProvider and out-of-band execution authority."""

from __future__ import annotations

import asyncio
import secrets
import time
import unittest

from toolspeed.core.types import (
    ApprovalGrant,
    ApprovalIssuer,
    ExecutionAuthorityContext,
    RuntimeAuthorityProvider,
)


class TestAuthorityProvider(unittest.IsolatedAsyncioTestCase):
    """Verifies cryptographic out-of-band authority provider integrity, validation, and concurrency."""

    def setUp(self) -> None:
        self.secret = secrets.token_bytes(32)
        self.provider = RuntimeAuthorityProvider(secret=self.secret, authority="trusted_system", run_id="run_test_001")

    def test_01_grant_binding_and_canonical_sha256(self) -> None:
        """Grants bind tool, arguments, tenant, subject, run_id, and full canonical SHA-256."""
        args = {"recipient": "Alice", "amount": 250.0, "note": "Payment"}
        grant = self.provider.issue(
            tool_name="execute_fund_transfer",
            arguments=args,
            ttl_seconds=60.0,
            subject="user_123",
            tenant="tenant_abc",
            run_id="run_test_001",
        )

        self.assertEqual(grant.tool_name, "execute_fund_transfer")
        self.assertEqual(grant.subject, "user_123")
        self.assertEqual(grant.tenant, "tenant_abc")
        self.assertEqual(grant.run_id, "run_test_001")
        self.assertEqual(len(grant.argument_fingerprint), 64)  # Full 256-bit SHA-256 hex string
        self.assertTrue(len(grant.signature) > 0)
        self.assertTrue(grant.single_use)

        # Verification succeeds with matching properties
        self.assertTrue(
            self.provider.verify(
                grant,
                tool_name="execute_fund_transfer",
                arguments=args,
                subject="user_123",
                tenant="tenant_abc",
                run_id="run_test_001",
            )
        )

    def test_02_rejection_of_unsigned_grants(self) -> None:
        """Unsigned grants must be strictly rejected."""
        args = {"amount": 100}
        fp = ApprovalGrant.compute_fingerprint("transfer", args)
        unsigned_grant = ApprovalGrant(
            approval_id="aid_fake",
            subject="sub",
            tool_name="transfer",
            argument_fingerprint=fp,
            expires_at=time.time() + 100,
            authority="trusted_system",
            signature="",  # Unsigned!
        )
        self.assertFalse(self.provider.verify(unsigned_grant, "transfer", args, subject="sub"))

    def test_03_rejection_of_forged_and_tampered_grants(self) -> None:
        """Grants with forged signatures, wrong secret, or tampered parameters must fail verification."""
        args = {"amount": 500}
        grant = self.provider.issue("transfer", args, subject="alice", tenant="t1", run_id="run_test_001")

        # Wrong secret
        other_provider = RuntimeAuthorityProvider(secret=secrets.token_bytes(32), run_id="run_test_001")
        self.assertFalse(other_provider.verify(grant, "transfer", args, subject="alice", tenant="t1"))

        # Wrong tool
        self.assertFalse(self.provider.verify(grant, "other_tool", args, subject="alice", tenant="t1"))

        # Changed arguments
        self.assertFalse(self.provider.verify(grant, "transfer", {"amount": 501}, subject="alice", tenant="t1"))

        # Wrong tenant
        self.assertFalse(self.provider.verify(grant, "transfer", args, subject="alice", tenant="wrong_tenant"))

        # Wrong subject
        self.assertFalse(self.provider.verify(grant, "transfer", args, subject="bob", tenant="t1"))

        # Wrong run_id
        self.assertFalse(
            self.provider.verify(grant, "transfer", args, subject="alice", tenant="t1", run_id="run_other")
        )

        # Forged signature
        tampered_sig = ApprovalGrant(
            approval_id=grant.approval_id,
            subject=grant.subject,
            tool_name=grant.tool_name,
            argument_fingerprint=grant.argument_fingerprint,
            expires_at=grant.expires_at,
            authority=grant.authority,
            nonce=grant.nonce,
            signature="deadbeef" * 8,
            tenant=grant.tenant,
            run_id=grant.run_id,
        )
        self.assertFalse(self.provider.verify(tampered_sig, "transfer", args, subject="alice", tenant="t1"))

    def test_04_rejection_of_expired_grants(self) -> None:
        """Grants past their expiry timestamp must be rejected."""
        args = {"amount": 10}
        past_time = time.time() - 100
        grant = self.provider.issue("transfer", args, ttl_seconds=-10.0, current_time=past_time)
        self.assertFalse(self.provider.verify(grant, "transfer", args))

    def test_05_rejection_of_legacy_truncated_fingerprint(self) -> None:
        """Legacy truncated (16-char) fingerprints must no longer be accepted."""
        args = {"amount": 100}
        full_fp = ApprovalGrant.compute_fingerprint("transfer", args)
        truncated_fp = full_fp[:16]

        truncated_grant = ApprovalGrant(
            approval_id="aid_trunc",
            subject="sub",
            tool_name="transfer",
            argument_fingerprint=truncated_fp,
            expires_at=time.time() + 100,
            authority="trusted_system",
            signature="any_sig",
        )
        # Even if signature matched, argument_fingerprint check must strictly require full 64-char match
        self.assertFalse(truncated_grant.matches("transfer", args))

    async def test_06_atomic_consumption_and_replay_prevention(self) -> None:
        """Single-use grants can be consumed exactly once; subsequent replay attempts must fail."""
        args = {"recipient": "Bob", "amount": 75.0}
        grant = self.provider.issue(
            tool_name="transfer",
            arguments=args,
            subject="alice",
            tenant="t1",
            single_use=True,
        )

        # First consumption must succeed
        consumed_1 = self.provider.consume(
            grant.approval_id,
            tool_name="transfer",
            arguments=args,
            subject="alice",
            tenant="t1",
        )
        self.assertTrue(consumed_1)

        # Replay attempt must fail
        consumed_2 = self.provider.consume(
            grant.approval_id,
            tool_name="transfer",
            arguments=args,
            subject="alice",
            tenant="t1",
        )
        self.assertFalse(consumed_2)

    async def test_07_concurrent_double_use_prevention(self) -> None:
        """Concurrent callers attempting to consume the same single-use grant must result in exactly one winner."""
        args = {"item_id": 999}
        grant = self.provider.issue(
            tool_name="purchase",
            arguments=args,
            subject="user_1",
            tenant="t_store",
            single_use=True,
        )

        async def attempt_consume() -> bool:
            await asyncio.sleep(0.001)
            return self.provider.consume(
                grant.approval_id,
                tool_name="purchase",
                arguments=args,
                subject="user_1",
                tenant="t_store",
            )

        # Fire 20 concurrent consumption requests
        results = await asyncio.gather(*[attempt_consume() for _ in range(20)])

        self.assertEqual(results.count(True), 1)
        self.assertEqual(results.count(False), 19)

    async def test_08_execution_authority_context_concurrency(self) -> None:
        """ExecutionAuthorityContext concurrency safety and atomic consumption."""
        ctx = ExecutionAuthorityContext(
            tenant="t_ctx",
            run_id="run_ctx",
            subject="sub_ctx",
            issuer_secret=self.secret,
            provider=self.provider,
        )

        issuer = ApprovalIssuer(secret=self.secret, run_id="run_ctx")
        g = issuer.issue(
            tool_name="critical_op",
            arguments={"id": 42},
            tenant="t_ctx",
            run_id="run_ctx",
            subject="sub_ctx",
            single_use=True,
        )
        ctx.add_grant(g)

        async def worker() -> bool:
            return ctx.verify_and_consume_grant("critical_op", {"id": 42})

        results = await asyncio.gather(*[worker() for _ in range(15)])
        self.assertEqual(results.count(True), 1)
        self.assertEqual(results.count(False), 14)


if __name__ == "__main__":
    unittest.main()
