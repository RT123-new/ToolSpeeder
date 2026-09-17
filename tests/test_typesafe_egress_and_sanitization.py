"""Tests verifying external egress sanitization and information barrier enforcement."""

from __future__ import annotations

import unittest

from toolspeed.core.sanitization import (
    EgressSecurityError,
    assert_no_egress_violations,
    derive_argument_shapes,
    sanitize_text,
    sanitize_value,
)


class TestEgressAndSanitization(unittest.TestCase):
    """Verifies that secrets, credentials, and oracle ground-truth cannot cross the egress boundary."""

    def test_bearer_token_redaction(self) -> None:
        raw = "Authorization header with Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.t-ID"
        clean = sanitize_text(raw)
        self.assertNotIn("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", clean)
        self.assertIn("[REDACTED", clean)

    def test_api_key_redaction(self) -> None:
        raw = "Configure client with TYPESAFE_API_KEY=ts_live_998877665544332211 and sk-proj_abcdef1234567890"
        clean = sanitize_text(raw)
        self.assertNotIn("ts_live_998877665544332211", clean)
        self.assertNotIn("sk-proj_abcdef1234567890", clean)
        self.assertIn("[REDACTED", clean)

    def test_secret_bearing_urls_redacted(self) -> None:
        raw = "Download file from https://user:supersecretpass@internal.vault.company.com/file?token=my_secret_token_12345"
        clean = sanitize_text(raw)
        self.assertNotIn("supersecretpass", clean)
        self.assertNotIn("my_secret_token_12345", clean)

    def test_oracle_ground_truth_stripped(self) -> None:
        payload = {
            "prompt": "Read customer document",
            "expected_output": "Secret Customer Record",
            "ground_truth": {"id": 101, "balance": 5000},
            "validator": "strict_equality",
            "oracle_canary": "canary_abc_123",
            "safe_metadata": "public_data",
        }
        clean = sanitize_value(payload)

        # Prohibited oracle keys must be completely removed from clean dict
        self.assertNotIn("expected_output", clean)
        self.assertNotIn("ground_truth", clean)
        self.assertNotIn("validator", clean)
        self.assertNotIn("oracle_canary", clean)
        self.assertIn("safe_metadata", clean)
        self.assertEqual(clean["safe_metadata"], "public_data")

    def test_assert_no_egress_violations_detects_leak(self) -> None:
        # A payload containing a raw Bearer token must raise EgressSecurityError
        leaky_data = {"user": "Alice", "auth": "Bearer secret_access_token_12345"}
        with self.assertRaises(EgressSecurityError):
            assert_no_egress_violations(leaky_data)

        # A payload containing an oracle ground-truth key must raise EgressSecurityError
        oracle_data = {"user": "Alice", "ground_truth": 42}
        with self.assertRaises(EgressSecurityError):
            assert_no_egress_violations(oracle_data)

    def test_argument_shape_derivation(self) -> None:
        args = {"query": "SELECT * FROM users", "limit": 10, "include_archived": False}
        shapes = derive_argument_shapes(args)
        self.assertEqual(shapes, {"query": "str", "limit": "int", "include_archived": "bool"})


if __name__ == "__main__":
    unittest.main()
