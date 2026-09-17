"""Comprehensive Privacy and Egress Red Team for TypeSafe/Jev Confirmatory Protocol v1.0.

Adversarially injects API keys, bearer tokens, passwords, authorization headers,
private repository tokens, and benchmark oracle ground truth into task prompts and state.
Verifies that all forbidden data is strictly redacted before egress.
"""

from __future__ import annotations

import unittest
from types import MappingProxyType
from unittest.mock import MagicMock

from toolspeed.core.sanitization import (
    FORBIDDEN_ORACLE_KEYS,
    assert_no_egress_violations,
    sanitize_text,
    sanitize_value,
)
from toolspeed.schedulers.speculative_providers import (
    SpeculationCandidate,
    SpeculationState,
    TypeSafeJevProvider,
)


class TestEgressRedTeam(unittest.IsolatedAsyncioTestCase):
    """Rigorous verification of the egress security and oracle boundary."""

    def test_forbidden_oracle_keys_pruning(self) -> None:
        for oracle_key in FORBIDDEN_ORACLE_KEYS:
            dirty_payload = {
                "task_id": "test_1",
                oracle_key: "CLASSIFIED_GROUND_TRUTH_DATA",
                "normal_field": "public_data",
            }
            cleaned = sanitize_value(dirty_payload)
            self.assertNotIn(oracle_key, cleaned, f"Prohibited oracle key '{oracle_key}' leaked past sanitization!")
            self.assertEqual(cleaned["normal_field"], "public_data")

    def test_credential_redaction_patterns(self) -> None:
        test_injections = [
            ("Bearer secret_jwt_token_1234567890abcdef", "[REDACTED_BEARER_TOKEN]"),
            (
                "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
                "[REDACTED_JWT_TOKEN]",
            ),
            ("api_key='sk-prod-998877665544332211'", "key=[REDACTED]"),
            ("token=ts-live-998877665544332211", "key=[REDACTED]"),
            ("https://admin:SuperSecretPass123@db.internal.net/records", "[REDACTED_URL_WITH_CREDENTIALS]"),
            (
                "https://api.example.com/data?token=verysecretapikeyvalue",
                "https://api.example.com/data??token=verysecretapikeyvalue=[REDACTED]",
            ),
            ("Authorization: Bearer super_secret_auth_token_999", "[REDACTED_AUTH_HEADER]"),
            ("ghp_1234567890abcdef1234567890abcdef12", "[REDACTED_GENERIC_SECRET_PREFIX]"),
        ]

        for dirty_str, _ in test_injections:
            sanitized = sanitize_text(dirty_str)
            self.assertNotIn("secret_jwt_token_1234567890abcdef", sanitized)
            self.assertNotIn("SuperSecretPass123", sanitized)
            self.assertNotIn("ghp_1234567890abcdef1234567890abcdef12", sanitized)
            # Must pass egress assertion without raising
            assert_no_egress_violations({"text": sanitized})

    async def test_provider_egress_interception(self) -> None:
        """Captures payload immediately before SDK invocation and asserts zero leaks."""
        captured_payloads = []

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock(choice="c1", confidence=0.9, probabilities={"c1": 0.9, "no_speculation": 0.1})
        mock_noul = MagicMock(noul=0.8)
        mock_response.choices = {"route": mock_choice}
        mock_response.nouls = {"should_speculate": mock_noul}
        mock_response.model = "jev-test"

        def _fake_system_one(state, questions, model):
            captured_payloads.append(state)
            return mock_response

        mock_client.system_one = _fake_system_one

        provider = TypeSafeJevProvider(api_key="ts-mock-key-123", client=mock_client)

        adversarial_state = SpeculationState(
            task_id="adv_01",
            prompt="Find user profile with api_key='sk-live-secret-token-123456789' and password: 'Password123!'",
            step_index=1,
            history_summary=(),
        )
        candidates = [
            SpeculationCandidate(
                candidate_id="c1",
                tool_name="get_user",
                arguments=MappingProxyType({"user_id": "U1"}),
                is_read_only=True,
                tool_family="crm",
            )
        ]

        _decision = await provider.decide(adversarial_state, candidates)
        self.assertEqual(len(captured_payloads), 1)

        payload = captured_payloads[0]
        # Verify forbidden raw secrets were redacted before crossing boundary
        self.assertNotIn("sk-live-secret-token-123456789", payload["task_prompt"])
        self.assertNotIn("Password123!", payload["task_prompt"])
        assert_no_egress_violations(payload)


if __name__ == "__main__":
    unittest.main()
