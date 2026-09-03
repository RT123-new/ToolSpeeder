"""Tests for Experiment E5a fair baselines and direct codec-to-JSON comparison."""

from __future__ import annotations

import unittest

from toolspeed.benchmarks.codec_bench import (
    CanonicalJSONCodec,
    benchmark_codecs_symmetric,
    get_bytecode_codec,
    get_json_codec,
)
from toolspeed.core.types import ToolCall
from toolspeed.schedulers.e5_action_bytecode import ActionBytecodeCodec


class TestExperimentE5a(unittest.TestCase):
    """Verifies E5a fair baseline: direct codec-to-JSON comparison, symmetric settings, and schema identity checks."""

    def test_01_direct_codec_to_json_comparison_not_agent_scheduler(self) -> None:
        """Verifies direct codec comparison benchmark without multi-turn agent scheduler."""
        call = ToolCall(
            call_id="c_e5a_01",
            tool_name="execute_analysis",
            arguments={"query": "SELECT * FROM users", "limit": 50, "include_archived": False},
        )
        res = benchmark_codecs_symmetric(call, schema_hash="schema_hash_e5a")

        self.assertTrue(res.round_trip_equal)
        self.assertGreater(res.json_bytes, 0)
        self.assertGreater(res.bytecode_bytes, 0)
        self.assertGreaterEqual(res.compression_ratio, 0.0)
        self.assertGreater(res.json_encode_ns, 0)
        self.assertGreater(res.bytecode_encode_ns, 0)

    def test_02_identical_serialization_settings_across_arms(self) -> None:
        """Verifies symmetric serialization policies across JSON and bytecode arms."""
        json_cfg = get_json_codec()
        bytecode_cfg = get_bytecode_codec()

        self.assertEqual(json_cfg.float_precision_policy, bytecode_cfg.float_precision_policy)
        self.assertEqual(json_cfg.key_sort_policy, bytecode_cfg.key_sort_policy)
        self.assertEqual(json_cfg.float_precision_policy, "ieee754_double")
        self.assertEqual(json_cfg.key_sort_policy, "canonical_lexicographic")

    def test_03_schema_identity_check_before_decode(self) -> None:
        """Verifies schema hash mismatch raises before decoding in both bytecode and JSON codecs."""
        call = ToolCall(
            call_id="c_schema_test",
            tool_name="transfer_funds",
            arguments={"amount": 250.75, "recipient": "acc_042"},
        )
        schema_good = "schema_hash_v1_auth"
        schema_bad = "schema_hash_v2_compromised"

        # Bytecode codec
        b_codec = ActionBytecodeCodec()
        packet_b = b_codec.encode(call, schema_hash=schema_good)

        # Good schema succeeds
        decoded_b = b_codec.decode(packet_b, expected_schema_hash=schema_good)
        self.assertEqual(decoded_b.arguments, call.arguments)

        # Mismatched schema raises ValueError
        with self.assertRaises(ValueError) as ctx:
            b_codec.decode(packet_b, expected_schema_hash=schema_bad)
        self.assertIn("Schema identity mismatch", str(ctx.exception))

        # JSON codec
        j_codec = CanonicalJSONCodec()
        packet_j = j_codec.encode(call, schema_hash=schema_good)

        decoded_j = j_codec.decode(packet_j, expected_schema_hash=schema_good)
        self.assertEqual(decoded_j.arguments, call.arguments)

        with self.assertRaises(ValueError) as ctx:
            j_codec.decode(packet_j, expected_schema_hash=schema_bad)
        self.assertIn("Schema identity mismatch", str(ctx.exception))

    def test_04_round_trip_fidelity_diverse_types(self) -> None:
        """Verifies encode/decode round-trip fidelity across complex nested types."""
        call = ToolCall(
            call_id="c_complex",
            tool_name="process_complex_payload",
            arguments={
                "str_val": "hello world 🚀",
                "int_val": 42,
                "float_val": -3.1415926535,
                "bool_val": True,
                "none_val": None,
                "list_val": [1, "two", 3.0, [4, 5]],
                "dict_val": {"nested_k": "nested_v", "inner_list": [True, False]},
            },
        )
        sh = "schema_fidelity_001"
        b_codec = ActionBytecodeCodec()
        j_codec = CanonicalJSONCodec()

        # Bytecode fidelity
        decoded_b = b_codec.decode(b_codec.encode(call, schema_hash=sh), expected_schema_hash=sh)
        self.assertEqual(decoded_b.arguments, call.arguments)
        self.assertEqual(decoded_b.tool_name, call.tool_name)

        # JSON fidelity
        decoded_j = j_codec.decode(j_codec.encode(call, schema_hash=sh), expected_schema_hash=sh)
        self.assertEqual(decoded_j.arguments, call.arguments)
        self.assertEqual(decoded_j.tool_name, call.tool_name)


if __name__ == "__main__":
    unittest.main()
