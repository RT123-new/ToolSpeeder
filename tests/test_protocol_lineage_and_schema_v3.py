"""Unit tests for protocol lineage, prospective v1.3 draft, and strict Draft 2020-12 schema validation."""

from __future__ import annotations

import json
import unittest

from toolspeed.core.protocol import (
    load_frozen_protocol,
    resolve_protocol_resource,
    resolve_schema_resource,
    validate_protocol_dict,
)


class TestProtocolLineageAndSchemaV3(unittest.TestCase):
    """Tests prospective protocol v1.3 draft and schema v3 strict validation."""

    def test_01_load_v1_1_retrospective_repair(self) -> None:
        """tool-speed-v1.1.json must load and be classified as retrospective_repair."""
        proto = load_frozen_protocol("tool-speed-v1.1.json")
        self.assertEqual(proto.plan_id, "tool-speed-v1.1")
        self.assertEqual(proto.status, "retrospective_repair")
        self.assertEqual(proto.plan_version, "1.1.1")
        self.assertFalse(proto.is_frozen)

    def test_02_load_v1_2_draft(self) -> None:
        """tool-speed-v1.2-draft.json must load and be classified as draft."""
        proto = load_frozen_protocol("tool-speed-v1.2-draft.json")
        self.assertEqual(proto.plan_id, "tool-speed-v1.2-draft")
        self.assertEqual(proto.status, "draft")
        self.assertFalse(proto.is_frozen)

    def test_03_load_v1_3_prospective_draft(self) -> None:
        """tool-speed-v1.3-draft.json must load and be classified as prospective_draft with fresh seeds."""
        proto = load_frozen_protocol("tool-speed-v1.3-draft.json")
        self.assertEqual(proto.plan_id, "tool-speed-v1.3-draft")
        self.assertEqual(proto.status, "prospective_draft")
        self.assertFalse(proto.is_frozen)
        # Verify 6 fresh seeds total (3 exploratory, 3 confirmatory)
        self.assertEqual(len(proto.seeds), 6)
        # Verify retrospective seeds (42, 137, 2026) are NOT reused
        for s in [42, 137, 2026]:
            self.assertNotIn(s, proto.seeds, f"Retrospective seed {s} was illegally reused in v1.3")

    def test_04_schema_v3_file_exists_and_is_strict_draft2020_12(self) -> None:
        """protocol-schema-v3.json must exist in package and benchmark-plans and specify Draft 2020-12."""
        schema_path = resolve_schema_resource("protocol-schema-v3.json")
        self.assertTrue(schema_path.exists())
        schema_data = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema_data.get("$schema"), "https://json-schema.org/draft/2020-12/schema")
        self.assertFalse(schema_data.get("additionalProperties", True))

    def test_05_v1_3_rejects_reused_retrospective_confirmatory_seeds(self) -> None:
        """v1.3 validator must reject confirmatory seeds reusing retrospective seeds 42, 137, 2026."""
        path = resolve_protocol_resource("tool-speed-v1.3-draft.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        # Tamper: inject seed 42 into confirmatory list
        data["seeds"]["confirmatory"] = [42, 7013, 7019]
        errors = validate_protocol_dict(data)
        has_seed_error = any("must not be reused" in e for e in errors)
        self.assertTrue(has_seed_error, "Validator failed to reject retrospective seed reuse in confirmatory arm")

    def test_06_v1_3_rejects_overlapping_exploratory_and_confirmatory_seeds(self) -> None:
        """v1.3 validator must reject overlapping exploratory and confirmatory seeds."""
        path = resolve_protocol_resource("tool-speed-v1.3-draft.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        # Tamper: overlap seed 101
        data["seeds"]["confirmatory"] = [101, 7013, 7019]
        errors = validate_protocol_dict(data)
        has_overlap_error = any("must not overlap" in e for e in errors)
        self.assertTrue(has_overlap_error, "Validator failed to reject overlapping seeds")

    def test_07_v1_3_rejects_missing_required_metrics(self) -> None:
        """v1.3 validator must reject missing required metrics without defaults."""
        path = resolve_protocol_resource("tool-speed-v1.3-draft.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        # Tamper: remove safety_violations_count
        data["required_metrics"] = ["p95_ccl_ms", "p95_speedup"]
        errors = validate_protocol_dict(data)
        has_metric_error = any("missing one or more of 9 required metrics" in e for e in errors)
        self.assertTrue(has_metric_error, "Validator failed to catch missing required metrics")

    def test_08_v1_3_rejects_missing_execution_modes(self) -> None:
        """v1.3 validator must require smoke, exploratory, and confirmatory execution modes."""
        path = resolve_protocol_resource("tool-speed-v1.3-draft.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        # Tamper: only specify smoke
        data["supported_execution_modes"] = ["smoke"]
        errors = validate_protocol_dict(data)
        has_mode_error = any("must support 'smoke', 'exploratory', and 'confirmatory'" in e for e in errors)
        self.assertTrue(has_mode_error, "Validator failed to catch missing execution modes")


if __name__ == "__main__":
    unittest.main()
