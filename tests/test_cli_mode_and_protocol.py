"""Unit tests for CLI protocol selection, execution modes, and strict confirmatory gates."""

from __future__ import annotations

import argparse
import unittest
from dataclasses import dataclass, field
from unittest.mock import patch

from toolspeed.cli import cmd_benchmark


@dataclass
class MockFrozenProtocol:
    plan_id: str = "test-frozen"
    plan_version: str = "1.0.0"
    status: str = "prospectively_frozen"
    is_frozen: bool = True
    seeds: list[int] = field(default_factory=lambda: [7001, 7013, 7019])
    seeds_dict: dict[str, list[int]] = field(default_factory=lambda: {"confirmatory": [7001, 7013, 7019]})


class TestCLIModeAndProtocol(unittest.TestCase):
    """Tests CLI protocol loading and fail-closed gates for confirmatory mode."""

    def test_01_invalid_mode_rejected(self) -> None:
        """CLI must reject unknown execution modes."""
        args = argparse.Namespace(
            backend="replay",
            protocol="tool-speed-v1.3-draft.json",
            mode="hypothetical",
            trials=10,
            seed=101,
            seeds=None,
            concurrency=1,
            out=None,
        )
        code = cmd_benchmark(args)
        self.assertEqual(code, 1)

    def test_02_confirmatory_rejects_draft_protocol(self) -> None:
        """Confirmatory mode must reject draft protocols."""
        args = argparse.Namespace(
            backend="replay",
            protocol="tool-speed-v1.3-draft.json",
            mode="confirmatory",
            trials=1000,
            seed=7001,
            seeds="7001,7013,7019",
            concurrency=1,
            out=None,
        )
        code = cmd_benchmark(args)
        self.assertEqual(code, 1)

    def test_03_confirmatory_rejects_retrospective_repair_protocol(self) -> None:
        """Confirmatory mode must reject retrospective repair protocol tool-speed-v1.1.json."""
        args = argparse.Namespace(
            backend="replay",
            protocol="tool-speed-v1.1.json",
            mode="confirmatory",
            trials=1000,
            seed=7001,
            seeds="7001,7013,7019",
            concurrency=1,
            out=None,
        )
        code = cmd_benchmark(args)
        self.assertEqual(code, 1)

    def test_04_confirmatory_rejects_insufficient_trials(self) -> None:
        """Confirmatory mode must enforce minimum trial counts (>= 1,000 for replay)."""
        with patch("toolspeed.cli.load_frozen_protocol", return_value=MockFrozenProtocol()):
            args = argparse.Namespace(
                backend="replay",
                protocol="test-frozen",
                mode="confirmatory",
                trials=500,  # Below 1000 minimum
                seed=7001,
                seeds="7001,7013,7019",
                concurrency=1,
                out=None,
            )
            code = cmd_benchmark(args)
            self.assertEqual(code, 1)

    def test_05_confirmatory_rejects_retrospective_seed_reuse(self) -> None:
        """Confirmatory mode must strictly forbid reusing seeds 42, 137, 2026."""
        proto = MockFrozenProtocol(
            seeds=[42, 7013, 7019],
            seeds_dict={"confirmatory": [42, 7013, 7019]},
        )
        with patch("toolspeed.cli.load_frozen_protocol", return_value=proto):
            args = argparse.Namespace(
                backend="replay",
                protocol="test-frozen",
                mode="confirmatory",
                trials=1000,
                seed=42,
                seeds="42,7013,7019",
                concurrency=1,
                out=None,
            )
            code = cmd_benchmark(args)
            self.assertEqual(code, 1)

    def test_06_confirmatory_rejects_fewer_than_3_seeds(self) -> None:
        """Confirmatory mode requires >= 3 distinct seeds."""
        proto = MockFrozenProtocol(
            seeds=[7001, 7013],
            seeds_dict={"confirmatory": [7001, 7013]},
        )
        with patch("toolspeed.cli.load_frozen_protocol", return_value=proto):
            args = argparse.Namespace(
                backend="replay",
                protocol="test-frozen",
                mode="confirmatory",
                trials=1000,
                seed=7001,
                seeds="7001,7013",
                concurrency=1,
                out=None,
            )
            code = cmd_benchmark(args)
            self.assertEqual(code, 1)

    def test_07_confirmatory_rejects_dirty_git_working_tree(self) -> None:
        """Confirmatory mode must reject dirty git working tree."""
        with (
            patch("toolspeed.cli.load_frozen_protocol", return_value=MockFrozenProtocol()),
            patch("toolspeed.cli.is_git_working_tree_dirty", return_value=True),
        ):
            args = argparse.Namespace(
                backend="replay",
                protocol="test-frozen",
                mode="confirmatory",
                trials=1000,
                seed=7001,
                seeds="7001,7013,7019",
                concurrency=1,
                out=None,
            )
            code = cmd_benchmark(args)
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
