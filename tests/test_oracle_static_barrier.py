"""Static AST regression test enforcing strict oracle barrier across all schedulers."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

FORBIDDEN_ORACLE_IDENTIFIERS = {
    "expected_output",
    "expected_outcome",
    "validator",
    "oracle_canary",
    "ground_truth",
}


class TestOracleStaticBarrier(unittest.TestCase):
    """Verifies that no scheduler code accesses oracle ground truth or canary data."""

    def test_schedulers_have_zero_oracle_access(self) -> None:
        schedulers_dir = Path(__file__).resolve().parent.parent / "toolspeed" / "schedulers"
        self.assertTrue(schedulers_dir.is_dir(), f"Schedulers directory missing: {schedulers_dir}")

        violations: list[str] = []

        for py_file in schedulers_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue

            content = py_file.read_text(encoding="utf-8")
            try:
                tree = ast.parse(content, filename=str(py_file))
            except SyntaxError as e:
                violations.append(f"{py_file.name}: Syntax error during parse: {e}")
                continue

            for node in ast.walk(tree):
                # Check attribute access, e.g. ctx.task.expected_output or self.validator
                if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ORACLE_IDENTIFIERS:
                    violations.append(f"{py_file.name}:{node.lineno}: Prohibited oracle attribute access '{node.attr}'")
                # Check exact string literals matching oracle keys
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if node.value in FORBIDDEN_ORACLE_IDENTIFIERS:
                        violations.append(
                            f"{py_file.name}:{node.lineno}: Prohibited oracle literal string '{node.value}'"
                        )
                # Check name access
                elif isinstance(node, ast.Name) and node.id in FORBIDDEN_ORACLE_IDENTIFIERS:
                    violations.append(f"{py_file.name}:{node.lineno}: Prohibited oracle identifier '{node.id}'")

        self.assertEqual(
            violations,
            [],
            "Static analysis found oracle barrier violations in schedulers:\n" + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
