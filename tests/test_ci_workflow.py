"""Tests for Phase 33: CI workflow validation, matrix coverage, adversarial suites, and wheel install smoke testing."""

from __future__ import annotations

import unittest
from pathlib import Path


class TestCIWorkflow(unittest.TestCase):
    """Validates CI workflow YAML structure, compatibility matrix, and non-tautological test steps."""

    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parent.parent
        self.ci_file = self.repo_root / ".github" / "workflows" / "ci.yml"
        self.pyproject_file = self.repo_root / "pyproject.toml"

    def test_01_ci_workflow_exists(self) -> None:
        """CI workflow file exists and is non-empty."""
        self.assertTrue(self.ci_file.exists(), f"Missing CI workflow at {self.ci_file}")
        content = self.ci_file.read_text(encoding="utf-8")
        self.assertGreater(len(content), 100)

    def test_02_python_matrix_coverage(self) -> None:
        """CI matrix tests across Python 3.10, 3.11, 3.12, and 3.13."""
        content = self.ci_file.read_text(encoding="utf-8")
        for py_ver in ("3.10", "3.11", "3.12", "3.13"):
            self.assertIn(
                f'"{py_ver}"',
                content,
                f"Missing Python {py_ver} in compatibility matrix",
            )

    def test_03_linting_and_security_analysis_steps(self) -> None:
        """CI includes ruff, mypy, and bandit security analysis."""
        content = self.ci_file.read_text(encoding="utf-8")
        self.assertIn("ruff check", content)
        self.assertIn("ruff format --check", content)
        self.assertIn("mypy toolspeed tests", content)
        self.assertIn("bandit -r toolspeed", content)

    def test_04_unit_adversarial_and_scientific_integrity_suites(self) -> None:
        """CI executes pytest with coverage, adversarial mutation tests, scientific integrity, and review findings."""
        content = self.ci_file.read_text(encoding="utf-8")
        self.assertIn("pytest --cov=toolspeed", content)
        self.assertIn("tests/test_behavioral_mutation_integrity.py", content)
        self.assertIn("tests/test_scientific_integrity.py", content)
        self.assertIn("tests/test_review_findings.py", content)

    def test_05_wheel_build_and_clean_environment_smoke_test(self) -> None:
        """CI builds distribution wheel, installs into clean environment, and verifies CLI smoke test."""
        content = self.ci_file.read_text(encoding="utf-8")
        self.assertIn("python -m build --wheel", content)
        self.assertIn("pip install dist/*.whl", content)
        self.assertIn("toolspeed --help", content)
        self.assertIn("toolspeed simulate", content)
        self.assertIn("toolspeed validate-bundle", content)

    def test_06_pyproject_contains_required_dev_dependencies(self) -> None:
        """pyproject.toml defines bandit and pytest-cov in dev dependencies."""
        pyproject_text = self.pyproject_file.read_text(encoding="utf-8")
        self.assertIn("bandit", pyproject_text)
        self.assertIn("pytest-cov", pyproject_text)


if __name__ == "__main__":
    unittest.main()
