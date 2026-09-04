"""Base interface and classes for ToolSpeed workload families."""

from __future__ import annotations

from abc import ABC, abstractmethod

from toolspeed.adapters.base import BaseToolAdapter
from toolspeed.core.types import (
    TaskInstance,
    TaskValidator,
    WorkloadSpec,
)


class BaseWorkload(ABC):
    """Abstract base class for standard workload families (W1 to W7)."""

    @abstractmethod
    def get_spec(self) -> WorkloadSpec:
        """Return the workload family specification and parameters."""
        ...

    @abstractmethod
    def generate_tasks(self, count: int = 10, seed: int | None = None) -> list[TaskInstance]:
        """Generate a deterministic sequence of benchmark task instances."""
        ...

    @abstractmethod
    def get_tools(self) -> list[BaseToolAdapter]:
        """Return the set of tool adapters required for this workload."""
        ...

    @abstractmethod
    def get_validator(self) -> TaskValidator:
        """Return the exact task validator for this workload."""
        ...
