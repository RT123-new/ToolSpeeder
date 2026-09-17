"""Base interfaces, data structures, and protocol for speculative decision providers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from toolspeed.core.sanitization import (
    assert_no_egress_violations,
    derive_argument_shapes,
    sanitize_text,
    sanitize_value,
)


@dataclass(frozen=True)
class SpeculationCandidate:
    """Immutable representation of an already-formed tool candidate eligible for speculative routing.

    Invariants:
    - Candidate arguments cannot be mutated by the decision provider.
    - Argument shapes and family metadata are safe for external egress transmission.
    """

    candidate_id: str
    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    is_read_only: bool = True
    tool_family: str = "default"
    argument_shapes: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Wrap mutable mappings in read-only MappingProxyType to guarantee immutability
        if not isinstance(self.arguments, MappingProxyType):
            object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))
        if not self.argument_shapes:
            shapes = derive_argument_shapes(self.arguments)
            object.__setattr__(self, "argument_shapes", MappingProxyType(shapes))
        elif not isinstance(self.argument_shapes, MappingProxyType):
            object.__setattr__(self, "argument_shapes", MappingProxyType(dict(self.argument_shapes)))
        if not isinstance(self.metadata, MappingProxyType):
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def candidate_hash(self) -> str:
        """Deterministic SHA-256 fingerprint of the candidate identity and arguments."""
        payload = {
            "candidate_id": self.candidate_id,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "is_read_only": self.is_read_only,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def to_sanitized_dict(self) -> dict[str, Any]:
        """Produces a sanitized representation safe for external inference."""
        data = {
            "candidate_id": self.candidate_id,
            "tool_name": self.tool_name,
            "tool_family": self.tool_family,
            "is_read_only": self.is_read_only,
            "argument_shapes": dict(self.argument_shapes),
            "sanitized_arguments": sanitize_value(dict(self.arguments)),
        }
        assert_no_egress_violations(data)
        return data


@dataclass(frozen=True)
class SpeculationState:
    """Bounded, immutable task and causal history state available at decision time."""

    task_id: str
    prompt: str
    step_index: int = 0
    history_summary: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, MappingProxyType):
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def state_hash(self) -> str:
        payload = {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "step_index": self.step_index,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def to_sanitized_dict(self) -> dict[str, Any]:
        """Produces a sanitized state representation safe for external egress."""
        data = {
            "task_id": self.task_id,
            "prompt": sanitize_text(self.prompt),
            "step_index": self.step_index,
            "history_summary": [sanitize_value(h) for h in self.history_summary],
        }
        assert_no_egress_violations(data)
        return data


@dataclass(frozen=True)
class SpeculationDecision:
    """Inspectable result produced by a speculative decision provider."""

    selected_candidate_id: str | None
    probability: float
    confidence: float
    should_speculate: bool
    provider: str
    latency_ms: float = 0.0
    fallback_used: bool = False
    error_class: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, MappingProxyType):
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_candidate_id": self.selected_candidate_id,
            "probability": self.probability,
            "confidence": self.confidence,
            "should_speculate": self.should_speculate,
            "provider": self.provider,
            "latency_ms": self.latency_ms,
            "fallback_used": self.fallback_used,
            "error_class": self.error_class,
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class SpeculationDecisionProvider(Protocol):
    """Protocol implemented by all speculative routing providers."""

    @property
    def provider_name(self) -> str:
        """Name identifying the provider."""
        ...

    async def decide(
        self,
        state: SpeculationState,
        candidates: Sequence[SpeculationCandidate],
        confidence_threshold: float = 0.70,
    ) -> SpeculationDecision:
        """Evaluate candidates and decide which (if any) should be speculatively executed."""
        ...
