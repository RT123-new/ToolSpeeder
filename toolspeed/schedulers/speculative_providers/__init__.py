"""Speculative Decision Providers package."""

from toolspeed.schedulers.speculative_providers.base import (
    SpeculationCandidate,
    SpeculationDecision,
    SpeculationDecisionProvider,
    SpeculationState,
)
from toolspeed.schedulers.speculative_providers.deterministic import DeterministicFrequencyProvider
from toolspeed.schedulers.speculative_providers.e3_current import E3CurrentDraftProvider
from toolspeed.schedulers.speculative_providers.no_speculation import NoSpeculationProvider
from toolspeed.schedulers.speculative_providers.replay import (
    ReplayRecord,
    ReplaySpeculationProvider,
    compute_composite_hash,
)
from toolspeed.schedulers.speculative_providers.system_one_llm import SystemOneLLMProvider
from toolspeed.schedulers.speculative_providers.typesafe_jev import TypeSafeJevProvider

__all__ = [
    "DeterministicFrequencyProvider",
    "E3CurrentDraftProvider",
    "NoSpeculationProvider",
    "ReplayRecord",
    "ReplaySpeculationProvider",
    "SpeculationCandidate",
    "SpeculationDecision",
    "SpeculationDecisionProvider",
    "SpeculationState",
    "SystemOneLLMProvider",
    "TypeSafeJevProvider",
    "compute_composite_hash",
]
