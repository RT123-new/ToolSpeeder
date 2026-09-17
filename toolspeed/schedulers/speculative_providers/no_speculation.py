"""No-speculation baseline provider (B0).

Unconditionally declines speculative execution to measure whether prediction overhead
is worse than simply waiting.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from toolspeed.schedulers.speculative_providers.base import (
    SpeculationCandidate,
    SpeculationDecision,
    SpeculationState,
)


class NoSpeculationProvider:
    """Baseline provider that never authorizes speculative tool dispatch."""

    @property
    def provider_name(self) -> str:
        return "no_speculation"

    async def decide(
        self,
        state: SpeculationState,
        candidates: Sequence[SpeculationCandidate],
        confidence_threshold: float = 0.70,
    ) -> SpeculationDecision:
        t0 = time.perf_counter()
        dur_ms = (time.perf_counter() - t0) * 1000.0
        return SpeculationDecision(
            selected_candidate_id=None,
            probability=0.0,
            confidence=0.0,
            should_speculate=False,
            provider=self.provider_name,
            latency_ms=dur_ms,
            fallback_used=False,
            metadata={"reason": "always_decline"},
        )
