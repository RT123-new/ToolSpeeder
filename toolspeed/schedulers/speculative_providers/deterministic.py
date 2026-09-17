"""Simple deterministic non-neural baseline provider (B2).

Uses deterministic keyword/name frequency matching without neural inference.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence

from toolspeed.schedulers.speculative_providers.base import (
    SpeculationCandidate,
    SpeculationDecision,
    SpeculationState,
)


class DeterministicFrequencyProvider:
    """Non-neural deterministic baseline using string/keyword heuristic scoring."""

    def __init__(self, latency_ms: float = 0.5) -> None:
        self.simulated_latency_ms = latency_ms

    @property
    def provider_name(self) -> str:
        return "deterministic_baseline"

    async def decide(
        self,
        state: SpeculationState,
        candidates: Sequence[SpeculationCandidate],
        confidence_threshold: float = 0.70,
    ) -> SpeculationDecision:
        t0 = time.perf_counter()
        prompt_lower = state.prompt.lower()

        best_candidate: SpeculationCandidate | None = None
        best_score = 0.0

        for cand in candidates:
            if not cand.is_read_only:
                continue

            score = 0.0
            tool_name = cand.tool_name.lower()
            if tool_name in prompt_lower:
                score += 0.5
            else:
                words = [w for w in re.split(r"[_\W]+", tool_name) if len(w) > 2]
                for w in words:
                    if w in prompt_lower:
                        score += 0.25

            # Match argument keys/values if present in prompt
            for k, v in cand.arguments.items():
                if str(k).lower() in prompt_lower:
                    score += 0.2
                if isinstance(v, str) and v.lower() in prompt_lower:
                    score += 0.3

            if score > best_score:
                best_score = score
                best_candidate = cand

        # Cap confidence in [0.0, 1.0]
        confidence = min(1.0, best_score)
        should_speculate = best_candidate is not None and confidence >= confidence_threshold

        dur_ms = ((time.perf_counter() - t0) * 1000.0) + self.simulated_latency_ms

        return SpeculationDecision(
            selected_candidate_id=best_candidate.candidate_id if best_candidate else None,
            probability=confidence,
            confidence=confidence,
            should_speculate=should_speculate,
            provider=self.provider_name,
            latency_ms=dur_ms,
            fallback_used=False,
            metadata={"heuristic_score": best_score},
        )
