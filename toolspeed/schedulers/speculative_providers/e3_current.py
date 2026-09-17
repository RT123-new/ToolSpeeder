"""Existing E3 predictor baseline provider (B1).

Wraps current E3 draft model / heuristic prediction behavior through the common provider abstraction.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from toolspeed.adapters.base import BaseLLMAdapter
from toolspeed.core.types import AgentTask, ToolCall, ToolSpec
from toolspeed.schedulers.speculative_providers.base import (
    SpeculationCandidate,
    SpeculationDecision,
    SpeculationState,
)


class E3CurrentDraftProvider:
    """Primary baseline: wraps existing E3 draft prediction through the provider interface."""

    def __init__(
        self,
        model_adapter: BaseLLMAdapter | None = None,
        draft_fn: Callable[[AgentTask, list[dict[str, Any]], list[ToolSpec]], ToolCall | None] | None = None,
        default_latency_ms: float = 70.0,
    ) -> None:
        self.model_adapter = model_adapter
        self.draft_fn = draft_fn
        self.default_latency_ms = default_latency_ms

    @property
    def provider_name(self) -> str:
        return "current_e3"

    async def decide(
        self,
        state: SpeculationState,
        candidates: Sequence[SpeculationCandidate],
        confidence_threshold: float = 0.70,
    ) -> SpeculationDecision:
        t0 = time.perf_counter()
        agent_task = AgentTask(prompt=state.prompt, task_id=state.task_id)
        history = [dict(h) for h in state.history_summary]

        predicted_call: ToolCall | None = None

        if self.draft_fn is not None:
            tool_specs = [
                ToolSpec(
                    name=c.tool_name,
                    description="",
                    parameters={},
                    is_read_only=c.is_read_only,
                )
                for c in candidates
            ]
            predicted_call = self.draft_fn(agent_task, history, tool_specs)
        elif self.model_adapter is not None:
            tool_specs = [
                ToolSpec(
                    name=c.tool_name,
                    description="",
                    parameters={},
                    is_read_only=c.is_read_only,
                )
                for c in candidates
            ]
            predicted_call = await self.model_adapter.predict_draft(agent_task, history, tool_specs)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_latency_ms = max(self.default_latency_ms if self.model_adapter is None else elapsed_ms, elapsed_ms)

        if predicted_call is None:
            return SpeculationDecision(
                selected_candidate_id=None,
                probability=0.0,
                confidence=0.0,
                should_speculate=False,
                provider=self.provider_name,
                latency_ms=total_latency_ms,
                fallback_used=False,
                metadata={"reason": "no_draft_prediction"},
            )

        pred_name = predicted_call.name or predicted_call.tool_name
        pred_confidence = float(getattr(predicted_call, "speculation_confidence", 0.85))

        # Match against eligible candidate IDs (provider never mutates candidate calls)
        matching_candidate: SpeculationCandidate | None = None
        for cand in candidates:
            if cand.tool_name == pred_name:
                if not cand.arguments or cand.arguments == predicted_call.arguments:
                    matching_candidate = cand
                    break
                elif matching_candidate is None:
                    matching_candidate = cand

        if matching_candidate is None:
            return SpeculationDecision(
                selected_candidate_id=None,
                probability=pred_confidence,
                confidence=pred_confidence,
                should_speculate=False,
                provider=self.provider_name,
                latency_ms=total_latency_ms,
                fallback_used=False,
                metadata={"reason": "draft_tool_not_in_candidates", "predicted_tool": pred_name},
            )

        should_speculate = matching_candidate.is_read_only and pred_confidence >= confidence_threshold

        return SpeculationDecision(
            selected_candidate_id=matching_candidate.candidate_id,
            probability=pred_confidence,
            confidence=pred_confidence,
            should_speculate=should_speculate,
            provider=self.provider_name,
            latency_ms=total_latency_ms,
            fallback_used=False,
            metadata={"predicted_tool": pred_name},
        )
