"""Optional System One LLM comparator provider (B4).

Uses official system-one-adapter-python to execute the exact same Choice and Noul
question schema through conventional LLM APIs (e.g. gpt-4o-mini, claude-3-5-haiku).
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Sequence
from typing import Any, cast

from toolspeed.core.sanitization import (
    assert_no_egress_violations,
    sanitize_text,
)
from toolspeed.schedulers.speculative_providers.base import (
    SpeculationCandidate,
    SpeculationDecision,
    SpeculationDecisionProvider,
    SpeculationState,
)

# Optional dependency check for system_one_adapter and typesafe_sdk
try:
    from system_one_adapter import AsyncSystemOneAdapterClient, Choice, Noul

    SYSTEM_ONE_ADAPTER_AVAILABLE = True
except ImportError:
    AsyncSystemOneAdapterClient = None  # type: ignore[assignment,misc]
    Choice = None  # type: ignore[assignment,misc]
    Noul = None  # type: ignore[assignment,misc]
    SYSTEM_ONE_ADAPTER_AVAILABLE = False


class SystemOneLLMProvider:
    """Experimental comparator executing System One questions via conventional LLMs."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        timeout_s: float = 2.0,
        client: Any | None = None,
        fallback_provider: SpeculationDecisionProvider | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.timeout_s = timeout_s
        self._injected_client = client
        self.fallback_provider = fallback_provider

    @property
    def provider_name(self) -> str:
        return f"llm_system_one_{self.provider}"

    @property
    def is_available(self) -> bool:
        return SYSTEM_ONE_ADAPTER_AVAILABLE

    @property
    def is_configured(self) -> bool:
        if self._injected_client is not None:
            return True
        if not self.is_available:
            return False
        if self.provider == "openai":
            return bool(os.environ.get("OPENAI_API_KEY"))
        elif self.provider == "anthropic":
            return bool(os.environ.get("ANTHROPIC_API_KEY"))
        return False

    async def decide(
        self,
        state: SpeculationState,
        candidates: Sequence[SpeculationCandidate],
        confidence_threshold: float = 0.70,
    ) -> SpeculationDecision:
        t0 = time.perf_counter()

        if not self.is_available:
            return await self._fallback(state, candidates, confidence_threshold, "SYSTEM_ONE_ADAPTER_UNAVAILABLE", t0)

        if not self.is_configured:
            return await self._fallback(
                state, candidates, confidence_threshold, f"{self.provider.upper()}_API_KEY_MISSING", t0
            )

        if not candidates:
            dur_ms = (time.perf_counter() - t0) * 1000.0
            return SpeculationDecision(
                selected_candidate_id=None,
                probability=0.0,
                confidence=0.0,
                should_speculate=False,
                provider=self.provider_name,
                latency_ms=dur_ms,
                fallback_used=False,
            )

        sanitized_state: Any = {
            "task_prompt": sanitize_text(state.prompt),
            "step_index": state.step_index,
            "candidates": [c.to_sanitized_dict() for c in candidates],
        }
        assert_no_egress_violations(sanitized_state)

        criteria: dict[str, str | None] = {}
        candidate_map: dict[str, SpeculationCandidate] = {}
        for c in candidates:
            candidate_map[c.candidate_id] = c
            arg_desc = ", ".join(f"{k}: {t}" for k, t in c.argument_shapes.items())
            criteria[c.candidate_id] = f"Execute tool '{c.tool_name}' ({c.tool_family}) with args [{arg_desc}]"
        criteria["no_speculation"] = "No candidate tool call is needed next or uncertainty is high."

        questions: dict[str, Any] = {
            "route": Choice(
                instructions="Which candidate tool call, if any, is most likely to be needed next by the agent?",
                criteria=criteria,
            ),
            "should_speculate": Noul(
                instructions="Is speculative execution of the selected candidate justified before model completes reasoning?",
            ),
        }

        try:
            if self._injected_client is not None:
                client = self._injected_client
                if hasattr(client, "__aenter__"):
                    async with client as cli:
                        response = await asyncio.wait_for(
                            cli.system_one(
                                state=sanitized_state,
                                questions=questions,
                                provider=cast(Any, self.provider),
                                model=self.model,
                            ),
                            timeout=self.timeout_s,
                        )
                else:
                    coro = client.system_one(
                        state=sanitized_state,
                        questions=questions,
                        provider=cast(Any, self.provider),
                        model=self.model,
                    )
                    response = await asyncio.wait_for(coro, timeout=self.timeout_s)
            else:
                async with AsyncSystemOneAdapterClient(
                    structured_outputs=True,
                    llm_answer_mode="probabilities",
                    normalize_probabilities=True,
                ) as cli:
                    response = await asyncio.wait_for(
                        cli.system_one(
                            state=sanitized_state,
                            questions=questions,
                            provider=cast(Any, self.provider),
                            model=self.model,
                        ),
                        timeout=self.timeout_s,
                    )

            dur_ms = (time.perf_counter() - t0) * 1000.0
            choice_ans = response.choices.get("route")
            noul_ans = response.nouls.get("should_speculate")

            if choice_ans is None or noul_ans is None:
                return await self._fallback(state, candidates, confidence_threshold, "MALFORMED_RESPONSE", t0)

            selected_label = choice_ans.choice
            route_confidence = float(choice_ans.confidence)
            route_probs = {k: float(v) for k, v in choice_ans.probabilities.items()}
            candidate_prob = route_probs.get(selected_label, 0.0)
            noul_prob = float(noul_ans.noul)

            if selected_label == "no_speculation" or selected_label not in candidate_map:
                return SpeculationDecision(
                    selected_candidate_id=None,
                    probability=candidate_prob,
                    confidence=route_confidence,
                    should_speculate=False,
                    provider=self.provider_name,
                    latency_ms=dur_ms,
                    fallback_used=False,
                    metadata={"selected_label": selected_label, "noul_prob": noul_prob},
                )

            cand = candidate_map[selected_label]
            if not cand.is_read_only:
                return SpeculationDecision(
                    selected_candidate_id=None,
                    probability=candidate_prob,
                    confidence=route_confidence,
                    should_speculate=False,
                    provider=self.provider_name,
                    latency_ms=dur_ms,
                    fallback_used=False,
                    error_class="UNSAFE_CANDIDATE_REJECTED",
                )

            should_speculate = route_confidence >= confidence_threshold and noul_prob >= 0.50

            return SpeculationDecision(
                selected_candidate_id=cand.candidate_id,
                probability=candidate_prob,
                confidence=route_confidence,
                should_speculate=should_speculate,
                provider=self.provider_name,
                latency_ms=dur_ms,
                fallback_used=False,
                metadata={
                    "noul_prob": noul_prob,
                    "probabilities": route_probs,
                    "model": self.model,
                    "provider": self.provider,
                },
            )

        except asyncio.TimeoutError:
            return await self._fallback(state, candidates, confidence_threshold, "TIMEOUT", t0)
        except Exception as e:
            return await self._fallback(state, candidates, confidence_threshold, f"ERROR_{type(e).__name__}", t0)

    async def _fallback(
        self,
        state: SpeculationState,
        candidates: Sequence[SpeculationCandidate],
        confidence_threshold: float,
        error_class: str,
        start_time: float,
    ) -> SpeculationDecision:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        if self.fallback_provider is not None:
            fallback_res = await self.fallback_provider.decide(state, candidates, confidence_threshold)
            return SpeculationDecision(
                selected_candidate_id=fallback_res.selected_candidate_id,
                probability=fallback_res.probability,
                confidence=fallback_res.confidence,
                should_speculate=fallback_res.should_speculate,
                provider=self.provider_name,
                latency_ms=elapsed_ms + fallback_res.latency_ms,
                fallback_used=True,
                error_class=error_class,
                metadata={"fallback_to": self.fallback_provider.provider_name},
            )
        return SpeculationDecision(
            selected_candidate_id=None,
            probability=0.0,
            confidence=0.0,
            should_speculate=False,
            provider=self.provider_name,
            latency_ms=elapsed_ms,
            fallback_used=True,
            error_class=error_class,
        )
