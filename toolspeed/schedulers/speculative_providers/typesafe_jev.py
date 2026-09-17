"""TypeSafe/Jev System One speculative routing provider (B3).

Integrates official typesafe-sdk Choice and Noul primitives to select among
immutable read-only candidate calls with calibrated probabilistic gating.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Sequence
from typing import Any

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

# Optional dependency check for typesafe_sdk
try:
    import typesafe_sdk
    from typesafe_sdk import (
        AsyncTypeSafeClient,
        Choice,
        Noul,
        Question,
        TypeSafeAPIConnectionError,
        TypeSafeAPIError,
        TypeSafeAPITimeoutError,
        TypeSafeAuthenticationError,
        TypeSafeError,
        TypeSafeRateLimitError,
    )
    TYPESAFE_SDK_AVAILABLE = True
except ImportError:
    typesafe_sdk = None  # type: ignore[assignment]
    AsyncTypeSafeClient = None  # type: ignore[assignment,misc]
    Choice = None  # type: ignore[assignment,misc]
    Noul = None  # type: ignore[assignment,misc]
    Question = Any  # type: ignore[assignment,misc]
    TypeSafeError = Exception  # type: ignore[assignment,misc]
    TypeSafeAPIError = Exception  # type: ignore[assignment,misc]
    TypeSafeAPIConnectionError = Exception  # type: ignore[assignment,misc]
    TypeSafeAPITimeoutError = Exception  # type: ignore[assignment,misc]
    TypeSafeRateLimitError = Exception  # type: ignore[assignment,misc]
    TypeSafeAuthenticationError = Exception  # type: ignore[assignment,misc]
    TYPESAFE_SDK_AVAILABLE = False


class TypeSafeJevProvider:
    """TypeSafe/Jev System One decision provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 2.0,  # 2.0s bounded timeout for live inference
        client: Any | None = None,
        fallback_provider: SpeculationDecisionProvider | None = None,
        noul_threshold: float = 0.30,
    ) -> None:
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        self.model = model
        self.timeout_s = timeout_s
        self._injected_client = client
        self.fallback_provider = fallback_provider
        self.noul_threshold = noul_threshold

    @property
    def provider_name(self) -> str:
        return "typesafe_jev"

    @property
    def is_available(self) -> bool:
        """Returns True if typesafe-sdk is importable."""
        return TYPESAFE_SDK_AVAILABLE

    @property
    def is_configured(self) -> bool:
        """Returns True if API key or injected client is present."""
        return bool(self._injected_client is not None or (self.is_available and self.api_key))

    async def decide(
        self,
        state: SpeculationState,
        candidates: Sequence[SpeculationCandidate],
        confidence_threshold: float = 0.70,
    ) -> SpeculationDecision:
        t0 = time.perf_counter()

        # 1. Check SDK availability
        if not self.is_available:
            return await self._fallback(
                state,
                candidates,
                confidence_threshold,
                error_class="TYPESAFE_SDK_UNAVAILABLE",
                start_time=t0,
            )

        # 2. Check API key configuration
        if not self.is_configured:
            return await self._fallback(
                state,
                candidates,
                confidence_threshold,
                error_class="TYPESAFE_API_KEY_MISSING",
                start_time=t0,
            )

        # 3. Empty candidates check
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
                metadata={"reason": "no_candidates"},
            )

        # 4. Construct sanitized prompt and state payload for TypeSafe
        sanitized_state: Any = {
            "task_prompt": sanitize_text(state.prompt),
            "step_index": state.step_index,
            "candidates": [c.to_sanitized_dict() for c in candidates],
        }
        assert_no_egress_violations(sanitized_state)

        # 5. Build question schema (Choice over immutable candidate IDs + Noul for justification)
        criteria: dict[str, str | None] = {}
        candidate_map: dict[str, SpeculationCandidate] = {}
        for c in candidates:
            candidate_map[c.candidate_id] = c
            arg_desc = ", ".join(f"{k}: {t}" for k, t in c.argument_shapes.items())
            criteria[c.candidate_id] = f"Execute tool '{c.tool_name}' ({c.tool_family}) with args [{arg_desc}]"
        criteria["no_speculation"] = "No candidate tool call is needed next or uncertainty is high."

        questions: dict[str, Question] = {
            "route": Choice(
                instructions="Which candidate tool call, if any, is most likely to be needed next by the agent?",
                criteria=criteria,
            ),
            "should_speculate": Noul(
                instructions="Is speculative execution of the selected candidate justified before model completes reasoning?",
            ),
        }

        # 6. Execute external inference under bounded timeout
        try:
            if self._injected_client is not None:
                client_ctx = self._injected_client
                if hasattr(client_ctx, "system_one"):
                    call_res = client_ctx.system_one(state=sanitized_state, questions=questions, model=self.model)
                    if asyncio.iscoroutine(call_res):
                        response = await asyncio.wait_for(call_res, timeout=self.timeout_s)
                    else:
                        response = call_res
                elif hasattr(client_ctx, "__aenter__"):
                    async with client_ctx as cli:
                        call_res = cli.system_one(state=sanitized_state, questions=questions, model=self.model)
                        if asyncio.iscoroutine(call_res):
                            response = await asyncio.wait_for(call_res, timeout=self.timeout_s)
                        else:
                            response = call_res
                else:
                    raise TypeError(f"Invalid client object: {type(client_ctx)}")
            else:
                async with AsyncTypeSafeClient(api_key=self.api_key, timeout=self.timeout_s) as client:
                    response = await asyncio.wait_for(
                        client.system_one(state=sanitized_state, questions=questions, model=self.model),
                        timeout=self.timeout_s,
                    )

            provider_latency_ms = (time.perf_counter() - t0) * 1000.0

            # 7. Parse response primitives
            choice_ans = response.choices.get("route")
            noul_ans = response.nouls.get("should_speculate")

            if choice_ans is None or noul_ans is None:
                return await self._fallback(
                    state,
                    candidates,
                    confidence_threshold,
                    error_class="MALFORMED_RESPONSE",
                    start_time=t0,
                )

            selected_label = choice_ans.choice
            route_confidence = float(choice_ans.confidence)
            route_probs = {k: float(v) for k, v in choice_ans.probabilities.items()}
            candidate_prob = route_probs.get(selected_label, 0.0)
            noul_prob = float(noul_ans.noul)

            # 8. Evaluation of recommendation
            if selected_label == "no_speculation" or selected_label not in candidate_map:
                return SpeculationDecision(
                    selected_candidate_id=None,
                    probability=candidate_prob,
                    confidence=route_confidence,
                    should_speculate=False,
                    provider=self.provider_name,
                    latency_ms=provider_latency_ms,
                    fallback_used=False,
                    metadata={
                        "selected_label": selected_label,
                        "noul_prob": noul_prob,
                        "model": getattr(response, "model", "jev"),
                    },
                )

            cand = candidate_map[selected_label]

            # Invariant: Jev may recommend, but only safe read-only candidates pass
            if not cand.is_read_only:
                return SpeculationDecision(
                    selected_candidate_id=None,
                    probability=candidate_prob,
                    confidence=route_confidence,
                    should_speculate=False,
                    provider=self.provider_name,
                    latency_ms=provider_latency_ms,
                    fallback_used=False,
                    error_class="UNSAFE_CANDIDATE_REJECTED",
                    metadata={"rejected_candidate_id": cand.candidate_id, "reason": "non_read_only"},
                )

            # Confidence-gated speculation condition:
            # Requires reported confidence >= threshold AND Noul probability >= noul_threshold
            should_speculate = route_confidence >= confidence_threshold and noul_prob >= self.noul_threshold

            return SpeculationDecision(
                selected_candidate_id=cand.candidate_id,
                probability=candidate_prob,
                confidence=route_confidence,
                should_speculate=should_speculate,
                provider=self.provider_name,
                latency_ms=provider_latency_ms,
                fallback_used=False,
                metadata={
                    "noul_prob": noul_prob,
                    "probabilities": route_probs,
                    "model": getattr(response, "model", "jev"),
                },
            )

        except asyncio.TimeoutError:
            return await self._fallback(
                state,
                candidates,
                confidence_threshold,
                error_class="TIMEOUT",
                start_time=t0,
            )
        except TypeSafeRateLimitError:
            return await self._fallback(
                state,
                candidates,
                confidence_threshold,
                error_class="RATE_LIMIT_429",
                start_time=t0,
            )
        except TypeSafeAuthenticationError:
            return await self._fallback(
                state,
                candidates,
                confidence_threshold,
                error_class="AUTH_FAILURE_401_403",
                start_time=t0,
            )
        except (TypeSafeAPIConnectionError, TypeSafeAPITimeoutError):
            return await self._fallback(
                state,
                candidates,
                confidence_threshold,
                error_class="CONNECTION_FAILURE",
                start_time=t0,
            )
        except TypeSafeAPIError as e:
            return await self._fallback(
                state,
                candidates,
                confidence_threshold,
                error_class=f"API_ERROR_{getattr(e, 'status_code', 'GENERIC')}",
                start_time=t0,
            )
        except Exception as e:
            return await self._fallback(
                state,
                candidates,
                confidence_threshold,
                error_class=f"UNHANDLED_{type(e).__name__}",
                start_time=t0,
            )

    async def _fallback(
        self,
        state: SpeculationState,
        candidates: Sequence[SpeculationCandidate],
        confidence_threshold: float,
        error_class: str,
        start_time: float,
    ) -> SpeculationDecision:
        """Executes fallback safely upon provider error or missing configuration."""
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
                metadata={"fallback_to": self.fallback_provider.provider_name, **dict(fallback_res.metadata)},
            )

        # Default fallback: safe no-speculation
        return SpeculationDecision(
            selected_candidate_id=None,
            probability=0.0,
            confidence=0.0,
            should_speculate=False,
            provider=self.provider_name,
            latency_ms=elapsed_ms,
            fallback_used=True,
            error_class=error_class,
            metadata={"fallback": "no_speculation"},
        )
