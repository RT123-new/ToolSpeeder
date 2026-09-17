"""Deterministic Replay Provider for Speculative Routing.

Records and replays sanitized decision fixtures to enable 100% deterministic,
network-independent harness verification and test reproducibility.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from toolspeed.core.sanitization import assert_no_egress_violations
from toolspeed.schedulers.speculative_providers.base import (
    SpeculationCandidate,
    SpeculationDecision,
    SpeculationState,
)


@dataclass(frozen=True)
class ReplayRecord:
    """A single recorded decision record."""

    composite_hash: str
    request_hash: str
    candidate_set_hash: str
    provider: str
    selected_candidate_id: str | None
    probability: float
    confidence: float
    should_speculate: bool
    observed_latency_ms: float
    probabilities: Mapping[str, float]
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = {
            "composite_hash": self.composite_hash,
            "request_hash": self.request_hash,
            "candidate_set_hash": self.candidate_set_hash,
            "provider": self.provider,
            "selected_candidate_id": self.selected_candidate_id,
            "probability": self.probability,
            "confidence": self.confidence,
            "should_speculate": self.should_speculate,
            "observed_latency_ms": self.observed_latency_ms,
            "probabilities": dict(self.probabilities),
            "metadata": dict(self.metadata),
        }
        assert_no_egress_violations(data)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplayRecord:
        assert_no_egress_violations(data)
        return cls(
            composite_hash=data["composite_hash"],
            request_hash=data["request_hash"],
            candidate_set_hash=data["candidate_set_hash"],
            provider=data["provider"],
            selected_candidate_id=data.get("selected_candidate_id"),
            probability=float(data.get("probability", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            should_speculate=bool(data.get("should_speculate", False)),
            observed_latency_ms=float(data.get("observed_latency_ms", 0.0)),
            probabilities=dict(data.get("probabilities", {})),
            metadata=dict(data.get("metadata", {})),
        )


def compute_composite_hash(state: SpeculationState, candidates: Sequence[SpeculationCandidate]) -> tuple[str, str, str]:
    """Computes (composite_hash, request_hash, candidate_set_hash)."""
    req_hash = state.state_hash
    cand_hashes = [c.candidate_hash for c in candidates]
    cand_set_raw = json.dumps(sorted(cand_hashes)).encode("utf-8")
    cand_set_hash = hashlib.sha256(cand_set_raw).hexdigest()
    comp_raw = f"{req_hash}:{cand_set_hash}".encode()
    composite_hash = hashlib.sha256(comp_raw).hexdigest()
    return composite_hash, req_hash, cand_set_hash


class ReplaySpeculationProvider:
    """Replays pre-recorded speculation decisions deterministically without network."""

    def __init__(
        self,
        records: Sequence[ReplayRecord | dict[str, Any]] | None = None,
        simulate_latency: bool = True,
        clock: Any = None,
        target_provider: str = "replay_jev",
    ) -> None:
        self._records_by_hash: dict[str, ReplayRecord] = {}
        self._sequential_records: list[ReplayRecord] = []
        self._seq_index = 0
        self.simulate_latency = simulate_latency
        self.clock = clock
        self._target_provider = target_provider

        if records:
            for r in records:
                rec = r if isinstance(r, ReplayRecord) else ReplayRecord.from_dict(r)
                self._records_by_hash[rec.composite_hash] = rec
                self._sequential_records.append(rec)

    @property
    def provider_name(self) -> str:
        return self._target_provider

    def add_record(self, record: ReplayRecord) -> None:
        self._records_by_hash[record.composite_hash] = record
        self._sequential_records.append(record)

    def export_records(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._sequential_records]

    async def decide(
        self,
        state: SpeculationState,
        candidates: Sequence[SpeculationCandidate],
        confidence_threshold: float = 0.70,
    ) -> SpeculationDecision:
        t0 = time.perf_counter()
        comp_hash, _req_hash, _cand_hash = compute_composite_hash(state, candidates)

        # Lookup by exact hash match first, then sequential fallback if configured
        matched_rec = self._records_by_hash.get(comp_hash)
        if matched_rec is None and self._seq_index < len(self._sequential_records):
            matched_rec = self._sequential_records[self._seq_index]
            self._seq_index += 1

        if matched_rec is None:
            dur_ms = (time.perf_counter() - t0) * 1000.0
            return SpeculationDecision(
                selected_candidate_id=None,
                probability=0.0,
                confidence=0.0,
                should_speculate=False,
                provider=self.provider_name,
                latency_ms=dur_ms,
                fallback_used=True,
                error_class="REPLAY_RECORD_NOT_FOUND",
                metadata={"composite_hash": comp_hash},
            )

        # Simulate observed replay latency
        target_latency = matched_rec.observed_latency_ms
        if self.simulate_latency and target_latency > 0:
            if self.clock is not None and hasattr(self.clock, "sleep_ms"):
                await self.clock.sleep_ms(target_latency)
            else:
                await asyncio.sleep(target_latency / 1000.0)

        dur_ms = ((time.perf_counter() - t0) * 1000.0) + (target_latency if not self.simulate_latency else 0.0)

        # Invariant: Replay candidate ID must exist in current candidate set
        candidate_ids = {c.candidate_id for c in candidates}
        selected_id = matched_rec.selected_candidate_id
        if selected_id is not None and selected_id not in candidate_ids:
            return SpeculationDecision(
                selected_candidate_id=None,
                probability=matched_rec.probability,
                confidence=matched_rec.confidence,
                should_speculate=False,
                provider=self.provider_name,
                latency_ms=dur_ms,
                fallback_used=True,
                error_class="REPLAY_CANDIDATE_ID_NOT_IN_ACTIVE_SET",
            )

        should_speculate = matched_rec.should_speculate and matched_rec.confidence >= confidence_threshold

        return SpeculationDecision(
            selected_candidate_id=selected_id if should_speculate else None,
            probability=matched_rec.probability,
            confidence=matched_rec.confidence,
            should_speculate=should_speculate,
            provider=self.provider_name,
            latency_ms=dur_ms,
            fallback_used=False,
            metadata={
                "replay": True,
                "recorded_provider": matched_rec.provider,
                "composite_hash": comp_hash,
                **dict(matched_rec.metadata),
            },
        )
