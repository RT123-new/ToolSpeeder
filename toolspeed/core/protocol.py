"""Authoritative Schema-Validated Benchmark Protocol Specification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ProtocolValidationError(Exception):
    """Raised when a benchmark protocol violates schema or integrity invariants."""


@dataclass(frozen=True)
class HypothesisThresholds:
    p95_speedup_target: float = 1.00
    min_success_rate: float = 0.95
    max_success_drop: float = 0.00
    min_p99_speedup: float = 0.95
    max_cost_multiplier: float = 1.10
    max_unapproved_side_effects: int = 0
    max_duplicate_mutations: int = 0


@dataclass(frozen=True)
class MechanismDefinition:
    workload_id: str
    name: str
    candidate: str
    primary_attribution_baseline: str
    practical_baseline: str
    informational_baselines: list[str] = field(default_factory=list)
    evaluates_hypothesis: str = "primary_attribution"
    hypotheses: HypothesisThresholds = field(default_factory=HypothesisThresholds)
    status: str = "ACTIVE"


@dataclass(frozen=True)
class BenchmarkProtocol:
    plan_id: str
    plan_version: str
    status: str
    description: str
    workload_fixture_version: str
    required_metric_policy_version: str
    evidence_levels: list[str]
    seeds: list[int]
    trials_per_seed_replay: int
    trials_per_seed_local: int
    smoke_trials: int
    bootstrap_resamples: int
    bootstrap_ci: float
    warmup_trials: int
    warmup_symmetric: bool
    warmup_isolated: bool
    warmup_excluded: bool
    missing_data_policy: str
    central_hypothesis_name: str
    central_hypothesis_description: str
    central_hypothesis_rule: str
    mechanisms: dict[str, MechanismDefinition]
    negative_controls: list[dict[str, Any]]
    positive_sensitivity_control: dict[str, Any]
    amendment_ledger: list[dict[str, Any]]
    raw_json: str
    protocol_hash: str


def validate_protocol_dict(data: dict[str, Any]) -> list[str]:
    """Validates raw protocol dictionary against schema invariants. Returns list of errors."""
    errors: list[str] = []
    required_fields = [
        "plan_id",
        "plan_version",
        "status",
        "workload_fixture_version",
        "required_metric_policy_version",
        "evidence_levels",
        "seeds",
        "trials_per_seed_replay",
        "trials_per_seed_local",
        "smoke_trials",
        "bootstrap_resamples",
        "bootstrap_ci",
        "warmup_policy",
        "missing_data_policy",
        "central_hypothesis",
        "mechanisms",
        "controls",
    ]
    for rf in required_fields:
        if rf not in data:
            errors.append(f"Missing required top-level field: '{rf}'")

    if "mechanisms" in data and isinstance(data["mechanisms"], dict):
        required_workloads = ["W1", "W2", "W3", "W4", "W5", "W6", "W7_SAFETY", "W7_LATENCY", "E5a"]
        for wl in required_workloads:
            if wl not in data["mechanisms"]:
                errors.append(f"Missing required mechanism in protocol: '{wl}'")
            else:
                m = data["mechanisms"][wl]
                if m.get("status") != "UNIMPLEMENTED":
                    if not m.get("candidate"):
                        errors.append(f"Mechanism '{wl}' missing 'candidate'")
                    if not m.get("primary_attribution_baseline"):
                        errors.append(f"Mechanism '{wl}' missing 'primary_attribution_baseline'")

    if "controls" in data and isinstance(data["controls"], dict):
        if "negative_controls" not in data["controls"] or not isinstance(data["controls"]["negative_controls"], list):
            errors.append("Missing 'controls.negative_controls' array")
        if "positive_sensitivity_control" not in data["controls"] or not isinstance(
            data["controls"]["positive_sensitivity_control"], dict
        ):
            errors.append("Missing 'controls.positive_sensitivity_control' object")

    return errors


def load_frozen_protocol(plan_path: str | Path | None = None) -> BenchmarkProtocol:
    """Loads and validates the authoritative benchmark protocol."""
    target_path = Path(plan_path) if plan_path is not None else Path("benchmark-plans/tool-speed-v1.1.json")
    if not target_path.exists():
        # Fallback to relative or package location if needed
        alt = Path(__file__).resolve().parent.parent.parent / "benchmark-plans" / "tool-speed-v1.1.json"
        if alt.exists():
            target_path = alt
        else:
            raise FileNotFoundError(f"Authoritative benchmark protocol not found at: {target_path}")

    raw_text = target_path.read_text(encoding="utf-8")
    data = json.loads(raw_text)
    errors = validate_protocol_dict(data)
    if errors:
        raise ProtocolValidationError(f"Protocol validation failed: {'; '.join(errors)}")

    protocol_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

    mechanisms: dict[str, MechanismDefinition] = {}
    for wl_id, m_dict in data.get("mechanisms", {}).items():
        hyp_dict = m_dict.get("hypotheses", {})
        thresh = HypothesisThresholds(
            p95_speedup_target=float(hyp_dict.get("p95_speedup_target", 1.0)),
            min_success_rate=float(hyp_dict.get("min_success_rate", 0.95)),
            max_success_drop=float(hyp_dict.get("max_success_drop", 0.0)),
            min_p99_speedup=float(hyp_dict.get("min_p99_speedup", 0.95)),
            max_cost_multiplier=float(hyp_dict.get("max_cost_multiplier", 1.10)),
            max_unapproved_side_effects=int(hyp_dict.get("max_unapproved_side_effects", 0)),
            max_duplicate_mutations=int(hyp_dict.get("max_duplicate_mutations", 0)),
        )
        mechanisms[wl_id] = MechanismDefinition(
            workload_id=wl_id,
            name=m_dict.get("name", wl_id),
            candidate=m_dict.get("candidate", ""),
            primary_attribution_baseline=m_dict.get("primary_attribution_baseline", ""),
            practical_baseline=m_dict.get("practical_baseline", ""),
            informational_baselines=list(m_dict.get("informational_baselines", [])),
            evaluates_hypothesis=m_dict.get("evaluates_hypothesis", "primary_attribution"),
            hypotheses=thresh,
            status=m_dict.get("status", "ACTIVE"),
        )

    warmup = data.get("warmup_policy", {})
    central = data.get("central_hypothesis", {})

    return BenchmarkProtocol(
        plan_id=data["plan_id"],
        plan_version=data["plan_version"],
        status=data.get("status", "prospectively_frozen"),
        description=data.get("description", ""),
        workload_fixture_version=data.get("workload_fixture_version", "1.1.0"),
        required_metric_policy_version=data.get("required_metric_policy_version", "2.1.0"),
        evidence_levels=list(data.get("evidence_levels", [])),
        seeds=list(data.get("seeds", [20260825])),
        trials_per_seed_replay=int(data.get("trials_per_seed_replay", 1000)),
        trials_per_seed_local=int(data.get("trials_per_seed_local", 200)),
        smoke_trials=int(data.get("smoke_trials", 10)),
        bootstrap_resamples=int(data.get("bootstrap_resamples", 2000)),
        bootstrap_ci=float(data.get("bootstrap_ci", 0.95)),
        warmup_trials=int(warmup.get("trials", 5)),
        warmup_symmetric=bool(warmup.get("symmetric", True)),
        warmup_isolated=bool(warmup.get("isolated", True)),
        warmup_excluded=bool(warmup.get("excluded_from_evidence", True)),
        missing_data_policy=data.get("missing_data_policy", "null_inconclusive"),
        central_hypothesis_name=central.get("name", ""),
        central_hypothesis_description=central.get("description", ""),
        central_hypothesis_rule=central.get("aggregation_rule", "conjunction_all_primary_mechanisms"),
        mechanisms=mechanisms,
        negative_controls=list(data.get("controls", {}).get("negative_controls", [])),
        positive_sensitivity_control=dict(data.get("controls", {}).get("positive_sensitivity_control", {})),
        amendment_ledger=list(data.get("amendment_ledger", [])),
        raw_json=raw_text,
        protocol_hash=protocol_hash,
    )


# Frozen protocol instance
FROZEN_PROTOCOL = load_frozen_protocol()
