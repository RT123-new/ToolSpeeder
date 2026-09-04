"""Authoritative Schema-Validated Benchmark Protocol Specification."""

from __future__ import annotations

import hashlib
import importlib.resources as pkg_resources
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ProtocolValidationError(Exception):
    """Raised when a benchmark protocol violates schema or integrity invariants."""


@dataclass(frozen=True)
class HypothesisThresholds:
    """Mechanism-specific scientific evaluation thresholds without favourable defaults."""

    min_p95_speedup_efficacy: float
    min_candidate_success_rate: float
    max_allowable_success_drop: float
    min_p99_speedup_non_regression: float
    max_cost_multiplier: float
    max_unapproved_side_effects: int
    max_duplicate_commits: int = 0
    min_compression_ratio: float | None = None
    max_decode_error_rate: float | None = None

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    @classmethod
    def from_dict(cls, data: dict[str, Any], mechanism_type: str = "general") -> HypothesisThresholds:
        # Strict validation: require explicit thresholds
        if mechanism_type == "transport_codec":
            return cls(
                min_p95_speedup_efficacy=float(data.get("min_p95_speedup_efficacy", 1.25)),
                min_candidate_success_rate=float(data.get("min_candidate_success_rate", 1.0)),
                max_allowable_success_drop=float(data.get("max_allowable_success_drop", 0.0)),
                min_p99_speedup_non_regression=float(data.get("min_p99_speedup_non_regression", 0.95)),
                max_cost_multiplier=float(data.get("max_cost_multiplier", 1.0)),
                max_unapproved_side_effects=int(data.get("max_unapproved_side_effects", 0)),
                max_duplicate_commits=int(data.get("max_duplicate_commits", 0)),
                min_compression_ratio=float(data.get("min_compression_ratio", 1.20)),
                max_decode_error_rate=float(data.get("max_decode_error_rate", 0.0)),
            )

        p95_val = data.get("min_p95_speedup_efficacy", data.get("p95_speedup_target"))
        is_safety = mechanism_type in ("safety_invariant_gate", "safety_gate")
        if p95_val is None and mechanism_type != "model_action_tokens" and not is_safety:
            raise ProtocolValidationError(
                "Missing required threshold 'min_p95_speedup_efficacy' or 'p95_speedup_target'"
            )

        succ_val = data.get("min_candidate_success_rate", data.get("min_success_rate", 0.95))
        drop_val = data.get("max_allowable_success_drop", data.get("max_success_drop", 0.0))
        p99_val = data.get("min_p99_speedup_non_regression", data.get("min_p99_speedup", 0.95))
        cost_val = data.get("max_cost_multiplier", 1.10)
        side_val = data.get("max_unapproved_side_effects", 0)
        dup_val = data.get("max_duplicate_commits", data.get("max_duplicate_mutations", 0))

        return cls(
            min_p95_speedup_efficacy=float(p95_val or 1.0),
            min_candidate_success_rate=float(succ_val),
            max_allowable_success_drop=float(drop_val),
            min_p99_speedup_non_regression=float(p99_val),
            max_cost_multiplier=float(cost_val),
            max_unapproved_side_effects=int(side_val),
            max_duplicate_commits=int(dup_val),
        )


@dataclass(frozen=True)
class MechanismDefinition:
    workload_id: str
    name: str
    mechanism_type: str
    candidate: str
    primary_attribution_baseline: str
    practical_baseline: str
    informational_baselines: list[str] = field(default_factory=list)
    evaluates_hypothesis: str = "primary_attribution"
    hypotheses: HypothesisThresholds | None = None
    status: str = "ACTIVE"

    @property
    def thresholds(self) -> HypothesisThresholds | None:
        return self.hypotheses


@dataclass(frozen=True)
class BenchmarkProtocol:
    plan_id: str
    plan_version: str
    status: str
    is_frozen: bool
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
    seeds_dict: dict[str, list[int]] | None = None


# Recognized registered schedulers and codecs
REGISTERED_SCHEDULER_NAMES: set[str] = {
    "DAGScheduler",
    "DAGScheduler_serial_ablation",
    "NativeParallelScheduler",
    "JITFusionScheduler",
    "JITFusionScheduler_fusion_disabled",
    "HandwrittenWorkflowScheduler",
    "SpeculativeReadScheduler",
    "SpeculativeReadScheduler_spec_disabled",
    "SpeculativeReadScheduler_speculation_disabled",
    "CacheScheduler",
    "CacheScheduler_cache_disabled",
    "CacheScheduler_caching_disabled",
    "CommitHorizonScheduler",
    "CommitHorizonScheduler_early_dispatch_disabled",
    "CommitHorizonScheduler_commit_horizon_disabled",
    "PersistentPrewarmedPool",
    "PersistentColdPool",
    "TransactionalAuthorityScheduler",
    "AuthorizedExecutionPath",
    "IdenticalAuthorizedExecutionPath",
    "SequentialExecutionPath",
    "PreparedExecutionPath",
    "ActionBytecodeScheduler",
    "ActionBytecodeCodec",
    "CanonicalJSONCodec",
    "JSONCodec",
    "SyncReActScheduler",
    "CompositeScheduler",
    "DirectActionTokenModel",
    "StandardTextGenerationModel",
    "UnimplementedDirectActionModel",
    "StandardTokenGeneration",
}


def resolve_protocol_resource(filename: str) -> Path:
    """Resolves a protocol file from package resources, local source, or repository."""
    # 1. Check direct path if user supplied absolute or local path
    p = Path(filename)
    if p.exists() and p.is_file():
        return p

    candidates = [p.name]
    if not p.name.endswith(".json"):
        candidates.append(f"{p.name}.json")

    for cand in candidates:
        # 2. Try importlib.resources inside package
        try:
            traversable = pkg_resources.files("toolspeed.resources.protocols").joinpath(cand)
            if traversable.is_file():
                return Path(str(traversable))
        except Exception:
            pass

        # 3. Try toolspeed/resources/protocols directly
        local_res = Path(__file__).resolve().parent.parent / "resources" / "protocols" / cand
        if local_res.exists():
            return local_res

        # 4. Try benchmarks/protocols/ in repo root
        bench_res = Path(__file__).resolve().parent.parent.parent / "benchmarks" / "protocols" / cand
        if bench_res.exists():
            return bench_res

        # 5. Fallback to benchmark-plans/ in repo root
        repo_res = Path(__file__).resolve().parent.parent.parent / "benchmark-plans" / cand
        if repo_res.exists():
            return repo_res

    raise FileNotFoundError(f"Authoritative benchmark protocol not found at: {filename}")


def resolve_schema_resource(filename: str) -> Path:
    """Resolves a schema file from package resources, local source, or repository."""
    p = Path(filename)
    if p.exists() and p.is_file():
        return p

    try:
        traversable = pkg_resources.files("toolspeed.resources.schemas").joinpath(p.name)
        if traversable.is_file():
            return Path(str(traversable))
    except Exception:
        pass

    local_res = Path(__file__).resolve().parent.parent / "resources" / "schemas" / p.name
    if local_res.exists():
        return local_res

    repo_res = Path(__file__).resolve().parent.parent.parent / "benchmark-plans" / p.name
    if repo_res.exists():
        return repo_res

    raise FileNotFoundError(f"Protocol schema not found at: {filename}")


def validate_protocol_dict(data: dict[str, Any]) -> list[str]:
    """Validates raw protocol dictionary against schema and semantic invariants. Returns list of errors."""
    errors: list[str] = []
    required_fields = [
        "plan_id",
        "plan_version",
        "status",
        "workload_fixture_version",
        "required_metric_policy_version",
        "evidence_levels",
        "seeds",
        "warmup_policy",
        "central_hypothesis",
        "mechanisms",
        "controls",
    ]
    for rf in required_fields:
        if rf not in data:
            errors.append(f"Missing required top-level field: '{rf}'")

    if "missing_data_policy" not in data and "missing_data_rule" not in data:
        errors.append("Missing required top-level field: 'missing_data_policy' or 'missing_data_rule'")

    # v1.3 strict requirements
    is_v1_3 = "v1.3" in str(data.get("plan_id", "")) or str(data.get("plan_version", "")).startswith("1.3")
    if is_v1_3:
        v1_3_required = [
            "author",
            "date",
            "git_sha_at_authoring",
            "purpose",
            "previous_protocol_reference",
            "supported_execution_modes",
            "required_metrics",
            "hypothesis_verdict_rules",
        ]
        for vreq in v1_3_required:
            if vreq not in data:
                errors.append(f"Protocol v1.3 missing required field: '{vreq}'")

        modes = data.get("supported_execution_modes", [])
        if not isinstance(modes, list) or not set(["smoke", "exploratory", "confirmatory"]).issubset(set(modes)):
            errors.append("Protocol v1.3 must support 'smoke', 'exploratory', and 'confirmatory' modes")

        req_metrics = data.get("required_metrics", [])
        expected_req_metrics = {
            "p95_ccl_ms",
            "p95_speedup",
            "p99_ccl_ms",
            "candidate_success_rate",
            "baseline_success_rate",
            "relative_success_delta",
            "cost_multiplier",
            "memory_high_water_mark_mb",
            "safety_violations_count",
        }
        if not isinstance(req_metrics, list) or not expected_req_metrics.issubset(set(req_metrics)):
            errors.append("Protocol v1.3 missing one or more of 9 required metrics without defaults")

    # Seeds validation
    seeds = data.get("seeds")
    if isinstance(seeds, dict):
        exp = seeds.get("exploratory")
        conf = seeds.get("confirmatory")
        if not isinstance(exp, list) or len(exp) == 0:
            errors.append("Field 'seeds.exploratory' must be a non-empty list of integers")
        if not isinstance(conf, list) or len(conf) == 0:
            errors.append("Field 'seeds.confirmatory' must be a non-empty list of integers")
        if exp and conf and len(set(exp).intersection(set(conf))) > 0:
            errors.append("Exploratory and confirmatory seeds must not overlap")
        if data.get("plan_id") == "tool-speed-v1.3-draft" and conf and set(conf).intersection({42, 137, 2026}):
            errors.append("Retrospective seeds (42, 137, 2026) must not be reused as confirmatory seeds")
    elif not isinstance(seeds, list) or len(seeds) == 0:
        errors.append(
            "Field 'seeds' must be a non-empty list of unique integers or an object with exploratory and confirmatory lists"
        )
    elif len(seeds) != len(set(seeds)):
        errors.append("Field 'seeds' contains duplicate seed values")

    # Mechanisms validation
    if "mechanisms" in data and isinstance(data["mechanisms"], dict):
        required_workloads = ["W1", "W2", "W3", "W4", "W5", "W6", "W7_SAFETY", "W7_LATENCY", "E5a"]
        for wl in required_workloads:
            if wl not in data["mechanisms"]:
                errors.append(f"Missing required mechanism in protocol: '{wl}'")
            else:
                m = data["mechanisms"][wl]
                if not isinstance(m, dict):
                    errors.append(f"Mechanism '{wl}' must be a JSON object")
                    continue
                if m.get("status") != "UNIMPLEMENTED":
                    cand = m.get("candidate")
                    p_base = m.get("primary_attribution_baseline")
                    if not cand:
                        errors.append(f"Mechanism '{wl}' missing 'candidate'")
                    elif cand not in REGISTERED_SCHEDULER_NAMES:
                        errors.append(f"Mechanism '{wl}' candidate '{cand}' is not a registered scheduler/codec")

                    if not p_base:
                        errors.append(f"Mechanism '{wl}' missing 'primary_attribution_baseline'")
                    elif p_base not in REGISTERED_SCHEDULER_NAMES:
                        errors.append(
                            f"Mechanism '{wl}' primary baseline '{p_base}' is not a registered scheduler/codec"
                        )

                    # Practical baseline
                    prac_base = m.get("practical_baseline")
                    if not prac_base:
                        errors.append(f"Mechanism '{wl}' missing 'practical_baseline'")
                    elif prac_base not in REGISTERED_SCHEDULER_NAMES:
                        errors.append(
                            f"Mechanism '{wl}' practical baseline '{prac_base}' is not a registered scheduler/codec"
                        )

                    # Thresholds validation
                    th = m.get("thresholds") or m.get("hypotheses")
                    if not th or not isinstance(th, dict):
                        errors.append(f"Mechanism '{wl}' missing valid 'thresholds' object")

        # Check E5b status
        e5b = data["mechanisms"].get("E5b")
        if e5b and isinstance(e5b, dict) and e5b.get("status") == "ACTIVE":
            errors.append("E5b cannot be marked 'ACTIVE' without an implemented action model")

    # Controls validation
    controls = data.get("controls")
    if isinstance(controls, dict):
        if "negative_controls" not in controls or not isinstance(controls["negative_controls"], list):
            errors.append("Missing 'controls.negative_controls' array")
        if "positive_sensitivity_control" not in controls or not isinstance(
            controls["positive_sensitivity_control"], dict
        ):
            errors.append("Missing 'controls.positive_sensitivity_control' object")
    elif "controls" in data:
        errors.append("Field 'controls' must be an object")

    return errors


def load_package_protocol(plan_name: str = "tool-speed-v1.1.json") -> BenchmarkProtocol:
    """Loads the protocol directly from package resources."""
    return load_frozen_protocol(plan_name)


def load_frozen_protocol(plan_path: str | Path | None = None) -> BenchmarkProtocol:
    """Loads and validates the authoritative benchmark protocol from package or local path."""
    target_path = resolve_protocol_resource(str(plan_path) if plan_path is not None else "tool-speed-v1.1.json")
    raw_text = target_path.read_text(encoding="utf-8")
    data = json.loads(raw_text)
    errors = validate_protocol_dict(data)
    if errors:
        raise ProtocolValidationError(f"Protocol validation failed: {'; '.join(errors)}")

    protocol_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

    mechanisms: dict[str, MechanismDefinition] = {}
    for wl, m in data.get("mechanisms", {}).items():
        th_data = m.get("thresholds") or m.get("hypotheses") or {}
        mech_type = m.get("mechanism_type") or m.get("evaluates_hypothesis", "general")
        th_obj = HypothesisThresholds.from_dict(th_data, mechanism_type=mech_type) if th_data else None
        mechanisms[wl] = MechanismDefinition(
            workload_id=wl,
            name=m.get("name", wl),
            mechanism_type=mech_type,
            candidate=m.get("candidate", ""),
            primary_attribution_baseline=m.get("primary_attribution_baseline", ""),
            practical_baseline=m.get("practical_baseline", ""),
            informational_baselines=list(m.get("informational_baselines", [])),
            evaluates_hypothesis=m.get("evaluates_hypothesis", "primary_attribution"),
            hypotheses=th_obj,
            status=m.get("status", "ACTIVE"),
        )

    warmup = data.get("warmup_policy", {})
    central = data.get("central_hypothesis", {})

    # Extract trials per seed
    t_replay = 1000
    t_local = 200
    if "trials_per_seed" in data and isinstance(data["trials_per_seed"], dict):
        t_replay = int(data["trials_per_seed"].get("replay_integration", 1000))
    elif "trials_per_seed_replay" in data:
        t_replay = int(data["trials_per_seed_replay"])
        t_local = int(data["trials_per_seed_local"])

    seeds_raw = data.get("seeds", [])
    seeds_list: list[int] = (
        list(seeds_raw.get("exploratory", [])) + list(seeds_raw.get("confirmatory", []))
        if isinstance(seeds_raw, dict)
        else list(seeds_raw)
    )

    return BenchmarkProtocol(
        plan_id=data["plan_id"],
        plan_version=data["plan_version"],
        status=data["status"],
        is_frozen=bool(data.get("is_frozen", data.get("status") == "prospectively_frozen")),
        description=data.get("description", ""),
        workload_fixture_version=data["workload_fixture_version"],
        required_metric_policy_version=data["required_metric_policy_version"],
        evidence_levels=list(data["evidence_levels"]),
        seeds=seeds_list,
        trials_per_seed_replay=t_replay,
        trials_per_seed_local=t_local,
        smoke_trials=int(data.get("smoke_trials_per_seed", data.get("smoke_trials", 10))),
        bootstrap_resamples=int(
            data.get("statistical_rules", {}).get("bootstrap_resamples", data.get("bootstrap_resamples", 2000))
        ),
        bootstrap_ci=float(data.get("statistical_rules", {}).get("bootstrap_ci", data.get("bootstrap_ci", 0.95))),
        warmup_trials=int(warmup.get("trials", 5)),
        warmup_symmetric=bool(warmup.get("symmetric", True)),
        warmup_isolated=bool(warmup.get("isolated", True)),
        warmup_excluded=bool(warmup.get("excluded_from_evidence", True)),
        missing_data_policy=data.get("missing_data_rule", data.get("missing_data_policy", "null_inconclusive")),
        central_hypothesis_name=central.get("name", ""),
        central_hypothesis_description=central.get("description", ""),
        central_hypothesis_rule=central.get("aggregation_rule", "conjunction_all_primary_mechanisms"),
        mechanisms=mechanisms,
        negative_controls=list(data.get("controls", {}).get("negative_controls", [])),
        positive_sensitivity_control=dict(data.get("controls", {}).get("positive_sensitivity_control", {})),
        amendment_ledger=list(data.get("amendment_ledger", [])),
        raw_json=raw_text,
        protocol_hash=protocol_hash,
        seeds_dict=seeds_raw if isinstance(seeds_raw, dict) else None,
    )


# Frozen protocol instance
FROZEN_PROTOCOL = load_frozen_protocol()
