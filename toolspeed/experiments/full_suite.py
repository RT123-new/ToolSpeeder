"""Full Suite Synthetic Benchmark Runner across E1-E5 and Workloads W1-W7."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from toolspeed.core.types import (
    ArtifactManifest,
    EvidenceLevel,
    strict_json_dumps,
)
from toolspeed.experiments.e1_dag_runner import E1DAGExperiment
from toolspeed.experiments.e2_fusion_runner import E2FusionExperiment
from toolspeed.experiments.e3_spec_runner import E3SpeculationExperiment
from toolspeed.experiments.e4_commit_runner import E4CommitHorizonExperiment
from toolspeed.experiments.e5_bytecode_runner import E5BytecodeExperiment
from toolspeed.experiments.runner import (
    ExperimentResult,
    LatencyProfile,
    WorkloadFamily,
    compute_summary,
    samples,
)


@dataclass
class WorkloadBenchmarkResult:
    """Benchmark outcome for an individual canonical workload."""

    workload_id: str
    name: str
    description: str
    primary_mechanisms: list[str]
    baseline_p50_ms: float
    candidate_p50_ms: float
    p50_speedup: float
    baseline_p95_ms: float
    candidate_p95_ms: float
    p95_speedup: float
    p95_reduction_pct: float
    success_rate: float
    cost_multiplier: float
    central_hypothesis_passed: bool
    evidence_level: EvidenceLevel = EvidenceLevel.SYNTHETIC

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "name": self.name,
            "description": self.description,
            "primary_mechanisms": self.primary_mechanisms,
            "baseline_p50_ms": self.baseline_p50_ms,
            "candidate_p50_ms": self.candidate_p50_ms,
            "p50_speedup": self.p50_speedup,
            "baseline_p95_ms": self.baseline_p95_ms,
            "candidate_p95_ms": self.candidate_p95_ms,
            "p95_speedup": self.p95_speedup,
            "p95_reduction_pct": self.p95_reduction_pct,
            "success_rate": self.success_rate,
            "cost_multiplier": self.cost_multiplier,
            "central_hypothesis_passed": self.central_hypothesis_passed,
            "evidence_level": self.evidence_level.value
            if isinstance(self.evidence_level, EvidenceLevel)
            else str(self.evidence_level),
        }


@dataclass
class SuiteResult:
    """Unified container for full experiment suite and workload benchmarks."""

    experiments: dict[str, ExperimentResult]
    workloads: dict[str, WorkloadBenchmarkResult]
    profile: LatencyProfile
    trials: int
    seed: int
    total_runtime_sec: float
    central_hypothesis_passed: bool
    evidence_log: list[dict[str, str]]
    evidence_level: EvidenceLevel = EvidenceLevel.SYNTHETIC
    manifest: ArtifactManifest | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_level": self.evidence_level.value
            if isinstance(self.evidence_level, EvidenceLevel)
            else str(self.evidence_level),
            "manifest": self.manifest.to_dict() if self.manifest else None,
            "experiments": {k: v.to_dict() for k, v in self.experiments.items()},
            "workloads": {k: v.to_dict() for k, v in self.workloads.items()},
            "profile": self.profile.to_dict(),
            "trials": self.trials,
            "seed": self.seed,
            "total_runtime_sec": self.total_runtime_sec,
            "central_hypothesis_passed": self.central_hypothesis_passed,
            "evidence_log": self.evidence_log,
        }

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(strict_json_dumps(self.to_dict(), indent=2), encoding="utf-8")

    def save_csvs(self, directory: str | Path) -> list[Path]:
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for exp_id, exp_res in self.experiments.items():
            csv_path = out_dir / f"{exp_id.lower()}_results.csv"
            if exp_res.rows:
                keys = list(exp_res.rows[0].keys())
                lines = [",".join(keys)]
                for r in exp_res.rows:
                    row_vals = [str(r.get(k, "")) for k in keys]
                    lines.append(",".join(row_vals))
                csv_path.write_text("\n".join(lines), encoding="utf-8")
                saved.append(csv_path)

        # Save workload summary CSV
        wl_csv = out_dir / "workload_summary.csv"
        wl_keys = [
            "workload_id",
            "name",
            "primary_mechanisms",
            "baseline_p50_ms",
            "candidate_p50_ms",
            "p50_speedup",
            "baseline_p95_ms",
            "candidate_p95_ms",
            "p95_speedup",
            "p95_reduction_pct",
            "success_rate",
            "central_hypothesis_passed",
            "evidence_level",
        ]
        wl_lines = [",".join(wl_keys)]
        for w in self.workloads.values():
            b50 = f"{w.baseline_p50_ms:.2f}" if w.baseline_p50_ms is not None else ""
            c50 = f"{w.candidate_p50_ms:.2f}" if w.candidate_p50_ms is not None else ""
            sp50 = f"{w.p50_speedup:.3f}" if w.p50_speedup is not None else ""
            b95 = f"{w.baseline_p95_ms:.2f}" if w.baseline_p95_ms is not None else ""
            c95 = f"{w.candidate_p95_ms:.2f}" if w.candidate_p95_ms is not None else ""
            sp95 = f"{w.p95_speedup:.3f}" if w.p95_speedup is not None else ""
            red95 = f"{w.p95_reduction_pct:.2f}" if w.p95_reduction_pct is not None else ""
            succ = f"{w.success_rate:.3f}" if w.success_rate is not None else "1.000"
            wl_lines.append(
                ",".join(
                    [
                        w.workload_id,
                        f'"{w.name}"',
                        f'"{"; ".join(w.primary_mechanisms)}"',
                        b50,
                        c50,
                        sp50,
                        b95,
                        c95,
                        sp95,
                        red95,
                        succ,
                        str(w.central_hypothesis_passed),
                        w.evidence_level.value
                        if isinstance(w.evidence_level, EvidenceLevel)
                        else str(w.evidence_level),
                    ]
                )
            )
        wl_csv.write_text("\n".join(wl_lines), encoding="utf-8")
        saved.append(wl_csv)
        return saved


class SuiteRunner:
    """Orchestrates multi-trial synthetic simulation across E1-E5 and W1-W7."""

    def __init__(
        self,
        profile: LatencyProfile | None = None,
        trials: int = 10_000,
        seed: int = 20260825,
        evidence_level: EvidenceLevel = EvidenceLevel.SYNTHETIC,
    ) -> None:
        self.profile = profile or LatencyProfile()
        self.trials = max(100, trials)
        self.seed = seed
        self.evidence_level = evidence_level

    def run_workload_benchmarks(self) -> dict[str, WorkloadBenchmarkResult]:
        """Execute analytical statistical simulation for each workload family W1-W7."""
        results: dict[str, WorkloadBenchmarkResult] = {}
        n = self.trials

        # W1: Independent fan-out reads (4 parallel reads + DAG scheduler)
        rng1 = np.random.default_rng(self.seed + 10001)
        dec1 = samples(rng1, self.profile.model_decision_ms, self.profile.sigma, n)
        fin1 = samples(rng1, self.profile.model_final_ms, self.profile.sigma, n)
        tools1 = samples(rng1, self.profile.tool_ms, self.profile.sigma, (n, 4))
        w1_base = dec1 + tools1.sum(axis=1) + fin1
        w1_cand = dec1 + tools1.max(axis=1) + fin1
        w1_s = compute_summary(w1_base, w1_cand)
        w1_red = (w1_s.baseline_p95_ms - w1_s.candidate_p95_ms) / max(1.0, w1_s.baseline_p95_ms) * 100.0
        results["W1"] = WorkloadBenchmarkResult(
            workload_id="W1",
            name=WorkloadFamily.W1_FANOUT.value,
            description="4 independent read tool calls executed concurrently vs sequentially",
            primary_mechanisms=["DAG Scheduler", "Parallel Fan-Out"],
            baseline_p50_ms=w1_s.baseline_p50_ms,
            candidate_p50_ms=w1_s.candidate_p50_ms,
            p50_speedup=w1_s.p50_speedup,
            baseline_p95_ms=w1_s.baseline_p95_ms,
            candidate_p95_ms=w1_s.candidate_p95_ms,
            p95_speedup=w1_s.p95_speedup,
            p95_reduction_pct=w1_red,
            success_rate=w1_s.candidate_success_rate,
            cost_multiplier=w1_s.cost_multiplier,
            central_hypothesis_passed=w1_red >= 10.0,
            evidence_level=self.evidence_level,
        )

        # W2: Deterministic dependent chains (4 steps compiled into fused runtime)
        rng2 = np.random.default_rng(self.seed + 10002)
        decs2 = samples(rng2, self.profile.model_decision_ms, self.profile.sigma, (n, 4))
        fin2 = samples(rng2, self.profile.model_final_ms, self.profile.sigma, n)
        tools2 = samples(rng2, self.profile.tool_ms, self.profile.sigma, (n, 4))
        over2 = samples(rng2, self.profile.program_runtime_overhead_ms, self.profile.sigma / 2.0, n)
        w2_base = decs2.sum(axis=1) + tools2.sum(axis=1) + fin2
        w2_cand = decs2[:, 0] + over2 + tools2.sum(axis=1) + fin2
        w2_s = compute_summary(w2_base, w2_cand)
        w2_red = (w2_s.baseline_p95_ms - w2_s.candidate_p95_ms) / max(1.0, w2_s.baseline_p95_ms) * 100.0
        results["W2"] = WorkloadBenchmarkResult(
            workload_id="W2",
            name=WorkloadFamily.W2_CHAINS.value,
            description="4-step sequential dependent tool chain fused into compiled runtime",
            primary_mechanisms=["JIT Workflow Fusion", "Model Hop Elimination"],
            baseline_p50_ms=w2_s.baseline_p50_ms,
            candidate_p50_ms=w2_s.candidate_p50_ms,
            p50_speedup=w2_s.p50_speedup,
            baseline_p95_ms=w2_s.baseline_p95_ms,
            candidate_p95_ms=w2_s.candidate_p95_ms,
            p95_speedup=w2_s.p95_speedup,
            p95_reduction_pct=w2_red,
            success_rate=w2_s.candidate_success_rate,
            cost_multiplier=w2_s.cost_multiplier,
            central_hypothesis_passed=w2_red >= 10.0,
            evidence_level=self.evidence_level,
        )

        # W3: Branching workflows
        rng3 = np.random.default_rng(self.seed + 10003)
        dec3 = samples(rng3, self.profile.model_decision_ms, self.profile.sigma, n)
        fin3 = samples(rng3, self.profile.model_final_ms, self.profile.sigma, n)
        tool3 = samples(rng3, self.profile.tool_ms, self.profile.sigma, n)
        draft3 = samples(rng3, self.profile.draft_model_ms, self.profile.sigma / 2.0, n)
        acc3 = 0.88
        correct3 = rng3.random(n) < acc3
        w3_base = dec3 + tool3 + fin3
        w3_cand = np.where(correct3, np.maximum(dec3, draft3 + tool3), dec3 + tool3) + fin3
        w3_s = compute_summary(w3_base, w3_cand)
        w3_red = (w3_s.baseline_p95_ms - w3_s.candidate_p95_ms) / max(1.0, w3_s.baseline_p95_ms) * 100.0
        results["W3"] = WorkloadBenchmarkResult(
            workload_id="W3",
            name=WorkloadFamily.W3_BRANCHING.value,
            description="Branching decision path with speculative pre-execution of dominant branch",
            primary_mechanisms=["Confidence-Gated Speculation", "Branch Prediction"],
            baseline_p50_ms=w3_s.baseline_p50_ms,
            candidate_p50_ms=w3_s.candidate_p50_ms,
            p50_speedup=w3_s.p50_speedup,
            baseline_p95_ms=w3_s.baseline_p95_ms,
            candidate_p95_ms=w3_s.candidate_p95_ms,
            p95_speedup=w3_s.p95_speedup,
            p95_reduction_pct=w3_red,
            success_rate=w3_s.candidate_success_rate,
            cost_multiplier=w3_s.cost_multiplier,
            central_hypothesis_passed=w3_red >= 10.0,
            evidence_level=self.evidence_level,
        )

        # W4: Repeated workflows with plan locality
        rng4 = np.random.default_rng(self.seed + 10004)
        dec4 = samples(rng4, self.profile.model_decision_ms, self.profile.sigma, n)
        fin4 = samples(rng4, self.profile.model_final_ms, self.profile.sigma, n)
        tools4 = samples(rng4, self.profile.tool_ms, self.profile.sigma, (n, 3))
        w4_base = dec4 * 3 + tools4.sum(axis=1) + fin4
        cache_hit_latency = 8.0
        w4_cand = cache_hit_latency + tools4.sum(axis=1) + fin4
        w4_s = compute_summary(w4_base, w4_cand)
        w4_red = (w4_s.baseline_p95_ms - w4_s.candidate_p95_ms) / max(1.0, w4_s.baseline_p95_ms) * 100.0
        results["W4"] = WorkloadBenchmarkResult(
            workload_id="W4",
            name=WorkloadFamily.W4_REPEATED.value,
            description="Repeated parameterized task reusing cached compiled plan AST",
            primary_mechanisms=["Plan Locality Cache", "JIT AST Reuse"],
            baseline_p50_ms=w4_s.baseline_p50_ms,
            candidate_p50_ms=w4_s.candidate_p50_ms,
            p50_speedup=w4_s.p50_speedup,
            baseline_p95_ms=w4_s.baseline_p95_ms,
            candidate_p95_ms=w4_s.candidate_p95_ms,
            p95_speedup=w4_s.p95_speedup,
            p95_reduction_pct=w4_red,
            success_rate=w4_s.candidate_success_rate,
            cost_multiplier=w4_s.cost_multiplier,
            central_hypothesis_passed=w4_red >= 10.0,
            evidence_level=self.evidence_level,
        )

        # W5: Large tool arguments and results
        rng5 = np.random.default_rng(self.seed + 10005)
        gen5 = samples(rng5, 800.0, self.profile.sigma, n)
        tool5 = samples(rng5, self.profile.tool_ms, self.profile.sigma, n)
        fin5 = samples(rng5, self.profile.model_final_ms, self.profile.sigma, n)
        w5_base = gen5 + tool5 + fin5
        cand_gen5 = (0.4 * gen5) + (0.6 * gen5 / 3.5)
        w5_cand = np.maximum(cand_gen5, 0.4 * gen5 + tool5) + fin5
        w5_s = compute_summary(w5_base, w5_cand)
        w5_red = (w5_s.baseline_p95_ms - w5_s.candidate_p95_ms) / max(1.0, w5_s.baseline_p95_ms) * 100.0
        results["W5"] = WorkloadBenchmarkResult(
            workload_id="W5",
            name=WorkloadFamily.W5_LARGE_PAYLOADS.value,
            description="Large JSON payload generation optimized via Action Bytecode + Commit Horizon",
            primary_mechanisms=["Action Bytecode", "Commit-Horizon Dispatch"],
            baseline_p50_ms=w5_s.baseline_p50_ms,
            candidate_p50_ms=w5_s.candidate_p50_ms,
            p50_speedup=w5_s.p50_speedup,
            baseline_p95_ms=w5_s.baseline_p95_ms,
            candidate_p95_ms=w5_s.candidate_p95_ms,
            p95_speedup=w5_s.p95_speedup,
            p95_reduction_pct=w5_red,
            success_rate=w5_s.candidate_success_rate,
            cost_multiplier=w5_s.cost_multiplier,
            central_hypothesis_passed=w5_red >= 10.0,
            evidence_level=self.evidence_level,
        )

        # W6: Cold-start code/browser sandboxes
        rng6 = np.random.default_rng(self.seed + 10006)
        dec6 = samples(rng6, self.profile.model_decision_ms, self.profile.sigma, n)
        cold_start6 = samples(rng6, 950.0, self.profile.sigma, n)
        warm_start6 = samples(rng6, 20.0, self.profile.sigma / 2.0, n)
        tool6 = samples(rng6, self.profile.tool_ms, self.profile.sigma, n)
        fin6 = samples(rng6, self.profile.model_final_ms, self.profile.sigma, n)
        w6_base = dec6 + cold_start6 + tool6 + fin6
        w6_cand = dec6 + warm_start6 + tool6 + fin6
        w6_s = compute_summary(w6_base, w6_cand)
        w6_red = (w6_s.baseline_p95_ms - w6_s.candidate_p95_ms) / max(1.0, w6_s.baseline_p95_ms) * 100.0
        results["W6"] = WorkloadBenchmarkResult(
            workload_id="W6",
            name=WorkloadFamily.W6_SANDBOX_COLDSTART.value,
            description="Cold-start sandbox execution mitigated by predictive sandbox prewarming pool",
            primary_mechanisms=["Predictive Sandbox Prewarming", "Warm Pool Recycling"],
            baseline_p50_ms=w6_s.baseline_p50_ms,
            candidate_p50_ms=w6_s.candidate_p50_ms,
            p50_speedup=w6_s.p50_speedup,
            baseline_p95_ms=w6_s.baseline_p95_ms,
            candidate_p95_ms=w6_s.candidate_p95_ms,
            p95_speedup=w6_s.p95_speedup,
            p95_reduction_pct=w6_red,
            success_rate=w6_s.candidate_success_rate,
            cost_multiplier=w6_s.cost_multiplier,
            central_hypothesis_passed=w6_red >= 10.0,
            evidence_level=self.evidence_level,
        )

        # W7: Side-effecting actions requiring approval
        rng7 = np.random.default_rng(self.seed + 10007)
        dec7 = samples(rng7, self.profile.model_decision_ms, self.profile.sigma, n)
        tool7 = samples(rng7, self.profile.tool_ms, self.profile.sigma, n)
        fin7 = samples(rng7, self.profile.model_final_ms, self.profile.sigma, n)
        w7_base = dec7 + tool7 + fin7
        w7_cand = np.maximum(dec7, 0.45 * dec7 + tool7) + fin7
        w7_s = compute_summary(w7_base, w7_cand)
        w7_red = (w7_s.baseline_p95_ms - w7_s.candidate_p95_ms) / max(1.0, w7_s.baseline_p95_ms) * 100.0
        results["W7"] = WorkloadBenchmarkResult(
            workload_id="W7",
            name=WorkloadFamily.W7_SIDE_EFFECTS.value,
            description="Side-effecting tool action with idempotency token and safe commit horizon",
            primary_mechanisms=["Idempotent Commit Horizon", "Side-Effect Safety Gate"],
            baseline_p50_ms=w7_s.baseline_p50_ms,
            candidate_p50_ms=w7_s.candidate_p50_ms,
            p50_speedup=w7_s.p50_speedup,
            baseline_p95_ms=w7_s.baseline_p95_ms,
            candidate_p95_ms=w7_s.candidate_p95_ms,
            p95_speedup=w7_s.p95_speedup,
            p95_reduction_pct=w7_red,
            success_rate=w7_s.candidate_success_rate,
            cost_multiplier=w7_s.cost_multiplier,
            central_hypothesis_passed=w7_red >= 10.0,
            evidence_level=self.evidence_level,
        )

        return results

    def run(self) -> SuiteResult:
        """Run analytical simulation for E1-E5 and W1-W7."""
        start_time = time.perf_counter()

        e1 = E1DAGExperiment(profile=self.profile, trials=self.trials, seed=self.seed).run()
        e2 = E2FusionExperiment(profile=self.profile, trials=self.trials, seed=self.seed).run()
        e3 = E3SpeculationExperiment(profile=self.profile, trials=self.trials, seed=self.seed).run()
        e4 = E4CommitHorizonExperiment(profile=self.profile, trials=self.trials, seed=self.seed).run()
        e5 = E5BytecodeExperiment(profile=self.profile, trials=self.trials, seed=self.seed).run()

        experiments = {
            "E1": e1,
            "E2": e2,
            "E3": e3,
            "E4": e4,
            "E5": e5,
        }

        workloads = self.run_workload_benchmarks()
        central_passed = all(w.central_hypothesis_passed for w in workloads.values())

        evidence_log = [
            {
                "experiment": "Synthetic Model Simulator",
                "tested": "Yes",
                "succeeded": "Analytical mechanisms conform to declared distributions",
                "failed": "Synthetic assumption only (not empirical proof)",
                "still_unproven": "Real OS and remote network I/O",
                "next_action": "Execute real benchmark harness on Replay and Local backends",
            },
            e1.verdict.evidence_log_row,
            e2.verdict.evidence_log_row,
            e3.verdict.evidence_log_row,
            e4.verdict.evidence_log_row,
            e5.verdict.evidence_log_row,
        ]

        manifest = ArtifactManifest.create(evidence_level=self.evidence_level, seed=self.seed)
        runtime = time.perf_counter() - start_time

        return SuiteResult(
            experiments=experiments,
            workloads=workloads,
            profile=self.profile,
            trials=self.trials,
            seed=self.seed,
            total_runtime_sec=runtime,
            central_hypothesis_passed=central_passed,
            evidence_log=evidence_log,
            evidence_level=self.evidence_level,
            manifest=manifest,
        )


def run_full_suite(
    profile: LatencyProfile | None = None,
    trials: int = 10_000,
    seed: int = 20260825,
) -> SuiteResult:
    return SuiteRunner(profile=profile, trials=trials, seed=seed).run()
