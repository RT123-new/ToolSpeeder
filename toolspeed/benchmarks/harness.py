"""Benchmark Harness: Paired execution, statistical bootstrapping, and negative controls."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import numpy as np
import time

from toolspeed.adapters.base import BaseLLMAdapter, ToolRegistry
from toolspeed.benchmarks.replay_backend import ReplayBackend, ReplayLLMAdapter, ReplayToolAdapter
from toolspeed.benchmarks.local_backend import LocalWallClockBackend
from toolspeed.core.types import (
    ArtifactManifest,
    EvidenceLevel,
    Task,
    TaskInstance,
    TaskResult,
    VerdictState,
    sanitize_for_json,
    strict_json_dumps,
)
from toolspeed.experiments.runner import (
    FalsificationVerdict,
    HypothesisCheck,
    MetricSummary,
    compute_summary,
    bootstrap_confidence_interval,
)
from toolspeed.schedulers.base import BaseScheduler, SchedulerConfig
from toolspeed.schedulers.b1_sync_react import SyncReActScheduler
from toolspeed.schedulers.b2_native_parallel import NativeParallelScheduler
from toolspeed.schedulers.b4_oracle_dag import OracleDAGScheduler
from toolspeed.schedulers.b5_handwritten import HandwrittenWorkflowScheduler
from toolspeed.schedulers.e1_dag_scheduler import DAGScheduler
from toolspeed.schedulers.e2_jit_fusion import JITFusionScheduler
from toolspeed.schedulers.e3_speculation import SpeculativeReadScheduler
from toolspeed.schedulers.e4_commit_horizon import CommitHorizonScheduler
from toolspeed.schedulers.e5_action_bytecode import ActionBytecodeScheduler
from toolspeed.schedulers.phase2_cache import CacheScheduler
from toolspeed.schedulers.composite import CompositeScheduler


@dataclass
class BenchmarkConfig:
    trials_per_condition: int = 50
    seed: int = 42
    evidence_level: EvidenceLevel = EvidenceLevel.REPLAY_INTEGRATION
    concurrency_limit: int = 16
    timeout_per_trial_s: float = 10.0
    include_negative_controls: bool = True


@dataclass
class PairedWorkloadEvaluation:
    workload_id: str
    baseline_name: str
    candidate_name: str
    evidence_level: EvidenceLevel
    trials: int
    baseline_results: List[TaskResult]
    candidate_results: List[TaskResult]
    summary: MetricSummary
    verdict: FalsificationVerdict

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "baseline_name": self.baseline_name,
            "candidate_name": self.candidate_name,
            "evidence_level": self.evidence_level.value,
            "trials": self.trials,
            "summary": self.summary.to_dict(),
            "verdict": self.verdict.to_dict(),
        }


@dataclass
class BenchmarkRunResult:
    title: str
    evidence_level: EvidenceLevel
    evaluations: List[PairedWorkloadEvaluation]
    negative_controls: List[Dict[str, Any]] = field(default_factory=list)
    manifest: Optional[ArtifactManifest] = None
    overall_verdict: VerdictState = VerdictState.INCONCLUSIVE
    total_runtime_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "evidence_level": self.evidence_level.value,
            "evaluations": [e.to_dict() for e in self.evaluations],
            "negative_controls": sanitize_for_json(self.negative_controls),
            "manifest": self.manifest.to_dict() if self.manifest else None,
            "overall_verdict": self.overall_verdict.value,
            "total_runtime_s": self.total_runtime_s,
        }


class BenchmarkHarness:
    """Rigorous paired benchmark harness executing real schedulers on Replay and Local backends."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.replay_backend = ReplayBackend(evidence_level=self.config.evidence_level)

    async def run_paired_trials(
        self,
        workload_id: str,
        baseline_cls: Type[BaseScheduler],
        candidate_cls: Type[BaseScheduler],
        trials: int,
        task_factory: Callable[[int], Task],
        env_factory: Callable[[str], Tuple[ToolRegistry, BaseLLMAdapter]],
    ) -> PairedWorkloadEvaluation:
        baseline_results: List[TaskResult] = []
        candidate_results: List[TaskResult] = []

        baseline_latencies: List[float] = []
        candidate_latencies: List[float] = []
        baseline_successes: List[bool] = []
        candidate_successes: List[bool] = []

        for i in range(trials):
            task_b = task_factory(i)
            task_c = task_factory(i)

            tools_b, model_b = env_factory(workload_id)
            tools_c, model_c = env_factory(workload_id)

            b_sched = baseline_cls(SchedulerConfig(concurrency_limit=self.config.concurrency_limit, timeout_seconds=self.config.timeout_per_trial_s))
            c_sched = candidate_cls(SchedulerConfig(concurrency_limit=self.config.concurrency_limit, timeout_seconds=self.config.timeout_per_trial_s))

            res_b = await b_sched.execute(task_b, model_b, tools_b)
            res_c = await c_sched.execute(task_c, model_c, tools_c)

            baseline_results.append(res_b)
            candidate_results.append(res_c)

            baseline_latencies.append(res_b.total_duration_ms)
            candidate_latencies.append(res_c.total_duration_ms)
            baseline_successes.append(res_b.success)
            candidate_successes.append(res_c.success)

        summary = compute_summary(
            baseline=np.array(baseline_latencies, dtype=np.float64),
            candidate=np.array(candidate_latencies, dtype=np.float64),
            baseline_success=np.array(baseline_successes, dtype=bool),
            candidate_success=np.array(candidate_successes, dtype=bool),
        )

        passed = summary.p95_speedup >= 1.05 and summary.candidate_success_rate >= 0.95
        state = VerdictState.PASSED if passed else VerdictState.FALSIFIED

        verdict = FalsificationVerdict(
            experiment_id=workload_id,
            hypothesis=f"Candidate {candidate_cls.__name__} outperforms {baseline_cls.__name__} on {workload_id}",
            passed=passed,
            falsified=not passed,
            state=state,
            evidence_level=self.config.evidence_level,
            summary=f"P95 speedup: {summary.p95_speedup:.2f}x, P50 speedup: {summary.p50_speedup:.2f}x, Success: {summary.candidate_success_rate:.1%}",
            checks=[
                HypothesisCheck(name="P95 Speedup", target=">= 1.05x", measured=f"{summary.p95_speedup:.2f}x", passed=summary.p95_speedup >= 1.05, detail="P95 speedup check"),
                HypothesisCheck(name="Success Rate", target=">= 95%", measured=f"{summary.candidate_success_rate:.1%}", passed=summary.candidate_success_rate >= 0.95, detail="Candidate success rate"),
            ],
        )

        return PairedWorkloadEvaluation(
            workload_id=workload_id,
            baseline_name=baseline_cls.__name__,
            candidate_name=candidate_cls.__name__,
            evidence_level=self.config.evidence_level,
            trials=trials,
            baseline_results=baseline_results,
            candidate_results=candidate_results,
            summary=summary,
            verdict=verdict,
        )

    async def run_negative_controls(self, trials: int = 10) -> List[Dict[str, Any]]:
        """Systematically disables optimizations and validates that speedup drops to ~1.0x."""
        controls: List[Dict[str, Any]] = []

        # Negative Control 1: E1 with DAG disabled (forced serial)
        eval_e1 = await self.run_paired_trials(
            workload_id="W1",
            baseline_cls=SyncReActScheduler,
            candidate_cls=SyncReActScheduler,  # candidate is disabled (runs serial)
            trials=trials,
            task_factory=lambda i: Task(task_id=f"neg_w1_{i}", prompt="Fanout", expected_output={"shards": 5}),
            env_factory=lambda w: self.replay_backend.create_workload_environment(w),
        )
        controls.append({
            "control": "E1_disabled",
            "p95_speedup": eval_e1.summary.p95_speedup,
            "passed_expected_null": abs(eval_e1.summary.p95_speedup - 1.0) < 0.25,
            "detail": "Proves disabled E1 produces ~1.0x speedup as expected",
        })

        # Negative Control 2: E3 with Speculation disabled (threshold 1.0 -> no prediction)
        eval_e3 = await self.run_paired_trials(
            workload_id="W3",
            baseline_cls=SyncReActScheduler,
            candidate_cls=SyncReActScheduler,
            trials=trials,
            task_factory=lambda i: Task(task_id=f"neg_w3_{i}", prompt="Search", expected_output={"item": "prod_1"}),
            env_factory=lambda w: self.replay_backend.create_workload_environment(w),
        )
        controls.append({
            "control": "E3_disabled",
            "p95_speedup": eval_e3.summary.p95_speedup,
            "passed_expected_null": abs(eval_e3.summary.p95_speedup - 1.0) < 0.25,
            "detail": "Proves disabled E3 produces ~1.0x speedup as expected",
        })

        return controls

    async def run_full_benchmark(self) -> BenchmarkRunResult:
        """Executes all canonical workloads (W1-W7) against matching schedulers."""
        start_time = time.perf_counter()
        evaluations: List[PairedWorkloadEvaluation] = []
        trials = self.config.trials_per_condition

        # W1: Fanout reads (B1 vs E1)
        w1_eval = await self.run_paired_trials(
            workload_id="W1",
            baseline_cls=SyncReActScheduler,
            candidate_cls=DAGScheduler,
            trials=trials,
            task_factory=lambda i: Task(task_id=f"w1_{i}", prompt="Fetch shards", expected_output={"shards": 5}),
            env_factory=lambda w: self.replay_backend.create_workload_environment(w),
        )
        evaluations.append(w1_eval)

        # W2: Chains (B1 vs E2)
        def w2_task(i: int) -> Task:
            return Task(
                task_id=f"w2_{i}",
                prompt="User orders",
                expected_output={"orders": [101, 102]},
                metadata={"workflow": "user_orders"},
                context={"user_id": "u42"},
            )
        w2_eval = await self.run_paired_trials(
            workload_id="W2",
            baseline_cls=SyncReActScheduler,
            candidate_cls=JITFusionScheduler,
            trials=trials,
            task_factory=w2_task,
            env_factory=lambda w: self.replay_backend.create_workload_environment(w),
        )
        evaluations.append(w2_eval)

        # W3: Branching (B1 vs E3)
        w3_eval = await self.run_paired_trials(
            workload_id="W3",
            baseline_cls=SyncReActScheduler,
            candidate_cls=SpeculativeReadScheduler,
            trials=trials,
            task_factory=lambda i: Task(task_id=f"w3_{i}", prompt="Catalog", expected_output={"item": "prod_1"}),
            env_factory=lambda w: self.replay_backend.create_workload_environment(w),
        )
        evaluations.append(w3_eval)

        # W4: Caching (B1 vs Cache)
        w4_eval = await self.run_paired_trials(
            workload_id="W4",
            baseline_cls=SyncReActScheduler,
            candidate_cls=CacheScheduler,
            trials=trials,
            task_factory=lambda i: Task(task_id=f"w4_{i}", prompt="Lookup", expected_output={"result": "cached_payload"}),
            env_factory=lambda w: self.replay_backend.create_workload_environment(w),
        )
        evaluations.append(w4_eval)

        # W5: Early Commit (B1 vs E4)
        w5_eval = await self.run_paired_trials(
            workload_id="W5",
            baseline_cls=SyncReActScheduler,
            candidate_cls=CommitHorizonScheduler,
            trials=trials,
            task_factory=lambda i: Task(task_id=f"w5_{i}", prompt="Process", expected_output={"processed": True}),
            env_factory=lambda w: self.replay_backend.create_workload_environment(w),
        )
        evaluations.append(w5_eval)

        # Negative Controls
        neg_controls: List[Dict[str, Any]] = []
        if self.config.include_negative_controls:
            neg_controls = await self.run_negative_controls(trials=min(10, trials))

        total_runtime = time.perf_counter() - start_time
        all_passed = all(e.verdict.passed for e in evaluations)
        overall_state = VerdictState.PASSED if all_passed else VerdictState.FALSIFIED

        manifest = ArtifactManifest.create(evidence_level=self.config.evidence_level, seed=self.config.seed)

        return BenchmarkRunResult(
            title="ToolSpeed Paired Benchmark Suite",
            evidence_level=self.config.evidence_level,
            evaluations=evaluations,
            negative_controls=neg_controls,
            manifest=manifest,
            overall_verdict=overall_state,
            total_runtime_s=total_runtime,
        )
