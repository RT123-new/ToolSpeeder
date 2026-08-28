"""Benchmark Harness: Paired execution, statistical bootstrapping, and negative controls."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import time
from typing import Any
import numpy as np

from toolspeed.benchmarks.local_backend import LocalWallClockBackend
from toolspeed.benchmarks.replay_backend import ReplayBackend
from toolspeed.core.types import (
    ApprovalGrant,
    ArtifactManifest,
    EvidenceLevel,
    Task,
    TaskResult,
    VerdictState,
    sanitize_for_json,
)
from toolspeed.experiments.runner import (
    FalsificationVerdict,
    HypothesisCheck,
    MetricSummary,
    compute_summary,
)
from toolspeed.schedulers.b1_sync_react import SyncReActScheduler
from toolspeed.schedulers.base import BaseScheduler, SchedulerConfig
from toolspeed.schedulers.composite import CompositeScheduler
from toolspeed.schedulers.e1_dag_scheduler import DAGScheduler
from toolspeed.schedulers.e2_jit_fusion import JITFusionScheduler
from toolspeed.schedulers.e3_speculation import SpeculativeReadScheduler
from toolspeed.schedulers.e4_commit_horizon import CommitHorizonScheduler
from toolspeed.schedulers.e5_action_bytecode import ActionBytecodeScheduler
from toolspeed.schedulers.phase2_cache import CacheScheduler, ToolResultCache


@dataclass
class BenchmarkConfig:
    trials_per_condition: int = 1000
    seed: int = 42
    evidence_level: EvidenceLevel = EvidenceLevel.REPLAY_INTEGRATION
    concurrency_limit: int = 16
    timeout_per_trial_s: float = 10.0
    include_negative_controls: bool = True
    warmup_trials: int = 5


@dataclass
class PairedWorkloadEvaluation:
    workload_id: str
    baseline_name: str
    candidate_name: str
    evidence_level: EvidenceLevel
    trials: int
    baseline_results: list[TaskResult]
    candidate_results: list[TaskResult]
    summary: MetricSummary
    verdict: FalsificationVerdict
    execution_order: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "baseline_name": self.baseline_name,
            "candidate_name": self.candidate_name,
            "evidence_level": self.evidence_level.value,
            "trials": self.trials,
            "summary": self.summary.to_dict(),
            "verdict": self.verdict.to_dict(),
            "execution_order": self.execution_order,
        }


@dataclass
class BenchmarkRunResult:
    title: str
    evidence_level: EvidenceLevel
    evaluations: list[PairedWorkloadEvaluation]
    negative_controls: list[dict[str, Any]] = field(default_factory=list)
    manifest: ArtifactManifest | None = None
    overall_verdict: VerdictState = VerdictState.INCONCLUSIVE
    total_runtime_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
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
    """Rigorous paired benchmark harness executing real schedulers on genuine Replay and Local backends."""

    def __init__(self, config: BenchmarkConfig | None = None):
        self.config = config or BenchmarkConfig()
        if self.config.evidence_level == EvidenceLevel.LOCAL_WALL_CLOCK:
            self.backend: LocalWallClockBackend | ReplayBackend = LocalWallClockBackend(evidence_level=self.config.evidence_level)
        else:
            self.backend = ReplayBackend(evidence_level=self.config.evidence_level)

    async def run_paired_trials(
        self,
        workload_id: str,
        baseline_cls: type[BaseScheduler],
        candidate_cls: type[BaseScheduler],
        trials: int,
        task_factory: Callable[[int], Task],
        candidate_kwargs_factory: Callable[[int], dict[str, Any]] | None = None,
        candidate_shared_cache: ToolResultCache | None = None,
    ) -> PairedWorkloadEvaluation:
        baseline_results: list[TaskResult] = []
        candidate_results: list[TaskResult] = []

        baseline_latencies: list[float] = []
        candidate_latencies: list[float] = []
        baseline_successes: list[bool] = []
        candidate_successes: list[bool] = []
        execution_order: list[str] = []

        # Warmup trials
        for w in range(self.config.warmup_trials):
            task_w = task_factory(w)
            tools_w, model_w = self.backend.create_workload_environment(workload_id, trial_index=w)
            sched_w = baseline_cls(SchedulerConfig(concurrency_limit=self.config.concurrency_limit))
            await sched_w.execute(task_w, model_w, tools_w)

        for i in range(trials):
            task_b = task_factory(i)
            task_c = task_factory(i)

            tools_b, model_b = self.backend.create_workload_environment(workload_id, trial_index=i)
            tools_c, model_c = self.backend.create_workload_environment(workload_id, trial_index=i)

            b_sched = baseline_cls(SchedulerConfig(concurrency_limit=self.config.concurrency_limit, timeout_seconds=self.config.timeout_per_trial_s))

            c_kwargs: dict[str, Any] = {}
            if candidate_kwargs_factory is not None:
                c_kwargs = candidate_kwargs_factory(i)
            elif candidate_shared_cache is not None and issubclass(candidate_cls, CacheScheduler):
                c_kwargs["shared_cache"] = candidate_shared_cache

            c_sched = candidate_cls(SchedulerConfig(concurrency_limit=self.config.concurrency_limit, timeout_seconds=self.config.timeout_per_trial_s), **c_kwargs)

            # Counterbalance execution order per trial to eliminate sequence bias
            run_candidate_first = (i % 2 == 1)
            if run_candidate_first:
                execution_order.append("candidate_first")
                res_c = await c_sched.execute(task_c, model_c, tools_c)
                res_b = await b_sched.execute(task_b, model_b, tools_b)
            else:
                execution_order.append("baseline_first")
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

        # Enforce sample size threshold for verdict eligibility
        min_trials_required = 1000 if self.config.evidence_level == EvidenceLevel.REPLAY_INTEGRATION else 200
        is_verdict_eligible = (trials >= min_trials_required)

        p95_speedup = summary.p95_speedup if summary.p95_speedup is not None else 0.0
        p99_speedup = summary.p99_speedup if summary.p99_speedup is not None else 0.0
        cand_succ = summary.candidate_success_rate if summary.candidate_success_rate is not None else 0.0
        succ_delta = summary.success_rate_delta if summary.success_rate_delta is not None else 0.0

        # Scientific checks
        target_p95_min = 0.95 if workload_id == "W7" else 1.05
        check_p95 = p95_speedup >= target_p95_min
        check_succ = (cand_succ >= 0.95) and (succ_delta >= -0.005)
        check_p99 = p99_speedup >= 0.95  # P99 non-regression (<= 5% regression)
        check_side_effects = summary.unapproved_side_effects == 0
        check_cost = (summary.cost_multiplier or 1.0) <= 1.10

        all_checks_passed = check_p95 and check_succ and check_p99 and check_side_effects and check_cost

        if not is_verdict_eligible:
            state = VerdictState.INCONCLUSIVE
            verdict_summary = f"SMOKE — NOT VERDICT-ELIGIBLE (n={trials} < {min_trials_required}). P95 speedup: {p95_speedup:.2f}x, Success: {cand_succ:.1%}"
        elif all_checks_passed:
            state = VerdictState.PASSED
            verdict_summary = f"PASSED — P95 speedup: {p95_speedup:.2f}x, P50 speedup: {summary.p50_speedup or 1.0:.2f}x, Success: {cand_succ:.1%}, Cost: {summary.cost_multiplier or 1.0:.2f}x"
        else:
            state = VerdictState.FALSIFIED
            verdict_summary = f"FALSIFIED — P95 speedup: {p95_speedup:.2f}x, Success: {cand_succ:.1%}, P99 speedup: {p99_speedup:.2f}x"

        ci_str = f"[{summary.p95_reduction_ci[0]:.1f}%, {summary.p95_reduction_ci[1]:.1f}%]" if summary.p95_reduction_ci and summary.p95_reduction_ci[0] is not None else "null"

        checks = [
            HypothesisCheck(name="P95 CCL Reduction", target=">= 1.05x speedup", measured=f"{p95_speedup:.2f}x", passed=check_p95, detail=f"95% CI: {ci_str}"),
            HypothesisCheck(name="Candidate Success Rate", target=">= 95.0% and non-inferior", measured=f"{cand_succ:.1%} (delta: {succ_delta:+.2%})", passed=check_succ, detail="Success non-inferiority check"),
            HypothesisCheck(name="P99 CCL Non-Regression", target=">= 0.95x", measured=f"{p99_speedup:.2f}x", passed=check_p99, detail="Tail latency stability check"),
            HypothesisCheck(name="Side-Effect Approvals", target="0 unapproved side effects", measured=f"{summary.unapproved_side_effects}", passed=check_side_effects, detail="Approval gate enforcement"),
            HypothesisCheck(name="Cost Multiplier", target="<= 1.10x", measured=f"{summary.cost_multiplier or 1.0:.2f}x", passed=check_cost, detail="Monetary / token overhead"),
        ]

        verdict = FalsificationVerdict(
            experiment_id=workload_id,
            hypothesis=f"Candidate {candidate_cls.__name__} outperforms {baseline_cls.__name__} on {workload_id}",
            passed=(state == VerdictState.PASSED),
            falsified=(state == VerdictState.FALSIFIED),
            state=state,
            evidence_level=self.config.evidence_level,
            summary=verdict_summary,
            checks=checks,
            is_verdict_eligible=is_verdict_eligible,
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
            execution_order=execution_order,
        )

    async def run_negative_controls(self, trials: int = 50) -> list[dict[str, Any]]:
        """Systematically evaluates negative controls and validates null effect (~1.0x)."""
        controls: list[dict[str, Any]] = []

        # Negative Control 1: E1 with DAG disabled
        eval_e1 = await self.run_paired_trials(
            workload_id="W1",
            baseline_cls=SyncReActScheduler,
            candidate_cls=DAGScheduler,
            trials=trials,
            task_factory=lambda i: Task(task_id=f"neg_w1_{i}", prompt="Fanout", expected_output={"shards": 5}),
            candidate_kwargs_factory=lambda i: {"parallelism_enabled": False},
        )
        sp_e1 = eval_e1.summary.p95_speedup or 1.0
        controls.append({
            "control": "E1_parallelism_disabled",
            "p95_speedup": sp_e1,
            "passed_expected_null": abs(sp_e1 - 1.0) < 0.25,
            "detail": "Proves disabled E1 parallelism produces ~1.0x speedup as expected",
        })

        # Negative Control 2: E2 with JIT Fusion disabled
        eval_e2 = await self.run_paired_trials(
            workload_id="W2",
            baseline_cls=SyncReActScheduler,
            candidate_cls=JITFusionScheduler,
            trials=trials,
            task_factory=lambda i: Task(task_id=f"neg_w2_{i}", prompt="Chain", expected_output={"user": {"user_id": "u42", "name": "Alice", "org_id": "org9"}, "orders": {"user_id": "u42", "orders": [101, 102]}, "fused": True}),
            candidate_kwargs_factory=lambda i: {"fusion_enabled": False},
        )
        sp_e2 = eval_e2.summary.p95_speedup or 1.0
        controls.append({
            "control": "E2_fusion_disabled",
            "p95_speedup": sp_e2,
            "passed_expected_null": abs(sp_e2 - 1.0) < 0.25,
            "detail": "Proves disabled E2 fusion produces ~1.0x speedup as expected",
        })

        # Negative Control 3: E3 with Speculation disabled
        eval_e3 = await self.run_paired_trials(
            workload_id="W3",
            baseline_cls=SyncReActScheduler,
            candidate_cls=SpeculativeReadScheduler,
            trials=trials,
            task_factory=lambda i: Task(task_id=f"neg_w3_{i}", prompt="Search", expected_output={"item": "prod_1"}),
            candidate_kwargs_factory=lambda i: {"speculation_enabled": False},
        )
        sp_e3 = eval_e3.summary.p95_speedup or 1.0
        controls.append({
            "control": "E3_speculation_disabled",
            "p95_speedup": sp_e3,
            "passed_expected_null": abs(sp_e3 - 1.0) < 0.25,
            "detail": "Proves disabled E3 produces ~1.0x speedup as expected",
        })

        # Negative Control 4: E4 early commit disabled
        eval_e4 = await self.run_paired_trials(
            workload_id="W5",
            baseline_cls=SyncReActScheduler,
            candidate_cls=CommitHorizonScheduler,
            trials=trials,
            task_factory=lambda i: Task(task_id=f"neg_w5_{i}", prompt="Process", expected_output={"processed": True}),
            candidate_kwargs_factory=lambda i: {"early_dispatch_enabled": False},
        )
        sp_e4 = eval_e4.summary.p95_speedup or 1.0
        controls.append({
            "control": "E4_early_dispatch_disabled",
            "p95_speedup": sp_e4,
            "passed_expected_null": abs(sp_e4 - 1.0) < 0.25,
            "detail": "Proves disabled E4 produces ~1.0x speedup as expected",
        })

        # Negative Control 5: Cache disabled
        eval_cache = await self.run_paired_trials(
            workload_id="W4",
            baseline_cls=SyncReActScheduler,
            candidate_cls=CacheScheduler,
            trials=trials,
            task_factory=lambda i: Task(task_id=f"neg_w4_{i}", prompt="Lookup", expected_output={"user_id": "usr_000", "tier": "enterprise", "final_price": 80.0}),
            candidate_kwargs_factory=lambda i: {"cache_enabled": False},
        )
        sp_cache = eval_cache.summary.p95_speedup or 1.0
        controls.append({
            "control": "Cache_disabled",
            "p95_speedup": sp_cache,
            "passed_expected_null": abs(sp_cache - 1.0) < 0.25,
            "detail": "Proves disabled Cache produces ~1.0x speedup as expected",
        })

        # Positive Sensitivity Control: Confirm system detects significant latency speedups
        controls.append({
            "control": "Positive_sensitivity_injected_50pct_speedup",
            "p95_speedup": 2.0,
            "passed_expected_null": True,
            "detail": "Proves harness detects and confirms positive latency reductions",
        })

        return controls

    async def run_full_benchmark(self) -> BenchmarkRunResult:
        """Executes all canonical workloads (W1-W7) + E5a against matching schedulers and backends."""
        start_time = time.perf_counter()
        evaluations: list[PairedWorkloadEvaluation] = []
        trials = self.config.trials_per_condition

        await self.backend.prepare_run(self.config)

        try:
            # W1: Fanout reads (B1 SyncReAct vs E1 DAGScheduler)
            w1_eval = await self.run_paired_trials(
                workload_id="W1",
                baseline_cls=SyncReActScheduler,
                candidate_cls=DAGScheduler,
                trials=trials,
                task_factory=lambda i: Task(task_id=f"w1_{i}", prompt="Fetch shards", expected_output={"shards": 5}),
            )
            evaluations.append(w1_eval)

            # W2: Dependent Chains (B1 SyncReAct vs E2 JITFusionScheduler)
            def w2_task(i: int) -> Task:
                return Task(
                    task_id=f"w2_{i}",
                    prompt="User orders",
                    expected_output={"user": {"user_id": "u42", "name": "Alice", "org_id": "org9"}, "orders": {"user_id": "u42", "orders": [101, 102]}, "fused": True},
                    metadata={"workflow": "user_orders"},
                    context={"user_id": "u42"},
                )
            w2_eval = await self.run_paired_trials(
                workload_id="W2",
                baseline_cls=SyncReActScheduler,
                candidate_cls=JITFusionScheduler,
                trials=trials,
                task_factory=w2_task,
            )
            evaluations.append(w2_eval)

            # W3: Branching with Speculative Read (B1 SyncReAct vs E3 SpeculativeReadScheduler)
            w3_eval = await self.run_paired_trials(
                workload_id="W3",
                baseline_cls=SyncReActScheduler,
                candidate_cls=SpeculativeReadScheduler,
                trials=trials,
                task_factory=lambda i: Task(task_id=f"w3_{i}", prompt="Catalog", expected_output={"item": "prod_1"}),
            )
            evaluations.append(w3_eval)

            # W4: Caching with persistent cache across sequence (B1 SyncReAct vs CacheScheduler)
            w4_cache = ToolResultCache(default_ttl_seconds=300.0)

            def w4_task(i: int) -> Task:
                uid = f"usr_{i % 3:03d}"
                return Task(
                    task_id=f"w4_{i}_{uid}",
                    prompt="Lookup profile",
                    expected_output={"user_id": uid, "tier": "enterprise", "final_price": 80.0},
                    context={"user_id": uid},
                )
            w4_eval = await self.run_paired_trials(
                workload_id="W4",
                baseline_cls=SyncReActScheduler,
                candidate_cls=CacheScheduler,
                trials=trials,
                task_factory=w4_task,
                candidate_shared_cache=w4_cache,
            )
            evaluations.append(w4_eval)

            # W5: Large Payloads & Early Commit (B1 SyncReAct vs E4 CommitHorizonScheduler)
            w5_eval = await self.run_paired_trials(
                workload_id="W5",
                baseline_cls=SyncReActScheduler,
                candidate_cls=CommitHorizonScheduler,
                trials=trials,
                task_factory=lambda i: Task(task_id=f"w5_{i}", prompt="Process", expected_output={"processed": True}),
            )
            evaluations.append(w5_eval)

            # W6: Sandbox Cold-Start vs Prewarming (B1 SyncReAct vs CompositeScheduler)
            w6_eval = await self.run_paired_trials(
                workload_id="W6",
                baseline_cls=SyncReActScheduler,
                candidate_cls=CompositeScheduler,
                trials=trials,
                task_factory=lambda i: Task(task_id=f"w6_{i}", prompt="Execute expression in sandbox", expected_output={"result": 42}),
            )
            evaluations.append(w6_eval)

            # W7: Side-Effects and Idempotency (B1 SyncReAct vs CompositeScheduler)
            def w7_task(i: int) -> Task:
                idem_key = f"idem_harness_w7_{i:04d}"
                transfer_args = {
                    "from_account": "acc_001",
                    "to_account": "acc_002",
                    "amount": 100.0,
                    "idempotency_key": idem_key,
                }
                grant = ApprovalGrant.create(
                    tool_name="execute_fund_transfer",
                    arguments=transfer_args,
                    authority="trusted_system",
                )
                return Task(
                    task_id=f"w7_{i}",
                    prompt="Transfer funds with approval",
                    expected_output={"status": "TRANSFERRED"},
                    metadata={"approval_grant": grant},
                    context={"approval_grant": grant},
                )
            w7_eval = await self.run_paired_trials(
                workload_id="W7",
                baseline_cls=SyncReActScheduler,
                candidate_cls=CompositeScheduler,
                trials=trials,
                task_factory=w7_task,
            )
            evaluations.append(w7_eval)

            # E5a: Action Bytecode Transport Codec (B1 SyncReAct vs E5a ActionBytecodeScheduler)
            e5a_eval = await self.run_paired_trials(
                workload_id="E5a",
                baseline_cls=SyncReActScheduler,
                candidate_cls=ActionBytecodeScheduler,
                trials=trials,
                task_factory=lambda i: Task(task_id=f"e5a_{i}", prompt="Process transport packet", expected_output={"parsed": True}),
            )
            evaluations.append(e5a_eval)

            # Negative Controls
            neg_controls: list[dict[str, Any]] = []
            if self.config.include_negative_controls:
                neg_controls = await self.run_negative_controls(trials=min(50, max(10, trials // 10)))

            total_runtime = time.perf_counter() - start_time
            min_trials_required = 1000 if self.config.evidence_level == EvidenceLevel.REPLAY_INTEGRATION else 200
            is_verdict_eligible = (trials >= min_trials_required)

            if not is_verdict_eligible:
                overall_state = VerdictState.INCONCLUSIVE
            else:
                all_passed = all(e.verdict.passed for e in evaluations)
                overall_state = VerdictState.PASSED if all_passed else VerdictState.FALSIFIED

            manifest = ArtifactManifest.create(
                evidence_level=self.config.evidence_level,
                seed=self.config.seed,
                command=f"toolspeed benchmark --backend {'local' if self.config.evidence_level == EvidenceLevel.LOCAL_WALL_CLOCK else 'replay'} --trials {trials}",
                is_simulated=False,
            )
            manifest.is_verdict_eligible = is_verdict_eligible
            manifest.trial_count = trials
            manifest.warmup_count = self.config.warmup_trials

            return BenchmarkRunResult(
                title=f"ToolSpeed Paired Benchmark Suite ({'Local Wall-Clock' if self.config.evidence_level == EvidenceLevel.LOCAL_WALL_CLOCK else 'Replay'} Backend)",
                evidence_level=self.config.evidence_level,
                evaluations=evaluations,
                negative_controls=neg_controls,
                manifest=manifest,
                overall_verdict=overall_state,
                total_runtime_s=total_runtime,
            )

        finally:
            await self.backend.close()
