"""Workload W3: Branching Workflows (Dynamic decision routing based on tool results)."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from toolspeed.adapters.base import BaseLLMAdapter, BaseToolAdapter, LLMDecision
from toolspeed.adapters.mock_tools import MockToolAdapter, MockToolConfig
from toolspeed.core.types import (
    ExecutionTrace,
    FunctionValidator,
    TaskInstance,
    TaskValidator,
    ToolCall,
    ToolSpec,
    WorkloadSpec,
)
from toolspeed.schedulers.base import BaseScheduler, SchedulerConfig
from toolspeed.schedulers.e3_speculation import SpeculativeReadScheduler
from toolspeed.workloads.base import BaseWorkload


class W3BranchingWorkload(BaseWorkload):
    """Workload Family 3: Branching Workflows.

    Evaluates dynamic branch resolution where intermediate tool returns determine
    downstream execution paths (low-risk approve vs medium stepup vs high-risk fraud).
    """

    def __init__(
        self,
        median_tool_ms: float = 350.0,
        sigma: float = 0.4,
    ):
        self.median_tool_ms = median_tool_ms
        self.sigma = sigma
        self._tx_registry: dict[str, dict[str, Any]] = {}

    def get_spec(self) -> WorkloadSpec:
        return WorkloadSpec(
            name="W3_Branching_Workflows",
            family="w3_branching",
            description="Dynamic conditional branching based on intermediate tool outputs.",
            parameters={
                "median_tool_ms": self.median_tool_ms,
                "sigma": self.sigma,
            },
        )

    def get_tools(self) -> list[BaseToolAdapter]:
        tools: list[BaseToolAdapter] = []

        # 1. Risk check
        tools.append(
            MockToolAdapter(
                MockToolConfig(
                    name="check_transaction_risk",
                    description="Evaluate transaction risk score (0-100).",
                    parameters={"type": "object", "properties": {"tx_id": {"type": "string"}}, "required": ["tx_id"]},
                    median_ms=self.median_tool_ms,
                    sigma=self.sigma,
                    handler=lambda args: self._tx_registry.get(
                        args.get("tx_id", ""), {"risk_score": 50, "amount": 100}
                    ),
                )
            )
        )

        # 2. Low-risk path: approve
        tools.append(
            MockToolAdapter(
                MockToolConfig(
                    name="approve_standard",
                    description="Approve low-risk transaction immediately.",
                    parameters={"type": "object", "properties": {"tx_id": {"type": "string"}}, "required": ["tx_id"]},
                    median_ms=self.median_tool_ms,
                    sigma=self.sigma,
                    handler=lambda args: {
                        "status": "APPROVED",
                        "tx_id": args.get("tx_id"),
                        "approval_code": f"APP_{args.get('tx_id')}",
                    },
                )
            )
        )

        # 3. Medium-risk path: step-up auth
        tools.append(
            MockToolAdapter(
                MockToolConfig(
                    name="request_stepup_auth",
                    description="Request step-up 2FA authentication challenge.",
                    parameters={"type": "object", "properties": {"tx_id": {"type": "string"}}, "required": ["tx_id"]},
                    median_ms=self.median_tool_ms,
                    sigma=self.sigma,
                    handler=lambda args: {
                        "tx_id": args.get("tx_id"),
                        "challenge_id": f"CHAL_{args.get('tx_id')}",
                        "otp_required": True,
                    },
                )
            )
        )
        tools.append(
            MockToolAdapter(
                MockToolConfig(
                    name="verify_stepup_response",
                    description="Verify step-up challenge resolution.",
                    parameters={
                        "type": "object",
                        "properties": {"challenge_id": {"type": "string"}, "code": {"type": "string"}},
                        "required": ["challenge_id"],
                    },
                    median_ms=self.median_tool_ms,
                    sigma=self.sigma,
                    handler=lambda args: {"status": "STEPUP_VERIFIED", "challenge_id": args.get("challenge_id")},
                )
            )
        )

        # 4. High-risk path: quarantine & fraud ticket
        tools.append(
            MockToolAdapter(
                MockToolConfig(
                    name="quarantine_transaction",
                    description="Quarantine high-risk fraudulent transaction.",
                    parameters={"type": "object", "properties": {"tx_id": {"type": "string"}}, "required": ["tx_id"]},
                    median_ms=self.median_tool_ms,
                    sigma=self.sigma,
                    is_side_effect=True,
                    handler=lambda args: {"status": "QUARANTINED", "tx_id": args.get("tx_id")},
                )
            )
        )
        tools.append(
            MockToolAdapter(
                MockToolConfig(
                    name="notify_fraud_team",
                    description="File fraud ticket with security operations.",
                    parameters={
                        "type": "object",
                        "properties": {"tx_id": {"type": "string"}, "risk_score": {"type": "integer"}},
                        "required": ["tx_id", "risk_score"],
                    },
                    median_ms=self.median_tool_ms,
                    sigma=self.sigma,
                    is_side_effect=True,
                    handler=lambda args: {"status": "TICKET_OPENED", "ticket_id": f"SEC_{args.get('tx_id')}"},
                )
            )
        )

        return tools

    def generate_tasks(self, count: int = 10, seed: int | None = None) -> list[TaskInstance]:
        rng = np.random.default_rng(seed)
        tasks: list[TaskInstance] = []

        for idx in range(count):
            tx_id = f"tx_route_{idx:04d}"
            category = rng.choice(["low", "medium", "high"])
            if category == "low":
                risk = int(rng.integers(5, 34))
                expected_branch = "low"
                expected_tools = ["check_transaction_risk", "approve_standard"]
                expected_status = "APPROVED"
            elif category == "medium":
                risk = int(rng.integers(35, 74))
                expected_branch = "medium"
                expected_tools = ["check_transaction_risk", "request_stepup_auth", "verify_stepup_response"]
                expected_status = "STEPUP_VERIFIED"
            else:
                risk = int(rng.integers(75, 99))
                expected_branch = "high"
                expected_tools = ["check_transaction_risk", "quarantine_transaction", "notify_fraud_team"]
                expected_status = "QUARANTINED_AND_FLAGGED"

            self._tx_registry[tx_id] = {
                "tx_id": tx_id,
                "risk_score": risk,
                "amount": float(rng.uniform(10.0, 5000.0)),
                "category": category,
            }

            task = TaskInstance(
                task_id=f"w3_task_{idx:04d}_{category}",
                workload_family="w3_branching",
                prompt=(
                    f"Process transaction '{tx_id}'. First check risk score. "
                    "If risk < 35: approve standard. "
                    "If 35 <= risk < 75: request stepup auth and verify. "
                    "If risk >= 75: quarantine transaction and notify fraud team."
                ),
                expected_tools=expected_tools,
                expected_output={
                    "tx_id": tx_id,
                    "branch": expected_branch,
                    "final_status": expected_status,
                    "risk_score": risk,
                },
                parameters={"tx_id": tx_id, "expected_branch": expected_branch, "risk_score": risk},
                context={"tx_data": self._tx_registry[tx_id]},
            )
            tasks.append(task)

        return tasks

    def get_validator(self) -> TaskValidator:
        def _validate(
            task: TaskInstance, output: Any, trace: ExecutionTrace | None
        ) -> tuple[bool, str, dict[str, Any]]:
            if not isinstance(output, dict):
                return False, f"Output must be a dict, got {type(output).__name__}", {}

            expected_branch = task.parameters.get("expected_branch")
            actual_branch = output.get("branch")
            if actual_branch != expected_branch:
                return False, f"Wrong branch taken. Expected '{expected_branch}', got '{actual_branch}'", {}

            expected_status = task.expected_output.get("final_status")
            actual_status = output.get("final_status")
            if actual_status != expected_status:
                return False, f"Expected final status '{expected_status}', got '{actual_status}'", {}

            if trace is not None:
                called_tools = [c.tool_name or c.name for c in trace.tool_calls]
                for exp_tool in task.expected_tools:
                    if exp_tool not in called_tools:
                        return False, f"Required tool '{exp_tool}' was not called on branch '{expected_branch}'", {}

            return True, "Branching workflow validation passed", {"branch": actual_branch, "status": actual_status}

        return FunctionValidator(_validate)


@dataclass(frozen=True)
class W3SpeculationFailurePoint:
    failure_rate: float
    baseline_duration_ms: float
    speculative_duration_ms: float
    speedup: float
    cancelled_leaked: bool


@dataclass
class W3SpeculationSweepReport:
    points: list[W3SpeculationFailurePoint]

    def verify_speculation_failure_invariants(self) -> tuple[bool, str]:
        """Verifies:

        - speedup is positive at 0% failure (> 1.0x)
        - speedup is negative at 100% failure (< 1.0x)
        - no unhandled exceptions or cancelled errors are leaked
        """
        if not self.points:
            return False, "No failure points evaluated"

        for p in self.points:
            if p.cancelled_leaked:
                return False, f"Cancelled draft leaked exception at failure rate {p.failure_rate:.2f}"

        sorted_pts = sorted(self.points, key=lambda x: x.failure_rate)

        p0 = next((p for p in sorted_pts if abs(p.failure_rate - 0.0) < 1e-4), None)
        p100 = next((p for p in sorted_pts if abs(p.failure_rate - 1.0) < 1e-4), None)

        if p0 is None or p100 is None:
            return False, "Missing 0% or 100% failure points"

        if p0.speedup <= 1.0:
            return False, f"Speedup at 0% failure ({p0.speedup:.2f}x) is not positive (expected > 1.0x)"

        if p100.speedup >= 1.0:
            return False, f"Speedup at 100% failure ({p100.speedup:.2f}x) is not negative (expected < 1.0x)"

        if p0.speedup <= p100.speedup:
            return (
                False,
                f"Speedup at 0% ({p0.speedup:.2f}x) not greater than at 100% ({p100.speedup:.2f}x)",
            )

        return True, "All W3 speculation failure invariants hold."


class W3DraftInjectingAdapter(BaseLLMAdapter):
    """Wraps a model adapter and injects draft failures deterministically or conditionally."""

    def __init__(self, inner: BaseLLMAdapter, inject_failure: bool = False) -> None:
        self.inner = inner
        self.inject_failure = inject_failure
        self.is_concurrency_safe = bool(getattr(inner, "is_concurrency_safe", True))

    async def decide(
        self,
        task: Any,
        history: list[dict[str, Any]],
        available_tools: list[ToolSpec],
    ) -> LLMDecision:
        return await self.inner.decide(task, history, available_tools)

    async def predict_draft(
        self,
        task: Any,
        history: list[dict[str, Any]],
        available_tools: list[ToolSpec],
    ) -> ToolCall | None:
        base_draft = await self.inner.predict_draft(task, history, available_tools)
        if base_draft is None:
            return None

        # Inject failure if specified
        if self.inject_failure:
            return ToolCall(
                name="audit_transaction",
                arguments={"customer_id": "divergent_customer_id"},
                speculation_confidence=0.95,
            )
        return copy.deepcopy(base_draft)


async def evaluate_w3_speculation_failure_sweep(
    backend: Any,
    baseline_scheduler: BaseScheduler,
    failure_rates: Sequence[float] = (0.0, 0.25, 0.50, 0.75, 1.0),
    seed: int = 42,
    trials_per_point: int = 4,
) -> W3SpeculationSweepReport:
    """Evaluates W3 single-slot speculative execution under injected draft failure rates."""
    points: list[W3SpeculationFailurePoint] = []

    for rate in failure_rates:
        base_durations: list[float] = []
        spec_durations: list[float] = []
        cancelled_leaked = False

        for t in range(trials_per_point):
            should_fail = (t / trials_per_point) < rate
            task_b = backend.generate_task("W3", trial_index=t, arm="baseline")
            task_c = backend.generate_task("W3", trial_index=t, arm="candidate")

            tools_b, model_b = backend.create_workload_environment("W3", trial_index=t, arm="baseline")
            tools_c, model_c = backend.create_workload_environment("W3", trial_index=t, arm="candidate")

            injected_model_c = W3DraftInjectingAdapter(model_c, inject_failure=should_fail)

            # Single concurrency slot for speculation under failure
            spec_sched = SpeculativeReadScheduler(
                SchedulerConfig(
                    concurrency_limit=1,
                    speculation_enabled=True,
                    speculation_contention_mode="single_slot",
                )
            )

            task_b_model = task_b.to_model_task() if hasattr(task_b, "to_model_task") else task_b
            task_c_model = task_c.to_model_task() if hasattr(task_c, "to_model_task") else task_c

            res_b = await baseline_scheduler.execute(task_b_model, model_b, tools_b)
            base_durations.append(res_b.total_duration_ms)

            try:
                res_c = await spec_sched.execute(task_c_model, injected_model_c, tools_c)
                spec_durations.append(res_c.total_duration_ms)
            except Exception:
                cancelled_leaked = True

        avg_b = float(np.mean(base_durations)) if base_durations else 1.0
        avg_c = float(np.mean(spec_durations)) if spec_durations else 1.0
        speedup = avg_b / avg_c if avg_c > 0 else 1.0

        points.append(
            W3SpeculationFailurePoint(
                failure_rate=rate,
                baseline_duration_ms=avg_b,
                speculative_duration_ms=avg_c,
                speedup=speedup,
                cancelled_leaked=cancelled_leaked,
            )
        )

    return W3SpeculationSweepReport(points=points)
