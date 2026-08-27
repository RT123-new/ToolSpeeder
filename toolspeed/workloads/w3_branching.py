"""Workload W3: Branching Workflows (Dynamic decision routing based on tool results)."""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from toolspeed.adapters.base import BaseToolAdapter
from toolspeed.adapters.mock_tools import MockToolAdapter, MockToolConfig
from toolspeed.core.types import (
    ExecutionTrace,
    FunctionValidator,
    TaskInstance,
    TaskValidator,
    WorkloadSpec,
)
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
                    handler=lambda args: self._tx_registry.get(args.get("tx_id", ""), {"risk_score": 50, "amount": 100}),
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
                    handler=lambda args: {"status": "APPROVED", "tx_id": args.get("tx_id"), "approval_code": f"APP_{args.get('tx_id')}"},
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
                    handler=lambda args: {"tx_id": args.get("tx_id"), "challenge_id": f"CHAL_{args.get('tx_id')}", "otp_required": True},
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

    def generate_tasks(self, count: int = 10, seed: Optional[int] = None) -> list[TaskInstance]:
        rng = np.random.default_rng(seed)
        tasks: list[TaskInstance] = []

        for idx in range(count):
            tx_id = f"tx_route_{idx:04d}"
            # Choose risk category: low (0-34), medium (35-74), high (75-100)
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
                expected_output={"tx_id": tx_id, "branch": expected_branch, "final_status": expected_status, "risk_score": risk},
                parameters={"tx_id": tx_id, "expected_branch": expected_branch, "risk_score": risk},
                context={"tx_data": self._tx_registry[tx_id]},
            )
            tasks.append(task)

        return tasks

    def get_validator(self) -> TaskValidator:
        def _validate(task: TaskInstance, output: Any, trace: Optional[ExecutionTrace]) -> Tuple[bool, str, dict[str, Any]]:
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
                called_tools = [c.tool_name for c in trace.tool_calls]
                for exp_tool in task.expected_tools:
                    if exp_tool not in called_tools:
                        return False, f"Required tool '{exp_tool}' was not called on branch '{expected_branch}'", {}

            return True, "Branching workflow validation passed", {"branch": actual_branch, "status": actual_status}

        return FunctionValidator(_validate)
