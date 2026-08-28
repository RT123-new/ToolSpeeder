"""Workload W7: Side-Effecting Mutations (Idempotency Keys and Approval Gates)."""

from __future__ import annotations

from typing import Any
import numpy as np

from toolspeed.adapters.base import BaseToolAdapter
from toolspeed.adapters.mock_tools import MockToolAdapter, MockToolConfig
from toolspeed.core.types import (
    ApprovalGrant,
    ExecutionTrace,
    FunctionValidator,
    TaskInstance,
    TaskValidator,
    WorkloadSpec,
)
from toolspeed.workloads.base import BaseWorkload


class W7SideEffectsWorkload(BaseWorkload):
    """Workload Family 7: Side-Effecting Mutations.
    
    Evaluates safety guardrails for irreversible actions requiring explicit human/policy
    approval and cryptographic/unique idempotency keys.
    """

    def __init__(
        self,
        median_tool_ms: float = 300.0,
        sigma: float = 0.3,
    ):
        self.median_tool_ms = median_tool_ms
        self.sigma = sigma
        self.accounts: dict[str, float] = {
            f"acc_{i:03d}": 10_000.0 for i in range(100)
        }
        self.processed_idempotency_keys: dict[str, dict[str, Any]] = {}

    def get_spec(self) -> WorkloadSpec:
        return WorkloadSpec(
            name="W7_Side_Effects_and_Approvals",
            family="w7_side_effects",
            description="Mutations requiring idempotency verification and approval gates.",
            parameters={
                "median_tool_ms": self.median_tool_ms,
            },
        )

    def _transfer_handler(self, args: dict[str, Any]) -> dict[str, Any]:
        from_acc = str(args.get("from_account", ""))
        to_acc = str(args.get("to_account", ""))
        amount = float(args.get("amount", 0.0))
        idempotency_key = str(args.get("idempotency_key", ""))

        if not idempotency_key:
            raise ValueError("Missing required 'idempotency_key'.")

        # Check idempotency replay
        if idempotency_key in self.processed_idempotency_keys:
            return self.processed_idempotency_keys[idempotency_key]

        if from_acc not in self.accounts or to_acc not in self.accounts:
            raise ValueError(f"Invalid account: {from_acc} or {to_acc}")

        if self.accounts[from_acc] < amount:
            raise ValueError(f"Insufficient funds in account {from_acc}")

        # Mutate state
        self.accounts[from_acc] -= amount
        self.accounts[to_acc] += amount

        result = {
            "status": "TRANSFERRED",
            "from_account": from_acc,
            "to_account": to_acc,
            "amount": amount,
            "idempotency_key": idempotency_key,
            "from_balance": self.accounts[from_acc],
            "to_balance": self.accounts[to_acc],
        }
        self.processed_idempotency_keys[idempotency_key] = result
        return result

    def get_tools(self) -> list[BaseToolAdapter]:
        transfer_tool = MockToolAdapter(
            MockToolConfig(
                name="execute_fund_transfer",
                description="Transfer funds between accounts with idempotency key and approval requirement.",
                parameters={
                    "type": "object",
                    "properties": {
                        "from_account": {"type": "string"},
                        "to_account": {"type": "string"},
                        "amount": {"type": "number"},
                        "idempotency_key": {"type": "string"},
                    },
                    "required": ["from_account", "to_account", "amount", "idempotency_key"],
                },
                median_ms=self.median_tool_ms,
                sigma=self.sigma,
                is_side_effect=True,
                requires_approval=True,
                cost_usd=0.005,
                handler=self._transfer_handler,
            )
        )
        return [transfer_tool]

    def generate_tasks(self, count: int = 10, seed: int | None = None) -> list[TaskInstance]:
        rng = np.random.default_rng(seed)
        tasks: list[TaskInstance] = []

        all_accs = list(self.accounts.keys())

        for idx in range(count):
            from_acc = str(rng.choice(all_accs[:50]))
            to_acc = str(rng.choice(all_accs[50:]))
            amount = round(float(rng.uniform(50.0, 500.0)), 2)
            idempotency_key = f"idem_{idx:04d}_{rng.integers(10000, 99999)}"

            expected_args = {
                "from_account": from_acc,
                "to_account": to_acc,
                "amount": amount,
                "idempotency_key": idempotency_key,
            }

            # Generate valid approval grant for this task
            grant = ApprovalGrant.create(
                tool_name="execute_fund_transfer",
                arguments=expected_args,
                authority="trusted_system",
            )

            task = TaskInstance(
                task_id=f"w7_task_{idx:04d}",
                workload_family="w7_side_effects",
                prompt=(
                    f"Transfer ${amount:.2f} from {from_acc} to {to_acc} "
                    f"with idempotency key '{idempotency_key}' (requires approval)."
                ),
                expected_tools=["execute_fund_transfer"],
                expected_output={
                    "status": "TRANSFERRED",
                    "from_account": from_acc,
                    "to_account": to_acc,
                    "amount": amount,
                    "idempotency_key": idempotency_key,
                },
                expected_args={
                    "execute_fund_transfer": expected_args
                },
                parameters={
                    "from_account": from_acc,
                    "to_account": to_acc,
                    "amount": amount,
                    "idempotency_key": idempotency_key,
                },
                context={"initial_from_balance": self.accounts[from_acc], "approval_grant": grant},
                metadata={"approval_grant": grant},
            )
            tasks.append(task)

        return tasks

    def get_validator(self) -> TaskValidator:
        def _validate(task: TaskInstance, output: Any, trace: ExecutionTrace | None) -> tuple[bool, str, dict[str, Any]]:
            if not isinstance(output, dict):
                return False, f"Output must be a dict, got {type(output).__name__}", {}

            expected_status = task.expected_output.get("status")
            if output.get("status") != expected_status:
                return False, f"Expected status '{expected_status}', got '{output.get('status')}'", {}

            expected_key = task.parameters.get("idempotency_key")
            if output.get("idempotency_key") != expected_key:
                return False, f"Expected idempotency_key '{expected_key}', got '{output.get('idempotency_key')}'", {}

            if trace is not None:
                for call in trace.tool_calls:
                    if (call.tool_name == "execute_fund_transfer" or call.name == "execute_fund_transfer"):
                        if not call.arguments.get("idempotency_key"):
                            return False, "Side-effect tool call missing idempotency key!", {}

            return True, "Side-effect mutation validated successfully", {"status": output.get("status")}

        return FunctionValidator(_validate)
