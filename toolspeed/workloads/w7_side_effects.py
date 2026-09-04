"""Workload W7: Side-Effecting Mutations (Idempotency Keys and Approval Gates)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from toolspeed.adapters.base import BaseToolAdapter
from toolspeed.adapters.mock_tools import MockToolAdapter, MockToolConfig
from toolspeed.core.types import (
    ApprovalGrant,
    ExecutionTrace,
    FunctionValidator,
    StateSnapshot,
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
        self.accounts: dict[str, float] = {f"acc_{i:03d}": 10_000.0 for i in range(100)}
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
                expected_args={"execute_fund_transfer": expected_args},
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
        def _validate(
            task: TaskInstance, output: Any, trace: ExecutionTrace | None
        ) -> tuple[bool, str, dict[str, Any]]:
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
                    if (
                        call.tool_name == "execute_fund_transfer" or call.name == "execute_fund_transfer"
                    ) and not call.arguments.get("idempotency_key"):
                        return False, "Side-effect tool call missing idempotency key!", {}

            return True, "Side-effect mutation validated successfully", {"status": output.get("status")}

        return FunctionValidator(_validate)


class ExternalStateLedger:
    """External authoritative state ledger for account balances with idempotency protection and atomic mutations."""

    def __init__(self, initial_balance: float = 10_000.0) -> None:
        self.accounts: dict[str, float] = {f"acc_{i:03d}": initial_balance for i in range(100)}
        self.processed_idempotency_keys: dict[str, dict[str, Any]] = {}
        self.mutation_history: list[dict[str, Any]] = []

    def snapshot(self) -> StateSnapshot:
        """Returns an immutable snapshot of all account balances."""
        return StateSnapshot(dict(self.accounts))

    def get_balance(self, account: str) -> float:
        return self.accounts.get(account, 0.0)

    def transfer(
        self,
        from_acc: str,
        to_acc: str,
        amount: float,
        idempotency_key: str,
        fail_before_commit: bool = False,
    ) -> dict[str, Any]:
        if not idempotency_key:
            raise ValueError("Missing required 'idempotency_key'.")

        # Idempotency check: prevent duplicate execution
        if idempotency_key in self.processed_idempotency_keys:
            return {
                **self.processed_idempotency_keys[idempotency_key],
                "idempotency_replay": True,
                "mutated": False,
            }

        if from_acc not in self.accounts or to_acc not in self.accounts:
            raise ValueError(f"Invalid account: {from_acc} or {to_acc}")

        if self.accounts[from_acc] < amount:
            raise ValueError(f"Insufficient funds in account {from_acc}")

        if fail_before_commit:
            # Simulate worker/network crash before commit
            raise RuntimeError("Crash injected before commit horizon: transaction aborted")

        # Atomic mutation on commit
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
            "idempotency_replay": False,
            "mutated": True,
        }
        self.processed_idempotency_keys[idempotency_key] = result
        self.mutation_history.append(result)
        return result


class W7aSafetyWorkload(BaseWorkload):
    """Workload W7a: Safety-Critical Side Effects.

    Verifies:
    - Query external state before and after.
    - Verify exactly one mutation on commit.
    - Verify zero mutations on rollback/crash.
    - Verify idempotency keys prevent duplicate execution.
    """

    def __init__(self) -> None:
        self.ledger = ExternalStateLedger()

    def get_spec(self) -> WorkloadSpec:
        return WorkloadSpec(
            name="W7a_Safety_Critical",
            family="w7a_safety",
            description="Safety-critical mutations with exact state query and zero premature effects.",
            parameters={"workload_id": "W7_SAFETY"},
        )

    def get_tools(self) -> list[BaseToolAdapter]:
        def _handler(args: dict[str, Any]) -> dict[str, Any]:
            return self.ledger.transfer(
                from_acc=str(args["from_account"]),
                to_acc=str(args["to_account"]),
                amount=float(args["amount"]),
                idempotency_key=str(args["idempotency_key"]),
                fail_before_commit=bool(args.get("fail_before_commit", False)),
            )

        tool = MockToolAdapter(
            MockToolConfig(
                name="execute_fund_transfer",
                description="Transfer funds with strict idempotency and exact state verification.",
                parameters={
                    "type": "object",
                    "properties": {
                        "from_account": {"type": "string"},
                        "to_account": {"type": "string"},
                        "amount": {"type": "number"},
                        "idempotency_key": {"type": "string"},
                        "fail_before_commit": {"type": "boolean"},
                    },
                    "required": ["from_account", "to_account", "amount", "idempotency_key"],
                },
                is_side_effect=True,
                requires_approval=True,
                handler=_handler,
            )
        )
        return [tool]

    def generate_tasks(self, count: int = 5, seed: int | None = None) -> list[TaskInstance]:
        tasks: list[TaskInstance] = []
        for i in range(count):
            from_acc = f"acc_{i:03d}"
            to_acc = f"acc_{(i + 50):03d}"
            key = f"idem_w7a_{i:04d}"
            task = TaskInstance(
                task_id=f"w7a_task_{i:04d}",
                workload_family="w7a_safety",
                prompt=f"Transfer $100.00 from {from_acc} to {to_acc} with key {key}",
                expected_tools=["execute_fund_transfer"],
                expected_output={"status": "TRANSFERRED"},
                parameters={"from_account": from_acc, "to_account": to_acc, "amount": 100.0, "idempotency_key": key},
            )
            tasks.append(task)
        return tasks

    def get_validator(self) -> TaskValidator:
        def _validate(
            task: TaskInstance, output: Any, trace: ExecutionTrace | None
        ) -> tuple[bool, str, dict[str, Any]]:
            if not isinstance(output, dict) or output.get("status") != "TRANSFERRED":
                return False, "Transfer failed or invalid status", {}
            return True, "W7a safety transfer passed", {}

        return FunctionValidator(_validate)


class W7bLatencyWorkload(BaseWorkload):
    """Workload W7b: Pure Latency Benchmark for Side-Effecting Mutations.

    Evaluates execution throughput and latency percentiles without failure injection.
    """

    def __init__(self, median_tool_ms: float = 5.0, sigma: float = 0.1) -> None:
        self.median_tool_ms = median_tool_ms
        self.sigma = sigma
        self.ledger = ExternalStateLedger()

    def get_spec(self) -> WorkloadSpec:
        return WorkloadSpec(
            name="W7b_Pure_Latency",
            family="w7b_latency",
            description="Pure latency benchmark for side-effect mutations.",
            parameters={"workload_id": "W7_LATENCY", "median_tool_ms": self.median_tool_ms},
        )

    def get_tools(self) -> list[BaseToolAdapter]:
        def _handler(args: dict[str, Any]) -> dict[str, Any]:
            return self.ledger.transfer(
                from_acc=str(args["from_account"]),
                to_acc=str(args["to_account"]),
                amount=float(args["amount"]),
                idempotency_key=str(args["idempotency_key"]),
            )

        tool = MockToolAdapter(
            MockToolConfig(
                name="execute_fund_transfer",
                description="Transfer funds pure latency benchmark.",
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
                handler=_handler,
            )
        )
        return [tool]

    def generate_tasks(self, count: int = 10, seed: int | None = None) -> list[TaskInstance]:
        tasks: list[TaskInstance] = []
        for i in range(count):
            from_acc = f"acc_{i:03d}"
            to_acc = f"acc_{(i + 50):03d}"
            key = f"idem_w7b_{i:04d}"
            task = TaskInstance(
                task_id=f"w7b_task_{i:04d}",
                workload_family="w7b_latency",
                prompt=f"Transfer $25.00 from {from_acc} to {to_acc} with key {key}",
                expected_tools=["execute_fund_transfer"],
                expected_output={"status": "TRANSFERRED"},
                parameters={"from_account": from_acc, "to_account": to_acc, "amount": 25.0, "idempotency_key": key},
            )
            tasks.append(task)
        return tasks

    def get_validator(self) -> TaskValidator:
        def _validate(
            task: TaskInstance, output: Any, trace: ExecutionTrace | None
        ) -> tuple[bool, str, dict[str, Any]]:
            if not isinstance(output, dict) or output.get("status") != "TRANSFERRED":
                return False, "Transfer failed or invalid status", {}
            return True, "W7b latency transfer passed", {}

        return FunctionValidator(_validate)


W7SafetyWorkload = W7aSafetyWorkload
W7LatencyWorkload = W7bLatencyWorkload


@dataclass(frozen=True)
class W7aSafetyReport:
    commit_verified: bool
    rollback_zero_mutations_verified: bool
    idempotency_dedup_verified: bool
    mutation_count_commit: int
    mutation_count_rollback: int
    mutation_count_replay: int
    all_passed: bool
    message: str


@dataclass(frozen=True)
class W7bLatencyReport:
    sample_count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


def evaluate_w7a_safety() -> W7aSafetyReport:
    """Executes exact-state verification for W7a Safety:

    1. Query external state before and after commit -> exactly 1 mutation.
    2. Query external state before and after crash -> exactly 0 mutations.
    3. Re-execute same idempotency key -> duplicate prevented (0 extra mutations).
    """
    ledger = ExternalStateLedger(initial_balance=1000.0)

    # Test 1: Commit verification (query external state before and after)
    state_before_commit = ledger.snapshot()
    res1 = ledger.transfer("acc_001", "acc_002", 150.0, "idem_commit_001")
    state_after_commit = ledger.snapshot()

    diff_001 = state_before_commit.get("acc_001") - state_after_commit.get("acc_001")
    diff_002 = state_after_commit.get("acc_002") - state_before_commit.get("acc_002")
    commit_verified = res1["mutated"] is True and abs(diff_001 - 150.0) < 1e-6 and abs(diff_002 - 150.0) < 1e-6
    mutations_on_commit = 1 if commit_verified else 0

    # Test 2: Crash/rollback verification (zero premature/orphaned effects under failure)
    state_before_crash = ledger.snapshot()
    crash_caught = False
    try:
        ledger.transfer("acc_001", "acc_002", 200.0, "idem_crash_002", fail_before_commit=True)
    except RuntimeError:
        crash_caught = True
    state_after_crash = ledger.snapshot()

    rollback_verified = (
        crash_caught and state_before_crash.data == state_after_crash.data and len(ledger.mutation_history) == 1
    )
    mutations_on_crash = 0 if state_before_crash.data == state_after_crash.data else 1

    # Test 3: Idempotency keys prevent duplicate execution
    state_before_replay = ledger.snapshot()
    res_replay = ledger.transfer("acc_001", "acc_002", 150.0, "idem_commit_001")
    state_after_replay = ledger.snapshot()

    idempotency_verified = (
        res_replay["mutated"] is False
        and res_replay.get("idempotency_replay") is True
        and state_before_replay.data == state_after_replay.data
        and len(ledger.mutation_history) == 1
    )
    mutations_on_replay = 0 if state_before_replay.data == state_after_replay.data else 1

    all_passed = commit_verified and rollback_verified and idempotency_verified
    message = (
        "W7a safety passed: 1 commit mutation, 0 crash mutations, 0 duplicate replay mutations."
        if all_passed
        else "W7a safety verification failed"
    )

    return W7aSafetyReport(
        commit_verified=commit_verified,
        rollback_zero_mutations_verified=rollback_verified,
        idempotency_dedup_verified=idempotency_verified,
        mutation_count_commit=mutations_on_commit,
        mutation_count_rollback=mutations_on_crash,
        mutation_count_replay=mutations_on_replay,
        all_passed=all_passed,
        message=message,
    )


def evaluate_w7b_latency(iterations: int = 50, sleep_ms: float = 0.5) -> W7bLatencyReport:
    """Evaluates pure latency benchmark for W7b mutations without failure injection."""
    ledger = ExternalStateLedger()
    latencies: list[float] = []

    for i in range(iterations):
        from_acc = f"acc_{(i * 2) % 100:03d}"
        to_acc = f"acc_{(i * 2 + 1) % 100:03d}"
        key = f"idem_bench_{i:04d}"

        start = time.perf_counter_ns()
        ledger.transfer(from_acc, to_acc, 10.0, key)
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)
        dur_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        latencies.append(dur_ms)

    lat_arr = np.array(latencies)
    return W7bLatencyReport(
        sample_count=len(latencies),
        mean_ms=float(np.mean(lat_arr)),
        p50_ms=float(np.percentile(lat_arr, 50)),
        p95_ms=float(np.percentile(lat_arr, 95)),
        p99_ms=float(np.percentile(lat_arr, 99)),
        min_ms=float(np.min(lat_arr)),
        max_ms=float(np.max(lat_arr)),
    )
