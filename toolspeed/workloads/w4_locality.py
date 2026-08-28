"""Workload W4: Repeated Workflows with High Plan Locality (Subplan and Result Caching)."""

from __future__ import annotations

from typing import Any
import numpy as np

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


class W4LocalityWorkload(BaseWorkload):
    """Workload Family 4: Repeated Workflows with High Plan Locality.
    
    Evaluates speedups from exact/semantic tool result caching and plan template reuse
    across workloads with Zipfian/skewed key locality.
    """

    def __init__(
        self,
        num_entities: int = 20,
        hot_share: float = 0.8,
        cache_ttl_s: float = 60.0,
        median_tool_ms: float = 500.0,
        sigma: float = 0.4,
    ):
        self.num_entities = num_entities
        self.hot_share = hot_share
        self.cache_ttl_s = cache_ttl_s
        self.median_tool_ms = median_tool_ms
        self.sigma = sigma

        self._user_db: dict[str, dict[str, Any]] = {
            f"usr_{i:03d}": {
                "user_id": f"usr_{i:03d}",
                "tier": "enterprise" if i < 5 else ("pro" if i < 12 else "basic"),
                "discount_pct": 25 if i < 5 else (10 if i < 12 else 0),
                "region": "us-east" if i % 2 == 0 else "eu-west",
            }
            for i in range(num_entities)
        }

    def get_spec(self) -> WorkloadSpec:
        return WorkloadSpec(
            name="W4_High_Plan_Locality",
            family="w4_locality",
            description="Workloads with high repetition and result cache locality.",
            parameters={
                "num_entities": self.num_entities,
                "hot_share": self.hot_share,
                "cache_ttl_s": self.cache_ttl_s,
                "median_tool_ms": self.median_tool_ms,
            },
        )

    def get_tools(self) -> list[BaseToolAdapter]:
        user_lookup = MockToolAdapter(
            MockToolConfig(
                name="lookup_user_profile",
                description="Fetch user account tier, region, and discount metadata.",
                parameters={"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]},
                median_ms=self.median_tool_ms,
                sigma=self.sigma,
                cache_ttl_s=self.cache_ttl_s,
                cost_usd=0.001,
                handler=lambda args: self._user_db.get(args.get("user_id", ""), {}),
            )
        )
        pricing_calc = MockToolAdapter(
            MockToolConfig(
                name="calculate_final_invoice",
                description="Compute final billing price given tier and base price.",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "tier": {"type": "string"},
                        "base_amount": {"type": "number"},
                        "discount_pct": {"type": "number"},
                    },
                    "required": ["user_id", "tier", "base_amount"],
                },
                median_ms=100.0,
                sigma=0.2,
                cost_usd=0.0002,
                handler=lambda args: {
                    "final_price": round(args.get("base_amount", 100.0) * (1.0 - args.get("discount_pct", 0) / 100.0), 2),
                    "currency": "USD",
                },
            )
        )
        return [user_lookup, pricing_calc]

    def generate_tasks(self, count: int = 10, seed: int | None = None) -> list[TaskInstance]:
        rng = np.random.default_rng(seed)
        tasks: list[TaskInstance] = []

        hot_count = max(1, int(self.num_entities * 0.2))
        hot_users = [f"usr_{i:03d}" for i in range(hot_count)]
        cold_users = [f"usr_{i:03d}" for i in range(hot_count, self.num_entities)]

        for idx in range(count):
            if rng.random() < self.hot_share and hot_users:
                user_id = str(rng.choice(hot_users))
            else:
                user_id = str(rng.choice(cold_users)) if cold_users else hot_users[0]

            base_price = 100.0 * float(rng.integers(1, 10))
            user_info = self._user_db[user_id]
            discount = user_info["discount_pct"]
            tier = user_info["tier"]
            final_price = round(base_price * (1.0 - discount / 100.0), 2)

            task = TaskInstance(
                task_id=f"w4_task_{idx:04d}_{user_id}",
                workload_family="w4_locality",
                prompt=f"Lookup profile for user '{user_id}' and calculate invoice for base amount ${base_price:.2f}.",
                expected_tools=["lookup_user_profile", "calculate_final_invoice"],
                expected_output={"user_id": user_id, "tier": tier, "final_price": final_price},
                parameters={"user_id": user_id, "base_amount": base_price, "is_hot": user_id in hot_users},
                context={"user_profile": user_info},
            )
            tasks.append(task)

        return tasks

    def get_validator(self) -> TaskValidator:
        def _validate(task: TaskInstance, output: Any, trace: ExecutionTrace | None) -> tuple[bool, str, dict[str, Any]]:
            if not isinstance(output, dict):
                return False, f"Output must be a dict, got {type(output).__name__}", {}

            expected_price = task.expected_output.get("final_price")
            actual_price = output.get("final_price")
            if actual_price != expected_price:
                return False, f"Expected final price {expected_price}, got {actual_price}", {}

            expected_tier = task.expected_output.get("tier")
            actual_tier = output.get("tier")
            if actual_tier != expected_tier:
                return False, f"Expected tier '{expected_tier}', got '{actual_tier}'", {}

            return True, "Locality validation passed", {"final_price": actual_price}

        return FunctionValidator(_validate)
