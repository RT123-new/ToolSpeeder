"""Workload W8: Speculative Routing Workload with Latency Sweeps.

Evaluates speculative routing decisions across variable tool latency bins,
candidate set sizes, and distractor candidates with controlled oracle ground-truth.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from toolspeed.adapters.base import BaseLLMAdapter, BaseToolAdapter, LLMDecision
from toolspeed.adapters.mock_models import MockScriptedLLM
from toolspeed.adapters.mock_tools import MockToolAdapter, MockToolConfig
from toolspeed.core.types import (
    FunctionValidator,
    TaskInstance,
    TaskValidator,
    ToolCall,
    WorkloadSpec,
)
from toolspeed.schedulers.speculative_providers.base import SpeculationCandidate
from toolspeed.workloads.base import BaseWorkload

TOOL_DEFINITIONS = [
    ("fetch_invoice_document", "Fetch and parse customer invoice document.", "doc_id", "INV-"),
    ("query_order_history", "Query past customer order history and transaction timestamps.", "customer_id", "CUST-"),
    ("get_inventory_status", "Query warehouse product inventory levels and backorder status.", "sku", "SKU-"),
    ("read_customer_account", "Retrieve customer profile, billing address, and account status.", "account_id", "ACC-"),
    (
        "lookup_product_catalog",
        "Lookup product catalog attributes, specifications, and categories.",
        "item_id",
        "ITEM-",
    ),
    ("check_shipping_status", "Retrieve logistics tracking details and estimated arrival.", "tracking_num", "TRK-"),
    ("get_pricing_tiers", "Fetch volume discounting rules and pricing tier matrix.", "tier_id", "TIER-"),
    ("inspect_policy_rules", "Inspect business rules and regional return policy guidelines.", "policy_id", "POL-"),
]


class W8SpeculativeRoutingWorkload(BaseWorkload):
    """Workload W8: Speculative Routing under Controlled Tool Latency & Distractors."""

    def __init__(
        self,
        tool_latency_ms: float = 250.0,
        candidate_count: int = 4,
        sigma: float = 0.05,
    ) -> None:
        self.tool_latency_ms = tool_latency_ms
        self.candidate_count = min(len(TOOL_DEFINITIONS), max(2, candidate_count))
        self.sigma = sigma

    def get_spec(self) -> WorkloadSpec:
        return WorkloadSpec(
            name="W8_Speculative_Routing",
            family="w8_speculation",
            description="Speculative read-only tool routing with configurable tool latency sweeps and distractors.",
            parameters={
                "tool_latency_ms": self.tool_latency_ms,
                "candidate_count": self.candidate_count,
                "sigma": self.sigma,
            },
        )

    def get_tools(self) -> list[BaseToolAdapter]:
        tools: list[BaseToolAdapter] = []

        def _make_handler(tool_n: str) -> Any:
            def _handler(args: dict[str, Any]) -> dict[str, Any]:
                return {"status": "success", "tool": tool_n, "data": args}

            return _handler

        for name, desc, arg_name, _ in TOOL_DEFINITIONS[: self.candidate_count]:
            tools.append(
                MockToolAdapter(
                    MockToolConfig(
                        name=name,
                        description=desc,
                        parameters={
                            "type": "object",
                            "properties": {arg_name: {"type": "string"}},
                            "required": [arg_name],
                        },
                        is_side_effect=False,
                        requires_approval=False,
                        median_ms=self.tool_latency_ms,
                        sigma=self.sigma,
                        handler=_make_handler(name),
                    )
                )
            )
        return tools

    def generate_tasks(self, count: int = 10, seed: int | None = None) -> list[TaskInstance]:
        rng = np.random.default_rng(seed or 42)
        tasks: list[TaskInstance] = []

        pool = TOOL_DEFINITIONS[: self.candidate_count]

        for i in range(count):
            # Select target tool deterministically
            target_idx = int(rng.integers(0, len(pool)))
            target_tool, desc, arg_name, prefix = pool[target_idx]
            entity_id = f"{prefix}{1000 + i}"

            prompt = f"Please {desc.lower()} Use identifier {entity_id}."
            expected_args = {target_tool: {arg_name: entity_id}}

            task = TaskInstance(
                task_id=f"w8_spec_task_{i:04d}",
                prompt=prompt,
                workload_family="w8_speculation",
                expected_tools=[target_tool],
                expected_args=expected_args,
                metadata={
                    "target_tool": target_tool,
                    "arg_name": arg_name,
                    "entity_id": entity_id,
                    "tool_latency_ms": self.tool_latency_ms,
                    "candidate_tools": [t[0] for t in pool],
                },
            )
            tasks.append(task)

        return tasks

    def create_model_for_task(
        self,
        task: TaskInstance,
        decision_delay_ms: float = 150.0,
    ) -> BaseLLMAdapter:
        """Constructs a scripted LLM adapter with simulated reasoning delay."""
        target_tool = task.metadata["target_tool"]
        arg_name = task.metadata["arg_name"]
        entity_id = task.metadata["entity_id"]

        call = ToolCall(
            name=target_tool,
            tool_name=target_tool,
            arguments={arg_name: entity_id},
        )
        decision1 = LLMDecision(
            reasoning=f"Reasoning about task {task.task_id}...",
            tool_calls=[call],
            duration_ms=decision_delay_ms,
        )
        decision2 = LLMDecision(
            reasoning="Received tool results, producing final answer.",
            tool_calls=[],
            final_answer={"status": "completed", "result": f"Done with {target_tool}"},
            duration_ms=20.0,
        )
        return MockScriptedLLM(
            decision_steps=[decision1, decision2],
            simulated_decision_ms=decision_delay_ms,
            simulated_draft_ms=10.0,
        )

    def create_candidates_for_task(
        self,
        task_or_ctx: Any,
        tools: Any = None,
    ) -> list[SpeculationCandidate]:
        """Constructs immutable SpeculationCandidate instances for a given task or execution context."""
        t = getattr(task_or_ctx, "task", task_or_ctx)
        meta = getattr(t, "metadata", {}) or {}
        entity_id = meta.get("entity_id")
        if not entity_id:
            prompt = getattr(t, "prompt", "")
            for _, _, _, prefix in TOOL_DEFINITIONS:
                if prefix in prompt:
                    part = prompt.split(prefix, 1)[1]
                    digits = "".join(c for c in part if c.isdigit())
                    if digits:
                        entity_id = f"{prefix}{digits}"
                        break
            if not entity_id:
                entity_id = "ID-1000"

        pool = TOOL_DEFINITIONS[: self.candidate_count]
        candidates: list[SpeculationCandidate] = []

        for idx, (name, _, arg_name, _) in enumerate(pool):
            candidates.append(
                SpeculationCandidate(
                    candidate_id=f"cand_{idx}_{name}",
                    tool_name=name,
                    arguments={arg_name: entity_id},
                    is_read_only=True,
                    tool_family="read_service",
                )
            )
        return candidates

    def get_validator(self) -> TaskValidator:
        def _validate(trace: Any) -> bool:
            return bool(getattr(trace, "success", False))

        return FunctionValidator(_validate)
