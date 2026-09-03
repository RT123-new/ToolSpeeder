"""Workloads module for ToolSpeed: standard benchmark families W1 through W7."""

from toolspeed.workloads.base import BaseWorkload
from toolspeed.workloads.w1_independent import (
    W1ConcurrencyPressurePoint,
    W1ConcurrencySweepReport,
    W1IndependentWorkload,
    evaluate_w1_concurrency_pressure,
)
from toolspeed.workloads.w2_chains import W2ChainsWorkload
from toolspeed.workloads.w3_branching import W3BranchingWorkload
from toolspeed.workloads.w4_locality import W4LocalityWorkload
from toolspeed.workloads.w5_large_payloads import W5LargePayloadsWorkload
from toolspeed.workloads.w6_cold_start import W6ColdStartWorkload
from toolspeed.workloads.w7_side_effects import W7SideEffectsWorkload

__all__ = [
    "BaseWorkload",
    "W1ConcurrencyPressurePoint",
    "W1ConcurrencySweepReport",
    "W1IndependentWorkload",
    "W2ChainsWorkload",
    "W3BranchingWorkload",
    "W4LocalityWorkload",
    "W5LargePayloadsWorkload",
    "W6ColdStartWorkload",
    "W7SideEffectsWorkload",
    "evaluate_w1_concurrency_pressure",
]
