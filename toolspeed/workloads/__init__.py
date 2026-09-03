"""Workloads module for ToolSpeed: standard benchmark families W1 through W7."""

from toolspeed.workloads.base import BaseWorkload
from toolspeed.workloads.w1_independent import (
    W1ConcurrencyPressurePoint,
    W1ConcurrencySweepReport,
    W1IndependentWorkload,
    evaluate_w1_concurrency_pressure,
)
from toolspeed.workloads.w2_chains import (
    CompiledExecutionPlan,
    CompiledPlanStep,
    W2ChainsWorkload,
    W2ComparisonResult,
    W2DynamicDependencyCompiler,
    evaluate_w2_compilation_vs_step_by_step,
    execute_compiled_plan,
)
from toolspeed.workloads.w3_branching import (
    W3BranchingWorkload,
    W3DraftInjectingAdapter,
    W3SpeculationFailurePoint,
    W3SpeculationSweepReport,
    evaluate_w3_speculation_failure_sweep,
)
from toolspeed.workloads.w4_locality import (
    W4CacheEvictionPoint,
    W4CacheEvictionSweepReport,
    W4LocalityWorkload,
    evaluate_w4_cache_eviction_pressure,
)
from toolspeed.workloads.w5_large_payloads import (
    W5LargePayloadsWorkload,
    W5StreamingComparisonPoint,
    W5StreamingSweepReport,
    evaluate_w5_streaming_vs_buffering,
)
from toolspeed.workloads.w6_cold_start import (
    W6AmortizationPoint,
    W6ColdStartWorkload,
    W6SubprocessSweepReport,
    WarmSubprocessPool,
    WarmSubprocessWorker,
    evaluate_w6_subprocess_warm_vs_cold,
    execute_cold_subprocess,
)
from toolspeed.workloads.w7_side_effects import W7SideEffectsWorkload

__all__ = [
    "BaseWorkload",
    "CompiledExecutionPlan",
    "CompiledPlanStep",
    "W1ConcurrencyPressurePoint",
    "W1ConcurrencySweepReport",
    "W1IndependentWorkload",
    "W2ChainsWorkload",
    "W2ComparisonResult",
    "W2DynamicDependencyCompiler",
    "W3BranchingWorkload",
    "W3DraftInjectingAdapter",
    "W3SpeculationFailurePoint",
    "W3SpeculationSweepReport",
    "W4CacheEvictionPoint",
    "W4CacheEvictionSweepReport",
    "W4LocalityWorkload",
    "W5LargePayloadsWorkload",
    "W5StreamingComparisonPoint",
    "W5StreamingSweepReport",
    "W6AmortizationPoint",
    "W6ColdStartWorkload",
    "W6SubprocessSweepReport",
    "W7SideEffectsWorkload",
    "WarmSubprocessPool",
    "WarmSubprocessWorker",
    "evaluate_w1_concurrency_pressure",
    "evaluate_w2_compilation_vs_step_by_step",
    "evaluate_w3_speculation_failure_sweep",
    "evaluate_w4_cache_eviction_pressure",
    "evaluate_w5_streaming_vs_buffering",
    "evaluate_w6_subprocess_warm_vs_cold",
    "execute_cold_subprocess",
    "execute_compiled_plan",
]
