"""Experiment modules and runners for ToolSpeed."""

from toolspeed.experiments.e1_dag_runner import E1DAGExperiment, run_e1_experiment
from toolspeed.experiments.e2_fusion_runner import E2FusionExperiment, run_e2_experiment
from toolspeed.experiments.e3_spec_runner import E3SpeculationExperiment, run_e3_experiment
from toolspeed.experiments.e4_commit_runner import E4CommitHorizonExperiment, run_e4_experiment
from toolspeed.experiments.e5_bytecode_runner import E5BytecodeExperiment, run_e5_experiment
from toolspeed.experiments.full_suite import (
    SuiteResult,
    SuiteRunner,
    WorkloadBenchmarkResult,
    run_full_suite,
)
from toolspeed.experiments.runner import (
    ExperimentResult,
    FalsificationVerdict,
    HypothesisCheck,
    LatencyProfile,
    MetricSummary,
    WorkloadFamily,
    bootstrap_confidence_interval,
    compute_percentiles,
    compute_summary,
    samples,
)

__all__ = [
    "E1DAGExperiment",
    "E2FusionExperiment",
    "E3SpeculationExperiment",
    "E4CommitHorizonExperiment",
    "E5BytecodeExperiment",
    "ExperimentResult",
    "FalsificationVerdict",
    "HypothesisCheck",
    "LatencyProfile",
    "MetricSummary",
    "SuiteResult",
    "SuiteRunner",
    "WorkloadBenchmarkResult",
    "WorkloadFamily",
    "bootstrap_confidence_interval",
    "compute_percentiles",
    "compute_summary",
    "run_e1_experiment",
    "run_e2_experiment",
    "run_e3_experiment",
    "run_e4_experiment",
    "run_e5_experiment",
    "run_full_suite",
    "samples",
]
