"""Experiment modules and runners for ToolSpeed."""

from toolspeed.experiments.runner import (
    LatencyProfile,
    MetricSummary,
    ExperimentResult,
    FalsificationVerdict,
    HypothesisCheck,
    WorkloadFamily,
    compute_summary,
    samples,
    compute_percentiles,
    bootstrap_confidence_interval,
)
from toolspeed.experiments.e1_dag_runner import E1DAGExperiment, run_e1_experiment
from toolspeed.experiments.e2_fusion_runner import E2FusionExperiment, run_e2_experiment
from toolspeed.experiments.e3_spec_runner import E3SpeculationExperiment, run_e3_experiment
from toolspeed.experiments.e4_commit_runner import E4CommitHorizonExperiment, run_e4_experiment
from toolspeed.experiments.e5_bytecode_runner import E5BytecodeExperiment, run_e5_experiment
from toolspeed.experiments.full_suite import (
    SuiteRunner,
    SuiteResult,
    WorkloadBenchmarkResult,
    run_full_suite,
)

__all__ = [
    "LatencyProfile",
    "MetricSummary",
    "ExperimentResult",
    "FalsificationVerdict",
    "HypothesisCheck",
    "WorkloadFamily",
    "compute_summary",
    "samples",
    "compute_percentiles",
    "bootstrap_confidence_interval",
    "E1DAGExperiment",
    "run_e1_experiment",
    "E2FusionExperiment",
    "run_e2_experiment",
    "E3SpeculationExperiment",
    "run_e3_experiment",
    "E4CommitHorizonExperiment",
    "run_e4_experiment",
    "E5BytecodeExperiment",
    "run_e5_experiment",
    "SuiteRunner",
    "SuiteResult",
    "WorkloadBenchmarkResult",
    "run_full_suite",
]
