"""ToolSpeed: System-level latency optimization and rigorous evaluation for AI agent tool calls."""

__version__ = "0.1.0"

from toolspeed.experiments.runner import (
    LatencyProfile,
    MetricSummary,
    ExperimentResult,
    FalsificationVerdict,
    HypothesisCheck,
    WorkloadFamily,
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
from toolspeed.visualization.charts import (
    generate_speedup_line_chart,
    generate_cdf_chart,
    generate_workload_bar_chart,
    ascii_sparkline,
    ascii_bar_chart,
    ascii_table,
)
from toolspeed.visualization.report import (
    generate_markdown_evidence_log,
    generate_html_dashboard,
    generate_json_summary,
    save_all_reports,
)

__all__ = [
    "LatencyProfile",
    "MetricSummary",
    "ExperimentResult",
    "FalsificationVerdict",
    "HypothesisCheck",
    "WorkloadFamily",
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
    "generate_speedup_line_chart",
    "generate_cdf_chart",
    "generate_workload_bar_chart",
    "ascii_sparkline",
    "ascii_bar_chart",
    "ascii_table",
    "generate_markdown_evidence_log",
    "generate_html_dashboard",
    "generate_json_summary",
    "save_all_reports",
]
