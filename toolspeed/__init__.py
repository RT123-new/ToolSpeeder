"""ToolSpeed: System-level latency optimization and rigorous evaluation for AI agent tool calls."""

__version__ = "0.1.0"

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
)
from toolspeed.visualization.charts import (
    ascii_bar_chart,
    ascii_sparkline,
    ascii_table,
    generate_cdf_chart,
    generate_speedup_line_chart,
    generate_workload_bar_chart,
)
from toolspeed.visualization.report import (
    generate_html_dashboard,
    generate_json_summary,
    generate_markdown_evidence_log,
    save_all_reports,
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
    "ascii_bar_chart",
    "ascii_sparkline",
    "ascii_table",
    "generate_cdf_chart",
    "generate_html_dashboard",
    "generate_json_summary",
    "generate_markdown_evidence_log",
    "generate_speedup_line_chart",
    "generate_workload_bar_chart",
    "run_e1_experiment",
    "run_e2_experiment",
    "run_e3_experiment",
    "run_e4_experiment",
    "run_e5_experiment",
    "run_full_suite",
    "save_all_reports",
]
