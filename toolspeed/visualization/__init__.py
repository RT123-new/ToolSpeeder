"""Visualization and reporting tools for ToolSpeed."""

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
