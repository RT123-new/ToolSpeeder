"""Visualization and reporting tools for ToolSpeed."""

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
    "ascii_bar_chart",
    "ascii_sparkline",
    "ascii_table",
    "generate_cdf_chart",
    "generate_html_dashboard",
    "generate_json_summary",
    "generate_markdown_evidence_log",
    "generate_speedup_line_chart",
    "generate_workload_bar_chart",
    "save_all_reports",
]
