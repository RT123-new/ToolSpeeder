"""Pure Python SVG and Terminal ASCII Chart Generators for ToolSpeed.

Zero external dependencies required (no matplotlib required).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

# Modern color palette
PALETTE = [
    "#3B82F6",  # Blue
    "#10B981",  # Emerald
    "#F59E0B",  # Amber
    "#8B5CF6",  # Violet
    "#EC4899",  # Pink
    "#06B6D4",  # Cyan
    "#EF4444",  # Red
    "#64748B",  # Slate
]


def _escape_xml(text: Any) -> str:
    """Escape text for XML/SVG safety."""
    s = str(text)
    return (
        s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&apos;")
    )


# ============================================================================
# Standalone SVG Chart Generators
# ============================================================================


def generate_speedup_line_chart(
    title: str,
    x_label: str,
    y_label: str,
    series: dict[str, Any] | Any,
    width: int = 800,
    height: int = 480,
    show_baseline_ref: bool = True,
    ref_y: float = 1.0,
) -> str:
    """Generate standalone SVG multi-line chart for parameter sweeps."""
    margin_top = 60
    margin_bottom = 65
    margin_left = 75
    margin_right = 160

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    # Find min/max ranges
    all_x: list[float] = []
    all_y: list[float] = []
    for pts in series.values():
        for x, y in pts:
            all_x.append(float(x))
            all_y.append(float(y))

    if not all_x or not all_y:
        all_x = [0.0, 1.0]
        all_y = [0.0, 1.0]

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    if show_baseline_ref:
        min_y = min(min_y, ref_y * 0.9)
        max_y = max(max_y, ref_y * 1.1)

    if min_x == max_x:
        max_x += 1.0
    if min_y == max_y:
        max_y += 1.0

    # Expand margins slightly
    y_pad = (max_y - min_y) * 0.08
    min_y = max(0.0, min_y - y_pad)
    max_y = max_y + y_pad

    def scale_x(val: float) -> float:
        return margin_left + ((val - min_x) / (max_x - min_x)) * plot_w

    def scale_y(val: float) -> float:
        return margin_top + plot_h - ((val - min_y) / (max_y - min_y)) * plot_h

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background:#0f172a;font-family:system-ui,-apple-system,sans-serif;">',
        "<style>",
        "  .title { fill: #f8fafc; font-size: 16px; font-weight: 600; }",
        "  .axis-label { fill: #94a3b8; font-size: 12px; }",
        "  .tick-label { fill: #64748b; font-size: 11px; }",
        "  .grid { stroke: #1e293b; stroke-width: 1; }",
        "  .axis-line { stroke: #334155; stroke-width: 1.5; }",
        "  .ref-line { stroke: #ef4444; stroke-width: 1.5; stroke-dasharray: 4,4; }",
        "  .legend-text { fill: #cbd5e1; font-size: 11px; }",
        "</style>",
        # Background card
        f'<rect width="{width}" height="{height}" rx="12" fill="#0f172a"/>',
        # Title
        f'<text x="{margin_left}" y="35" class="title">{_escape_xml(title)}</text>',
    ]

    # Grid & Y ticks
    num_y_ticks = 6
    for i in range(num_y_ticks):
        y_val = min_y + (i / (num_y_ticks - 1)) * (max_y - min_y)
        py = scale_y(y_val)
        svg_parts.append(f'<line x1="{margin_left}" y1="{py}" x2="{margin_left + plot_w}" y2="{py}" class="grid"/>')
        svg_parts.append(
            f'<text x="{margin_left - 10}" y="{py + 4}" text-anchor="end" class="tick-label">{y_val:.2f}</text>'
        )

    # X ticks
    num_x_ticks = 5
    for i in range(num_x_ticks):
        x_val = min_x + (i / (num_x_ticks - 1)) * (max_x - min_x)
        px = scale_x(x_val)
        svg_parts.append(f'<line x1="{px}" y1="{margin_top}" x2="{px}" y2="{margin_top + plot_h}" class="grid"/>')
        svg_parts.append(
            f'<text x="{px}" y="{margin_top + plot_h + 20}" text-anchor="middle" class="tick-label">{x_val:.1f}</text>'
        )

    # Axis Lines
    svg_parts.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" class="axis-line"/>'
    )
    svg_parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" class="axis-line"/>'
    )

    # Reference Line (1.0 Baseline)
    if show_baseline_ref and min_y <= ref_y <= max_y:
        ref_py = scale_y(ref_y)
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{ref_py}" x2="{margin_left + plot_w}" y2="{ref_py}" class="ref-line"/>'
        )
        svg_parts.append(
            f'<text x="{margin_left + plot_w + 8}" y="{ref_py + 4}" fill="#ef4444" font-size="10px">1.0x Baseline</text>'
        )

    # Axis Labels
    svg_parts.append(
        f'<text x="{margin_left + plot_w / 2}" y="{height - 15}" text-anchor="middle" class="axis-label">{_escape_xml(x_label)}</text>'
    )
    svg_parts.append(
        f'<text x="20" y="{margin_top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 20 {margin_top + plot_h / 2})" class="axis-label">{_escape_xml(y_label)}</text>'
    )

    # Plot Series Lines and Points
    legend_y = margin_top + 10
    for idx, (label, points) in enumerate(series.items()):
        color = PALETTE[idx % len(PALETTE)]
        sorted_pts = sorted(points, key=lambda p: p[0])
        if not sorted_pts:
            continue

        path_d = []
        for i, (x, y) in enumerate(sorted_pts):
            px, py = scale_x(x), scale_y(y)
            path_d.append(f"{'M' if i == 0 else 'L'}{px:.1f},{py:.1f}")

        svg_parts.append(
            f'<path d="{" ".join(path_d)}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"/>'
        )

        # Points
        for x, y in sorted_pts:
            px, py = scale_x(x), scale_y(y)
            svg_parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}" stroke="#0f172a" stroke-width="1.5"><title>{label}: ({x:.2f}, {y:.2f})</title></circle>'
            )

        # Legend
        svg_parts.append(
            f'<line x1="{width - margin_right + 15}" y1="{legend_y}" x2="{width - margin_right + 35}" y2="{legend_y}" stroke="{color}" stroke-width="2.5"/>'
        )
        svg_parts.append(f'<circle cx="{width - margin_right + 25}" cy="{legend_y}" r="3" fill="{color}"/>')
        svg_parts.append(
            f'<text x="{width - margin_right + 42}" y="{legend_y + 4}" class="legend-text">{_escape_xml(label)}</text>'
        )
        legend_y += 22

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def generate_cdf_chart(
    title: str,
    datasets: dict[str, Sequence[float]],
    width: int = 800,
    height: int = 480,
    x_unit: str = "ms",
) -> str:
    """Generate standalone SVG Cumulative Distribution Function (CDF) chart."""
    margin_top = 60
    margin_bottom = 65
    margin_left = 75
    margin_right = 160

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    # Find global min/max
    all_vals: list[float] = []
    for data in datasets.values():
        all_vals.extend([float(v) for v in data])

    if not all_vals:
        all_vals = [0.0, 1000.0]

    min_x = max(0.0, min(all_vals))
    max_x = max(all_vals)
    if min_x == max_x:
        max_x += 100.0

    def scale_x(val: float) -> float:
        return margin_left + ((val - min_x) / (max_x - min_x)) * plot_w

    def scale_y(prob: float) -> float:
        return margin_top + plot_h - (prob * plot_h)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background:#0f172a;font-family:system-ui,-apple-system,sans-serif;">',
        "<style>",
        "  .title { fill: #f8fafc; font-size: 16px; font-weight: 600; }",
        "  .axis-label { fill: #94a3b8; font-size: 12px; }",
        "  .tick-label { fill: #64748b; font-size: 11px; }",
        "  .grid { stroke: #1e293b; stroke-width: 1; }",
        "  .axis-line { stroke: #334155; stroke-width: 1.5; }",
        "  .p95-line { stroke: #64748b; stroke-width: 1; stroke-dasharray: 2,2; }",
        "  .legend-text { fill: #cbd5e1; font-size: 11px; }",
        "</style>",
        f'<rect width="{width}" height="{height}" rx="12" fill="#0f172a"/>',
        f'<text x="{margin_left}" y="35" class="title">{_escape_xml(title)}</text>',
    ]

    # Probabilities ticks: 0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0
    y_probs = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0]
    for p in y_probs:
        py = scale_y(p)
        svg_parts.append(f'<line x1="{margin_left}" y1="{py}" x2="{margin_left + plot_w}" y2="{py}" class="grid"/>')
        svg_parts.append(
            f'<text x="{margin_left - 10}" y="{py + 4}" text-anchor="end" class="tick-label">{int(p * 100)}%</text>'
        )

    # P95 reference line
    p95_y = scale_y(0.95)
    svg_parts.append(
        f'<line x1="{margin_left}" y1="{p95_y}" x2="{margin_left + plot_w}" y2="{p95_y}" class="p95-line"/>'
    )
    svg_parts.append(f'<text x="{margin_left + plot_w + 8}" y="{p95_y + 3}" fill="#94a3b8" font-size="10px">P95</text>')

    # X ticks
    num_x_ticks = 6
    for i in range(num_x_ticks):
        x_val = min_x + (i / (num_x_ticks - 1)) * (max_x - min_x)
        px = scale_x(x_val)
        svg_parts.append(f'<line x1="{px}" y1="{margin_top}" x2="{px}" y2="{margin_top + plot_h}" class="grid"/>')
        svg_parts.append(
            f'<text x="{px}" y="{margin_top + plot_h + 20}" text-anchor="middle" class="tick-label">{x_val:.0f}{x_unit}</text>'
        )

    svg_parts.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" class="axis-line"/>'
    )
    svg_parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" class="axis-line"/>'
    )

    svg_parts.append(
        f'<text x="{margin_left + plot_w / 2}" y="{height - 15}" text-anchor="middle" class="axis-label">Latency ({_escape_xml(x_unit)})</text>'
    )
    svg_parts.append(
        f'<text x="20" y="{margin_top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 20 {margin_top + plot_h / 2})" class="axis-label">Cumulative Probability</text>'
    )

    # Plot CDF curves
    legend_y = margin_top + 10
    for idx, (label, data) in enumerate(datasets.items()):
        color = PALETTE[idx % len(PALETTE)]
        sorted_data = np.sort(np.asarray(data, dtype=float))
        n_pts = len(sorted_data)
        if n_pts == 0:
            continue

        # Downsample for SVG path performance (up to 200 points)
        step = max(1, n_pts // 200)
        sampled_indices = list(range(0, n_pts, step))
        if (n_pts - 1) not in sampled_indices:
            sampled_indices.append(n_pts - 1)

        path_d = []
        for i_pt, idx_val in enumerate(sampled_indices):
            x = sorted_data[idx_val]
            prob = (idx_val + 1) / n_pts
            px = scale_x(x)
            py = scale_y(prob)
            path_d.append(f"{'M' if i_pt == 0 else 'L'}{px:.1f},{py:.1f}")

        svg_parts.append(f'<path d="{" ".join(path_d)}" fill="none" stroke="{color}" stroke-width="2.5"/>')

        # Legend
        svg_parts.append(
            f'<line x1="{width - margin_right + 15}" y1="{legend_y}" x2="{width - margin_right + 35}" y2="{legend_y}" stroke="{color}" stroke-width="2.5"/>'
        )
        svg_parts.append(
            f'<text x="{width - margin_right + 42}" y="{legend_y + 4}" class="legend-text">{_escape_xml(label)}</text>'
        )
        legend_y += 22

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def generate_workload_bar_chart(
    workload_data: list[dict[str, Any]],
    width: int = 850,
    height: int = 420,
) -> str:
    """Generate standalone SVG horizontal bar chart for W1-W7 speedup comparisons."""
    margin_top = 60
    margin_bottom = 45
    margin_left = 180
    margin_right = 100

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    n_bars = len(workload_data)
    bar_h = min(32, (plot_h / max(1, n_bars)) - 8)

    max_val = max([float(w.get("p95_speedup", 1.0)) for w in workload_data] + [2.0])
    max_x = math.ceil(max_val * 1.15 * 10) / 10.0

    def scale_x(val: float) -> float:
        return margin_left + (val / max_x) * plot_w

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background:#0f172a;font-family:system-ui,-apple-system,sans-serif;">',
        "<style>",
        "  .title { fill: #f8fafc; font-size: 16px; font-weight: 600; }",
        "  .label { fill: #cbd5e1; font-size: 12px; font-weight: 500; }",
        "  .val-label { fill: #f8fafc; font-size: 11px; font-weight: 600; }",
        "  .grid { stroke: #1e293b; stroke-width: 1; }",
        "  .axis-line { stroke: #334155; stroke-width: 1.5; }",
        "  .thresh-line { stroke: #10B981; stroke-width: 1.5; stroke-dasharray: 4,4; }",
        "</style>",
        f'<rect width="{width}" height="{height}" rx="12" fill="#0f172a"/>',
        f'<text x="{margin_left}" y="35" class="title">Workload Family Latency Speedup (P95 CCL)</text>',
    ]

    # Grid lines for speedup factors (1.0x, 1.5x, 2.0x, etc.)
    num_ticks = int(max_x * 2) + 1
    for i in range(num_ticks):
        x_val = i * 0.5
        if x_val > max_x:
            break
        px = scale_x(x_val)
        svg_parts.append(f'<line x1="{px}" y1="{margin_top}" x2="{px}" y2="{margin_top + plot_h}" class="grid"/>')
        svg_parts.append(
            f'<text x="{px}" y="{margin_top + plot_h + 18}" text-anchor="middle" fill="#64748b" font-size="11px">{x_val:.1f}x</text>'
        )

    # 1.10x Falsification Threshold Line (10% CCL improvement)
    thresh_x = scale_x(1.111)  # 1 / (1 - 0.10) ~ 1.11x
    svg_parts.append(
        f'<line x1="{thresh_x}" y1="{margin_top}" x2="{thresh_x}" y2="{margin_top + plot_h}" class="thresh-line"/>'
    )
    svg_parts.append(
        f'<text x="{thresh_x + 4}" y="{margin_top - 8}" fill="#10B981" font-size="10px">10% Falsification Target (1.11x)</text>'
    )

    # Bars
    slot_h = plot_h / max(1, n_bars)
    for i, w in enumerate(workload_data):
        y = margin_top + i * slot_h + (slot_h - bar_h) / 2
        speedup = float(w.get("p95_speedup", 1.0))
        pct_red = float(w.get("p95_reduction_pct", 0.0))
        bar_w = max(4.0, (speedup / max_x) * plot_w)
        name = w.get("workload_id", f"W{i + 1}")
        desc = w.get("name", "")
        # Color based on performance
        color = "#10B981" if pct_red >= 10.0 else "#F59E0B"

        svg_parts.append(
            f'<text x="{margin_left - 12}" y="{y + bar_h / 2 + 4}" text-anchor="end" class="label">{_escape_xml(name)}</text>'
        )
        svg_parts.append(
            f'<rect x="{margin_left}" y="{y}" width="{bar_w}" height="{bar_h}" rx="4" fill="{color}"><title>{_escape_xml(desc)}: {speedup:.2f}x ({pct_red:.1f}% reduction)</title></rect>'
        )
        svg_parts.append(
            f'<text x="{margin_left + bar_w + 8}" y="{y + bar_h / 2 + 4}" class="val-label">{speedup:.2f}x ({pct_red:.1f}%)</text>'
        )

    svg_parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" class="axis-line"/>'
    )
    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


# ============================================================================
# Terminal ASCII Chart Generators
# ============================================================================


def ascii_sparkline(values: Sequence[float], width: int | None = None) -> str:
    """Generate compact Unicode sparkline (e.g.  ▂▃▅▆▇█)."""
    if not values:
        return ""
    vals = [float(v) for v in values]
    if width and len(vals) > width:
        # Downsample
        step = len(vals) / width
        vals = [vals[int(i * step)] for i in range(width)]

    min_v, max_v = min(vals), max(vals)
    if min_v == max_v:
        return "▄" * len(vals)

    bars = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    result = []
    for v in vals:
        norm = (v - min_v) / (max_v - min_v)
        idx = min(len(bars) - 1, int(norm * len(bars)))
        result.append(bars[idx])
    return "".join(result)


def ascii_bar_chart(
    data: dict[str, float],
    max_bar_width: int = 35,
    unit: str = "",
) -> str:
    """Generate clean horizontal ASCII bar chart for terminal output."""
    if not data:
        return ""
    max_val = max(data.values()) if data else 1.0
    if max_val <= 0:
        max_val = 1.0

    max_key_len = max(len(k) for k in data)
    lines = []
    for key, val in data.items():
        bar_len = int((val / max_val) * max_bar_width) if max_val > 0 else 0
        bar_str = "█" * bar_len + "░" * (max_bar_width - bar_len)
        lines.append(f"{key.ljust(max_key_len)} │ {bar_str} │ {val:.2f}{unit}")
    return "\n".join(lines)


def ascii_table(
    headers: list[str],
    rows: list[list[Any]],
    alignments: list[str] | None = None,
) -> str:
    """Generate beautiful formatted ASCII text table with box borders."""
    if not headers:
        return ""

    num_cols = len(headers)
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i in range(min(num_cols, len(row))):
            col_widths[i] = max(col_widths[i], len(str(row[i])))

    if not alignments:
        alignments = ["left"] * num_cols

    def format_cell(val: Any, width: int, align: str) -> str:
        s = str(val)
        if align == "right":
            return s.rjust(width)
        elif align == "center":
            return s.center(width)
        return s.ljust(width)

    top_border = "┌─" + "─┬─".join("─" * w for w in col_widths) + "─┐"
    header_sep = "├─" + "─┼─".join("─" * w for w in col_widths) + "─┤"
    bottom_border = "└─" + "─┴─".join("─" * w for w in col_widths) + "─┘"

    header_cells = [format_cell(h, col_widths[i], alignments[i]) for i, h in enumerate(headers)]
    header_row = "│ " + " │ ".join(header_cells) + " │"

    body_rows = []
    for row in rows:
        cells = []
        for i in range(num_cols):
            val = row[i] if i < len(row) else ""
            cells.append(format_cell(val, col_widths[i], alignments[i]))
        body_rows.append("│ " + " │ ".join(cells) + " │")

    return "\n".join([top_border, header_row, header_sep, *body_rows, bottom_border])
