"""Visualization module for the MediFlow translation benchmark.

Generates a self-contained Plotly HTML report with five charts:
  1. Avg Quality Score vs Avg Cost/File scatter (log x-axis)
  2. Avg Quality Score vs Avg Time/File scatter
  3. Avg Errors per File by MQM Category grouped bar
  4. Score Distribution box plot (0.0–1.0 per scenario)
  5. Avg Quality Score by Document Type grouped bar
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

from src.benchmark.db import BenchmarkDB

_MQM_CATEGORIES = ("accuracy", "terminology", "audience", "linguistic", "locale")
_DOC_TYPES = ("form", "summary", "prescript", "referral")


def _doc_type_from_id(doc_id: str) -> str:
    """Extract the document type prefix from a document ID.

    Args:
        doc_id: Document identifier string, e.g. "Form_001".

    Returns:
        Lowercase type prefix, e.g. "form".
    """
    return doc_id.split("_")[0].lower()


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate per-document rows into per-scenario summary statistics.

    Args:
        rows: Raw result rows from BenchmarkDB.load_all_results(), each
              containing scenario_name, quality_score, errors, cost_usd,
              and elapsed_sec columns.

    Returns:
        Dict keyed by scenario_name. Each value is a dict with:
          - avg_quality_score (float)
          - avg_cost_per_file (float)
          - avg_time_per_file (float)
          - avg_errors_by_category: dict[str, float] for each MQM category
          - quality_scores: list[float] — one per document

    Raises:
        KeyError: If a row is missing one or more required fields.
    """
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row["scenario_name"]
        if name not in buckets:
            buckets[name] = {
                "quality_scores": [],
                "costs": [],
                "times": [],
                "error_counts": {cat: [] for cat in _MQM_CATEGORIES},
                "scores_by_type": {t: [] for t in _DOC_TYPES},
            }
        b = buckets[name]
        required_keys = ("doc_id", "quality_score", "cost_usd", "elapsed_sec")
        missing = [k for k in required_keys if k not in row]
        if missing:
            raise KeyError(
                f"Result row for scenario '{name}' is missing required fields {missing}; "
                f"got keys: {list(row.keys())}"
            )
        b["quality_scores"].append(row["quality_score"])
        b["costs"].append(row["cost_usd"])
        b["times"].append(row["elapsed_sec"])
        doc_type = _doc_type_from_id(row["doc_id"])
        if doc_type in b["scores_by_type"]:
            b["scores_by_type"][doc_type].append(row["quality_score"])

        errors = row.get("errors") or []
        cat_count = {cat: 0 for cat in _MQM_CATEGORIES}
        for err in errors:
            cat = err.get("category", "").lower()
            if cat in cat_count:
                cat_count[cat] += 1
        for cat in _MQM_CATEGORIES:
            b["error_counts"][cat].append(cat_count[cat])

    for b in buckets.values():
        n = len(b["quality_scores"])
        b["avg_quality_score"] = sum(b["quality_scores"]) / n
        b["avg_cost_per_file"] = sum(b["costs"]) / n
        b["avg_time_per_file"] = sum(b["times"]) / n
        b["avg_errors_by_category"] = {
            cat: sum(b["error_counts"][cat]) / n for cat in _MQM_CATEGORIES
        }
        b["avg_score_by_type"] = {
            t: (sum(scores) / len(scores) if scores else 0.0)
            for t, scores in b["scores_by_type"].items()
        }

    return buckets


def _score_vs_cost_chart(
    scenarios: list[str],
    agg: dict[str, dict[str, Any]],
) -> go.Figure:
    """Build a scatter chart of avg quality score vs avg cost per file (log x-axis).

    Args:
        scenarios: Ordered list of scenario names.
        agg: Aggregated stats from _aggregate().

    Returns:
        A Plotly Figure with one point per scenario.
    """
    costs = [agg[s]["avg_cost_per_file"] for s in scenarios]
    scores = [agg[s]["avg_quality_score"] for s in scenarios]
    fig = go.Figure(
        go.Scatter(
            x=costs,
            y=scores,
            mode="markers+text",
            text=scenarios,
            textposition="top center",
            marker={"size": 12},
        )
    )
    fig.update_layout(
        title="Avg Quality Score vs Avg Cost per File",
        xaxis_title="Avg cost per file USD (log scale)",
        yaxis_title="Avg quality score (0.0–1.0)",
        xaxis_type="log",
    )
    return fig


def _score_vs_time_chart(
    scenarios: list[str],
    agg: dict[str, dict[str, Any]],
) -> go.Figure:
    """Build a scatter chart of avg quality score vs avg time per file.

    Args:
        scenarios: Ordered list of scenario names.
        agg: Aggregated stats from _aggregate().

    Returns:
        A Plotly Figure with one point per scenario.
    """
    times = [agg[s]["avg_time_per_file"] for s in scenarios]
    scores = [agg[s]["avg_quality_score"] for s in scenarios]
    fig = go.Figure(
        go.Scatter(
            x=times,
            y=scores,
            mode="markers+text",
            text=scenarios,
            textposition="top center",
            marker={"size": 12},
        )
    )
    fig.update_layout(
        title="Avg Quality Score vs Avg Time per File",
        xaxis_title="Avg time per file (seconds)",
        yaxis_title="Avg quality score (0.0–1.0)",
    )
    return fig


def _error_breakdown_chart(
    scenarios: list[str],
    agg: dict[str, dict[str, Any]],
) -> go.Figure:
    """Build a grouped bar chart of avg MQM error counts per file by category.

    Args:
        scenarios: Ordered list of scenario names.
        agg: Aggregated stats from _aggregate().

    Returns:
        A Plotly Figure with one bar group per MQM category.
    """
    fig = go.Figure()
    for cat in _MQM_CATEGORIES:
        fig.add_trace(
            go.Bar(
                name=cat.capitalize(),
                x=scenarios,
                y=[agg[s]["avg_errors_by_category"][cat] for s in scenarios],
            )
        )
    fig.update_layout(
        title="Avg Errors per File by Category",
        yaxis_title="Avg error count per file",
        barmode="group",
    )
    return fig


def _distribution_chart(
    scenarios: list[str],
    agg: dict[str, dict[str, Any]],
) -> go.Figure:
    """Build a box plot of quality score distribution across eval docs.

    Args:
        scenarios: Ordered list of scenario names.
        agg: Aggregated stats from _aggregate().

    Returns:
        A Plotly Figure with one box per scenario.
    """
    fig = go.Figure()
    for s in scenarios:
        fig.add_trace(
            go.Box(
                y=agg[s]["quality_scores"],
                name=s,
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
            )
        )
    fig.update_layout(
        title="Score Distribution Across Eval Documents",
        yaxis_title="Quality score (0.0–1.0)",
    )
    return fig


def _chart_score_by_type(
    scenarios: list[str],
    agg: dict[str, dict[str, Any]],
) -> go.Figure:
    """Build a grouped bar chart of avg quality score by document type.

    Args:
        scenarios: Ordered list of scenario names.
        agg: Aggregated stats from _aggregate().

    Returns:
        A Plotly Figure with one bar group per scenario, X = doc types.
    """
    fig = go.Figure()
    for s in scenarios:
        fig.add_trace(
            go.Bar(
                name=s,
                x=list(_DOC_TYPES),
                y=[agg[s]["avg_score_by_type"][t] for t in _DOC_TYPES],
            )
        )
    fig.update_layout(
        title="Avg Quality Score by Document Type",
        xaxis_title="Document type",
        yaxis_title="Avg quality score (0.0–1.0)",
        barmode="group",
        yaxis={"range": [0, 1]},
    )
    return fig


def generate_report(db: BenchmarkDB, output_path: Path) -> None:
    """Query the database and write a self-contained Plotly HTML report.

    Produces five charts in a single HTML file (no server required):
      1. Avg Quality Score vs Avg Cost/File scatter
      2. Avg Quality Score vs Avg Time/File scatter
      3. Avg Errors per File by MQM Category grouped bar
      4. Score Distribution box plot
      5. Avg Quality Score by Document Type grouped bar

    Args:
        db: Open BenchmarkDB instance to read all results from.
        output_path: Filesystem path where the HTML file will be written.

    Returns:
        None

    Raises:
        ValueError: If the database contains no result rows.
    """
    rows = db.load_all_results()
    if not rows:
        raise ValueError(
            "No results found in the database. Run at least one scenario before generating a report."
        )

    scenarios = list(dict.fromkeys(row["scenario_name"] for row in rows))
    agg = _aggregate(rows)

    charts = [
        _score_vs_cost_chart(scenarios, agg),
        _score_vs_time_chart(scenarios, agg),
        _error_breakdown_chart(scenarios, agg),
        _distribution_chart(scenarios, agg),
        _chart_score_by_type(scenarios, agg),
    ]

    # Combine all figures into one HTML string.
    # First chart includes the Plotly JS (from CDN); subsequent charts are div-only.
    html_parts = []
    for i, fig in enumerate(charts):
        include_js = "cdn" if i == 0 else False
        html_parts.append(pio.to_html(fig, include_plotlyjs=include_js, full_html=False))

    full_html = (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head><meta charset='utf-8'>"
        "<title>MediFlow Benchmark Results</title></head>\n"
        "<body>\n"
        + "\n".join(html_parts)
        + "\n</body>\n</html>"
    )

    output_path.write_text(full_html, encoding="utf-8")
