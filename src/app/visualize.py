"""Visualization module for the MediFlow translation pipeline.

Generates a self-contained Plotly HTML report with six charts comparing
one or more named translation runs:

  1. Avg Quality Score by Run — bar chart, cross-run quality comparison
  2. Score by Document Type — grouped box plot, one box per (run, doc type)
  3. MQM Error Breakdown — stacked bar by error category and severity per run
  4. Section-level Scores — scatter of section index vs quality score per run
  5. Avg Time per File — bar chart of mean elapsed seconds per run
  6. Avg Time per File Type — grouped bar of mean elapsed seconds by doc type
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go
import plotly.io as pio

from src.app.db import BenchmarkDB

if TYPE_CHECKING:
    pass

_MQM_CATEGORIES = ("accuracy", "terminology", "audience", "linguistic", "locale")
_MQM_SEVERITIES = ("critical", "major", "minor")
_DOC_TYPES = ("form", "summary", "prescript", "referral")


def _doc_type_from_id(doc_id: str) -> str:
    """Extract the document type prefix from a document ID.

    Args:
        doc_id: Document identifier string, e.g. ``"Form_001"``.

    Returns:
        Lowercase type prefix, e.g. ``"form"``.
    """
    return doc_id.split("_")[0].lower()


def _aggregate(
    rows: list[dict[str, Any]],
    db: BenchmarkDB,
) -> dict[str, dict[str, Any]]:
    """Aggregate per-document rows into per-run summary statistics.

    Loads section-level data for each result row via ``db.load_section_results``
    so that chart 4 (section index vs quality score) can be populated.

    Args:
        rows: Raw result rows from :meth:`BenchmarkDB.load_all_results`, each
              containing ``run_name``, ``id``, ``doc_id``, ``quality_score``,
              ``errors``, ``cost_usd``, and ``elapsed_sec`` columns.
        db: Open database instance used to fetch per-section rows for chart 4.

    Returns:
        Dict keyed by ``run_name``. Each value is a dict with:

        - ``quality_scores`` (``list[float]``): document-level scores
        - ``times`` (``list[float]``): elapsed seconds per document
        - ``scores_by_type`` (``dict[str, list[float]]``): scores grouped by doc type
        - ``times_by_type`` (``dict[str, list[float]]``): elapsed seconds by doc type
        - ``error_counts`` (``dict[tuple[str,str], list[int]]``): per-(cat, sev) error
          counts per document
        - ``section_scores`` (``list[tuple[int, float]]``): ``(section_index, score)``
          pairs across all documents in this run
        - ``avg_quality_score``, ``avg_time_per_file``, ``avg_score_by_type``,
          ``avg_time_by_type``, ``avg_errors_by_cat_sev`` — derived summary stats

    Raises:
        KeyError: If a row is missing one or more required fields.
    """
    buckets: dict[str, dict[str, Any]] = {}

    for row in rows:
        name = row["run_name"]
        if name not in buckets:
            buckets[name] = {
                "quality_scores": [],
                "times": [],
                "scores_by_type": {t: [] for t in _DOC_TYPES},
                "times_by_type": {t: [] for t in _DOC_TYPES},
                # Keyed by (category, severity) — 5×3 = 15 combinations.
                "error_counts": {
                    (cat, sev): []
                    for cat in _MQM_CATEGORIES
                    for sev in _MQM_SEVERITIES
                },
                "section_scores": [],
            }

        b = buckets[name]

        required_keys = ("id", "doc_id", "quality_score", "elapsed_sec")
        missing = [k for k in required_keys if k not in row]
        if missing:
            raise KeyError(
                f"Result row for run '{name}' is missing required fields {missing}; "
                f"got keys: {list(row.keys())}"
            )

        b["quality_scores"].append(row["quality_score"])
        b["times"].append(row["elapsed_sec"])

        doc_type = _doc_type_from_id(row["doc_id"])
        if doc_type in b["scores_by_type"]:
            b["scores_by_type"][doc_type].append(row["quality_score"])
            b["times_by_type"][doc_type].append(row["elapsed_sec"])

        # Tally error counts by (category, severity) for the stacked bar chart.
        errors = row.get("errors") or []
        cat_sev_count: dict[tuple[str, str], int] = {
            (cat, sev): 0
            for cat in _MQM_CATEGORIES
            for sev in _MQM_SEVERITIES
        }
        for err in errors:
            cat = err.get("category", "").lower()
            sev = err.get("severity", "").lower()
            if (cat, sev) in cat_sev_count:
                cat_sev_count[(cat, sev)] += 1
        for key in b["error_counts"]:
            b["error_counts"][key].append(cat_sev_count[key])

        # Fetch section rows for chart 4; each result may have many sections.
        section_rows = db.load_section_results(row["id"])
        for sr in section_rows:
            b["section_scores"].append((sr["section_index"], sr["quality_score"]))

    # Derive per-run summary statistics after all rows have been bucketed.
    for b in buckets.values():
        n = len(b["quality_scores"])
        b["avg_quality_score"] = sum(b["quality_scores"]) / n
        b["avg_time_per_file"] = sum(b["times"]) / n
        b["avg_score_by_type"] = {
            t: (sum(scores) / len(scores) if scores else 0.0)
            for t, scores in b["scores_by_type"].items()
        }
        b["avg_time_by_type"] = {
            t: (sum(times) / len(times) if times else 0.0)
            for t, times in b["times_by_type"].items()
        }
        b["avg_errors_by_cat_sev"] = {
            key: sum(counts) / n
            for key, counts in b["error_counts"].items()
        }

    return buckets


def _quality_by_run_chart(
    runs: list[str],
    agg: dict[str, dict[str, Any]],
) -> go.Figure:
    """Build a bar chart of avg quality score per run.

    Args:
        runs: Ordered list of run names.
        agg: Aggregated stats from :func:`_aggregate`.

    Returns:
        A Plotly Figure with one bar per run.
    """
    scores = [agg[r]["avg_quality_score"] for r in runs]
    fig = go.Figure(
        go.Bar(
            x=runs,
            y=scores,
            text=[f"{s:.3f}" for s in scores],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Avg Quality Score by Run",
        yaxis_title="Avg quality score (0.0–1.0)",
        yaxis={"range": [0, 1.1]},
    )
    return fig


def _score_by_doc_type_chart(
    runs: list[str],
    agg: dict[str, dict[str, Any]],
) -> go.Figure:
    """Build a grouped box plot of quality score distributions by document type.

    One ``go.Box`` trace is added per ``(run, doc_type)`` pair. The
    ``legendgroup`` / ``showlegend`` pattern ensures only the first doc type
    for each run appears in the legend, avoiding 16 duplicate entries.

    Args:
        runs: Ordered list of run names.
        agg: Aggregated stats from :func:`_aggregate`.

    Returns:
        A Plotly Figure with boxes grouped by doc type, coloured by run.
    """
    fig = go.Figure()
    for run in runs:
        for i, doc_type in enumerate(_DOC_TYPES):
            scores = agg[run]["scores_by_type"][doc_type]
            if not scores:
                continue
            fig.add_trace(
                go.Box(
                    y=scores,
                    # x repeated to tell Plotly which category this box belongs to.
                    x=[doc_type] * len(scores),
                    name=run,
                    legendgroup=run,
                    # Show the legend entry only for the first doc type so the
                    # same run name doesn't appear four times in the legend.
                    showlegend=(i == 0),
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=-1.8,
                )
            )
    fig.update_layout(
        title="Score Distribution by Document Type",
        xaxis_title="Document type",
        yaxis_title="Quality score (0.0–1.0)",
        boxmode="group",
        yaxis={"range": [0, 1.05]},
    )
    return fig


def _error_breakdown_chart(
    runs: list[str],
    agg: dict[str, dict[str, Any]],
) -> go.Figure:
    """Build a stacked bar chart of avg MQM error counts by category and severity.

    With 15 (category × severity) combinations, grouped bars would be
    unreadable; stacking reveals both total error volume and composition.

    Args:
        runs: Ordered list of run names.
        agg: Aggregated stats from :func:`_aggregate`.

    Returns:
        A Plotly Figure with one stacked bar per run.
    """
    fig = go.Figure()
    # Iterate severities outermost so segments are grouped visually by severity.
    for sev in _MQM_SEVERITIES:
        for cat in _MQM_CATEGORIES:
            y = [agg[r]["avg_errors_by_cat_sev"].get((cat, sev), 0.0) for r in runs]
            fig.add_trace(
                go.Bar(
                    name=f"{cat}/{sev}",
                    x=runs,
                    y=y,
                )
            )
    fig.update_layout(
        title="MQM Error Breakdown by Category and Severity",
        yaxis_title="Avg errors per file",
        barmode="stack",
    )
    return fig


def _section_scores_chart(
    runs: list[str],
    agg: dict[str, dict[str, Any]],
) -> go.Figure:
    """Build a scatter plot of section index vs quality score across all documents.

    Each point represents one section of one document. Reveals whether
    quality degrades at later sections (positional bias in translation).

    Args:
        runs: Ordered list of run names.
        agg: Aggregated stats from :func:`_aggregate`.

    Returns:
        A Plotly Figure with one scatter trace per run.
    """
    fig = go.Figure()
    for run in runs:
        pairs = agg[run]["section_scores"]
        if not pairs:
            continue
        indices = [p[0] for p in pairs]
        scores = [p[1] for p in pairs]
        fig.add_trace(
            go.Scatter(
                x=indices,
                y=scores,
                mode="markers",
                name=run,
                opacity=0.6,
                marker={"size": 6},
            )
        )
    fig.update_layout(
        title="Section-level Quality Scores",
        xaxis_title="Section index",
        yaxis_title="Quality score (0.0–1.0)",
        yaxis={"range": [0, 1.05]},
    )
    return fig


def _avg_time_per_file_chart(
    runs: list[str],
    agg: dict[str, dict[str, Any]],
) -> go.Figure:
    """Build a bar chart of mean elapsed seconds per run.

    Args:
        runs: Ordered list of run names.
        agg: Aggregated stats from :func:`_aggregate`.

    Returns:
        A Plotly Figure with one bar per run.
    """
    times = [agg[r]["avg_time_per_file"] for r in runs]
    fig = go.Figure(
        go.Bar(
            x=runs,
            y=times,
            text=[f"{t:.1f}s" for t in times],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Avg Time per File",
        yaxis_title="Mean elapsed seconds",
    )
    return fig


def _avg_time_by_doc_type_chart(
    runs: list[str],
    agg: dict[str, dict[str, Any]],
) -> go.Figure:
    """Build a grouped bar chart of mean elapsed seconds by document type per run.

    Args:
        runs: Ordered list of run names.
        agg: Aggregated stats from :func:`_aggregate`.

    Returns:
        A Plotly Figure with one bar group per run.
    """
    fig = go.Figure()
    for run in runs:
        times = [agg[run]["avg_time_by_type"].get(t, 0.0) for t in _DOC_TYPES]
        fig.add_trace(
            go.Bar(
                name=run,
                x=list(_DOC_TYPES),
                y=times,
            )
        )
    fig.update_layout(
        title="Avg Time per File by Document Type",
        xaxis_title="Document type",
        yaxis_title="Mean elapsed seconds",
        barmode="group",
    )
    return fig


def generate_report(db: BenchmarkDB, output_path: Path) -> None:
    """Query the database and write a self-contained Plotly HTML report.

    Produces six charts in a single HTML file (no server required):

      1. Avg Quality Score by Run — cross-run quality comparison bar chart
      2. Score by Document Type — grouped box plot per doc type, coloured by run
      3. MQM Error Breakdown — stacked bar of error categories × severities per run
      4. Section-level Scores — scatter of section index vs quality score per run
      5. Avg Time per File — bar chart of mean elapsed seconds per run
      6. Avg Time per File Type — grouped bar of mean elapsed seconds by doc type

    Args:
        db: Open :class:`BenchmarkDB` instance to read all results from.
        output_path: Filesystem path where the HTML file will be written.

    Returns:
        None

    Raises:
        ValueError: If the database contains no result rows.
    """
    rows = db.load_all_results()
    if not rows:
        raise ValueError(
            "No results found in the database. "
            "Run at least one translation run before generating a report."
        )

    # dict.fromkeys preserves insertion order while deduplicating run names.
    runs = list(dict.fromkeys(row["run_name"] for row in rows))
    agg = _aggregate(rows, db)

    charts = [
        _quality_by_run_chart(runs, agg),
        _score_by_doc_type_chart(runs, agg),
        _error_breakdown_chart(runs, agg),
        _section_scores_chart(runs, agg),
        _avg_time_per_file_chart(runs, agg),
        _avg_time_by_doc_type_chart(runs, agg),
    ]

    # First chart includes the Plotly JS bundle from CDN; subsequent charts
    # emit only their <div> markup, keeping the file size reasonable.
    html_parts = []
    for i, fig in enumerate(charts):
        include_js = "cdn" if i == 0 else False
        html_parts.append(pio.to_html(fig, include_plotlyjs=include_js, full_html=False))

    full_html = (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head><meta charset='utf-8'>"
        "<title>MediFlow Translation Results</title></head>\n"
        "<body>\n"
        + "\n".join(html_parts)
        + "\n</body>\n</html>"
    )

    output_path.write_text(full_html, encoding="utf-8")
