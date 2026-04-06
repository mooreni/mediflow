# Quick smoke-test for src/benchmark/visualize.py.
#
# Creates synthetic benchmark data covering all 5 scenarios and 10 documents,
# then calls generate_report() and checks the produced HTML for correctness.
#
# Usage: python scripts/test_visualize.py

import random
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Synthetic data configuration
# ---------------------------------------------------------------------------

SCENARIOS = [
    ("s1_google_translate",          "Google Translate zero-shot"),
    ("s2_gemini_flash_zeroshot",     "Gemini Flash zero-shot"),
    ("s3_gemini_pro_dspy_predict",   "Gemini Pro DSPy Predict"),
    ("s4_gemini_pro_bootstrap",      "Gemini Pro BootstrapFewShot"),
    ("s5_dual_agent",                "Dual Agent (translator + critic)"),
]

DOC_IDS = [f"Form_{i:03d}" for i in range(1, 11)]

# Quality score profiles per scenario (0.0–1.0)
SCORE_PROFILES = {
    "s1_google_translate":        0.55,
    "s2_gemini_flash_zeroshot":   0.65,
    "s3_gemini_pro_dspy_predict": 0.72,
    "s4_gemini_pro_bootstrap":    0.80,
    "s5_dual_agent":              0.88,
}

COST_PER_DOC = {
    "s1_google_translate":        0.0004,
    "s2_gemini_flash_zeroshot":   0.0012,
    "s3_gemini_pro_dspy_predict": 0.008,
    "s4_gemini_pro_bootstrap":    0.009,
    "s5_dual_agent":              0.018,
}

ELAPSED_PER_DOC = {
    "s1_google_translate":        1.2,
    "s2_gemini_flash_zeroshot":   3.5,
    "s3_gemini_pro_dspy_predict": 8.0,
    "s4_gemini_pro_bootstrap":    9.0,
    "s5_dual_agent":              18.0,
}

_MQM_CATEGORIES = ("accuracy", "terminology", "audience", "linguistic", "locale")


def _make_result(scenario_name: str, doc_id: str, seed: int) -> dict:
    """Build a synthetic result dict matching BenchmarkDB.load_all_results() shape.

    Args:
        scenario_name: Name of the translation scenario.
        doc_id: Document identifier.
        seed: Deterministic seed for score jitter.

    Returns:
        Dict with all columns expected by visualize._aggregate().
    """
    rng = random.Random(seed)
    base_score = SCORE_PROFILES[scenario_name]
    quality_score = min(1.0, max(0.0, base_score + rng.uniform(-0.1, 0.1)))

    # Synthetic MQM errors: better scenarios have fewer errors
    n_errors = max(0, int((1.0 - base_score) * 10 + rng.uniform(-2, 2)))
    errors = []
    for k in range(n_errors):
        errors.append({
            "span": f"sample span {k}",
            "category": rng.choice(_MQM_CATEGORIES),
            "severity": rng.choice(["critical", "major", "minor"]),
            "justification": "synthetic error for testing",
        })

    return {
        "scenario_name": scenario_name,
        "doc_id": doc_id,
        "quality_score": quality_score,
        "errors": errors,
        "cost_usd": COST_PER_DOC[scenario_name],
        "elapsed_sec": ELAPSED_PER_DOC[scenario_name] + rng.uniform(-0.5, 0.5),
    }


def build_rows() -> list[dict]:
    """Build all synthetic result rows.

    Returns:
        List of result dicts — one per (scenario, document) pair.
    """
    rows = []
    for i, (name, _) in enumerate(SCENARIOS):
        for j, doc_id in enumerate(DOC_IDS):
            rows.append(_make_result(name, doc_id, seed=i * 100 + j))
    return rows


# ---------------------------------------------------------------------------
# Minimal BenchmarkDB stub
# ---------------------------------------------------------------------------

class _FakeDB:
    """Minimal stub of BenchmarkDB that returns pre-built rows."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def load_all_results(self) -> list[dict]:
        return self._rows


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def analyse(rows: list[dict]) -> None:
    """Print a human-readable analysis of the synthetic benchmark data.

    Args:
        rows: Synthetic result rows.

    Returns:
        None
    """
    from src.benchmark.visualize import _aggregate

    agg = _aggregate(rows)
    scenario_order = [name for name, _ in SCENARIOS]

    col = 32
    print(f"\n{'Scenario':<{col}} {'Avg Score':>10} {'Avg Cost/File':>14} "
          f"{'Avg Time/File':>14} {'Avg Errors':>11}")
    print("-" * (col + 10 + 14 + 14 + 11 + 5))

    for name in scenario_order:
        b = agg[name]
        total_errors = sum(b["avg_errors_by_category"].values())
        print(
            f"{name:<{col}} "
            f"{b['avg_quality_score']:>10.3f} "
            f"{b['avg_cost_per_file']:>14.4f} "
            f"{b['avg_time_per_file']:>14.2f} "
            f"{total_errors:>11.2f}"
        )


def check_html(html_path: Path) -> None:
    """Verify the generated HTML contains expected chart markers.

    Args:
        html_path: Path to the generated HTML file.

    Returns:
        None
    """
    text = html_path.read_text(encoding="utf-8")
    checks = {
        "Has Plotly CDN script tag": "cdn.plot.ly" in text or "plotly" in text.lower(),
        "Contains all 5 scenario names": all(name in text for name, _ in SCENARIOS),
        "Avg Cost chart present": "Avg Cost per File" in text,
        "Avg Time chart present": "Avg Time per File" in text,
        "Error breakdown chart present": "Avg Errors per File by Category" in text,
        "Score distribution chart present": "Score Distribution" in text,
        "File size > 10 KB (non-empty output)": html_path.stat().st_size > 10_000,
    }
    print("\nHTML checks:")
    all_pass = True
    for label, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}")
        if not passed:
            all_pass = False

    if not all_pass:
        print("\nSome checks FAILED.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Build synthetic data, call generate_report(), and analyse the output."""
    from src.benchmark.visualize import generate_report

    rows = build_rows()
    db = _FakeDB(rows)

    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "benchmark_results.html"

        print("Generating report...", flush=True)
        generate_report(db, output_path)
        print(f"Report written ({output_path.stat().st_size:,} bytes)")

        check_html(output_path)

    print("\n--- Synthetic benchmark analysis ---")
    analyse(rows)

    print("\nAll checks passed. visualize.py is working correctly.")


if __name__ == "__main__":
    main()
