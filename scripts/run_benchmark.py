# CLI entry point for the MediFlow translation benchmark.
#
# Runs all 4 translation scenarios (or a single named one) against the eval
# documents and persists results to benchmark.db. Can also generate an
# interactive HTML visualisation report from existing DB results.
#
# Usage:
#   python scripts/run_benchmark.py                                # all scenarios
#   python scripts/run_benchmark.py --scenario s1_google_translate # one scenario
#   python scripts/run_benchmark.py --visualize                    # charts only
#
# Environment variables (via .env or shell):
#   GOOGLE_CLOUD_PROJECT  — required for all Gemini and Google Translate calls
#   GOOGLE_CLOUD_LOCATION — optional, defaults to "global"

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import os

# Validate required environment variables before any heavy imports.
if "GOOGLE_CLOUD_PROJECT" not in os.environ:
    print("ERROR: GOOGLE_CLOUD_PROJECT is not set. Add it to .env or your shell environment.")
    sys.exit(1)

from src.benchmark.db import BenchmarkDB

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_DB_PATH = _PROJECT_ROOT / "benchmark.db"
_REPORT_PATH = _PROJECT_ROOT / "benchmark_results.html"


_SCENARIO_REGISTRY: dict[str, tuple[str, str]] = {
    "s1_google_translate": ("src.benchmark.scenarios.s1_google_translate", "GoogleTranslateScenario"),
    "s2_gemini_flash_zeroshot": ("src.benchmark.scenarios.s2_gemini_flash_zeroshot", "GeminiFlashZeroShotScenario"),
    "s3_gemini_pro_dspy_predict": ("src.benchmark.scenarios.s3_gemini_pro_dspy_predict", "GeminiProDSPyPredictScenario"),
    "s4_gemini_pro_bootstrap": ("src.benchmark.scenarios.s4_gemini_pro_bootstrap", "GeminiProBootstrapScenario"),
    "s5_gemini_flash_dspy_predict": ("src.benchmark.scenarios.s5_gemini_flash_dspy_predict", "GeminiFlashDSPyPredictScenario"),
    "s6_gemini_flash_cot": ("src.benchmark.scenarios.s6_gemini_flash_cot", "GeminiFlashCoTScenario"),
}
_DEFAULT_SCENARIOS = [
    "s1_google_translate",
    "s2_gemini_flash_zeroshot",
    "s3_gemini_pro_dspy_predict",
    "s4_gemini_pro_bootstrap",
    "s5_gemini_flash_dspy_predict",
    "s6_gemini_flash_cot",
]


def _build_scenarios(names: list[str] | None) -> list:
    """Instantiate only the requested scenarios (or all if names is None).

    Imports each scenario module on demand so that DSPy and Vertex AI are not
    loaded when running --visualize or encountering an early argument error.

    Args:
        names: List of scenario name strings to instantiate, or None for all.

    Returns:
        List of instantiated TranslationScenario objects in registry order.
    """
    import importlib

    keys = names if names is not None else _DEFAULT_SCENARIOS
    scenarios = []
    known = ", ".join(_SCENARIO_REGISTRY.keys())
    for k in keys:
        if k not in _SCENARIO_REGISTRY:
            raise KeyError(
                f"Unknown scenario {k!r}; expected one of: {known}"
            )
        module_path, class_name = _SCENARIO_REGISTRY[k]
        module = importlib.import_module(module_path)
        scenarios.append(getattr(module, class_name)())
    return scenarios


def _parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments.

    Args:
        None

    Returns:
        Parsed argparse.Namespace with fields: scenario (str|None), visualize (bool).
    """
    parser = argparse.ArgumentParser(
        description="Run the MediFlow translation benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_benchmark.py\n"
            "  python scripts/run_benchmark.py --scenario s1_google_translate\n"
            "  python scripts/run_benchmark.py --visualize"
        ),
    )
    parser.add_argument(
        "--scenario",
        metavar="NAME",
        help="Run only this scenario (e.g. s1_google_translate). Omit to run all.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate benchmark_results.html from existing DB results; skip all scenarios.",
    )
    return parser.parse_args()


def _run_visualize(db: BenchmarkDB) -> None:
    """Generate the HTML visualisation report from existing DB results.

    Args:
        db: Open database instance to read results from.

    Returns:
        None

    Raises:
        ImportError: If the visualize module (Step 13) has not been implemented yet.
    """
    try:
        from src.benchmark.visualize import generate_report
    except ImportError as exc:
        raise ImportError(
            "Visualisation module not found. Implement Step 13 (src/benchmark/visualize.py) first."
        ) from exc

    generate_report(db, _REPORT_PATH)
    print(f"Report written to {_REPORT_PATH}")


def main() -> None:
    """Entry point: parse args, load data, and run the requested benchmark action.

    Runs all scenarios sequentially unless --scenario filters to one, or
    --visualize skips translation and generates charts from existing DB rows.

    Args:
        None

    Returns:
        None
    """
    args = _parse_args()

    db = BenchmarkDB(_DB_PATH)
    db.create_tables()

    if args.visualize:
        _run_visualize(db)
        return

    if args.scenario is not None and args.scenario not in _SCENARIO_REGISTRY:
        known = ", ".join(_SCENARIO_REGISTRY.keys())
        print(f"ERROR: Unknown scenario {args.scenario!r}. Known scenarios: {known}")
        sys.exit(1)

    from src.benchmark.dataset import load_eval_docs, load_train_docs
    from src.benchmark.runner import run_scenario

    names_to_run = [args.scenario] if args.scenario else None
    scenarios_to_run = _build_scenarios(names_to_run)

    eval_docs = load_eval_docs(_DATA_DIR)
    train_docs = load_train_docs(_DATA_DIR)

    print(f"Eval docs: {len(eval_docs)} (10 per type × 4 types) | Train docs: {len(train_docs)} (8 per type × 4 types)")
    print(f"DB: {_DB_PATH}")
    print()

    for scenario in scenarios_to_run:
        print(f"=== {scenario.name} ===")
        run_scenario(
            scenario=scenario,
            train_docs=train_docs,
            eval_docs=eval_docs,
            db=db,
        )
        print()

    print("Done.")


if __name__ == "__main__":
    main()
