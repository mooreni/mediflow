"""Pytest configuration for the MediFlow test suite.

Defines the --run-integration flag that gates tests which call the real
Gemini API. Without the flag those tests are automatically skipped so the
normal fast unit-test suite does not require API credentials.
"""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the --run-integration CLI option.

    Args:
        parser: The pytest argument parser to extend.

    Returns:
        None
    """
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that call real LLM APIs (requires GOOGLE_CLOUD_PROJECT).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip all tests marked @pytest.mark.integration unless --run-integration is set.

    Args:
        config: The pytest configuration object.
        items:  The collected test items to inspect and potentially skip.

    Returns:
        None
    """
    if not config.getoption("--run-integration"):
        skip_marker = pytest.mark.skip(reason="pass --run-integration to run API tests")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_marker)
