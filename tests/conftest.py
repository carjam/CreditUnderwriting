from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-notebook",
        action="store_true",
        default=False,
        help="Run end-to-end notebook execution (slow; several minutes).",
    )


def _should_run_notebook_e2e(config: pytest.Config) -> bool:
    if os.environ.get("SKIP_NOTEBOOK_E2E", "").lower() in ("1", "true", "yes"):
        return False
    if config.getoption("--run-notebook"):
        return True
    if os.environ.get("RUN_NOTEBOOK_E2E", "").lower() in ("1", "true", "yes"):
        return True
    if os.environ.get("CI", "").lower() in ("true", "1", "yes"):
        return True
    return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _should_run_notebook_e2e(config):
        return
    skip_nb = pytest.mark.skip(
        reason=(
            "Notebook E2E skipped (use --run-notebook, set RUN_NOTEBOOK_E2E=1, or run in CI; "
            "CI: set SKIP_NOTEBOOK_E2E=1 to disable)"
        )
    )
    for item in items:
        if "notebook_e2e" in item.keywords:
            item.add_marker(skip_nb)


@pytest.fixture(autouse=True)
def deterministic_seed() -> None:
    random.seed(42)
    np.random.seed(42)
