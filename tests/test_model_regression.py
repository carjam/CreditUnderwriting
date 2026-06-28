from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._model_reference import run_reference_model


BASELINE_PATH = Path(__file__).parent / "baselines" / "model_quality_baseline.json"


@pytest.mark.regression
def test_model_quality_against_baseline() -> None:
    """
    Regression guardrail:
    If environment, dependency, or logic changes degrade model quality below
    agreed baseline floors, this test fails.
    """
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    run = run_reference_model()

    assert run.roc_auc >= baseline["roc_auc_min"]
    assert run.accuracy >= baseline["accuracy_min"]
