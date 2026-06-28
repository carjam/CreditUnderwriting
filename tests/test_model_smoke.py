from __future__ import annotations

import pytest

from tests._model_reference import run_reference_model


@pytest.mark.smoke
def test_reference_model_smoke_quality() -> None:
    """
    Fast check that core sklearn model behavior is healthy in this environment.
    """
    run = run_reference_model()
    assert run.n_test > 100
    assert run.roc_auc >= 0.97
    assert run.accuracy >= 0.93
