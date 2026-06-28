"""Notebook integrity and (optional) end-to-end execution checks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Credit_Underwriting_Decisioning-Lending_Club.ipynb"
DATA_CSV = ROOT / "data" / "loans.csv"

# Allow headroom beyond ExecutePreprocessor.timeout for kernel startup and teardown.
_SUBPROCESS_TIMEOUT_SEC = 960
_NBCONVERT_CELL_TIMEOUT_SEC = 900


@pytest.mark.unit
def test_notebook_file_is_valid_nbformat() -> None:
    """Catches corrupt or schema-invalid notebooks before any execution."""
    assert NOTEBOOK.is_file(), f"Missing notebook: {NOTEBOOK}"
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)


@pytest.mark.unit
def test_bundled_dataset_present_for_notebook() -> None:
    """The notebook expects data/loans.csv when run from the repo root."""
    assert DATA_CSV.is_file(), (
        f"Missing {DATA_CSV}; clone should include bundled sample or set LENDING_CLUB_DATA_PATH"
    )


@pytest.mark.notebook_e2e
def test_lending_club_notebook_executes_end_to_end(tmp_path: Path) -> None:
    """
    Full `nbconvert --execute` from repo root so paths like data/loans.csv resolve.

    Skipped locally unless --run-notebook, RUN_NOTEBOOK_E2E=1, or CI is set.
    """
    assert NOTEBOOK.is_file()
    assert DATA_CSV.is_file()

    out_name = "Credit_Underwriting_Decisioning-Lending_Club.executed.ipynb"
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(NOTEBOOK),
        "--output-dir",
        str(tmp_path),
        "--output",
        out_name,
        f"--ExecutePreprocessor.timeout={_NBCONVERT_CELL_TIMEOUT_SEC}",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_SEC,
        check=False,
    )
    executed = tmp_path / out_name
    assert proc.returncode == 0, (
        f"nbconvert failed (exit {proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert executed.is_file(), f"Expected output notebook at {executed}"
