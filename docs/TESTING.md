# Testing and Regression Framework

This repository includes a lightweight test framework for regression checks plus **notebook integrity** and optional **full notebook execution**.

## Test Layers

- **Unit tests** (`tests/test_decisioning.py`)
  - Validate decision-layer logic in `src/decisioning.py`:
    - risk tier mapping
    - approve/review/decline mapping
    - threshold sweep math
    - simple capital helper functions
- **Policy config tests** (`tests/test_policy_config.py`)
  - Validate `config/policy.default.yaml` parses into expected threshold ranges.
  - Verify policy application output columns and decision mapping behavior.

- **Notebook checks** (`tests/test_notebook.py`)
  - **Always run (fast):**
    - `nbformat` schema validation on `Credit_Underwriting_Decisioning-Lending_Club.ipynb`
    - presence of bundled `data/loans.csv` expected by the notebook
  - **End-to-end (slow):** runs `jupyter nbconvert --execute` from the repo root so paths like `data/loans.csv` resolve. Marked `notebook_e2e`.

- **Smoke test** (`tests/test_model_smoke.py`)
  - Verifies the local ML toolchain behaves correctly with a deterministic reference model run.
  - Uses `sklearn.datasets.load_breast_cancer` (no external data dependency).

- **Regression test** (`tests/test_model_regression.py`)
  - Compares deterministic model quality to baseline minimums in:
    - `tests/baselines/model_quality_baseline.json`
  - Fails if quality drops below floor values.

## Run Tests

Default (local): fast suite; **notebook E2E is skipped** so `pytest` stays quick.

```bash
pytest
```

Run the **full notebook** execution test (several minutes):

```bash
pytest --run-notebook
```

Or:

```bash
set RUN_NOTEBOOK_E2E=1
pytest
```

On **CI** (`CI=true`, e.g. GitHub Actions), the notebook E2E test runs automatically. To disable it in CI only:

```bash
set SKIP_NOTEBOOK_E2E=1
```

Run only smoke tests:

```bash
pytest -m smoke
```

Run only regression checks:

```bash
pytest -m regression
```

Fast checks only (exclude slow E2E by marker):

```bash
pytest -m "not notebook_e2e"
```

## Updating Baselines Intentionally

If you intentionally change baseline expectations:

1. edit `tests/baselines/model_quality_baseline.json`
2. run `pytest -m regression`
3. include a commit note explaining why the floor changed

## Notes

- Unit/smoke/regression tests validate **decisioning code correctness** and **environment/model health**.
- **Notebook E2E** validates that the main analysis notebook still executes with the bundled dataset and current dependencies (see `requirements.txt` for `nbconvert` / `nbformat`).
