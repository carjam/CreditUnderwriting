# Runbook

## Environment

- Python: `3.13.x` tested locally; CI uses `3.12`.
- Install: `pip install -r requirements.txt`
- Deterministic test seeds are enforced in `tests/conftest.py`.

## Data snapshot (bundled sample)

- File: `data/loans.csv`
- Source: public Lending Club sample mirror (documented in commit history)
- Raw rows: `6,305`
- Rows after filtering to binary loan_status: `5,942`
- Class balance: ~81.1% Fully Paid, 18.9% Charged Off
- Example issue date in sample: `Dec-2014`

If you use a different CSV, set `LENDING_CLUB_DATA_PATH` before running the notebook.

## Pipeline summary

```
data/loans.csv
  └─ preprocessing (encode, scale, impute)
        └─ stratified 80/20 split on original data
              ├─ X_train → SMOTE → X_train_balanced (for model fitting)
              │     └─ CalibratedClassifierCV (cv=3, isotonic) → calibrated_model
              └─ X_test (original distribution — evaluation only)
                    └─ decisioning layer (tiers, thresholds, EL, UL)
```

SMOTE is applied **after** the split. The test set is never augmented.

## Standard execution path

1. Fast checks (decisioning unit tests, smoke, regression, notebook schema):
   ```
   pytest
   ```
2. Full notebook execution check (several minutes):
   ```
   pytest --run-notebook
   ```
3. Interactive notebook:
   - Open `Credit_Underwriting_Decisioning-Lending_Club.ipynb`
   - **Kernel → Restart & Run All** (must run from clean kernel; cells depend on execution order)

## Policy-driven decisioning outside Jupyter

Use the CLI with a score file that contains `p_good`:

```
python scripts/run_decisioning.py --scores-csv <scores.csv> --policy config/policy.default.yaml
```

Optional: include `y_true` in `scores.csv` to compute default-rate, break-even comparison, lifetime EL, and portfolio UL in CLI output.

## Known performance numbers (corrected pipeline)

For reference when comparing runs. These are from the corrected SMOTE + calibration pipeline on the bundled `data/loans.csv`:

| Metric | Value |
|---|---|
| XGBoost ROC-AUC (honest test set) | 0.9991 |
| XGBoost + calibration ROC-AUC | 0.9981 |
| Recall (Charged Off, honest test) | 96.4% |
| Brier score (calibrated) | 0.0055 |
| Near-prime band [0.50–0.70] loans | 6 of 1,189 |

> The near-perfect AUC is likely a feature leakage artifact (see `docs/MODEL_CARD.md`). These numbers should not be cited as evidence of production model quality.
