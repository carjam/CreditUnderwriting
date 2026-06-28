# Model Card (Prototype)

## Purpose

Demonstrate **credit risk decisioning** on top of classifier outputs: tiers, approve/review/decline, threshold tradeoffs, calibration, SHAP views, and a directional capital lens. **Not** a production underwriting or capital system.

## Scope

| In scope | Out of scope |
|----------|----------------|
| Notebook-based training/eval on a fixed Lending Club–style sample | Live scoring API, model registry, or governance sign-off |
| Policy logic in `src/decisioning.py` + `config/policy.default.yaml` | Calibrated PD as regulatory default probability |
| Reproducibility via pinned `requirements.txt` and tests | Out-of-time validation on current vintages |
| Break-even DR, lifetime EL, and UL helpers (illustrative) | CECL/IFRS 9 compliant ACL computation |

## Data

- **Default artifact:** `data/loans.csv` (public Lending Club sample mirror; **6,305** raw rows, **5,942** after filtering to binary loan_status; **2014-era** issue dates).
- **Override:** set `LENDING_CLUB_DATA_PATH` before the notebook load cell.
- **Label:** binary `loan_status` mapped to Fully Paid (1) vs Charged Off (0) after filtering.
- **Class balance:** ~81.1% Fully Paid / 18.9% Charged Off in the original data.

## Pipeline methodology

### SMOTE and train/test split

SMOTE is applied **after** the stratified train/test split, exclusively to the training fold:

```
X, y  (original, imbalanced — 81.1% good)
  └─ train_test_split (stratified, 80/20, random_state=42)
        ├─ X_train, y_train  →  SMOTE  →  X_train_smote (balanced, used to fit models)
        └─ X_test,  y_test   (original distribution — used ONLY for evaluation)
```

**Previous state (corrected):** The original notebook applied SMOTE to the full dataset before splitting. This caused synthetic minority-class samples—generated from the neighborhood of test-set observations—to leak into training, inflating apparent performance. The corrected order means `X_test` reflects real loan prevalence.

### Probability calibration

`CalibratedClassifierCV(best_model, cv=3, method='isotonic')` is fit on the SMOTE-balanced training data. The result (`calibrated_model`) is used in the Portfolio Decisioning Layer. Reliability diagrams (before and after) are rendered in the notebook's Calibration Assessment section.

## Measured performance (corrected pipeline)

All metrics are on the **honest, unaugmented test set** (81.1% good loans) using XGBoost as `best_model`.

| Metric | XGBoost (uncalibrated) | XGBoost + isotonic calibration |
|--------|------------------------|-------------------------------|
| Accuracy | 99.3% | — |
| ROC-AUC | 0.9991 | 0.9981 |
| Precision (Fully Paid) | 99.2% | 99.2% |
| Recall (Fully Paid) | 100.0% | 99.9% |
| Recall (Charged Off) | **96.4%** | 96.4% |
| Brier score | 0.0066 | **0.0055** |

**Logistic baseline (reference):** ROC-AUC ~0.66 — consistent with the original notebook observation; logistic regression has weaker separation on this feature set.

**Cross-validation (training fold only):** CV accuracy of ~91–92% is the score that appears in the notebook's tuning cells. It is computed on the SMOTE-balanced training folds—not on the held-out test set. The final test-set accuracy (99.3%) is higher because the tuned XGBoost fits the 2014-era patterns very tightly.

### Near-binary score distribution — important caveat

The XGBoost model assigns scores in a near-binary pattern:

| Score band | Count (test set, n=1,189) | Observed DR |
|---|---|---|
| ≥ 0.70 (prime) | 965 | **0.52%** |
| 0.50–0.70 (near-prime) | **6** | 50.0% |
| < 0.50 (subprime) | 218 | **99.5%** |

Only 6 of 1,189 test loans fall in the [0.50, 0.70] mid-range, and 99.6% of predictions cluster at p ≥ 0.95 or p ≤ 0.05. A genuine credit model operating on a priori features should produce a spread of intermediate probabilities. This bimodal distribution is a strong indicator of **feature leakage** (see Known Limitations).

## Expected loss and capital helpers

| Helper | Function | What it computes |
|--------|----------|-----------------|
| Single-period EL | `simple_expected_loss_per_approved` | DR × LGD × avg_loan — annual snapshot |
| Lifetime EL | `lifetime_expected_loss` | Σ conditional_PD_t × LGD × EAD_t over loan tenor |
| Unexpected Loss | `unexpected_loss` | Portfolio loss std-dev via Vasicek single-factor (ρ=0.12) |
| Break-even DR | `break_even_default_rate` | Revenue / (LGD × avg_loan); threshold above which approvals destroy value |

At the policy threshold (approve_min=0.65) on the corrected test set:

| Metric | Value |
|---|---|
| Approval rate | 81.3% (967 loans) |
| Default rate among approved | 0.52% |
| Break-even DR | **66.7%** (revenue $3k / LGD 45% × $10k) |
| Single-period EL/loan | $23 |
| Lifetime EL/loan (36 months) | $36 |
| Net margin/loan | $2,964 (illustrative revenue $3,000 − lifetime EL) |
| Portfolio UL (ρ=12%) | $108,522 (1.1% of notional) |
| RAROC | ~26× |

The RAROC of 26× and DR of 0.52% are **inconsistent with real-world consumer lending** (typical RAROC 1.2–2.5×, typical prime consumer DR 1–4%) and are likely artifacts of feature leakage rather than genuine model performance. See Known Limitations.

All values are **illustrative** (assumed avg_loan, LGD=0.45 Basel II floor, tenor=36mo). They are **not** CECL/IFRS 9 compliant. See `docs/CECL_MAPPING.md`.

## LGD assumption

`lgd = 0.45` is the **Basel II/III supervisory floor** for senior unsecured consumer credit. Actual LGD varies:

| Loan type | Typical LGD range |
|-----------|------------------|
| Secured (auto, mortgage) | 10–30% |
| Unsecured personal (with collections) | 40–65% |
| Charged-off, no collections | 70–90% |

Replace with portfolio-specific empirical recovery data for any production use.

## Known limitations

### Feature leakage hypothesis (primary concern)

The near-perfect AUC (~0.999) and near-binary score distribution are inconsistent with a model trained solely on a priori borrower attributes. Two features are the most likely sources:

- **`sub_grade`:** Lending Club assigns sub_grade after reviewing the full application including FICO score, debt, and income. Crucially, the platform's grade reflects its *own expected credit loss* — making sub_grade a proxy for the label itself. The model is partially predicting the outcome using the platform's own prediction.
- **`int_rate`:** 96% correlated with sub_grade and similarly set with knowledge of expected risk. Including both sub_grade and int_rate gives the model two near-identical proxies for the outcome.

**Recommended test:** Re-run with `sub_grade` and `int_rate` removed from the feature set. If AUC drops to 0.65–0.75, the leakage hypothesis is confirmed. Features to retain: `annual_inc`, `dti`, `emp_length`, `inq_last_6mths`, `revol_util`, `open_acc`, `term`, `loan_amnt`, `home_ownership_*`, `purpose_*`.

### Other limitations

- Calibration improves Brier score from 0.0066 → 0.0055 but cannot fix leakage-inflated probabilities; calibrated scores should still be treated skeptically until leakage is ruled out.
- **No** CECL/IFRS 9/regulatory capital treatment — see `docs/CECL_MAPPING.md`.
- Bundled data is **2014-era**; no claim about current market or any live portfolio.
- No out-of-time validation; all quality figures are from the same origination year as training.
- No model drift monitoring, override tracking, or governance documentation for production use.

## Adverse action and review disposition

Loans in the **"review"** bucket require a documented disposition process for ECOA/Reg B compliance. See `docs/ADVERSE_ACTION.md` for workflow guidance and required disclosures.

## Reproduction

1. `pip install -r requirements.txt`
2. Run `Credit_Underwriting_Decisioning-Lending_Club.ipynb` end-to-end (clean kernel), or `pytest --run-notebook` (slow).
3. Record **git commit** and **data file** when reporting metrics.
