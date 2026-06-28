# Portfolio Credit Risk Decisioning

This document describes how this repository elevates an existing underwriting ML notebook into a **portfolio-oriented credit risk decisioning narrative**.

## What the Decisioning Layer Adds

| Layer | Implementation | Notes |
|-------|---------------|-------|
| **SMOTE fix** | Split-first, SMOTE on training fold only | Corrects data leakage; test set reflects real 81.1% good-loan prevalence |
| **Probability calibration** | `CalibratedClassifierCV` (isotonic, cv=3) | Improves Brier score 0.0066 → 0.0055; reliability diagrams rendered in notebook |
| **Decision rules** | `decisions()` — approve / review / decline | Thresholds: approve ≥ 0.65, review 0.50–0.65, decline < 0.50 |
| **Risk tiers** | `risk_tier()` — prime / near-prime / subprime | Cutoffs: prime ≥ 0.70, near-prime 0.50–0.70, subprime < 0.50 |
| **Tier calibration** | `calibrate_tiers()` | Returns empirical DR per tier; validates monotone ordering |
| **Threshold simulation** | `threshold_sweep()` | Sweeps approval cutoffs; reports approval rate and default rate among approved |
| **Break-even DR** | `break_even_default_rate()` | Revenue / (LGD × avg_loan); annotated as red dashed line on threshold sweep chart |
| **Lifetime EL** | `lifetime_expected_loss()` | Accumulates PD × LGD × EAD_t over loan tenor (IFRS 9 lifetime ECL structure) |
| **Portfolio UL** | `unexpected_loss()` | Vasicek single-factor std-dev of aggregate loss (Basel retail ρ=0.12) |
| **SHAP** | Global beeswarm + one-loan waterfall | Feature importance and individual-loan explanation for adverse action context |

## Where to Run It

Open `Credit_Underwriting_Decisioning-Lending_Club.ipynb` and run all cells from a clean kernel. The Portfolio Credit Risk Decisioning Layer at the end depends on `best_model`, `calibrated_model`, `X_train`, `X_test`, and `y_test` being defined by the preceding cells.

## Measured Portfolio Outcomes (Corrected Pipeline)

The following are from the actual XGBoost + calibration run on the honest test set (n=1,189 loans, 81.1% good-loan prevalence):

| Metric | Value |
|---|---|
| Approval rate at threshold 0.65 | 81.3% (967 loans) |
| Default rate among approved | 0.52% |
| Break-even default rate | 66.7% |
| All thresholds 0.35–0.92 profitable? | Yes — observed DR never approaches break-even |
| Prime tier (≥0.70): n / DR | 965 / 0.52% |
| Near-prime (0.50–0.70): n / DR | 6 / 50.0% |
| Subprime (<0.50): n / DR | 218 / 99.5% |
| Lifetime EL/loan (36 months) | $36 |
| Portfolio UL (ρ=12%) | $108,522 (1.1% of notional) |

> **Interpretation note:** The near-binary score distribution (only 6 loans in the [0.50, 0.70] band) and the near-zero observed DR among approved loans are almost certainly artifacts of feature leakage from `sub_grade` / `int_rate` — see `docs/MODEL_CARD.md`. The capital framing (EL, UL, break-even) is structurally correct and reusable; the input DR just needs to come from a leakage-free model for the numbers to be meaningful.

## Reuse, Not Rebuild

The decisioning functions in `src/decisioning.py` are pure, stateless functions that consume any array of `p_good` scores. Replacing the underlying model (after the leakage investigation) requires no changes to the policy layer.

## CECL and Compliance Context

The EL and UL outputs are **directional internal metrics**, not accounting provisions. See `docs/CECL_MAPPING.md` for the full ASC 326 / IFRS 9 gap map, and `docs/ADVERSE_ACTION.md` for the ECOA/Reg B adverse action workflow.
