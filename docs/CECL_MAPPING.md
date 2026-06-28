# CECL / IFRS 9 Gap Map

This document maps each ASC 326 (CECL) and IFRS 9 requirement to the current
state of the repo.  Its purpose is to prevent stakeholders from treating the
repo's illustrative EL figures as CECL-compliant Allowance for Credit Losses (ACL).

---

## ASC 326 (CECL) — Requirement-by-requirement status

| ASC 326 Requirement | Repo Implementation | Status | Gap / Action Required |
|---------------------|---------------------|--------|-----------------------|
| **Pool segmentation** — pools of financial assets sharing similar risk characteristics | Risk tiers (prime / near-prime / subprime) based on P(good) | Partial | Tiers are score-based, not characteristic-based (grade, term, product type). Production pools should be segmented by origination channel, loan purpose, and vintage. |
| **Lifetime loss estimate** — measure credit losses over the contractual term | `lifetime_expected_loss(pd_annual, lgd, avg_loan_amount, tenor_months)` | Partial | Uses a simplified linear-amortization EAD schedule and constant annual PD. Production requires period-specific conditional PD curves (survival curves) and EAD by scheduled amortization. |
| **Reasonable and supportable forecasts** — incorporate macro forward-looking information | Not implemented | **Missing** | No macro variables (unemployment, GDP, credit spreads) are fed into the EL model. A CECL-compliant ACL model must adjust PD/LGD estimates using an economic scenario or multiple weighted scenarios. |
| **Reversion to historical loss rates** — beyond the supportable forecast horizon, revert to long-run historical average | Not implemented | **Missing** | Required for any period beyond the model's forecast horizon. Implement as: `EL_t = EL_modeled` for t ≤ N_forecast; `EL_t = EL_historical` for t > N_forecast. |
| **Discounted cash flow (DCF) or loss-rate method** — measurement basis must be documented | Simplified: `PD × LGD × EAD` (non-discounted) | Partial | ASC 326 allows both methods; if using DCF, cash flows must be discounted at the effective interest rate. The current single-period helper omits discounting. |
| **Q-factor (qualitative) adjustments** — overlays for factors not captured in quantitative models | Not implemented | **Missing** | Regulators expect overlays for: changes in lending standards, external competition, concentration risk, model uncertainty. |
| **Vintage / cohort tracking** — performance should be tracked by origination cohort | Not implemented | **Missing** | Critical for observing seasoning curves and adjusting loss rates for newer vintages. |
| **Disclosure** — required disclosures about credit quality, ACL roll-forward, vintage analysis | Not implemented | **Missing** | ASC 326-20-50 requires extensive tabular disclosures. Out of scope for this prototype. |

---

## IFRS 9 — Three-Stage ECL Model

| IFRS 9 Stage | Description | Repo Equivalent | Gap |
|---|---|---|---|
| **Stage 1** — no significant increase in credit risk (SICR) since origination | 12-month ECL | `simple_expected_loss_per_approved` (annual single-period) | Missing: 12-month ECL should be PD_12mo × LGD × EAD, where PD_12mo is the through-the-cycle 12-month PD. Current PD is point-in-time from model output. |
| **Stage 2** — significant increase in credit risk | Lifetime ECL | `lifetime_expected_loss` | Partial: function exists but uses constant annual PD rather than a PD term structure. SICR trigger not defined. |
| **Stage 3** — credit-impaired | Lifetime ECL (individual assessment) | Not implemented | No individual impairment assessment or write-off criteria defined. |
| **Discount rate** — effective interest rate for DCF | Not implemented | **Missing** | IFRS 9 requires discounting ECL at the instrument's effective interest rate. |
| **Macro scenarios** (probability-weighted) | Not implemented | **Missing** | IAS 36 / IFRS 9 require at least three economic scenarios (base, upside, downside) with probability weights. |

---

## What the Repo EL Numbers Represent

The numbers produced by `simple_expected_loss_per_approved` and
`lifetime_expected_loss` are best understood as **directional internal stress
metrics**, not accounting provisions:

- They use a **point-in-time** model PD (from the ML classifier) rather than a
  **through-the-cycle** PD calibrated to long-run default rates.
- They apply a fixed **Basel II supervisory LGD** (0.45), not an empirically
  estimated recovery rate.
- They incorporate **no macro forecasts** or scenario weights.
- They are **not discounted** at the effective interest rate.

For regulatory reporting or financial statement ACL purposes, a full CECL
methodology with the above components must be implemented by qualified
accounting and credit risk professionals.

---

## Recommended Path to CECL Alignment

1. **Probability calibration** — ensure PD outputs are actuarially calibrated
   (reliability diagram, Hosmer-Lemeshow test). See `docs/MODEL_CARD.md`.
2. **PD term structure** — estimate conditional survival curves (Kaplan-Meier or
   parametric hazard model) by vintage to produce period-specific PDs.
3. **LGD model** — replace the supervisory floor with a portfolio-specific
   recovery rate model (regression on collateral, time-to-recovery, recovery channel).
4. **EAD schedule** — implement actual amortization schedules per loan contract
   rather than linear-amortization approximation.
5. **Macro overlay** — build a macro-conditional PD adjustment using a transition
   matrix or regression on leading indicators (unemployment, credit spreads).
6. **Segmentation** — pool loans by product type, origination channel, and vintage;
   run the EL model by pool rather than as a single aggregate.
7. **Vintage / cohort tracking** — track cumulative default rates by origination
   cohort to validate the model against observed loss experience.
8. **Q-factors** — document qualitative overlays approved by credit committee.
9. **Audit trail** — maintain a complete ACL roll-forward with supporting
   calculation workpapers; subject to external audit under ASC 326 / IFRS 9.
