# Adverse Action and Review Disposition

This document describes the required workflow for handling loan applications
that receive a **"review"** or **"decline"** decision from the underwriting model,
as required by ECOA (Equal Credit Opportunity Act) and Regulation B.

---

## Regulatory Context

**ECOA / Regulation B (12 CFR Part 202)** requires that:

1. Applicants who are denied credit, or approved on less favorable terms than
   requested, receive an **adverse action notice** within 30 days.
2. The notice must state the **specific reasons** for the adverse action (or
   inform the applicant of their right to request reasons).
3. Reasons must be **legitimate credit factors** — not protected characteristics
   (race, color, religion, national origin, sex, marital status, age, receipt of
   public assistance, or exercise of ECRA rights).

**Reg B § 202.9** — specific reasons must be drawn from the actual decisioning
factors that most influenced the denial, not generic boilerplate.

---

## Decision Outcomes and Required Actions

| Decision | Regulatory Status | Required Action | Timing |
|---|---|---|---|
| **Approve** | Standard approval | Issue commitment letter; disclose APR and terms (Reg Z / TILA) | Before consummation |
| **Review** | Counteroffer or pending | Human underwriter must review and either approve, approve with conditions, or decline within 30 days | 30 days from application |
| **Decline** | Adverse action | Issue ECOA adverse action notice with specific reasons | 30 days from application |

---

## Review Bucket Disposition Workflow

The **"review"** tier (`review_min ≤ p_good < approve_min`, currently 0.50–0.65)
sits in the ambiguous middle.  Without a documented disposition process, every
review case that ultimately results in a decline triggers adverse action obligations.

### Recommended Workflow

```
Applicant → Model scores p_good ∈ [0.50, 0.65) → "Review" flag
                │
                ▼
    Human Underwriter Reviews:
    ├── Additional documentation requested? (pay stubs, bank statements)
    ├── SHAP explanation consulted for top adverse factors
    ├── Compensating factors checked (low DTI, long employment, large down payment)
    └── Decision:
         ├── Approve → Issue commitment; log override reason
         ├── Approve-with-conditions → Counter-offer (reduced amount, higher rate)
         │   └── If applicant declines counter → Adverse action notice
         └── Decline → Adverse action notice with specific reasons (see below)
```

### Override Logging Requirements

All review outcomes (especially approvals and conditions) must be logged with:

- Application ID
- Model decision (review)
- Human disposition (approve / decline / counter-offer)
- Override reason (free text, selected from approved taxonomy)
- Reviewer ID and timestamp
- Final terms (if approved)

Override rates and patterns should be **monitored quarterly** for disparate impact
signals (approval rate differentials by protected class proxies).

---

## Adverse Action Notice — Required Content

Per Reg B § 202.9(b)(2), written adverse action notices must include:

1. **Statement of action taken** — "We are unable to offer you credit on the terms you requested."
2. **Name and address of creditor**
3. **ECOA Notice** — "The federal Equal Credit Opportunity Act prohibits creditors from discriminating..."
4. **Statement of specific reasons** OR disclosure that applicant may request specific reasons within 60 days

### Mapping SHAP Factors to Adverse Action Reasons

The repo generates SHAP explanations (`shap.TreeExplainer`) for individual loans.
Top negative SHAP contributors for declined/review loans are the natural candidates
for adverse action reason codes.

**Approved adverse action reason categories (illustrative):**

| SHAP-contributing feature | Reg B reason code (example) |
|---|---|
| `sub_grade` (low credit tier) | Credit score below minimum threshold |
| `annual_inc` (low income) | Insufficient income to service requested debt |
| `dti` (high debt-to-income) | Debt-to-income ratio too high |
| `inq_last_6mths` (recent inquiries) | Too many recent credit inquiries |
| `revol_util` (high utilization) | Proportion of balances to credit limits too high |
| `delinq_2yrs` (past delinquency) | Delinquent past or present credit obligations |

**Important:** Adverse action reasons must reflect the applicant's actual credit
profile — not model internals.  A reason derived from a SHAP value is valid only
if it corresponds to a verifiable fact about the applicant.

---

## Fair Lending Monitoring

The decisioning layer should be monitored for **disparate impact** on protected
classes.  Under the **80% (four-fifths) rule** (EEOC / CFPB guidance):

```
Approval rate (protected group) / Approval rate (control group) < 0.80 → Potential disparate impact
```

**Minimum monitoring cadence:** Quarterly.

**Required actions if disparate impact is detected:**
1. Root-cause analysis (feature distribution differences vs. model coefficients)
2. Model recalibration or threshold adjustment
3. Legal / compliance review
4. Possible HMDA / CRA reporting implications

---

## Scope Note

This document describes the **regulatory framework** applicable to a production
credit decisioning system built on this prototype's logic.  The prototype itself
does not implement adverse action notice generation, override logging, or fair
lending monitoring.  These are required additions before any production deployment.

See also:
- `docs/CECL_MAPPING.md` — accounting / financial reporting requirements
- `docs/MODEL_CARD.md` — model assumptions, calibration status, known limitations
