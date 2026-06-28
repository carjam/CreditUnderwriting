import json
from pathlib import Path

path = Path(__file__).resolve().parent.parent / "Credit_Underwriting_Decisioning-Lending_Club.ipynb"
nb = json.loads(path.read_text(encoding="utf-8"))

md1 = """## Portfolio Credit Risk Decisioning Layer

This section **does not retrain models**. It reuses the fitted classifier from above (`best_model`, typically XGBoost after the full notebook run) to demonstrate:

- **Approval thresholds** and **risk tiers** (prime / near-prime / subprime)
- **Mapped decisions**: approve / review / decline
- **Threshold simulation**: approval rate vs. default rate among approved (directional)
- **SHAP** global importance and one-loan explanation
- **Simple capital framing**: illustrative expected loss using assumed average loan size and LGD

*Note: Train/test splits use SMOTE-balanced features from the assignment pipeline; portfolio rates are **illustrative** of tradeoffs, not live-book statistics.*"""

code1 = r'''# --- Portfolio decisioning (reuse trained classifier; no retraining) ---
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd()
for p in (ROOT, ROOT.parent):
    if (p / "src" / "decisioning.py").exists():
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        break

from src.decisioning import (
    decisions,
    portfolio_notional_exposure,
    risk_tier,
    simple_expected_loss_per_approved,
    threshold_sweep,
)

decision_classifier = best_model

# P(class=1) = P(Fully Paid) = "good"
p_good = decision_classifier.predict_proba(X_test)[:, 1]
y_arr = np.asarray(y_test).ravel()

tier = risk_tier(p_good, prime_cut=0.70, near_cut=0.50)
action = decisions(p_good, approve_min=0.65, review_min=0.50)

decision_df = pd.DataFrame({
    "p_good": p_good,
    "actual_good": y_arr,
    "tier": tier,
    "decision": action,
})
print(decision_df["tier"].value_counts())
print(decision_df["decision"].value_counts())

# Threshold simulation
thresholds = np.linspace(0.35, 0.92, 30)
sim = threshold_sweep(p_good, y_arr, thresholds)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(sim["threshold"], sim["approval_rate"], label="Approval rate")
ax.plot(sim["threshold"], sim["default_rate_among_approved"], label="Default rate (among approved)")
ax.set_xlabel("Approval threshold on P(good)")
ax.set_ylabel("Rate")
ax.legend()
ax.grid(True, alpha=0.3)
plt.title("Directional threshold tradeoff (test set)")
plt.tight_layout()
plt.show()

# --- Simple capital framing (illustrative; clarity > precision) ---
AVG_LOAN = 15_000.0
LGD = 0.45

pick = sim.loc[(sim["threshold"] - 0.65).abs().idxmin()]
el_per_loan = simple_expected_loss_per_approved(
    pick["default_rate_among_approved"], AVG_LOAN, LGD
)
notional = portfolio_notional_exposure(int(pick["n_approved"]), AVG_LOAN)

print("Illustrative policy point near threshold 0.65:")
print(pick.to_string())
print(f"Expected loss per approved loan (directional): ${el_per_loan:,.0f}")
print(f"Notional approved (n × avg loan): ${notional:,.0f}")
'''

code2 = r'''# --- SHAP explainability (global + one loan) ---
import shap

# Sample for speed; use DataFrame so feature names flow to plots
X_sample = X_test.iloc[:800] if hasattr(X_test, "iloc") else X_test

# SHAP TreeExplainer requires the base tree model, not a CalibratedClassifierCV wrapper
shap_model = decision_classifier
if hasattr(decision_classifier, 'calibrated_classifiers_'):
    # Extract base estimator from CalibratedClassifierCV
    shap_model = decision_classifier.calibrated_classifiers_[0].estimator

explainer = shap.TreeExplainer(shap_model)

# Prefer new SHAP API (callable explainer); fall back to shap_values for older versions
try:
    shap_explanation = explainer(X_sample)
    shap.plots.beeswarm(shap_explanation, max_display=15)
    shap.plots.waterfall(shap_explanation[0], max_display=14)
except Exception:
    raw = explainer.shap_values(X_sample)
    if isinstance(raw, list):
        raw = raw[1]
    shap.summary_plot(raw, X_sample, max_display=15, show=True)
    shap.waterfall_plot(
        shap.Explanation(
            values=raw[0],
            base_values=explainer.expected_value,
            data=X_sample.iloc[0].values,
            feature_names=list(X_sample.columns),
        ),
        max_display=14,
        show=True,
    )
'''


def to_source(s: str) -> list:
    lines = s.split("\n")
    return [line + "\n" for line in lines]


insert_at = 95
new_cells = [
    {"cell_type": "markdown", "metadata": {}, "source": to_source(md1)},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": to_source(code1)},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": to_source(code2)},
]

for i, c in enumerate(new_cells):
    c.setdefault("id", f"portfolio-decisioning-{i}")

nb["cells"] = nb["cells"][:insert_at] + new_cells + nb["cells"][insert_at:]
path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Inserted", len(new_cells), "cells at", insert_at, "total cells", len(nb["cells"]))
