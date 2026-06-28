from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.decisioning import (
    DecisionPolicy,
    apply_policy,
    break_even_default_rate,
    lifetime_expected_loss,
    portfolio_notional_exposure,
    simple_expected_loss_per_approved,
    unexpected_loss,
)


def _load_policy(path: Path) -> DecisionPolicy:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return DecisionPolicy(
        approve_min=float(raw["approve_min"]),
        review_min=float(raw["review_min"]),
        prime_cut=float(raw["prime_cut"]),
        near_cut=float(raw["near_cut"]),
        avg_loan_amount=float(raw.get("avg_loan_amount", 10000)),
        lgd=float(raw.get("lgd", 0.45)),
        tenor_months=int(raw.get("tenor_months", 36)),
        revenue_per_loan=float(raw.get("revenue_per_loan", 3000.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply underwriting decision policy to model scores.")
    parser.add_argument("--scores-csv", required=True, help="CSV containing p_good column")
    parser.add_argument("--policy", default="config/policy.default.yaml", help="YAML policy path")
    parser.add_argument("--output", default="outputs/decisions.csv", help="Output CSV path")
    args = parser.parse_args()

    scores = pd.read_csv(args.scores_csv)
    if "p_good" not in scores.columns:
        raise ValueError("scores CSV must include a `p_good` column")

    policy = _load_policy(Path(args.policy))
    decisions_df = apply_policy(scores["p_good"].to_numpy(), policy)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    decisions_df.to_csv(out_path, index=False)

    approved = decisions_df["decision"] == "approve"
    approval_rate = float(np.mean(approved))
    n_approved = int(np.sum(approved))
    default_rate_approved = float("nan")
    if "y_true" in scores.columns and n_approved > 0:
        y = scores["y_true"].to_numpy(dtype=int)
        default_rate_approved = float(np.mean(y[approved.to_numpy()] == 0))

    el_single = simple_expected_loss_per_approved(default_rate_approved, policy.avg_loan_amount, policy.lgd)
    be_dr = break_even_default_rate(policy.revenue_per_loan, policy.lgd, policy.avg_loan_amount)
    notional = portfolio_notional_exposure(n_approved, policy.avg_loan_amount)

    print(f"policy={Path(args.policy)}")
    print(f"rows={len(decisions_df)} approval_rate={approval_rate:.4f} n_approved={n_approved}")
    print(f"break_even_default_rate={be_dr:.4f}")
    if np.isnan(default_rate_approved):
        print("default_rate_among_approved=nan (provide y_true column for loss metrics)")
        print("expected_loss_per_approved_single_period=nan")
        print(f"expected_loss_per_approved_lifetime_{policy.tenor_months}mo=nan")
    else:
        el_lifetime = lifetime_expected_loss(
            pd_annual=default_rate_approved,
            lgd=policy.lgd,
            avg_loan_amount=policy.avg_loan_amount,
            tenor_months=policy.tenor_months,
        )
        ul = unexpected_loss(
            n_loans=n_approved,
            pd=default_rate_approved,
            lgd=policy.lgd,
            avg_loan_amount=policy.avg_loan_amount,
        )
        above_below = "ABOVE" if default_rate_approved > be_dr else "below"
        print(f"default_rate_among_approved={default_rate_approved:.4f} ({above_below} break-even)")
        print(f"expected_loss_per_approved_single_period={el_single:.2f}")
        print(f"expected_loss_per_approved_lifetime_{policy.tenor_months}mo={el_lifetime:.2f}")
        print(f"portfolio_unexpected_loss_rho12pct={ul:.2f}  ({ul/notional:.2%} of notional)")
    print(f"portfolio_notional_exposure={notional:.2f}")
    print(f"wrote={out_path}")


if __name__ == "__main__":
    main()
