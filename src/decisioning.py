"""
Decision layer and portfolio-style simulation on top of existing classifier outputs.

No model training here — consumes P(good) and labels only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DecisionPolicy:
    """Configurable decision and tier thresholds."""

    approve_min: float = 0.65
    review_min: float = 0.50
    prime_cut: float = 0.70
    near_cut: float = 0.50
    avg_loan_amount: float = 10000.0
    lgd: float = 0.45
    # Loan tenor and revenue used for lifetime EL and break-even analysis.
    # revenue_per_loan: illustrative interest income over the full loan term
    # (e.g. 10% annual rate × $10k × 3yr ≈ $3,000; adjust to actual pricing).
    tenor_months: int = 36
    revenue_per_loan: float = 3000.0


def risk_tier(p_good: np.ndarray, prime_cut: float = 0.70, near_cut: float = 0.50) -> np.ndarray:
    """Map P(Fully Paid) to tiers: prime >= prime_cut, near-prime [near_cut, prime_cut), else subprime."""
    p = np.asarray(p_good, dtype=float)
    tiers = np.full(p.shape, "subprime", dtype=object)
    tiers[(p >= near_cut) & (p < prime_cut)] = "near-prime"
    tiers[p >= prime_cut] = "prime"
    return tiers


def decisions(
    p_good: np.ndarray,
    approve_min: float = 0.65,
    review_min: float = 0.50,
) -> np.ndarray:
    """approve / review / decline from P(good), higher is better."""
    p = np.asarray(p_good, dtype=float)
    out = np.full(p.shape, "decline", dtype=object)
    out[(p >= review_min) & (p < approve_min)] = "review"
    out[p >= approve_min] = "approve"
    return out


def threshold_sweep(
    p_good: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """
    y_true: 1 = good (Fully Paid), 0 = bad (Charged Off).
    For each approval threshold on p_good, compute approval rate and default rate among approved.
    """
    p = np.asarray(p_good, dtype=float)
    y = np.asarray(y_true, dtype=int)
    rows = []
    for t in thresholds:
        approved = p >= t
        n = approved.sum()
        approval_rate = approved.mean()
        if n == 0:
            default_rate_approved = np.nan
        else:
            default_rate_approved = (y[approved] == 0).mean()
        rows.append(
            {
                "threshold": t,
                "approval_rate": approval_rate,
                "default_rate_among_approved": default_rate_approved,
                "n_approved": int(n),
            }
        )
    return pd.DataFrame(rows)


def simple_expected_loss_per_approved(
    default_rate_among_approved: float,
    avg_loan_amount: float,
    lgd: float,
) -> float:
    """Directional expected loss per approved loan (single-period, illustrative)."""
    if np.isnan(default_rate_among_approved):
        return float("nan")
    return float(default_rate_among_approved * lgd * avg_loan_amount)


def portfolio_notional_exposure(n_approved: int, avg_loan_amount: float) -> float:
    return float(n_approved * avg_loan_amount)


def break_even_default_rate(
    revenue_per_loan: float,
    lgd: float,
    avg_loan_amount: float,
) -> float:
    """
    Default rate at which expected revenue exactly offsets expected loss.

    break_even_DR = revenue_per_loan / (lgd × avg_loan_amount)

    Approve loans below this default rate and you generate positive expected
    margin. Loans with empirical DR above break-even destroy value even if
    the model approves them.
    """
    denom = lgd * avg_loan_amount
    if denom == 0.0:
        return float("inf")
    return float(revenue_per_loan / denom)


def lifetime_expected_loss(
    pd_annual: float,
    lgd: float,
    avg_loan_amount: float,
    tenor_months: int = 36,
) -> float:
    """
    Lifetime EL over the loan term using a declining-balance EAD schedule
    (equal-principal amortization).

    EL = Σ_{t=1}^{N} [ pd_monthly × (1 - pd_monthly)^(t-1) × LGD × EAD_t ]

    where:
        pd_monthly = 1 - (1 - pd_annual)^(1/12)   exact monthly conversion
        EAD_t      = avg_loan_amount × (N - t + 1) / N   linear amortization

    The single-period helper uses an annual point-in-time DR, which understates
    lifetime loss for multi-year loans.  This function accumulates loss exposure
    period-by-period, consistent with IFRS 9 lifetime ECL methodology.
    """
    if np.isnan(pd_annual):
        return float("nan")
    if pd_annual <= 0.0:
        return 0.0
    if pd_annual >= 1.0:
        return float(lgd * avg_loan_amount)

    pd_monthly = 1.0 - (1.0 - pd_annual) ** (1.0 / 12.0)
    el = 0.0
    for t in range(1, tenor_months + 1):
        ead_t = avg_loan_amount * (tenor_months - t + 1) / tenor_months
        survival = (1.0 - pd_monthly) ** (t - 1)
        el += pd_monthly * survival * lgd * ead_t
    return float(el)


def unexpected_loss(
    n_loans: int,
    pd: float,
    lgd: float,
    avg_loan_amount: float,
    rho: float = 0.12,
) -> float:
    """
    Portfolio Unexpected Loss — standard deviation of aggregate credit loss —
    using a Vasicek single-factor (Gaussian copula) approximation.

    UL = sqrt( n × PD × (1-PD) × (1 + (n-1) × ρ) ) × LGD × avg_loan_amount

    rho=0.12 is the Basel II/III retail correlation floor for qualifying
    revolving exposures.  Residential mortgages use rho=0.15; corporate
    loans range from 0.12–0.24.

    UL represents the capital buffer needed to absorb losses beyond EL at a
    chosen confidence level.  RAROC = (Revenue - EL) / UL provides a
    risk-adjusted view that EL alone cannot.
    """
    if n_loans <= 0 or pd <= 0.0 or pd >= 1.0:
        return 0.0
    variance = n_loans * pd * (1.0 - pd) * (1.0 + (n_loans - 1) * rho)
    return float(np.sqrt(max(variance, 0.0)) * lgd * avg_loan_amount)


def calibrate_tiers(
    p_good: np.ndarray,
    y_true: np.ndarray,
    prime_cut: float = 0.70,
    near_cut: float = 0.50,
) -> pd.DataFrame:
    """
    Return observed default rate within each risk tier on a labelled sample.

    y_true: 1 = good (Fully Paid), 0 = bad (Charged Off).

    Tier boundaries are validated against empirical default rates.
    A well-calibrated tier system should show monotonically increasing DR
    from prime → near-prime → subprime.

    Columns: tier, n, observed_default_rate, p_good_min, p_good_max
    """
    p = np.asarray(p_good, dtype=float)
    y = np.asarray(y_true, dtype=int)
    tiers_arr = risk_tier(p, prime_cut=prime_cut, near_cut=near_cut)

    rows = []
    for tier_name, lo, hi in [
        ("prime", prime_cut, 1.0),
        ("near-prime", near_cut, prime_cut),
        ("subprime", 0.0, near_cut),
    ]:
        mask = tiers_arr == tier_name
        n = int(mask.sum())
        dr = float((y[mask] == 0).mean()) if n > 0 else float("nan")
        rows.append(
            {
                "tier": tier_name,
                "n": n,
                "observed_default_rate": dr,
                "p_good_min": lo,
                "p_good_max": hi,
            }
        )
    return pd.DataFrame(rows)


def apply_policy(p_good: np.ndarray, policy: DecisionPolicy) -> pd.DataFrame:
    """
    Apply a policy to model scores and return per-applicant decision outputs.
    """
    p = np.asarray(p_good, dtype=float)
    out = pd.DataFrame({"p_good": p})
    out["risk_tier"] = risk_tier(p, prime_cut=policy.prime_cut, near_cut=policy.near_cut)
    out["decision"] = decisions(p, approve_min=policy.approve_min, review_min=policy.review_min)
    return out
