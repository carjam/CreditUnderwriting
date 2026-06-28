from __future__ import annotations

import numpy as np

from src.decisioning import (
    break_even_default_rate,
    calibrate_tiers,
    decisions,
    lifetime_expected_loss,
    portfolio_notional_exposure,
    risk_tier,
    simple_expected_loss_per_approved,
    threshold_sweep,
    unexpected_loss,
)


# ---------------------------------------------------------------------------
# Original tests (unchanged)
# ---------------------------------------------------------------------------

def test_risk_tier_mapping() -> None:
    p_good = np.array([0.2, 0.5, 0.69, 0.7, 0.95])
    got = risk_tier(p_good, prime_cut=0.70, near_cut=0.50).tolist()
    assert got == ["subprime", "near-prime", "near-prime", "prime", "prime"]


def test_decision_mapping() -> None:
    p_good = np.array([0.40, 0.50, 0.64, 0.65, 0.91])
    got = decisions(p_good, approve_min=0.65, review_min=0.50).tolist()
    assert got == ["decline", "review", "review", "approve", "approve"]


def test_threshold_sweep_outputs_rates() -> None:
    p_good = np.array([0.9, 0.8, 0.6, 0.4])
    y_true = np.array([1, 1, 0, 0])
    thresholds = np.array([0.5, 0.7, 0.95])
    sim = threshold_sweep(p_good, y_true, thresholds)

    assert sim.shape[0] == 3
    # at 0.5: first 3 approved, one bad -> 1/3 default among approved
    row = sim.loc[sim["threshold"] == 0.5].iloc[0]
    assert row["approval_rate"] == 0.75
    assert np.isclose(row["default_rate_among_approved"], 1 / 3)

    # at 0.95: none approved -> nan default among approved
    row = sim.loc[sim["threshold"] == 0.95].iloc[0]
    assert row["approval_rate"] == 0.0
    assert np.isnan(row["default_rate_among_approved"])


def test_simple_capital_helpers() -> None:
    el = simple_expected_loss_per_approved(0.04, avg_loan_amount=15000, lgd=0.45)
    assert np.isclose(el, 270.0)

    exposure = portfolio_notional_exposure(n_approved=120, avg_loan_amount=15000)
    assert exposure == 1_800_000.0


# ---------------------------------------------------------------------------
# Break-even default rate (Recommendation 3)
# ---------------------------------------------------------------------------

def test_break_even_default_rate_basic() -> None:
    # revenue=$1,500, lgd=0.45, avg_loan=$10,000 → 1,500 / (0.45 × 10,000) = 1/3
    bdr = break_even_default_rate(revenue_per_loan=1500, lgd=0.45, avg_loan_amount=10000)
    assert np.isclose(bdr, 1 / 3)


def test_break_even_default_rate_scales_with_revenue() -> None:
    bdr_low = break_even_default_rate(1000, lgd=0.45, avg_loan_amount=10000)
    bdr_high = break_even_default_rate(2000, lgd=0.45, avg_loan_amount=10000)
    assert bdr_high > bdr_low


def test_break_even_default_rate_decreases_with_higher_lgd() -> None:
    bdr_low_lgd = break_even_default_rate(1500, lgd=0.30, avg_loan_amount=10000)
    bdr_high_lgd = break_even_default_rate(1500, lgd=0.60, avg_loan_amount=10000)
    # Higher LGD → each default costs more → break-even DR is lower
    assert bdr_high_lgd < bdr_low_lgd


# ---------------------------------------------------------------------------
# Lifetime expected loss (Recommendation 4)
# ---------------------------------------------------------------------------

def test_lifetime_el_positive_for_positive_pd() -> None:
    el = lifetime_expected_loss(pd_annual=0.05, lgd=0.45, avg_loan_amount=10000, tenor_months=36)
    assert el > 0.0


def test_lifetime_el_zero_for_zero_pd() -> None:
    el = lifetime_expected_loss(pd_annual=0.0, lgd=0.45, avg_loan_amount=10000, tenor_months=36)
    assert el == 0.0


def test_lifetime_el_greater_than_single_period_for_long_tenor() -> None:
    # Lifetime EL over 36 months should exceed a single-period annual EL
    # because it accumulates exposure across all periods.
    single = simple_expected_loss_per_approved(0.05, avg_loan_amount=10000, lgd=0.45)
    lifetime = lifetime_expected_loss(0.05, lgd=0.45, avg_loan_amount=10000, tenor_months=36)
    assert lifetime > single


def test_lifetime_el_bounded_above_by_full_loss() -> None:
    # Lifetime EL can never exceed LGD × avg_loan_amount (100% default on day 1)
    lifetime = lifetime_expected_loss(0.5, lgd=0.45, avg_loan_amount=10000, tenor_months=36)
    assert lifetime <= 0.45 * 10000


def test_lifetime_el_increases_with_tenor() -> None:
    el_12 = lifetime_expected_loss(0.05, lgd=0.45, avg_loan_amount=10000, tenor_months=12)
    el_60 = lifetime_expected_loss(0.05, lgd=0.45, avg_loan_amount=10000, tenor_months=60)
    assert el_60 > el_12


def test_lifetime_el_increases_with_pd() -> None:
    el_low = lifetime_expected_loss(0.02, lgd=0.45, avg_loan_amount=10000, tenor_months=36)
    el_high = lifetime_expected_loss(0.10, lgd=0.45, avg_loan_amount=10000, tenor_months=36)
    assert el_high > el_low


# ---------------------------------------------------------------------------
# Unexpected loss (Recommendation 7)
# ---------------------------------------------------------------------------

def test_unexpected_loss_positive_for_valid_inputs() -> None:
    ul = unexpected_loss(n_loans=100, pd=0.05, lgd=0.45, avg_loan_amount=10000, rho=0.12)
    assert ul > 0.0


def test_unexpected_loss_zero_for_no_loans() -> None:
    assert unexpected_loss(0, pd=0.05, lgd=0.45, avg_loan_amount=10000) == 0.0


def test_unexpected_loss_zero_for_zero_pd() -> None:
    assert unexpected_loss(100, pd=0.0, lgd=0.45, avg_loan_amount=10000) == 0.0


def test_unexpected_loss_increases_with_correlation() -> None:
    ul_low = unexpected_loss(100, pd=0.05, lgd=0.45, avg_loan_amount=10000, rho=0.02)
    ul_high = unexpected_loss(100, pd=0.05, lgd=0.45, avg_loan_amount=10000, rho=0.20)
    assert ul_high > ul_low


def test_unexpected_loss_increases_with_portfolio_size() -> None:
    ul_small = unexpected_loss(50, pd=0.05, lgd=0.45, avg_loan_amount=10000, rho=0.12)
    ul_large = unexpected_loss(500, pd=0.05, lgd=0.45, avg_loan_amount=10000, rho=0.12)
    assert ul_large > ul_small


def test_unexpected_loss_known_value() -> None:
    # 1 loan: UL = sqrt(1 × 0.1 × 0.9 × (1 + 0×0.12)) × 0.5 × 1000
    # = sqrt(0.09) × 500 = 0.3 × 500 = 150
    ul = unexpected_loss(n_loans=1, pd=0.1, lgd=0.5, avg_loan_amount=1000, rho=0.12)
    assert np.isclose(ul, 150.0)


# ---------------------------------------------------------------------------
# Tier calibration (Recommendation 6)
# ---------------------------------------------------------------------------

def test_calibrate_tiers_returns_all_three_tiers() -> None:
    p_good = np.array([0.2, 0.55, 0.75, 0.8, 0.3])
    y_true = np.array([0, 1, 1, 0, 0])
    result = calibrate_tiers(p_good, y_true)
    assert set(result["tier"].tolist()) == {"prime", "near-prime", "subprime"}


def test_calibrate_tiers_correct_default_rates() -> None:
    p_good = np.array([0.2, 0.55, 0.75, 0.8, 0.3])
    y_true = np.array([0, 1, 1, 0, 0])
    result = calibrate_tiers(p_good, y_true, prime_cut=0.70, near_cut=0.50)

    # prime: p=[0.75, 0.80], y=[1, 0] → DR = 0.5
    prime = result.loc[result["tier"] == "prime"].iloc[0]
    assert np.isclose(prime["observed_default_rate"], 0.5)

    # near-prime: p=[0.55], y=[1] → DR = 0.0
    near = result.loc[result["tier"] == "near-prime"].iloc[0]
    assert np.isclose(near["observed_default_rate"], 0.0)

    # subprime: p=[0.2, 0.3], y=[0, 0] → DR = 1.0
    sub = result.loc[result["tier"] == "subprime"].iloc[0]
    assert np.isclose(sub["observed_default_rate"], 1.0)


def test_calibrate_tiers_columns() -> None:
    p_good = np.array([0.2, 0.55, 0.8])
    y_true = np.array([0, 1, 1])
    result = calibrate_tiers(p_good, y_true)
    expected_cols = {"tier", "n", "observed_default_rate", "p_good_min", "p_good_max"}
    assert expected_cols.issubset(set(result.columns))


def test_calibrate_tiers_nan_for_empty_tier() -> None:
    # All loans in prime tier → near-prime and subprime have n=0 → DR=nan
    p_good = np.array([0.80, 0.85, 0.90])
    y_true = np.array([1, 1, 0])
    result = calibrate_tiers(p_good, y_true)
    near = result.loc[result["tier"] == "near-prime"].iloc[0]
    assert np.isnan(near["observed_default_rate"])


# ---------------------------------------------------------------------------
# Edge-case guards
# ---------------------------------------------------------------------------

def test_lifetime_el_nan_for_nan_input() -> None:
    el = lifetime_expected_loss(pd_annual=float("nan"), lgd=0.45, avg_loan_amount=10000)
    assert np.isnan(el)


def test_break_even_default_rate_infinite_for_zero_lgd() -> None:
    be = break_even_default_rate(revenue_per_loan=3000, lgd=0.0, avg_loan_amount=10000)
    assert be == float("inf")


def test_break_even_default_rate_infinite_for_zero_loan_amount() -> None:
    be = break_even_default_rate(revenue_per_loan=3000, lgd=0.45, avg_loan_amount=0.0)
    assert be == float("inf")
