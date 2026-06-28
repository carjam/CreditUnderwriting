from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from src.decisioning import DecisionPolicy, apply_policy


def test_default_policy_config_parses() -> None:
    policy_path = Path("config/policy.default.yaml")
    raw = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    policy = DecisionPolicy(
        approve_min=float(raw["approve_min"]),
        review_min=float(raw["review_min"]),
        prime_cut=float(raw["prime_cut"]),
        near_cut=float(raw["near_cut"]),
        avg_loan_amount=float(raw["avg_loan_amount"]),
        lgd=float(raw["lgd"]),
        tenor_months=int(raw.get("tenor_months", 36)),
        revenue_per_loan=float(raw.get("revenue_per_loan", 3000.0)),
    )
    assert 0.0 < policy.review_min < policy.approve_min < 1.0
    assert 0.0 < policy.near_cut < policy.prime_cut < 1.0
    assert policy.tenor_months > 0
    assert policy.revenue_per_loan > 0.0
    assert 0.0 < policy.lgd < 1.0


def test_apply_policy_outputs_expected_columns() -> None:
    policy = DecisionPolicy()
    out = apply_policy(np.array([0.2, 0.55, 0.8]), policy)
    assert list(out.columns) == ["p_good", "risk_tier", "decision"]
    assert out["decision"].tolist() == ["decline", "review", "approve"]
