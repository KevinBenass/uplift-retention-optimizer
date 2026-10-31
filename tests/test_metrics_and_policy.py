import numpy as np
import pytest

from uplift_optimizer.config import Economics
from uplift_optimizer.evaluate.curves import decile_table, qini_coefficient, qini_curve, uplift_at_k
from uplift_optimizer.evaluate.policy import (
    Outcomes,
    budget_policy,
    expected_incremental_profit,
    incremental_profit_terms,
    profit_curve,
    top_fraction,
)


@pytest.fixture
def rct():
    """Randomized data where only customers with x > 0.5 respond to treatment."""
    rng = np.random.default_rng(0)
    n = 40_000
    x = rng.random(n)
    t = rng.integers(0, 2, n)
    p = 0.3 + 0.3 * (x > 0.5) * t
    y = (rng.random(n) < p).astype(int)
    return x, t, y


def test_qini_ranks_informative_score_above_random(rct):
    x, t, y = rct
    good = qini_coefficient(x, t, y)
    rand = qini_coefficient(np.random.default_rng(1).random(len(x)), t, y)
    bad = qini_coefficient(-x, t, y)
    assert good > 0.08 and abs(rand) < 0.03 and bad < 0  # normalized by the observable optimum


def test_qini_curve_endpoints(rct):
    x, t, y = rct
    frac, q = qini_curve(x, t, y)
    assert frac[0] == 0 and frac[-1] == 1 and q[0] == 0
    # Final value = total treated responders minus scaled control responders = overall effect.
    expected = y[t == 1].sum() - y[t == 0].sum() * t.sum() / (1 - t).sum()
    assert q[-1] == pytest.approx(expected)


def test_uplift_at_k_and_deciles(rct):
    x, t, y = rct
    assert uplift_at_k(x, t, y, 0.3) == pytest.approx(0.3, abs=0.03)
    assert uplift_at_k(x, t, y, 1.0) == pytest.approx(0.15, abs=0.02)
    table = decile_table(x * 0.3, t, y, n_bins=10)
    assert len(table) == 10
    assert table.loc[table.bin <= 4, "observed_uplift"].mean() > 0.25
    assert table.loc[table.bin >= 7, "observed_uplift"].abs().mean() < 0.05


def test_ipw_policy_value_matches_ground_truth(rct):
    x, t, y = rct
    econ = Economics(gross_margin=0.5, contact_cost=1.0, discount_value=10.0, redemption_rate=1.0)
    revenue = y * 100.0
    o = Outcomes(t=t, y=y, revenue=revenue)
    policy = (x > 0.5).astype(float)
    estimate = incremental_profit_terms(policy, o, econ).mean()
    # Truth: half the customers are contacted. Each gains 0.3 extra purchases worth 50 margin,
    # pays the offer on 0.6 purchases (10 each) and costs 1 to contact.
    truth = 0.5 * (0.3 * 50 - 0.6 * 10 - 1)
    assert estimate == pytest.approx(truth, abs=0.6)


def test_profit_curve_consistent_with_policy_terms(rct):
    x, t, y = rct
    econ = Economics()
    o = Outcomes(t=t, y=y, revenue=y * 20.0)
    curve = profit_curve(x, o, econ, n_points=11)
    at_30 = curve.loc[curve.fraction.round(2) == 0.3, "incremental_profit_per_customer"].iloc[0]
    assert at_30 == pytest.approx(incremental_profit_terms(top_fraction(x, 0.3), o, econ).mean())


def test_budget_policy_respects_cap_and_sign():
    score = np.array([5.0, 4.0, -1.0, 3.0, -2.0, 1.0])
    assert budget_policy(score, 0.5).tolist() == [1, 1, 0, 1, 0, 0]
    assert budget_policy(score, 1.0).sum() == 4


def test_expected_profit_penalizes_sure_things():
    econ = Economics(gross_margin=0.2, contact_cost=1.0, discount_value=10.0, redemption_rate=0.5)
    basket = np.array([100.0, 100.0])
    # Same uplift, but the first customer is a near-certain buyer who will redeem the offer.
    profit = expected_incremental_profit(np.array([0.1, 0.1]), np.array([0.95, 0.3]), basket, econ)
    assert profit[0] < profit[1]
