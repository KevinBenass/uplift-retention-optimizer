"""From scores to a targeting decision, valued in currency on randomized holdout data.

Profit of one customer under each arm (margin m, contact cost c, offer cost d, redemption
rate r, realized post-period revenue R, purchase indicator Y):
    treated:  m * R - r * d * Y - c      (every treated buyer may redeem, even sure things)
    control:  m * R

Policy value is estimated off-policy with inverse propensity weighting: a customer the
policy would contact contributes their profit if they were in fact treated (weighted 1/e),
one it would not contact contributes their profit if they were in control (weighted
1/(1-e)). Randomized assignment makes this unbiased for any policy, including ones that
were never run. Confidence intervals come from a paired bootstrap over customers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from uplift_optimizer.config import Economics


@dataclass
class Outcomes:
    t: np.ndarray
    y: np.ndarray
    revenue: np.ndarray

    @property
    def e(self) -> float:
        return float(self.t.mean())


def arm_profits(o: Outcomes, econ: Economics) -> tuple[np.ndarray, np.ndarray]:
    treated = (
        econ.gross_margin * o.revenue
        - econ.redemption_rate * econ.discount_value * o.y
        - econ.contact_cost
    )
    control = econ.gross_margin * o.revenue
    return treated, control


def ipw_terms(policy: np.ndarray, o: Outcomes, econ: Economics) -> np.ndarray:
    """Per-customer IPW contributions; their mean is the policy's profit per customer."""
    p_t, p_c = arm_profits(o, econ)
    e = o.e
    return policy * o.t / e * p_t + (1 - policy) * (1 - o.t) / (1 - e) * p_c


def incremental_profit_terms(policy: np.ndarray, o: Outcomes, econ: Economics) -> np.ndarray:
    """Per-customer contribution to profit versus running no campaign at all."""
    return ipw_terms(policy, o, econ) - ipw_terms(np.zeros_like(policy), o, econ)


def expected_incremental_profit(
    uplift: np.ndarray, p_treated: np.ndarray, basket: np.ndarray, econ: Economics
) -> np.ndarray:
    """Model-based value of contacting each customer: the quantity to rank on.

    Incremental margin from caused purchases, minus the offer cost paid by every treated
    purchaser who redeems (caused or not), minus the contact cost.
    """
    return (
        econ.gross_margin * basket * uplift
        - econ.redemption_rate * econ.discount_value * p_treated
        - econ.contact_cost
    )


def top_fraction(score: np.ndarray, fraction: float) -> np.ndarray:
    k = round(fraction * len(score))
    policy = np.zeros(len(score), dtype=float)
    policy[np.argsort(-score, kind="stable")[:k]] = 1.0
    return policy


def budget_policy(profit_score: np.ndarray, max_fraction: float) -> np.ndarray:
    """Contact customers with positive expected incremental profit, best first, up to budget."""
    policy = top_fraction(profit_score, max_fraction)
    return policy * (profit_score > 0)


def profit_curve(
    score: np.ndarray, o: Outcomes, econ: Economics, n_points: int = 51
) -> pd.DataFrame:
    """Incremental profit per customer when contacting the top x% by ``score``."""
    order = np.argsort(-score, kind="stable")
    p_t, p_c = arm_profits(o, econ)
    e = o.e
    gain = (
        o.t / e * p_t - (1 - o.t) / (1 - e) * p_c
    )  # contribution of switching a customer to "contact"
    cum = np.r_[0, np.cumsum(gain[order])]
    fracs = np.linspace(0, 1, n_points)
    ks = (fracs * len(score)).round().astype(int)
    return pd.DataFrame(
        {"fraction": fracs, "incremental_profit_per_customer": cum[ks] / len(score)}
    )


def bootstrap_policy_table(
    policies: dict[str, np.ndarray], o: Outcomes, econ: Economics, n_boot: int = 500, seed: int = 0
) -> pd.DataFrame:
    """Incremental profit per 1,000 customers for each policy with a 95% CI."""
    rng = np.random.default_rng(seed)
    n = len(o.t)
    idx = rng.integers(0, n, size=(n_boot, n))
    terms = {name: incremental_profit_terms(p, o, econ) for name, p in policies.items()}
    rows = []
    for name, policy in policies.items():
        tm = terms[name]
        boots = tm[idx].mean(axis=1) * 1000
        targeted = policy.astype(bool)
        tr, ct = targeted & (o.t == 1), targeted & (o.t == 0)
        observed_uplift = o.y[tr].mean() - o.y[ct].mean() if tr.any() and ct.any() else np.nan
        rows.append(
            {
                "policy": name,
                "contacted": policy.mean(),
                "uplift_in_targeted": observed_uplift,
                "profit_per_1000": tm.mean() * 1000,
                "ci_low": np.percentile(boots, 2.5),
                "ci_high": np.percentile(boots, 97.5),
            }
        )
    return pd.DataFrame(rows)
