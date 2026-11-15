# Memo: who should receive the retention offer

**To:** Head of CRM · **Data:** randomized campaign, `x5` · **Model:** S-learner

## Recommendation
Send the offer to **9% of customers**, chosen by expected incremental profit, instead of the current churn-risk list. On held-out data this earns **922 RUB per 1,000 customers** (95% CI -223 to 1,878), versus -40 RUB for churn-risk targeting at the same budget cap and -3,449 RUB for contacting everybody. For a base of 100,000 customers that is about **96,221 RUB per campaign** over the current approach.

## Why the usual lists earn less
- **Contacting everyone** pays the offer to customers who would have bought anyway: -3,449 RUB per 1,000.
- **Churn-risk targeting** picks people who are leaving but often do not respond to an offer (uplift among them: +4.4 pp).
- **Response-model targeting** picks loyal buyers, the most expensive 'sure things' (uplift among them: +2.5 pp).
- **Uplift targeting** picks customers whose behavior the offer changes (uplift among them: +10.0 pp).

## Who responds
Predicted uplift is higher for customers with higher age, points_burn_ratio, share_tx_points_spent and lower days_issue_to_redeem, days_since_issue, n_tx. See the driver chart in the README for the full picture.

## How confident are we
- Profit is measured on customers the model never saw, from a randomized test, so it is a causal estimate.
- Assumptions: margin 22%, contact cost 2.5 RUB, offer cost 15.0 RUB, redemption 50%. If the offer cost doubles, the profitable share falls to 4%.

## Next steps
1. Run the next campaign as champion/challenger: 80% uplift targeting, 20% current list, plus a 10% no-contact holdout.
2. Refresh the model monthly and monitor the uplift-decile chart for drift.
3. Test a cheaper offer for customers whose uplift is positive but whose expected profit is negative at the current offer cost.
