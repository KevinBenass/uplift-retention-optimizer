"""What drives uplift? SHAP values on a surrogate of the chosen uplift model.

Meta-learners combine several models, so their SHAP values are awkward to read directly.
Instead a single LightGBM regressor is fit to the chosen model's uplift predictions (a
faithful surrogate, R² reported) and explained with exact TreeSHAP via LightGBM's
``pred_contrib``. The output answers the question a marketer asks: what kind of customer
does the offer actually move?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score


def uplift_drivers(
    X: pd.DataFrame, uplift: np.ndarray, top: int = 12
) -> tuple[pd.DataFrame, float, np.ndarray]:
    surrogate = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        verbose=-1,
        random_state=0,
    )
    surrogate.fit(X, uplift)
    fidelity = r2_score(uplift, surrogate.predict(X))
    contrib = surrogate.predict(X, pred_contrib=True)[:, :-1]  # last column = expected value
    mean_abs = np.abs(contrib).mean(axis=0)
    # Direction: does a higher feature value push predicted uplift up or down?
    direction = []
    for j, col in enumerate(X.columns):
        x = X[col].to_numpy(dtype=float)
        ok = ~np.isnan(x)
        corr = np.corrcoef(x[ok], contrib[ok, j])[0, 1] if ok.sum() > 2 and x[ok].std() > 0 else 0.0
        direction.append(corr)
    table = (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs, "direction": direction})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )
    return table, fidelity, contrib


def segment_profile(
    df: pd.DataFrame, uplift: np.ndarray, features: list[str], q: float = 0.2
) -> pd.DataFrame:
    """Median feature values of the top-q vs bottom-q customers by predicted uplift."""
    hi = uplift >= np.quantile(uplift, 1 - q)
    lo = uplift <= np.quantile(uplift, q)
    return pd.DataFrame(
        {
            "feature": features,
            f"top {int(q * 100)}% uplift": [df.loc[hi, f].median() for f in features],
            f"bottom {int(q * 100)}% uplift": [df.loc[lo, f].median() for f in features],
            "all customers": [df[f].median() for f in features],
        }
    )
