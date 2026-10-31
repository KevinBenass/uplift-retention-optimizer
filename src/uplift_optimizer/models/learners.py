"""Uplift meta-learners, all built on the same LightGBM base learner.

Using one base learner for every strategy isolates what this project compares: how each
strategy turns outcome models into a treatment-effect estimate.

  S-learner            one model with treatment as a feature; tau = f(x,1) - f(x,0).
                       Simple, but regularization can shrink the treatment effect to zero.
  T-learner            separate models for treated and control; tau = mu1 - mu0.
                       Unbiased in the limit but noisy: two models' errors add up.
  X-learner            imputes individual effects from the other arm's model, then
                       regresses them (Künzel et al., 2019). Strong when arms are unbalanced
                       or effects are simple.
  DR-learner           regresses a doubly-robust pseudo-outcome with cross-fitting
                       (Kennedy, 2020). Robust to outcome-model error when the propensity
                       is known, which it is in a randomized campaign.
  Transformed outcome  regresses Y*(T - e) / (e(1 - e)), whose expectation is tau
                       (Athey & Imbens, 2016). One model, but high variance.

The propensity e is the treatment share of the training data: assignment was randomized,
so it is known by design rather than estimated.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import StratifiedKFold


class UpliftModel(Protocol):
    name: str

    def fit(self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> UpliftModel: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


def _clf(params: dict[str, Any], seed: int = 0) -> LGBMClassifier:
    return LGBMClassifier(**params, random_state=seed)


def _reg(params: dict[str, Any], seed: int = 0) -> LGBMRegressor:
    return LGBMRegressor(**params, random_state=seed)


def _p1(model: LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


class SLearner:
    name = "S-learner"

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params

    def fit(self, X, t, y):
        self.model = _clf(self.params).fit(X.assign(_treatment=t), y)
        return self

    def predict(self, X):
        return _p1(self.model, X.assign(_treatment=1)) - _p1(self.model, X.assign(_treatment=0))


class TLearner:
    name = "T-learner"

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params

    def fit(self, X, t, y):
        self.mu1 = _clf(self.params).fit(X[t == 1], y[t == 1])
        self.mu0 = _clf(self.params).fit(X[t == 0], y[t == 0])
        return self

    def predict(self, X):
        return _p1(self.mu1, X) - _p1(self.mu0, X)

    def predict_outcomes(self, X) -> tuple[np.ndarray, np.ndarray]:
        return _p1(self.mu0, X), _p1(self.mu1, X)


class XLearner:
    name = "X-learner"

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params

    def fit(self, X, t, y):
        self.e = float(t.mean())
        mu1 = _clf(self.params).fit(X[t == 1], y[t == 1])
        mu0 = _clf(self.params).fit(X[t == 0], y[t == 0])
        d1 = y[t == 1] - _p1(mu0, X[t == 1])
        d0 = _p1(mu1, X[t == 0]) - y[t == 0]
        self.tau1 = _reg(self.params).fit(X[t == 1], d1)
        self.tau0 = _reg(self.params).fit(X[t == 0], d0)
        return self

    def predict(self, X):
        # Weight by propensity: lean on the arm whose imputation used the larger sample.
        return self.e * self.tau0.predict(X) + (1 - self.e) * self.tau1.predict(X)


class DRLearner:
    name = "DR-learner"

    def __init__(self, params: dict[str, Any], n_folds: int = 2, seed: int = 0) -> None:
        self.params = params
        self.n_folds = n_folds
        self.seed = seed

    def fit(self, X, t, y):
        e = float(t.mean())
        mu0 = np.zeros(len(X))
        mu1 = np.zeros(len(X))
        folds = StratifiedKFold(self.n_folds, shuffle=True, random_state=self.seed)
        strata = t * 2 + y
        # Cross-fitting: nuisance predictions for each row come from models that never saw it.
        for train_idx, pred_idx in folds.split(X, strata):
            Xtr, ttr, ytr = X.iloc[train_idx], t[train_idx], y[train_idx]
            m1 = _clf(self.params).fit(Xtr[ttr == 1], ytr[ttr == 1])
            m0 = _clf(self.params).fit(Xtr[ttr == 0], ytr[ttr == 0])
            mu1[pred_idx] = _p1(m1, X.iloc[pred_idx])
            mu0[pred_idx] = _p1(m0, X.iloc[pred_idx])
        pseudo = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
        self.model = _reg(self.params).fit(X, pseudo)
        return self

    def predict(self, X):
        return self.model.predict(X)


class TransformedOutcome:
    name = "Transformed outcome"

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params

    def fit(self, X, t, y):
        e = float(t.mean())
        self.model = _reg(self.params).fit(X, y * (t - e) / (e * (1 - e)))
        return self

    def predict(self, X):
        return self.model.predict(X)


LEARNERS = {
    "s_learner": SLearner,
    "t_learner": TLearner,
    "x_learner": XLearner,
    "dr_learner": DRLearner,
    "transformed_outcome": TransformedOutcome,
}


def make_learner(name: str, params: dict[str, Any]) -> UpliftModel:
    if name not in LEARNERS:
        raise KeyError(f"Unknown learner {name!r}; options: {sorted(LEARNERS)}")
    return LEARNERS[name](params)
