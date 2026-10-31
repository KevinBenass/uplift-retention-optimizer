"""The targeting rules a marketing team would use without uplift modeling.

Each exposes ``predict`` returning a score where higher = contact first, so it can be
evaluated with exactly the same curves and policy code as the uplift models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


class RandomTargeting:
    name = "Random"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def fit(self, X, t, y):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.random.default_rng(self.seed).random(len(X))


class ResponseTargeting:
    """Contact those most likely to buy when contacted: the classic 'propensity' model.

    It rewards sure things, who generate conversions the campaign did not cause.
    """

    name = "Response model"

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params

    def fit(self, X, t, y):
        self.model = LGBMClassifier(**self.params, random_state=0).fit(X[t == 1], y[t == 1])
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


class ChurnRiskTargeting:
    """Contact those least likely to buy without contact: the classic retention model.

    It rewards lost causes, who are at risk but do not respond to the offer.
    """

    name = "Churn-risk model"

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params

    def fit(self, X, t, y):
        self.model = LGBMClassifier(**self.params, random_state=0).fit(X[t == 0], y[t == 0])
        return self

    def predict(self, X):
        return 1 - self.model.predict_proba(X)[:, 1]
