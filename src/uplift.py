"""Machine learning for A/B tests: feature prep, a T-learner uplift model, and uplift evaluation.

Uplift = how much MORE likely a customer is to visit BECAUSE they got the email:
    uplift(x) = P(visit | email, x) - P(visit | no email, x)
A T-learner ("two models") fits one model on emailed customers and one on the control
group, then subtracts their predictions.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone

FEATURES_NUM = ["recency", "history", "mens", "womens", "newbie"]
FEATURES_CAT = ["zip_code", "channel"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Customer features known BEFORE the email was sent (pre-treatment only).
    history_segment is dropped because it just bins `history`."""
    X = pd.get_dummies(df[FEATURES_NUM + FEATURES_CAT], columns=FEATURES_CAT, drop_first=True, dtype=int)
    return X


class TLearner:
    """Two-model uplift estimator."""

    def __init__(self, model):
        self.model_t = clone(model)   # trained on the emailed group
        self.model_c = clone(model)   # trained on the control group

    def fit(self, X: pd.DataFrame, y, treated) -> "TLearner":
        y, treated = np.asarray(y), np.asarray(treated).astype(bool)
        self.model_t.fit(X[treated], y[treated])
        self.model_c.fit(X[~treated], y[~treated])
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_t.predict_proba(X)[:, 1] - self.model_c.predict_proba(X)[:, 1]