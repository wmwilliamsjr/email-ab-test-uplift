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


def uplift_by_decile(uplift_pred, y, treated, n_bins: int = 10) -> pd.DataFrame:
    """Sort customers by predicted uplift (highest first), cut into bins, and compare the
    ACTUAL visit rate of emailed vs. control customers inside each bin."""
    d = pd.DataFrame({"pred": np.asarray(uplift_pred), "y": np.asarray(y), "t": np.asarray(treated).astype(int)})
    d["bin"] = pd.qcut(d["pred"].rank(method="first", ascending=False), n_bins, labels=range(1, n_bins + 1))
    g = d.groupby("bin", observed=True).agg(customers=("y", "size"), predicted_uplift=("pred", "mean"))
    rates = d.pivot_table(index="bin", columns="t", values="y", aggfunc="mean", observed=True)
    g["rate_email"], g["rate_control"] = rates[1], rates[0]
    g["actual_uplift"] = g["rate_email"] - g["rate_control"]
    return g


def qini_curve(uplift_pred, y, treated) -> pd.DataFrame:
    """Qini curve: target customers from highest to lowest predicted uplift and track the
    extra visits gained versus control, scaled to the treated count."""
    d = pd.DataFrame({"pred": np.asarray(uplift_pred), "y": np.asarray(y), "t": np.asarray(treated).astype(int)})
    d = d.sort_values("pred", ascending=False).reset_index(drop=True)
    n_t, n_c = d["t"].cumsum(), (1 - d["t"]).cumsum()
    y_t, y_c = (d["y"] * d["t"]).cumsum(), (d["y"] * (1 - d["t"])).cumsum()
    gain = y_t - y_c * (n_t / n_c.replace(0, np.nan)).fillna(0)
    out = pd.DataFrame({"share_targeted": (np.arange(len(d)) + 1) / len(d), "incremental_visits": gain})
    return pd.concat([pd.DataFrame({"share_targeted": [0.0], "incremental_visits": [0.0]}), out], ignore_index=True)


def qini_coefficient(curve: pd.DataFrame) -> float:
    """Area between the model's Qini curve and the random-targeting line (higher = better)."""
    x, yv = curve["share_targeted"].to_numpy(), curve["incremental_visits"].to_numpy()
    random_line = x * yv[-1]
    area = getattr(np, "trapezoid", None) or np.trapz      # name differs by numpy version
    return float(area(yv - random_line, x))


def share_of_gain_at(curve: pd.DataFrame, share: float) -> float:
    """Share of all incremental visits captured by emailing only the top `share` of customers."""
    total = curve["incremental_visits"].iloc[-1]
    at = curve.loc[curve["share_targeted"] <= share, "incremental_visits"].iloc[-1]
    return float(at / total) if total else float("nan")