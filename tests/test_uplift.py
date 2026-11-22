"""Run with:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uplift import TLearner, qini_coefficient, qini_curve, share_of_gain_at, uplift_by_decile  # noqa: E402


@pytest.fixture
def toy():
    # Email helps ONLY customers with responsive = 1
    rng = np.random.default_rng(0)
    n = 8000
    X = pd.DataFrame({"responsive": rng.integers(0, 2, n), "noise": rng.normal(size=n)})
    t = rng.integers(0, 2, n)
    p = 0.10 + 0.20 * t * X["responsive"]
    y = (rng.random(n) < p).astype(int)
    return X, y, t


def test_tlearner_finds_responsive_customers(toy):
    X, y, t = toy
    up = TLearner(LogisticRegression()).fit(X, y, t).predict_uplift(X)
    assert up[X["responsive"] == 1].mean() > up[X["responsive"] == 0].mean() + 0.1


def test_top_decile_beats_bottom(toy):
    X, y, t = toy
    up = TLearner(LogisticRegression()).fit(X, y, t).predict_uplift(X)
    table = uplift_by_decile(up, y, t)
    assert table["actual_uplift"].iloc[0] > table["actual_uplift"].iloc[-1]


def test_good_model_beats_random_targeting(toy):
    X, y, t = toy
    up = TLearner(LogisticRegression()).fit(X, y, t).predict_uplift(X)
    curve = qini_curve(up, y, t)
    assert qini_coefficient(curve) > 0
    assert share_of_gain_at(curve, 0.5) > 0.6     # half the customers carry most of the gain


def test_random_scores_give_near_zero_qini(toy):
    X, y, t = toy
    rand = np.random.default_rng(1).random(len(y))
    assert abs(qini_coefficient(qini_curve(rand, y, t))) < 60