"""Run with:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ab_tools import bootstrap_diff, holm_adjust, sample_size_per_group, srm_check, two_proportion_test  # noqa: E402


def test_sample_size_textbook_example():
    # 40% vs 50%, alpha 0.05, power 0.80 -> about 385 per group
    assert sample_size_per_group(0.40, 0.10) == pytest.approx(385, abs=2)


def test_smaller_effects_need_more_customers():
    assert sample_size_per_group(0.106, 0.01) > sample_size_per_group(0.106, 0.02)


def test_srm_three_equal_groups_pass():
    assert srm_check([1000, 1000, 1000]) == pytest.approx(1.0)


def test_srm_lopsided_groups_fail():
    assert srm_check([1200, 1000, 800]) < 0.001


def test_two_proportion_no_difference():
    r = two_proportion_test(100, 1000, 100, 1000)
    assert r["diff"] == 0 and r["p_value"] == pytest.approx(1.0)


def test_two_proportion_ci_contains_diff():
    r = two_proportion_test(200, 1000, 250, 1000)
    assert r["ci_low"] < r["diff"] < r["ci_high"] and r["p_value"] < 0.05


def test_holm_adjust():
    assert holm_adjust([0.04, 0.01, 0.03]) == pytest.approx([0.06, 0.03, 0.06])


def test_bootstrap_centers_on_true_difference():
    a = np.array([0] * 800 + [1] * 200)
    b = np.array([0] * 700 + [1] * 300)
    assert bootstrap_diff(a, b, n_boot=500).mean() == pytest.approx(0.10, abs=0.01)