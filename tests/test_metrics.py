import numpy as np
import pandas as pd
import pytest
from legacy import (
    LEGACY_NON_FEATURES,
    legacy_add_features,
    legacy_altman_z,
    legacy_build_label,
    legacy_capture_rate,
    legacy_cluster_bootstrap_ci,
    legacy_fit_prepare,
    legacy_paired_bootstrap_diff_ci,
    legacy_pr_auc,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from corp_default.metrics import (
    accuracy_ratio,
    capture_curve,
    capture_rate,
    cluster_bootstrap_ci,
    paired_bootstrap_diff_ci,
    psi,
    reliability_table,
    trapezoid_pr_auc,
    wilson_interval,
    within_year_percentile,
)

# eight companies, two rows each; two positives in different companies
y = np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
s = np.linspace(0.0, 1.0, 16)
groups = np.repeat([f"C_{i}" for i in range(8)], 2)


def test_capture_rate_on_a_hand_example():
    y = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    scores = list(range(10, 0, -1))
    table = capture_rate(y, scores, budgets=(0.1, 0.5))
    assert list(table["defaults_caught"]) == [1, 2]
    assert list(table["capture_rate"]) == [0.5, 1.0]
    assert list(table["lift"]) == pytest.approx([5.0, 2.0])


def test_capture_curve_runs_from_nothing_to_everything():
    reviewed, captured = capture_curve([1, 0, 0, 1], [0.9, 0.8, 0.1, 0.2])
    assert list(reviewed) == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert list(captured) == [0.0, 0.5, 0.5, 1.0, 1.0]


def test_accuracy_ratio():
    assert accuracy_ratio([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.5)


def test_trapezoid_pr_auc_equals_legacy():
    y_hand, s_hand = [1, 0, 1, 0, 0, 1], [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    assert trapezoid_pr_auc(y_hand, s_hand) == pytest.approx(legacy_pr_auc(y_hand, s_hand))


def test_wilson_interval():
    assert wilson_interval(0, 10) == pytest.approx((0.0, 0.27753), abs=1e-4)
    assert wilson_interval(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-4)


def test_wilson_interval_refuses_an_empty_bin():
    with pytest.raises(ValueError, match="n"):
        wilson_interval(0, 0)


def test_within_year_percentile_ranks_inside_each_year():
    pct = within_year_percentile(
        pd.Series([1.0, 2.0, 10.0, 20.0]), pd.Series([2005, 2005, 2006, 2006])
    )
    assert list(pct) == [0.5, 1.0, 0.5, 1.0]


def test_psi_is_zero_for_the_same_sample_and_positive_for_a_shift():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-12)
    assert psi(x, x + 1.0) > 0.25


def test_psi_ignores_missing_values():
    x = np.random.default_rng(0).normal(size=5000)
    with_gaps = np.concatenate([x, [np.nan] * 50])
    assert psi(with_gaps, x) == pytest.approx(0.0, abs=1e-12)


def test_paired_difference_of_identical_scores_is_zero():
    lo, hi, n = paired_bootstrap_diff_ci(y, s, s, groups, roc_auc_score, n_boot=50)
    assert (lo, hi) == (0.0, 0.0)
    assert 0 < n <= 50


def test_bootstrap_counts_the_resamples_it_could_use():
    # six companies, the only positive sits in one of them: some resamples have no positive
    y_one = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    s_one = np.linspace(0.0, 1.0, 12)
    g_one = np.repeat([f"C_{i}" for i in range(6)], 2)
    _lo, _hi, usable = cluster_bootstrap_ci(y_one, s_one, g_one, roc_auc_score, n_boot=200)
    assert 0 < usable < 200


def test_bootstrap_with_no_usable_resample_raises():
    with pytest.raises(ValueError, match="usable"):
        cluster_bootstrap_ci(
            [0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4], ["A", "A", "B", "B"], roc_auc_score
        )


def test_reliability_table_by_hand():
    prob = np.array([0.1] * 10 + [0.6] * 10)
    y_rel = np.array([1] + [0] * 9 + [1] * 6 + [0] * 4)
    table = reliability_table(y_rel, prob, bins=2)
    assert list(table["n"]) == [10, 10]
    assert list(table["mean_prob"]) == pytest.approx([0.1, 0.6])
    assert list(table["rate"]) == pytest.approx([0.1, 0.6])
    assert (table["lo"] < table["rate"]).all() and (table["rate"] < table["hi"]).all()


@pytest.mark.needs_data
def test_bootstraps_equal_legacy(raw):
    # legacy Altman and logistic scores on the legacy VALIDATION window 2009-2011: identical intervals
    frame = legacy_add_features(legacy_build_label(raw.copy(), 2))
    cols = [c for c in frame.columns if c not in LEGACY_NON_FEATURES]
    train = frame[frame["year"] <= 2006]
    val = frame[frame["year"].between(2009, 2011)]
    prepare, *_ = legacy_fit_prepare(train, cols)
    logit = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=42),
    )
    logit.fit(prepare(train), train["target"])
    altman = -legacy_altman_z(prepare(val)).to_numpy()
    logistic = logit.predict_proba(prepare(val))[:, 1]
    y_val, g_val = val["target"], val["company_name"]

    for scores in (altman, logistic):
        for metric in (roc_auc_score, legacy_pr_auc):
            assert cluster_bootstrap_ci(y_val, scores, g_val, metric, n_boot=200) == (
                legacy_cluster_bootstrap_ci(y_val, scores, g_val, metric, n_boot=200)
            )
        pd.testing.assert_frame_equal(
            capture_rate(y_val, scores), legacy_capture_rate(y_val, scores)
        )
    assert paired_bootstrap_diff_ci(y_val, logistic, altman, g_val, roc_auc_score, n_boot=200) == (
        legacy_paired_bootstrap_diff_ci(y_val, logistic, altman, g_val, roc_auc_score, n_boot=200)
    )
