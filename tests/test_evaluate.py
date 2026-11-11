import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from corp_default.evaluate import fold_table, paired_table, scoreboard, walk_forward_oof
from corp_default.metrics import cluster_bootstrap_ci
from corp_default.splits import CRISIS_FOLDS, HORIZON, walk_forward_folds


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    # 300 companies, 1999 to 2011; the chance of failure rises with `signal`
    rng = np.random.default_rng(42)
    rows = [(f"C_{c}", year) for c in range(300) for year in range(1999, 2012)]
    frame = pd.DataFrame(rows, columns=["company_name", "year"])
    frame["signal"] = rng.normal(size=len(frame))
    probability = 1 / (1 + np.exp(-(frame["signal"] * 2.0 - 3.0)))
    frame["target"] = (rng.random(len(frame)) < probability).astype(int)
    return frame


def by_signal(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    return valid["signal"].to_numpy()


def constant(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    return np.zeros(len(valid))


def test_walk_forward_oof_returns_exactly_the_fold_years(panel):
    oof = walk_forward_oof(by_signal, panel, walk_forward_folds(panel))
    expected = panel.loc[panel["year"].between(2005, 2011), ["company_name", "year"]]
    expected = expected.sort_values("year", kind="stable")  # folds stack year by year
    assert list(oof.columns) == ["company_name", "year", "target", "score", "pct"]
    pd.testing.assert_frame_equal(
        oof[["company_name", "year"]].reset_index(drop=True), expected.reset_index(drop=True)
    )


def test_fit_predict_never_sees_a_training_year_inside_the_embargo(panel):
    seen = []

    def spy(train, valid):
        seen.append((int(train["year"].max()), set(valid["year"])))
        return np.zeros(len(valid))

    walk_forward_oof(spy, panel, walk_forward_folds(panel))
    assert len(seen) == 7
    for last_train_year, valid_years in seen:
        (valid_year,) = valid_years
        assert last_train_year + HORIZON < valid_year


def test_walk_forward_oof_refuses_the_wrong_number_of_scores(panel):
    with pytest.raises(ValueError, match="scores"):
        walk_forward_oof(lambda train, valid: np.zeros(3), panel, walk_forward_folds(panel))


def test_a_planted_signal_beats_a_constant(panel):
    folds = walk_forward_folds(panel)
    scores = {
        "signal": walk_forward_oof(by_signal, panel, folds),
        "constant": walk_forward_oof(constant, panel, folds),
    }
    board = scoreboard(scores, n_boot=50).set_index("model")
    assert board.loc["signal", "roc_auc"] > 0.75
    assert board.loc["constant", "roc_auc"] == pytest.approx(0.5)
    assert board.loc["signal", "roc_auc_lo"] > board.loc["constant", "roc_auc_hi"]
    assert board.loc["signal", "capture_10"] > board.loc["constant", "capture_10"]
    assert board.loc["signal", "accuracy_ratio"] == pytest.approx(
        2 * board.loc["signal", "roc_auc"] - 1
    )


def test_scoreboard_intervals_equal_the_cluster_bootstrap(panel):
    oof = walk_forward_oof(by_signal, panel, walk_forward_folds(panel))
    board = scoreboard({"signal": oof}, n_boot=50).iloc[0]
    lo, hi, usable = cluster_bootstrap_ci(
        oof["target"], oof["pct"], oof["company_name"], roc_auc_score, n_boot=50
    )
    assert (board["roc_auc_lo"], board["roc_auc_hi"], board["usable"]) == (lo, hi, usable)


def test_scoreboard_refuses_models_scored_on_different_rows(panel):
    folds = walk_forward_folds(panel)
    full = walk_forward_oof(by_signal, panel, folds)
    with pytest.raises(ValueError, match="rows"):
        scoreboard({"a": full, "b": full.iloc[1:]}, n_boot=10)


def test_paired_table_compares_each_rung_with_the_one_before(panel):
    folds = walk_forward_folds(panel)
    scores = {
        "constant": walk_forward_oof(constant, panel, folds),
        "signal": walk_forward_oof(by_signal, panel, folds),
        "signal again": walk_forward_oof(by_signal, panel, folds),
    }
    table = paired_table(scores, ["constant", "signal", "signal again"], n_boot=50)
    assert list(table["model"]) == ["signal", "signal again"]
    assert list(table["against"]) == ["constant", "signal"]
    first, second = table.iloc[0], table.iloc[1]
    assert first["roc_auc_diff_lo"] > 0
    assert (second["roc_auc_diff"], second["roc_auc_diff_lo"], second["roc_auc_diff_hi"]) == (
        0.0,
        0.0,
        0.0,
    )


def test_fold_table_gives_each_fold_and_the_crisis_and_calm_pools(panel):
    oof = walk_forward_oof(by_signal, panel, walk_forward_folds(panel))
    table = fold_table(oof, n_boot=50).set_index("fold")
    assert list(table.index) == [str(y) for y in range(2005, 2012)] + ["crisis folds", "calm folds"]
    assert list(table["crisis"]) == [True] * 3 + [False] * 4 + [True, False]
    assert table.loc["crisis folds", "rows"] == table.loc[["2005", "2006", "2007"], "rows"].sum()
    crisis = oof[oof["year"].isin(CRISIS_FOLDS)]
    lo, hi, usable = cluster_bootstrap_ci(
        crisis["target"], crisis["pct"], crisis["company_name"], roc_auc_score, n_boot=50
    )
    row = table.loc["crisis folds"]
    assert (row["roc_auc_lo"], row["roc_auc_hi"], row["usable"]) == (lo, hi, usable)
    assert {"capture_5", "capture_5_lo", "capture_5_hi"} <= set(table.columns)
