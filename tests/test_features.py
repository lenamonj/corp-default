import numpy as np
import pandas as pd
import pytest
from legacy import (
    LEGACY_NON_FEATURES,
    legacy_add_features,
    legacy_build_label,
    legacy_fit_prepare,
)

from corp_default.data import RAW_ITEMS
from corp_default.features import FAMILIES, add_history, add_ratios, fit_prepare


def toy(**items) -> pd.DataFrame:
    row = {"company_name": "C_1", "year": 2001, "status_label": "alive"}
    row.update({name: 10.0 for name in RAW_ITEMS})
    row["total_assets"] = 100.0
    row.update(items)
    return pd.DataFrame([row])


def gap_company() -> pd.DataFrame:
    rows = [
        toy(year=year, net_income=income)
        for year, income in [(2001, 5.0), (2002, -3.0), (2005, 8.0)]
    ]
    return pd.concat(rows, ignore_index=True)


def test_families_are_in_order_and_name_every_ratio_once():
    assert list(FAMILIES) == [
        "leverage",
        "profitability",
        "liquidity",
        "coverage",
        "activity",
        "market",
        "size",
    ]
    names = [name for members in FAMILIES.values() for name in members]
    assert len(names) == len(set(names)) == 24
    assert set(names) <= set(add_ratios(toy()).columns)


def test_current_ratio_is_current_assets_over_current_liabilities():
    frame = add_ratios(toy(current_assets=30.0, total_current_liabilities=20.0, total_assets=100.0))
    assert frame["current_ratio"].iloc[0] == 1.5
    assert frame["current_assets_to_assets"].iloc[0] == 0.3


def test_ratios_are_never_infinite():
    frame = add_ratios(
        toy(total_assets=0.0, net_sales=0.0, total_liabilities=0.0, total_current_liabilities=0.0)
    )
    assert not np.isinf(frame.select_dtypes("number")).any().any()
    assert np.isnan(frame["roa"].iloc[0])


def test_flags_and_guarded_ratios():
    frame = add_ratios(toy(total_lt_debt=0.0, ebitda=-1.0, market_cap=-5.0))
    assert frame["no_lt_debt"].iloc[0] == 1
    assert frame["negative_ebitda"].iloc[0] == 1
    assert np.isnan(frame["debt_to_ebitda"].iloc[0])
    assert np.isnan(frame["log_market_cap"].iloc[0])
    assert frame["log_assets"].iloc[0] == pytest.approx(np.log(100.0))


def test_add_ratios_does_not_change_its_input():
    frame = toy()
    before = frame.copy()
    add_ratios(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_history_never_pairs_across_a_gap():
    # one company reporting in 2001, 2002, 2005
    frame = add_history(add_ratios(gap_company()))
    row = frame.set_index("year")
    assert np.isnan(row.loc[2005, "d1_roa"])
    assert row.loc[2002, "d1_roa"] == pytest.approx(row.loc[2002, "roa"] - row.loc[2001, "roa"])
    assert np.isnan(row.loc[2001, "d1_roa"])


def test_history_columns_by_lag():
    frame = add_history(add_ratios(gap_company())).set_index("year")
    assert frame.loc[2002, "loss_1y_ago"] == 0
    assert np.isnan(frame.loc[2005, "loss_1y_ago"])
    assert np.isnan(frame.loc[2005, "loss_2y_ago"])
    assert frame.loc[2002, "g1_assets"] == pytest.approx(0.0)
    assert list(frame["years_observed"]) == [1, 2, 3]
    for lag in (1, 2):
        for name in ("sales", "assets", "market_cap"):
            assert f"g{lag}_{name}" in frame.columns


def test_history_refuses_a_repeated_company_year():
    frame = add_ratios(pd.concat([toy(), toy()], ignore_index=True))
    with pytest.raises(ValueError, match="company_name, year"):
        add_history(frame)


@pytest.mark.needs_data
def test_shared_ratios_equal_legacy(raw):
    ours = add_ratios(raw)
    theirs = legacy_add_features(raw.copy())
    same = {
        "roa": "roa",
        "net_margin": "net_margin",
        "ebitda_margin": "ebitda_margin",
        "gross_margin": "gross_margin",
        "asset_turnover": "asset_turnover",
        "negative_ebitda": "negative_ebitda",
        "debt_to_ebitda": "debt_to_ebitda",
        "working_capital_to_assets": "working_capital_to_assets",
        "retained_earnings_to_assets": "retained_earnings_to_assets",
        "ebit_to_assets": "ebit_to_assets",
        "mve_to_liabilities": "mve_to_liabilities",
        "lt_debt_to_assets": "debt_to_assets",
        "liabilities_to_assets": "total_liabilities_to_assets",
        "current_assets_to_assets": "current_ratio",
    }
    for name, legacy_name in same.items():
        np.testing.assert_array_equal(
            ours[name].to_numpy(dtype=float),
            theirs[legacy_name].to_numpy(dtype=float),
            err_msg=name,
        )


@pytest.mark.needs_data
def test_fit_prepare_equals_legacy_on_the_legacy_features(raw):
    # legacy features, legacy train window 1999-2006: bounds and prepared matrix must be identical
    frame = legacy_add_features(legacy_build_label(raw.copy(), 2))
    cols = [c for c in frame.columns if c not in LEGACY_NON_FEATURES]
    train = frame[frame["year"] <= 2006]
    val = frame[frame["year"].between(2009, 2011)]
    prepare, lower, upper, medians = fit_prepare(train, cols)
    legacy_prepare, legacy_lower, legacy_upper, legacy_medians = legacy_fit_prepare(train, cols)
    pd.testing.assert_series_equal(lower, legacy_lower)
    pd.testing.assert_series_equal(upper, legacy_upper)
    pd.testing.assert_series_equal(medians, legacy_medians)
    pd.testing.assert_frame_equal(prepare(val), legacy_prepare(val))


def test_fit_prepare_can_leave_missing_values_for_trees():
    train_with_nan = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0, 100.0]})
    prepare, *_ = fit_prepare(train_with_nan, ["x"], impute=False)
    assert prepare(train_with_nan)["x"].isna().any()
    filled, *_ = fit_prepare(train_with_nan, ["x"])
    assert not filled(train_with_nan)["x"].isna().any()
