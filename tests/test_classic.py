import numpy as np
import pandas as pd
import pytest
from legacy import legacy_add_features, legacy_altman_z

from corp_default import classic
from corp_default.classic import ALTMAN_ZONES, altman_z, altman_z2, altman_zone
from corp_default.features import add_ratios


def company() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "current_assets": 50.0,
                "total_current_liabilities": 30.0,
                "total_assets": 100.0,
                "retained_earnings": 10.0,
                "ebit": 8.0,
                "market_cap": 60.0,
                "total_liabilities": 40.0,
                "net_sales": 120.0,
            }
        ]
    )


def ratios(frame: pd.DataFrame) -> pd.DataFrame:
    ta, tl = frame["total_assets"], frame["total_liabilities"]
    return frame.assign(
        working_capital_to_assets=(frame["current_assets"] - frame["total_current_liabilities"])
        / ta,
        retained_earnings_to_assets=frame["retained_earnings"] / ta,
        ebit_to_assets=frame["ebit"] / ta,
        mve_to_liabilities=frame["market_cap"] / tl,
        asset_turnover=frame["net_sales"] / ta,
    )


def test_altman_z_by_hand():
    # X1 = (50 - 30) / 100 = 0.2, X2 = 10 / 100 = 0.1, X3 = 8 / 100 = 0.08,
    # X4 = 60 / 40 = 1.5, X5 = 120 / 100 = 1.2
    # Z = 1.2 * 0.2 + 1.4 * 0.1 + 3.3 * 0.08 + 0.6 * 1.5 + 1.0 * 1.2 = 2.744
    assert altman_z(ratios(company())).iloc[0] == pytest.approx(2.744)


def test_altman_z2_by_hand():
    # X4 is book equity over total liabilities: (100 - 40) / 40 = 1.5
    # Z'' = 6.56 * 0.2 + 3.26 * 0.1 + 6.72 * 0.08 + 1.05 * 1.5 = 3.7506
    assert altman_z2(ratios(company())).iloc[0] == pytest.approx(3.7506)


def test_altman_z2_is_missing_without_liabilities():
    frame = ratios(company().assign(total_liabilities=0.0))
    assert np.isnan(altman_z2(frame).iloc[0])


def test_altman_zones_follow_altman_1968():
    assert ALTMAN_ZONES == {"distress_below": 1.81, "safe_above": 2.99}
    zones = altman_zone(pd.Series([1.5, 1.81, 2.5, 2.99, 3.5, np.nan]))
    assert list(zones.iloc[:5]) == ["distress", "grey", "grey", "grey", "safe"]
    assert pd.isna(zones.iloc[5])


def test_no_unverified_score_is_written():
    # Zmijewski (1984) coefficients are under "Not verified" in SOURCES.md
    assert not hasattr(classic, "zmijewski")


@pytest.mark.needs_data
def test_altman_z_equals_legacy_on_the_validation_window(raw):
    val = raw[raw["year"].between(2009, 2011)]
    np.testing.assert_array_equal(
        altman_z(add_ratios(val)).to_numpy(), legacy_altman_z(legacy_add_features(val.copy()))
    )
