import pandas as pd
import pytest

from corp_default.labels import build_label
from corp_default.splits import (
    CRISIS_FINAL_YEARS,
    CRISIS_FOLDS,
    DEV_END,
    FOLD_YEARS,
    HOLDOUT,
    HORIZON,
    SealedHoldoutError,
    development,
    holdout,
    walk_forward_folds,
)


def years(first: int, last: int, companies: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"company_name": f"C_{c}", "year": y}
            for y in range(first, last + 1)
            for c in range(companies)
        ]
    )


def test_fixed_values():
    assert (HORIZON, DEV_END, HOLDOUT) == (2, 2011, (2014, 2016))
    assert FOLD_YEARS == tuple(range(2005, 2012))


def test_holdout_is_sealed():
    with pytest.raises(SealedHoldoutError, match="Chapter 7"):
        holdout(years(1999, 2016))


def test_holdout_unsealed_is_2014_to_2016():
    assert sorted(holdout(years(1999, 2016), unseal=True)["year"].unique()) == [2014, 2015, 2016]


def test_development_is_up_to_2011():
    assert development(years(1999, 2016))["year"].max() == DEV_END


def test_folds_respect_the_embargo():
    frame = years(1999, 2016)
    folds = walk_forward_folds(frame)
    assert [fold.year for fold in folds] == list(range(2005, 2012))
    for fold in folds:
        assert frame.loc[fold.train, "year"].max() + HORIZON < fold.year
        assert frame.loc[fold.train, "year"].max() == fold.year - 3
        assert set(frame.loc[fold.valid, "year"]) == {fold.year}


def test_development_and_holdout_share_no_year_and_leave_an_embargo():
    assert DEV_END + HORIZON < HOLDOUT[0]


@pytest.mark.needs_data
def test_folds_on_the_panel_match_the_review(raw):
    frame = development(build_label(raw, HORIZON))
    folds = walk_forward_folds(frame)
    positives = [int(frame.loc[fold.valid, "target"].sum()) for fold in folds]
    assert positives == [105, 114, 79, 57, 59, 47, 51]
    assert sum(positives) == 512
    first = frame.loc[folds[0].train]
    assert (len(first), int(first["target"].sum())) == (20_045, 155)


def test_crisis_folds_are_those_whose_labels_reach_the_2008_and_2009_filings():
    # A final filed year precedes the filing year, so 2008 and 2009 filings are final years 2007, 2008.
    assert CRISIS_FINAL_YEARS == (2007, 2008)
    assert CRISIS_FOLDS == (2005, 2006, 2007)


@pytest.mark.needs_data
def test_crisis_event_positives_by_fold(raw):
    frame = development(build_label(raw, HORIZON))
    crisis = frame["target"].eq(1) & frame["last_year"].isin(CRISIS_FINAL_YEARS)
    by_year = crisis.groupby(frame["year"]).sum()
    assert [int(by_year.loc[year]) for year in FOLD_YEARS] == [55, 114, 57, 0, 0, 0, 0]
