"""Development rows, walk-forward folds, and a holdout that stays sealed until Chapter 7."""

from typing import NamedTuple

import pandas as pd

HORIZON = 2
DEV_END = 2011
HOLDOUT = (2014, 2016)
FOLD_YEARS = tuple(range(2005, DEV_END + 1))

# The 2007 to 2009 recession's filings fall in 2008 and 2009. A final filed year is the fiscal year
# before the filing, so those events are final years 2007 and 2008. A fold is a crisis fold when its
# label window, final years Y + 1 to Y + HORIZON, reaches either. Fixed before any fold is scored.
CRISIS_FINAL_YEARS = (2007, 2008)
CRISIS_FOLDS = tuple(
    year
    for year in FOLD_YEARS
    if set(range(year + 1, year + HORIZON + 1)) & set(CRISIS_FINAL_YEARS)
)


class SealedHoldoutError(RuntimeError):
    """Raised when the holdout is read before Chapter 7 opens it."""


class Fold(NamedTuple):
    year: int
    train: pd.Index
    valid: pd.Index


def development(frame: pd.DataFrame) -> pd.DataFrame:
    """Fiscal years up to DEV_END: everything a model may be built and chosen on."""
    return frame.loc[frame["year"] <= DEV_END]


def holdout(frame: pd.DataFrame, *, unseal: bool = False) -> pd.DataFrame:
    """Fiscal years 2014 to 2016. Refuses unless unseal=True, which only Chapter 7 passes."""
    if not unseal:
        raise SealedHoldoutError(
            f"the {HOLDOUT[0]} to {HOLDOUT[1]} holdout is sealed until Chapter 7 opens it"
        )
    return frame.loc[frame["year"].between(*HOLDOUT)]


def walk_forward_folds(
    frame: pd.DataFrame, horizon: int = HORIZON, fold_years: tuple[int, ...] = FOLD_YEARS
) -> list[Fold]:
    """One fold per validation year Y: train on years up to Y - horizon - 1, validate on Y.

    Every training label resolves within `horizon` years, so it is known before Y begins.
    """
    return [
        Fold(
            year=year,
            train=frame.index[frame["year"] <= year - horizon - 1],
            valid=frame.index[frame["year"] == year],
        )
        for year in fold_years
    ]
