"""Ratios by family, history aligned on the calendar, and preprocessing fitted on training rows only."""

from collections.abc import Callable

import numpy as np
import pandas as pd

FAMILIES = {
    "leverage": [
        "liabilities_to_assets",
        "lt_debt_to_assets",
        "book_equity_to_assets",
        "debt_to_ebitda",
        "no_lt_debt",
    ],
    "profitability": [
        "roa",
        "ebit_to_assets",
        "ebitda_to_assets",
        "net_margin",
        "ebitda_margin",
        "gross_margin",
        "negative_ebitda",
        "retained_earnings_to_assets",
    ],
    "liquidity": [
        "current_ratio",
        "quick_ratio",
        "working_capital_to_assets",
        "current_assets_to_assets",
        "current_liabilities_to_liabilities",
    ],
    "coverage": ["ffo_to_liabilities"],
    "activity": ["asset_turnover"],
    "market": ["mve_to_liabilities", "market_to_book_assets"],
    "size": ["log_assets", "log_market_cap"],
}

HISTORY_RATIOS = [
    "roa",
    "liabilities_to_assets",
    "working_capital_to_assets",
    "mve_to_liabilities",
    "ebitda_margin",
    "ebitda_to_assets",
]
GROWTH_ITEMS = {"sales": "net_sales", "assets": "total_assets", "market_cap": "market_cap"}


def _nonzero(series: pd.Series) -> pd.Series:
    """A denominator with zeros blanked, so a ratio is NaN rather than infinite."""
    return series.replace(0, np.nan)


def _log_positive(series: pd.Series) -> pd.Series:
    return np.log(series.where(series > 0))


def add_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    """Every ratio in FAMILIES, added as columns. A zero denominator gives NaN."""
    ta = _nonzero(frame["total_assets"])
    tl = _nonzero(frame["total_liabilities"])
    tcl = _nonzero(frame["total_current_liabilities"])
    sales = _nonzero(frame["net_sales"])
    ebitda = frame["ebitda"]
    return frame.assign(
        liabilities_to_assets=frame["total_liabilities"] / ta,
        lt_debt_to_assets=frame["total_lt_debt"] / ta,
        book_equity_to_assets=(frame["total_assets"] - frame["total_liabilities"]) / ta,
        debt_to_ebitda=(frame["total_lt_debt"] / ebitda).where(ebitda > 0),
        no_lt_debt=(frame["total_lt_debt"] == 0).astype(int),
        roa=frame["net_income"] / ta,
        ebit_to_assets=frame["ebit"] / ta,
        ebitda_to_assets=ebitda / ta,
        net_margin=frame["net_income"] / sales,
        ebitda_margin=ebitda / sales,
        gross_margin=frame["gross_profit"] / sales,
        negative_ebitda=(ebitda <= 0).astype(int),
        retained_earnings_to_assets=frame["retained_earnings"] / ta,
        current_ratio=frame["current_assets"] / tcl,
        quick_ratio=(frame["current_assets"] - frame["inventory"]) / tcl,
        working_capital_to_assets=(frame["current_assets"] - frame["total_current_liabilities"])
        / ta,
        current_assets_to_assets=frame["current_assets"] / ta,
        current_liabilities_to_liabilities=frame["total_current_liabilities"] / tl,
        ffo_to_liabilities=(frame["net_income"] + frame["depreciation_amort"]) / tl,
        asset_turnover=frame["net_sales"] / ta,
        mve_to_liabilities=frame["market_cap"] / tl,
        market_to_book_assets=(frame["market_cap"] + frame["total_liabilities"]) / ta,
        log_assets=_log_positive(frame["total_assets"]),
        log_market_cap=_log_positive(frame["market_cap"]),
    )


def add_history(frame: pd.DataFrame, lags: tuple[int, ...] = (1, 2)) -> pd.DataFrame:
    """Changes, growth, and past losses against the same company `lag` calendar years earlier.

    Rows are paired by (company_name, year - lag), never by position, so a reporting gap gives
    NaN instead of pairing two years that are not `lag` apart.
    """
    keys = ["company_name", "year"]
    if frame.duplicated(keys).any():
        raise ValueError("add_history needs one row per (company_name, year)")
    by_key = frame.set_index(keys)
    logs = pd.DataFrame({name: _log_positive(by_key[item]) for name, item in GROWTH_ITEMS.items()})
    added = {}
    for lag in lags:
        earlier = pd.MultiIndex.from_arrays([frame["company_name"], frame["year"] - lag])
        past = by_key.reindex(earlier)
        past_logs = logs.reindex(earlier)
        for name in HISTORY_RATIOS:
            added[f"d{lag}_{name}"] = frame[name].to_numpy() - past[name].to_numpy()
        for name in GROWTH_ITEMS:
            added[f"g{lag}_{name}"] = logs[name].to_numpy() - past_logs[name].to_numpy()
        past_income = past["net_income"].to_numpy()
        added[f"loss_{lag}y_ago"] = np.where(
            np.isnan(past_income), np.nan, (past_income < 0).astype(float)
        )
    added["years_observed"] = frame.sort_values("year").groupby("company_name").cumcount() + 1
    return frame.assign(**added)


def fit_prepare(
    train: pd.DataFrame, cols: list[str], impute: bool = True
) -> tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.Series, pd.Series, pd.Series]:
    """Winsorisation bounds at the 1st and 99th percentiles and fill medians, from `train` only.

    impute=False leaves missing values in place for tree models, which route them themselves.
    """
    lower = train[cols].quantile(0.01)
    upper = train[cols].quantile(0.99)
    medians = train[cols].clip(lower, upper, axis=1).median()

    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        clipped = frame[cols].clip(lower, upper, axis=1)
        return clipped.fillna(medians) if impute else clipped

    return prepare, lower, upper, medians
