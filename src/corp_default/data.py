"""Loading the panel, its industry codes, and the span of each company's history."""

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA_PATH = REPO / "data" / "american_bankruptcy.csv"
INDUSTRY_PATH = REPO / "data" / "industry.csv"

COLUMN_MAP = {
    "X1": "current_assets",
    "X2": "cogs",
    "X3": "depreciation_amort",
    "X4": "ebitda",
    "X5": "inventory",
    "X6": "net_income",
    "X7": "total_receivables",
    "X8": "market_cap",
    "X9": "net_sales",
    "X10": "total_assets",
    "X11": "total_lt_debt",
    "X12": "ebit",
    "X13": "gross_profit",
    "X14": "total_current_liabilities",
    "X15": "retained_earnings",
    "X16": "total_revenue",
    "X17": "total_liabilities",
    "X18": "total_operating_expenses",
}
RAW_ITEMS = [name for name in COLUMN_MAP.values() if name != "total_revenue"]

_FETCH = "Run `uv run python -m tools.get_data` from the repo root, or see data/README.md."


def require_file(path: Path) -> None:
    """Stop with the one command that fetches the data when a file is not there."""
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found at {path}. {_FETCH}")


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """The panel with readable column names; total_revenue is dropped because it equals net_sales."""
    require_file(path)
    frame = pd.read_csv(path).rename(columns=COLUMN_MAP)
    return frame.drop(columns=["total_revenue"])


def load_industry(path: Path = INDUSTRY_PATH, data_path: Path = DATA_PATH) -> pd.DataFrame:
    """SIC division and major group per company, refused unless it covers the panel exactly."""
    require_file(path)
    require_file(data_path)
    codes = pd.read_csv(path).drop_duplicates()
    repeated = codes.loc[codes["company_name"].duplicated(), "company_name"]
    if not repeated.empty:
        raise ValueError(f"companies with more than one industry code: {sorted(set(repeated))[:5]}")
    companies = set(pd.read_csv(data_path, usecols=["company_name"])["company_name"])
    unmatched = companies.symmetric_difference(codes["company_name"])
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} companies are in only one of {path.name} and {data_path.name}, "
            f"for example {sorted(unmatched)[:5]}"
        )
    return codes.set_index("company_name")


def panel_spans(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per company: first and last year, years present, missing years inside the span."""
    grouped = frame.groupby("company_name")
    spans = grouped["year"].agg(first_year="min", last_year="max", n_years="nunique")
    spans["n_gap_years"] = spans["last_year"] - spans["first_year"] + 1 - spans["n_years"]
    spans["failed"] = grouped["status_label"].agg(lambda s: bool((s == "failed").any()))
    return spans
