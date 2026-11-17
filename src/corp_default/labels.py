"""The two-year hazard label and the event table of each company's history."""

import pandas as pd

PANEL_END = 2018


def build_label(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Label each firm-year: does the company's last filing fall 1 to `horizon` years later?

    Drops the final-year rows of failed companies, which have no future to predict, and every
    row whose outcome window runs past PANEL_END, where the outcome is not yet known.
    """
    last_year = frame.groupby("company_name")["year"].max().rename("last_year")
    frame = frame.join(last_year, on="company_name")
    frame = frame.assign(
        failed=frame["status_label"] == "failed",
        years_to_final=frame["last_year"] - frame["year"],
    )
    terminal = frame["failed"] & (frame["years_to_final"] == 0)
    frame = frame.loc[~terminal & (frame["year"] + horizon <= PANEL_END)]
    in_window = frame["failed"] & frame["years_to_final"].between(1, horizon)
    return frame.assign(target=in_window.astype(int))


def event_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per company: entry, exit, duration in years, and how the history ends.

    "failed" is a bankruptcy filing, "reporting" is still filing when the panel ends, and
    "left" is any other exit (acquisition, delisting, going private: the file does not say).
    """
    grouped = frame.groupby("company_name")
    events = grouped["year"].agg(entry_year="min", exit_year="max").reset_index()
    failed = grouped["status_label"].agg(lambda s: bool((s == "failed").any())).to_numpy()
    reporting = events["exit_year"] == PANEL_END
    events["duration"] = events["exit_year"] - events["entry_year"] + 1
    events["event"] = "left"
    events.loc[reporting, "event"] = "reporting"
    events.loc[failed, "event"] = "failed"
    return events
