"""The functions of the single notebook this casebook replaced (cells 6, 12, 20, 23, and 36 of it).

The reference the library is checked against. Never imported by the library.
Only the names carry a legacy_ prefix; the bodies are as published.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

PANEL_END = 2018


def legacy_build_label(frame, horizon):
    """Discrete-time hazard label at a given horizon.

    Returns the frame with `target` set, minus the terminal-year rows of failed
    companies and any row whose outcome window runs past PANEL_END. Written as a
    function because the horizon sensitivity check relabels at 1 and 3 years.
    """
    frame = frame.join(
        frame.groupby("company_name")["year"].max().rename("last_year"), on="company_name"
    )
    frame["failed"] = frame["status_label"] == "failed"
    frame["years_to_final"] = frame["last_year"] - frame["year"]
    frame = frame[~(frame["failed"] & (frame["years_to_final"] == 0))]
    frame = frame[frame["year"] + horizon <= PANEL_END].copy()
    frame["target"] = (frame["failed"] & frame["years_to_final"].between(1, horizon)).astype(int)
    return frame


def legacy_add_features(frame):
    """Engineered credit ratios. Called once here and again by the horizon check."""
    ta = frame["total_assets"].replace(0, np.nan)
    sales = frame["net_sales"].replace(0, np.nan)
    tl = frame["total_liabilities"].replace(0, np.nan)

    frame["current_ratio"] = frame["current_assets"] / ta
    frame["debt_to_assets"] = frame["total_lt_debt"] / ta
    frame["net_margin"] = frame["net_income"] / sales
    frame["ebitda_margin"] = frame["ebitda"] / sales
    frame["roa"] = frame["net_income"] / ta
    frame["gross_margin"] = frame["gross_profit"] / sales
    frame["asset_turnover"] = frame["net_sales"] / ta

    # debt_to_ebitda is meaningless when EBITDA is negative: the ratio flips sign and a
    # distressed firm sorts as conservatively levered. Blank it and carry the
    # information in an explicit flag instead.
    frame["negative_ebitda"] = (frame["ebitda"] <= 0).astype(int)
    frame["debt_to_ebitda"] = np.where(
        frame["ebitda"] > 0, frame["total_lt_debt"] / frame["ebitda"], np.nan
    )

    # Altman Z components. asset_turnover above is already the fifth term (Sales / TA).
    frame["working_capital_to_assets"] = (
        frame["current_assets"] - frame["total_current_liabilities"]
    ) / ta
    frame["retained_earnings_to_assets"] = frame["retained_earnings"] / ta
    frame["ebit_to_assets"] = frame["ebit"] / ta
    frame["mve_to_liabilities"] = frame["market_cap"] / tl

    # Gearing on total liabilities, not long-term debt alone.
    frame["total_liabilities_to_assets"] = frame["total_liabilities"] / ta
    return frame


LEGACY_NON_FEATURES = [
    "company_name",
    "status_label",
    "year",
    "target",
    "failed",
    "years_to_final",
    "last_year",
]


def legacy_fit_prepare(train_frame, cols):
    """Fit winsorization bounds and fill medians on the TRAINING split only.

    Returns a `prepare` closure plus the fitted statistics so they can be asserted on.
    Written as a function so the horizon check refits cleanly on its own train split
    rather than reusing statistics fitted here.
    """
    lower = train_frame[cols].quantile(0.01)
    upper = train_frame[cols].quantile(0.99)
    medians = train_frame[cols].clip(lower, upper, axis=1).median()

    def prepare(frame):
        return frame[cols].clip(lower, upper, axis=1).fillna(medians)

    return prepare, lower, upper, medians


def legacy_altman_z(frame):
    """Altman (1968) public-firm Z-score. Higher is safer."""
    return (
        1.2 * frame["working_capital_to_assets"]
        + 1.4 * frame["retained_earnings_to_assets"]
        + 3.3 * frame["ebit_to_assets"]
        + 0.6 * frame["mve_to_liabilities"]
        + 1.0 * frame["asset_turnover"]
    )


def legacy_pr_auc(y_true, scores):
    prec, rec, _ = precision_recall_curve(y_true, scores)
    return auc(rec, prec)


def legacy_capture_rate(y_true, scores, budgets=(0.01, 0.05, 0.10)):
    """Share of positives captured within the top-k% highest-scoring rows."""
    y_true = np.asarray(y_true)
    ranked = y_true[np.argsort(-np.asarray(scores), kind="mergesort")]
    total_pos = ranked.sum()
    rows = []
    for b in budgets:
        k = max(1, int(round(b * len(ranked))))  # noqa: RUF046 (as published)
        caught = ranked[:k].sum()
        rows.append(
            {
                "budget": f"{b:.0%}",
                "n_reviewed": k,
                "defaults_caught": int(caught),
                "capture_rate": caught / total_pos,
                "precision": caught / k,
                "lift": (caught / k) / y_true.mean(),
            }
        )
    return pd.DataFrame(rows)


def legacy_cluster_bootstrap_ci(y_true, scores, groups, metric, n_boot=1000, seed=42):
    """95% CI resampling whole companies, since firm-years within a company correlate."""
    rng = np.random.default_rng(seed)
    y_true, scores, groups = np.asarray(y_true), np.asarray(scores), np.asarray(groups)
    uniq = np.unique(groups)
    index_of = {g: np.flatnonzero(groups == g) for g in uniq}
    stats = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index_of[g] for g in drawn])
        y_b = y_true[idx]
        if y_b.sum() == 0 or y_b.sum() == len(y_b):
            continue  # degenerate resample, no positives or no negatives
        stats.append(metric(y_b, scores[idx]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)), len(stats)


def legacy_paired_bootstrap_diff_ci(
    y_true, scores_a, scores_b, groups, metric, n_boot=1000, seed=42
):
    """95% CI for metric(a) - metric(b) over the SAME company resamples."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores_a, scores_b = np.asarray(scores_a), np.asarray(scores_b)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    index_of = {g: np.flatnonzero(groups == g) for g in uniq}
    diffs = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index_of[g] for g in drawn])
        y_b = y_true[idx]
        if y_b.sum() == 0 or y_b.sum() == len(y_b):
            continue
        diffs.append(metric(y_b, scores_a[idx]) - metric(y_b, scores_b[idx]))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), len(diffs)
