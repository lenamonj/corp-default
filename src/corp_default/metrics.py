"""Ranking metrics a credit desk reads, intervals that respect companies, drift, and calibration."""

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def capture_rate(y_true, scores, budgets=(0.01, 0.05, 0.10)) -> pd.DataFrame:
    """Share of positives caught when the top `budget` share of rows by score is reviewed."""
    y_true = np.asarray(y_true)
    ranked = y_true[np.argsort(-np.asarray(scores), kind="mergesort")]
    total_pos = ranked.sum()
    rows = []
    for budget in budgets:
        k = max(1, round(budget * len(ranked)))
        caught = ranked[:k].sum()
        rows.append(
            {
                "budget": f"{budget:.0%}",
                "n_reviewed": k,
                "defaults_caught": int(caught),
                "capture_rate": caught / total_pos,
                "precision": caught / k,
                "lift": (caught / k) / y_true.mean(),
            }
        )
    return pd.DataFrame(rows)


def capture_curve(y_true, scores) -> tuple[np.ndarray, np.ndarray]:
    """Share of rows reviewed against share of positives captured, from (0, 0) to (1, 1)."""
    y_true = np.asarray(y_true)
    ranked = y_true[np.argsort(-np.asarray(scores), kind="mergesort")]
    reviewed = np.arange(len(ranked) + 1) / len(ranked)
    captured = np.concatenate([[0.0], np.cumsum(ranked) / ranked.sum()])
    return reviewed, captured


def accuracy_ratio(y_true, scores) -> float:
    """The Gini, or accuracy ratio of the cumulative accuracy profile: 2 * ROC-AUC - 1."""
    return 2.0 * roc_auc_score(y_true, scores) - 1.0


def trapezoid_pr_auc(y_true, scores) -> float:
    """Area under the precision-recall curve by the trapezoid rule. Optimistic; kept to show why."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return float(auc(recall, precision))


def within_year_percentile(scores: pd.Series, years: pd.Series) -> pd.Series:
    """Each score's percentile rank among the scores of its own year."""
    return scores.groupby(years).rank(pct=True)


def _company_resamples(y_true: np.ndarray, groups, n_boot: int, seed: int) -> Iterator[np.ndarray]:
    """Row indices of each usable company resample, in the published notebook's draw order.

    A resample with no positive or no negative is skipped; the caller counts what it receives.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    index_of = pd.Series(np.arange(len(groups))).groupby(groups).indices
    for _ in range(n_boot):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_of[g] for g in drawn])
        positives = y_true[idx].sum()
        if 0 < positives < len(idx):
            yield idx


def _interval(stats: list[float], n_boot: int) -> tuple[float, float, int]:
    if not stats:
        raise ValueError(
            f"no usable resample in {n_boot}: every one lacked a positive or a negative"
        )
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)), len(stats)


def cluster_bootstrap_ci(
    y_true, scores, groups, metric, n_boot=1000, seed=42
) -> tuple[float, float, int]:
    """95% interval of `metric` resampling whole companies, and how many resamples were usable."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    stats = [
        metric(y_true[idx], scores[idx]) for idx in _company_resamples(y_true, groups, n_boot, seed)
    ]
    return _interval(stats, n_boot)


def paired_bootstrap_diff_ci(
    y_true, scores_a, scores_b, groups, metric, n_boot=1000, seed=42
) -> tuple[float, float, int]:
    """95% interval of metric(a) - metric(b) on the same company resamples, and the usable count."""
    y_true, scores_a, scores_b = np.asarray(y_true), np.asarray(scores_a), np.asarray(scores_b)
    diffs = [
        metric(y_true[idx], scores_a[idx]) - metric(y_true[idx], scores_b[idx])
        for idx in _company_resamples(y_true, groups, n_boot, seed)
    ]
    return _interval(diffs, n_boot)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n <= 0:
        raise ValueError(f"wilson_interval needs n > 0, got n = {n}")
    p = k / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def psi(expected, actual, bins: int = 10) -> float:
    """Population stability index of `actual` against `expected`, on missing-free values.

    Bin edges are the quantiles of `expected`. Empty bins are floored at 1e-6 so the log is finite.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected, actual = expected[~np.isnan(expected)], actual[~np.isnan(actual)]
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)[1:-1]))
    e = np.bincount(np.searchsorted(edges, expected, side="right"), minlength=len(edges) + 1)
    a = np.bincount(np.searchsorted(edges, actual, side="right"), minlength=len(edges) + 1)
    e = np.maximum(e / e.sum(), 1e-6)
    a = np.maximum(a / a.sum(), 1e-6)
    return float(np.sum((a - e) * np.log(a / e)))


def reliability_table(y_true, prob, bins: int = 10) -> pd.DataFrame:
    """Realised rate against mean predicted probability in quantile bins, with Wilson intervals."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "prob": np.asarray(prob, dtype=float)})
    frame["bin"] = pd.qcut(frame["prob"], bins, duplicates="drop")
    table = frame.groupby("bin", observed=True).agg(
        n=("y", "size"), events=("y", "sum"), mean_prob=("prob", "mean")
    )
    table["rate"] = table["events"] / table["n"]
    bounds = [wilson_interval(int(k), int(n)) for k, n in zip(table["events"], table["n"])]
    table["lo"] = [lo for lo, _ in bounds]
    table["hi"] = [hi for _, hi in bounds]
    return table[["n", "mean_prob", "rate", "lo", "hi"]].reset_index(drop=True)
