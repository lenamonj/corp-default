"""Walk-forward out-of-fold scoring, and the scoreboard and paired table every chapter shares."""

from collections.abc import Callable
from itertools import pairwise

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from corp_default.metrics import capture_rate, company_resamples, within_year_percentile
from corp_default.splits import Fold

BUDGETS = (0.01, 0.05, 0.10)
SEED = 42
KEYS = ["company_name", "year"]


def walk_forward_oof(
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    frame: pd.DataFrame,
    folds: list[Fold],
) -> pd.DataFrame:
    """Score every validation row with a model fitted on its fold's training rows only.

    fit_predict(train_rows, valid_rows) returns one score per validation row, higher is riskier.
    pct is the score's percentile within its own year.
    """
    parts = []
    for fold in folds:
        train, valid = frame.loc[fold.train], frame.loc[fold.valid]
        scores = np.asarray(fit_predict(train, valid))
        if len(scores) != len(valid):
            raise ValueError(
                f"fit_predict returned {len(scores)} scores for {len(valid)} rows in fold {fold.year}"
            )
        parts.append(valid[[*KEYS, "target"]].assign(score=scores))
    oof = pd.concat(parts, ignore_index=True)
    return oof.assign(pct=within_year_percentile(oof["score"], oof["year"]))


def _aligned(scores: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Every model's rows in the first model's order; refuses models scored on different rows."""
    names = list(scores)
    first = scores[names[0]].reset_index(drop=True)
    index = pd.MultiIndex.from_frame(first[KEYS])
    aligned = {names[0]: first}
    for name in names[1:]:
        other = scores[name].set_index(KEYS)
        if len(other) != len(first) or not other.index.sort_values().equals(index.sort_values()):
            raise ValueError(f"{name} and {names[0]} are scored on different rows")
        other = other.reindex(index).reset_index()
        if not other["target"].equals(first["target"]):
            raise ValueError(f"{name} and {names[0]} carry different targets on the same rows")
        aligned[name] = other
    return aligned


def _ranking_metrics(y: np.ndarray, pct: np.ndarray) -> dict[str, float]:
    values = {"roc_auc": roc_auc_score(y, pct), "ap": average_precision_score(y, pct)}
    values["accuracy_ratio"] = 2 * values["roc_auc"] - 1
    table = capture_rate(y, pct, BUDGETS)
    for budget, capture, lift in zip(BUDGETS, table["capture_rate"], table["lift"]):
        values[f"capture_{round(budget * 100)}"] = capture
        values[f"lift_{round(budget * 100)}"] = lift
    return values


def _with_intervals(point: dict, draws: list[dict], n_boot: int) -> dict:
    """Each metric beside its 2.5 and 97.5 percentiles over the usable resamples."""
    if not draws:
        raise ValueError(f"no usable resample in {n_boot}")
    row = {}
    for key, value in point.items():
        stats = [draw[key] for draw in draws]
        row[key] = value
        row[f"{key}_lo"] = float(np.percentile(stats, 2.5))
        row[f"{key}_hi"] = float(np.percentile(stats, 97.5))
    row["usable"] = len(draws)
    return row


def scoreboard(scores: dict[str, pd.DataFrame], n_boot: int = 1000) -> pd.DataFrame:
    """One row per model on identical rows, every metric on `pct` with its cluster-bootstrap interval.

    Each company resample is drawn once and every metric is computed on it, in the same draw
    order as metrics.cluster_bootstrap_ci, so the intervals equal what that function gives.
    """
    rows = []
    for name, oof in _aligned(scores).items():
        y, pct = oof["target"].to_numpy(), oof["pct"].to_numpy()
        point = _ranking_metrics(y, pct)
        draws = [
            _ranking_metrics(y[idx], pct[idx])
            for idx in company_resamples(y, oof["company_name"], n_boot, SEED)
        ]
        rows.append(
            {"model": name, "rows": len(y), "positives": int(y.sum())}
            | _with_intervals(point, draws, n_boot)
        )
    return pd.DataFrame(rows)


def _differences(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc_diff": roc_auc_score(y, a) - roc_auc_score(y, b),
        "ap_diff": average_precision_score(y, a) - average_precision_score(y, b),
    }


def paired_table(
    scores: dict[str, pd.DataFrame], ladder: list[str], n_boot: int = 1000
) -> pd.DataFrame:
    """Each rung of the ladder against the rung before it, on the same company resamples."""
    aligned = _aligned({name: scores[name] for name in ladder})
    rows = []
    for before, rung in pairwise(ladder):
        y = aligned[rung]["target"].to_numpy()
        a, b = aligned[rung]["pct"].to_numpy(), aligned[before]["pct"].to_numpy()
        point = _differences(y, a, b)
        groups = aligned[rung]["company_name"]
        draws = [
            _differences(y[idx], a[idx], b[idx])
            for idx in company_resamples(y, groups, n_boot, SEED)
        ]
        rows.append({"model": rung, "against": before} | _with_intervals(point, draws, n_boot))
    return pd.DataFrame(rows)
