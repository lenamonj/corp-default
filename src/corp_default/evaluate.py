"""Walk-forward out-of-fold scoring, and the scoreboard and paired table every chapter shares."""

from collections.abc import Callable
from itertools import pairwise

import numpy as np
import pandas as pd

from corp_default.metrics import company_resamples, within_year_percentile
from corp_default.splits import CRISIS_FOLDS, Fold

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


def _levels(pct: np.ndarray) -> np.ndarray:
    """Each score's rank among the distinct scores, 0 for the lowest. Equal scores share a level."""
    return np.unique(pct, return_inverse=True)[1]


def _roc_auc_and_ap(y: np.ndarray, level: np.ndarray) -> tuple[float, float]:
    """ROC-AUC and average precision from the count of rows at each score level.

    They equal scikit-learn's roc_auc_score and average_precision_score, ties included. Counting
    needs no sort, which keeps a thousand resamples quick.
    """
    rows = np.bincount(level)
    pos = np.bincount(level, weights=y, minlength=len(rows))
    neg = rows - pos
    roc_auc = (pos * (np.cumsum(neg) - neg / 2)).sum() / (pos.sum() * neg.sum())
    caught = np.cumsum(pos[::-1])  # from the riskiest level down
    reviewed = np.cumsum(rows[::-1])
    ap = (pos[::-1] * caught / np.maximum(reviewed, 1)).sum() / pos.sum()
    return float(roc_auc), float(ap)


def _captured(
    y: np.ndarray, pct: np.ndarray, budgets: tuple[float, ...]
) -> list[tuple[float, float]]:
    """Capture rate and lift at each budget: the numbers of metrics.capture_rate without its table."""
    ranked = y[np.argsort(-pct, kind="mergesort")]
    total_pos, base_rate = ranked.sum(), y.mean()
    pairs = []
    for budget in budgets:
        k = max(1, round(budget * len(ranked)))
        caught = ranked[:k].sum()
        pairs.append((caught / total_pos, (caught / k) / base_rate))
    return pairs


def _ranking_metrics(y: np.ndarray, pct: np.ndarray, level: np.ndarray) -> dict[str, float]:
    roc_auc, ap = _roc_auc_and_ap(y, level)
    values = {"roc_auc": roc_auc, "ap": ap, "accuracy_ratio": 2 * roc_auc - 1}
    for budget, (capture, lift) in zip(BUDGETS, _captured(y, pct, BUDGETS)):
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

    Each company resample is drawn once and every model and metric is computed on it, in the draw
    order of metrics.cluster_bootstrap_ci, so the intervals equal what that function gives.
    """
    aligned = _aligned(scores)
    first = next(iter(aligned.values()))
    y = first["target"].to_numpy()
    ranked = {name: oof["pct"].to_numpy() for name, oof in aligned.items()}
    levels = {name: _levels(pct) for name, pct in ranked.items()}
    draws = {name: [] for name in ranked}
    for idx in company_resamples(y, first["company_name"], n_boot, SEED):
        for name, pct in ranked.items():
            draws[name].append(_ranking_metrics(y[idx], pct[idx], levels[name][idx]))
    return pd.DataFrame(
        [
            {"model": name, "rows": len(y), "positives": int(y.sum())}
            | _with_intervals(_ranking_metrics(y, pct, levels[name]), draws[name], n_boot)
            for name, pct in ranked.items()
        ]
    )


def _differences(a: tuple[float, float], b: tuple[float, float]) -> dict[str, float]:
    """ROC-AUC and average precision of one model less those of another."""
    return {"roc_auc_diff": a[0] - b[0], "ap_diff": a[1] - b[1]}


def paired_table(
    scores: dict[str, pd.DataFrame], ladder: list[str], n_boot: int = 1000
) -> pd.DataFrame:
    """Each rung of the ladder against the rung before it, on the same company resamples."""
    aligned = _aligned({name: scores[name] for name in ladder})
    first = aligned[ladder[0]]
    y = first["target"].to_numpy()
    levels = {name: _levels(oof["pct"].to_numpy()) for name, oof in aligned.items()}
    steps = list(pairwise(ladder))
    draws = {step: [] for step in steps}
    for idx in company_resamples(y, first["company_name"], n_boot, SEED):
        drawn = {name: _roc_auc_and_ap(y[idx], level[idx]) for name, level in levels.items()}
        for before, rung in steps:
            draws[before, rung].append(_differences(drawn[rung], drawn[before]))
    point = {name: _roc_auc_and_ap(y, level) for name, level in levels.items()}
    return pd.DataFrame(
        [
            {"model": rung, "against": before}
            | _with_intervals(_differences(point[rung], point[before]), draws[before, rung], n_boot)
            for before, rung in steps
        ]
    )


def _fold_metrics(y: np.ndarray, pct: np.ndarray, level: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": _roc_auc_and_ap(y, level)[0],
        "capture_5": _captured(y, pct, (0.05,))[0][0],
    }


def fold_table(oof: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """ROC-AUC and capture at 5% for each fold year, then pooled over crisis and calm folds.

    Average precision is left out: it moves with the base rate, which differs between folds.
    """
    crisis = oof["year"].isin(CRISIS_FOLDS)
    parts = [
        (str(year), year in CRISIS_FOLDS, oof["year"] == year) for year in sorted(set(oof["year"]))
    ]
    parts += [("crisis folds", True, crisis), ("calm folds", False, ~crisis)]
    rows = []
    for label, is_crisis, mask in parts:
        part = oof.loc[mask]
        y, pct = part["target"].to_numpy(), part["pct"].to_numpy()
        level = _levels(pct)
        draws = [
            _fold_metrics(y[idx], pct[idx], level[idx])
            for idx in company_resamples(y, part["company_name"], n_boot, SEED)
        ]
        rows.append(
            {"fold": label, "crisis": is_crisis, "rows": len(y), "positives": int(y.sum())}
            | _with_intervals(_fold_metrics(y, pct, level), draws, n_boot)
        )
    return pd.DataFrame(rows)
