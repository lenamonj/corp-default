"""Kaplan-Meier with delayed entry, and cumulative incidence when there is more than one way out."""

import numpy as np
import pandas as pd


def kaplan_meier(duration, event, entry=None) -> pd.DataFrame:
    """Survival at each distinct duration, with Greenwood standard errors.

    A subject is at risk at time t when entry < t <= duration; entry defaults to 0. Delayed entry
    handles a company first seen part-way through its life: it cannot fail before it is observed.
    """
    duration = np.asarray(duration)
    event = np.asarray(event).astype(bool)
    entry = np.zeros_like(duration) if entry is None else np.asarray(entry)
    if (duration <= entry).any():
        raise ValueError("every duration must be later than its entry")
    times = np.unique(duration)
    at_risk = np.array([((entry < t) & (duration >= t)).sum() for t in times])
    events = np.array([(event & (duration == t)).sum() for t in times])
    survival = np.cumprod(1 - events / at_risk)
    # Greenwood; a step where every subject at risk fails leaves survival at 0 with se 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(at_risk > events, events / (at_risk * (at_risk - events)), 0.0)
    se = survival * np.sqrt(np.cumsum(terms))
    return pd.DataFrame(
        {"time": times, "at_risk": at_risk, "events": events, "survival": survival, "se": se}
    )


def cumulative_incidence(
    duration, cause, causes: tuple[str, ...] = ("failed", "left")
) -> pd.DataFrame:
    """Aalen-Johansen cumulative incidence of each cause; any other value of `cause` is censored.

    survival is the chance of no exit of any listed cause by time t, so survival and the
    incidence columns sum to one at every time.
    """
    duration = np.asarray(duration)
    cause = np.asarray(cause)
    times = np.unique(duration)
    at_risk = np.array([(duration >= t).sum() for t in times])
    by_cause = {
        c: np.array([((duration == t) & (cause == c)).sum() for t in times]) for c in causes
    }
    all_events = sum(by_cause.values())
    survival = np.cumprod(1 - all_events / at_risk)
    before = np.concatenate([[1.0], survival[:-1]])
    table = {"time": times, "at_risk": at_risk, "survival": survival}
    for c in causes:
        table[f"cif_{c}"] = np.cumsum(before * by_cause[c] / at_risk)
    return pd.DataFrame(table)
