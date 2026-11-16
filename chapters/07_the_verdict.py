# %% [markdown]
# # Chapter 7: The Verdict
#
# The models are frozen, the choice is made, and the years no model has seen are opened once.
#
# > This casebook is educational content created in a personal capacity
#
# > Nothing here is investment advice or a recommendation to buy or sell any security
#
# > The data is a public research dataset with anonymised company names

# %% [markdown]
# ## The Question
#
# Six chapters built a label, a set of ratios, a yardstick, and seven models.
# Every choice among them was made on the same development years, and a choice made on data flatters itself.
#
# Three later years have been kept sealed the whole time.
# What does the sealed holdout say?

# %% [markdown]
# ## What You Will Learn
#
# By the end you can:
#
# - fix a model, a choice, and a comparison before a test set is opened;
# - turn a ranking into a probability of default, and say what the probability leaves out;
# - read a holdout result with its interval, and tell a coarse ordering from a fine one;
# - explain a score by ratio family, and for one company;
# - write the limits of a study in plain words.

# %% [markdown]
# ## What You Need
#
# Chapter 3, which wrote the `features` table, and Chapters 4 to 6, which wrote the out-of-fold scores of the seven rungs and the record of how each is built.
#
# This is the only chapter that reads the outcomes of the holdout years.
# It does so in one cell.
#
# It writes the results that the front page of this casebook quotes.

# %% [markdown]
# ## Your Route

# %% [markdown]
# # Setting Up
#
# Make sure the Jupyter kernel `corp-default` is selected before you run a cell.
# If it is not in the list, or an import fails, the [Quick Start](../README.md#quick-start) shows how to register it and refresh the list.

# %% [markdown]
# ## The Data Dictionary
#
# The chapter reads `features`, the out-of-fold scores, and the record of the seven models.
# It builds two columns of its own.
#
# | Name | What it holds |
# | --- | --- |
# | **PD** | A probability of default: the chance that a firm-year is followed by a final filing within two years |
# | **Grade** | One of seven bands of a score's percentile within its year, from the safest, one, to the riskiest, seven |
#
# Five words are new.
#
# | Term | What it means |
# | --- | --- |
# | **Calibration** | Turning a score into a probability that matches how often the event happens |
# | **Brier score** | The mean squared gap between a probability and the outcome, lower being better |
# | **Master scale** | A fixed set of grades, each with its own probability of default |
# | **Holdout** | The observation years {{ch7.holdout_first}} to {{ch7.holdout_last}}, sealed since Chapter 4 |
# | **SHAP value** | The part of one row's score that is credited to one of its inputs |
#
# Q. What does the chapter need before it starts?

# %%
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from corp_default import store, viz
from corp_default.evaluate import paired_table, scoreboard
from corp_default.features import FAMILIES
from corp_default.metrics import wilson_interval
from corp_default.splits import DEV_END, FOLD_YEARS, HOLDOUT, development

store.open_chapter(7)
viz.use_house_style()
SOURCE = "Lombardo et al. (2022), Kaggle copy"
OOF_SOURCE = f"{SOURCE}; out-of-fold rows, {FOLD_YEARS[0]} to {FOLD_YEARS[-1]}"
COLORS = viz.MODEL_COLORS

features = store.load_frame("features")
oof_scores = store.load_frame("oof_scores")
frozen = store.load_json("frozen")

# %% [markdown]
# **Reading.** The table of Chapter 3, the scores of Chapters 4 to 6, and the record of each model are in memory.
# `store.put` carries a number from a cell into the text, so no number in this chapter is typed by hand.
#
# Q. What do the scores so far look like?

# %%
oof_scores.head()

# %% [markdown]
# **Reading.** Each row is one firm-year scored by one model: the score, and its percentile within the year.
#
# Q. How large are the tables, and how do the seven rungs rank the same rows?

# %%
ladder = {
    name: part.drop(columns="model").reset_index(drop=True)
    for name, part in oof_scores.groupby("model", sort=False)
}
dev = development(features).reset_index(drop=True)
RATIOS = [name for members in FAMILIES.values() for name in members]

store.put("holdout_first", HOLDOUT[0])
store.put("holdout_last", HOLDOUT[1])
store.put("dev_first", int(dev["year"].min()))
store.put("dev_last", DEV_END)
store.put("n_rungs", len(ladder))
store.put("n_oof", len(next(iter(ladder.values()))))

print(features.shape, dev.shape)
oof_scores.info()
pd.DataFrame({name: oof["pct"] for name, oof in ladder.items()}).describe().T.round(3)

# %% [markdown]
# **Reading.** The {{ch7.n_rungs}} rungs each score the same {{ch7.n_oof:,}} out-of-fold rows.
# Every percentile runs evenly from zero to one, by construction, so the rungs differ only in which firm-years they put at the top.
#
# Q. What does the record say about each model, and which one was chosen?

# %%
record = pd.DataFrame(
    [
        {
            "rung": k + 1,
            "model": name,
            "kind": spec["kind"],
            "inputs": len(spec.get("features", spec.get("members", []))),
            "chosen": spec["chosen"],
        }
        for k, (name, spec) in enumerate(frozen.items())
    ]
).set_index("rung")
CHOSEN = record.loc[record["chosen"], "model"].item()
CLASSIC = record["model"].iloc[0]
walked = paired_table({CLASSIC: ladder[CLASSIC], CHOSEN: ladder[CHOSEN]}, [CLASSIC, CHOSEN]).iloc[0]
walked_board = scoreboard({CHOSEN: ladder[CHOSEN]}).iloc[0]

store.put("chosen", CHOSEN)
store.put("classic", CLASSIC)
store.put("walked_auc", walked_board["roc_auc"])
store.put("walked_capture_5", walked_board["capture_5"])
store.put("walked_step", walked["roc_auc_diff"])
store.put("walked_step_low", walked["roc_auc_diff_lo"])
store.put("walked_step_high", walked["roc_auc_diff_hi"])

record

# %% [markdown]
# **Reading.** The record holds every setting that Chapters 4 to 6 chose, and it marks one model as chosen: "{{ch7.chosen}}".
#
# Three things are fixed here, before any outcome of the holdout years is read.
#
# - **The models.** Each of the seven rungs is refitted on the development years, {{ch7.dev_first}} to {{ch7.dev_last}}, with the settings in the record and nothing tuned again.
# - **The choice.** The model named in Chapter 6 by two tests written before their results: the highest mean average precision over the seven folds, and the simpler model unless a blend beats it with an interval above zero.
# - **The primary comparison.** The chosen model against "{{ch7.classic}}", as a paired difference in ROC-AUC on the holdout, with the share of failures the chosen model finds at the middle budget beside it.
#
# On the walk-forward that difference is {{ch7.walked_step:.3f}}, with an interval from {{ch7.walked_step_low:.3f}} to {{ch7.walked_step_high:.3f}}.
# The chosen model scores {{ch7.walked_auc:.3f}} there and finds {{ch7.walked_capture_5:.1%}} of failures at the middle budget.
#
# Everything else the chapter reports about the holdout is description, and is to be read as such.
#
# **Next, a probability for every score, still on the development years.**

# %% [markdown]
# # From Scores to Probabilities
#
# A ranking tells a desk which names to read first.
# A credit decision also needs a level: a **probability of default**, or PD.
#
# The model's own output is a poor PD.
# Chapter 5 found it too low on average, because each fold's model learned the failure rate of its training years.
#
# So the PD is built from the scale the ladder already uses, the percentile within the year.
# That scale means the same thing for every fold's model and for the final one.
#
# Two standard maps from a score to a probability are tried.
# **Platt scaling** fits a smooth curve, a logistic regression on the score, and **isotonic regression** fits a staircase that may only step up.
#
# Each is **cross-fitted** by year: the map for one year is fitted on the other six.
# The rule is written first: the map with the lower Brier score on those cross-fitted probabilities is the one that goes forward.
#
# Q. Which map does the rule choose?


# %%
def logit(p: np.ndarray) -> np.ndarray:
    """Log odds, with the ends held just inside zero and one."""
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def platt(rows: pd.DataFrame) -> Callable:
    """A logistic curve from percentile to probability, fitted on `rows`."""
    curve = LogisticRegression(C=np.inf, max_iter=1000).fit(
        logit(rows["pct"])[:, None], rows["target"]
    )
    return lambda pct: curve.predict_proba(logit(pct)[:, None])[:, 1]


def isotonic(rows: pd.DataFrame) -> Callable:
    """A rising staircase from percentile to probability, fitted on `rows`."""
    return IsotonicRegression(out_of_bounds="clip").fit(rows["pct"], rows["target"]).predict


def cross_fit(make_map: Callable, rows: pd.DataFrame) -> np.ndarray:
    """A probability for every row, from a map fitted on the other years."""
    out = np.zeros(len(rows))
    for year in sorted(rows["year"].unique()):
        held = (rows["year"] == year).to_numpy()
        out[held] = make_map(rows.loc[~held])(rows.loc[held, "pct"])
    return out


chosen_oof = ladder[CHOSEN]
MAPS = {"Platt scaling": platt, "Isotonic regression": isotonic}
crossed = {name: cross_fit(make_map, chosen_oof) for name, make_map in MAPS.items()}
brier = pd.Series(
    {name: np.mean((pd_ - chosen_oof["target"]) ** 2) for name, pd_ in crossed.items()}
)
base_rate = chosen_oof["target"].mean()
brier["One probability for every row"] = np.mean((base_rate - chosen_oof["target"]) ** 2)
MAP = brier.drop("One probability for every row").idxmin()
pd_oof = crossed[MAP]
# The reading says isotonic regression is chosen and both maps beat a single number. Checked here.
assert MAP == "Isotonic regression"
assert brier[list(MAPS)].max() < brier["One probability for every row"]

store.put("map", MAP.lower())
store.put("brier_chosen", brier[MAP])
store.put("brier_other", brier[list(MAPS)].max())
store.put("brier_flat", brier["One probability for every row"])
store.put("base_rate", base_rate)

brier.round(5).to_frame("Brier score, cross-fitted by year")

# %% [markdown]
# **Reading.** The rule chooses {{ch7.map}}, at {{ch7.brier_chosen:.5f}} against {{ch7.brier_other:.5f}}.
# Giving every row the same {{ch7.base_rate:.1%}} scores {{ch7.brier_flat:.5f}}, so most of a Brier score here is the rarity of failure and the maps differ in its last digits.
#
# A probability is checked by sorting rows into bins and comparing each bin's mean probability with the share that failed.
# The library's function for that is short.


# %%
# taught: corp_default.metrics.reliability_table
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


# %% [markdown]
# **Reading.** `qcut` makes bins of equal size, and each bin gets its mean probability, its realised rate, and a Wilson interval for that rate.
#
# Q. Do the cross-fitted probabilities match what happened?

# %%
fig, ax = plt.subplots(figsize=(viz.EXHIBIT_WIDTH_IN, 4.2))
tables = {name: reliability_table(chosen_oof["target"], pd_) for name, pd_ in crossed.items()}
for name, table in tables.items():
    color = COLORS["trees"] if name == MAP else viz.MIST
    ax.vlines(table["mean_prob"], table["lo"], table["hi"], color=color, linewidth=1.2)
    ax.scatter(table["mean_prob"], table["rate"], s=30, color=color, zorder=3, label=name)
ends = [min(t["mean_prob"].min() for t in tables.values()) * 0.8, 0.2]
ax.plot(ends, ends, color=viz.INK, linewidth=0.8)
viz.log_axis(ax, "x")
viz.log_axis(ax, "y")
ax.set_xlabel("Mean probability of default in the bin")
ax.set_ylabel("Share of the bin that failed")
ax.legend(frameon=False, loc="upper left")
inside = (tables[MAP]["lo"] <= tables[MAP]["mean_prob"]) & (
    tables[MAP]["mean_prob"] <= tables[MAP]["hi"]
)

store.put("bins_inside", int(inside.sum()))
store.put("n_bins", len(tables[MAP]))
store.put("top_bin_pd", tables[MAP]["mean_prob"].iloc[-1])
store.put("top_bin_rate", tables[MAP]["rate"].iloc[-1])

viz.exhibit(
    fig,
    f"In {int(inside.sum())} of {len(inside)} bins the probability sits inside the interval of what happened",
    "Each dot is a bin of firm-years, with a 95% Wilson interval. On the line, probability and "
    "outcome agree.",
    OOF_SOURCE,
)

# %% [markdown]
# **Reading.** In Exhibit 1 a dot on the line is a bin whose probability matched its outcome.
# For the chosen map {{ch7.bins_inside}} of {{ch7.n_bins}} bins have the probability inside the interval, and the riskiest bin is given {{ch7.top_bin_pd:.1%}} and failed at {{ch7.top_bin_rate:.1%}}.
#
# That is the pooled picture, all seven years together.
#
# Q. Does the probability hold year by year?

# %%
by_year = (
    chosen_oof.assign(pd=pd_oof)
    .groupby("year")
    .agg(
        rows=("target", "size"),
        failed=("target", "sum"),
        predicted=("pd", "mean"),
        rate=("target", "mean"),
    )
)
bounds = np.array(
    [wilson_interval(int(k), int(n)) for k, n in zip(by_year["failed"], by_year["rows"])]
)
own_rows = MAPS[MAP](chosen_oof)(
    chosen_oof["pct"]
)  # the naive check: the map judged on the rows it was fitted on
gap = (by_year["predicted"] - by_year["rate"]).abs()

fig, ax = plt.subplots(figsize=(viz.EXHIBIT_WIDTH_IN, 3.4))
ax.vlines(by_year.index, bounds[:, 0], bounds[:, 1], color=viz.INK, linewidth=1.2)
ax.scatter(by_year.index, by_year["rate"], s=30, color=viz.INK, zorder=3)
ax.plot(by_year.index, by_year["predicted"], color=COLORS["trees"], linewidth=1.8)
ax.annotate("mean probability", (by_year.index[-1], by_year["predicted"].iloc[-1]), xytext=(0, 8),
            textcoords="offset points", color=COLORS["trees"], size=9, ha="right")  # fmt: skip
ax.annotate("share that failed", (by_year.index[0], by_year["rate"].iloc[0]), xytext=(10, 0),
            textcoords="offset points", color=viz.INK, size=9, va="center")  # fmt: skip
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
ax.set_ylim(0, None)
ax.set_xlabel("Observation year")
ax.grid(axis="x", visible=False)

store.put("pooled_gap", abs(own_rows.mean() - base_rate))
store.put("worst_year", int(gap.idxmax()))
store.put("worst_gap", gap.max())
store.put("worst_predicted", by_year.loc[gap.idxmax(), "predicted"])
store.put("worst_rate", by_year.loc[gap.idxmax(), "rate"])
store.put("flat_low", by_year["predicted"].min())
store.put("flat_high", by_year["predicted"].max())
store.put("rate_low", by_year["rate"].min())
store.put("rate_high", by_year["rate"].max())
# The reading says the probability barely moves while the failure rate halves. Checked here.
assert by_year["predicted"].max() / by_year["predicted"].min() < 1.2
assert by_year["rate"].max() / by_year["rate"].min() > 2

viz.exhibit(
    fig,
    f"The mean probability stays near {base_rate:.1%} while the share that failed runs from "
    f"{by_year['rate'].min():.1%} to {by_year['rate'].max():.1%}",
    "Cross-fitted probability of the chosen model against outcomes, by observation year, with 95% "
    "Wilson intervals.",
    OOF_SOURCE,
)

# %% [markdown]
# <!-- pitfall: pooled_calibration -->

# %% [markdown]
# **Reading.** Exhibit 2 shows what a pooled check hides.
# The mean probability stays between {{ch7.flat_low:.1%}} and {{ch7.flat_high:.1%}}, and the share that failed runs from {{ch7.rate_low:.1%}} to {{ch7.rate_high:.1%}}.
#
# In {{ch7.worst_year}} the probability says {{ch7.worst_predicted:.1%}} and {{ch7.worst_rate:.1%}} failed.
# A percentile within the year cannot know whether the year is a good one or a bad one, and no input of this model can tell it.
#
# So this PD is an average over the cycle, right across years and wrong in most single ones.
# The casebook carries it forward with that label on it.
#
# **Check.** The file holds accounts and a market value, and no interest rates or credit spreads. Why does that matter for the level of a PD more than for a ranking?
#
# <details>
# <summary>Show the answer</summary>
#
# A ranking compares companies with each other in the same year, and a bad year is bad for all of them.
# The level is how many of them fail, which moves with the economy.
#
# Nothing in a company's own accounts says where the economy is going, so the level needs information this file does not have.
#
# </details>
#
# **Next, grades.**

# %% [markdown]
# # A Master Scale
#
# A desk rarely works with a probability to four decimals.
# It works with **grades**, a short ladder of bands, each with a PD attached.
#
# The scale here has seven grades, set as bands of the percentile within the year.
# Grade one is the safest {{ch7.band_1:.0%}} of firm-years, and grade seven the riskiest {{ch7.band_7:.0%}}.
#
# The bands are narrow at the risky end, where the PD changes fastest and a desk looks hardest.
# They are fixed here, and the PD of a grade is the mean cross-fitted PD of its firm-years.
#
# Q. Do the grades order the failures on the development years?

# %%
GRADE_EDGES = (
    0.30,
    0.60,
    0.80,
    0.90,
    0.95,
    0.98,
)  # upper ends of grades one to six, as percentiles


def grade_of(pct: pd.Series) -> np.ndarray:
    """The grade, one to seven, of each percentile within its year."""
    return np.digitize(pct, GRADE_EDGES, right=True) + 1


def grade_table(rows: pd.DataFrame, pds: np.ndarray) -> pd.DataFrame:
    """Firm-years, failures, realised rate with its interval, and mean PD, for each grade."""
    table = (
        rows.assign(pd=pds, grade=grade_of(rows["pct"]))
        .groupby("grade")
        .agg(rows=("target", "size"), failed=("target", "sum"), pd=("pd", "mean"))
    )
    bounds = [wilson_interval(int(k), int(n)) for k, n in zip(table["failed"], table["rows"])]
    return table.assign(
        rate=table["failed"] / table["rows"],
        lo=[lo for lo, _ in bounds],
        hi=[hi for _, hi in bounds],
    )


def grade_ladder(table: pd.DataFrame, pds: pd.Series) -> plt.Figure:
    """Realised rate by grade with intervals, against the PD attached to each grade."""
    fig, ax = plt.subplots(figsize=(viz.EXHIBIT_WIDTH_IN, 3.6))
    ax.vlines(table.index, table["lo"], table["hi"], color=viz.INK, linewidth=1.2)
    ax.scatter(table.index, table["rate"], s=30, color=viz.INK, zorder=3, label="Share that failed")
    ax.scatter(pds.index, pds, s=120, marker="_", color=COLORS["trees"], linewidth=2.2, zorder=4,
               label="PD of the grade")  # fmt: skip
    viz.log_axis(ax, "y")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.set_xlabel("Grade, safest to riskiest")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="x", visible=False)
    return fig


grades_dev = grade_table(chosen_oof, pd_oof)
GRADE_PD = grades_dev["pd"]
step_up = grades_dev["rate"].diff().iloc[1:] > 0
# The reading says the rate rises with the grade except from five to six, where the intervals overlap.
assert list(step_up.index[~step_up]) == [6]
assert grades_dev.loc[6, "hi"] > grades_dev.loc[5, "lo"]

store.put("band_1", GRADE_EDGES[0])
store.put("band_7", 1 - GRADE_EDGES[-1])
store.put("grade_1_rate", grades_dev.loc[1, "rate"])
store.put("grade_7_rate", grades_dev.loc[7, "rate"])
store.put("grade_5_rate", grades_dev.loc[5, "rate"])
store.put("grade_6_rate", grades_dev.loc[6, "rate"])
store.put("grade_7_pd", GRADE_PD[7])

viz.exhibit(
    grade_ladder(grades_dev, GRADE_PD),
    f"From grade one to grade seven the share that failed rises from {grades_dev.loc[1, 'rate']:.1%} "
    f"to {grades_dev.loc[7, 'rate']:.1%}",
    "Out-of-fold firm-years of the chosen model by grade, with 95% Wilson intervals. A bar marks "
    "each grade's PD.",
    OOF_SOURCE,
)
grades_dev.round(4)

# %% [markdown]
# **Reading.** In Exhibit 3 the share that failed climbs from {{ch7.grade_1_rate:.1%}} in grade one to {{ch7.grade_7_rate:.1%}} in grade seven.
# One step is out of order: grade six failed at {{ch7.grade_6_rate:.1%}} and grade five at {{ch7.grade_5_rate:.1%}}, with intervals that overlap.
#
# The two are kept as separate grades, since merging them now would be a choice made after the result.
#
# The map, the bands, and the PD of each grade are now fixed.
# **Next, the holdout.**
