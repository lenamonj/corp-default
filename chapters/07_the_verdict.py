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
import shap
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator, PercentFormatter
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from xgboost import XGBClassifier

from corp_default import store, viz
from corp_default.classic import altman_z
from corp_default.evaluate import KEYS, SEED, paired_table, scoreboard
from corp_default.features import FAMILIES, fit_prepare
from corp_default.labels import PANEL_END
from corp_default.metrics import (
    capture_curve,
    cluster_bootstrap_ci,
    company_resamples,
    wilson_interval,
    within_year_percentile,
)
from corp_default.nets import TabularNet, predict, seed_everything, train_net
from corp_default.splits import DEV_END, FOLD_YEARS, HOLDOUT, development, holdout

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
# - **The choice.** The model named in Chapter 6 by two tests, both fixed before the holdout was opened: the highest mean average precision over the seven folds, and the simpler model unless a blend beats it with an interval above zero.
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
ax.annotate("mean PD", (by_year.index[-1], by_year["predicted"].iloc[-1]), xytext=(0, 8),
            textcoords="offset points", color=COLORS["trees"], size=9, ha="right")  # fmt: skip
ax.annotate("share that failed", (by_year.index[0], by_year["rate"].iloc[0]), xytext=(10, -22),
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
    f"The mean PD stays near {base_rate:.1%} while the share that failed runs from "
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


# %% [markdown]
# # Opening the Holdout
#
# First the seven models are fitted for the last time.
# Each recipe is read from the record, and each is fitted on every development row, {{ch7.dev_first}} to {{ch7.dev_last}}.
#
# The number of boosting rounds comes from the walk-forward, as Chapter 5 left it.
# The network chooses its epochs as it did in every fold, by stopping on the last two training years, and is then fitted on all of them.
#
# Q. How is each model fitted?


# %%
def fit_classic(train: pd.DataFrame, spec: dict) -> Callable:
    return lambda rows: -altman_z(rows).to_numpy()  # a low Z is the risky end


def fit_logistic(train: pd.DataFrame, spec: dict) -> Callable:
    prepare, *_ = fit_prepare(train, spec["features"])
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=spec.get("C", np.inf), max_iter=2000)
    )
    model.fit(prepare(train), train["target"])
    return lambda rows: model.predict_proba(prepare(rows))[:, 1]


def fit_trees(train: pd.DataFrame, spec: dict) -> Callable:
    prepare, *_ = fit_prepare(train, spec["features"], impute=False)
    model = XGBClassifier(
        n_estimators=spec["rounds"], n_jobs=1, random_state=SEED, **spec["params"]
    )
    model.fit(prepare(train), train["target"])
    score = lambda rows: model.predict_proba(prepare(rows))[:, 1]
    score.model, score.prepare = model, prepare  # kept for the part that explains the score
    return score


def fit_network(train: pd.DataFrame, spec: dict) -> Callable:
    columns = spec["features"]
    scale = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=SEED)
    scale.fit(train[columns])
    blanks = [name for name in columns if train[name].isna().any()]

    def inputs(rows: pd.DataFrame) -> torch.Tensor:
        bell = pd.DataFrame(scale.transform(rows[columns]), columns=columns, index=rows.index)
        flags = {f"{name}_blank": rows[name].isna().astype(float) for name in blanks}
        return torch.tensor(bell.fillna(0.0).assign(**flags).to_numpy(dtype=np.float32))

    data = (
        inputs(train),
        torch.tensor(train["target"].to_numpy(dtype=np.float32)),
        torch.ones(len(train)),
    )
    stop = torch.tensor((train["year"] > train["year"].max() - spec["stop_years"]).to_numpy())
    members = []
    for seed in spec["seeds"]:
        seed_everything(seed)
        make = lambda: TabularNet(data[0].shape[1], spec["hidden"], spec["dropout"])
        early, held = tuple(part[~stop] for part in data), tuple(part[stop] for part in data)
        stopped = train_net(make(), early, held, seed=seed, **spec["settings"])
        seed_everything(seed)
        model = make()
        train_net(
            model, data, None, seed=seed, **{**spec["settings"], "epochs": stopped["best_epoch"]}
        )
        members.append(model)
    return lambda rows: np.mean([predict(model, inputs(rows)) for model in members], axis=0)


FIT = {
    "published score": fit_classic,
    "logistic regression": fit_logistic,
    "gradient boosting": fit_trees,
    "tabular network": fit_network,
}
BLENDS = {name: spec["members"] for name, spec in frozen.items() if "members" in spec}

# Seven models on every development row. The network takes about a minute.
fitted = {name: FIT[spec["kind"]](dev, spec) for name, spec in frozen.items() if name not in BLENDS}


def score_rows(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every rung's score for `rows`, ranked within each year, in the form the ladder keeps."""

    def ranked(scores: np.ndarray) -> pd.DataFrame:
        table = rows[[*KEYS, "target"]].assign(score=scores)
        return table.assign(pct=within_year_percentile(table["score"], table["year"]))

    scored = {name: ranked(score(rows)) for name, score in fitted.items()}
    for name, members in BLENDS.items():
        scored[name] = ranked(np.mean([scored[member]["pct"] for member in members], axis=0))
    return {name: scored[name] for name in frozen}


# The final map from percentile to PD, fitted on all seven walk-forward years.
to_pd = MAPS[MAP](chosen_oof)

store.put("n_dev", len(dev))
store.put("dev_failures", int(dev["target"].sum()))
len(fitted), list(BLENDS)

# %% [markdown]
# **Reading.** Six models are fitted on {{ch7.n_dev:,}} firm-years with {{ch7.dev_failures}} failures, and the blend is the mean of two of them.
# Nothing in that cell looked at a year after {{ch7.dev_last}}.
#
# The next cell is the only one in the casebook that does.
# The holdout is the observation years {{ch7.holdout_first}} to {{ch7.holdout_last}}, and the two years between are left out because their labels overlap the last development years.
#
# **Predict 1.** The chosen model scored {{ch7.walked_auc:.3f}} on the walk-forward, and that number helped to choose it. Will its ROC-AUC on the holdout be higher, lower, or about the same?
#
# Q. What does the holdout say?

# %%
sealed = holdout(features, unseal=True).reset_index(drop=True)
verdict = score_rows(sealed)
board = scoreboard(verdict).set_index("model")
primary = paired_table(
    {CLASSIC: verdict[CLASSIC], CHOSEN: verdict[CHOSEN]}, [CLASSIC, CHOSEN]
).iloc[0]
top = board.loc[CHOSEN]

store.put("n_holdout", len(sealed))
store.put("holdout_failures", int(sealed["target"].sum()))
store.put("holdout_rate", sealed["target"].mean())
store.put("holdout_auc", top["roc_auc"])
store.put("holdout_auc_low", top["roc_auc_lo"])
store.put("holdout_auc_high", top["roc_auc_hi"])
store.put("holdout_capture_5", top["capture_5"])
store.put("holdout_capture_5_low", top["capture_5_lo"])
store.put("holdout_capture_5_high", top["capture_5_hi"])
store.put("holdout_lift_5", top["lift_5"])
store.put("classic_auc", board.loc[CLASSIC, "roc_auc"])
store.put("primary_step", primary["roc_auc_diff"])
store.put("primary_step_low", primary["roc_auc_diff_lo"])
store.put("primary_step_high", primary["roc_auc_diff_hi"])
# The reading says the holdout score is above the walk-forward one and the primary step is clear of zero.
assert top["roc_auc"] > walked_board["roc_auc"]
assert primary["roc_auc_diff_lo"] > 0

board[
    ["rows", "positives", "roc_auc", "roc_auc_lo", "roc_auc_hi", "ap", "capture_5", "lift_5"]
].round(3)

# %% [markdown]
# **Reading.** The chosen model scores {{ch7.holdout_auc:.3f}} on the holdout, with an interval from {{ch7.holdout_auc_low:.3f}} to {{ch7.holdout_auc_high:.3f}}.
# That is above its walk-forward score, which answers Predict 1.
#
# It does not show that the walk-forward score was free of flattery: the two samples differ in more than the model, as the next part says.
#
# The primary comparison is settled.
# The chosen model is ahead of the published score by {{ch7.primary_step:.3f}}, with an interval from {{ch7.primary_step_low:.3f}} to {{ch7.primary_step_high:.3f}}.
#
# The holdout has {{ch7.n_holdout:,}} firm-years and {{ch7.holdout_failures}} failures.
# The model's top twentieth of each year holds {{ch7.holdout_capture_5:.1%}} of them, with an interval from {{ch7.holdout_capture_5_low:.1%}} to {{ch7.holdout_capture_5_high:.1%}}.
#
# **Next, the other rungs.**

# %% [markdown]
# # The Ladder
#
# The ladder asked where the predictive power comes from, one rung at a time.
# On the walk-forward it had {{ch7.n_oof:,}} rows and {{ch7.walk_failures}} failures to answer with.
#
# The holdout has {{ch7.holdout_failures}} failures.
# That is enough to order models that are far apart, and too few to settle a step of a hundredth.
#
# **Predict 2.** On the walk-forward, adding history to the trees was a step down. Will the holdout agree?
#
# Q. What does each rung score on the holdout, and what does each add to the one below?

# %%
steps = paired_table(verdict, list(verdict)).set_index("model")
walk_board = pd.Series(
    {name: roc_auc_score(oof["target"], oof["pct"]) for name, oof in ladder.items()}
)
both = pd.DataFrame({"walk-forward": walk_board, "holdout": board["roc_auc"]})
both["holdout minus walk-forward"] = both["holdout"] - both["walk-forward"]

KINDS = ("classic", "logistic", "logistic", "trees", "trees", "networks", "blend")
fig, (level, change) = plt.subplots(1, 2, figsize=(viz.EXHIBIT_WIDTH_IN, 4.6), sharey=True)
place = np.arange(len(verdict))
for y_at, (name, row), kind in zip(place, board.iterrows(), KINDS):
    level.plot(
        [row["roc_auc_lo"], row["roc_auc_hi"]], [y_at, y_at], color=COLORS[kind], linewidth=1.6
    )
    level.scatter(row["roc_auc"], y_at, s=46, color=COLORS[kind], zorder=3)
    level.scatter(walk_board[name], y_at, s=40, facecolor="none", edgecolor=viz.SLATE, zorder=2)
    level.annotate(f"{row['roc_auc']:.3f}", (row["roc_auc"], y_at), xytext=(0, 8),
                   textcoords="offset points", ha="center", size=9)  # fmt: skip
for y_at, (name, row), kind in zip(place[1:], steps.iterrows(), KINDS[1:]):
    change.plot([row["roc_auc_diff_lo"], row["roc_auc_diff_hi"]], [y_at, y_at], color=COLORS[kind],
                linewidth=1.6)  # fmt: skip
    change.scatter(row["roc_auc_diff"], y_at, s=46, color=COLORS[kind], zorder=3)
    change.annotate(f"{row['roc_auc_diff']:+.3f}", (row["roc_auc_diff"], y_at), xytext=(0, 8),
                    textcoords="offset points", ha="center", size=9)  # fmt: skip
change.axvline(0, color=viz.INK, linewidth=0.8)
level.set_yticks(place, [f"{k + 1}  {name}" for k, name in enumerate(verdict)])
level.set_ylim(-0.6, len(verdict) - 0.4)
level.set_title("ROC-AUC on the holdout")
level.xaxis.set_major_locator(MultipleLocator(0.05))
change.set_title("What the rung adds to the one below")
for ax in (level, change):
    ax.grid(axis="y", visible=False)

settled = steps["roc_auc_diff_lo"].gt(0) | steps["roc_auc_diff_hi"].lt(0)
store.put("walk_failures", int(chosen_oof["target"].sum()))
store.put("n_settled", int(settled.sum()))
store.put("n_steps", len(steps))
store.put("best_rung", board["roc_auc"].idxmax())
store.put("best_rung_auc", board["roc_auc"].max())
store.put("history_step", steps.loc["Plus history", "roc_auc_diff"])
store.put("history_step_low", steps.loc["Plus history", "roc_auc_diff_lo"])
store.put("history_step_high", steps.loc["Plus history", "roc_auc_diff_hi"])
store.put("shift_low", both["holdout minus walk-forward"].min())
store.put("shift_high", both["holdout minus walk-forward"].max())
# The readings say: every rung is higher than on the walk-forward, the broad order holds, history's
# step holds zero, the chosen model is not the highest rung, and two steps are clear of zero.
assert both["holdout minus walk-forward"].min() > 0
assert board.loc[CLASSIC, "roc_auc"] < board.loc["All ratios, logistic", "roc_auc"] < top["roc_auc"]
assert steps.loc["Plus history", "roc_auc_diff_lo"] < 0 < steps.loc["Plus history", "roc_auc_diff"]
assert board["roc_auc"].idxmax() != CHOSEN
assert settled.sum() == 2
REFIT = "Five ratios, refitted"
assert steps.loc[REFIT, "roc_auc_diff_lo"] < 0 < steps.loc[REFIT, "roc_auc_diff"]
store.put("refit_step", steps.loc[REFIT, "roc_auc_diff"])

viz.exhibit(
    fig,
    f"On the holdout the chosen model scores {top['roc_auc']:.3f}, {primary['roc_auc_diff']:.3f} above the "
    "published score",
    "Holdout scores with 95% cluster-bootstrap intervals. Open circles: the walk-forward score. "
    "Right: paired differences.",
    f"{SOURCE}; observation years {HOLDOUT[0]} to {HOLDOUT[1]}",
    save_as="ladder",
)
both.round(3)

# %% [markdown]
# **Reading.** Exhibit 4 is the central picture of the casebook.
# The broad order of the walk-forward holds: the published score at the foot, the logistic regression on every ratio above it, the trees above that.
#
# Every rung scores higher here than on the walk-forward, the open circles, by {{ch7.shift_low:.3f}} to {{ch7.shift_high:.3f}}.
# The two samples differ in more than the models, with no recession in the holdout and more years to train on, so levels are not compared across them.
#
# The fine order does not hold.
# History is a step up here, {{ch7.history_step:.3f}}, with an interval from {{ch7.history_step_low:.3f}} to {{ch7.history_step_high:.3f}} that holds zero, so Predict 2 has no firm answer.
#
# That fits what Chapter 5 saw fold by fold: history lost in the folds that trained on the file's first years and gained in the later ones, and the final model trains on all of them.
# The refit of Altman's five ratios, a step down in Chapter 4, is a step of {{ch7.refit_step:.3f}} here, with an interval that holds zero as well.
#
# The highest rung on the holdout is "{{ch7.best_rung}}", at {{ch7.best_rung_auc:.3f}}, and it is not the model chosen.
# Only {{ch7.n_settled}} of the {{ch7.n_steps}} steps have an interval clear of zero.
#
# **Check.** The rung with history scores highest on the holdout. Why not switch to it now?
#
# <details>
# <summary>Show the answer</summary>
#
# The holdout is a test only while nothing is chosen on it.
# Pick its winner and it becomes one more validation window, with nothing left to test the pick.
#
# The margin is also inside its interval, so the winner here may be chance.
#
# </details>
#
# **Next, what a desk would have found.**

# %% [markdown]
# # Capture Curves
#
# A desk does not review a ROC-AUC.
# It reviews a share of its names each year, and what matters is how many of the failures were on that list.
#
# Q. How many failures does each review budget find, and how sure is that?


# %%
def capture_band(oof: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """The 95% band of the capture curve on `grid`, resampling whole companies."""
    y, pct = oof["target"].to_numpy(), oof["pct"].to_numpy()
    curves = []
    for rows in company_resamples(y, oof["company_name"], N_BOOT, SEED):
        reviewed, captured = capture_curve(y[rows], pct[rows])
        curves.append(np.interp(grid, reviewed, captured))
    return np.percentile(curves, [2.5, 97.5], axis=0)


N_BOOT = 1000
grid = np.linspace(0, 0.3, 61)
fig, ax = plt.subplots(figsize=(viz.EXHIBIT_WIDTH_IN, 3.8))
for name, kind in ((CLASSIC, "classic"), (CHOSEN, "trees")):
    reviewed, captured = capture_curve(verdict[name]["target"], verdict[name]["pct"])
    low, high = capture_band(verdict[name], grid)
    ax.fill_between(grid, low, high, color=COLORS[kind], alpha=0.18, linewidth=0)
    ax.plot(grid, np.interp(grid, reviewed, captured), color=COLORS[kind], linewidth=1.8)
    ax.annotate(name, (grid[-1], np.interp(grid[-1], reviewed, captured)), xytext=(-4, 8),
                textcoords="offset points", color=COLORS[kind], size=9, ha="right")  # fmt: skip
ax.plot(grid, grid, color=viz.INK, linewidth=0.8, linestyle="--")
ax.annotate("names picked at random", (grid[-1], grid[-1]), xytext=(-4, -18),
            textcoords="offset points", color=viz.SLATE, size=9, ha="right")  # fmt: skip
ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.set_xlabel("Share of each year's firm-years reviewed")
ax.set_ylabel("Share of failures found")

budgets = pd.DataFrame(
    {
        f"{name}, top {share}%": board.loc[
            name,
            [f"capture_{share}", f"capture_{share}_lo", f"capture_{share}_hi", f"lift_{share}"],
        ].to_numpy()
        for share in (1, 5, 10)
        for name in (CLASSIC, CHOSEN)
    },
    index=["failures found", "low", "high", "lift"],
).T
for share in (1, 10):
    store.put(f"holdout_capture_{share}", top[f"capture_{share}"])
store.put("classic_capture_5", board.loc[CLASSIC, "capture_5"])
# The desk's list is every firm-year above the 95th percentile of its year, the list the case files use.
on_list = verdict[CHOSEN]["pct"] > 0.95
found = int((on_list & (verdict[CHOSEN]["target"] == 1)).sum())
# The list finds the same failures as the capture rate at the middle budget. Checked here.
assert found == round(top["capture_5"] * sealed["target"].sum())
store.put("found_5", found)
store.put("reviewed_5", int(on_list.sum()))

viz.exhibit(
    fig,
    f"The top 5% of each year holds {top['capture_5']:.0%} of the failures, and "
    f"{board.loc[CLASSIC, 'capture_5']:.0%} by the published score",
    "Capture curves on the holdout with 95% cluster-bootstrap bands.",
    f"{SOURCE}; observation years {HOLDOUT[0]} to {HOLDOUT[1]}",
)
budgets.round(3)

# %% [markdown]
# **Reading.** In Exhibit 5 the chosen model's curve climbs at once.
# The top hundredth of each year holds {{ch7.holdout_capture_1:.1%}} of the failures, the top twentieth {{ch7.holdout_capture_5:.1%}}, and the top tenth {{ch7.holdout_capture_10:.1%}}.
#
# A desk that read the top twentieth of each year, {{ch7.reviewed_5}} of the {{ch7.n_holdout:,}} firm-years, would have had {{ch7.found_5}} of the {{ch7.holdout_failures}} failures on its list.
#
# The published score finds {{ch7.classic_capture_5:.1%}} at the same budget, which is no better than names picked at random.
# Its ROC-AUC comes from ordering the broad middle, and it fails where a desk looks, at the top.
#
# **Next, the probabilities.**

# %% [markdown]
# # Did the Probabilities Hold
#
# The map and the grades were fixed before the holdout was opened.
# Each holdout firm-year now gets its PD from its percentile, and its grade from the same bands.
#
# Q. Did each grade fail at the rate attached to it?

# %%
pd_holdout = to_pd(verdict[CHOSEN]["pct"])
grades_holdout = grade_table(verdict[CHOSEN], pd_holdout)
inside_grade = (grades_holdout["lo"] <= GRADE_PD) & (GRADE_PD <= grades_holdout["hi"])
years_holdout = (
    verdict[CHOSEN]
    .assign(pd=pd_holdout)
    .groupby("year")
    .agg(predicted=("pd", "mean"), rate=("target", "mean"))
)
brier_holdout = np.mean((pd_holdout - sealed["target"]) ** 2)
brier_flat = np.mean((sealed["target"].mean() - sealed["target"]) ** 2)

store.put("mean_pd_holdout", pd_holdout.mean())
store.put("grades_inside", int(inside_grade.sum()))
store.put("grade_1_holdout", grades_holdout.loc[1, "rate"])
store.put("grade_7_holdout", grades_holdout.loc[7, "rate"])
store.put("grade_7_pd_fixed", GRADE_PD[7])
store.put("brier_holdout", brier_holdout)
store.put("brier_holdout_flat", brier_flat)
# The readings say: the rate rises at every grade, grade seven fails above its PD, the mean PD is
# above the failure rate in each year, and the map beats one probability for all.
assert (grades_holdout["rate"].diff().iloc[1:] > 0).all()
assert grades_holdout.loc[7, "lo"] > GRADE_PD[7]
assert (years_holdout["predicted"] > years_holdout["rate"]).all()
assert brier_holdout < brier_flat

print(years_holdout.round(4))
viz.exhibit(
    grade_ladder(grades_holdout, GRADE_PD),
    f"The grades keep their order, and grade seven fails at {grades_holdout.loc[7, 'rate']:.0%} "
    f"against a PD of {GRADE_PD[7]:.0%}",
    "Holdout firm-years by grade, with 95% Wilson intervals. A bar marks the PD fixed beforehand.",
    f"{SOURCE}; observation years {HOLDOUT[0]} to {HOLDOUT[1]}",
)
grades_holdout.round(4)

# %% [markdown]
# **Reading.** The grades held their order: in Exhibit 6 the share that failed rises at every step, from {{ch7.grade_1_holdout:.1%}} to {{ch7.grade_7_holdout:.1%}}.
# The levels held less well, and the PD fixed beforehand sits inside the interval for {{ch7.grades_inside}} of the seven grades.
#
# Grade seven failed at {{ch7.grade_7_holdout:.1%}} against a PD of {{ch7.grade_7_pd_fixed:.1%}}, and the middle grades failed less often than their PDs said.
# The model separated the riskiest firm-years more sharply here than on the walk-forward, which is the higher ROC-AUC seen from another side.
#
# Across all firm-years the mean PD is {{ch7.mean_pd_holdout:.1%}} and {{ch7.holdout_rate:.1%}} failed, and in each of the three years the PD was the higher of the two.
# The Brier score is {{ch7.brier_holdout:.5f}}, against {{ch7.brier_holdout_flat:.5f}} for one probability given to every row.
#
# **Next, the same result cut several ways.**

# %% [markdown]
# # Robustness
#
# One number on one sample is a thin result.
# The same score is now read on parts of the holdout, and against two other labels.
#
# The cuts were listed before the holdout was opened: companies the model trained on and companies it never saw, each year alone, the sample without its **unconfirmed** failures, and failure within one year and within three.
# An unconfirmed failure is one whose last filing is in the file's last year, so no later row confirms that it stopped.
#
# Q. Does the result hold in each cut?

# %%
hold = verdict[CHOSEN]
seen = sealed["company_name"].isin(dev["company_name"]).to_numpy()
unconfirmed = ((sealed["target"] == 1) & (sealed["last_year"] == PANEL_END)).to_numpy()
cuts = {"All holdout firm-years": (np.ones(len(sealed), dtype=bool), sealed["target"])}
cuts["Companies seen in development"] = (seen, sealed["target"])
cuts["Companies never seen"] = (~seen, sealed["target"])
for year in range(HOLDOUT[0], HOLDOUT[1] + 1):
    cuts[f"Observation year {year}"] = ((sealed["year"] == year).to_numpy(), sealed["target"])
cuts["Without unconfirmed failures"] = (~unconfirmed, sealed["target"])
cuts["Failure within one year"] = (np.ones(len(sealed), dtype=bool), sealed["target_1"])
cuts["Failure within three years"] = (sealed["target_3"].notna().to_numpy(), sealed["target_3"])

rows = []
for name, (keep, outcome) in cuts.items():
    y, pct = outcome[keep].to_numpy().astype(int), hold.loc[keep, "pct"].to_numpy()
    low, high, _ = cluster_bootstrap_ci(
        y, pct, sealed.loc[keep, "company_name"].to_numpy(), roc_auc_score
    )
    rows.append({"cut": name, "rows": int(keep.sum()), "failures": int(y.sum()),
                 "roc_auc": roc_auc_score(y, pct), "low": low, "high": high})  # fmt: skip
robust = pd.DataFrame(rows).set_index("cut")

fig, ax = plt.subplots(figsize=(viz.EXHIBIT_WIDTH_IN, 4.2))
place = np.arange(len(robust))[::-1]
ax.hlines(place, robust["low"], robust["high"], color=COLORS["trees"], linewidth=1.6)
ax.scatter(robust["roc_auc"], place, s=46, color=COLORS["trees"], zorder=3)
ax.axvline(robust["roc_auc"].iloc[0], color=viz.INK, linewidth=0.8, linestyle="--")
ax.set_yticks(place, [f"{name} ({failures})" for name, failures in robust["failures"].items()])
ax.set_xlabel("ROC-AUC of the chosen model")
ax.grid(axis="y", visible=False)

# The reading says every company never seen first filed after the development years. Checked here.
first_filed = features.groupby("company_name")["year"].min()
assert (first_filed[sealed.loc[~seen, "company_name"]] > DEV_END).all()
store.put("seen_share", seen.mean())
store.put("unseen_auc", robust.loc["Companies never seen", "roc_auc"])
store.put("unseen_low", robust.loc["Companies never seen", "low"])
store.put("unseen_high", robust.loc["Companies never seen", "high"])
store.put("seen_auc", robust.loc["Companies seen in development", "roc_auc"])
store.put("unseen_failures", int(robust.loc["Companies never seen", "failures"]))
store.put("n_unconfirmed", int(unconfirmed.sum()))
store.put("confirmed_auc", robust.loc["Without unconfirmed failures", "roc_auc"])
store.put("one_year_auc", robust.loc["Failure within one year", "roc_auc"])
store.put("three_year_auc", robust.loc["Failure within three years", "roc_auc"])
store.put("year_auc_low", robust.filter(like="Observation year", axis=0)["roc_auc"].min())
store.put("year_auc_high", robust.filter(like="Observation year", axis=0)["roc_auc"].max())
store.put(
    "cuts_holding",
    int(((robust["low"] <= top["roc_auc"]) & (top["roc_auc"] <= robust["high"])).sum()),
)
store.put("n_cuts", len(robust))
# The readings say: every cut's interval holds the whole-holdout score, never-seen companies score
# lower than seen ones, and the score does not fall without the unconfirmed failures.
assert ((robust["low"] <= top["roc_auc"]) & (top["roc_auc"] <= robust["high"])).all()
assert (
    robust.loc["Companies never seen", "roc_auc"]
    < robust.loc["Companies seen in development", "roc_auc"]
)
assert robust.loc["Without unconfirmed failures", "roc_auc"] > top["roc_auc"]

viz.exhibit(
    fig,
    f"Companies never seen score {robust.loc['Companies never seen', 'roc_auc']:.3f}, and companies "
    f"seen before {robust.loc['Companies seen in development', 'roc_auc']:.3f}",
    "ROC-AUC with 95% cluster-bootstrap intervals. The number after each cut is its failures. "
    "Dashed: the whole holdout.",
    f"{SOURCE}; observation years {HOLDOUT[0]} to {HOLDOUT[1]}",
)
robust.round(3)

# %% [markdown]
# **Reading.** In Exhibit 7 every cut's interval holds the score of the whole holdout, the dashed line.
#
# The cut to watch is the companies the model never saw: {{ch7.unseen_auc:.3f}}, with an interval from {{ch7.unseen_low:.3f}} to {{ch7.unseen_high:.3f}}, against {{ch7.seen_auc:.3f}} for companies from the development years.
# Only {{ch7.unseen_failures}} failures stand behind it, so the interval is wide, but a new name is the case a desk cares about most.
#
# {{ch7.seen_share:.0%}} of holdout firm-years belong to companies from the development years, so the headline leans on familiar names.
# Every company never seen first filed after {{ch7.dev_last}}, so the gap may come from newer firms as much as from unfamiliar ones.
#
# Without the {{ch7.n_unconfirmed}} unconfirmed failures the score is {{ch7.confirmed_auc:.3f}}, so they are not what holds the result up.
# The model ranks failure within one year best, at {{ch7.one_year_auc:.3f}}, and failure within three years at {{ch7.three_year_auc:.3f}}.
#
# **Next, why the model ranks as it does.**

# %% [markdown]
# # What Drives the Score
#
# A score that cannot be explained is hard to act on.
# A **SHAP value** splits one firm-year's score among its inputs: how far each ratio pushed the score up or down from the average.
#
# Ratios in one family carry much the same information, so the first reading adds SHAP values up by family.
# A second method checks it: shuffle a whole family's columns among the firm-years and see how far the ROC-AUC falls.
#
# Q. Which families of ratios drive the score?

# %%
tree = fitted[CHOSEN]
inputs_holdout = tree.prepare(sealed)
explained = shap.TreeExplainer(tree.model)(inputs_holdout)
credit = pd.DataFrame(explained.values, columns=RATIOS)
FAMILY_OF = {name: family for family, members in FAMILIES.items() for name in members}

by_family = credit.abs().mean().groupby(FAMILY_OF).sum()
shap_share = (by_family / by_family.sum()).sort_values()

rng = np.random.default_rng(SEED)
REPEATS = 20
base_auc = roc_auc_score(sealed["target"], hold["pct"])
drops = {}
for family, members in FAMILIES.items():
    falls = []
    for _ in range(REPEATS):
        shuffled = inputs_holdout.copy()
        shuffled[members] = shuffled[members].to_numpy()[rng.permutation(len(shuffled))]
        pct = within_year_percentile(
            pd.Series(tree.model.predict_proba(shuffled)[:, 1]), sealed["year"]
        )
        falls.append(base_auc - roc_auc_score(sealed["target"], pct))
    drops[family] = np.mean(falls)
drops = pd.Series(drops)

fig, (left, right) = plt.subplots(1, 2, figsize=(viz.EXHIBIT_WIDTH_IN, 3.6), sharey=True)
left.barh(shap_share.index, shap_share, color=COLORS["trees"])
right.barh(shap_share.index, drops[shap_share.index], color=COLORS["trees"])
left.set_title("Share of the SHAP credit")
right.set_title("ROC-AUC lost when the family is shuffled")
left.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
for ax in (left, right):
    ax.grid(axis="y", visible=False)

store.put("top_family", shap_share.index[-1])
store.put("top_family_share", shap_share.iloc[-1])
store.put("second_family", shap_share.index[-2])
store.put("second_family_share", shap_share.iloc[-2])
store.put("top_family_drop", drops[shap_share.index[-1]])
store.put("top_by_shuffle", drops.idxmax())
# The readings say: leverage leads by SHAP, the market family by shuffling, and coverage and
# activity are last by both.
assert shap_share.index[-1] == "leverage" and drops.idxmax() == "market"
assert (
    set(shap_share.index[:2])
    == set(drops.abs().sort_values().index[:2])
    == {"coverage", "activity"}
)

viz.exhibit(
    fig,
    f"The {shap_share.index[-1]} ratios carry {shap_share.iloc[-1]:.0%} of the credit, and the "
    f"{shap_share.index[-2]} ratios {shap_share.iloc[-2]:.0%}",
    f"Left: mean absolute SHAP value, summed by family. Right: mean of {REPEATS} shuffles.",
    f"{SOURCE}; observation years {HOLDOUT[0]} to {HOLDOUT[1]}",
)

# %% [markdown]
# **Reading.** Exhibit 8 gives two answers that agree on the leaders and not on their order.
# By SHAP credit the {{ch7.top_family}} ratios lead with {{ch7.top_family_share:.0%}}, then {{ch7.second_family}} with {{ch7.second_family_share:.0%}}.
#
# By shuffling, the {{ch7.top_by_shuffle}} family costs the most.
# SHAP shares credit among ratios that say similar things, and shuffling asks what is lost when a family goes missing, so the market's view looks like the information least repeated elsewhere.
#
# Coverage and activity matter least by either method.
#
# Q. Which single ratios matter most, and in which direction?

# %%
TOP = credit.abs().mean().sort_values(ascending=False).index[:8]
rank_of = inputs_holdout[RATIOS].rank(pct=True)
ramp = LinearSegmentedColormap.from_list("ramp", ["#F1E4C3", COLORS["trees"], "#3B2A05"])
jitter = np.random.default_rng(SEED).normal(0, 0.11, len(credit))

fig, ax = plt.subplots(figsize=(viz.EXHIBIT_WIDTH_IN, 4.6))
for y_at, name in enumerate(TOP[::-1]):
    ax.scatter(credit[name], y_at + jitter, c=rank_of[name], cmap=ramp, s=5, linewidths=0,
               rasterized=True)  # fmt: skip
ax.axvline(0, color=viz.INK, linewidth=0.8)
ax.set_yticks(range(len(TOP)), TOP[::-1])
ax.set_xlabel("SHAP value: push on the score, in log odds")
ax.grid(axis="y", visible=False)

direction = pd.Series({name: credit[name].corr(rank_of[name], method="spearman") for name in TOP})
store.put("top_ratio", TOP[0])
store.put("second_ratio", TOP[1])
store.put("third_ratio", TOP[2])
store.put("top_ratio_share", credit[TOP[0]].abs().mean() / credit.abs().mean().sum())
# The reading says each leading ratio pushes the way a credit analyst expects. Checked here.
RAISES_RISK_WHEN_HIGH = {"debt_to_ebitda", "liabilities_to_assets"}
assert all((direction[name] > 0) == (name in RAISES_RISK_WHEN_HIGH) for name in TOP)
assert TOP[0] == "mve_to_liabilities"

viz.exhibit(
    fig,
    f"{TOP[0]} moves the score most, then {TOP[1]} and {TOP[2]}",
    "One dot for each holdout firm-year and ratio. Dark is a high value of the ratio, pale a low one.",
    f"{SOURCE}; observation years {HOLDOUT[0]} to {HOLDOUT[1]}",
)
direction.round(2).to_frame("rank correlation of the ratio with its SHAP value")

# %% [markdown]
# **Reading.** In Exhibit 9 a ratio whose dark dots sit on the right pushes the score up when it is high.
# `{{ch7.top_ratio}}` leads, with {{ch7.top_ratio_share:.0%}} of all the credit: a small market value beside the liabilities pushes the score up, and a large one pushes it down.
#
# Debt over EBITDA and liabilities over assets raise the score when they are high, and the other six lower it.
# Each direction is the one a credit analyst would expect.
#
# Q. How does the push change along each of the three leading ratios?

# %%
fig, axes = plt.subplots(1, 3, figsize=(viz.EXHIBIT_WIDTH_IN, 3.4), sharey=True)
partners = {}
for ax, name in zip(axes, TOP[:3]):
    partner = RATIOS[shap.utils.potential_interactions(explained[:, name], explained)[0]]
    partners[name] = partner
    ax.scatter(rank_of[name], credit[name], c=rank_of[partner], cmap=ramp, s=5, linewidths=0,
               rasterized=True)  # fmt: skip
    ax.axhline(0, color=viz.INK, linewidth=0.8)
    ax.set_title(name)
    ax.set_xlabel(f"Percentile of the ratio\ncolour: {partner}")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
axes[0].set_ylabel("SHAP value, in log odds")

middle = credit.loc[rank_of[TOP[0]].between(0.45, 0.55), TOP[0]].median()
lower_fall = credit.loc[rank_of[TOP[0]] < 0.1, TOP[0]].median() - middle
upper_fall = middle - credit.loc[rank_of[TOP[0]] > 0.9, TOP[0]].median()
# The reading says the push falls steeply through the lower half and is nearly flat above it.
assert upper_fall < 0.25 * lower_fall
viz.exhibit(
    fig,
    f"The push from {TOP[0]} falls {lower_fall:.2f} over its lower half and {upper_fall:.2f} over "
    "its upper",
    "Holdout firm-years. Colour is the percentile of the ratio that most changes the picture, "
    "dark for high.",
    f"{SOURCE}; observation years {HOLDOUT[0]} to {HOLDOUT[1]}",
)

# %% [markdown]
# **Reading.** Exhibit 10 shows shapes a straight line could not draw.
# The push from `{{ch7.top_ratio}}` falls steeply through the lower half of its range and then goes flat: past the middle, more market value buys little more safety.
#
# The other two are closer to steps, with debt over EBITDA adding risk only at its high end and the quick ratio only at its low end.
# Thresholds of this kind are what the trees added to the logistic regression in Chapter 5.
#
# **Next, three companies.**

# %% [markdown]
# # Three Case Files
#
# Averages hide companies, so three are read closely.
# They are chosen by a rule, so that no one can pick the flattering ones.
#
# Among the failures the model put in its top {{ch7.budget:.0%}}, the one at the median score is the **caught** case.
# Among the failures it did not, the one at the median score is the **missed** case, and among the flagged firm-years that did not fail, the one at the median score is the **false alarm**.
#
# Q. Who are they, and what did the model see?

# %%
BUDGET = 0.05
flagged = (hold["pct"] > 1 - BUDGET).to_numpy()
failed_soon = (sealed["target"] == 1).to_numpy()


def median_case(keep: np.ndarray) -> int:
    """The row at the median score among the rows kept."""
    ordered = hold.loc[keep].sort_values(["pct", "company_name"])
    return int(ordered.index[len(ordered) // 2])


CASES = {
    "Caught": median_case(failed_soon & flagged),
    "Missed": median_case(failed_soon & ~flagged),
    "False alarm": median_case(~failed_soon & flagged),
}
# Every firm-year of the panel scored by the final model. The development years are years it trained on.
everything = features[[*KEYS]].assign(score=tree(features))
everything = everything.assign(pct=within_year_percentile(everything["score"], everything["year"]))

fig, axes = plt.subplots(len(CASES), 2, figsize=(viz.EXHIBIT_WIDTH_IN, 7.4), width_ratios=(1, 1.25))
files = []
for (path, bars), (label, row) in zip(axes, CASES.items()):
    company, year = sealed.loc[row, "company_name"], int(sealed.loc[row, "year"])
    history = everything.loc[everything["company_name"] == company].sort_values("year")
    path.plot(history["year"], history["pct"], color=COLORS["trees"], linewidth=1.8)
    path.scatter(year, hold.loc[row, "pct"], s=46, color=viz.INK, zorder=3)
    path.axhline(1 - BUDGET, color=viz.SLATE, linewidth=0.8, linestyle="--")
    path.axvspan(HOLDOUT[0] - 0.5, HOLDOUT[1] + 0.5, color=viz.RULE, alpha=0.35, linewidth=0)
    path.set_xlim(int(features["year"].min()) - 0.5, HOLDOUT[1] + 0.5)
    path.set_ylim(0, 1.02)
    path.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    path.set_title(f"{label}: {company} in {year}")
    path.set_ylabel("Percentile in the year")
    pushes = credit.loc[row].sort_values(key=np.abs, ascending=False).iloc[:6][::-1]
    values = inputs_holdout.loc[row, pushes.index]
    names = [
        f"{name} = {value:,.2f}" if pd.notna(value) else f"{name}: blank"
        for name, value in values.items()
    ]
    bars.barh(names, pushes, color=COLORS["trees"])
    bars.axvline(0, color=viz.INK, linewidth=0.8)
    bars.set_title("Largest pushes, in log odds")
    bars.grid(axis="y", visible=False)
    files.append({"case": label, "company": company, "year": year, "percentile": hold.loc[row, "pct"],
                  "grade": int(grade_of(hold.loc[[row], "pct"])[0]), "PD": pd_holdout[row],
                  "last filing": int(sealed.loc[row, "last_year"]), "status": sealed.loc[row, "status_label"],
                  "main push": pushes.index[-1]})  # fmt: skip
files = pd.DataFrame(files).set_index("case")

store.put("budget", BUDGET)
store.put("n_caught", int((failed_soon & flagged).sum()))
store.put("n_missed", int((failed_soon & ~flagged).sum()))
store.put("n_alarms", int((~failed_soon & flagged).sum()))
for label, row in files.iterrows():
    key = label.lower().replace(" ", "_")
    store.put(f"{key}_pct", row["percentile"])
    store.put(f"{key}_push", row["main push"])
    store.put(f"{key}_year", int(row["year"]))
    store.put(f"{key}_last", int(row["last filing"]))
store.put("alarm_status", files.loc["False alarm", "status"])
# The readings say: the caught case climbed for three years, the missed case is an unconfirmed
# failure, and the false alarm left the file a year later without being marked as failed.
caught_path = everything.loc[
    everything["company_name"] == files.loc["Caught", "company"]
].set_index("year")["pct"]
assert caught_path.loc[
    files.loc["Caught", "year"] - 3 : files.loc["Caught", "year"]
].is_monotonic_increasing
assert files.loc["Missed", "last filing"] == PANEL_END
assert files.loc["False alarm", "status"] == "alive"
assert files.loc["False alarm", "last filing"] == files.loc["False alarm", "year"] + 1

viz.exhibit(
    fig,
    f"The caught case ranked above {files.loc['Caught', 'percentile']:.0%} of its year, and the "
    f"missed one above {files.loc['Missed', 'percentile']:.0%}",
    "Left: percentile through time, a dot at the case year, the holdout shaded. Right: SHAP values.",
    f"{SOURCE}; chosen by rule from observation years {HOLDOUT[0]} to {HOLDOUT[1]}",
)
files.round(3)

# %% [markdown]
# **Reading.** Of the {{ch7.reviewed_5}} firm-years on the desk's list, {{ch7.n_caught}} failed within two years and {{ch7.n_alarms}} did not, and the list missed {{ch7.n_missed}} failures.
#
# The caught case ranked above {{ch7.caught_pct:.0%}} of its year in {{ch7.caught_year}} and filed for the last time in {{ch7.caught_last}}.
# In Exhibit 11 its rank climbs for three years before that, and its largest push is a market value that is a small fraction of its liabilities.
#
# The missed case ranked above {{ch7.missed_pct:.0%}}, high and not on the list.
# Its losses pushed the score up, and its small size and a market value above its liabilities pushed it down; it is also one of the unconfirmed failures.
#
# The false alarm looks like the caught case on paper.
# It filed for the last time in {{ch7.false_alarm_last}}, a year after the flag, and the file marks it "{{ch7.alarm_status}}", which Chapter 2 warned is not the same as healthy.
#
# **Next, how early the model sees a failure coming.**

# %% [markdown]
# # How Early Does It See
#
# A warning one year before the final filing helps less than one three years before.
# The holdout holds firm-years of failed companies from one to four years ahead of their final filing.
#
# Q. How high does the model rank a failing company as the final filing nears?

# %%
doomed = sealed["failed"].astype(bool).to_numpy() & (sealed["years_to_final"] >= 1).to_numpy()
ahead = (
    hold.loc[doomed]
    .assign(years=sealed.loc[doomed, "years_to_final"].astype(int), flag=flagged[doomed])
    .groupby("years")
    .agg(
        rows=("pct", "size"),
        median=("pct", "median"),
        low=("pct", lambda s: s.quantile(0.25)),
        high=("pct", lambda s: s.quantile(0.75)),
        flagged=("flag", "sum"),
    )
)
flag_bounds = np.array(
    [wilson_interval(int(k), int(n)) for k, n in zip(ahead["flagged"], ahead["rows"])]
)
ahead["share flagged"] = ahead["flagged"] / ahead["rows"]

fig, (rank, share) = plt.subplots(1, 2, figsize=(viz.EXHIBIT_WIDTH_IN, 3.4))
rank.vlines(ahead.index, ahead["low"], ahead["high"], color=COLORS["trees"], linewidth=1.6)
rank.scatter(ahead.index, ahead["median"], s=46, color=COLORS["trees"], zorder=3)
rank.axhline(0.5, color=viz.INK, linewidth=0.8, linestyle="--")
rank.set_title("Median percentile, with the middle half")
rank.set_ylim(0, 1.02)
share.vlines(
    ahead.index, flag_bounds[:, 0], flag_bounds[:, 1], color=COLORS["trees"], linewidth=1.6
)
share.scatter(ahead.index, ahead["share flagged"], s=46, color=COLORS["trees"], zorder=3)
share.axhline(BUDGET, color=viz.INK, linewidth=0.8, linestyle="--")
share.set_title(f"Share in the top {BUDGET:.0%}")
for ax in (rank, share):
    ax.invert_xaxis()
    ax.set_xticks(ahead.index)
    ax.set_xlabel("Years before the final filing")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="x", visible=False)

for years in (ahead.index.min(), ahead.index.max()):
    store.put(f"median_{years}_out", ahead.loc[years, "median"])
    store.put(f"flagged_{years}_out", ahead.loc[years, "share flagged"])
store.put("farthest", int(ahead.index.max()))
# The readings say the rank rises as the final filing nears, and nothing four years out is flagged.
assert ahead["median"].is_monotonic_decreasing
assert ahead.loc[ahead.index.max(), "flagged"] == 0

viz.exhibit(
    fig,
    f"{ahead.index.max()} years out the median failing company ranks above "
    f"{ahead.loc[ahead.index.max(), 'median']:.0%}, one year out above {ahead.loc[1, 'median']:.0%}",
    "Holdout firm-years of companies that later failed. Dashed: what a company picked at random "
    "would show.",
    f"{SOURCE}; observation years {HOLDOUT[0]} to {HOLDOUT[1]}",
)
ahead.round(3)

# %% [markdown]
# **Reading.** In Exhibit 12 the warning builds over years.
# {{ch7.farthest}} years before the final filing the median failing company already ranks above {{ch7.median_4_out:.0%}} of its year, where a company picked at random would rank above half.
#
# One year out it ranks above {{ch7.median_1_out:.0%}}.
# The list is slower than the rank: none of the firm-years {{ch7.farthest}} years out is in the top twentieth, and one year out {{ch7.flagged_1_out:.0%}} are.
#
# So the model sees trouble early, as a drift up the ranking, and names it late.
#
# **Next, what all of this rests on.**

# %% [markdown]
# # Limits
#
# A result is only as good as the account of what it rests on.
# The cell below measures four of the limits, and writes the results the front page quotes.
#
# Q. What should a reader hold against these numbers?

# %%
events = store.load_frame("event_table")  # Chapter 2's table of how each company leaves the file
quiet = events.loc[events["event"] != "failed", "event"]
bounds_low, bounds_high = (fit_prepare(dev, RATIOS, impute=False)[k] for k in (1, 2))
at_ceiling = (sealed[RATIOS] > bounds_high).mean()

store.put("still_reporting", (quiet == "reporting").mean())
store.put("ceiling_ratio", at_ceiling.idxmax())
store.put("ceiling_share", at_ceiling.max())
store.put("model_age_low", HOLDOUT[0] - DEV_END)
store.put("model_age_high", HOLDOUT[1] - DEV_END)

metrics = {
    "scoreboard": board.reset_index().to_dict("records"),
    "paired": steps.reset_index().to_dict("records"),
    "ladder": both.reset_index(names="model").to_dict("records"),
    "calibration": grades_holdout.assign(pd_fixed=GRADE_PD).reset_index().to_dict("records"),
    "robustness": robust.reset_index().to_dict("records"),
}
claims = {
    "holdout_rows": f"{len(sealed):,}",
    "holdout_failures": f"{int(sealed['target'].sum())}",
    "chosen_model": CHOSEN,
    "holdout_auc": f"{top['roc_auc']:.3f}",
    "holdout_auc_interval": f"{top['roc_auc_lo']:.3f} to {top['roc_auc_hi']:.3f}",
    "walk_forward_auc": f"{walked_board['roc_auc']:.3f}",
    "classic_auc": f"{board.loc[CLASSIC, 'roc_auc']:.3f}",
    "primary_step": f"{primary['roc_auc_diff']:+.3f}",
    "primary_step_interval": f"{primary['roc_auc_diff_lo']:+.3f} to {primary['roc_auc_diff_hi']:+.3f}",
    "capture_5": f"{top['capture_5']:.1%}",
    "capture_5_interval": f"{top['capture_5_lo']:.1%} to {top['capture_5_hi']:.1%}",
    "lift_5": f"{top['lift_5']:.1f}",
    "unseen_auc": f"{robust.loc['Companies never seen', 'roc_auc']:.3f}",
}
store.write_results(metrics, claims)

pd.Series(claims).to_frame("what the front page quotes")

# %% [markdown]
# **Reading.** The results are written, and these are the limits to read them with.
#
# The label:
#
# - It is derived, not given: the file marks a company as failed, and the year of failure is taken to follow its last filing, on the paper's own rule.
# - Bankruptcy is not default: a missed coupon or a distressed exchange is not in the file.
# - A company that stops filing without failing is counted as a survivor, and only {{ch7.still_reporting:.0%}} of the companies never marked as failed are still filing in the last year.
# - {{ch7.n_unconfirmed}} of the holdout's {{ch7.holdout_failures}} failures cannot be confirmed by a later row.
#
# The sample:
#
# - These are small listed companies, not a universe of bond issuers.
# - Early failures are thin: the first years of the file hold few, and the dot-com bust is never a validation year.
# - {{ch7.seen_share:.0%}} of holdout firm-years belong to companies the model trained on.
# - Market value is among the inputs, so this is a market and accounting model, not an accounting one.
# - Clipping bounds come from the development years, and {{ch7.ceiling_share:.1%}} of the holdout values of `{{ch7.ceiling_ratio}}` sit above the upper bound.
#
# The evidence:
#
# - The holdout years have been scored more than once in the making of this study, so the walk-forward is the primary evidence and the holdout a check on it.
# - The holdout holds no recession, and the evidence under stress comes from crisis folds that also took part in selection.
# - Those crisis folds have the smallest training sets, so stress and sample size cannot be told apart.
# - The panel holds one crisis.
# - With {{ch7.holdout_failures}} failures the holdout supports a coarse ordering of models, not gains from rung to rung.
# - The model is {{ch7.model_age_low}} to {{ch7.model_age_high}} years old when the holdout scores it.
# - The PD is an average over the cycle and will be wrong in most single years.
#
# **That is the verdict, and the casebook ends here.**

# %% [markdown]
# # Findings
#
# 1. On the holdout the chosen model has a ROC-AUC of {{ch7.holdout_auc:.3f}}, with an interval from {{ch7.holdout_auc_low:.3f}} to {{ch7.holdout_auc_high:.3f}}, above its walk-forward score of {{ch7.walked_auc:.3f}} (Exhibit 4).
# 2. The primary comparison is settled: the chosen model is {{ch7.primary_step:.3f}} ahead of the published Altman Z, with an interval above zero (Exhibit 4).
# 3. The top twentieth of each year holds {{ch7.holdout_capture_5:.1%}} of the failures, where the published score finds {{ch7.classic_capture_5:.1%}} (Exhibit 5).
# 4. The broad order of the ladder holds and the fine order does not: {{ch7.n_settled}} of {{ch7.n_steps}} steps are clear of zero, and history is a step of {{ch7.history_step:.3f}} where the walk-forward had a step down (Exhibit 4).
# 5. A PD built from the percentile is an average over the cycle: by year it stays between {{ch7.flat_low:.1%}} and {{ch7.flat_high:.1%}} while the share that failed runs from {{ch7.rate_low:.1%}} to {{ch7.rate_high:.1%}} (Exhibit 2).
# 6. The seven grades keep their order on the holdout, and the PD fixed beforehand is inside the interval for {{ch7.grades_inside}} of them (Exhibit 6).
# 7. On companies it never saw the model scores {{ch7.unseen_auc:.3f}}, with an interval from {{ch7.unseen_low:.3f}} to {{ch7.unseen_high:.3f}}, against {{ch7.seen_auc:.3f}} on companies from the development years (Exhibit 7).
# 8. Leverage, profitability, and the market's view drive the score, and market value over liabilities is the leading ratio (Exhibit 8, Exhibit 9).
# 9. A failing company ranks above {{ch7.median_4_out:.0%}} of its year {{ch7.farthest}} years before its final filing and above {{ch7.median_1_out:.0%}} one year before (Exhibit 12).

# %% [markdown]
# # Your Turn
#
# Three extensions, each with a hint folded under it.
#
# **1. A fresher model.** The model is three to five years old when it is scored. How much does it gain if it is refitted each year, as a desk would refit it?
#
# <details>
# <summary>Hint</summary>
#
# For each holdout year, fit `fit_trees` on the rows of `features` up to three years earlier and score that year alone.
# Join the three years and compare with `verdict[CHOSEN]` through `paired_table`.
#
# </details>
#
# **2. A level for the cycle.** The PD is flat across years. Can last year's failure rate move it in the right direction?
#
# <details>
# <summary>Hint</summary>
#
# The failures known at the start of a year are those with a final filing two or more years back.
# Scale `pd_holdout` by that known rate over `base_rate`, and compare the Brier score with `brier_holdout`.
#
# </details>
#
# **3. Another case.** The three case files sit at the median of their groups. What does the worst miss look like?
#
# <details>
# <summary>Hint</summary>
#
# Sort `hold.loc[failed_soon & ~flagged]` by `pct` and take the first row.
# Draw its path with `everything` and its pushes with `credit`, as the case files do.
#
# </details>
