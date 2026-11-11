"""The proof sheet: every drawn component and four real exhibits, for the identity review.

    uv run python -m tools.casebook.proof

Writes artifacts/proof/, plus artifacts/proof/dark/ with each picture on GitHub's dark page colour.
Exhibits use development years only (observation years up to 2011).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from chapters.diagrams import label_timeline
from corp_default import viz
from corp_default.data import REPO, load_raw
from corp_default.features import FAMILIES, add_ratios
from corp_default.labels import build_label, event_table
from corp_default.metrics import wilson_interval
from corp_default.splits import DEV_END, development
from corp_default.survival import kaplan_meier
from tools.casebook.components import cover, next_card, opener, pitfall, route, step_curve
from tools.casebook.studio import LANDING_WIDTH, Studio

OUT = REPO / "artifacts" / "proof"
DARK_PAGE = "#0D1117"
SOURCE = "Lombardo et al. (2022), Kaggle copy; development years 1999 to 2011"

CHAPTER_ONE_PARTS = [
    ("Setting Up", 2),
    ("Where the Data Comes From", 4),
    ("The Data Dictionary", 5),
    ("The Shape of the Panel", 5),
    ("Late Arrivals and Gaps", 5),
    ("Who These Companies Are", 6),
    ("Do the Numbers Add Up", 5),
    ("Signed Numbers Need a Signed Scale", 5),
    ("What the File Cannot Tell Us", 3),
]
SPEC_PALETTE = {
    "failed": "#D9482B",
    "survived": "#3F5F7F",
    "censored": "#8B909A",
    "classic": "#8B909A",
    "logistic": "#3F5F7F",
    "trees": "#C28F2C",
    "networks": "#7B5EA7",
    "blend": "#16181D",
}


def house_marks(events: pd.DataFrame) -> dict[str, dict]:
    """The two candidate house marks, as step-curve arguments, computed from the event table."""
    failed_only = kaplan_meier(events["duration"], (events["event"] == "failed").astype(int))
    any_exit = kaplan_meier(events["duration"], (events["event"] != "reporting").astype(int))
    a_points = [(0, 1.0), *zip(failed_only["time"], failed_only["survival"])]
    b_points = [(0, 1.0), *zip(any_exit["time"], any_exit["survival"])]
    low = float(np.floor(failed_only["survival"].min() * 20) / 20)
    failure_times = failed_only.loc[failed_only["events"] > 0, "time"].tolist()
    return {
        "A": {
            "points": a_points,
            "y_range": (low, 1.0),
            "labels": ((1.0, "1.00"), (low, f"{low:.2f}")),
        },
        "B": {"points": b_points, "y_range": (0.0, 1.0), "ticks": failure_times},
    }


def mark_svg(mark: dict, chapter: int | None, width: int = 1200, height: int = 160) -> str:
    times = [t for t, _ in mark["points"]]
    marker = None if chapter is None else times[0] + (times[-1] - times[0]) * (chapter - 1) / 6
    return step_curve(
        mark["points"],
        width,
        height,
        (96, 18, width - 64, height - 22),
        marker=marker,
        ticks=mark.get("ticks"),
        y_range=mark["y_range"],
        labels=mark.get("labels", ()),
    )


def hero(mark: dict, counts: dict[str, int]) -> str:
    """A draft of the landing hero at 830 px. Phase 11 builds the real one."""
    facts = "".join(f"<span><b>{value:,}</b> {name}</span>" for name, value in counts.items())
    style = (
        "<style>.cb-hero{height:440px;background:var(--carbon);color:var(--on-carbon);border:1px solid var(--carbon-rule)}"
        ".cb-hero .cb-kicker{position:absolute;top:32px;left:40px;font:500 11px/14px var(--mono);"
        "letter-spacing:.18em;color:var(--on-carbon-mute)}"
        ".cb-hero h1{position:absolute;top:60px;left:40px;font:800 46px/52px var(--display);"
        "color:var(--on-carbon)}"
        ".cb-hero p{position:absolute;top:122px;left:40px;width:700px;font:400 18px/26px var(--sans);"
        "color:var(--on-carbon-soft)}"
        ".cb-hero .cb-mark{position:absolute;top:186px;left:0;right:0;height:170px}"
        ".cb-hero .cb-facts{position:absolute;left:40px;right:40px;bottom:0;height:64px;display:flex;"
        "gap:32px;align-items:center;border-top:1px solid var(--carbon-rule);"
        "font:500 12px/16px var(--mono);letter-spacing:.12em;color:var(--on-carbon-mute)}"
        ".cb-hero .cb-facts b{font:700 16px/20px var(--mono);color:var(--on-carbon)}</style>"
    )
    return (
        f'{style}<div class="cb cb-hero"><div class="cb-kicker">A CASEBOOK IN SEVEN CHAPTERS</div>'
        f"<h1>Corporate Default Prediction</h1>"
        f"<p>A casebook on predicting bankruptcy in US public companies, 1999 to 2018.</p>"
        f'<div class="cb-mark">{mark_svg(mark, None, width=830, height=170)}</div>'
        f'<div class="cb-facts">{facts}</div></div>'
    )


def palette_plate() -> str:
    """The spec palette beside the validated one, for the identity decision."""
    rows = ""
    for label, palette in (
        ("SPEC", SPEC_PALETTE),
        ("VALIDATED", {
            "failed": viz.FAILED, "survived": viz.SURVIVED, "censored": viz.CENSORED,
            **viz.MODEL_COLORS,
        }),
    ):  # fmt: skip
        cells = "".join(
            f'<div style="flex:1;display:flex;flex-direction:column;gap:6px">'
            f'<div style="height:44px;background:{color}"></div>'
            f'<span class="cb-mono" style="font-size:11px;color:var(--slate)">{name.upper()}</span>'
            f'<span class="cb-mono" style="font-size:12px;color:var(--ink)">{color}</span></div>'
            for name, color in palette.items()
        )
        rows += (
            f'<div style="display:flex;gap:12px;align-items:flex-start;margin-top:20px">'
            f'<span class="cb-mono" style="flex:none;width:96px;padding-top:14px;font-size:12px;'
            f'letter-spacing:.14em;color:var(--slate)">{label}</span>{cells}</div>'
        )
    return f'<div class="cb" style="height:280px;padding:28px 40px">{rows}</div>'


def exhibit_event_study(frame: pd.DataFrame, out) -> None:
    """Liabilities to assets before the final filing, failed companies against survivors."""
    dev = frame.loc[frame["year"] <= DEV_END]
    failed = dev.loc[dev["failed"] & (dev["last_year"] <= DEV_END)]
    path = failed.loc[failed["years_to_final"] <= 8].groupby("years_to_final")[
        "liabilities_to_assets"
    ]
    quantiles = path.quantile([0.25, 0.5, 0.75]).unstack()
    survivors = dev.loc[~dev["failed"], "liabilities_to_assets"].median()
    x = -quantiles.index.to_numpy()
    fig, ax = plt.subplots(figsize=(viz.EXHIBIT_WIDTH_IN, 4.2))
    ax.fill_between(x, quantiles[0.25], quantiles[0.75], color=viz.FAILED, alpha=0.15, linewidth=0,
                    label="Failed companies, middle half")  # fmt: skip
    ax.plot(x, quantiles[0.5], color=viz.FAILED, marker="o", markersize=5,
            label="Failed companies, median")  # fmt: skip
    ax.axhline(survivors, color=viz.SURVIVED, linewidth=1.5, linestyle=(0, (4, 3)),
               label=f"Surviving firm-years, median {survivors:.2f}")  # fmt: skip
    ax.legend(loc="upper left")
    ax.set_xticks(range(-8, 1))
    ax.set_xlabel("Years before the final filing")
    ax.set_ylabel("Total liabilities / total assets")
    viz.exhibit(
        fig,
        "Leverage climbs for years before a failure",
        f"Median leverage of failed companies rises from {quantiles.loc[8, 0.5]:.2f} eight years "
        f"out to {quantiles.loc[0, 0.5]:.2f} in the final filing.",
        SOURCE,
    )
    fig.savefig(out)
    plt.close(fig)


def exhibit_roa_rate(labelled: pd.DataFrame, out) -> None:
    """Two-year failure rate by decile of return on assets, with Wilson intervals."""
    dev = development(labelled).dropna(subset=["roa"])
    deciles = pd.qcut(dev["roa"], 10, labels=False) + 1
    table = dev.groupby(deciles)["target"].agg(["sum", "size"])
    rates = table["sum"] / table["size"]
    bounds = np.array(
        [wilson_interval(int(k), int(n)) for k, n in zip(table["sum"], table["size"])]
    )
    base = dev["target"].mean()
    above = [int(d) for d in rates.index if rates[d] > base]
    fig, ax = plt.subplots(figsize=(viz.EXHIBIT_WIDTH_IN, 4.2))
    ax.axhline(base, color=viz.MIST, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(10.45, base, f"base rate {base:.1%}", color=viz.SLATE, size=8.5, va="bottom")
    ax.errorbar(table.index, rates, yerr=[rates - bounds[:, 0], bounds[:, 1] - rates],
                fmt="o", color=viz.FAILED, markersize=6, elinewidth=1.5, capsize=0)  # fmt: skip
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.5, 11.8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax.set_xlabel("Decile of return on assets (1 = lowest)")
    ax.set_ylabel("Two-year failure rate")
    viz.exhibit(
        fig,
        "Failure risk sits across the lower deciles of return on assets, not only the lowest",
        f"Wilson 95% intervals. Deciles above the base rate: {', '.join(map(str, above))}.",
        SOURCE,
    )
    fig.savefig(out)
    plt.close(fig)


def exhibit_km_cohorts(events: pd.DataFrame, out) -> None:
    """Kaplan-Meier survival from failure by entry cohort, with Greenwood bands."""
    cohorts = {
        "Listed by 1999": events["entry_year"] == 1999,
        "Entered 2000 to 2005": events["entry_year"].between(2000, 2005),
        "Entered 2006 to 2011": events["entry_year"].between(2006, 2011),
    }
    shades = ["#9DB4CC", "#5F86AE", "#173F66"]  # one hue, light to dark: entry time is ordered
    fig, ax = plt.subplots(figsize=(viz.EXHIBIT_WIDTH_IN, 4.2))
    for (name, mask), color in zip(cohorts.items(), shades):
        part = events.loc[mask]
        km = kaplan_meier(part["duration"], (part["event"] == "failed").astype(int))
        times = np.concatenate([[0], km["time"]])
        surv = np.concatenate([[1.0], km["survival"]])
        se = np.concatenate([[0.0], km["se"]])
        ax.fill_between(times, surv - 1.96 * se, surv + 1.96 * se, step="post", color=color,
                        alpha=0.18, linewidth=0)  # fmt: skip
        ax.step(times, surv, where="post", color=color, label=name)
    ax.set_xlim(0, 21)
    ax.set_xticks(range(0, 21, 5))
    ax.legend(loc="lower left")
    ax.set_xlabel("Years since first filing in the panel")
    ax.set_ylabel("Share not yet failed")
    viz.exhibit(
        fig,
        "Companies already listed in 1999 fail at a different pace",
        "Failure only; other exits are censored. The 1999 cohort's earlier years are unseen.",
        "Lombardo et al. (2022), Kaggle copy; every year, exits only",
    )
    fig.savefig(out)
    plt.close(fig)


def exhibit_correlation(frame: pd.DataFrame, out) -> None:
    """Spearman correlation of the ratios, ordered by hierarchical clustering."""
    binary = {"no_lt_debt", "negative_ebitda"}
    names = [name for members in FAMILIES.values() for name in members if name not in binary]
    corr = frame.loc[frame["year"] <= DEV_END, names].corr(method="spearman")
    distance = squareform(1 - corr.abs().to_numpy(), checks=False)
    order = hierarchy.leaves_list(hierarchy.linkage(distance, method="average"))
    corr = corr.iloc[order, order]
    cmap = LinearSegmentedColormap.from_list("cb_diverging", [viz.MODEL_COLORS["trees"], viz.SHEET,
                                                              viz.SURVIVED])  # fmt: skip
    fig, ax = plt.subplots(figsize=(viz.EXHIBIT_WIDTH_IN, 6.4))
    image = ax.imshow(corr.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), corr.columns, rotation=90)
    ax.set_yticks(range(len(names)), corr.index)
    ax.grid(False)
    ax.tick_params(length=0)
    fig.colorbar(image, ax=ax, shrink=0.7, pad=0.02, label="Spearman correlation")
    viz.exhibit(
        fig,
        "The ratios fall into a few families",
        "Continuous ratios, ordered by clustering on absolute correlation. Blocks are redundant groups.",
        SOURCE,
    )
    for axis in fig.axes:
        axis.tick_params(labelsize=7)
    fig.savefig(out)
    plt.close(fig)


def on_dark(path) -> None:
    """The picture on GitHub's dark page colour, with a margin, to check its edges."""
    image = Image.open(path).convert("RGB")
    canvas = Image.new("RGB", (image.width + 160, image.height + 160), DARK_PAGE)
    canvas.paste(image, (80, 80))
    dark = OUT / "dark"
    dark.mkdir(parents=True, exist_ok=True)
    canvas.save(dark / path.name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    events = event_table(raw)
    marks = house_marks(events)
    counts = {
        "FIRM-YEARS": len(raw),
        "COMPANIES": raw["company_name"].nunique(),
        "FAILURES": int((events["event"] == "failed").sum()),
    }
    written = []
    with Studio() as studio:
        for name, mark in marks.items():
            written.append(studio.png(cover(1, "The Panel", "What exactly is in this file?", 45,
                                            mark_svg(mark, 1)), OUT / f"cover_ch1_mark{name}.png"))  # fmt: skip
            written.append(studio.png(cover(4, "The Yardstick", "How will we know a model is good?",
                                            55, mark_svg(mark, 4)), OUT / f"cover_ch4_mark{name}.png"))  # fmt: skip
            written.append(
                studio.png(hero(mark, counts), OUT / f"hero_mark{name}.png", LANDING_WIDTH)
            )
        written.append(studio.png(route(CHAPTER_ONE_PARTS, "ABOUT 45 MIN"), OUT / "route_ch1.png"))
        for index, goal, minutes in [
            (1, "Load the panel and the house style.", 2),
            (5, "Measure who arrives late and who leaves gaps.", 5),
            (9, "Name what the file cannot tell us.", 3),
        ]:
            written.append(studio.png(opener(index, 9, goal, minutes), OUT / f"opener_{index}.png"))
        written.append(studio.png(pitfall(
            "SAMPLE: a random split against a split by time", "RANDOM ROW SPLIT, ROC-AUC",
            "0.91", "SPLIT BY YEAR, ROC-AUC", "0.78",
            "SAMPLE numbers. A random split lets a company's own later years leak into training.",
        ), OUT / "pitfall_sample_split.png"))  # fmt: skip
        written.append(studio.png(pitfall(
            "SAMPLE: eighteen columns, fourteen independent", "COLUMNS IN THE FILE", "18",
            "INDEPENDENT ITEMS", "14", "SAMPLE numbers. Four columns are sums of the others.",
        ), OUT / "pitfall_sample_columns.png"))  # fmt: skip
        written.append(studio.png(label_timeline(), OUT / "diagram_label_timeline.png"))
        written.append(studio.png(next_card("NEXT", "CHAPTER 2", "The Event",
                                            "What does failed mean, and when did it happen?"),
                                  OUT / "next_card.png"))  # fmt: skip
        written.append(studio.png(palette_plate(), OUT / "palette_spec_and_validated.png"))

    viz.use_house_style()
    frame = add_ratios(raw)
    last_year = frame.groupby("company_name")["year"].transform("max")
    frame = frame.assign(
        failed=frame["status_label"] == "failed",
        last_year=last_year,
        years_to_final=last_year - frame["year"],
    )
    exhibit_event_study(frame, OUT / "exhibit_event_study.png")
    exhibit_roa_rate(add_ratios(build_label(raw, 2)), OUT / "exhibit_roa_rate.png")
    exhibit_km_cohorts(events, OUT / "exhibit_km_cohorts.png")
    exhibit_correlation(frame, OUT / "exhibit_correlation.png")
    written += [
        OUT / f"exhibit_{n}.png" for n in ("event_study", "roa_rate", "km_cohorts", "correlation")
    ]
    for path in written:
        on_dark(path)
        print(f"{path.name}: {Image.open(path).size[0]} px wide")


if __name__ == "__main__":
    main()
