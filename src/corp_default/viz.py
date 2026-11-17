"""The chart house style: palettes, fonts, and the exhibit frame every chart in the casebook wears."""

import dataclasses
import re
import warnings

import matplotlib as mpl
from cycler import cycler
from IPython import get_ipython
from IPython.core.pylabtools import print_figure
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter

from corp_default.data import REPO

FONTS = REPO / "assets" / "fonts"
GALLERY = REPO / "assets" / "readme" / "gallery"

SHEET = "#F7F7F5"
RULE = "#CFCFC9"
INK = "#16181D"
SLATE = "#4F5560"
MIST = "#8B909A"

# Outcome palette: failure, survival, and the neutral for censored or any other exit.
FAILED = "#D9482B"
SURVIVED = "#295F92"
CENSORED = "#878C96"
# Model palette. classic and blend are neutrals; the three hues pass the dataviz checks on the sheet.
MODEL_COLORS = {
    "classic": "#878C96",
    "logistic": "#295F92",
    "trees": "#B48219",
    "networks": "#8C6FB9",
    "blend": "#16181D",
}
# Two groups that are neither outcomes nor models: ink for the subject, this hue for the other.
ACCENT = "#0B8A6F"

# Groups in time order, such as cohorts or eras: one hue, and the later the darker.
COHORTS = ("#A9BFD6", "#5F86AE", "#1F4E7A")

# What a chart calls each column. Code names stay in the code and in the data dictionaries.
LABELS = {
    "liabilities_to_assets": "Liabilities / assets",
    "lt_debt_to_assets": "Long-term debt / assets",
    "book_equity_to_assets": "Book equity / assets",
    "debt_to_ebitda": "Debt / EBITDA",
    "no_lt_debt": "No long-term debt",
    "roa": "Return on assets",
    "ebit_to_assets": "EBIT / assets",
    "ebitda_to_assets": "EBITDA / assets",
    "net_margin": "Net margin",
    "ebitda_margin": "EBITDA margin",
    "gross_margin": "Gross margin",
    "negative_ebitda": "Negative EBITDA",
    "retained_earnings_to_assets": "Retained earnings / assets",
    "current_ratio": "Current ratio",
    "quick_ratio": "Quick ratio",
    "working_capital_to_assets": "Working capital / assets",
    "current_assets_to_assets": "Current assets / assets",
    "current_liabilities_to_liabilities": "Current liabilities / liabilities",
    "ffo_to_liabilities": "Funds from operations / liabilities",
    "asset_turnover": "Sales / assets",
    "mve_to_liabilities": "Market value / liabilities",
    "market_to_book_assets": "Market value plus liabilities / assets",
    "log_assets": "Total assets (log)",
    "log_market_cap": "Market value (log)",
    "years_observed": "Years observed",
    "cogs": "Cost of goods sold",
    "depreciation_amort": "Depreciation and amortization",
    "ebitda": "EBITDA",
    "ebit": "EBIT",
    "market_cap": "Market value",
    "total_lt_debt": "Long-term debt",
    "total_current_liabilities": "Current liabilities",
    "total_receivables": "Receivables",
    "sales": "Sales",
    "assets": "Total assets",
}
# Shorter names for the title of a narrow panel.
SHORT_LABELS = {
    "ffo_to_liabilities": "FFO / liabilities",
    "current_liabilities_to_liabilities": "Current / total liabilities",
    "market_to_book_assets": "Market-to-book, assets",
}
_LAG_WORDS = {"1": "one year", "2": "two years"}

EXHIBIT_WIDTH_IN = 8.0  # 1200 px at 150 dpi
DPI = 150
TOP_IN, FOOT_IN = 0.95, 0.4  # room the exhibit frame adds above and below the axes

# matplotlib reads JetBrains Mono's weights as 436 and 558; register the files at the weights
# they serve, so asking for normal or bold text finds them without a fallback.
MONO_WEIGHTS = {"JetBrainsMono-Medium.ttf": 400, "JetBrainsMono-Bold.ttf": 700}

_fonts = {"display": "Archivo", "sans": "Public Sans", "mono": "JetBrains Mono"}
_count = 0


def _register(files: list) -> None:
    manager = font_manager.fontManager
    known = {entry.fname for entry in manager.ttflist}
    for path in files:
        if str(path) in known:
            continue
        manager.addfont(str(path))
        if path.name in MONO_WEIGHTS:
            # A file can register under more than one family name; fix the one the charts ask for.
            manager.ttflist = [
                dataclasses.replace(entry, weight=MONO_WEIGHTS[path.name])
                if entry.fname == str(path) and entry.name == "JetBrains Mono"
                else entry
                for entry in manager.ttflist
            ]


def notebook_png(fig: Figure) -> tuple[str, dict]:
    """A figure as a notebook shows it: the whole canvas, with the exhibit's label as alt text.

    Jupyter's default crops each figure to its ink (bbox_inches="tight"), which would change an
    exhibit's width from one chart to the next.
    """
    return print_figure(fig, "png", bbox_inches=None, base64=True), {"alt": fig.get_label()}


def use_house_style() -> None:
    """Set the house style, register the fonts, and start exhibit numbering again at 1."""
    global _count, _fonts
    _count = 0
    shell = get_ipython()
    if shell is not None:
        # The inline backend installs its own PNG formatter when it loads; ours must follow it.
        import matplotlib_inline.backend_inline  # noqa: F401

        shell.display_formatter.formatters["image/png"].for_type(Figure, notebook_png)
    files = sorted(FONTS.glob("*/*.ttf"))
    if files:
        _register(files)
        _fonts = {"display": "Archivo", "sans": "Public Sans", "mono": "JetBrains Mono"}
    else:
        warnings.warn(
            f"no font files under assets/fonts ({FONTS}); charts fall back to DejaVu Sans",
            UserWarning,
            stacklevel=2,
        )
        _fonts = {"display": "DejaVu Sans", "sans": "DejaVu Sans", "mono": "DejaVu Sans Mono"}
    mpl.rcParams.update(
        {
            "figure.facecolor": SHEET,
            "axes.facecolor": SHEET,
            "savefig.facecolor": SHEET,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "figure.figsize": (EXHIBIT_WIDTH_IN, 4.5),
            "font.family": [_fonts["sans"]],
            "font.size": 10,
            "text.color": INK,
            "axes.edgecolor": RULE,
            "axes.labelcolor": SLATE,
            "axes.titlesize": 11,
            "axes.titleweight": 600,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": RULE,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "xtick.color": RULE,
            "ytick.color": RULE,
            "xtick.labelcolor": SLATE,
            "ytick.labelcolor": SLATE,
            "lines.linewidth": 2.0,
            # No default cycle: a series without a colour is ink, so every colour is a choice.
            "axes.prop_cycle": cycler(color=[INK]),
            "legend.frameon": False,
        }
    )


def exhibit(
    fig: Figure, title: str, note: str, source: str = "", save_as: str | None = None
) -> None:
    """Frame a finished chart: an EXHIBIT n tag and title at the top left, one line on what to
    notice, and a source line at the foot. save_as also writes assets/readme/gallery/<name>.png."""
    global _count
    _count += 1
    fig.set_label(f"Exhibit {_count}: {title}. {note}")
    # Tick labels take their face first, so the layout is fitted to the labels as drawn: numbers
    # in the mono face, and an axis of words in the text face.
    for ax in fig.axes:
        for which, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            words = any(
                re.search(r"[A-Za-z]{3}", tick.get_text()) for tick in axis.get_ticklabels()
            )
            ax.tick_params(
                axis=which,
                labelfontfamily=_fonts["sans"] if words else _fonts["mono"],
                labelsize=9.5 if words else 8.5,
            )
    # Let a constrained layout set the margins around tick labels, freeze it, then grow the figure
    # and move every axes down into the new room.
    if fig.get_layout_engine() is None:
        # A colorbar made at full size sits in a grid whose spacer rows have zero height, which a
        # constrained layout cannot divide by.
        for ax in fig.axes:
            if spec := ax.get_subplotspec():
                grid = spec.get_gridspec()
                grid.set_height_ratios([ratio or 1e-3 for ratio in grid.get_height_ratios()])
                grid.set_width_ratios([ratio or 1e-3 for ratio in grid.get_width_ratios()])
        fig.set_layout_engine("constrained", w_pad=0.15, h_pad=0.1)
    fig.draw_without_rendering()
    fig.set_layout_engine("none")
    old_height = fig.get_figheight()
    new_height = old_height + TOP_IN + FOOT_IN
    for ax in fig.axes:
        box = ax.get_position()
        ax.set_position(
            [
                box.x0,
                (box.y0 * old_height + FOOT_IN) / new_height,
                box.width,
                box.height * old_height / new_height,
            ]
        )
    fig.set_figheight(new_height)

    def at(inches_from_top: float) -> float:
        return 1 - inches_from_top / new_height

    left = 0.32 / fig.get_figwidth()
    fig.text(left, at(0.22), f"EXHIBIT {_count}", family=_fonts["mono"], weight=700, size=8.5,
             color=SLATE, va="top")  # fmt: skip
    heading = fig.text(left, at(0.40), title, family=_fonts["display"], weight=600, size=14,
                       color=INK, va="top")  # fmt: skip
    subheading = fig.text(left, at(0.68), note, size=10, color=SLATE, va="top")
    # A line that runs past the right margin is cut off in the notebook, so it is refused here.
    renderer = fig.canvas.get_renderer()
    limit = fig.bbox.x1 - left * fig.bbox.width
    for name, text in (("title", heading), ("note", subheading)):
        if text.get_window_extent(renderer).x1 > limit:
            raise ValueError(f"the exhibit's {name} is too long for the page: {text.get_text()!r}")
    if source:
        fig.text(left, 0.12 / new_height, f"Source: {source}", size=8, color=MIST, va="bottom")
    if save_as:
        GALLERY.mkdir(parents=True, exist_ok=True)
        fig.savefig(GALLERY / f"{save_as}.png")


def label(name: str, short: bool = False) -> str:
    """The name a chart shows for a column: `d2_roa` is "Return on assets, 2-year change".

    short gives the name that fits over a narrow panel.
    """
    if short and name in SHORT_LABELS:
        return SHORT_LABELS[name]
    if name in LABELS:
        return LABELS[name]
    if match := re.fullmatch(r"([dg])([12])_(.+)", name):
        kind, lag, base = match.groups()
        return f"{label(base)}, {lag}-year {'change' if kind == 'd' else 'growth'}"
    if match := re.fullmatch(r"loss_([12])y_ago", name):
        return f"Loss {_LAG_WORDS[match.group(1)]} ago"
    return name.replace("_", " ").capitalize()


def plain_number(value: float, position: int | None = None) -> str:
    """A tick label a reader can say aloud: 10,000 or 0.01, never a power of ten."""
    return f"{value:,.10f}".rstrip("0").rstrip(".")


def log_axis(ax, axis: str = "x") -> None:
    """A log scale whose ticks read as plain numbers. Minor ticks carry no label: on an axis of
    under one decade matplotlib would write them as powers of ten."""
    if axis == "x":
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(plain_number)
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(plain_number)
        ax.yaxis.set_minor_formatter(NullFormatter())


def symlog_axis(ax, axis: str = "x", linthresh: float = 1.0) -> None:
    """A symmetric-log scale for a signed quantity: linear within +/- linthresh, log beyond it.
    Its ticks read as plain numbers."""
    if axis == "x":
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.xaxis.set_major_formatter(plain_number)
        ax.axvline(0, color=RULE, linewidth=0.8, zorder=0)
    else:
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.yaxis.set_major_formatter(plain_number)
        ax.axhline(0, color=RULE, linewidth=0.8, zorder=0)
