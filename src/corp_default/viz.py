"""The chart house style: palettes, fonts, and the exhibit frame every chart in the casebook wears."""

import dataclasses
import warnings

import matplotlib as mpl
from cycler import cycler
from IPython import get_ipython
from IPython.core.pylabtools import print_figure
from matplotlib import font_manager
from matplotlib.figure import Figure

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
    # Tick labels take the mono face first, so the layout is fitted to the labels as drawn.
    for ax in fig.axes:
        ax.tick_params(labelfontfamily=_fonts["mono"], labelsize=8.5)
    # Let a constrained layout set the margins around tick labels, freeze it, then grow the figure
    # and move every axes down into the new room.
    if fig.get_layout_engine() is None:
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
    fig.text(left, at(0.40), title, family=_fonts["display"], weight=600, size=14, color=INK,
             va="top")  # fmt: skip
    fig.text(left, at(0.68), note, size=10, color=SLATE, va="top")
    if source:
        fig.text(left, 0.12 / new_height, f"Source: {source}", size=8, color=MIST, va="bottom")
    if save_as:
        GALLERY.mkdir(parents=True, exist_ok=True)
        fig.savefig(GALLERY / f"{save_as}.png")


def symlog_axis(ax, axis: str = "x", linthresh: float = 1.0) -> None:
    """A symmetric-log scale for a signed quantity: linear within +/- linthresh, log beyond it."""
    if axis == "x":
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.axvline(0, color=RULE, linewidth=0.8, zorder=0)
    else:
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.axhline(0, color=RULE, linewidth=0.8, zorder=0)
