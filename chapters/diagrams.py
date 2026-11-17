"""The casebook's drawn diagrams, one function each. Each returns the HTML of a figure plate."""

from corp_default import store
from corp_default.labels import PANEL_END
from corp_default.splits import (
    CRISIS_FOLDS,
    DEV_END,
    HOLDOUT,
    HORIZON,
    development,
    walk_forward_folds,
)
from tools.casebook.components import arrows, box, figure_plate, stage, timebar, words

YEAR_WIDTH = 110
LEFT = 60
FIRST_YEAR = 1999  # the file's first fiscal year


def _x(index: int) -> int:
    return LEFT + index * YEAR_WIDTH


def label_timeline() -> str:
    """One company's filings, one row, and the two years its label looks ahead."""
    years = list(range(2002, 2010))
    final = years.index(2008)
    observed = years.index(2006)
    parts = [words(_x(i), 0, YEAR_WIDTH, str(year), "m") for i, year in enumerate(years)]
    for i in range(1, final):
        parts.append(box(_x(i) + 6, 30, YEAR_WIDTH - 12, 34, "tint"))
    parts.append(box(_x(final) + 6, 30, YEAR_WIDTH - 12, 34, "ink"))
    parts.append(words(_x(1), 72, YEAR_WIDTH * (final - 1), "Annual filings", "s"))
    parts.append(words(_x(final) - 20, 72, YEAR_WIDTH + 40, "Last filing", "s b"))
    parts.append(timebar(_x(observed) + 6, 110, YEAR_WIDTH - 12, 30, "valid", "YEAR t"))
    parts.append(
        timebar(_x(observed + 1) + 6, 110, 2 * YEAR_WIDTH - 12, 30, "label", "t + 1 TO t + 2")
    )
    parts.append(arrows(1000, 230, (_x(observed) + 55, 104, _x(observed) + 55, 68)))
    parts.append(
        words(
            LEFT,
            166,
            8 * YEAR_WIDTH,
            "The 2006 row is positive: the last filing falls in the two years after it.",
        )
    )
    parts.append(
        words(
            LEFT,
            194,
            8 * YEAR_WIDTH,
            "The 2007 row is positive too. The 2008 row is dropped: nothing follows it to predict.",
        )
    )
    return figure_plate(
        stage(1000, 230, *parts),
        "The label of one firm-year: does the last filing fall one or two years later?",
    )


def split_map() -> str:
    """The whole file on one line of years: where models are built, the gap, and the sealed years."""
    wide = 44
    years = list(range(FIRST_YEAR, PANEL_END + 1))

    def x(year: int) -> int:
        return LEFT + (year - FIRST_YEAR) * wide

    def span(first: int, last: int) -> tuple[int, int]:
        return x(first) + 2, (last - first + 1) * wide - 4

    parts = [words(x(year), 0, wide, str(year), "m") for year in years]
    spans = [
        (FIRST_YEAR, DEV_END, "train", "DEVELOPMENT", "Every model is built and chosen here"),
        (DEV_END + 1, HOLDOUT[0] - 1, "embargo", "", "Left out"),
        (HOLDOUT[0], HOLDOUT[1], "holdout", "SEALED", "Opened in Chapter 7"),
    ]
    for first, last, kind, label, note in spans:
        left, width = span(first, last)
        parts.append(timebar(left, 34, width, 36, kind, label))
        parts.append(words(left - 30, 80, width + 60, f"{first} to {last}", "b"))
        parts.append(words(left - 30, 102, width + 60, note, "s"))
    left, width = span(HOLDOUT[1] + 1, PANEL_END)
    parts.append(box(left, 34, width, 36))
    parts.append(words(left - 30, 80, width + 60, f"{HOLDOUT[1] + 1} to {PANEL_END}", "b"))
    parts.append(words(left - 30, 102, width + 60, "No label yet", "s"))
    return figure_plate(
        stage(1000, 130, *parts),
        "The split by time: build on the early years, leave a gap, and test once on the late years.",
    )


def embargo() -> str:
    """Why the last training year sits three years before the validation year."""
    wide = 150
    names = [f"Y - {back}" for back in range(5, 0, -1)] + ["Y"]

    def x(index: int) -> int:
        return LEFT + index * wide

    last_train = len(names) - 2 - HORIZON  # the index of year Y - HORIZON - 1
    parts = [words(x(i), 0, wide, name, "m") for i, name in enumerate(names)]
    parts.append(timebar(x(0) + 3, 34, (last_train + 1) * wide - 6, 34, "train", "TRAIN"))
    parts.append(timebar(x(last_train + 1) + 3, 34, HORIZON * wide - 6, 34, "embargo"))
    parts.append(timebar(x(len(names) - 1) + 3, 34, wide - 6, 34, "valid", "VALIDATE"))
    parts.append(
        timebar(
            x(last_train + 1) + 3,
            100,
            HORIZON * wide - 6,
            30,
            "label",
            "LABEL OF THE LAST TRAINING ROW",
        )
    )
    parts.append(arrows(1000, 250, (x(last_train) + wide // 2, 72, x(last_train + 1) + 20, 98)))
    parts.append(
        words(
            LEFT,
            140,
            len(names) * wide,
            f"A row of year {names[last_train]} has a label that is settled by the end of {names[-2]}.",
        )
    )
    parts.append(
        timebar(
            x(last_train + 2) + 3,
            180,
            HORIZON * wide - 6,
            30,
            "label",
            "LABEL OF A ROW ONE YEAR LATER",
        )
    )
    parts.append(
        words(
            LEFT,
            220,
            len(names) * wide,
            f"A row of year {names[last_train + 1]} needs the filings of year Y, so no model for year Y may train on it.",
            "s",
        )
    )
    return figure_plate(
        stage(1000, 250, *parts),
        "A label that looks two years ahead needs a two-year gap before the validation year.",
    )


def fold_map() -> str:
    """Every walk-forward fold as a row: its training years, the gap, its validation year, its positives."""
    wide, tall = 56, 30
    dev = development(store.load_frame("labelled_panel"))
    folds = walk_forward_folds(dev)
    years = list(range(FIRST_YEAR, DEV_END + 1))
    right = LEFT + len(years) * wide + 16

    def x(year: int) -> int:
        return LEFT + (year - FIRST_YEAR) * wide

    parts = [words(x(year), 0, wide, str(year), "m") for year in years]
    parts.append(words(right, 0, 200, "POSITIVES TO FIND", "m left"))
    for row, fold in enumerate(folds):
        y = 32 + row * (tall + 8)
        last_train = fold.year - HORIZON - 1
        trained = len(fold.train)
        parts.append(
            timebar(
                x(FIRST_YEAR) + 2,
                y,
                (last_train - FIRST_YEAR + 1) * wide - 4,
                tall,
                "train",
                f"TRAIN ON {trained:,} ROWS",
            )
        )
        parts.append(timebar(x(last_train + 1) + 2, y, HORIZON * wide - 4, tall, "embargo"))
        parts.append(timebar(x(fold.year) + 2, y, wide - 4, tall, "valid", str(fold.year)))
        found = int(dev.loc[fold.valid, "target"].sum())
        crisis = ", a crisis fold" if fold.year in CRISIS_FOLDS else ""
        parts.append(words(right, y + 5, 200, f"{found}{crisis}", "left"))
    height = 32 + len(folds) * (tall + 8)
    return figure_plate(
        stage(1000, height, *parts),
        f"Walking forward in {len(folds)} folds: each trains on the years up to "
        f"{HORIZON + 1} before the year it validates on.",
    )


def sequence_tensor() -> str:
    """Two rows of 2008 as five-year windows: a company with a reporting gap, and a young one."""
    years = list(range(2004, 2009))
    wide, tall, left = 140, 34, 280
    rows = [
        ("Filed in 2004, 2005, and 2008", (True, True, False, False, True)),
        ("First filed in 2007", (False, False, False, True, True)),
    ]
    parts = [words(left + i * wide, 0, wide, str(year), "m") for i, year in enumerate(years)]
    for row, (name, filed) in enumerate(rows):
        y = 34 + row * 104
        parts.append(words(0, y + 8, left - 20, name, "left b"))
        parts.append(words(0, y + tall + 10, left - 20, "Mask", "left s"))
        for i, on in enumerate(filed):
            kind, label = ("train", "RATIOS") if on else ("embargo", "")
            parts.append(timebar(left + i * wide + 3, y, wide - 6, tall, kind, label))
            parts.append(words(left + i * wide, y + tall + 10, wide, "1" if on else "0", "m"))
    return figure_plate(
        stage(1000, 34 + len(rows) * 104 - 30, *parts),
        "A row becomes five years of ratios, oldest first, and a mask that marks the years filed.",
    )
