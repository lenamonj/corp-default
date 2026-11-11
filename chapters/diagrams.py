"""The casebook's drawn diagrams, one function each. Each returns the HTML of a figure plate."""

from tools.casebook.components import arrows, box, figure_plate, stage, timebar, words

YEAR_WIDTH = 110
LEFT = 60


def _x(index: int) -> int:
    return LEFT + index * YEAR_WIDTH


def label_timeline() -> str:
    """One company's filings, one observation year, and the two years its label looks ahead."""
    years = list(range(2002, 2010))
    final = years.index(2008)
    observed = years.index(2006)
    parts = [words(_x(i), 0, YEAR_WIDTH, str(year), "m") for i, year in enumerate(years)]
    for i in range(1, final):
        parts.append(box(_x(i) + 6, 30, YEAR_WIDTH - 12, 34, "tint"))
    parts.append(box(_x(final) + 6, 30, YEAR_WIDTH - 12, 34, "ink"))
    parts.append(words(_x(1), 72, YEAR_WIDTH * (final - 1), "Annual filings", "s"))
    parts.append(words(_x(final) - 20, 72, YEAR_WIDTH + 40, "Final filing", "s b"))
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
            "The 2006 row is positive: the final filing falls in the two years after it.",
        )
    )
    parts.append(
        words(
            LEFT, 194, 8 * YEAR_WIDTH, "The 2008 row is dropped: nothing after it to predict.", "s"
        )
    )
    return figure_plate(
        stage(1000, 230, *parts),
        "The label of one firm-year: does the final filing fall one or two years later?",
    )
