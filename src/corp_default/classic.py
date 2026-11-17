"""The classic scores, each with its published coefficients and source. Higher is safer.

Zmijewski (1984) is not here: its coefficients are under "Not verified" in SOURCES.md.
"""

import numpy as np
import pandas as pd

ALTMAN_ZONES = {"distress_below": 1.81, "safe_above": 2.99}


def altman_z(frame: pd.DataFrame) -> pd.Series:
    """Altman (1968) Z for public manufacturers, on the ratio columns of features.add_ratios.

    Z = 1.2 X1 + 1.4 X2 + 3.3 X3 + 0.6 X4 + 1.0 X5, with X1 working capital, X2 retained
    earnings, X3 EBIT, and X5 sales, each over total assets, and X4 market value of equity over
    total liabilities. Source: Altman (2000), pp. 9 and 13, in SOURCES.md.
    """
    return (
        1.2 * frame["working_capital_to_assets"]
        + 1.4 * frame["retained_earnings_to_assets"]
        + 3.3 * frame["ebit_to_assets"]
        + 0.6 * frame["mve_to_liabilities"]
        + 1.0 * frame["asset_turnover"]
    )


def altman_z2(frame: pd.DataFrame) -> pd.Series:
    """Altman Z'' for non-manufacturers: no sales term, book equity in place of market value.

    Z'' = 6.56 X1 + 3.26 X2 + 6.72 X3 + 1.05 X4, X4 book equity over total liabilities.
    Source: Altman (2000), p. 27, in SOURCES.md. Its zones are verified only for the
    emerging-market form with a +3.25 constant, so none are given here.
    """
    liabilities = frame["total_liabilities"].replace(0, np.nan)
    book_equity_to_liabilities = (frame["total_assets"] - frame["total_liabilities"]) / liabilities
    return (
        6.56 * frame["working_capital_to_assets"]
        + 3.26 * frame["retained_earnings_to_assets"]
        + 6.72 * frame["ebit_to_assets"]
        + 1.05 * book_equity_to_liabilities
    )


def altman_zone(z: pd.Series) -> pd.Series:
    """Altman (1968) p. 606: above 2.99 safe, below 1.81 distress, between them gray."""
    zones = pd.Series(pd.NA, index=z.index, dtype="string")
    zones.loc[z < ALTMAN_ZONES["distress_below"]] = "distress"
    zones.loc[z.between(ALTMAN_ZONES["distress_below"], ALTMAN_ZONES["safe_above"])] = "gray"
    zones.loc[z > ALTMAN_ZONES["safe_above"]] = "safe"
    return zones
