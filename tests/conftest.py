from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def raw():
    # Imported here so the harness collects before data.py exists and on a clone without the CSV.
    from corp_default.data import DATA_PATH, load_raw

    if not DATA_PATH.exists():
        pytest.skip("data/american_bankruptcy.csv is not present")
    return load_raw()


@pytest.fixture(scope="session")
def industry():
    from corp_default.data import INDUSTRY_PATH, load_industry

    if not INDUSTRY_PATH.exists():
        pytest.skip("data/industry.csv is not present")
    return load_industry()


@pytest.fixture(scope="session")
def toy_mark():
    """A house mark from four made-up companies, so the fixture chapter needs no data."""
    import pandas as pd

    from tools.casebook.components import house_mark

    events = pd.DataFrame(
        {"duration": [1, 2, 3, 4], "event": ["failed", "left", "failed", "reporting"]}
    )
    return house_mark(events)


@pytest.fixture(scope="session")
def built(tmp_path_factory, toy_mark):
    """The fixture chapter, built once for the session: the written notebook and its build line."""
    from tools.casebook.build import build

    return build(0, mark=toy_mark, chapters=FIXTURES, out=tmp_path_factory.mktemp("notebooks"))
