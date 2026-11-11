import pytest


@pytest.fixture(scope="session")
def raw():
    # Imported here so the harness collects before data.py exists and on a clone without the CSV.
    from corp_default.data import DATA_PATH, load_raw

    if not DATA_PATH.exists():
        pytest.skip("data/american_bankruptcy.csv is not present")
    return load_raw()
