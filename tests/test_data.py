import pandas as pd
import pytest

from corp_default.data import COLUMN_MAP, RAW_ITEMS, load_industry, load_raw, panel_spans


def test_load_raw_names_the_file_and_the_data_page_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="american_bankruptcy.csv") as error:
        load_raw(tmp_path / "american_bankruptcy.csv")
    assert "data/README.md" in str(error.value)
    assert "uv run python -m tools.get_data" in str(error.value)


def test_load_industry_names_the_file_and_the_fetch_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="industry.csv") as error:
        load_industry(tmp_path / "industry.csv")
    assert "uv run python -m tools.get_data" in str(error.value)


def test_raw_items_are_the_column_map_without_total_revenue():
    assert len(COLUMN_MAP) == 18
    assert RAW_ITEMS == [name for name in COLUMN_MAP.values() if name != "total_revenue"]


def _write_pair(tmp_path, companies, industry_rows):
    data = tmp_path / "american_bankruptcy.csv"
    pd.DataFrame({"company_name": companies}).to_csv(data, index=False)
    industry = tmp_path / "industry.csv"
    pd.DataFrame(industry_rows, columns=["company_name", "division", "major_group"]).to_csv(
        industry, index=False
    )
    return industry, data


def test_load_industry_refuses_a_company_with_two_divisions(tmp_path):
    industry, data = _write_pair(
        tmp_path, ["C_1", "C_2"], [("C_1", "D", 37), ("C_1", "E", 48), ("C_2", "D", 20)]
    )
    with pytest.raises(ValueError, match="C_1"):
        load_industry(industry, data)


def test_load_industry_refuses_a_company_it_cannot_match(tmp_path):
    industry, data = _write_pair(
        tmp_path, ["C_1", "C_2", "C_3"], [("C_1", "D", 37), ("C_2", "D", 20)]
    )
    with pytest.raises(ValueError, match="1 companies"):
        load_industry(industry, data)


@pytest.mark.needs_data
def test_load_raw_shape_and_columns(raw):
    assert raw.shape == (78_682, 20)
    assert "total_revenue" not in raw.columns
    assert raw["company_name"].nunique() == 8_971


@pytest.mark.needs_data
def test_load_industry_covers_every_company_once(raw, industry):
    assert industry.index.is_unique
    assert set(industry.index) == set(raw["company_name"])
    assert set(industry["division"]) <= set("ABCDEFGHIJ")
    assert pd.api.types.is_integer_dtype(industry["major_group"])


@pytest.mark.needs_data
def test_panel_spans_counts(raw):
    spans = panel_spans(raw)
    assert int((spans["first_year"] == 1999).sum()) == 5_308
    assert int((spans["n_gap_years"] > 0).sum()) == 710
    assert int(spans["failed"].sum()) == 609


def test_panel_spans_on_a_gap_company():
    frame = pd.DataFrame(
        {
            "company_name": ["A", "A", "A", "B"],
            "year": [2001, 2002, 2005, 2003],
            "status_label": ["failed", "failed", "failed", "alive"],
        }
    )
    spans = panel_spans(frame)
    assert spans.loc["A"].to_dict() == {
        "first_year": 2001,
        "last_year": 2005,
        "n_years": 3,
        "n_gap_years": 2,
        "failed": True,
    }
    assert spans.loc["B", "n_gap_years"] == 0
    assert not spans.loc["B", "failed"]
