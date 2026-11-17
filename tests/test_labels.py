import pandas as pd
import pytest
from legacy import legacy_build_label

from corp_default.labels import PANEL_END, build_label, event_table


@pytest.mark.needs_data
def test_build_label_counts(raw):
    frame = build_label(raw, horizon=2)
    assert len(frame) == 72_393
    assert int(frame["target"].sum()) == 1_123
    assert len(raw) - len(frame) == 6_289


@pytest.mark.needs_data
@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_build_label_equals_legacy(raw, horizon):
    pd.testing.assert_frame_equal(
        build_label(raw, horizon), legacy_build_label(raw.copy(), horizon)
    )


@pytest.mark.needs_data
def test_positives_by_year_match_the_review(raw):
    by_year = build_label(raw, 2).groupby("year")["target"].sum()
    assert by_year.loc[1999] == 16 and by_year.loc[2006] == 114


@pytest.mark.needs_data
def test_event_table(raw):
    events = event_table(raw)
    assert len(events) == 8_971
    assert (events["event"] == "failed").sum() == 609
    assert set(events["event"]) == {"failed", "left", "reporting"}


def test_build_label_does_not_change_its_input():
    frame = pd.DataFrame(
        {"company_name": ["A", "A"], "year": [2001, 2002], "status_label": ["failed", "failed"]}
    )
    before = frame.copy()
    build_label(frame, 2)
    pd.testing.assert_frame_equal(frame, before)


def test_event_table_by_hand():
    frame = pd.DataFrame(
        {
            "company_name": ["F", "F", "L", "L", "R"],
            "year": [2001, 2003, 2005, 2006, PANEL_END],
            "status_label": ["failed", "failed", "alive", "alive", "alive"],
        }
    )
    events = event_table(frame).set_index("company_name")
    assert events.loc["F"].to_dict() == {
        "entry_year": 2001,
        "exit_year": 2003,
        "duration": 3,
        "event": "failed",
    }
    assert events.loc["L", "event"] == "left"
    assert events.loc["R", "event"] == "reporting"
    assert events.loc["R", "duration"] == 1


def test_build_label_by_hand():
    def company(name, first, last, status):
        return [
            {"company_name": name, "year": y, "status_label": status}
            for y in range(first, last + 1)
        ]

    frame = pd.DataFrame(
        company("F", 2010, 2015, "failed")  # fails after its 2015 filing
        + company("E", 2015, PANEL_END, "failed")  # fails at the panel's end
        + company("A", 2014, PANEL_END, "alive")
    )
    labeled = build_label(frame, horizon=2)
    got = {(row.company_name, row.year): row.target for row in labeled.itertuples()}
    assert got == {
        ("F", 2010): 0, ("F", 2011): 0, ("F", 2012): 0, ("F", 2013): 1, ("F", 2014): 1,
        ("E", 2015): 0, ("E", 2016): 1,
        ("A", 2014): 0, ("A", 2015): 0, ("A", 2016): 0,
    }  # fmt: skip
