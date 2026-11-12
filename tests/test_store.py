import json

import numpy as np
import pandas as pd
import pytest

from corp_default import store
from corp_default.store import render


@pytest.fixture
def folders(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(store, "RESULTS", tmp_path / "results")
    return tmp_path


def test_render_fills_and_formats():
    known = {"ch1.n": 78682, "ch1.rate": 0.0155}
    assert render("n = {{ch1.n:,}} and {{ch1.rate:.1%}}", known) == "n = 78,682 and 1.6%"


def test_render_refuses_an_unknown_key():
    with pytest.raises(KeyError, match="ch9.missing"):
        render("{{ch9.missing}}", {})


def test_load_frame_says_which_chapter_to_run(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ARTIFACTS", tmp_path)
    with pytest.raises(FileNotFoundError, match="run chapter 2 first"):
        store.load_frame("labelled_panel")


def test_load_json_says_which_chapter_to_run(folders):
    with pytest.raises(FileNotFoundError, match="run chapter 4 first"):
        store.load_json("frozen")


def test_put_before_open_chapter_raises(folders, monkeypatch):
    monkeypatch.setattr(store, "_chapter", None)
    with pytest.raises(RuntimeError, match="open_chapter"):
        store.put("n", 1)


def test_values_carry_between_chapters_and_reopening_clears(folders):
    store.open_chapter(1)
    store.put("n", np.int64(78682))
    store.put("rate", np.float64(0.0155))
    store.open_chapter(2)
    store.put("label", "two years")
    assert store.values() == {"ch1.n": 78682, "ch1.rate": 0.0155, "ch2.label": "two years"}
    assert store.values(upto=1) == {"ch1.n": 78682, "ch1.rate": 0.0155}
    assert isinstance(store.values()["ch1.n"], int)
    store.open_chapter(1)
    assert store.values(upto=1) == {}


def test_put_refuses_a_value_that_is_not_a_number_or_text(folders):
    store.open_chapter(1)
    with pytest.raises(TypeError, match="ch1.bad"):
        store.put("bad", [1, 2])


def test_frames_and_json_round_trip(folders):
    frame = pd.DataFrame({"company_name": ["C_1", "C_2"], "year": [2001, 2002]})
    store.save_frame("event_table", frame)
    pd.testing.assert_frame_equal(store.load_frame("event_table"), frame)
    store.save_json("frozen", {"logistic": {"C": 0.1}})
    assert store.load_json("frozen") == {"logistic": {"C": 0.1}}


def test_a_frame_of_long_floats_is_read_back_exactly(folders):
    # pandas' default parser misses the last digit of many 17-digit floats, so a chapter that
    # read a hand-off and wrote it again would change it
    rng = np.random.default_rng(0)
    scores = rng.normal(size=5000) * 10.0 ** rng.integers(-6, 6, size=5000)
    store.save_frame("oof_scores", pd.DataFrame({"score": scores}))
    assert store.load_frame("oof_scores")["score"].tolist() == scores.tolist()


def test_write_results_sorts_keys(folders):
    store.write_results({"b": 1, "a": 2}, {"z": "z", "y": "y"})
    text = (folders / "results" / "metrics.json").read_text(encoding="utf-8")
    assert list(json.loads(text)) == ["a", "b"]
    assert json.loads((folders / "results" / "claims.json").read_text(encoding="utf-8")) == {
        "y": "y",
        "z": "z",
    }
