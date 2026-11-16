import hashlib

import pandas as pd
import pytest

from tools import get_data


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _no_download(url, dest):
    raise AssertionError(f"downloaded {url}")


def test_get_data_refuses_a_hash_mismatch(tmp_path):
    bad = tmp_path / "file.csv"
    bad.write_text("x")
    with pytest.raises(ValueError, match="SHA-256"):
        get_data.verify(bad, "0" * 64)


def test_verify_accepts_the_right_hash(tmp_path):
    good = tmp_path / "file.csv"
    good.write_text("x")
    get_data.verify(good, _sha(good))


def test_a_present_file_with_the_right_hash_is_not_downloaded_again(tmp_path, monkeypatch):
    present = tmp_path / "american_bankruptcy.csv"
    present.write_text("company_name\nC_1\n")
    monkeypatch.setattr(get_data, "KAGGLE_SHA256", _sha(present))
    monkeypatch.setattr(get_data, "download", _no_download)
    get_data.fetch_kaggle(present)


def test_a_present_file_with_the_wrong_hash_is_refused_and_kept(tmp_path, monkeypatch):
    present = tmp_path / "american_bankruptcy.csv"
    present.write_text("edited")
    monkeypatch.setattr(get_data, "download", _no_download)
    with pytest.raises(ValueError, match="SHA-256"):
        get_data.fetch_kaggle(present)
    assert present.read_text() == "edited"


def _authors_file(path):
    pd.DataFrame(
        {
            "company_name": ["C_1", "C_1", "C_2"],
            "fyear": [1999.0, 2000.0, 1999.0],
            "X1": [511267, 485856, 1.0],
            "Division": ["D", "D", "E"],
            "MajorGroup": [37, 37, 48],
        }
    ).to_csv(path, index=False)


def test_fetch_industry_keeps_only_the_three_columns(tmp_path, monkeypatch):
    reference = tmp_path / "reference.csv"
    _authors_file(reference)
    monkeypatch.setattr(get_data, "AUTHORS_SHA256", _sha(reference))
    monkeypatch.setattr(get_data, "download", lambda url, dest: _authors_file(dest))
    dest = tmp_path / "industry.csv"
    get_data.fetch_industry(dest)
    written = pd.read_csv(dest)
    assert list(written.columns) == ["company_name", "division", "major_group"]
    assert written.to_dict("records") == [
        {"company_name": "C_1", "division": "D", "major_group": 37},
        {"company_name": "C_2", "division": "E", "major_group": 48},
    ]


def test_a_download_with_the_wrong_hash_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(get_data, "download", lambda url, dest: dest.write_text("tampered"))
    dest = tmp_path / "industry.csv"
    with pytest.raises(ValueError, match="SHA-256"):
        get_data.fetch_industry(dest)
    assert not dest.exists()
