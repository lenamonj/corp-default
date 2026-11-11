"""Fetch the panel and its industry codes into data/, each checked against SOURCES.md.

uv run python -m tools.get_data
"""

import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from corp_default.data import DATA_PATH, INDUSTRY_PATH

KAGGLE_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "utkarshx27/american-companies-bankruptcy-prediction-dataset"
)
KAGGLE_MEMBER = "american_bankruptcy.csv"
KAGGLE_SHA256 = "cff2c899a97ecd629415cb22f59186000e74e1c0a78cfae036c0a53025419b5e"

AUTHORS_URL = (
    "https://raw.githubusercontent.com/sowide/bankruptcy_dataset/"
    "279e4275101e5df7a39e8e6b3179b66c442557f3/american_bankruptcy_dataset.csv"
)
AUTHORS_SHA256 = "d64ed85ae786d75113455ae238feb24015d19716c10ec1de3a90a52eddc7b04a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> None:
    """Raise unless the file's SHA-256 is the one SOURCES.md records. Deletes nothing."""
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 of {path} is {actual}, expected {expected} (see SOURCES.md)")


def download(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "corp-default-get-data"})
    with urllib.request.urlopen(request, timeout=120) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)


def fetch_kaggle(dest: Path = DATA_PATH) -> None:
    """The Kaggle file, version 3. A file already present is checked, never replaced."""
    if dest.exists():
        verify(dest, KAGGLE_SHA256)
        print(f"{dest.name}: present, SHA-256 matches")
        return
    with tempfile.TemporaryDirectory() as scratch:
        archive = Path(scratch) / "kaggle.zip"
        download(KAGGLE_URL, archive)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extract(KAGGLE_MEMBER, scratch)
        csv = Path(scratch) / KAGGLE_MEMBER
        verify(csv, KAGGLE_SHA256)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(csv, dest)
    print(f"{dest.name}: downloaded, SHA-256 matches")


def fetch_industry(dest: Path = INDUSTRY_PATH) -> None:
    """SIC division and major group per company from the authors' 2024 file; the rest is discarded."""
    if dest.exists():
        print(f"{dest.name}: present")
        return
    with tempfile.TemporaryDirectory() as scratch:
        source = Path(scratch) / "authors_2024.csv"
        download(AUTHORS_URL, source)
        verify(source, AUTHORS_SHA256)
        codes = (
            pd.read_csv(source, usecols=["company_name", "Division", "MajorGroup"])
            .rename(columns={"Division": "division", "MajorGroup": "major_group"})
            .drop_duplicates()
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    codes.to_csv(dest, index=False)
    print(f"{dest.name}: written, {len(codes):,} companies")


def main() -> None:
    fetch_kaggle()
    fetch_industry()


if __name__ == "__main__":
    main()
