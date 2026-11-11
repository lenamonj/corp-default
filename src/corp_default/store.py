"""What chapters hand each other: frames and JSON in artifacts/, narrative values, and results/."""

import json
import re
from numbers import Integral, Real
from pathlib import Path

import pandas as pd

from corp_default.data import REPO

ARTIFACTS = REPO / "artifacts"
RESULTS = REPO / "results"
WRITTEN_BY = {"labelled_panel": 2, "event_table": 2, "features": 3, "oof_scores": 4, "frozen": 4}

_PLACEHOLDER = re.compile(r"\{\{\s*(ch\d+\.\w+)(?::([^}]*))?\s*\}\}")
_chapter: int | None = None


def _values_file(number: int) -> Path:
    return ARTIFACTS / "values" / f"{number:02d}.json"


def open_chapter(number: int) -> None:
    """Start a chapter: its values file is emptied, and put() writes to it from now on."""
    global _chapter
    _chapter = number
    path = _values_file(number)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")


def put(key: str, value) -> None:
    """Record a value as chN.key for the chapter text to quote."""
    if _chapter is None:
        raise RuntimeError("call store.open_chapter(N) before store.put")
    name = f"ch{_chapter}.{key}"
    # bool is an Integral; numpy scalars register as Integral and Real.
    if isinstance(value, Integral):
        value = int(value)
    elif isinstance(value, Real):
        value = float(value)
    elif not isinstance(value, str):
        raise TypeError(f"{name} must be an int, a float, or a str, not {type(value).__name__}")
    path = _values_file(_chapter)
    known = json.loads(path.read_text(encoding="utf-8"))
    known[name] = value
    path.write_text(json.dumps(known, indent=2, sort_keys=True), encoding="utf-8")


def values(upto: int | None = None) -> dict:
    """Every chapter's values, up to and including chapter `upto`."""
    known = {}
    for path in sorted((ARTIFACTS / "values").glob("*.json")):
        if upto is None or int(path.stem) <= upto:
            known.update(json.loads(path.read_text(encoding="utf-8")))
    return known


def render(text: str, known: dict) -> str:
    """Fill {{chN.key}} and {{chN.key:format_spec}} from `known`; an unknown key raises."""

    def fill(match: re.Match) -> str:
        key, spec = match.group(1), match.group(2) or ""
        if key not in known:
            raise KeyError(f"unknown placeholder {key}")
        return format(known[key], spec)

    return _PLACEHOLDER.sub(fill, text)


def _artifact(name: str, suffix: str) -> Path:
    path = ARTIFACTS / f"{name}{suffix}"
    if not path.exists():
        raise FileNotFoundError(
            f"artifacts/{path.name} is missing: run chapter {WRITTEN_BY[name]} first"
        )
    return path


def save_frame(name: str, frame: pd.DataFrame) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(ARTIFACTS / f"{name}.csv", index=False)


def load_frame(name: str) -> pd.DataFrame:
    return pd.read_csv(_artifact(name, ".csv"))


def save_json(name: str, obj: dict) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / f"{name}.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(name: str) -> dict:
    return json.loads(_artifact(name, ".json").read_text(encoding="utf-8"))


def write_results(metrics: dict, claims: dict[str, str]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, obj in (("metrics", metrics), ("claims", claims)):
        text = json.dumps(obj, indent=2, sort_keys=True) + "\n"
        (RESULTS / f"{name}.json").write_text(text, encoding="utf-8")
