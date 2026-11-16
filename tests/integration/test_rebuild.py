"""A rebuild of every chapter writes the same files, and stays inside its budgets.

    uv run pytest -q -m integration tests/integration/test_rebuild.py

This builds all seven chapters from their sources, which takes about twenty-five minutes and needs
the data. The built notebooks and the results are committed, so "the same files" is checked against
what is on disk before the build starts.
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
BUILD_LINE = re.compile(
    r"^(\d\d_\w+): (\d+) parts, (\d+) exhibits, (\d+) minutes, ([\d.]+) MB, (\d+) s$", re.MULTILINE
)
MAX_MB = 3.0
MAX_MINUTES_TO_READ = 60
MAX_SECONDS_FOR_ALL = 45 * 60
ALLOWED_OVER_TEN_MINUTES = {"05_the_trees"}  # Jeff, 2026-11-11: training takes the time it takes


def kept_files() -> list[Path]:
    return [
        *sorted((REPO / "notebooks").glob("*.ipynb")),
        *sorted((REPO / "results").glob("*.json")),
    ]


def digests() -> dict[str, str]:
    return {
        path.relative_to(REPO).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in kept_files()
    }


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    if not (REPO / "data" / "american_bankruptcy.csv").exists():
        pytest.skip("data/american_bankruptcy.csv is not present")
    before = digests()
    metrics_before = json.loads((REPO / "results" / "metrics.json").read_text(encoding="utf-8"))
    run = subprocess.run(
        [sys.executable, "-m", "tools.casebook.build", "--all"],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,  # the assertion below reports the build's own output
    )
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    lines = [
        {"chapter": c, "parts": int(p), "exhibits": int(e), "minutes": int(m), "mb": float(mb), "seconds": int(s)}
        for c, p, e, m, mb, s in BUILD_LINE.findall(run.stdout)
    ]  # fmt: skip
    return {"before": before, "after": digests(), "metrics_before": metrics_before, "lines": lines}


def test_every_chapter_was_built(rebuilt):
    assert len(rebuilt["lines"]) == 7


def test_a_rebuild_writes_the_same_notebooks_and_results(rebuilt):
    changed = [
        name for name, digest in rebuilt["after"].items() if rebuilt["before"].get(name) != digest
    ]
    assert changed == []


def test_the_metrics_agree_to_the_stated_tolerance(rebuilt):
    after = json.loads((REPO / "results" / "metrics.json").read_text(encoding="utf-8"))
    for table, rows in rebuilt["metrics_before"].items():
        for old, new in zip(rows, after[table], strict=True):
            for key, value in old.items():
                if isinstance(value, float):
                    assert new[key] == pytest.approx(value, abs=1e-9), (table, key)
                else:
                    assert new[key] == value, (table, key)


def test_every_chapter_is_inside_its_budgets(rebuilt):
    over = [
        line
        for line in rebuilt["lines"]
        if line["mb"] > MAX_MB
        or line["minutes"] > MAX_MINUTES_TO_READ
        or (line["seconds"] > 600 and line["chapter"] not in ALLOWED_OVER_TEN_MINUTES)
    ]
    assert over == []
    assert sum(line["seconds"] for line in rebuilt["lines"]) <= MAX_SECONDS_FOR_ALL
