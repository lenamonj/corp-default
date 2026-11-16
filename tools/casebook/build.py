"""Build a chapter: its source becomes an executed, filled, designed notebook.

    uv run python -m tools.casebook.build 03          # one chapter
    uv run python -m tools.casebook.build --all       # every chapter that has a source, in order

In order: the jupytext source is read into a notebook, executed from the repo root, its
placeholders are filled from the numbers the chapter stored, the design is applied, and the result
is written to notebooks/. The build is the only path to a notebook, so its outputs are always current.
"""

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import NamedTuple

import jupytext
import nbformat
from nbclient import NotebookClient

from corp_default import store
from corp_default.data import REPO, load_raw
from corp_default.labels import event_table
from tools.casebook.apply import apply
from tools.casebook.components import house_mark
from tools.casebook.studio import Studio

CHAPTERS = REPO / "chapters"
NOTEBOOKS = REPO / "notebooks"
KERNELSPEC = {"display_name": "corp-default (uv)", "language": "python", "name": "corp-default"}
CELL_TIMEOUT = 600  # seconds: no one cell may take a chapter's whole budget


class Built(NamedTuple):
    """One built notebook and the line the build prints for it."""

    path: Path
    parts: int
    exhibits: int
    minutes: int
    megabytes: float
    seconds: float

    def __str__(self) -> str:
        return (
            f"{self.path.stem}: {self.parts} parts, {self.exhibits} exhibits, "
            f"{self.minutes} minutes, {self.megabytes:.2f} MB, {self.seconds:.0f} s"
        )


def chapter_numbers(chapters: Path = CHAPTERS) -> list[int]:
    """The chapters that have a source, in order."""
    return sorted(int(path.name[:2]) for path in chapters.glob("[0-9][0-9]_*.py"))


def _filled(item, known: dict):
    """A plan with every placeholder in its words filled."""
    if isinstance(item, dict):
        return {key: _filled(value, known) for key, value in item.items()}
    if isinstance(item, list):
        return [_filled(value, known) for value in item]
    return store.render(item, known) if isinstance(item, str) else item


def build(
    number: int, mark: dict | None = None, chapters: Path = CHAPTERS, out: Path = NOTEBOOKS
) -> Built:
    """Build chapter `number` from `chapters` into `out`. `mark` is the house mark for its cover."""
    started = time.perf_counter()
    sources = sorted(chapters.glob(f"{number:02d}_*.py"))
    if len(sources) != 1:
        raise FileNotFoundError(
            f"{chapters} needs exactly one source named {number:02d}_*.py, and it has {len(sources)}"
        )
    source = sources[0]
    plan = json.loads((chapters / "plans" / f"{number:02d}.json").read_text(encoding="utf-8"))
    nb = jupytext.read(source, fmt="py:percent")
    # Timings are left out so that a rebuild of an unchanged chapter writes the same file.
    client = NotebookClient(
        nb,
        timeout=CELL_TIMEOUT,
        kernel_name="python3",
        record_timing=False,
        extra_arguments=["--log-level=ERROR"],  # the kernel's own start-up notices
        resources={"metadata": {"path": str(REPO)}},
    )
    with warnings.catch_warnings():
        # On Windows pyzmq reaches the kernel through a selector thread and says so on every run.
        warnings.filterwarnings("ignore", message="Proactor event loop does not implement")
        client.execute()
    known = store.values(upto=number)
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            cell.source = store.render(cell.source, known)
    if mark is None:
        mark = house_mark(event_table(load_raw()))
    with Studio() as studio:
        parts, exhibits, minutes = apply(nb, _filled(plan, known), mark, studio)
    # The language version is kept to its minor number, so a rebuild on another patch of the same
    # Python writes the same file.
    language = dict(nb.metadata["language_info"])
    language["version"] = ".".join(language["version"].split(".")[:2])
    nb.metadata = {"kernelspec": KERNELSPEC, "language_info": language}
    for place, cell in enumerate(nb.cells):
        cell.id = f"cell-{place:03d}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{source.stem}.ipynb"
    nbformat.write(nb, path)
    megabytes = path.stat().st_size / 1e6
    return Built(path, parts, exhibits, minutes, megabytes, time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build casebook chapters into notebooks/.")
    parser.add_argument("chapter", type=int, nargs="?", help="the chapter number, such as 03")
    parser.add_argument("--all", action="store_true", help="every chapter that has a source")
    args = parser.parse_args()
    if args.all == (args.chapter is not None):
        parser.error("give one chapter number, or --all")
    mark = house_mark(event_table(load_raw()))
    for number in chapter_numbers() if args.all else [args.chapter]:
        print(build(number, mark), flush=True)


if __name__ == "__main__":
    main()
