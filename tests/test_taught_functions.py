"""A function a chapter teaches is typed in full on screen and stays identical to the library.

A code cell whose first line is `# taught: corp_default.<module>.<name>` must define that function
or class with the same syntax tree as the library's. The list below is the spec's blueprints: each
name is taught exactly once, in its chapter, as soon as that chapter's source exists.
"""

import ast
import inspect
import re
from pathlib import Path

import jupytext
import pytest

REPO = Path(__file__).resolve().parents[1]
CHAPTERS = REPO / "chapters"
LIBRARY = REPO / "src" / "corp_default"
MARK = re.compile(r"# taught: corp_default\.(\w+)\.(\w+)")
SOURCES = {int(path.name[:2]): path for path in sorted(CHAPTERS.glob("[0-9][0-9]_*.py"))}

TAUGHT = {
    1: ["data.load_raw", "data.load_industry", "data.panel_spans"],
    2: [
        "labels.build_label",
        "labels.event_table",
        "survival.kaplan_meier",
        "survival.cumulative_incidence",
    ],
    3: ["features.add_ratios", "metrics.wilson_interval", "features.add_history", "metrics.psi"],
    4: [
        "splits.development",
        "splits.walk_forward_folds",
        "features.fit_prepare",
        "metrics.capture_rate",
        "metrics.capture_curve",
        "metrics.cluster_bootstrap_ci",
        "metrics.paired_bootstrap_diff_ci",
        "classic.altman_z",
    ],
    5: ["evaluate.walk_forward_oof"],
    6: [
        "nets.seed_everything",
        "nets.TabularNet",
        "nets.train_net",
        "nets.build_sequences",
        "nets.SequenceHazardNet",
        "nets.cumulative_pd",
    ],
    7: ["metrics.reliability_table"],
}


def taught_cells(source: Path) -> list[tuple[str, str]]:
    """Each taught cell of a chapter source, as ("module.name", the cell's code)."""
    found = []
    for cell in jupytext.read(source, fmt="py:percent").cells:
        mark = MARK.fullmatch(cell.source.split("\n")[0].strip())
        if cell.cell_type == "code" and mark:
            found.append((f"{mark.group(1)}.{mark.group(2)}", cell.source))
    return found


def definition(code: str, name: str) -> str | None:
    """The syntax tree of the function or class `name` defined at the top level of `code`."""
    for node in ast.parse(code).body:
        if isinstance(node, ast.FunctionDef | ast.ClassDef) and node.name == name:
            return ast.dump(node)
    return None


def same_as_library(taught: str, code: str) -> bool:
    module, name = taught.split(".")
    library = definition((LIBRARY / f"{module}.py").read_text(encoding="utf-8"), name)
    return library is not None and definition(code, name) == library


def _chapter(tmp_path: Path, cell: str) -> Path:
    source = tmp_path / "01_test.py"
    source.write_text(f"# %% [markdown]\n# Words.\n\n# %%\n{cell}", encoding="utf-8")
    return source


def test_a_verbatim_copy_matches_and_comments_do_not_matter(tmp_path):
    from corp_default.metrics import wilson_interval

    copy = inspect.getsource(wilson_interval).replace("\n", "\n\n", 1) + "\n# a comment after\n"
    cells = taught_cells(
        _chapter(tmp_path, f"# taught: corp_default.metrics.wilson_interval\n{copy}")
    )
    assert [taught for taught, _ in cells] == ["metrics.wilson_interval"]
    assert same_as_library(*cells[0])


def test_a_changed_copy_does_not_match(tmp_path):
    from corp_default.metrics import wilson_interval

    changed = inspect.getsource(wilson_interval).replace("1.96", "1.64")
    assert changed != inspect.getsource(wilson_interval)
    cells = taught_cells(
        _chapter(tmp_path, f"# taught: corp_default.metrics.wilson_interval\n{changed}")
    )
    assert not same_as_library(*cells[0])


def test_a_cell_that_names_a_function_and_does_not_define_it_does_not_match(tmp_path):
    cells = taught_cells(_chapter(tmp_path, "# taught: corp_default.metrics.psi\nx = 1\n"))
    assert not same_as_library(*cells[0])


@pytest.mark.parametrize("number", list(SOURCES))
def test_every_taught_cell_is_identical_to_the_library(number):
    for taught, code in taught_cells(SOURCES[number]):
        assert same_as_library(taught, code), f"{taught} differs from src/corp_default"


@pytest.mark.parametrize("number", list(SOURCES))
def test_a_chapter_teaches_each_of_its_functions_exactly_once(number):
    taught = [name for name, _ in taught_cells(SOURCES[number])]
    assert sorted(taught) == sorted(TAUGHT[number])
