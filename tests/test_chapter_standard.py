"""Every built notebook against the chapter standard, rule by rule.

The fixture chapter is always checked, so the rules run even while notebooks/ is empty. Each rule
is also run on a broken copy of the fixture: a rule that cannot fail proves nothing when it passes.
"""

import re
from pathlib import Path

import nbformat
import pytest
from chapter_rules import MAX_BYTES, RULES, Chapter

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
BUILT = {path.stem: path for path in sorted(NOTEBOOKS.glob("*.ipynb"))}


@pytest.fixture(scope="module", params=["00_fixture", *BUILT])
def chapter(request) -> Chapter:
    if request.param == "00_fixture":
        return Chapter.read(request.getfixturevalue("built").path)
    return Chapter.read(BUILT[request.param])


@pytest.mark.parametrize("rule", list(RULES))
def test_chapter_meets_the_standard(chapter, rule):
    assert RULES[rule](chapter) == []


def _markdown(source: str):
    return nbformat.v4.new_markdown_cell(source)


def _replace(ch: Chapter, old: str, new: str) -> None:
    """Change words in the one cell that holds them."""
    hits = [cell for cell in ch.cells if old in cell.source]
    assert len(hits) == 1, f"{old!r} is in {len(hits)} cells"
    hits[0].source = hits[0].source.replace(old, new)


def _relabel_first_chart(ch: Chapter) -> None:
    chart = next(
        output
        for cell in ch.cells
        if cell.cell_type == "code"
        for output in cell.outputs
        if "image/png" in output.get("data", {})
    )
    chart.metadata["image/png"]["alt"] = "Exhibit 2: Out of order. A note."


def _shrink_first_chart(ch: Chapter) -> None:
    import base64
    import io

    from PIL import Image

    chart = next(
        output
        for cell in ch.cells
        if cell.cell_type == "code"
        for output in cell.outputs
        if "image/png" in output.get("data", {})
    )
    small = io.BytesIO()
    Image.new("RGB", (1183, 10)).save(small, format="PNG")
    chart.data["image/png"] = base64.b64encode(small.getvalue()).decode("ascii")


def _add_error(ch: Chapter) -> None:
    code = next(cell for cell in ch.cells if cell.cell_type == "code")
    code.outputs.append(
        nbformat.v4.new_output("error", ename="ValueError", evalue="broken", traceback=[])
    )


def _drop_pitfall_from_chapter_two(ch: Chapter) -> None:
    ch.number = 2
    ch.cells[:] = [cell for cell in ch.cells if not cell.source.startswith("![Pitfall: ")]


def _cover_says(ch: Chapter, minutes: int) -> None:
    """Rewrite the time on the cover."""
    cover = ch.cells[0]
    cover.source, changed = re.subn(
        r"About \d+ minutes\.", f"About {minutes} minutes.", cover.source
    )
    assert changed == 1


# One way to break the fixture chapter for each rule.
BREAKS = {
    "cover_first": lambda ch: ch.cells.pop(0),
    "title_cell": lambda ch: _replace(ch, "investment advice", "advice"),
    "front_matter": lambda ch: _replace(ch, "- how a pitfall becomes a plate.", ""),
    "route_drawn": lambda ch: _replace(ch, "**Time.** About", "About"),
    "parts_and_openers": lambda ch: ch.cells.pop(ch.parts()[1] - 1),
    "kernel_line": lambda ch: _replace(ch, "the Jupyter kernel `corp-default`", "the right kernel"),
    "short_paragraphs": lambda ch: ch.cells.insert(3, _markdown("One.\nTwo.\nThree.\nFour.")),
    "exhibits_numbered": _relabel_first_chart,
    "exhibits_cited_exist": lambda ch: _replace(ch, "(Exhibit 2)", "(Exhibit 9)"),
    "picture_widths": _shrink_first_chart,
    "predicts": lambda ch: _replace(ch, "**Predict 2.**", "**Predict.**"),
    "checks": lambda ch: _replace(ch, "**Check.** Why does the plate", "Why does the plate"),
    "pitfall_plate": _drop_pitfall_from_chapter_two,
    "findings": lambda ch: _replace(ch, " (Exhibit 1).", "."),
    "your_turn": lambda ch: _replace(ch, "**3. A log scale.**", "<summary>Hint</summary>"),
    "next_card_last": lambda ch: ch.cells.append(_markdown("One more thing.")),
    "no_placeholder_left": lambda ch: ch.cells.insert(3, _markdown("It was {{ch0.gone}}.")),
    "no_error_output": _add_error,
    "no_style_no_img": lambda ch: ch.cells.insert(3, _markdown('<p style="color:red">Red</p>')),
    "size": lambda ch: setattr(ch, "size", MAX_BYTES + 1),
    "reading_time": lambda ch: _cover_says(ch, 65),
}


def test_the_readme_has_the_section_the_kernel_line_links_to():
    readme = (NOTEBOOKS.parent / "README.md").read_text(encoding="utf-8")
    assert "\n## Quick Start\n" in readme


def test_every_rule_has_a_way_to_fail():
    assert set(BREAKS) == set(RULES)


@pytest.mark.parametrize("rule", list(RULES))
def test_a_rule_reports_a_broken_chapter(rule, built):
    broken = Chapter.read(built.path)
    BREAKS[rule](broken)
    assert RULES[rule](broken) != []


BARE_LOG = ('set_xscale("log")', 'set_yscale("log")')


def bare_log_scales(folder: Path) -> list[str]:
    """Chapter sources that set a log scale without the house style's readable ticks."""
    return [
        path.name
        for path in sorted(folder.glob("[0-9][0-9]_*.py"))
        if any(call in path.read_text(encoding="utf-8") for call in BARE_LOG)
    ]


def test_a_bare_log_scale_in_a_chapter_is_named(tmp_path):
    (tmp_path / "03_the_signal.py").write_text('ax.set_xscale("log")\n', encoding="utf-8")
    (tmp_path / "04_the_yardstick.py").write_text('viz.log_axis(ax, "x")\n', encoding="utf-8")
    assert bare_log_scales(tmp_path) == ["03_the_signal.py"]


def test_every_log_axis_in_a_chapter_goes_through_the_house_style():
    assert bare_log_scales(NOTEBOOKS.parent / "chapters") == []


def test_a_chapter_of_sixty_minutes_is_within_the_ceiling(built):
    ch = Chapter.read(built.path)
    _cover_says(ch, 60)
    assert RULES["reading_time"](ch) == []


def test_a_chapter_over_sixty_minutes_is_told_its_time_and_the_ceiling(built):
    ch = Chapter.read(built.path)
    _cover_says(ch, 65)
    assert RULES["reading_time"](ch) == ["the chapter takes about 65 minutes, and the most is 60"]


def test_a_cover_that_gives_no_time_is_named(built):
    ch = Chapter.read(built.path)
    ch.cells[0].source = re.sub(r" About \d+ minutes\.", "", ch.cells[0].source)
    assert RULES["reading_time"](ch) == ["the cover does not say how long the chapter takes"]
