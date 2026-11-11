"""The build, on the fixture chapter: a miniature chapter that needs no data and builds in seconds."""

import base64
import io
import json
import re

import jupytext
import nbformat
import pytest
from conftest import FIXTURES
from PIL import Image

PICTURE = re.compile(r"!\[([^\]]*)\]\(data:image/png;base64,([^)]+)\)")
PARTS = ["Setting Up", "A Number in the Prose", "Two Charts and a Plate"]
GOALS = [
    "Open the store and set the style.",
    "Carry a number into the prose.",
    "Draw two exhibits and a plate.",
]


@pytest.fixture(scope="module")
def nb(built):
    return nbformat.read(built.path, as_version=4)


def _markdown(nb) -> list[str]:
    return [cell.source for cell in nb.cells if cell.cell_type == "markdown"]


def _charts(nb) -> list:
    return [
        output
        for cell in nb.cells
        if cell.cell_type == "code"
        for output in cell.outputs
        if "image/png" in output.get("data", {})
    ]


def _width(encoded: str) -> int:
    return Image.open(io.BytesIO(base64.b64decode(encoded))).width


def test_the_first_cell_is_the_cover(nb):
    first = nb.cells[0]
    assert first.cell_type == "markdown"
    assert first.source.startswith(
        "![Chapter 0 of 7: The Fixture. Does the build carry every part across? About "
    )
    assert nb.cells[1].source.startswith("# Chapter 0: The Fixture")


def test_each_part_has_an_opener_directly_above_it(nb):
    for place, (title, goal) in enumerate(zip(PARTS, GOALS), start=1):
        at = next(i for i, cell in enumerate(nb.cells) if cell.source.startswith(f"# {title}"))
        above = nb.cells[at - 1]
        assert above.cell_type == "markdown"
        assert above.source.startswith(f"![Part {place} of 3. {goal} About "), title
        assert PICTURE.fullmatch(above.source), "an opener has a cell of its own"


def test_findings_and_your_turn_take_no_opener(nb):
    for heading in ("# Findings", "# Your Turn"):
        at = next(i for i, cell in enumerate(nb.cells) if cell.source.startswith(heading))
        assert not nb.cells[at - 1].source.startswith("![Part ")


def test_a_placeholder_is_filled_with_the_value_the_chapter_put(nb):
    text = "\n".join(_markdown(nb))
    assert "The values sum to 42." in text
    assert "{{" not in text


def test_an_unknown_placeholder_fails_the_build_with_its_key(tmp_path, toy_mark):
    from tools.casebook.build import build

    (tmp_path / "plans").mkdir()
    (tmp_path / "00_broken.py").write_text(
        "# %% [markdown]\n# The answer is {{ch0.missing}}.\n", encoding="utf-8"
    )
    (tmp_path / "plans" / "00.json").write_text("{}", encoding="utf-8")
    with pytest.raises(KeyError, match="ch0.missing"):
        build(0, mark=toy_mark, chapters=tmp_path, out=tmp_path / "out")


def test_a_pitfall_marker_becomes_a_picture_whose_alt_text_carries_both_numbers(nb):
    plates = [source for source in _markdown(nb) if source.startswith("![Pitfall: ")]
    assert len(plates) == 1
    alt = PICTURE.fullmatch(plates[0]).group(1)
    assert "Mean: 22.0." in alt and "Median: 3.0." in alt
    assert "<!-- pitfall" not in "\n".join(_markdown(nb))


def test_a_diagram_marker_becomes_its_picture(nb):
    plates = [source for source in _markdown(nb) if source.startswith("![Diagram: ")]
    assert len(plates) == 1
    assert "The label of one firm-year" in PICTURE.fullmatch(plates[0]).group(1)


def test_two_charts_are_numbered_exhibit_1_and_exhibit_2(nb):
    alts = [chart.metadata["image/png"]["alt"] for chart in _charts(nb)]
    first = "Exhibit 1: One value towers over the rest. The fifth observation sets the scale of the axis."
    second = "Exhibit 2: Four of five values sit below the mean. A symmetric-log axis keeps both signs readable."
    assert alts == [first, second]


def test_an_exhibit_is_1200_pixels_wide_in_the_notebook(nb):
    assert [_width(chart.data["image/png"]) for chart in _charts(nb)] == [1200, 1200]


def test_every_drawn_picture_is_2400_pixels_wide(nb):
    widths = [_width(data) for source in _markdown(nb) for _, data in PICTURE.findall(source)]
    assert len(widths) == 8  # cover, route, three openers, pitfall, diagram, next card
    assert set(widths) == {2400}


def test_the_route_sits_under_its_heading_with_a_line_of_time(nb):
    route = next(source for source in _markdown(nb) if source.startswith("## Your Route"))
    alt = PICTURE.search(route).group(1)
    assert alt.startswith("Your route through Chapter 0: 1 Setting Up, ")
    assert "3 Two Charts and a Plate, " in alt
    assert re.search(
        r"\*\*Time\.\*\* About \d+ minutes to read and run, then about 9 for Your Turn\.", route
    )


def test_the_last_cell_is_the_next_card_as_a_link(nb):
    last = nb.cells[-1].source
    assert last.startswith("[![Next: Chapter 1. The Panel. What exactly is in this file?](data:")
    assert last.endswith("](01_the_panel.ipynb)")


def test_the_written_notebook_has_no_style_attribute_and_no_img_tag(nb):
    for cell in nb.cells:
        assert not re.search(r"<img|<[a-z][^>]*\sstyle=", cell.source), cell.source[:80]


def test_the_notebook_is_written_for_the_corp_default_kernel_without_timings(nb):
    assert nb.metadata.kernelspec.name == "corp-default"
    assert "jupytext" not in nb.metadata
    assert all("execution" not in cell.metadata for cell in nb.cells)


def test_a_rebuild_of_an_unchanged_chapter_writes_the_same_file(built, tmp_path, toy_mark):
    from tools.casebook.build import build

    again = build(0, mark=toy_mark, chapters=FIXTURES, out=tmp_path)
    assert again.path.read_bytes() == built.path.read_bytes()


def test_no_cell_has_an_error_output(nb):
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            assert output.output_type != "error"


def test_the_build_line_gives_parts_exhibits_minutes_megabytes_seconds(built):
    assert (built.parts, built.exhibits) == (3, 2)
    # three parts of at least a minute each and nine minutes of Your Turn, rounded to five
    assert built.minutes >= 10 and built.minutes % 5 == 0
    assert 0 < built.megabytes < 3
    assert built.seconds > 0
    assert re.fullmatch(
        r"00_fixture: 3 parts, 2 exhibits, \d+ minutes, \d+\.\d\d MB, \d+ s", str(built)
    )


def test_a_plan_with_one_goal_too_few_fails_naming_the_count(toy_mark):
    from tools.casebook.apply import PlanError, apply
    from tools.casebook.studio import Studio

    plan = json.loads((FIXTURES / "plans" / "00.json").read_text(encoding="utf-8"))
    plan["goals"] = plan["goals"][:-1]
    nb = jupytext.read(FIXTURES / "00_fixture.py", fmt="py:percent")
    with Studio() as studio, pytest.raises(PlanError, match="2 goals and the notebook has 3 parts"):
        apply(nb, plan, toy_mark, studio)


def test_a_pitfall_the_plan_does_not_describe_stops_the_build(toy_mark):
    from tools.casebook.apply import PlanError, apply
    from tools.casebook.studio import Studio

    plan = json.loads((FIXTURES / "plans" / "00.json").read_text(encoding="utf-8"))
    plan["pitfalls"] = {}
    nb = jupytext.read(FIXTURES / "00_fixture.py", fmt="py:percent")
    with Studio() as studio, pytest.raises(PlanError, match="skew"):
        apply(nb, plan, toy_mark, studio)


def test_a_title_the_plan_and_the_notebook_disagree_on_stops_the_build(toy_mark):
    from tools.casebook.apply import PlanError, apply
    from tools.casebook.studio import Studio

    plan = json.loads((FIXTURES / "plans" / "00.json").read_text(encoding="utf-8"))
    plan["title"] = "The Other"
    nb = jupytext.read(FIXTURES / "00_fixture.py", fmt="py:percent")
    with Studio() as studio, pytest.raises(PlanError, match="# Chapter 0: The Other"):
        apply(nb, plan, toy_mark, studio)


def test_a_plan_key_the_build_does_not_know_stops_it(toy_mark):
    from tools.casebook.apply import PlanError, apply
    from tools.casebook.studio import Studio

    plan = json.loads((FIXTURES / "plans" / "00.json").read_text(encoding="utf-8"))
    plan["subtitle"] = "Not a key"
    nb = jupytext.read(FIXTURES / "00_fixture.py", fmt="py:percent")
    with Studio() as studio, pytest.raises(PlanError, match="subtitle"):
        apply(nb, plan, toy_mark, studio)


def _cell(kind: str, source: str, alts: tuple[str, ...] = ()):
    if kind == "markdown":
        return nbformat.v4.new_markdown_cell(source)
    outputs = [
        nbformat.v4.new_output(
            "display_data", data={"image/png": ""}, metadata={"image/png": {"alt": alt}}
        )
        for alt in alts
    ]
    return nbformat.v4.new_code_cell(source, outputs=outputs)


def test_the_time_rule_counts_words_code_cells_and_exhibits():
    from tools.casebook.apply import minutes

    cells = [
        _cell("markdown", "word " * 300),  # 300 words at 150 a minute: 2 minutes
        _cell("code", "x = 1"),  # half a minute
        _cell(
            "code", "chart()", alts=("Exhibit 1: A. B.",)
        ),  # half a minute and one for the exhibit
    ]
    assert minutes(cells) == pytest.approx(2 + 0.5 + 0.5 + 1)


def test_the_time_rule_reads_a_picture_by_its_alt_text_and_skips_markup():
    from tools.casebook.apply import minutes

    picture = "![Five words of alt text](data:image/png;base64,AAAA)"
    folded = "<details>\n<summary>Hint</summary>\n\nTwo words\n\n</details>"
    assert minutes([_cell("markdown", picture)]) == pytest.approx(5 / 150)
    assert minutes([_cell("markdown", folded)]) == pytest.approx(3 / 150)
    assert minutes([_cell("markdown", "<!-- pitfall: skew -->")]) == 0


def test_chapter_seven_links_its_next_card_to_the_landing_page():
    from tools.casebook.apply import next_link

    assert next_link(3, "The Yardstick") == ("CHAPTER 4", "04_the_yardstick.ipynb")
    assert next_link(7, "The Casebook") == ("THE LANDING PAGE", "../README.md")


def test_all_builds_the_chapters_that_exist_in_order(tmp_path):
    from tools.casebook.build import chapter_numbers

    for name in ("02_the_event.py", "01_the_panel.py", "diagrams.py"):
        (tmp_path / name).write_text("", encoding="utf-8")
    assert chapter_numbers(tmp_path) == [1, 2]


def test_check_finds_every_picture_and_exhibit_in_reading_order(built):
    from tools.casebook.check import pictures

    found = pictures(built.path)
    starts = [alt.split(" ")[0].rstrip(":.") for alt, _ in found]
    assert starts == [
        "Chapter",
        "Your",
        "Part",
        "Part",
        "Part",
        "Exhibit",
        "Pitfall",
        "Exhibit",
        "Diagram",
        "Next",
    ]


def test_check_saves_sheets_1200_pixels_wide(built, tmp_path):
    from tools.casebook.check import pictures, save_sheets

    sheets = save_sheets(pictures(built.path), tmp_path, "00_fixture")
    assert sheets and all(path.name.startswith("00_fixture_sheet") for path in sheets)
    assert {Image.open(path).width for path in sheets} == {1200}
    assert all(Image.open(path).height <= 2400 for path in sheets)


def test_check_lists_each_part_with_its_goal_and_minutes(built):
    from tools.casebook.check import parts

    listed = parts(built.path)
    assert len(listed) == 3
    assert listed[0].startswith("Part 1 of 3. Open the store and set the style. About ")


def test_check_text_shows_a_picture_as_its_alt_text(built):
    from tools.casebook.check import as_text

    text = as_text(built.path)
    assert "[picture: Part 1 of 3. Open the store and set the style." in text
    assert "[exhibit: Exhibit 1: One value towers over the rest." in text
    assert "The values sum to 42." in text
    assert "base64" not in text
