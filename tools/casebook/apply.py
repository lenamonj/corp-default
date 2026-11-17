"""Put an executed chapter into the casebook design, from the notebook itself and a small plan.

build.py calls apply() on every build, so the design is drawn fresh each time and no chapter source
holds a picture. What the notebook already says is read from it: its parts, their length, its
exhibits. The plan, chapters/plans/NN.json, holds only what a person has to write: the title and
question for the cover, one goal for each part, the words on each pitfall plate, and the next card.

Anything the plan and the notebook disagree on stops the build with a sentence that says what to fix.
"""

import base64
import html
import re
import tempfile
from pathlib import Path

import nbformat

from chapters import diagrams
from tools.casebook import components as c
from tools.casebook.studio import Studio

# The time rule of the chapter standard.
WORDS_A_MINUTE = 150
MINUTES_A_CODE_CELL = 0.5
MINUTES_AN_EXHIBIT = 1
MINUTES_AN_EXTENSION = 3

MAX_PARTS = 12
REQUIRED = {"number", "title", "question", "goals", "next"}
KEYS = REQUIRED | {"pitfalls"}
MARKER = re.compile(r"<!-- (pitfall|diagram): (\w+) -->")
HINT = "<summary>Hint</summary>"  # each extension of Your Turn folds one hint
_NOT_WORDS = re.compile(r"\(data:image[^)]*\)|<[^>]+>")  # picture data and markup are not read
_WORD = re.compile(r"\S*\w\S*")


class PlanError(ValueError):
    """The plan and the notebook disagree. The message says what to change."""


def minutes(cells: list) -> float:
    """About how long a run of cells takes: its words at reading pace, each code cell, each exhibit."""
    words = sum(
        len(_WORD.findall(_NOT_WORDS.sub(" ", cell.source)))
        for cell in cells
        if cell.cell_type == "markdown"
    )
    code = [cell for cell in cells if cell.cell_type == "code"]
    exhibits = sum(
        "image/png" in output.get("data", {}) for cell in code for output in cell.outputs
    )
    return words / WORDS_A_MINUTE + len(code) * MINUTES_A_CODE_CELL + exhibits * MINUTES_AN_EXHIBIT


def next_link(number: int, title: str) -> tuple[str, str]:
    """The plate on the next card and where the card leads: the next chapter, or the landing page."""
    if number == c.CHAPTERS:
        return "THE LANDING PAGE", "../README.md"
    return f"CHAPTER {number + 1}", f"{number + 1:02d}_{title.lower().replace(' ', '_')}.ipynb"


def _heading(cell) -> str:
    return cell.source.split("\n")[0].strip()


def _find(cells: list, heading: str) -> int:
    """The index of the one markdown cell whose first line is the heading."""
    hits = [
        i
        for i, cell in enumerate(cells)
        if cell.cell_type == "markdown" and _heading(cell) == heading
    ]
    if len(hits) != 1:
        raise PlanError(
            f"the notebook needs exactly one markdown cell starting {heading!r}, and it has {len(hits)}"
        )
    return hits[0]


def _about(count: int) -> str:
    return f"About {count} minute{'' if count == 1 else 's'}."


def diagram_alt(drawn: str) -> str:
    """What a diagram says: its own alt text where it carries one, and its caption otherwise."""
    own = re.search(r'data-alt="([^"]*)"', drawn)
    if own:
        return html.unescape(own.group(1))
    return f"Diagram: {html.unescape(re.search(r'class="cb-cap">([^<]*)<', drawn).group(1))}"


def apply(nb, plan: dict, mark: dict, studio: Studio) -> tuple[int, int, int]:
    """Put the notebook's cells into the design, in place. Returns its parts, exhibits, and minutes."""
    unknown, missing = set(plan) - KEYS, REQUIRED - set(plan)
    if unknown or missing:
        raise PlanError(
            f"the plan has unknown keys {sorted(unknown)} and lacks {sorted(missing)}: "
            f"it takes {sorted(KEYS)}"
        )
    cells = nb.cells
    number, title, question = plan["number"], plan["title"], plan["question"]
    _find(cells, f"# Chapter {number}: {title}")
    route_at, findings_at, turn_at = (
        _find(cells, heading) for heading in ("## Your Route", "# Findings", "# Your Turn")
    )
    starts = [
        i
        for i in range(route_at + 1, findings_at)
        if cells[i].cell_type == "markdown" and cells[i].source.startswith("# ")
    ]
    names = [_heading(cells[i])[2:] for i in starts]
    goals = plan["goals"]
    if len(goals) != len(starts):
        raise PlanError(
            f"the plan gives {len(goals)} goals and the notebook has {len(starts)} parts: {names}"
        )
    if len(starts) > MAX_PARTS:
        raise PlanError(f"a chapter has at most {MAX_PARTS} parts, and this one has {len(starts)}")
    marked = {}
    for i, cell in enumerate(cells):
        found = MARKER.fullmatch(cell.source.strip()) if cell.cell_type == "markdown" else None
        if found:
            marked[i] = found.groups()
    pitfalls = plan.get("pitfalls", {})
    plates = {key for kind, key in marked.values() if kind == "pitfall"}
    if plates != set(pitfalls):
        raise PlanError(
            f"the notebook marks the pitfalls {sorted(plates)} and the plan describes {sorted(pitfalls)}"
        )
    for kind, key in marked.values():
        if kind == "diagram" and not hasattr(diagrams, key):
            raise PlanError(f"chapters/diagrams.py has no diagram named {key}")

    # The time rule, on the cells as written and executed, before any picture goes in.
    part_minutes = [
        max(1, round(minutes(cells[a:b]))) for a, b in zip(starts, [*starts[1:], findings_at])
    ]
    around = minutes(cells[: starts[0]]) + minutes(cells[findings_at:turn_at])
    lesson = sum(part_minutes) + round(around)
    turn = MINUTES_AN_EXTENSION * sum(cell.source.count(HINT) for cell in cells[turn_at:])
    total = 5 * round((lesson + turn) / 5)
    exhibits = sum(
        "image/png" in output.get("data", {})
        for cell in cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )

    with tempfile.TemporaryDirectory() as folder:

        def picture(drawn: str, alt: str) -> str:
            png = studio.png(drawn, Path(folder) / "picture.png").read_bytes()
            return f"![{alt}](data:image/png;base64,{base64.b64encode(png).decode('ascii')})"

        def plate(kind: str, key: str) -> str:
            if kind == "diagram":
                drawn = getattr(diagrams, key)()
                return picture(drawn, diagram_alt(drawn))
            item = pitfalls[key]
            drawn = c.pitfall(
                item["name"],
                item["naive_label"].upper(),
                item["naive_value"],
                item["honest_label"].upper(),
                item["honest_value"],
                item["why"],
            )
            return picture(
                drawn,
                f"Pitfall: {item['name']}. {item['naive_label']}: {item['naive_value']}. "
                f"{item['honest_label']}: {item['honest_value']}. {item['why']}",
            )

        # The fixture chapter is numbered 0; it takes the first chapter's place on the mark.
        cover = picture(
            c.cover(number, title, question, total, c.mark_curve(mark, max(number, 1))),
            f"Chapter {number} of {c.CHAPTERS}: {title}. {question} About {total} minutes.",
        )
        stops = "; ".join(
            f"{k} {name}, {_about(spent)[6:-1]}"
            for k, (name, spent) in enumerate(zip(names, part_minutes), start=1)
        )
        route = picture(
            c.route(list(zip(names, part_minutes)), f"ABOUT {total} MIN"),
            f"Your route through Chapter {number}: {stops}.",
        )
        designed = [nbformat.v4.new_markdown_cell(cover)]
        for i, cell in enumerate(cells):
            if i in starts:
                k = starts.index(i)
                opener = picture(
                    c.opener(k + 1, len(starts), goals[k], part_minutes[k]),
                    f"Part {k + 1} of {len(starts)}. {goals[k]} {_about(part_minutes[k])}",
                )
                designed.append(nbformat.v4.new_markdown_cell(opener))
            if i == route_at:
                cell.source = (
                    f"{cell.source.rstrip()}\n\n{route}\n\n**Time.** About {lesson} minutes to read "
                    f"and run, then about {turn} for Your Turn."
                )
            if i in marked:
                cell.source = plate(*marked[i])
            designed.append(cell)
        label, link = next_link(number, plan["next"]["title"])
        card = picture(
            c.next_card("NEXT", label, plan["next"]["title"], plan["next"]["line"]),
            f"Next: {label.capitalize()}. {plan['next']['title']}. {plan['next']['line']}",
        )
        designed.append(nbformat.v4.new_markdown_cell(f"[{card}]({link})"))
    nb.cells = designed
    return len(starts), exhibits, total
