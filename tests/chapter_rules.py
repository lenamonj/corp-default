"""The chapter standard (spec section 7) as rules a built notebook is checked against.

Each rule takes a Chapter and returns the faults it finds, as sentences; an empty list is a pass.
test_chapter_standard.py runs every rule on every built notebook, and on broken copies of the
fixture chapter, so each rule is seen to fail before it is trusted to pass.
"""

import base64
import io
import re
from dataclasses import dataclass
from pathlib import Path

import nbformat
from PIL import Image

# One sentence to a quoted line, each a block of its own.
DISCLAIMER = (
    "This casebook is educational content created in a personal capacity",
    "Nothing here is investment advice or a recommendation to buy or sell any security",
    "The data is a public research dataset with anonymised company names",
)
MAX_PARAGRAPH = 2  # sentences, written one to a line
NOT_PROSE = re.compile(r"\||- |#|>|<|!\[|\[!\[|\d+\. ")  # tables, lists, headings, quotes, markup
PICTURE = re.compile(r"!\[([^\]]*)\]\(data:image/png;base64,([^)]+)\)")
NEXT_CARD = re.compile(r"\[!\[Next: [^\]]*\]\(data:image/png;base64,[^)]+\)\]\(([^)]+)\)")
FRONT = ["## The Question", "## What You Will Learn", "## What You Need", "## Your Route"]
CHAPTERS = 7
MAX_PARTS = 12
MAX_BYTES = 3_000_000
SHOW_ANSWER = "<summary>Show the answer</summary>"
KERNEL = "Jupyter kernel `corp-default`"
QUICK_START = "(../README.md#quick-start)"  # where the README says how to register the kernel
HINT = "<summary>Hint</summary>"


@dataclass
class Chapter:
    """A built notebook: its number, its cells, and its size on disk."""

    number: int
    cells: list
    size: int

    @classmethod
    def read(cls, path: Path) -> "Chapter":
        cells = nbformat.read(path, as_version=4).cells
        return cls(int(path.name[:2]), cells, path.stat().st_size)

    def find(self, heading: str) -> list[int]:
        """The markdown cells whose first line is the heading."""
        return [
            i
            for i, cell in enumerate(self.cells)
            if cell.cell_type == "markdown" and cell.source.split("\n")[0].strip() == heading
        ]

    def prose(self) -> str:
        """Every typed word: the markdown cells, without their pictures."""
        typed = [cell.source for cell in self.cells if cell.cell_type == "markdown"]
        return PICTURE.sub("", "\n\n".join(typed))

    def drawn(self) -> list[tuple[str, str]]:
        """Every drawn picture, in order, as (alt text, encoded PNG)."""
        return [
            found
            for cell in self.cells
            if cell.cell_type == "markdown"
            for found in PICTURE.findall(cell.source)
        ]

    def exhibits(self) -> list[tuple[str, str]]:
        """Every chart a code cell shows, in order, as (alt text, encoded PNG)."""
        return [
            (
                output.get("metadata", {}).get("image/png", {}).get("alt", ""),
                output.data["image/png"],
            )
            for cell in self.cells
            if cell.cell_type == "code"
            for output in cell.outputs
            if "image/png" in output.get("data", {})
        ]

    def parts(self) -> list[int]:
        """The part headings: every '# ' cell between the route and the findings."""
        route, findings = self.find("## Your Route"), self.find("# Findings")
        if len(route) != 1 or len(findings) != 1:
            return []
        return [
            i
            for i in range(route[0] + 1, findings[0])
            if self.cells[i].cell_type == "markdown" and self.cells[i].source.startswith("# ")
        ]


def _width(encoded: str) -> int:
    return Image.open(io.BytesIO(base64.b64decode(encoded))).width


def cover_first(ch: Chapter) -> list[str]:
    first = ch.cells[0]
    found = PICTURE.fullmatch(first.source) if first.cell_type == "markdown" else None
    if found and found.group(1).startswith(f"Chapter {ch.number} of {CHAPTERS}: "):
        return []
    return ["the first cell is not the cover of this chapter"]


def title_cell(ch: Chapter) -> list[str]:
    lines = [line for line in ch.cells[1].source.split("\n") if line.strip()]
    faults = []
    if not lines[0].startswith(f"# Chapter {ch.number}: "):
        faults.append(f"the second cell does not start '# Chapter {ch.number}: Title'")
    if len(lines) < 3 or lines[1].startswith(">"):
        faults.append("the title cell has no one-line purpose under its heading")
    quoted = "\n\n".join(f"> {line}" for line in DISCLAIMER)
    if quoted not in ch.cells[1].source:
        faults.append("the title cell lacks the disclaimer, as three quoted lines")
    return faults


def front_matter(ch: Chapter) -> list[str]:
    where = [ch.find(heading) for heading in FRONT]
    faults = [
        f"{heading} starts {len(hits)} cells, not one"
        for heading, hits in zip(FRONT, where)
        if len(hits) != 1
    ]
    if faults:
        return faults
    order = [hits[0] for hits in where]
    if order != sorted(order):
        faults.append(f"the opening sections are out of order: {FRONT}")
    learn = ch.cells[where[1][0]].source
    objectives = sum(line.startswith("- ") for line in learn.split("\n"))
    if not 3 <= objectives <= 5:
        faults.append(f"What You Will Learn lists {objectives} objectives, not three to five")
    return faults


def route_drawn(ch: Chapter) -> list[str]:
    at = ch.find("## Your Route")
    source = ch.cells[at[0]].source if len(at) == 1 else ""
    found = PICTURE.search(source)
    faults = []
    if not found or not found.group(1).startswith(f"Your route through Chapter {ch.number}: "):
        faults.append("the route is not drawn under ## Your Route")
    if not re.search(r"^\*\*Time\.\*\* About \d+ minutes", source, re.MULTILINE):
        faults.append("the route has no line of time")
    return faults


def parts_and_openers(ch: Chapter) -> list[str]:
    parts = ch.parts()
    if not parts:
        return ["the chapter has no parts between its route and its findings"]
    faults = []
    if len(parts) > MAX_PARTS:
        faults.append(f"the chapter has {len(parts)} parts, and the most is {MAX_PARTS}")
    if ch.cells[parts[0]].source.split("\n")[0].strip() != "# Setting Up":
        faults.append("the first part is not # Setting Up")
    for place, at in enumerate(parts, start=1):
        above = ch.cells[at - 1]
        found = PICTURE.fullmatch(above.source) if above.cell_type == "markdown" else None
        if not found or not found.group(1).startswith(f"Part {place} of {len(parts)}. "):
            faults.append(f"part {place} has no opener of its own directly above it")
    return faults


def kernel_line(ch: Chapter) -> list[str]:
    at = ch.find("# Setting Up")
    lines = [line for line in ch.cells[at[0]].source.split("\n") if line.strip()] if at else []
    if len(lines) > 2 and KERNEL in lines[1] and QUICK_START in lines[2]:
        return []
    return ["Setting Up does not open with the kernel line and its link to the Quick Start"]


def short_paragraphs(ch: Chapter) -> list[str]:
    """A paragraph is one or two sentences. Prose is written one sentence to a line, so a
    paragraph is a run of prose lines with no blank line between them."""
    faults = []
    for place, cell in enumerate(ch.cells):
        if cell.cell_type != "markdown":
            continue
        run = []
        for line in [*cell.source.split("\n"), ""]:
            if line.strip() and not NOT_PROSE.match(line.strip()):
                run.append(line)
                continue
            if len(run) > MAX_PARAGRAPH:
                faults.append(
                    f"cell {place} runs {len(run)} sentences together, starting {run[0][:50]!r}"
                )
            run = []
    return faults


def exhibits_numbered(ch: Chapter) -> list[str]:
    return [
        f"chart {place} is labelled {alt[:40]!r}: every chart is an exhibit, numbered in order"
        for place, (alt, _) in enumerate(ch.exhibits(), start=1)
        if not alt.startswith(f"Exhibit {place}: ")
    ]


def exhibits_cited_exist(ch: Chapter) -> list[str]:
    count = len(ch.exhibits())
    cited = {int(number) for number in re.findall(r"Exhibit (\d+)", ch.prose())}
    return [
        f"the text cites Exhibit {n}, and there are {count}" for n in sorted(cited) if n > count
    ]


def picture_widths(ch: Chapter) -> list[str]:
    faults = [
        f"the drawn picture {alt[:40]!r} is not 2400 px wide"
        for alt, data in ch.drawn()
        if _width(data) != 2400
    ]
    faults += [
        f"{alt[:40]!r} is not 1200 px wide" for alt, data in ch.exhibits() if _width(data) != 1200
    ]
    return faults


def predicts(ch: Chapter) -> list[str]:
    numbers = [int(n) for n in re.findall(r"\*\*Predict (\d+)\.\*\*", ch.prose())]
    if len(numbers) < 2:
        return [f"the chapter has {len(numbers)} **Predict N.** prompts, and it needs at least two"]
    if numbers != list(range(1, len(numbers) + 1)):
        return [f"the Predict prompts are numbered {numbers}, not from 1 in order"]
    return []


def checks(ch: Chapter) -> list[str]:
    prose = ch.prose()
    asked, folded = prose.count("**Check.**"), prose.count(SHOW_ANSWER)
    if asked < 2:
        return [f"the chapter has {asked} **Check.** questions, and it needs at least two"]
    if folded < asked:
        return [f"{asked} checks and only {folded} answers folded under 'Show the answer'"]
    return []


def pitfall_plate(ch: Chapter) -> list[str]:
    if ch.number < 2 or any(alt.startswith("Pitfall: ") for alt, _ in ch.drawn()):
        return []
    return ["from Chapter 2 on, a chapter has at least one pitfall plate"]


def findings(ch: Chapter) -> list[str]:
    at = ch.find("# Findings")
    if len(at) != 1:
        return [f"# Findings starts {len(at)} cells, not one"]
    items = re.findall(r"^(\d+)\. (.+)$", ch.cells[at[0]].source, re.MULTILINE)
    if [int(number) for number, _ in items] != list(range(1, len(items) + 1)) or not items:
        return ["the findings are not a list numbered from 1"]
    return [
        f"finding {number} cites no exhibit"
        for number, words in items
        if not re.search(r"Exhibit \d+", words)
    ]


def your_turn(ch: Chapter) -> list[str]:
    at, before = ch.find("# Your Turn"), ch.find("# Findings")
    if len(at) != 1 or len(before) != 1 or at[0] < before[0]:
        return ["# Your Turn must start one cell, after # Findings"]
    hints = sum(cell.source.count(HINT) for cell in ch.cells[at[0] :])
    return [] if hints == 3 else [f"Your Turn folds {hints} hints, and it has three extensions"]


def next_card_last(ch: Chapter) -> list[str]:
    found = NEXT_CARD.fullmatch(ch.cells[-1].source)
    if not found:
        return ["the last cell is not the next card, as a link"]
    link = found.group(1)
    if ch.number == CHAPTERS:
        return (
            [] if link == "../README.md" else ["the last chapter's card must link to ../README.md"]
        )
    if re.fullmatch(rf"{ch.number + 1:02d}_\w+\.ipynb", link):
        return []
    return [f"the next card links to {link}, not to chapter {ch.number + 1}'s notebook"]


def no_placeholder_left(ch: Chapter) -> list[str]:
    left = [
        cell.source[:60]
        for cell in ch.cells
        if cell.cell_type == "markdown" and "{{" in cell.source
    ]
    return [f"an unfilled placeholder in {source!r}" for source in left]


def no_error_output(ch: Chapter) -> list[str]:
    return [
        f"cell {place} ended in an error"
        for place, cell in enumerate(ch.cells)
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]


def no_style_no_img(ch: Chapter) -> list[str]:
    return [
        f"cell {place} carries an <img> tag or a style attribute"
        for place, cell in enumerate(ch.cells)
        if re.search(r"<img|<[a-z][^>]*\sstyle=", cell.source)
    ]


def size(ch: Chapter) -> list[str]:
    if ch.size <= MAX_BYTES:
        return []
    return [f"the notebook is {ch.size / 1e6:.2f} MB, and the most is {MAX_BYTES / 1e6:.0f} MB"]


RULES = {
    rule.__name__: rule
    for rule in (
        cover_first,
        title_cell,
        front_matter,
        route_drawn,
        parts_and_openers,
        kernel_line,
        short_paragraphs,
        exhibits_numbered,
        exhibits_cited_exist,
        picture_widths,
        predicts,
        checks,
        pitfall_plate,
        findings,
        your_turn,
        next_card_last,
        no_placeholder_left,
        no_error_output,
        no_style_no_img,
        size,
    )
}
