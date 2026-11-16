"""The landing page: the pictures of README.md and the blocks of it that carry numbers.

    uv run python -m tools.casebook.landing

GitHub shows a README in a column 830 px wide and strips every style, so the page keeps the
casebook's rule: words are typed, structure is drawn. This file draws the structure into
assets/readme/landing/ and fills every block of README.md that sits between a pair of markers,
`<!-- name:start -->` and `<!-- name:end -->`. Those blocks hold every number on the page.

Drawing needs the chapters to have been built, since the panel's counts come from their stored
values and the house mark from Chapter 2's event table. The counts are saved to
results/landing.json, so the blocks can be filled, and tested, from results/ alone.
"""

import json
from html import escape
from pathlib import Path

from PIL import Image

from corp_default import store
from corp_default.labels import PANEL_END
from corp_default.splits import HOLDOUT
from tools.casebook import components as c
from tools.casebook.studio import LANDING_WIDTH, Studio

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "assets" / "readme" / "landing"
GALLERY = REPO / "assets" / "readme" / "gallery"
RESULTS = REPO / "results"
PLANS = REPO / "chapters" / "plans"
STYLE = f'<link rel="stylesheet" href="{Path(__file__).with_name("landing.css").as_uri()}">'
TITLE = c.SERIES
LINE = "A casebook on predicting bankruptcy in US public companies"
SOCIAL_WIDTH = 640  # drawn at twice that: 1280 by 640, the size GitHub asks for

LEARN = {
    1: "Read a panel before modelling it: who is in it, when, and what the columns can say",
    2: "Turn a company-level flag into a dated event, and count the exits that are not failures",
    3: "Find where failure shows in the accounts, how early, and how the ground shifts",
    4: "Build the yardstick first: splits by time, an embargo, ranking metrics, honest intervals",
    5: "Tune boosted trees without fooling yourself, and test what history adds",
    6: "Prepare inputs for a network, train it repeatably, and read a term structure of default",
    7: "Open a sealed holdout once, calibrate, explain the score, and state the limits",
}
STEPS = [
    ("FIRST", "Get the data", "One command fetches the file and checks it.", "data/README.md"),
    ("NEXT", "Chapter 1: The Panel", "What exactly is in this file?", "notebooks/01_the_panel.ipynb"),
    ("SHORTCUT", "Chapter 7: The Verdict", "Only here for the result? The holdout, opened once.", "notebooks/07_the_verdict.ipynb"),
]  # fmt: skip
ARROW = (
    '<svg class="cb-go" width="40" height="40" viewBox="0 0 40 40">'
    '<path d="M10 20 H28 M21 13 L28 20 L21 27"/></svg>'
)


# ------------------------------------------------------------ what the page knows
def chapters() -> list[dict]:
    """Each chapter's number, title, question, and notebook, from its plan."""
    listed = []
    for path in sorted(PLANS.glob("[0-9][0-9].json")):
        plan = json.loads(path.read_text(encoding="utf-8"))
        name = "_".join(plan["title"].lower().split())
        listed.append(
            {
                "number": plan["number"],
                "title": plan["title"],
                "question": plan["question"],
                "notebook": f"notebooks/{plan['number']:02d}_{name}.ipynb",
            }
        )
    return listed


def gather_facts() -> dict:
    """The panel's counts, from the values the built chapters stored."""
    values = store.values()
    return {
        "firm_years": values["ch1.firm_years"],
        "companies": values["ch1.companies"],
        "failed_companies": values["ch1.failed_companies"],
        "base_rate": values["ch2.base_rate"],
        "first_year": values["ch7.dev_first"],
        "last_year": PANEL_END,
        "chapters": len(chapters()),
    }


def facts() -> dict:
    return json.loads((RESULTS / "landing.json").read_text(encoding="utf-8"))


def years(known: dict) -> str:
    return f"{known['first_year']} to {known['last_year']}"


# ------------------------------------------------------------ the pictures
def hero(mark: dict, known: dict) -> str:
    """The cover of the casebook: its name, one line, the house mark, and three counts."""
    counts = [
        (known["firm_years"], "FIRM-YEARS"),
        (known["companies"], "COMPANIES"),
        (known["failed_companies"], "FAILURES"),
    ]
    shown = "".join(f"<span><b>{value:,}</b> {name}</span>" for value, name in counts)
    return (
        f'<div class="cb cb-hero"><div class="cb-kicker">A CASEBOOK IN {known["chapters"]} CHAPTERS</div>'
        f"<h1>{escape(TITLE)}</h1><p>{escape(LINE)}, {years(known)}.</p>"
        f'<div class="cb-mark">{c.mark_curve(mark, None, width=LANDING_WIDTH, height=170)}</div>'
        f'<div class="cb-facts">{shown}<span class="cb-by">Created by <b>{escape(c.AUTHOR)}</b></span>'
        f"</div></div>"
    )


def number_facts(known: dict) -> list[tuple[str, str, str]]:
    return [
        (f"{known['firm_years']:,}", "FIRM-YEARS", years(known)),
        (f"{known['companies']:,}", "COMPANIES", "US listed"),
        (f"{known['failed_companies']:,}", "FAILED", "Marked bankrupt"),
        (f"{known['base_rate']:.1%}", "BASE RATE", "Fail within two years"),
        (str(known["chapters"]), "CHAPTERS", "Each a notebook"),
    ]


def numbers(known: dict) -> str:
    """Five facts about the casebook on one plate."""
    shown = "".join(
        f'<div class="cb-fact"><span class="cb-n">{escape(n)}</span>'
        f'<span class="cb-what">{escape(what)}</span><span class="cb-more">{escape(more)}</span></div>'
        for n, what, more in number_facts(known)
    )
    return f'<div class="cb cb-numbers">{shown}</div>'


def step(label: str, title: str, line: str) -> str:
    """One of the three cards under Start here. Each is a link in the README."""
    return (
        f'<div class="cb cb-step"><div class="cb-edge"></div><span class="cb-label">{escape(label)}</span>'
        f'<div class="cb-what"><span class="cb-title">{escape(title)}</span>'
        f'<span class="cb-line">{escape(line)}</span></div>{ARROW}</div>'
    )


def plate(body: str, caption: str) -> str:
    return c.figure_plate(body, caption).replace(
        'class="cb cb-plate"', 'class="cb cb-plate cb-land"', 1
    )


def route() -> str:
    """The chapters in the order they are read, each with its question."""
    rows = "".join(
        f'<div class="cb-row"><div class="cb-node"></div><span class="cb-no">{chapter["number"]:02d}</span>'
        f'<span class="cb-title">{escape(chapter["title"])}</span>'
        f'<span class="cb-question">{escape(chapter["question"])}</span></div>'
        for chapter in chapters()
    )
    return plate(
        f'<div class="cb-chapters">{rows}</div>',
        "The casebook in the order it is read, one question to a chapter.",
    )


def inside() -> str:
    """One exhibit from each chapter that has one in the gallery, as its notebook draws it."""
    shown = [chapter for chapter in chapters() if (GALLERY / f"ch{chapter['number']}.png").exists()]
    charts = "".join(
        f'<div class="cb-chart"><img src="{(GALLERY / f"ch{chapter['number']}.png").as_uri()}" alt="">'
        f'<span class="cb-label"><b>CHAPTER {chapter["number"]}</b>{escape(chapter["title"])}</span></div>'
        for chapter in shown
    )
    return plate(
        f'<div class="cb-inside">{charts}</div>',
        "One exhibit from each of these chapters, as the notebook's own code draws it.",
    )


def evidence() -> str:
    """The ladder as Chapter 7 draws it on the holdout, set on a plate at the page's width."""
    wide, tall = Image.open(GALLERY / "ladder.png").size
    height = round(734 * tall / wide)  # a whole number of pixels, or the studio refuses the picture
    return plate(
        f'<div class="cb-evidence"><img src="{(GALLERY / "ladder.png").as_uri()}" '
        f'width="734" height="{height}" alt=""></div>',
        f"The ladder on the holdout, observation years {HOLDOUT[0]} to {HOLDOUT[1]}, from Chapter 7.",
    )


def closing() -> str:
    first = chapters()[0]
    return c.next_card("START", f"CHAPTER {first['number']}", first["title"], first["question"])


def social(mark: dict, known: dict) -> str:
    """The card a link to the repo shows."""
    return (
        f'<div class="cb cb-social"><div class="cb-kicker">A CASEBOOK IN {known["chapters"]} CHAPTERS</div>'
        f"<h1>{escape(TITLE)}</h1><p>{escape(LINE)}, {years(known)}.</p>"
        f'<div class="cb-mark">{c.mark_curve(mark, None, width=SOCIAL_WIDTH, height=120)}</div>'
        f'<div class="cb-by">Created by <b>{escape(c.AUTHOR)}</b></div></div>'
    )


# ------------------------------------------------------------ the blocks of the README
def picture(name: str, alt: str) -> str:
    return (
        f'<img src="assets/readme/landing/{name}.png" width="{LANDING_WIDTH}" alt="{escape(alt)}">'
    )


def results_table(metrics: dict, chosen: str) -> str:
    """Every rung on both samples: one markdown table from results/metrics.json."""
    walked = {row["model"]: row["walk-forward"] for row in metrics["ladder"]}
    steps = {row["model"]: row for row in metrics["paired"]}
    lines = [
        (
            "| Rung | Model | Walk-forward ROC-AUC | Holdout ROC-AUC | 95% interval "
            "| Step from the rung below | Average precision | Top 5% finds | Lift |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rung, row in enumerate(metrics["scoreboard"], start=1):
        name = f"**{row['model']}** (chosen)" if row["model"] == chosen else row["model"]
        step_ = steps.get(row["model"])
        added = (
            f"{step_['roc_auc_diff']:+.3f} ({step_['roc_auc_diff_lo']:+.3f} to {step_['roc_auc_diff_hi']:+.3f})"
            if step_
            else ""
        )
        lines.append(
            f"| {rung} | {name} | {walked[row['model']]:.3f} | {row['roc_auc']:.3f} "
            f"| {row['roc_auc_lo']:.3f} to {row['roc_auc_hi']:.3f} | {added} | {row['ap']:.3f} "
            f"| {row['capture_5']:.1%} | {row['lift_5']:.1f} |"
        )
    return "\n".join(lines)


def chapter_table() -> str:
    lines = ["| Chapter | The question | What you learn |", "| --- | --- | --- |"]
    for chapter in chapters():
        lines.append(
            f"| [{chapter['number']}. {chapter['title']}]({chapter['notebook']}) | {chapter['question']} "
            f"| {LEARN[chapter['number']]} |"
        )
    return "\n".join(lines)


def evidence_words(claims: dict) -> str:
    return "\n".join(
        [
            (
                f"The holdout is the observation years {HOLDOUT[0]} to {HOLDOUT[1]}: {claims['holdout_rows']} "
                f"firm-years with {claims['holdout_failures']} failures, opened once."
            ),
            "",
            (
                f'The chosen model, "{claims["chosen_model"]}", has a ROC-AUC of {claims["holdout_auc"]} there, '
                f"with a 95% interval from {claims['holdout_auc_interval']}."
            ),
            f"On the walk-forward that chose it, it scored {claims['walk_forward_auc']}.",
            "",
            f"The published Altman Z scores {claims['classic_auc']} on the same rows.",
            (
                f"The paired difference is {claims['primary_step']}, with an interval from "
                f"{claims['primary_step_interval']}."
            ),
            "",
            (
                f"Reviewing the top 5% of each year finds {claims['capture_5']} of the failures, with an interval "
                f"from {claims['capture_5_interval']}, a lift of {claims['lift_5']} over names picked at random."
            ),
            "",
            f"On companies the model never saw in training, it scores {claims['unseen_auc']}.",
        ]
    )


def blocks() -> dict[str, str]:
    """Every generated block of the README, from results/ and the chapter plans."""
    known = facts()
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    claims = json.loads((RESULTS / "claims.json").read_text(encoding="utf-8"))
    counted = ", ".join(f"{n} {what.lower()}" for n, what, _ in number_facts(known)[:3])
    said = ". ".join(f"{n} {what.lower()}, {more.lower()}" for n, what, more in number_facts(known))
    return {
        "hero": picture(
            "hero",
            f"{TITLE}. {LINE}, {years(known)}. {counted}. The house mark is the share of companies "
            f"not yet failed, stepping down over {known['last_year'] - known['first_year'] + 1} years. Created by {c.AUTHOR}.",
        ),
        "numbers": picture("numbers", f"The casebook in numbers. {said}."),
        "chapters": chapter_table(),
        "evidence": picture(
            "ladder",
            f"The ladder on the holdout: the ROC-AUC of each of the rungs with its interval, and what each "
            f"adds to the one below. The chosen model scores {claims['holdout_auc']}, "
            f"{claims['primary_step']} against the published Altman Z.",
        )
        + "\n\n"
        + evidence_words(claims),
        "results": results_table(metrics, claims["chosen_model"]),
    }


def fill(text: str, made: dict[str, str]) -> str:
    """`text` with each named block replaced between its markers."""
    for name, block in made.items():
        start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
        if start not in text or end not in text:
            raise ValueError(f"the page has no {name}:start and {name}:end markers")
        head, rest = text.split(start, 1)
        _, tail = rest.split(end, 1)
        text = f"{head}{start}\n{block}\n{end}{tail}"
    return text


# ------------------------------------------------------------ draw and fill
def draw() -> None:
    """Draw every picture, save the counts they carry, and fill the README's blocks."""
    known = gather_facts()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "landing.json").write_text(
        json.dumps(known, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    mark = c.house_mark(store.load_frame("event_table"))
    pictures = {
        "hero": hero(mark, known),
        "numbers": numbers(known),
        "route": route(),
        "inside": inside(),
        "ladder": evidence(),
        "next": closing(),
    }
    pictures.update({f"step{k}": step(*parts[:3]) for k, parts in enumerate(STEPS, start=1)})
    with Studio() as studio:
        for name, html in pictures.items():
            print(studio.png(STYLE + html, OUT / f"{name}.png", LANDING_WIDTH).name)
        print(studio.png(STYLE + social(mark, known), OUT.parent / "social.png", SOCIAL_WIDTH).name)
    readme = REPO / "README.md"
    readme.write_text(
        fill(readme.read_text(encoding="utf-8"), blocks()), encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    draw()
