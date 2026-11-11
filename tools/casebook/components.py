"""The drawn components of the casebook. Each function returns the HTML of one component, styled by
design.css; studio.Studio.png draws it. README.md in this folder says where each one goes.

Positions inside a component are set by number, so a builder places words where they belong and the
studio's fit check refuses any that do not fit.
"""

import itertools
from html import escape

from tools.casebook.studio import WIDTH

AUTHOR = "Jeff Lenamon"
SERIES = "Corporate Default Prediction"
OPENER_WORDS = 8  # an opener's goal is one short line
TIMEBAR_KINDS = ("train", "embargo", "valid", "holdout", "label")


def step_curve(
    points: list[tuple[float, float]],
    width: int,
    height: int,
    box: tuple[int, int, int, int],
    marker: float | None = None,
    ticks: list[float] | None = None,
    y_range: tuple[float, float] | None = None,
    labels: tuple[tuple[float, str], ...] = (),
    color: str = "var(--signal-bright)",
) -> str:
    """The house mark: a survival curve that steps down, as an SVG.

    points are (time, survival) in time order; box is (left, top, right, bottom) in pixels.
    marker is a time to mark as "you are here"; ticks are times marked with a short line under the
    curve; labels are (survival, words) set at the left edge, for a truncated axis.
    """
    left, top, right, bottom = box
    times = [t for t, _ in points]
    low, high = y_range or (min(s for _, s in points), max(s for _, s in points))

    def x_of(t: float) -> float:
        return left + (right - left) * (t - times[0]) / (times[-1] - times[0])

    def y_of(s: float) -> float:
        return bottom - (bottom - top) * (s - low) / (high - low)

    def survival_at(t: float) -> float:
        return [s for time, s in points if time <= t][-1]

    # Post-step: survival holds until the next time, then drops.
    path = f"M{x_of(times[0]):.1f} {y_of(points[0][1]):.1f}"
    for (_, before), (t, s) in itertools.pairwise(points):
        path += f" L{x_of(t):.1f} {y_of(before):.1f} L{x_of(t):.1f} {y_of(s):.1f}"
    drawn = [f'<line class="axis" x1="{left}" x2="{right}" y1="{bottom}" y2="{bottom}"/>']
    for value, text in labels:
        drawn.append(
            f'<text class="label" x="{left - 24}" y="{y_of(value) + 4:.1f}" '
            f'text-anchor="end">{escape(text)}</text>'
        )
    for t in ticks or []:
        x = x_of(t)
        drawn.append(
            f'<line class="tick" x1="{x:.1f}" x2="{x:.1f}" y1="{bottom}" y2="{bottom - 10}"/>'
        )
    drawn.append(f'<path class="step" d="{path}" style="stroke:{color}"/>')
    if marker is not None:
        x, y = x_of(marker), y_of(survival_at(marker))
        drawn.append(f'<circle class="halo" cx="{x:.1f}" cy="{y:.1f}" r="12"/>')
        drawn.append(f'<circle class="here" cx="{x:.1f}" cy="{y:.1f}" r="5"/>')
    return (
        f'<svg class="curve" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f"{''.join(drawn)}</svg>"
    )


def cover(
    number: int, title: str, question: str, minutes: int, curve: str, chapters: int = 7
) -> str:
    """A chapter's first picture: its numeral, title, question, and its place on the house mark."""
    return (
        f'<div class="cb cover"><header><span>{escape(SERIES.upper())}</span>'
        f"<span>CHAPTER {number} OF {chapters}</span></header>"
        f'<main><div class="numeral">{number:02d}</div><div class="words">'
        f'<h1>{escape(title)}</h1><p class="question">{escape(question)}</p></div></main>'
        f'<div class="mark">{curve}</div>'
        f"<footer><span>Created by <b>{escape(AUTHOR)}</b></span>"
        f"<span>ABOUT {minutes} MIN</span></footer></div>"
    )


def _route_layout(count: int) -> tuple[float, float, float]:
    """Margin, gap between stops, and label width, so labels on one side never meet."""
    margin = 100.0
    for _ in range(3):
        gap = (WIDTH - 2 * margin) / max(count - 1, 1)
        label = min(2 * gap - 16, 240.0)
        margin = max(100.0, label / 2 + 48)  # inside the header's 48 px margins
    return margin, gap, label


def route(parts: list[tuple[str, int]], total: str) -> str:
    """The chapter's parts in order on a ruled rail, minutes under each, labels above and below."""
    margin, gap, label = _route_layout(len(parts))
    stops = []
    for i, (title, minutes) in enumerate(parts):
        x = margin + gap * i
        side = "above" if i % 2 == 0 else "below"
        stops.append(
            f'<i class="stop" style="left:{x - 5:.1f}px"></i>'
            f'<div class="part {side}" style="left:{x - label / 2:.1f}px;width:{label:.1f}px">'
            f'<span class="no">{i + 1:02d}</span><span class="name">{escape(title)}</span>'
            f'<span class="min">{minutes} MIN</span></div>'
        )
    return (
        f'<div class="cb route"><header><span>YOUR ROUTE</span><span>{escape(total)}</span></header>'
        f'<div class="rail" style="left:{margin:.1f}px;width:{gap * (len(parts) - 1):.1f}px"></div>'
        f"{''.join(stops)}</div>"
    )


def opener(index: int, count: int, goal: str, minutes: int) -> str:
    """The strip before a part: which part of how many, its goal in a few words, about how long."""
    if len(goal.split()) > OPENER_WORDS:
        raise ValueError(
            f"opener goal has {len(goal.split())} words; keep it to eight words: {goal!r}"
        )
    marks = "".join(
        f'<i class="{"done" if i < index else "now" if i == index else ""}"></i>'
        for i in range(1, count + 1)
    )
    return (
        f'<div class="cb opener"><span class="bar"></span>'
        f'<span class="part">PART {index:02d} / {count:02d}</span>'
        f'<span class="goal">{escape(goal)}</span><span class="ticks">{marks}</span>'
        f'<span class="time">ABOUT {minutes} MIN</span></div>'
    )


def pitfall(
    name: str, naive_label: str, naive_value: str, honest_label: str, honest_value: str, why: str
) -> str:
    """A pitfall plate: the naive number beside the honest number, both computed, and why they differ."""
    return (
        f'<div class="cb pitfall"><div class="band"><span class="what">PITFALL</span>'
        f'<span class="name">{escape(name)}</span></div><div class="sides">'
        f'<div class="side naive"><span class="label">{escape(naive_label)}</span>'
        f'<span class="value">{escape(naive_value)}</span></div>'
        f'<div class="side honest"><span class="label">{escape(honest_label)}</span>'
        f'<span class="value">{escape(honest_value)}</span></div></div>'
        f'<p class="why">{escape(why)}</p></div>'
    )


def stage(width: int, height: int, *parts: str) -> str:
    """A diagram: boxes, words, time bars, and arrows placed by position."""
    return f'<div class="stage" style="width:{width}px;height:{height}px">{"".join(parts)}</div>'


def box(x: int, y: int, w: int, h: int, kind: str = "") -> str:
    """A diagram box. kind: "" (hairline), tint (sheet deep), ink (solid)."""
    return f'<div class="box {kind}" style="left:{x}px;top:{y}px;width:{w}px;height:{h}px"></div>'


def words(x: int, y: int, w: int, text: str, kind: str = "") -> str:
    """Words in a diagram, centred in a strip w wide. kind: b (bold), m (mono), s (small), left."""
    return f'<div class="t {kind}" style="left:{x}px;top:{y}px;width:{w}px">{escape(text)}</div>'


def arrows(width: int, height: int, *lines: tuple) -> str:
    """Arrows over a diagram, each (x1, y1, x2, y2); a fifth item "dash" draws a dashed line, no head."""
    drawn = ""
    for line in lines:
        x1, y1, x2, y2 = line[:4]
        kind = "dash" if len(line) > 4 else "arrow"
        head = "" if kind == "dash" else ' marker-end="url(#head)"'
        drawn += f'<line class="{kind}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"{head}/>'
    return (
        f'<svg class="arrows" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><defs>'
        f'<marker id="head" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto"><path d="M0 0 L10 5 L0 10 z" class="head"/></marker></defs>{drawn}</svg>'
    )


def timebar(x: int, y: int, w: int, h: int, kind: str, label: str = "") -> str:
    """A span of time in a diagram. kind: train, embargo (hatched), valid, holdout (ink), label."""
    if kind not in TIMEBAR_KINDS:
        raise ValueError(f"timebar kind must be one of {TIMEBAR_KINDS}, not {kind!r}")
    return (
        f'<div class="bar {kind}" style="left:{x}px;top:{y}px;width:{w}px;height:{h}px">'
        f"<span>{escape(label)}</span></div>"
    )


def figure_plate(body: str, caption: str) -> str:
    """The sheet frame a diagram sits in, with its caption at the foot."""
    return (
        f'<div class="cb plate"><div class="body">{body}</div>'
        f'<div class="cap">{escape(caption)}</div></div>'
    )


def next_card(label: str, plate: str, title: str, line: str) -> str:
    """The carbon card that closes a chapter: where to go next."""
    return (
        f'<div class="cb next"><div class="where"><span class="label">{escape(label)}</span>'
        f'<span class="plate">{escape(plate)}</span></div><div class="what">'
        f'<span class="title">{escape(title)}</span><span class="line">{escape(line)}</span></div>'
        f'<svg class="go" width="56" height="56" viewBox="0 0 56 56">'
        f'<path d="M16 28 H38 M30 20 L38 28 L30 36"/></svg></div>'
    )
