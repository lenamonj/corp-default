"""The writing rules, on every file a reader sees: no dashes, no emojis, no version history,
and no link in a chapter that SOURCES.md does not record."""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CHAPTER_FILES = sorted(
    path
    for folder in (REPO / "chapters", REPO / "tests" / "fixtures")
    for path in folder.rglob("*")
    if path.suffix in {".py", ".json"}
)
PAGES = [
    REPO / name
    for name in ("README.md", "SOURCES.md", "data/README.md", "tools/casebook/README.md")
]
DASH = re.compile("[\u2013\u2014]")
EMOJI = re.compile("[\U0001f000-\U0001faff\u2600-\u27bf\ufe0f\u200d]")
HISTORY = ("previous version", "old label", "used to", "no longer")
LINK = re.compile(r"https?://[^\s)>\]\"'`]+")


def _id(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def dashes_and_emojis(text: str) -> list[str]:
    """Each line that holds an em dash, an en dash, or an emoji."""
    return [line for line in text.split("\n") if DASH.search(line) or EMOJI.search(line)]


def history(text: str) -> list[str]:
    """Each phrase that tells the reader about an earlier version."""
    return [phrase for phrase in HISTORY if phrase in text.lower()]


def links(text: str) -> list[str]:
    return [link.rstrip(".,;:") for link in LINK.findall(text)]


def test_the_checks_find_what_they_look_for():
    assert dashes_and_emojis("plain - hyphen\nan em\u2014dash\nan en\u2013dash\na \u2705 mark") == [
        "an em\u2014dash",
        "an en\u2013dash",
        "a \u2705 mark",
    ]
    assert history("The label is No Longer constant, as it used to be.") == ["used to", "no longer"]
    assert links("See https://example.org/a_b. Also (http://example.org/c).") == [
        "https://example.org/a_b",
        "http://example.org/c",
    ]


@pytest.mark.parametrize("path", [*CHAPTER_FILES, *PAGES], ids=_id)
def test_no_dash_and_no_emoji(path):
    assert dashes_and_emojis(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("path", [*CHAPTER_FILES, REPO / "README.md"], ids=_id)
def test_no_version_history(path):
    assert history(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("path", CHAPTER_FILES, ids=_id)
def test_every_link_in_a_chapter_is_in_sources(path):
    sources = (REPO / "SOURCES.md").read_text(encoding="utf-8")
    missing = [link for link in links(path.read_text(encoding="utf-8")) if link not in sources]
    assert missing == []
