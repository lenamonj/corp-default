"""The landing page: its pictures, its links, and the numbers it quotes from results/."""

import json
import re
from pathlib import Path

import pytest
from PIL import Image

from tools.casebook import landing

REPO = Path(__file__).resolve().parents[1]
README = (REPO / "README.md").read_text(encoding="utf-8")
IMG = re.compile(r"<img\b[^>]*>")
METRICS = {
    "scoreboard": [
        {"model": "A", "roc_auc": 0.7, "roc_auc_lo": 0.65, "roc_auc_hi": 0.75, "ap": 0.03,
         "capture_5": 0.05, "lift_5": 1.0},
        {"model": "B", "roc_auc": 0.8, "roc_auc_lo": 0.76, "roc_auc_hi": 0.84, "ap": 0.1,
         "capture_5": 0.356, "lift_5": 7.12},
    ],
    "paired": [{"model": "B", "roc_auc_diff": 0.1, "roc_auc_diff_lo": 0.06, "roc_auc_diff_hi": 0.14}],
    "ladder": [{"model": "A", "walk-forward": 0.69}, {"model": "B", "walk-forward": 0.77}],
}  # fmt: skip


def attribute(tag: str, name: str) -> str | None:
    found = re.search(rf'\b{name}="([^"]*)"', tag)
    return found.group(1) if found else None


def local_targets() -> list[str]:
    """Every relative path the page points at: pictures, markdown links, and anchors' targets."""
    targets = [attribute(tag, "src") for tag in IMG.findall(README)]
    targets += re.findall(r'<a\b[^>]*href="([^"]+)"', README)
    targets += re.findall(r"\]\(([^)\s]+)\)", README)
    return [target for target in targets if not target.startswith(("http", "#", "mailto:"))]


def test_results_table_by_hand():
    table = landing.results_table(METRICS, chosen="B").split("\n")
    assert table[0].startswith("| Rung | Model | Walk-forward ROC-AUC | Holdout ROC-AUC |")
    assert table[2] == "| 1 | A | 0.690 | 0.700 | 0.650 to 0.750 |  | 0.030 | 5.0% | 1.0 |"
    assert table[3] == (
        "| 2 | **B** (chosen) | 0.770 | 0.800 | 0.760 to 0.840 | +0.100 (+0.060 to +0.140) "
        "| 0.100 | 35.6% | 7.1 |"
    )


def test_fill_replaces_a_block_and_keeps_the_rest():
    text = "top\n<!-- results:start -->\nold\n<!-- results:end -->\nfoot\n"
    assert landing.fill(text, {"results": "new"}) == (
        "top\n<!-- results:start -->\nnew\n<!-- results:end -->\nfoot\n"
    )


def test_fill_refuses_a_page_without_the_markers():
    with pytest.raises(ValueError, match="results:start"):
        landing.fill("no markers here", {"results": "new"})


def test_the_generated_blocks_are_current():
    assert landing.fill(README, landing.blocks()) == README


def test_every_claim_is_on_the_page():
    claims = json.loads((REPO / "results" / "claims.json").read_text(encoding="utf-8"))
    assert [key for key, claim in claims.items() if claim not in README] == []


def test_every_picture_exists_at_twice_the_page_width():
    pictures = [attribute(tag, "src") for tag in IMG.findall(README)]
    assert len(pictures) >= 9
    for picture in pictures:
        assert not picture.startswith("http"), picture
        assert Image.open(REPO / picture).width == 2 * landing.LANDING_WIDTH, picture


def test_every_picture_has_a_width_and_says_what_it_shows():
    for tag in IMG.findall(README):
        assert attribute(tag, "width") == str(landing.LANDING_WIDTH), tag
        assert len(attribute(tag, "alt") or "") > 20, tag


def test_the_page_carries_no_style_attribute():
    # GitHub strips a style attribute. A badge's address may still say style=for-the-badge.
    assert not re.search(r"\sstyle\s*=\s*[\"']", README)


def test_every_relative_link_resolves():
    missing = [t for t in local_targets() if not (REPO / t.split("#")[0]).exists()]
    assert missing == []


def test_nothing_links_to_the_retired_notebook():
    assert "corp_default.ipynb" not in README
    assert not (REPO / "corp_default.ipynb").exists()


def test_every_chapter_is_linked():
    for number in range(1, 8):
        assert re.search(rf"\(notebooks/{number:02d}_\w+\.ipynb\)", README), number
