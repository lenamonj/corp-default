import itertools
from pathlib import Path

import pytest
from matplotlib import font_manager

FONTS = Path(__file__).resolve().parents[1] / "assets" / "fonts"
FILES = {
    "archivo": ["Archivo-SemiBold.ttf", "Archivo-ExtraBold.ttf"],
    "public-sans": ["PublicSans-Regular.ttf", "PublicSans-SemiBold.ttf"],
    "jetbrains-mono": ["JetBrainsMono-Medium.ttf", "JetBrainsMono-Bold.ttf"],
}
FAMILY_NAMES = {
    "archivo": "Archivo",
    "public-sans": "Public Sans",
    "jetbrains-mono": "JetBrains Mono",
}


@pytest.mark.parametrize(
    ("folder", "name"), [(folder, name) for folder, names in FILES.items() for name in names]
)
def test_font_loads_and_has_its_licence_beside_it(folder, name):
    path = FONTS / folder / name
    assert path.exists(), path
    assert font_manager.FontProperties(fname=path).get_name() == FAMILY_NAMES[folder]
    licence = FONTS / folder / "OFL.txt"
    assert "SIL Open Font License" in licence.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def studio():
    from tools.casebook.studio import Studio

    with Studio() as drawing:
        yield drawing


def test_a_component_is_saved_at_twice_its_size(studio, tmp_path):
    from PIL import Image

    out = studio.png('<div class="cb" style="height:100px"></div>', tmp_path / "plain.png")
    assert Image.open(out).size == (2400, 200)


def test_the_landing_width_is_830(studio, tmp_path):
    from PIL import Image

    from tools.casebook.studio import LANDING_WIDTH

    out = studio.png(
        '<div class="cb" style="height:50px"></div>', tmp_path / "l.png", LANDING_WIDTH
    )
    assert Image.open(out).size == (1660, 100)


def test_a_word_wider_than_its_box_is_refused_by_name(studio, tmp_path):
    html = (
        '<div class="cb" style="height:100px"><span style="display:block;width:60px;'
        'white-space:nowrap;overflow:hidden">Supercalifragilistic</span></div>'
    )
    with pytest.raises(ValueError, match="Supercalifragilistic"):
        studio.png(html, tmp_path / "wide.png")


def test_a_part_pixel_height_is_refused(studio, tmp_path):
    with pytest.raises(ValueError, match="whole number of pixels"):
        studio.png('<div class="cb" style="height:100.5px"></div>', tmp_path / "half.png")


def test_the_design_css_carries_the_tokens_and_the_fonts():
    from tools.casebook.studio import CSS

    css = CSS.read_text(encoding="utf-8")
    for token in ("#121316", "#F7F7F5", "#16181D", "#D9482B", "#295F92", "#FF6A4D", "#8FB1D1"):
        assert token in css
    for family in ("Archivo", "Public Sans", "JetBrains Mono"):
        assert f'font-family: "{family}"' in css


# The longest words the spec gives each component.
LONGEST_QUESTION = "Where in the accounts does failure show, and how early?"
TWELVE_PARTS = [
    ("Setting Up", 2),
    ("From Dollars to Ratios", 6),
    ("Looking at Skewed Ratios", 4),
    ("Anatomy of a Failure", 6),
    ("Failure Rate Along Each Ratio", 5),
    ("Why the Worst-Looking Firms Are Not the Riskiest", 5),
    ("Ranking the Signals One at a Time", 4),
    ("Redundancy", 3),
    ("The Risk Map", 3),
    ("History Matters", 5),
    ("The Ground Is Moving", 5),
    ("Do Raw Dollars and Industry Earn Their Place", 4),
]
SURVIVAL = [(0, 1.0), (1, 0.97), (2, 0.95), (3, 0.95), (4, 0.91), (5, 0.88), (6, 0.88)]


def _width(path):
    from PIL import Image

    return Image.open(path).size[0]


def _curve(marker=None):
    from tools.casebook.components import step_curve

    return step_curve(SURVIVAL, 1200, 160, (64, 20, 1136, 140), marker=marker, ticks=[1, 4])


@pytest.mark.parametrize(
    ("number", "title", "question"),
    [(1, "The Panel", "What exactly is in this file?"), (4, "The Yardstick", LONGEST_QUESTION)],
)
def test_cover_draws_at_both_ends_of_the_spec(studio, tmp_path, number, title, question):
    from tools.casebook.components import cover

    html = cover(number, title, question, 45, _curve(marker=3))
    assert _width(studio.png(html, tmp_path / "cover.png")) == 2400


@pytest.mark.parametrize("count", [3, 9, 12])
def test_route_draws_up_to_twelve_parts(studio, tmp_path, count):
    from tools.casebook.components import route

    html = route(TWELVE_PARTS[:count], "ABOUT 50 MIN")
    assert _width(studio.png(html, tmp_path / "route.png")) == 2400


def test_opener_draws_and_refuses_a_goal_over_eight_words(studio, tmp_path):
    from tools.casebook.components import opener

    html = opener(12, 12, "Tell drift from signal before trusting a score.", 5)
    assert _width(studio.png(html, tmp_path / "opener.png")) == 2400
    with pytest.raises(ValueError, match="eight words"):
        opener(1, 9, "One two three four five six seven eight nine.", 3)


def test_pitfall_draws(studio, tmp_path):
    from tools.casebook.components import pitfall

    html = pitfall(
        "A row shift across a reporting gap",
        "ROWS MIS-PAIRED BY SHIFT",
        "1,234",
        "MIS-PAIRED BY A CALENDAR JOIN",
        "0",
        "A shift pairs neighbours in the file, not neighbours in time.",
    )
    assert _width(studio.png(html, tmp_path / "pitfall.png")) == 2400


def test_diagram_parts_draw_on_a_figure_plate(studio, tmp_path):
    from tools.casebook.components import arrows, box, figure_plate, stage, timebar, words

    body = stage(
        1000,
        160,
        box(0, 0, 200, 60),
        words(0, 20, 200, "Observation year t", "b"),
        timebar(220, 0, 300, 28, "train", "TRAIN"),
        timebar(530, 0, 120, 28, "embargo", "EMBARGO"),
        timebar(660, 0, 120, 28, "valid", "VALID"),
        timebar(790, 0, 200, 28, "holdout", "HOLDOUT"),
        timebar(220, 60, 200, 28, "label", "LABEL WINDOW"),
        arrows(1000, 160, (100, 60, 100, 120), (220, 100, 420, 100, "dash")),
    )
    html = figure_plate(body, "One firm-year and the two years its label looks ahead.")
    assert _width(studio.png(html, tmp_path / "diagram.png")) == 2400


def test_timebar_refuses_an_unknown_kind():
    from tools.casebook.components import timebar

    with pytest.raises(ValueError, match="kind"):
        timebar(0, 0, 10, 10, "test")


def test_next_card_draws(studio, tmp_path):
    from tools.casebook.components import next_card

    html = next_card("NEXT", "CHAPTER 4", "The Yardstick", "How will we know a model is good?")
    assert _width(studio.png(html, tmp_path / "next.png")) == 2400


def test_step_curve_only_moves_right_and_down_for_a_survival_input():
    import re

    path = re.search(r'<path class="cb-step" d="([^"]+)"', _curve()).group(1)
    numbers = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", path)]
    xs, ys = numbers[0::2], numbers[1::2]
    assert all(b >= a for a, b in itertools.pairwise(xs))
    assert all(b >= a for a, b in itertools.pairwise(ys))  # SVG y grows downward
    assert xs[0] == 64 and xs[-1] == 1136


def test_carbon_components_carry_a_carbon_rule_edge_for_dark_pages():
    import re

    from tools.casebook.studio import CSS

    css = CSS.read_text(encoding="utf-8")
    for selector in (".cb-cover", ".cb-next"):
        rule = re.search(re.escape(selector) + r" \{([^}]*)\}", css).group(1)
        assert "border: 1px solid var(--carbon-rule)" in rule, selector


def test_house_mark_is_the_share_not_yet_failed_on_a_labelled_truncated_axis(toy_mark):
    # Four companies: failed at 1, left at 2, failed at 3, still reporting at 4. The cumulative
    # incidence of failure is 1/4 at year 1 and 1/4 + (1/2)(1/2) = 1/2 at year 3.
    times = [t for t, _ in toy_mark["points"]]
    shares = [s for _, s in toy_mark["points"]]
    assert times == [0, 1, 2, 3, 4]
    assert shares == pytest.approx([1.0, 0.75, 0.75, 0.5, 0.5])
    assert toy_mark["y_range"] == (0.5, 1.0)
    assert toy_mark["labels"] == ((1.0, "1.00"), (0.5, "0.50"))


def test_mark_curve_marks_the_chapter_from_the_first_to_the_last(toy_mark):
    import re

    from tools.casebook.components import mark_curve

    def here(chapter):
        found = re.search(r'<circle class="cb-here" cx="([^"]+)"', mark_curve(toy_mark, chapter))
        return found and float(found.group(1))

    assert here(1) == 96 and here(7) == 1136
    assert here(4) == pytest.approx((96 + 1136) / 2)
    assert here(None) is None
