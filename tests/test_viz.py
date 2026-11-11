import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib import font_manager
from palette_checks import contrast, failures

from corp_default import viz

SHEET = "#F7F7F5"


@pytest.fixture(autouse=True)
def fresh_style():
    yield
    plt.close("all")
    matplotlib.rcdefaults()


def _texts(fig) -> list[str]:
    return [text.get_text() for text in fig.texts]


def test_house_style_sets_the_sheet_and_registers_public_sans():
    viz.use_house_style()
    assert matplotlib.rcParams["figure.facecolor"] == SHEET
    found = font_manager.findfont(font_manager.FontProperties(family="Public Sans"))
    assert Path(found).name == "PublicSans-Regular.ttf"
    assert matplotlib.rcParams["font.family"] == ["Public Sans"]


def test_exhibits_are_numbered_and_the_house_style_resets_them():
    viz.use_house_style()
    first, second = plt.figure(), plt.figure()
    first.add_subplot().plot([0, 1])
    second.add_subplot().plot([0, 1])
    viz.exhibit(first, "First", "What to notice.")
    viz.exhibit(second, "Second", "What to notice.", source="Kaggle file")
    assert "EXHIBIT 1" in _texts(first) and "EXHIBIT 2" in _texts(second)
    assert "Source: Kaggle file" in _texts(second)
    viz.use_house_style()
    third = plt.figure()
    third.add_subplot().plot([0, 1])
    viz.exhibit(third, "Third", "What to notice.")
    assert "EXHIBIT 1" in _texts(third)


def test_an_exhibit_is_1200_pixels_wide_and_can_be_saved_for_the_landing_page(
    tmp_path, monkeypatch
):
    from PIL import Image

    monkeypatch.setattr(viz, "GALLERY", tmp_path)
    viz.use_house_style()
    fig = plt.figure(figsize=(viz.EXHIBIT_WIDTH_IN, 4))
    fig.add_subplot().plot([0, 1])
    viz.exhibit(fig, "Saved", "What to notice.", save_as="ch0")
    assert Image.open(tmp_path / "ch0.png").size[0] == 1200


def test_missing_fonts_warn_once_and_fall_back_to_dejavu(tmp_path, monkeypatch):
    monkeypatch.setattr(viz, "FONTS", tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        viz.use_house_style()
    font_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(font_warnings) == 1
    assert "assets/fonts" in str(font_warnings[0].message)
    assert matplotlib.rcParams["font.family"] == ["DejaVu Sans"]


def test_the_outcome_palette_passes_the_checks_on_the_sheet():
    assert failures([viz.FAILED, viz.SURVIVED], SHEET) == []
    # censored is the neutral slot: no hue by design, but it must still read against the sheet
    assert contrast(viz.CENSORED, SHEET) >= 3.0


def test_the_model_palette_passes_the_checks_on_the_sheet():
    hues = [viz.MODEL_COLORS[name] for name in ("logistic", "trees", "networks")]
    assert failures(hues, SHEET) == []
    for neutral in ("classic", "blend"):
        assert contrast(viz.MODEL_COLORS[neutral], SHEET) >= 3.0


def test_an_exhibit_carries_its_number_title_and_note_as_its_label():
    viz.use_house_style()
    fig = plt.figure()
    fig.add_subplot().plot([0, 1])
    viz.exhibit(fig, "Leverage climbs", "Watch the last three years.")
    assert fig.get_label() == "Exhibit 1: Leverage climbs. Watch the last three years."


def test_a_notebook_shows_the_whole_exhibit_with_its_label_as_alt_text():
    import base64
    import io

    from PIL import Image

    viz.use_house_style()
    fig = plt.figure(figsize=(viz.EXHIBIT_WIDTH_IN, 3))
    fig.add_subplot().plot([0, 1])
    viz.exhibit(fig, "Narrow content", "A tight crop would trim the margins.")
    data, metadata = viz.notebook_png(fig)
    assert Image.open(io.BytesIO(base64.b64decode(data))).width == 1200
    assert metadata == {"alt": "Exhibit 1: Narrow content. A tight crop would trim the margins."}
