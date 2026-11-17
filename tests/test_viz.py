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


def test_long_tick_labels_stay_inside_the_exhibit():
    # Tick labels are set in the wider mono face, so the layout must be fitted with it in place.
    viz.use_house_style()
    fig, ax = plt.subplots(figsize=(viz.EXHIBIT_WIDTH_IN, 3))
    ax.barh(["Transportation, Communications, Electric, Gas, and Sanitary Services"], [1])
    viz.exhibit(fig, "Long labels", "A note.")
    fig.canvas.draw()
    assert min(label.get_window_extent().x0 for label in ax.get_yticklabels()) >= 0


def test_plain_number_is_a_number_a_reader_can_say():
    assert viz.plain_number(1_000_000) == "1,000,000"
    assert viz.plain_number(100) == "100"
    assert viz.plain_number(0.01) == "0.01"
    assert viz.plain_number(1e-8) == "0.00000001"
    assert viz.plain_number(0) == "0"
    assert viz.plain_number(-10_000) == "-10,000"
    assert viz.plain_number(2.5) == "2.5"


def _tick_labels(fig, axis) -> list[str]:
    fig.canvas.draw()
    return [label.get_text() for label in axis.get_ticklabels() if label.get_text()]


def test_a_log_axis_shows_plain_numbers_not_powers_of_ten():
    viz.use_house_style()
    fig, ax = plt.subplots()
    ax.plot([0.01, 1_000_000], [1, 2])
    viz.log_axis(ax, "x")
    labels = _tick_labels(fig, ax.xaxis)
    assert {"0.01", "1", "100", "10,000", "1,000,000"} <= set(labels)
    assert not any("10^" in label or "mathdefault" in label for label in labels)


def test_a_log_axis_under_one_decade_labels_no_minor_tick_in_powers_of_ten():
    viz.use_house_style()
    fig, ax = plt.subplots()
    ax.plot([0.4, 8], [1, 2])
    viz.log_axis(ax, "x")
    fig.canvas.draw()
    minor = [label.get_text() for label in ax.xaxis.get_minorticklabels()]
    assert not any(minor)
    assert "1" in _tick_labels(fig, ax.xaxis)


def test_a_log_axis_can_be_the_y_axis():
    viz.use_house_style()
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 1e-6])
    viz.log_axis(ax, "y")
    assert {"1", "0.01", "0.0001", "0.000001"} <= set(_tick_labels(fig, ax.yaxis))


def test_a_symlog_axis_shows_plain_numbers_on_both_sides_of_zero():
    viz.use_house_style()
    fig, ax = plt.subplots()
    ax.plot([-10_000, 10_000], [1, 2])
    viz.symlog_axis(ax, "x")
    ax.set_xticks([-10_000, -100, 0, 100, 10_000])
    assert _tick_labels(fig, ax.xaxis) == ["-10,000", "-100", "0", "100", "10,000"]


def test_an_exhibit_takes_a_colorbar_at_its_default_size():
    fig, ax = plt.subplots(figsize=(8, 4))
    image = ax.imshow([[0, 1], [2, 3]], aspect="auto")
    bar = fig.colorbar(image, ax=ax)
    viz.exhibit(fig, "A heatmap", "The bar runs most of the chart's height.")
    assert bar.ax.get_position().height > 0.75 * ax.get_position().height


def test_the_accent_passes_the_checks_beside_every_hue():
    hues = [viz.SURVIVED, *[viz.MODEL_COLORS[name] for name in ("trees", "networks")]]
    assert failures([*hues, viz.ACCENT], SHEET) == []
    assert failures([viz.FAILED, viz.ACCENT], SHEET) == []


def test_every_model_input_and_item_has_a_name_a_reader_can_say():
    from corp_default.data import COLUMN_MAP
    from corp_default.features import feature_names

    for name in [*feature_names(), *COLUMN_MAP.values()]:
        shown = viz.label(name)
        assert "_" not in shown and shown[0].isupper(), name
    assert viz.label("mve_to_liabilities") == "Market value / liabilities"
    assert viz.label("d2_liabilities_to_assets") == "Liabilities / assets, 2-year change"
    assert viz.label("g1_market_cap") == "Market value, 1-year growth"
    assert viz.label("loss_2y_ago") == "Loss two years ago"
