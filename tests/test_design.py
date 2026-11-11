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
    for token in ("#121316", "#F7F7F5", "#16181D", "#D9482B", "#3F5F7F", "#FF6A4D", "#8FB1D1"):
        assert token in css
    for family in ("Archivo", "Public Sans", "JetBrains Mono"):
        assert f'font-family: "{family}"' in css
