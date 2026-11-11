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
