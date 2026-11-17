"""The computable palette checks of the dataviz method, in Python, for the test suite.

Lightness band, chroma floor, contrast against the surface, and the normal-vision floor. The
color-blind separation check was run with the dataviz skill's own validator; the ledger has it.
"""

import itertools
import math

LIGHT_BAND = (0.43, 0.77)
CHROMA_FLOOR = 0.10
CONTRAST_FLOOR = 3.0
NORMAL_FLOOR = 15.0  # OKLab Delta E x 100


def _linear(channel: float) -> float:
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def _rgb(hex_color: str) -> tuple[float, float, float]:
    return tuple(_linear(int(hex_color[i : i + 2], 16) / 255) for i in (1, 3, 5))


def oklab(hex_color: str) -> tuple[float, float, float]:
    r, g, b = _rgb(hex_color)
    l_ = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m_ = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s_ = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def contrast(a: str, b: str) -> float:
    def luminance(hex_color: str) -> float:
        r, g, bl = _rgb(hex_color)
        return 0.2126 * r + 0.7152 * g + 0.0722 * bl

    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def failures(palette: list[str], surface: str) -> list[str]:
    """Every check a categorical palette fails on a light surface; empty when it passes."""
    found = []
    for color in palette:
        lightness, a, b = oklab(color)
        if not LIGHT_BAND[0] <= lightness <= LIGHT_BAND[1]:
            found.append(f"{color} lightness {lightness:.3f}")
        if math.hypot(a, b) < CHROMA_FLOOR:
            found.append(f"{color} chroma {math.hypot(a, b):.3f}")
        if contrast(color, surface) < CONTRAST_FLOOR:
            found.append(f"{color} contrast {contrast(color, surface):.2f}")
    for first, second in itertools.combinations(palette, 2):
        distance = 100 * math.dist(oklab(first), oklab(second))
        if distance < NORMAL_FLOOR:
            found.append(f"{first} and {second} Delta E {distance:.1f}")
    return found
