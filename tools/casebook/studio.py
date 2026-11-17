"""The studio: a headless Chromium page that draws a component at twice its size and saves a PNG.

Needs the author tools (playwright, Pillow) and, once, `uv run python -m playwright install chromium`.
"""

import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
CSS = HERE / "design.css"
WIDTH = 1200  # a notebook picture, on screen
LANDING_WIDTH = 830  # the column GitHub shows a README in
SCALE = 2  # drawn at twice the size, so it stays sharp on a high-resolution display

# Run in the page before a component is drawn. It names a part that runs outside the picture and
# words wider than their own box. Shapes inside an <svg> are placed by number and are not checked.
FIT_CHECK = """() => {
  const root = document.querySelector('.cb');
  const box = root.getBoundingClientRect();
  const say = (el) => '"' + (el.textContent.trim() || el.className).slice(0, 48) + '"';
  const found = new Set();
  for (const el of root.querySelectorAll('*')) {
    if (el.closest('svg')) continue;
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) continue;
    if (r.right > box.right + 0.5 || r.left < box.left - 0.5
        || r.bottom > box.bottom + 0.5 || r.top < box.top - 0.5)
      found.add(say(el) + ' runs outside the picture');
    else if (el.children.length === 0 && el.clientWidth > 0 && el.scrollWidth > el.clientWidth + 1)
      found.add(say(el) + ' is wider than its box');
  }
  return [...found].slice(0, 4);
}"""


class Studio:
    """One headless Chromium page. Use as a context manager and draw as many components as needed."""

    def __enter__(self):
        # Imported here so the HTML builders work on a machine without a browser.
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        try:
            self._browser = self._playwright.chromium.launch()
        except Exception as error:
            self._playwright.stop()
            raise RuntimeError(
                "Chromium is not available to Playwright. "
                "Run `uv run python -m playwright install chromium` once, then try again."
            ) from error
        self._page = self._browser.new_page(
            viewport={"width": WIDTH, "height": 800}, device_scale_factor=SCALE
        )
        self._folder = tempfile.TemporaryDirectory()
        return self

    def __exit__(self, *error):
        self._browser.close()
        self._playwright.stop()
        self._folder.cleanup()

    def png(self, html: str, out: Path, width: int = WIDTH, colors: int = 256) -> Path:
        """Draw one component `width` px wide and save it at twice that, with at most `colors` colors."""
        page_file = Path(self._folder.name) / "component.html"
        page_file.write_text(
            f'<!doctype html><html><head><meta charset="utf-8">'
            f'<link rel="stylesheet" href="{CSS.as_uri()}"></head>'
            f'<body style="width:{width}px">{html}</body></html>',
            encoding="utf-8",
        )
        self._page.set_viewport_size({"width": width, "height": 800})
        self._page.goto(page_file.as_uri())
        self._page.evaluate("document.fonts.ready")
        block = self._page.locator(".cb").first
        height = block.bounding_box()["height"]
        if height != int(height):
            raise ValueError(
                f"{out.name} is {height} px tall: its height must be a whole number of pixels"
            )
        faults = self._page.evaluate(FIT_CHECK)
        if faults:
            raise ValueError(
                f"{out.name} does not fit: {'; '.join(faults)}. Shorten the words; never shrink the type."
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        block.screenshot(path=str(out))
        image = Image.open(out).convert("RGB")
        quantized = image.quantize(
            colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
        )
        quantized.save(out, optimize=True)
        return out
