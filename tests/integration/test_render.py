"""Every built notebook renders: pictures load, nothing spills sideways, no placeholder or traceback.

    uv run pytest -q -m integration tests/integration/test_render.py

Each notebook is converted to HTML and opened in a browser at a desktop width and a phone width.
A full-page screenshot of each goes to artifacts/render/ to be read.
"""

import re
from pathlib import Path

import nbformat
import pytest
from nbconvert import HTMLExporter
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
NOTEBOOKS = sorted((REPO / "notebooks").glob("[0-9][0-9]_*.ipynb"))
RENDER = REPO / "artifacts" / "render"
WIDTHS = {"desktop": 1280, "phone": 390}


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        chromium = playwright.chromium.launch()
        yield chromium
        chromium.close()


@pytest.fixture(scope="module", params=NOTEBOOKS, ids=lambda path: path.stem)
def page_file(request, tmp_path_factory) -> Path:
    body, _ = HTMLExporter().from_notebook_node(nbformat.read(request.param, as_version=4))
    out = tmp_path_factory.mktemp("render") / f"{request.param.stem}.html"
    out.write_text(body, encoding="utf-8")
    return out


def test_there_are_seven_notebooks():
    assert len(NOTEBOOKS) == 7


@pytest.mark.parametrize("name", WIDTHS)
def test_a_notebook_renders(browser, page_file, name):
    width = WIDTHS[name]
    page = browser.new_page(viewport={"width": width, "height": 900})
    page.goto(page_file.as_uri())
    page.wait_for_load_state("networkidle")
    RENDER.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(RENDER / f"{page_file.stem}_{name}.png"), full_page=True)
    broken = page.evaluate(
        "Array.from(document.images).filter(i => !i.complete || i.naturalWidth === 0).length"
    )
    text = page.inner_text("body")
    spill = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    page.close()
    assert broken == 0
    assert "{{" not in text
    assert "Traceback (most recent call last)" not in text
    if name == "desktop":
        assert spill <= 0, f"the page is {spill} px wider than the window"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_no_cell_holds_an_error(path):
    nb = nbformat.read(path, as_version=4)
    errors = [
        cell.source[:60]
        for cell in nb.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]
    assert errors == []


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_every_relative_link_in_a_notebook_resolves(path):
    nb = nbformat.read(path, as_version=4)
    markdown = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "markdown")
    links = [
        link
        for link in re.findall(r"\]\(([^)\s]+)\)", markdown)
        if not link.startswith(("http", "#", "data:", "attachment:"))
    ]
    assert links, "a chapter links at least to the next one"
    missing = [link for link in links if not (path.parent / link.split("#")[0]).exists()]
    assert missing == []
