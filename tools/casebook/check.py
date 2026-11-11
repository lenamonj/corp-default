"""Check one built chapter: its parts and minutes, its tests, and every picture on sheets to look at.

    uv run python -m tools.casebook.check 03 --sheet artifacts/sheets/03
    uv run python -m tools.casebook.check 03 --text   # the notebook as plain text, to read

A notebook file is mostly picture data, so it is never read by opening the file: --text prints its
words, with each picture as its alt text, and --sheet saves the pictures themselves. Looking at the
sheets is still a person's job. The exit code is 0 only when the chapter's tests pass.
"""

import argparse
import base64
import io
import re
import subprocess
import sys
from pathlib import Path

import nbformat
from PIL import Image

from corp_default.data import REPO

NOTEBOOKS = REPO / "notebooks"
PICTURE = re.compile(r"!\[([^\]]*)\]\(data:image/png;base64,([^)]+)\)")
SHEET_WIDTH, SHEET_HEIGHT, GAP = 1200, 2400, 24  # a sheet is short enough to read in one look
OUTPUT_LIMIT = 600  # characters of a printed output shown by --text


def _image(encoded: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


def pictures(notebook: Path) -> list[tuple[str, Image.Image]]:
    """Every drawn picture and every exhibit of the notebook, in reading order, as (alt, image)."""
    found = []
    for cell in nbformat.read(notebook, as_version=4).cells:
        if cell.cell_type == "markdown":
            found += [(alt, _image(data)) for alt, data in PICTURE.findall(cell.source)]
        elif cell.cell_type == "code":
            found += [
                (output.metadata["image/png"]["alt"], _image(output.data["image/png"]))
                for output in cell.outputs
                if "image/png" in output.get("data", {})
            ]
    return found


def parts(notebook: Path) -> list[str]:
    """Each part as its opener states it: its place, its goal, its minutes."""
    return [alt for alt, _ in pictures(notebook) if alt.startswith("Part ")]


def save_sheets(found: list[tuple[str, Image.Image]], folder: Path, stem: str) -> list[Path]:
    """The pictures stacked 1200 px wide on white, cut into sheets short enough to look at."""
    folder.mkdir(parents=True, exist_ok=True)
    scaled = [
        image.resize((SHEET_WIDTH, round(image.height * SHEET_WIDTH / image.width)), Image.LANCZOS)
        for _, image in found
    ]
    sheets, batch, height = [], [], GAP
    for image in [*scaled, None]:
        if image is None or (batch and height + image.height + GAP > SHEET_HEIGHT):
            sheet = Image.new("RGB", (SHEET_WIDTH, height), "#FFFFFF")
            y = GAP
            for part in batch:
                sheet.paste(part, (0, y))
                y += part.height + GAP
            path = folder / f"{stem}_sheet{len(sheets) + 1:02d}.png"
            sheet.save(path)
            sheets.append(path)
            batch, height = [], GAP
        if image is not None:
            batch.append(image)
            height += image.height + GAP
    return sheets


def as_text(notebook: Path) -> str:
    """The notebook as plain text: each cell numbered, each picture as its alt text, outputs cut short."""
    lines = []
    for place, cell in enumerate(nbformat.read(notebook, as_version=4).cells):
        source = PICTURE.sub(lambda found: f"[picture: {found.group(1)}]", cell.source)
        lines += [f"--- cell {place}, {cell.cell_type}", source.strip()]
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "image/png" in data:
                lines.append(f"[exhibit: {output.metadata['image/png']['alt']}]")
                continue
            words = output.get("text") or data.get("text/plain") or "\n".join(output.traceback)
            words = re.sub(r"\x1b\[[0-9;]*m", "", words).strip()
            kind = ", a table" if "text/html" in data else ""
            cut = " ..." if len(words) > OUTPUT_LIMIT else ""
            lines.append(f"[output{kind}] {words[:OUTPUT_LIMIT]}{cut}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check one built chapter of the casebook.")
    parser.add_argument("chapter", type=int, help="the chapter number, such as 03")
    parser.add_argument("--sheet", type=Path, help="a folder for sheets of the chapter's pictures")
    parser.add_argument("--text", action="store_true", help="print the notebook as text and stop")
    args = parser.parse_args()
    built = sorted(NOTEBOOKS.glob(f"{args.chapter:02d}_*.ipynb"))
    if len(built) != 1:
        print(f"notebooks/ has {len(built)} notebooks for chapter {args.chapter}: build it first")
        return 1
    notebook = built[0]
    if args.text:
        sys.stdout.reconfigure(encoding="utf-8")
        print(as_text(notebook))
        return 0
    found = pictures(notebook)
    exhibits = sum(alt.startswith("Exhibit ") for alt, _ in found)
    size = notebook.stat().st_size / 1e6
    print(f"{notebook.name}: {len(found)} pictures, {exhibits} of them exhibits, {size:.2f} MB")
    for line in parts(notebook):
        print("  ", line)
    if args.sheet:
        for path in save_sheets(found, args.sheet, notebook.stem):
            print("sheet:", path)
    tests = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_chapter_standard.py",
            "-q",
            "-k",
            notebook.stem,
        ],
        cwd=REPO,
        capture_output=True,
        check=False,  # a failing test is reported below, not raised
        text=True,
    )
    print(tests.stdout.strip().splitlines()[-1] if tests.stdout.strip() else tests.stderr.strip())
    if tests.returncode != 0:
        print(tests.stdout[-3000:])
    return tests.returncode


if __name__ == "__main__":
    raise SystemExit(main())
