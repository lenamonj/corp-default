"""The holdout stays sealed: chapters 1 to 6 never pass the argument that opens it."""

from pathlib import Path

CHAPTERS = Path(__file__).resolve().parents[1] / "chapters"


def unsealed(chapters: Path) -> list[str]:
    """The sources of chapters 1 to 6 that mention the argument that opens the holdout."""
    return [
        path.name
        for path in sorted(chapters.glob("0[1-6]*.py"))
        if "unseal" in path.read_text(encoding="utf-8")
    ]


def test_a_chapter_that_opens_the_holdout_early_is_named(tmp_path):
    (tmp_path / "02_the_event.py").write_text("frame = development(panel)\n", encoding="utf-8")
    (tmp_path / "03_the_signal.py").write_text("holdout(panel, unseal=True)\n", encoding="utf-8")
    (tmp_path / "07_the_verdict.py").write_text("holdout(panel, unseal=True)\n", encoding="utf-8")
    assert unsealed(tmp_path) == ["03_the_signal.py"]


def test_chapters_one_to_six_never_open_the_holdout():
    assert unsealed(CHAPTERS) == []
