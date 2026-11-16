"""Kept apart from test_design.py: a live Studio there holds a Playwright loop this test cannot share."""

import pytest


def test_a_missing_chromium_says_how_to_install_it(monkeypatch):
    from playwright.sync_api import BrowserType

    from tools.casebook.studio import Studio

    def no_browser(self, **kwargs):
        raise RuntimeError("Executable doesn't exist")

    monkeypatch.setattr(BrowserType, "launch", no_browser)
    with pytest.raises(RuntimeError, match="python -m playwright install chromium"):
        Studio().__enter__()
