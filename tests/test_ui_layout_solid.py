# ruff: noqa: E501
"""The page frame is centered and shared by header and content, and step cards match each other."""
import re
from pathlib import Path

STATIC = Path(__file__).parent.parent / "photocure" / "ui" / "static"


def _css() -> str:
    return re.sub(r"/\*.*?\*/", "", (STATIC / "style.css").read_text(encoding="utf-8"), flags=re.S)


def _rule(selector: str) -> str:
    m = re.search(r"(?m)^" + re.escape(selector) + r"\s*\{([^}]*)\}", _css())
    assert m, selector
    return m.group(1)


def test_main_is_centered_and_wide_enough_for_big_screens():
    body = _rule("main")
    assert "max-width: 1680px" in body and "margin-inline: auto" in body and "width: 100%" in body


def test_header_shares_the_same_frame_as_the_content():
    body = _rule(".topbar")
    assert "margin: 0 auto 24px" in body and "max-width: 1616px" in body and "width: calc(100% - 64px)" in body
    for query in ("max-width: 960px", "max-width: 640px"):
        block = re.search(r"@media\s*\(" + re.escape(query) + r"\)\s*\{(.*?)\n\}", _css(), re.S).group(1)
        assert re.search(r"\.topbar\s*\{[^}]*width: auto", block), query


def test_step_cards_have_a_visible_border_and_the_active_one_is_not_scaled():
    assert "var(--line)" in _rule(".step")
    active = _rule(".step.active")
    assert "scale(" not in active and "translateY(-2px)" in active and "z-index" in active
