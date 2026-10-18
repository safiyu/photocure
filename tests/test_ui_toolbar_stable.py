"""The filter toolbar and sliders do not move when the page gains or loses its scrollbar."""
import re
from pathlib import Path

STATIC = Path(__file__).parent.parent / "photocure" / "ui" / "static"


def _css() -> str:
    return re.sub(r"/\*.*?\*/", "", (STATIC / "style.css").read_text(encoding="utf-8"), flags=re.S)


def _rule(selector: str) -> str:
    m = re.search(r"(?m)^" + re.escape(selector) + r"\s*\{([^}]*)\}", _css())
    assert m, selector
    return m.group(1)


def test_page_reserves_the_scrollbar_gutter():
    body = _rule("html")
    assert "scrollbar-gutter: stable" in body and "overflow-y: scroll" in body


def test_scrubber_is_a_fixed_grid_not_a_wrapping_row():
    body = _rule(".scrubber")
    assert "display: grid" in body and "repeat(2, minmax(0, 1fr))" in body
    rng = _rule(".scrubber input[type=range]")
    assert "min(300px" not in rng and "width: 100%" in rng


def test_row_count_has_a_stable_width():
    body = _rule(".rowcount")
    assert "min-width" in body and "text-align: right" in body
