import re
from pathlib import Path

CSS = (Path(__file__).resolve().parents[1] / "photocure" / "ui" / "static" / "style.css").read_text(encoding="utf-8")


def _rule(selector: str) -> str:
    m = re.search(r"(?m)^" + re.escape(selector) + r"\s*\{([^}]*)\}", CSS)
    assert m, selector
    return m.group(1)


def test_toolbar_selects_keep_the_arrow_settings_of_the_base_select_rule():
    """The `background` shorthand would reset the no-repeat, size and position of the arrow."""
    body = _rule(".toolbar select")
    assert not re.search(r"(?<![-\w])background\s*:", body)
    assert "background-color" in body
