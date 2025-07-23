# ruff: noqa: E501
"""Category dropdowns are alphabetical, and the grid fades in when its photos are replaced."""
import re
from pathlib import Path

STATIC = Path(__file__).parent.parent / "photocure" / "ui" / "static"


def _js() -> str:
    return (STATIC / "app.js").read_text(encoding="utf-8")


def _css() -> str:
    return re.sub(r"/\*.*?\*/", "", (STATIC / "style.css").read_text(encoding="utf-8"), flags=re.S)


def test_category_dropdowns_are_sorted_by_label_with_all_first():
    body = _js().split("function fillCatFilters", 1)[1].split("\n  }\n", 1)[0]
    assert "localeCompare" in body and "catLabel" in body
    assert "<option value=\"\">All categories</option>' + opts" in body


def test_number_key_shortcuts_keep_the_original_category_order():
    js = _js()
    assert "state.cats = sortCats(" in js, "shortcuts and menus still follow the fixed order"


def test_card_and_image_fade_keyframes_play_once_and_only_after_a_reset():
    css = _css()
    for name in ("card-in", "img-in"):
        assert re.search(r"@keyframes\s+" + name + r"\b", css), name
    card = re.search(r"\.grid\.anim-in\s+\.card\s*\{([^}]*)\}", css).group(1)
    img = re.search(r"\.grid\.anim-in\s+\.thumb img\s*\{([^}]*)\}", css).group(1)
    assert "card-in" in card and "img-in" in img
    assert "infinite" not in card + img
    assert not re.search(r"(?m)^\.card\s*\{[^}]*animation", css), "plain edits must not replay the fade"


def test_reset_flashes_the_grid_and_always_clears_the_class():
    js = _js()
    assert "function flashGrid(" in js
    assert "g.reset = () => {" in js and "flashGrid(host)" in js.split("g.reset = () => {", 1)[1].split("};", 1)[0]
    fn = js.split("function flashGrid(", 1)[1].split("\n  }\n", 1)[0]
    assert "anim-in" in fn and "setTimeout" in fn and "classList.remove" in fn
