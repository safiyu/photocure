# ruff: noqa: E501
"""Utility tab: nav item, kind chips, grid, count, re-check button, and Junk without screenshots."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).parent.parent / "photocure" / "ui" / "static"
needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def _query(kind: str) -> str:
    src = re.search(r"function utilityQuery\([^)]*\)\s*\{.*?\n  \}", _read("app.js"), re.S)
    assert src, "utilityQuery not found"
    return subprocess.run(["node", "-e", f"{src.group(0)}\nconsole.log(utilityQuery({kind!r}))"],
                          capture_output=True, text=True, check=True).stdout.strip()


@needs_node
def test_all_kinds_ask_for_the_whole_family_and_one_kind_asks_for_its_category():
    assert _query("") == "group=utility"
    assert _query("slide") == "category=slide"
    assert _query("utility") == "category=utility"


def test_nav_item_panel_and_chips_exist():
    html = _read("index.html")
    nav = re.search(r"<nav>(.*?)</nav>", html, re.S).group(1)
    assert 'data-tab="utility"' in nav and 'data-count="utility"' in nav and "<span>Utility</span>" in nav
    tab = re.search(r'<section id="tab-utility"[^>]*>(.*?)</section>', html, re.S).group(1)
    for needle in ('id="utilitygrid"', 'id="utilitygrid-more"', 'id="utilitycount"', 'id="btn-rescore"'):
        assert needle in tab, needle
    kinds = re.findall(r'data-ukind="([^"]*)"', tab)
    assert kinds == ["", "screenshot", "document", "slide", "utility"]
    assert 'id="i-inbox"' in html


def test_junk_filter_no_longer_offers_screenshots_or_documents():
    junk = re.search(r'<select id="junkfilter".*?</select>', _read("index.html"), re.S).group(0)
    assert 'value="screenshot"' not in junk and 'value="document"' not in junk
    assert 'value="graphic"' in junk and 'value="duplicate"' in junk


def test_js_wires_grid_counts_titles_and_rescore():
    js = _read("app.js")
    assert 'makeGrid("utility"' in js and "group=utility" in js
    assert "utility: [" in js.split("const TITLES", 1)[1].split("};", 1)[0]
    assert 'set("utility"' in js.split("function renderCounts", 1)[1].split("\n  }\n", 1)[0]
    assert "state.utility = new Set(" in js
    assert '"utility"' in js.split("const CAT_ORDER", 1)[1].split("];", 1)[0]
    assert "Label or sign" in js
    handler = js.split('$("#btn-rescore").onclick', 1)[1].split("};", 1)[0]
    assert '"/api/rescore"' in handler and "followJob(" in handler and "sortNow(" in handler and "grids.utility.reset()" in handler


def test_junk_empty_state_no_longer_promises_screenshots_or_documents():
    js = _read("app.js")
    text = re.search(r'makeGrid\("junk".*?empty: \{[^}]*\}', js, re.S).group(0)
    assert "creenshot" not in text and "ocument" not in text


def test_chip_and_category_styles_exist():
    css = re.sub(r"/\*.*?\*/", "", _read("style.css"), flags=re.S)
    assert "--c-utility:" in css
    assert ".chip-btn {" in css and '.chip-btn[aria-pressed="true"]' in css
