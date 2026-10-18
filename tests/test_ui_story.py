# ruff: noqa: E501
"""Event stories view: a balanced selection with scale badges, a coverage line and a fallback note."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).parent.parent / "photocure" / "ui" / "static"
needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def _coverage_text(cov: dict) -> str:
    src = re.search(r"function coverageText\([^)]*\)\s*\{.*?\n  \}", _read("app.js"), re.S)
    names = re.search(r"const SCALE_NAMES = \{.*?\};", _read("app.js"), re.S)
    assert src and names, "coverageText or SCALE_NAMES not found"
    return subprocess.run(["node", "-e", f"{names.group(0)}\n{src.group(0)}\nconsole.log(coverageText({json.dumps(cov)}))"],
                          capture_output=True, text=True, check=True).stdout.strip()


@needs_node
def test_coverage_line_names_each_scale_with_its_count_in_story_order():
    assert _coverage_text({"medium": 3, "wide": 1, "closeup": 2}) == "1 wide, 2 close-up, 3 medium"
    assert _coverage_text({}) == ""


def test_story_option_exists_next_to_the_other_views():
    html = _read("index.html")
    select = re.search(r'<select id="hl-view".*?</select>', html, re.S).group(0)
    assert 'value="story"' in select and "balanced" in select.lower()
    assert 'value="events"' in select and 'value="all"' in select


def test_story_view_asks_for_a_balanced_selection_and_shows_the_fallback_note():
    js = _read("app.js")
    body = js.split("async function renderEventHighlights", 1)[1].split("\n  }\n", 1)[0]
    assert "balanced=true" in body and "per_event=" in body
    for needle in ("ev-cover", "coverageText(", "badge scale", "data.note"):
        assert needle in body, needle
    assert js.count("isEventView()") >= 3, "every place that checked for the events view must accept stories"


def test_story_styles_exist():
    css = re.sub(r"/\*.*?\*/", "", _read("style.css"), flags=re.S)
    for sel in (".ev-cover {", ".scale-chip {", ".badge.scale {"):
        assert sel in css, sel
    for scale in ("wide", "closeup", "action", "ambient", "medium"):
        assert f'.badge.scale[data-scale="{scale}"]' in css, scale
