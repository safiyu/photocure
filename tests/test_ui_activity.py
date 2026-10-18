# ruff: noqa: E501
"""Activity timeline: status icon, bold title with detail, time on the right, filter and clear."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).parent.parent / "photocure" / "ui" / "static"


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def _split(msg: str) -> dict:
    src = re.search(r"function splitLog\([^)]*\)\s*\{.*?\n  \}", _read("app.js"), re.S)
    assert src, "splitLog not found in app.js"
    out = subprocess.run(["node", "-e", f"{src.group(0)}\nconsole.log(JSON.stringify(splitLog({json.dumps(msg)})))"],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


@needs_node
def test_split_log_separates_a_headline_from_its_detail():
    r = _split("Scan finished: 3,194 files seen, 3,194 new, 162 duplicates in 9.97s")
    assert r == {"title": "Scan finished", "detail": "3,194 files seen, 3,194 new, 162 duplicates in 9.97s"}


@needs_node
def test_split_log_leaves_plain_messages_and_paths_alone():
    assert _split("Scanning /home/me/Pictures...") == {"title": "Scanning /home/me/Pictures...", "detail": ""}
    assert _split("C:\\Photos is not readable")["detail"] == ""
    long_prefix = "x" * 60 + ": something"
    assert _split(long_prefix)["detail"] == ""


def test_both_activity_panels_use_the_shared_component():
    html = _read("index.html")
    assert html.count('<section class="activity"') == 2
    for needle in ('data-act-filter="problems"', "data-act-clear", 'role="log"', "activity-count"):
        assert html.count(needle) == 2, needle
    assert "Nothing has run yet" in html


def test_log_rows_are_built_as_icon_body_time_and_the_old_pills_are_gone():
    js, css = _read("app.js"), _read("style.css")
    body = js.split("function logTo", 1)[1].split("\n  }\n", 1)[0]
    for cls in ("log-ico", "log-title", "log-detail", "log-time", "running"):
        assert cls in body, cls
    assert "splitLog(" in body
    assert "time-pill" not in js + css and "active-highlight" not in js + css


def test_filter_and_clear_are_wired():
    js = _read("app.js")
    assert "function bindActivity(" in js and "bindActivity();" in js
    body = js.split("function bindActivity", 1)[1].split("\n  }\n", 1)[0]
    assert "data-act-filter" in body and "data-act-clear" in body and "dataset.filter" in body


def test_activity_styles_cover_rows_filter_and_empty_state():
    css = re.sub(r"/\*.*?\*/", "", _read("style.css"), flags=re.S)
    for sel in (".activity {", ".logline {", ".log-ico {", ".log-time {", ".logline-empty {"):
        assert sel in css, sel
    assert re.search(r'\.activity\[data-filter="problems"\]\s+\.logline:not\(\.err\):not\(\.warn\)', css)
