# ruff: noqa: E501
"""Burst detail shows the group matrix: a banner, per-frame badges, subject dots and a score bar."""
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


def _summary(group):
    src = re.search(r"function groupSummary\([^)]*\)\s*\{.*?\n  \}", _read("app.js"), re.S)
    assert src, "groupSummary not found"
    names = {1: "a.jpg", 2: "b.jpg"}
    call = f"groupSummary({json.dumps(group)}, (id) => ({json.dumps(names)})[id])"
    out = subprocess.run(["node", "-e", f"{src.group(0)}\nconsole.log(JSON.stringify({call}))"],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


@needs_node
def test_perfect_frame_is_celebrated():
    r = _summary({"status": "perfect", "best_id": 1, "partner_id": None})
    assert r["tone"] == "ok" and "Everyone's eyes open" in r["text"] and "a.jpg" in r["text"]


@needs_node
def test_composite_names_both_frames():
    r = _summary({"status": "composite", "best_id": 1, "partner_id": 2})
    assert r["tone"] == "warn" and "a.jpg" in r["text"] and "b.jpg" in r["text"] and "combine" in r["text"]


@needs_node
def test_best_effort_says_no_frame_is_perfect():
    r = _summary({"status": "best_effort", "best_id": 2, "partner_id": None})
    assert r["tone"] == "warn" and "closest" in r["text"] and "b.jpg" in r["text"]


@needs_node
def test_single_subject_bursts_show_nothing():
    assert _summary({"status": "not_group", "best_id": None, "partner_id": None}) is None


def test_burst_detail_renders_banner_badges_dots_and_bar():
    js = _read("app.js")
    body = js.split("async function showBurst", 1)[1].split("\n  }\n", 1)[0]
    for needle in ("data.group", "gm-banner", "gm-dot", "gm-bar", "All eyes open", "Composite pair", "Closest"):
        assert needle in body, needle


def test_group_styles_exist():
    css = re.sub(r"/\*.*?\*/", "", _read("style.css"), flags=re.S)
    for sel in (".gm-banner {", ".gm-banner.ok {", ".gm-banner.warn {", ".gm-dots {", ".gm-dot {", ".gm-bar {"):
        assert sel in css, sel
    for state in ("open", "possible", "closed", "missing"):
        assert f'.gm-dot[data-s="{state}"]' in css, state
