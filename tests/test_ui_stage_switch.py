"""The header switch: New, Library and Trash, each with a count, active segment in the brand pink."""
from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "photocure" / "ui" / "static"


def _read(name: str) -> str:
    return (STATIC / name).read_text()


def test_switch_has_three_stages_with_icons_and_counts():
    html = _read("index.html")
    sw = re.search(r'<div class="top-mode-switch" id="stage-switch".*?</div>', html, re.S).group(0)
    assert 'role="group"' in sw
    for stage in ("new", "library", "trash"):
        assert f'data-stage="{stage}"' in sw, stage
        assert f'data-stage-count="{stage}"' in sw, stage
    assert sw.count("<svg") == 3
    assert [t for t in re.findall(r"<span>(\w+)</span>", sw)] == ["New", "Library", "Trash"]
    assert "btn-mode-" not in html and "badge-mode-" not in html


def _stages(html: str, tab: str) -> set[str]:
    m = re.search(rf'<button[^>]*data-tab="{tab}"[^>]*>', html)
    assert m, tab
    s = re.search(r'data-stages="([^"]*)"', m.group(0))
    assert s, f"{tab} declares no stages"
    return set(s.group(1).split())


def test_nav_buttons_declare_which_stages_show_them():
    html = _read("index.html")
    for tab in ("categories", "people", "bursts"):
        assert _stages(html, tab) == {"new", "library", "trash"}, tab
    for tab in ("search", "highlights", "events", "utility"):
        assert _stages(html, tab) == {"new", "library"}, tab
    for tab in ("library", "uncertain", "duplicates", "junk", "actions"):
        assert _stages(html, tab) == {"new"}, tab


def test_js_uses_stage_state_and_set_stage():
    js = _read("app.js")
    assert "state.scope" not in js
    assert "function setStage" in js
    assert "btn-mode-" not in js and "badge-mode-" not in js
    assert "stage_counts" in js and "data-stage-count" in js
    assert "document.body.dataset.stage" in js
    assert "missing_ids" in js or "libraryMissing" in js


def test_active_segment_is_pink_with_navy_ink():
    css = _read("style.css")
    block = re.search(r"\.top-mode-btn\.active\s*\{([^}]*)\}", css).group(1)
    assert "var(--accent)" in block and "var(--accent-ink)" in block
    assert "250, 59, 145" in block  # the pink glow
