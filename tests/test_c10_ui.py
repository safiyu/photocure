"""C10: Import layout rework (hero gauge, blob cards, dock, sidebar) static assertions."""
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
STATIC = ROOT / "photocure" / "ui" / "static"


def _css() -> str:
    return (STATIC / "style.css").read_text(encoding="utf-8")


def _html() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


def _js() -> str:
    return (STATIC / "app.js").read_text(encoding="utf-8")


def _step(html: str, step_id: str) -> str:
    m = re.search(rf'<li class="step" id="{step_id}">(.*?)</li>', html, re.DOTALL)
    assert m, f"{step_id} <li> not found"
    return m.group(1)


# ------------------------------------------------------------------ triple chevrons

def test_step_flow_has_three_chevrons():
    html = _html()
    for step_id in ("step-scan", "step-infer"):
        assert _step(html, step_id).count('class="chev"') == 3, f"{step_id} needs 3 chevrons"
    rule = re.search(r"\.step-flow\s+\.chev:nth-child\(3\)\s*\{([^}]*)\}", _css())
    assert rule, ".step-flow .chev:nth-child(3) rule missing"
    assert ".7s" in rule.group(1), "third chevron must be delayed .7s to extend the stagger"


# ------------------------------------------------------------------ sidebar

def _nav(html: str) -> str:
    m = re.search(r"<nav>(.*?)</nav>", html, re.DOTALL)
    assert m, "<nav> not found"
    return m.group(1)


def test_nav_finish_group_dropped_and_actions_moved_into_review():
    nav = _nav(_html())
    assert "<h2>Finish</h2>" not in nav, "the Finish group header must be gone"
    review = re.search(r"<h2>Review</h2>(.*)", nav, re.DOTALL)
    assert review, "Review navgroup not found"
    assert 'data-tab="actions"' in review.group(1), "Apply and undo must live in the Review group"
    assert "<span>Apply and undo</span>" in review.group(1)


def test_nav_junk_label_is_short():
    nav = _nav(_html())
    assert "<span>Junk</span>" in nav
    assert "<span>Junk review</span>" not in nav
    assert 'data-tab="junk"' in nav, "route key must not change"


def test_dark_mode_row_has_visible_label():
    html = _html()
    assert re.search(r'class="theme-text"[^>]*>\s*Dark Mode\s*<', html), "visible Dark Mode text missing"
    assert 'id="theme-toggle"' in html, "the accessible switch must stay"


def test_device_chip_is_short_with_full_detail_in_title():
    js = _js()
    assert '.textContent = "Running on " + label' not in js, "chip text must be the short GPU/CPU form"
    render = re.search(r"function renderChip\(\)\s*\{(.*?)\n  \}", js, re.DOTALL)
    assert render, "renderChip not found"
    body = render.group(1)
    assert "chip.title" in body and "Running on" in body, "full detail belongs in the title"
    assert "aria-label" in body, "chip must keep an accessible name with the full detail"
    chip_rule = re.search(r"\.devchip\s*\{([^}]*)\}", _css())
    assert chip_rule and "width: 100%" not in chip_rule.group(1), ".devchip must be a compact pill"


# ------------------------------------------------------------------ action and summary dock

def _library_tab(html: str) -> str:
    m = re.search(r'<section id="tab-library"[^>]*>(.*?)</section>', html, re.DOTALL)
    assert m, "#tab-library not found"
    return m.group(1)


def test_dock_markup_holds_summary_and_actions():
    tab = _library_tab(_html())
    dock = re.search(r'<div class="dock"[^>]*>(.*)', tab, re.DOTALL)
    assert dock, ".dock not found in the Import tab"
    body = dock.group(1)
    for needle in ('id="dock-total"', 'id="btn-sort-dock"', "data-sortnow",
                   'id="btn-ingest-all"', 'id="btn-import-reset"'):
        assert needle in body, f"{needle} must live inside the dock"
    assert "Run All Three Steps" in body and "Reset Library" in body and "Sort Photos" in body
    assert 'class="runall"' not in tab, "the old .runall row must be gone"
    assert 'id="library-status"' in tab, "status caption must be kept for the JS that writes to it"


def test_dock_css_is_a_pill_bar():
    css = _css()
    rule = re.search(r"\.dock\s*\{([^}]*)\}", css)
    assert rule and "border-radius" in rule.group(1), ".dock must be a rounded bar"
    assert ".runall" not in css, "dead .runall rules must be removed"


def test_dock_total_follows_overview_and_sort_disables_when_empty():
    js = _js()
    render = re.search(r"function renderComp\(ov\)\s*\{(.*?)\n  \}", js, re.DOTALL)
    assert render, "renderComp not found"
    body = render.group(1)
    assert "dock-total" in body, "renderComp must write the dock total"
    assert re.search(r"btn-sort-dock[^;]*;?[^;]*disabled\s*=\s*!total", body, re.DOTALL), (
        "Sort Photos must be disabled when the library is empty"
    )


def test_reset_restores_button_markup_not_plain_text():
    js = _js()
    assert 'b.textContent = b.id === "btn-import-reset"' not in js, (
        "restoring plain text would wipe the dock button icon and casing"
    )
    assert "innerHTML = saved" in js
