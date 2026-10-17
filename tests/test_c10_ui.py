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


# ------------------------------------------------------------------ hero header and gauge

def _hero(html: str) -> str:
    m = re.search(r'<header class="topbar hero">(.*?)</header>', html, re.DOTALL)
    assert m, '<header class="topbar hero"> not found'
    return m.group(1)


def test_progress_card_lives_in_the_hero_and_keeps_error_text():
    hero = _hero(_html())
    assert 'id="progress-card"' in hero, "progress card must sit inside the hero header"
    assert 'id="progress-msg"' in hero, "errors render in #progress-msg, which must stay visible"
    assert 'id="btn-stop-progress"' in hero, "Stop button must move with the card"
    assert _html().count('id="progress-card"') == 1, "progress card must not be duplicated"


def test_hero_gauge_is_an_accessible_progressbar():
    hero = _hero(_html())
    tag = re.search(r'<div[^>]*id="hero-gauge"[^>]*>', hero)
    assert tag, "#hero-gauge not found in the hero"
    for needle in ('role="progressbar"', "aria-valuemin", "aria-valuemax", "aria-valuenow", "aria-label", "hidden"):
        assert needle in tag.group(0), f"gauge needs {needle}"
    assert 'id="gauge-arc"' in hero and 'pathLength="100"' in hero


def test_old_topbar_pill_is_gone_everywhere():
    for name, text in (("html", _html()), ("js", _js())):
        for stale in ("topbar-progress", "btn-stop-topbar", "topbar-fill", "topbar-spinner", "topbar-status-icon"):
            assert stale not in text, f"{stale} still referenced in {name}"
    assert "topbar-track" not in _css() and "topbar-fill" not in _css()


def test_render_progress_drives_the_gauge_and_hides_it_when_idle():
    js = _js()
    assert 'show($("#hero-gauge"), visible)' in js, "gauge must hide when no task is visible"
    assert "gauge-arc" in js and "aria-valuenow" in js
    assert re.search(r"Phase\\s\+\\d\+", js), "phase label must be parsed from the progress message"


def test_gauge_css_and_hero_is_not_sticky():
    css = _css()
    arc = re.search(r"(?m)^\.gauge-arc\s*\{([^}]*)\}", css)
    assert arc and "stroke-dasharray" in arc.group(1) and "transition" in arc.group(1)
    assert re.search(r"prefers-reduced-motion:\s*reduce\)\s*\{[^@]*transition-duration:\s*\.001ms", css), (
        "the blanket reduced-motion rule must cover the gauge transition"
    )
    topbar = re.search(r"\.topbar\s*\{([^}]*)\}", css)
    assert topbar and "position: sticky" not in topbar.group(1), "the hero card scrolls with the page"


# ------------------------------------------------------------------ blob cards and dark gradient

def test_step_cards_have_organic_elliptical_radii():
    css = _css()
    rule = re.search(r"(?m)^\.step\s*\{([^}]*)\}", css)
    assert rule, ".step base rule not found"
    radius = re.search(r"border-radius:\s*([^;]+);", rule.group(1))
    assert radius and "/" in radius.group(1), "blob shape needs elliptical radii (a '/' in border-radius)"
    assert "var(--r-lg)" not in radius.group(1)


def test_active_step_is_larger_with_scooped_cta():
    css = _css()
    active = re.search(r"\.step\.active\s*\{([^}]*)\}", css)
    assert active and "scale(" in active.group(1) and "z-index" in active.group(1), (
        "the active card must be raised and scaled"
    )
    cta = re.search(r"\.step\.active\s+\.actions\s+\.btn\.primary\s*\{([^}]*)\}", css)
    assert cta and "border-radius" in cta.group(1) and "gradient" in cta.group(1), (
        "the active CTA needs the scooped gradient shape"
    )
    tablet = re.search(r"@media\s*\(max-width:\s*960px\)\s*\{(.*?)\n\}", css, re.DOTALL)
    assert tablet and re.search(r"\.step\.active\s*\{[^}]*transform:\s*none", tablet.group(1)), (
        "stacked layout must not scale the active card past the page padding"
    )


def test_dark_theme_has_gradient_background_and_glass_cards():
    css = _css()
    body = re.search(r':root\[data-theme="dark"\]\s+body\s*\{([^}]*)\}', css)
    assert body and "gradient" in body.group(1), "dark body needs a gradient background"
    glass = re.search(r':root\[data-theme="dark"\]\s+\.step[^{]*\{([^}]*)\}', css)
    assert glass and "gradient" in glass.group(1), "dark cards need a glass gradient fill"
    assert "border-color" not in glass.group(1), (
        "glass rule must not override the active step's electric border"
    )
    base_body = re.search(r"(?m)^body\s*\{([^}]*)\}", css)
    assert base_body and "var(--table)" in base_body.group(1), "light theme body stays flat"


# ------------------------------------------------------------------ legibility fixes found in the browser

def test_gauge_track_is_visible_against_dark_cards():
    rule = re.search(r"(?m)^\.gauge-track\s*\{([^}]*)\}", _css())
    assert rule, "dedicated .gauge-track rule not found"
    assert "stroke" in rule.group(1) and "var(--well)" not in rule.group(1), (
        "--well is nearly the card colour in dark mode, so the empty arc vanished"
    )


def test_version_pill_is_readable_in_dark_theme():
    rule = re.search(r':root\[data-theme="dark"\]\s+\.version-pill\s*\{([^}]*)\}', _css())
    assert rule and re.search(r"color:\s*#fff", rule.group(1)), (
        "indigo text on the indigo dark pill is illegible; the hero makes it prominent"
    )


def test_chevrons_stay_above_the_raised_active_card():
    css = _css()
    flow = re.search(r"(?m)^\.step-flow\s*\{([^}]*)\}", css)
    active = re.search(r"\.step\.active\s*\{([^}]*)\}", css)
    assert flow and active
    z_flow = int(re.search(r"z-index:\s*(\d+)", flow.group(1)).group(1))
    z_active = int(re.search(r"z-index:\s*(\d+)", active.group(1)).group(1))
    assert z_flow > z_active, "the raised active card hid two of the three chevrons leading into it"
