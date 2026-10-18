"""Modern components on the new tokens: inputs, seg control, nav indicator, cards, stars, bulk bar, toast."""
import re
from pathlib import Path

STATIC = Path(__file__).parent.parent / "photocure" / "ui" / "static"
BRAND_HEX = ("#3b82f6", "#60a5fa", "#93c5fd", "#2563eb", "#f472b6", "#ec4899",
             "#f59e0b", "#d97706", "#fbbf24", "#38bdf8", "#818cf8")


def _raw_css() -> str:
    return (STATIC / "style.css").read_text(encoding="utf-8")


def _css() -> str:
    return re.sub(r"/\*.*?\*/", "", _raw_css(), flags=re.S)


def _html() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


def _rules():
    return re.findall(r"([^{}]+)\{([^{}]*)\}", _css())


def _body(selector_part: str) -> str:
    exact = selector_part.endswith(" {")
    key = selector_part[:-2] if exact else selector_part
    for selector, body in _rules():
        if (selector.strip() == key) if exact else (key in selector):
            return body
    raise AssertionError(f"no rule with selector {'equal to' if exact else 'containing'} {key!r}")


def test_inputs_have_focus_ring_and_data_uri_chevron():
    assert "data:image/svg+xml" in _body("select {")
    assert "var(--focus)" in _body("input:focus-visible")


def test_seg_markup_and_css():
    tag = re.search(r'<div[^>]*id="scrub-presets"[^>]*>', _html()).group(0)
    assert "seg" in re.search(r'class="([^"]*)"', tag).group(1).split()
    presets = re.findall(r'<button[^>]*data-preset="[^"]*"[^>]*>', _html())
    assert len(presets) == 5 and all("seg-btn" in p and "preset-btn" in p for p in presets)
    assert "transition: transform" in _body(".seg::before")
    assert "var(--link)" in _body(".seg-btn.active")


def test_hl_view_still_select():
    assert '<select id="hl-view"' in _html()


def test_nav_indicator_present():
    nav = re.search(r"<nav>(.*?)</nav>", _html(), re.S).group(1)
    assert 'id="nav-indicator"' in nav
    assert "transition: transform" in _body(".nav-indicator")


def test_cards_have_brand_hairline():
    for selector, body in _rules():
        if ".sect::before" in selector:
            for part in (".step::before", ".dup-group::before", ".diffpanel::before", ".accel::before"):
                assert part in selector, part
            assert "var(--grad-brand)" in body
            return
    raise AssertionError("shared ::before hairline rule not found")


def test_bulk_bar_not_pinned_to_sidebar_width():
    assert "252px" not in _body(".bulk {")
    assert "backdrop-filter" in _body(".bulk {")


def test_toast_has_status_rail():
    assert "var(--ok)" in _body(".toast.ok")
    assert "var(--err)" in _body(".toast.err")
    assert "border-left" in _body(".toast {")


def test_star_squares_and_digit_captions():
    assert _html().count('class="star-n"') == 5
    assert "var(--commit" in _body(".lb-rate .star.on")


def test_progress_fill_and_gauge_use_brand_gradient():
    assert "var(--grad-brand)" in _body(".progress-fill {")
    assert "#38bdf8" not in _html() and "#ec4899" not in _html()
    assert "var(--blue-400)" in _html() and "var(--pink-500)" in _html()


def test_modal_cards_are_18px():
    assert "border-radius: 18px" in _body(".modal-card {")
    assert "border-radius: 18px" in _body(".confirm-content {")


def test_no_brand_hex_outside_root_blocks():
    css = _css()
    css = re.sub(r"(?m)^:root(\[data-theme=\"dark\"\])?\s*\{.*?\n\}", "", css, count=2, flags=re.S)
    css = re.sub(r"(?m)^\s*--c-[\w-]+:[^;]*;", "", css).lower()
    left = [h for h in BRAND_HEX if h in css]
    assert not left, f"hard-coded brand hex outside :root: {left}"
