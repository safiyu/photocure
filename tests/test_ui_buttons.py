# ruff: noqa: E501
"""Unified pill button family: base, variants, states, role mapping in markup and JS."""
import re
from pathlib import Path

from tests.test_ui_tokens import _contrast, _tok

STATIC = Path(__file__).parent.parent / "photocure" / "ui" / "static"
VARIANTS = ("primary", "accent", "commit", "secondary", "ghost", "danger", "sm", "lg", "loading")


def _css() -> str:
    return re.sub(r"/\*.*?\*/", "", (STATIC / "style.css").read_text(encoding="utf-8"), flags=re.S)


def _html() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


def _js() -> str:
    return (STATIC / "app.js").read_text(encoding="utf-8")


def _rule(selector: str) -> str:
    m = re.search(r"(?m)^" + re.escape(selector) + r"\s*\{([^}]*)\}", _css())
    assert m, f"rule {selector} not found"
    return m.group(1)


def _classes(button_id: str) -> list:
    m = re.search(r'<button[^>]*id="' + re.escape(button_id) + r'"[^>]*>', _html())
    assert m, f"#{button_id} not found"
    c = re.search(r'class="([^"]*)"', m.group(0))
    return c.group(1).split() if c else []


def test_btn_base_is_pill_and_neutral():
    body = _rule(".btn")
    assert "border-radius: 999px" in body
    assert "height: 38px" in body
    assert "var(--sheet)" in body and "var(--line)" in body


def test_variants_exist():
    css = _css()
    for v in VARIANTS:
        assert re.search(r"\.btn\." + v + r"\s*\{", css), f".btn.{v} rule missing"


def test_variants_use_role_tokens():
    for variant, role in (("primary", "primary"), ("accent", "accent"), ("commit", "commit")):
        body = _rule(f".btn.{variant}")
        assert f"var(--{role})" in body and f"var(--{role}-ink)" in body, variant


def test_states_present():
    css = _css()
    active = re.search(r"\.btn:active[^{]*\{([^}]*)\}", css)
    assert active and "scale(.97)" in active.group(1)
    focus = re.search(r"\.btn:focus-visible[^{]*\{([^}]*)\}", css)
    assert focus and "var(--focus)" in focus.group(1)
    assert re.search(r"\.btn\.loading::before\s*\{", css), "loading spinner pseudo-element missing"


def test_secondary_text_is_aa_on_its_fill_in_light():
    body = _rule(".btn.secondary")
    assert "var(--link)" in body and "var(--primary-soft)" in body
    assert _contrast(_tok("light", "--link"), _tok("light", "--primary-soft")) >= 4.5


def test_active_step_cta_keeps_dome_for_primary_and_accent():
    css = _css()
    primary = re.search(r"\.step\.active\s+\.actions\s+\.btn\.primary\s*\{([^}]*)\}", css)
    assert primary and "border-radius" in primary.group(1) and "gradient" in primary.group(1)
    accent = re.search(r"\.step\.active\s+\.actions\s+\.btn\.accent[^{]*\{([^}]*)\}", css)
    assert accent and "border-radius" in accent.group(1) and "gradient" in accent.group(1)


def test_markup_roles():
    expected = {
        "btn-scan": "accent", "btn-infer": "primary",
        "btn-ingest-all": "commit", "btn-import-reset": "danger", "btn-apply": "commit",
        "btn-rebuild": "secondary", "btn-diff": "secondary", "btn-undo": "danger",
        "btn-dup-all": "commit", "btn-save-settings": "primary", "btn-events": "primary",
        "btn-kw-remove": "danger", "btn-reset-library": "danger",
    }
    for button_id, role in expected.items():
        assert role in _classes(button_id), f"#{button_id} should be .{role}, has {_classes(button_id)}"


def test_confirm_warning_maps_to_commit():
    assert re.search(r'warning:\s*"commit"', _js()), "showConfirmDialog must map warning to commit"


def test_busy_marks_the_started_button_as_loading():
    js = _js()
    apply_busy = re.search(r"function applyBusy\(busy\)\s*\{(.*?)\n  \}", js, re.S)
    assert apply_busy and "loading" in apply_busy.group(1) and "busyBtn" in apply_busy.group(1)
    assert "state.busyBtn" in js.split("function applyBusy")[0] or "busyBtn:" in js, "state.busyBtn must exist"


def test_focus_ring_is_an_outline_that_variant_shadows_cannot_override():
    body = _rule(".btn:focus-visible")
    assert "outline: 2px solid var(--focus)" in body and "outline-offset" in body
    assert "outline: none" not in body


def test_pressed_state_comes_after_every_variant_hover_so_it_wins():
    css = _css()
    press = css.index(".btn:active:not(:disabled)")
    for hover in (".btn.primary:hover", ".btn.secondary:hover", ".btn.danger:hover", ".btn.ghost:hover"):
        assert css.index(hover) < press, f"{hover} would override the pressed state"
    body = re.search(r"\.btn:active:not\(:disabled\)[^{]*\{([^}]*)\}", css).group(1)
    assert "scale(.97)" in body and "box-shadow: none" in body


def test_closing_overlays_do_not_catch_clicks():
    css = _css()
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        if re.search(r"\.(modal|lb|similar|bulk|confirm-modal)\.closing\s*$", selector.strip().split(",")[-1]) and "animation" in body:
            assert "pointer-events: none" in body, selector.strip()
