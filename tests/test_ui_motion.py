# ruff: noqa: E501
"""Subtle, reduced-motion-safe animation: keyframes, gated loops, and the JS helpers."""
import re
from pathlib import Path

STATIC = Path(__file__).parent.parent / "photocure" / "ui" / "static"
KEYFRAMES = ("panel-in", "pop", "ov-in", "ov-out", "ov-fade-in", "ov-fade-out", "toast-in", "toast-out", "tick", "bulk-in", "bulk-out")


def _css() -> str:
    return re.sub(r"/\*.*?\*/", "", (STATIC / "style.css").read_text(encoding="utf-8"), flags=re.S)


def _js() -> str:
    return (STATIC / "app.js").read_text(encoding="utf-8")


def _fn(js: str, name: str) -> str:
    m = re.search(r"(?:async )?function " + re.escape(name) + r"\([^)]*\)\s*\{(.*?)\n  \}", js, re.S)
    assert m, f"function {name} not found"
    return m.group(1)


def test_keyframes_defined():
    css = _css()
    for name in KEYFRAMES:
        assert re.search(r"@keyframes\s+" + re.escape(name) + r"\b", css), f"@keyframes {name} missing"


def test_panel_enter_ends_visible_and_does_not_loop():
    css = _css()
    frames = re.search(r"@keyframes\s+panel-in\s*\{((?:[^{}]|\{[^{}]*\})*)\}", css, re.S).group(1)
    to = frames.split("to", 1)[1]
    assert "opacity: 1" in to and "transform: none" in to
    rule = re.search(r"\.panel\.enter\s*\{([^}]*)\}", css).group(1)
    assert "animation: panel-in" in rule and "infinite" not in rule and "forwards" not in rule


def test_chevrons_always_flow_with_a_stagger():
    css = _css()
    assert re.search(r"\.step-flow\s+\.chev\s*\{", css)
    for n, delay in ((1, ""), (2, ".35s"), (3, ".7s")):
        rule = re.search(r"(?m)^\.step-flow \.chev:nth-child\(" + str(n) + r"\)\s*\{([^}]*)\}", css)
        assert rule, f"ungated .step-flow .chev:nth-child({n}) rule missing"
        assert "chev-flow" in rule.group(1) and "infinite" in rule.group(1) and delay in rule.group(1)


def test_chevrons_are_centered_in_the_gap_between_cards():
    css = _css()
    flow = re.search(r"(?m)^\.step-flow\s*\{([^}]*)\}", css).group(1)
    assert "right: -8px" in flow, "gap is 16px, so the flow sits 8px past the card edge"
    tablet = re.search(r"@media\s*\(max-width:\s*960px\)\s*\{(.*?)\n\}", css, re.S).group(1)
    assert re.search(r"\.step-flow\s*\{[^}]*bottom:\s*-8px", tablet)


def test_helpers_exist():
    js = _js()
    for name in ("reducedMotion", "afterAnim", "openOverlay", "closeOverlay", "positionNavIndicator"):
        assert re.search(r"function " + name + r"\(", js), f"{name} missing"


def test_after_anim_has_fallbacks():
    body = _fn(_js(), "afterAnim")
    assert "reducedMotion()" in body, "must resolve immediately under reduced motion"
    assert "setTimeout" in body, "must not depend on animationend alone"
    assert "animationend" in body


def test_close_overlay_is_idempotent_and_reopen_safe():
    body = _fn(_js(), "closeOverlay")
    assert "closing" in body and "_closeSeq" in body
    open_body = _fn(_js(), "openOverlay")
    assert "_closeSeq" in open_body and "closing" in open_body


def test_overlays_use_the_helpers():
    js = _js()
    for sel in ("#picker", "#help", "#similar-panel", "#lightbox"):
        assert f'$("{sel}").hidden = true' not in js, f"{sel} still hidden directly"
        assert f'closeOverlay($("{sel}"))' in js, f"{sel} must close via closeOverlay"
        assert f'openOverlay($("{sel}"))' in js, f"{sel} must open via openOverlay"
    confirm = _fn(js, "showConfirmDialog")
    assert "closeOverlay(modal)" in confirm and "openOverlay(modal)" in confirm


def test_toast_exit_has_a_timeout_fallback():
    body = _fn(_js(), "toast")
    assert 'classList.add("out")' in body and "afterAnim(" in body and ".remove()" in body


def test_tab_switch_and_progress_hooks():
    js = _js()
    show = _fn(js, "showTab")
    assert 'classList.remove("enter")' in show and 'classList.add("enter")' in show and "positionNavIndicator()" in show
    assert 'toggle("task-running"' in _fn(js, "renderProgress")
    assert "tick" in _fn(js, "renderCounts")


def test_blanket_reduced_motion_rule_kept():
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    assert re.search(r"prefers-reduced-motion:\s*reduce\)\s*\{[^@]*transition-duration:\s*\.001ms", css)


def test_stale_tab_fade_timer_cannot_drop_a_newer_fade():
    show = _fn(_js(), "showTab")
    assert "_enterSeq" in show
    assert show.count("_enterSeq") >= 3, "the token must be bumped and compared in the timer"


def test_no_old_amber_or_pill_leftovers_in_the_log_styles():
    css = (STATIC / "style.css").read_text(encoding="utf-8").lower()
    assert "#b45309" not in css and "time-pill" not in css


def test_star_focus_ring_keeps_the_star_radius():
    rule = re.search(r"\.lb-rate \.star:focus-visible\s*\{([^}]*)\}", _css()).group(1)
    assert "border-radius" not in rule and "outline: 2px solid var(--focus)" in rule
