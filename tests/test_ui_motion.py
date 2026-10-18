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


def test_chevron_loop_is_gated_to_running_or_next_step():
    css = _css()
    assert re.search(r"\.step-flow\s+\.chev\s*\{", css)
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        if "chev-flow" in body and "infinite" in body:
            assert "task-running" in selector or ".step.done:has(+ .step.active)" in selector, selector.strip()
    third = re.search(r"\.step-flow\s+\.chev:nth-child\(3\)\s*\{([^}]*)\}", css)
    assert third and ".7s" in third.group(1)


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
