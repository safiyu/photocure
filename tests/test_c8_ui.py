"""C8: UI/UX modern revamp — dark-slate workstation static assertions.

Covers:
  * C8 design tokens present in BOTH the light and dark root blocks
    (surfaces, electric-blue accent + glow, sage / crimson status, visualizer),
    with the baseline C5 tokens left intact (zero backward-compat regression).
  * Hardware-style theme switch accessibility (#theme-toggle label + role="switch"
    checkbox, focusable, aria-labelled).
  * Equalizer spectrum markup (exactly 24 bars in #progress-card) plus the CSS
    that scopes the animation to the active card and freezes it under
    prefers-reduced-motion.
  * 3-step workflow state classes (.step.done / .step.active / .step.locked).
  * Tablet / mobile responsive breakpoints (steps stack to 1 column at <= 960px).
  * Zero external (http/https) URLs anywhere in the static bundle.
"""
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
STATIC = ROOT / "photocure" / "ui" / "static"
STYLE = STATIC / "style.css"
INDEX = STATIC / "index.html"


def _style() -> str:
    return STYLE.read_text(encoding="utf-8")


def _index() -> str:
    return INDEX.read_text(encoding="utf-8")


def _root_block(css: str, selector: str) -> str:
    """Return the declaration body of the first rule matching `selector`."""
    match = re.search(r"""\s""" + re.escape(selector) + r"""\s*\{([^}]+)\}""", css)
    assert match, f"CSS rule block not found: {selector}"
    return match.group(1)


# ------------------------------------------------------------------ design tokens

# C8 tokens that must exist with the same value in BOTH themes (additive; the
# baseline C5 tokens are asserted separately so we prove they were not touched).
_C8_THEME_TOKENS = [
    "--canvas-bg",
    "--sheet-subtle",
    "--sheet-border",
    "--sheet-border-light",
    "--accent-electric",
    "--accent-electric-hover",
    "--accent-glow",
    "--accent-pill",
    "--ok-sage",
    "--ok-sage-bg",
    "--ok-sage-border",
    "--danger-crimson",
    "--danger-crimson-bg",
    "--danger-crimson-border",
    "--viz-bar",
    "--viz-bar-faint",
]


def test_c8_tokens_present_in_both_themes():
    """Every C8 surface / accent / status / visualizer token is declared in both
    the light and dark root blocks."""
    css = _style()
    for selector in (":root", ':root[data-theme="dark"]'):
        block = _root_block(css, selector)
        for token in _C8_THEME_TOKENS:
            assert re.search(rf"(?m)^\s*{token}\s*:", block), (
                f"C8 token {token} not declared in {selector}"
            )


def test_c8_token_values_match_spec():
    """Spot-check the spec-mandated hex / rgba values for the key C8 tokens."""
    css = _style()
    light = _root_block(css, ":root")
    dark = _root_block(css, ':root[data-theme="dark"]')

    # Dark theme is the target of the revamp — exact spec values.
    for token, value in [
        ("--canvas-bg", "#0b0e14"),
        ("--sheet-subtle", "#0f141f"),
        ("--sheet-border", "#1e2638"),
        ("--accent-electric", "#3b82f6"),
        ("--accent-electric-hover", "#60a5fa"),
        ("--accent-glow", "0 0 20px rgba(59, 130, 246, 0.28)"),
        ("--accent-pill", "rgba(59, 130, 246, 0.16)"),
        ("--ok-sage", "#22c55e"),
        ("--ok-sage-bg", "#14291f"),
        ("--danger-crimson", "#ef4444"),
        ("--danger-crimson-bg", "#261619"),
        ("--viz-bar", "#38bdf8"),
        ("--viz-bar-faint", "rgba(56, 189, 248, 0.22)"),
    ]:
        assert re.search(rf"(?m)^\s*{token}\s*:\s*{re.escape(value)}\s*;", dark), (
            f"Dark-theme {token} should be {value}"
        )

    # Light theme keeps high-contrast slate surfaces.
    for token, value in [
        ("--canvas-bg", "#f1f5f9"),
        ("--accent-electric", "#3b82f6"),
        ("--accent-electric-hover", "#2563eb"),
    ]:
        assert re.search(rf"(?m)^\s*{token}\s*:\s*{re.escape(value)}\s*;", light), (
            f"Light-theme {token} should be {value}"
        )


def test_baseline_c5_tokens_preserved():
    """The C5 baseline tokens that other suites assert must survive the revamp."""
    css = _style()
    light = _root_block(css, ":root")
    dark = _root_block(css, ':root[data-theme="dark"]')

    for token, value in [
        ("--table", "#f8fafc"),
        ("--sheet", "#ffffff"),
        ("--ink", "#0f172a"),
        ("--mut", "#475569"),
        ("--faint", "#5f6e84"),
    ]:
        assert re.search(rf"(?m)^\s*{token}\s*:\s*{re.escape(value)}\s*;", light), (
            f"Light-theme baseline {token} was altered"
        )

    for token, value in [
        ("--table", "#0c0e12"),
        ("--sheet", "#14171f"),
        ("--ink", "#f1f3f7"),
        ("--mut", "#949eb2"),
        ("--faint", "#7e8a9f"),
    ]:
        assert re.search(rf"(?m)^\s*{token}\s*:\s*{re.escape(value)}\s*;", dark), (
            f"Dark-theme baseline {token} was altered"
        )


def test_terminal_tokens_are_dark_fixed():
    """The terminal feed is always-dark: its tokens hold dark values in the
    light root block (they must NOT be re-declared as theme-flipping values)."""
    css = _style()
    light = _root_block(css, ":root")

    for token, value in [
        ("--terminal-bg", "#0e121a"),
        ("--terminal-border", "#1a2233"),
    ]:
        assert re.search(rf"(?m)^\s*{token}\s*:\s*{re.escape(value)}\s*;", light), (
            f"Terminal token {token} should be dark-fixed at {value}"
        )


# ------------------------------------------------------------------ theme switch

def test_theme_switch_markup_and_accessibility():
    """#theme-toggle is a focusable <label> wrapping a role="switch" checkbox and
    a decorative .slider knob — accessible via Tab / Space / Enter."""
    html = _index()

    label_match = re.search(
        r'<label[^>]*id="theme-toggle"[^>]*>', html
    )
    assert label_match, "No <label id=\"theme-toggle\"> found in index.html"
    label_tag = label_match.group(0)
    assert 'tabindex="0"' in label_tag, "theme-toggle label must be keyboard-focusable"
    assert "aria-label" in label_tag, "theme-toggle label must carry an aria-label"

    switch_match = re.search(
        r'<input[^>]*id="theme-switch"[^>]*>', html
    )
    assert switch_match, "No <input id=\"theme-switch\"> found in index.html"
    switch_tag = switch_match.group(0)
    assert 'type="checkbox"' in switch_tag, "theme-switch must be a checkbox"
    assert 'role="switch"' in switch_tag, "theme-switch must declare role=\"switch\""
    assert "aria-checked" in switch_tag, "theme-switch must declare aria-checked"

    assert re.search(r'class="switch"', html), "switch container class not found"
    assert re.search(r'class="slider"', html), "switch slider knob not found"


def test_theme_switch_css_present():
    """The sliding switch has real styles and a reduced-motion guard on its knob."""
    css = _style()
    assert re.search(r"\.switch\s*\{", css), ".switch base rule not found"
    assert re.search(r"\.switch\s+\.slider\s*\{", css), ".switch .slider rule not found"
    assert (
        re.search(
            r"@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{[^}]*\.switch\s+\.slider",
            css,
        )
        is not None
    ), "switch slider transition must be disabled under prefers-reduced-motion"


# ------------------------------------------------------------------ equalizer spectrum

def test_spectrum_markup_has_exactly_24_bars():
    """#progress-card embeds a .spectrum-bars container with exactly 24 bars and
    a live .progress-ratio counter."""
    html = _index()

    assert 'id="progress-card"' in html, "#progress-card not found"
    assert 'class="spectrum-bars"' in html, ".spectrum-bars container not found"
    assert 'class="progress-ratio"' in html, ".progress-ratio counter not found"

    bar_count = len(re.findall(r'class="spectrum-bar"', html))
    assert bar_count == 24, f"Expected exactly 24 .spectrum-bar spans, found {bar_count}"


def test_spectrum_css_animation_scoped_and_guarded():
    """The eq-pulse animation only runs while #progress-card.active, and the whole
    wave is frozen under prefers-reduced-motion (no continuous GPU/CPU work)."""
    css = _style()

    assert re.search(r"\.spectrum-bars\s*\{", css), ".spectrum-bars rule not found"
    assert re.search(r"\.spectrum-bar\s*\{", css), ".spectrum-bar rule not found"
    assert "@keyframes eq-pulse" in css, "eq-pulse keyframes not found"

    # Animation is scoped to the active card (not global).
    assert re.search(
        r"#progress-card\.active\s+\.spectrum-bar\s*\{[^}]*eq-pulse",
        css,
    ), "eq-pulse animation must be scoped to #progress-card.active"

    # Reduced-motion guard freezes the bars entirely.
    assert re.search(
        r"@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{[^}]*"
        r"\.spectrum-bar[^}]*animation:\s*none\s*!important",
        css,
    ), "spectrum bars must be frozen under prefers-reduced-motion"


# ------------------------------------------------------------------ 3-step workflow

def test_step_state_classes_present():
    """The 3-step grid carries .step.done / .step.active (blue glow) / .step.locked
    (dimmed) state styles."""
    css = _style()

    # Grid is the spec's responsive 3-column track.
    assert re.search(
        r"\.steps\s*\{[^}]*repeat\(3,\s*minmax\(0,\s*1fr\)\)", css
    ), ".steps must use repeat(3, minmax(0, 1fr))"

    # Done: sage circular badge.
    assert re.search(r"\.step\.done\s+\.step-num\s*\{", css), ".step.done badge rule not found"
    assert "var(--ok-sage)" in css, "step.done badge should use --ok-sage"

    # Active: electric-blue border + atmospheric glow.
    active = re.search(r"\.step\.active\s*\{([^}]*)\}", css)
    assert active, ".step.active rule not found"
    assert "var(--accent-electric)" in active.group(1), ".step.active must use --accent-electric"
    assert "var(--accent-glow)" in active.group(1), ".step.active must use --accent-glow"

    # Locked: dimmed.
    locked = re.search(r"\.step\.locked\s*\{([^}]*)\}", css)
    assert locked, ".step.locked rule not found"
    assert re.search(r"opacity:\s*0\.55", locked.group(1)), ".step.locked must dim to 0.55"


def test_step_grid_collapses_to_single_column_on_tablet():
    """At <= 960px the 3-step grid stacks into a single column."""
    css = _style()

    tablet = re.search(
        r"@media\s*\(max-width:\s*960px\)\s*\{(.*)", css, re.S
    )
    assert tablet, "960px tablet media query not found"
    assert re.search(r"\.steps\s*\{[^}]*grid-template-columns:\s*1fr", tablet.group(1)), (
        ".steps must collapse to a single column at <= 960px"
    )


# ------------------------------------------------------------------ zero external URLs

def test_zero_external_urls_across_static_bundle():
    """No static file references a remote http/https URL (self-hosted only)."""
    offenders = []
    for f in sorted(STATIC.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() not in {".html", ".css", ".js", ".mjs", ".json", ".svg"}:
            continue
        text = f.read_text(encoding="utf-8")
        urls = re.findall(r"https?://(?!github\.com/)", text)
        if urls:
            offenders.append((str(f.relative_to(STATIC)), urls))

    assert not offenders, f"External URLs found in static bundle: {offenders}"


def test_css_uses_only_self_hosted_font_urls():
    """Every url() in the stylesheet points at a local /static/ asset, never remote."""
    css = _style()

    remote = [
        m
        for m in re.findall(r"url\(\s*['\"]?([^'\")]+)", css)
        if re.match(r"^https?://", m) or m.startswith("//")
    ]
    assert not remote, f"Remote url() references found: {remote}"

    for ref in re.findall(r"url\(\s*['\"]?(/static/[^'\")]+)", css):
        # CSS url()s use the served route /static/... which maps to the static
        # dir itself on disk — strip the /static/ prefix to resolve the file.
        rel = ref[len("/static/"):]
        assert (STATIC / rel).exists(), f"Referenced local asset missing: {ref}"


if __name__ == "__main__":
    test_c8_tokens_present_in_both_themes()
    test_c8_token_values_match_spec()
    test_baseline_c5_tokens_preserved()
    test_terminal_tokens_are_dark_fixed()
    test_theme_switch_markup_and_accessibility()
    test_theme_switch_css_present()
    test_spectrum_markup_has_exactly_24_bars()
    test_spectrum_css_animation_scoped_and_guarded()
    test_step_state_classes_present()
    test_step_grid_collapses_to_single_column_on_tablet()
    test_zero_external_urls_across_static_bundle()
    test_css_uses_only_self_hosted_font_urls()
    print("✓ All C8 tests passed")
