"""C9: chevron connectors + unified sect card shape — static assertions."""
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


def test_step_flow_markup_inside_step1_and_step2():
    html = _index()
    scan_block = re.search(
        r'<li class="step" id="step-scan">(.*?)</li>', html, re.DOTALL
    )
    assert scan_block, "step-scan <li> not found"
    assert 'class="step-flow"' in scan_block.group(1), "step-scan must contain .step-flow"
    assert 'aria-hidden="true"' in scan_block.group(1), ".step-flow must be aria-hidden"
    assert scan_block.group(1).count('class="chev"') == 2, "step-scan .step-flow must hold 2 chevrons"

    infer_block = re.search(
        r'<li class="step" id="step-infer">(.*?)</li>', html, re.DOTALL
    )
    assert infer_block, "step-infer <li> not found"
    assert 'class="step-flow"' in infer_block.group(1), "step-infer must contain .step-flow"

    cluster_block = re.search(
        r'<li class="step" id="step-cluster">(.*?)</li>', html, re.DOTALL
    )
    assert cluster_block and 'class="step-flow"' not in cluster_block.group(1), (
        "step-cluster (last) must NOT contain a trailing .step-flow"
    )


def test_step_flow_css_rule_and_keyframes():
    css = _style()
    assert re.search(r"\.step-flow\s*\{", css), ".step-flow rule missing"
    assert "@keyframes chev-flow" in css, "@keyframes chev-flow missing"
    assert re.search(r"\.step-flow\s+\.chev\s*\{", css), ".step-flow .chev rule missing"


def test_step_flow_state_aware_colours():
    css = _style()
    done = re.search(r"\.step\.done\s+\.step-flow\s*\{([^}]*)\}", css)
    assert done and "var(--ok-sage)" in done.group(1), (
        ".step.done .step-flow must use --ok-sage"
    )
    active_sibling = re.search(
        r"\.step:has\(\s*\+\s*\.step\.active\s*\)\s+\.step-flow\s*\{([^}]*)\}", css
    )
    assert active_sibling and "var(--accent-electric)" in active_sibling.group(1), (
        ".step:has(+ .step.active) .step-flow must use --accent-electric"
    )
    done_pos = re.search(r"\.step\.done\s+\.step-flow", css).start()
    active_pos = re.search(
        r"\.step:has\(\s*\+\s*\.step\.active\s*\)\s+\.step-flow", css
    ).start()
    assert active_pos > done_pos, (
        "active-pointing rule must come AFTER done rule so it wins on same specificity"
    )


def test_step_flow_reduced_motion_freezes_animation():
    css = _style()
    reduced = re.findall(
        r"@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{([^{}]*(?:\{[^}]*\}[^{}]*)*)\}",
        css,
    )
    assert any(".step-flow" in block and "animation: none" in block for block in reduced), (
        ".step-flow .chev must have animation disabled under prefers-reduced-motion"
    )


def test_step_flow_rotates_on_mobile():
    css = _style()
    tablet = re.search(
        r"@media\s*\(max-width:\s*960px\)\s*\{([^{}]*(?:\{[^}]*\}[^{}]*)*)\}", css
    )
    assert tablet and re.search(r"\.step-flow[^{]*\{[^}]*rotate\(90deg\)", tablet.group(1)), (
        ".step-flow must rotate 90deg at <= 960px so chevrons point down when stacked"
    )


def test_sect_padding_matches_step():
    css = _style()
    sect = re.search(r"\.sect\s*\{([^}]*)\}", css)
    assert sect, ".sect rule not found"
    assert re.search(r"padding:\s*20px\s+22px", sect.group(1)), (
        ".sect padding must match .step (20px 22px)"
    )
    assert "var(--r-lg)" in sect.group(1), ".sect must keep --r-lg radius"
    assert "1px solid var(--line)" in sect.group(1), ".sect must keep 1px --line border"


def test_sect_header_has_accent_marker():
    css = _style()
    head = re.search(r"\.sect\s*>\s*h2\s*\{([^}]*)\}", css)
    assert head, ".sect > h2 rule not found"
    body = head.group(1)
    assert "border-left" in body and "var(--accent-electric)" in body, (
        ".sect > h2 must carry a left accent bar in --accent-electric"
    )
    assert "padding-left" in body, ".sect > h2 must indent after the accent bar"


def test_sect_danger_border_still_wins():
    css = _style()
    danger = re.search(r"\.sect\.danger\s*\{([^}]*)\}", css)
    assert danger, ".sect.danger rule must still exist"
    assert "var(--err)" in danger.group(1) or "var(--danger-crimson)" in danger.group(1), (
        ".sect.danger must keep its crimson/err border"
    )
    sect_pos = re.search(r"\.sect\s*\{", css).start()
    danger_pos = re.search(r"\.sect\.danger\s*\{", css).start()
    assert danger_pos > sect_pos, ".sect.danger must come AFTER .sect so its border wins"
