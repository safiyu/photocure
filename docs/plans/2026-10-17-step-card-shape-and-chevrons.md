# Step-card shape + animated chevrons implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add animated `>>` chevron connectors between the Import page's three step cards, and bring Settings and Apply-and-undo sections up to the same visual shape as the step cards (including the Workers 3-up fields rendered as mini step-cards).

**Architecture:** Purely a static-asset refresh: additive CSS rules in `photocure/ui/static/style.css` plus two tiny markup additions in `photocure/ui/static/index.html` (chevron connectors inside step 1 and step 2 as absolutely-positioned overlays, so the existing `.steps` 3-column grid and its C8 test coverage are untouched). The chevrons are state-aware via CSS `:has()` on sibling `.step` classes; animation is scoped and respects `prefers-reduced-motion`. The `.sect` class is unified visually with `.step` (padding, header accent) so Settings and Apply-and-undo pick up the same shape without restructuring page flow.

**Tech Stack:** Vanilla HTML + CSS. SVG chevrons use the existing `#i-right` sprite symbol. Tests use pytest with the project's static-assertion pattern (`tests/test_c8_ui.py` is the reference).

**Spec:** This plan + the mockup in the 2026-10-17 conversation (user showed the current Photocure Import page as the target shape; asked for `>>` marks with animation between the step cards and for the same card shape to be applied to all Settings sections and Apply-and-undo).

## Global Constraints

- Preserve all `test_c8_ui.py` assertions: `.steps { grid-template-columns: repeat(3, minmax(0, 1fr)) }`, `.step.active` / `.step.done` / `.step.locked` rules, and the `<= 960px` stack-to-one-column media query.
- Respect `@media (prefers-reduced-motion: reduce)` — the new animation must freeze when reduced-motion is on.
- No external URLs (http/https) in static assets — `test_c8_ui.py` asserts zero external URLs in the bundle.
- Follow session coding style: comments at most 1 line.
- No new fonts, images, or JS dependencies.

## Review Focus

- **Reduced-motion users** on the Import page — the chevron animation must stop (static chevrons, no transform/opacity cycling). Pinned by Task 1 Step 1 test.
- **Mobile users at ≤ 960px** — chevrons would collide with stacked cards if left in their desktop position; they must rotate 90° and point down. Pinned by Task 1 Step 1 test.
- **Keyboard/screen-reader users** — chevron connectors are decoration, not content, so they must be `aria-hidden="true"` and not focusable. Pinned by Task 1 Step 1 test.
- **Done→active flow colour** — a chevron sandwiched between a `.done` step and an `.active` step must prefer the active-pointing colour (accent-electric) over the done-trailing colour (ok-sage), otherwise the user loses the "you are here" cue. Pinned by Task 1 Step 1 test.
- **Danger sect (Reset library, Undo)** — unifying `.sect` padding/header must not override the existing crimson border on `.sect.danger`. Pinned by Task 2 Step 1 test.

---

## File Structure

- `photocure/ui/static/index.html` — add one `<div class="step-flow">` inside `#step-scan` and inside `#step-infer`; each holds two `<svg class="chev">` using `#i-right`. No other markup changes.
- `photocure/ui/static/style.css` — three additive blocks, each marked with a C9 banner comment: (1) `.step-flow` rules + `@keyframes chev-flow` + state-aware colouring + reduced-motion / mobile overrides, (2) `.sect` shape unification + header accent, (3) `.form-grid.workers .field` mini-card treatment.
- `tests/test_c9_ui.py` — new file, mirrors `tests/test_c8_ui.py` structure (reads `style.css` and `index.html` and asserts CSS selectors / HTML elements exist).

---

### Task 1: Animated `>>` chevrons between step cards

**Files:**
- Modify: `photocure/ui/static/index.html` (inside `<li class="step" id="step-scan">` and `<li class="step" id="step-infer">`, before the closing `</li>`)
- Modify: `photocure/ui/static/style.css` (new block at end, before the responsive media queries)
- Create: `tests/test_c9_ui.py`

**Interfaces:**
- Consumes: existing sprite symbol `#i-right` (already defined in `index.html`); existing `.step`, `.step.done`, `.step.active` classes.
- Produces: new CSS class `.step-flow` (chevron container) and `.step-flow .chev` (individual chevron); new `@keyframes chev-flow`. These names are referenced by Task 2 and Task 3 tests only to confirm they weren't accidentally removed.

- [ ] **Step 1: Write the failing tests in `tests/test_c9_ui.py`**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_c9_ui.py -v`
Expected: all 5 tests FAIL (markup missing, CSS rules missing).

- [ ] **Step 3: Add `.step-flow` markup to `index.html`**

Inside `<li class="step" id="step-scan">`, immediately before its closing `</li>` (after the existing `<div class="actions">...</div>`), insert:

```html
<div class="step-flow" aria-hidden="true">
  <svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><use href="#i-right"/></svg>
  <svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><use href="#i-right"/></svg>
</div>
```

Do the same inside `<li class="step" id="step-infer">`. Do NOT add one inside `#step-cluster`.

- [ ] **Step 4: Add `.step-flow` CSS at end of `style.css`, before the responsive media queries**

The rules this block must contain (names and values pinned by Step 1 tests):

- `.step { position: relative; }` — so the chevron overlay anchors to its card.
- `.step-flow { position: absolute; right: -14px; top: 50%; transform: translate(50%, -50%); display: flex; gap: 2px; z-index: 2; pointer-events: none; color: var(--faint); }`
- `.step-flow .chev { width: 16px; height: 16px; flex: none; }`
- Two staggered animations: `.step-flow .chev:nth-child(1) { animation: chev-flow 1.4s ease-in-out infinite; }` and `.step-flow .chev:nth-child(2) { animation: chev-flow 1.4s ease-in-out .35s infinite; }`
- `@keyframes chev-flow { 0%, 100% { opacity: .25; transform: translateX(-3px); } 50% { opacity: 1; transform: translateX(2px); } }`
- Done-trailing rule BEFORE active-pointing rule: `.step.done .step-flow { color: var(--ok-sage); }` then `.step:has(+ .step.active) .step-flow { color: var(--accent-electric); }`.

Reduced-motion override — add inside the EXISTING `@media (prefers-reduced-motion: reduce)` block near the spectrum-bars rules (around current line 718–720):

```css
.step-flow .chev { animation: none !important; opacity: .55; transform: none !important; }
```

Mobile rotation — add inside the EXISTING `@media (max-width: 960px)` block (around current line 807–829), after the `.steps { grid-template-columns: 1fr; }` line:

```css
.step-flow { right: 50%; top: auto; bottom: -22px; transform: translate(50%, 50%) rotate(90deg); }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_c9_ui.py -v`
Expected: all 5 tests PASS. Also run `pytest tests/test_c8_ui.py -v` to confirm no C8 regression.

- [ ] **Step 6: Commit**

```bash
git add photocure/ui/static/index.html photocure/ui/static/style.css tests/test_c9_ui.py
git commit -m "feat(ui): animated chevron connectors between step cards"
```

---

### Task 2: Unify `.sect` visual shape with step-cards

**Files:**
- Modify: `photocure/ui/static/style.css` (edit the existing `.sect` rule around current line 487 and add a header accent)
- Modify: `tests/test_c9_ui.py` (append new tests)

**Interfaces:**
- Consumes: existing `.sect`, `.sect.danger` classes used by Settings and Apply-and-undo panels in `index.html` (unchanged markup).
- Produces: `.sect` now has `padding: 20px 22px` (matching `.step`) and a new `.sect > h2` left-accent treatment. The `.sect.danger` border rule still wins.

- [ ] **Step 1: Append failing tests to `tests/test_c9_ui.py`**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_c9_ui.py -v -k sect`
Expected: 2 of 3 FAIL (padding differs; `.sect > h2` lacks accent).

- [ ] **Step 3: Edit `.sect` rule in `style.css`**

Change the existing `.sect` padding from `22px 24px` to `20px 22px` (keep bg, border, radius, margin). Change `.sect > h2` to include `padding-left: 10px; border-left: 3px solid var(--accent-electric);` while keeping the existing font-size. Leave `.sect.danger` unchanged (already after `.sect`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_c9_ui.py -v`
Expected: all tests so far PASS. Run full UI suite: `pytest tests/test_c7_ui.py tests/test_c8_ui.py tests/test_c9_ui.py -v`
Expected: no regression.

- [ ] **Step 5: Commit**

```bash
git add photocure/ui/static/style.css tests/test_c9_ui.py
git commit -m "feat(ui): unify .sect shape with step-cards (header accent)"
```

---

### Task 3: Workers 3-up fields as mini step-cards

**Files:**
- Modify: `photocure/ui/static/style.css` (add a `.form-grid.workers .field` block after the existing `.workers .field-head` rule around line 950)
- Modify: `tests/test_c9_ui.py` (append new test)

**Interfaces:**
- Consumes: existing `.form-grid.workers .field` structure (no HTML changes).
- Produces: `.form-grid.workers .field` cards with the same border, radius, and padding tokens as `.step`.

- [ ] **Step 1: Append failing test to `tests/test_c9_ui.py`**

```python
def test_workers_fields_become_mini_cards():
    css = _style()
    rule = re.search(r"\.form-grid\.workers\s+\.field\s*\{([^}]*)\}", css)
    assert rule, ".form-grid.workers .field card rule not found"
    body = rule.group(1)
    assert "var(--sheet)" in body, "workers .field must use --sheet background"
    assert "1px solid var(--line)" in body, "workers .field must use 1px --line border"
    assert "var(--r-lg)" in body, "workers .field must use --r-lg radius"
    assert "padding:" in body, "workers .field must declare padding"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_c9_ui.py::test_workers_fields_become_mini_cards -v`
Expected: FAIL (rule not found).

- [ ] **Step 3: Add `.form-grid.workers .field` rule to `style.css`**

Place it right after the existing `.workers .field-head` rule. Required declarations: `background: var(--sheet); border: 1px solid var(--line); border-radius: var(--r-lg); padding: 14px 16px;`. Leave the inner `.field-head`, `input`, and `.help` child rules alone.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_c9_ui.py -v`
Expected: every C9 test PASS. Then run the full UI test suite: `pytest tests/test_c5_ui.py tests/test_c7_ui.py tests/test_c8_ui.py tests/test_c9_ui.py -v`
Expected: zero regressions.

- [ ] **Step 5: Commit**

```bash
git add photocure/ui/static/style.css tests/test_c9_ui.py
git commit -m "feat(ui): workers fields rendered as mini step-cards"
```
