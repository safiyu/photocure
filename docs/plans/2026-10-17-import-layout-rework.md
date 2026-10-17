# Import layout rework implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Rework the Import page and shell to match the second mockup (2026-10-17 conversation): hero header with gauge, blob-shaped step cards, triple chevrons, a bottom action dock, sidebar touch-ups, dark gradient background.

**Architecture:** Mostly `style.css` + `index.html`. Small `app.js` edits keep JS-driven elements alive: the hero gauge replaces the topbar progress pill, the dock total hooks into `renderComp`, the device chip text gets shorter. All existing element ids that JS or tests use are preserved unless a task says otherwise.

**Tech Stack:** Vanilla HTML/CSS/JS, pytest static assertions (`tests/test_c10_ui.py`, same pattern as `test_c8_ui.py`/`test_c9_ui.py`). Test runner: `.venv/bin/pytest`.

**Spec:** The second mockup image and the scope chat: user chose option A, all six changes.

## Global Constraints

- Keep `test_c5/c7/c8/c9` green: `#theme-toggle` label with `tabindex="0"`, `aria-label`, inner `#theme-switch` role=switch; `#progress-card` with exactly 24 `.spectrum-bar`; `.sidebar-footer`; `.steps` = `repeat(3, minmax(0, 1fr))`; `.step.active` uses `--accent-electric` + `--accent-glow`; `.step.locked` opacity `0.55`; no external URLs.
- Respect `prefers-reduced-motion` for every new animation.
- Comments one line max. No em dashes. No co-author trailer on commits (user rule overrides the harness reminder).
- Decorative SVG is `aria-hidden`.

## Design calls (mine, not the user's)

- Blob shape is approximated with asymmetric elliptical `border-radius`, not SVG-masked concave scoops.
- The mockup nav has no "Apply and undo". It is a real feature, so it moves into the Review group and the "Finish" header goes away.
- Only the nav label becomes "Junk"; the page title and toasts keep "Junk review".
- Chip text becomes `GPU` or `CPU`; the full "Running on <device>" moves to `title` and `aria-label`.
- The gradient applies to dark theme only; light theme stays flat.
- `#library-status` help line is kept, as a small caption under the dock.
- The old topbar progress pill is removed; the gauge and the progress card's own Stop button replace it.

## Review Focus

- Progress error text still visible (it renders in `#progress-msg`, which stays in the hero card).
- Hero with no active task shows only title (gauge and card hidden, no empty gap).
- Dock "Sort Photos" disabled when library is empty.
- Phone width: hero and dock wrap instead of overflowing.
- Light theme: new glass/dock/hero styling stays legible.

## Tasks

### Task 1: Triple chevrons
**Files:** Modify `photocure/ui/static/index.html` (both `.step-flow` blocks), `photocure/ui/static/style.css` (chev rules), `tests/test_c9_ui.py` (count 2 to 3), Create `tests/test_c10_ui.py`.
- [ ] Update `test_step_flow_markup_inside_step1_and_step2` in `tests/test_c9_ui.py` to expect 3 chevrons. Add to `test_c10_ui.py`: `test_step_flow_has_three_chevrons` (both step-scan and step-infer blocks hold 3 `class="chev"`; CSS has `.step-flow .chev:nth-child(3)` with a `.7s` delay). Run, see RED.
- [ ] Add a third chevron to both blocks; add the nth-child(3) rule. Run c9+c10, see GREEN. Commit.

### Task 2: Sidebar touch-ups
**Files:** Modify `index.html` (nav, sprite `i-moon` already exists, foot), `style.css`, `app.js` (`renderChip`), `tests/test_c10_ui.py`.
- [ ] Tests: no `<h2>Finish</h2>`; the `data-tab="actions"` button sits inside the Review navgroup; nav label `<span>Junk</span>` and no `<span>Junk review</span>`; `class="theme-text"` containing `Dark Mode`; app.js no longer assigns `"Running on " + label` to `#device-chip-text`, and sets `chip.title` and `aria-label` containing `Running on`; CSS `.devchip` is not `width: 100%`. RED.
- [ ] Implement: markup moves; `.theme-row` = moon icon + `Dark Mode` text + the existing `#theme-toggle` label; `renderChip` text `GPU`/`CPU`; `.devchip` compact pill (`width: auto; align-self: flex-start`). GREEN, full UI suite, commit.

### Task 3: Action and summary dock
**Files:** Modify `index.html` (replace the Import tab's `.comp-host` and `.runall`; add sprite `i-undo`), `style.css` (`.dock*`, remove `.runall` rules), `app.js` (`renderComp`), `tests/test_c10_ui.py`.
- [ ] Tests: `class="dock"` exists inside `#tab-library`; ids `dock-total`, `btn-sort-dock` (with `data-sortnow`), `btn-ingest-all` and `btn-import-reset` are inside the dock; `class="runall"` absent from html; `id="library-status"` still present; app.js `renderComp` sets `dock-total` and disables `btn-sort-dock` when total is 0; CSS `.dock {` has `border-radius`. RED.
- [ ] Implement. Buttons keep their ids and `data-busy`. Labels: `Run All Three Steps` (play icon), `Reset Library` (undo icon), `Sort Photos`. GREEN, suite, commit.

### Task 4: Hero header with gauge
**Files:** Modify `index.html` (topbar becomes `.topbar.hero`; move `#progress-card` into it; add `#hero-gauge`; delete `#topbar-progress`), `style.css`, `app.js` (`renderProgress`, stop-button wiring), `tests/test_c10_ui.py`.
- [ ] Tests: `#progress-card` appears inside `<header class="topbar hero"`; `id="hero-gauge"` has `role="progressbar"`; html and app.js contain no `topbar-progress`/`btn-stop-topbar`/`topbar-fill`; app.js sets `gauge-arc` dash and `aria-valuenow`; CSS `.gauge-arc` has `stroke-dasharray` and a transition, and reduced-motion disables it; `.topbar` is not `position: sticky`. RED.
- [ ] Implement: gauge = semicircle SVG using `pathLength="100"`; phase text from `/Phase\s+\d+\s*\/\s*\d+/` in `p.message`, else task name; `show()` hides the gauge when idle. Clean every topbar-pill reference out of `app.js` (grep `topbar`). Compact progress card inside hero (no border/background), thin track kept. GREEN, suite, commit.

### Task 5: Blob step cards and dark gradient
**Files:** Modify `style.css`, `tests/test_c10_ui.py`.
- [ ] Tests: `.step` rule has an elliptical `border-radius` (contains `/`); `.step.active` has `transform` with `scale(`; `:root[data-theme="dark"] body` has `gradient`; `.step.active .actions .btn.primary` rule exists. RED.
- [ ] Implement asymmetric radii (max about 46px so corner content is not clipped), larger active card (scale, higher z-index, scooped-top full-width Analyze button), dark-theme body gradient and glass card backgrounds for `.step`, `.sect`, `.hero`, `.dock`. GREEN, full UI suite plus broader `-k` suite, commit.

### Final
- [ ] Run the app, screenshot Import (idle, and with a fake running progress via the page console if feasible), Settings, Apply and undo in both themes. Dispatch one fresh reviewer on the whole branch. Fix Critical/Important with TDD.
