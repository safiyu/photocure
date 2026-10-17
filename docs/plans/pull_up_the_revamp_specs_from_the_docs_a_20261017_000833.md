<!-- myagy:approved=1 -->
<!-- myagy:idea=pull up the revamp specs from the docs and implement them -->
<!-- myagy:dirty_before=["docs/specs_c7_ui_recommendations.md", "docs/specs_ui_revamp.md", "docs/ui_mockup.jpeg", "docs/ui_revamp_specification.md", "static/app.js", "static/index.html", "static/styles.css", "static/virtual_grid.mjs"] -->

# C8: UI/UX Modern Revamp Implementation Specification & Milestone Plan

Based on [docs/specs_ui_revamp.md](file:///home/safiyu/Projects/photocure/docs/specs_ui_revamp.md), [docs/ui_revamp_specification.md](file:///home/safiyu/Projects/photocure/docs/ui_revamp_specification.md), and [docs/ui_mockup.jpeg](file:///home/safiyu/Projects/photocure/docs/ui_mockup.jpeg), this specification defines the complete implementation of the dark-slate workstation visual overhaul for Photocure's primary workspace canvas and navigation.

---

## 1. Architecture & Approach (Trade-offs & Decisions)

### 1.1 Design System & CSS Token Architecture
- **Additive Token Scoping**: Base surface, text, and accent tokens asserted by [tests/test_c5_ui.py](file:///home/safiyu/Projects/photocure/tests/test_c5_ui.py#L31-L50) (`--table`, `--sheet`, `--ink`, `--mut`, `--faint`) and [tests/test_c7_ui.py](file:///home/safiyu/Projects/photocure/tests/test_c7_ui.py#L29-L46) (`--warn: #b45309`) must be strictly preserved in `:root` and `:root[data-theme="dark"]` to ensure zero backward-compatibility regressions.
- **Extended Token Palette**:
  - Surfaces: `--canvas-bg: #0b0e14`, `--sheet-subtle: #0f141f`, `--sheet-border: #1e2638`, `--sheet-border-light: rgba(255, 255, 255, 0.08)`.
  - Electric Blue Accent & Glow: `--accent-electric: #3b82f6`, `--accent-electric-hover: #60a5fa`, `--accent-glow: 0 0 20px rgba(59, 130, 246, 0.28)`, `--accent-pill: rgba(59, 130, 246, 0.16)`.
  - Semantic Status: Sage Green (`--ok-sage: #22c55e`, `--ok-sage-bg: #14291f`, `--ok-sage-border: rgba(34, 197, 94, 0.35)`), Crimson (`--danger-crimson: #ef4444`, `--danger-crimson-bg: #261619`, `--danger-crimson-border: rgba(239, 68, 68, 0.35)`).
  - Equalizer Visualizer: `--viz-bar: #38bdf8`, `--viz-bar-faint: rgba(56, 189, 248, 0.22)`.
- **WCAG 2.1 AA Compliance**: All text-on-surface pairings in both themes maintain contrast $\ge 4.5:1$ for normal text and $\ge 3:1$ for large text/icons. Light theme retains high-contrast slate surfaces while dark theme uses obsidian `#0b0e14` and `#131926`.

### 1.2 Sidebar Navigation & Hardware-Style Dark Mode Switch
- **Active State Capsule Pills**: Active route item (`.navitem.active`) is styled with capsule border radius (`20px`), `--accent-pill` background, `1px solid rgba(59, 130, 246, 0.3)` border, and high-contrast white text and icons.
- **Tabular Count Badges**: Monospace count badges (`.count`) utilize `font-family: var(--font-mono)` and `font-variant-numeric: tabular-nums` enclosed in subtle pill containers.
- **Device Status Chip**: `#device-chip` updated to a sage green capsule pill (`--ok-sage-bg`, `--ok-sage-border`) with a glowing green pulse dot (`.dot`) and label `Running on <Device>`.
- **Theme Switcher Transition**:
  - In [photocure/ui/static/index.html](file:///home/safiyu/Projects/photocure/photocure/ui/static/index.html#L85-L88), `#theme-toggle` is upgraded to an accessible hardware-style sliding toggle containing an `<input type="checkbox" id="theme-switch">` with a round knob `.slider`.
  - `#theme-toggle` container preserves its ID and `tabindex="0"` to satisfy existing CI assertions in [test_c5_ui.py](file:///home/safiyu/Projects/photocure/tests/test_c5_ui.py#L135-L148) while responding to `click`, `Space`, and `Enter`.
  - State stays bidirectional and synchronized with `localStorage.getItem("pcur-theme")`.

### 1.3 Equalizer Spectrum Visualizer & AI Inference Progress Card
- **Spectrum Architecture**:
  - Embedded in `#progress-card` adjacent to the progress header: a 24-bar flex container (`.spectrum-bars`).
  - Bars animate with `@keyframes eq-pulse` with staggered animation delays (`0.05s` to `1.15s`) when the task is active (`#progress-card.active`).
  - Strict motion sensitivity guard: `@media (prefers-reduced-motion: reduce)` disables keyframe transforms and sets static normalized heights.
- **Progress Card Layout**:
  - Left: Pulsing cobalt spinner or status checkmark, bold task title, and phase subtitle.
  - Center: Audio/AI frequency visualizer wave (`.spectrum-bars`).
  - Right: Live tabular ratio (`.progress-ratio`, e.g., `135 / 3,194`), percentage badge, stop button, and dismiss `[✕]` button.
  - Bottom: Full-width progress track with cobalt fill.

### 1.4 Three-Step Workflow Engine Grid
- **Responsive 3-Column Layout**: Responsive CSS Grid (`grid-template-columns: repeat(3, minmax(0, 1fr))`) collapsing to 1 column at $\le 960\text{px}$.
- **Step 1 ("Find your photos")**:
  - Completed state styling: Green circular badge `(1)` in `--ok-sage`, top-right `X indexed` pill badge.
  - Input styling: Slate background with inline magnifying glass SVG icon.
  - Action buttons: Capsule `[📁 Browse Folder]` button with dark-slate fill and `[📷 Scan]`.
- **Step 2 ("Analyze")**:
  - Focused active state styling: Electric blue border (`1.5px solid var(--accent-electric)`), atmospheric glow (`box-shadow: 0 0 20px rgba(59, 130, 246, 0.28)`), and blue circular badge `(2)`.
  - Context menu button `...` for inference settings.
  - Dropdown chip for device selection (`[ microchip | NVIDIA GPU | ▼ ]`) with sublink `Devices and workers in Settings`.
  - Primary action: Electric blue pill button `[ ▷ Analyze Now ]`.
- **Step 3 ("Group bursts")**:
  - Pending/locked state styling: Dimmed opacity (`0.55`), muted badge `(3)`, and top-right `Needs step 2` status badge.
  - Visual asset: Centered isometric 3D wireframe SVG representing 3 layered burst photos.
  - Action button: Disabled capsule `[ ☵ Group ]` with trailing lock icon `🔒`.

### 1.5 Terminal-Grade Activity Log
- **Pulse Header**: Section header "Activity" accompanied by an electric-blue EKG/pulse heartbeat SVG glyph 〽️.
- **Monospace Tabular Time Pills**:
  - [photocure/ui/static/app.js](file:///home/safiyu/Projects/photocure/photocure/ui/static/app.js#L308-L318) `logTo()` wraps the timestamp in a `<span class="time-pill">` formatted with `JetBrains Mono`.
  - Status classes: `.time-pill.info` (blue tint), `.time-pill.ok` (green tint), `.time-pill.muted` (gray tint).
- **Completion Highlighting**:
  - Rows signalling completed phases (e.g., `Scan finished...`, `Analyze finished...`) receive class `.logline.active-highlight` (sage border, translucent green background, and trailing checkmark `✓`).

---

## 2. Files to Modify & Files to Create

### Files to Modify
- [photocure/ui/static/style.css](file:///home/safiyu/Projects/photocure/photocure/ui/static/style.css):
  - Add C8 tokens to `:root` and `:root[data-theme="dark"]` (`--canvas-bg`, `--accent-electric`, `--accent-glow`, `--ok-sage`, `--danger-crimson`, `--viz-bar`, etc.).
  - Add sliding switch styles (`.switch`, `.slider`), active navigation capsule styling, and tabular count badges.
  - Add equalizer spectrum styles (`.spectrum-bars`, `.spectrum-bar`, `@keyframes eq-pulse`).
  - Add 3-step grid styling, glowing `.step.active` and completed `.step.done` states, and isometric burst card SVG styling.
  - Add terminal log styling (`.time-pill`, `.active-highlight`, scrollbar refinements).
  - Add media queries for responsive stacking ($\le 960\text{px}$) and `@media (prefers-reduced-motion: reduce)`.
- [photocure/ui/static/index.html](file:///home/safiyu/Projects/photocure/photocure/ui/static/index.html):
  - Add SVG icon definitions for pulse wave (`#i-pulse`), search input glyph (`#i-search-inline`), play triangle (`#i-play`), lock (`#i-lock`), and isometric cards.
  - Update `#device-chip` structure and replace `#theme-toggle` markup with the accessible hardware-style slider.
  - Embed `.spectrum-bars` container and `.progress-ratio` in `#progress-card`.
  - Update `.steps` list items with input search icon, device select chip, options ellipsis, isometric burst SVG, and lock icon.
  - Add pulse icon to the Activity section header.
- [photocure/ui/static/app.js](file:///home/safiyu/Projects/photocure/photocure/ui/static/app.js):
  - In `renderProgress()`, update live processed counts in `.progress-ratio`, set spectrum bar active states, and manage equalizer animation classes.
  - In `logTo()`, format timestamps into `.time-pill` elements and detect completion messages to apply `.active-highlight` and checkmark glyphs.
  - In `applyTheme()` and initialization, bind the sliding toggle `#theme-switch` alongside `#theme-toggle`, handling keyboard and click events.
  - Update step state transitions (marking Step 1 `.done`, Step 2 `.active`, and Step 3 `.locked` based on library state).
- [tests/browser/scenarios.mjs](file:///home/safiyu/Projects/photocure/tests/browser/scenarios.mjs):
  - Add C8 browser scenarios verifying theme switch toggle interaction, equalizer spectrum presence, step card visual states, and terminal `.time-pill` rendering.

### Files to Create
- `tests/test_c8_ui.py`:
  - Static assertions verifying presence of C8 CSS tokens, spectrum visualizer DOM elements, hardware switch accessibility attributes, and absence of external URLs.

---

## Milestones

- [x] Add extended C8 CSS design tokens to `photocure/ui/static/style.css` without altering baseline C5/C7 tokens.
- [x] Implement sliding dark mode toggle switch in `photocure/ui/static/index.html` and wire keyboard/click events in `photocure/ui/static/app.js`.
- [x] Restyle sidebar navigation with capsule pills, uppercase category headers, and monospace count badges.
- [x] Update `#device-chip` with sage green pulse dot, capsule container, and live GPU/CPU status styling.
- [x] Implement AI Inference Equalizer Spectrum Visualizer with CSS keyframes and `@media (prefers-reduced-motion)` guards in `photocure/ui/static/style.css`.
- [x] Embed visualizer container, live ratio counter, and dismiss button into `#progress-card` in `photocure/ui/static/index.html`.
- [x] Update `renderProgress()` in `photocure/ui/static/app.js` to drive equalizer animation states and live ratio text.
- [x] Overhaul 3-step workflow engine grid in `photocure/ui/static/index.html` with search input icon, device selector, 3D burst illustration, and lock icon.
- [x] Add step card status styles (`.step.done`, `.step.active` with blue glow, and `.step.locked`) in `photocure/ui/static/style.css`.
- [x] Update step dynamic state logic in `photocure/ui/static/app.js` based on library and scan counts.
- [x] Restyle batch action buttons (`[ Run all three steps ]`, `[ Reset library ]`) and safety status text.
- [x] Update Terminal Activity feed header with pulse EKG glyph and restyle log container.
- [x] Upgrade `logTo()` in `photocure/ui/static/app.js` to render tabular `.time-pill` badges and `.active-highlight` completion rows.
- [x] Add tablet/mobile responsive breakpoints ($\le 960\text{px}$) for 3-step grid and canvas layout.
- [x] Create `tests/test_c8_ui.py` covering C8 tokens, switch accessibility, equalizer markup, and zero external URLs.
- [x] Extend `tests/browser/scenarios.mjs` with browser assertions for theme switch, step card states, and log timestamp pills.
- [x] Run full test suites (`.venv/bin/pytest tests/` and `node --test tests/virtual_grid.test.mjs`) to verify zero regressions.

---

## 4. Risk Mitigation & Verification Strategy

| Risk | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Breaking C5 Token Tests** | `test_c5_ui.py::test_wcag_aa_color_tokens` asserts exact hex codes for `--table`, `--sheet`, `--ink`, `--mut`, `--faint` | Retain original hex tokens in `:root` and `:root[data-theme="dark"]`; define C8 properties as distinct additive variables (`--canvas-bg`, `--sheet-subtle`, `--accent-electric`, etc.). |
| **Breaking Theme Toggle Test** | `test_c5_ui.py::test_theme_persistence_html_structure` looks for `'id="theme-toggle"'` | Keep `id="theme-toggle"` on the switch wrapper element; wire both the container and inner checkbox to theme state. |
| **Accessibility & Keyboard Trap Regressions** | Moving to custom switch or adding buttons could trap focus or break WAI-ARIA | Ensure the switch is a semantic `<label class="switch"><input type="checkbox">` accessible via Tab, Space, and Enter; all action buttons remain native `<button>`. |
| **Excessive GPU/CPU from Animations** | Equalizer wave keyframes and glow effects running continuously could degrade browser performance | Scope `@keyframes eq-pulse` to run only while `#progress-card.active` is true; freeze all animations when `@media (prefers-reduced-motion: reduce)` is set. |
| **Zero External Network Rule Violation** | External SVG or icon fonts introduced accidentally | All new icons (pulse, play, lock, search, 3D isometric stack) embedded directly as inline SVGs in `index.html`. |

### Verification Protocol
1. **Static Suite Verification**:
   ```bash
   .venv/bin/pytest -q tests/test_c5_ui.py tests/test_c7_ui.py tests/test_c8_ui.py
   ```
2. **Virtual Grid Unit Invariants**:
   ```bash
   node --test tests/virtual_grid.test.mjs
   ```
3. **Comprehensive Pytest Suite**:
   ```bash
   .venv/bin/pytest tests/
   ```
4. **Browser Scenarios (Playwright / CDP)**:
   Verify scenario runs including the new C8 visual states, theme toggle, and log formatting.
