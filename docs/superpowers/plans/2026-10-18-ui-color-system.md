# UI Color System, Buttons and Motion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-skin the whole Photocure web UI with a logo-derived blue, pink and amber token system, one pill button family, consistent components, and subtle reduced-motion-safe animation.

**Architecture:** Tokens first (so everything else can reference roles, not hex), then dead-CSS cleanup, then the button family, then components, then motion (CSS keyframes plus a few small JS hooks), then a manual walk of every tab in both themes. Static pytest assertions pin the new palette and contrast; existing structural tests stay green.

**Tech Stack:** Vanilla CSS and JS (`photocure/ui/static/`), pytest static-asset tests, Playwright browser scenarios in `tests/browser/`.

**Spec:** `docs/superpowers/specs/2026-10-18-ui-color-system-design.md` (read it first; section numbers below refer to it)

## Global Constraints

- Work on branch `develop`. Commit after each task. Do not push. Commit messages carry no `Co-Authored-By` trailer (user rule).
- No em dashes anywhere (code, comments, messages). Code comments are one line max.
- Baseline tokens unchanged: light `--table #f8fafc`, `--sheet #ffffff`, `--ink #0f172a`, `--mut #475569`, `--faint #5f6e84`; dark `--table #0c0e12`, `--sheet #14171f`, `--ink #f1f3f7`, `--mut #949eb2`, `--faint #7e8a9f`; fonts Plus Jakarta Sans, JetBrains Mono, Righteous.
- Brand stops (spec 3.1): blue `#175be1 / #2da4f9 / #36bdfb`, pink `#e32b8c / #fa3b91 / #ff5aa5`, amber `#f08000 / #f69212 / #fcce40`, ink on brand fills `--navy #0b1a33`. Values may be nudged a few points only to reach AA.
- Text contrast >= 4.5:1 (large text >= 3:1) in both themes. Brand fills use navy ink, never white.
- No external URLs in `style.css` or `index.html`. Selects use a data-URI chevron.
- Do not change ids, `data-tab`, `data-busy`, `data-goto`, or any selector `app.js` queries. New classes are additive. `#hl-view` stays a `<select>`.
- Keep passing: `tests/test_c5_ui.py`, `tests/test_c10_ui.py`, `tests/test_ui.py`, Node grid tests, browser scenarios. The `.step.active .actions .btn.primary` rule must keep `border-radius` and `gradient`; `body { background: ... var(--table) ...}` must stay literal; the reduced-motion blanket rule with `.001ms` must stay.
- Motion: 120ms presses, 160 to 200ms state, 220ms overlays; enter easing `cubic-bezier(.2,.8,.2,1)`; nothing loops while idle.

## Review Focus

1. Reduced motion on, then close a dialog or toast: it must hide immediately, never waiting on an `animationend` that will not fire (Task 5 test).
2. Switching tabs ten times quickly: the final panel is fully visible, no stuck `.enter` at opacity 0 (Task 5 test).
3. Background tab or hidden element where `animationend` never arrives: overlays and toasts still close via a timeout fallback (Task 5 test).
4. Reopening a dialog while its `.closing` animation is running, or closing twice: ends in the correct visible or hidden state, no duplicate handlers (Task 5 test).
5. Brand fill with navy ink, `--link`, `--warn`, and `.btn.secondary` text on both themes: AA, and a visible focus ring on every button variant (Tasks 1, 3 tests).

---

### Task 1: Token system and old palette tests

**Files:**
- Modify: `photocure/ui/static/style.css` (both `:root` blocks, every `var(--accent*)`, `--duo-*`, `--ok-sage*`, `--danger-crimson*`, `--viz-bar*`, `--tp-*`, `--accent-electric*` use)
- Modify: `tests/test_c7_ui.py` (`test_wcag_light_theme_warn_token`), `tests/test_c8_ui.py`, `tests/test_c9_ui.py`
- Create: `tests/test_ui_tokens.py`

**Interfaces:**
- Produces: CSS custom properties `--blue-50/100/400/500/600/700`, `--pink-*`, `--amber-*`, `--navy`, `--primary`, `--primary-hover`, `--primary-ink`, `--accent`, `--accent-hover`, `--accent-ink`, `--commit`, `--commit-hover`, `--commit-ink`, `--focus`, `--link`, `--primary-soft`, `--accent-soft`, `--commit-soft`, `--grad-brand`, `--canvas-bg` (dark `#0a0f1e`), `--warn` (amber-700 light, amber-400 dark). Keeps `--ok`, `--warn`, `--err` and their `-soft` pairs.
- Consumes: nothing.

- [ ] **Step 1: Run the baseline.** Run `pytest -q tests/test_c5_ui.py tests/test_c7_ui.py tests/test_c8_ui.py tests/test_c9_ui.py tests/test_c10_ui.py tests/test_ui.py`. Expected: all pass. Note the count.
- [ ] **Step 2: Write `tests/test_ui_tokens.py`.** Helpers `_block(theme)` returning the `:root` or `:root[data-theme="dark"]` declarations as a dict, `_contrast(hex_a, hex_b) -> float` (WCAG relative luminance). Tests: `test_role_tokens_defined_in_both_themes` (every token in Interfaces except the scales exists in both blocks), `test_removed_tokens_are_gone` (none of `--accent-electric`, `--duo-`, `--ok-sage`, `--danger-crimson`, `--viz-bar`, `--tp-`, `--accent-glow`, `--accent-pill` appear in the CSS), `test_role_values` (light `--primary == --blue-500`, `--accent == --pink-500`, `--commit == --amber-500`; dark use the `-400` stops per spec 3.2; dark `--canvas-bg == #0a0f1e`), `test_brand_fills_pass_aa_with_navy` (for both themes and each of primary, accent, commit and their hover values: `_contrast(fill, "#0b1a33") >= 4.5`), `test_link_and_warn_pass_aa_on_sheet` (light `--link` and `--warn` against `#ffffff`, dark against `#14171f`, each >= 4.5), `test_no_indigo_left` (no `#4f46e5`, `#6366f1`, `#8b5cf6`, `#7c3aed` in the CSS).
- [ ] **Step 3: Run it to verify it fails.** Run `pytest -q tests/test_ui_tokens.py`. Expected: FAIL on missing tokens.
- [ ] **Step 4: Rewrite the two `:root` blocks.** Add scales and roles per spec 3.1 and 3.2, delete the removed tokens, repoint every existing use. Audit each `var(--accent)` by role: focus outlines to `--focus`, range `accent-color` and toggles and radios to `--primary`, nav and selection and favourites to `--accent`, links to `--link`. Re-tune the body glows (spec 3.3) while leaving `background: var(--table)` literal. If `--warn` amber-700 `#c56500` misses 4.5 on white, darken it until it passes.
- [ ] **Step 5: Edit the three old suites.** Delete only color-pinning assertions: in `test_c7_ui.py` remove `test_wcag_light_theme_warn_token` (superseded by Step 2); in `test_c8_ui.py` remove the removed-token entries from the token list and the `test_c8_token_values_match_spec` rows for them, the `--ok-sage`, `--accent-electric`, `--accent-glow` asserts inside `test_step_state_classes_present`, and the `--canvas-bg` rows; keep `test_baseline_c5_tokens_preserved`, switch, spectrum, step-state class and zero-URL tests; in `test_c9_ui.py` change the `--ok-sage`, `--accent-electric` and `--danger-crimson` asserts to only check that a `border-left` or `color` declaration exists, keep everything else.
- [ ] **Step 6: Run the suites.** Run the Step 1 command plus `tests/test_ui_tokens.py`. Expected: all pass.
- [ ] **Step 7: Commit.** `git add -u tests photocure/ui/static/style.css && git add tests/test_ui_tokens.py && git commit -m "feat(ui): logo-derived blue, pink and amber token system"`

### Task 2: CSS cleanup

**Files:**
- Modify: `photocure/ui/static/style.css`
- Test: `tests/test_ui_tokens.py` (extend)

**Interfaces:** Consumes Task 1 tokens. Produces a stylesheet with one definition each of `.banner`, `.spinner`, `.devchip`, no undefined variables, no `!important` on brand buttons, a flat `.topbar`.

- [ ] **Step 1: Add failing tests** `test_every_var_reference_is_defined` (collect `var(--x)` names, assert each is declared in a `:root` block, ignoring ones with a fallback), `test_single_definition_of_banner_spinner_devchip` (count `^\.banner\s*\{`, `^\.spinner\s*\{`, `^\.devchip\s*\{` equals 1), `test_no_important_on_button_ids` (no `!important` in rules whose selector contains `#btn-`), `test_topbar_is_flat` (`.topbar` rule has no `inset` box-shadow and no `text-shadow` on `.topbar h1`).
- [ ] **Step 2: Run to verify they fail.** `pytest -q tests/test_ui_tokens.py`.
- [ ] **Step 3: Clean the CSS.** Merge duplicate `.banner`, `.spinner`, `.devchip` blocks into one; replace `var(--serif)` with `var(--font-display)` and `var(--sans)` with `var(--font-body)`; delete the "Light theme high-contrast typography" override block and make `.step`, `.sect` light fills at least 90% opaque white instead; delete per-id button color rules (`#btn-scan`, `#btn-cluster`, `#btn-ingest-all`, `.dock-actions .btn.*`, `.dock-sort`) only after Task 3 supplies replacements, so in this task just leave them and skip the `#btn-` test until Task 3; flatten `.topbar` per spec 3.3; delete the five-duos comment.
- [ ] **Step 4: Run all UI suites.** Expected: pass (`test_no_important_on_button_ids` is marked `xfail` here and un-marked in Task 3).
- [ ] **Step 5: Commit.** `git commit -am "refactor(ui): remove duplicate and undefined CSS, flatten topbar"`

### Task 3: Button family

**Files:**
- Modify: `photocure/ui/static/style.css`, `photocure/ui/static/index.html`, `photocure/ui/static/app.js` (strings that build `class="btn"` near the former lines 432, 2254, 2255; `showConfirmDialog` ok-class line; `applyBusy`)
- Test: `tests/test_ui_buttons.py` (create)

**Interfaces:**
- Produces: classes `.btn` (neutral base), `.btn.sm`, `.btn.lg`, `.btn.primary`, `.btn.accent`, `.btn.commit`, `.btn.secondary`, `.btn.ghost`, `.btn.danger`, `.btn.loading`, `.btn.busy`; `applyBusy(busy: boolean): void` adds `.loading` to the button that started the task (tracked in `state.busyBtn: HTMLElement | null`, set by `runTask`) and `.busy` to the others.
- Consumes: Task 1 role tokens.

- [ ] **Step 1: Write failing tests** in `tests/test_ui_buttons.py`: `test_btn_base_is_pill_and_neutral` (`.btn` rule has `border-radius: 999px`, `height: 38px`, `var(--sheet)`), `test_variants_exist` (rules for each variant in the Interfaces list), `test_variants_use_role_tokens` (primary uses `--primary` and `--primary-ink`, accent `--accent`/`--accent-ink`, commit `--commit`/`--commit-ink`), `test_states_present` (`.btn:active` has `scale(.97)`, `.btn:focus-visible` references `--focus`, `.btn.loading` exists), `test_active_step_cta_keeps_dome` (rule `.step.active .actions .btn.primary` has `border-radius` and `gradient`; a sibling rule covers `.btn.accent`), `test_markup_roles` (in `index.html`: `#btn-scan` has class `accent`, `#btn-infer` and `#btn-cluster` `primary`, `#btn-ingest-all` `commit`, `#btn-import-reset` `danger`, `#btn-apply` `commit`, `#btn-rebuild` `secondary`), `test_confirm_warning_maps_to_commit` (`app.js` contains a mapping from `warning` to `commit` near `showConfirmDialog`).
- [ ] **Step 2: Run to verify they fail.** `pytest -q tests/test_ui_buttons.py`.
- [ ] **Step 3: Implement the button CSS** per spec 4 (sizes, variants, shared states, inner highlight, hue-matched shadows). Delete all per-id button rules and the `!important` ones; extend the active-step dome rule to `.btn.accent` with a `linear-gradient(180deg, lighter-stop, role-color)`.
- [ ] **Step 4: Map every button in `index.html`** to a role using spec 4's list, and map any not listed by the same logic (record unlisted ones in the commit body). Change the `app.js` string-built buttons the same way.
- [ ] **Step 5: In `showConfirmDialog`**, set `ok.className = "btn " + ({danger: "danger", warning: "commit", primary: "primary"}[variant])`. In `applyBusy` and `runTask`, implement `.loading` as above; the `::before` spinner is CSS only.
- [ ] **Step 6: Un-xfail `test_no_important_on_button_ids`** and run `pytest -q tests/test_ui_buttons.py tests/test_ui_tokens.py tests/test_c10_ui.py tests/test_c5_ui.py tests/test_ui.py`. Expected: pass.
- [ ] **Step 7: Commit.** `git commit -am "feat(ui): unified pill button family"`

### Task 4: Components

**Files:**
- Modify: `photocure/ui/static/style.css`, `photocure/ui/static/index.html` (add `.seg` class names to `#scrub-presets`, `#nav-indicator` element inside `<nav>`)
- Test: `tests/test_ui_components.py` (create)

**Interfaces:**
- Produces: `.seg` and `.seg-btn` (`#scrub-presets` becomes `class="scrub-presets seg"` and its buttons `class="preset-btn seg-btn"`, keeping existing `.preset-btn` hooks), `<span id="nav-indicator" class="nav-indicator" aria-hidden="true"></span>` as the first child of `<nav>`, `.card::before` brand hairline on `.sect, .step, .dup-group, .diffpanel, .accel`.
- Consumes: Tasks 1 and 3.

- [ ] **Step 1: Write failing tests**: `test_inputs_have_focus_ring_and_data_uri_chevron` (select rule contains `data:image/svg+xml`; `input:focus-visible` or `:focus` rule uses `--focus`), `test_seg_markup_and_css` (`#scrub-presets` has class `seg`; `.seg-btn.active` rule exists), `test_hl_view_still_select` (`<select id="hl-view"` present), `test_nav_indicator_present` (`id="nav-indicator"` inside `<nav>` and `.nav-indicator` rule with `transition: transform`), `test_cards_have_brand_hairline` (a `::before` rule using `--grad-brand` for the listed selectors), `test_bulk_bar_not_pinned_to_sidebar_width` (`.bulk` rule does not contain `252px`), `test_toast_has_status_rail` (`.toast.ok`, `.toast.err` set `border-left` or a rail pseudo-element using `--ok`, `--err`).
- [ ] **Step 2: Run to verify they fail.** `pytest -q tests/test_ui_components.py`.
- [ ] **Step 3: Implement components** per spec 5: inputs and selects, toggles and theme switch knob, `.seg`, nav pill tokens plus indicator styling, count badges, card hairline, selection and locked outlines, lightbox star squares with 10px digit captions (markup in `#lb-rate`, keeping `data-star`), glass bulk bar centered on the content column, toast rail, modal and confirm card radius and footer spacing, activity log pill token mapping, progress fill and gauge arc on `--grad-brand`.
- [ ] **Step 4: Run all UI suites** (Task 3 Step 6 list plus `tests/test_ui_components.py`). Expected: pass.
- [ ] **Step 5: Commit.** `git commit -am "feat(ui): modern components on the new tokens"`

### Task 5: Motion

**Files:**
- Modify: `photocure/ui/static/style.css`, `photocure/ui/static/app.js`
- Test: `tests/test_ui_motion.py` (create); browser scenario added to `tests/browser/scenarios.mjs`

**Interfaces:**
- Produces in `app.js`: `reducedMotion(): boolean`; `afterAnim(el: Element, fallbackMs: number): Promise<void>` (resolves on `animationend`, on the timeout, or immediately when `reducedMotion()`); `openOverlay(el: HTMLElement): void` (removes `.closing`, clears `hidden`); `closeOverlay(el: HTMLElement): Promise<void>` (adds `.closing`, awaits `afterAnim(el, 260)`, then sets `hidden` only if `.closing` is still present; no-op if already hidden); `positionNavIndicator(): void` (moves `#nav-indicator` to the active in-nav item, hides it when Settings or no item is active; also called on window `resize`). CSS: `@keyframes panel-in`, `pop`, `ov-in`, `ov-out`, `toast-in`, `toast-out`, `tick`, `.panel.enter`, `.closing`, `.toast.out`, `.tick`, `body.task-running`.
- Consumes: Tasks 1, 3, 4.

- [ ] **Step 1: Write failing static tests** in `tests/test_ui_motion.py`: `test_keyframes_defined` (each keyframe name above), `test_panel_enter_ends_visible` (`panel-in` final state opacity 1, translateY 0, and `.panel.enter` does not use `animation-fill-mode: backwards` alone in a way that leaves it hidden), `test_chevrons_gated` (the infinite `chev-flow` animation is only declared under `body.task-running` or `.step.active`/`.step.done` selectors; the `.7s` third-chevron delay stays), `test_helpers_exist` (`app.js` defines `reducedMotion`, `afterAnim`, `openOverlay`, `closeOverlay`, `positionNavIndicator`), `test_after_anim_has_fallbacks` (the `afterAnim` body references `prefers-reduced-motion` via `reducedMotion()` and `setTimeout`), `test_overlays_use_closeoverlay` (`#picker`, `#help`, `#similar-panel`, `#confirm-modal`, and `#lightbox` close paths call `closeOverlay`), `test_toast_exit_fallback` (`toast()` adds `out` and still calls `el.remove()` through a timeout), `test_blanket_reduced_motion_rule_kept`.
- [ ] **Step 2: Run to verify they fail.** `pytest -q tests/test_ui_motion.py`.
- [ ] **Step 3: Implement CSS keyframes and rules** per the spec 6 table (durations and easing from Global Constraints).
- [ ] **Step 4: Implement the JS helpers** with the signatures above. `closeOverlay` must be idempotent and must check `.closing` after the await so a reopen during the animation wins.
- [ ] **Step 5: Wire the hooks.** `showTab`: add `.enter` to the new panel, remove it on `animationend` (also via `afterAnim`), remove it from all other panels first so rapid switching cannot stack; call `positionNavIndicator()`. `renderProgress`: toggle `body.task-running` from `p.active`. `setStepState` (the code at the former line 2741): the `done` toggle needs no JS, the pop is CSS. `renderComp`: add `.tick` to count badges whose text changed, removed on `animationend`. `applyPicks`: toggle a `.show` class around `bulkbar.hidden` so it rises and fades. `toast`: on dismiss add `.out`, remove after `afterAnim`. Replace direct `hidden = true` closes of the overlays listed in the test with `closeOverlay(...)` and opens with `openOverlay(...)`; lightbox image fades via `onload` toggling an `.in` class.
- [ ] **Step 6: Add browser scenarios** to `tests/browser/scenarios.mjs`: (a) click through 10 tabs with no delay, assert the last panel has computed opacity 1 after 400ms; (b) with `reducedMotion: 'reduce'` emulated, open and close the confirm dialog and assert it is `hidden` within 50ms; (c) open the help dialog, close it, reopen within 100ms, assert it is visible and not `.closing`. Run them with the project's existing browser-scenario runner (`tests/test_browser_c4.py` shows how).
- [ ] **Step 7: Run everything.** `pytest -q tests/` and the Node grid tests. Expected: pass.
- [ ] **Step 8: Commit.** `git commit -am "feat(ui): subtle reduced-motion-safe animation"`

### Task 6: Full-app verification and screenshots

**Files:**
- Modify: `docs/preview_*.png`, `docs/shot_*.png` (refresh); `README.md` only if a screenshot filename changes.

**Interfaces:** Consumes everything above.

- [ ] **Step 1: Start the app.** `.venv/bin/photocure serve --no-open --port 8080`.
- [ ] **Step 2: Walk every tab in light and dark:** Import, Library, Search, Bursts, Highlights, Events, Finalize, Duplicates, Junk, Apply and undo, Settings; plus lightbox, folder picker, shortcuts, confirm dialog, bulk bar, toast. For each: blue, pink and amber roles look as intended, no leftover indigo or violet, focus ring visible on Tab, no text below AA, no animation running while idle. Fix any miss in `style.css` and re-run the suites.
- [ ] **Step 3: Record contrast ratios** for brand fills with navy ink, `--link`, `--warn`, and light `.btn.secondary` text in the commit body (compute with the helper from `tests/test_ui_tokens.py`).
- [ ] **Step 4: Refresh screenshots** in `docs/` for Import (light and dark), Library, lightbox, duplicates and actions.
- [ ] **Step 5: Final run.** `pytest -q tests/` and `ruff check .`. Expected: pass.
- [ ] **Step 6: Commit.** `git add docs && git commit -m "docs(ui): refresh screenshots for the new color system"`
