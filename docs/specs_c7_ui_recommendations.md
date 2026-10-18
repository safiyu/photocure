# C7: UI Refinements, Accessibility (WCAG 2.1 AA) & Ergonomics Commit Group

**Goal**: Close the remaining accessibility, focus-management, and usability gaps found in the post-C5 UI review. Every item below is small, local, and testable; the whole group must land as one commit (or a tight series of commits with the `C7:` subject prefix) directly on `develop`, following the Rules of Engagement in `docs/specs.txt` §1.

**Source**: UI/UX review of the C5 revamp (focus management, modal trapping, contrast, native dialogs, touch ergonomics). No item below touches the ledger, sidecars, model pipeline, or security boundaries — this group is UI-layer only.

**Precondition**: C1–C6 acceptance suites green on `develop` HEAD, including the C5 browser scenarios (`tests/test_browser_c4.py` pattern) and `ruff`/syntax CI gate.

---

## 7.1 (H1) Lightbox Focus Restoration & Focus Trap

**Issue**: `closeLightbox()` in `photocure/ui/static/app.js` (≈line 798) closes the viewer opened from the grid and calls `scrollToIndex`, but never calls `.focus()` on the card. The previously focused element (`#lb-close`) becomes hidden, so the browser drops focus to `document.body` — disorienting keyboard and screen-reader users. While open, Tab focus also cycles into the obscured background page (no trap in `#lightbox`).

**Requirements**:
1. In `closeLightbox()`, after `scrollToIndex`, focus the matching `.card[data-id]` with `.focus({ preventScroll: true })`. When the lightbox was opened from a non-grid context (e.g. similar-photos, burst), restore focus to the element that opened it (save `document.activeElement` in `openLightbox`, restore if still in the DOM, else fall back to `#main`).
2. Implement a focus trap for `#lightbox` while `.open`: Tab/Shift+Tab cycle within the lightbox controls; Escape already closes (keep); on close, restore per rule 1.
3. Extract the trap into a shared helper (e.g. `trapFocus(container, onEscape)`) and reuse it by `showConfirmDialog` — the confirm dialog already has a bespoke trap; de-duplicate rather than fork a second implementation.

**Acceptance**:
- Keyboard-only: open a photo from the grid, navigate inside the lightbox with Tab (focus never lands on background page), press Escape, focus is back on the originating card (assert via a new Node/browser scenario, or `document.activeElement` check in the C5 browser-test harness).
- No focus drop to `<body>` after close (assert `document.activeElement.classList.contains('card')`).

---

## 7.2 (H2) Replace Native `window.prompt()` with In-App Input Dialog

**Issue**: Folder creation (`picker-new`, `app.js` ≈line 996) and event renaming (`data-rename`, `app.js` ≈line 1326) use synchronous `window.prompt()`. It freezes background listeners, ignores the theme, and can be suppressed in headless/webview contexts.

**Requirements**:
1. Extend `showConfirmDialog()` (≈line 47) with an optional input mode: `{ input: { value, placeholder, label } }` returning the entered string (or `null` on cancel/Escape). Keep the existing focus trap, Escape handling, and focus-restore behavior (reuse the 7.1 shared trap helper).
2. Both call sites switch to the input variant: new-folder creation requires a non-empty, path-safe name (reuse the existing validation the backend applies; reject `/`, null bytes, empty); event rename pre-fills the current name and is a no-op when unchanged.
3. Style: input uses the existing dialog design tokens (`.version-pill`-class focus ring, sheet background); `type="text"`, `autocomplete="off"`.

**Acceptance**:
- No `window.prompt` / `window.confirm` / `window.alert` occurrences remain in `photocure/ui/static/` (add a `test_c5_ui.py`-style static assertion).
- Browser scenario: create a folder through the dialog; empty and path-unsafe names are rejected client-side with inline feedback; rename pre-fills and cancels cleanly on Escape.

---

## 7.3 (M3) WCAG AA Contrast for Warning Text (Light Theme)

**Issue**: `--warn: #d97706` on `--sheet: #ffffff` is ≈3.2:1 — below the 4.5:1 AA requirement for regular text. Used in `.reason` captions (`.reason`, `style.css` ≈line 637) and `.navitem .count.alert` (≈line 148), plus `.catline .tag`.

**Requirements**:
1. In the light theme block of `style.css`, change `--warn` (line 38) to `#b45309` (≈4.6:1 on white) — or darker if other uses (e.g. 40px+ large text or icon-only contexts) demand the lighter hue; dark theme (`#f59e0b`, line 87) is already compliant and stays.
2. If any usage needs the lighter amber only for large/bold elements, introduce `--warn-strong` for text and keep `--warn` for accents, and update each usage explicitly.
3. Audit all `--warn` text-on-sheet pairings after the change; record the measured ratios in the commit body.

**Acceptance**:
- Every regular-size text token pairing in both themes ≥ 4.5:1; large text ≥ 3:1 (verify by hand, note values in commit body).
- Existing contrast test (C5 §6.2 token checks) extended to assert the new light `--warn` ratio if the test suite parses tokens; otherwise a static assertion pinning `--warn` in the light block to `#b45309` or darker.

---

## 7.4 (M4) Persistent "Reset Filters" Button in Library Toolbar

**Issue**: The Library tab has nine filter controls, but "Clear filters" is only rendered in the zero-results empty state (`app.js` ≈line 420). With matches present, users must reset all controls by hand.

**Requirements**:
1. Add a `#reset-filters` button to the Library `.toolbar` (`index.html` ≈line 196), hidden by default.
2. Show it (aria-pressed not needed; simple visible/hidden) whenever any of: search query, category, manual-override, near-duplicate, camera, rating, country, favourite, date-range (either bound away from the library's auto-detected min/max) differs from the default.
3. Clicking it resets all nine controls, clears the composition strip selection, re-runs the current query, and hides itself. Reuse the existing empty-state reset logic — extract it into one `resetAllFilters()` function both call.

**Acceptance**:
- Setting any single filter makes the button appear; clearing it makes the button disappear.
- Reset returns the grid to the unfiltered state with one network round-trip (assert request count in browser scenario).
- Keyboard accessible (it's a `<button>`), focus ring visible.

---

## 7.5 (M5) Native Button for Header Version Badge

**Issue**: `#header-version-badge` (`index.html` line 100) is a `<span role="button" tabindex="0">`; Enter/Space do not activate it in most browsers, so keyboard users cannot reach Settings/About.

**Requirements**:
1. Change the element to `<button id="header-version-badge" class="version-pill" type="button" title="..." data-goto="settings">` (keep the id, classes, and `data-goto` wiring).
2. Adjust `.version-pill` CSS for the button reset (`background`, `border`, `padding`, `font`, `cursor`) so the rendered look is unchanged in both themes.

**Acceptance**:
- Tab reaches the badge; Enter and Space both navigate to Settings (browser scenario).
- No visual regression (screenshot comparison optional; at minimum verify computed styles of pill dimensions are unchanged).

---

## 7.6 (L6) Touch Swipe Navigation in Lightbox

**Issue**: On touch devices, moving between photos requires hitting the small `#lb-prev` / `#lb-next` buttons.

**Requirements**:
1. On `.lb-stage` (`index.html` ≈line 465): record `touchstart` X/Y, and on `touchend` treat a horizontal delta ≥ 50px (and ≥ 2× the vertical delta) as a swipe; ≥ 0 to the left → `lbStep(1)`, to the right → `lbStep(-1)`.
2. Use pointer events (`pointerdown`/`pointerup` with `pointerType === 'touch'`) so mouse drags are unaffected.
3. Suppress the swipe when the gesture started on a control inside the stage (rating stars, buttons) — `e.target.closest('button, .lb-rate')` check.
4. Threshold (50px) is a starting default marked `(tune)` in code comment; keep it local to the handler per Rule 5 (UI gesture constant, not a model threshold — no config entry needed, but do not scatter a second magic number).

**Acceptance**:
- Browser scenario (Playwright `touchscreen`/pointer emulation): simulated left/right swipes step the lightbox; vertical swipes and swipes over the rating row do nothing.
- Mouse drag on the stage changes nothing.

---

## 7.7 (L7) Immediate Busy-State Release on Task Error

**Issue**: In `runTask()` (`app.js` ≈line 1920), `applyBusy(true)` disables `data-busy` buttons; on failure the `finally` clears `state.taskRunning` but the UI stays disabled until the next `pollProgress` tick (up to 3s).

**Requirements**:
1. In `runTask`'s `finally`, call `pollProgress()` immediately after clearing `state.taskRunning` (it is idempotent and already schedules its own next tick — verify no double-timer accumulation: `pollProgress` must remain the single timer owner).
2. Ensure `applyBusy` reflects the polled state, not a local boolean, so a racing in-flight poll cannot re-stick the busy flag.

**Acceptance**:
- Force a failing task (browser scenario with a mocked 500 or a malformed plan): `data-busy` buttons are re-enabled within one tick (< 100ms of the error response).
- No change to the success path (progress bar still driven by `pollProgress` as today).

---

## 7.8 (L8) Date-Range Quick Presets

**Issue**: The dual-thumb scrubber spans the library's full timestamp range; on multi-year libraries a few pixels of thumb movement skips weeks, making month-level filtering awkward.

**Requirements**:
1. Next to the existing range scrubber in the Library toolbar, add a compact segmented preset control: `All`, `This year`, `Last 90 days`, `Last 30 days`, `Custom` (the default state, where the scrubber behaves exactly as today).
2. Presets compute their bounds from the same library min/max the scrubber already uses (from the facets response) and clamp; "This year" starts Jan 1 of the current local year.
3. Moving either thumb manually switches the preset to `Custom`. Selecting a preset animates the thumbs to the new bounds (instant under `prefers-reduced-motion`) and re-runs the query.
4. Preset selection is part of the "any filter differs" state for the 7.4 reset button; reset returns the preset to `All`.

**Acceptance**:
- Each preset produces the correct `from`/`to` query parameters (assert in the existing facet/filter browser scenarios; date boundaries inclusive of the whole end day, consistent with C4 §5.4 behavior).
- Scrubber-first interaction still works unchanged with no preset selected (`Custom`).
- Keyboard accessible (native `<button>`s in the segmented control).

---

## 7.9 Roving Focus for the Virtual Grid (Accessibility Follow-Up)

**Issue**: Every rendered `.card` has `tabindex="0"`, plus its nested buttons — ~80 rendered cards means ~400+ tab stops before a keyboard user can leave the grid. `moveCursor()` (in `virtual_grid.mjs`) moves the visual `.sel` cursor but not DOM focus, so screen readers don't announce items during arrow navigation.

**Requirements**:
1. Implement the WAI-ARIA grid pattern: the grid container gets `role="grid"` (or `role="list"` if the simpler model fits the markup), rows `role="row"`, cards `role="gridcell"`/`role="listitem"`; exactly one card has `tabindex="0"` (the roving position), all others `tabindex="-1"`.
2. `moveCursor()` (arrow keys, `Home`/`End`, `PageUp`/`PageDown`) moves the roving `tabindex` to the target card and calls `card.focus({ preventScroll: false })` so the SR announcement happens; `j`/`k` triage navigation continues to work on the same cursor.
3. Focus restoration on re-render/scroll recycling: when the roving card's node is recycled out of view and a new node is created for the same record id, the `tabindex="0"` must transfer with the record id, not the DOM node.
4. Enter/Space on the focused card opens the lightbox (existing behavior via the card's primary button must remain reachable in one Tab from the card — if the card itself becomes focusable, its nested buttons remain individually tabbable only when the card is focused, i.e. nested `tabindex="-1"` until the card receives focus — document the chosen pattern in the commit body).
5. Keep the DOM-node-count invariant from C4 (virtualization test: rendered nodes bounded by viewport rows ± overscan) — the roving tabindex must not force extra rows into view.

**Acceptance**:
- Keyboard-only scenario: Tab enters the grid once (one stop), arrows navigate N items with SR-appropriate DOM focus changes, Enter opens the lightbox, Escape returns focus to the same card.
- Existing virtual-grid Node unit tests extended: roving tabindex moves on cursor moves, survives scroll recycling, and exactly one `tabindex="0"` card exists at all times.

---

## 7.10 Modal Focus Traps for `#picker` and `#help`

**Issue**: The folder picker (`#picker`, `index.html` ≈line 514) and shortcuts help (`#help`) overlays don't trap focus; Tab escapes into the background page.

**Requirements**:
1. Apply the shared `trapFocus()` helper from 7.1 to both overlays: on open, focus the first actionable control; Tab cycles within; Escape closes (picker already supports Escape — keep); on close, restore focus to the opener.
2. No behavior change to the picker's permission checks, file counts, or navigation — trapping only.

**Acceptance**:
- Browser scenario: with either overlay open, 20 consecutive Tab presses never land on a hidden/background element.
- Focus returns to the opener on close (both Escape and explicit close button).

---

## 7.11 Acceptance Criteria & Test Suite for C7

Automated (all must pass before the group is considered done, per the per-group CI gate):

```bash
# Backend + UI static suites (unchanged command set, extended with C7 assertions)
pytest -q tests/

# Node virtual-grid unit tests (extended per §7.9)
node --test photocure/ui/static/*.test.mjs 2>/dev/null || node --test tests/node/

# New/extended browser scenarios (C5-style Playwright harness)
#   - 7.1 lightbox focus restore + trap
#   - 7.2 prompt replacement (create folder, rename, empty/path-unsafe rejection)
#   - 7.4 reset-filters button appears/disappears/resets in one round-trip
#   - 7.5 version-badge Enter/Space activation
#   - 7.6 touch swipe navigation (pointer events, touch type only)
#   - 7.7 busy-state release on forced task error
#   - 7.8 date preset query parameters
#   - 7.9 roving focus single-tab-stop grid navigation
#   - 7.10 picker/help focus containment
```

Static assertions to add (extend `tests/test_c5_ui.py` or its C7 successor):
1. `window.prompt|window.confirm|window.alert` absent from `photocure/ui/static/` (7.2).
2. Light-theme `--warn` is `#b45309` or darker (7.3).
3. `#header-version-badge` is a `<button>` (7.5).
4. Exactly one `tabindex="0"` in rendered grid DOM (covered by Node test, 7.9).

Manual QA checklist (run on a real machine before the commit is considered final):
1. Full keyboard-only pass of Library → open photo → rate → close → grid focus intact.
2. Screen reader (Orca or NVDA) announces the card title/counter while arrowing through 20 grid items.
3. Light theme: read `.reason` captions and the "Needs a look" alert counts at arm's length — no washed-out amber.
4. Phone (or 375px emulation): swipe through 10 lightbox photos without touching the nav buttons.
5. Force-kill a running task (stop the scan mid-flight): toolbar buttons re-enable immediately, progress bar shows the terminal state.

---

## 7.12 Ordering & Sizing

Suggested implementation order (each step independently testable; all land in the C7 commit group):

| # | Item | Priority | Est. size |
|---|------|----------|-----------|
| 1 | 7.3 contrast token | M | trivial |
| 2 | 7.5 version badge button | M | trivial |
| 3 | 7.7 busy-state release | L | trivial |
| 4 | 7.2 shared input dialog + prompt removal | H | small |
| 5 | 7.10 picker/help focus traps (needs 7.1's helper) | M | small |
| 6 | 7.1 lightbox focus restore + trap | H | small |
| 7 | 7.4 reset-filters button | M | small |
| 8 | 7.8 date presets | L | medium |
| 9 | 7.6 touch swipe | L | small |
| 10 | 7.9 roving grid focus | M | medium — schedule last; it touches `virtual_grid.mjs` recycling and needs the most careful regression testing |

Notes:
- Items 1–3 are independent one-liners-to-twenty-liners; batch them to reduce CI churn.
- 7.9 is the only item that may warrant its own commit within the C7 group (`C7: roving tabindex for virtual grid`) if the recycling interaction reveals bugs — split only, never reorder.
- Nothing in this group changes `SCHEMA_VERSION`, the ledger, sidecars, models, or the security model. No `NOTICE` changes required (no new assets or dependencies).
