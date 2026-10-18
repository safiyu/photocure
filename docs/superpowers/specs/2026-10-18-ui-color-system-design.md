# Photocure UI: logo-matched color system, button family, and motion

Date: 2026-10-18
Status: draft for approval
Scope: the whole web UI (`photocure/ui/static/index.html`, `style.css`, `app.js`, `virtual_grid.mjs` only where a class name is emitted)

## 1. Intent

Make the UI read as one system built from the logo: blue, hot pink and warm amber on white (light) or deep navy (dark), with modern pill buttons and components and small, purposeful motion. Today five accents compete (indigo, electric blue, pink, violet, amber), buttons are styled per id with `!important`, and most state changes snap with no transition.

### Goals
- Every color in the UI comes from one token set derived from the logo. No hard-coded brand hex outside `:root` blocks.
- One button family with consistent sizes, variants and states, used everywhere (Import, toolbars, lightbox, modals, Settings, Apply and undo, bulk bar).
- Subtle motion on the moments that matter: tab change, nav move, press, done, open, close, toast. All of it respects `prefers-reduced-motion`.
- Remove dead and duplicate CSS found in the review.

### Non-goals
- No layout changes to the Import three-step flow, dock, hero or sidebar structure. `tests/test_c10_ui.py` pins them and they stay.
- No new features, no API or backend changes, no change to the virtual grid's rendering or DOM count.
- No new fonts or external assets. `tests/test_c5_ui.py` forbids external URLs.

### Anti-goals (would make this a failure even if it ships)
- Any text pairing below WCAG AA (4.5:1 normal, 3:1 large). The current light theme already needed contrast fixes once; this must not reintroduce them.
- Motion that runs while the app is idle. The current always-on chevron loop is the example to remove, not copy.
- Breaking any existing test in `tests/test_c5_ui.py`, `tests/test_c10_ui.py`, `tests/test_ui.py`, or the Node virtual-grid tests.

## 2. Logo analysis (source of truth for the palette)

Measured from `photocure/ui/static/logo.png`, opaque pixels only:

| Family | Share | Dark stop | Mid | Light stop |
|---|---|---|---|---|
| Blue | 45% | `#175be1` royal | `#2da4f9` | `#36bdfb` sky |
| Pink | 26% | `#e32b8c` magenta | `#fa3b91` | `#ff5aa5` (lightened) |
| Amber | 22% | `#f08000` orange | `#f69212` | `#fcce40` yellow |
| White | 5% | | `#edf0f8` | |

Two things follow from the proportions and should shape the UI:
1. Blue is the dominant brand color, pink is the signature accent, amber is the warm highlight. The UI mirrors that weighting: blue carries most interactive chrome, pink is used for the moments that should feel like "Photocure", amber for commit and attention.
2. The logo fills are saturated and light. White text fails AA on the sky blue, the hot pink and all of the amber. So brand-filled buttons use dark navy ink text, the way the current Scan and Run All buttons already do. This is a system rule, not a per-button exception.

## 3. Token system

All tokens live in `:root` and `:root[data-theme="dark"]` in `style.css`. The baseline tokens asserted by `tests/test_c5_ui.py` keep their exact values (`--table`, `--sheet`, `--ink`, `--mut`, `--faint` in both themes, and the three font tokens).

### 3.1 Brand scales

```
--blue-50:  #eaf6ff   --blue-100: #cfeafe   --blue-400: #36bdfb
--blue-500: #2b9af6   --blue-600: #1d6fe6   --blue-700: #175be1
--pink-50:  #fff0f7   --pink-100: #ffd6e9   --pink-400: #ff5aa5
--pink-500: #fa3b91   --pink-600: #e32b8c   --pink-700: #c21f74
--amber-50: #fff6e5   --amber-100: #ffe6b8  --amber-400: #fcce40
--amber-500: #f6a315  --amber-600: #f08000  --amber-700: #c56500
--navy:     #0b1a33   (ink used on brand fills, both themes)
```

Builder may nudge any value by a few points to hit a contrast target, but the hue and the 50/100/400/500/600/700 structure stay.

### 3.2 Semantic roles

| Role | Token | Light | Dark | Used for |
|---|---|---|---|---|
| Primary action | `--primary` / `--primary-hover` / `--primary-ink` | blue-500 / blue-600 / navy | blue-400 / blue-500 / navy | `.btn.primary`, progress fill, gauge, device chips, log info pills |
| Brand accent | `--accent` / `--accent-hover` / `--accent-ink` | pink-500 / pink-600 / navy | pink-400 / pink-500 / navy | active nav pill, Scan, favourite, selection ring, gradient titles, brand glow |
| Commit / attention | `--commit` / `--commit-hover` / `--commit-ink` | amber-500 / amber-600 / navy | amber-400 / amber-500 / navy | Run All, Review and apply, star ratings, locked badge, warn states, `.count.alert` |
| Focus ring | `--focus` | blue-600 | blue-400 | every `:focus-visible` |
| Link | `--link` | blue-700 | blue-400 | `.link`, inline anchors |
| Soft fills | `--primary-soft`, `--accent-soft`, `--commit-soft` | the 50/100 tints | 14 to 18% alpha of the 500 | toggles on, chips, selected rows, banners |
| Brand gradient | `--grad-brand` | `linear-gradient(135deg, var(--blue-400), var(--pink-500))` | same | hero accents, gauge arc, gradient text, active nav border |

The existing `--accent` token is currently indigo. It is repurposed to pink. Everything that today reads `var(--accent)` for focus rings, range `accent-color`, toggles and radios is re-pointed per the table (focus to `--focus`, form controls to `--primary`). Builder audits every `var(--accent)` use and assigns it a role; none keep indigo.

Removed: `--accent-electric*`, `--accent-glow`, `--accent-pill`, all nine `--duo-*`, `--viz-bar*`, `--ok-sage*`, `--danger-crimson*`, `--tp-*`. Their uses map onto the roles above or onto the kept semantic tokens `--ok`, `--warn`, `--err` and their `-soft` pairs. `--warn` becomes the amber-700 / amber-400 pair so warning text stays AA.

Category colors `--c-*` are unchanged (they are data colors, not chrome). `--c-people` stays pink-500 so the legend matches the brand.

### 3.3 Surfaces

- Light canvas: keep the current body radial glows but tune them to the logo stops (pink-500 at 8% alpha top-left, blue-400 at 10% top-right). `body { background: var(--table) }` must remain literally present for `test_dark_theme_has_gradient_background_and_glass_cards`.
- Dark canvas: `--canvas-bg: #0a0f1e` (navy, not the current purple `#131024`), same two glows at 14 to 16%. Dark cards keep the glass gradient the test requires.
- Topbar: drop the embossed inset shadows and the engraved `text-shadow`. It becomes a flat area with the page title, no card chrome, so the step cards and dock are the only raised surfaces on Import. The test only requires that `.topbar` is not sticky.

## 4. Button family

One `.btn` base plus modifiers. Every `<button>` in the app that is not an icon-only control uses it; per-id color rules (`#btn-scan`, `#btn-cluster`, `#btn-ingest-all`, `.dock-actions .btn.*`) are deleted.

```
.btn            pill (border-radius: 999px), height 38px, padding 0 18px, font-weight 600,
                transition: background .15s, border-color .15s, box-shadow .15s, transform .12s
                bare .btn is the neutral button: bg --sheet, border 1px --line, text --ink,
                hover bg --well and border --faint (Cancel, Browse, Close, New folder, and any
                button app.js builds with class "btn" alone)
.btn.sm         32px, padding 0 14px, 13px text
.btn.lg         46px, padding 0 24px, 15px text
.btn.primary    bg --primary, ink --primary-ink, inner 1px highlight, soft shadow in the same hue
.btn.accent     bg --accent, ink --accent-ink
.btn.commit     bg --commit, ink --commit-ink
.btn.secondary  bg --primary-soft, border 1px --primary at 35%, text --primary (darkened to AA in light)
.btn.ghost      transparent, text --mut, hover --well
.btn.danger     bg --err-soft, border --err at 45%, text --err; hover solid --err with white text
```

States, shared by all variants:
- hover: `translateY(-1px)` plus one shade darker (`*-hover`) and a slightly larger shadow
- active: `scale(.97)`, shadow removed
- focus-visible: `0 0 0 3px color-mix(var(--focus) 35%, transparent)` outside the border
- disabled / `.busy`: opacity .5, no transform, `cursor: not-allowed`
- `.loading`: text stays, a 14px spinner is injected before it, pointer events off. `applyBusy()` in `app.js` adds `.loading` to the `data-busy` button that started the task and plain `.busy` to the others.

Import step CTAs: Scan is `.btn.accent`, Analyze is `.btn.primary`, Group is `.btn.primary` (violet is gone). Run All is `.btn.commit`, Reset is `.btn.danger`. The active-step CTA keeps the scooped dome (`.step.active .actions .btn.primary` and the same rule extended to `.btn.accent`), because `test_active_step_is_larger_with_scooped_cta` requires `border-radius` and `gradient` in that rule. The dome uses `linear-gradient(180deg, lighter-stop, role-color)` so it looks like a lit button, not a second shape language.

Other buttons mapped to roles (builder applies the same logic to any not listed):
- Search submit, Find events, Learn from my choices, Save settings, picker "Use this folder", auth Continue: `.primary`
- Review and apply, Choose keeper, Keep the suggested photo in every group: `.commit`
- Rebuild plan, Preview changes, Verify, Repair, Download page, Show more, Back to saved: `.secondary`
- Lightbox Lock / Find similar / Back to AI choice, bulk Lock / Clear, Close: `.ghost` or `.sm .secondary` depending on surface (ghost on dark surfaces)
- Undo this change, Clear indexed library, Remove them: `.danger`
- `showConfirmDialog` sets `ok.className = "btn " + variant` with `danger | warning | primary`. Add a one-line map in that function so `warning` renders as `.commit`; the call sites do not change.

## 5. Components

- Inputs and selects: 38px, radius 12px, border `--line`, focus shows the same 3px soft ring as buttons plus border in `--focus`. Selects use an inline SVG chevron (data URI, no external URL) tinted `--mut`.
- Toggles (`.toggle`): already pills; on-state uses `--primary-soft` and `--primary` text. The theme switch knob becomes `--grad-brand` when on.
- Segmented control: extract `.scrub-presets` into a reusable `.seg` with `.seg-btn`, used for the date presets only. The Highlights `all / events` `<select id="hl-view">` stays a select (app.js reads its `.value` and a browser scenario dispatches `change` on it); it just picks up the new select styling. Active segment has a sliding white (light) or navy (dark) pill behind it, animated with `transform`.
- Nav: `.navitem.active` keeps the pill but moves to tokens (`--accent-soft` fill, 1px `--grad-brand` border via `border-image` or a pseudo-element, text `--accent-700` in light and white in dark). A 3px rounded indicator on the left edge of the active item slides between the items inside `<nav>` (`transition: transform .2s`). Settings lives in `.side-foot` outside `<nav>`; when it is active it shows the pill style only, and the nav indicator hides.
- Count badges: unchanged shape, `.alert` uses `--commit-soft` and amber-700 text.
- Cards (`.sect`, `.step`, `.dup-group`, `.diffpanel`, `.accel`, settings worker fields): 1px border in `--line` plus a very faint top-edge brand gradient line (`::before`, 1px, `--grad-brand` at 35% alpha) so every card shares one signature.
- Thumbnails: selection outline (`.card.sel`, `.picked`) uses `--accent`; best-of-burst stays `--ok`; locked stays amber via `--commit`.
- Star rating in lightbox: five `.star` buttons become 34px rounded squares in a row; the lit state fills `--commit-soft` with `--commit` glyph. Digits 1 to 5 appear as 10px captions under the stars (they are already keyboard shortcuts).
- Bulk bar: floating glass pill (`backdrop-filter`, `--sheet` at 85%, 1px `--line`, soft shadow), centered under `main` using `left: 50%` inside a wrapper that spans the content column, so the hard-coded `252px` offset goes away.
- Toasts: `--sheet` background, `--ink` text, a 4px left rail in `--ok`, `--err` or `--primary`, matching icon. They slide in from the right and fade out (see motion).
- Modals and confirm dialog: card radius 18px, header, body, footer spacing unified; the primary button in the footer uses the role of the action (destructive confirm uses `.danger`).
- Activity log: keep the terminal look. Time pills map to `--primary-soft` (info), `--ok-soft` (ok), `--commit-soft` (warn), muted. Light theme keeps the white log background.
- Progress card and gauge: fill and arc use `--grad-brand`; done uses `--ok`; error uses `--err`. Spectrum bars: first third `--blue-400`, middle white, last third `--pink-500` (that already matches the logo, keep it).

## 6. Motion

Durations: 120ms for presses, 160 to 200ms for state and tab changes, 220ms for overlays. Easing `cubic-bezier(.2,.8,.2,1)` for enters, `ease-in` for exits. Nothing loops while idle.

| Moment | Effect | Where |
|---|---|---|
| Tab switch | outgoing panel fades, incoming fades in and rises 6px | `showTab()` adds `.enter` class to the new `.panel`, removed on `animationend`; CSS keyframe `panel-in` |
| Nav move | indicator slides to the active item | pure CSS transform on an indicator element positioned from the active item's `offsetTop` (set in `showTab()`) |
| Button press | `scale(.97)` on `:active` | CSS only |
| Step done | the check icon pops (`scale .6 to 1`, slight overshoot) once when `.done` is added | CSS animation on `.step.done .step-num .ic` |
| Step chevrons | flow only while a task is running (`body.task-running`, set in `renderProgress`) or while the previous step is done and the next is active; otherwise static at 55% opacity | existing keyframe, gated by selector; keep the `.7s` third delay the test asserts |
| Count badge | brief highlight pulse when its number changes | `renderComp()` toggles `.tick` on changed badges |
| Modals, confirm, picker, help | backdrop fades, card scales 96 to 100% and fades in; reverse on close | add `.closing` before setting `hidden`, remove after `animationend` (fall back to immediate hide under reduced motion) |
| Lightbox | backdrop fade, image cross-fades on navigate (`opacity` on `#lb-img` load), side panel slides in 12px | `openLightbox`, `closeLightbox`, image `onload` |
| Toast | slides in from the right 12px; on dismiss fades and collapses height | `toast()` adds `.out` before removal |
| Bulk bar | rises 10px and fades when selection goes from 0 to n and back | class toggle where `#bulkbar.hidden` is set today |
| Progress fill, gauge arc | keep existing transitions | unchanged |
| Skeleton shimmer, spinner, devchip pulse | keep; they only run while loading or when the device is live | unchanged |

Reduced motion: the blanket rule at the end of `style.css` already forces `.001ms` durations and single iterations and must stay (a test asserts it). JS that waits on `animationend` must also finish immediately when `matchMedia("(prefers-reduced-motion: reduce)")` matches, so hiding never depends on an event that will not arrive in time.

## 7. Cleanup

- Remove the second `.banner`, `.spinner` and `.devchip` definitions; keep one of each and merge needed properties.
- Replace `var(--serif)` and `var(--sans)` (undefined) with `var(--font-display)` and `var(--font-body)`.
- Delete the "Light theme high-contrast typography" override block; instead set light `--mut` usage so it passes AA where it was failing (the baseline `--mut: #475569` already passes on white; the block was compensating for translucent card fills, so the fix is to make `.step` and `.sect` fills opaque enough, 90%+ white, rather than override text colors).
- Remove all `!important` on brand buttons.
- Remove the five-duos comment block and anything else that documents the old palette.

## 8. Constraints the builder must respect

- All assertions in `tests/test_c5_ui.py`, `tests/test_c10_ui.py`, `tests/test_ui.py`, the browser scenarios and the Node grid tests keep passing. The ones that touch styling: baseline tokens, font tokens, `@supports (backdrop-filter`, reduced-motion rule, `.step` elliptical radii without `var(--r-lg)`, `.step.active` with `scale(` and `z-index`, `.step.active .actions .btn.primary` with `border-radius` and `gradient`, tablet `.step.active { transform: none }`, dark body gradient, dark `.step` glass gradient without `border-color`, light `body` referencing `var(--table)`, `.gauge-track` with `stroke` and no `var(--well)`, dark `.version-pill` `color: #fff`, `.step-flow` z-index above `.step.active`, `.dock` with `border-radius`, `.devchip` without `width: 100%`.
- Older suites pin the old palette by name: `tests/test_c7_ui.py` (light `--warn: #b45309`), `tests/test_c8_ui.py` (`--accent-electric*`, `--accent-glow`, `--accent-pill`, `--ok-sage*`, `--danger-crimson*`, `--viz-bar*`, dark `--canvas-bg: #0b0e14`, electric tokens inside `.step.active`) and `tests/test_c9_ui.py` (`.sect > h2` using `--accent-electric`, `.step.done .step-flow` using `--ok-sage`). Decision: delete only the assertions in those three files that pin a colour token or value; keep every other test in them (switch markup, reduced motion, structure, JS behaviour). My call on top of that: add a small `tests/test_ui_tokens.py` that pins the light and dark values of `--primary`, `--accent`, `--commit` and `--focus`, so the new palette has the same regression net the old one had.
- No change to ids, `data-tab` keys, `data-busy`, `data-goto`, or any selector `app.js` queries. New classes are additive.
- Light and dark themes both ship complete. Every new color pairing is checked against AA; the builder records the ratios for brand fills with navy ink and for `--link`, `--warn` and `.secondary` text in the commit body.
- Style the real markup, then open the app (`photocure serve --no-open`) and walk every tab in both themes before calling it done. Screenshots in `docs/` are refreshed as part of the work.

## 9. Left to the builder

- Exact alpha values for glows and soft fills, within the ranges above.
- Whether the nav indicator is a separate element or a pseudo-element, as long as it slides.
- How the `.loading` spinner is injected (CSS `::before` is fine).
- Order of work. A sensible order is tokens, cleanup, buttons, components, motion, screenshots.
- Updating the three existing UI spec documents in `docs/` is optional; this document supersedes their palette sections.
