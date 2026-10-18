# C8: Photocure Modern UI/UX Overhaul Specification (Import & Workspace Canvas)

> **Status**: Ready for Implementation  
> **Target Branch**: `develop`  
> **Source Design**: `/home/safiyu/Downloads/new.jpeg`  
> **Compliance**: WCAG 2.1 AA, No External CDN Dependencies, Zero API Regressions  

---

## 1. Executive Summary

This specification defines the complete UI/UX overhaul of Photocure's primary workspace, aligning the interface with the modern, high-contrast, dark-mode design specified in `/home/safiyu/Downloads/new.jpeg`. 

The design transforms Photocure from a basic utility interface into a sleek, dark-slate workstation with:
- **Atmospheric Glow & Focused State Hierarchy**: Clear visual focus on active steps (e.g. Step 2 "Analyze" with electric blue border and soft atmospheric glow).
- **Audio / AI Equalizer Spectrum Visualizer**: Integrated frequency wave/spectrum bar visualizer embedded into the AI Inference progress card.
- **High-Affordance Workflow Cards**: Redesigned three-column step engine with distinct status indicators (Completed, In Progress, Locked/Pending with isometric burst illustration).
- **Modernized Dark Mode Sidebar**: Enhanced navigation capsule pills, device chip with glowing green status dot, and a native hardware-style sliding Dark Mode toggle switch.
- **Terminal-Grade Activity Panel**: High-readability event stream featuring tabular timestamp pills (`.time-pill`), active row status highlighting, and inline completion glyphs.

---

## 2. Visual Architecture & Design System Tokens

All styling adheres to Photocure's self-contained font bundle (`Plus Jakarta Sans` and `JetBrains Mono`) and strictly preserves WCAG 2.1 AA text contrast (≥ 4.5:1 for regular text, ≥ 3.0:1 for large/accent text).

### 2.1 Color Palette & CSS Custom Properties

Existing base tokens (`--table`, `--sheet`, `--ink`, `--mut`, `--faint`) are retained for backwards compatibility with test assertions in `tests/test_c5_ui.py`. New design tokens are introduced into `:root` and `:root[data-theme="dark"]`:

```css
:root[data-theme="dark"] {
  /* Core Base Surfaces (Obsidian Slate) */
  --table: #0c0e12;               /* Root canvas background */
  --canvas-bg: #0b0e14;           /* Deep canvas */
  --sheet: #131926;               /* Primary card surface */
  --sheet-subtle: #0f141f;        /* Sidebar & secondary panels */
  --sheet-border: #1e2638;        /* Subtle component borders */
  --sheet-border-light: rgba(255, 255, 255, 0.08);

  /* Typography */
  --ink: #f1f3f7;                 /* Primary high-contrast text */
  --mut: #949eb2;                 /* Muted descriptions */
  --faint: #626d82;               /* Secondary captions & counts */

  /* Electric Blue Accent (Active Focus & AI Tasks) */
  --accent-electric: #3b82f6;     /* Primary neon cobalt */
  --accent-electric-hover: #60a5fa;
  --accent-glow: 0 0 20px rgba(59, 130, 246, 0.28);
  --accent-pill: rgba(59, 130, 246, 0.16);

  /* Semantic Feedback Tokens */
  --ok-sage: #22c55e;             /* Completed green */
  --ok-sage-bg: #14291f;          /* Completed chip background */
  --ok-sage-border: rgba(34, 197, 94, 0.35);

  --danger-crimson: #ef4444;      /* Destructive actions */
  --danger-crimson-bg: #261619;   /* Destructive pill background */
  --danger-crimson-border: rgba(239, 68, 68, 0.35);

  /* Equalizer Spectrum Visualizer */
  --viz-bar: #38bdf8;
  --viz-bar-faint: rgba(56, 189, 248, 0.22);
}
```

---

## 3. Component Specifications

### 3.1 Sidebar Navigation (`aside#side`)

```
+----------------------------------------------------+
| [o] Photocure                                      |
|     Intelligent image curator                      |
|                                                    |
| GET STARTED                                        |
| [ (icon) Import ]   <-- Active capsule pill        |
|                                                    |
| BROWSE                                             |
|   (grid)   Library                    [ 3,194 ]    |
|   (layers) Bursts                                  |
|   (star)   Highlights                     [ 3 ]    |
|   (cal)    Events                         [ 2 ]    |
|                                                    |
| REVIEW                                             |
|   (eye)    Needs a look                            |
|   (trash)  Junk review                             |
|                                                    |
| FINISH                                             |
|   (swap)   Apply and undo                          |
|                                                    |
|----------------------------------------------------|
| [*] Running on NVIDIA GPU   (Sage green chip)      |
| [gear] Settings                                    |
| [moon] [?]   Dark Mode [ (o) ]  (Switch slider)    |
| Photocure 0.1.0 - by safiyu                        |
+----------------------------------------------------+
```

1. **Active State (`Import`)**:
   - Background: `--accent-pill` (`rgba(59, 130, 246, 0.16)`).
   - Border: `1px solid rgba(59, 130, 246, 0.3)`.
   - Text & Icon: Solid crisp white (`#ffffff`).
   - Radius: `var(--r-sm)` (8px) or capsule `20px`.
2. **Category Headings**:
   - Font: Uppercase, `letter-spacing: 0.05em`, size `11px`, color `--faint`.
3. **Tabular Count Badges**:
   - Rendered using monospace/tabular font (`font-variant-numeric: tabular-nums`).
   - Badges on Library (`3,194`), Highlights (`3`), and Events (`2`) with subtle pill container.
4. **Device Status Chip (`#device-chip`)**:
   - Capsule container styled with sage green border and soft dark green background (`--ok-sage-bg`).
   - Glowing emerald status dot (`•`) with subtle pulse animation.
   - Text: `Running on <Device>` (e.g. `Running on NVIDIA GPU`).
5. **Theme Switch (`#theme-switch`)**:
   - Replaces the simple icon button with an accessible HTML `<label class="switch">` containing a hidden checkbox and an animated slider track with round knob.
   - Keyboard accessible (`tabindex="0"`, responds to Space/Enter and click).
   - Syncs seamlessly with `localStorage.getItem("pcur-theme")` and updates `document.documentElement.dataset.theme`.

---

### 3.2 AI Inference Progress Card with Equalizer Spectrum (`#progress-card`)

```
+----------------------------------------------------------------------------------------------------+
| (o) AI Inference: Running quality + faces + CLIP semantics...   ||||||||||||||||||||||||||||    [x]|
| Phase 1/2: 135 / 3,194 photos (2% overall)                                              135 / 3,194|
| [=================================================================================================]|
+----------------------------------------------------------------------------------------------------+
```

1. **Container**:
   - Background: `#131926`, border: `1px solid #1e2638`.
   - Rounded corners: `12px`.
   - Padding: `16px 20px`.
2. **Left Status Details**:
   - Pulsing electric-blue spinner or circular progress ring.
   - Task label: `AI Inference: Running quality + faces + CLIP semantics...` (bold, 14px, white).
   - Subtitle: `Phase 1/2: 135 / 3,194 photos (2% overall)` (13px, color `--mut`).
3. **Equalizer Spectrum Visualizer (`.spectrum-viz`)**:
   - Set of 24–32 vertical bar elements (`.spectrum-bar`) styled with CSS heights ranging from 4px to 24px.
   - Animated when `p.active` using CSS keyframe wave pulses (`animation: eq-pulse 1.2s ease-in-out infinite alternate`).
   - Guarded by `@media (prefers-reduced-motion: reduce)` which freezes the animation into static equalized bars.
4. **Controls & Metrics**:
   - Right-side dismiss `[✕]` button.
   - Live processed count: `135 / 3,194` in crisp tabular numbers.
   - Horizontal progress bar spanning the bottom of the card with electric-blue progress fill.

---

### 3.3 Library Overview & Distribution Bar (`.comp-host`)

```
3,194 photos in your library
[====================================================================================================]
* Unrouted 3,194   Photos are placed in categories when a plan is built   [ Sort photos now ]
```

1. **Header**:
   - Bold numerical counter: `3,194 photos in your library`.
2. **Progress Strip (`.comp-strip`)**:
   - Single continuous bar or segmented category bar with subtle rounded ends (`height: 8px`).
   - Muted slate-blue track background with vibrant fill.
3. **Status Line & Sort Trigger**:
   - Dot indicator: `• Unrouted 3,194`.
   - Explanation text: `Photos are placed in categories when a plan is built`.
   - Action Button: `[ Sort photos now ]` dark capsule button with 1px border.

---

### 3.4 Three-Step Workflow Engine (`.steps`)

The layout features three equal-width cards in a responsive CSS Grid (`grid-template-columns: repeat(3, minmax(0, 1fr))`).

```
+---------------------------+  +---------------------------+  +---------------------------+
| (1) Find your photos      |  | (2) Analyze           ... |  | (3) Group bursts          |
|              3,194 indexed|  |                           |  |               Needs step 2|
|                           |  |                           |  |                           |
| Pick the folder that      |  | Finds faces, checks       |  | Finds shots taken seconds |
| holds your photos...      |  | sharpness and exposure... |  | apart and keeps best...   |
|                           |  |                           |  |                           |
| Photo folder              |  | Run on                    |  |       [=== Layer 3 ===]   |
| [/home/safiyu/Pictures Q] |  | [ [*] NVIDIA GPU      v ] |  |     [=== Layer 2 ===]     |
|                           |  | Devices and workers in... |  |   [=== Layer 1 ===]       |
| [ Browse Folder ] [ Scan ]|  | [ > Analyze Now ]         |  | [ Group ]             (L) |
+---------------------------+  +---------------------------+  +---------------------------+
  Status: Completed (Green)      Status: Active (Blue Glow)     Status: Pending / Locked
```

#### Step 1: "Find your photos"
- **Badge**: Green circle with number `1` (`--ok-sage`).
- **Status Indicator**: `3,194 indexed` top right badge in sage green.
- **Form Input**:
  - Label: `Photo folder`.
  - Input field with dark slate background, inline search/magnifying glass icon on right edge.
- **Actions**:
  - `[ Browse Folder ]`: Capsule button with folder icon and solid dark-slate fill.
  - `[ Scan ]`: Secondary button with camera/scan framing icon.

#### Step 2: "Analyze" (Active / Highlighted)
- **Active Border & Glow**:
  - Border: `1.5px solid var(--accent-electric)`.
  - Box Shadow: `0 0 20px rgba(59, 130, 246, 0.28)`.
- **Badge**: Blue circle with number `2`.
- **Context Menu**: Top-right horizontal ellipsis button `...` for advanced inference settings.
- **Device Selector Chip**:
  - Custom styled `<select>` or custom dropdown pill: Microchip icon + `NVIDIA GPU` + chevron caret.
  - Sublink: `Devices and workers in Settings` (`color: var(--accent-electric)`).
- **Action**:
  - `[ ▷ Analyze Now ]`: Outlined or filled electric blue pill button with play triangle glyph.

#### Step 3: "Group bursts" (Pending / Locked)
- **Visual State**: Dimmed opacity (`opacity: 0.55`) indicating prerequisite dependency.
- **Badge**: Muted gray circle with number `3`.
- **Top-right Tag**: `Needs step 2`.
- **Visual Illustration**: Isometric 3D wireframe wirecard stack (SVG/CSS illustration of 3 layered burst photos).
- **Action**:
  - Disabled `[ Group ]` button with lock glyph `🔒` on the trailing edge.

---

### 3.5 Batch Action Strip (`.runall`)

```
[ Run all three steps ]   [ Reset library ]   Choose a folder, then run all three steps. Nothing is moved until you apply a plan.
```

1. **`[ Run all three steps ]`**:
   - Capsule button (`border-radius: 20px`), solid slate-blue background, crisp white typography.
2. **`[ Reset library ]`**:
   - Capsule button with dark terracotta background (`--danger-crimson-bg`), red border (`--danger-crimson-border`), and crimson text (`--danger-crimson`).
3. **Safety Caption**:
   - Muted helper text assuring user that files are untouched until plan application.

---

### 3.6 Terminal Activity Feed (`#library-log`)

```
Activity ~-~-~
+----------------------------------------------------------------------------------------------------+
| [ 9:04:36 PM ] Analyzing photos (cuda)...                                                          |
| [ 9:04:34 PM ] Scan finished: 3,194 files seen, 3,194 new, 162 duplicates in 9.97s              [v]|
| [ 9:04:24 PM ] Scanning /home/safiyu/Pictures...                                                   |
| [ 9:04:39 PM ] Scan finished: 0 files seen, 0 new, 0 duplicates in 0s                              |
+----------------------------------------------------------------------------------------------------+
```

1. **Section Header**:
   - Title: `Activity`.
   - Accompanying Icon: Glowing pulse / EKG heartbeat line 〽️ (`color: #60a5fa`).
2. **Log Container**:
   - Dark background (`#0e121a`), border `1px solid #1a2233`, radius `10px`, max-height `320px`, smooth scrollbar.
3. **Log Row Formatting**:
   - **Timestamp Badge (`.time-pill`)**:
     - Monospace tabular font (`JetBrains Mono`).
     - Distinct pill background based on log type:
       - Info/Progress: Light blue tint (`rgba(59, 130, 246, 0.15)`, text `#93c5fd`).
       - Success/Finished: Sage green tint (`rgba(34, 197, 94, 0.15)`, text `#86efac`).
       - Muted: Subtle gray tint (`rgba(148, 158, 178, 0.12)`, text `#949eb2`).
   - **Active Completed Row Highlight**:
     - When a major job completes (e.g. Scan finished), the row displays a subtle green outline (`border: 1px solid rgba(34, 197, 94, 0.35)`), soft green background tint, and a trailing checkmark glyph `✓` on the right side.

---

## 4. Implementation Details & File Touchpoints

### 4.1 `photocure/ui/static/index.html`
- Update sidebar navigation markup:
  - Add `switch` component for Dark Mode toggle.
  - Add count badges on Library, Highlights, Events.
  - Update `#device-chip` with green status dot.
- Update `#progress-card`:
  - Add container for equalizer visualizer bars (`<div class="spectrum-bars">...</div>`).
  - Add live progress counter `<span class="progress-ratio"></span>`.
- Update `.steps` list items:
  - Add search icon into `.with-btn` input container.
  - Add "Browse Folder" button styling.
  - Update Step 2 with options menu button and device select chip.
  - Add 3D stacked burst cards SVG to Step 3.
  - Add lock icon to Step 3 action button.
- Update `#library-log` header with SVG pulse line icon.

### 4.2 `photocure/ui/static/style.css`
- Add `:root[data-theme="dark"]` custom properties without mutating the test-asserted baseline tokens.
- Add `.spectrum-bars` and `@keyframes eq-pulse`.
- Add active glow `.step.active` and completed `.step.done` styles.
- Add `.switch` toggle component styling.
- Add `.time-pill` styling and `.logline.active-highlight`.
- Add `@media (max-width: 960px)` responsive breakpoints (stacks 3-column steps into 1 column on tablet/mobile).
- Add `@media (prefers-reduced-motion: reduce)` overrides.

### 4.3 `photocure/ui/static/app.js`
- `renderProgress(p)`:
  - Render live ratios (e.g. `p.done + " / " + p.total`).
  - Toggle spectrum bars animation based on `p.active`.
- `logLine(msg, type)`:
  - Format timestamps into `<span class="time-pill">` pills with tabular numbers.
  - Add checkmark icon for successful scan/infer completion lines.
- `initThemeSwitch()`:
  - Wire toggle switch to localStorage and document dataset theme.
- Update step states:
  - Highlight Step 2 with `.active` class when scanning is completed or infer is running.
  - Add `.locked` state to Step 3 when analysis has not run.

---

## 5. Verification & Acceptance Criteria

1. **Zero External Requests**:
   - `tests/test_c5_ui.py::test_zero_external_urls` passes (no remote CDN fonts, CDNs, or external scripts).
2. **WCAG 2.1 AA Contrast**:
   - All text pairings on `--sheet` and `--canvas-bg` maintain ≥ 4.5:1 contrast.
   - Form controls have visible 2px focus indicators (`:focus-visible`).
3. **Theme & Motion Compliance**:
   - Dark Mode switch toggles theme seamlessly between dark and light.
   - `@media (prefers-reduced-motion: reduce)` disables equalizer bar animations and pulse effects.
4. **Behavioral Invariance**:
   - Folder browsing, scanning, device selection, GPU inference, clustering, and library reset perform identically with no backend API changes.
5. **Automated Test Suite**:
   - `.venv/bin/pytest tests/test_c5_ui.py tests/test_ui.py` passes 100% green.
