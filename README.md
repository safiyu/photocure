<p align="center"><img src="docs/logo.png" width="140" alt="Photocure logo"></p>

# Photocure Studio

**Sort, cull, edit and tidy your photo library, privately, on your own computer.**

Photocure analyzes your photos locally, identifies subject matter, picks the sharpest shot in every burst, catches duplicates, and organizes everything into a clean folder structure. Nothing leaves your machine, and your files stay untouched until you review and confirm.

<p align="center">
  <img src="docs/shot_light.png" width="48%" alt="Photocure Studio Pipeline page, light theme">
  <img src="docs/shot_dark.png" width="48%" alt="Photocure Studio Pipeline page, dark theme">
</p>

## Contents

[What it does](#what-it-does) · [Install](#install) · [Quickstart](#quickstart) · [Library stages](#library-stages) · [The app](#the-app) · [Photo editor](#photo-editor) · [Shortcuts](#keyboard-shortcuts) · [Safety](#your-photos-stay-safe) · [Extras](#optional-extras) · [Command line](#command-line) · [Troubleshooting](#troubleshooting)

## What it does

- **Smart categorization:** Sorts photos by scene and subject (people, food, animals, landscapes, vehicles, architecture, art, and more) without manual tagging.
- **Burst culling & face inspection:** Identifies the best shot in every burst or retake. Detects closed eyes and blinks, and can blend open eyes across group frames into a composite photo.
- **Duplicate detection:** Catches exact matches and similar retakes, recommending the best copy to keep.
- **Utility triage:** Automatically separates receipts, whiteboards, screenshots, and labels from your memories.
- **Highlight curation:** Ranks top photos and composes balanced event stories.
- **Natural search:** Find photos by text description (e.g. "golden retriever running on the beach").
- **Built-in editor:** Polish keeper shots, crop, adjust lighting, denoise, or extract text right inside the app.
- **Safe, non-destructive file management:** Preview all changes before applying, with complete one-click rollback.

Works with JPEG, PNG, WebP, TIFF, HEIC and common RAW formats (CR2, NEF, ARW, DNG). Libraries of 100,000+ photos scroll smoothly.

**New in 2.x:** Built-in [photo editor](#photo-editor), burst collages, library reanalysis, direct Trash staging, and enhanced burst detection.

## Install

You need a Mac, Linux or Windows computer, a few GB of disk for the AI model, and an internet connection for the first setup only (runs completely offline afterwards). Apple Silicon and NVIDIA GPUs provide accelerated inference, but standard CPUs work as well. Python 3.11+ is required for source installs.

**Debian or Ubuntu**

```bash
curl -LO https://github.com/safiyu/photocure/releases/latest/download/photocure_amd64.deb
sudo apt install ./photocure_amd64.deb
photocure setup
```

**Mac, Windows or any Linux (from source)**

```bash
git clone https://github.com/safiyu/photocure.git && cd photocure
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -e .
photocure setup
```

`photocure setup` detects your hardware, downloads the models (with confirmation), and verifies the setup. Useful flags: `--yes`, `--dry-run`, `--cpu-only`, `--no-faces`, `--no-open`.

**Docker** (ideal for a NAS or home server):

```bash
PHOTOS_DIR=/path/to/photos OUTPUT_DIR=/path/to/organized docker compose up
```

Open <http://localhost:8080/>. For GPU acceleration, add `-f docker-compose.gpu.yml` (NVIDIA Container Toolkit), `docker-compose.rocm.yml` (AMD), or `docker-compose.intel.yml`. Uses `/output/Organized` by default so it never prompts for a library path. For Portainer, see [docker-compose.stack.yml](docker-compose.stack.yml).

## Quickstart

Start the app with `photocure start` (or `./start.sh`) and open <http://localhost:8080/>:

1. **Scan & Analyze:** Pick your source photo folder and click **Scan**, then **Analyze Now**.
2. **Review & Triage:** Inspect bursts, accept duplicate suggestions, and mark unwanted shots.
3. **Plan & Apply:** Preview proposed moves or copies in **Apply and undo**, then apply with one click.

To stop, press `Ctrl+C` or run `photocure stop`.

## Library stages

The stage switcher at the top right navigates your photos through three states:

- **New:** Scanned photos pending organization. Analyze, inspect suggestions, and prepare your plan here.
- **Library:** Finished photos organized into your target folder. Recategorize, rename people clusters, edit photos, or run **Reanalyse library** to discover missed bursts as an undoable operation.
- **Trash:** Staged removals awaiting deletion. Photos can be restored, returned to origin (re-staged as New), or permanently emptied.

To relocate your organized library later, use **Move library to another folder** in Settings: Photocure copies and verifies all assets before updating paths.

## The app

- **Import:** Select a folder, run **Scan** and **Analyze Now** (or **Run All**). Optional **Keep watching this folder** auto-imports incoming files.
- **Library:** Full catalog view sorted chronologically. Filter by category, camera, rating, country, favorites, or date. Open any photo to rate, favorite, lock, recategorize, inspect detected faces, or edit.
  - **Mark for delete:** Available on individual photos, sections, or selection sets. Dimmed items stage into Trash upon clicking **Finalize** or applying your plan.
  - **Clean export:** Exports sanitized, share-ready copies with EXIF/location metadata stripped, optional neutral filenames, web gallery generation, and automated redaction (blurring license plates, payment cards, screens, and ID documents).
- **Search:** Natural language semantic search across your entire collection.
- **Bursts:** Groups high-speed sequences and retakes. Flags the best overall frame and checks for closed eyes:
  - **Synchronized 1:1 Loupe:** Compare 2–4 candidate frames with locked pan and 100% zoom (shortcut **Z**) to verify eye sharpness and fine detail.
  - **Create fixed photo:** Blends open eyes across complementary group frames (`IMG_xxxx_fixed.jpg`).
  - **Export layered PSD:** Generates aligned layers for manual retouching in Photoshop or GIMP.
  - **Make collage:** Merges 2–5 burst frames into a custom grid layout up to 4K resolution.
  - **Make best shot:** Manually overrides the selected winning frame.
- **Highlights:** Surface your best shots using an aesthetic scoring engine combined with your star ratings and favorites. Includes event story views, an exportable **Year in review** web page, and a **Target Keeper Budget** ("Cull to $N$ Photos" or Top %) that selects a visually balanced cut across narrative scales.
- **Events:** Automatically clusters photos into discrete events by time and GPS proximity.
- **Insights:** Visual disk space analytics: live reclaimable storage meters (duplicates, non-winning burst retakes, junk), camera and category breakdowns, burst reduction efficiency, and date resolution audits.
- **Utility:** Isolates functional documents (receipts, whiteboards, notes, screenshots, package labels) so they don't clutter your photographic memories. Features one-click text extraction (OCR) directly in photo previews.
- **Finalize:** Swiftly triage edge cases and uncertain classifications with numeric hotkeys; the classifier learns from your adjustments.
- **Duplicates:** Side-by-side comparison of duplicate and near-duplicate images with one-click resolution.
- **Junk:** Central review for confirmed duplicates, memes, severely blurred shots, and unreadable files.
- **Apply and undo:** Review planned disk operations (Move or Copy mode), verify paths, and commit changes with complete atomic rollback support. Live Photo companion clips (`.mov`, `.mp4`) and sidecars are tracked and moved atomically with parent images. Supports downstream webhooks or shell commands on completion.
- **Settings:** Configure library directory, collision handling (files are never overwritten), digiKam/Lightroom sidecar sync, Google Takeout metadata sidecar ingest, multi-camera clock drift alignment, downstream webhook triggers, worker count, and appearance.

## Photo editor

Once you've culled a burst or rescued a memory, you can polish it on the spot without opening an external editor. Press **Edit** in the photo viewer to access the toolkit:

| Category | Tools | What it does |
|---|---|---|
| **Polish Memories** | **Crop & Rotate** | Freeform and aspect ratio crops, 90° rotations, flips, and fine leveling |
| | **Light** | Exposure, contrast, and saturation adjustments |
| | **Looks** | 7 skin-safe color grades (Vivid, Portrait, Film, Cinematic, Golden, Matte, Mono) with strength control; no hue distortion so skin tones remain natural |
| | **Detail & Red-Eye** | Selective sharpening, softening, and red-eye removal |
| | **Remaster** | AI denoise, low-light rescue, and super-resolution upscaling for older, grainy, or under-exposed shots |
| **Quick Fun & Annotate** | **Remove BG** | AI subject isolation with fallback edge cutout |
| | **Draw & Stickers** | Freehand pen, highlighter, eraser, and emoji stickers for family notes and quick sharing |
| | **QR Code** | Embed scannable QR code stickers directly onto photos |

- **Unified Apply bar:** Live previews for Crop, Light, Looks, and Detail with instant Undo/Redo (up to 10 steps).
- **Split compare:** Press **Compare** (or shortcut **C**) for a draggable side-by-side view against the unedited original.
- **Non-destructive save options:**
  - **Save as copy:** Writes `name_edited.ext` alongside the original file.
  - **Overwrite original:** Replaces the file with safety checks (refuses RAW/HEIF and aborts if conversion is lossy).
  - **Download:** Directly exports any format and quality without modifying library files.
- **Hardware-friendly canvas:** Editing is rendered on an accelerated 4K-capped canvas.
- **Utility OCR:** Text extraction from receipts, whiteboards, and notes is available directly in the Utility tab previews (integrates Tesseract or `pip install "photocure[ocr]"`).

## Keyboard shortcuts

Press **?** anywhere in the app for the full shortcut cheat sheet.

| Key | What it does |
|---|---|
| Arrow keys, Enter or Space | Navigate grid, open highlighted photo |
| **1** to **5**, **0** | Set star rating, clear rating |
| **x**, **f**, **l** | Reject, favorite, lock (toggle) |
| **j** and **k** | Next / previous photo in Finalize and Junk |
| **1** to **9** | Assign category in Finalize and Junk |
| **u** | Undo last category assignment |
| **z** | Toggle synchronized 1:1 Loupe in burst view |
| Esc | Close lightbox or dialog |

## Your photos stay safe

- **Preview before action:** Nothing moves or copies until you review the plan in **Apply and undo**. Every operation is recorded in an undo journal for complete single-click rollback.
- **Protected originals:** Scanning, analyzing, rating, and triage operations never modify your files. RAW and HEIF files are never overwritten in the editor.
- **Trash safety net:** Rejected or deleted photos remain in the Photocure Trash until you deliberately purge them.
- **Metadata compatibility:** Ratings and reject flags write to standard `.xmp` sidecars recognized by Lightroom, digiKam, and Apple Photos (disable via `PCUR_WRITE_SIDECARS=false` if desired).
- **Collision proofing:** Existing target files are never overwritten; duplicate names are safely suffixed.
- **100% private and local:** No telemetry, no external API calls, and no cloud uploads.

## Optional extras

**Offline Geocoding:** Fill in city, region, and country names from GPS EXIF coordinates offline:
```bash
photocure geocode
```

**Custom Taxonomy:** Extend or customize categorization rules with a YAML or JSON file:
```bash
export PCUR_TAXONOMY_FILE=/full/path/to/my_categories.yaml
```

```yaml
classes:
  - name: concert
    folder: Events/Concerts        # destination subfolder
    group: people                  # visual group
    date_grouped: true             # organize into year/month/day
    prompts:                       # semantic descriptions for the AI classifier
      - a concert with a stage and bright lights
```

**AI Agent Integration (MCP):** Connect local AI agents (such as Claude Desktop) via the Model Context Protocol. The server provides 8 read-only inspection tools (overview, search, browse, details, photo view, burst review) and 2 safe creative tools (`create_composite`, `clean_export`):

```bash
pip install "photocure[mcp]"
claude mcp add photocure -- photocure mcp --db /full/path/to/photocure.db
```

## Command line

| Command | What it does |
|---|---|
| `photocure setup` / `start` / `stop` | Initialize environment, start or stop server |
| `photocure search "dog on the beach"` | Semantic search by text prompt |
| `photocure highlights --year 2024 --top 50 --out DIR` | Generate year-in-review web summary (`--copy` copies originals) |
| `photocure keywords` | Sync IPTC/XMP tags directly into image files (`--remove` reverts) |
| `photocure geocode` | Populate reverse-geocoded place names from GPS |
| `photocure mcp --db FILE` | Start the Model Context Protocol agent server |
| `photocure doctor` / `devices` | Verify model weights, dependencies, and accelerator devices |
| `photocure plan` / `apply` | CLI-driven plan generation and execution (`--operation move\|copy`) |
| `photocure trash-purge` | Permanently empty the trash staging area |
| `photocure version` | Print version information |

## Troubleshooting

- **"The AI model is switched off":** An environment override is forcing the stub backend. Run `unset PCUR_BACKEND` (or remove the variable on Windows), restart, and run **Analyze Now**.
- **Model weights missing:** Run `photocure setup` to download the required local models.
- **Sign-in authentication prompt:** The one-time token is output in your console upon running `photocure start`.
- **Port conflicts:** If 8080 is occupied, Photocure automatically binds to the next available port, or specify `--port <PORT>`.
- **Inference performance:** Verify GPU access using `photocure devices` and adjust workers under Settings -> Processing device.
- **System diagnostics:** Run `photocure doctor` for an end-to-end environment audit.

---

Photocure is released under the MIT license.
