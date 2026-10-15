<p align="center"><img src="docs/logo.png" width="140" alt="Photocure logo"></p>

# Photocure

**Intelligent image curator.**

An intelligent, **offline**, GPU-accelerated local photo curation engine. It scans
large heterogeneous photo libraries (JPEG, PNG, HEIC, WebP, RAW), scores quality,
detects faces, runs zero-shot semantic classification with OpenCLIP, groups
burst shots, and routes everything into a category-first folder tree - all
**transactionally reversible** with a 100% undo guarantee.

No files are ever mutated before a reviewed plan. Every filesystem operation is
journalized in an action ledger and can be rolled back byte-for-byte.

---

## How it works

```
scan → infer → cluster → plan → apply → (undo)
```

| Stage | What happens |
|---|---|
| **1. Scan** | Multi-threaded recursive walk. Two-tier dedup (xxHash64 head/tail → SHA-256 only on collision). Date cascade: EXIF `DateTimeOriginal` → `DateTimeDigitized` → `DateTime` → XMP sidecar → strict filename regex → `mtime` → `Unknown_Date/` (never 1970). Rescans are incremental (size/mtime), `--prune-missing` marks vanished files, and a 64-bit pHash links near-duplicates. |
| **2. Infer** | Sequential GPU phases - never both models in VRAM at once, and **never the whole library in RAM** (frames are decoded in `infer_chunk_size` chunks and released per chunk). Phase A: faces (InsightFace/SCRFD, Haar fallback, stub). Memory sweep. Phase B: OpenCLIP ViT-B-16 zero-shot classification + 512-dim L2-normalized embeddings. Quality metrics (Laplacian blur, luminance, clipped-tail ratios) on the same decoded frame. |
| **3. Cluster** | Burst detection: `Δt ≤ 30 s AND cosine(e_i, e_j) ≥ 0.88`, graph-constrained temporal sliding window (`O(N log N)` sort + in-window pairwise, union-find transitivity). Best-shot score: `0.40·blur + 0.30·face_sharp + 0.20·eye_open + 0.10·log(resolution)`. |
| **4. Plan** | Pure decision matrix over DB state → destination path graph + HTML/CLI audit report. Manual buckets and locked rows win over the matrix. The plan is **persisted** with a content hash of its inputs. **Zero filesystem mutation.** |
| **5. Apply** | Executes exactly the persisted plan (`--plan-id`) and refuses if the state store changed since, or if the plan was already applied (`--force` overrides). `move` / `copy` / `hardlink` / `symlink` / junk-to-trash staging, with collision renaming (`_dup<sha6>`). Each op: ledger row `PENDING` (committed to SQLite) → FS mutation → `COMMITTED` (committed again). |
| **6. Undo** | Reverse-chronological rollback of the ledger with pre-flight SHA-256 verification (refuses to proceed on mismatch, `--no-verify` skips it). Rows left `PENDING` by a crash are reconciled against the disk first. |

**VRAM budget (RTX 4060 8 GiB shared with the desktop):** faces ≈ 1.2 GiB,
CLIP ≈ 2.5 GiB, sequential phases → peak < 4 GiB, with
`torch.cuda.set_per_process_memory_fraction(0.75)` capping PyTorch's share.

### Decision matrix

```
SHA-256 duplicate?           → Junk_Review/Duplicates/
decode failed?               → Junk_Review/Unreadable/
near-duplicate (pHash)?      → Junk_Review/Near_Duplicates/
dark / blurry / blown out    → Junk_Review/Accidental/   (not for screenshots and documents)
face detected (conf > 0.65)? → People/YYYY/MM/DD/
top CLIP = screenshot/doc/   → Junk_Review/Screenshots|Documents|Graphics/
meme/graphic
top CLIP = illustration      → Non-Human/Art_and_Diagrams/
top CLIP = slide             → Non-Human/Slides_and_Notes/
top CLIP = food              → Non-Human/Food/YYYY/MM/DD/
top CLIP = pet/animal        → Non-Human/Animals/YYYY/MM/DD/
top CLIP = landscape         → Non-Human/Scenery/YYYY/MM/DD/
top CLIP = vehicle           → Non-Human/Vehicles/YYYY/MM/DD/
top CLIP = building/street   → Non-Human/Architecture/YYYY/MM/DD/
top CLIP = plant/flower      → Non-Human/Plants/YYYY/MM/DD/
top CLIP = product           → Non-Human/Products/
not confident enough         → Uncertain_Review/
in burst?                    → best shot kept, alternates → Bursts/Burst_<ID>/alternates/
```

- **Group aware confidence:** look-alike classes (for example screenshot, document and
  slide) are judged together. A photo that is clearly "some kind of screen content" is no
  longer Uncertain just because screenshot and document scored close to each other.
- **Screenshots and documents** skip the dark/blurry quality gate, since a dark mode
  screenshot is not an accident.
- **Learn from corrections:** every photo you move by hand teaches its category. Once a
  category has `PCUR_TAXONOMY_FEEDBACK_MIN_EXAMPLES` examples, `pc learn` (or the
  dashboard) re-scores the rest using those examples as well as the zero-shot prompts.
- **Placeholder results are redone:** photos analyzed while the AI model was missing are
  tagged, and `pc infer` redoes them once the model is installed.
- **Custom classes:** edit the taxonomy file (`PCUR_TAXONOMY_FILE`) to add your own.

### Organization layouts

- **Option 1 - `category_first` (default):** `People/2024/05/12/…`,
  `Non-Human/{Food,Animals,Scenery,Vehicles,Architecture,Plants,Products,Art_and_Diagrams,Slides_and_Notes}/…`, `Junk_Review/…`, `Bursts/…`.
- **Option 2 - `in_place`:** originals stay put; hardlinks/symlinks are created
  in the target tree (no extra storage for hardlinks).

---

## Install

```bash
cd photocure
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
photocure setup
```

`photocure setup` does the rest. It detects your hardware (NVIDIA, AMD, Intel, Apple
or CPU), installs the matching PyTorch build, OpenCLIP and the face detector, downloads
the model weights, checks that the model loads, then starts the web UI and opens your
browser. It only installs what is missing, so running it again is quick.

| Option | What it does |
|---|---|
| `--dry-run` | Show the plan and the download size, install nothing |
| `--yes` | Do not ask before downloading |
| `--no-faces` | Skip face detection (it can need a C++ compiler on some systems) |
| `--cpu-only` | Ignore any GPU and install the CPU build |
| `--no-serve`, `--no-open` | Install only, or start the UI without opening a browser |
| `--reinstall` | Replace PyTorch, for example after changing GPU |
| `--verify` / `--no-verify` | Force or skip loading the model to check it |

Set `PCUR_ACCEL=cuda`, `rocm`, `xpu` or `cpu` to skip detection, and `PCUR_TORCH_INDEX` to
use a different PyTorch wheel index, such as an older CUDA build. If the optional face
package fails to install, setup remembers and does not retry on every start. The model
weights (about 350 MB) download once and then work offline. Without them, photos are
classified by a **placeholder** that cannot recognise anything, so most end up Uncertain.

Prefer to install by hand? These extras still work:

```bash
pip install -e ".[ml]"      # AI model: Apple Silicon, NVIDIA or CPU
pip install -e ".[faces]"   # optional, better face detection (insightface)
pip install -e ".[gpu]"     # NVIDIA only: ml + CUDA face detection
pip install -e ".[dev]"     # + pytest, httpx
```

Run `photocure doctor` any time to see what is installed. Set `PCUR_BACKEND=stub` to
force the placeholder, for example in tests.

## Quick start

```bash
alias pc=".venv/bin/photocure"

# 0. Check the AI model, GPU and face detector are ready
pc doctor --db photocure.db

# 1. Index the library (idempotent; re-scans skip known paths in < 1 s for 10k files)
pc scan /media/safiyu/Photos --db photocure.db   # add -w N to set scan threads, 0 means auto

# 2. Quality + faces + CLIP (resumable; rows analyzed by the placeholder are redone once the AI model is installed)
pc infer --db photocure.db --device auto -b 32

# 3. Burst grouping + best-shot selection
pc cluster --db photocure.db

# 4. Dry-run routing plan (nothing is touched), persisted + HTML audit report + cost diff
pc plan --db photocure.db -o ./Organized --operation move --report plan.html

# 5. Review plan.html, then execute exactly that plan (records a transaction)
pc apply --db photocure.db --plan-id PLAN_... -y

# 6. Regret it? Full byte-for-byte rollback:
pc undo --db photocure.db

# Other
pc learn --db photocure.db              # re-sort using the categories you set by hand
pc transactions --db photocure.db       # ledger overview
pc verify-txn --db photocure.db         # re-hash committed targets, report drift
pc repair --db photocure.db             # fix images.path from the ledger
pc trash-purge --db photocure.db -o ./Organized --older-than-days 30
pc similar /path/img.jpg -k 10              # nearest neighbours by CLIP embedding
pc events --gap-hours 2                     # group images into events
pc people                                   # cluster faces into people
pc export-onnx -o ./onnx                    # export models
pc stats --db photocure.db          # state-store summary
pc serve --db photocure.db -p 8080  # web review dashboard
pc version
```

### CLI reference

| Command | Key options |
|---|---|
| `scan <target>` | `--db`, `--workers/-w` (0 or omitted means auto), `--quiet/-q`, `--prune-missing`, `--watch` |
| `devices` | `--db`; lists every GPU and the CPU with availability, the one in use, the worker counts, and install steps for hardware that is present but not usable |
| `doctor` | `--db`; checks the AI model, device, face detector, cached weights and placeholder rows, and prints the fix commands |
| `infer` | `--device` (auto\|cuda\|rocm\|xpu\|mps\|directml\|integrated\|cpu, default: settings file, then `PCUR_DEVICE_DEVICE`), `--device-index`, `--decode-workers`, `--threads` (0 or omitted means auto), `--batch-size/-b` (default from config), `--limit/-n`, `--paths`, `--retry-failed`, `--reanalyze/--no-reanalyze` (default: redo placeholder results whenever the AI model is installed) |
| `learn` | `--rebuild/--no-rebuild`; learns from photos you moved by hand and re-scores the rest |
| `cluster` | `--window-seconds`, `--similarity-threshold` (defaults from `PCUR_BURST_*`); results are reused by `plan`/`apply` |
| `plan` | `--output-dir/-o`, `--mode` (category_first\|in_place), `--operation` (move\|copy\|hardlink\|symlink), `--report/-r`, `--plan-id`, `--exclude-duplicates`, `--recluster`, `--window-seconds`, `--similarity-threshold` |
| `apply` | `--plan-id` (apply that persisted plan), `--operation` (explicit override), `--only-category` (repeatable), `--trash`, `--force`, `--yes/-y`; without `--plan-id` a fresh plan is built from `--output-dir/-o`, `--mode`, `--exclude-duplicates` and the thresholds |
| `undo` | `--transaction-id/-t` (default: latest transaction that still has committed rows), `--no-verify` |
| `verify-txn` | `--transaction-id/-t`; exit code 2 when drift or missing targets are found |
| `repair` | fixes `images.path` from the ledger |
| `trash-purge` | `--output-dir/-o`, `--older-than-days` (default `PCUR_TRASH_RETENTION_DAYS`, 30), `--yes/-y` |
| `transactions` | list ledger transactions |
| `similar <path>` | `-k` |
| `events` | `--gap-hours`, `--gps-km` |
| `people` | cluster face embeddings |
| `export-onnx` | `-o DIR` |
| `stats` | `--db` |
| `serve` | `--host`, `--port/-p`, `--reload` |

Plan semantics: `apply --plan-id` loads the stored actions, so duplicates excluded at
`plan` time stay excluded. In `in_place` mode `move` is always turned into `hardlink`,
also when `--operation move` is passed. Trash staging only affects junk buckets with
the `move` operation: files land in `<root>/.trash/<txn>/<original layout>`, stay fully
undoable, and `trash-purge` deletes them for good (ledger rows become `purged`).

## Web dashboard

`pc serve --db photocure.db` opens the review dashboard on http://127.0.0.1:8080.

- **Import:** a three step flow (scan, analyze, group bursts) with a folder picker, so you never paste a path. A strip at the top shows your library by category and filters when clicked.
- **Browse and review:** Library, Bursts, Highlights, Events, People, Needs a look, Junk review. Click a photo for a large viewer with arrow key navigation. Tick photos (Shift selects a range) to move or lock many at once.
- **Apply and undo:** build a plan, preview collisions, free space and per category counts, then apply. Undo and verify live on the same page.
- **Shortcuts:** press `?` in the dashboard. Light and dark themes follow your system and can be switched in the sidebar.

The folder picker lists folders on the machine running the server. Set `PCUR_BROWSE_ROOTS` (comma separated) to confine it to specific folders, and `PCUR_DEFAULT_OUTPUT` to prefill the destination.

## Docker

```bash
PHOTOS_DIR=/path/to/photos OUTPUT_DIR=/path/to/organized docker compose up --build
docker compose logs -f      # watch the first start, then open http://localhost:8080
```

One image works on every machine. On the first start the container runs the same setup as
`photocure setup`: it detects the hardware from the devices it can see (NVIDIA device
files, `/dev/kfd` for AMD, an Intel render node, otherwise CPU), installs the matching
PyTorch build into the data volume, and downloads the model weights. That is a few GB and
the web page appears when it finishes. Later starts reuse the install and are quick.

- **Volumes:** your photos appear at `/photos`, organized output goes to `/output`, and the database, thumbnails, installed libraries and model weights live in the `curator-data` volume. The picker can only see `/photos` and `/output`.
- **GPU access:** the container needs the device passed through. Add one more compose file, nothing else changes:

  | Hardware | Command | Status |
  |---|---|---|
  | CPU | `docker compose up --build` | works without any override |
  | NVIDIA | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build` | needs the NVIDIA Container Toolkit. Set `PCUR_TORCH_INDEX` for an older driver |
  | AMD | `docker compose -f docker-compose.yml -f docker-compose.rocm.yml up --build` | **untested**, passes `/dev/kfd` and `/dev/dri`, set `VIDEO_GID` and `RENDER_GID` from `getent group video render` |
  | Intel | `docker compose -f docker-compose.yml -f docker-compose.intel.yml up --build` | **untested**, passes `/dev/dri`, set `RENDER_GID`, and set `APT_PACKAGES` to install the Intel compute runtime |

- **Settings:** `PCUR_ACCEL` forces a hardware type, `PCUR_AUTO_SETUP=0` turns the install at start off for offline or pre-built images, and `PCUR_TORCH_INDEX` picks a different wheel index.
- **Changing hardware:** if you move the same volume to a different GPU, run once with `PCUR_ACCEL` set, or delete the volume. Setup replaces a PyTorch build that does not match the detected hardware.
- **Offline or pre-built image:** build with `--build-arg EXTRAS=ml` (and `TORCH_INDEX` for a GPU) and start with `PCUR_AUTO_SETUP=0`.
- **Read-only library:** change `:rw` to `:ro` on the photos volume in `docker-compose.yml` if you only scan and copy. Moving originals needs write access.
- **Permissions:** files are written as `PUID:PGID` (default 1000). Set them to your host user if moves fail.
- **Faces:** the InsightFace package is tried and kept optional. If it cannot build in the slim image, setup records that and uses the basic OpenCV detector instead.
- **Untested:** none of the container paths were run against a Docker daemon, and the AMD and Intel ones were not run on real hardware. The detection and install logic is covered by tests and was run for real outside Docker.
- **Security:** the page embeds a per launch token, so anyone who can reach the port can change your files. The compose file publishes the port on 127.0.0.1 only. Keep it that way, or put an authenticating proxy in front.

## Hardware support

The AI model runs on whatever you have. The default `auto` picks the best available device in
this order: NVIDIA, AMD, Intel, Apple, DirectML, then CPU. Run `photocure devices` to see
every device, why an unavailable one is unavailable, and which one is in use.

| Hardware | Install torch | Device value |
|---|---|---|
| NVIDIA (CUDA) | `pip install torch --index-url https://download.pytorch.org/whl/cu124` (change cu124 to match your driver) | `cuda` |
| AMD (ROCm, Linux) | `pip install torch --index-url https://download.pytorch.org/whl/rocm6.2` | `rocm` |
| Intel Arc and integrated Iris Xe | `pip install torch --index-url https://download.pytorch.org/whl/xpu` | `xpu` |
| Apple Silicon (Metal) | `pip install torch` (the default wheel includes it) | `mps` |
| Windows, any DirectX 12 GPU | `pip install torch-directml` | `directml` |
| CPU only | `pip install torch` | `cpu` |

After installing torch for your hardware, run `pip install "photocure[ml]"` and then
`photocure doctor`. The device `integrated` means "an integrated GPU": Intel first, then Apple. It never picks a
discrete NVIDIA or AMD card, and DirectML must be chosen by name. If the device you ask for is
not available the app falls back to the next best one and prints why when you run `infer`.
The `gpu` extra in `pyproject.toml` is NVIDIA only. `device_index` chooses between several GPUs.

Set the device on the command line (`--device`), in the dashboard settings, or with
`PCUR_DEVICE_DEVICE`. Precedence: command line flag, then the settings file, then the
environment, then defaults. The settings file sits next to the database and is named
`<database name>.settings.json`, for example `photocure.settings.json`.

## Workers and threads

| Setting | What it does | Auto |
|---|---|---|
| Scan threads (`--workers`, `PCUR_SCAN_WORKERS`) | read and hash files while indexing | `min(32, cpu_count * 2)` |
| Decode threads (`--decode-workers`, `PCUR_DEVICE_DECODE_WORKERS`) | decode and resize photos ahead of the model | `min(8, cpu_count)` |
| Torch threads (`--threads`, `PCUR_DEVICE_TORCH_THREADS`) | CPU threads inside the model | `min(cpu_count, 8)` on CPU, otherwise 2 |

Every setting uses 0 to mean auto and accepts values up to 128.

- **Scanning** waits on the disk more than the processor, so more threads than cores helps on an
  SSD. On a spinning disk or a network share, lower it (try 4) because parallel reads fight over
  the head or the link.
- **Decoding** is the usual bottleneck on a fast GPU. Each decode thread holds one full size
  photo in memory while it works, which can be 100 MB or more for large RAW and 48 MP files, so
  lower it on a machine with little RAM.
- **Torch threads** only matter on the CPU. Using every core finishes sooner but makes a laptop
  hot and loud, so lower it if the fan or temperature matters.

## Configuration

All settings are Pydantic models overridable via environment variables
(`.env` supported). Prefixes: `PCUR_`, `PCUR_DEVICE_`, `PCUR_QUALITY_`,
`PCUR_BURST_`, `PCUR_TAXONOMY_`.

| Variable | Default | Meaning |
|---|---|---|
| `PCUR_DEVICE_DEVICE` | `auto` | `auto` / `cuda` / `rocm` / `xpu` / `mps` / `directml` / `integrated` / `cpu`, see Hardware support. An unknown value falls back to `auto` with a warning |
| `PCUR_DEVICE_DEVICE_INDEX` | `0` | which GPU to use when there is more than one |
| `PCUR_DEVICE_DECODE_WORKERS` | `0` | threads decoding photos ahead of the model, 0 means auto |
| `PCUR_DEVICE_TORCH_THREADS` | `0` | CPU threads for torch, 0 means auto |
| `PCUR_SCAN_WORKERS` | `0` | scan threads, 0 means auto |
| `PCUR_DEVICE_CUDA_MEMORY_FRACTION` | `0.75` | PyTorch VRAM cap (headroom for Xorg/Jellyfin) |
| `PCUR_DEVICE_FACE_BATCH_SIZE` | `32` | face batch |
| `PCUR_DEVICE_CLIP_BATCH_SIZE` | `64` | CLIP batch |
| `PCUR_DEVICE_INFER_CHUNK_SIZE` | `512` | decoded frames held in RAM at once (streaming bound) |
| `PCUR_BURST_WINDOW_SECONDS` | `30` | burst temporal window |
| `PCUR_BURST_SIMILARITY_THRESHOLD` | `0.88` | burst cosine threshold |
| `PCUR_BURST_MIN_GROUP_SIZE` | `2` | minimum burst size |
| `PCUR_TAXONOMY_MIN_CONFIDENCE` | `0.40` | below → `Uncertain_Review/` |
| `PCUR_TAXONOMY_MIN_MARGIN` | `0.08` | top-1 − top-2 below → `Uncertain_Review/` |
| `PCUR_TAXONOMY_FACE_CONFIDENCE_MIN` | `0.65` | face gate |
| `PCUR_TAXONOMY_MIN_CLASS_FLOOR` | `0.25` | top-1 below this is never routed, even when its group agrees |
| `PCUR_TAXONOMY_FEEDBACK_WEIGHT` | `0.6` | weight of what was learned from your corrections versus zero-shot |
| `PCUR_TAXONOMY_FEEDBACK_MIN_EXAMPLES` | `2` | corrected photos needed in a category before it is learned |
| `PCUR_BACKEND` | - | `stub` / `clip` / `insightface` (test/CI override) |
| `PCUR_TRASH_DIR_NAME` | `.trash` | staging folder under the organize root |
| `PCUR_TRASH_RETENTION_DAYS` | `30` | default for `trash-purge` |

## Architecture

```
photocure/
├── config.py              # Pydantic settings (PCUR_* env overrides)
├── cli.py                 # Typer CLI: scan/infer/cluster/plan/apply/undo/stats/serve
├── db/                    # SQLite (WAL) models, session, batch repository
├── scanner/               # file_scanner (threads, resumable), exif_reader (date cascade), xmp, phash, watcher, hasher (xxh64/sha256)
├── quality/               # blur (Laplacian), brightness (HSV/luminance), image loader
├── inference/             # backends (capability detect), face_engine, semantic_engine (OpenCLIP), pipeline (Phase A→sweep→B)
├── clustering/            # burst_detector (temporal+cosine, union-find), best_shot_selector
├── rules/                 # taxonomy (prompts/CategoryType), decision_matrix, organizer (path graph, HTML report)
├── engine/                # action_engine (move/copy/hardlink/symlink + ledger), rollback (atomic undo)
└── ui/                    # FastAPI review dashboard + static/ (vanilla JS)
```

### State store

SQLite in WAL mode. `images` (metadata, hashes, `taken_at` cascade source,
routing, `locked`, `manual_bucket`, `phash`, `rating`, `event_id`) + child tables `quality`, `faces`, `semantics` (float16-quantized
512-dim embeddings, JSON int16 bit patterns), `bursts`, `person_clusters`, `event_groups`, `plans` + `plan_actions` (persisted, content-hashed plans), `action_ledger`
(transaction journal: `pending → committed → rolled_back | failed | purged`; each
row records the directory it created so undo only prunes what apply made).

The schema is versioned with `PRAGMA user_version`; `init_db` adds any columns missing
from older stores automatically.

## Troubleshooting

**Everything is Uncertain.** The AI model is probably not installed, so a placeholder
classifier ran. Run `photocure doctor`. If it reports "placeholder", install the model
with `pip install "photocure[ml]"` and run `photocure infer`, which redoes those
photos. If the model is installed and a lot is still Uncertain, move a few photos to the
right category in the dashboard and run `photocure learn`.

## Testing

```bash
.venv/bin/python -m pytest tests/ -q
```

- `test_scanner.py` / `test_hasher.py` - EXIF cascade, two-tier dedup, resumption
- `test_quality.py` - blur/brightness synthetic frames
- `test_inference.py` - stub determinism + real OpenCLIP path (auto-skipped if
  weights are not cached); EAR geometry; multi-chunk streaming (no image lost
  across chunk boundaries)
- `test_clustering.py` - burst windows, transitive chaining, best-shot scenario
- `test_rules.py` - decision matrix routing
- `test_ledger_recovery.py` / `test_plans.py` - crash recovery, partial failure,
  persisted plans and drift refusal, in-place safety, trash, verify, repair, schema migration
- `test_action_engine.py` - ledger lifecycle, collision rename, rollback hash
  verification, and the **rollback-invariance test** (N files → apply → undo →
  byte-for-byte directory equivalence)

## Safety guarantees

1. **`plan` is read-only** - it only writes the HTML report and the DB.
2. **Every mutation is journaled before it happens** (`PENDING` → mutate → `COMMITTED`).
3. **`undo` verifies SHA-256 before touching anything** and reverses in
   reverse-chronological order; it refuses to proceed on hash mismatch.
4. **No data loss on partial failure** - the ledger is write-ahead and durable per action.
   A crash mid-`apply` leaves committed rows plus at most one `PENDING` row; `undo`
   reconciles that row against the disk and rolls back the whole prefix.
5. **Duplicates are never deleted** - only routed to `Junk_Review/Duplicates/`.
