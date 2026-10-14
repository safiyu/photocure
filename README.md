# Photo Curator

An intelligent, **offline**, GPU-accelerated local photo curation engine. It scans
large heterogeneous photo libraries (JPEG, PNG, HEIC, WebP, RAW), scores quality,
detects faces, runs zero-shot semantic classification with OpenCLIP, groups
burst shots, and routes everything into a category-first folder tree — all
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
| **1. Scan** | Multi-threaded recursive walk. Two-tier dedup (xxHash64 head/tail → SHA-256 only on collision). 4-tier date cascade: EXIF `DateTimeOriginal` → EXIF `CreateDate` → strict filename regex → `mtime` → `Unknown_Date/` (never 1970). |
| **2. Infer** | Sequential GPU phases — never both models in VRAM at once. Phase A: faces (InsightFace/SCRFD, Haar fallback, stub). Memory sweep. Phase B: OpenCLIP ViT-B-16 zero-shot classification + 512-dim L2-normalized embeddings. Quality metrics (Laplacian blur, luminance, clipped-tail ratios) on the same decoded frame. |
| **3. Cluster** | Burst detection: `Δt ≤ 30 s AND cosine(e_i, e_j) ≥ 0.88`, graph-constrained temporal sliding window (`O(N log N)` sort + in-window pairwise, union-find transitivity). Best-shot score: `0.40·blur + 0.30·face_sharp + 0.20·eye_open + 0.10·log(resolution)`. |
| **4. Plan** | Pure decision matrix over DB state → destination path graph + HTML/CLI audit report. **Zero filesystem mutation.** |
| **5. Apply** | `move` / `copy` / `hardlink` / `symlink` with collision renaming (`_dup<sha6>`). Each op: ledger row `PENDING` → FS mutation → `COMMITTED`. |
| **6. Undo** | Reverse-chronological rollback of the ledger with pre-flight SHA-256 verification (refuses to proceed on mismatch). |

**VRAM budget (RTX 4060 8 GiB shared with the desktop):** faces ≈ 1.2 GiB,
CLIP ≈ 2.5 GiB, sequential phases → peak < 4 GiB, with
`torch.cuda.set_per_process_memory_fraction(0.75)` capping PyTorch's share.

### Decision matrix

```
SHA-256 duplicate?          → Junk_Review/Duplicates/
face detected (conf > 0.65)? → People/YYYY/MM/DD/
top CLIP = screenshot/doc/  → Junk_Review/Screenshots|Documents/
meme/graphic
top CLIP = food             → Non-Human/Food/YYYY/MM/DD/
top CLIP = pet/animal       → Non-Human/Animals/YYYY/MM/DD/
top CLIP = landscape/etc.   → Non-Human/Scenery/YYYY/MM/DD/
low confidence / small      → Uncertain_Review/
margin
in burst?                   → best shot kept, alternates → Bursts/Burst_<ID>/alternates/
```

### Organization layouts

- **Option 1 — `category_first` (default):** `People/2024-05-12/…`,
  `Non-Human/{Food,Animals,Scenery}/…`, `Junk_Review/…`, `Bursts/…`.
- **Option 2 — `in_place`:** originals stay put; hardlinks/symlinks are created
  in the target tree (no extra storage for hardlinks).

---

## Install

```bash
cd /home/safiyu/Projects/photo-curator
python3 -m venv .venv
.venv/bin/pip install -e .            # core (CPU + stub backends)
.venv/bin/pip install -e ".[gpu]"     # + torch, OpenCLIP, InsightFace
.venv/bin/pip install -e ".[dev]"     # + pytest, httpx
```

The core install is fully functional without a GPU: face/CLIP fall back to
deterministic **stub** backends so the whole pipeline (dedup, quality, routing,
plan, apply, undo) is exercisable end-to-end. Set `PCUR_BACKEND=stub` to force
stubs even when the real models are installed.

## Quick start

```bash
alias pc=".venv/bin/photo-curator"

# 1. Index the library (idempotent; re-scans skip known paths in < 1 s for 10k files)
pc scan /media/safiyu/Photos --db photo_curator.db -w 8

# 2. Quality + faces + CLIP (resumable; already-inferred rows are skipped)
pc infer --db photo_curator.db --device auto -b 32

# 3. Burst grouping + best-shot selection
pc cluster --db photo_curator.db

# 4. Dry-run routing plan (nothing is touched) + HTML audit report
pc plan --db photo_curator.db -o ./Organized --operation move --report plan.html

# 5. Review plan.html, then execute (records a transaction)
pc apply --db photo_curator.db --operation move -y

# 6. Regret it? Full byte-for-byte rollback:
pc undo --db photo_curator.db

# Other
pc stats --db photo_curator.db          # state-store summary
pc serve --db photo_curator.db -p 8080  # web review dashboard
pc version
```

### CLI reference

| Command | Key options |
|---|---|
| `scan <target>` | `--db`, `--workers/-w` (default 8), `--quiet/-q` |
| `infer` | `--device` (cuda\|cpu\|auto), `--batch-size/-b`, `--limit/-n`, `--paths` |
| `cluster` | `--window-seconds`, `--similarity-threshold` |
| `plan` | `--output-dir/-o`, `--mode` (category_first\|in_place), `--operation` (move\|copy\|hardlink\|symlink), `--report/-r`, `--plan-id`, `--exclude-duplicates` |
| `apply` | `--plan-id`, `--operation`, `--output-dir/-o`, `--mode`, `--yes/-y` |
| `undo` | `--transaction-id/-t` (default: latest), `--no-verify` |
| `stats` | `--db` |
| `serve` | `--host`, `--port/-p`, `--reload` |

## Configuration

All settings are Pydantic models overridable via environment variables
(`.env` supported). Prefixes: `PCUR_`, `PCUR_DEVICE_`, `PCUR_QUALITY_`,
`PCUR_BURST_`, `PCUR_TAXONOMY_`.

| Variable | Default | Meaning |
|---|---|---|
| `PCUR_DEVICE_DEVICE` | `cuda` | `cuda` / `cpu` / `auto` |
| `PCUR_DEVICE_CUDA_MEMORY_FRACTION` | `0.75` | PyTorch VRAM cap (headroom for Xorg/Jellyfin) |
| `PCUR_DEVICE_FACE_BATCH_SIZE` | `32` | face batch |
| `PCUR_DEVICE_CLIP_BATCH_SIZE` | `64` | CLIP batch |
| `PCUR_BURST_WINDOW_SECONDS` | `30` | burst temporal window |
| `PCUR_BURST_SIMILARITY_THRESHOLD` | `0.88` | burst cosine threshold |
| `PCUR_BURST_MIN_GROUP_SIZE` | `2` | minimum burst size |
| `PCUR_TAXONOMY_MIN_CONFIDENCE` | `0.40` | below → `Uncertain_Review/` |
| `PCUR_TAXONOMY_MIN_MARGIN` | `0.08` | top-1 − top-2 below → `Uncertain_Review/` |
| `PCUR_TAXONOMY_FACE_CONFIDENCE_MIN` | `0.65` | face gate |
| `PCUR_BACKEND` | — | `stub` / `clip` / `insightface` (test/CI override) |

## Architecture

```
photo_curator/
├── config.py              # Pydantic settings (PCUR_* env overrides)
├── cli.py                 # Typer CLI: scan/infer/cluster/plan/apply/undo/stats/serve
├── db/                    # SQLite (WAL) models, session, batch repository
├── scanner/               # file_scanner (threads, resumable), exif_reader (4-tier cascade), hasher (xxh64/sha256)
├── quality/               # blur (Laplacian), brightness (HSV/luminance), image loader
├── inference/             # backends (capability detect), face_engine, semantic_engine (OpenCLIP), pipeline (Phase A→sweep→B)
├── clustering/            # burst_detector (temporal+cosine, union-find), best_shot_selector
├── rules/                 # taxonomy (prompts/CategoryType), decision_matrix, organizer (path graph, HTML report)
├── engine/                # action_engine (move/copy/hardlink/symlink + ledger), rollback (atomic undo)
└── ui/                    # FastAPI review dashboard + static/ (vanilla JS)
```

### State store

SQLite in WAL mode. `images` (metadata, hashes, `taken_at` cascade source,
routing) + child tables `quality`, `faces`, `semantics` (float16-quantized
512-dim embeddings, JSON int16 bit patterns), `bursts`, `action_ledger`
(transaction journal: `pending → committed → rolled_back`).

## Testing

```bash
.venv/bin/python -m pytest tests/ -q
```

- `test_scanner.py` / `test_hasher.py` — EXIF cascade, two-tier dedup, resumption
- `test_quality.py` — blur/brightness synthetic frames
- `test_inference.py` — stub determinism + real OpenCLIP path (auto-skipped if
  weights are not cached); EAR geometry
- `test_clustering.py` — burst windows, transitive chaining, best-shot scenario
- `test_rules.py` — decision matrix routing
- `test_action_engine.py` — ledger lifecycle, collision rename, rollback hash
  verification, and the **rollback-invariance test** (N files → apply → undo →
  byte-for-byte directory equivalence)

## Safety guarantees

1. **`plan` is read-only** — it only writes the HTML report and the DB.
2. **Every mutation is journaled before it happens** (`PENDING` → mutate → `COMMITTED`).
3. **`undo` verifies SHA-256 before touching anything** and reverses in
   reverse-chronological order; it refuses to proceed on hash mismatch.
4. **No data loss on partial failure** — a crash mid-`apply` leaves a
   transaction with uncommitted entries; `undo` rolls back the committed prefix.
5. **Duplicates are never deleted** — only routed to `Junk_Review/Duplicates/`.
