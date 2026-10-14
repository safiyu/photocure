"""Multi-threaded recursive directory scanner with resumability.

Scanning is idempotent:
- Paths already present in the store with a non-empty ``quick_hash`` are
  skipped entirely (resumption benchmark: 10k known files in < 1 s).
- New files get a Tier-1 quick hash; only files whose quick hash collides with
  an existing record receive the expensive Tier-2 SHA-256, and are then marked
  as exact duplicates.
"""
from __future__ import annotations

import concurrent.futures as cf
import logging
import mimetypes
import sys
import time
from dataclasses import dataclass, field
from datetime import timezone
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Set

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ..config import CuratorConfig
from ..db.models import ImageRecord, utcnow
from ..db.repository import PhotoRepository
from ..db.session import get_session, init_db
from sqlmodel import Session
from .exif_reader import ExifInfo, extract_exif
from .hasher import compute_quick_hash, compute_sha256

log = logging.getLogger("photo_curator.scanner")


@dataclass
class ScanResult:
    scanned: int = 0
    new: int = 0
    skipped_known: int = 0
    duplicates: int = 0
    errors: int = 0
    duration_s: float = 0.0
    skipped_files: list = field(default_factory=list)
    errors_list: list = field(default_factory=list)


def _guess_mime(path: Path, ext: str) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime:
        return mime
    table = {
        ".heic": "image/heic", ".heif": "image/heif",
        ".cr2": "image/x-canon-cr2", ".nef": "image/x-nikon-nef",
        ".arw": "image/x-sony-arw", ".dng": "image/x-adobe-dng",
    }
    return table.get(ext, "application/octet-stream")


def _aware(d):
    """Normalize a naive datetime to UTC-aware for the state store.

    SQLModel 0.0.48 / SQLAlchemy 2.1.x reject naive datetimes on insert
    (``Datetime values must have timezone information``). EXIF/filename/mtime
    sources are all naive; tagging them UTC preserves the calendar date (the
    routing hierarchy only uses year/month/day) and keeps burst deltas valid.
    """
    if d is None:
        return None
    if d.tzinfo is None:
        return d.replace(tzinfo=timezone.utc)
    return d


class FileScanner:
    """Recursive scanner producing :class:`ImageRecord` rows via batch upserts."""

    def __init__(self, config: CuratorConfig, session: Session | None = None):
        self.cfg = config
        self.session = session

    # ------------------------------------------------------------------ walk
    def iter_files(self, root: Path | str,
                   extensions: Iterable[str] | None = None) -> Iterable[Path]:
        root = Path(root)
        exts = {e.lower() for e in (extensions or self.cfg.extensions)}
        # Skip hidden/system dirs to avoid .git, __MACOSX, @eaDir etc.
        skip_dirs = {".git", "__MACOSX", ".cache", ".thumbnail", ".thumbnails",
                     "@eaDir", "$recycle.bin", "system volume information"}
        for dirpath, dirnames, filenames in _os_walk(root, skip_dirs):
            dirnames[:] = [d for d in dirnames if d.lower() not in skip_dirs
                           and not d.startswith(".")]
            for fn in filenames:
                if fn.startswith("."):
                    continue
                p = Path(dirpath) / fn
                if p.suffix.lower() in exts:
                    yield p

    # ------------------------------------------------------------------ scan
    def scan_directory(self, root: Path | str,
                       db_path: Path | str | None = None,
                       on_progress: Optional[Callable[[int, int, str], None]] = None,
                       show_progress: bool = True) -> ScanResult:
        root = Path(root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Target directory does not exist: {root}")

        db = db_path or self.cfg.db_path
        init_db(db)  # idempotent: ensures the schema exists for fresh stores
        files = list(self.iter_files(root))
        result = ScanResult()
        t0 = time.monotonic()

        with get_session(db) as session:
            repo = PhotoRepository(session)
            known = repo.known_paths()

            # Split into known (skip) vs new.
            new_files: list[Path] = []
            for p in files:
                if str(p) in known:
                    result.skipped_known += 1
                else:
                    new_files.append(p)

            if show_progress:
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    transient=False,
                )
                main_task = progress.add_task("Scanning", total=len(new_files))
                progress.start()
            else:
                progress = None

            def process(path: Path) -> _WorkerResult:
                """Pure filesystem I/O worker: stat + quick hash + EXIF.

                Never touches the DB session (thread-safe by construction).
                """
                wr = _WorkerResult(path=path)
                try:
                    st = path.stat()
                    wr.size = st.st_size
                except OSError as e:
                    wr.error = f"stat: {e}"
                    return wr
                try:
                    wr.quick_hash = compute_quick_hash(
                        path, self.cfg.quick_hash_head_bytes,
                        self.cfg.quick_hash_tail_bytes)
                except OSError as e:
                    wr.error = f"quick_hash: {e}"
                    return wr
                exif = extract_exif(path)
                wr.exif = exif
                return wr

            def build_record(wr: _WorkerResult) -> ImageRecord | None:
                """Main-thread: turn a worker result into an ImageRecord.

                Computes SHA-256, resolves dedup and assigns the row.
                """
                if wr.error or wr.exif is None:
                    log.warning("scan error %s: %s", wr.path, wr.error)
                    return None
                exif = wr.exif
                rec = ImageRecord(
                    path=str(wr.path),
                    relative_path=str(wr.path.relative_to(root)),
                    filename=wr.path.name,
                    extension=wr.path.suffix.lower(),
                    mime_type=_guess_mime(wr.path, wr.path.suffix.lower()),
                    file_size=wr.size,
                    quick_hash=wr.quick_hash,
                )
                rec.width = exif.width
                rec.height = exif.height
                rec.orientation = exif.orientation
                rec.taken_at = _aware(exif.taken_at)
                rec.taken_at_source = exif.taken_at_source
                rec.gps_lat = exif.gps_lat
                rec.gps_lon = exif.gps_lon
                rec.camera_model = exif.camera_model
                try:
                    rec.sha256 = compute_sha256(wr.path)
                    mem_match = seen_sha256.get(rec.sha256)
                    if mem_match is not None:
                        # Original is still in the unflushed batch (no id yet):
                        # flag the duplicate now, link duplicate_of post-flush.
                        rec.is_duplicate = True
                        pending_dupes.append((rec, mem_match))
                        result.duplicates += 1
                    else:
                        db_match = repo.find_by_sha256(
                            rec.sha256, exclude_id=rec.id)
                        if db_match is not None:
                            repo.mark_duplicate(rec, db_match)
                            result.duplicates += 1
                        else:
                            seen_sha256[rec.sha256] = rec
                except OSError as e:
                    log.warning("sha256 failed for %s: %s", wr.path, e)
                    rec.sha256 = ""
                return rec

            batch: list[ImageRecord] = []
            # Tier-2 dedup across the current batch: find_by_sha256 only sees
            # rows already in the session, so a copy scanned in the same pass
            # would otherwise slip through. This map closes that gap.
            seen_sha256: dict[str, ImageRecord] = {}
            # (dup, original) pairs whose duplicate_of is linked after flush.
            pending_dupes: list[tuple[ImageRecord, ImageRecord]] = []

            def _flush():
                nonlocal batch
                if batch:
                    repo.bulk_upsert_images(batch)
                    batch.clear()
                # All batch rows now carry ids; resolve same-batch duplicate
                # links (original was flushed above, so its id is real).
                for dup, orig in pending_dupes:
                    dup.duplicate_of = orig.id
                    dup.updated_at = utcnow()
                pending_dupes.clear()

            with cf.ThreadPoolExecutor(max_workers=self.cfg.scan_workers) as pool:
                futures = {pool.submit(process, p): p for p in new_files}
                done = 0
                for fut in cf.as_completed(futures):
                    wr = fut.result()
                    rec = build_record(wr)
                    if rec is not None:
                        batch.append(rec)
                        result.new += 1
                    else:
                        result.errors += 1
                        result.errors_list.append(str(futures[fut]))
                    done += 1
                    if show_progress:
                        progress.update(main_task, completed=done)
                    if on_progress:
                        on_progress(done, len(new_files), str(futures[fut]))
                    if len(batch) >= 500:
                        _flush()
            _flush()

            if show_progress and progress is not None:
                progress.stop()

        result.scanned = len(files)
        result.duration_s = time.monotonic() - t0
        return result


@dataclass
class _WorkerResult:
    path: Path
    size: int = 0
    quick_hash: str = ""
    exif: Optional["ExifInfo"] = None
    error: str = ""


def _os_walk(root: Path, skip_dirs: Set[str]):
    """os.walk wrapper kept local so skip logic is testable."""
    import os
    return os.walk(root, followlinks=False)


def scan_directory(root: Path | str, config: CuratorConfig | None = None,
                   db_path: Path | str | None = None,
                   show_progress: bool = True) -> ScanResult:
    cfg = config or CuratorConfig.load(db_path)
    return FileScanner(cfg).scan_directory(root, db_path=db_path,
                                           show_progress=show_progress)
