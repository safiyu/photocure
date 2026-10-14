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
import time
from dataclasses import dataclass, field
from datetime import timezone
from pathlib import Path
from collections.abc import Callable, Iterable

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


def _safe_sha256(path: Path) -> str:
    """Full-file SHA-256 that degrades to ``""`` on I/O failure.

    A single unreadable file must not abort a scan; the caller treats an empty
    result as "could not verify" rather than a crash.
    """
    try:
        return compute_sha256(path)
    except OSError as e:
        log.warning("sha256 failed for %s: %s", path, e)
        return ""


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
                       on_progress: Callable[[int, int, str], None] | None = None,
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
            # Tier-1 fingerprints already in the store. A new file only needs a
            # full-file SHA-256 if its quick-hash is in this set (or collides
            # with an earlier file in this same scan pass) -- see _resolve_dedup.
            known_qh = repo.known_quick_hashes()

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

                Resolves dedup (SHA-256 is computed *lazily*, only on a Tier-1
                quick-hash collision -- see :meth:`_resolve_dedup`) and assigns
                the row.
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
                self._resolve_dedup(rec, wr.path, repo, seen_qh, pending_dupes,
                                    known_qh, result)
                return rec

            batch: list[ImageRecord] = []
            # Same-pass originals keyed by Tier-1 quick-hash. Exact duplicates
            # always share the quick-hash (same size + head/tail), so this map
            # lets a later copy in the same scan pass find its original without
            # a DB round-trip -- and lets _resolve_dedup backfill the original's
            # (lazily-skipped) SHA-256 only when a real duplicate turns up.
            seen_qh: dict[str, ImageRecord] = {}
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

    # --------------------------------------------------------------- dedup
    def _resolve_dedup(self, rec: ImageRecord, path: Path, repo,
                       seen_qh: dict, pending_dupes: list,
                       known_qh: set, result: "ScanResult") -> None:
        """Lazy two-tier dedup for a single new record (mutates ``rec``).

        Tier-1 (cheap head/tail, always) has already run in the worker. Tier-2
        (the expensive full-file SHA-256) is read **only** when a quick-hash
        collision forces verification::

            same-pass quick-hash hit  -> hash both sides, dup if they match
            in-store quick-hash hit   -> hash this file, verify vs the store
            first sighting (no hit)   -> store NO hash (deferred to apply time)

        Net effect: a library of *unique* photos pays only the O(1) head/tail
        cost and never triggers a full-file read during scan; the expensive
        read happens solely for files sharing a fingerprint with something
        already known (i.e. real or near-duplicates).
        """
        # 1. Same-pass original (in the current/previous batch).
        orig = seen_qh.get(rec.quick_hash)
        if orig is not None:
            # A same-pass file shares this fingerprint. Confirm the bytes match
            # (a quick-hash collision can still be distinct content) before
            # calling it a duplicate. Backfill the original's deferred hash.
            if not orig.sha256:
                orig.sha256 = _safe_sha256(Path(orig.path))
            sha = _safe_sha256(path)
            rec.sha256 = sha
            if sha and sha == orig.sha256:
                rec.is_duplicate = True
                pending_dupes.append((rec, orig))
                result.duplicates += 1
            # Mismatch => genuinely different file; keep it, hash now stored.
            return

        # 2. In-store original from an earlier scan pass.
        if rec.quick_hash in known_qh:
            sha = _safe_sha256(path)
            rec.sha256 = sha
            if sha:
                match = self._find_db_original(sha, repo, rec)
                if match is not None:
                    repo.mark_duplicate(rec, match)
                    result.duplicates += 1
                    return
            seen_qh[rec.quick_hash] = rec
            return

        # 3. First sighting of this fingerprint: no full-file read (lazy
        #    Tier-2). The SHA-256 stays deferred until a collision forces
        #    verification, or the action engine computes it at apply time.
        seen_qh[rec.quick_hash] = rec

    @staticmethod
    def _find_db_original(sha: str, repo, rec: ImageRecord) -> ImageRecord | None:
        """Locate a definitive SHA-256 match already persisted (any path)."""
        # Fast path: an earlier pass stored this exact sha256.
        hit = repo.find_by_sha256(sha, exclude_id=rec.id)
        if hit is not None:
            return hit
        # Otherwise confirm against the quick-hash candidates. A prior *lazy*
        # run may have left the candidate's sha256 empty, so backfill it.
        for cand in repo.find_by_quick_hash(rec.quick_hash, exclude_id=rec.id):
            if not cand.sha256:
                cand.sha256 = _safe_sha256(Path(cand.path))
                repo.s.flush()
            if cand.sha256 == sha:
                return cand
        return None


@dataclass
class _WorkerResult:
    path: Path
    size: int = 0
    quick_hash: str = ""
    exif: ExifInfo | None = None
    error: str = ""


def _os_walk(root: Path, skip_dirs: set[str]):
    """os.walk wrapper kept local so skip logic is testable."""
    import os
    return os.walk(root, followlinks=False)


def scan_directory(root: Path | str, config: CuratorConfig | None = None,
                   db_path: Path | str | None = None,
                   show_progress: bool = True) -> ScanResult:
    cfg = config or CuratorConfig.load(db_path)
    return FileScanner(cfg).scan_directory(root, db_path=db_path,
                                           show_progress=show_progress)
