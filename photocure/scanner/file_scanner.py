"""Multi-threaded recursive directory scanner with incremental rescans.

Scanning is idempotent and incremental:
- Known paths whose size and mtime are unchanged are skipped (resumption
  benchmark: 10k known files in < 1 s). Modified files are re-processed in
  place and their derived state (hashes, children, duplicate links) is reset.
- New files get a Tier-1 quick hash; only files whose quick hash collides with
  an existing record receive the expensive Tier-2 SHA-256, and are then marked
  as exact duplicates. Missing originals are never used as a duplicate target.
- Every new file also gets a 64-bit perceptual hash; near-duplicates (Hamming
  distance <= 6, different bytes) point the lower-resolution record at the
  higher-resolution one through ``near_dup_of``.
- With ``prune_missing`` vanished paths under the root are marked
  ``scan_status="missing"``; they are restored if the file reappears.
"""
from __future__ import annotations

import concurrent.futures as cf
import logging
import mimetypes
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import timezone
from pathlib import Path

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sqlmodel import Session
from sqlmodel import select as sselect

from ..config import CuratorConfig
from ..db.models import ImageRecord, utcnow
from ..db.repository import PhotoRepository
from ..db.session import get_session, init_db
from .exif_reader import ExifInfo, extract_exif, taken_at_utc
from .hasher import compute_quick_hash, compute_sha256
from .phash import DEFAULT_THRESHOLD, NearDupIndex, compute_phash

log = logging.getLogger("photocure.scanner")

_IN_CHUNK = 400


@dataclass
class ScanResult:
    scanned: int = 0
    new: int = 0
    modified: int = 0
    skipped_known: int = 0
    skipped_library: int = 0
    duplicates: int = 0
    near_duplicates: int = 0
    geocoded: int = 0
    missing: int = 0
    errors: int = 0
    duration_s: float = 0.0
    # Files not indexed because their content is not a decodable image.
    skipped_files: list = field(default_factory=list)
    # Files too fresh to trust (still being written); retried on a later scan.
    deferred_files: list = field(default_factory=list)
    errors_list: list = field(default_factory=list)
    # Directories the walk could not read (permission or I/O errors).
    walk_errors: list = field(default_factory=list)


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


def _as_utc_ts(d) -> float:
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()


@dataclass
class _WorkerResult:
    path: Path
    size: int = 0
    mtime: float = 0.0
    quick_hash: str = ""
    phash: str | None = None
    exif: ExifInfo | None = None
    error: str = ""


# Returned by build_record for a file already in the library or trash: not an error, not new.
_SKIPPED = object()


@dataclass
class _Known:
    id: int
    size: int
    stamp: float | None
    status: str
    created: float | None = None
    stage: str = "new"


class FileScanner:
    """Recursive scanner producing :class:`ImageRecord` rows via batch upserts."""

    def __init__(self, config: CuratorConfig, session: Session | None = None):
        self.cfg = config
        self.session = session

    # ------------------------------------------------------------------ walk
    def iter_files(self, root: Path | str,
                   extensions: Iterable[str] | None = None,
                   errors: list | None = None,
                   skipped: list | None = None,
                   cancel_check: Callable[[], bool] | None = None) -> Iterable[Path]:
        root = Path(root)
        exts = {e.lower() for e in (extensions or self.cfg.extensions)}
        # Skip hidden/system/build dirs to avoid .git, __MACOSX, build, dist, _gzip etc.
        skip_dirs = {".git", "__MACOSX", ".cache", ".thumbnail", ".thumbnails",
                     "@eaDir", "$recycle.bin", "system volume information",
                     "build", "dist", "node_modules", "_gzip", "ui-gzip",
                     ".venv", "venv", "__pycache__", ".tox", "target"}
        root_resolved = None
        lib_root = Path(self.cfg.library_root).expanduser().resolve() if self.cfg.library_root else None
        for dirpath, dirnames, filenames in _os_walk(root, skip_dirs, errors):
            if cancel_check and cancel_check():
                raise InterruptedError("Scan cancelled by user")
            dirnames[:] = [d for d in dirnames if d.lower() not in skip_dirs
                           and not d.startswith(".") and not _is_library_bundle(d)]
            if lib_root:
                dirnames[:] = [d for d in dirnames if Path(dirpath, d).resolve() != lib_root
                               and not Path(dirpath, d).resolve().is_relative_to(lib_root)]
            for fn in filenames:
                if cancel_check and cancel_check():
                    raise InterruptedError("Scan cancelled by user")
                if fn.startswith("."):
                    continue
                p = Path(dirpath) / fn
                if p.suffix.lower() not in exts:
                    continue
                if os.path.islink(p):
                    # Never follow a symlink that leaves the scan root.
                    root_resolved = root_resolved or root.resolve()
                    try:
                        p.resolve().relative_to(root_resolved)
                    except (ValueError, OSError):
                        if skipped is not None:
                            skipped.append(str(p))
                        continue
                yield p

    # ------------------------------------------------------------------ scan
    def scan_directory(self, root: Path | str,
                       db_path: Path | str | None = None,
                       on_progress: Callable[[int, int, str], None] | None = None,
                       show_progress: bool = True,
                       prune_missing: bool = False,
                       compute_phashes: bool = True,
                       settle_s: float = 0.0,
                       near_dup_threshold: int = DEFAULT_THRESHOLD,
                       cancel_check: Callable[[], bool] | None = None) -> ScanResult:
        root = Path(root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Target directory does not exist: {root}")
        if self.cfg.library_root:
            lib_root = Path(self.cfg.library_root).expanduser().resolve()
            if root == lib_root or root.is_relative_to(lib_root):
                raise ValueError(f"Cannot scan inside library root: {root}")
        result = ScanResult()
        files = list(self.iter_files(root, errors=result.walk_errors,
                                     skipped=result.skipped_files,
                                     cancel_check=cancel_check))
        return self._run(root, files, result, db_path, on_progress, show_progress,
                         prune_missing, compute_phashes, settle_s, near_dup_threshold,
                         cancel_check=cancel_check)

    def scan_paths(self, root: Path | str, paths: Iterable[Path | str],
                   db_path: Path | str | None = None,
                   show_progress: bool = False,
                   compute_phashes: bool = True,
                   settle_s: float = 0.0,
                   near_dup_threshold: int = DEFAULT_THRESHOLD,
                   cancel_check: Callable[[], bool] | None = None) -> ScanResult:
        """Scan only the given files (used by watch mode); never prunes."""
        root = Path(root).expanduser().resolve()
        exts = {e.lower() for e in self.cfg.extensions}
        lib_root = Path(self.cfg.library_root).expanduser().resolve() if self.cfg.library_root else None
        files: list[Path] = []
        for p in paths:
            if cancel_check and cancel_check():
                raise InterruptedError("Scan cancelled by user")
            p = Path(p)
            try:
                p = p.resolve() if not p.is_symlink() else p.absolute()
                p.relative_to(root)
            except (ValueError, OSError):
                continue
            if lib_root is not None and (p == lib_root or lib_root in p.parents):
                continue  # the library is Photocure's own output, never a source
            rel = p.relative_to(root).parts
            if (any(part.startswith(".") or _is_library_bundle(part) for part in rel)
                    or p.suffix.lower() not in exts):
                continue
            if p.is_file():
                files.append(p)
        return self._run(root, sorted(set(files)), ScanResult(), db_path, None,
                         show_progress, False, compute_phashes, settle_s,
                         near_dup_threshold, cancel_check=cancel_check)

    def expand_dir(self, directory: Path | str) -> list[Path]:
        """Indexable files under ``directory`` (used when a whole folder appears)."""
        return list(self.iter_files(directory))

    def mark_missing(self, root: Path | str, paths: Iterable[Path | str],
                     db_path: Path | str | None = None) -> int:
        """Mark records at or under each vanished path as missing; returns the count."""
        root = Path(root).expanduser().resolve()
        prefix = str(root) + os.sep
        targets = [str(p) for p in paths]
        db = db_path or self.cfg.db_path
        init_db(db)
        gone: list[int] = []
        with get_session(db) as session:
            for path, k in self._known_rows(session).items():
                if k.status == "missing" or not path.startswith(prefix):
                    continue
                if any(path == t or path.startswith(t + os.sep) for t in targets) \
                        and not os.path.exists(path):
                    gone.append(k.id)
            for chunk in _chunks(gone, _IN_CHUNK):
                for row in session.exec(sselect(ImageRecord).where(
                        ImageRecord.id.in_(chunk))).all():  # type: ignore[attr-defined]
                    row.scan_status = "missing"
                    row.updated_at = utcnow()
        return len(gone)

    # ------------------------------------------------------------------- run
    def _known_rows(self, session: Session) -> dict[str, _Known]:
        cols = [ImageRecord.path, ImageRecord.id, ImageRecord.file_size,
                ImageRecord.scan_status, ImageRecord.file_mtime, ImageRecord.created_at, ImageRecord.stage]
        out: dict[str, _Known] = {}
        for path, iid, size, status, mtime, created, stage in session.exec(sselect(*cols)).all():
            out[path] = _Known(iid, size or 0, mtime, status,
                               _as_utc_ts(created) if created is not None else None, stage)
        return out

    @staticmethod
    def _is_modified(k: _Known, st: os.stat_result) -> bool:
        if st.st_size != k.size:
            return True
        if k.stamp is not None:
            return abs(st.st_mtime - k.stamp) > 1e-3
        # Row predates stored mtimes: anything written after it was created is newer.
        return k.created is not None and st.st_mtime > k.created + 0.5

    def _run(self, root: Path, files: list[Path], result: ScanResult,
             db_path, on_progress, show_progress: bool, prune_missing: bool,
             compute_phashes: bool, settle_s: float,
             near_dup_threshold: int,
             cancel_check: Callable[[], bool] | None = None) -> ScanResult:
        db = db_path or self.cfg.db_path
        init_db(db)  # idempotent: ensures the schema exists for fresh stores
        t0 = time.monotonic()
        now = time.time()

        with get_session(db) as session:
            repo = PhotoRepository(session)
            known = self._known_rows(session)

            # Classify: unchanged (skip), restored, modified (re-process), new.
            work: list[tuple[Path, int | None]] = []
            restored: list[int] = []
            for p in files:
                k = known.get(str(p))
                if k is None and settle_s <= 0:
                    work.append((p, None))
                    continue
                try:
                    st = p.stat()
                except OSError:
                    if k is not None:
                        result.skipped_known += 1
                    continue
                if settle_s > 0 and (now - st.st_mtime) < settle_s:
                    result.deferred_files.append(str(p))
                    continue
                if k is None:
                    work.append((p, None))
                elif self._is_modified(k, st):
                    work.append((p, k.id))
                else:
                    result.skipped_known += 1
                    if k.status == "missing":
                        restored.append(k.id)

            modified_ids = {i for _, i in work if i is not None}
            if modified_ids:
                self._invalidate(session, modified_ids)
                session.commit()  # release the write lock before the slow reads
            for chunk in _chunks(restored, _IN_CHUNK):
                for row in session.exec(sselect(ImageRecord).where(
                        ImageRecord.id.in_(chunk))).all():  # type: ignore[attr-defined]
                    row.scan_status = "scanned"
                    row.updated_at = utcnow()

            if work:
                self._process(root, work, repo, result, on_progress, show_progress,
                              compute_phashes, near_dup_threshold, modified_ids,
                              cancel_check=cancel_check)

            if prune_missing:
                self._prune(session, root, {str(p) for p in files}, known, result)

            if self.cfg.geocode:
                from .geo import geocode_pending
                result.geocoded = geocode_pending(session, self.cfg.geocode_batch)

        result.scanned = len(files)
        result.duration_s = time.monotonic() - t0
        return result

    # ----------------------------------------------------------- invalidation
    @staticmethod
    def _invalidate(session: Session, ids: set[int]) -> None:
        """Reset derived state of rows whose files changed on disk."""
        repo = PhotoRepository(session)
        for chunk in _chunks(sorted(ids), _IN_CHUNK):
            for row in session.exec(sselect(ImageRecord).where(
                    ImageRecord.id.in_(chunk))).all():  # type: ignore[attr-defined]
                row.quick_hash = ""
                row.sha256 = ""
                row.phash = None
                row.is_duplicate = False
                row.duplicate_of = None
                row.near_dup_of = None
                row.burst_id = None
                row.is_best_shot = False
                row.is_alternate = False
                row.infer_status = "pending"
                row.route_status = "unrouted"
                row.category = None
                row.subcategory = None
                row.target_path = None
                row.city = row.region = row.country_code = ""  # the GPS may have changed
                row.scan_status = "scanned"
                row.quality = None
                for face in row.faces:
                    if face.person_id is not None:
                        repo.decrement_person_cluster(face.person_id)
                row.faces = []
                row.delete_approved = False  # an old approval must not apply to new content
                row.manual_bucket = None
                row.semantic = None
                row.updated_at = utcnow()
            # Rows that pointed at a changed original no longer match its bytes.
            for dep in session.exec(sselect(ImageRecord).where(
                    ImageRecord.duplicate_of.in_(chunk))).all():  # type: ignore[attr-defined]
                dep.is_duplicate = False
                dep.duplicate_of = None
                dep.updated_at = utcnow()
            for dep in session.exec(sselect(ImageRecord).where(
                    ImageRecord.near_dup_of.in_(chunk))).all():  # type: ignore[attr-defined]
                dep.near_dup_of = None
                dep.updated_at = utcnow()
        session.flush()

    # ----------------------------------------------------------------- prune
    @staticmethod
    def _prune(session: Session, root: Path, seen: set[str],
               known: dict[str, _Known], result: ScanResult) -> None:
        prefix = str(root) + os.sep
        blocked = tuple(str(Path(d)) + os.sep for d in result.walk_errors)
        gone = [k.id for path, k in known.items()
                if path.startswith(prefix) and path not in seen
                and k.status != "missing" and k.stage == "new"  # finished photos live in the library, not here
                and not (blocked and path.startswith(blocked))]
        for chunk in _chunks(gone, _IN_CHUNK):
            for row in session.exec(sselect(ImageRecord).where(
                    ImageRecord.id.in_(chunk))).all():  # type: ignore[attr-defined]
                row.scan_status = "missing"
                row.updated_at = utcnow()
        result.missing = len(gone)

    # --------------------------------------------------------------- process
    def _process(self, root: Path, work: list[tuple[Path, int | None]],
                 repo: PhotoRepository, result: ScanResult, on_progress,
                 show_progress: bool, compute_phashes: bool,
                 near_dup_threshold: int, modified_ids: set[int],
                 cancel_check: Callable[[], bool] | None = None) -> None:
        session = repo.s
        # Tier-1 fingerprints in the store; a new file needs a full SHA-256
        # only if its quick-hash is in this set or collides within this pass.
        known_qh = repo.known_quick_hashes()
        index = self._build_index(session, near_dup_threshold)
        near_touched: set[int] = set()

        # Same-pass originals keyed by quick-hash. Exact duplicates always share
        # it, so later copies find their original without a DB round-trip.
        seen_qh: dict[str, list[ImageRecord]] = {}
        # (dup, original) pairs whose duplicate_of is linked after flush.
        pending_dupes: list[tuple[ImageRecord, ImageRecord]] = []
        batch: list[ImageRecord] = []
        indexable: list[ImageRecord] = []

        def worker(path: Path) -> _WorkerResult:
            """Pure filesystem I/O worker: stat + quick hash + EXIF + pHash.

            Never touches the DB session (thread-safe by construction).
            """
            wr = _WorkerResult(path=path)
            try:
                st = path.stat()
                wr.size, wr.mtime = st.st_size, st.st_mtime
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
            wr.exif = extract_exif(path)
            if compute_phashes and wr.exif.width and wr.exif.height:
                wr.phash = compute_phash(path)
            return wr

        def fill(rec: ImageRecord, wr: _WorkerResult) -> None:
            exif = wr.exif
            rec.path = str(wr.path)
            rec.relative_path = str(wr.path.relative_to(root))
            rec.filename = wr.path.name
            rec.extension = wr.path.suffix.lower()
            rec.mime_type = _guess_mime(wr.path, rec.extension)
            rec.file_size = wr.size
            rec.quick_hash = wr.quick_hash
            rec.width = exif.width
            rec.height = exif.height
            rec.orientation = exif.orientation
            rec.taken_at = taken_at_utc(exif)
            rec.taken_at_source = exif.taken_at_source
            rec.tz_offset_minutes = exif.tz_offset_minutes
            rec.gps_lat = exif.gps_lat
            rec.gps_lon = exif.gps_lon
            rec.camera_model = exif.camera_model
            if exif.rating is not None:
                rec.rating = exif.rating
            rec.phash = wr.phash
            rec.file_mtime = wr.mtime

        def build_record(wr: _WorkerResult, existing_id: int | None) -> ImageRecord | None:
            """Main-thread: turn a worker result into a persisted/pending row."""
            if wr.error or wr.exif is None or not wr.exif.width or not wr.exif.height:
                if wr.exif is not None:
                    log.debug("skip non-image file %s: %s", wr.path, wr.exif.error)
                    result.skipped_files.append(str(wr.path))
                else:
                    log.warning("scan error %s: %s", wr.path, wr.error)
                if existing_id is not None:
                    row = repo.get_by_id(existing_id)
                    if row is not None:
                        row.infer_status = "failed"
                return None
            if existing_id is not None:
                rec = repo.get_by_id(existing_id)
                if rec is None:
                    rec = ImageRecord(path=str(wr.path), filename=wr.path.name)
                    existing_id = None
            else:
                rec = ImageRecord(path=str(wr.path), filename=wr.path.name)
            fill(rec, wr)
            if existing_id is None and rec.quick_hash in known_qh \
                    and self._find_in_library(rec, wr.path, repo) is not None:
                result.skipped_library += 1
                return _SKIPPED
            self._resolve_dedup(rec, wr.path, repo, seen_qh, pending_dupes,
                                known_qh, result)
            if existing_id is None:
                batch.append(rec)
            else:
                rec.updated_at = utcnow()
            indexable.append(rec)
            return rec

        def flush() -> None:
            if batch:
                repo.bulk_upsert_images(batch)
                batch.clear()
            # All batch rows now carry ids; resolve same-batch duplicate links.
            for dup, orig in pending_dupes:
                dup.duplicate_of = orig.id
                dup.updated_at = utcnow()
            pending_dupes.clear()
            for rec in indexable:
                if rec.id is None or not rec.phash or rec.is_duplicate:
                    continue
                for dup_id, keeper_id in index.add(
                        rec.id, rec.phash, (rec.width or 0) * (rec.height or 0),
                        rec.file_size or 0, rec.quick_hash, rec.sha256):
                    row = repo.get_by_id(dup_id)
                    if row is not None and row.near_dup_of != keeper_id:
                        row.near_dup_of = keeper_id
                        row.updated_at = utcnow()
                    near_touched.add(dup_id)
            indexable.clear()
            session.commit()  # short write transactions so UI/engine writers are not starved

        progress = None
        if show_progress:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(), TaskProgressColumn(),
                TimeElapsedColumn(), TimeRemainingColumn(), transient=False)
            main_task = progress.add_task("Scanning", total=len(work))
            progress.start()
        try:
            with cf.ThreadPoolExecutor(max_workers=self.cfg.scan_workers) as pool:
                futures = {pool.submit(worker, p): (p, eid) for p, eid in work}
                done = 0
                for fut in cf.as_completed(futures):
                    if cancel_check and cancel_check():
                        for f in futures:
                            f.cancel()
                        pool.shutdown(wait=False, cancel_futures=True)
                        flush()
                        raise InterruptedError("Scan cancelled by user")
                    path, eid = futures[fut]
                    wr = fut.result()
                    rec = build_record(wr, eid)
                    if rec is _SKIPPED:
                        pass
                    elif rec is not None:
                        if eid is None:
                            result.new += 1
                        else:
                            result.modified += 1
                    else:
                        result.errors += 1
                        result.errors_list.append(str(path))
                    done += 1
                    if progress is not None:
                        progress.update(main_task, completed=done)
                    if on_progress:
                        on_progress(done, len(work), str(path))
                    if len(batch) + len(indexable) >= 50:
                        flush()
            flush()
        finally:
            if progress is not None:
                progress.stop()
        result.near_duplicates = len(near_touched)

    @staticmethod
    def _build_index(session: Session, threshold: int) -> NearDupIndex:
        index = NearDupIndex(threshold)
        rows = session.exec(sselect(
            ImageRecord.id, ImageRecord.phash, ImageRecord.width, ImageRecord.height,
            ImageRecord.file_size, ImageRecord.quick_hash, ImageRecord.sha256,
            ImageRecord.near_dup_of,
        ).where(ImageRecord.phash != None,  # noqa: E711
                ImageRecord.is_duplicate == False,  # noqa: E712
                ImageRecord.scan_status != "missing")).all()
        index.seed((i, ph, (w or 0) * (h or 0), sz, qh, sha, nd)
                   for i, ph, w, h, sz, qh, sha, nd in rows)
        return index

    # --------------------------------------------------------------- dedup
    def _resolve_dedup(self, rec: ImageRecord, path: Path, repo,
                       seen_qh: dict, pending_dupes: list,
                       known_qh: set, result: ScanResult) -> None:
        """Lazy two-tier dedup for a single record (mutates ``rec``).

        Tier-1 (cheap head/tail, always) has already run in the worker. Tier-2
        (the expensive full-file SHA-256) is read **only** when a quick-hash
        collision forces verification::

            same-pass quick-hash hit  -> hash all sides, dup if one matches
            in-store quick-hash hit   -> hash this file, verify vs the store
            first sighting (no hit)   -> store NO hash (deferred to apply time)

        Net effect: a library of *unique* photos pays only the O(1) head/tail
        cost and never triggers a full-file read during scan; the expensive
        read happens solely for files sharing a fingerprint with something
        already known (i.e. real or near-duplicates).
        """
        # 1. Same-pass originals sharing this fingerprint (all are compared).
        group = seen_qh.get(rec.quick_hash)
        if group:
            sha = _safe_sha256(path)
            rec.sha256 = sha
            for orig in group:
                if not orig.sha256:
                    orig.sha256 = _safe_sha256(Path(orig.path))
                if sha and sha == orig.sha256:
                    rec.is_duplicate = True
                    pending_dupes.append((rec, orig))
                    result.duplicates += 1
                    return
            # Mismatch on every candidate => genuinely different file; keep it.
            group.append(rec)
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
            seen_qh.setdefault(rec.quick_hash, []).append(rec)
            return

        # 3. First sighting of this fingerprint: no full-file read (lazy Tier-2).
        seen_qh.setdefault(rec.quick_hash, []).append(rec)

    @staticmethod
    def _find_in_library(rec: ImageRecord, path: Path, repo) -> ImageRecord | None:
        """A library or trash photo with this exact content (trash rows are marked missing)."""
        origin = repo.s.exec(sselect(ImageRecord).where(
            ImageRecord.origin_path == str(path), ImageRecord.stage.in_(("library", "trash")),
            ImageRecord.route_status != "purged")).first()
        if origin is not None and origin.file_size == rec.file_size:
            return origin  # the photo came from this very path and is the same size: no need to read it
        stmt = sselect(ImageRecord).where(
            ImageRecord.quick_hash == rec.quick_hash,
            ImageRecord.stage.in_(("library", "trash")),
            ImageRecord.route_status != "purged")
        cands = repo.s.exec(stmt).all()
        if not cands:
            return None
        sha = rec.sha256 or _safe_sha256(path)
        if not sha:
            return None
        rec.sha256 = sha
        for cand in cands:
            if not cand.sha256:
                cand.sha256 = _safe_sha256(Path(cand.path))
                repo.s.flush()
            if cand.sha256 == sha:
                return cand
        return None

    @staticmethod
    def _find_db_original(sha: str, repo, rec: ImageRecord) -> ImageRecord | None:
        """Locate a definitive SHA-256 match already persisted (any live path)."""
        stmt = sselect(ImageRecord).where(ImageRecord.sha256 == sha,
                                          ImageRecord.scan_status != "missing",
                                          ImageRecord.is_duplicate == False)  # noqa: E712
        if rec.id is not None:
            stmt = stmt.where(ImageRecord.id != rec.id)
        hit = repo.s.exec(stmt).first()
        if hit is not None:
            return hit
        # A prior *lazy* run may have left a candidate's sha256 empty: backfill.
        for cand in repo.find_by_quick_hash(rec.quick_hash, exclude_id=rec.id):
            if cand.scan_status == "missing" or cand.is_duplicate:
                continue
            if not cand.sha256:
                cand.sha256 = _safe_sha256(Path(cand.path))
                repo.s.flush()
            if cand.sha256 == sha:
                return cand
        return None


def _chunks(seq, n: int):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


_LIBRARY_BUNDLE_SUFFIXES = (".photoslibrary", ".aplibrary", ".migratedphotolibrary", ".lrdata")


def _is_library_bundle(name: str) -> bool:
    # Managed photo-app libraries hold derivatives and must not be indexed or moved.
    return name.lower().endswith(_LIBRARY_BUNDLE_SUFFIXES)


def _os_walk(root: Path, skip_dirs: set[str], errors: list | None = None):
    """os.walk wrapper that reports unreadable directories into ``errors``."""
    def onerror(e: OSError) -> None:
        if errors is not None:
            errors.append(str(e.filename))
    return os.walk(root, followlinks=False, onerror=onerror)


def scan_directory(root: Path | str, config: CuratorConfig | None = None,
                   db_path: Path | str | None = None,
                   show_progress: bool = True,
                   prune_missing: bool = False,
                   **kwargs) -> ScanResult:
    cfg = config or CuratorConfig.load(db_path)
    return FileScanner(cfg).scan_directory(root, db_path=db_path,
                                           show_progress=show_progress,
                                           prune_missing=prune_missing, **kwargs)


# Alias used by library callers and the CLI.
scan_library = scan_directory
