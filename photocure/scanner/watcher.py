"""Watch mode: continuously ingest new and changed photos from a folder.

Uses ``watchdog`` filesystem events when it is installed and falls back to
polling otherwise. Both modes debounce: a file is only scanned once it has
been quiet for ``debounce_s`` seconds, so half-synced files are not indexed.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from ..config import CuratorConfig
from .file_scanner import FileScanner, ScanResult

log = logging.getLogger("photocure.scanner.watcher")


def _merge(total: ScanResult, part: ScanResult) -> None:
    for name in ("scanned", "new", "modified", "skipped_known", "duplicates",
                 "near_duplicates", "missing", "errors"):
        setattr(total, name, getattr(total, name) + getattr(part, name))
    total.duration_s += part.duration_s
    total.errors_list.extend(part.errors_list)
    total.skipped_files.extend(part.skipped_files)
    total.walk_errors.extend(part.walk_errors)


def _watchdog_available() -> bool:
    try:
        import watchdog.observers  # noqa: F401
    except ImportError:
        return False
    return True


def watch(target: Path | str, db_path: Path | str | None = None,
          cfg: CuratorConfig | None = None, debounce_s: float = 2.0,
          stop_event: threading.Event | None = None,
          poll_interval: float | None = None,
          use_watchdog: bool | None = None,
          initial_scan: bool = True,
          prune_missing: bool = False,
          on_scan: Callable[[ScanResult], None] | None = None,
          show_progress: bool = False) -> ScanResult:
    """Ingest ``target`` until ``stop_event`` is set; returns the summed result."""
    root = Path(target).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Target directory does not exist: {root}")
    cfg = cfg or CuratorConfig.load(Path(db_path) if db_path else None)
    db = db_path or cfg.db_path
    stop = stop_event or threading.Event()
    scanner = FileScanner(cfg)
    total = ScanResult()

    def record(res: ScanResult) -> None:
        _merge(total, res)
        if on_scan is not None:
            on_scan(res)

    if initial_scan:
        record(scanner.scan_directory(root, db_path=db, show_progress=show_progress,
                                      prune_missing=prune_missing,
                                      settle_s=debounce_s))

    if use_watchdog is None:
        use_watchdog = _watchdog_available()
    if use_watchdog:
        _watch_events(scanner, root, db, debounce_s, stop, record)
    else:
        interval = poll_interval if poll_interval is not None else max(debounce_s, 1.0)
        _watch_poll(scanner, root, db, debounce_s, interval, stop, record,
                    prune_missing)
    return total


def _watch_poll(scanner: FileScanner, root: Path, db, debounce_s: float,
                interval: float, stop: threading.Event,
                record: Callable[[ScanResult], None], prune_missing: bool) -> None:
    while not stop.wait(interval):
        try:
            res = scanner.scan_directory(root, db_path=db, show_progress=False,
                                         prune_missing=prune_missing,
                                         settle_s=debounce_s)
        except Exception:
            log.exception("watch poll scan failed")
            continue
        if res.new or res.modified or res.missing or res.errors:
            record(res)


def _watch_events(scanner: FileScanner, root: Path, db, debounce_s: float,
                  stop: threading.Event,
                  record: Callable[[ScanResult], None]) -> None:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    pending: dict[Path, float] = {}
    lock = threading.Lock()

    def touch(p) -> None:
        with lock:
            pending[Path(p)] = time.monotonic()

    class _Handler(FileSystemEventHandler):
        def on_created(self, event):
            if not event.is_directory:
                touch(event.src_path)

        def on_modified(self, event):
            if not event.is_directory:
                touch(event.src_path)

        def on_moved(self, event):
            if not event.is_directory:
                touch(event.dest_path)

    observer = Observer()
    observer.schedule(_Handler(), str(root), recursive=True)
    observer.start()
    tick = max(0.05, min(debounce_s / 4 if debounce_s else 0.05, 0.5))
    try:
        while not stop.wait(tick):
            now = time.monotonic()
            with lock:
                ready = [p for p, t in pending.items() if now - t >= debounce_s]
                for p in ready:
                    del pending[p]
            if not ready:
                continue
            try:
                res = scanner.scan_paths(root, ready, db_path=db, show_progress=False)
            except Exception:
                log.exception("watch scan failed")
                continue
            record(res)
    finally:
        observer.stop()
        observer.join(timeout=5)
