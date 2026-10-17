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
from contextlib import AbstractContextManager, nullcontext
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
          show_progress: bool = False,
          scan_gate: Callable[[], AbstractContextManager] | None = None) -> ScanResult:
    """Ingest ``target`` until ``stop_event`` is set; returns the summed result.

    ``scan_gate`` returns a context manager held around every scan, so an embedding app can keep the
    watcher from writing while another task runs. It may raise ``InterruptedError`` to end the watch.
    """
    gate = scan_gate or nullcontext
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

    def first_scan() -> ScanResult:
        with gate():
            res = scanner.scan_directory(root, db_path=db, show_progress=show_progress,
                                         prune_missing=prune_missing, settle_s=debounce_s)
        record(res)
        return res

    if use_watchdog is None:
        use_watchdog = _watchdog_available()
    if use_watchdog:
        # Events must flow while the first scan runs, or a file landing in between is never seen.
        _watch_events(scanner, root, db, debounce_s, stop, record, prune_missing,
                      first_scan if initial_scan else None, gate)
    else:
        if initial_scan:
            first_scan()
        interval = poll_interval if poll_interval is not None else max(debounce_s, 1.0)
        _watch_poll(scanner, root, db, debounce_s, interval, stop, record,
                    prune_missing, gate)
    return total


def _watch_poll(scanner: FileScanner, root: Path, db, debounce_s: float,
                interval: float, stop: threading.Event,
                record: Callable[[ScanResult], None], prune_missing: bool,
                gate: Callable[[], AbstractContextManager] = nullcontext) -> None:
    while not stop.wait(interval):
        try:
            with gate():
                res = scanner.scan_directory(root, db_path=db, show_progress=False,
                                             prune_missing=prune_missing,
                                             settle_s=debounce_s)
        except InterruptedError:
            return
        except Exception:
            log.exception("watch poll scan failed")
            continue
        if res.new or res.modified or res.missing or res.errors:
            record(res)


def _watch_events(scanner: FileScanner, root: Path, db, debounce_s: float,
                  stop: threading.Event,
                  record: Callable[[ScanResult], None],
                  prune_missing: bool = False,
                  first_scan: Callable[[], ScanResult] | None = None,
                  gate: Callable[[], AbstractContextManager] = nullcontext) -> None:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    pending: dict[Path, float] = {}
    removed: dict[Path, float] = {}
    new_dirs: dict[Path, float] = {}
    lock = threading.Lock()

    def touch(p) -> None:
        with lock:
            pending[Path(p)] = time.monotonic()

    def gone(p) -> None:
        if prune_missing:
            with lock:
                removed[Path(p)] = time.monotonic()

    def appeared_dir(p) -> None:
        with lock:
            new_dirs[Path(p)] = time.monotonic()

    class _Handler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory:
                appeared_dir(event.src_path)
            else:
                touch(event.src_path)

        def on_modified(self, event):
            if not event.is_directory:
                touch(event.src_path)

        def on_deleted(self, event):
            gone(event.src_path)

        def on_moved(self, event):
            gone(event.src_path)
            if event.is_directory:
                appeared_dir(event.dest_path)
            else:
                touch(event.dest_path)

    def take_ready(table: dict[Path, float], now: float) -> list[Path]:
        ready = [p for p, t in table.items() if now - t >= debounce_s]
        for p in ready:
            del table[p]
        return ready

    observer = Observer()
    observer.schedule(_Handler(), str(root), recursive=True)
    observer.start()
    tick = max(0.05, min(debounce_s / 4 if debounce_s else 0.05, 0.5))
    try:
        if first_scan is not None:
            # Files that arrive meanwhile are already queued by the observer. Files written too
            # recently to scan are deferred, and nothing else would bring them back in this mode.
            for f in first_scan().deferred_files:
                touch(f)
        while not stop.wait(tick):
            now = time.monotonic()
            with lock:
                ready = take_ready(pending, now)
                ready_removed = take_ready(removed, now)
                ready_dirs = take_ready(new_dirs, now)
            for d in ready_dirs:
                for f in scanner.expand_dir(d) if d.is_dir() else []:
                    touch(f)
            if ready_removed:
                try:
                    with gate():
                        n = scanner.mark_missing(root, ready_removed, db_path=db)
                except InterruptedError:
                    return
                except Exception:
                    log.exception("watch prune failed")
                else:
                    if n:
                        record(ScanResult(missing=n))
            if not ready:
                continue
            try:
                with gate():
                    res = scanner.scan_paths(root, ready, db_path=db, show_progress=False)
            except InterruptedError:
                return
            except Exception:
                log.exception("watch scan failed")
                continue
            record(res)
    finally:
        observer.stop()
        observer.join(timeout=5)
