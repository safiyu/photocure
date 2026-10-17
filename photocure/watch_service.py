"""Watch mode inside the app: scan new photos as they arrive, analyse them, and file them for review.

One background thread follows a folder with :func:`photocure.scanner.watcher.watch`. Every scan and
every analysis runs only while it holds the app's busy lock, so the watch never writes to the
library while a scan, an analysis or an apply started by the user is running.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .config import CuratorConfig

log = logging.getLogger("photocure.watch")


def default_process(db_path: Path | str, cfg: CuratorConfig) -> dict:
    """Analyse every pending photo, then give the new ones a category."""
    from .inference.pipeline import InferencePipeline
    from .services import Planner
    res = InferencePipeline(cfg).run(db_path=db_path, show_progress=False)
    routed = Planner(cfg, db_path=Path(db_path)).route_new()
    return {"processed": res.processed, "routed": routed}


class WatchService:
    def __init__(self, db_path: Path | str, cfg_factory: Callable[[], CuratorConfig], *,
                 busy: threading.Lock, process_fn: Callable[[CuratorConfig], dict] | None = None,
                 debounce_s: float = 2.0, poll_interval: float | None = None,
                 use_watchdog: bool | None = None, retry_s: float = 3.0,
                 on_cancel: Callable[[], None] | None = None, stop_timeout: float = 8.0):
        self.db_path = db_path
        self._cfg_factory = cfg_factory
        self._busy = busy
        self._process = process_fn or (lambda c: default_process(db_path, c))
        self._debounce_s = debounce_s
        self._poll_interval = poll_interval
        self._use_watchdog = use_watchdog
        self._retry_s = retry_s
        self._on_cancel = on_cancel          # asks a running analysis to stop early
        self._stop_timeout = stop_timeout
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._root: Path | None = None
        self._state = "stopped"
        self._photos = 0
        self._batches = 0
        self._last_event: str | None = None
        self._last_error: str | None = None

    # ----------------------------------------------------------------- control
    def start(self, path: Path | str) -> dict:
        """Watch ``path``, replacing any watch already running.

        Raises ``FileNotFoundError`` for a missing folder and ``RuntimeError`` while the previous watch
        is still finishing an analysis that could not be interrupted in time.
        """
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Folder does not exist: {root}")
        if not self.stop():
            raise RuntimeError("The previous watch is still finishing. Try again in a moment.")
        with self._lock:
            self._stop = threading.Event()
            self._root = root
            self._photos = self._batches = 0
            self._last_event = self._last_error = None
            self._state = "idle"
            self._thread = threading.Thread(target=self._run, args=(root, self._stop),
                                            name="photocure-watch", daemon=True)
            self._thread.start()
        return self.status()

    def stop(self) -> bool:
        """Stop watching. Returns False when an analysis is still winding down."""
        with self._lock:
            thread, stop, analysing = self._thread, self._stop, self._state == "analyzing"
            stop.set()
        if thread is not None and thread.is_alive():
            if analysing and self._on_cancel is not None:
                try:
                    self._on_cancel()
                except Exception:
                    log.debug("cancel hook failed", exc_info=True)
            thread.join(timeout=self._stop_timeout)
            if thread.is_alive():
                return False
        with self._lock:
            if self._thread is thread:
                self._thread = None
            self._state = "stopped"
        return True

    def note_failure(self, root: str, error: str) -> None:
        """Record why a saved watch could not be resumed."""
        with self._lock:
            self._root = Path(root)
            self._last_error = error
            self._state = "stopped"

    def status(self) -> dict:
        with self._lock:
            alive = self._thread is not None and self._thread.is_alive()
            state = "stopped" if not alive else ("stopping" if self._stop.is_set() else self._state)
            return {"running": alive, "state": state,
                    "path": str(self._root) if self._root else "", "photos_added": self._photos,
                    "batches": self._batches, "last_event": self._last_event,
                    "last_error": self._last_error}

    # ------------------------------------------------------------------ worker
    def _set(self, stop: threading.Event, **kw) -> None:
        """Update status, unless a newer watch has replaced the one that is reporting."""
        with self._lock:
            if stop is not self._stop:
                return
            for k, v in kw.items():
                setattr(self, "_" + k, v)

    @contextmanager
    def _gate(self, stop: threading.Event):
        """Hold the busy lock for the length of a scan, waiting for other tasks to finish first."""
        while True:
            if stop.is_set():
                raise InterruptedError("watch stopped")
            if self._busy.acquire(timeout=0.2):
                break
            self._set(stop, state="waiting")
        try:
            self._set(stop, state="scanning")
            yield
        finally:
            self._busy.release()
            self._set(stop, state="idle")

    def _run(self, root: Path, stop: threading.Event) -> None:
        from .scanner.watcher import watch

        def on_scan(res) -> None:
            added = int(res.new) + int(res.modified)
            if added <= 0:
                return
            with self._lock:
                if stop is self._stop:
                    self._photos += added
                    self._batches += 1
                    self._last_event = datetime.now(timezone.utc).isoformat()
            self._analyse(stop)

        try:
            watch(root, db_path=self.db_path, cfg=self._cfg_factory(), debounce_s=self._debounce_s,
                  stop_event=stop, poll_interval=self._poll_interval,
                  use_watchdog=self._use_watchdog, initial_scan=True, on_scan=on_scan,
                  scan_gate=lambda: self._gate(stop))
        except InterruptedError:
            pass  # stopped while waiting for the busy lock
        except Exception as e:
            log.exception("watch stopped")
            self._set(stop, last_error=f"{e.__class__.__name__}: {e}")

    def _analyse(self, stop: threading.Event) -> None:
        """Run the analysis once the busy lock is free; a failure is recorded, not raised."""
        while not stop.is_set():
            if not self._busy.acquire(timeout=0.2):
                self._set(stop, state="waiting")
                continue
            try:
                self._set(stop, state="analyzing")
                self._process(self._cfg_factory())
                self._set(stop, last_error=None)
            except Exception as e:
                log.warning("watch analysis failed: %s", e)
                self._set(stop, last_error=f"{e.__class__.__name__}: {e}")
            finally:
                self._busy.release()
                self._set(stop, state="idle")
            return
