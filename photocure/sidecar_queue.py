"""Debounced background writer for XMP sidecars.

A burst of rating changes (holding a key down a row of photos) becomes one write per photo,
a moment after the last change. Writes never block the request and never raise to the caller.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from .db.models import ImageRecord
from .db.session import get_session
from .sidecar_sync import WRITTEN, SyncResult, sync_sidecar

log = logging.getLogger("photocure.sidecars")


def desired_tags(rec: ImageRecord) -> tuple[str, ...]:
    """The photocure-* tags that describe this photo right now."""
    tags = []
    if rec.is_best_shot:
        tags.append("photocure-best-shot")
    if rec.has_blinks:
        tags.append("photocure-blink")
    if rec.rating == -1:
        tags.append("photocure-rejected")
    return tuple(tags)


class SidecarQueue:
    def __init__(self, db_path: Path | str, debounce: float = 1.5):
        self.db_path = db_path
        self.debounce = debounce
        self._lock = threading.Lock()
        # Held while a sidecar is written AND its database record is committed, and by anyone who
        # compares the two. Without it a request can see the new file next to the old record and
        # wrongly decide another program edited the sidecar.
        self.io_lock = threading.RLock()
        self._pending: dict[int, bool] = {}   # image_id -> overwrite a foreign edit
        self._timer: threading.Timer | None = None
        self.last_results: dict[int, SyncResult] = {}

    def schedule(self, image_id: int, overwrite: bool = False) -> None:
        with self._lock:
            self._pending[image_id] = self._pending.get(image_id, False) or overwrite
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce, self.flush)
            self._timer.daemon = True
            self._timer.start()

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._pending)

    def flush(self) -> dict[int, SyncResult]:
        """Write every queued sidecar now. Returns the outcome per photo."""
        with self._lock:
            batch, self._pending = self._pending, {}
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        results: dict[int, SyncResult] = {}
        # Keywords follow the setting as it is now, so switching it off strips them on the next write.
        from .features import load_features
        from .highlights import HighlightWeights
        from .keywords import keyword_tags
        feats = load_features(self.db_path)
        weights = HighlightWeights.from_features(feats)
        for image_id, overwrite in batch.items():
            try:
                with self.io_lock, get_session(self.db_path) as s:  # the commit happens inside the lock
                    rec = s.get(ImageRecord, image_id)
                    if rec is None:
                        continue
                    rating = rec.rating if rec.rating is not None else 0
                    tags = desired_tags(rec)
                    if feats["write_keywords"]:
                        tags += keyword_tags(s, rec, weights)
                    res = sync_sidecar(rec, rating=rating, tags=tags,
                                       explicit=True, overwrite_foreign=overwrite)
            except Exception as e:  # a failed sidecar must never take the server down
                res = SyncResult("error", f"{e.__class__.__name__}: {e}")
            results[image_id] = res
            if res.status != WRITTEN:
                log.warning("sidecar for photo %s not written: %s %s", image_id, res.status, res.detail)
        self.last_results.update(results)
        return results

    def close(self) -> None:
        self.flush()
