"""Decide whether Photocure may write a photo's XMP sidecar, and record what it wrote.

A sidecar that changed outside Photocure is never overwritten by an automated pass.
An explicit user action wins, but only after the caller has confirmed with the user.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .db.models import ImageRecord
from .scanner.xmp import (OWNED_TAG_PREFIX, get_on_disk_owned_tags, get_on_disk_rating,
                          update_xmp_sidecar)

WRITTEN = "written"
EXISTING_FOREIGN = "existing_foreign_rating"  # first run found a rating we did not write
EDITED_ELSEWHERE = "edited_elsewhere"  # on-disk values differ from what we last wrote
NEEDS_CONFIRMATION = "needs_confirmation"  # explicit action over a foreign edit
SKIPPED = "sidecar_skipped"
SKIPPED_DUAL = "sidecar_skipped_dual"


@dataclass
class SyncResult:
    status: str
    detail: str = ""


def _join(tags) -> str:
    return ",".join(sorted(tags))


def foreign_state(img: ImageRecord) -> str | None:
    """EXISTING_FOREIGN, EDITED_ELSEWHERE, or None when the sidecar is ours or absent."""
    disk_rating = get_on_disk_rating(img.path)
    disk_tags = get_on_disk_owned_tags(img.path)
    if img.xmp_written_rating is None and img.xmp_written_tags is None:
        return EXISTING_FOREIGN if disk_rating is not None else None
    written_tags = set(filter(None, (img.xmp_written_tags or "").split(",")))
    if disk_rating != img.xmp_written_rating or disk_tags != written_tags:
        return EDITED_ELSEWHERE
    return None


def sync_sidecar(img: ImageRecord, *, rating: int | None = None, tags: tuple[str, ...] = (),
                 explicit: bool = False, overwrite_foreign: bool = False) -> SyncResult:
    """Write ``rating`` and ``tags`` (photocure-* only) for ``img`` when that is safe.

    ``explicit`` marks a user action. Over a foreign edit it returns NEEDS_CONFIRMATION until
    the caller repeats it with ``overwrite_foreign=True``.
    """
    tags = tuple(t for t in tags if t.startswith(OWNED_TAG_PREFIX))
    state = foreign_state(img)
    if state and not explicit:
        if state == EDITED_ELSEWHERE:
            return SyncResult(EDITED_ELSEWHERE, "sidecar changed outside Photocure; left untouched")
        rating = None  # keep the existing rating, still record our own tags
    elif state and explicit and not overwrite_foreign:
        return SyncResult(NEEDS_CONFIRMATION, state)

    old_owned = get_on_disk_owned_tags(img.path)
    ok, info = update_xmp_sidecar(Path(img.path), rating=rating,
                                  add_tags=tags, remove_tags=tuple(old_owned - set(tags)))
    if not ok:
        return SyncResult(SKIPPED_DUAL if info == "sidecar_skipped_dual" else SKIPPED, info)
    if rating is not None:
        img.xmp_written_rating = rating
    img.xmp_written_tags = _join(tags)
    return SyncResult(EXISTING_FOREIGN if state and not explicit else WRITTEN, info)
