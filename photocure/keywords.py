"""Keyword export: write category, highlight and event keywords into XMP sidecars.

Every keyword starts with ``photocure-`` so Photocure only ever adds, changes or removes its own
tags. Other programs' keywords, ratings and sidecars edited elsewhere are left alone, using the
same safety rules as rating sync.
"""
from __future__ import annotations

import re
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

from sqlmodel import col, select

from .config import CuratorConfig
from .db.models import EventGroup, ImageRecord, SemanticScore
from .db.session import get_session
from .features import load_features
from .highlights import HighlightWeights, highlight_score
from .scanner.xmp import find_sidecar
from .sidecar_sync import EDITED_ELSEWHERE, EXISTING_FOREIGN, SKIPPED, SKIPPED_DUAL, WRITTEN, sync_sidecar

HIGHLIGHT_MIN = 0.65
_MAX_LABEL = 64
_DEFAULT_EVENT = re.compile(r"^event\d+$", re.IGNORECASE)


def is_highlight(rec: ImageRecord, sem: SemanticScore | None, w: HighlightWeights) -> bool:
    """True for a scored, unrejected, non-duplicate photo whose ranking score is high enough."""
    if sem is None or rec.is_duplicate or rec.is_alternate:
        return False
    score = highlight_score(sem.aesthetic, sem.preference, rec.rating, bool(rec.is_favorite), w)
    return score is not None and score >= HIGHLIGHT_MIN


def _clean_label(label: str | None) -> str:
    text = "".join(ch for ch in (label or "") if ch.isprintable()).replace(",", " ")
    return " ".join(text.split())[:_MAX_LABEL].strip()


def build_keywords(rec: ImageRecord, highlight: bool, event_label: str | None) -> tuple[str, ...]:
    """The keywords that describe this photo, sorted. Default event names are not worth a keyword."""
    tags = set()
    cat = rec.manual_bucket or rec.category
    if cat and cat != "unrouted":
        tags.add(f"photocure-cat-{cat}")
    label = _clean_label(event_label)
    if label and not _DEFAULT_EVENT.match(label):
        tags.add(f"photocure-event-{label}")
    if highlight:
        tags.add("photocure-highlight")
    return tuple(sorted(tags))


def keyword_tags(session, rec: ImageRecord, w: HighlightWeights) -> tuple[str, ...]:
    ev = session.get(EventGroup, rec.event_id) if rec.event_id is not None else None
    return build_keywords(rec, is_highlight(rec, rec.semantic, w), ev.label if ev else None)


def write_all(db_path: Path | str, cfg: CuratorConfig, *, keywords: bool = True,
              weights: HighlightWeights | None = None, io_lock=None, on_progress=None,
              cancel_check=None) -> dict:
    """Write (or with ``keywords=False`` remove) keywords in every photo's sidecar.

    Removal only touches sidecars that already exist. Ratings are never changed, and a sidecar that
    was edited outside Photocure is skipped. Returns counts per outcome.
    """
    from .sidecar_queue import desired_tags
    if not cfg.write_sidecars:
        raise ValueError("Sidecar writing is turned off in the settings.")
    w = weights or HighlightWeights.from_features(load_features(db_path))
    with get_session(db_path) as s:
        ids = [int(i) for i in s.exec(select(ImageRecord.id).where(
            col(ImageRecord.scan_status) != "missing").order_by(ImageRecord.id)).all()]
    total = len(ids)
    out = Counter(total=total, written=0, kept_foreign_rating=0, edited_elsewhere=0, skipped=0,
                  errors=0, no_sidecar=0, nothing_to_write=0)
    out["cancelled"] = False
    outcomes = {WRITTEN: "written", EXISTING_FOREIGN: "kept_foreign_rating",
                EDITED_ELSEWHERE: "edited_elsewhere", SKIPPED: "skipped", SKIPPED_DUAL: "skipped"}
    for n, image_id in enumerate(ids, 1):
        if cancel_check and cancel_check():
            out["cancelled"] = True
            break
        try:
            with (io_lock or nullcontext()), get_session(db_path) as s:
                rec = s.get(ImageRecord, image_id)
                if rec is None or not Path(rec.path).exists():
                    out["skipped"] += 1
                elif not keywords and find_sidecar(rec.path) is None:
                    out["no_sidecar"] += 1
                else:
                    tags = desired_tags(rec) + (keyword_tags(s, rec, w) if keywords else ())
                    if not tags and find_sidecar(rec.path) is None:
                        out["nothing_to_write"] += 1  # do not litter the library with empty sidecars
                    else:
                        res = sync_sidecar(rec, rating=None, tags=tags, explicit=False)
                        out[outcomes.get(res.status, "errors")] += 1
        except Exception:
            out["errors"] += 1
        if on_progress:
            on_progress(n, total)
    return dict(out)
