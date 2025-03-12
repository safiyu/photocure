"""Event clustering: group photos into events by time gaps and optional GPS jumps.

A new event starts when the gap to the previous photo exceeds ``gap_hours`` or,
when ``gps_km`` is set, the photo is farther than that from the last photo that
had GPS. Photos dated from file mtime are ignored since copy time is not capture
time. Events with fewer than ``min_images`` photos are not recorded.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from sqlmodel import col, select as sselect

from ..db.models import EventGroup, ImageRecord

UNRELIABLE_TIME_SOURCES = frozenset({"mtime", "unknown"})


def _ts(v: datetime) -> float:
    if v.tzinfo is None:
        v = v.replace(tzinfo=timezone.utc)
    return v.timestamp()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def cluster_events(session, gap_hours: float = 2.0, gps_km: float | None = None,
                   min_images: int = 2) -> int:
    """Rebuild every :class:`EventGroup`; returns the number of events written."""
    for g in session.exec(sselect(EventGroup)).all():
        session.delete(g)
    for img in session.exec(
            sselect(ImageRecord).where(col(ImageRecord.event_id).is_not(None))).all():
        img.event_id = None
    session.flush()

    rows = [r for r in session.exec(
        sselect(ImageRecord).where(col(ImageRecord.taken_at).is_not(None))
    ).all() if not r.is_duplicate
        and r.taken_at_source not in UNRELIABLE_TIME_SOURCES]
    rows.sort(key=lambda r: (_ts(r.taken_at), r.id or 0))

    gap = float(gap_hours) * 3600.0
    groups: list[list[ImageRecord]] = []
    cur: list[ImageRecord] = []
    last_gps: tuple[float, float] | None = None
    prev_ts = None
    for r in rows:
        t = _ts(r.taken_at)
        split = False
        if prev_ts is not None and t - prev_ts > gap:
            split = True
        if (not split and gps_km is not None and last_gps is not None
                and r.gps_lat is not None and r.gps_lon is not None
                and haversine_km(last_gps[0], last_gps[1], r.gps_lat, r.gps_lon) > gps_km):
            split = True
        if split and cur:
            groups.append(cur)
            cur = []
        cur.append(r)
        prev_ts = t
        if r.gps_lat is not None and r.gps_lon is not None:
            last_gps = (r.gps_lat, r.gps_lon)
    if cur:
        groups.append(cur)

    written = 0
    for members in groups:
        if len(members) < max(1, min_images):
            continue
        ev = EventGroup(start=members[0].taken_at, end=members[-1].taken_at,
                        image_count=len(members))
        session.add(ev)
        session.flush()
        ev.label = f"Event{ev.id}"
        for m in members:
            m.event_id = ev.id
        written += 1
    session.flush()
    return written
