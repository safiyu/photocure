"""Offline reverse geocoding of GPS positions (optional extra: photocure[geo]).

Uses the reverse_geocoder package, which looks up the nearest city in a bundled GeoNames table:
no network and no per-photo process pool (mode=1 runs in this process). Positions are looked up in
batches so the table is queried once per batch, not once per photo.
"""
from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable

from sqlmodel import select

from ..db.models import ImageRecord, utcnow

log = logging.getLogger("photocure.geo")


def available() -> bool:
    try:
        return importlib.util.find_spec("reverse_geocoder") is not None
    except (ImportError, ValueError):
        return False


def _default_search():
    import reverse_geocoder as rg

    def search(coords):
        return rg.search(coords, mode=1)
    return search


def geocode_pending(session, batch: int = 500, search: Callable | None = None, limit: int = 0) -> int:
    """Fill city, region and country_code for photos that have GPS but no place yet.

    Returns how many photos were filled. Does nothing (returns 0) when reverse_geocoder is not
    installed. The caller owns the commit.
    """
    if search is None:
        if not available():
            return 0
        search = _default_search()
    stmt = (select(ImageRecord).where(ImageRecord.gps_lat.is_not(None), ImageRecord.gps_lon.is_not(None),
                                      ImageRecord.country_code == "").order_by(ImageRecord.id))
    if limit:
        stmt = stmt.limit(limit)
    rows = list(session.exec(stmt).all())
    done = 0
    for i in range(0, len(rows), max(1, batch)):
        chunk = rows[i:i + batch]
        try:
            hits = search([(float(r.gps_lat), float(r.gps_lon)) for r in chunk])
        except Exception as e:  # a bad coordinate must not fail the whole scan
            log.warning("reverse geocoding failed for a batch of %d photos (%s: %s)", len(chunk),
                        e.__class__.__name__, e)
            continue
        for rec, hit in zip(chunk, hits):
            cc = str(hit.get("cc") or "").upper()
            if not cc:
                continue
            rec.city, rec.region, rec.country_code = str(hit.get("name") or ""), str(hit.get("admin1") or ""), cc
            rec.updated_at = utcnow()
            done += 1
        session.flush()
    return done
