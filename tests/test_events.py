"""Event clustering by time gap and optional GPS distance."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import select

from photocure.clustering.events import cluster_events, haversine_km
from photocure.db.models import EventGroup, ImageRecord

T0 = datetime(2024, 5, 12, 9, 0, 0, tzinfo=timezone.utc)


def _add(session, name, minutes, source="exif:dtoriginal", lat=None, lon=None, **kw):
    r = ImageRecord(path=f"/lib/{name}.jpg", filename=f"{name}.jpg",
                    taken_at=T0 + timedelta(minutes=minutes), taken_at_source=source,
                    gps_lat=lat, gps_lon=lon, **kw)
    session.add(r)
    session.flush()
    return r


def test_gap_splits_events(session):
    morning = [_add(session, f"m{i}", i * 10) for i in range(4)]
    evening = [_add(session, f"e{i}", 600 + i * 10) for i in range(3)]
    assert cluster_events(session, gap_hours=2.0) == 2
    assert len({r.event_id for r in morning}) == 1
    assert len({r.event_id for r in evening}) == 1
    assert morning[0].event_id != evening[0].event_id
    events = session.exec(select(EventGroup).order_by(EventGroup.start)).all()
    assert [e.image_count for e in events] == [4, 3]
    assert events[0].start == T0 and events[0].end == T0 + timedelta(minutes=30)
    assert events[0].label == f"Event{events[0].id}"


def test_gap_threshold_is_inclusive_of_the_limit(session):
    a = _add(session, "a", 0)
    b = _add(session, "b", 120)       # exactly 2 h later: same event
    c = _add(session, "c", 241)       # 121 min after b: new event
    d = _add(session, "d", 242)
    cluster_events(session, gap_hours=2.0)
    assert a.event_id == b.event_id
    assert c.event_id == d.event_id != a.event_id


def test_gps_jump_splits_events(session):
    paris = [_add(session, f"p{i}", i, lat=48.85, lon=2.35) for i in range(3)]
    london = [_add(session, f"l{i}", 5 + i, lat=51.5, lon=-0.12) for i in range(3)]
    assert cluster_events(session, gap_hours=2.0, gps_km=None) == 1
    assert cluster_events(session, gap_hours=2.0, gps_km=50.0) == 2
    assert paris[0].event_id != london[0].event_id


def test_missing_gps_does_not_split(session):
    a = _add(session, "a", 0, lat=48.85, lon=2.35)
    b = _add(session, "b", 1)                      # no GPS
    c = _add(session, "c", 2, lat=48.86, lon=2.36)
    cluster_events(session, gap_hours=2.0, gps_km=5.0)
    assert a.event_id == b.event_id == c.event_id is not None


def test_mtime_unknown_and_duplicates_are_ignored(session):
    keep = [_add(session, f"k{i}", i) for i in range(2)]
    skipped = [_add(session, "m", 1, source="mtime"),
               _add(session, "u", 1, source="unknown"),
               _add(session, "d", 1, is_duplicate=True)]
    assert cluster_events(session) == 1
    assert all(r.event_id is None for r in skipped)
    assert all(r.event_id is not None for r in keep)


def test_singletons_are_not_events(session):
    solo = _add(session, "solo", 0)
    assert cluster_events(session) == 0
    assert solo.event_id is None
    assert cluster_events(session, min_images=1) == 1
    assert solo.event_id is not None


def test_recluster_replaces_previous_events(session):
    rows = [_add(session, f"r{i}", m) for i, m in enumerate((0, 10, 100, 110))]
    assert cluster_events(session, gap_hours=2.0) == 1
    assert cluster_events(session, gap_hours=1.0) == 2
    assert len(session.exec(select(EventGroup)).all()) == 2
    assert all(r.event_id is not None for r in rows)
    assert cluster_events(session, gap_hours=48.0) == 1
    assert len(session.exec(select(EventGroup)).all()) == 1


def test_naive_timestamps_are_treated_as_utc():
    from photocure.clustering.events import _ts
    naive = datetime(2024, 5, 12, 9, 0, 0)
    assert _ts(naive) == _ts(naive.replace(tzinfo=timezone.utc))


def test_empty_library(session):
    assert cluster_events(session) == 0


def test_haversine_known_distance():
    assert haversine_km(48.8566, 2.3522, 51.5074, -0.1278) == pytest.approx(343.5, rel=0.01)
    assert haversine_km(10, 10, 10, 10) == 0.0
