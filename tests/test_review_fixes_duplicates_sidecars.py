"""Fixes from the final review of duplicate resolution and sidecar writes."""
from __future__ import annotations

import pytest

from photocure import duplicates
from photocure.db import models as M
from photocure.db.session import get_session, init_db
from photocure.scanner.xmp import read_xmp
from photocure.sidecar_queue import SidecarQueue

from .conftest import make_image


def _rows(session, *specs):
    out = []
    for i, kw in enumerate(specs, 1):
        r = M.ImageRecord(path=f"/p/{i}.jpg", filename=f"{i}.jpg", **kw)
        session.add(r)
        out.append(r)
    session.flush()
    return out


def test_resolve_group_refuses_a_keeper_that_is_in_the_trash(session):
    keep, other = _rows(session, dict(sha256="S", stage="trash", scan_status="missing", route_status="trashed"),
                        dict(sha256="S"))
    with pytest.raises(ValueError, match="trash|missing"):
        duplicates.resolve_group(session, keep.id, [other.id], "exact")
    assert other.is_duplicate is False


def test_resolve_group_exact_needs_identical_content(session):
    keep, other = _rows(session, dict(sha256="A"), dict(sha256="B"))
    with pytest.raises(ValueError, match="identical"):
        duplicates.resolve_group(session, keep.id, [other.id], "exact")


def test_a_near_group_cannot_keep_a_photo_already_junked_as_an_exact_duplicate(session):
    keep, other = _rows(session, dict(sha256="A", is_duplicate=True), dict(sha256="B"))
    with pytest.raises(ValueError, match="duplicate"):
        duplicates.resolve_group(session, keep.id, [other.id], "near")


def test_resolve_all_skips_a_group_whose_keeper_an_earlier_group_just_junked(session, monkeypatch):
    a, a2, n = _rows(session, dict(sha256="X"), dict(sha256="X"), dict(sha256="Y", near_dup_of=2))
    groups = [{"kind": "exact", "keeper_id": a.id, "members": [{"id": a.id}, {"id": a2.id}]},
              {"kind": "near", "keeper_id": a2.id, "members": [{"id": a2.id}, {"id": n.id}]}]
    monkeypatch.setattr(duplicates, "find_groups", lambda *args, **kw: groups)
    out = duplicates.resolve_all(session)
    assert out["groups"] == 1 and out["discarded"] == 1
    assert a2.is_duplicate is True and a2.manual_bucket == "duplicate"  # still junked, not half undone
    assert n.manual_bucket is None and n.near_dup_of == a2.id  # the distinct photo was left alone


@pytest.fixture()
def photo(tmp_db, tmp_path):
    init_db(tmp_db)
    path = make_image(tmp_path / "a.jpg", rng_seed=1)
    with get_session(tmp_db) as s:
        r = M.ImageRecord(path=str(path), filename="a.jpg", infer_status="inferred", is_best_shot=True)
        s.add(r)
        s.commit()
        return type("P", (), {"db": tmp_db, "path": path, "id": r.id})


def _write(photo):
    q = SidecarQueue(photo.db, debounce=60)
    q.schedule(photo.id)
    return q.flush()[photo.id]


def test_an_unrated_photo_never_gets_a_zero_star_rating_written(photo):
    assert _write(photo).status == "written"
    info = read_xmp(photo.path)
    assert info is not None and info.rating is None  # the best-shot tag went in, no Rating


def test_clearing_a_rating_still_clears_it_in_the_sidecar(photo):
    with get_session(photo.db) as s:
        r = s.get(M.ImageRecord, photo.id)
        r.rating = 3
        s.commit()
    _write(photo)
    assert read_xmp(photo.path).rating == 3
    with get_session(photo.db) as s:
        s.get(M.ImageRecord, photo.id).rating = None  # the user cleared it
        s.commit()
    _write(photo)
    assert read_xmp(photo.path).rating == 0
