"""Every listing filters by stage: new, library, trash or all (new and library together)."""
from __future__ import annotations

import pytest

from photocure.db import models as M
from photocure.db.session import get_session

from .test_ui_api import _seed_face, _seed_person, env  # noqa: F401  (shared fixture and seeders)


def _set(env, image_id, stage, **kw):
    with get_session(env.db) as s:
        r = s.get(M.ImageRecord, image_id)
        r.stage = stage
        if stage == "library":
            r.route_status = "executed"
        if stage == "trash":
            r.route_status, r.scan_status, r.delete_approved = "trashed", "missing", True
        for k, v in kw.items():
            setattr(r, k, v)
        s.commit()


def _ids(env, url):
    return {i["id"] for i in env.client.get(url).json()["items"]}


@pytest.fixture()
def staged(env):
    ids = sorted(i["id"] for i in env.client.get("/api/images?limit=50").json()["items"])
    a, b, c, d = ids
    _set(env, c, "library")
    _set(env, d, "trash")
    return type("S", (), {"env": env, "a": a, "b": b, "c": c, "d": d})


def test_images_filter_by_stage(staged):
    e = staged.env
    assert _ids(e, "/api/images?stage=new") == {staged.a, staged.b}
    assert _ids(e, "/api/images?stage=library") == {staged.c}
    assert _ids(e, "/api/images?stage=trash") == {staged.d}
    assert _ids(e, "/api/images") == {staged.a, staged.b, staged.c}  # all = new + library
    assert _ids(e, "/api/images?stage=all") == {staged.a, staged.b, staged.c}


def test_images_reject_unknown_stage(staged):
    assert staged.env.client.get("/api/images?stage=archive").status_code == 422


def test_purged_trash_rows_are_not_listed(staged):
    _set(staged.env, staged.d, "trash", route_status="purged")
    assert _ids(staged.env, "/api/images?stage=trash") == set()


def test_overview_stage_counts_and_scoped_totals(staged):
    e = staged.env
    ov = e.client.get("/api/overview").json()
    assert ov["stage_counts"] == {"new": 2, "library": 1, "trash": 1}
    assert e.client.get("/api/overview?stage=new").json()["total"] == 2
    assert e.client.get("/api/overview?stage=library").json()["total"] == 1
    assert e.client.get("/api/overview?stage=trash").json()["total"] == 1


def _burst(env, name, members):
    with get_session(env.db) as s:
        bg = M.BurstGroup(name=name, image_count=len(members), best_shot_id=members[0])
        s.add(bg)
        s.flush()
        for i, iid in enumerate(members):
            r = s.get(M.ImageRecord, iid)
            r.burst_id, r.is_best_shot, r.is_alternate = bg.id, i == 0, i != 0
        s.commit()
        return bg.id


def test_bursts_hide_groups_with_no_member_in_stage(staged):
    e = staged.env
    # c and d need a partner in their own stage for a burst to survive pruning
    extra = M.ImageRecord(path="/x/extra.jpg", filename="extra.jpg", infer_status="inferred",
                          stage="library", route_status="executed")
    with get_session(e.db) as s:
        s.add(extra)
        s.commit()
        extra_id = extra.id
    b_new = _burst(e, "Burst_new", [staged.a, staged.b])
    b_lib = _burst(e, "Burst_lib", [staged.c, extra_id])
    new_ids = {b["id"] for b in e.client.get("/api/bursts?stage=new").json()["items"]}
    lib_ids = {b["id"] for b in e.client.get("/api/bursts?stage=library").json()["items"]}
    assert new_ids == {b_new} and lib_ids == {b_lib}


def test_people_listing_follows_stage(staged):
    e = staged.env
    p_new = _seed_person(e, name="Newbie", face_count=1)
    p_lib = _seed_person(e, name="Libby", face_count=1)
    _seed_face(e, staged.a, person_id=p_new)
    _seed_face(e, staged.c, person_id=p_lib)
    with get_session(e.db) as s:
        t = M.PersonCluster(name="Binned", face_count=1, stage="trash")
        s.add(t)
        s.commit()
        p_trash = t.id
    _seed_face(e, staged.d, person_id=p_trash)

    def people(q=""):
        return {p["id"] for p in e.client.get(f"/api/people{q}").json()["items"]}

    assert people("?stage=new") == {p_new}
    assert people("?stage=library") == {p_lib}
    assert people("?stage=trash") == {p_trash}
    assert people() == {p_new, p_lib}


def test_person_photos_respect_stage(staged):
    e = staged.env
    pid = _seed_person(e, name="Both", face_count=2)
    _seed_face(e, staged.a, person_id=pid)
    _seed_face(e, staged.c, person_id=pid)

    def photos(stage):
        j = e.client.get(f"/api/people/{pid}/photos?stage={stage}").json()
        return {p["id"] for p in j["photos"]}

    assert photos("new") == {staged.a}
    assert photos("library") == {staged.c}
    assert photos("all") == {staged.a, staged.c}


def test_delete_person_in_new_stage_keeps_library_faces(staged):
    e = staged.env
    pid = _seed_person(e, name="Shared", face_count=2)
    f_new = _seed_face(e, staged.a, person_id=pid)
    f_lib = _seed_face(e, staged.c, person_id=pid)
    assert e.client.delete(f"/api/people/{pid}?stage=new").status_code == 200
    with get_session(e.db) as s:
        assert s.get(M.FaceDetection, f_new).ignored is True
        assert s.get(M.FaceDetection, f_lib).ignored is False
        assert s.get(M.FaceDetection, f_lib).person_id == pid
