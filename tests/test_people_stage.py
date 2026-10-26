"""People per stage: trash clusters stay apart, and Restore matches faces back to library people."""
from __future__ import annotations

import pytest

from photocure.db import models as M
from photocure.db.repository import quantize_embedding
from photocure.db.session import get_session
from photocure.people_actions import attach_faces_to_library, recluster_people

from .test_ui_api import env  # noqa: F401  (shared fixture)

A = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
A_CLOSE = [0.97, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
B = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _image(s, name="x.jpg", **kw):
    r = M.ImageRecord(path=f"/p/{name}", filename=name, infer_status="inferred", **kw)
    s.add(r)
    s.flush()
    return r


def _face(s, image_id, emb, person_id=None, ignored=False, sharp=1.0):
    f = M.FaceDetection(image_id=image_id, bbox_x1=0, bbox_y1=0, bbox_x2=1, bbox_y2=1, confidence=0.9,
                        x1=0.1, y1=0.1, x2=0.3, y2=0.3, person_id=person_id, ignored=ignored,
                        face_sharpness=sharp, embedding_json=quantize_embedding(emb))
    s.add(f)
    s.flush()
    return f


def _cluster(s, emb, count=1, stage="library", name=None):
    c = M.PersonCluster(name=name, face_count=count, stage=stage, centroid_json=quantize_embedding(emb))
    s.add(c)
    s.flush()
    return c


def test_attach_matches_a_known_library_person(session):
    lib = _cluster(session, A, count=3, name="Alice")
    img = _image(session, stage="library")
    f = _face(session, img.id, A_CLOSE)
    out = attach_faces_to_library(session, img.id)
    assert out == {"matched": 1, "created": 0}
    assert f.person_id == lib.id and lib.face_count == 4 and f.ignored is False


def test_attach_creates_a_new_library_person_for_an_unknown_face(session):
    _cluster(session, A, count=2, name="Alice")
    img = _image(session, stage="library")
    f = _face(session, img.id, B)
    out = attach_faces_to_library(session, img.id)
    assert out == {"matched": 0, "created": 1}
    new = session.get(M.PersonCluster, f.person_id)
    assert new.stage == "library" and new.face_count == 1 and new.name is None


def test_attach_never_puts_two_faces_of_one_photo_on_the_same_person(session):
    lib = _cluster(session, A, count=2, name="Alice")
    img = _image(session, stage="library")
    f1 = _face(session, img.id, A_CLOSE, sharp=2.0)
    f2 = _face(session, img.id, A, sharp=1.0)
    out = attach_faces_to_library(session, img.id)
    assert out == {"matched": 1, "created": 1}
    assert f1.person_id == lib.id and f2.person_id not in (None, lib.id)


def test_attach_takes_a_face_out_of_its_trash_cluster_and_drops_the_empty_one(session):
    trash = _cluster(session, A, count=1, stage="trash")
    img = _image(session, stage="trash", route_status="trashed", scan_status="missing")
    f = _face(session, img.id, A, person_id=trash.id)
    tid = trash.id
    attach_faces_to_library(session, img.id)
    assert f.person_id is not None and f.person_id != tid
    assert session.get(M.PersonCluster, tid) is None


def test_attach_re_attaches_a_face_that_was_ignored_while_trashed(session):
    img = _image(session, stage="trash", route_status="trashed", scan_status="missing")
    f = _face(session, img.id, A, ignored=True)
    attach_faces_to_library(session, img.id)
    assert f.ignored is False and f.person_id is not None


def test_trash_recluster_creates_trash_clusters_only(env):
    with get_session(env.db) as s:
        lib_img = _image(s, "lib.jpg", stage="library", route_status="executed")
        lib_c = _cluster(s, A, count=1, name="Alice")
        lib_f = _face(s, lib_img.id, A, person_id=lib_c.id)
        t1 = _image(s, "t1.jpg", stage="trash", route_status="trashed", scan_status="missing")
        t2 = _image(s, "t2.jpg", stage="trash", route_status="trashed", scan_status="missing")
        ft1 = _face(s, t1.id, B)
        ft2 = _face(s, t2.id, B)
        s.commit()
        lib_c_id, lib_f_id, ft1_id, ft2_id = lib_c.id, lib_f.id, ft1.id, ft2.id
    out = recluster_people(env.db, stage="trash")
    assert out["clusters_created"] == 1 and out["faces_clustered"] == 2
    with get_session(env.db) as s:
        c1, c2 = s.get(M.FaceDetection, ft1_id), s.get(M.FaceDetection, ft2_id)
        assert c1.person_id == c2.person_id is not None
        assert s.get(M.PersonCluster, c1.person_id).stage == "trash"
        lib = s.get(M.PersonCluster, lib_c_id)
        assert lib.stage == "library" and lib.face_count == 1 and lib.name == "Alice"
        assert s.get(M.FaceDetection, lib_f_id).person_id == lib_c_id


def test_recluster_endpoint_takes_a_stage(env):
    with get_session(env.db) as s:
        t = _image(s, "t.jpg", stage="trash", route_status="trashed", scan_status="missing")
        _face(s, t.id, B)
        s.commit()
    r = env.client.post("/api/people/recluster?stage=trash")
    assert r.status_code == 200 and r.json()["clusters_created"] == 1
    assert env.client.post("/api/people/recluster?stage=library").status_code == 422


def test_people_endpoint_new_stage_hides_trash_clusters(env):
    with get_session(env.db) as s:
        t = _image(s, "t.jpg", stage="trash", route_status="trashed", scan_status="missing")
        c = _cluster(s, B, count=1, stage="trash", name="Binned")
        _face(s, t.id, B, person_id=c.id)
        s.commit()
    new = {p["name"] for p in env.client.get("/api/people?stage=new").json()["items"]}
    trash = {p["name"] for p in env.client.get("/api/people?stage=trash").json()["items"]}
    assert "Binned" not in new and "Binned" in trash
