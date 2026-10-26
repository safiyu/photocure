"""Fixes from the final review of the scanner, analysis and people code."""
from __future__ import annotations

import types
from pathlib import Path

import pytest

import photocure.scanner.file_scanner as fs
from photocure.clustering.face_identity import ClusterCentroid
from photocure.db import models as M
from photocure.db.repository import PhotoRepository, quantize_embedding
from photocure.db.session import get_session, init_db
from photocure.inference.pipeline import InferencePipeline
from photocure.people_actions import recluster_people
from photocure.scanner.file_scanner import FileScanner
from photocure.scanner.xmp import parse_xmp

from .conftest import make_image
from .test_scan_skip_library import _apply, _count, _rescan, world  # noqa: F401  (shared fixture)


def test_pruning_a_folder_never_marks_library_photos_missing(cfg, tmp_db, tmp_path):
    root = (tmp_path / "Photos").resolve()
    lib = root / "Library"
    make_image(root / "new1.jpg", rng_seed=1)
    kept = make_image(lib / "People" / "kept.jpg", rng_seed=2)
    cfg.library_root = lib
    init_db(tmp_db)
    with get_session(tmp_db) as s:
        s.add(M.ImageRecord(path=str(kept.resolve()), filename="kept.jpg", stage="library", route_status="executed",
                            scan_status="scanned", infer_status="inferred"))
        s.commit()
    FileScanner(cfg).scan_directory(root, db_path=tmp_db, show_progress=False, prune_missing=True)
    with get_session(tmp_db) as s:
        row = s.exec(M.ImageRecord.__table__.select().where(M.ImageRecord.filename == "kept.jpg")).first()
        assert row.scan_status == "scanned"


def test_rescanning_a_copied_source_does_not_read_every_file_again(world, monkeypatch):  # noqa: F811
    w = world
    _apply(w, "copy")
    calls = []
    real = fs._safe_sha256
    monkeypatch.setattr(fs, "_safe_sha256", lambda p: calls.append(p) or real(p))
    res = _rescan(w)
    assert res.skipped_library == 4 and res.new == 0
    assert calls == []  # known by origin path and size, so nothing is hashed
    assert _count(w) == 4


_XMP = (b'<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?><x:xmpmeta xmlns:x="adobe:ns:meta/">'
        b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        b'<rdf:Description xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="%s"/></rdf:RDF>'
        b'</x:xmpmeta><?xpacket end="w"?>')


@pytest.mark.parametrize("rating", [b"inf", b"1e999", b"nan", b"-inf"])
def test_a_hostile_xmp_rating_does_not_abort_the_scan(rating):
    assert parse_xmp(_XMP % rating).rating is None


def test_a_new_copy_never_points_at_a_row_that_is_itself_a_duplicate(session):
    original = M.ImageRecord(path="/p/a.jpg", filename="a.jpg", sha256="S", is_duplicate=False)
    chain = M.ImageRecord(path="/p/b.jpg", filename="b.jpg", sha256="S", is_duplicate=True)
    session.add_all([chain, original])
    session.flush()
    rec = M.ImageRecord(path="/p/c.jpg", filename="c.jpg", sha256="S")
    hit = FileScanner._find_db_original("S", PhotoRepository(session), rec)
    assert hit is not None and hit.id == original.id


EMB = [1.0, 0.0, 0.0, 0.0]


def test_recluster_keeps_a_named_person_whose_centroid_is_empty(tmp_db):
    init_db(tmp_db, drop=True)
    with get_session(tmp_db) as s:
        lib_img = M.ImageRecord(path="/p/lib.jpg", filename="lib.jpg", stage="library", route_status="executed",
                                infer_status="inferred")
        new_img = M.ImageRecord(path="/p/new.jpg", filename="new.jpg", stage="new", infer_status="inferred")
        s.add_all([lib_img, new_img])
        s.flush()
        grandma = M.PersonCluster(name="Grandma", face_count=1, stage="library", centroid_json="[]")
        s.add(grandma)
        s.flush()
        s.add(M.FaceDetection(image_id=lib_img.id, person_id=grandma.id, bbox_x1=0, bbox_y1=0, bbox_x2=1,
                              bbox_y2=1, confidence=0.9, x1=0.1, y1=0.1, x2=0.2, y2=0.2))
        s.add(M.FaceDetection(image_id=new_img.id, bbox_x1=0, bbox_y1=0, bbox_x2=1, bbox_y2=1, confidence=0.9,
                              x1=0.1, y1=0.1, x2=0.2, y2=0.2, embedding_json=quantize_embedding(EMB)))
        s.commit()
        gid = grandma.id
    recluster_people(tmp_db, stage="new")
    with get_session(tmp_db) as s:
        assert s.get(M.PersonCluster, gid) is not None and s.get(M.PersonCluster, gid).name == "Grandma"


def test_one_person_never_gets_two_faces_of_the_same_photo_through_the_remap_path(cfg, session):
    pipeline = InferencePipeline(cfg)
    repo = PhotoRepository(session)
    img = M.ImageRecord(path="/p/x.jpg", filename="x.jpg", infer_status="inferred")
    session.add(img)
    session.flush()
    p1 = repo.create_person_cluster([1.0, 0.0])
    # the photo already had a face of person 1, which is what a continuing face remaps to
    session.add(M.FaceDetection(image_id=img.id, person_id=p1.id, bbox_x1=0, bbox_y1=0, bbox_x2=1, bbox_y2=1,
                                confidence=0.9, x1=0.1, y1=0.1, x2=0.2, y2=0.2,
                                embedding_json=quantize_embedding([1.0, 0.0])))
    session.flush()
    centroids = {p1.id: ClusterCentroid(p1.id, [1.0, 0.0], 1)}
    # face 0 joins person 1 by centroid only (cosine 0.6); face 1 is the true continuation (cosine 0.99)
    faces = [types.SimpleNamespace(embedding=[0.6, 0.8]), types.SimpleNamespace(embedding=[0.99, 0.14])]
    out = pipeline._assign_people(repo, centroids, img.id, faces)
    assert out[1] == p1.id  # the continuing face keeps its person
    assert out[0] is not None and out[0] != p1.id
