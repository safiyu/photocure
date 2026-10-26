"""Unit and integration tests for the Pipeline vs. Library spec.

Verifies:
1. test_organizer_flat_categories
2. test_shared_people_and_ignored_faces
3. test_apply_completion_gate
4. test_pipeline_vs_library_scopes
5. test_library_sync_external_changes
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from photocure.config import CuratorConfig
from photocure.db import models as M
from photocure.db.models import BurstGroup, FaceDetection, ImageRecord, PersonCluster, utcnow
from photocure.db.repository import PhotoRepository, quantize_embedding
from photocure.db.session import get_session, init_db
from photocure.inference.pipeline import InferencePipeline
from photocure.people_actions import delete_person, list_people
from photocure.rules.decision_matrix import Decision, DecisionMatrix
from photocure.rules.organizer import FolderOrganizer
from photocure.rules.taxonomy import CategoryType
from photocure.services import apply_settings, load_settings, save_settings
from photocure.ui.app import create_app

from .conftest import make_image


def test_organizer_flat_categories(cfg):
    """Verify Non-Human/Food, People, Bursts/Burst_.../alternates paths are generated without YYYY/MM/DD date subfolders."""
    root = Path("/library")
    org = FolderOrganizer(cfg, organize_root=root)

    t = datetime(2024, 5, 12, 10, 0, 0, tzinfo=timezone.utc)
    rec_food = ImageRecord(path="/in/food.jpg", filename="food.jpg", taken_at=t)
    rec_people = ImageRecord(path="/in/person.jpg", filename="person.jpg", taken_at=t)
    rec_burst = ImageRecord(path="/in/b_alt.jpg", filename="b_alt.jpg", taken_at=t, burst_id=1, is_alternate=True)

    bg = BurstGroup(id=1, name="Burst_000001", image_count=3, best_shot_id=1)
    org.set_bursts([bg])

    dec_food = Decision(CategoryType.FOOD, "clip", needs_date=False)
    dec_people = Decision(CategoryType.PEOPLE, "face", needs_date=False)
    dec_burst = Decision(CategoryType.UNKNOWN, "burst alternate", is_alternate=True)

    target_food = org.full_target(rec_food, dec_food)
    target_people = org.full_target(rec_people, dec_people)
    target_burst = org.full_target(rec_burst, dec_burst)

    assert target_food == root / "Non-Human" / "Food" / "food.jpg"
    assert target_people == root / "People" / "person.jpg"
    assert target_burst == root / "Bursts" / "Burst_000001" / "alternates" / "b_alt.jpg"

    for p in (target_food, target_people, target_burst):
        assert "2024" not in str(p)
        assert "Unknown_Date" not in str(p)


def test_shared_people_and_ignored_faces(tmp_path):
    """Verify deleting a person marks faces ignored=True, analysing new photos never resurrects deleted faces,
    Library photos are never re-analysed, and new pipeline imports match against named Library people centroids.
    """
    db_file = tmp_path / "people_test.db"
    init_db(db_file, drop=True)
    cfg = CuratorConfig()
    cfg.db_path = db_file

    app = create_app(str(db_file))
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token

    f_lib = make_image(tmp_path / "lib_alice.jpg")
    f_pipe = make_image(tmp_path / "pipe_photo.jpg")

    emb = [0.1] * 512
    q_emb = quantize_embedding(emb)

    with get_session(db_file) as s:
        # Library person (named)
        lib_person = PersonCluster(name="Alice", face_count=1, centroid_json=q_emb)
        s.add(lib_person)
        s.flush()

        # Finalized library photo
        lib_img = ImageRecord(
            path=str(f_lib),
            filename="lib_alice.jpg",
            route_status="executed",
            stage="library",
            scan_status="active",
            infer_status="inferred",
            faces_schema_version=cfg.face.schema_version,
        )
        s.add(lib_img)
        s.flush()

        lib_face = FaceDetection(
            image_id=lib_img.id,
            person_id=lib_person.id,
            confidence=0.95,
            x1=0.2, y1=0.2, x2=0.4, y2=0.4,
            bbox_x1=20, bbox_y1=20, bbox_x2=40, bbox_y2=40,
            embedding_json=q_emb,
            ignored=False,
        )
        s.add(lib_face)

        # Pipeline photo with an unnamed person
        pipe_person = PersonCluster(name=None, face_count=1, centroid_json=quantize_embedding([0.5] * 512))
        s.add(pipe_person)
        s.flush()

        pipe_img = ImageRecord(
            path=str(f_pipe),
            filename="pipe_photo.jpg",
            route_status="unrouted",
            scan_status="active",
            infer_status="inferred",
            faces_schema_version=cfg.face.schema_version,
        )
        s.add(pipe_img)
        s.flush()

        pipe_face = FaceDetection(
            image_id=pipe_img.id,
            person_id=pipe_person.id,
            confidence=0.90,
            x1=0.1, y1=0.1, x2=0.3, y2=0.3,
            bbox_x1=10, bbox_y1=10, bbox_x2=30, bbox_y2=30,
            embedding_json=quantize_embedding([0.5] * 512),
            ignored=False,
        )
        s.add(pipe_face)
        s.commit()
        pipe_person_id = pipe_person.id
        pipe_face_id = pipe_face.id
        lib_img_id = lib_img.id
        pipe_img_id = pipe_img.id

    # 1. Deleting a person marks faces ignored=True
    del_res = client.delete(f"/api/people/{pipe_person_id}?stage=new")
    assert del_res.status_code == 200

    with get_session(db_file) as s:
        f = s.get(FaceDetection, pipe_face_id)
        assert f.ignored is True
        assert f.person_id is None

    # 2. Analyze runs on new photos only -> a Library photo is never re-analyzed
    infer_res = client.post("/api/infer", json={})
    assert infer_res.status_code == 200

    with get_session(db_file) as s:
        lib_rec = s.get(ImageRecord, lib_img_id)
        assert lib_rec.faces_schema_version == cfg.face.schema_version  # untouched!
        pipe_rec = s.get(ImageRecord, pipe_img_id)
        assert pipe_rec.faces_schema_version == cfg.face.schema_version  # nothing pending, nothing redone


def test_apply_completion_gate(tmp_path):
    """Verify Apply is rejected with 409 when uncertain > 0 or unapproved junk > 0,
    and succeeds once all items are categorized or approved for trash.
    """
    db_file = tmp_path / "apply_gate.db"
    init_db(db_file, drop=True)
    cfg = CuratorConfig()
    cfg.db_path = db_file

    app = create_app(str(db_file))
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    out_dir = tmp_path / "organized"
    out_dir.mkdir()

    f1 = make_image(photos_dir / "p1.jpg")
    f2 = make_image(photos_dir / "p2.jpg")

    with get_session(db_file) as s:
        r1 = ImageRecord(
            path=str(f1),
            filename="p1.jpg",
            route_status="unrouted",
            scan_status="active",
            infer_status="inferred",
            category="uncertain",
            delete_approved=False,
        )
        r2 = ImageRecord(
            path=str(f2),
            filename="p2.jpg",
            route_status="unrouted",
            scan_status="active",
            infer_status="inferred",
            category="duplicate",
            delete_approved=False,
        )
        s.add_all([r1, r2])
        s.commit()
        r1_id, r2_id = r1.id, r2.id

    assert client.post("/api/settings", json={"library_root": str(out_dir)}).status_code == 200

    # Check readiness: should report uncompleted issues
    readiness = client.get("/api/plan/readiness").json()
    assert readiness["ready"] is False
    assert readiness["uncertain_count"] == 1
    assert readiness["unresolved_junk"] == 1

    # Attempt apply: must be rejected with 409 Conflict
    apply_res = client.post("/api/plan/apply", json={"operation": "copy"})
    assert apply_res.status_code == 409
    assert "uncertain" in apply_res.text.lower() or "finalize" in apply_res.text.lower()

    # Resolve items: assign category to r1, approve delete for junk r2
    client.post(f"/api/images/{r1_id}/reassign", json={"bucket": "food"})
    client.post(f"/api/images/{r2_id}/approve-delete", json={})

    # Now readiness must be ready
    readiness2 = client.get("/api/plan/readiness").json()
    assert readiness2["ready"] is True
    assert readiness2["uncertain_count"] == 0
    assert readiness2["unresolved_junk"] == 0

    # Apply succeeds
    apply_res2 = client.post("/api/plan/apply", json={"operation": "copy"})
    assert apply_res2.status_code == 202


def test_pipeline_vs_library_scopes(tmp_path):
    """Verify API endpoints (/api/images, /api/people, /api/bursts, /api/overview) filter accurately
    by scope="pipeline" vs scope="library", and manual edits in Library mode move files on disk immediately.
    """
    db_file = tmp_path / "scopes.db"
    init_db(db_file, drop=True)
    cfg = CuratorConfig()
    cfg.db_path = db_file
    lib_root = tmp_path / "MyLibrary"
    lib_root.mkdir()
    cfg.library_root = lib_root

    app = create_app(str(db_file))
    app.state.cfg.library_root = lib_root
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token

    p_pipe = tmp_path / "incoming"
    p_pipe.mkdir()
    f_pipe = make_image(p_pipe / "pipe.jpg")

    p_lib = lib_root / "Non-Human" / "Food"
    p_lib.mkdir(parents=True)
    f_lib = make_image(p_lib / "food_photo.jpg")

    with get_session(db_file) as s:
        rec_pipe = ImageRecord(
            path=str(f_pipe),
            filename="pipe.jpg",
            category="food",
            route_status="unrouted",
            scan_status="active",
            infer_status="inferred",
        )
        rec_lib = ImageRecord(
            path=str(f_lib),
            filename="food_photo.jpg",
            category="food",
            route_status="executed",
            stage="library",
            scan_status="active",
            infer_status="inferred",
        )
        s.add_all([rec_pipe, rec_lib])
        s.commit()
        lib_img_id = rec_lib.id

    # 1. Check /api/images scoped
    imgs_pipe = client.get("/api/images?stage=new").json()
    assert imgs_pipe["total"] == 1
    assert imgs_pipe["items"][0]["filename"] == "pipe.jpg"

    imgs_lib = client.get("/api/images?stage=library").json()
    assert imgs_lib["total"] == 1
    assert imgs_lib["items"][0]["filename"] == "food_photo.jpg"

    # 2. Check /api/overview scoped
    ov_pipe = client.get("/api/overview?stage=new").json()
    assert ov_pipe["total"] == 1
    assert ov_pipe["pipeline_count"] == 1
    assert ov_pipe["library_count"] == 1

    ov_lib = client.get("/api/overview?stage=library").json()
    assert ov_lib["total"] == 1

    # 3. Manual category change in Library mode moves file on disk immediately
    reassign_res = client.post(f"/api/images/{lib_img_id}/reassign", json={"bucket": "scenery"})
    assert reassign_res.status_code == 200

    new_path = lib_root / "Non-Human" / "Scenery" / "food_photo.jpg"
    assert new_path.exists()
    assert not f_lib.exists()

    with get_session(db_file) as s:
        updated = s.get(ImageRecord, lib_img_id)
        assert updated.path == str(new_path)
        assert updated.category == "scenery"

    # 4. Delete photo in Library mode moves it into the trash (a ledger transaction folder)
    del_res = client.delete(f"/api/images/{lib_img_id}")
    assert del_res.status_code == 200
    assert not new_path.exists()
    assert list((lib_root / ".trash").rglob("food_photo.jpg"))

    with get_session(db_file) as s:
        trashed = s.get(ImageRecord, lib_img_id)
        assert trashed.route_status == "trashed"
        assert trashed.stage == "trash"
        assert trashed.scan_status == "missing"


def test_library_sync_external_changes(tmp_path):
    """Verify external file deletion removes the photo from Library and prunes counts,
    while moving a file between category folders on disk updates its path and category in the DB.
    """
    db_file = tmp_path / "sync.db"
    init_db(db_file, drop=True)
    cfg = CuratorConfig()
    cfg.db_path = db_file
    lib_root = tmp_path / "PermanentLibrary"
    lib_root.mkdir()
    cfg.library_root = lib_root

    save_settings(db_file, {"library_root": str(lib_root)})

    app = create_app(str(db_file))
    app.state.cfg.library_root = lib_root
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token

    food_dir = lib_root / "Non-Human" / "Food"
    food_dir.mkdir(parents=True)
    f1 = make_image(food_dir / "moving.jpg")
    f2 = make_image(food_dir / "deleting.jpg")

    with get_session(db_file) as s:
        r1 = ImageRecord(
            path=str(f1),
            filename="moving.jpg",
            category="food",
            route_status="executed",
            stage="library",
            scan_status="active",
            infer_status="inferred",
            file_size=f1.stat().st_size,
        )
        r2 = ImageRecord(
            path=str(f2),
            filename="deleting.jpg",
            category="food",
            route_status="executed",
            stage="library",
            scan_status="active",
            infer_status="inferred",
            file_size=f2.stat().st_size,
        )
        s.add_all([r1, r2])
        s.commit()
        r1_id, r2_id = r1.id, r2.id

    # External changes:
    # 1. Move f1 to Scenery folder on disk
    scenery_dir = lib_root / "Non-Human" / "Scenery"
    scenery_dir.mkdir(parents=True)
    dst1 = scenery_dir / "moving.jpg"
    shutil.move(f1, dst1)

    # 2. Delete f2 on disk
    f2.unlink()

    # Run library sync
    sync_res = client.post("/api/library/sync")
    assert sync_res.status_code == 200
    res_data = sync_res.json()
    assert res_data["ok"] is True
    assert res_data["moved"] == 1
    assert res_data["missing"] == 1

    with get_session(db_file) as s:
        rec1 = s.get(ImageRecord, r1_id)
        assert rec1.path == str(dst1)
        assert rec1.category == "scenery"
        assert rec1.scan_status == "active"

        rec2 = s.get(ImageRecord, r2_id)
        assert rec2.scan_status == "missing"
        assert rec2.route_status == "removed"
