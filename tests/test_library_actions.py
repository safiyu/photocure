"""Manual edits in Library: category moves and deletes go through the ledger."""
from __future__ import annotations

from pathlib import Path

import pytest

from photocure import library_actions
from photocure.db import models as M
from photocure.db.session import get_session
from photocure.engine.rollback import RollbackEngine

from .test_ui_api import _seed_face, _seed_person, env, wait_job  # noqa: F401  (shared fixture)


@pytest.fixture()
def lib(env):
    """All four photos applied into the library root (moved, nothing trashed)."""
    r = env.client.post("/api/plan/apply", json={"operation": "move", "trash": False})
    assert r.status_code == 202
    assert wait_job(env.client, r.json()["job_id"])["status"] == "done"
    items = env.client.get("/api/images?stage=library&limit=50").json()["items"]
    assert len(items) == 4
    return env


def _row(env, iid):
    with get_session(env.db) as s:
        r = s.get(M.ImageRecord, iid)
        return {c: getattr(r, c) for c in ("path", "stage", "scan_status", "route_status",
                                           "manual_bucket", "category", "delete_approved")}


def _first(env):
    return env.client.get("/api/images?stage=library&limit=50").json()["items"][0]["id"]


def test_move_to_category_moves_file_and_writes_ledger(lib):
    iid = _first(lib)
    before = Path(_row(lib, iid)["path"])
    out = library_actions.move_to_category(lib.db, lib.app.state.cfg, iid, "scenery")
    assert out["moved"] is True
    after = Path(_row(lib, iid)["path"])
    assert after == lib.out.resolve() / "Non-Human" / "Scenery" / before.name
    assert after.exists() and not before.exists()
    with get_session(lib.db) as s:
        e = s.query(M.ActionLedger).filter_by(transaction_id=out["transaction_id"]).one()
        assert e.status == "committed" and e.operation_type == "move"
    rb = RollbackEngine(lib.app.state.cfg, db_path=lib.db).rollback_transaction(out["transaction_id"])
    assert rb.failures == 0 and before.exists()


def test_move_to_category_creates_the_missing_folder(lib):
    iid = _first(lib)
    target_dir = lib.out.resolve() / "Utility_Reference" / "Documents"
    assert not target_dir.exists()
    library_actions.move_to_category(lib.db, lib.app.state.cfg, iid, "document")
    assert target_dir.is_dir() and any(target_dir.iterdir())


def test_move_to_same_category_is_a_no_op(lib):
    iid = _first(lib)
    cat = _row(lib, iid)["category"]
    out = library_actions.move_to_category(lib.db, lib.app.state.cfg, iid, cat)
    assert out["moved"] is False


def test_move_to_category_missing_file_is_409_and_bucket_unchanged(lib):
    iid = _first(lib)
    Path(_row(lib, iid)["path"]).unlink()
    r = lib.client.post(f"/api/images/{iid}/reassign", json={"bucket": "food"})
    assert r.status_code == 409
    assert _row(lib, iid)["manual_bucket"] is None


def test_reassign_in_library_moves_the_file_and_sets_the_category(lib):
    iid = _first(lib)
    r = lib.client.post(f"/api/images/{iid}/reassign", json={"bucket": "scenery"})
    assert r.status_code == 200
    row = _row(lib, iid)
    assert row["category"] == "scenery" and row["manual_bucket"] == "scenery"
    assert Path(row["path"]).parent == lib.out.resolve() / "Non-Human" / "Scenery"


def test_delete_in_library_moves_into_a_txn_trash_folder_and_detaches_faces(lib):
    iid = _first(lib)
    pid = _seed_person(lib, name=None, face_count=1)
    fid = _seed_face(lib, iid, person_id=pid)
    r = lib.client.delete(f"/api/images/{iid}")
    assert r.status_code == 200
    row = _row(lib, iid)
    path = Path(row["path"])
    assert row["stage"] == "trash" and row["scan_status"] == "missing" and row["route_status"] == "trashed"
    assert path.exists()
    trash = lib.out.resolve() / ".trash"
    assert trash in path.parents and path.parent.parent != trash  # .trash/<txn>/<category>/name
    with get_session(lib.db) as s:
        f = s.get(M.FaceDetection, fid)
        assert f.ignored is True and f.person_id is None
        assert s.get(M.PersonCluster, pid) is None  # unnamed and now empty


def test_delete_of_a_new_photo_only_marks_it(env):
    iid = env.client.get("/api/images").json()["items"][0]["id"]
    path = Path(_row(env, iid)["path"])
    r = env.client.delete(f"/api/images/{iid}")
    assert r.status_code == 200
    row = _row(env, iid)
    assert row["delete_approved"] is True and row["stage"] == "new"
    assert path.exists() and Path(row["path"]) == path


def test_library_sync_never_pulls_a_photo_out_of_library(lib):
    iid = _first(lib)
    Path(_row(lib, iid)["path"]).unlink()
    r = lib.client.post("/api/library/sync")
    assert r.status_code == 200
    j = r.json()
    assert iid in j["missing_ids"]
    row = _row(lib, iid)
    assert row["stage"] == "library" and row["scan_status"] == "missing"


def test_best_shot_in_library_keeps_alternates_where_they_are(lib):
    ids = [i["id"] for i in lib.client.get("/api/images?stage=library&limit=50").json()["items"]][:2]
    with get_session(lib.db) as s:
        bg = M.BurstGroup(name="Burst_lib", image_count=2, best_shot_id=ids[0])
        s.add(bg)
        s.flush()
        for n, iid in enumerate(ids):
            r = s.get(M.ImageRecord, iid)
            r.burst_id, r.is_best_shot, r.is_alternate = bg.id, n == 0, n != 0
        s.commit()
        bid = bg.id
    paths_before = {i: _row(lib, i)["path"] for i in ids}
    r = lib.client.post(f"/api/bursts/{bid}/best-shot", json={"image_id": ids[1]})
    assert r.status_code == 200
    assert not (lib.out.resolve() / "Bursts").exists()
    assert Path(_row(lib, ids[0])["path"]).exists() and Path(_row(lib, ids[1])["path"]).exists()
    assert _row(lib, ids[0])["path"] == paths_before[ids[0]]
