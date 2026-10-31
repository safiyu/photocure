"""Trash stage actions: restore, return to origin, delete permanently, empty trash."""
from __future__ import annotations

from pathlib import Path

import pytest
from sqlmodel import select

from photocure import library_actions, trash_actions
from photocure.db import models as M
from photocure.db.session import get_session

from .test_ui_api import env, wait_job  # noqa: F401  (shared fixture)


def _apply(env, op="move", trash=True):
    r = env.client.post("/api/plan/apply", json={"operation": op, "trash": trash})
    assert r.status_code == 202
    assert wait_job(env.client, r.json()["job_id"])["status"] == "done"


def _row(env, iid):
    with get_session(env.db) as s:
        r = s.get(M.ImageRecord, iid)
        return {c: getattr(r, c) for c in ("path", "stage", "scan_status", "route_status", "manual_bucket",
                                           "category", "delete_approved", "infer_status", "origin_path")}


def _mark_delete(env, iid):
    with get_session(env.db) as s:
        s.get(M.ImageRecord, iid).delete_approved = True
        s.commit()


def _ids(env):
    return sorted(i["id"] for i in env.client.get("/api/images?limit=50").json()["items"])


@pytest.fixture()
def trashed(env):
    """Four photos applied by move; the first one was approved for delete and is in the trash."""
    ids = _ids(env)
    _mark_delete(env, ids[0])
    _apply(env, "move", trash=True)
    assert _row(env, ids[0])["stage"] == "trash"
    return env, ids


def _cfg(env):
    return env.app.state.cfg


def test_restore_places_the_photo_by_its_current_category(trashed):
    env, ids = trashed
    with get_session(env.db) as s:
        s.get(M.ImageRecord, ids[0]).manual_bucket = "scenery"
        s.commit()
    out = trash_actions.restore(env.db, _cfg(env), [ids[0]])
    assert out["restored"] == 1 and out["skipped"] == 0
    row = _row(env, ids[0])
    assert Path(row["path"]) == env.out.resolve() / "Non-Human" / "Scenery" / Path(row["path"]).name
    assert Path(row["path"]).exists()
    assert (row["stage"], row["scan_status"], row["route_status"], row["delete_approved"]) == (
        "library", "scanned", "executed", False)
    with get_session(env.db) as s:
        e = s.exec(select(M.ActionLedger).where(M.ActionLedger.transaction_id == out["transaction_id"])).one()
        assert e.status == "committed"


def test_restore_respects_the_collision_policy_and_never_overwrites(trashed):
    env, ids = trashed
    with get_session(env.db) as s:
        rec = s.get(M.ImageRecord, ids[0])
        rec.manual_bucket = "scenery"
        name = Path(rec.path).name
        s.commit()
    target = env.out.resolve() / "Non-Human" / "Scenery" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"someone else's file")
    trash_actions.restore(env.db, _cfg(env), [ids[0]])
    assert target.read_bytes() == b"someone else's file"
    assert Path(_row(env, ids[0])["path"]) != target
    assert Path(_row(env, ids[0])["path"]).exists()


def test_restore_skips_a_file_that_is_gone_and_says_so(trashed):
    env, ids = trashed
    Path(_row(env, ids[0])["path"]).unlink()
    out = trash_actions.restore(env.db, _cfg(env), [ids[0]])
    assert out["restored"] == 0 and out["skipped"] == 1
    assert out["skipped_reasons"] == {"file_missing": 1}
    assert _row(env, ids[0])["stage"] == "trash"


def test_return_to_origin_moves_the_file_back_and_resets_the_photo(trashed):
    env, ids = trashed
    origin = Path(_row(env, ids[0])["origin_path"])
    assert not origin.exists()
    out = trash_actions.return_to_origin(env.db, _cfg(env), [ids[0]])
    assert out["returned"] == 1 and out["skipped"] == 0
    row = _row(env, ids[0])
    assert Path(row["path"]) == origin and origin.exists()
    assert (row["stage"], row["route_status"], row["infer_status"], row["origin_path"]) == (
        "new", "unrouted", "pending", None)
    assert row["scan_status"] == "scanned" and row["delete_approved"] is False


def test_return_skips_an_origin_that_is_occupied(env):
    ids = _ids(env)
    _mark_delete(env, ids[0])
    _apply(env, "copy", trash=True)  # the original stays where it was
    origin = Path(_row(env, ids[0])["origin_path"])
    trash_path = Path(_row(env, ids[0])["path"])
    assert origin.exists() and trash_path.exists()
    out = trash_actions.return_to_origin(env.db, _cfg(env), [ids[0]])
    assert out["returned"] == 0 and out["skipped_reasons"] == {"origin_occupied": 1}
    assert trash_path.exists() and _row(env, ids[0])["stage"] == "trash"


def test_return_skips_a_missing_origin_folder(trashed):
    env, ids = trashed
    origin = Path(_row(env, ids[0])["origin_path"])
    with get_session(env.db) as s:
        s.get(M.ImageRecord, ids[0]).origin_path = str(origin.parent / "gone_folder" / origin.name)
        s.commit()
    out = trash_actions.return_to_origin(env.db, _cfg(env), [ids[0]])
    assert out["returned"] == 0 and out["skipped_reasons"] == {"origin_missing": 1}


def test_return_all_when_ids_is_none(trashed):
    env, ids = trashed
    library_actions.trash_image(env.db, _cfg(env), ids[1])
    out = trash_actions.return_to_origin(env.db, _cfg(env), None)
    assert out["returned"] == 2
    assert {_row(env, i)["stage"] for i in ids[:2]} == {"new"}


def test_delete_permanently_unlinks_and_marks_purged(trashed):
    env, ids = trashed
    path = Path(_row(env, ids[0])["path"])
    out = trash_actions.delete_permanently(env.db, _cfg(env), [ids[0]])
    assert out["deleted"] == 1 and out["bytes"] > 0
    assert not path.exists()
    row = _row(env, ids[0])
    assert (row["route_status"], row["scan_status"]) == ("purged", "missing")
    with get_session(env.db) as s:
        assert s.exec(select(M.ActionLedger).where(M.ActionLedger.image_id == ids[0],
                                                   M.ActionLedger.status == "purged")).all()
    trash_root = env.out.resolve() / ".trash"
    assert not any(p.is_file() for p in trash_root.rglob("*")) if trash_root.exists() else True


def test_empty_trash_clears_every_row_and_folder(trashed):
    env, ids = trashed
    library_actions.trash_image(env.db, _cfg(env), ids[1])
    trash_root = env.out.resolve() / ".trash"
    (trash_root / "stray.txt").write_text("left behind")
    out = trash_actions.empty_trash(env.db, _cfg(env))
    assert out["deleted"] == 2
    assert not trash_root.exists()
    assert {_row(env, i)["route_status"] for i in ids[:2]} == {"purged"}


def test_trash_endpoints_409_when_busy(trashed):
    from photocure.ui import app as ui_app
    env, ids = trashed
    assert ui_app._BUSY.acquire(blocking=False)
    try:
        for path, body in [("/api/trash/restore", {"ids": [ids[0]]}), ("/api/trash/return", {"ids": [ids[0]]}),
                           ("/api/trash/delete", {"ids": [ids[0]]}), ("/api/trash/empty", {})]:
            assert env.client.post(path, json=body).status_code == 409, path
    finally:
        ui_app._BUSY.release()
    assert _row(env, ids[0])["stage"] == "trash"


def test_trash_endpoints_run_as_jobs(trashed):
    env, ids = trashed
    r = env.client.post("/api/trash/restore", json={"ids": [ids[0]]})
    assert r.status_code == 202
    j = wait_job(env.client, r.json()["job_id"])
    assert j["status"] == "done" and j["result"]["restored"] == 1
    assert _row(env, ids[0])["stage"] == "library"
    library_actions.trash_image(env.db, _cfg(env), ids[0])
    r = env.client.post("/api/trash/return", json={"ids": None})
    assert wait_job(env.client, r.json()["job_id"])["result"]["returned"] == 1
    library_actions.trash_image(env.db, _cfg(env), ids[1])
    r = env.client.post("/api/trash/empty", json={})
    assert wait_job(env.client, r.json()["job_id"])["result"]["deleted"] == 1


def test_finalize_deletes_moves_marked_to_trash(env):
    ids = _ids(env)
    # Mark two photos for delete in stage 'new'
    _mark_delete(env, ids[0])
    _mark_delete(env, ids[1])
    assert _row(env, ids[0])["stage"] == "new"
    assert _row(env, ids[1])["stage"] == "new"

    out = trash_actions.finalize_deletes(env.db, _cfg(env), stage="new")
    assert out["ok"] is True
    assert out["trashed"] == 2
    assert out["transaction_id"] != ""
    assert _row(env, ids[0])["stage"] == "trash"
    assert _row(env, ids[1])["stage"] == "trash"
    assert _row(env, ids[2])["stage"] == "new"


def test_api_finalize_endpoint(env):
    ids = _ids(env)
    _mark_delete(env, ids[2])
    res = env.client.post("/api/finalize", json={"stage": "new", "section": "categories"})
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["trashed"] == 1
    assert _row(env, ids[2])["stage"] == "trash"

