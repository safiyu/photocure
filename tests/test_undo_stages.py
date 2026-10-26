"""Undo puts a photo back in the stage it came from, not always in New."""
from __future__ import annotations

from pathlib import Path

import pytest
from sqlmodel import select

from photocure import library_actions, trash_actions
from photocure.db import models as M
from photocure.db.session import get_session
from photocure.engine.rollback import RollbackEngine

from .test_ui_api import env, wait_job  # noqa: F401  (shared fixture)


def _row(env, iid):
    with get_session(env.db) as s:
        r = s.get(M.ImageRecord, iid)
        return {c: getattr(r, c) for c in ("path", "stage", "scan_status", "route_status", "origin_path",
                                           "delete_approved", "manual_bucket", "category")}


def _undo(env, txn):
    return RollbackEngine(env.app.state.cfg, db_path=env.db).rollback_transaction(txn)


@pytest.fixture()
def lib(env):
    """Four photos moved into the library."""
    r = env.client.post("/api/plan/apply", json={"operation": "move", "trash": False})
    assert wait_job(env.client, r.json()["job_id"])["status"] == "done"
    ids = sorted(i["id"] for i in env.client.get("/api/images?stage=library&limit=50").json()["items"])
    assert len(ids) == 4
    env.ids = ids
    return env


def test_undoing_a_manual_delete_puts_the_photo_back_in_the_library_unmarked(lib):
    iid = lib.ids[0]
    before = _row(lib, iid)
    out = library_actions.trash_image(lib.db, lib.app.state.cfg, iid)
    assert _row(lib, iid)["stage"] == "trash"
    rb = _undo(lib, out["transaction_id"])
    assert rb.failures == 0
    after = _row(lib, iid)
    assert after["stage"] == "library" and after["scan_status"] == "scanned"
    assert after["route_status"] == "executed" and after["delete_approved"] is False
    assert after["origin_path"] == before["origin_path"] and after["origin_path"]
    assert after["path"] == before["path"] and Path(after["path"]).exists()


def test_undoing_a_category_move_keeps_the_photo_a_library_photo(lib):
    iid = lib.ids[0]
    before = _row(lib, iid)
    out = library_actions.move_to_category(lib.db, lib.app.state.cfg, iid, "scenery")
    _undo(lib, out["transaction_id"])
    after = _row(lib, iid)
    assert after["stage"] == "library" and after["path"] == before["path"]
    assert after["origin_path"] == before["origin_path"]
    assert Path(after["path"]).is_relative_to(lib.out.resolve())


def test_undoing_a_restore_sends_the_photo_back_to_the_trash_and_it_stays_visible(lib):
    iid = lib.ids[0]
    library_actions.trash_image(lib.db, lib.app.state.cfg, iid)
    out = trash_actions.restore(lib.db, lib.app.state.cfg, [iid])
    assert _row(lib, iid)["stage"] == "library"
    _undo(lib, out["transaction_id"])
    row = _row(lib, iid)
    assert (row["stage"], row["scan_status"], row["route_status"]) == ("trash", "missing", "trashed")
    assert Path(row["path"]).exists() and (lib.out.resolve() / ".trash") in Path(row["path"]).parents
    assert iid in {i["id"] for i in lib.client.get("/api/images?stage=trash&limit=50").json()["items"]}
    # Emptying the trash now deletes it the proper way, with the row marked purged
    trash_actions.empty_trash(lib.db, lib.app.state.cfg)
    assert _row(lib, iid)["route_status"] == "purged" and not Path(row["path"]).exists()


def test_undoing_a_return_to_origin_sends_the_photo_back_to_the_trash(lib):
    iid = lib.ids[0]
    library_actions.trash_image(lib.db, lib.app.state.cfg, iid)
    out = trash_actions.return_to_origin(lib.db, lib.app.state.cfg, [iid])
    assert out["returned"] == 1 and _row(lib, iid)["stage"] == "new"
    _undo(lib, out["transaction_id"])
    row = _row(lib, iid)
    assert (row["stage"], row["scan_status"]) == ("trash", "missing")
    assert Path(row["path"]).exists() and row["origin_path"]


def test_undoing_an_apply_still_returns_photos_to_new(env):
    r = env.client.post("/api/plan/apply", json={"operation": "move", "trash": False})
    job = wait_job(env.client, r.json()["job_id"])
    iid = sorted(i["id"] for i in env.client.get("/api/images?stage=library&limit=50").json()["items"])[0]
    assert _undo(env, job["result"]["transaction_id"]).failures == 0
    row = _row(env, iid)
    assert (row["stage"], row["route_status"], row["origin_path"]) == ("new", "unrouted", None)
    assert Path(row["path"]).exists() and not Path(row["path"]).is_relative_to(env.out.resolve())


def test_ledger_rows_from_before_the_change_still_undo_to_new(env):
    r = env.client.post("/api/plan/apply", json={"operation": "move", "trash": False})
    txn = wait_job(env.client, r.json()["job_id"])["result"]["transaction_id"]
    with get_session(env.db) as s:
        for e in s.exec(select(M.ActionLedger).where(M.ActionLedger.transaction_id == txn)).all():
            e.prev_stage = ""  # an old row carries no previous state
        s.commit()
    assert _undo(env, txn).failures == 0
    with get_session(env.db) as s:
        assert {r.stage for r in s.exec(select(M.ImageRecord)).all()} == {"new"}
