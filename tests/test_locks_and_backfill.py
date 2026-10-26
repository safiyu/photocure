"""File actions wait for each other, trash deletion only touches what it owns, and old data backfills sanely."""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from photocure import library_actions, library_migration, trash_actions
from photocure.db import models as M
from photocure.db.session import SCHEMA_VERSION, get_session, init_db
from photocure.engine.locks import LEDGER_WRITING
from photocure.ui import app as ui_app
from photocure.ui.app import create_app

from .test_ui_api import env, wait_job  # noqa: F401  (shared fixture)


@pytest.fixture()
def lib(env):
    r = env.client.post("/api/plan/apply", json={"operation": "move", "trash": False})
    assert wait_job(env.client, r.json()["job_id"])["status"] == "done"
    env.ids = sorted(i["id"] for i in env.client.get("/api/images?stage=library&limit=50").json()["items"])
    return env


def _row(env, iid):
    with get_session(env.db) as s:
        r = s.get(M.ImageRecord, iid)
        return {c: getattr(r, c) for c in ("path", "stage", "scan_status", "route_status", "manual_bucket")}


def test_library_edits_are_refused_while_a_job_runs(lib):
    iid = lib.ids[0]
    before = _row(lib, iid)
    assert ui_app._BUSY.acquire(blocking=False)
    try:
        assert lib.client.delete(f"/api/images/{iid}").status_code == 409
        assert lib.client.post(f"/api/images/{iid}/reassign", json={"bucket": "scenery"}).status_code == 409
        assert lib.client.post("/api/library/sync").status_code == 409
    finally:
        ui_app._BUSY.release()
    assert _row(lib, iid) == before and Path(before["path"]).exists()


def test_a_new_photo_can_still_be_edited_while_a_job_runs(env):
    iid = env.client.get("/api/images").json()["items"][0]["id"]
    assert ui_app._BUSY.acquire(blocking=False)
    try:
        assert env.client.post(f"/api/images/{iid}/reassign", json={"bucket": "food"}).status_code == 200
    finally:
        ui_app._BUSY.release()


def test_the_library_move_holds_the_ledger_lock_for_the_whole_copy(lib, monkeypatch):
    seen = []
    real = library_migration._copy_verified

    def spy(src, dst):
        seen.append(LEDGER_WRITING.locked())
        return real(src, dst)

    monkeypatch.setattr(library_migration, "_copy_verified", spy)
    out = library_migration.migrate_library(lib.db, lib.app.state.cfg, lib.tmp / "Moved")
    assert out["ok"] and seen and all(seen)
    assert not LEDGER_WRITING.locked()


def test_empty_trash_waits_for_other_file_actions(lib):
    library_actions.trash_image(lib.db, lib.app.state.cfg, lib.ids[0])
    done = threading.Event()
    LEDGER_WRITING.acquire()
    try:
        t = threading.Thread(target=lambda: (trash_actions.empty_trash(lib.db, lib.app.state.cfg), done.set()))
        t.start()
        time.sleep(0.4)
        assert not done.is_set()
    finally:
        LEDGER_WRITING.release()
    t.join(timeout=10)
    assert done.is_set()


def test_empty_trash_keeps_a_file_that_a_live_row_still_points_at(lib):
    iid = lib.ids[0]
    library_actions.trash_image(lib.db, lib.app.state.cfg, iid)
    path = Path(_row(lib, iid)["path"])
    with get_session(lib.db) as s:  # a row that is not in the trash but whose file sits there
        s.get(M.ImageRecord, iid).stage = "new"
        s.commit()
    stray = path.parent / "stray.txt"
    stray.write_text("untracked")
    trash_actions.empty_trash(lib.db, lib.app.state.cfg)
    assert path.exists()
    assert not stray.exists()


def test_delete_permanently_never_unlinks_a_file_outside_the_trash(lib):
    iid = lib.ids[0]
    path = Path(_row(lib, iid)["path"])
    with get_session(lib.db) as s:  # a trash row whose file is really in the library
        r = s.get(M.ImageRecord, iid)
        r.stage, r.route_status, r.scan_status = "trash", "trashed", "missing"
        s.commit()
    out = trash_actions.delete_permanently(lib.db, lib.app.state.cfg, [iid])
    assert out["deleted"] == 0 and out["skipped"] == 1
    assert out["skipped_reasons"] == {"outside_trash": 1}
    assert path.exists() and _row(lib, iid)["route_status"] == "trashed"


def test_restore_reports_a_move_that_failed(lib, monkeypatch):
    iid = lib.ids[0]
    library_actions.trash_image(lib.db, lib.app.state.cfg, iid)
    from photocure.engine.action_engine import ActionEngine

    def boom(self, action, target, sha=""):
        raise OSError("no space left")

    monkeypatch.setattr(ActionEngine, "_apply", boom)
    out = trash_actions.restore(lib.db, lib.app.state.cfg, [iid])
    assert out["restored"] == 0 and out["failed"] == 1
    assert "no space left" in out["errors"][0]
    assert _row(lib, iid)["stage"] == "trash"


def test_startup_moves_a_row_found_in_the_trash_folder_to_the_trash_stage(tmp_path):
    db = tmp_path / "s.db"
    init_db(db)
    trash_file = tmp_path / "Lib" / ".trash" / "T1" / "x.jpg"
    with get_session(db) as s:
        s.add(M.ImageRecord(path=str(trash_file), filename="x.jpg", stage="library", route_status="executed"))
        s.commit()
    with TestClient(create_app(str(db))):
        pass
    with get_session(db) as s:
        r = s.exec(select(M.ImageRecord)).one()
        assert (r.stage, r.scan_status, r.route_status) == ("trash", "missing", "trashed")


def test_upgrade_backfills_origin_from_the_first_move_in_the_ledger(tmp_path):
    db = tmp_path / "old.db"
    init_db(db)
    con = sqlite3.connect(db)
    con.execute("DROP INDEX ix_images_stage")
    con.execute("DROP INDEX ix_person_clusters_stage")
    con.execute("ALTER TABLE images DROP COLUMN stage")
    con.execute("ALTER TABLE images DROP COLUMN origin_path")
    con.execute("ALTER TABLE person_clusters DROP COLUMN stage")
    base = ("INSERT INTO images (path, relative_path, filename, extension, mime_type, file_size, quick_hash, sha256,"
            " width, height, orientation, taken_at_source, camera_model, city, region, country_code, is_favorite,"
            " scan_status, infer_status, is_duplicate, is_best_shot, is_alternate, route_status, locked,"
            " delete_approved, has_blinks, blink_count, faces_schema_version, created_at, updated_at)"
            " VALUES (?, '', ?, '', '', 0, '', '', 0, 0, 1, '', '', '', '', '', 0, 'scanned', 'pending', 0, 0, 0,"
            " 'executed', 0, 0, 0, 0, 0, '2026-01-01', '2026-01-01')")
    con.execute(base, ("/lib/a.jpg", "a.jpg"))
    con.execute(base, ("/lib/b.jpg", "b.jpg"))
    con.execute("INSERT INTO action_ledger (transaction_id, image_id, source_path, target_path, operation_type,"
                " created_dir, prev_relative_path, sha256, file_size, actual_operation, status, created_at,"
                " prev_stage, prev_scan_status, prev_route_status, prev_delete_approved)"
                " VALUES ('T1', 1, '/photos/a.jpg', '/lib/a.jpg', 'move', '', '', '', 0, 'move', 'committed',"
                " '2026-01-01', '', '', '', 0)")
    con.execute("PRAGMA user_version=10")
    con.commit()
    con.close()
    init_db(db)
    con = sqlite3.connect(db)
    got = dict(con.execute("SELECT filename, origin_path FROM images"))
    assert got["a.jpg"] == "/photos/a.jpg"
    assert got["b.jpg"] is None  # unknown origin: Return to origin will skip it, never guess the library path
    assert con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
