"""Moving the library to another folder: copy, verify, then rewrite the database in one go."""
from __future__ import annotations

from pathlib import Path

import pytest
from sqlmodel import select

from photocure import library_migration
from photocure.db import models as M
from photocure.db.session import get_session
from photocure.services import load_settings
from photocure.ui import app as ui_app
from photocure.ui.app import create_app

from .test_ui_api import env, wait_job  # noqa: F401  (shared fixture)


@pytest.fixture()
def lib(env):
    """Three photos in the library and one in the trash, all under the library root."""
    ids = sorted(i["id"] for i in env.client.get("/api/images?limit=50").json()["items"])
    with get_session(env.db) as s:
        s.get(M.ImageRecord, ids[0]).delete_approved = True
        s.commit()
    r = env.client.post("/api/plan/apply", json={"operation": "move", "trash": True})
    assert wait_job(env.client, r.json()["job_id"])["status"] == "done"
    env.ids = ids
    env.new_root = env.tmp / "NewLibrary"
    return env


def _cfg(env):
    return env.app.state.cfg


def _files(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


def test_preview_counts_files_and_rejects_a_non_empty_target(lib):
    old = lib.out.resolve()
    pv = library_migration.preview_migration(lib.db, _cfg(lib), lib.new_root)
    assert pv["ok"] is True and pv["error"] is None
    assert pv["files"] == len(_files(old)) == 4 and pv["bytes"] > 0
    assert pv["enough_space"] is True and pv["free_bytes"] > pv["bytes"]
    lib.new_root.mkdir()
    (lib.new_root / "something.txt").write_text("x")
    pv = library_migration.preview_migration(lib.db, _cfg(lib), lib.new_root)
    assert pv["ok"] is False and "empty" in pv["error"]


def test_preview_rejects_a_folder_inside_the_library_or_around_it(lib):
    old = lib.out.resolve()
    assert library_migration.preview_migration(lib.db, _cfg(lib), old / "inner")["ok"] is False
    assert library_migration.preview_migration(lib.db, _cfg(lib), old.parent)["ok"] is False


def test_migration_copies_verifies_and_rewrites_paths(lib):
    old, new = lib.out.resolve(), lib.new_root
    before = _files(old)
    out = library_migration.migrate_library(lib.db, _cfg(lib), new)
    assert out["ok"] is True and out["copied"] == 4 and out["failed"] == []
    assert out["old_root"] == str(old) and out["new_root"] == str(new)
    assert _files(new) == before and _files(old) == before  # nothing is deleted from the old folder
    with get_session(lib.db) as s:
        rows = s.exec(select(M.ImageRecord).where(M.ImageRecord.stage.in_(("library", "trash")))).all()
        assert len(rows) == 4
        for r in rows:
            assert Path(r.path).is_relative_to(new) and Path(r.path).exists()
            if r.target_path:
                assert Path(r.target_path).is_relative_to(new)
        for e in s.exec(select(M.ActionLedger)).all():
            assert not str(e.target_path).startswith(str(old))
    assert load_settings(lib.db)["library_root"] == str(new)


def test_migration_failure_leaves_the_database_and_setting_untouched(lib, monkeypatch):
    old, new = lib.out.resolve(), lib.new_root
    real = library_migration._copy_verified
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("disk went away")
        return real(src, dst)

    monkeypatch.setattr(library_migration, "_copy_verified", flaky)
    out = library_migration.migrate_library(lib.db, _cfg(lib), new)
    assert out["ok"] is False and len(out["failed"]) == 1
    with get_session(lib.db) as s:
        for r in s.exec(select(M.ImageRecord).where(M.ImageRecord.stage.in_(("library", "trash")))).all():
            assert Path(r.path).is_relative_to(old)
    assert load_settings(lib.db)["library_root"] == str(old)


def test_migrate_endpoint_runs_as_a_job_and_the_new_root_sticks(lib):
    r = lib.client.post("/api/library/migrate", json={"path": str(lib.new_root)})
    assert r.status_code == 202
    j = wait_job(lib.client, r.json()["job_id"])
    assert j["status"] == "done" and j["result"]["ok"] is True
    assert lib.client.get("/api/settings").json()["library_root"] == str(lib.new_root)
    first = lib.client.get("/api/images?stage=library&limit=5").json()["items"][0]
    assert Path(first["path"]).is_relative_to(lib.new_root)


def test_preview_endpoint_validates_the_folder(lib):
    ok = lib.client.get("/api/library/migrate/preview", params={"path": str(lib.new_root)})
    assert ok.status_code == 200 and ok.json()["files"] == 4
    assert lib.client.get("/api/library/migrate/preview", params={"path": "relative/x"}).status_code == 400
    assert lib.client.get("/api/library/migrate/preview", params={"path": "/etc/photos"}).status_code == 400


def test_migrate_endpoint_is_busy_aware(lib):
    assert ui_app._BUSY.acquire(blocking=False)
    try:
        assert lib.client.post("/api/library/migrate", json={"path": str(lib.new_root)}).status_code == 409
    finally:
        ui_app._BUSY.release()


def test_migrate_endpoint_is_forbidden_when_the_root_comes_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("PCUR_LIBRARY_ROOT", str(tmp_path / "EnvLib"))
    from fastapi.testclient import TestClient
    app = create_app(str(tmp_path / "ui.db"))
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token
    r = client.post("/api/library/migrate", json={"path": str(tmp_path / "Other")})
    assert r.status_code == 403


def test_move_library_modal_and_wiring_exist():
    static = Path(__file__).resolve().parent.parent / "photocure" / "ui" / "static"
    html = (static / "index.html").read_text()
    js = (static / "app.js").read_text()
    for needle in ('id="movelib"', 'id="movelib-path"', 'id="movelib-browse"', 'id="movelib-preview"',
                   'id="movelib-start"'):
        assert needle in html, needle
    assert html.index('id="movelib"') < html.index('id="picker"')  # the folder picker opens above it
    assert '$("#btn-move-library").onclick = openMoveLibrary' in js
    assert "/api/library/migrate/preview" in js and '"/api/library/migrate"' in js
    assert "is yours to delete" in js
    assert 'id="movelib-start" type="button" disabled' in html  # needs a good preview first
