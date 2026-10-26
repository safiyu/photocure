"""The stage and origin_path columns, person stage, per-stage stats and the backfill."""
from __future__ import annotations

import sqlite3

from photocure.db import models as M
from photocure.db.session import SCHEMA_VERSION, init_db


def _img(path: str, **kw) -> M.ImageRecord:
    return M.ImageRecord(path=path, filename=path.rsplit("/", 1)[-1], **kw)


def test_new_image_defaults_to_stage_new(session):
    rec = _img("/p/a.jpg")
    session.add(rec)
    session.flush()
    assert rec.stage == "new"
    assert rec.origin_path is None


def test_person_cluster_defaults_to_library(session):
    c = M.PersonCluster()
    session.add(c)
    session.flush()
    assert c.stage == "library"


def test_stage_constants():
    assert (M.STAGE_NEW, M.STAGE_LIBRARY, M.STAGE_TRASH) == ("new", "library", "trash")
    assert M.STAGES == ("new", "library", "trash")


def test_migrate_backfills_stage_from_route_status(tmp_path):
    db = tmp_path / "old.db"
    init_db(db)
    con = sqlite3.connect(db)
    con.execute("DROP INDEX ix_images_stage")
    con.execute("DROP INDEX ix_person_clusters_stage")
    con.execute("ALTER TABLE images DROP COLUMN stage")
    con.execute("ALTER TABLE images DROP COLUMN origin_path")
    con.execute("ALTER TABLE person_clusters DROP COLUMN stage")
    for name, route in (("a", "executed"), ("b", "trashed"), ("c", "unrouted"), ("d", "planned")):
        con.execute(
            "INSERT INTO images (path, relative_path, filename, extension, mime_type, file_size,"
            " quick_hash, sha256, width, height, orientation, taken_at_source, camera_model, city,"
            " region, country_code, is_favorite, scan_status, infer_status, is_duplicate,"
            " is_best_shot, is_alternate, route_status, locked, delete_approved, has_blinks,"
            " blink_count, faces_schema_version, created_at, updated_at)"
            " VALUES (?, '', ?, '', '', 0, '', '', 0, 0, 1, '', '', '', '', '', 0, 'scanned',"
            " 'pending', 0, 0, 0, ?, 0, 0, 0, 0, 0, '2026-01-01', '2026-01-01')",
            (f"/lib/{name}.jpg", f"{name}.jpg", route))
    con.execute("PRAGMA user_version=10")
    con.commit()
    con.close()
    init_db(db)
    con = sqlite3.connect(db)
    got = {r[0]: (r[1], r[2]) for r in con.execute("SELECT filename, stage, origin_path FROM images")}
    assert got["a.jpg"] == ("library", "/lib/a.jpg")
    assert got["b.jpg"] == ("trash", None)
    assert got["c.jpg"] == ("new", None)
    assert got["d.jpg"] == ("new", None)
    assert con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert SCHEMA_VERSION == 12


def test_migrate_leaves_existing_stages_alone_on_rerun(tmp_path):
    db = tmp_path / "again.db"
    init_db(db)
    con = sqlite3.connect(db)
    con.execute(
        "INSERT INTO images (path, relative_path, filename, extension, mime_type, file_size,"
        " quick_hash, sha256, width, height, orientation, taken_at_source, camera_model, city,"
        " region, country_code, is_favorite, scan_status, infer_status, is_duplicate,"
        " is_best_shot, is_alternate, route_status, locked, delete_approved, has_blinks,"
        " blink_count, faces_schema_version, created_at, updated_at, stage)"
        " VALUES ('/x/a.jpg', '', 'a.jpg', '', '', 0, '', '', 0, 0, 1, '', '', '', '', '', 0,"
        " 'scanned', 'pending', 0, 0, 0, 'executed', 0, 0, 0, 0, 0, '2026-01-01', '2026-01-01', 'trash')")
    con.commit()
    con.close()
    init_db(db)
    con = sqlite3.connect(db)
    assert con.execute("SELECT stage FROM images").fetchone()[0] == "trash"


def test_stats_counts_per_stage(repo, session):
    for i, stage in enumerate(("new", "new", "library", "trash")):
        session.add(_img(f"/p/{i}.jpg", stage=stage))
    session.add(_img("/p/gone.jpg", stage="library", scan_status="missing"))
    session.flush()
    assert repo.stats()["stages"] == {"new": 2, "library": 1, "trash": 1}
