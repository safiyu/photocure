"""C2 acceptance: hardlink/copy provenance + undo, collision policies, orphaned recovery."""
from __future__ import annotations

import os

import pytest

from photocure.db.repository import PhotoRepository
from photocure.db.session import get_session
from photocure.engine.action_engine import ActionEngine
from photocure.engine.rollback import RollbackEngine
from photocure.rules.organizer import PlannedAction
from photocure.rules.taxonomy import CategoryType

from .world import build_world


def _acts(w, op, n=None):
    return [PlannedAction(image_id=i, source=p, target=w.out / "A" / p.name,
                          category=CategoryType.PEOPLE, operation=op)
            for i, p in list(zip(w.ids, sorted(w.paths)))[:n]]


def _ledger(w, tid):
    with get_session(w.db) as s:
        rows = PhotoRepository(s).ledger_by_txn(tid)
        return [(r.actual_operation, r.target_dev, r.target_ino, r.status) for r in rows]


def _occupy(w):
    p = sorted(w.paths)[0]
    (w.out / "A").mkdir(parents=True, exist_ok=True)
    victim = w.out / "A" / p.name
    victim.write_bytes(b"already here")
    return p, victim


def test_policy_sequence(tmp_path):
    w = build_world(tmp_path, n=1)
    p, victim = _occupy(w)
    (w.out / "A" / f"{p.stem} (1){p.suffix}").write_bytes(b"also here")
    rep = ActionEngine(w.cfg, db_path=w.db).execute(_acts(w, "copy"))
    assert rep.failures == 0 and (w.out / "A" / f"{p.stem} (2){p.suffix}").exists()
    assert victim.read_bytes() == b"already here"


def test_policy_date_suffix(tmp_path):
    w = build_world(tmp_path, n=1)
    w.cfg.collision_policy = "date_suffix"
    p, victim = _occupy(w)
    rep = ActionEngine(w.cfg, db_path=w.db).execute(_acts(w, "copy"))
    assert rep.failures == 0
    names = [x.name for x in (w.out / "A").iterdir() if x != victim]
    assert len(names) == 1 and names[0].startswith(p.stem + "_2024") and names[0].endswith(p.suffix)


def test_policy_skip_counts_separately_from_errors(tmp_path):
    w = build_world(tmp_path, n=1)
    w.cfg.collision_policy = "skip"
    p, victim = _occupy(w)
    rep = ActionEngine(w.cfg, db_path=w.db).execute(_acts(w, "move"))
    assert rep.skipped_collisions == 1 and rep.skipped == 1 and rep.failures == 0
    assert p.exists() and victim.read_bytes() == b"already here"
    assert _ledger(w, rep.transaction_id) == []


def test_unknown_collision_policy_is_rejected():
    from photocure.config import CuratorConfig
    with pytest.raises(ValueError):
        CuratorConfig(collision_policy="sha_dup")


def test_older_database_gains_the_new_columns(tmp_path):
    import sqlite3

    from photocure.db.session import SCHEMA_VERSION, init_db
    db = tmp_path / "old.db"
    init_db(db)
    con = sqlite3.connect(db)
    for col in ("actual_operation", "target_dev", "target_ino"):
        con.execute(f"ALTER TABLE action_ledger DROP COLUMN {col}")
    for col in ("xmp_written_rating", "xmp_written_tags", "faces_schema_version", "blink_count"):
        con.execute(f"ALTER TABLE images DROP COLUMN {col}")
    for col in ("eye_closed_score", "eye_status", "is_blink", "x1"):
        con.execute(f"ALTER TABLE face_detections DROP COLUMN {col}")
    con.execute("PRAGMA user_version=3")
    con.commit()
    con.close()
    init_db(db)  # migrate on startup
    con = sqlite3.connect(db)
    ledger = {r[1] for r in con.execute("PRAGMA table_info(action_ledger)")}
    images = {r[1] for r in con.execute("PRAGMA table_info(images)")}
    assert {"actual_operation", "target_dev", "target_ino"} <= ledger
    faces = {r[1] for r in con.execute("PRAGMA table_info(face_detections)")}
    assert {"xmp_written_rating", "xmp_written_tags", "faces_schema_version", "blink_count"} <= images
    assert {"eye_closed_score", "eye_status", "is_blink", "x1"} <= faces
    assert con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
