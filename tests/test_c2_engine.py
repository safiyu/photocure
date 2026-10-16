"""C2 acceptance: hardlink/copy provenance + undo, collision policies, orphaned recovery."""
from __future__ import annotations

import errno
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


def test_hardlink_undo_survives_source_deletion(tmp_path):
    w = build_world(tmp_path, n=2)
    rep = ActionEngine(w.cfg, db_path=w.db).execute(_acts(w, "hardlink"), transaction_id="TX_H")
    assert rep.failures == 0 and rep.copy_fallbacks == 0
    for op, dev, ino, status in _ledger(w, "TX_H"):
        assert op == "hardlink" and dev is not None and ino is not None and status == "committed"
    for p in w.paths:
        p.unlink()  # user cleans up the import folder
    out = RollbackEngine(w.cfg, db_path=w.db).rollback_transaction("TX_H")
    assert out.failures == 0 and out.reversed == 2
    assert not any((w.out / "A").glob("*.jpg")) if (w.out / "A").exists() else True


def test_hardlink_undo_refuses_when_target_was_replaced(tmp_path):
    w = build_world(tmp_path, n=1)
    ActionEngine(w.cfg, db_path=w.db).execute(_acts(w, "hardlink"), transaction_id="TX_R")
    target = w.out / "A" / sorted(w.paths)[0].name
    target.unlink()
    target.write_bytes(b"the user's own file")  # different inode now
    out = RollbackEngine(w.cfg, db_path=w.db).rollback_transaction("TX_R")
    assert out.failures == 1 and target.read_bytes() == b"the user's own file"


def test_cross_filesystem_falls_back_to_copy_and_undo_verifies(tmp_path, monkeypatch):
    w = build_world(tmp_path, n=2)
    real = os.link

    def exdev(src, dst, *a, **k):
        raise OSError(errno.EXDEV, "cross-device link")
    monkeypatch.setattr(os, "link", exdev)
    rep = ActionEngine(w.cfg, db_path=w.db).execute(_acts(w, "hardlink"), transaction_id="TX_C")
    monkeypatch.setattr(os, "link", real)
    assert rep.failures == 0 and rep.copy_fallbacks == 2
    assert {r[0] for r in _ledger(w, "TX_C")} == {"copy"}
    for p in w.paths:
        p.unlink()
    out = RollbackEngine(w.cfg, db_path=w.db).rollback_transaction("TX_C")
    assert out.failures == 0 and out.reversed == 2


def test_copy_fallback_undo_refuses_modified_copy(tmp_path, monkeypatch):
    w = build_world(tmp_path, n=1)
    monkeypatch.setattr(os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EXDEV, "x")))
    ActionEngine(w.cfg, db_path=w.db).execute(_acts(w, "hardlink"), transaction_id="TX_M")
    target = w.out / "A" / sorted(w.paths)[0].name
    target.write_bytes(b"edited after the fact")
    out = RollbackEngine(w.cfg, db_path=w.db).rollback_transaction("TX_M")
    assert out.failures == 1 and target.exists()


def test_disk_full_is_not_masked_by_copy_fallback(tmp_path, monkeypatch):
    w = build_world(tmp_path, n=1)
    monkeypatch.setattr(os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.ENOSPC, "full")))
    rep = ActionEngine(w.cfg, db_path=w.db).execute(_acts(w, "hardlink"))
    assert rep.failures == 1 and rep.copy_fallbacks == 0


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
    for col in ("xmp_written_rating", "xmp_written_tags"):
        con.execute(f"ALTER TABLE images DROP COLUMN {col}")
    con.execute("PRAGMA user_version=3")
    con.commit()
    con.close()
    init_db(db)  # migrate on startup
    con = sqlite3.connect(db)
    ledger = {r[1] for r in con.execute("PRAGMA table_info(action_ledger)")}
    images = {r[1] for r in con.execute("PRAGMA table_info(images)")}
    assert {"actual_operation", "target_dev", "target_ino"} <= ledger
    assert {"xmp_written_rating", "xmp_written_tags"} <= images
    assert con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 4
