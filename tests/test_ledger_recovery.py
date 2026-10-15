"""Crash recovery, partial failure and rollback-safety tests for the ledger."""
from __future__ import annotations

import os

import pytest

from photocure.db.repository import PhotoRepository
from photocure.db.session import get_session
from photocure.engine.action_engine import ActionEngine
from photocure.engine.rollback import RollbackEngine
from photocure.rules.organizer import PlannedAction
from photocure.rules.taxonomy import CategoryType

from .world import build_world, get_img


def _actions(world, out, op="move"):
    acts = []
    for i, p in zip(world.ids, sorted(world.paths)):
        acts.append(PlannedAction(image_id=i, source=p, target=out / "People" / p.name,
                                  category=CategoryType.PEOPLE, operation=op))
    return acts


def test_crash_keeps_committed_prefix_and_undo_restores(tmp_path, monkeypatch):
    w = build_world(tmp_path)
    acts = _actions(w, w.out)
    real = ActionEngine._apply
    calls = {"n": 0}

    def flaky(self, action, target, sha=""):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt  # simulates a kill mid-apply
        return real(self, action, target, sha)

    monkeypatch.setattr(ActionEngine, "_apply", flaky)
    with pytest.raises(KeyboardInterrupt):
        ActionEngine(w.cfg, db_path=w.db).execute(acts, transaction_id="TX_CRASH")
    monkeypatch.undo()

    with get_session(w.db) as s:
        rows = PhotoRepository(s).ledger_by_txn("TX_CRASH")
        assert [r.status for r in rows] == ["committed", "committed", "pending"]

    rep = RollbackEngine(w.cfg, db_path=w.db).rollback_transaction("TX_CRASH")
    assert rep.reversed == 2 and rep.failures == 0
    assert all(p.exists() for p in w.paths)
    assert not (w.out / "People").exists()
    for i, p in zip(w.ids, sorted(w.paths)):
        assert get_img(w, i).path == str(p)


def test_crash_after_move_before_commit_is_recovered(tmp_path, monkeypatch):
    w = build_world(tmp_path, n=2)
    acts = _actions(w, w.out)
    real = PhotoRepository.ledger_commit
    calls = {"n": 0}

    def boom(self, entry):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real(self, entry)

    monkeypatch.setattr(PhotoRepository, "ledger_commit", boom)
    with pytest.raises(KeyboardInterrupt):
        ActionEngine(w.cfg, db_path=w.db).execute(acts, transaction_id="TX_C2")
    monkeypatch.undo()
    # second file was physically moved but its row is still pending
    assert not sorted(w.paths)[1].exists()
    rep = RollbackEngine(w.cfg, db_path=w.db).rollback_transaction("TX_C2")
    assert rep.reversed == 2
    assert all(p.exists() for p in w.paths)


def test_partial_copy_left_by_crash_is_removed(tmp_path):
    w = build_world(tmp_path, n=1)
    src = w.paths[0]
    target = w.out / "x" / src.name
    target.parent.mkdir(parents=True)
    target.write_bytes(b"half")
    with get_session(w.db) as s:
        PhotoRepository(s).ledger_add("TX_P", w.ids[0], str(src), str(target),
                                      "copy", "deadbeef" * 8, 1234)
    rep = RollbackEngine(w.cfg, db_path=w.db).rollback_transaction("TX_P")
    assert not target.exists() and src.exists()
    assert any("partial" in n for n in rep.notes)


def test_failure_does_not_flip_previous_committed_entry(tmp_path):
    w = build_world(tmp_path, n=2)
    acts = _actions(w, w.out)
    sorted(w.paths)[1].unlink()  # second source vanishes before apply
    rep = ActionEngine(w.cfg, db_path=w.db).execute(acts, transaction_id="TX_F")
    assert rep.executed == 1 and rep.failures == 1
    with get_session(w.db) as s:
        rows = PhotoRepository(s).ledger_by_txn("TX_F")
        assert [r.status for r in rows] == ["committed"]
    undo = RollbackEngine(w.cfg, db_path=w.db).rollback_transaction("TX_F")
    assert undo.reversed == 1
    assert sorted(w.paths)[0].exists()


def test_ledger_names_the_collision_renamed_target(tmp_path):
    w = build_world(tmp_path, n=1)
    src = w.paths[0]
    target = w.out / "People" / src.name
    target.parent.mkdir(parents=True)
    target.write_bytes(b"pre-existing user file")
    acts = [PlannedAction(image_id=w.ids[0], source=src, target=target,
                          category=CategoryType.PEOPLE, operation="move")]
    ActionEngine(w.cfg, db_path=w.db).execute(acts, transaction_id="TX_COL")
    with get_session(w.db) as s:
        row = PhotoRepository(s).ledger_by_txn("TX_COL")[0]
        assert row.target_path != str(target) and "_dup" in row.target_path
    RollbackEngine(w.cfg, db_path=w.db).rollback_transaction("TX_COL")
    assert target.read_bytes() == b"pre-existing user file"
    assert src.exists()


def test_undo_prunes_only_created_dirs(tmp_path):
    w = build_world(tmp_path, n=1)
    keep = w.out / "keep"
    keep.mkdir(parents=True)
    src = w.paths[0]
    acts = [PlannedAction(image_id=w.ids[0], source=src,
                          target=keep / "People" / "2024" / src.name,
                          category=CategoryType.PEOPLE)]
    ActionEngine(w.cfg, db_path=w.db).execute(acts, transaction_id="TX_D")
    RollbackEngine(w.cfg, db_path=w.db).rollback_transaction("TX_D")
    assert keep.exists() and w.out.exists()
    assert not (keep / "People").exists()


def test_symlink_at_source_slot_is_not_clobbered(tmp_path):
    w = build_world(tmp_path, n=1)
    src = w.paths[0]
    acts = _actions(w, w.out)[:1]
    ActionEngine(w.cfg, db_path=w.db).execute(acts, transaction_id="TX_S")
    decoy = tmp_path / "decoy.txt"
    decoy.write_text("x")
    os.symlink(decoy, src)
    rep = RollbackEngine(w.cfg, db_path=w.db).rollback_transaction("TX_S")
    assert rep.failures == 1 and rep.reversed == 0
    assert src.is_symlink() and (w.out / "People" / src.name).exists()


def test_cross_device_move_refuses_symlink_source(tmp_path, monkeypatch):
    w = build_world(tmp_path, n=1)
    real = tmp_path / "real.jpg"
    real.write_bytes(b"data")
    link = tmp_path / "link.jpg"
    os.symlink(real, link)
    monkeypatch.setattr(ActionEngine, "_same_device", staticmethod(lambda a, b: False))
    with pytest.raises(RuntimeError, match="symlink"):
        ActionEngine._move(link, tmp_path / "t.jpg", "")
    assert link.is_symlink() and real.exists()


def test_cross_device_move_checks_hash_against_ledger_sha(tmp_path, monkeypatch):
    src = tmp_path / "a.bin"
    src.write_bytes(b"payload")
    monkeypatch.setattr(ActionEngine, "_same_device", staticmethod(lambda a, b: False))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        ActionEngine._move(src, tmp_path / "b.bin", "0" * 64)
    assert src.exists() and not (tmp_path / "b.bin").exists()


def test_space_check_blocks_copy(tmp_path, monkeypatch):
    import shutil
    from collections import namedtuple
    src = tmp_path / "a.bin"
    src.write_bytes(b"x" * 100)
    U = namedtuple("U", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: U(10, 10, 5))
    with pytest.raises(OSError, match="free space"):
        ActionEngine._copy(src, tmp_path / "b.bin")
