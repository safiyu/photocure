"""Regression tests for the engine/services review findings."""
from __future__ import annotations

import time
from types import SimpleNamespace

from photocure.db.repository import PhotoRepository
from photocure.db.session import get_session
from photocure.rules.decision_matrix import DecisionMatrix
from photocure.rules.taxonomy import CategoryType
from photocure.services import Planner

from .world import build_world, get_img, set_bucket


def _files(root):
    return sorted(p.name for p in root.rglob("*.jpg"))


def test_reapply_move_does_not_rename_organized_photos(tmp_path):
    w = build_world(tmp_path, n=3)
    p1 = w.planner.build_plan(organize_root=w.out, operation="move")
    w.planner.execute_plan(p1)
    before = _files(w.out)
    p2 = w.planner.build_plan(organize_root=w.out, operation="move")
    rep = w.planner.execute_plan(p2)
    assert rep.executed == 0 and rep.in_place == 3 and rep.failures == 0
    assert _files(w.out) == before
    assert not any("(1)" in n for n in _files(w.out))
    for i in w.ids:
        assert get_img(w, i).route_status == "executed"


def test_reapply_hardlink_in_place_is_idempotent(tmp_path):
    w = build_world(tmp_path, n=2)
    from photocure.rules.organizer import OrganizeMode
    for _ in range(3):
        p = w.planner.build_plan(organize_root=w.out, operation="move",
                                 mode=OrganizeMode.IN_PLACE)
        w.planner.execute_plan(p)
    assert len(_files(w.root)) == 2
    assert not list(w.root.rglob("* (*).jpg"))


def test_reapply_copy_does_not_duplicate_library(tmp_path):
    w = build_world(tmp_path, n=3)
    w.planner.execute_plan(w.planner.build_plan(organize_root=w.out, operation="copy"))
    rep = w.planner.execute_plan(w.planner.build_plan(organize_root=w.out,
                                                      operation="copy"))
    assert rep.executed == 0 and rep.in_place == 3
    assert len(_files(w.out)) == 3


def test_post_mutation_db_error_leaves_ledger_pending_and_recoverable(
        tmp_path, monkeypatch):
    w = build_world(tmp_path, n=1)
    p = w.planner.build_plan(organize_root=w.out, operation="move")

    def boom(self, entry):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(PhotoRepository, "ledger_commit", boom)
    rep = w.planner.execute_plan(p)
    monkeypatch.undo()
    assert rep.failures == 1 and rep.executed == 0
    assert p.actions[0].target.exists()
    with get_session(w.db) as s:
        rows = PhotoRepository(s).ledger_by_txn(rep.transaction_id)
        assert [r.status for r in rows] == ["pending"]
    undo = w.planner.undo(rep.transaction_id)
    assert undo.reversed == 1
    assert w.paths[0].exists() and not p.actions[0].target.exists()


def test_failed_mutation_still_marks_ledger_failed(tmp_path, monkeypatch):
    from photocure.engine.action_engine import ActionEngine
    w = build_world(tmp_path, n=1)
    p = w.planner.build_plan(organize_root=w.out, operation="move")

    def boom(self, action, target, sha=""):
        raise OSError("disk exploded")

    monkeypatch.setattr(ActionEngine, "_apply", boom)
    rep = w.planner.execute_plan(p)
    with get_session(w.db) as s:
        rows = PhotoRepository(s).ledger_by_txn(rep.transaction_id)
        assert [r.status for r in rows] == ["failed"]


def test_purge_trash_marks_ledger_with_relative_output_dir(tmp_path, monkeypatch):
    w = build_world(tmp_path, n=1)
    set_bucket(w, w.ids[0], "screenshot")
    p = w.planner.build_plan(organize_root=w.out, operation="move")
    rep = w.planner.execute_plan(p, trash=True)
    monkeypatch.chdir(tmp_path)
    res = w.planner.purge_trash(type(w.out)("out"), older_than_days=30,
                                now=time.time() + 31 * 86400)
    assert res["removed_transactions"] == [rep.transaction_id]
    with get_session(w.db) as s:
        rows = PhotoRepository(s).ledger_by_txn(rep.transaction_id)
        assert rows[0].status == "purged"
    assert get_img(w, w.ids[0]).scan_status == "missing"


def test_only_category_partial_apply_does_not_need_force(tmp_path):
    w = build_world(tmp_path, n=4)
    set_bucket(w, w.ids[0], "food")
    set_bucket(w, w.ids[1], "screenshot")
    p = w.planner.build_plan(organize_root=w.out, operation="move")
    first = w.planner.execute_plan(p.plan_id, only_category="food")
    assert first.executed == 1
    assert w.planner.load_plan(p.plan_id).applied_transaction in (None, "")
    second = w.planner.execute_plan(p.plan_id, only_category="screenshot")
    assert second.executed == 1 and second.failures == 0
    rest = w.planner.execute_plan(p.plan_id)
    assert rest.executed == 2 and rest.in_place == 2
    assert w.planner.load_plan(p.plan_id).applied_transaction == rest.transaction_id


def test_partial_apply_still_detects_later_drift(tmp_path):
    import pytest
    from photocure.services import PlanDriftError
    w = build_world(tmp_path, n=3)
    set_bucket(w, w.ids[0], "food")
    p = w.planner.build_plan(organize_root=w.out, operation="move")
    w.planner.execute_plan(p.plan_id, only_category="food")
    set_bucket(w, w.ids[1], "screenshot")
    with pytest.raises(PlanDriftError, match="stale"):
        w.planner.execute_plan(p.plan_id)


def test_locked_or_manual_burst_alternate_stays_an_alternate():
    m = DecisionMatrix()
    manual = SimpleNamespace(id=1, manual_bucket="burst_alternate", locked=False,
                             category=None, subcategory="")
    locked = SimpleNamespace(id=2, manual_bucket=None, locked=True,
                             category="burst_alternate", subcategory="")
    for rec in (manual, locked):
        d = Planner._decide(m, rec)
        assert d.category == CategoryType.BURST_ALTERNATE and d.is_alternate
    plain = SimpleNamespace(id=3, manual_bucket="food", locked=False,
                            category=None, subcategory="")
    assert not Planner._decide(m, plain).is_alternate


def test_dry_run_collision_flags_do_not_leak_into_apply(tmp_path):
    w = build_world(tmp_path, n=2)
    p = w.planner.build_plan(organize_root=w.out, operation="copy")
    for a in p.actions:
        a.collides = True  # as left behind by a dry run
    rep = w.planner.execute_plan(p)
    assert rep.executed == 2 and rep.collisions == 0
