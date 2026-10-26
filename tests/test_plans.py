"""Persisted plans, routing overrides, in-place safety, trash, verify, repair, schema."""
from __future__ import annotations

import os
import sqlite3
import time

import pytest

from photocure.db.models import BurstGroup
from photocure.db.repository import PhotoRepository
from photocure.db.session import SCHEMA_VERSION, close_engine, get_session, init_db
from photocure.rules.organizer import OrganizeMode
from photocure.services import PlanDriftError

from .world import approve_delete, build_world, get_img, set_bucket


def _src(w, i):
    return sorted(w.paths)[i]


def test_plan_is_persisted_and_apply_by_id_runs_exactly_it(tmp_path):
    w = build_world(tmp_path)
    p = w.planner.build_plan(organize_root=w.out, operation="copy")
    loaded = w.planner.load_plan(p.plan_id)
    assert loaded.total == p.total == 4
    assert [str(a.target) for a in loaded.actions] == [str(a.target) for a in p.actions]
    assert loaded.input_hash == p.input_hash
    rep = w.planner.execute_plan(p.plan_id)
    assert rep.executed == 4 and rep.failures == 0
    assert w.planner.load_plan(p.plan_id).applied_transaction == rep.transaction_id


def test_apply_refuses_stale_plan_unless_forced(tmp_path):
    w = build_world(tmp_path)
    p = w.planner.build_plan(organize_root=w.out, operation="copy")
    set_bucket(w, w.ids[0], "food")  # reviewer changes something after planning
    with pytest.raises(PlanDriftError, match="stale"):
        w.planner.execute_plan(p.plan_id)
    assert not w.out.exists()
    rep = w.planner.execute_plan(p.plan_id, force=True)
    assert rep.executed == 4


def test_plan_cannot_be_applied_twice(tmp_path):
    w = build_world(tmp_path)
    p = w.planner.build_plan(organize_root=w.out, operation="copy")
    w.planner.execute_plan(p.plan_id)
    with pytest.raises(PlanDriftError, match="already applied"):
        w.planner.execute_plan(p.plan_id)


def test_move_updates_db_path_and_undo_restores_it(tmp_path):
    w = build_world(tmp_path)
    p = w.planner.build_plan(organize_root=w.out, operation="move")
    rep = w.planner.execute_plan(p)
    for a in p.actions:
        img = get_img(w, a.image_id)
        assert img.path == str(a.target) and os.path.exists(img.path)
    # a fresh plan after the move must still see every file at its new place
    p2 = w.planner.build_plan(organize_root=w.out, operation="move", persist=False)
    assert all(a.source.exists() for a in p2.actions)
    w.planner.undo(rep.transaction_id)
    for i, src in zip(w.ids, sorted(w.paths)):
        assert get_img(w, i).path == str(src) and src.exists()


def test_repair_paths_fixes_stale_rows(tmp_path):
    w = build_world(tmp_path)
    p = w.planner.build_plan(organize_root=w.out, operation="move")
    w.planner.execute_plan(p)
    with get_session(w.db) as s:  # simulate a pre-fix store: path never updated
        for a in p.actions:
            PhotoRepository(s).get_by_id(a.image_id).path = str(a.source)
    assert w.planner.repair_paths() == 4
    for a in p.actions:
        assert get_img(w, a.image_id).path == str(a.target)


def test_undo_no_verify_flag_is_honoured(tmp_path):
    w = build_world(tmp_path, n=1)
    p = w.planner.build_plan(organize_root=w.out, operation="move")
    rep = w.planner.execute_plan(p)
    target = p.actions[0].target
    target.write_bytes(target.read_bytes() + b"edited")
    bad = w.planner.undo(rep.transaction_id, verify=True)
    assert bad.failures == 1 and bad.reversed == 0 and target.exists()
    ok = w.planner.undo(rep.transaction_id, verify=False)
    assert ok.reversed == 1 and w.paths[0].exists()


def test_second_undo_targets_the_previous_transaction(tmp_path):
    w = build_world(tmp_path, n=2)
    set_bucket(w, w.ids[0], "food")
    set_bucket(w, w.ids[1], "animal")
    p1 = w.planner.build_plan(organize_root=w.out, operation="move")
    t1 = w.planner.execute_plan(p1, only_category="food").transaction_id
    p2 = w.planner.build_plan(organize_root=w.out, operation="move")
    t2 = w.planner.execute_plan(p2, only_category="animal").transaction_id
    first = w.planner.undo()
    second = w.planner.undo()
    assert first.transaction_id == t2 and second.transaction_id == t1
    assert second.reversed == 1
    assert all(p.exists() for p in w.paths)
    third = w.planner.undo()
    assert third.reversed == 0


def test_manual_bucket_is_honoured_and_not_clobbered(tmp_path):
    w = build_world(tmp_path, n=2)
    set_bucket(w, w.ids[0], "food")
    set_bucket(w, w.ids[1], "not-a-category")
    p = w.planner.build_plan(organize_root=w.out, operation="copy")
    by_img = {a.image_id: a for a in p.actions}
    assert "Non-Human/Food" in str(by_img[w.ids[0]].target).replace(os.sep, "/")
    assert by_img[w.ids[0]].reason == "manual override"
    assert by_img[w.ids[1]].reason != "manual override"
    assert get_img(w, w.ids[0]).manual_bucket == "food"
    assert p.by_category["food"] == 1


def test_locked_rows_keep_their_category(tmp_path):
    w = build_world(tmp_path, n=1)
    with get_session(w.db) as s:
        img = PhotoRepository(s).get_by_id(w.ids[0])
        img.category = "animal"
        img.locked = True
    p = w.planner.build_plan(organize_root=w.out, operation="copy")
    assert p.actions[0].category.value == "animal"
    assert get_img(w, w.ids[0]).category == "animal"


def test_existing_bursts_are_reused_and_alternates_not_counted_uncertain(tmp_path):
    w = build_world(tmp_path, n=3)
    with get_session(w.db) as s:
        repo = PhotoRepository(s)
        bg = repo.upsert_burst(BurstGroup(name="Burst_20240501_0000", image_count=2))
        a, b = repo.get_by_id(w.ids[0]), repo.get_by_id(w.ids[1])
        a.burst_id = b.burst_id = bg.id
        a.is_best_shot = True
        b.is_alternate = True
    p = w.planner.build_plan(organize_root=w.out, operation="copy")
    assert p.burst_count == 1
    assert get_img(w, w.ids[1]).is_alternate and get_img(w, w.ids[0]).is_best_shot
    assert p.by_category.get("burst_alternate") == 1
    assert p.by_category.get("uncertain") == 2  # best shot + the lone image
    r = w.planner.build_plan(organize_root=w.out, operation="copy", recluster=True,
                             persist=False)
    assert r.burst_count == 0  # explicit recluster recomputes (distinct images)


def test_only_category_filter(tmp_path):
    w = build_world(tmp_path, n=3)
    set_bucket(w, w.ids[0], "food")
    p = w.planner.build_plan(organize_root=w.out, operation="move")
    rep = w.planner.execute_plan(p, only_category=["food"])
    assert rep.executed == 1 and rep.skipped == 2
    assert not _src(w, 0).exists() and _src(w, 1).exists() and _src(w, 2).exists()


def test_trash_stages_junk_and_purge_removes_it(tmp_path):
    w = build_world(tmp_path, n=3)
    set_bucket(w, w.ids[0], "graphic")
    approve_delete(w, w.ids[0])
    set_bucket(w, w.ids[1], "food")
    p = w.planner.build_plan(organize_root=w.out, operation="move")
    rep = w.planner.execute_plan(p, trash=True)
    trash_dir = w.out / ".trash" / rep.transaction_id
    assert (trash_dir / "Junk_Review" / "Graphics" / _src(w, 0).name).exists()
    assert (w.out / "Non-Human" / "Food" / "2024" / "05" / "02").exists() or \
        any((w.out / "Non-Human" / "Food").rglob("*.jpg"))
    # trash is journaled, so undo still restores the junk file
    undo = w.planner.undo(rep.transaction_id)
    assert undo.reversed == 3 and _src(w, 0).exists()
    assert not (w.out / ".trash" / rep.transaction_id).exists()


def test_unapproved_junk_is_not_staged_in_trash(tmp_path):
    w = build_world(tmp_path, n=2)
    set_bucket(w, w.ids[0], "graphic")
    set_bucket(w, w.ids[1], "graphic")
    approve_delete(w, w.ids[1])
    p = w.planner.build_plan(organize_root=w.out, operation="move")
    rep = w.planner.execute_plan(p, trash=True)
    trash_dir = w.out / ".trash" / rep.transaction_id
    assert rep.unapproved_junk == 1
    assert (trash_dir / "Junk_Review" / "Graphics" / _src(w, 1).name).exists()
    assert not list(trash_dir.rglob(_src(w, 0).name))
    assert any((w.out / "Junk_Review" / "Graphics").rglob(_src(w, 0).name))


def test_reassigning_clears_delete_approval(tmp_path):
    w = build_world(tmp_path, n=1)
    set_bucket(w, w.ids[0], "graphic")
    approve_delete(w, w.ids[0])
    assert get_img(w, w.ids[0]).delete_approved
    set_bucket(w, w.ids[0], "food")
    assert not get_img(w, w.ids[0]).delete_approved


def test_purge_trash_respects_retention_and_marks_ledger(tmp_path):
    w = build_world(tmp_path, n=1)
    set_bucket(w, w.ids[0], "graphic")
    approve_delete(w, w.ids[0])
    p = w.planner.build_plan(organize_root=w.out, operation="move")
    rep = w.planner.execute_plan(p, trash=True)
    kept = w.planner.purge_trash(w.out, older_than_days=30)
    assert kept["removed_transactions"] == []
    res = w.planner.purge_trash(w.out, older_than_days=30,
                                now=time.time() + 31 * 86400)
    assert res["removed_transactions"] == [rep.transaction_id] and res["files"] == 1
    assert not (w.out / ".trash").exists()
    with get_session(w.db) as s:
        rows = PhotoRepository(s).ledger_by_txn(rep.transaction_id)
        assert rows[0].status == "purged"
    assert w.planner.undo(rep.transaction_id).reversed == 0


def test_verify_transaction_reports_drift_and_missing(tmp_path):
    w = build_world(tmp_path, n=3)
    p = w.planner.build_plan(organize_root=w.out, operation="copy")
    rep = w.planner.execute_plan(p)
    clean = w.planner.verify_transaction(rep.transaction_id)
    assert clean["checked"] == 3 and clean["ok"] == 3
    p.actions[0].target.write_bytes(b"tampered")
    p.actions[1].target.unlink()
    res = w.planner.verify_transaction(rep.transaction_id)
    assert len(res["drift"]) == 1 and len(res["missing"]) == 1 and res["ok"] == 1


def test_dry_run_diff_reports_bytes_collisions_and_space(tmp_path):
    w = build_world(tmp_path, n=2)
    p = w.planner.build_plan(organize_root=w.out, operation="copy")
    p.actions[0].target.parent.mkdir(parents=True)
    p.actions[0].target.write_bytes(b"in the way")
    rep = w.planner.dry_run(p)
    assert rep.collisions == 1 and rep.bytes_needed == rep.bytes_total > 0
    assert rep.space_ok and rep.free_bytes
    mv = w.planner.build_plan(organize_root=w.out, operation="move", persist=False)
    assert w.planner.dry_run(mv).bytes_needed == 0  # same-device move is free


def test_latest_transaction_skips_fully_rolled_back(tmp_path):
    w = build_world(tmp_path, n=1)
    p = w.planner.build_plan(organize_root=w.out, operation="copy")
    rep = w.planner.execute_plan(p)
    w.planner.undo(rep.transaction_id)
    with get_session(w.db) as s:
        assert PhotoRepository(s).latest_transaction() is None


def test_pending_inference_includes_faces_done_and_skips_missing(tmp_path):
    w = build_world(tmp_path, n=3)
    with get_session(w.db) as s:
        repo = PhotoRepository(s)
        a, b, c = (repo.get_by_id(i) for i in w.ids)
        a.infer_status, b.infer_status, c.infer_status = "faces_done", "inferred", "pending"
        c.scan_status = "missing"
        s.flush()
        pend = {r.id for r in repo.images_pending_inference()}
    assert pend == {w.ids[0]}


def test_reset_failed_inference(tmp_path):
    w = build_world(tmp_path, n=2)
    with get_session(w.db) as s:
        repo = PhotoRepository(s)
        repo.get_by_id(w.ids[0]).infer_status = "failed"
        assert repo.reset_failed_inference() == 1
        assert repo.get_by_id(w.ids[0]).infer_status == "pending"


def test_schema_migration_adds_missing_columns(tmp_path):
    db = tmp_path / "old.db"
    init_db(db, drop=True)
    close_engine(db)
    con = sqlite3.connect(db)
    con.execute("DROP INDEX IF EXISTS ix_images_phash")
    con.execute("ALTER TABLE images DROP COLUMN phash")
    con.execute("ALTER TABLE images DROP COLUMN locked")
    con.execute("ALTER TABLE action_ledger DROP COLUMN created_dir")
    con.execute("PRAGMA user_version=1")
    con.commit()
    con.close()
    init_db(db)
    con = sqlite3.connect(db)
    cols = {r[1] for r in con.execute("PRAGMA table_info(images)")}
    led = {r[1] for r in con.execute("PRAGMA table_info(action_ledger)")}
    ver = con.execute("PRAGMA user_version").fetchone()[0]
    con.close()
    assert {"phash", "locked", "rating", "event_id"} <= cols
    assert "created_dir" in led and ver == SCHEMA_VERSION
    with get_session(db) as s:  # ORM works against the migrated table
        assert PhotoRepository(s).all_images() == []


def test_any_marked_photo_is_trashed_and_unmarked_ones_are_not(tmp_path):
    w = build_world(tmp_path, n=3)
    set_bucket(w, w.ids[0], "food")  # not junk: any photo can be marked
    approve_delete(w, w.ids[0])
    p = w.planner.build_plan(organize_root=w.out, operation="move")
    rep = w.planner.execute_plan(p, trash=True)
    trash = w.out / ".trash" / rep.transaction_id
    assert list(trash.rglob(_src(w, 0).name))  # marked: staged in the trash folder
    assert not list(trash.rglob(_src(w, 1).name))  # unmarked: organized normally
    assert any(w.out.rglob(_src(w, 1).name))


def test_marked_photo_is_left_out_without_trash(tmp_path):
    w = build_world(tmp_path, n=2)
    approve_delete(w, w.ids[0])
    p = w.planner.build_plan(organize_root=w.out, operation="move")
    rep = w.planner.execute_plan(p, trash=False)
    assert rep.marked_left_out == 1 and _src(w, 0).exists()  # not moved, not deleted
    assert not any(w.out.rglob(_src(w, 0).name)) and not (w.out / ".trash").exists()
    w2 = build_world(tmp_path / "copy", n=2)
    approve_delete(w2, w2.ids[0])
    p2 = w2.planner.build_plan(organize_root=w2.out, operation="copy")
    rep2 = w2.planner.execute_plan(p2, trash=False)
    assert rep2.marked_left_out == 1 and _src(w2, 0).exists()
    assert not any(w2.out.rglob(_src(w2, 0).name))


def test_marked_photo_is_copied_into_the_trash_when_copying(tmp_path):
    w = build_world(tmp_path, n=2)
    approve_delete(w, w.ids[0])
    p = w.planner.build_plan(organize_root=w.out, operation="copy")
    rep = w.planner.execute_plan(p, trash=True)
    assert rep.marked_left_out == 0 and _src(w, 0).exists()  # the original stays put
    trash = w.out / ".trash" / rep.transaction_id
    assert list(trash.rglob(_src(w, 0).name))
    assert not [f for f in w.out.rglob(_src(w, 0).name) if trash not in f.parents]


def test_marked_burst_alternate_is_trashed_unmarked_stays(tmp_path):
    from photocure.db import models as M
    w = build_world(tmp_path, n=3)
    with get_session(w.db) as s:
        bg = M.BurstGroup(name="Burst_t", image_count=3, best_shot_id=w.ids[0])
        s.add(bg)
        s.flush()
        for i, iid in enumerate(w.ids):
            r = s.get(M.ImageRecord, iid)
            r.burst_id, r.is_best_shot, r.is_alternate = bg.id, i == 0, i != 0
        s.commit()
    approve_delete(w, w.ids[1])
    p = w.planner.build_plan(organize_root=w.out, operation="move")
    rep = w.planner.execute_plan(p, trash=True)
    trash = w.out / ".trash" / rep.transaction_id
    assert list(trash.rglob(_src(w, 1).name))
    assert not list(trash.rglob(_src(w, 2).name)) and not list(trash.rglob(_src(w, 0).name))
    assert rep.unapproved_junk == 0


def test_trashed_photo_removed_from_index_and_undo_restores_it(tmp_path):
    w = build_world(tmp_path, n=3)
    set_bucket(w, w.ids[0], "graphic")
    approve_delete(w, w.ids[0])
    set_bucket(w, w.ids[1], "food")
    p = w.planner.build_plan(organize_root=w.out, operation="move")
    rep = w.planner.execute_plan(p, trash=True)

    # Trashed photo is marked missing/trashed and excluded from stats and subsequent plans
    trashed = get_img(w, w.ids[0])
    assert trashed.scan_status == "missing" and trashed.route_status == "trashed"
    with get_session(w.db) as s:
        assert PhotoRepository(s).stats()["total_images"] == 2
        assert PhotoRepository(s).count_category("graphic") == 0

    p2 = w.planner.build_plan(organize_root=w.out, operation="move", persist=False)
    assert p2.total == 0  # applied and trashed photos are not planned again
    assert w.ids[0] not in {a.image_id for a in p2.actions}

    # Undo restores the trashed photo to disk and back into the active index
    undo = w.planner.undo(rep.transaction_id)
    assert undo.reversed == 3
    restored = get_img(w, w.ids[0])
    assert restored.scan_status == "scanned" and restored.route_status == "unrouted"
    assert restored.path == str(_src(w, 0)) and _src(w, 0).exists()
    with get_session(w.db) as s:
        assert PhotoRepository(s).stats()["total_images"] == 3
        assert PhotoRepository(s).count_category("graphic") == 1
