"""Apply and undo drive a photo's stage; copy repoints the row; trash works for move and copy."""
from __future__ import annotations

from pathlib import Path

import pytest

from photocure.config import CuratorConfig
from photocure.db.models import ImageRecord
from photocure.db.repository import PhotoRepository
from photocure.db.session import get_session, init_db
from photocure.engine.action_engine import ActionEngine
from photocure.engine.rollback import RollbackEngine
from photocure.rules.organizer import PlannedAction
from photocure.rules.taxonomy import CategoryType
from photocure.scanner.file_scanner import FileScanner

from .conftest import make_image


@pytest.fixture()
def world(tmp_db, tmp_path):
    cfg = CuratorConfig()
    cfg.db_path = tmp_db
    cfg.device.device = "cpu"
    src = tmp_path / "src"
    for i in range(1, 5):
        make_image(src / f"IMG_2024{i:04d}_153045.jpg", rng_seed=i)
    init_db(tmp_db, drop=True)
    FileScanner(cfg).scan_directory(src, db_path=tmp_db, show_progress=False)
    with get_session(tmp_db) as s:
        recs = PhotoRepository(s).all_images()
        for r in recs:
            r.infer_status = "inferred"
        s.commit()
        rows = [(r.id, Path(r.path)) for r in recs]
    root = tmp_path / "Library"
    return type("W", (), {"cfg": cfg, "db": tmp_db, "rows": rows, "root": root, "src": src})


def _actions(w, op="move"):
    return [PlannedAction(image_id=i, source=p, target=w.root / "People" / p.name,
                          operation=op, category=CategoryType.PEOPLE, reason="t")
            for i, p in w.rows]


def _mark_delete(w, image_id):
    with get_session(w.db) as s:
        s.get(ImageRecord, image_id).delete_approved = True
        s.commit()


def _row(w, image_id):
    with get_session(w.db) as s:
        r = s.get(ImageRecord, image_id)
        return {c: getattr(r, c) for c in ("path", "stage", "origin_path", "scan_status",
                                           "route_status", "target_path")}


def _run(w, op="move", trash=False):
    return ActionEngine(w.cfg, db_path=w.db).execute(_actions(w, op), organize_root=w.root, trash=trash)


def test_move_apply_sets_library_stage_and_origin(world):
    w = world
    iid, old = w.rows[0]
    rep = _run(w)
    assert rep.failures == 0
    row = _row(w, iid)
    assert row["stage"] == "library"
    assert row["origin_path"] == str(old)
    assert Path(row["path"]).is_relative_to(w.root)
    assert not old.exists()


def test_copy_apply_repoints_row_to_library_copy(world):
    w = world
    iid, old = w.rows[0]
    rep = _run(w, "copy")
    assert rep.failures == 0
    row = _row(w, iid)
    assert old.exists()
    assert Path(row["path"]).is_relative_to(w.root) and Path(row["path"]).exists()
    assert row["stage"] == "library"
    assert row["origin_path"] == str(old)


def test_copy_with_trash_copies_junk_into_trash(world):
    w = world
    iid, old = w.rows[0]
    _mark_delete(w, iid)
    rep = _run(w, "copy", trash=True)
    assert rep.failures == 0 and rep.marked_left_out == 0
    row = _row(w, iid)
    assert row["stage"] == "trash"
    assert row["scan_status"] == "missing"
    assert (w.root / ".trash") in Path(row["path"]).parents
    assert Path(row["path"]).exists() and old.exists()
    assert row["origin_path"] == str(old)


def test_move_with_trash_moves_junk_into_the_trash_stage(world):
    w = world
    iid, old = w.rows[0]
    _mark_delete(w, iid)
    _run(w, "move", trash=True)
    row = _row(w, iid)
    assert row["stage"] == "trash" and row["scan_status"] == "missing"
    assert row["route_status"] == "trashed"
    assert Path(row["path"]).exists() and not old.exists()


def test_apply_without_trash_leaves_marked_photo_in_new(world):
    w = world
    iid, old = w.rows[0]
    _mark_delete(w, iid)
    rep = _run(w, "move", trash=False)
    assert rep.marked_left_out == 1
    row = _row(w, iid)
    assert row["stage"] == "new" and old.exists()


@pytest.mark.parametrize("op", ["move", "copy"])
def test_undo_returns_rows_to_new(world, op):
    w = world
    iid, old = w.rows[0]
    rep = _run(w, op)
    lib_path = Path(_row(w, iid)["path"])
    out = RollbackEngine(w.cfg, db_path=w.db).rollback_transaction(rep.transaction_id)
    assert out.failures == 0
    row = _row(w, iid)
    assert row["stage"] == "new"
    assert row["path"] == str(old)
    assert row["origin_path"] is None
    assert row["route_status"] == "unrouted"
    assert old.exists() and not lib_path.exists()


def test_cross_device_move_still_lands_in_library(world, monkeypatch):
    w = world
    iid, old = w.rows[0]
    monkeypatch.setattr(ActionEngine, "_same_device", staticmethod(lambda a, b: False))
    rep = _run(w)
    assert rep.failures == 0
    row = _row(w, iid)
    assert row["stage"] == "library"
    assert Path(row["path"]).is_relative_to(w.root) and Path(row["path"]).exists()
    assert not old.exists()


def test_reapplying_an_applied_photo_keeps_it_in_library(world):
    w = world
    iid, old = w.rows[0]
    _run(w)
    lib_path = Path(_row(w, iid)["path"])
    again = [PlannedAction(image_id=iid, source=lib_path, target=lib_path,
                           operation="move", category=CategoryType.PEOPLE, reason="t")]
    rep = ActionEngine(w.cfg, db_path=w.db).execute(again, organize_root=w.root)
    assert rep.in_place == 1
    row = _row(w, iid)
    assert row["stage"] == "library" and row["path"] == str(lib_path)
