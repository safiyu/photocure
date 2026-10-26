"""A rescan skips files whose content is already in the library or the trash."""
from __future__ import annotations

from pathlib import Path

import pytest
from sqlmodel import select

from photocure.config import CuratorConfig
from photocure.db.models import ImageRecord
from photocure.db.repository import PhotoRepository
from photocure.db.session import get_session, init_db
from photocure.engine.action_engine import ActionEngine
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
    return type("W", (), {"cfg": cfg, "db": tmp_db, "rows": rows, "root": tmp_path / "Library", "src": src})


def _apply(w, op="copy", trash=False):
    acts = [PlannedAction(image_id=i, source=p, target=w.root / "People" / p.name,
                          operation=op, category=CategoryType.PEOPLE, reason="t")
            for i, p in w.rows]
    rep = ActionEngine(w.cfg, db_path=w.db).execute(acts, organize_root=w.root, trash=trash)
    assert rep.failures == 0
    return rep


def _count(w):
    with get_session(w.db) as s:
        return len(s.exec(select(ImageRecord)).all())


def _rescan(w):
    return FileScanner(w.cfg).scan_directory(w.src, db_path=w.db, show_progress=False)


def test_rescan_of_copied_source_adds_no_row(world):
    w = world
    _apply(w, "copy")
    res = _rescan(w)
    assert _count(w) == 4
    assert res.new == 0 and res.skipped_library == 4


def test_rescan_of_trashed_copy_adds_no_row(world):
    w = world
    with get_session(w.db) as s:
        s.get(ImageRecord, w.rows[0][0]).delete_approved = True
        s.commit()
    _apply(w, "copy", trash=True)
    res = _rescan(w)
    assert _count(w) == 4
    assert res.new == 0 and res.skipped_library == 4


def test_different_file_with_same_name_is_still_new(world):
    w = world
    _apply(w, "copy")
    name = w.rows[0][1].name
    make_image(w.src / name, rng_seed=99)  # same name, different content
    res = _rescan(w)
    assert res.new == 1 and res.skipped_library == 3
    assert _count(w) == 5


def test_purged_photos_no_longer_block_a_rescan(world):
    w = world
    _apply(w, "copy")
    with get_session(w.db) as s:
        r = s.get(ImageRecord, w.rows[0][0])
        r.stage, r.route_status, r.scan_status = "trash", "purged", "missing"
        s.commit()
    res = _rescan(w)
    assert res.new == 1 and res.skipped_library == 3
