"""Tests for Live Photo and companion video/sidecar handling."""
import os
from pathlib import Path

import pytest

from photocure.config import CuratorConfig
from photocure.db.repository import PhotoRepository
from photocure.db.session import get_session, init_db
from photocure.engine.action_engine import ActionEngine, find_companion_files, new_transaction_id
from photocure.engine.rollback import RollbackEngine
from photocure.rules.organizer import PlannedAction
from tests.conftest import make_image


def test_find_companion_files(tmp_path):
    img = tmp_path / "IMG_1001.HEIC"
    img.write_bytes(b"image")
    mov = tmp_path / "IMG_1001.MOV"
    mov.write_bytes(b"live video")
    xmp = tmp_path / "IMG_1001.xmp"
    xmp.write_bytes(b"<xmp/>")

    comps = find_companion_files(img)
    comp_names = {c.name for c in comps}
    assert "IMG_1001.MOV" in comp_names
    assert "IMG_1001.xmp" in comp_names


def test_action_engine_moves_live_photo_and_rolls_back(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path, drop=True)
    cfg = CuratorConfig()
    cfg.db_path = db_path

    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    src_dir.mkdir()
    dst_dir.mkdir()

    img_path = make_image(src_dir / "IMG_20240101_120000.jpg")
    mov_path = src_dir / "IMG_20240101_120000.mov"
    mov_path.write_bytes(b"mov payload")

    with get_session(db_path) as s:
        from photocure.db.models import ImageRecord
        rec = ImageRecord(path=str(img_path), filename=img_path.name, file_size=img_path.stat().st_size,
                          quick_hash="qh1", sha256="sha1")
        s.add(rec)
        s.commit()
        s.refresh(rec)
        img_id = rec.id

    action = PlannedAction(
        source=img_path,
        target=dst_dir / "Photos" / "2024" / "IMG_20240101_120000.jpg",
        operation="move",
        category="photo",
        image_id=img_id,
    )

    engine = ActionEngine(cfg, db_path=db_path)
    rep = engine.execute([action], organize_root=dst_dir)
    assert rep.executed == 1

    # Verify both image and companion MOV moved to destination
    assert not img_path.exists()
    assert not mov_path.exists()
    dst_img = dst_dir / "Photos" / "2024" / "IMG_20240101_120000.jpg"
    dst_mov = dst_dir / "Photos" / "2024" / "IMG_20240101_120000.mov"
    assert dst_img.exists()
    assert dst_mov.exists()
    assert dst_mov.read_bytes() == b"mov payload"

    # Now verify RollbackEngine brings both files back to original src_dir!
    rb = RollbackEngine(cfg, db_path=db_path)
    rb_rep = rb.rollback_transaction(rep.transaction_id)
    assert rb_rep.reversed >= 1

    assert img_path.exists()
    assert mov_path.exists()
    assert mov_path.read_bytes() == b"mov payload"
    assert not dst_img.exists()
    assert not dst_mov.exists()


def _engine_env(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path, drop=True)
    cfg = CuratorConfig()
    cfg.db_path = db_path
    return cfg, db_path


def _add_photo(db_path, img_path, qh):
    from photocure.db.models import ImageRecord
    with get_session(db_path) as s:
        rec = ImageRecord(path=str(img_path), filename=img_path.name, file_size=img_path.stat().st_size,
                          quick_hash=qh, sha256=f"sha-{qh}")
        s.add(rec)
        s.commit()
        s.refresh(rec)
        return rec.id


def _two_live_photos_same_stem(tmp_path):
    a = make_image(tmp_path / "A" / "IMG_001.jpg", rng_seed=1)
    (tmp_path / "A" / "IMG_001.MOV").write_bytes(b"video A")
    b = make_image(tmp_path / "B" / "IMG_001.jpg", rng_seed=2)
    (tmp_path / "B" / "IMG_001.MOV").write_bytes(b"video B")
    return a, b


def test_companion_collision_gets_sequence_name(tmp_path):
    cfg, db_path = _engine_env(tmp_path)
    a, b = _two_live_photos_same_stem(tmp_path)
    out = tmp_path / "out"
    ids = [_add_photo(db_path, a, "qa"), _add_photo(db_path, b, "qb")]
    actions = [PlannedAction(source=p, target=out / "Photos" / "IMG_001.jpg", operation="move",
                             category="photo", image_id=i) for p, i in zip((a, b), ids)]

    rep = ActionEngine(cfg, db_path=db_path).execute(actions, organize_root=out)

    assert rep.executed == 2 and rep.companion_failures == 0
    movs = sorted(p.name for p in out.rglob("*.mov"))
    assert movs == ["IMG_001 (1).mov", "IMG_001.mov"]
    assert {p.read_bytes() for p in out.rglob("*.mov")} == {b"video A", b"video B"}
    assert not list((tmp_path / "B").glob("*.MOV"))


def test_companion_ledger_row_has_no_image_id(tmp_path):
    from sqlmodel import select
    from photocure.db.models import ActionLedger
    cfg, db_path = _engine_env(tmp_path)
    img = make_image(tmp_path / "A" / "IMG_002.jpg")
    (tmp_path / "A" / "IMG_002.mov").write_bytes(b"v")
    img_id = _add_photo(db_path, img, "q2")
    out = tmp_path / "out"
    rep = ActionEngine(cfg, db_path=db_path).execute(
        [PlannedAction(source=img, target=out / "IMG_002.jpg", operation="move", category="photo", image_id=img_id)],
        organize_root=out)

    with get_session(db_path) as s:
        rows = s.exec(select(ActionLedger).where(ActionLedger.transaction_id == rep.transaction_id)
                      .order_by(ActionLedger.id)).all()
        parent = [r for r in rows if r.source_path.endswith(".jpg")][0]
        comp = [r for r in rows if r.source_path.endswith(".mov")]
        assert comp and comp[0].image_id is None and comp[0].companion_of == parent.id
        assert comp[0].status == "committed"


def test_find_companion_files_dedups_case_variants(tmp_path):
    img = tmp_path / "IMG.jpg"
    img.write_bytes(b"i")
    (tmp_path / "IMG.MOV").write_bytes(b"m")
    probe = tmp_path / "IMG.mov"
    if not probe.exists():  # case-sensitive filesystem: nothing to dedup
        pytest.skip("case-sensitive filesystem")
    assert len(find_companion_files(img)) == 1
