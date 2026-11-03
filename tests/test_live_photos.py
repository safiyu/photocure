"""Tests for Live Photo and companion video/sidecar handling."""
import os
from pathlib import Path

import pytest

from photocure.config import CuratorConfig
from photocure.db.repository import PhotoRepository
from photocure.db.session import get_session, init_db
from photocure.engine.action_engine import ActionEngine, find_companion_files, new_transaction_id
from photocure.engine.rollback import RollbackEngine
from photocure.library_actions import _move_sidecar
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


def test_library_move_sidecar_moves_live_photo(tmp_path):
    src = tmp_path / "A" / "photo.heic"
    src.parent.mkdir()
    src.write_bytes(b"heic")
    mov = tmp_path / "A" / "photo.mov"
    mov.write_bytes(b"mov")

    dst = tmp_path / "B" / "photo.heic"
    dst.parent.mkdir()

    _move_sidecar(src, dst)
    assert not mov.exists()
    assert (tmp_path / "B" / "photo.mov").exists()
    assert (tmp_path / "B" / "photo.mov").read_bytes() == b"mov"
