from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

import pytest

from photocure.db.models import ActionLedger, FaceDetection, ImageRecord, PersonCluster
from photocure.db.repository import PhotoRepository
from photocure.db.session import get_session
from photocure.engine.action_engine import ActionEngine, _copy_or_cleanup
from photocure.engine.rollback import RollbackEngine
from photocure.scanner.file_scanner import FileScanner
from photocure.scanner.hasher import compute_sha256
from photocure.scanner.xmp import update_xmp_sidecar

from .world import build_world, get_img


def test_move_onto_stale_missing_row_path(tmp_path):
    w = build_world(tmp_path, n=1)
    p = w.planner.build_plan(organize_root=w.out, operation="move")
    target = str(p.actions[0].target)
    with get_session(w.db) as s:
        s.add(ImageRecord(path=target, filename="x.jpg", scan_status="missing"))
    rep = w.planner.execute_plan(p, force=True)
    assert rep.failures == 0 and rep.executed == 1
    assert get_img(w, w.ids[0]).path == target
    with get_session(w.db) as s:
        rows = PhotoRepository(s).ledger_by_txn(rep.transaction_id)
        assert [r.status for r in rows] == ["committed"]


def test_post_move_commit_failure_records_moved(tmp_path, monkeypatch):
    w = build_world(tmp_path, n=1)
    p = w.planner.build_plan(organize_root=w.out, operation="move")

    def boom(session, target, keep_id):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(ActionEngine, "_retire_stale_row", staticmethod(boom))
    rep = w.planner.execute_plan(p)
    monkeypatch.undo()
    assert rep.failures == 1
    with get_session(w.db) as s:
        rows = PhotoRepository(s).ledger_by_txn(rep.transaction_id)
        assert [r.status for r in rows] == ["committed"]
    assert w.planner.undo(rep.transaction_id).reversed == 1
    assert w.paths[0].exists()


def test_reconcile_never_deletes_only_copy(tmp_path):
    target = tmp_path / "t.jpg"
    target.write_bytes(b"edited")
    with get_session(tmp_path / "r.db") as s:
        repo = PhotoRepository(s)
        from photocure.db.session import init_db
        init_db(tmp_path / "r.db")
        e = repo.ledger_add("tx", None, str(tmp_path / "gone.jpg"), str(target), "move", "deadbeef", 6)
        RollbackEngine.__new__(RollbackEngine)._reconcile_pending("tx", repo, type("R", (), {"notes": []})())
        assert target.exists()
        assert e.status == "pending"


def test_partial_copy_removed_on_failure(tmp_path, monkeypatch):
    src = tmp_path / "s.bin"
    src.write_bytes(b"x" * 100)
    tgt = tmp_path / "t.bin"

    def boom(a, b, follow_symlinks=True):
        Path(b).write_bytes(b"x")
        raise OSError("ENOSPC")

    monkeypatch.setattr(shutil, "copy2", boom)
    with pytest.raises(OSError):
        _copy_or_cleanup(src, tgt)
    assert not tgt.exists()
    tgt.write_bytes(b"keep")
    with pytest.raises(OSError):
        _copy_or_cleanup(src, tgt)
    assert tgt.read_bytes() == b"x"  # pre-existing file is left alone (not unlinked)


def test_resolve_group_clears_keeper_delete_approval(tmp_path):
    from photocure.duplicates import resolve_group
    w = build_world(tmp_path, n=2)
    with get_session(w.db) as s:
        for i in w.ids:
            s.get(ImageRecord, i).delete_approved = True
        resolve_group(s, keep_id=w.ids[0], discard_ids=[w.ids[1]], kind="exact")
    assert get_img(w, w.ids[0]).delete_approved is False


def test_modified_file_clears_approval_and_face_counts(tmp_path):
    w = build_world(tmp_path, n=1)
    with get_session(w.db) as s:
        pc = PersonCluster(name=None, face_count=1, centroid_json=b"")
        s.add(pc)
        s.flush()
        s.add(FaceDetection(image_id=w.ids[0], person_id=pc.id, bbox_x1=0, bbox_y1=0, bbox_x2=1, bbox_y2=1, confidence=0.9))
        r = s.get(ImageRecord, w.ids[0])
        r.delete_approved = True
        r.manual_bucket = "junk"
        pid = pc.id
    f = w.paths[0]
    f.write_bytes(f.read_bytes() + b"\0")
    os.utime(f, (1, 1))
    FileScanner(w.cfg).scan_directory(w.root, db_path=w.db, show_progress=False)
    r = get_img(w, w.ids[0])
    assert r.delete_approved is False and r.manual_bucket is None
    with get_session(w.db) as s:
        cluster = s.get(PersonCluster, pid)
        assert cluster is None or cluster.face_count == 0


def test_rescan_keeps_ui_rating_without_sidecar(tmp_path):
    w = build_world(tmp_path, n=1)
    with get_session(w.db) as s:
        s.get(ImageRecord, w.ids[0]).rating = 4
    f = w.paths[0]
    os.utime(f, (1, 1))
    FileScanner(w.cfg).scan_directory(w.root, db_path=w.db, show_progress=False)
    assert get_img(w, w.ids[0]).rating == 4


def test_xmp_keeps_xpacket_and_mode(tmp_path):
    photo = tmp_path / "a.jpg"
    photo.write_bytes(b"x")
    sc = tmp_path / "a.xmp"
    sc.write_bytes(b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
                   b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
                   b'<rdf:Description rdf:about=""/></rdf:RDF></x:xmpmeta>\n<?xpacket end="w"?>')
    os.chmod(sc, 0o640)
    ok, _ = update_xmp_sidecar(photo, rating=3)
    assert ok
    data = sc.read_bytes()
    assert data.startswith(b"<?xpacket begin=") and data.rstrip().endswith(b'<?xpacket end="w"?>')
    assert b"<?xml" not in data
    assert stat.S_IMODE(sc.stat().st_mode) == 0o640


def test_sha256_chunked(tmp_path):
    import hashlib
    f = tmp_path / "f"
    f.write_bytes(os.urandom(3 * 1024 * 1024 + 5))
    assert compute_sha256(f) == hashlib.sha256(f.read_bytes()).hexdigest()
