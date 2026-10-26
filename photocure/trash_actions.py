"""Trash stage actions: restore into the library, return to the source folder, delete for good."""
from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from sqlmodel import select

from .config import CuratorConfig
from .db import models as M
from .db.models import STAGE_LIBRARY, STAGE_NEW, STAGE_TRASH
from .db.repository import PhotoRepository
from .db.session import get_session
from .engine.action_engine import ActionEngine
from .engine.locks import LEDGER_WRITING
from .library import library_target, require_library_root, trash_dir
from .library_actions import _category_enum, _effective_category, _move_sidecar
from .people_actions import attach_faces_to_library
from .rules.organizer import PlannedAction


def _trashed(s, ids: list[int] | None) -> list[M.ImageRecord]:
    q = select(M.ImageRecord).where(M.ImageRecord.stage == STAGE_TRASH,
                                    M.ImageRecord.route_status != "purged")
    if ids is not None:
        q = q.where(M.ImageRecord.id.in_(list(ids)))
    return list(s.exec(q.order_by(M.ImageRecord.id)).all())


def _clear_marks(dbp: Path, image_ids: list[int]) -> None:
    """The engine leaves out photos marked for delete, so the mark is cleared before a move."""
    with get_session(dbp) as s:
        for iid in image_ids:
            s.get(M.ImageRecord, iid).delete_approved = False
        s.commit()


def _tick(on_progress: Callable | None, done: int, total: int, name: str) -> None:
    if on_progress:
        on_progress(done, total, name)


def restore(dbp: Path, cfg: CuratorConfig, ids: list[int], *, cancel_check: Callable | None = None,
            on_progress: Callable | None = None, root: Path | None = None) -> dict:
    """Move trashed photos into the library by their current category and re-attach their faces."""
    root = Path(root) if root else require_library_root(dbp, cfg)
    skipped = {"file_missing": 0}
    failed, errors = 0, []
    actions: list[PlannedAction] = []
    with get_session(dbp) as s:
        recs = _trashed(s, ids)
        for rec in recs:
            src = Path(rec.path)
            if not src.exists():
                skipped["file_missing"] += 1
                continue
            category = _effective_category(rec)
            actions.append(PlannedAction(
                image_id=rec.id, source=src, target=library_target(cfg, root, rec, category),
                operation="move", category=_category_enum(cfg, category), reason="restored from trash",
                reuse_existing=False))
    txn = ""
    restored = 0
    if actions:
        _clear_marks(dbp, [a.image_id for a in actions])
        rep = ActionEngine(cfg, db_path=dbp).execute(actions, organize_root=root, cancel_check=cancel_check)
        txn = rep.transaction_id
        failed, errors = rep.failures, list(rep.errors)[:20]
        with get_session(dbp) as s:
            for n, a in enumerate(actions, 1):
                final = rep.resolved_targets.get(a.image_id)
                if final is None:
                    s.get(M.ImageRecord, a.image_id).delete_approved = True  # it stays in the trash, still marked
                    continue
                rec = s.get(M.ImageRecord, a.image_id)
                rec.scan_status = "scanned"
                rec.delete_approved = False
                attach_faces_to_library(s, rec.id)
                _move_sidecar(a.source, Path(final))
                restored += 1
                _tick(on_progress, n, len(actions), rec.filename)
            PhotoRepository(s).prune_bursts()
            s.commit()
    reasons = {k: v for k, v in skipped.items() if v}
    return {"ok": True, "restored": restored, "skipped": sum(reasons.values()) + (len(actions) - restored),
            "skipped_reasons": reasons, "failed": failed, "errors": errors, "transaction_id": txn}


def return_to_origin(dbp: Path, cfg: CuratorConfig, ids: list[int] | None, *,
                     cancel_check: Callable | None = None, on_progress: Callable | None = None) -> dict:
    """Move trashed photos back to the folder they were first scanned from; they count as new again."""
    skipped = {"origin_missing": 0, "origin_occupied": 0, "file_missing": 0}
    failed, errors = 0, []
    actions: list[PlannedAction] = []
    with get_session(dbp) as s:
        for rec in _trashed(s, ids):
            src = Path(rec.path)
            origin = Path(rec.origin_path) if rec.origin_path else None
            if origin is None or not origin.parent.is_dir():
                skipped["origin_missing"] += 1
            elif os.path.lexists(origin):
                skipped["origin_occupied"] += 1
            elif not src.exists():
                skipped["file_missing"] += 1
            else:
                actions.append(PlannedAction(
                    image_id=rec.id, source=src, target=origin, operation="move",
                    category=_category_enum(cfg, _effective_category(rec)), reason="returned to origin",
                    reuse_existing=False))
    txn = ""
    returned = 0
    if actions:
        _clear_marks(dbp, [a.image_id for a in actions])
        rep = ActionEngine(cfg, db_path=dbp).execute(actions, cancel_check=cancel_check)
        txn = rep.transaction_id
        failed, errors = rep.failures, list(rep.errors)[:20]
        with get_session(dbp) as s:
            for n, a in enumerate(actions, 1):
                if a.image_id not in rep.resolved_targets:
                    s.get(M.ImageRecord, a.image_id).delete_approved = True  # it stays in the trash, still marked
                    continue
                rec = s.get(M.ImageRecord, a.image_id)
                rec.stage, rec.route_status, rec.scan_status = STAGE_NEW, "unrouted", "scanned"
                rec.infer_status, rec.origin_path, rec.target_path = "pending", None, None
                rec.delete_approved = False
                _move_sidecar(a.source, Path(rep.resolved_targets[a.image_id]))
                returned += 1
                _tick(on_progress, n, len(actions), rec.filename)
            s.commit()
    reasons = {k: v for k, v in skipped.items() if v}
    return {"ok": True, "returned": returned, "skipped": sum(reasons.values()) + (len(actions) - returned),
            "skipped_reasons": reasons, "failed": failed, "errors": errors, "transaction_id": txn}


def _prune_empty_dirs(start: Path, stop: Path) -> None:
    cur = start
    while cur != stop and stop in cur.parents:
        try:
            cur.rmdir()
        except OSError:
            return
        cur = cur.parent


def delete_permanently(dbp: Path, cfg: CuratorConfig, ids: list[int], *,
                       cancel_check: Callable | None = None, on_progress: Callable | None = None,
                       root: Path | None = None) -> dict:
    """Unlink trashed files for good; the rows stay as purged so a rescan can tell."""
    root = Path(root).resolve() if root else require_library_root(dbp, cfg)
    with LEDGER_WRITING:  # no other file action runs while files are being removed
        return _delete_locked(dbp, ids, root, cancel_check, on_progress)


def _delete_locked(dbp: Path, ids: list[int], root: Path, cancel_check, on_progress) -> dict:
    trash_root = trash_dir(root)
    deleted = skipped = freed = 0
    with get_session(dbp) as s:
        recs = _trashed(s, ids)
        for n, rec in enumerate(recs, 1):
            if cancel_check and cancel_check():
                break
            path = Path(rec.path)
            if trash_root not in path.parents:  # never unlink something that is not in the trash folder
                skipped += 1
                continue
            if path.exists():
                try:
                    freed += path.stat().st_size
                    path.unlink()
                except OSError:
                    continue
            sidecar = path.with_suffix(".xmp")
            if sidecar.exists():
                try:
                    sidecar.unlink()
                except OSError:
                    pass
            _prune_empty_dirs(path.parent, trash_root)
            rec.route_status, rec.scan_status = "purged", "missing"
            for e in s.exec(select(M.ActionLedger).where(M.ActionLedger.image_id == rec.id,
                                                         M.ActionLedger.status == "committed")).all():
                if trash_root in Path(e.target_path).parents:
                    e.status = "purged"
            deleted += 1
            _tick(on_progress, n, len(recs), rec.filename)
        s.commit()
    return {"ok": True, "deleted": deleted, "bytes": freed, "skipped": skipped,
            "skipped_reasons": {"outside_trash": skipped} if skipped else {}}


def empty_trash(dbp: Path, cfg: CuratorConfig, *, cancel_check: Callable | None = None,
                on_progress: Callable | None = None, root: Path | None = None) -> dict:
    """Delete everything in the trash, plus files in the trash folder that no photo row points at."""
    root = Path(root).resolve() if root else require_library_root(dbp, cfg)
    trash_root = trash_dir(root)
    with LEDGER_WRITING:
        with get_session(dbp) as s:
            ids = [r.id for r in _trashed(s, None)]
        out = _delete_locked(dbp, ids, root, cancel_check, on_progress)
        if trash_root.is_dir():
            with get_session(dbp) as s:
                tracked = {r.path for r in s.exec(select(M.ImageRecord).where(M.ImageRecord.route_status != "purged")).all()}
            for p in sorted(trash_root.rglob("*"), key=lambda q: len(q.parts), reverse=True):
                if p.is_file() and str(p) not in tracked:
                    try:
                        out["bytes"] += p.stat().st_size
                        p.unlink()
                    except OSError:
                        pass
                elif p.is_dir():
                    try:
                        p.rmdir()
                    except OSError:
                        pass
            try:
                trash_root.rmdir()
            except OSError:
                pass
    return out
