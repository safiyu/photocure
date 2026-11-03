"""Single-photo moves inside the library, each recorded in the ledger so undo keeps working."""
from __future__ import annotations

import shutil
from pathlib import Path

from sqlmodel import select

from .config import CuratorConfig
from .db import models as M
from .db.models import STAGE_LIBRARY, STAGE_TRASH
from .db.session import get_session
from .engine.action_engine import ActionEngine
from .library import library_target, require_library_root
from .people_actions import detach_faces
from .rules.organizer import PlannedAction
from .rules.taxonomy import CategoryType, get_taxonomy


def _category_enum(cfg: CuratorConfig, category: str) -> CategoryType:
    return get_taxonomy(cfg).category_of(category) or CategoryType.UNKNOWN


def _move_sidecar(src: Path, dst: Path) -> None:
    # Standard XMP sidecars
    for ext in (".xmp", ".XMP"):
        xmp = src.with_suffix(ext)
        if xmp.exists():
            try:
                shutil.move(xmp, dst.with_suffix(ext.lower()))
            except OSError:
                pass
    # Live Photo companion video files (.mov, .mp4)
    for ext in (".mov", ".MOV", ".mp4", ".MP4"):
        vid = src.with_suffix(ext)
        if vid.exists() and vid.resolve() != src.resolve():
            try:
                shutil.move(vid, dst.with_suffix(ext.lower()))
            except OSError:
                pass


def _effective_category(rec: M.ImageRecord) -> str:
    return rec.manual_bucket or rec.category or "unknown"


def move_to_category(dbp: Path, cfg: CuratorConfig, image_id: int, category: str) -> dict:
    """Move a library photo into its category folder, creating the folder when needed."""
    root = require_library_root(dbp, cfg)
    with get_session(dbp) as s:
        rec = s.get(M.ImageRecord, image_id)
        if rec is None:
            raise LookupError(f"no photo with id {image_id}")
        if rec.stage != STAGE_LIBRARY:
            raise ValueError("only library photos move to a category folder")
        src = Path(rec.path)
        target = library_target(cfg, root, rec, category)
    if not src.exists():
        raise FileNotFoundError(f"file is missing on disk: {src}")
    if target == src:
        return {"ok": True, "moved": False, "path": str(src)}
    action = PlannedAction(image_id=image_id, source=src, target=target, operation="move",
                           category=_category_enum(cfg, category), reason="manual category change")
    rep = ActionEngine(cfg, db_path=dbp).execute([action], organize_root=root)
    if rep.failures:
        raise OSError(rep.errors[0])
    final = Path(rep.resolved_targets.get(image_id, target))
    _move_sidecar(src, final)
    return {"ok": True, "moved": True, "transaction_id": rep.transaction_id, "path": str(final)}


def trash_image(dbp: Path, cfg: CuratorConfig, image_id: int) -> dict:
    """Send a photo to the trash. Library photos move into <root>/.trash/<txn>/; new photos are
    only marked for delete and leave when the plan is applied."""
    root = require_library_root(dbp, cfg)
    with get_session(dbp) as s:
        rec = s.get(M.ImageRecord, image_id)
        if rec is None:
            raise LookupError(f"no photo with id {image_id}")
        rec.delete_approved = True
        if rec.stage != STAGE_LIBRARY:
            s.commit()
            return {"ok": True, "trashed": False, "marked": True}
        src = Path(rec.path)
        category = _effective_category(rec)
        target = library_target(cfg, root, rec, category)
        s.commit()

    txn = ""
    if src.exists():
        action = PlannedAction(image_id=image_id, source=src, target=target, operation="move",
                               category=_category_enum(cfg, category), reason="deleted by hand")
        rep = ActionEngine(cfg, db_path=dbp).execute([action], organize_root=root, trash=True)
        if rep.failures:
            raise OSError(rep.errors[0])
        txn = rep.transaction_id
        _move_sidecar(src, Path(rep.resolved_targets.get(image_id, target)))
        with get_session(dbp) as s:  # the mark was only set to route this move; undoing it must unmark
            for e in s.exec(select(M.ActionLedger).where(M.ActionLedger.transaction_id == txn)).all():
                e.prev_delete_approved = False
            s.commit()

    with get_session(dbp) as s:
        rec = s.get(M.ImageRecord, image_id)
        if rec.stage != STAGE_TRASH:  # the file was already gone: trash the row anyway
            rec.stage, rec.route_status, rec.scan_status = STAGE_TRASH, "trashed", "missing"
        detach_faces(s, image_id)
        from .db.repository import PhotoRepository
        PhotoRepository(s).prune_bursts()
        s.commit()
    return {"ok": True, "trashed": True, "transaction_id": txn}


def refile_library(dbp: Path, cfg: CuratorConfig, *, root: Path | None = None,
                   cancel_check=None) -> dict:
    """Decide the category of every library photo again and move the files whose folder changed.

    Used after the library was analysed again, when the AI may see things differently. Photos you
    sorted by hand, kept (locked) or marked for delete stay where they are, and so do frames that
    sit in a burst: those are settled by finalizing the burst. The moves are one transaction, so
    Undo puts every file back.
    """
    from sqlalchemy.orm import selectinload

    from .db.repository import PhotoRepository
    from .rules.decision_matrix import DecisionMatrix
    from .rules.organizer import FolderOrganizer

    root = Path(root).resolve() if root else require_library_root(dbp, cfg)
    changes: dict[int, tuple[str, str]] = {}  # image id -> (category, subcategory) once settled
    actions: list[PlannedAction] = []
    unchanged = 0
    with get_session(dbp) as s:
        matrix = DecisionMatrix(cfg)
        organizer = FolderOrganizer(cfg, organize_root=root)
        organizer.set_bursts(list(PhotoRepository(s).all_bursts()))
        rows = s.exec(select(M.ImageRecord).where(
            M.ImageRecord.stage == STAGE_LIBRARY,
            M.ImageRecord.scan_status != "missing",
            M.ImageRecord.locked == False,  # noqa: E712
            M.ImageRecord.delete_approved == False,  # noqa: E712
            M.ImageRecord.manual_bucket.is_(None),
            M.ImageRecord.is_alternate == False,  # noqa: E712
        ).options(selectinload(M.ImageRecord.semantic), selectinload(M.ImageRecord.quality),
                  selectinload(M.ImageRecord.faces))).all()
        for r in rows:
            if cancel_check and cancel_check():
                break
            dec = matrix.evaluate_record(r)
            cat = dec.category.value
            target = organizer.full_target(r, dec)
            src = Path(r.path)
            if src.parent == target.parent or not src.exists():
                if (r.category, r.subcategory or "") != (cat, dec.subcategory or ""):
                    changes[r.id] = (cat, dec.subcategory or "")
                else:
                    unchanged += 1
                continue
            actions.append(PlannedAction(image_id=r.id, source=src, target=target, operation="move",
                                         category=dec.category, reason="library analysed again"))
            changes[r.id] = (cat, dec.subcategory or "")

    moved, failed, txn = 0, 0, ""
    if cancel_check and cancel_check():
        return {"ok": False, "stopped": True, "checked": len(changes) + unchanged,
                "recategorised": 0, "moved": 0, "failed": 0, "transaction_id": ""}
    if actions:
        rep = ActionEngine(cfg, db_path=dbp).execute(actions, organize_root=root)
        txn = rep.transaction_id
        moved, failed = rep.executed, rep.failures
        for a in actions:
            if a.image_id in rep.resolved_targets:
                _move_sidecar(a.source, Path(rep.resolved_targets[a.image_id]))
            else:
                changes.pop(a.image_id, None)  # the file stayed, so its category must too
    with get_session(dbp) as s:
        for iid, (cat, sub) in changes.items():
            rec = s.get(M.ImageRecord, iid)
            if rec is not None:
                rec.category, rec.subcategory = cat, sub
        s.commit()
    return {"ok": True, "checked": len(changes) + unchanged, "recategorised": len(changes),
            "moved": moved, "failed": failed, "transaction_id": txn}
