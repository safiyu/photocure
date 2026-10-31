"""Actions that write NEW files, shared by the web app and the MCP server.

Both only ever add files: a fixed photo or layered file next to an original, or a clean copy
folder somewhere outside the library. No original is changed, moved or deleted.
"""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy.orm import selectinload
from sqlmodel import col, select

from .clustering.group_shots import group_matrix_payload
from .config import CuratorConfig
from .db import models as M
from .db.session import get_session


class ActionError(ValueError):
    """A refusal with a message fit to show the user and the HTTP status the web app should use."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _existing(*paths) -> str | None:
    return next((p for p in paths if p and os.path.exists(p)), None)


def composite_burst(db_path: Path | str, burst_id: int, mode: str = "fixed") -> dict:
    """Blend the open eyes of the partner frame into the best frame of a burst."""
    from . import composite as comp
    from .quality.image_loader import load_image
    if mode not in ("fixed", "layered"):
        raise ActionError("mode must be 'fixed' or 'layered'")
    with get_session(db_path) as s:
        if s.get(M.BurstGroup, burst_id) is None:
            raise ActionError("burst not found", 404)
        members = s.exec(select(M.ImageRecord).where(
            M.ImageRecord.burst_id == burst_id, M.ImageRecord.infer_status == "inferred",
            M.ImageRecord.scan_status != "missing"  # never blend in or write next to a trashed frame
        ).options(selectinload(M.ImageRecord.quality), selectinload(M.ImageRecord.faces))).all()
        grp = group_matrix_payload(members)
        if grp["status"] != "composite":
            raise ActionError("this burst has no pair of frames that can be combined")
        by_id = {m.id: m for m in members}
        base, donor = by_id[grp["best_id"]], by_id[grp["partner_id"]]

        def source(r) -> Path:
            p = _existing(r.path, r.target_path)
            if p is None:
                raise ActionError(f"{r.filename} is no longer on disk", 404)
            return Path(p)

        def faces(r):
            return [comp.FaceRef((f.x1, f.y1, f.x2, f.y2), f.eye_status)
                    for f in r.faces or [] if None not in (f.x1, f.y1, f.x2, f.y2)]

        bp, dp = source(base), source(donor)
        bb = load_image(bp, max_side=None, orientation=base.orientation).bgr
        db_ = load_image(dp, max_side=None, orientation=donor.orientation).bgr
        res = comp.compose_open_eyes(bb, db_, faces(base), faces(donor))
        if not res.ok:
            raise ActionError(res.reason or "the frames could not be combined", 422)
        if mode == "fixed":
            out = comp.write_fixed_jpeg(res.bgr, bp, comp.fixed_jpeg_path(bp))
        else:
            out = comp.write_layered_psd(comp.layered_psd_path(bp), bb, res.aligned_donor,
                                         res.mask, merged_bgr=res.bgr)
        return {"ok": True, "mode": mode, "path": str(out), "applied": res.applied,
                "skipped": res.skipped}


def clean_export_selection(cfg: CuratorConfig, ids: list[int], output_dir: str, *,
                           policy: str = "blur", html: bool = True, rename: bool = True,
                           backend=None) -> dict:
    """Scan (unless policy is include) and write metadata-free copies into a new folder."""
    from .clean_export import ExportItem, PiiScanner, PiiUnavailable, export_bundle
    from .quality.image_loader import load_image
    from .ui.app import validate_output_dir
    if policy not in ("blur", "skip", "include"):
        raise ActionError("policy must be blur, skip or include")
    if not ids:
        raise ActionError("choose at least one photo")
    try:
        root = validate_output_dir(output_dir, cfg.db_path, False)
    except Exception as e:  # the web helper raises HTTPException
        raise ActionError(str(getattr(e, "detail", e))) from None
    with get_session(cfg.db_path) as s:
        rows = []
        for i in range(0, len(ids), 500):  # SQLite caps bound variables per statement
            rows += [(r.id, r.filename, r.path, r.target_path, r.orientation, r.extension,
                      r.taken_at.isoformat() if r.taken_at else None) for r in s.exec(
                select(M.ImageRecord).where(col(M.ImageRecord.id).in_(ids[i:i + 500]))).all()]
    if not rows:
        raise ActionError("none of those photos exist", 404)
    findings: dict = {}
    flagged: list[dict] = []
    scan_skipped: list[dict] = []
    if policy != "include":
        try:
            scanner = PiiScanner(cfg, backend=backend)
        except PiiUnavailable as e:
            raise ActionError(f"{e} Use policy 'include' to export without the check.") from None
        try:
            for iid, name, path, target, orient, _ext, _taken in rows:
                src = _existing(path, target)
                if src is None:
                    continue
                try:
                    f = scanner.scan_image(
                        iid, load_image(src, max_side=1600, orientation=orient).bgr)
                except Exception as e:
                    scan_skipped.append({"id": iid, "filename": name, "error": str(e)[:200]})
                    continue
                if f is not None:
                    findings[iid] = f
                    flagged.append({"id": iid, "filename": name, "labels": f.labels})
        finally:
            scanner.release()
    items = [ExportItem(iid, _existing(path, target) or path, orient, ext, taken)
             for iid, _n, path, target, orient, ext, taken in rows]
    res = export_bundle(items, root, findings=findings, policy=policy, html=html, rename=rename)
    return {"ok": not res.failures, "folder": str(res.folder), "exported": res.exported,
            "flagged": res.flagged, "blurred": res.blurred, "skipped_flagged": res.skipped_flagged,
            "flagged_photos": flagged, "failures": res.failures[:20],
            "scan_skipped": scan_skipped[:20]}


def finalize_burst(db_path: Path | str, cfg: CuratorConfig, burst_id: int, *, operation: str = "move") -> dict:
    """Finalize a single burst: move marked-for-delete frames to trash and finalize kept frames into the library."""
    from .library import require_library_root
    from .db.repository import PhotoRepository
    from .rules.decision_matrix import DecisionMatrix
    from .rules.organizer import FolderOrganizer
    from .engine.action_engine import ActionEngine, PlannedAction
    from .library_actions import _category_enum, _move_sidecar
    from .people_actions import detach_faces

    dbp = Path(db_path)
    root = require_library_root(dbp, cfg)

    with get_session(dbp) as s:
        bg = s.get(M.BurstGroup, burst_id)
        if bg is None:
            raise ActionError("burst not found", 404)
        members = s.exec(
            select(M.ImageRecord).where(
                M.ImageRecord.burst_id == burst_id,
                M.ImageRecord.scan_status != "missing",
                M.ImageRecord.stage != M.STAGE_TRASH,
            ).options(
                selectinload(M.ImageRecord.quality),
                selectinload(M.ImageRecord.faces),
                selectinload(M.ImageRecord.semantic),
            )
        ).all()
        if not members:
            return {"ok": True, "burst_id": burst_id, "trashed": 0, "finalized": 0, "message": "no active members"}

        matrix = DecisionMatrix(cfg)
        organizer = FolderOrganizer(cfg, organize_root=root)
        organizer.set_bursts([bg])

        actions: list[PlannedAction] = []
        missing_sources = []
        for r in members:
            src = Path(r.path)
            if not src.exists() and r.target_path and Path(r.target_path).exists():
                src = Path(r.target_path)
            if not src.exists():
                missing_sources.append(r)
                continue

            dec = matrix.evaluate_record(r)
            cat = _category_enum(cfg, r.manual_bucket) if r.manual_bucket else dec.category
            target = organizer.full_target(r, dec)
            op = "move" if r.delete_approved else operation
            actions.append(PlannedAction(
                image_id=r.id,
                source=src,
                target=target,
                operation=op,
                category=cat,
                reason="burst finalized (marked for delete)" if r.delete_approved else "burst finalized",
                is_alternate=r.is_alternate,
                burst_name=bg.name if r.is_alternate else None,
            ))

    txn = ""
    trashed_count = 0
    finalized_count = 0

    if actions:
        engine = ActionEngine(cfg, db_path=dbp)
        rep = engine.execute(actions, organize_root=root, trash=True)
        if rep.failures:
            raise ActionError(f"failed to move burst photos: {rep.errors[0]}", 500)
        txn = rep.transaction_id

        for a in actions:
            dest = Path(rep.resolved_targets.get(a.image_id, a.target))
            _move_sidecar(a.source, dest)

    with get_session(dbp) as s:
        repo = PhotoRepository(s)
        for r_miss in missing_sources:
            rec = s.get(M.ImageRecord, r_miss.id)
            if rec and rec.delete_approved:
                rec.stage = M.STAGE_TRASH
                rec.route_status = "trashed"
                rec.scan_status = "missing"
                detach_faces(s, rec.id)
                trashed_count += 1
        for a in actions:
            rec = s.get(M.ImageRecord, a.image_id)
            if rec:
                if rec.delete_approved and rec.stage == M.STAGE_TRASH:
                    detach_faces(s, a.image_id)
                    trashed_count += 1
                elif rec.stage == M.STAGE_LIBRARY:
                    finalized_count += 1
        if txn:
            for e in s.exec(select(M.ActionLedger).where(M.ActionLedger.transaction_id == txn)).all():
                rec = s.get(M.ImageRecord, e.image_id)
                if rec and rec.stage == M.STAGE_TRASH:
                    e.prev_delete_approved = False
        repo.prune_bursts()
        burst_pruned = s.get(M.BurstGroup, burst_id) is None
        s.commit()

    return {
        "ok": True,
        "burst_id": burst_id,
        "trashed": trashed_count,
        "finalized": finalized_count,
        "transaction_id": txn,
        "pruned": burst_pruned,
    }
