"""FastAPI review dashboard backend.

Endpoints:
    GET  /api/stats                        state-store statistics
    GET  /api/plan                         current routing plan summary
    GET  /api/images?category=&q=          list images (+ manual bucket filter)
    GET  /api/thumbnail/{image_id}         cached JPEG thumbnail
    GET  /api/bursts                       list burst groups
    GET  /api/burst/{burst_id}             members of one burst group
    POST /api/images/{image_id}/reassign   manual bucket reassignment
    POST /api/plan/rebuild                 rebuild the plan (dry-run)
    POST /api/plan/apply                   execute the plan (transaction)
    POST /api/undo                         roll back a transaction
    GET  /api/transactions                 list known transactions
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import CuratorConfig
from ..db.repository import PhotoRepository
from ..db.session import get_session, init_db
from ..quality.image_loader import load_image, to_thumbnail_jpg
from ..rules.organizer import OrganizeMode
from ..services import Planner

STATIC_DIR = Path(__file__).parent / "static"


class ReassignReq(BaseModel):
    bucket: str | None = None  # None clears a manual override
    category: str | None = None  # explicit category override


class PlanReq(BaseModel):
    output_dir: str = "Organized"
    operation: str = "move"
    mode: str = "category_first"


class UndoReq(BaseModel):
    transaction_id: str | None = None


def create_app(db_path: str | None = None) -> FastAPI:
    dbp = Path(db_path or os.environ.get("PCUR_DB_PATH", "photo_curator.db"))
    init_db(dbp)
    cfg = CuratorConfig.load(dbp)

    app = FastAPI(title="Photo Curator Review", version="0.1.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
        allow_headers=["*"])
    app.state.db = dbp
    app.state.cfg = cfg

    # ------------------------------------------------------------------ api
    @app.get("/api/stats")
    def stats():
        with get_session(dbp) as s:
            return PhotoRepository(s).stats()

    @app.get("/api/plan")
    def plan():
        """Summarize the currently-planned routing (route_status == planned)."""
        with get_session(dbp) as s:
            repo = PhotoRepository(s)
            recs = repo.all_images(route_status="planned")
            by_cat: dict = {}
            for r in recs:
                key = r.manual_bucket or (r.category or "unrouted")
                by_cat[key] = by_cat.get(key, 0) + 1
            return {
                "total_planned": len(recs),
                "by_category": by_cat,
                "organize_root": str(cfg.organize_root),
            }

    @app.get("/api/images")
    def images(
        category: str | None = Query(None),
        manual: bool = Query(False),
        q: str | None = Query(None),
        limit: int = Query(400, le=5000),
        offset: int = Query(0),
    ):
        with get_session(dbp) as s:
            repo = PhotoRepository(s)
            recs = repo.all_images()
            out = []
            for r in recs:
                if category and (r.manual_bucket or r.category) != category:
                    continue
                if manual and not r.manual_bucket:
                    continue
                if q:
                    ql = q.lower()
                    if ql not in r.filename.lower() and \
                       ql not in r.relative_path.lower():
                        continue
                sem = getattr(r, "semantic", None)
                qm = getattr(r, "quality", None)
                out.append({
                    "id": r.id,
                    "filename": r.filename,
                    "relative_path": r.relative_path,
                    "path": r.path,
                    "category": r.category,
                    "manual_bucket": r.manual_bucket,
                    "route_status": r.route_status,
                    "target_path": r.target_path,
                    "is_duplicate": r.is_duplicate,
                    "is_alternate": r.is_alternate,
                    "is_best_shot": r.is_best_shot,
                    "burst_id": r.burst_id,
                    "width": r.width, "height": r.height,
                    "taken_at": r.taken_at.isoformat() if r.taken_at else None,
                    "taken_at_source": r.taken_at_source,
                    "camera_model": r.camera_model,
                    "top1": sem.top1_category if sem else None,
                    "top1_score": round(sem.top1_score, 3) if sem else None,
                    "margin": round(sem.score_margin, 3) if sem else None,
                    "quality_score": round(qm.quality_score, 3) if qm else None,
                    "is_blurry": bool(qm.is_blurry) if qm else False,
                    "is_too_dark": bool(qm.is_too_dark) if qm else False,
                    "is_overexposed": bool(qm.is_overexposed) if qm else False,
                    "face_count": len(getattr(r, "faces", None) or []),
                })
            total = len(out)
            out = out[offset:offset + limit]
            return {"total": total, "items": out}

    @app.get("/api/thumbnail/{image_id}")
    def thumbnail(image_id: int):
        thumb_dir = cfg.cache_dir / "thumbs"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        cache = thumb_dir / f"{image_id:09d}.jpg"
        if cache.exists():
            return FileResponse(cache, media_type="image/jpeg")
        with get_session(dbp) as s:
            rec = PhotoRepository(s).get_by_id(image_id)
            if rec is None:
                raise HTTPException(404, "image not found")
            path = rec.path
            orientation = rec.orientation
        try:
            img = load_image(path, max_side=384, orientation=orientation)
            data = to_thumbnail_jpg(img.bgr, max_side=384)
        except Exception as e:
            raise HTTPException(500, f"thumbnail decode failed: {e}")
        cache.write_bytes(data)
        return Response(data, media_type="image/jpeg")

    @app.get("/api/bursts")
    def bursts():
        with get_session(dbp) as s:
            repo = PhotoRepository(s)
            out = []
            for b in repo.all_bursts():
                best = repo.get_by_id(b.best_shot_id) if b.best_shot_id else None
                out.append({
                    "id": b.id, "name": b.name, "image_count": b.image_count,
                    "best_shot_id": b.best_shot_id,
                    "best_shot": best.filename if best else None,
                    "start_time": b.start_time.isoformat() if b.start_time else None,
                })
            return {"count": len(out), "items": out}

    @app.get("/api/burst/{burst_id}")
    def burst_detail(burst_id: int):
        with get_session(dbp) as s:
            repo = PhotoRepository(s)
            bg = next((b for b in repo.all_bursts() if b.id == burst_id), None)
            if bg is None:
                raise HTTPException(404, "burst not found")
            members = [r for r in repo.all_images()
                       if r.burst_id == burst_id and r.infer_status == "inferred"]
            items = []
            for r in members:
                qm = getattr(r, "quality", None)
                faces = getattr(r, "faces", None) or []
                best_face = max(faces, key=lambda f: f.confidence, default=None)
                items.append({
                    "id": r.id, "filename": r.filename, "is_best_shot": r.is_best_shot,
                    "is_alternate": r.is_alternate,
                    "taken_at": r.taken_at.isoformat() if r.taken_at else None,
                    "width": r.width, "height": r.height,
                    "sharpness_norm": round(qm.sharpness_norm, 3) if qm else None,
                    "face_count": len(faces),
                    "face_confidence": round(best_face.confidence, 3) if best_face else None,
                    "ear": best_face.eye_aspect_ratio if best_face else None,
                })
            return {"burst": bg.name, "members": items}

    @app.post("/api/images/{image_id}/reassign")
    def reassign(image_id: int, body: ReassignReq):
        with get_session(dbp) as s:
            repo = PhotoRepository(s)
            rec = repo.get_by_id(image_id)
            if rec is None:
                raise HTTPException(404, "image not found")
            rec.manual_bucket = body.bucket
            if body.category:
                rec.category = body.category
            return {"ok": True, "id": image_id,
                    "manual_bucket": rec.manual_bucket,
                    "category": rec.category}

    @app.post("/api/plan/rebuild")
    def rebuild_plan(body: PlanReq):
        planner = Planner(cfg, db_path=dbp)
        m = OrganizeMode.IN_PLACE if body.mode == "in_place" else \
            OrganizeMode.CATEGORY_FIRST
        p = planner.build_plan(organize_root=Path(body.output_dir),
                               operation=body.operation, mode=m)
        return {"plan_id": p.plan_id, "total": p.total,
                "by_category": p.by_category, "burst_count": p.burst_count,
                "organize_root": p.organize_root}

    @app.post("/api/plan/apply")
    def apply_plan(body: PlanReq):
        planner = Planner(cfg, db_path=dbp)
        m = OrganizeMode.IN_PLACE if body.mode == "in_place" else \
            OrganizeMode.CATEGORY_FIRST
        p = planner.build_plan(organize_root=Path(body.output_dir),
                               operation=body.operation, mode=m)
        if p.total == 0:
            return {"ok": False, "error": "nothing to apply"}
        rep = planner.execute_plan(p, operation=body.operation)
        return {
            "ok": rep.failures == 0,
            "transaction_id": rep.transaction_id,
            "executed": rep.executed,
            "collisions": rep.collisions,
            "failures": rep.failures,
            "errors": rep.errors[:50],
        }

    @app.post("/api/undo")
    def undo(body: UndoReq):
        planner = Planner(cfg, db_path=dbp)
        tid = body.transaction_id
        if not tid:
            with get_session(dbp) as s:
                tid = PhotoRepository(s).latest_transaction()
        if not tid:
            return {"ok": False, "error": "no transactions found"}
        rep = planner.undo(tid)
        return {
            "ok": rep.failures == 0,
            "transaction_id": tid,
            "reversed": rep.reversed,
            "skipped": rep.skipped,
            "failures": rep.failures,
            "errors": rep.errors[:50],
        }

    @app.get("/api/transactions")
    def transactions():
        from sqlalchemy import select as _sel
        from ..db.models import ActionLedger
        with get_session(dbp) as s:
            rows = s.exec(_sel(ActionLedger)).all()
            txns: dict = {}
            for r in rows:
                d = txns.setdefault(r.transaction_id, {
                    "transaction_id": r.transaction_id,
                    "count": 0, "committed": 0, "rolled_back": 0,
                    "failed": 0, "created_at": r.created_at.isoformat(),
                })
                d["count"] += 1
                d[r.status] = d.get(r.status, 0) + 1
            return {"items": list(txns.values())}

    # ---------------------------------------------------------------- static
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/healthz")
    def health():
        return {"ok": True}

    return app


# Module-level app for `uvicorn photo_curator.ui.app:app`.
app = create_app()
