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
import threading
from datetime import datetime, timezone
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


class ScanReq(BaseModel):
    path: str
    workers: int = 8


class InferReq(BaseModel):
    device: str = "auto"
    batch_size: int = 32


class ClusterReq(BaseModel):
    window_seconds: int = 30
    similarity_threshold: float = 0.88


class ProgressManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.state = {
            "active": False,
            "task": "",
            "step": "",
            "done": 0,
            "total": 0,
            "percent": 0,
            "message": "",
            "detail": "",
            "error": None,
        }

    def start(self, task: str, step: str = "", total: int = 0, message: str = ""):
        with self._lock:
            self.state = {
                "active": True,
                "task": task,
                "step": step or task,
                "done": 0,
                "total": total,
                "percent": 0,
                "message": message,
                "detail": f"0 / {total}" if total > 0 else "",
                "error": None,
            }

    def update(self, done: int, total: int, message: str = "", detail: str = ""):
        with self._lock:
            if not self.state["active"]:
                return
            pct = int(done / max(total, 1) * 100) if total > 0 else 0
            self.state["done"] = done
            self.state["total"] = total
            self.state["percent"] = min(pct, 100)
            if message:
                self.state["message"] = message
            if detail:
                self.state["detail"] = detail
            elif total > 0:
                self.state["detail"] = f"{done} / {total}"

    def finish(self, message: str = "Completed successfully"):
        with self._lock:
            self.state["active"] = False
            self.state["percent"] = 100
            self.state["message"] = message

    def error(self, err_msg: str):
        with self._lock:
            self.state["active"] = False
            self.state["error"] = err_msg
            self.state["message"] = f"Error: {err_msg}"

    def get(self) -> dict:
        with self._lock:
            return dict(self.state)


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
    progress_mgr = ProgressManager()
    app.state.progress = progress_mgr

    # ------------------------------------------------------------------ api
    @app.get("/api/stats")
    def stats():
        with get_session(dbp) as s:
            return PhotoRepository(s).stats()

    @app.get("/api/progress")
    def get_progress():
        return progress_mgr.get()

    @app.post("/api/scan")
    def scan_folder(body: ScanReq):
        target = Path(body.path).expanduser().resolve()
        if not target.exists() or not target.is_dir():
            raise HTTPException(400, f"Directory does not exist: {body.path}")

        progress_mgr.start("Scanning", f"Scanning directory: {target.name}", 0, "Discovering image files…")
        from ..scanner.file_scanner import FileScanner
        cfg.scan_workers = body.workers
        scanner = FileScanner(cfg)

        def on_prog(done, total, curr):
            fn = Path(curr).name
            progress_mgr.update(done, total, f"Indexing: {fn}", detail=f"{done} / {total}")

        try:
            res = scanner.scan_directory(target, db_path=dbp, on_progress=on_prog, show_progress=False)
            progress_mgr.finish(f"Scan complete: {res.scanned} files seen, {res.new} new indexed in {res.duration_s:.1f}s")
            return {
                "ok": True,
                "path": str(target),
                "scanned": res.scanned,
                "new": res.new,
                "duplicates": res.duplicates,
                "skipped_known": res.skipped_known,
                "errors": res.errors,
                "duration_s": round(res.duration_s, 2),
            }
        except Exception as e:
            progress_mgr.error(str(e))
            raise

    @app.post("/api/infer")
    def infer_pending(body: InferReq = InferReq()):
        progress_mgr.start("AI Inference", "Running quality + faces + CLIP semantics…", 0, "Initializing AI models…")
        from ..inference.backends import detect_backend
        from ..inference.pipeline import InferencePipeline
        cfg.device.device = body.device
        if body.batch_size:
            cfg.device.face_batch_size = body.batch_size
            cfg.device.clip_batch_size = max(body.batch_size, 1)
        bs = detect_backend(cfg)
        pipeline = InferencePipeline(cfg, backend=bs)

        def on_prog(done, total, curr):
            fn = Path(curr).name if not curr.startswith("Phase") else curr
            progress_mgr.update(done, total, fn, detail=f"{done} / {total}" if total > 0 else "")

        try:
            res = pipeline.run(db_path=dbp, on_progress=on_prog, show_progress=False)
            progress_mgr.finish(f"Inference complete: {res.processed} processed, {res.faces_total} faces in {res.duration_s:.1f}s")
            return {
                "ok": True,
                "processed": res.processed,
                "faces_total": res.faces_total,
                "quality_computed": res.quality_computed,
                "semantic_computed": res.semantic_computed,
                "failures": res.failures,
                "duration_s": round(res.duration_s, 2),
            }
        except Exception as e:
            progress_mgr.error(str(e))
            raise

    @app.post("/api/cluster")
    def cluster_bursts(body: ClusterReq = ClusterReq()):
        progress_mgr.start("Burst Clustering", "Detecting bursts and selecting best shots…", 0, "Grouping photos by timestamp & similarity…")
        from ..clustering.burst_detector import BurstDetector
        from ..clustering.best_shot_selector import BestShotSelector, _candidate_from_record
        from ..db.models import BurstGroup
        _EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
        cfg.burst.window_seconds = body.window_seconds
        cfg.burst.similarity_threshold = body.similarity_threshold

        progress_mgr.start("Clustering", "Burst detection and best-shot selection…", 1, "Analyzing timestamps and visual similarity…")
        try:
            with get_session(dbp) as s:
                repo = PhotoRepository(s)
                records = repo.all_images()
                for r in records:
                    r.burst_id = None
                    r.is_best_shot = False
                    r.is_alternate = False
                for b in repo.all_bursts():
                    repo.s.delete(b)
                s.flush()
                detector = BurstDetector(cfg)
                results = detector.cluster_records(records)
                by_id = {r.id: r for r in records}
                sel = BestShotSelector(cfg)
                n = 0
                for res in results:
                    members = [by_id[m] for m in res.image_ids if m in by_id]
                    if len(members) < cfg.burst.min_group_size:
                        continue
                    best_id = sel.pick_best_shot(
                        [_candidate_from_record(m) for m in members]).image_id
                    earliest = min(members, key=lambda m: (m.taken_at is None,
                                                           m.taken_at or _EPOCH))
                    stamp = (earliest.taken_at.strftime("%Y%m%d")
                             if earliest.taken_at else "Unknown")
                    bg = repo.upsert_burst(BurstGroup(
                        name=f"Burst_{stamp}_{n:04d}",
                        image_count=len(members),
                        best_shot_id=best_id,
                        start_time=earliest.taken_at,
                        end_time=max((m.taken_at for m in members), default=earliest.taken_at),
                    ))
                    for m in members:
                        m.burst_id = bg.id
                        m.is_best_shot = (m.id == best_id)
                        m.is_alternate = (m.id != best_id)
                    n += 1
                progress_mgr.finish(f"Clustering complete: {n} burst groups detected")
                return {"ok": True, "burst_groups": n}
        except Exception as e:
            progress_mgr.error(str(e))
            raise

    @app.post("/api/ingest")
    def ingest_folder(body: ScanReq):
        progress_mgr.start("Ingest Pipeline", "Starting full ingest (Scan + Infer + Cluster)…", 0, "Initializing…")
        try:
            scan_res = scan_folder(body)
            infer_res = infer_pending(InferReq())
            cluster_res = cluster_bursts(ClusterReq())
            progress_mgr.finish("Full ingest pipeline complete!")
            return {
                "ok": True,
                "scan": scan_res,
                "infer": infer_res,
                "cluster": cluster_res,
            }
        except Exception as e:
            progress_mgr.error(str(e))
            raise

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
        from sqlmodel import select as _sel
        from ..db.models import ActionLedger
        with get_session(dbp) as s:
            rows = s.exec(_sel(ActionLedger)).all()
            txns: dict = {}
            for r in rows:
                d = txns.setdefault(r.transaction_id, {
                    "transaction_id": r.transaction_id,
                    "count": 0, "committed": 0, "rolled_back": 0,
                    "failed": 0, "created_at": r.created_at.isoformat() if r.created_at else "",
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
