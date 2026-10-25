"""FastAPI review dashboard backend.

Read endpoints are open on localhost; every mutating (POST) endpoint requires the
per-launch token in the ``X-PCUR-Token`` header (injected into index.html).

Endpoints:
    GET  /api/config                       defaults, categories
    GET  /api/stats | /api/progress | /api/plan
    GET  /api/images                       filtered + paginated listing
    GET  /api/images/{id}/similar          embedding neighbours
    GET  /api/search?q=                    natural-language search over embeddings
    GET  /api/highlights/events | /years   best of each event, years with scored photos
    GET  /api/highlights/export            year in review as a self-contained HTML page
    GET  /api/features  POST /api/features highlight weights, keyword export, watch folder
    GET  /api/duplicates                   exact and near duplicate groups with a suggested keeper
    GET  /api/keywords  POST /api/keywords/write   XMP keyword export (background job)
    GET  /api/watch  POST /api/watch       follow a folder: scan, analyse and file new photos
    POST /api/duplicates/resolve | /dismiss | /resolve-all
    GET  /api/preferences                  taste model status
    POST /api/preferences/learn | /reset   train the taste model from ratings, or forget it
    GET  /thumbnail/{id}?tier=&v=          cached WebP thumbnail (micro, medium, preview)
    GET  /api/bursts | /api/burst/{id}
    GET  /api/events                       clusters
    GET  /api/transactions                 ledger summary (+ latest undoable)
    GET  /api/transactions/{tid}/verify    re-hash committed targets
    GET  /api/jobs/{id}                    background apply/undo job + live log
    POST /api/scan | /api/infer | /api/cluster | /api/ingest
    POST /api/images/{id}/reassign | /api/images/{id}/lock
    POST /api/plan/rebuild | /api/plan/diff | /api/plan/apply
    POST /api/undo | /api/repair | /api/learn | /api/settings
    POST /api/events/cluster | /api/events/{id}/rename
"""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from datetime import time as dtime
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func, or_
from sqlalchemy import update as sa_update
from sqlalchemy.orm import selectinload
from sqlmodel import col, select

from .. import __version__
from ..auth import COOKIE_MAX_AGE, COOKIE_NAME, AuthManager, build_allowed_hosts, origin_allowed
from ..clustering.group_shots import burst_face_matrix, group_matrix_payload
from ..config import CuratorConfig, default_db_path
from ..db import models as M
from ..engine.locks import LEDGER_WRITING as _LEDGER_WRITING
from ..db.repository import PhotoRepository
from ..db.session import get_session, init_db, snapshot_database
from ..rules.organizer import OrganizeMode
from ..rules.taxonomy import CategoryType
from ..services import Planner

STATIC_DIR = Path(__file__).parent / "static"
TOKEN_HEADER = "X-PCUR-Token"
TOKEN_PLACEHOLDER = "__PCUR_TOKEN__"
OPERATIONS = ("move", "copy", "hardlink", "symlink")
CATEGORY_VALUES = tuple(c.value for c in CategoryType)
JUNK_VALUES = tuple(c.value for c in CategoryType if c.is_junk)
UTILITY_VALUES = tuple(c.value for c in CategoryType if c.is_utility)

# Only one long-running task at a time across the whole process.
_BUSY = threading.Lock()


def _pid_file(port: int) -> Path:
    from ..lifecycle import pid_file_path
    return pid_file_path(port)

_BLOCKED_DIRS = (
    "/etc", "/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/boot", "/dev",
    "/proc", "/sys", "/System", "/Library", "/Applications", "/private/etc",
    "/var/lib", "/var/log", "/var/db",
)


class ReassignReq(BaseModel):
    bucket: str | None = None  # None clears a manual override
    category: str | None = None  # alias for bucket


class LockReq(BaseModel):
    locked: bool = True


class RenamePersonReq(BaseModel):
    name: str


class BulkDeletePeopleReq(BaseModel):
    person_ids: list[int]


class ReassignFaceReq(BaseModel):
    person_id: int | None = None
    new_name: str | None = None


class PlanReq(BaseModel):
    output_dir: str
    operation: Literal["move", "copy", "hardlink", "symlink"] = "move"
    mode: Literal["category_first", "in_place"] = "category_first"
    only_category: str | None = None
    trash: bool = False
    include_duplicates: bool = True


class UndoReq(BaseModel):
    transaction_id: str | None = None
    verify: bool = True


DeviceName = Literal["auto", "cuda", "rocm", "xpu", "mps", "directml", "integrated", "cpu"]
Count = Annotated[int, Field(ge=0, le=128)]


class RunOverrides(BaseModel):
    """Optional per request overrides; anything left out falls back to saved settings."""

    device: DeviceName | None = None
    device_index: Count | None = None
    decode_workers: Count | None = None
    torch_threads: Count | None = None
    scan_workers: Count | None = None


class ScanReq(RunOverrides):
    path: str
    workers: Count | None = None


class InferReq(RunOverrides):
    batch_size: int = 32
    reanalyze: bool = False


class ClusterReq(BaseModel):
    window_seconds: int = 30
    similarity_threshold: float = 0.88
    only_new: bool = False


class EventClusterReq(BaseModel):
    gap_hours: float = 2.0
    gps_km: float = 1.0


class RenameReq(BaseModel):
    name: str


def infer_progress_detail(done: int, total: int) -> str:
    """Photos handled in the current phase; the pipeline counts two steps per photo."""
    photos = total // 2
    if photos <= 0:
        return ""
    return f"{done if done <= photos else done - photos} / {photos} photos"


class ProgressManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._run = 0
        self.state = {
            "active": False, "task": "", "step": "", "done": 0, "total": 0,
            "percent": 0, "message": "", "detail": "", "error": None,
            "stopping": False, "stopped": False, "run": 0,
        }

    def start(self, task: str, step: str = "", total: int = 0, message: str = ""):
        self._stop_event.clear()
        with self._lock:
            self._run += 1
            self.state = {
                "active": True, "task": task, "step": step or task, "done": 0,
                "total": total, "percent": 0, "message": message,
                "detail": f"0 / {total}" if total > 0 else "", "error": None,
                "stopping": False, "stopped": False, "run": self._run,
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

    def request_stop(self):
        self._stop_event.set()
        with self._lock:
            if self.state["active"]:
                self.state["stopping"] = True
                self.state["message"] = "Stopping..."

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def finish(self, message: str = "Completed successfully", stopped: bool = False):
        with self._lock:
            self.state["active"] = False
            self.state["stopping"] = False
            self.state["stopped"] = stopped
            self.state["percent"] = 100
            self.state["message"] = message

    def stop(self, message: str = "Stopped by user"):
        self.finish(message, stopped=True)

    def error(self, err_msg: str):
        with self._lock:
            self.state["active"] = False
            self.state["stopping"] = False
            self.state["stopped"] = False
            self.state["error"] = err_msg
            self.state["message"] = f"Error: {err_msg}"

    def reset(self):
        self._stop_event.clear()
        with self._lock:
            self.state = {
                "active": False, "task": "", "step": "", "done": 0, "total": 0,
                "percent": 0, "message": "", "detail": "", "error": None,
                "stopping": False, "stopped": False, "run": self._run,
            }

    def get(self) -> dict:
        with self._lock:
            return dict(self.state)


# --------------------------------------------------------------------- helpers
def _restart_args(orig_argv: list[str]) -> list[str]:
    """Original command line for a restart, forced to --no-open so no second tab appears."""
    args = [a for a in orig_argv[1:] if a != "--open"]
    if "--no-open" not in args and any(a in ("serve", "start", "setup") for a in args):
        args.append("--no-open")
    return args


def _accepts(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _not_impl(what: str) -> HTTPException:
    return HTTPException(501, f"{what} is not available in this build")


def _check_category(value: str | None, field: str = "category") -> str | None:
    if value is None or value == "":
        return None
    if value not in CATEGORY_VALUES:
        raise HTTPException(400, f"invalid {field} '{value}', expected one of "
                                 f"{', '.join(CATEGORY_VALUES)}")
    return value


def _nearest_existing(p: Path) -> Path:
    cur = p
    while not cur.exists() and cur != cur.parent:
        cur = cur.parent
    return cur


def _library_root(dbp: Path) -> Path | None:
    """Common directory of all indexed images, or None when unknown/too shallow."""
    moved = select(M.ActionLedger.target_path).where(
        M.ActionLedger.status == "committed",
        M.ActionLedger.operation_type == "move")
    with get_session(dbp) as s:
        lo, hi = s.exec(select(func.min(M.ImageRecord.path),
                               func.max(M.ImageRecord.path)).where(
            M.ImageRecord.scan_status != "missing",
            M.ImageRecord.path.not_in(moved))).one()
    if not lo or not hi:
        return None
    try:
        root = os.path.commonpath([lo, hi])
    except ValueError:
        return None
    if root in (lo, hi):
        root = os.path.dirname(root)
    rp = Path(root)
    return rp if len(rp.parts) >= 3 else None


def validate_output_dir(raw: str, dbp: Path, in_place: bool) -> Path:
    """Reject relative, system, shallow, home and in-library output roots."""
    if not raw or not raw.strip():
        raise HTTPException(400, "output_dir is required")
    p = Path(raw.strip()).expanduser()
    if not p.is_absolute():
        raise HTTPException(400, "output_dir must be an absolute path")
    p = p.resolve()
    if len(p.parts) < 3 or p == Path.home().resolve():
        raise HTTPException(400, f"output_dir is too broad: {p}")
    for b in _BLOCKED_DIRS:
        bp = Path(b)
        if p == bp or bp in p.parents:
            raise HTTPException(400, f"output_dir is inside a system directory: {p}")
    try:
        hidden = p.relative_to(Path.home().resolve()).parts[:1]
    except ValueError:
        hidden = ()
    if hidden and hidden[0].startswith("."):
        raise HTTPException(400, f"output_dir is inside a hidden home folder: {p}")
    if _confined() and not _within(p, _browse_roots()):
        raise HTTPException(403, "output_dir is outside the allowed locations")
    from ..auth import token_file_path
    tdir = token_file_path(CuratorConfig.load(dbp).cache_dir).parent.resolve()
    if p == tdir or (tdir.name == "photocure" and tdir in p.parents):
        raise HTTPException(400, "output_dir must not be the photocure data directory")
    if not in_place:
        root = _library_root(dbp)
        if root is not None and (p == root or root in p.parents):
            raise HTTPException(400, f"output_dir must not be inside the library ({root})")
    return p


def _cat_label(name: str | None) -> str:
    t = str(name or "").replace("_", " ")
    return t[:1].upper() + t[1:]


def _guesses(sem) -> list[dict]:
    """Top three class guesses as [{name, score}] from the stored distribution."""
    import json
    try:
        dist = json.loads(sem.distribution or "{}")
    except (ValueError, TypeError):
        dist = {}
    ranked = sorted(((k, float(v)) for k, v in dist.items()), key=lambda kv: -kv[1])[:3]
    if not ranked:
        ranked = [(sem.top1_category, float(sem.top1_score))]
        if sem.top2_category:
            ranked.append((sem.top2_category, float(sem.top2_score)))
    return [{"name": k, "score": round(v, 3)} for k, v in ranked if k]


def _explain_uncertain(sem, cfg: CuratorConfig | None) -> dict:
    """Why a photo is uncertain; uses the rules module when it provides one."""
    if sem is None or not sem.top1_category:
        return {"kind": "no_data", "text": "This photo has not been analysed yet."}
    if cfg is not None:
        try:
            from ..rules.decision_matrix import explain_uncertain
            from ..rules.taxonomy import get_taxonomy
            out = explain_uncertain(sem, get_taxonomy(cfg), cfg)
            if out and out.get("text"):
                return {"kind": out.get("kind", ""), "text": out["text"]}
        except Exception:
            pass
    tax = cfg.taxonomy if cfg is not None else None
    min_conf = float(getattr(tax, "min_confidence", 0.40))
    min_margin = float(getattr(tax, "min_margin", 0.08))
    p1, p2 = round(sem.top1_score * 100), round(sem.top2_score * 100)
    if sem.top1_score < min_conf:
        return {"kind": "low_confidence",
                "text": f"Not confident: the best guess is {_cat_label(sem.top1_category)} "
                        f"at {p1}%, below the {round(min_conf * 100)}% needed."}
    if sem.score_margin < min_margin:
        return {"kind": "close_call",
                "text": f"Close call between {_cat_label(sem.top1_category)} ({p1}%) and "
                        f"{_cat_label(sem.top2_category)} ({p2}%)."}
    return {"kind": "other", "text": "The routing rules could not place this photo."}


def _item(r: M.ImageRecord, cfg: CuratorConfig | None = None) -> dict:
    sem = r.semantic
    qm = r.quality
    out = {
        "id": r.id, "fp": _fingerprint(r), "filename": r.filename, "relative_path": r.relative_path,
        "path": r.path, "category": r.category, "manual_bucket": r.manual_bucket,
        "route_status": r.route_status, "target_path": r.target_path,
        "is_duplicate": r.is_duplicate, "is_alternate": r.is_alternate,
        "is_best_shot": r.is_best_shot, "burst_id": r.burst_id,
        "width": r.width, "height": r.height,
        "taken_at": r.taken_at.isoformat() if r.taken_at else None,
        "taken_at_source": r.taken_at_source, "camera_model": r.camera_model,
        "locked": bool(r.locked), "rating": r.rating, "event_id": r.event_id,
        "delete_approved": bool(r.delete_approved),
        "near_dup_of": r.near_dup_of,
        "top1": sem.top1_category if sem else None,
        "top1_score": round(sem.top1_score, 3) if sem else None,
        "margin": round(sem.score_margin, 3) if sem else None,
        "aesthetic": round(sem.aesthetic, 3) if sem and sem.aesthetic is not None else None,
        "preference": round(sem.preference, 3) if sem and sem.preference is not None else None,
        "quality_score": round(qm.quality_score, 3) if qm else None,
        "is_blurry": bool(qm.is_blurry) if qm else False,
        "is_too_dark": bool(qm.is_too_dark) if qm else False,
        "is_overexposed": bool(qm.is_overexposed) if qm else False,
        "face_count": len(r.faces or []),
        "has_blinks": bool(r.has_blinks), "blink_count": int(r.blink_count or 0),
        "is_favorite": bool(r.is_favorite), "city": r.city, "region": r.region,
        "country_code": r.country_code,
        "top2": sem.top2_category if sem else None,
        "top2_score": round(sem.top2_score, 3) if sem else None,
    }
    if r.category == "uncertain" and not r.manual_bucket:
        out["uncertain_reason"] = _explain_uncertain(sem, cfg)
        out["guesses"] = _guesses(sem) if sem else []
    return out


_QUALITY_ISSUES = ("blurry", "dark", "overexposed")
_ISSUES = _QUALITY_ISSUES + ("blink", "duplicate")


def _issue_clause(issue: str):
    """Photos with any of the comma-separated problems (blurry, dark, overexposed, blink, duplicate)."""
    I, Q = M.ImageRecord, M.QualityMetric
    names = [n.strip() for n in issue.split(",") if n.strip()]
    bad = [n for n in names if n not in _ISSUES]
    if not names or bad:
        raise HTTPException(422, f"issue must be a comma-separated list of {', '.join(_ISSUES)}")
    flag = {"blurry": Q.is_blurry, "dark": Q.is_too_dark, "overexposed": Q.is_overexposed}
    clauses = []
    for n in dict.fromkeys(names):
        if n in flag:
            clauses.append(I.id.in_(select(Q.image_id).where(flag[n] == True)))  # noqa: E712
        elif n == "blink":
            clauses.append(I.has_blinks == True)  # noqa: E712
        else:
            clauses.append(or_(I.is_duplicate == True, I.near_dup_of.is_not(None)))  # noqa: E712
    return or_(*clauses)


def _by_ids(s, ids, stmt_fn=None, chunk: int = 500) -> list:
    """ImageRecord rows for ids, queried in chunks to stay under SQLite's variable limit."""
    ids = list(ids)
    rows: list = []
    for i in range(0, len(ids), chunk):
        stmt = select(M.ImageRecord).where(M.ImageRecord.id.in_(ids[i:i + chunk]))
        rows += s.exec(stmt_fn(stmt) if stmt_fn else stmt).all()
    return rows


def _burst_neighbourhood(s, new_ids, window_s: float) -> tuple[list, list]:
    """Records and burst groups that clustering the given new photos can change.

    Each new photo's time is widened by the burst window; bursts touching those
    intervals are pulled in with their own span widened too, until none more join.
    """
    I, B = M.ImageRecord, M.BurstGroup
    new = _by_ids(s, new_ids)
    wd = timedelta(seconds=window_s)
    spans = [(r.taken_at - wd, r.taken_at + wd) for r in new if r.taken_at is not None]
    records = {r.id: r for r in new}
    bursts: dict[int, M.BurstGroup] = {}
    while spans:
        spans.sort()
        merged = [list(spans[0])]
        for lo, hi in spans[1:]:
            if lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        spans = [tuple(m) for m in merged]
        grew = False
        for lo, hi in spans:
            for r in s.exec(select(I).where(I.taken_at >= lo, I.taken_at <= hi)).all():
                records[r.id] = r
                if r.burst_id is not None and r.burst_id not in bursts:
                    b = s.get(B, r.burst_id)
                    if b is not None:
                        bursts[b.id] = b
                        for t in (b.start_time, b.end_time):
                            if t is not None:
                                spans.append((t - wd, t + wd))
                                grew = True
        if not grew:
            break
    for bid in bursts:
        for r in s.exec(select(I).where(I.burst_id == bid)).all():
            records[r.id] = r
    return list(records.values()), list(bursts.values())


def _load_items(s, ids: list[int], cfg: CuratorConfig | None = None) -> list[dict]:
    if not ids:
        return []
    rows = _by_ids(s, ids, lambda st: st.options(
        selectinload(M.ImageRecord.quality), selectinload(M.ImageRecord.semantic),
        selectinload(M.ImageRecord.faces)))
    by_id = {r.id: r for r in rows}
    return [_item(by_id[i], cfg) for i in ids if i in by_id]


_FP_RE = re.compile(r"^[0-9a-f]{10,64}$")


def _fingerprint(rec: M.ImageRecord) -> str:
    """Identifies this version of the photo; derived only from the database row."""
    return hashlib.sha256(f"{rec.quick_hash}|{rec.sha256}|{rec.file_size}|{rec.filename}"
                          .encode()).hexdigest()[:16]


class RatingReq(BaseModel):
    rating: int = Field(ge=0, le=5)  # 0 clears
    overwrite: bool = False


class RejectReq(BaseModel):
    reject: bool = True
    overwrite: bool = False


class FavoriteReq(BaseModel):
    favorite: bool | None = None  # omitted: toggle


class WatchReq(BaseModel):
    enabled: bool
    path: str | None = None


class KeywordsReq(BaseModel):
    remove: bool = False


class DupResolveReq(BaseModel):
    keep_id: int
    discard_ids: list[int] = Field(min_length=1, max_length=500)
    kind: Literal["exact", "near"]


class DupDismissReq(BaseModel):
    ids: list[int] = Field(min_length=1, max_length=500)


class MkdirReq(BaseModel):
    parent: str
    name: str


class BestShotReq(BaseModel):
    image_id: int


class CompositeReq(BaseModel):
    mode: Literal["fixed", "layered"] = "fixed"


class CleanScanReq(BaseModel):
    ids: list[int] = Field(min_length=1, max_length=5000)


class CleanExportReq(BaseModel):
    ids: list[int] = Field(min_length=1, max_length=5000)
    output_dir: str
    policy: Literal["blur", "skip", "include"] = "blur"
    html: bool = True
    rename: bool = True


class ApproveDeleteReq(BaseModel):
    approved: bool | None = None  # omitted: toggle


class ApproveDeletesReq(BaseModel):
    ids: list[int] | None = Field(default=None, max_length=5000)  # omitted: every proposed deletion
    approved: bool = True
    only_new: bool = False  # with no ids: only junk among the photos the last Analyze processed


# --------------------------------------------------------------- folder browser
def _browse_roots() -> list[Path]:
    """Folders the picker may show; PCUR_BROWSE_ROOTS confines it (used in Docker)."""
    raw = os.environ.get("PCUR_BROWSE_ROOTS", "").strip()
    if raw:
        found = [Path(p.strip()).expanduser() for p in raw.split(",") if p.strip()]
        return [r for r in found if r.is_dir()]
    roots = [Path.home()]
    if os.name == "nt":
        roots += [Path(f"{d}:/") for d in "CDEFGHIJKLMNOPQRSTUVWXYZ"
                  if Path(f"{d}:/").exists()]
    else:
        roots.append(Path("/"))
        for base in ("/Volumes", "/media", "/mnt"):
            b = Path(base)
            if b.is_dir():
                try:
                    roots += sorted(c for c in b.iterdir() if c.is_dir())
                except OSError:
                    pass
    seen, out = set(), []
    for r in roots:
        if str(r) not in seen:
            seen.add(str(r))
            out.append(r)
    return out


def _confined() -> bool:
    return bool(os.environ.get("PCUR_BROWSE_ROOTS", "").strip())


def _within(path: Path, roots: list[Path]) -> bool:
    try:
        rp = path.resolve()
    except OSError:
        return False
    return any(rp == r.resolve() or r.resolve() in rp.parents for r in roots)


def create_app(db_path: str | None = None) -> FastAPI:
    dbp = Path(db_path or os.environ.get("PCUR_DB_PATH") or default_db_path())
    init_db(dbp)
    cfg = CuratorConfig.load(dbp)
    db_fp = hashlib.sha1(str(dbp.resolve()).encode()).hexdigest()[:8]

    app = FastAPI(title="Photocure Review", version=__version__)
    app.state.db = dbp
    app.state.cfg = cfg
    auth_mgr = AuthManager(cfg.cache_dir)
    app.state.auth_mgr = auth_mgr
    app.state.token = auth_mgr.token
    app.state.allowed_hosts = build_allowed_hosts(cfg.host, cfg.port, cfg.allowed_hosts)
    from ..sidecar_queue import SidecarQueue
    app.state.sidecars = SidecarQueue(dbp, cfg.sidecar_debounce_seconds)
    progress_mgr = ProgressManager()
    app.state.progress = progress_mgr
    jobs: dict[str, dict] = {}
    app.state.jobs = jobs
    encoder_lock = threading.Lock()
    from ..watch_service import WatchService

    def req_cfg() -> CuratorConfig:
        return app.state.cfg.model_copy(deep=True)

    def run_cfg(body: RunOverrides | None = None) -> CuratorConfig:
        """Per request cfg: saved settings first, then explicit overrides from the body."""
        c = req_cfg()
        try:
            from ..services import apply_settings, load_settings
            c = apply_settings(c, load_settings(dbp))
            over: dict = {}
            for k in ("device", "device_index", "decode_workers", "torch_threads",
                      "scan_workers"):
                v = getattr(body, k, None)
                if v is not None:
                    over[k] = v
            legacy = getattr(body, "workers", None)
            if legacy is not None and "scan_workers" not in over:
                over["scan_workers"] = legacy
            if over:
                c = apply_settings(c, over)
        except ImportError:
            pass
        except ValueError as e:
            raise HTTPException(400, str(e))
        return c

    def detect_for(c: CuratorConfig):
        """Backend for this run; re-detects when the requested device changed."""
        from ..inference import backends as bk
        st = bk.detect_backend(c)
        if (getattr(st, "requested", c.device.device) != c.device.device
                or getattr(st, "device_index", c.device.device_index) != c.device.device_index):
            bk.reset_backend_cache()
            st = bk.detect_backend(c)
        return st

    # Every request: Host allowlist. Everything except the static shell, /healthz and
    # /auth/exchange also needs a valid session; mutating requests need a matching Origin.
    _OPEN = ("/", "/healthz", "/auth/exchange", "/api/about", "/licenses")

    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = request.headers.get("host", "")
        host_no_port = host.split(":")[0] if host else ""
        if (
            "*" not in app.state.allowed_hosts
            and host not in app.state.allowed_hosts
            and host_no_port not in app.state.allowed_hosts
        ):
            return JSONResponse({"detail": "invalid host header"}, status_code=400)
        path = request.url.path
        if path not in _OPEN and not path.startswith("/static/"):
            token = request.headers.get("x-pcur-token") or request.cookies.get(COOKIE_NAME)
            if not app.state.auth_mgr.validate(token):
                return JSONResponse({"detail": "missing or invalid token"}, status_code=401)
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            origin = request.headers.get("origin") or request.headers.get("referer")
            if origin:
                if not origin_allowed(origin, app.state.allowed_hosts, host):
                    return JSONResponse({"detail": "cross-origin request rejected"},
                                        status_code=403)
            elif (not request.headers.get("x-pcur-token")
                  and request.headers.get("sec-fetch-site", "same-origin")
                  not in ("same-origin", "none")):
                return JSONResponse({"detail": "cross-origin request rejected"}, status_code=403)
        return await call_next(request)

    auth: list = []  # enforced app-wide by the guard middleware

    @contextmanager
    def exclusive():
        if not _BUSY.acquire(blocking=False):
            raise HTTPException(409, "another task is already running")
        try:
            yield
        finally:
            _BUSY.release()

    def jlog(job: dict, msg: str, level: str = ""):
        job["log"].append({"t": time.time(), "level": level, "msg": msg})

    def spawn(kind: str, fn, cancellable: bool = True) -> dict:
        """Run fn(job) in a thread; the caller already holds _BUSY, released here."""
        job = {"id": uuid.uuid4().hex[:10], "kind": kind, "status": "running",
               "log": [], "result": None, "error": None}
        jobs[job["id"]] = job
        while len(jobs) > 20:
            old = next((k for k, j in list(jobs.items()) if j["status"] not in ("running", "stopping")),
                       None)
            if old is None:
                break
            jobs.pop(old, None)

        def run():
            try:
                job["result"] = fn(job)
                if cancellable and (progress_mgr.is_stopped() or job.get("status") == "stopping"):
                    job["status"] = "stopped"
                    jlog(job, "stopped by user", "warn")
                    progress_mgr.stop("Process stopped by user")
                else:
                    job["status"] = "done"
            except InterruptedError:
                job["status"] = "stopped"
                jlog(job, "stopped by user", "warn")
                progress_mgr.stop("Process stopped by user")
            except Exception as e:
                job["error"] = str(e)
                job["status"] = "error"
                jlog(job, f"failed: {e}", "err")
                progress_mgr.error(str(e))
            finally:
                _BUSY.release()

        try:
            threading.Thread(target=run, daemon=True).start()
        except Exception:
            _BUSY.release()
            raise
        return job

    def watch_ledger(job: dict, stop: threading.Event, total: int, *, after_id: int = 0,
                     tid: str | None = None):
        """Poll the ledger and stream one log line per action state change."""
        seen: dict[int, str] = {}

        def pump():
            with get_session(dbp) as s:
                q = select(M.ActionLedger)
                q = q.where(M.ActionLedger.transaction_id == tid) if tid \
                    else q.where(M.ActionLedger.id > after_id)
                rows = s.exec(q.order_by(M.ActionLedger.id)).all()
                for r in rows:
                    if seen.get(r.id) != r.status:
                        seen[r.id] = r.status
                        if r.status == "pending":
                            continue
                        lvl = "err" if r.status == "failed" else ""
                        jlog(job, f"{r.status} {r.operation_type} "
                                  f"{Path(r.source_path).name} -> {r.target_path}"
                                  + (f" ({r.error})" if r.error else ""), lvl)
            done = sum(1 for v in seen.values() if v in ("committed", "rolled_back", "failed"))
            progress_mgr.update(min(done, total), total)

        def loop():
            while not stop.is_set():
                try:
                    pump()
                except Exception:
                    pass
                stop.wait(0.25)
            try:
                pump()
            except Exception:
                pass

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        return t

    def make_planner(c: CuratorConfig) -> Planner:
        return Planner(c, db_path=dbp)

    def build(planner: Planner, body: PlanReq, root: Path, persist: bool):
        m = OrganizeMode.IN_PLACE if body.mode == "in_place" else OrganizeMode.CATEGORY_FIRST
        kw = {"organize_root": root, "operation": body.operation, "mode": m}
        if _accepts(planner.build_plan, "include_duplicates"):
            kw["include_duplicates"] = body.include_duplicates
        if _accepts(planner.build_plan, "persist"):
            kw["persist"] = persist
        return planner.build_plan(**kw)

    def latest_undoable() -> str | None:
        with get_session(dbp) as s:
            top = s.exec(select(func.max(M.ActionLedger.id)).where(
                M.ActionLedger.status == "committed")).one()
            if top is None:
                return None
            return s.exec(select(M.ActionLedger.transaction_id).where(
                M.ActionLedger.id == top)).first()

    # ------------------------------------------------------------------ read api
    def backend_info() -> dict:
        """Which AI backends are active, and how many rows the placeholder analysed."""
        info: dict = {}
        try:
            info = dict(make_planner(req_cfg()).backend_info())
        except Exception:
            try:
                from ..services import describe_backends
                info = dict(describe_backends(req_cfg()))
            except Exception:
                info = {}
        if not info:
            try:
                from ..inference.backends import detect_backend
                bs = detect_backend(req_cfg())
                info = {"device": bs.device, "semantic_backend": bs.semantic_backend,
                        "face_backend": bs.face_backend}
            except Exception:
                info = {"device": "cpu", "semantic_backend": "stub", "face_backend": "stub"}
        sem = info.get("semantic_backend", "stub")
        info.setdefault("semantic_is_stub", sem == "stub")
        info.setdefault("face_is_fallback", info.get("face_backend") != "yunet")
        info.setdefault("hint", 'pip install torch torchvision open-clip-torch')
        info.setdefault("stub_rows", 0)
        return info

    @app.post("/auth/exchange")
    async def exchange_otc(request: Request):
        """Trade a one-time code (or the persistent token) for the session cookie."""
        try:
            code = (await request.json()).get("code")
        except Exception:
            raise HTTPException(400, "invalid request body")
        mgr = app.state.auth_mgr
        client = request.client.host if request.client else ""
        if mgr.locked_out(client):
            raise HTTPException(429, "too many attempts, try again later")
        token = mgr.exchange(code if isinstance(code, str) else None)
        if not token:
            mgr.record_failure(client)
            raise HTTPException(401, "invalid or expired code")
        resp = Response(status_code=204)
        secure = (request.url.scheme == "https"
                  or request.headers.get("x-forwarded-proto", "").lower() == "https")
        resp.set_cookie(COOKIE_NAME, token, httponly=True, samesite="strict", path="/",
                        secure=secure, max_age=COOKIE_MAX_AGE)
        return resp

    @app.get("/api/config")
    def get_config():
        tax = app.state.cfg.taxonomy
        return {
            "default_output_dir": os.environ.get("PCUR_DEFAULT_OUTPUT")
            or str((dbp.resolve().parent / "Organized")),
            "categories": [{"value": c.value, "junk": c.is_junk, "utility": c.is_utility}
                           for c in CategoryType],
            "backend": backend_info(),
            "thresholds": {"min_confidence": tax.min_confidence,
                           "min_margin": tax.min_margin},
        }

    @app.get("/api/about")
    def get_about():
        info = backend_info()
        device = info.get("device", "unknown")
        models = []
        try:
            manifest_path = Path(__file__).parent.parent / "models" / "MANIFEST.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                    models_dir = manifest_path.parent
                    for m in manifest.get("models", []):
                        file = m.get("file", "")
                        if file and (models_dir / file).exists():
                            models.append({"name": m.get("name", file), "file": file})
        except Exception:
            pass
        return {
            "version": __version__,
            "author": "Safiyu",
            "license": "MIT License",
            "device": device,
            "models": models,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        }

    @app.get("/licenses")
    def get_licenses():
        notice_path = Path(__file__).parent.parent.parent / "NOTICE"
        if notice_path.exists():
            with open(notice_path) as f:
                content = f.read()
            return HTMLResponse(f"<pre>{content}</pre>")
        return HTMLResponse("<pre>License information not available.</pre>")

    def settings_payload() -> dict:
        """Saved settings, what will really run, and every accelerator with its status."""
        try:
            from ..config import resolve_workers
            from ..services import apply_settings, load_settings
        except ImportError:
            raise _not_impl("settings")
        saved = load_settings(dbp)
        eff_cfg = apply_settings(req_cfg(), saved)
        info = backend_info()
        accelerators = info.get("accelerators")
        if accelerators is None:
            try:
                from ..inference.backends import list_accelerators
                accelerators = list_accelerators()
            except Exception:
                accelerators = []
        selected = info.get("selected") or info.get("device") or "cpu"
        effective = dict(resolve_workers(eff_cfg, info.get("device")))
        effective["device"] = selected
        effective["device_index"] = eff_cfg.device.device_index
        return {"settings": saved, "effective": effective, "accelerators": accelerators,
                "selected": selected,
                "requested": info.get("requested") or eff_cfg.device.device,
                "fallback_reason": info.get("fallback_reason", "")}

    @app.get("/api/settings")
    def get_settings():
        return settings_payload()

    @app.post("/api/settings", dependencies=auth)
    def post_settings(payload: dict = Body(...)):
        try:
            from ..services import apply_settings, save_settings
        except ImportError:
            raise _not_impl("settings")
        try:
            saved = save_settings(dbp, payload)
        except ValueError as e:
            raise HTTPException(400, str(e))
        # Re-detect now so the page shows the device that will run, unless a job is busy.
        if _BUSY.acquire(blocking=False):
            try:
                from ..inference import backends as bk
                bk.reset_backend_cache()
                bk.detect_backend(apply_settings(req_cfg(), saved))
            except Exception:
                pass
            finally:
                _BUSY.release()
        return settings_payload()

    @app.get("/api/stats")
    def stats():
        with get_session(dbp) as s:
            return PhotoRepository(s).stats()

    @app.get("/api/progress")
    def get_progress():
        return {**progress_mgr.get(), "busy": _BUSY.locked()}

    @app.get("/api/plan")
    def plan():
        """Summarize the currently planned routing (route_status == planned)."""
        eff = func.coalesce(M.ImageRecord.manual_bucket, M.ImageRecord.category)
        with get_session(dbp) as s:
            rows = s.exec(select(eff, func.count()).where(
                M.ImageRecord.scan_status != "missing",
                M.ImageRecord.route_status == "planned").group_by(eff)).all()
            by_cat = {(k or "unrouted"): int(n) for k, n in rows}
            return {"total_planned": sum(by_cat.values()), "by_category": by_cat,
                    "organize_root": str(cfg.organize_root)}

    @app.get("/api/images")
    def images(
        category: str | None = Query(None),
        categories: str | None = Query(None),
        group: Literal["junk", "utility"] | None = Query(None),
        manual: bool = Query(False),
        q: str | None = Query(None),
        near_dup: bool = Query(False),
        highlights: bool = Query(False),
        min_aesthetic: float | None = Query(None),
        locked: bool | None = Query(None),
        event_id: int | None = Query(None),
        camera: str | None = Query(None, max_length=200),
        rating: int | None = Query(None, ge=-1, le=5),
        min_rating: int | None = Query(None, ge=1, le=5),
        country: str | None = Query(None, min_length=2, max_length=2),
        favorite: bool | None = Query(None),
        issue: str | None = Query(None, max_length=100),
        date_from: date | None = Query(None),
        date_to: date | None = Query(None),
        sort: Literal["id", "taken"] = Query("id"),
        limit: int = Query(200, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ):
        I = M.ImageRecord
        eff = func.coalesce(I.manual_bucket, I.category)
        conds = [I.scan_status != "missing"]
        if issue:
            conds.append(_issue_clause(issue))
        if category:
            conds.append(eff == category)
        if categories:
            conds.append(eff.in_([c for c in categories.split(",") if c]))
        if group == "junk":
            conds.append(eff.in_(JUNK_VALUES))
        elif group == "utility":
            conds.append(eff.in_(UTILITY_VALUES))
        if manual:
            conds.append(I.manual_bucket.is_not(None))
        if q:
            like = "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            conds.append(I.filename.ilike(like, escape="\\")
                         | I.relative_path.ilike(like, escape="\\"))
        if near_dup:
            conds.append(I.near_dup_of.is_not(None))
        if locked is not None:
            conds.append(I.locked == locked)
        if event_id is not None:
            conds.append(I.event_id == event_id)
        # Every value below reaches SQL as a bound parameter, never as text.
        if camera:
            conds.append(I.camera_model == camera)
        if rating is not None:
            conds.append(func.coalesce(I.rating, 0) == rating)  # null counts as unrated (0)
        if min_rating is not None:
            conds.append(I.rating >= min_rating)
        if country:
            conds.append(I.country_code == country.upper())
        if favorite is not None:
            conds.append(I.is_favorite == favorite)
        # taken_at is stored as UTC, so a date range is a range of UTC calendar days.
        if date_from is not None:
            conds.append(I.taken_at >= datetime.combine(date_from, dtime.min, tzinfo=timezone.utc))
        if date_to is not None:
            conds.append(I.taken_at < datetime.combine(date_to, dtime.min, tzinfo=timezone.utc) + timedelta(days=1))
        need_sem = highlights or min_aesthetic is not None
        if need_sem:
            conds.append(M.SemanticScore.aesthetic.is_not(None))
        if highlights:
            from ..highlights import candidate_filters
            conds.extend(candidate_filters())
        if min_aesthetic is not None:
            conds.append(M.SemanticScore.aesthetic >= min_aesthetic)

        def apply(stmt):
            if need_sem:
                stmt = stmt.join(M.SemanticScore, M.SemanticScore.image_id == I.id)
            for c in conds:
                stmt = stmt.where(c)
            return stmt

        with get_session(dbp) as s:
            total = s.exec(apply(select(func.count()).select_from(I))).one()
            if highlights:
                from ..highlights import highlight_expr
                expr = highlight_expr(hl_weights())
                ranked = s.exec(apply(select(I.id, expr)).order_by(expr.desc(), I.id)
                                .offset(offset).limit(limit)).all()
                ids = [i for i, _ in ranked]
                items = _load_items(s, list(ids), app.state.cfg)
                hl = {int(i): float(v) for i, v in ranked}
                for it in items:
                    it["highlight"] = round(hl.get(it["id"], 0.0), 3)
                return {"total": int(total), "offset": offset, "limit": limit, "items": items}
            order = [I.taken_at.asc().nulls_last(), I.id] if sort == "taken" else [I.id]
            ids = s.exec(apply(select(I.id)).order_by(*order)
                         .offset(offset).limit(limit)).all()
            return {"total": int(total), "offset": offset, "limit": limit,
                    "items": _load_items(s, list(ids), app.state.cfg)}

    def hl_weights():
        from ..features import load_features
        from ..highlights import HighlightWeights
        return HighlightWeights.from_features(load_features(dbp))

    @app.get("/api/features")
    def get_features():
        from ..features import DEFAULTS, load_features
        return {"features": load_features(dbp), "defaults": dict(DEFAULTS)}

    @app.post("/api/features", dependencies=auth)
    def post_features(payload: dict = Body(...)):
        from ..features import DEFAULTS, save_features
        try:
            saved = save_features(dbp, payload)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"features": saved, "defaults": dict(DEFAULTS)}

    def near_cosine() -> float:
        return float(getattr(app.state.cfg.taxonomy, "near_dup_min_cosine", 0.93))

    @app.get("/api/duplicates")
    def get_duplicates(kind: Literal["exact", "near"] | None = Query(None),
                       limit: int = Query(30, ge=1, le=200), offset: int = Query(0, ge=0),
                       reviewed: bool = Query(False)):
        from .. import duplicates as dups
        with get_session(dbp) as s:
            found = dups.find_groups(s, include_reviewed=True, near_min_cosine=near_cosine())
            everything = [g for g in found if (kind is None or g["kind"] == kind)
                          and (reviewed or not g["reviewed"])]
            page = everything[offset:offset + limit]
            items = {it["id"]: it for it in _load_items(
                s, [m["id"] for g in page for m in g["members"]], app.state.cfg)}
            for g in page:
                for m in g["members"]:
                    m["item"] = items.get(m["id"])
            return {"summary": dups.summarize([g for g in found if not g["reviewed"]]),
                    "total": len(everything), "offset": offset, "limit": limit, "groups": page}

    @app.post("/api/duplicates/resolve", dependencies=auth)
    def resolve_duplicates(body: DupResolveReq):
        from .. import duplicates as dups
        with exclusive(), get_session(dbp) as s:
            try:
                result = dups.resolve_group(s, body.keep_id, body.discard_ids, body.kind)
            except LookupError as e:
                raise HTTPException(404, str(e))
            except ValueError as e:
                raise HTTPException(400, str(e))
            s.flush()
            return {"result": result, "summary": dups.summary(s, near_cosine())}

    @app.post("/api/duplicates/dismiss", dependencies=auth)
    def dismiss_duplicates(body: DupDismissReq):
        from .. import duplicates as dups
        with exclusive(), get_session(dbp) as s:
            cut = dups.dismiss_near(s, body.ids)
            return {"result": {"unlinked": cut}, "summary": dups.summary(s, near_cosine())}

    @app.post("/api/duplicates/resolve-all", dependencies=auth)
    def resolve_all_duplicates(payload: dict = Body(default={})):
        from .. import duplicates as dups
        with exclusive(), get_session(dbp) as s:
            result = dups.resolve_all(s, near_cosine())
            return {"result": result, "summary": dups.summary(s, near_cosine())}

    def watch_process(c: CuratorConfig) -> dict:
        res = do_infer(c, InferReq())
        routed = make_planner(c).route_new()
        return {"processed": res.get("processed", 0) if isinstance(res, dict) else 0, "routed": routed}

    app.state.watch = WatchService(dbp, lambda: run_cfg(None), busy=_BUSY, process_fn=watch_process,
                                   debounce_s=cfg.watch_debounce_seconds,
                                   on_cancel=progress_mgr.request_stop)

    @app.get("/api/watch")
    def get_watch():
        return app.state.watch.status()

    @app.post("/api/watch", dependencies=auth)
    def set_watch(body: WatchReq):
        from ..features import load_features, save_features
        if not body.enabled:
            app.state.watch.stop()  # if an analysis is still winding down, the status says "stopping"
            save_features(dbp, {"watch_enabled": False})
            return app.state.watch.status()
        path = (body.path or load_features(dbp)["watch_path"]).strip()
        if not path:
            raise HTTPException(400, "Choose a folder to watch.")
        if _confined() and not _within(Path(path).expanduser(), _browse_roots()):
            raise HTTPException(403, "that folder is outside the allowed locations")
        try:
            status = app.state.watch.start(path)
        except FileNotFoundError as e:
            raise HTTPException(400, str(e))
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        save_features(dbp, {"watch_enabled": True, "watch_path": status["path"]})
        return status

    @app.get("/api/keywords")
    def keywords_status():
        from ..features import load_features
        with get_session(dbp) as s:
            photos = s.exec(select(func.count()).select_from(M.ImageRecord)
                            .where(M.ImageRecord.scan_status != "missing")).one()
        return {"enabled": bool(load_features(dbp)["write_keywords"]),
                "sidecars": bool(app.state.cfg.write_sidecars), "photos": int(photos)}

    @app.post("/api/keywords/write", dependencies=auth, status_code=202)
    def write_keywords(body: KeywordsReq = KeywordsReq()):
        """Write, or remove, Photocure keywords in the XMP sidecars (background job)."""
        from .. import keywords as kw
        if not app.state.cfg.write_sidecars:
            raise HTTPException(400, "Sidecar writing is turned off in the settings.")
        if not _BUSY.acquire(blocking=False):
            raise HTTPException(409, "another task is already running")

        def work(job: dict) -> dict:
            verb = "Removing" if body.remove else "Writing"
            progress_mgr.start("Keywords", f"{verb} keywords in XMP sidecars", 0, "Starting...")
            jlog(job, f"{verb.lower()} keywords")

            def on_prog(done, total):
                progress_mgr.update(done, total, f"{verb} keywords", detail=f"{done} / {total}")

            try:
                res = kw.write_all(dbp, app.state.cfg, keywords=not body.remove,
                                   io_lock=app.state.sidecars.io_lock, on_progress=on_prog,
                                   cancel_check=progress_mgr.is_stopped)
            except Exception as e:
                progress_mgr.error(str(e))
                raise
            if res.get("cancelled"):
                progress_mgr.stop("Keyword export stopped by user")
            else:
                progress_mgr.finish(f"Keywords done: {res['written']} sidecars updated")
            jlog(job, f"{res['written']} updated, {res['edited_elsewhere']} left alone "
                      f"because another program edited them")
            return {"ok": True, **res}

        job = spawn("keywords", work)
        return {"ok": True, "job_id": job["id"]}

    @app.get("/api/preferences")
    def get_preferences():
        from ..inference.preference import status
        with get_session(dbp) as s:
            return status(s)

    @app.post("/api/preferences/learn", dependencies=auth)
    def learn_preferences():
        from ..inference import preference as pref
        with exclusive(), get_session(dbp) as s:
            res = pref.train_preferences(s)
            st = pref.status(s)
        return {"status": res.status, "n_positive": res.n_positive,
                "n_negative": res.n_negative, "cv_accuracy": round(res.cv_accuracy, 3),
                "scored": res.scored, "preferences": st}

    @app.post("/api/preferences/reset", dependencies=auth)
    def reset_preferences():
        from ..inference import preference as pref
        with exclusive(), get_session(dbp) as s:
            pref.reset_preferences(s)
            return pref.status(s)

    @app.get("/api/highlights/events")
    def highlights_by_event(per_event: int = Query(1, ge=1, le=12),
                            min_score: float | None = Query(None, ge=0, le=2),
                            balanced: bool = Query(False)):
        from .. import narrative
        from .. import search as search_mod
        from ..highlights import best_of_events
        with get_session(dbp) as s:
            scale_of = None
            if balanced:
                def scale_of(ids):
                    with encoder_lock:
                        encoder = getattr(app.state, "text_encoder", None)
                        if encoder is None:
                            encoder = app.state.text_encoder = search_mod.default_encoder(run_cfg())
                    return narrative.scale_labels(s, ids, encoder)
            events = best_of_events(s, hl_weights(), per_event, min_score, scale_of)
            items = {it["id"]: it for it in _load_items(
                s, [i for e in events for i, _ in e["items"]], app.state.cfg)}
            out = []
            for e in events:
                scales = e.get("scales")
                row_items = []
                for image_id, score in e["items"]:
                    if image_id in items:
                        row = {**items[image_id], "highlight": round(score, 3)}
                        if scales is not None:
                            row["scale"] = scales.get(image_id)
                        row_items.append(row)
                entry = {"event_id": e["event_id"], "label": e["label"],
                         "start": e["start"].isoformat() if e["start"] else None,
                         "end": e["end"].isoformat() if e["end"] else None,
                         "total": e["total"], "count": len(row_items), "items": row_items}
                if scales is not None:
                    entry["coverage"] = e["coverage"]
                out.append(entry)
        used = any("coverage" in e for e in out)
        note = ("" if used or not balanced else
                "Event stories need the AI model and analyzed photos. Showing top photos instead.")
        return {"events": out, "balanced": used, "note": note}

    @app.get("/api/highlights/years")
    def highlight_years():
        from ..highlights import years_available
        with get_session(dbp) as s:
            return {"years": [{"year": y, "count": c} for y, c in years_available(s)]}

    @app.get("/api/highlights/export")
    def highlights_export(year: int = Query(..., ge=1800, le=9999), n: int = Query(50, ge=1, le=500)):
        from ..highlights import export_html, top_of_year
        with get_session(dbp) as s:
            ranked = top_of_year(s, hl_weights(), year, n)
            if not ranked:
                raise HTTPException(404, f"No scored photos from {year}.")
            page = export_html(s, ranked, f"Best of {year}")
        return Response(page, media_type="text/html; charset=utf-8", headers={
            "Content-Disposition": f'attachment; filename="highlights-{year}.html"'})

    @app.get("/api/facets")
    def facets():
        """What the library contains, for the filter controls and the date scrubber."""
        I = M.ImageRecord
        active = I.scan_status != "missing"
        with get_session(dbp) as s:
            cams = s.exec(select(I.camera_model, func.count()).where(active, I.camera_model != "")
                          .group_by(I.camera_model).order_by(func.count().desc(), I.camera_model)).all()
            rated = dict(s.exec(select(func.coalesce(I.rating, 0), func.count()).where(active)
                                .group_by(func.coalesce(I.rating, 0))).all())
            places = s.exec(select(I.country_code, func.count()).where(active, I.country_code != "")
                            .group_by(I.country_code).order_by(func.count().desc(), I.country_code)).all()
            lo, hi = s.exec(select(func.min(I.taken_at), func.max(I.taken_at))
                            .where(active, I.taken_at.is_not(None))).one()
            favs = s.exec(select(func.count()).select_from(I).where(active, I.is_favorite == True)).one()  # noqa: E712
            total = s.exec(select(func.count()).select_from(I).where(active)).one()
            issues = {n: int(s.exec(select(func.count()).select_from(I).where(active, _issue_clause(n))).one())
                      for n in _ISSUES}
            issues["any"] = int(s.exec(select(func.count()).select_from(I).where(
                active, _issue_clause(",".join(_ISSUES)))).one())
        return {
            "total": int(total), "favorites": int(favs), "issues": issues,
            "cameras": [{"name": n, "count": int(c)} for n, c in cams],
            "ratings": {str(k): int(rated.get(k, 0)) for k in (-1, 0, 1, 2, 3, 4, 5)},
            "countries": [{"code": c, "count": int(n)} for c, n in places],
            "taken_min": str(lo)[:10] if lo else None, "taken_max": str(hi)[:10] if hi else None,
        }

    def _set_rating(image_id: int, value: int | None, overwrite: bool):
        """Change a photo's rating. Over a sidecar edited outside Photocure this asks first."""
        from ..sidecar_sync import foreign_state
        # The queue's lock keeps this comparison from seeing a half-finished background write.
        with app.state.sidecars.io_lock, get_session(dbp) as s:
            rec = s.get(M.ImageRecord, image_id)
            if rec is None:
                raise HTTPException(404, "image not found")
            if app.state.cfg.write_sidecars and not overwrite:
                state = foreign_state(rec)
                if state:
                    return JSONResponse({
                        "ok": False, "needs_confirmation": True, "reason": state,
                        "message": "This photo's sidecar was modified outside Photocure."}, status_code=409)
            rec.rating = value
            rec.updated_at = datetime.now(timezone.utc)
            s.flush()
        if app.state.cfg.write_sidecars:
            app.state.sidecars.schedule(image_id, overwrite)
        with get_session(dbp) as s:
            return {"ok": True, "item": _load_items(s, [image_id], app.state.cfg)[0]}

    @app.post("/api/images/{image_id}/rating", dependencies=auth)
    def rate_image(image_id: int, body: RatingReq):
        return _set_rating(image_id, body.rating or None, body.overwrite)

    @app.post("/api/images/{image_id}/reject", dependencies=auth)
    def reject_image(image_id: int, body: RejectReq = RejectReq()):
        return _set_rating(image_id, -1 if body.reject else None, body.overwrite)

    @app.post("/api/images/{image_id}/favorite", dependencies=auth)
    def favorite_image(image_id: int, body: FavoriteReq = FavoriteReq()):
        with get_session(dbp) as s:
            rec = s.get(M.ImageRecord, image_id)
            if rec is None:
                raise HTTPException(404, "image not found")
            rec.is_favorite = (not rec.is_favorite) if body.favorite is None else body.favorite
            rec.updated_at = datetime.now(timezone.utc)
            s.flush()
        with get_session(dbp) as s:
            return {"ok": True, "item": _load_items(s, [image_id], app.state.cfg)[0]}

    @app.post("/api/images/{image_id}/approve-delete", dependencies=auth)
    def approve_delete(image_id: int, body: ApproveDeleteReq = ApproveDeleteReq()):
        """Mark one photo for deletion (or clear the mark): it is left out of the output."""
        with get_session(dbp) as s:
            rec = s.get(M.ImageRecord, image_id)
            if rec is None:
                raise HTTPException(404, "image not found")
            rec.delete_approved = (not rec.delete_approved) if body.approved is None else body.approved
            rec.updated_at = datetime.now(timezone.utc)
            s.flush()
        with get_session(dbp) as s:
            return {"ok": True, "item": _load_items(s, [image_id], app.state.cfg)[0]}

    @app.post("/api/deletes/approve", dependencies=auth)
    def approve_deletes(body: ApproveDeletesReq = ApproveDeletesReq()):
        """Mark every junk photo for deletion, or exactly the given ids."""
        I = M.ImageRecord
        active = I.scan_status != "missing"
        with exclusive(), get_session(dbp) as s:
            if body.ids is None and body.only_new:
                junk = func.coalesce(I.manual_bucket, I.category).in_(JUNK_VALUES)
                recs = _by_ids(s, getattr(app.state, "last_analyzed_ids", ()),
                               lambda st: st.where(active, junk))
            elif body.ids is None:
                recs = s.exec(select(I).where(
                    active, func.coalesce(I.manual_bucket, I.category).in_(JUNK_VALUES))).all()
            else:
                recs = _by_ids(s, body.ids, lambda st: st.where(active))
            changed = 0
            for rec in recs:
                if rec.delete_approved != body.approved:
                    rec.delete_approved = body.approved
                    changed += 1
            total = s.exec(select(func.count()).select_from(I).where(
                active, I.delete_approved == True)).one()  # noqa: E712
            return {"ok": True, "changed": changed, "approved_total": int(total)}

    @app.get("/api/images/{image_id}/similar")
    def similar(image_id: int, k: int = Query(12, ge=1, le=100)):
        try:
            from ..clustering.similarity import find_similar
        except ImportError:
            raise _not_impl("similarity search")
        with get_session(dbp) as s:
            if s.get(M.ImageRecord, image_id) is None:
                raise HTTPException(404, "image not found")
            pairs = list(find_similar(s, image_id, k))
            items = _load_items(s, [int(i) for i, _ in pairs], app.state.cfg)
            score = {int(i): float(sc) for i, sc in pairs}
            for it in items:
                it["score"] = round(score.get(it["id"], 0.0), 4)
            return {"id": image_id, "items": items}

    @app.get("/api/search")
    def search_photos(q: str = Query(..., max_length=300), k: int = Query(60, ge=1, le=300),
                      duplicates: bool = Query(False), strict: bool = Query(True)):
        from .. import search as search_mod
        text = " ".join(q.split())
        if not text:
            raise HTTPException(400, "Type what you are looking for.")
        with encoder_lock:  # two cold searches must not each load a CLIP text tower
            encoder = getattr(app.state, "text_encoder", None)
            if encoder is None:
                encoder = app.state.text_encoder = search_mod.default_encoder(run_cfg())
        try:
            with get_session(dbp) as s:
                res = search_mod.search_ranked(s, encoder, text, k, include_duplicates=duplicates, strict=strict)
                items = _load_items(s, [i for i, _ in res.hits], app.state.cfg)
                score = dict(res.hits)
                for it in items:
                    it["score"] = round(float(score.get(it["id"], 0.0)), 4)
        except search_mod.SearchUnavailable as e:
            raise HTTPException(503, str(e))
        return {"query": text, "items": items, "strong": res.strong, "more": res.more, "parts": res.parts}

    @app.get("/thumbnail/{image_id}")
    def thumbnail(image_id: int, tier: Literal["micro", "medium", "preview"] = Query("micro"),
                  v: str | None = Query(None)):
        """WebP thumbnail in one of three sizes. The file is chosen from the database row only;
        ``v`` is a cache-buster and must look like a hex fingerprint."""
        if v is not None and not _FP_RE.fullmatch(v):
            raise HTTPException(400, "invalid cache key")
        with get_session(dbp) as s:
            rec = s.get(M.ImageRecord, image_id)
            if rec is None:
                raise HTTPException(404, "image not found")
            cands = [p for p in (rec.path, rec.target_path) if p]
            fp = _fingerprint(rec)
            ext, orientation = rec.extension, rec.orientation
        tier_dir = cfg.cache_dir / "thumbs" / tier
        cache = tier_dir / f"{db_fp}_{image_id}_{fp}.webp"
        headers = {"Cache-Control": "public, max-age=31536000, immutable"}
        if cache.exists():
            return FileResponse(cache, media_type="image/webp", headers=headers)
        path = next((p for p in cands if os.path.exists(p)), None)
        if path is None:
            raise HTTPException(404, "image file not found on disk")
        from ..thumbs import render_tier
        try:
            data = render_tier(path, ext or Path(path).suffix, orientation, tier)
        except Exception as e:
            logging.getLogger("photocure.ui").warning("thumbnail decode failed: %s", e)
            raise HTTPException(500, "thumbnail decode failed")
        tier_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(f".{uuid.uuid4().hex[:6]}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, cache)
        return Response(data, media_type="image/webp", headers=headers)

    @app.get("/api/images/{image_id}/faces")
    def image_faces(image_id: int):
        with get_session(dbp) as s:
            rec = s.get(M.ImageRecord, image_id)
            if rec is None:
                raise HTTPException(404, "image not found")
            fp = _fingerprint(rec)
            faces = s.exec(select(M.FaceDetection).where(M.FaceDetection.image_id == image_id)
                           .order_by(M.FaceDetection.id)).all()
            person_ids = {f.person_id for f in faces if f.person_id is not None}
            names = {p.id: p.name for p in (
                s.exec(select(M.PersonCluster).where(col(M.PersonCluster.id).in_(person_ids))).all()
                if person_ids else [])}
            return {
                "image_id": image_id, "filename": rec.filename, "width": rec.width,
                "height": rec.height, "orientation": rec.orientation,
                "taken_at": rec.taken_at.isoformat() if rec.taken_at else None,
                "items": [{
                    "id": f.id, "confidence": round(f.confidence, 3),
                    "box": [f.x1, f.y1, f.x2, f.y2] if f.x1 is not None else None,
                    "eye_status": f.eye_status, "is_blink": bool(f.is_blink),
                    "eye_closed_score": (round(f.eye_closed_score, 3)
                                         if f.eye_closed_score is not None else None),
                    "sharpness": round(f.face_sharpness, 1) if f.face_sharpness is not None else None,
                    "url": f"/api/faces/{f.id}.jpg?v={fp}" if f.x1 is not None else None,
                    "person_id": f.person_id,
                    "person_name": names.get(f.person_id) if f.person_id is not None else None,
                } for f in faces],
            }

    @app.get("/api/people")
    def list_people_endpoint(sort: Literal["similarity", "count", "name"] = "similarity"):
        """Known people clusters, arranged by similarity (default), count, or name."""
        from ..people_actions import list_people
        return {"items": list_people(dbp, sort=sort)}

    @app.delete("/api/people/{person_id}", dependencies=auth)
    def delete_person_endpoint(person_id: int):
        """Delete a person cluster and unassign all its associated faces."""
        from ..people_actions import ActionError, delete_person
        try:
            return delete_person(dbp, person_id)
        except ActionError as e:
            raise HTTPException(e.status, str(e)) from None

    @app.post("/api/people/bulk-delete", dependencies=auth)
    def bulk_delete_people_endpoint(body: BulkDeletePeopleReq):
        """Delete multiple person clusters in a single atomic transaction."""
        from ..people_actions import bulk_delete_people
        return bulk_delete_people(dbp, body.person_ids)

    @app.post("/api/people/{person_id}/rename", dependencies=auth)
    def rename_person_endpoint(person_id: int, body: RenamePersonReq):
        """Rename a person, or merge into an existing person with the same name."""
        from ..people_actions import ActionError, rename_person
        try:
            return rename_person(dbp, person_id, body.name)
        except ActionError as e:
            raise HTTPException(e.status, str(e)) from None

    @app.post("/api/people/{source_id}/merge-into/{target_id}", dependencies=auth)
    def merge_people_endpoint(source_id: int, target_id: int):
        """Merge all faces from source cluster into target cluster (drag-and-drop merge)."""
        from ..people_actions import ActionError, merge_people
        try:
            return merge_people(dbp, source_id, target_id)
        except ActionError as e:
            raise HTTPException(e.status, str(e)) from None

    @app.post("/api/people/recluster", dependencies=auth)
    def recluster_people_endpoint(threshold: float = Query(0.64, ge=0.0, le=1.0)):
        """Recluster all detected faces with strict identity threshold (default 0.64)."""
        from ..people_actions import recluster_people
        return recluster_people(dbp, threshold=threshold)

    @app.get("/api/people/{person_id}/photos")
    def person_photos_endpoint(person_id: int):
        """All distinct photos that contain a face belonging to *person_id*, newest first."""
        with get_session(dbp) as s:
            person = s.get(M.PersonCluster, person_id)
            if person is None:
                raise HTTPException(404, "person not found")
            # Fetch all face rows for this person that have a valid crop box
            faces = s.exec(
                select(M.FaceDetection)
                .where(M.FaceDetection.person_id == person_id)
                .where(M.FaceDetection.x1.isnot(None))
            ).all()
            # Deduplicate: keep one face (sharpest) per image
            best: dict[int, M.FaceDetection] = {}
            for f in faces:
                prev = best.get(f.image_id)
                if prev is None or (f.face_sharpness or 0) > (prev.face_sharpness or 0):
                    best[f.image_id] = f
            # Load image records
            image_ids = list(best.keys())
            if not image_ids:
                return {"person": {"id": person.id, "name": person.name, "face_count": person.face_count},
                        "photos": []}
            recs = sorted((r for r in _by_ids(s, image_ids) if r.scan_status != "missing"),
                          key=lambda r: (r.taken_at is not None, r.taken_at),
                          reverse=True)
            photos = []
            for r in recs:
                face = best.get(r.id)
                fp = _fingerprint(r)
                photos.append({
                    "id": r.id,
                    "filename": r.filename,
                    "taken_at": r.taken_at.isoformat() if r.taken_at else None,
                    "thumb_url": f"/thumbnail/{r.id}?tier=medium&v={fp}",
                    "face_id": face.id if face else None,
                    "face_url": f"/api/faces/{face.id}.jpg" if face else None,
                })
            return {
                "person": {"id": person.id, "name": person.name, "face_count": person.face_count},
                "photos": photos,
            }

    @app.post("/api/faces/{face_id}/reassign", dependencies=auth)
    def reassign_face_endpoint(face_id: int, body: ReassignFaceReq):
        """Move one face to a different (or brand-new) person."""
        from ..people_actions import ActionError, reassign_face
        try:
            return reassign_face(dbp, face_id, person_id=body.person_id, new_name=body.new_name)
        except ActionError as e:
            raise HTTPException(e.status, str(e)) from None

    @app.get("/api/faces/{face_name}")
    def face_crop(face_name: str, v: str | None = Query(None)):
        """JPEG crop of one face. The path comes from the database row, never from the request;
        ``v`` is only a cache-buster and must look like a hex fingerprint."""
        if v is not None and not _FP_RE.fullmatch(v):
            raise HTTPException(400, "invalid cache key")
        stem, dot, ext = face_name.rpartition(".")
        if ext != "jpg" or not re.fullmatch(r"[0-9]+", stem):
            raise HTTPException(404, "face not found")
        face_id = int(stem)
        with get_session(dbp) as s:
            face = s.get(M.FaceDetection, face_id)
            rec = s.get(M.ImageRecord, face.image_id) if face is not None else None
            if face is None or rec is None:
                raise HTTPException(404, "face not found")
            if face.x1 is None:
                raise HTTPException(404, "this photo needs to be analysed again")
            box = (face.x1, face.y1, face.x2, face.y2)
            fp = _fingerprint(rec)
            cands = [p for p in (rec.path, rec.target_path) if p]
            orientation = rec.orientation
        faces_dir = cfg.cache_dir / "faces"
        cache = faces_dir / f"{face_id}_{fp}.jpg"
        headers = {"Cache-Control": "public, max-age=31536000, immutable"}
        if cache.exists():
            return FileResponse(cache, media_type="image/jpeg", headers=headers)
        path = next((p for p in cands if os.path.exists(p)), None)
        if path is None:
            raise HTTPException(404, "image file not found on disk")
        try:
            import cv2

            from ..inference.face_engine import crop_face, shrink_to
            from ..quality.image_loader import load_image
            full = load_image(path, max_side=None, orientation=orientation)
            crop = shrink_to(crop_face(full.bgr, box, app.state.cfg.face.crop_margin), 512)
            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise RuntimeError("could not encode the crop")
        except Exception as e:
            logging.getLogger("photocure.ui").warning("face crop failed: %s", e)
            raise HTTPException(500, "face crop failed")
        faces_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(f".{uuid.uuid4().hex[:6]}.tmp")
        tmp.write_bytes(buf.tobytes())
        os.replace(tmp, cache)
        return Response(buf.tobytes(), media_type="image/jpeg", headers=headers)

    @app.post("/api/bursts/{burst_id}/reject-blinks", dependencies=auth)
    def reject_blinks(burst_id: int):
        """Send the closed-eyed alternates of a burst to Junk_Review/Blinks. Reversible; no file moves."""
        with get_session(dbp) as s:
            if s.get(M.BurstGroup, burst_id) is None:
                raise HTTPException(404, "burst not found")
            members = s.exec(select(M.ImageRecord).where(
                M.ImageRecord.burst_id == burst_id,
                M.ImageRecord.infer_status == "inferred")).all()
            if members and all(m.has_blinks for m in members):
                return {"ok": True, "moved": 0, "ids": [],
                        "note": "Every frame has a blink, so the best one is kept."}
            repo = PhotoRepository(s)
            ids = []
            for m in members:
                if m.is_best_shot or m.locked or not m.has_blinks:
                    continue  # the best shot, locked photos and non-blinking frames stay put
                if m.manual_bucket == "junk_blinks":
                    continue
                repo.set_manual_bucket(m.id, "junk_blinks")
                ids.append(m.id)
            return {"ok": True, "moved": len(ids), "ids": ids}

    @app.post("/api/bursts/{burst_id}/restore-blinks", dependencies=auth)
    def restore_blinks(burst_id: int):
        with get_session(dbp) as s:
            if s.get(M.BurstGroup, burst_id) is None:
                raise HTTPException(404, "burst not found")
            repo = PhotoRepository(s)
            ids = []
            for m in s.exec(select(M.ImageRecord).where(M.ImageRecord.burst_id == burst_id,
                                                        M.ImageRecord.manual_bucket == "junk_blinks")).all():
                repo.set_manual_bucket(m.id, None)
                ids.append(m.id)
            return {"ok": True, "restored": len(ids), "ids": ids}

    app.state.pii = {}  # image id -> Finding from the last sensitive-detail scan

    @app.post("/api/export/clean/scan", dependencies=auth, status_code=202)
    def clean_scan(body: CleanScanReq):
        """Look for plates, cards, screens, house numbers and ID documents in the selection."""
        from ..clean_export import PiiScanner, PiiUnavailable
        from ..quality.image_loader import load_image
        c = run_cfg(None)
        if not _BUSY.acquire(blocking=False):
            raise HTTPException(409, "another task is already running")

        def work(job: dict) -> dict:
            with get_session(dbp) as s:
                recs = [(r.id, r.filename, r.path, r.target_path, r.orientation)
                        for r in _by_ids(s, body.ids)]
            try:
                scanner = PiiScanner(c, backend=detect_for(c))
            except PiiUnavailable as e:
                return {"ok": False, "unavailable": str(e), "scanned": 0, "flagged": []}
            progress_mgr.start("Privacy scan", "Looking for sensitive details", len(recs), "Scanning...")
            flagged = []
            try:
                for n, (iid, name, path, target, orient) in enumerate(recs, 1):
                    if progress_mgr.is_stopped():
                        break
                    app.state.pii[iid] = False  # scanned, nothing found, unless a finding replaces it
                    src = next((x for x in (path, target) if x and os.path.exists(x)), None)
                    if src is not None:
                        try:
                            bgr = load_image(src, max_side=1600, orientation=orient).bgr
                            f = scanner.scan_image(iid, bgr)
                        except Exception as e:
                            jlog(job, f"could not scan {name}: {e}", "warn")
                            f = None
                        if f is not None:
                            app.state.pii[iid] = f
                            flagged.append({"id": iid, "filename": name, "labels": f.labels,
                                            "score": round(f.score, 3), "localized": f.localized})
                    progress_mgr.update(n, len(recs), name, detail=f"{n} / {len(recs)}")
            finally:
                scanner.release()
            progress_mgr.finish(f"Scan complete: {len(flagged)} photos flagged")
            return {"ok": True, "unavailable": "", "scanned": len(recs), "flagged": flagged}

        return {"ok": True, "job_id": spawn("pii-scan", work)["id"]}

    @app.post("/api/export/clean", dependencies=auth, status_code=202)
    def clean_export(body: CleanExportReq):
        """Write metadata-free copies to a new folder, optionally blurring or leaving out flagged photos."""
        from ..clean_export import ExportItem, export_bundle
        root = validate_output_dir(body.output_dir, dbp, False)
        findings = {i: app.state.pii[i] for i in body.ids if app.state.pii.get(i)}
        if body.policy != "include" and any(i not in app.state.pii for i in body.ids):
            raise HTTPException(409, "scan these photos for sensitive details first, "
                                     "or choose to export without that check")
        if not _BUSY.acquire(blocking=False):
            raise HTTPException(409, "another task is already running")

        def work(job: dict) -> dict:
            with get_session(dbp) as s:
                items = [ExportItem(
                    r.id, next((x for x in (r.path, r.target_path) if x and os.path.exists(x)), r.path),
                    r.orientation, r.extension, r.taken_at.isoformat() if r.taken_at else None)
                    for r in _by_ids(s, body.ids)]
            progress_mgr.start("Clean export", "Writing clean copies", len(items), "Exporting...")

            def on_prog(done, total, name):
                progress_mgr.update(done, total, name, detail=f"{done} / {total}")

            res = export_bundle(items, root, findings=findings, policy=body.policy, html=body.html,
                                rename=body.rename, on_progress=on_prog,
                                cancel_check=progress_mgr.is_stopped)
            progress_mgr.finish(f"Exported {res.exported} clean photos")
            jlog(job, f"exported {res.exported} photos to {res.folder}")
            return {"ok": not res.failures, "folder": str(res.folder), "exported": res.exported,
                    "skipped_flagged": res.skipped_flagged, "blurred": res.blurred,
                    "flagged": res.flagged, "failures": res.failures[:20]}

        return {"ok": True, "job_id": spawn("clean-export", work)["id"]}

    @app.post("/api/bursts/{burst_id}/composite", dependencies=auth)
    def composite_burst(burst_id: int, body: CompositeReq = CompositeReq()):
        """Swap open eyes from the partner frame into the best frame, saved next to the original."""
        from ..burst_actions import ActionError, composite_burst as run
        with exclusive():
            try:
                return run(dbp, burst_id, body.mode)
            except ActionError as e:
                raise HTTPException(e.status, str(e)) from None

    @app.post("/api/bursts/{burst_id}/best-shot", dependencies=auth)
    def set_best_shot(burst_id: int, body: BestShotReq):
        """Pick which frame in a burst is the best shot. The rest become alternates."""
        with get_session(dbp) as s:
            bg = s.get(M.BurstGroup, burst_id)
            if bg is None:
                raise HTTPException(404, "burst not found")
            members = s.exec(select(M.ImageRecord).where(
                M.ImageRecord.burst_id == burst_id)).all()
            if not any(m.id == body.image_id for m in members):
                raise HTTPException(400, "image is not part of this burst")
            for m in members:
                m.is_best_shot = (m.id == body.image_id)
                m.is_alternate = (m.id != body.image_id)
                if m.id == body.image_id:
                    m.delete_approved = False  # choosing a frame to keep clears its delete mark
            bg.best_shot_id = body.image_id
            s.flush()
            return {"ok": True, "burst_id": burst_id, "best_shot_id": body.image_id}

    @app.get("/api/overview")
    def overview():
        """Library composition by effective category, for the composition strip."""
        eff = func.coalesce(M.ImageRecord.manual_bucket, M.ImageRecord.category)
        active = M.ImageRecord.scan_status != "missing"
        with get_session(dbp) as s:
            rows = s.exec(select(eff, func.count()).where(active).group_by(eff)).all()
            by_cat = {(k or "unrouted"): int(n) for k, n in rows}
            stats_ = PhotoRepository(s).stats()
            manual = s.exec(select(func.count()).select_from(M.ImageRecord).where(
                active, M.ImageRecord.manual_bucket.is_not(None))).one()
        return {"by_category": by_cat, "total": sum(by_cat.values()), "stats": stats_,
                "manual_count": int(manual or 0)}

    @app.get("/api/browse", dependencies=auth)
    def browse(path: str | None = Query(None), hidden: bool = Query(False)):
        """List sub-folders of ``path`` so the UI can offer a folder picker."""
        roots = _browse_roots()
        if not roots:
            raise HTTPException(404, "no browsable folders are configured")
        target = Path(path).expanduser() if path else roots[0]
        if not target.is_absolute():
            raise HTTPException(400, "path must be absolute")
        if _confined() and not _within(target, roots):
            raise HTTPException(403, "that folder is outside the allowed locations")
        if not target.is_dir():
            raise HTTPException(404, "folder not found")
        target = target.resolve()
        exts = {e.lower() for e in cfg.extensions}
        entries, photos, error, truncated = [], 0, None, False
        try:
            with os.scandir(target) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False) or (e.is_symlink() and e.is_dir()):
                            if e.name.startswith(".") and not hidden:
                                continue
                            if len(entries) >= 2000:
                                truncated = True
                                continue
                            entries.append({"name": e.name, "path": str(Path(e.path))})
                        elif e.is_file() and Path(e.name).suffix.lower() in exts:
                            photos += 1
                    except OSError:
                        continue
        except PermissionError:
            error = "You do not have permission to open this folder."
        except OSError as ex:
            error = str(ex)
        entries.sort(key=lambda d: d["name"].lower())
        parent = None
        if target.parent != target and (not _confined() or _within(target.parent, roots)):
            parent = str(target.parent)
        return {
            "path": str(target), "parent": parent, "entries": entries, "photos": photos,
            "writable": os.access(target, os.W_OK), "error": error, "truncated": truncated,
            "roots": [{"name": "Home" if r == Path.home()
                      else ("Computer" if str(r) == "/" else (r.name or str(r))),
                       "path": str(r)} for r in roots],
        }

    @app.post("/api/browse/mkdir", dependencies=auth)
    def browse_mkdir(body: MkdirReq):
        name = body.name.strip()
        if (not name or name in (".", "..") or any(c in name for c in '/\\:*?"<>|')
                or len(name) > 200):
            raise HTTPException(400, "folder name contains characters that are not allowed")
        parent = Path(body.parent).expanduser()
        roots = _browse_roots()
        if not parent.is_absolute() or not parent.is_dir():
            raise HTTPException(404, "parent folder not found")
        if _confined() and not _within(parent, roots):
            raise HTTPException(403, "that folder is outside the allowed locations")
        new = parent.resolve() / name
        try:
            new.mkdir(exist_ok=False)
        except FileExistsError:
            raise HTTPException(409, "a folder with that name already exists")
        except OSError as ex:
            raise HTTPException(403, f"could not create folder: {ex}")
        return {"path": str(new)}

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
            bg = s.get(M.BurstGroup, burst_id)
            if bg is None:
                raise HTTPException(404, "burst not found")
            members = s.exec(select(M.ImageRecord).where(
                M.ImageRecord.burst_id == burst_id,
                M.ImageRecord.scan_status != "missing",
                M.ImageRecord.infer_status == "inferred").options(
                selectinload(M.ImageRecord.quality),
                selectinload(M.ImageRecord.faces))).all()
            items = []
            for r in members:
                qm = r.quality
                faces = r.faces or []
                best_face = max(faces, key=lambda f: f.confidence, default=None)
                items.append({
                    "id": r.id, "filename": r.filename, "is_best_shot": r.is_best_shot,
                    "is_alternate": r.is_alternate,
                    "taken_at": r.taken_at.isoformat() if r.taken_at else None,
                    "width": r.width, "height": r.height,
                    "orientation": r.orientation,
                    "sharpness_norm": round(qm.sharpness_norm, 3) if qm else None,
                    "face_count": len(faces),
                    "face_confidence": round(best_face.confidence, 3) if best_face else None,
                    "has_blinks": bool(r.has_blinks), "blink_count": int(r.blink_count or 0),
                    "eye_closed": (round(best_face.eye_closed_score, 3)
                                   if best_face and best_face.eye_closed_score is not None else None),
                    "manual_bucket": r.manual_bucket,
                    "delete_approved": bool(r.delete_approved),
                })
            blinkers = [m for m in members if m.has_blinks]
            movable = [m for m in blinkers if not m.is_best_shot and not m.locked
                       and m.manual_bucket != "junk_blinks"]
            # Offered only when some frames blink and some do not: if every frame blinks the
            # best one is still kept and there is nothing sensible to reject.
            can_reject = bool(movable) and len(blinkers) < len(members)
            return {"burst": bg.name, "members": items, "can_reject_blinks": can_reject,
                    "reject_count": len(movable) if can_reject else 0,
                    "rejected_blinks": sum(1 for m in members if m.manual_bucket == "junk_blinks"),
                    "group": group_matrix_payload(members),
                    "face_matrix": burst_face_matrix(members)}

    @app.get("/api/events")
    def events():
        with get_session(dbp) as s:
            covers = dict(s.exec(select(M.ImageRecord.event_id, func.min(M.ImageRecord.id))
                                 .where(M.ImageRecord.scan_status != "missing",
                                        M.ImageRecord.event_id.is_not(None))
                                 .group_by(M.ImageRecord.event_id)).all())
            rows = s.exec(select(M.EventGroup).order_by(M.EventGroup.start)).all()
            return {"count": len(rows), "items": [{
                "id": e.id, "label": e.label, "image_count": e.image_count,
                "start": e.start.isoformat() if e.start else None,
                "end": e.end.isoformat() if e.end else None,
                "cover_id": covers.get(e.id),
            } for e in rows]}

    @app.get("/api/transactions")
    def transactions():
        with get_session(dbp) as s:
            rows = s.exec(select(M.ActionLedger.transaction_id, M.ActionLedger.status,
                                 func.count(), func.max(M.ActionLedger.id),
                                 func.min(M.ActionLedger.created_at))
                          .group_by(M.ActionLedger.transaction_id,
                                    M.ActionLedger.status)).all()
        txns: dict = {}
        for tid, status, n, max_id, created in rows:
            d = txns.setdefault(tid, {
                "transaction_id": tid, "count": 0, "committed": 0, "rolled_back": 0,
                "failed": 0, "pending": 0, "max_id": 0, "created_at": "",
            })
            d["count"] += n
            d[status] = d.get(status, 0) + n
            d["max_id"] = max(d["max_id"], max_id or 0)
            if created and (not d["created_at"] or created.isoformat() < d["created_at"]):
                d["created_at"] = created.isoformat()
        items = sorted(txns.values(), key=lambda d: d["max_id"], reverse=True)
        for d in items:
            d["undoable"] = d["committed"] > 0
            d.pop("max_id")
        return {"items": items, "latest_undoable": latest_undoable()}

    @app.get("/api/transactions/{tid}/verify")
    def verify_txn(tid: str):
        planner = make_planner(req_cfg())
        if not hasattr(planner, "verify_transaction"):
            raise _not_impl("transaction verification")
        if tid == "latest":
            tid = latest_undoable() or ""
            if not tid:
                raise HTTPException(404, "no undoable transactions")
        return {"transaction_id": tid, "report": planner.verify_transaction(tid)}

    def orphaned_rows() -> list[dict]:
        with get_session(dbp) as s:
            rows = s.exec(select(M.ActionLedger.transaction_id, func.count())
                          .where(M.ActionLedger.status == "pending")
                          .group_by(M.ActionLedger.transaction_id)).all()
        return [{"transaction_id": t, "pending_actions": int(n)} for t, n in rows]

    @app.get("/api/transactions/orphaned")
    def get_orphaned_transactions():
        """Transactions left PENDING by a crash, power loss or kill mid-apply."""
        tx = orphaned_rows()
        return {"count": len(tx), "transactions": tx}

    @app.post("/api/transactions/{tid}/resume", dependencies=auth)
    def resume_transaction(tid: str):
        """Settle PENDING rows by inspecting the disk: finished work is committed,
        partial targets are removed and marked failed (sources are never touched)."""
        from ..engine.rollback import RollbackEngine, RollbackReport
        if not _BUSY.acquire(blocking=False):
            raise HTTPException(409, "another task is already running")
        try:
            rep = RollbackReport(transaction_id=tid)
            with _LEDGER_WRITING, get_session(dbp) as s:
                RollbackEngine(req_cfg(), dbp)._reconcile_pending(tid, PhotoRepository(s), rep)
            return {"ok": True, "transaction_id": tid, "notes": list(rep.notes)}
        finally:
            _BUSY.release()

    @app.post("/api/transactions/{tid}/rollback", dependencies=auth)
    def rollback_orphaned(tid: str):
        from ..engine.rollback import RollbackEngine
        if not _BUSY.acquire(blocking=False):
            raise HTTPException(409, "another task is already running")
        try:
            rep = RollbackEngine(req_cfg(), dbp).rollback_transaction(tid)
            return {"ok": rep.failures == 0, "transaction_id": tid, "reversed": rep.reversed,
                    "failures": rep.failures, "errors": list(rep.errors)[:50]}
        finally:
            _BUSY.release()

    def cleanup():
        app.state.watch.stop()
        app.state.sidecars.close()  # write anything still waiting out its debounce
        _pid_file(cfg.port).unlink(missing_ok=True)

    @app.post("/api/restart", dependencies=auth)
    def restart_server(force: bool = Query(False)):
        if _LEDGER_WRITING.locked():
            return JSONResponse({"ok": False, "error": "A filesystem transaction is actively writing. "
                                 "Server restart cannot be forced during plan execution or undo."},
                                status_code=409)
        if _BUSY.locked() and not force:
            return JSONResponse({"ok": False, "error": "A background task is running. "
                                 "Restart anyway with force=true."}, status_code=409)

        def restart_daemon():
            import sys
            time.sleep(0.25)
            if _BUSY.locked():
                progress_mgr.request_stop()  # let a forced restart wind the job down first
                deadline = time.monotonic() + 15
                while _BUSY.locked() and time.monotonic() < deadline:
                    time.sleep(0.2)
            if _LEDGER_WRITING.locked():
                return
            cleanup()
            os.execv(sys.executable, [sys.executable, *_restart_args(sys.orig_argv)])
        threading.Thread(target=restart_daemon, daemon=True).start()
        return {"ok": True, "message": "Server restarting..."}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str, since: int = Query(0, ge=0)):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        log = job["log"]
        return {"id": job["id"], "kind": job["kind"], "status": job["status"],
                "result": job["result"], "error": job["error"],
                "log": log[since:], "next": len(log)}

    # ----------------------------------------------------------------- write api
    def do_scan(c: CuratorConfig, body: ScanReq, finish: bool = True) -> dict:
        target = Path(body.path).expanduser().resolve()
        if not target.exists() or not target.is_dir():
            raise HTTPException(400, f"Directory does not exist: {body.path}")
        if _confined() and not _within(target, _browse_roots()):
            raise HTTPException(403, "that folder is outside the allowed locations")
        progress_mgr.start("Scanning", f"Scanning directory: {target.name}", 0,
                           "Discovering image files...")
        from ..scanner.file_scanner import FileScanner
        scanner = FileScanner(c)

        def on_prog(done, total, curr):
            progress_mgr.update(done, total, f"Indexing: {Path(curr).name}",
                                detail=f"{done} / {total}")

        try:
            res = scanner.scan_directory(target, db_path=dbp, on_progress=on_prog,
                                         show_progress=False,
                                         cancel_check=progress_mgr.is_stopped)
        except InterruptedError:
            progress_mgr.stop("Scan stopped by user")
            return {"ok": False, "stopped": True, "message": "Scan stopped by user"}
        except Exception as e:
            progress_mgr.error(str(e))
            raise
        if finish:
            progress_mgr.finish(f"Scan complete: {res.scanned} files seen, {res.new} new "
                                f"indexed in {res.duration_s:.1f}s")
        return {"ok": True, "path": str(target), "scanned": res.scanned, "new": res.new,
                "duplicates": res.duplicates, "skipped_known": res.skipped_known,
                "errors": res.errors, "duration_s": round(res.duration_s, 2)}

    def do_infer(c: CuratorConfig, body: InferReq, finish: bool = True) -> dict:
        progress_mgr.start("AI Inference", "Running quality + faces + CLIP semantics...",
                           0, "Initializing AI models...")
        from ..inference.pipeline import InferencePipeline
        if body.batch_size:
            c.device.face_batch_size = body.batch_size
            c.device.clip_batch_size = max(body.batch_size, 1)

        def on_prog(done, total, curr):
            fn = Path(curr).name if not curr.startswith("Phase") else curr
            progress_mgr.update(done, total, fn, detail=infer_progress_detail(done, total))

        try:
            pipeline = InferencePipeline(c, backend=detect_for(c))
            if body.reanalyze:
                with get_session(dbp) as s:
                    I = M.ImageRecord
                    upd = {"faces_schema_version": 0}
                    if pipeline.bs.face_backend == "stub" or not c.face.enabled:
                        upd["infer_status"] = "pending"  # the stale-face path is off, so redo from scratch
                    s.exec(sa_update(I).where(I.infer_status == "inferred").values(**upd))
            res = pipeline.run(db_path=dbp, on_progress=on_prog, show_progress=False,
                               cancel_check=progress_mgr.is_stopped)
        except InterruptedError:
            progress_mgr.stop("Inference stopped by user")
            return {"ok": False, "stopped": True, "message": "Inference stopped by user"}
        except Exception as e:
            progress_mgr.error(str(e))
            raise
        app.state.last_analyzed_ids = set(getattr(res, "processed_ids", ()))
        if finish:
            progress_mgr.finish(f"Inference complete: {res.processed} processed, "
                                f"{res.faces_total} faces in {res.duration_s:.1f}s")
        return {"ok": True, "processed": res.processed, "faces_total": res.faces_total,
                "new_ids_count": len(app.state.last_analyzed_ids),
                "quality_computed": res.quality_computed,
                "semantic_computed": res.semantic_computed, "failures": res.failures,
                "duration_s": round(res.duration_s, 2)}

    def do_cluster(c: CuratorConfig, body: ClusterReq, finish: bool = True) -> dict:
        from ..clustering.best_shot_selector import BestShotSelector, _candidate_from_record
        from ..clustering.burst_detector import BurstDetector
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        c.burst.window_seconds = body.window_seconds
        c.burst.similarity_threshold = body.similarity_threshold
        progress_mgr.start("Clustering", "Burst detection and best-shot selection...", 1,
                           "Analyzing timestamps and visual similarity...")
        try:
            with get_session(dbp) as s:
                repo = PhotoRepository(s)
                new_ids = getattr(app.state, "last_analyzed_ids", None)
                if body.only_new and new_ids:
                    records, bursts = _burst_neighbourhood(
                        s, new_ids, float(c.burst.window_seconds))
                else:
                    records, bursts = repo.all_images(), repo.all_bursts()
                old_best = {b.best_shot_id for b in bursts if b.best_shot_id}
                for r in records:
                    r.burst_id = None
                    r.is_best_shot = False
                    r.is_alternate = False
                for b in bursts:
                    repo.s.delete(b)
                s.flush()
                active = [r for r in records if r.scan_status != "missing"]
                results = BurstDetector(c).cluster_records(active)
                by_id = {r.id: r for r in active}
                sel = BestShotSelector(c)
                n = seq = 0
                for res in results:
                    if progress_mgr.is_stopped():  # the session rolls back, keeping the old groups
                        raise InterruptedError("Clustering stopped by user")
                    members = [by_id[m] for m in res.image_ids if m in by_id]
                    if len(members) < c.burst.min_group_size:
                        continue
                    kept = [m.id for m in members if m.id in old_best]
                    best_id = kept[0] if len(kept) == 1 else sel.pick_best_shot(
                        [_candidate_from_record(m) for m in members]).image_id
                    earliest = min(members, key=lambda m: (m.taken_at is None,
                                                           m.taken_at or epoch))
                    stamp = (earliest.taken_at.strftime("%Y%m%d")
                             if earliest.taken_at else "Unknown")
                    while s.exec(select(M.BurstGroup.id).where(  # kept groups own their names
                            M.BurstGroup.name == f"Burst_{stamp}_{seq:04d}")).first() is not None:
                        seq += 1
                    bg = repo.upsert_burst(M.BurstGroup(
                        name=f"Burst_{stamp}_{seq:04d}", image_count=len(members),
                        best_shot_id=best_id, start_time=earliest.taken_at,
                        end_time=max((m.taken_at for m in members), default=earliest.taken_at),
                    ))
                    for m in members:
                        m.burst_id = bg.id
                        m.is_best_shot = (m.id == best_id)
                        m.is_alternate = (m.id != best_id)
                    n += 1
                    seq += 1
        except InterruptedError:
            progress_mgr.stop("Clustering stopped by user")
            return {"ok": False, "stopped": True, "message": "Clustering stopped by user"}
        except Exception as e:
            progress_mgr.error(str(e))
            raise
        if finish:
            progress_mgr.finish(f"Clustering complete: {n} burst groups detected")
        return {"ok": True, "burst_groups": n}

    @app.post("/api/scan", dependencies=auth)
    def scan_folder(body: ScanReq):
        with exclusive():
            return do_scan(run_cfg(body), body)

    @app.post("/api/infer", dependencies=auth)
    def infer_pending(body: InferReq = InferReq()):
        with exclusive():
            return do_infer(run_cfg(body), body)

    @app.post("/api/cluster", dependencies=auth)
    def cluster_bursts(body: ClusterReq = ClusterReq()):
        with exclusive():
            return do_cluster(req_cfg(), body)

    @app.post("/api/ingest", dependencies=auth)
    def ingest_folder(body: ScanReq):
        with exclusive():
            progress_mgr.start("Ingest Pipeline", "Starting full ingest (Scan + Infer + Cluster)",
                               0, "Initializing...")
            try:
                c = run_cfg(body)
                scan_res = do_scan(c, body, finish=False)
                if progress_mgr.is_stopped() or (isinstance(scan_res, dict) and scan_res.get("stopped")):
                    progress_mgr.stop("Ingest stopped by user")
                    return {"ok": False, "stopped": True, "message": "Ingest stopped by user"}
                infer_res = do_infer(c, InferReq(), finish=False)
                if progress_mgr.is_stopped() or (isinstance(infer_res, dict) and infer_res.get("stopped")):
                    progress_mgr.stop("Ingest stopped by user")
                    return {"ok": False, "stopped": True, "message": "Ingest stopped by user"}
                cluster_res = do_cluster(c, ClusterReq(only_new=bool(infer_res.get("new_ids_count"))),
                                         finish=False)
                if progress_mgr.is_stopped() or (isinstance(cluster_res, dict) and cluster_res.get("stopped")):
                    progress_mgr.stop("Ingest stopped by user")
                    return {"ok": False, "stopped": True, "message": "Ingest stopped by user"}
            except InterruptedError:
                progress_mgr.stop("Ingest stopped by user")
                return {"ok": False, "stopped": True, "message": "Ingest stopped by user"}
            except Exception as e:
                progress_mgr.error(str(e))
                raise
            progress_mgr.finish("Full ingest pipeline complete!")
            return {"ok": True, "scan": scan_res, "infer": infer_res, "cluster": cluster_res}

    @app.post("/api/stop", dependencies=auth)
    def stop_current_task():
        progress_mgr.request_stop()
        for j in list(jobs.values()):
            if j.get("status") == "running":
                j["status"] = "stopping"
                jlog(j, "stop requested by user", "warn")
        return {"ok": True, "stopping": True, "message": "Stop requested"}

    @app.post("/api/reset", dependencies=auth)
    def reset_library():
        with exclusive():
            with get_session(dbp) as s:
                repo = PhotoRepository(s)
                counts = repo.clear_library(keep_ledger=False)
            thumb_dir = cfg.cache_dir / "thumbs"
            if thumb_dir.exists():
                stale = [*thumb_dir.glob(f"*/{db_fp}_*.webp"), *thumb_dir.glob(f"{db_fp}_*.jpg")]
                for th in stale:
                    try:
                        th.unlink(missing_ok=True)
                    except OSError:
                        pass
            progress_mgr.reset()
            return {"ok": True, "deleted": counts}

    @app.post("/api/images/{image_id}/reassign", dependencies=auth)
    def reassign(image_id: int, body: ReassignReq):
        bucket = _check_category(body.bucket, "bucket")
        category = _check_category(body.category, "category")
        effective = bucket or category
        with get_session(dbp) as s:
            rec = s.get(M.ImageRecord, image_id)
            if rec is None:
                raise HTTPException(404, "image not found")
            PhotoRepository(s).set_manual_bucket(image_id, effective)
            return {"ok": True, "id": image_id, "manual_bucket": rec.manual_bucket,
                    "category": rec.category, "locked": bool(rec.locked)}

    @app.post("/api/images/{image_id}/lock", dependencies=auth)
    def lock_image(image_id: int, body: LockReq = LockReq()):
        with get_session(dbp) as s:
            rec = s.get(M.ImageRecord, image_id)
            if rec is None:
                raise HTTPException(404, "image not found")
            repo = PhotoRepository(s)
            if hasattr(repo, "set_locked"):
                repo.set_locked(image_id, body.locked)
            else:
                rec.locked = body.locked
            return {"ok": True, "id": image_id, "locked": bool(rec.locked)}

    @app.post("/api/plan/rebuild", dependencies=auth)
    def rebuild_plan(body: PlanReq):
        _check_category(body.only_category, "only_category")
        root = validate_output_dir(body.output_dir, dbp, body.mode == "in_place")
        with exclusive():
            p = build(make_planner(req_cfg()), body, root, persist=True)
            return {"plan_id": p.plan_id, "total": p.total, "by_category": p.by_category,
                    "burst_count": p.burst_count, "organize_root": p.organize_root}

    @app.post("/api/plan/diff", dependencies=auth)
    def plan_diff(body: PlanReq):
        only = _check_category(body.only_category, "only_category")
        root = validate_output_dir(body.output_dir, dbp, body.mode == "in_place")
        with exclusive():
            p = build(make_planner(req_cfg()), body, root, persist=False)
        actions = [a for a in p.actions if not only or str(
            getattr(a.category, "value", a.category)) == only]
        targets_seen: set[str] = set()
        collisions, missing, cross, need_bytes, total_bytes = [], 0, 0, 0, 0
        for a in actions:
            src, dst = Path(a.source), Path(a.target)
            try:
                st = src.stat()
            except OSError:
                missing += 1
                continue
            total_bytes += st.st_size
            key = str(dst)
            same_file = False
            if dst.exists():
                try:
                    same_file = os.path.samefile(src, dst)
                except OSError:
                    same_file = False
            if (dst.exists() and not same_file) or key in targets_seen:
                collisions.append(key)
            targets_seen.add(key)
            anc = _nearest_existing(dst.parent)
            crossdev = False
            try:
                crossdev = anc.stat().st_dev != st.st_dev
            except OSError:
                pass
            if a.operation in ("move", "hardlink") and crossdev:
                cross += 1
            if a.operation == "copy" or (a.operation == "move" and crossdev):
                need_bytes += st.st_size
        free = shutil.disk_usage(_nearest_existing(root)).free
        ids = [a.image_id for a in actions if a.image_id]
        marked = 0
        with get_session(dbp) as s:
            for i in range(0, len(ids), 500):
                marked += int(s.exec(select(func.count()).select_from(M.ImageRecord).where(
                    M.ImageRecord.id.in_(ids[i:i + 500]),
                    M.ImageRecord.delete_approved == True)).one())  # noqa: E712
        by_cat: dict[str, int] = {}
        for a in actions:
            k = str(getattr(a.category, "value", a.category))
            by_cat[k] = by_cat.get(k, 0) + 1
        return {
            "plan_id": p.plan_id, "total": len(actions), "by_category": by_cat,
            "operation": body.operation, "mode": body.mode, "organize_root": str(root),
            "collisions": len(collisions), "collision_samples": collisions[:20],
            "cross_device": cross, "missing_sources": missing,
            "bytes_total": total_bytes, "bytes_to_write": need_bytes,
            "free_bytes": free, "enough_space": need_bytes <= free, "marked_for_delete": marked,
        }

    @app.post("/api/plan/apply", dependencies=auth, status_code=202)
    def apply_plan(body: PlanReq):
        only = _check_category(body.only_category, "only_category")
        root = validate_output_dir(body.output_dir, dbp, body.mode == "in_place")
        planner = make_planner(run_cfg())
        if body.trash and not _accepts(planner.execute_plan, "trash"):
            raise _not_impl("trash staging")
        if not _BUSY.acquire(blocking=False):
            raise HTTPException(409, "another task is already running")

        def work(job: dict) -> dict:
            # Pre-execution backup (C1 §2.5). A failed backup warns but does not block the apply.
            backup_warning = ""
            if cfg.auto_db_backup:
                try:
                    snapshot_database(dbp, keep=10)
                    jlog(job, "database snapshot created")
                except Exception as e:
                    backup_warning = f"Could not back up the database first: {e}"
                    jlog(job, backup_warning, "warn")
            progress_mgr.start("Apply", "Building plan", 0, "Building plan...")
            jlog(job, "building plan")
            p = build(planner, body, root, persist=True)
            kw: dict = {}
            if only:
                if _accepts(planner.execute_plan, "only_category"):
                    kw["only_category"] = only
                else:
                    p.actions = [a for a in p.actions if str(
                        getattr(a.category, "value", a.category)) == only]
                    p.total = len(p.actions)
            if body.trash:
                kw["trash"] = True
            total = sum(1 for a in p.actions if not only or str(
                getattr(a.category, "value", a.category)) == only)
            if total == 0:
                progress_mgr.finish("Nothing to apply")
                jlog(job, "nothing to apply")
                return {"ok": False, "error": "nothing to apply"}
            with get_session(dbp) as s:
                after = s.exec(select(func.max(M.ActionLedger.id))).one() or 0
            jlog(job, f"applying {total} actions ({body.operation}, plan {p.plan_id})")
            progress_mgr.start("Apply", f"Applying {total} actions", total, "Applying...")
            stop = threading.Event()
            watcher = watch_ledger(job, stop, total, after_id=after)
            kw["cancel_check"] = progress_mgr.is_stopped
            try:
                rep = planner.execute_plan(p, **kw)
            finally:
                stop.set()
                watcher.join(timeout=5)
            ok = rep.failures == 0
            if progress_mgr.is_stopped():
                progress_mgr.stop(f"Applied {rep.executed} files (stopped by user)")
                jlog(job, f"stopped by user: {rep.executed} executed", "warn")
            else:
                progress_mgr.finish(f"Applied {rep.executed} files, {rep.failures} failed")
                jlog(job, f"done: {rep.executed} executed, {rep.failures} failed "
                          f"(txn {rep.transaction_id})", "" if ok else "err")
            return {"ok": ok, "transaction_id": rep.transaction_id,
                    "executed": rep.executed, "collisions": rep.collisions,
                    "backup_warning": backup_warning,
                    "skipped_collisions": getattr(rep, "skipped_collisions", 0),
                    "unapproved_junk": getattr(rep, "unapproved_junk", 0),
                    "marked_left_out": getattr(rep, "marked_left_out", 0),
                    "copy_fallbacks": getattr(rep, "copy_fallbacks", 0),
                    "failures": rep.failures, "errors": list(rep.errors)[:50]}

        job = spawn("apply", work)
        return {"ok": True, "job_id": job["id"]}

    @app.post("/api/undo", dependencies=auth, status_code=202)
    def undo(body: UndoReq):
        tid = body.transaction_id or latest_undoable()
        if not tid:
            return JSONResponse({"ok": False, "error": "no transactions found"},
                                status_code=200)
        planner = make_planner(req_cfg())
        if not body.verify and not _accepts(planner.undo, "verify"):
            raise _not_impl("undo without verification")
        if not _BUSY.acquire(blocking=False):
            raise HTTPException(409, "another task is already running")

        def work(job: dict) -> dict:
            with get_session(dbp) as s:
                total = s.exec(select(func.count()).select_from(M.ActionLedger).where(
                    M.ActionLedger.transaction_id == tid,
                    M.ActionLedger.status == "committed")).one()
            progress_mgr.start("Undo", f"Undoing {total} actions", total, "Rolling back...")
            jlog(job, f"undoing transaction {tid} ({total} committed actions)")
            stop = threading.Event()
            watcher = watch_ledger(job, stop, total, tid=tid)
            undo_kw: dict = {"cancel_check": progress_mgr.is_stopped}
            if _accepts(planner.undo, "verify"):
                undo_kw["verify"] = body.verify
            try:
                rep = planner.undo(tid, **undo_kw)
            finally:
                stop.set()
                watcher.join(timeout=5)
            ok = rep.failures == 0
            if progress_mgr.is_stopped():
                progress_mgr.stop(f"Reversed {rep.reversed} actions (stopped by user)")
                jlog(job, f"stopped by user: {rep.reversed} reversed", "warn")
            else:
                progress_mgr.finish(f"Reversed {rep.reversed} actions, {rep.failures} failed")
                jlog(job, f"done: {rep.reversed} reversed, {rep.skipped} skipped, "
                          f"{rep.failures} failed", "" if ok else "err")
            return {"ok": ok, "transaction_id": tid, "reversed": rep.reversed,
                    "skipped": rep.skipped, "failures": rep.failures,
                    "errors": list(rep.errors)[:50]}

        job = spawn("undo", work)
        return {"ok": True, "job_id": job["id"], "transaction_id": tid}

    @app.post("/api/rescore", dependencies=auth)
    def rescore():
        """Re-classify every analyzed photo from its stored embedding, with no image decoding."""
        if backend_info().get("semantic_is_stub", True):
            return JSONResponse({"updated": 0, "note": "Re-checking needs the AI model, which is "
                                                       "not installed. Run: photocure setup"})
        if not _BUSY.acquire(blocking=False):
            raise HTTPException(409, "another task is already running")

        def work(job: dict) -> dict:
            from ..inference.semantic_engine import rescore_from_embeddings
            progress_mgr.start("Re-checking", "Re-checking photos with the latest categories", 0,
                               "Reading stored embeddings...")
            with get_session(dbp) as s:
                updated = int(rescore_from_embeddings(s, run_cfg()))
            progress_mgr.finish(f"Re-checked {updated} photos")
            jlog(job, f"re-checked {updated} photos")
            return {"ok": True, "updated": updated}

        job = spawn("rescore", work, cancellable=False)
        return JSONResponse({"ok": True, "job_id": job["id"]}, status_code=202)

    @app.post("/api/learn", dependencies=auth, status_code=202)
    def learn():
        """Re-sort uncertain photos using the user's manual choices (background job)."""
        planner = make_planner(req_cfg())
        if not hasattr(planner, "learn_from_corrections"):
            raise _not_impl("learning from corrections")
        if not _BUSY.acquire(blocking=False):
            raise HTTPException(409, "another task is already running")

        def work(job: dict) -> dict:
            progress_mgr.start("Learning", "Re-sorting from your choices", 0,
                               "Reading your corrections...")
            jlog(job, "learning from your choices")
            res = dict(planner.learn_from_corrections())
            moved = int(res.get("moved_out_of_uncertain", 0))
            progress_mgr.finish(f"Re-sorted {int(res.get('updated', 0))} photos, "
                                f"{moved} left Needs a look")
            jlog(job, f"updated {int(res.get('updated', 0))}, moved {moved} out of uncertain")
            return {"ok": True, **res}

        job = spawn("learn", work, cancellable=False)
        return {"ok": True, "job_id": job["id"]}

    @app.post("/api/repair", dependencies=auth)
    def repair():
        planner = make_planner(req_cfg())
        if not hasattr(planner, "repair_paths"):
            raise _not_impl("path repair")
        with exclusive():
            return {"ok": True, "repaired": int(planner.repair_paths())}

    @app.post("/api/events/cluster", dependencies=auth)
    def cluster_events_ep(body: EventClusterReq = EventClusterReq()):
        try:
            from ..clustering.events import cluster_events
        except ImportError:
            raise _not_impl("event clustering")
        with exclusive(), get_session(dbp) as s:
            return {"ok": True, "events": int(cluster_events(s, body.gap_hours, body.gps_km))}

    @app.post("/api/events/{event_id}/rename", dependencies=auth)
    def rename_event(event_id: int, body: RenameReq):
        with get_session(dbp) as s:
            ev = s.get(M.EventGroup, event_id)
            if ev is None:
                raise HTTPException(404, "event not found")
            ev.label = body.name.strip() or None
            return {"ok": True, "id": event_id, "label": ev.label}

    # ---------------------------------------------------------------- static
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.on_event("startup")
    async def startup_event():
        pid_file = _pid_file(cfg.port)
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(os.getpid()))
        try:
            with get_session(dbp) as s:
                tpat = f"%/{cfg.trash_dir_name}/%"
                for r in s.exec(select(M.ImageRecord).where(
                        M.ImageRecord.scan_status != "missing",
                        M.ImageRecord.path.like(tpat))).all():
                    r.scan_status = "missing"
                    r.route_status = "trashed"
        except Exception:
            pass
        from .. import models as bundled
        for m in bundled.load_manifest():
            if not bundled.verify_model(m["file"]):
                logging.getLogger("photocure.ui").warning(
                    "bundled model %s is missing or does not match MANIFEST.json; "
                    "the feature that uses it is switched off", m["file"])
        for t in orphaned_rows():
            logging.getLogger("photocure.ui").warning(
                "incomplete transaction %s (%d pending actions) from an unexpected shutdown",
                t["transaction_id"], t["pending_actions"])
        from ..features import load_features
        saved = load_features(dbp)
        if saved["watch_enabled"] and saved["watch_path"]:
            try:
                app.state.watch.start(saved["watch_path"])
            except FileNotFoundError as e:
                logging.getLogger("photocure.ui").warning("watched folder not resumed: %s", e)
                app.state.watch.note_failure(saved["watch_path"], str(e))

    @app.on_event("shutdown")
    async def shutdown_event():
        cleanup()
        encoder = getattr(app.state, "text_encoder", None)
        if encoder is not None:
            encoder.close()

    try:
        _stamp = max(int(p.stat().st_mtime) for p in STATIC_DIR.iterdir() if p.suffix in (".js", ".mjs", ".css"))
    except (OSError, ValueError):
        _stamp = 0
    _asset_v = f"{__version__}.{_stamp}" if _stamp else __version__

    @app.get("/", response_class=HTMLResponse)
    def index():
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        html = html.replace("?v={{VERSION}}", f"?v={_asset_v}")
        html = html.replace(TOKEN_PLACEHOLDER, "").replace("{{VERSION}}", __version__)
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    @app.get("/healthz")
    def health():
        return {"ok": True}

    return app


_app_singleton: FastAPI | None = None
_app_lock = threading.Lock()


def __getattr__(name: str):
    """Lazy module-level ``app`` so importing this module has no side effects."""
    global _app_singleton
    if name == "app":
        with _app_lock:
            if _app_singleton is None:
                _app_singleton = create_app()
            return _app_singleton
    raise AttributeError(name)


def run(host: str = "127.0.0.1", port: int = 8080, db_path: str | None = None,
        reload: bool = False) -> None:
    """Serve the dashboard (loopback by default) using the app factory."""
    import uvicorn

    if db_path:
        os.environ["PCUR_DB_PATH"] = str(db_path)
    uvicorn.run("photocure.ui.app:create_app", host=host, port=port,
                reload=reload, factory=True)
