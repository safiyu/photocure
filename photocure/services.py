"""High-level orchestration services shared by the CLI and the web UI.

Ties together the clustering, decision and organizer steps to produce a
complete, reviewable :class:`Plan`, and renders the dry-run audit report.
"""
from __future__ import annotations

import hashlib
import json
import html
import logging
import os
import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rich.console import Console
from sqlmodel import select
from rich.table import Table

from .clustering.best_shot_selector import BestShotSelector, _candidate_from_record
from .clustering.blinks import apply_blinks
from .clustering.burst_detector import BurstDetector
from .config import (
    DEVICE_CHOICES,
    MAX_WORKERS,
    CuratorConfig,
    auto_scan_workers,
    normalize_device,
    resolve_workers,
)
from .db.models import (STAGE_LIBRARY, STAGE_NEW, STAGE_TRASH, ActionLedger, BurstGroup,
                        PlanActionRow, PlanRecord)
from .db.repository import PhotoRepository
from .db.session import get_session
from .engine.action_engine import ActionEngine, ExecutionReport
from .engine.hash import file_sha256
from .engine.rollback import RollbackEngine, RollbackReport
from .rules.decision_matrix import Decision, DecisionMatrix
from .rules.organizer import FolderOrganizer, OrganizeMode, PlannedAction
from .rules.taxonomy import CategoryType

console = Console()
log = logging.getLogger("photocure.services")

_EPOCH = datetime(1970, 1, 1)


class PlanDriftError(ValueError):
    """Raised when a persisted plan no longer matches the state store."""


def describe_backends(cfg: CuratorConfig) -> dict:
    """Active device and backends, tolerant of a missing or broken torch."""
    try:
        from .inference import backends as bk
        fn = getattr(bk, "describe_backend", None)
        bs = bk.detect_backend(cfg)
        if fn is not None:
            return dict(fn(bs))
        return {
            "device": bs.device,
            "semantic_backend": bs.semantic_backend,
            "face_backend": bs.face_backend,
            "semantic_is_stub": bs.semantic_backend == "stub",
            "face_is_fallback": bs.face_backend != "yunet",
            "hint": (f'No AI model is installed. Run: {bk.ml_install_cmd()}'
                     if bs.semantic_backend == "stub" else ""),
        }
    except Exception as e:  # keep the UI and CLI alive if torch is broken
        log.warning("backend detection failed: %s", e)
        return {"device": "cpu", "semantic_backend": "stub", "face_backend": "stub",
                "semantic_is_stub": True, "face_is_fallback": True,
                "hint": f"Backend detection failed: {e}"}


# Settings store. Precedence: CLI flag > this settings file > environment > defaults.
SETTINGS_KEYS = ("device", "device_index", "decode_workers", "torch_threads", "scan_workers",
                 "collision_policy", "library_root", "default_operation")
COLLISION_POLICIES = ("sequence", "date_suffix", "skip")
OPERATIONS = ("move", "copy")


def settings_path(db_path: Path | str) -> Path:
    """The settings file lives next to the database: <stem>.settings.json."""
    p = Path(db_path)
    return p.with_name(f"{p.stem}.settings.json")


def _validate_settings(data: dict, allow_none: bool = False) -> dict:
    """Check keys and values; raises ValueError with a message fit for the user."""
    out: dict = {}
    for key, value in data.items():
        if key not in SETTINGS_KEYS:
            raise ValueError(f"Unknown setting '{key}'.")
        if value is None:
            if not allow_none:
                raise ValueError(f"'{key}' needs a value.")
            out[key] = None
        elif key == "device":
            out[key] = normalize_device(value)
        elif key == "collision_policy":
            v = str(value).strip().lower()
            if v not in COLLISION_POLICIES:
                raise ValueError(f"Unknown collision policy '{value}'. "
                                 f"Choose one of: {', '.join(COLLISION_POLICIES)}.")
            out[key] = v
        elif key == "default_operation":
            v = str(value).strip().lower()
            if v not in OPERATIONS:
                raise ValueError(f"Unknown operation '{value}'. Choose one of: {', '.join(OPERATIONS)}.")
            out[key] = v
        elif key == "library_root":
            out[key] = str(Path(value).expanduser().resolve()) if value else None
        else:
            if isinstance(value, bool):
                raise ValueError(f"'{key}' must be a whole number.")
            try:
                iv = int(value)
                if isinstance(value, float) and value != iv:
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError(f"'{key}' must be a whole number.") from None
            if not 0 <= iv <= MAX_WORKERS:
                raise ValueError(f"'{key}' must be between 0 and {MAX_WORKERS}.")
            out[key] = iv
    return out


def load_settings(db_path: Path | str) -> dict:
    """Stored settings only (keys never set are absent). Corrupt files read as empty."""
    path = settings_path(db_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    good: dict = {}
    for key, value in raw.items():
        try:
            good.update(_validate_settings({key: value}))
        except ValueError as e:
            log.warning("ignoring bad stored setting: %s", e)
    return good


def save_settings(db_path: Path | str, values: dict, merge: bool = True, force: bool = False) -> dict:
    """Validate and store settings. A value of None clears that key. Returns what is stored.

    Once a library folder is stored it only changes with ``force`` (the library migration).
    """
    checked = _validate_settings(values, allow_none=True)
    stored = load_settings(db_path)
    if ("library_root" in checked and stored.get("library_root")
            and checked["library_root"] != stored["library_root"] and not force):
        raise ValueError("library folder is already set; use Move library to change it")
    current = stored if merge else {}
    for key, value in checked.items():
        if value is None:
            current.pop(key, None)
        else:
            current[key] = value
    path = settings_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return current


def apply_settings(cfg: CuratorConfig, settings: dict) -> CuratorConfig:
    """A copy of ``cfg`` with ``settings`` applied. Raises ValueError on bad values."""
    checked = _validate_settings(settings)
    new = cfg.model_copy(deep=True)
    if "device" in checked:
        new.device.device = checked["device"]
    for key in ("device_index", "decode_workers", "torch_threads"):
        if key in checked:
            setattr(new.device, key, checked[key])
    if "scan_workers" in checked:
        new.scan_workers = checked["scan_workers"] or auto_scan_workers()
    if "collision_policy" in checked:
        new.collision_policy = checked["collision_policy"]
    if "library_root" in checked:
        new.library_root = Path(checked["library_root"]) if checked["library_root"] else None
    if "default_operation" in checked:
        new.default_operation = checked["default_operation"]
    return new


def effective_settings(db_path: Path | str, cfg: CuratorConfig | None = None,
                       resolved_device: str | None = None) -> dict:
    """What will run, where each value came from, and the resolved worker counts."""
    stored = load_settings(db_path)
    base = apply_settings(cfg or CuratorConfig.load(Path(db_path)), stored)
    values = {
        "device": base.device.device,
        "device_index": base.device.device_index,
        "decode_workers": base.device.decode_workers,
        "torch_threads": base.device.torch_threads,
        "scan_workers": 0 if stored.get("scan_workers") == 0 else base.scan_workers,
        "collision_policy": base.collision_policy,
        "library_root": str(base.library_root) if base.library_root else None,
        "default_operation": base.default_operation,
    }
    return {
        "values": values,
        "stored": stored,
        "sources": {k: ("settings" if k in stored else "environment or default")
                    for k in SETTINGS_KEYS},
        "resolved": resolve_workers(base, resolved_device),
        "choices": list(DEVICE_CHOICES),
        "collision_choices": list(COLLISION_POLICIES),
        "path": str(settings_path(db_path)),
    }


@dataclass
class Plan:
    plan_id: str
    mode: str
    operation: str
    organize_root: str
    created_at: float
    total: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    actions: list[PlannedAction] = field(default_factory=list)
    burst_count: int = 0
    dry_run: bool = True
    report_path: str | None = None
    input_hash: str = ""
    include_duplicates: bool = True
    applied_transaction: str | None = None


def _alt_key() -> str:
    member = getattr(CategoryType, "BURST_ALTERNATE", None)
    return member.value if member is not None else "burst_alternate"


def compute_inputs_hash(records) -> str:
    """Stable hash of every routing input; changes when the DB state changes."""
    h = hashlib.sha256()
    for r in sorted(records, key=lambda x: x.id or 0):
        if getattr(r, "scan_status", "scanned") == "missing":
            continue
        sem = getattr(r, "semantic", None)
        q = getattr(r, "quality", None)
        faces = getattr(r, "faces", None) or []
        row = (
            r.id, r.path, r.file_size, r.sha256, r.infer_status, r.is_duplicate,
            r.manual_bucket, r.locked,
            r.taken_at.isoformat() if r.taken_at else "",
            r.burst_id, r.is_best_shot, r.is_alternate,
            sem.top1_category if sem else "",
            round(sem.top1_score, 4) if sem else 0,
            round(sem.score_margin, 4) if sem else 0,
            (q.is_blurry, q.is_too_dark, q.is_overexposed) if q else (),
            sorted(round(f.confidence, 3) for f in faces),
        )
        h.update(repr(row).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


class Planner:
    """Cluster + decide + organize into a reviewable, persisted plan."""

    def __init__(self, config: CuratorConfig | None = None,
                 db_path: Path | None = None):
        self.cfg = config or CuratorConfig()
        self.db_path = db_path or self.cfg.db_path

    # ------------------------------------------------------------- cluster
    def cluster(self, window_seconds: int | None = None,
                similarity_threshold: float | None = None) -> int:
        """(Re)cluster bursts and pick best shots; returns the burst count."""
        if window_seconds is not None:
            self.cfg.burst.window_seconds = int(window_seconds)
        if similarity_threshold is not None:
            self.cfg.burst.similarity_threshold = float(similarity_threshold)
        with get_session(self.db_path) as session:
            repo = PhotoRepository(session)
            return self._cluster(repo, repo.all_images())

    def _retire_legacy_trashed(self, records) -> int:
        """Ensure any row whose path already lives inside .trash is in the trash stage."""
        retired = 0
        trash_name = self.cfg.trash_dir_name
        for r in records:
            if (getattr(r, "scan_status", "scanned") != "missing" and r.stage != STAGE_TRASH
                    and trash_name in Path(r.path).parts):
                r.stage = STAGE_TRASH
                r.scan_status = "missing"
                r.route_status = "trashed"
                retired += 1
        return retired

    def _cluster(self, repo: PhotoRepository, records) -> int:
        self._retire_legacy_trashed(records)
        for r in records:
            r.burst_id = None
            r.is_best_shot = False
            r.is_alternate = False
        for b in repo.all_bursts():
            repo.s.delete(b)
        repo.s.flush()

        active = [r for r in records if getattr(r, "scan_status", "scanned") != "missing"]
        burst_results = BurstDetector(self.cfg).cluster_records(active)
        by_id = {r.id: r for r in active}
        groups = []
        for res in burst_results:
            members = [by_id[m] for m in res.image_ids if m in by_id]
            if len(members) >= self.cfg.burst.min_group_size:
                groups.append(members)
        # Blinks are judged per burst (a subject's own frames set their baseline) before the
        # best shot is chosen, so the selector sees the closed-eye counts.
        apply_blinks(active, groups, self.cfg.face)
        burst_count = 0
        for members in groups:
            best_id = BestShotSelector(self.cfg).pick_best_shot(
                [_candidate_from_record(m, self.cfg.taxonomy.face_confidence_min,
                                        self.cfg.burst.face_sharp_ref)
                 for m in members]).image_id
            earliest = min(members, key=lambda m: (m.taken_at is None,
                                                   m.taken_at or _EPOCH))
            stamp = earliest.taken_at.strftime("%Y%m%d") if earliest.taken_at \
                else "Unknown"
            bg = repo.upsert_burst(BurstGroup(
                name=f"Burst_{stamp}_{burst_count:04d}", image_count=len(members),
                best_shot_id=best_id,
                start_time=earliest.taken_at,
                end_time=max((m.taken_at for m in members if m.taken_at),
                             default=earliest.taken_at),
            ))
            burst_count += 1
            for m in members:
                m.burst_id = bg.id
                m.is_best_shot = m.id == best_id
                m.is_alternate = m.id != best_id
        return burst_count

    # ---------------------------------------------------------------- plan
    def _confirm_near_duplicates(self, records) -> int:
        """Drop perceptual-hash near-duplicate links the CLIP embeddings do not support.

        Flat graphics and screenshots hash alike even when the content differs, so a
        link is kept only when both photos have embeddings that are close enough.
        Links without embeddings on either side are left alone. Returns links cleared.
        """
        import numpy as np

        from .db.repository import dequantize_embedding

        floor = float(getattr(self.cfg.taxonomy, "near_dup_min_cosine", 0.93))
        by_id = {r.id: r for r in records}
        cache: dict[int, np.ndarray | None] = {}

        def emb(rec):
            if rec.id not in cache:
                sem = getattr(rec, "semantic", None)
                v = dequantize_embedding(sem.embedding) if sem is not None else None
                a = np.asarray(v, dtype=np.float32) if v else None
                n = float(np.linalg.norm(a)) if a is not None else 0.0
                cache[rec.id] = a / n if n > 1e-9 else None
            return cache[rec.id]

        cleared = 0
        for r in records:
            tgt = getattr(r, "near_dup_of", None)
            if tgt is None or tgt not in by_id:
                continue
            a, b = emb(r), emb(by_id[tgt])
            if a is None or b is None or a.shape != b.shape:
                continue
            if float(a @ b) < floor:
                r.near_dup_of = None
                cleared += 1
        if cleared:
            log.info("cleared %d near-duplicate links not confirmed by embeddings", cleared)
        return cleared

    def build_plan(self,
                   organize_root: Path | None = None,
                   operation: str = "move",
                   mode: OrganizeMode = OrganizeMode.CATEGORY_FIRST,
                   plan_id: str | None = None,
                   include_duplicates: bool = True,
                   persist: bool = True,
                   recluster: bool | None = None,
                   window_seconds: int | None = None,
                   similarity_threshold: float | None = None) -> Plan:
        """Decide + organize. Existing bursts are reused unless ``recluster``.

        ``recluster=None`` reclusters only when no bursts exist yet, or when a
        threshold override is given.
        """
        if operation not in OPERATIONS:
            raise ValueError(f"unsupported operation: {operation}")
        from .library import require_library_root
        root = Path(organize_root) if organize_root else require_library_root(self.db_path, self.cfg)
        pid = plan_id or f"PLAN_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        if window_seconds is not None:
            self.cfg.burst.window_seconds = int(window_seconds)
        if similarity_threshold is not None:
            self.cfg.burst.similarity_threshold = float(similarity_threshold)
        override = window_seconds is not None or similarity_threshold is not None
        mode = OrganizeMode(mode)
        with get_session(self.db_path) as session:
            repo = PhotoRepository(session)
            records = repo.all_images()
            self._retire_legacy_trashed(records)
            self._confirm_near_duplicates(records)

            has_bursts = bool(repo.all_bursts())
            if recluster or override or not has_bursts:
                burst_count = self._cluster(repo, records)
            else:
                burst_count = len(repo.all_bursts())

            matrix = DecisionMatrix(self.cfg)
            organizer = FolderOrganizer(self.cfg, organize_root=root, mode=mode)
            organizer.set_bursts(repo.all_bursts())

            actions: list[PlannedAction] = []
            by_category: dict[str, int] = {}
            for r in records:
                if r.infer_status not in ("inferred", "failed") or r.scan_status == "missing":
                    continue
                if r.stage != STAGE_NEW:  # finished and trashed photos are not planned again
                    continue
                if r.is_duplicate and not include_duplicates:
                    continue
                dec = self._decide(matrix, r)
                target = organizer.full_target(r, dec)
                actions.append(PlannedAction(
                    image_id=r.id, source=Path(r.path), target=target,
                    operation=operation,
                    category=dec.category, reason=dec.reason,
                    is_alternate=dec.is_alternate,
                    burst_name=organizer.burst_name_for(r) if dec.is_alternate else None,
                ))
                key = _alt_key() if dec.is_alternate else dec.category.value
                by_category[key] = by_category.get(key, 0) + 1
                repo.set_routing(r, key, dec.subcategory, str(target),
                                 route_status="planned")
            session.flush()
            input_hash = compute_inputs_hash(records)

        plan = Plan(
            plan_id=pid, mode=mode.value, operation=operation,
            organize_root=str(root), created_at=time.time(), total=len(actions),
            by_category=by_category, actions=actions, burst_count=burst_count,
            input_hash=input_hash, include_duplicates=include_duplicates,
        )
        if persist:
            self._persist_plan(plan)
        return plan

    @staticmethod
    def _decide(matrix: DecisionMatrix, r):
        """Manual overrides and locked categories win over the matrix."""
        if r.manual_bucket:
            try:
                cat = CategoryType(r.manual_bucket)
            except ValueError:
                log.warning("ignoring invalid manual bucket %r for image %s",
                            r.manual_bucket, r.id)
            else:
                return Decision(cat, reason="manual override", subcategory="manual",
                                needs_date=cat.is_date_grouped,
                                is_alternate=cat == CategoryType.BURST_ALTERNATE)
        if r.locked and r.category:
            try:
                cat = CategoryType(r.category)
            except ValueError:
                pass
            else:
                return Decision(cat, reason="locked", subcategory=r.subcategory or "",
                                needs_date=cat.is_date_grouped,
                                is_alternate=cat == CategoryType.BURST_ALTERNATE)
        return matrix.evaluate_record(r)

    def route_new(self) -> int:
        """Give analysed photos that have no category yet one, so they show up for review.

        No plan is built and nothing on disk changes. Hand-sorted and locked photos are left to
        their owners. Returns how many photos were filed.
        """
        from sqlalchemy.orm import selectinload
        from .db.models import ImageRecord, utcnow
        I = ImageRecord
        filed = 0
        with get_session(self.db_path) as session:
            matrix = DecisionMatrix(self.cfg)
            rows = session.exec(
                select(I).where(I.category.is_(None), I.infer_status.in_(("inferred", "failed")),
                                I.scan_status != "missing", I.manual_bucket.is_(None), I.locked == False)  # noqa: E712
                .options(selectinload(I.semantic), selectinload(I.quality), selectinload(I.faces))).all()
            for r in rows:
                dec = self._decide(matrix, r)
                r.category = _alt_key() if dec.is_alternate else dec.category.value
                r.subcategory = dec.subcategory
                r.updated_at = utcnow()
                filed += 1
        return filed

    # --------------------------------------------------------- persistence
    def _persist_plan(self, plan: Plan) -> None:
        with get_session(self.db_path) as session:
            session.add(PlanRecord(
                plan_id=plan.plan_id, mode=plan.mode, operation=plan.operation,
                organize_root=plan.organize_root,
                include_duplicates=plan.include_duplicates,
                input_hash=plan.input_hash, burst_count=plan.burst_count))
            for i, a in enumerate(plan.actions):
                session.add(PlanActionRow(
                    plan_id=plan.plan_id, seq=i, image_id=a.image_id,
                    source=str(a.source), target=str(a.target),
                    operation=a.operation, category=a.category.value,
                    reason=a.reason, is_alternate=a.is_alternate,
                    burst_name=a.burst_name))

    def load_plan(self, plan_id: str) -> Plan:
        with get_session(self.db_path) as session:
            rec = session.exec(select(PlanRecord).where(
                PlanRecord.plan_id == plan_id)).first()
            if rec is None:
                raise KeyError(f"unknown plan id: {plan_id}")
            rows = session.exec(select(PlanActionRow).where(
                PlanActionRow.plan_id == plan_id).order_by(PlanActionRow.seq)).all()
            actions = []
            by_category: dict[str, int] = {}
            for r in rows:
                try:
                    cat = CategoryType(r.category)
                except ValueError:
                    cat = CategoryType.UNCERTAIN
                actions.append(PlannedAction(
                    image_id=r.image_id, source=Path(r.source), target=Path(r.target),
                    category=cat, operation=r.operation, reason=r.reason,
                    is_alternate=r.is_alternate, burst_name=r.burst_name))
                key = _alt_key() if r.is_alternate else r.category
                by_category[key] = by_category.get(key, 0) + 1
            return Plan(
                plan_id=rec.plan_id, mode=rec.mode, operation=rec.operation,
                organize_root=rec.organize_root,
                created_at=rec.created_at.timestamp() if rec.created_at else 0.0,
                total=len(actions), by_category=by_category, actions=actions,
                burst_count=rec.burst_count, input_hash=rec.input_hash,
                include_duplicates=rec.include_duplicates,
                applied_transaction=rec.applied_transaction)

    def list_plans(self) -> list[dict]:
        with get_session(self.db_path) as session:
            rows = session.exec(select(PlanRecord).order_by(
                PlanRecord.id.desc())).all()  # type: ignore[attr-defined]
            return [{"plan_id": r.plan_id, "mode": r.mode, "operation": r.operation,
                     "organize_root": r.organize_root,
                     "applied_transaction": r.applied_transaction,
                     "created_at": r.created_at.isoformat() if r.created_at else ""}
                    for r in rows]

    def check_plan(self, plan: Plan) -> None:
        """Raise :class:`PlanDriftError` if the plan is stale or already applied."""
        if not plan.input_hash:
            return
        with get_session(self.db_path) as session:
            rec = session.exec(select(PlanRecord).where(
                PlanRecord.plan_id == plan.plan_id)).first()
            if rec is not None and rec.applied_transaction:
                raise PlanDriftError(
                    f"plan {plan.plan_id} was already applied "
                    f"({rec.applied_transaction}); build a new plan or use force")
            current = compute_inputs_hash(PhotoRepository(session).all_images())
        if current != plan.input_hash:
            raise PlanDriftError(
                f"plan {plan.plan_id} is stale: the state store changed since it "
                "was built; rebuild the plan or use force")

    # -------------------------------------------------------------- execute
    def execute_plan(self, plan_or_plan_id: Plan | str,
                     only_category=None, trash: bool = False,
                     force: bool = False,
                     operation: str | None = None,
                     cancel_check: Callable[[], bool] | None = None) -> ExecutionReport:
        """Apply a persisted (or in-memory) plan exactly as built.

        ``operation`` is an explicit override only ("move" or "copy").
        """
        plan = (self.load_plan(plan_or_plan_id)
                if isinstance(plan_or_plan_id, str) else plan_or_plan_id)
        if not force:
            self.check_plan(plan)
        if operation:
            if operation not in OPERATIONS:
                raise ValueError(f"unsupported operation: {operation}")
            for a in plan.actions:
                a.operation = operation
        engine = ActionEngine(self.cfg, db_path=self.db_path)
        rep = engine.execute(plan.actions, organize_root=Path(plan.organize_root),
                             only_category=only_category, trash=trash,
                             cancel_check=cancel_check)
        if rep.executed:
            partial = rep.filtered > 0 or (
                rep.executed + rep.failures + rep.skipped) < len(plan.actions)
            with get_session(self.db_path) as session:
                rec = session.exec(select(PlanRecord).where(
                    PlanRecord.plan_id == plan.plan_id)).first()
                if rec is not None and not partial:
                    rec.applied_transaction = rep.transaction_id
                elif rec is not None and plan.input_hash:
                    # Partial apply moved files: rebase the drift hash so the rest applies without force.
                    fresh = compute_inputs_hash(PhotoRepository(session).all_images())
                    rec.input_hash = fresh
                    plan.input_hash = fresh
            if not partial:
                plan.applied_transaction = rep.transaction_id
        return rep

    def dry_run(self, plan: Plan | str) -> ExecutionReport:
        if isinstance(plan, str):
            plan = self.load_plan(plan)
        return ActionEngine(self.cfg, db_path=self.db_path).dry_run(plan.actions)

    # ---------------------------------------------------------------- undo
    def undo(self, transaction_id: str | None = None,
             verify: bool = True,
             cancel_check: Callable[[], bool] | None = None) -> RollbackReport:
        tid = transaction_id
        if not tid:
            with get_session(self.db_path) as session:
                tid = PhotoRepository(session).latest_transaction()
        if not tid:
            return RollbackReport(transaction_id="",
                                  notes=["no transactions found"])
        return RollbackEngine(self.cfg, db_path=self.db_path).rollback_transaction(
            tid, verify=verify, cancel_check=cancel_check)

    def list_transactions(self) -> list[dict]:
        with get_session(self.db_path) as session:
            return PhotoRepository(session).list_transactions()

    # ------------------------------------------------------ verify / repair
    def verify_transaction(self, transaction_id: str | None = None) -> dict:
        """Re-hash every committed target and report drift."""
        out = {"transaction_id": transaction_id, "checked": 0, "ok": 0,
               "drift": [], "missing": []}
        with get_session(self.db_path) as session:
            repo = PhotoRepository(session)
            tid = transaction_id or repo.latest_transaction()
            out["transaction_id"] = tid
            if not tid:
                return out
            for e in repo.ledger_by_txn(tid):
                if e.status != "committed":
                    continue
                out["checked"] += 1
                t = Path(e.target_path)
                if not os.path.lexists(t):
                    out["missing"].append(e.target_path)
                    continue
                if e.operation_type == "symlink":
                    good = t.is_symlink() and t.exists()
                    actual = ""
                else:
                    try:
                        actual = file_sha256(t)
                    except OSError:
                        actual = ""
                    good = (not e.sha256) or actual == e.sha256
                if good:
                    out["ok"] += 1
                else:
                    out["drift"].append({"path": e.target_path,
                                         "expected": e.sha256, "actual": actual})
        return out

    def repair_paths(self) -> int:
        """Fix images.path from the ledger (committed moves and rollbacks)."""
        fixed = 0
        trash_name = self.cfg.trash_dir_name
        with get_session(self.db_path) as session:
            repo = PhotoRepository(session)
            final: dict[int, tuple[str, str]] = {}
            cands: dict[int, list[tuple[str, str]]] = {}
            for e in session.exec(select(ActionLedger).order_by(
                    ActionLedger.id)).all():  # type: ignore[arg-type]
                if e.image_id is None or e.operation_type != "move":
                    continue
                if e.status == "committed":
                    final[e.image_id] = (e.target_path, "committed")
                    cands.setdefault(e.image_id, []).append((e.target_path, "committed"))
                    cands[e.image_id].append((e.source_path, "rolled_back"))
                elif e.status == "rolled_back":
                    final[e.image_id] = (e.source_path, "rolled_back")
                    cands.setdefault(e.image_id, []).append((e.source_path, "rolled_back"))
                elif e.status == "failed":
                    cands.setdefault(e.image_id, []).append((e.source_path, "rolled_back"))
                    cands[e.image_id].append((e.target_path, "committed"))
            for image_id, (loc, st) in final.items():
                img = repo.get_by_id(image_id)
                if img is None:
                    continue
                if not os.path.lexists(loc):
                    found = next(((p, s) for p, s in reversed(cands.get(image_id, []))
                                  if os.path.lexists(p)), None)
                    if found is None:
                        continue
                    loc, st = found
                clash = repo.get_by_path(loc)
                if clash is not None and clash.id != img.id:
                    continue
                changed = False
                if img.path != loc:
                    img.path = loc
                    changed = True
                in_trash = trash_name in Path(loc).parts
                if st == "committed" and in_trash:
                    if img.stage != STAGE_TRASH or img.route_status != "trashed":
                        img.stage = STAGE_TRASH
                        img.scan_status = "missing"
                        img.route_status = "trashed"
                        changed = True
                elif st == "rolled_back" or (not in_trash and img.route_status == "trashed"):
                    if img.scan_status == "missing" or img.route_status == "trashed":
                        img.scan_status = "scanned"
                        img.route_status = "unrouted" if st == "rolled_back" else "executed"
                        img.stage = STAGE_NEW if st == "rolled_back" else STAGE_LIBRARY
                        if st == "rolled_back":
                            img.target_path = None
                        changed = True
                if changed:
                    fixed += 1
        return fixed

    # ------------------------------------------------------- backend and learning
    def backend_info(self) -> dict:
        """Which AI backends are active, plus how many rows the stub produced."""
        info = describe_backends(self.cfg)
        with get_session(self.db_path) as session:
            info["stub_rows"] = PhotoRepository(session).count_stub_rows()
        return info

    def learn_from_corrections(self, rebuild: bool = True) -> dict:
        """Re-score every photo using the user's corrections, then refresh routing."""
        from .inference import semantic_engine as se
        fn = getattr(se, "learn_from_corrections", None)
        if fn is None:
            raise RuntimeError("Learning from corrections is not available in this build.")
        with get_session(self.db_path) as session:
            repo = PhotoRepository(session)
            before = repo.count_category("uncertain")
            summary = dict(fn(session, self.cfg))
        out = {**summary, "uncertain_before": before, "plan_id": None}
        if rebuild:
            from .library import library_root
            root = library_root(self.db_path, self.cfg) or self.cfg.organize_root
            out["plan_id"] = self.build_plan(organize_root=root, persist=True).plan_id
        with get_session(self.db_path) as session:
            after = PhotoRepository(session).count_category("uncertain")
        out["uncertain_after"] = after
        out["moved_out_of_uncertain"] = max(0, before - after)
        return out

    # ---------------------------------------------------------------- trash
    def empty_trash(self, root: Path | None = None) -> dict:
        """Delete everything in the trash; nothing leaves it on a timer."""
        from .trash_actions import empty_trash
        return empty_trash(self.db_path, self.cfg, root=root)

    # --------------------------------------------------------------- report
    def render_html_report(self, plan: Plan, output: Path | str,
                           extra: dict | None = None) -> Path:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = _plan_to_rows(self.db_path, plan)
        out.write_text(_render_html(plan, rows, extra or {}), encoding="utf-8")
        return out

    def print_summary(self, plan: Plan) -> None:
        t = Table(title=f"Plan {plan.plan_id}  ({plan.mode} / {plan.operation})")
        t.add_column("Category", style="cyan")
        t.add_column("Count", justify="right")
        t.add_column("Target folder", style="dim")
        for cat, n in sorted(plan.by_category.items(), key=lambda kv: -kv[1]):
            try:
                folder = CategoryType(cat).folder
            except ValueError:
                folder = "Bursts" if cat == _alt_key() else cat
            t.add_row(cat, str(n), folder)
        t.add_row("[bold]Total[/bold]", f"[bold]{plan.total}[/bold]",
                  f"{plan.burst_count} burst groups")
        console.print(t)
        console.print(f"[dim]Organize root: {plan.organize_root}[/dim]")


def _plan_to_rows(db_path: Path, plan: Plan) -> list[dict]:
    with get_session(db_path) as session:
        repo = PhotoRepository(session)
        rows = []
        for a in plan.actions:
            img = repo.get_by_id(a.image_id)
            if img is None:
                continue
            sem = getattr(img, "semantic", None)
            q = getattr(img, "quality", None)
            rows.append({
                "image_id": a.image_id,
                "filename": img.filename,
                "relative_path": img.relative_path,
                "category": a.category.value,
                "reason": a.reason,
                "target": str(a.target),
                "operation": a.operation,
                "is_alternate": a.is_alternate,
                "is_duplicate": img.is_duplicate,
                "taken_at": img.taken_at.isoformat() if img.taken_at else "",
                "width": img.width, "height": img.height,
                "top1": sem.top1_category if sem else "",
                "top1_score": round(sem.top1_score, 3) if sem else 0,
                "margin": round(sem.score_margin, 3) if sem else 0,
                "quality_score": round(q.quality_score, 3) if q else 0,
                "is_blurry": bool(q.is_blurry) if q else False,
                "is_too_dark": bool(q.is_too_dark) if q else False,
                "is_overexposed": bool(q.is_overexposed) if q else False,
                "has_face": bool(getattr(img, "faces", None)),
            })
        return rows


def _rel_target(target: str, root: str) -> str:
    try:
        return str(Path(target).relative_to(root))
    except ValueError:
        return target


def _render_html(plan: Plan, rows: list[dict], extra: dict) -> str:
    total = plan.total
    cats = plan.by_category

    def pct(n):
        return f"{(100.0 * n / total):.1f}%" if total else "0.0%"

    cat_cards = "".join(
        f'<div class="card"><div class="card-n">{v}</div>'
        f'<div class="card-l">{html.escape(c)}</div>'
        f'<div class="card-p">{pct(v)}</div></div>'
        for c, v in sorted(cats.items(), key=lambda kv: -kv[1])
    )

    trs = []
    for r in rows:
        badges = []
        if r["is_duplicate"]:
            badges.append('<span class="b dup">DUP</span>')
        if r["is_alternate"]:
            badges.append('<span class="b alt">ALT</span>')
        if r["has_face"]:
            badges.append('<span class="b face">face</span>')
        if r["is_blurry"]:
            badges.append('<span class="b blur">blurry</span>')
        if r["is_too_dark"]:
            badges.append('<span class="b dark">dark</span>')
        if r["is_overexposed"]:
            badges.append('<span class="b over">over</span>')
        taken = r["taken_at"][:19].replace("T", " ") if r["taken_at"] else "-"
        clip = (f'{html.escape(r["top1"])} {r["top1_score"]}' if r["top1"] else "-")
        trs.append(
            "<tr>"
            f'<td class="fn" title="{html.escape(r["relative_path"])}">{html.escape(r["filename"])}</td>'
            f'<td class="cat" data-cat="{html.escape(r["category"])}">{html.escape(r["category"])}</td>'
            f'<td>{"".join(badges)}</td>'
            f'<td class="num">{r["width"]}×{r["height"]}</td>'
            f'<td class="num">{clip}</td>'
            f'<td class="num">{r["quality_score"]}</td>'
            f'<td class="date">{taken}</td>'
            f'<td class="reason" title="{html.escape(r["reason"])}">{html.escape(r["reason"][:60])}</td>'
            f'<td class="target" title="{html.escape(r["target"])}">{html.escape(_rel_target(r["target"], plan.organize_root))}</td>'
            f'<td class="op">{html.escape(r["operation"])}</td>'
            "</tr>"
        )

    meta_html = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in extra.items()
    )
    cats_json = json_dumps(cats)  # raw JSON; keys are CategoryType values (safe)

    return _HTML_TEMPLATE.format(
        plan_id=html.escape(plan.plan_id),
        mode=html.escape(plan.mode),
        operation=html.escape(plan.operation),
        total=total,
        burst_count=plan.burst_count,
        dupes=sum(1 for r in rows if r["is_duplicate"]),
        ncats=len(cats),
        meta=meta_html,
        cards=cat_cards,
        rows="".join(trs),
        cats_json=cats_json,
        root=html.escape(plan.organize_root),
    )


def json_dumps(obj) -> str:
    import json
    return json.dumps(obj)


# ``{}`` in the JS below are escaped as ``{{ }}`` for str.format.
_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Photocure - Plan {plan_id}</title>
<style>
:root{{--bg:#0f1117;--panel:#171a23;--panel2:#1e2230;--fg:#e7e9f0;--mut:#8b93a7;
--acc:#5b8cff;--line:#262b3a}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}}
header{{padding:22px 26px;border-bottom:1px solid var(--line);
display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:12px}}
h1{{margin:0;font-size:20px}}
.sub{{color:var(--mut);font-size:13px;margin-top:4px}}
.stats{{display:flex;gap:10px;flex-wrap:wrap}}
.stat{{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:10px 14px;min-width:96px}}
.stat b{{display:block;font-size:20px}}
.stat span{{color:var(--mut);font-size:12px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px;padding:18px 26px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}}
.card-n{{font-size:26px;font-weight:600;color:var(--acc)}}
.card-l{{text-transform:capitalize}}
.card-p{{color:var(--mut);font-size:12px}}
.controls{{display:flex;gap:10px;padding:0 26px 8px;flex-wrap:wrap;align-items:center}}
.controls input,.controls select{{background:var(--panel2);border:1px solid var(--line);
color:var(--fg);border-radius:8px;padding:8px 10px;font-size:13px}}
.controls input{{min-width:260px}}
table{{width:calc(100% - 52px);margin:0 26px 12px;border-collapse:collapse;font-size:13px}}
th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}}
th{{position:sticky;top:0;background:var(--panel);color:var(--mut);font-weight:600;cursor:pointer}}
td.num{{font-variant-numeric:tabular-nums;color:var(--mut);white-space:nowrap}}
td.date{{white-space:nowrap;color:var(--mut)}}
td.fn{{max-width:320px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
td.target{{max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--acc)}}
td.reason{{max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--mut)}}
tr:hover td{{background:rgba(91,140,255,.06)}}
.b{{display:inline-block;font-size:10px;padding:2px 6px;border-radius:6px;margin-right:4px;font-weight:600}}
.b.dup{{background:#3a2530;color:#ff9db4}}
.b.alt{{background:#25324a;color:#8ab4ff}}
.b.face{{background:#1f3a2b;color:#7fe0a6}}
.b.blur{{background:#3a3320;color:#f2d17a}}
.b.dark{{background:#2b2f3a;color:#aab4cf}}
.b.over{{background:#3a2a24;color:#f2b48a}}
.cat{{text-transform:capitalize}}
table.meta{{width:calc(100% - 52px);margin:0 26px 10px;border-collapse:collapse}}
table.meta th{{text-align:left;color:var(--mut);font-weight:500;width:200px;padding:4px 8px}}
table.meta td{{padding:4px 8px;color:var(--fg)}}
footer{{padding:0 26px 30px;color:var(--mut);font-size:12px}}
</style>
</head><body>
<header>
  <div>
    <h1>Photocure - Dry-Run Plan</h1>
    <div class="sub">Plan <b>{plan_id}</b> · mode <b>{mode}</b> · operation <b>{operation}</b></div>
  </div>
  <div class="stats">
    <div class="stat"><b>{total}</b><span>files to route</span></div>
    <div class="stat"><b>{burst_count}</b><span>burst groups</span></div>
    <div class="stat"><b>{dupes}</b><span>duplicates</span></div>
    <div class="stat"><b>{ncats}</b><span>categories</span></div>
  </div>
</header>
<table class="meta"><tbody>{meta}</tbody></table>
<div class="cards">{cards}</div>
<div class="controls">
  <input id="q" type="search" placeholder="Search filename, path or reason…">
  <select id="cat"><option value="">All categories</option></select>
  <span id="count" style="color:var(--mut)"></span>
</div>
<table id="tbl">
<thead><tr>
<th data-k="0">File</th><th data-k="1">Category</th><th>Flags</th>
<th data-k="3">Size</th><th data-k="4">CLIP</th>
<th data-k="5">Quality</th><th data-k="6">Taken</th>
<th>Reason</th><th>Target</th><th>Op</th></tr></thead>
<tbody id="rows">{rows}</tbody></table>
<footer>Generated by <b>photocure</b> · Organize root:
<code>{root}</code> · Dry-run: nothing has been moved.</footer>
<script id="cats" type="application/json">{cats_json}</script>
<script>
(function(){{
  var rows=Array.from(document.querySelectorAll('#rows tr'));
  var sel=document.getElementById('cat'), q=document.getElementById('q'),
      count=document.getElementById('count');
  var cats=JSON.parse(document.getElementById('cats').textContent);
  Object.keys(cats).sort().forEach(function(c){{
    var o=document.createElement('option');o.value=c;o.textContent=c;sel.appendChild(o);
  }});
  var sortKey=null, dir=1;
  var isNum={{3:1,4:1,5:1,6:1}};
  function apply(){{
    var f=0;
    rows.forEach(function(tr){{
      var cat=tr.querySelector('.cat').getAttribute('data-cat');
      var hay=tr.textContent.toLowerCase();
      var ok=(sel.value===''||cat===sel.value)&&(q.value===''||hay.indexOf(q.value.toLowerCase())>=0);
      tr.style.display=ok?'':'none'; if(ok)f++;
    }});
    if(sortKey!==null){{
      rows.sort(function(a,b){{
        var av=a.cells[sortKey].textContent.trim(), bv=b.cells[sortKey].textContent.trim();
        if(isNum[sortKey]){{ av=parseFloat(av)||0; bv=parseFloat(bv)||0; }}
        return av<bv?-dir:av>bv?dir:0;
      }});
      rows.forEach(function(tr){{ document.getElementById('rows').appendChild(tr); }});
    }}
    count.textContent=f+' / '+rows.length+' rows';
  }}
  Array.from(document.querySelectorAll('th[data-k]')).forEach(function(th){{
    th.addEventListener('click',function(){{
      var k=parseInt(th.getAttribute('data-k'),10);
      dir=(sortKey===k)?-dir:1; sortKey=k; apply();
    }});
  }});
  sel.addEventListener('change',apply); q.addEventListener('input',apply); apply();
}})();
</script>
</body></html>
"""
