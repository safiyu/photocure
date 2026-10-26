"""High-performance batch repository on top of SQLModel/SQLAlchemy."""
from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterator, Sequence

from sqlalchemy import and_, func, or_
from sqlalchemy.orm import joinedload
from sqlmodel import Session, col, delete as sdelete, select as sselect, update as supdate

from ..db.models import (
    ActionLedger,
    BurstGroup,
    EventGroup,
    FaceDetection,
    ImageRecord,
    PersonCluster,
    PlanActionRow,
    PlanRecord,
    QualityMetric,
    SemanticScore,
    utcnow,
)

_EMBED_DIM = 512


def _f16(x: float) -> int:
    """Quantize a float to float16 bits and store as int16 (lossy but compact)."""
    import numpy as np

    v = np.float16(max(-65504.0, min(65504.0, float(x))))
    return int(v.view(np.int16))


def quantize_embedding(vec: Sequence[float]) -> str:
    """Serialize a normalized embedding as JSON of int16 float16 bit patterns."""
    return json.dumps([_f16(v) for v in vec])


def dequantize_embedding(blob: str) -> list[float] | None:
    """Deserialize an embedding produced by :func:`quantize_embedding`."""
    import numpy as np

    if not blob:
        return None
    try:
        raw = json.loads(blob)
    except (ValueError, TypeError):
        return None
    if not raw:
        return []
    if isinstance(raw[0], float):
        return [float(v) for v in raw]
    arr = np.fromiter(raw, dtype=np.int16, count=len(raw)).view(np.float16)
    return [float(v) for v in arr]


def stage_clause(stage: str | None):
    """SQL filter for one stage. "all" (or None) is new and library photos together.

    Trashed rows are marked missing so features skip them, so only the trash stage
    ignores scan_status; purged rows are gone from every stage.
    """
    I = ImageRecord
    if stage == "trash":
        return and_(I.stage == "trash", I.route_status != "purged")
    active = I.scan_status != "missing"
    if stage in ("new", "library"):
        return and_(active, I.stage == stage)
    return active


class PhotoRepository:
    """All state-store access goes through this class."""

    def __init__(self, session: Session):
        self.s = session

    # ------------------------------------------------------------------ images
    def upsert_image(self, rec: ImageRecord) -> ImageRecord:
        """Insert or update (by ``path``) a single image record."""
        existing = self.get_by_path(rec.path)
        if existing is None:
            self.s.add(rec)
            self.s.flush()
            return rec
        for k, v in rec.model_dump(exclude_none=True).items():
            if k in ("id", "created_at", "locked", "manual_bucket"):
                continue
            setattr(existing, k, v)
        existing.updated_at = utcnow()
        self.s.flush()
        return existing

    def bulk_upsert_images(self, records: Sequence[ImageRecord], commit_every: int = 500) -> int:
        """Upsert many records; returns number of rows written."""
        written = 0
        for i, rec in enumerate(records):
            self.upsert_image(rec)
            written += 1
            if commit_every and (i + 1) % commit_every == 0:
                self.s.commit()
        return written

    def get_by_path(self, path: str) -> ImageRecord | None:
        return self.s.exec(
            sselect(ImageRecord).where(ImageRecord.path == str(path))
        ).first()

    def get_by_id(self, image_id: int, with_faces: bool = False) -> ImageRecord | None:
        stmt = sselect(ImageRecord).where(ImageRecord.id == image_id)
        if with_faces:
            stmt = stmt.options(joinedload(ImageRecord.faces))
        return self.s.exec(stmt).first()

    def known_paths(self) -> set[str]:
        """All paths already in the store (fast resumption check)."""
        rows = self.s.exec(sselect(ImageRecord.path)).all()
        return set(rows)

    def known_quick_hashes(self) -> set[str]:
        """All Tier-1 quick-hashes already in the store.

        Used for *lazy* Tier-2 hashing: a new file only needs the expensive
        full-file SHA-256 if its quick-hash appears in this set (or collides
        with an earlier file in the same scan pass).
        """
        rows = self.s.exec(sselect(ImageRecord.quick_hash)).all()
        return {r for r in rows if r}

    def images_pending_inference(self, limit: int = 0) -> list[ImageRecord]:
        stmt = sselect(ImageRecord).where(
            col(ImageRecord.infer_status).in_(["pending", "faces_done"]),
            ImageRecord.scan_status != "missing")
        if limit:
            stmt = stmt.limit(limit)
        return list(self.s.exec(stmt).all())

    def reset_failed_inference(self) -> int:
        """Put rows whose inference failed back into the pending queue."""
        rows = self.s.exec(sselect(ImageRecord).where(
            ImageRecord.infer_status == "failed")).all()
        for r in rows:
            r.infer_status = "pending"
        self.s.flush()
        return len(rows)

    def count_stub_rows(self) -> int:
        """Semantic rows made by the stub (or an unknown backend)."""
        return int(self.s.exec(
            sselect(func.count()).select_from(SemanticScore)
            .where(col(SemanticScore.backend).in_(["", "stub"]))
        ).one() or 0)

    def count_labeled_examples(self) -> int:
        """Photos the user re-categorized that have a stored embedding."""
        return int(self.s.exec(
            sselect(func.count()).select_from(ImageRecord)
            .join(SemanticScore, SemanticScore.image_id == ImageRecord.id)
            .where(col(ImageRecord.manual_bucket).is_not(None),
                   ImageRecord.scan_status != "missing",
                   SemanticScore.embedding != "")
        ).one() or 0)

    def count_category(self, category: str) -> int:
        """Rows whose effective category (manual override first) matches."""
        eff = func.coalesce(ImageRecord.manual_bucket, ImageRecord.category)
        return int(self.s.exec(
            sselect(func.count()).select_from(ImageRecord).where(
                ImageRecord.scan_status != "missing", eff == category)
        ).one() or 0)

    def count_inferred(self) -> int:
        """Number of rows already marked inferred (resumption skip counter)."""
        return int(self.s.exec(
            sselect(func.count()).select_from(ImageRecord)
            .where(ImageRecord.scan_status != "missing",
                   ImageRecord.infer_status == "inferred")
        ).one() or 0)

    def all_images(self, category: str | None = None,
                   route_status: str | None = None) -> list[ImageRecord]:
        stmt = sselect(ImageRecord)
        if category:
            stmt = stmt.where(ImageRecord.category == category)
        if route_status:
            stmt = stmt.where(ImageRecord.route_status == route_status)
        return list(self.s.exec(stmt).all())

    def iter_images(self, chunk: int = 500, **filters) -> Iterator[ImageRecord]:
        stmt = sselect(ImageRecord)
        if filters.get("category"):
            stmt = stmt.where(ImageRecord.category == filters["category"])
        rows = self.s.exec(stmt).all()
        for i in range(0, len(rows), chunk):
            yield from rows[i:i + chunk]

    # --------------------------------------------------------------- dedup
    def find_by_sha256(self, sha256: str, exclude_id: int | None = None
                       ) -> ImageRecord | None:
        stmt = sselect(ImageRecord).where(ImageRecord.sha256 == sha256)
        if exclude_id is not None:
            stmt = stmt.where(ImageRecord.id != exclude_id)
        return self.s.exec(stmt).first()

    def find_by_quick_hash(self, quick_hash: str, exclude_id: int | None = None
                           ) -> list[ImageRecord]:
        """All records sharing a Tier-1 quick-hash (duplicate candidates)."""
        stmt = sselect(ImageRecord).where(ImageRecord.quick_hash == quick_hash)
        if exclude_id is not None:
            stmt = stmt.where(ImageRecord.id != exclude_id)
        return list(self.s.exec(stmt).all())

    def mark_duplicate(self, dup: ImageRecord, original: ImageRecord) -> None:
        dup.is_duplicate = True
        dup.duplicate_of = original.id
        dup.updated_at = utcnow()
        self.s.flush()

    # --------------------------------------------------------------- quality
    def upsert_quality(self, qm: QualityMetric) -> QualityMetric:
        existing = self.s.exec(
            sselect(QualityMetric).where(QualityMetric.image_id == qm.image_id)
        ).first()
        if existing is None:
            self.s.add(qm)
        else:
            for k, v in qm.model_dump(exclude_none=True).items():
                if k in ("id", "computed_at"):
                    continue
                setattr(existing, k, v)
        self.s.flush()
        return qm

    # ----------------------------------------------------------------- faces
    def replace_faces(self, image_id: int, faces: Sequence[FaceDetection]) -> None:
        existing = self.s.exec(
            sselect(FaceDetection).where(FaceDetection.image_id == image_id)
        ).all()
        for row in existing:
            self.s.delete(row)
        for f in faces:
            f.image_id = image_id
            self.s.add(f)
        self.s.flush()

    def faces_for_image(self, image_id: int) -> list[FaceDetection]:
        return list(self.s.exec(
            sselect(FaceDetection).where(FaceDetection.image_id == image_id)
        ).all())

    # ---------------------------------------------------------------- people
    def all_person_clusters(self) -> list[PersonCluster]:
        return list(self.s.exec(sselect(PersonCluster)).all())

    def create_person_cluster(self, embedding: Sequence[float], stage: str = "library") -> PersonCluster:
        p = PersonCluster(name=None, face_count=1, stage=stage, centroid_json=quantize_embedding(embedding))
        self.s.add(p)
        self.s.flush()
        return p

    def update_person_cluster(self, person_id: int, centroid: Sequence[float], face_count: int) -> None:
        p = self.s.get(PersonCluster, person_id)
        if p is None:
            raise ValueError(f"no person cluster with id {person_id}")
        p.centroid_json = quantize_embedding(centroid)
        p.face_count = face_count
        if p.face_count <= 0 and not p.name:
            self.s.delete(p)
        self.s.flush()

    def decrement_person_cluster(self, person_id: int) -> None:
        """Lower a cluster's face_count by one, floored at 0. A no-op if the cluster is gone."""
        p = self.s.get(PersonCluster, person_id)
        if p is None:
            return
        p.face_count = max(0, p.face_count - 1)
        if p.face_count <= 0 and not p.name:
            self.s.delete(p)
        self.s.flush()

    def rename_person(self, person_id: int, name: str | None) -> PersonCluster:
        p = self.s.get(PersonCluster, person_id)
        if p is None:
            raise ValueError(f"no person cluster with id {person_id}")
        p.name = name
        self.s.flush()
        return p

    def find_person_by_name(self, name: str) -> PersonCluster | None:
        target = name.strip().lower()
        return self.s.exec(
            sselect(PersonCluster).where(
                func.lower(func.trim(PersonCluster.name)) == target)
        ).first()

    # -------------------------------------------------------------- semantic
    def upsert_semantic(self, sem: SemanticScore) -> SemanticScore:
        existing = self.s.exec(
            sselect(SemanticScore).where(SemanticScore.image_id == sem.image_id)
        ).first()
        if existing is None:
            self.s.add(sem)
        else:
            for k, v in sem.model_dump(exclude_none=True).items():
                if k in ("id", "computed_at"):
                    continue
                setattr(existing, k, v)
            # A re-analysis replaces these even when empty, or an old score would outlive its embedding.
            existing.aesthetic = sem.aesthetic
            existing.aesthetic_model = sem.aesthetic_model
            existing.preference = sem.preference
        self.s.flush()
        return sem

    # --------------------------------------------------------------- bursts
    def upsert_burst(self, bg: BurstGroup) -> BurstGroup:
        existing = self.s.exec(
            sselect(BurstGroup).where(BurstGroup.name == bg.name)
        ).first()
        if existing is None:
            self.s.add(bg)
            self.s.flush()
            return bg
        for k, v in bg.model_dump(exclude_none=True).items():
            if k in ("id", "created_at"):
                continue
            setattr(existing, k, v)
        self.s.flush()
        return existing

    def get_burst_by_name(self, name: str) -> BurstGroup | None:
        return self.s.exec(sselect(BurstGroup).where(BurstGroup.name == name)).first()

    def all_bursts(self) -> list[BurstGroup]:
        return list(self.s.exec(sselect(BurstGroup)).all())

    def prune_bursts(self) -> int:
        """Clean up burst groups whose members were deleted, trashed, or reduced below 2.

        - Disassociates stale members (missing/trashed/purged) from burst groups.
        - Burst groups with 0 active members are deleted.
        - Burst groups with 1 active member have their burst flags cleared on that member
          and the burst group is deleted (a single frame cannot form a burst).
        - Burst groups with >= 2 active members have image_count updated and ensure
          best_shot_id points to an active member with member flags synchronized.
        Returns the number of burst groups removed.
        """
        bursts = self.all_bursts()
        deleted = 0
        for bg in bursts:
            # Clear burst flags from any images that are missing, trashed, or purged
            stale_members = self.s.exec(
                sselect(ImageRecord).where(
                    ImageRecord.burst_id == bg.id,
                    or_(
                        ImageRecord.scan_status == "missing",
                        col(ImageRecord.route_status).in_(["trashed", "purged"]),
                    ),
                )
            ).all()
            for sm in stale_members:
                sm.burst_id = None
                sm.is_best_shot = False
                sm.is_alternate = False

            members = self.s.exec(
                sselect(ImageRecord).where(
                    ImageRecord.burst_id == bg.id,
                    ImageRecord.scan_status != "missing",
                    ~col(ImageRecord.route_status).in_(["trashed", "purged"]),
                )
            ).all()
            if len(members) == 0:
                self.s.delete(bg)
                deleted += 1
            elif len(members) == 1:
                lone = members[0]
                lone.burst_id = None
                lone.is_best_shot = False
                lone.is_alternate = False
                self.s.delete(bg)
                deleted += 1
            else:
                if bg.image_count != len(members):
                    bg.image_count = len(members)
                active_ids = {m.id for m in members}
                if bg.best_shot_id not in active_ids:
                    best = max(members, key=lambda r: getattr(getattr(r, "quality", None), "sharpness_norm", 0.0) or 0.0)
                    bg.best_shot_id = best.id
                for m in members:
                    m.is_best_shot = (m.id == bg.best_shot_id)
                    m.is_alternate = (m.id != bg.best_shot_id)
        if deleted:
            self.s.flush()
        return deleted

    # --------------------------------------------------------------- routing
    def set_routing(self, image: ImageRecord, category: str, subcategory: str,
                    target_path: str, route_status: str = "planned") -> None:
        # Manual overrides and locked rows keep their category.
        if not (image.manual_bucket or image.locked):
            image.category = category
        image.subcategory = subcategory
        image.target_path = target_path
        image.route_status = route_status
        image.updated_at = utcnow()
        self.s.flush()

    def clear_routing(self) -> int:
        """Reset all planned routing (used before regenerating a plan)."""
        rows = self.s.exec(
            sselect(ImageRecord).where(col(ImageRecord.route_status).in_(
                ["planned", "executed"]))
        ).all()
        for r in rows:
            r.route_status = "unrouted"
            r.target_path = None
        self.s.flush()
        return len(rows)

    def set_locked(self, image_id: int, locked: bool = True) -> None:
        img = self.get_by_id(image_id)
        if img is not None:
            img.locked = bool(locked)
            img.updated_at = utcnow()
            self.s.flush()

    def set_manual_bucket(self, image_id: int, bucket: str | None) -> None:
        img = self.get_by_id(image_id)
        if img is not None:
            img.manual_bucket = bucket
            img.delete_approved = False  # a re-sorted photo needs a fresh approval
            img.updated_at = utcnow()
            self.s.flush()

    # --------------------------------------------------------------- ledger
    def ledger_begin(self, transaction_id: str) -> None:
        """No-op hook kept for API symmetry (ledger rows are appended lazily)."""
        return None

    def ledger_add(self, txn: str, image_id: int | None, source: str, target: str,
                   op: str, sha256: str = "", size: int = 0) -> ActionLedger:
        row = ActionLedger(
            transaction_id=txn, image_id=image_id, source_path=str(source),
            target_path=str(target), operation_type=op, sha256=sha256,
            file_size=size, status="pending",
        )
        self.s.add(row)
        self.s.flush()
        return row

    def ledger_commit(self, entry: ActionLedger) -> None:
        entry.status = "committed"
        entry.committed_at = utcnow()
        self.s.flush()

    def ledger_fail(self, entry: ActionLedger, error: str) -> None:
        entry.status = "failed"
        entry.error = error
        self.s.flush()

    def ledger_pending(self, transaction_id: str) -> list[ActionLedger]:
        return list(self.s.exec(
            sselect(ActionLedger).where(
                ActionLedger.transaction_id == transaction_id,
                ActionLedger.status == "pending")
            .order_by(ActionLedger.id.asc())  # type: ignore[arg-type]
        ).all())

    def list_transactions(self) -> list[dict]:
        rows = self.s.exec(
            sselect(ActionLedger.transaction_id, ActionLedger.status,
                    func.count(), func.max(ActionLedger.id))
            .group_by(ActionLedger.transaction_id, ActionLedger.status)
        ).all()
        out: dict[str, dict] = {}
        for tid, status, n, mx in rows:
            d = out.setdefault(tid, {"transaction_id": tid, "last_id": 0, "counts": {}})
            d["counts"][status] = int(n)
            d["last_id"] = max(d["last_id"], int(mx or 0))
        return sorted(out.values(), key=lambda d: d["last_id"], reverse=True)

    def ledger_by_txn(self, transaction_id: str) -> list[ActionLedger]:
        rows = self.s.exec(
            sselect(ActionLedger)
            .where(ActionLedger.transaction_id == transaction_id)
            .order_by(ActionLedger.id.asc())  # type: ignore[arg-type]
        ).all()
        return rows

    def ledger_committed_desc(self, transaction_id: str) -> list[ActionLedger]:
        """Committed actions in reverse chronological order (for rollback)."""
        rows = self.s.exec(
            sselect(ActionLedger)
            .where(
                ActionLedger.transaction_id == transaction_id,
                ActionLedger.status == "committed",
            )
            .order_by(ActionLedger.id.desc())  # type: ignore[arg-type]
        ).all()
        return rows

    def mark_rolled_back(self, entry: ActionLedger) -> None:
        entry.status = "rolled_back"
        self.s.flush()

    def latest_transaction(self) -> str | None:
        row = self.s.exec(
            sselect(func.max(ActionLedger.id))
            .where(ActionLedger.status.in_(["committed", "pending"]))  # type: ignore[attr-defined]
        ).one()
        if row is None:
            return None
        return self.s.exec(
            sselect(ActionLedger.transaction_id)
            .where(ActionLedger.id == row)
        ).first()

    # -------------------------------------------------------------- stats
    def stats(self) -> dict:
        active = ImageRecord.scan_status != "missing"
        total = self.s.exec(sselect(func.count()).select_from(ImageRecord).where(active)).one()
        pending = self.s.exec(
            sselect(func.count()).select_from(ImageRecord)
            .where(active, ImageRecord.infer_status == "pending")
        ).one()
        dupes = self.s.exec(
            sselect(func.count()).select_from(ImageRecord)
            .where(active, ImageRecord.is_duplicate == True)  # noqa: E712
        ).one()
        planned = self.s.exec(
            sselect(func.count()).select_from(ImageRecord)
            .where(active, ImageRecord.route_status == "planned")
        ).one()
        self.prune_bursts()
        bursts = self.s.exec(sselect(func.count()).select_from(BurstGroup)).one()
        people = self.s.exec(
            sselect(func.count()).select_from(PersonCluster)
            .where(PersonCluster.face_count > 0)
        ).one()
        stage_rows = self.s.exec(
            sselect(ImageRecord.stage, func.count()).where(active).group_by(ImageRecord.stage)
        ).all()
        stages = {"new": 0, "library": 0, "trash": 0}
        stages.update({k: int(n) for k, n in stage_rows if k in ("new", "library")})
        # Trashed rows are marked missing so features skip them; the trash is counted by stage.
        stages["trash"] = int(self.s.exec(
            sselect(func.count()).select_from(ImageRecord)
            .where(ImageRecord.stage == "trash", ImageRecord.route_status != "purged")
        ).one() or 0)
        return {
            "stages": stages,
            "total_images": int(total or 0),
            "pending_inference": int(pending or 0),
            "duplicates": int(dupes or 0),
            "planned_actions": int(planned or 0),
            "burst_groups": int(bursts or 0),
            "people": int(people or 0),
        }

    # ----------------------------------------------------------- thumbnails
    def thumbnail_path(self, cache_dir: Path, image_id: int) -> Path:
        d = Path(cache_dir) / "thumbs"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{image_id:09d}.jpg"

    # -------------------------------------------------------- clear / reset
    def clear_library(self, keep_ledger: bool = False) -> dict[str, int]:
        """Wipe all indexed records, clusters, plans, and optionally ledger entries."""
        counts: dict[str, int] = {}
        models = [
            SemanticScore,
            FaceDetection,
            QualityMetric,
            PlanActionRow,
            PlanRecord,
        ]
        if not keep_ledger:
            models.append(ActionLedger)
        models.extend([
            BurstGroup,
            PersonCluster,
            EventGroup,
            ImageRecord,
        ])

        if keep_ledger:
            self.s.exec(supdate(ActionLedger).values(image_id=None))

        for model in models:
            res = self.s.exec(sdelete(model))
            counts[model.__tablename__] = int(
                res.rowcount if res.rowcount is not None and res.rowcount >= 0 else 0
            )

        self.s.commit()
        return counts
