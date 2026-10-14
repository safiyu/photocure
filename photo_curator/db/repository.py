"""High-performance batch repository on top of SQLModel/SQLAlchemy."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import joinedload
from sqlmodel import Session, col, select as sselect

from ..db.models import (
    ActionLedger,
    BurstGroup,
    FaceDetection,
    ImageRecord,
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


def dequantize_embedding(blob: str) -> Optional[List[float]]:
    """Deserialize an embedding produced by :func:`quantize_embedding`."""
    import numpy as np

    if not blob:
        return None
    try:
        raw = json.loads(blob)
    except (ValueError, TypeError):
        return None
    arr = np.fromiter(raw, dtype=np.int16, count=len(raw)).view(np.float16)
    return [float(v) for v in arr]


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
            if k in ("id", "created_at"):
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

    def get_by_path(self, path: str) -> Optional[ImageRecord]:
        return self.s.exec(
            sselect(ImageRecord).where(ImageRecord.path == str(path))
        ).first()

    def get_by_id(self, image_id: int, with_faces: bool = False) -> Optional[ImageRecord]:
        stmt = sselect(ImageRecord).where(ImageRecord.id == image_id)
        if with_faces:
            stmt = stmt.options(joinedload(ImageRecord.faces))
        return self.s.exec(stmt).first()

    def known_paths(self) -> set[str]:
        """All paths already in the store (fast resumption check)."""
        rows = self.s.exec(sselect(ImageRecord.path)).all()
        return set(rows)

    def images_pending_inference(self, limit: int = 0) -> List[ImageRecord]:
        stmt = sselect(ImageRecord).where(ImageRecord.infer_status == "pending")
        if limit:
            stmt = stmt.limit(limit)
        return list(self.s.exec(stmt).all())

    def count_inferred(self) -> int:
        """Number of rows already marked inferred (resumption skip counter)."""
        return int(self.s.exec(
            sselect(func.count()).select_from(ImageRecord)
            .where(ImageRecord.infer_status == "inferred")
        ).one() or 0)

    def all_images(self, category: Optional[str] = None,
                   route_status: Optional[str] = None) -> List[ImageRecord]:
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
    def find_by_sha256(self, sha256: str, exclude_id: Optional[int] = None
                       ) -> Optional[ImageRecord]:
        stmt = sselect(ImageRecord).where(ImageRecord.sha256 == sha256)
        if exclude_id is not None:
            stmt = stmt.where(ImageRecord.id != exclude_id)
        return self.s.exec(stmt).first()

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

    def get_burst_by_name(self, name: str) -> Optional[BurstGroup]:
        return self.s.exec(sselect(BurstGroup).where(BurstGroup.name == name)).first()

    def all_bursts(self) -> List[BurstGroup]:
        return list(self.s.exec(sselect(BurstGroup)).all())

    # --------------------------------------------------------------- routing
    def set_routing(self, image: ImageRecord, category: str, subcategory: str,
                    target_path: str, route_status: str = "planned") -> None:
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

    def set_manual_bucket(self, image_id: int, bucket: Optional[str]) -> None:
        img = self.get_by_id(image_id)
        if img is not None:
            img.manual_bucket = bucket
            img.updated_at = utcnow()
            self.s.flush()

    # --------------------------------------------------------------- ledger
    def ledger_begin(self, transaction_id: str) -> None:
        """No-op hook kept for API symmetry (ledger rows are appended lazily)."""
        return None

    def ledger_add(self, txn: str, image_id: Optional[int], source: str, target: str,
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

    def ledger_by_txn(self, transaction_id: str) -> List[ActionLedger]:
        rows = self.s.exec(
            sselect(ActionLedger)
            .where(ActionLedger.transaction_id == transaction_id)
            .order_by(ActionLedger.id.asc())  # type: ignore[arg-type]
        ).all()
        return rows

    def ledger_committed_desc(self, transaction_id: str) -> List[ActionLedger]:
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

    def latest_transaction(self) -> Optional[str]:
        row = self.s.exec(
            sselect(func.max(ActionLedger.id))
        ).one()
        if row is None:
            return None
        return self.s.exec(
            sselect(ActionLedger.transaction_id)
            .where(ActionLedger.id == row)
        ).first()

    # -------------------------------------------------------------- stats
    def stats(self) -> dict:
        total = self.s.exec(sselect(func.count()).select_from(ImageRecord)).one()
        pending = self.s.exec(
            sselect(func.count()).select_from(ImageRecord)
            .where(ImageRecord.infer_status == "pending")
        ).one()
        dupes = self.s.exec(
            sselect(func.count()).select_from(ImageRecord)
            .where(ImageRecord.is_duplicate == True)  # noqa: E712
        ).one()
        planned = self.s.exec(
            sselect(func.count()).select_from(ImageRecord)
            .where(ImageRecord.route_status == "planned")
        ).one()
        bursts = self.s.exec(sselect(func.count()).select_from(BurstGroup)).one()
        return {
            "total_images": int(total or 0),
            "pending_inference": int(pending or 0),
            "duplicates": int(dupes or 0),
            "planned_actions": int(planned or 0),
            "burst_groups": int(bursts or 0),
        }

    # ----------------------------------------------------------- thumbnails
    def thumbnail_path(self, cache_dir: Path, image_id: int) -> Path:
        d = Path(cache_dir) / "thumbs"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{image_id:09d}.jpg"
