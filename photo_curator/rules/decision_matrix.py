"""Decision matrix: deterministic routing from image features to a bucket.

Implements spec section 4.B (and the milestone-4 routing rules). Precedence:

    1. Exact SHA-256 duplicate            -> Junk_Review/Duplicates/
    2. Burst alternate (non-best)         -> Bursts/<burst>/alternates/
    3. Pathological quality (dark/blurry) -> Junk_Review/Accidental/
    4. Human face (conf > 0.65)           -> People/YYYY/MM/DD/
    5. Low-confidence / low-margin CLIP   -> Uncertain_Review/
    6. Top CLIP category                  -> Screenshots / Documents / Food /
                                             Animals / Scenery
    7. Fallback                           -> Uncertain_Review/
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..config import CuratorConfig, TaxonomySettings
from ..db.models import FaceDetection, ImageRecord
from ..rules.taxonomy import CategoryType


@dataclass
class Decision:
    category: CategoryType
    reason: str
    subcategory: str = ""
    confidence: float = 0.0
    needs_date: bool = False
    # When True the image is a burst alternate (routed to the burst folder).
    is_alternate: bool = False
    diagnostics: dict = field(default_factory=dict)


def _date_components(taken_at: datetime | None):
    """Return (year, month, day) strings, or None if no usable date."""
    if taken_at is None:
        return None
    # Guard against the epoch default (1970) per spec.
    if taken_at.year < 1980:
        return None
    return (f"{taken_at.year:04d}", f"{taken_at.month:02d}", f"{taken_at.day:02d}")


class DecisionMatrix:
    """Evaluate a single :class:`ImageRecord` and produce a :class:`Decision`."""

    def __init__(self, config: CuratorConfig | None = None):
        self.cfg = config or CuratorConfig()
        self.tax: TaxonomySettings = self.cfg.taxonomy

    # ------------------------------------------------------------------ api
    def evaluate_record(self, rec: ImageRecord) -> Decision:
        # 1. Exact duplicate.
        if rec.is_duplicate:
            return Decision(
                CategoryType.DUPLICATE,
                reason="exact sha256 match",
                subcategory="duplicate",
            )

        # 2. Burst alternate (set by the clustering step).
        if rec.is_alternate:
            return Decision(
                CategoryType.UNCERTAIN,  # overridden by organizer for bursts
                reason="burst alternate",
                is_alternate=True,
            )

        # Gather child metrics.
        q = getattr(rec, "quality", None)
        sem = getattr(rec, "semantic", None)
        faces: list[FaceDetection] = getattr(rec, "faces", None) or []

        # 3. Pathological quality -> accidental junk.
        if self.cfg.quality.force_review_quality and q is not None:
            if q.is_too_dark or q.is_overexposed or q.is_blurry:
                which = []
                if q.is_too_dark:
                    which.append("too_dark")
                if q.is_overexposed:
                    which.append("overexposed")
                if q.is_blurry:
                    which.append("blurry")
                return Decision(
                    CategoryType.ACCIDENTAL,
                    reason="quality:" + "+".join(which),
                    subcategory="+".join(which),
                    confidence=float(q.quality_score),
                )

        # 4. Face routing (highest-confidence face).
        best_face = max(faces, key=lambda f: f.confidence, default=None)
        has_face = best_face is not None and best_face.confidence > \
            self.tax.face_confidence_min
        if has_face:
            return Decision(
                CategoryType.PEOPLE,
                reason=f"face conf {best_face.confidence:.2f}",
                subcategory="portraits",
                confidence=best_face.confidence,
                needs_date=True,
            )

        # 5/6. Semantic routing with confidence + margin gating.
        if sem is not None and sem.top1_category:
            top1 = sem.top1_score
            margin = sem.score_margin
            if top1 < self.tax.min_confidence or margin < self.tax.min_margin:
                return Decision(
                    CategoryType.UNCERTAIN,
                    reason=(f"low confidence {top1:.2f} "
                            f"or margin {margin:.2f}"),
                    subcategory=sem.top1_category,
                    confidence=top1,
                    diagnostics={"top1": sem.top1_category,
                                 "top1_score": top1,
                                 "top2": sem.top2_category,
                                 "margin": margin},
                )
            cat = self._map_category(sem.top1_category)
            if cat is None:
                return Decision(
                    CategoryType.UNCERTAIN,
                    reason="unmapped category",
                    subcategory=sem.top1_category,
                    confidence=top1,
                )
            needs_date = cat.is_date_grouped
            return Decision(
                cat,
                reason=f"clip top1 {sem.top1_category} {top1:.2f}",
                subcategory=sem.top1_category,
                confidence=top1,
                needs_date=needs_date,
                diagnostics={"top1": sem.top1_category, "top1_score": top1,
                             "top2": sem.top2_category, "margin": margin},
            )

        # 7. No semantic data at all.
        return Decision(
            CategoryType.UNCERTAIN,
            reason="no semantic data",
        )

    @staticmethod
    def _map_category(name: str) -> CategoryType | None:
        try:
            return CategoryType(name)
        except ValueError:
            return None

    # ---------------------------------------------------------- bulk helper
    def evaluate_many(self, records: list[ImageRecord]) -> list[Decision]:
        return [self.evaluate_record(r) for r in records]
