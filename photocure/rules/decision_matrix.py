"""Decision matrix: deterministic routing from image features to a bucket.

Implements spec section 4.B (and the milestone-4 routing rules). Precedence:

    1. Exact SHA-256 duplicate            -> Junk_Review/Duplicates/
    2. Decode failed                      -> Junk_Review/Unreadable/
    3. Burst alternate (non-best)         -> Bursts/<burst>/alternates/
    4. Perceptual near-duplicate          -> Junk_Review/Near_Duplicates/
       (never for burst members, the burst logic owns those)
    5. Pathological quality (dark/blurry) -> Junk_Review/Accidental/
       (skipped when CLIP is confident the frame is screen content, since
       flat dark UI screenshots and documents are not accidents; a confident
       face also protects a photo from the blurry and overexposed verdicts, so
       a soft photo of a person is flagged for review, not junked)
    6. Human face (conf > 0.65)           -> People/
    7. Low-confidence / close-call CLIP   -> Uncertain_Review/
       (confidence is judged per group of look-alike classes, see taxonomy)
    8. Top CLIP category                  -> Screenshots / Documents / Food /
                                             Animals / Scenery / custom classes
    9. Fallback                           -> Uncertain_Review/
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

from ..config import CuratorConfig, TaxonomySettings
from ..db.models import FaceDetection, ImageRecord
from ..rules.taxonomy import CategoryType, Taxonomy, get_taxonomy


@dataclass
class Decision:
    category: CategoryType
    reason: str
    subcategory: str = ""
    confidence: float = 0.0
    needs_date: bool = False
    # When True the image is a burst alternate (routed to the burst folder).
    is_alternate: bool = False
    # Relative folder override (custom YAML classes); None uses category.folder.
    folder: str | None = None
    diagnostics: dict = field(default_factory=dict)


SCREEN_GROUP = "screen"

# Confident classes that win over a detected face: avatars and thumbnails inside screenshots,
# graphics and documents are not people photos. Slides, whiteboards and labels are photos of
# real places, so a presenter's face still routes to People.
FACE_OVERRIDDEN_BY = frozenset({"screenshot", "document", "graphic", "illustration"})


def _label(name: str) -> str:
    t = str(name or "").replace("_", " ").strip()
    return t[:1].upper() + t[1:]


def _pct(x: float) -> int:
    return int(round(float(x) * 100))


def _parse_distribution(sem) -> dict[str, float]:
    raw = getattr(sem, "distribution", None)
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items()}
    try:
        data = json.loads(raw or "{}")
        return {str(k): float(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except (TypeError, ValueError):
        return {}


def assess_semantics(sem, tax: Taxonomy, tcfg: TaxonomySettings) -> dict:
    """Judge a semantic result per group of look-alike classes.

    Confident means: the top class clears ``min_class_floor``, its group's
    summed probability clears ``min_confidence``, and it beats the best class
    outside the group (or the runner up when that is in another group) by
    ``min_margin``. Without a stored distribution the old top1/margin test runs.
    """
    min_conf = float(getattr(tcfg, "min_confidence", 0.40))
    min_margin = float(getattr(tcfg, "min_margin", 0.08))
    floor = float(getattr(tcfg, "min_class_floor", 0.25))
    base = {"kind": "no_data", "top1": "", "top1_score": 0.0, "top2": "",
            "top2_score": 0.0, "group": "", "group_score": 0.0, "margin": 0.0,
            "confident": False, "text": "Not analyzed yet. Run Analyze first."}
    if sem is None or not getattr(sem, "top1_category", ""):
        return base

    top1 = sem.top1_category
    p1 = float(sem.top1_score)
    top2 = getattr(sem, "top2_category", "") or ""
    p2 = float(getattr(sem, "top2_score", 0.0) or 0.0)
    dist = _parse_distribution(sem)
    out = dict(base, top1=top1, top1_score=p1, top2=top2, top2_score=p2)

    if not dist:
        margin = float(sem.score_margin)
        confident = p1 >= min_conf and margin >= min_margin
        out.update(group=tax.group_of(top1), group_score=p1, margin=margin,
                   confident=confident)
        if confident:
            out.update(kind="ok", text=f"{_label(top1)} at {_pct(p1)}%.")
        elif p1 < min_conf:
            out.update(kind="low_confidence", text=(
                f"Best guess is {_label(top1)} at {_pct(p1)}%, "
                f"below the {_pct(min_conf)}% needed."))
        else:
            out.update(kind="close_call", text=(
                f"Close call between {_label(top1)} ({_pct(p1)}%) and "
                f"{_label(top2)} ({_pct(p2)}%)."))
        return out

    group = tax.group_of(top1)
    group_score = sum(v for n, v in dist.items() if tax.group_of(n) == group)
    outside = [(v, n) for n, v in dist.items() if tax.group_of(n) != group]
    other_p, other_n = max(outside, default=(0.0, ""))
    same_group_runner = bool(top2) and tax.group_of(top2) == group
    if same_group_runner:
        margin, rival, rival_p = p1 - other_p, other_n, other_p
    else:
        margin, rival, rival_p = p1 - p2, top2, p2
    confident = p1 >= floor and group_score >= min_conf and margin >= min_margin
    out.update(group=group, group_score=group_score, margin=margin, confident=confident)
    if confident:
        out.update(kind="ok", text=f"{_label(top1)} at {_pct(p1)}%.")
    elif group_score < min_conf or p1 < floor:
        shown = group_score if group_score < min_conf else p1
        needed = min_conf if group_score < min_conf else floor
        out.update(kind="low_confidence", text=(
            f"Best guess is {_label(top1)} at {_pct(shown)}%, "
            f"below the {_pct(needed)}% needed."))
    else:
        out.update(kind="close_call", text=(
            f"Close call between {_label(top1)} ({_pct(p1)}%) and "
            f"{_label(rival)} ({_pct(rival_p)}%)."))
    return out


def explain_uncertain(sem, tax: Taxonomy | None = None, cfg: CuratorConfig | None = None) -> dict:
    """Plain reason a photo is (or is not) uncertain, for the review screens."""
    cfg = cfg or CuratorConfig()
    tax = tax or get_taxonomy(cfg)
    a = assess_semantics(sem, tax, cfg.taxonomy)
    return {k: a[k] for k in ("kind", "top1", "top1_score", "top2", "top2_score", "text")}


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
        self.taxonomy: Taxonomy = get_taxonomy(self.cfg)

    # ------------------------------------------------------------------ api
    def evaluate_record(self, rec: ImageRecord) -> Decision:
        # 1. Exact duplicate.
        if rec.is_duplicate:
            return Decision(
                CategoryType.DUPLICATE,
                reason="exact sha256 match",
                subcategory="duplicate",
            )

        # 2. Decode failure: surface it instead of silently dropping the file.
        if rec.infer_status == "failed":
            return Decision(
                CategoryType.UNREADABLE,
                reason="decode failed",
                subcategory="unreadable",
            )

        # 3. Burst alternate (set by the clustering step).
        if rec.is_alternate:
            return Decision(
                CategoryType.BURST_ALTERNATE,
                reason="burst alternate",
                is_alternate=True,
            )

        # 4. Perceptual near-duplicate (burst members are handled by bursts).
        if getattr(rec, "near_dup_of", None) is not None and rec.burst_id is None:
            return Decision(
                CategoryType.NEAR_DUPLICATE,
                reason=f"near-duplicate of image {rec.near_dup_of}",
                subcategory="near_duplicate",
            )

        # Gather child metrics.
        q = getattr(rec, "quality", None)
        sem = getattr(rec, "semantic", None)
        faces: list[FaceDetection] = [f for f in (getattr(rec, "faces", None) or []) if not getattr(f, "ignored", False)]

        # Judge semantics first so the quality gate can spare screen content.
        assessment = (assess_semantics(sem, self.taxonomy, self.tax)
                      if sem is not None and sem.top1_category else None)
        confident_screen = bool(assessment and assessment["confident"]
                                and assessment["group"] == SCREEN_GROUP)
        confident_flat = bool(confident_screen and assessment["top1"] in FACE_OVERRIDDEN_BY)

        # Face routing needs the best face before the quality gate so it can protect people.
        best_face = max(faces, key=lambda f: f.confidence, default=None)
        has_face = best_face is not None and best_face.confidence > \
            self.tax.face_confidence_min

        # 5. Pathological quality -> accidental junk (not for screen content).
        if (self.cfg.quality.force_review_quality and q is not None
                and not confident_screen):
            protected = has_face and not confident_flat
            bad_dark = q.is_too_dark
            bad_over = q.is_overexposed and not protected
            bad_blur = q.is_blurry and not protected
            if bad_dark or bad_over or bad_blur:
                which = []
                if bad_dark:
                    which.append("too_dark")
                if bad_over:
                    which.append("overexposed")
                if bad_blur:
                    which.append("blurry")
                return Decision(
                    CategoryType.ACCIDENTAL,
                    reason="quality:" + "+".join(which),
                    subcategory="+".join(which),
                    confidence=float(q.quality_score),
                )

        # 6. Face routing (highest-confidence face).
        # A confident screenshot can contain avatars or thumbnails; the face must not win.
        if has_face and not confident_flat:
            return Decision(
                CategoryType.PEOPLE,
                reason=f"face conf {best_face.confidence:.2f}",
                subcategory="portraits",
                confidence=best_face.confidence,
                needs_date=True,
            )

        # 7. Semantic routing with group-aware confidence gating.
        if assessment is not None:
            top1 = sem.top1_score
            margin = assessment["margin"]
            if not assessment["confident"]:
                return Decision(
                    CategoryType.UNCERTAIN,
                    reason=(f"low confidence {top1:.2f} "
                            f"or margin {margin:.2f}"),
                    subcategory=sem.top1_category,
                    confidence=top1,
                    diagnostics={
                        **{k: assessment[k] for k in (
                            "kind", "top1", "top1_score", "top2", "top2_score", "text")},
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
            needs_date = self.taxonomy.is_date_grouped(sem.top1_category)
            return Decision(
                cat,
                reason=f"clip top1 {sem.top1_category} {top1:.2f}",
                subcategory=sem.top1_category,
                confidence=top1,
                needs_date=needs_date,
                folder=self.taxonomy.folder_override_of(sem.top1_category),
                diagnostics={"top1": sem.top1_category, "top1_score": top1,
                             "top2": sem.top2_category, "margin": margin,
                             "group": assessment["group"],
                             "group_score": assessment["group_score"]},
            )

        # 8. No semantic data at all.
        return Decision(
            CategoryType.UNCERTAIN,
            reason="no semantic data",
            diagnostics=explain_uncertain(None, self.taxonomy, self.cfg),
        )

    def _map_category(self, name: str) -> CategoryType | None:
        # Rule-assigned buckets must never come from the model.
        cat = self.taxonomy.category_of(name)
        if cat is None or cat not in self.taxonomy.categories:
            return None
        return cat

    # ---------------------------------------------------------- bulk helper
    def evaluate_many(self, records: list[ImageRecord]) -> list[Decision]:
        return [self.evaluate_record(r) for r in records]
