"""Best-shot selector for burst groups.

Composite score (spec section 2.D)::

    Score = w_blur * S_laplacian
          + w_face_sharp * S_face
          + w_eye * S_eye_open
          + w_res * log(width * height)

Each component is normalized to ``[0, 1]``. Components that are unknown for a
candidate (eye openness without 106 landmarks, aesthetic without an embedding)
are left out and the remaining weights are renormalized, so an unknown value
is never scored as zero. The highest-scoring photo is kept as the primary
shot; the rest are alternates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from collections.abc import Sequence

import numpy as np

from ..config import BurstThresholds, CuratorConfig
from ..db.models import FaceDetection


@dataclass
class ShotCandidate:
    """Normalized view of one shot used by the scoring engine."""

    image_id: int
    # 0..1 Laplacian sharpness (from quality.sharpness_norm).
    sharpness_norm: float = 0.0
    # 0..1 normalized face-crop sharpness (0 if no face).
    face_sharpness_norm: float = 0.0
    # Canonical EAR of the face, or None when unknown (no 106 landmarks).
    eye_open: float | None = None
    # 0..1 aesthetic score from the CLIP embedding, None when not computed.
    aesthetic: float | None = None
    width: int = 0
    height: int = 0
    has_face: bool = False
    # Tie-breakers / diagnostics.
    taken_ts: float = 0.0
    confidence: float = 0.0
    breakdown: dict = field(default_factory=dict)

    @property
    def log_pixels(self) -> float:
        if self.width <= 0 or self.height <= 0:
            return 0.0
        return float(math.log(max(1, self.width * self.height)))


# Reference Laplacian variance used to normalize a face-crop sharpness value.
_FACE_SHARP_REF = 300.0
# Resolution normalization anchors (log of megapixel-ish pixel counts).
_RES_MIN = math.log(1024 * 768)
_RES_MAX = math.log(4000 * 3000)


def _normalize_face_sharpness(v: float | None) -> float:
    if v is None:
        return 0.0
    return float(np.clip(v / _FACE_SHARP_REF, 0.0, 1.0))


# Default weight of the aesthetic term when the config has none.
DEFAULT_AESTHETIC_WEIGHT = 0.10


def _eye_open_score(ear: float | None, t: BurstThresholds) -> float | None:
    """Map a canonical EAR to an openness score in [0, 1]; None if unknown."""
    if ear is None:
        return None
    if t.ear_open <= t.ear_closed:
        return 1.0
    return float(np.clip((ear - t.ear_closed) / (t.ear_open - t.ear_closed), 0.0, 1.0))


def _resolution_score(width: int, height: int) -> float:
    if width <= 0 or height <= 0:
        return 0.0
    if _RES_MAX <= _RES_MIN:
        return 0.5
    return float(np.clip((math.log(width * height) - _RES_MIN) /
                         (_RES_MAX - _RES_MIN), 0.0, 1.0))


class BestShotSelector:
    """Rank burst candidates and pick the best shot."""

    def __init__(self, config: CuratorConfig | None = None,
                 thresholds: BurstThresholds | None = None):
        self.cfg = config or CuratorConfig()
        self.t = thresholds or self.cfg.burst

    def score(self, c: ShotCandidate) -> tuple[float, dict]:
        """Return ``(composite_score, breakdown)`` for a single candidate."""
        w = self.t
        s_blur = float(np.clip(c.sharpness_norm, 0.0, 1.0))
        s_face = float(np.clip(c.face_sharpness_norm, 0.0, 1.0)) if c.has_face else 0.0
        s_eye = _eye_open_score(c.eye_open, w) if c.has_face else None
        s_res = _resolution_score(c.width, c.height)
        s_aes = None if c.aesthetic is None else float(np.clip(c.aesthetic, 0.0, 1.0))

        terms = [(w.weight_blur, s_blur), (w.weight_face_sharp, s_face),
                 (w.weight_resolution, s_res)]
        if s_eye is not None:
            terms.append((w.weight_eye, s_eye))
        if s_aes is not None:
            terms.append((getattr(w, "weight_aesthetic", DEFAULT_AESTHETIC_WEIGHT), s_aes))
        total_w = sum(wt for wt, _ in terms)
        composite = sum(wt * v for wt, v in terms) / total_w if total_w > 0 else 0.0
        breakdown = {
            "blur": s_blur, "face": s_face,
            "eye": s_eye if s_eye is not None else 0.0,
            "eye_known": s_eye is not None,
            "aesthetic": s_aes if s_aes is not None else 0.0,
            "resolution": s_res, "composite": float(composite),
        }
        return float(composite), breakdown

    def pick_best_shot(self, candidates: Sequence[ShotCandidate]) -> ShotCandidate | None:
        """Return the best candidate (or None for an empty list)."""
        if not candidates:
            return None
        best: ShotCandidate | None = None
        best_key = None
        for c in candidates:
            _, br = self.score(c)
            c.breakdown = br
            # Key: composite, then face presence, then eye, then sharpness,
            # then resolution, then earliest (stable & deterministic).
            key = (
                br["composite"],
                1.0 if c.has_face else 0.0,
                br["eye"],
                br["blur"],
                br["resolution"],
                -c.taken_ts,
            )
            if best_key is None or key > best_key:
                best, best_key = c, key
        return best

    def rank(self, candidates: Sequence[ShotCandidate]) -> list[ShotCandidate]:
        """Return candidates sorted best-first."""
        for c in candidates:
            _, br = self.score(c)
            c.breakdown = br
        return sorted(
            candidates,
            key=lambda c: (
                c.breakdown.get("composite", 0.0),
                1.0 if c.has_face else 0.0,
                c.breakdown.get("eye", 0.0),
                c.breakdown.get("blur", 0.0),
                c.breakdown.get("resolution", 0.0),
                -c.taken_ts,
            ),
            reverse=True,
        )


def _candidate_from_record(rec, face_min: float = 0.65) -> ShotCandidate:
    """Build a :class:`ShotCandidate` from an :class:`ImageRecord` row.

    ``face_min`` is the same face-confidence gate the decision matrix uses.
    """
    from datetime import timezone
    q = getattr(rec, "quality", None)
    sharp = q.sharpness_norm if q is not None else 0.0
    faces: list[FaceDetection] = getattr(rec, "faces", None) or []
    # Pick the highest-confidence face for sharpness/EAR.
    best_face = None
    best_conf = -1.0
    for f in faces:
        if f.confidence > best_conf:
            best_face, best_conf = f, f.confidence
    has_face = best_face is not None and best_face.confidence > face_min
    sem = getattr(rec, "semantic", None)
    aesthetic = getattr(sem, "aesthetic", None) if sem is not None else None
    ts = 0.0
    if rec.taken_at is not None:
        if rec.taken_at.tzinfo is None:
            rec.taken_at = rec.taken_at.replace(tzinfo=timezone.utc)
        ts = rec.taken_at.timestamp()
    return ShotCandidate(
        image_id=rec.id,
        sharpness_norm=sharp,
        face_sharpness_norm=_normalize_face_sharpness(
            best_face.face_sharpness) if has_face else 0.0,
        eye_open=(best_face.eye_aspect_ratio if has_face else None),
        width=rec.width,
        height=rec.height,
        has_face=has_face,
        taken_ts=ts,
        confidence=best_conf if has_face else 0.0,
        aesthetic=aesthetic,
    )


def pick_best_shot(candidates: Sequence[ShotCandidate],
                   config: CuratorConfig | None = None) -> ShotCandidate | None:
    return BestShotSelector(config).pick_best_shot(candidates)


def pick_best_shot_records(records: Sequence,
                           config: CuratorConfig | None = None) -> int | None:
    """Return the image_id of the best shot among DB ``records``."""
    cfg = config or CuratorConfig()
    sel = BestShotSelector(cfg)
    face_min = cfg.taxonomy.face_confidence_min
    cands = [_candidate_from_record(r, face_min) for r in records]
    best = sel.pick_best_shot(cands)
    return best.image_id if best is not None else None
