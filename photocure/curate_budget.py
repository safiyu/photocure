"""Target Keeper Budget: cull to N photos or top X% per event or stage.

Leverages aesthetic scores, burst winners, and narrative scale balancing
to keep a diverse, appealing cut of keepers and stage the rest for review or deletion.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from sqlmodel import Session, col, select

from .clustering.events import UNRELIABLE_TIME_SOURCES
from .config import CuratorConfig
from .db.models import ImageRecord, SemanticScore
from .highlights import HighlightWeights, highlight_score
from .narrative import SCALES, balance_selection, scale_labels

log = logging.getLogger("photocure.curate_budget")


@dataclass
class CurateCandidate:
    id: int
    filename: str
    file_size: int
    aesthetic: float | None
    preference: float | None
    rating: int | None
    is_favorite: bool
    locked: bool
    is_best_shot: bool
    is_alternate: bool
    burst_id: int | None
    event_id: int | None
    scale: str = "medium"
    final_score: float = 0.5
    guaranteed_keeper: bool = False


def budget_curate(
    session: Session,
    cfg: CuratorConfig | None = None,
    target_count: int | None = None,
    target_percent: float | None = None,
    event_id: int | None = None,
    stage: str = "new",
    action: Literal["preview", "stage_delete", "star_keepers"] = "preview",
) -> dict:
    """Calculate and optionally apply a target keeper budget."""
    cfg = cfg or CuratorConfig()
    weights = HighlightWeights()

    # 1. Fetch eligible candidates in the given stage/scope
    q = select(ImageRecord).where(ImageRecord.scan_status != "missing")
    if stage and stage != "all":
        q = q.where(ImageRecord.stage == stage)
    if event_id is not None:
        q = q.where(ImageRecord.event_id == event_id)
    # Exclude exact duplicates and existing junk
    q = q.where(ImageRecord.is_duplicate == False)  # noqa: E712

    records = session.exec(q).all()
    if not records:
        return {
            "ok": True,
            "total": 0,
            "budget": 0,
            "kept_count": 0,
            "culled_count": 0,
            "kept_ids": [],
            "culled_ids": [],
            "scale_coverage": {},
            "reclaimable_bytes": 0,
            "action_applied": action,
        }

    rec_ids = [r.id for r in records if r.id is not None]

    # 2. Fetch semantic scores for candidates
    scores_stmt = select(SemanticScore).where(col(SemanticScore.image_id).in_(rec_ids))
    scores_by_id = {s.image_id: s for s in session.exec(scores_stmt).all()}

    # 3. Fetch scale labels for candidates if available
    scales_by_id: dict[int, str] = {}
    try:
        from .search import make_text_encoder
        encoder = make_text_encoder(cfg.backend)
        scales = scale_labels(session, rec_ids, encoder)
        if scales:
            scales_by_id = scales
    except Exception as e:
        log.debug("Scale classification unavailable for budget curation: %s", e)

    # 4. Build candidate objects and compute ranking scores
    candidates: list[CurateCandidate] = []
    for r in records:
        sem = scores_by_id.get(r.id)
        aes = sem.aesthetic if sem else None
        pref = sem.preference if sem else None

        # Favorite, locked, or 4-5 stars are guaranteed keepers
        guaranteed = bool(r.is_favorite or r.locked or (r.rating and r.rating >= 4))

        # Base ranking score
        score = highlight_score(aes, pref, r.rating, r.is_favorite, weights)
        if score is None:
            score = 0.5 if aes is None else float(aes)

        # Burst logic: best shot gets a priority boost; alternates take a culling penalty
        if r.is_best_shot:
            score += 0.20
        elif r.is_alternate:
            score -= 0.15

        candidates.append(CurateCandidate(
            id=r.id,
            filename=r.filename,
            file_size=r.file_size or 0,
            aesthetic=aes,
            preference=pref,
            rating=r.rating,
            is_favorite=r.is_favorite,
            locked=r.locked,
            is_best_shot=r.is_best_shot,
            is_alternate=r.is_alternate,
            burst_id=r.burst_id,
            event_id=r.event_id,
            scale=scales_by_id.get(r.id, "medium"),
            final_score=float(score),
            guaranteed_keeper=guaranteed,
        ))

    total = len(candidates)

    # 5. Compute budget N
    if target_percent is not None:
        pct = max(0.0, min(100.0, float(target_percent)))
        budget_n = max(1, round(total * pct / 100.0))
    elif target_count is not None:
        budget_n = max(1, min(total, int(target_count)))
    else:
        # Default: keep top 25%
        budget_n = max(1, round(total * 0.25))

    # 6. Partition into guaranteed keepers vs rest
    guaranteed_list = [c for c in candidates if c.guaranteed_keeper]
    pool = [c for c in candidates if not c.guaranteed_keeper]

    # Convert pool to items for balance_selection: (id, score, scale)
    pool_items = [(c.id, c.final_score, c.scale) for c in pool]
    slots_needed = max(0, budget_n - len(guaranteed_list))

    chosen_from_pool_items = balance_selection(pool_items, slots_needed)
    chosen_pool_ids = {it[0] for it in chosen_from_pool_items}

    kept_candidates = guaranteed_list + [c for c in pool if c.id in chosen_pool_ids]
    kept_ids = {c.id for c in kept_candidates}

    culled_candidates = [c for c in candidates if c.id not in kept_ids]
    culled_ids = [c.id for c in culled_candidates]

    reclaimable_bytes = sum(c.file_size for c in culled_candidates)

    # Scale coverage of kept photos
    scale_coverage: dict[str, int] = {s: 0 for s in SCALES}
    for c in kept_candidates:
        s = c.scale if c.scale in scale_coverage else "medium"
        scale_coverage[s] += 1
    scale_coverage = {k: v for k, v in scale_coverage.items() if v > 0}

    # 7. Apply action to DB if requested
    if action == "stage_delete" and culled_ids:
        # Mark culled photos for delete (delete_approved = True), unmark keepers
        for r in records:
            if r.id in kept_ids and r.delete_approved:
                r.delete_approved = False
            elif r.id in culled_ids and not r.delete_approved:
                r.delete_approved = True
        session.commit()
    elif action == "star_keepers" and kept_ids:
        for r in records:
            if r.id in kept_ids:
                if not r.rating or r.rating < 4:
                    r.rating = 5
        session.commit()

    return {
        "ok": True,
        "total": total,
        "budget": budget_n,
        "kept_count": len(kept_ids),
        "culled_count": len(culled_ids),
        "kept_ids": sorted(list(kept_ids)),
        "culled_ids": sorted(culled_ids),
        "scale_coverage": scale_coverage,
        "reclaimable_bytes": reclaimable_bytes,
        "action_applied": action,
    }

