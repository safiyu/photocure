"""Library Insights: calculate storage footprint, space reclaimable, and curation progress."""
from __future__ import annotations

from sqlmodel import Session, func, select

from .db.models import BurstGroup, ImageRecord
from .rules.taxonomy import CategoryType

JUNK_VALUES = tuple(c.value for c in CategoryType if c.is_junk)
UTILITY_VALUES = tuple(c.value for c in CategoryType if c.is_utility)


def get_library_insights(session: Session, stage: str = "all") -> dict:
    """Compute comprehensive library metrics and storage reclamation potential."""
    I = ImageRecord

    # 1. Total storage and count
    total_count, total_bytes = session.exec(
        select(func.count(I.id), func.coalesce(func.sum(I.file_size), 0)).where(I.scan_status != "missing")
        if stage == "all" else
        select(func.count(I.id), func.coalesce(func.sum(I.file_size), 0)).where(I.scan_status != "missing", I.stage == stage)
    ).one()

    # 2. Exact Duplicates
    dup_q = select(func.count(I.id), func.coalesce(func.sum(I.file_size), 0)).where(
        I.scan_status != "missing", I.is_duplicate == True  # noqa: E712
    )
    if stage != "all":
        dup_q = dup_q.where(I.stage == stage)
    dup_count, dup_bytes = session.exec(dup_q).one()

    # 3. Burst Alternates (non-winning frames)
    alt_q = select(func.count(I.id), func.coalesce(func.sum(I.file_size), 0)).where(
        I.scan_status != "missing", I.is_alternate == True  # noqa: E712
    )
    if stage != "all":
        alt_q = alt_q.where(I.stage == stage)
    alt_count, alt_bytes = session.exec(alt_q).one()

    # 4. Junk Photos (categories + rejected rating)
    eff_cat = func.coalesce(I.manual_bucket, I.category)
    junk_q = select(func.count(I.id), func.coalesce(func.sum(I.file_size), 0)).where(
        I.scan_status != "missing",
        (eff_cat.in_(JUNK_VALUES) | (I.rating == -1)),
        I.is_duplicate == False,  # noqa: E712
        I.is_alternate == False,  # noqa: E712
    )
    if stage != "all":
        junk_q = junk_q.where(I.stage == stage)
    junk_count, junk_bytes = session.exec(junk_q).one()

    # 5. Utility Photos
    util_q = select(func.count(I.id), func.coalesce(func.sum(I.file_size), 0)).where(
        I.scan_status != "missing",
        eff_cat.in_(UTILITY_VALUES),
    )
    if stage != "all":
        util_q = util_q.where(I.stage == stage)
    util_count, util_bytes = session.exec(util_q).one()

    # 6. Marked for Delete
    marked_q = select(func.count(I.id), func.coalesce(func.sum(I.file_size), 0)).where(
        I.scan_status != "missing",
        I.delete_approved == True,  # noqa: E712
    )
    if stage != "all":
        marked_q = marked_q.where(I.stage == stage)
    marked_count, marked_bytes = session.exec(marked_q).one()

    # 7. Potential Reclaimable
    reclaimable_bytes = int(dup_bytes) + int(alt_bytes) + int(junk_bytes)
    reclaimable_pct = (reclaimable_bytes / total_bytes * 100.0) if total_bytes > 0 else 0.0

    # 8. By Stage
    stage_rows = session.exec(
        select(I.stage, func.count(I.id), func.coalesce(func.sum(I.file_size), 0))
        .where(I.scan_status != "missing")
        .group_by(I.stage)
    ).all()
    by_stage = {stg: {"count": int(c), "bytes": int(b)} for stg, c, b in stage_rows}

    # 9. By Category
    cat_q = select(eff_cat, func.count(I.id), func.coalesce(func.sum(I.file_size), 0)).where(
        I.scan_status != "missing", eff_cat.is_not(None), eff_cat != ""
    )
    if stage != "all":
        cat_q = cat_q.where(I.stage == stage)
    cat_rows = session.exec(
        cat_q.group_by(eff_cat)
        .order_by(func.count(I.id).desc())
        .limit(10)
    ).all()
    by_category = [{"name": c or "unassigned", "count": int(cnt), "bytes": int(b)} for c, cnt, b in cat_rows]

    # 10. By Camera
    cam_q = select(I.camera_model, func.count(I.id), func.coalesce(func.sum(I.file_size), 0)).where(
        I.scan_status != "missing", I.camera_model != ""
    )
    if stage != "all":
        cam_q = cam_q.where(I.stage == stage)
    cam_rows = session.exec(
        cam_q.group_by(I.camera_model)
        .order_by(func.count(I.id).desc())
        .limit(8)
    ).all()
    by_camera = [{"name": c, "count": int(cnt), "bytes": int(b)} for c, cnt, b in cam_rows]

    # 11. Date Sources
    date_q = select(I.taken_at_source, func.count(I.id)).where(I.scan_status != "missing")
    if stage != "all":
        date_q = date_q.where(I.stage == stage)
    date_rows = session.exec(
        date_q.group_by(I.taken_at_source)
    ).all()
    date_sources = {src or "unknown": int(cnt) for src, cnt in date_rows}

    # 12. Bursts Summary
    tb_q = select(func.count(func.distinct(I.burst_id))).where(I.scan_status != "missing", I.burst_id.is_not(None))
    bf_q = select(func.count(I.id)).where(I.scan_status != "missing", I.burst_id.is_not(None))
    if stage != "all":
        tb_q = tb_q.where(I.stage == stage)
        bf_q = bf_q.where(I.stage == stage)
    total_bursts = session.exec(tb_q).one()
    burst_frames = session.exec(bf_q).one()
    avg_burst_len = (burst_frames / total_bursts) if total_bursts > 0 else 0.0

    return {
        "total_photos": int(total_count),
        "total_bytes": int(total_bytes),
        "reclaimable_bytes": reclaimable_bytes,
        "reclaimable_percent": round(reclaimable_pct, 1),
        "duplicates": {"count": int(dup_count), "bytes": int(dup_bytes)},
        "burst_alternates": {"count": int(alt_count), "bytes": int(alt_bytes)},
        "junk": {"count": int(junk_count), "bytes": int(junk_bytes)},
        "utility": {"count": int(util_count), "bytes": int(util_bytes)},
        "marked_for_delete": {"count": int(marked_count), "bytes": int(marked_bytes)},
        "by_stage": by_stage,
        "by_category": by_category,
        "by_camera": by_camera,
        "date_sources": date_sources,
        "bursts": {
            "total_bursts": int(total_bursts),
            "total_frames": int(burst_frames),
            "avg_frames": round(avg_burst_len, 1),
            "keepers": int(total_bursts),
        },
    }
