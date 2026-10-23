"""Duplicate review: find groups of exact and near duplicates, suggest a keeper, record the choice.

An exact group is files with the same SHA-256. A near group is photos linked by perceptual hash
(``near_dup_of``) and confirmed by their CLIP embeddings when both have one. Burst frames are left
to the burst review. Choosing a keeper marks the others with a manual duplicate bucket, so the
routing rules and any later plan treat them as duplicates without further steps.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import selectinload
from sqlmodel import col, select

from .db.models import ImageRecord
from .db.repository import dequantize_embedding

DUP_BUCKETS = ("duplicate", "near_duplicate")
_BUCKET = {"exact": "duplicate", "near": "near_duplicate"}
DEFAULT_NEAR_COSINE = 0.93


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _unit(rec) -> np.ndarray | None:
    sem = rec.semantic
    if sem is None or sem.backend == "stub":
        return None
    v = dequantize_embedding(sem.embedding)
    if not v:
        return None
    a = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-9 else None


def _member(rec) -> dict:
    sem, qm = rec.semantic, rec.quality
    return {"id": rec.id, "filename": rec.filename, "path": rec.path,
            "width": rec.width or 0, "height": rec.height or 0, "file_size": rec.file_size or 0,
            "rating": rec.rating, "favorite": bool(rec.is_favorite),
            "quality_score": qm.quality_score if qm else None,
            "aesthetic": sem.aesthetic if sem else None,
            "taken_at": rec.taken_at.isoformat() if rec.taken_at else None,
            "manual_bucket": rec.manual_bucket, "similarity": None, "suggested": False}


def _keep_key(m: dict) -> tuple:
    """Higher is better: not rejected, favourite, stars, pixels, quality, appeal, size, oldest."""
    return (m["rating"] != -1, bool(m["favorite"]), max(m["rating"] or 0, 0),
            m["width"] * m["height"], m["quality_score"] or 0.0, m["aesthetic"] or 0.0,
            m["file_size"], -m["id"])


def suggest_keeper(members: list[dict]) -> int:
    return max(members, key=_keep_key)["id"]


def _undecided(recs) -> list:
    """Members the user has not already placed elsewhere by hand."""
    return [r for r in recs if r.manual_bucket in (None, *DUP_BUCKETS)]


def _build(kind: str, group_id: str, recs: list, embs: dict[int, np.ndarray | None]) -> dict | None:
    recs = _undecided(recs)
    if len(recs) < 2:
        return None
    members = {r.id: _member(r) for r in recs}
    open_ids = [r.id for r in recs if r.manual_bucket not in DUP_BUCKETS]
    reviewed = len(open_ids) <= 1
    if reviewed and not open_ids:
        return None
    keeper_id = open_ids[0] if reviewed else suggest_keeper(list(members.values()))
    ordered = [members[keeper_id]] + sorted(
        (m for i, m in members.items() if i != keeper_id), key=_keep_key, reverse=True)
    base = embs.get(keeper_id)
    for m in ordered:
        e = embs.get(m["id"])
        m["similarity"] = (1.0 if m["id"] == keeper_id else
                           (round(float(base @ e), 4) if base is not None and e is not None else None))
    ordered[0]["suggested"] = True
    return {"id": group_id, "kind": kind, "keeper_id": keeper_id, "members": ordered,
            "reclaimable_bytes": sum(m["file_size"] for m in ordered[1:]), "reviewed": reviewed}


def _exact_groups(session) -> list[dict]:
    I = ImageRecord
    shared = (select(I.sha256).where(I.sha256 != "", I.scan_status != "missing")
              .group_by(I.sha256).having(func.count() > 1))
    recs = session.exec(select(I).where(col(I.sha256).in_(shared), I.scan_status != "missing")
                        .options(selectinload(I.semantic), selectinload(I.quality))
                        .order_by(I.id)).all()
    by_sha: dict[str, list] = {}
    for r in recs:
        by_sha.setdefault(r.sha256, []).append(r)
    out = []
    for sha, rows in by_sha.items():
        g = _build("exact", f"exact:{sha[:12]}", rows, {r.id: _unit(r) for r in rows})
        if g:
            out.append(g)
    return out


def _near_groups(session, min_cosine: float) -> list[dict]:
    I = ImageRecord
    linked = session.exec(
        select(I).where(col(I.near_dup_of).is_not(None), I.is_duplicate == False,  # noqa: E712
                        col(I.burst_id).is_(None), I.scan_status != "missing")
        .options(selectinload(I.semantic), selectinload(I.quality))).all()
    if not linked:
        return []
    by_id = {r.id: r for r in linked}
    wanted = {r.near_dup_of for r in linked} - set(by_id)
    if wanted:
        for r in session.exec(select(I).where(col(I.id).in_(wanted), I.is_duplicate == False,  # noqa: E712
                                              col(I.burst_id).is_(None), I.scan_status != "missing")
                              .options(selectinload(I.semantic), selectinload(I.quality))).all():
            by_id[r.id] = r
    embs = {i: _unit(r) for i, r in by_id.items()}
    parent = {i: i for i in by_id}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in linked:
        t = r.near_dup_of
        if t not in by_id:
            continue
        a, b = embs.get(r.id), embs.get(t)
        if a is not None and b is not None and float(a @ b) < min_cosine:
            continue  # the planner would drop this link too
        parent[find(r.id)] = find(t)
    comps: dict[int, list] = {}
    for i, r in by_id.items():
        comps.setdefault(find(i), []).append(r)
    out = []
    for rows in comps.values():
        rows.sort(key=lambda r: r.id)
        g = _build("near", f"near:{rows[0].id}", rows, embs)
        if g:
            out.append(g)
    return out


def find_groups(session, kind: str | None = None, include_reviewed: bool = False,
                near_min_cosine: float = DEFAULT_NEAR_COSINE) -> list[dict]:
    """Duplicate groups, biggest saving first. Reviewed groups are hidden unless asked for."""
    if kind not in (None, "exact", "near"):
        raise ValueError(f"unknown kind {kind!r}")
    groups = []
    if kind in (None, "exact"):
        groups += _exact_groups(session)
    if kind in (None, "near"):
        groups += _near_groups(session, near_min_cosine)
    if not include_reviewed:
        groups = [g for g in groups if not g["reviewed"]]
    groups.sort(key=lambda g: (-g["reclaimable_bytes"], g["id"]))
    return groups


def summarize(groups: list[dict]) -> dict:
    """Counts over groups already found, so a caller that has them need not search again."""
    return {"groups": len(groups),
            "exact_groups": sum(g["kind"] == "exact" for g in groups),
            "near_groups": sum(g["kind"] == "near" for g in groups),
            "images": sum(len(g["members"]) for g in groups),
            "reclaimable_bytes": sum(g["reclaimable_bytes"] for g in groups)}


def summary(session, near_min_cosine: float = DEFAULT_NEAR_COSINE) -> dict:
    return summarize(find_groups(session, near_min_cosine=near_min_cosine))


def resolve_group(session, keep_id: int, discard_ids: list[int], kind: str) -> dict:
    """Keep one photo and mark the others as duplicates of it. The caller owns the commit."""
    if kind not in _BUCKET:
        raise ValueError(f"unknown kind {kind!r}")
    discard_ids = list(dict.fromkeys(discard_ids))
    if not discard_ids:
        raise ValueError("nothing to discard")
    if keep_id in discard_ids:
        raise ValueError("the photo to keep cannot also be discarded")
    I = ImageRecord
    rows = {r.id: r for r in session.exec(select(I).where(col(I.id).in_([keep_id, *discard_ids]))).all()}
    missing = [i for i in (keep_id, *discard_ids) if i not in rows]
    if missing:
        raise LookupError(f"unknown photo {missing[0]}")
    keeper = rows[keep_id]
    now = _now()
    link = "duplicate_of" if kind == "exact" else "near_dup_of"
    # Anything that pointed at a photo being discarded now points at the keeper.
    for dep in session.exec(select(I).where(col(getattr(I, link)).in_(discard_ids))).all():
        if dep.id not in rows:
            setattr(dep, link, keep_id)
            dep.updated_at = now
    for i in discard_ids:
        r = rows[i]
        if kind == "exact":
            r.is_duplicate, r.duplicate_of = True, keep_id
        else:
            r.near_dup_of = keep_id
        r.manual_bucket = _BUCKET[kind]
        r.updated_at = now
    if kind == "exact":
        keeper.is_duplicate, keeper.duplicate_of = False, None
    else:
        keeper.near_dup_of = None
    if keeper.manual_bucket in DUP_BUCKETS:
        keeper.manual_bucket = None
    keeper.delete_approved = False
    keeper.updated_at = now
    session.flush()
    return {"kept": keep_id, "discarded": discard_ids,
            "reclaimable_bytes": sum(rows[i].file_size or 0 for i in discard_ids)}


def dismiss_near(session, ids: list[int]) -> int:
    """Say these photos are not duplicates: cut their near-duplicate links. Returns links cut."""
    I = ImageRecord
    cut = 0
    for r in session.exec(select(I).where(col(I.id).in_(ids), col(I.near_dup_of).is_not(None))).all():
        r.near_dup_of = None
        if r.manual_bucket == "near_duplicate":
            r.manual_bucket = None
        r.updated_at = _now()
        cut += 1
    session.flush()
    return cut


def resolve_all(session, near_min_cosine: float = DEFAULT_NEAR_COSINE) -> dict:
    """Apply the suggested keeper to every open group."""
    done = discarded = saved = 0
    for g in find_groups(session, near_min_cosine=near_min_cosine):
        others = [m["id"] for m in g["members"] if m["id"] != g["keeper_id"]]
        out = resolve_group(session, g["keeper_id"], others, g["kind"])
        done += 1
        discarded += len(out["discarded"])
        saved += out["reclaimable_bytes"]
    return {"groups": done, "discarded": discarded, "reclaimable_bytes": saved}
