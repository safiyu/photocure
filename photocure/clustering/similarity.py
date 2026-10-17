"""Embedding similarity search: brute-force cosine over stored CLIP embeddings."""
from __future__ import annotations

import numpy as np
from sqlmodel import col, select as sselect

from ..db.models import ImageRecord, SemanticScore
from ..db.repository import dequantize_embedding

_CHUNK = 20000


def _unit(v: list[float] | np.ndarray) -> np.ndarray | None:
    a = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-9 else None


def top_k(session, q: np.ndarray, k: int, *, exclude_id: int | None = None,
          filters: tuple = (), join_images: bool = False,
          min_score: float | None = None) -> list[tuple[int, float]]:
    """Top-``k`` stored embeddings by cosine to the unit vector ``q`` as ``(image_id, cosine)``.

    Scans in keyset-paginated chunks so memory stays bounded for large libraries.
    ``filters`` are extra SQL conditions; ``join_images`` joins the image row so they can use it.
    """
    if k <= 0:
        return []
    best_ids = np.empty(0, dtype=np.int64)
    best_sims = np.empty(0, dtype=np.float32)
    last = 0
    while True:
        stmt = sselect(SemanticScore.image_id, SemanticScore.embedding)
        if join_images:
            stmt = stmt.join(ImageRecord, ImageRecord.id == SemanticScore.image_id)
        for cond in filters:
            stmt = stmt.where(cond)
        rows = session.exec(
            stmt.where(col(SemanticScore.image_id) > last)
            .order_by(col(SemanticScore.image_id)).limit(_CHUNK)).all()
        if not rows:
            break
        last = rows[-1][0]
        ids, vecs = [], []
        for iid, blob in rows:
            if iid == exclude_id:
                continue
            v = _unit(dequantize_embedding(blob) or [])
            if v is None or v.shape != q.shape:
                continue
            ids.append(iid)
            vecs.append(v)
        if not ids:
            continue
        sims = np.stack(vecs) @ q
        if min_score is not None:
            keep = sims >= min_score
            ids = [i for i, ok in zip(ids, keep) if ok]
            sims = sims[keep]
            if not ids:
                continue
        best_ids = np.concatenate([best_ids, np.asarray(ids, dtype=np.int64)])
        best_sims = np.concatenate([best_sims, sims])
        if len(best_ids) > k:
            keep_idx = np.argpartition(-best_sims, k - 1)[:k]
            best_ids, best_sims = best_ids[keep_idx], best_sims[keep_idx]
    order = np.lexsort((best_ids, -best_sims))
    return [(int(best_ids[i]), float(best_sims[i])) for i in order[:k]]


def find_similar(session, image_id: int, k: int = 10) -> list[tuple[int, float]]:
    """Top-``k`` most similar images to ``image_id`` as ``(image_id, cosine)``.

    Raises ``ValueError`` when the query image has no embedding.
    """
    row = session.exec(
        sselect(SemanticScore).where(SemanticScore.image_id == image_id)).first()
    q = _unit(dequantize_embedding(row.embedding) or []) if row is not None else None
    if q is None:
        raise ValueError(f"image {image_id} has no embedding")
    return top_k(session, q, k, exclude_id=image_id)
