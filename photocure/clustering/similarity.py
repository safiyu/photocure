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
    from ..inference.embedding_cache import matrix
    all_ids, vecs = matrix(session)
    if len(all_ids) == 0 or vecs.shape[1] != q.shape[0]:
        return []
    keep = np.ones(len(all_ids), dtype=bool)
    if filters:
        stmt = sselect(SemanticScore.image_id)
        if join_images:
            stmt = stmt.join(ImageRecord, ImageRecord.id == SemanticScore.image_id)
        for cond in filters:
            stmt = stmt.where(cond)
        eligible = set(session.exec(stmt).all())
        keep = np.fromiter((int(i) in eligible for i in all_ids), dtype=bool, count=len(all_ids))
    if exclude_id is not None:
        keep &= all_ids != exclude_id
    if not keep.any():
        return []
    ids = all_ids[keep]
    sims = vecs[keep].astype(np.float32) @ q.astype(np.float32)
    if min_score is not None:
        m = sims >= min_score
        ids, sims = ids[m], sims[m]
        if len(ids) == 0:
            return []
    if len(ids) > k:
        top = np.argpartition(-sims, k - 1)[:k]
        ids, sims = ids[top], sims[top]
    order = np.lexsort((ids, -sims))
    return [(int(ids[i]), float(sims[i])) for i in order[:k]]


def find_similar(session, image_id: int, k: int = 10) -> list[tuple[int, float]]:
    """Top-``k`` most similar images to ``image_id`` as ``(image_id, cosine)``.

    Raises ``ValueError`` when the query image has no embedding.
    """
    row = session.exec(
        sselect(SemanticScore).where(SemanticScore.image_id == image_id)).first()
    q = _unit(dequantize_embedding(row.embedding) or []) if row is not None else None
    if q is None:
        raise ValueError(f"image {image_id} has no embedding")
    return top_k(session, q, k, exclude_id=image_id, join_images=True,
                 filters=(ImageRecord.scan_status != "missing",))
