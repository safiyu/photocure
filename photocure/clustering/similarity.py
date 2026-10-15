"""Embedding similarity search: brute-force cosine over stored CLIP embeddings."""
from __future__ import annotations

import numpy as np
from sqlmodel import col, select as sselect

from ..db.models import SemanticScore
from ..db.repository import dequantize_embedding

_CHUNK = 20000


def _unit(v: list[float] | np.ndarray) -> np.ndarray | None:
    a = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-9 else None


def find_similar(session, image_id: int, k: int = 10) -> list[tuple[int, float]]:
    """Top-``k`` most similar images to ``image_id`` as ``(image_id, cosine)``.

    Scans embeddings in keyset-paginated chunks so memory stays bounded for
    large libraries. Raises ``ValueError`` when the query image has no embedding.
    """
    row = session.exec(
        sselect(SemanticScore).where(SemanticScore.image_id == image_id)).first()
    q = _unit(dequantize_embedding(row.embedding) or []) if row is not None else None
    if q is None:
        raise ValueError(f"image {image_id} has no embedding")
    if k <= 0:
        return []

    best_ids = np.empty(0, dtype=np.int64)
    best_sims = np.empty(0, dtype=np.float32)
    last = 0
    while True:
        rows = session.exec(
            sselect(SemanticScore.image_id, SemanticScore.embedding)
            .where(col(SemanticScore.image_id) > last)
            .order_by(col(SemanticScore.image_id))
            .limit(_CHUNK)
        ).all()
        if not rows:
            break
        last = rows[-1][0]
        ids, vecs = [], []
        for iid, blob in rows:
            if iid == image_id:
                continue
            v = _unit(dequantize_embedding(blob) or [])
            if v is None or v.shape != q.shape:
                continue
            ids.append(iid)
            vecs.append(v)
        if not ids:
            continue
        sims = np.stack(vecs) @ q
        best_ids = np.concatenate([best_ids, np.asarray(ids, dtype=np.int64)])
        best_sims = np.concatenate([best_sims, sims])
        if len(best_ids) > k:
            keep = np.argpartition(-best_sims, k - 1)[:k]
            best_ids, best_sims = best_ids[keep], best_sims[keep]
    order = np.lexsort((best_ids, -best_sims))
    return [(int(best_ids[i]), float(best_sims[i])) for i in order[:k]]
