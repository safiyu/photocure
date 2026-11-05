"""Process-level cache of every photo's CLIP embedding as one float16 matrix.

Search and similarity used to decode a JSON blob per row on every query; at 100k photos that is
seconds of pure parsing. The matrix is rebuilt when the semantic table changes (row count, newest
id or newest ``computed_at``) and dropped explicitly when the library is cleared.
"""
from __future__ import annotations

import threading

import numpy as np
from sqlalchemy import func
from sqlmodel import select

from ..db.models import SemanticScore
from ..db.repository import dequantize_embedding

_lock = threading.Lock()
_cache: dict[str, tuple[tuple, np.ndarray, np.ndarray]] = {}


def _key(session) -> str:
    return str(session.get_bind().url)


def _version(session) -> tuple:
    return tuple(session.exec(select(func.count(SemanticScore.id), func.max(SemanticScore.id),
                                     func.max(SemanticScore.computed_at))).one())


def matrix(session) -> tuple[np.ndarray, np.ndarray]:
    """``(ids int64 (N,), vecs float16 (N, D))`` of unit embeddings for every non-placeholder row."""
    key, version = _key(session), _version(session)
    with _lock:
        hit = _cache.get(key)
        if hit is not None and hit[0] == version:
            return hit[1], hit[2]
    ids: list[int] = []
    vecs: list[np.ndarray] = []
    dim = None
    for iid, blob, backend in session.exec(
            select(SemanticScore.image_id, SemanticScore.embedding, SemanticScore.backend)
            .order_by(SemanticScore.image_id)).all():
        if backend == "stub" or not blob:
            continue
        v = dequantize_embedding(blob)
        if not v:
            continue
        arr = np.asarray(v, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm < 1e-9:
            continue
        if dim is None:
            dim = arr.shape[0]
        if arr.shape[0] != dim:
            continue
        ids.append(int(iid))
        vecs.append((arr / norm).astype(np.float16))
    out_ids = np.asarray(ids, dtype=np.int64)
    out_vecs = np.stack(vecs) if vecs else np.empty((0, dim or 0), dtype=np.float16)
    with _lock:
        _cache[key] = (version, out_ids, out_vecs)
    return out_ids, out_vecs


def invalidate() -> None:
    with _lock:
        _cache.clear()
