"""Aesthetic score from the stored CLIP embedding (no extra GPU pass).

A linear head over the L2-normalized 512-dim image embedding produces a 0..1
score. Real weights (``.npz`` with ``w`` and optional ``b``, or a ``.npy`` vector)
load from ``PCUR_AESTHETIC_WEIGHTS``. Without a weights file a deterministic
stub head is used so the pipeline stays reproducible; its scores are NOT a
meaningful aesthetic judgement and only exist to exercise the plumbing.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

log = logging.getLogger("photocure.aesthetic")

EMBED_DIM = 512
_STUB_SEED = 20240601
_SCALE = 8.0  # sharpens the sigmoid for unit-norm inputs


class AestheticHead:
    """Linear head: sigmoid(scale * (w . e) + b)."""

    def __init__(self, weights: np.ndarray | None = None, bias: float = 0.0,
                 stub: bool = False):
        if weights is None:
            rng = np.random.default_rng(_STUB_SEED)
            weights = rng.standard_normal(EMBED_DIM).astype(np.float32)
            weights /= np.linalg.norm(weights)
            stub = True
        self.w = np.asarray(weights, dtype=np.float32).ravel()
        self.b = float(bias)
        self.stub = stub

    @classmethod
    def load(cls, path: str | Path | None = None) -> AestheticHead:
        chosen = path or os.environ.get("PCUR_AESTHETIC_WEIGHTS")
        if not chosen:
            return cls()
        p = Path(chosen).expanduser()
        try:
            if p.suffix == ".npz":
                data = np.load(p)
                return cls(data["w"], float(data["b"]) if "b" in data else 0.0)
            return cls(np.load(p))
        except Exception as e:
            log.warning("aesthetic weights %s unreadable (%s); using stub head", p, e)
            return cls()

    def score(self, embedding: Sequence[float] | np.ndarray) -> float:
        e = np.asarray(embedding, dtype=np.float32).ravel()
        if e.shape != self.w.shape:
            raise ValueError(f"embedding dim {e.shape[0]} != head dim {self.w.shape[0]}")
        n = float(np.linalg.norm(e))
        if n < 1e-9:
            return 0.5
        z = _SCALE * float(np.dot(self.w, e / n)) + self.b
        return float(1.0 / (1.0 + np.exp(-z)))

    def score_many(self, embeddings: np.ndarray) -> np.ndarray:
        e = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        z = _SCALE * ((e / norms) @ self.w) + self.b
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


_head: AestheticHead | None = None


def get_head() -> AestheticHead:
    global _head
    if _head is None:
        _head = AestheticHead.load()
    return _head


def reset_head() -> None:
    global _head
    _head = None


def score_embedding(embedding: Sequence[float] | np.ndarray) -> float:
    return get_head().score(embedding)


def backfill_aesthetic(session, batch: int = 2000) -> int:
    """Fill ``SemanticScore.aesthetic`` for rows that have an embedding but no score."""
    from sqlmodel import select as sselect

    from ..db.models import SemanticScore
    from ..db.repository import dequantize_embedding

    head = get_head()
    updated = 0
    rows = session.exec(
        sselect(SemanticScore).where(SemanticScore.aesthetic.is_(None))  # type: ignore[union-attr]
    ).all()
    for row in rows:
        emb = dequantize_embedding(row.embedding)
        if not emb or len(emb) != head.w.shape[0]:
            continue
        row.aesthetic = head.score(emb)
        updated += 1
        if updated % batch == 0:
            session.flush()
    session.flush()
    return updated


def highlights(session, min_score: float = 0.6, limit: int = 100) -> list[tuple[int, float]]:
    """Top aesthetic images as ``(image_id, score)``, best first.

    Duplicates, burst alternates and unreadable rows are excluded.
    """
    from sqlmodel import col, select as sselect

    from ..db.models import ImageRecord, SemanticScore

    stmt = (
        sselect(SemanticScore.image_id, SemanticScore.aesthetic)
        .join(ImageRecord, ImageRecord.id == SemanticScore.image_id)
        .where(
            col(SemanticScore.aesthetic) >= min_score,
            ImageRecord.is_duplicate == False,  # noqa: E712
            ImageRecord.is_alternate == False,  # noqa: E712
        )
        .order_by(col(SemanticScore.aesthetic).desc(), col(SemanticScore.image_id))
        .limit(limit)
    )
    return [(int(i), float(a)) for i, a in session.exec(stmt).all()]
