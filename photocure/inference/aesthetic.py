"""Aesthetic score from the stored CLIP embedding (no extra GPU pass).

A linear head over the L2-normalized 512-dim image embedding produces a 0..1
score. The bundled head is the LAION aesthetic predictor v1 for CLIP ViT-B/16
(MIT); it rates photos 1..10 and that rating is mapped onto 0..1. A custom head
loads from ``PCUR_AESTHETIC_WEIGHTS`` (``.npz`` with ``w`` and optional ``b``,
``kind`` and ``name``, or a bare ``.npy`` vector). Without any usable weights a
deterministic stub keeps the plumbing exercisable; its scores are never stored.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .. import models

log = logging.getLogger("photocure.aesthetic")

EMBED_DIM = 512
_STUB_SEED = 20240601
_SCALE = 8.0  # sharpens the sigmoid for unit-norm inputs
# sigmoid: sigmoid(scale * w.e + b). linear10: a 1..10 rating (w.e + b) mapped to 0..1.
KINDS = ("sigmoid", "linear10")
_stub_warned = False


class AestheticHead:
    """Linear head over a unit embedding; ``kind`` says how the raw value becomes 0..1."""

    def __init__(self, weights: np.ndarray | None = None, bias: float = 0.0,
                 stub: bool = False, kind: str = "sigmoid", name: str = ""):
        if weights is None:
            rng = np.random.default_rng(_STUB_SEED)
            weights = rng.standard_normal(EMBED_DIM).astype(np.float32)
            weights /= np.linalg.norm(weights)
            stub = True
        if kind not in KINDS:
            raise ValueError(f"unknown aesthetic head kind {kind!r}")
        self.w = np.asarray(weights, dtype=np.float32).ravel()
        self.b = float(bias)
        self.stub = stub
        self.kind = kind
        self.name = "stub" if stub else (name or "custom")

    @classmethod
    def load(cls, path: str | Path | None = None) -> AestheticHead:
        """Explicit path or env var first, then the bundled model, else the stub."""
        chosen = path or os.environ.get("PCUR_AESTHETIC_WEIGHTS")
        if chosen:
            return cls._from_file(Path(chosen).expanduser()) or cls()
        bundled = models.model_path(models.AESTHETIC)
        head = cls._from_file(bundled) if bundled is not None else None
        if head is None:
            _warn_stub_once()
            return cls()
        return head

    @classmethod
    def _from_file(cls, p: Path) -> AestheticHead | None:
        try:
            if p.suffix == ".npz":
                data = np.load(p)
                return cls(data["w"], float(data["b"]) if "b" in data else 0.0,
                           kind=str(data["kind"]) if "kind" in data else "sigmoid",
                           name=str(data["name"]) if "name" in data else p.stem)
            return cls(np.load(p), name=p.stem)
        except Exception as e:
            log.warning("aesthetic weights %s unreadable (%s); using stub head", p, e)
            return None

    def _squash(self, raw: np.ndarray) -> np.ndarray:
        if self.kind == "linear10":
            return np.clip((raw - 1.0) / 9.0, 0.0, 1.0)
        return 1.0 / (1.0 + np.exp(-(_SCALE * raw)))

    def score(self, embedding: Sequence[float] | np.ndarray) -> float:
        e = np.asarray(embedding, dtype=np.float32).ravel()
        if e.shape != self.w.shape:
            raise ValueError(f"embedding dim {e.shape[0]} != head dim {self.w.shape[0]}")
        n = float(np.linalg.norm(e))
        if n < 1e-9:
            return 0.5
        raw = float(np.dot(self.w, e / n)) + (self.b / _SCALE if self.kind == "sigmoid" else self.b)
        return float(self._squash(np.float64(raw)))

    def score_many(self, embeddings: np.ndarray) -> np.ndarray:
        e = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        raw = (e / norms) @ self.w + (self.b / _SCALE if self.kind == "sigmoid" else self.b)
        return self._squash(raw.astype(np.float64)).astype(np.float32)


def _warn_stub_once() -> None:
    global _stub_warned
    if not _stub_warned:
        _stub_warned = True
        log.warning("No aesthetic model is available (bundled file missing or altered); "
                    "highlights will stay empty")


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


def _paged(session, model, cond, batch: int):
    """Yield rows matching ``cond`` in id order, ``batch`` at a time, freeing each batch afterwards."""
    from sqlmodel import col, select as sselect
    last = 0
    while True:
        rows = session.exec(sselect(model).where(col(model.id) > last, cond)
                            .order_by(col(model.id)).limit(batch)).all()
        if not rows:
            return
        last = rows[-1].id
        yield rows
        session.flush()
        for row in rows:  # embeddings are large, so do not keep a whole library in memory
            session.expunge(row)


def sync_aesthetic(session, batch: int = 2000) -> int:
    """Bring ``SemanticScore.aesthetic`` in line with the active head.

    Rows scored by another (or unknown) head are re-scored from their stored
    embedding; placeholder embeddings and a stub head leave the score empty.
    Returns the number of rows changed. The caller owns the commit.
    """
    from sqlalchemy import and_
    from sqlmodel import col, or_

    from ..db.models import SemanticScore
    from ..db.repository import dequantize_embedding

    head = get_head()
    S = SemanticScore
    scored = or_(col(S.aesthetic).is_not(None), col(S.aesthetic_model) != "")
    updated = 0
    # Scores that should not exist: everything under a stub head, placeholder embeddings otherwise.
    stale = scored if head.stub else and_(S.backend == "stub", scored)
    for rows in _paged(session, S, stale, batch):
        for row in rows:
            row.aesthetic, row.aesthetic_model = None, ""
            updated += 1
    if head.stub:
        return updated
    needs = and_(S.backend != "stub",
                 or_(col(S.aesthetic).is_(None), col(S.aesthetic_model) != head.name))
    for rows in _paged(session, S, needs, batch):
        for row in rows:
            emb = dequantize_embedding(row.embedding)
            if not emb or len(emb) != head.w.shape[0]:
                continue
            row.aesthetic, row.aesthetic_model = head.score(emb), head.name
            updated += 1
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
