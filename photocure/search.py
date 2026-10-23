"""Natural-language photo search: a CLIP text embedding against the stored image embeddings."""
from __future__ import annotations

import logging
import re
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from sqlmodel import col
from sqlmodel import select as sselect

from .clustering.similarity import _CHUNK, _unit, top_k
from .config import CuratorConfig
from .db.models import ImageRecord, SemanticScore
from .db.repository import dequantize_embedding

log = logging.getLogger("photocure.search")

# Each query is encoded plain and as a caption, then pooled (CLIP reads captions best).
_TEMPLATES = ("{}", "a photo of {}")


class SearchUnavailable(RuntimeError):
    """Raised when text search cannot run, with a message fit to show the user."""


def search_embeddings(session, query: np.ndarray, k: int = 60, *,
                      include_duplicates: bool = False,
                      min_score: float | None = None) -> list[tuple[int, float]]:
    """Best ``k`` photos for a unit-length text embedding as ``(image_id, cosine)``.

    Placeholder embeddings carry no meaning, so they are never searched. Duplicates, burst
    alternates and missing files are hidden unless ``include_duplicates`` is set.
    """
    q = _unit(query)
    if q is None:
        raise ValueError("empty query vector")
    filters = [col(SemanticScore.backend) != "stub", col(ImageRecord.scan_status) != "missing"]
    if not include_duplicates:
        filters += [col(ImageRecord.is_duplicate) == False,  # noqa: E712
                    col(ImageRecord.is_alternate) == False]  # noqa: E712
    return top_k(session, q, k, filters=tuple(filters), join_images=True, min_score=min_score)


# Strong matches: at least Z_FLOOR robust z-scores above the library's typical similarity and
# within Z_MARGIN of the best hit. A few results are always shown so the page is never empty.
Z_FLOOR = 2.0
Z_MARGIN = 2.0
MIN_SHOWN = 10
_MAX_PARTS = 3
_CONNECTORS = re.compile(
    r"\s*(?:,|\b(?:with|and|of|in|on|at|near|beside|by|while|holding|wearing|next to)\b)\s*", re.I)
_FILLER = re.compile(r"^(?:a|an|the|some|my|our|his|her|their|photos?|pictures?|images?|me|us)\b\s*", re.I)


def split_concepts(text: str) -> list[str]:
    """The separate things a query asks for ("playing with a dog" gives playing and dog).

    Empty when the query is one idea, so single-concept searches behave exactly as before.
    """
    parts: list[str] = []
    for raw in _CONNECTORS.split(" ".join(text.split())):
        piece = raw.strip()
        while True:
            stripped = _FILLER.sub("", piece).strip()
            if stripped == piece:
                break
            piece = stripped
        if piece and piece.lower() not in {x.lower() for x in parts}:
            parts.append(piece)
    return parts[:_MAX_PARTS] if len(parts) > 1 else []


@dataclass
class SearchResult:
    hits: list[tuple[int, float]]          # (image_id, cosine to the full query), best first
    strong: int = 0                        # how many cleared the strong-match bar (capped at k)
    more: int = 0                          # weaker matches held back (0 when everything is shown)
    parts: list[str] = field(default_factory=list)


def _scan(session, queries: np.ndarray, filters: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Cosine of every searchable photo against each query row: ids (N,), sims (N, P)."""
    ids: list[int] = []
    blocks: list[np.ndarray] = []
    last = 0
    while True:
        rows = session.exec(
            sselect(SemanticScore.image_id, SemanticScore.embedding)
            .join(ImageRecord, ImageRecord.id == SemanticScore.image_id)
            .where(*filters, col(SemanticScore.image_id) > last)
            .order_by(col(SemanticScore.image_id)).limit(_CHUNK)).all()
        if not rows:
            break
        last = rows[-1][0]
        keep_ids, vecs = [], []
        for iid, blob in rows:
            v = _unit(dequantize_embedding(blob) or [])
            if v is None or v.shape[0] != queries.shape[1]:
                continue
            keep_ids.append(iid)
            vecs.append(v)
        if keep_ids:
            ids.extend(keep_ids)
            blocks.append(np.stack(vecs) @ queries.T)
    if not ids:
        return np.empty(0, dtype=np.int64), np.empty((0, queries.shape[0]), dtype=np.float32)
    return np.asarray(ids, dtype=np.int64), np.concatenate(blocks).astype(np.float32)


def _robust_z(sims: np.ndarray) -> np.ndarray:
    """Per-column z-scores from the median and MAD, so phrases with different baselines compare."""
    med = np.median(sims, axis=0)
    mad = np.median(np.abs(sims - med), axis=0) * 1.4826
    return (sims - med) / np.maximum(mad, 1e-3)


def search_ranked(session, encoder, text: str, k: int = 60, *, include_duplicates: bool = False,
                  strict: bool = True, extra_filters: tuple = ()) -> SearchResult:
    """Search that needs every part of the query, and stops at the strong matches.

    A photo scores the average of its full-phrase z-score and its weakest part's z-score, so
    "playing with a dog" no longer fills up with every playing photo. With ``strict`` the list
    ends where the matches stop being strong (never below ``MIN_SHOWN`` results).
    """
    parts = split_concepts(text)
    vecs = np.stack([encoder.encode(text)] + [encoder.encode(p) for p in parts])
    filters = [col(SemanticScore.backend) != "stub", col(ImageRecord.scan_status) != "missing"]
    if not include_duplicates:
        filters += [col(ImageRecord.is_duplicate) == False,  # noqa: E712
                    col(ImageRecord.is_alternate) == False]  # noqa: E712
    ids, sims = _scan(session, vecs, (*filters, *extra_filters))
    if len(ids) == 0:
        return SearchResult([], parts=parts)
    z = _robust_z(sims)
    score = z[:, 0] if not parts else 0.5 * z[:, 0] + 0.5 * z[:, 1:].min(axis=1)
    order = np.lexsort((ids, -score))
    cutoff = max(Z_FLOOR, float(score[order[0]]) - Z_MARGIN)
    strong = int((score >= cutoff).sum())
    pool = min(k, len(ids))
    shown = pool if not strict else min(pool, max(strong, MIN_SHOWN))
    hits = [(int(ids[i]), float(sims[i, 0])) for i in order[:shown]]
    return SearchResult(hits, strong=min(strong, pool), more=pool - shown, parts=parts)


class TextEncoder:
    """Lazy CLIP text tower shared by searches, freed after ``idle_s`` seconds unused."""

    def __init__(self, engine_factory: Callable[[], object], idle_s: float = 120.0,
                 cache_size: int = 64):
        self._factory = engine_factory
        self._idle_s = idle_s
        self._cache_size = cache_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._engine = None
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    def encode(self, text: str) -> np.ndarray:
        text = " ".join(str(text).split())
        if not text:
            raise ValueError("empty query")
        with self._lock:
            hit = self._cache.get(text)
            if hit is not None:
                self._cache.move_to_end(text)
                self._arm()
                return hit.copy()
            if self._engine is None:
                self._engine = self._factory()
            if not self._engine.text_search_available():
                self._release_locked()
                raise SearchUnavailable(
                    "Search needs the AI model, which is not installed. Run: photocure setup")
            feats = np.asarray(self._engine.encode_queries(
                [t.format(text) for t in _TEMPLATES]), dtype=np.float32)
            from .inference.semantic_engine import ensemble_text_features
            vec = ensemble_text_features([feats])[0]
            self._cache[text] = vec
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            self._arm()
            return vec.copy()

    def _arm(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self._idle_s, self.release)
        self._timer.daemon = True
        self._timer.start()

    def _release_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        engine, self._engine = self._engine, None
        if engine is not None:
            try:
                engine.release()
            except Exception:
                log.debug("text encoder release failed", exc_info=True)

    def release(self) -> None:
        with self._lock:
            self._release_locked()

    def close(self) -> None:
        self.release()


def default_encoder(cfg: CuratorConfig | None = None) -> TextEncoder:
    """An encoder on the active backend; reports the placeholder as unavailable."""
    from .inference.backends import detect_backend
    from .inference.semantic_engine import SemanticEngine
    c = cfg or CuratorConfig()
    return TextEncoder(lambda: SemanticEngine(c, backend=detect_backend(c)))
