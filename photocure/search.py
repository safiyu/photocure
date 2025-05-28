"""Natural-language photo search: a CLIP text embedding against the stored image embeddings."""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable

import numpy as np
from sqlmodel import col

from .clustering.similarity import _unit, top_k
from .config import CuratorConfig
from .db.models import ImageRecord, SemanticScore

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
