"""Burst detection: graph-constrained temporal cosine sliding window.

Two photos are in the same burst iff:

    ``abs(t_i - t_j) <= window_seconds``  AND  ``cosine(e_i, e_j) >= sim_thr``

Complexity: images are sorted by timestamp once (``O(N log N)``). For each
image we only compare against its *temporal neighbours* within the window
(using a two-pointer sweep), so the cosine work is bounded by the average
burst density rather than the full ``O(N^2)`` pair set.

Groups are formed with union-find over the resulting edges. A union is refused
when the merged group would span more than ``max_burst_duration_s``, so a long
event of similar frames cannot chain into a single "burst". Inputs dated from
file mtime (or unknown) are skipped: copy times are not capture times.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from collections.abc import Sequence

import numpy as np

from ..config import BurstThresholds, CuratorConfig
from ..db.repository import dequantize_embedding

# Fallback until BurstThresholds grows the field of the same name.
DEFAULT_MAX_BURST_DURATION_S = 120.0
# Timestamp sources that say nothing about when the photo was taken.
UNRELIABLE_TIME_SOURCES = frozenset({"mtime", "unknown"})

# A record-like view the detector consumes (decouples from SQLModel rows).
@dataclass
class BurstInput:
    image_id: int
    taken_at: datetime | None
    embedding: list[float] | None  # normalized 512-dim
    # Optional pre-computed sharpness so the selector need not re-read pixels.
    sharpness_norm: float = 0.0
    # Which cascade tier produced ``taken_at`` (see ImageRecord.taken_at_source).
    taken_at_source: str = ""


class _UnionFind:
    __slots__ = ("parent", "lo", "hi")

    def __init__(self, n: int, times: np.ndarray | None = None):
        self.parent = list(range(n))
        # Per-root time span, used to cap a group's duration.
        self.lo = list(times) if times is not None else [0.0] * n
        self.hi = list(times) if times is not None else [0.0] * n

    def union_capped(self, a: int, b: int, max_span: float) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        lo = min(self.lo[ra], self.lo[rb])
        hi = max(self.hi[ra], self.hi[rb])
        if hi - lo > max_span:
            return False
        self.parent[rb] = ra
        self.lo[ra], self.hi[ra] = lo, hi
        return True

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _ts(v: datetime | None) -> float | None:
    if v is None:
        return None
    if v.tzinfo is None:
        v = v.replace(tzinfo=timezone.utc)
    return v.timestamp()


class BurstDetector:
    """Cluster near-duplicate burst shots via temporal + cosine constraints."""

    def __init__(self, config: CuratorConfig | None = None,
                 thresholds: BurstThresholds | None = None):
        self.cfg = config or CuratorConfig()
        self.t = thresholds or self.cfg.burst

    def cluster(self, items: Sequence[BurstInput]) -> list[list[int]]:
        """Return a list of groups; each group is a list of image_ids.

        Only groups with >= ``min_group_size`` members are returned; singleton
        photos are omitted (they are not bursts).
        """
        n = len(items)
        if n < self.t.min_group_size:
            return []

        # Index items that have a trustworthy timestamp and an embedding.
        order = list(range(n))
        order.sort(key=lambda i: (_ts(items[i].taken_at) is None,
                                  _ts(items[i].taken_at) or 0.0))

        usable = [i for i in order if items[i].taken_at is not None
                  and items[i].embedding
                  and items[i].taken_at_source not in UNRELIABLE_TIME_SOURCES]
        if len(usable) < self.t.min_group_size:
            return []

        embs = np.array([np.asarray(items[i].embedding, dtype=np.float32)
                         for i in usable], dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms
        times = np.array([_ts(items[i].taken_at) for i in usable], dtype=np.float64)

        window = float(self.t.window_seconds)
        thr = self.t.similarity_threshold
        max_span = float(getattr(self.t, "max_burst_duration_s",
                                 DEFAULT_MAX_BURST_DURATION_S))
        n_usable = len(usable)
        uf = _UnionFind(n_usable, times)
        # Every in-window pair (i, k) is compared once, as one matrix-vector
        # product per i (times are sorted, so the window is a contiguous slice).
        ends = np.searchsorted(times, times + window, side="right")
        for i in range(n_usable - 1):
            k_end = int(ends[i])
            if k_end <= i + 1:
                continue
            sims = embs[i + 1:k_end] @ embs[i]
            for off in np.flatnonzero(sims >= thr):
                uf.union_capped(i, i + 1 + int(off), max_span)

        # Collect groups.
        groups: dict[int, list[int]] = {}
        for idx, gi in enumerate(usable):
            root = uf.find(idx)
            groups.setdefault(root, []).append(gi)

        # Keep only real bursts and convert to original image_ids, sorted by time.
        result: list[list[int]] = []
        for members in groups.values():
            if len(members) < self.t.min_group_size:
                continue
            # ``members`` already holds *original* item indices (gi).
            member_items = sorted(
                members,
                key=lambda gi: _ts(items[gi].taken_at) or 0.0,
            )
            result.append([items[gi].image_id for gi in member_items])
        return result

    def cluster_records(self, records: Sequence) -> list[BurstGroupResult]:
        """Convenience: build BurstInputs from DB records and run the detector.

        ``records`` may be raw :class:`ImageRecord` rows; their ``semantic``
        and ``quality`` child rows are read via lazy attributes when present.
        """
        inputs = []
        for r in records:
            if r.infer_status != "inferred":
                continue
            if r.is_duplicate:
                continue  # duplicates are handled by the decision matrix
            emb = None
            sem = getattr(r, "semantic", None)
            if sem is not None and sem.embedding:
                emb = dequantize_embedding(sem.embedding)
            sharp = 0.0
            q = getattr(r, "quality", None)
            if q is not None:
                sharp = q.sharpness_norm
            inputs.append(BurstInput(
                image_id=r.id,
                taken_at=r.taken_at,
                embedding=emb,
                sharpness_norm=sharp,
                taken_at_source=r.taken_at_source or "",
            ))
        groups = self.cluster(inputs)
        return [BurstGroupResult(image_ids) for image_ids in groups]


@dataclass
class BurstGroupResult:
    image_ids: list[int]


def cluster_bursts(records: Sequence, config: CuratorConfig | None = None
                   ) -> list[BurstGroupResult]:
    cfg = config or CuratorConfig()
    return BurstDetector(cfg).cluster_records(records)
