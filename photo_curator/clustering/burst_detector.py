"""Burst detection: graph-constrained temporal cosine sliding window.

Two photos are in the same burst iff:

    ``abs(t_i - t_j) <= window_seconds``  AND  ``cosine(e_i, e_j) >= sim_thr``

Complexity: images are sorted by timestamp once (``O(N log N)``). For each
image we only compare against its *temporal neighbours* within the window
(using a two-pointer sweep), so the cosine work is bounded by the average
burst density rather than the full ``O(N^2)`` pair set.

Groups are formed with union-find over the resulting edges.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from collections.abc import Sequence

import numpy as np

from ..config import BurstThresholds, CuratorConfig
from ..db.repository import dequantize_embedding

# A record-like view the detector consumes (decouples from SQLModel rows).
@dataclass
class BurstInput:
    image_id: int
    taken_at: datetime | None
    embedding: list[float] | None  # normalized 512-dim
    # Optional pre-computed sharpness so the selector need not re-read pixels.
    sharpness_norm: float = 0.0


class _UnionFind:
    __slots__ = ("parent",)

    def __init__(self, n: int):
        self.parent = list(range(n))

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

        # Index items that have both a timestamp and an embedding.
        order = list(range(n))
        # Sort by timestamp (None timestamps sort last and are excluded).
        order.sort(key=lambda i: (_ts(items[i].taken_at) is None,
                                  _ts(items[i].taken_at) or 0.0))

        usable = [i for i in order if items[i].taken_at is not None
                  and items[i].embedding]
        if len(usable) < self.t.min_group_size:
            return []

        # Build normalized embedding matrix for the usable subset.
        embs = np.array([np.asarray(items[i].embedding, dtype=np.float32)
                         for i in usable], dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms
        times = np.array([_ts(items[i].taken_at) for i in usable], dtype=np.float64)

        uf = _UnionFind(len(usable))
        # For each i, sweep the forward in-window neighbours with a fresh
        # pointer (reset per i) so every pair (i, k) with
        # ``times[k] - times[i] <= window`` is compared exactly once. A single
        # monotonically-advancing pointer would skip in-window neighbours for
        # later i and break union-find transitivity. Complexity is O(N * W)
        # where W is the average in-window fan-out (small for real bursts).
        window = float(self.t.window_seconds)
        thr = self.t.similarity_threshold
        n_usable = len(usable)
        for i in range(n_usable):
            k = i + 1
            while k < n_usable and (times[k] - times[i]) <= window:
                if float(np.dot(embs[i], embs[k])) >= thr:
                    uf.union(i, k)
                k += 1

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
