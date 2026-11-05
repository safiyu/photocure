"""Burst detection: graph-constrained temporal cosine sliding window.

Two photos are in the same burst iff they are close in time AND look alike. How close
"close" is depends on how long apart they were taken, because the longer the gap the more
alike the photos must be before they count as the same moment:

    gap <= window_seconds        cosine >= similarity_threshold        (a true burst)
    gap <= near_window_seconds   cosine >= near_similarity_threshold   (quick retakes)
    gap <= far_window_seconds    cosine >= far_similarity_threshold    (retakes within hours)

Photos with no trustworthy capture time (copy time, or none at all: chat apps and
downloads strip it) cannot be placed on a timeline, so they are grouped on look alone,
with a much stricter threshold and a complete-linkage check so unrelated photos cannot
chain together.

Complexity: images are sorted by timestamp once (``O(N log N)``). For each image we only
compare against its *temporal neighbours* within the widest window (a contiguous slice),
so the cosine work is bounded by the average photo density rather than the full
``O(N^2)`` pair set. Photos without a time are compared in blocks.

Groups are formed with union-find over the resulting edges, tightest tier first. A union is
refused when the merged group would span more than the tier's limit, so a long event of
similar frames cannot chain into a single "burst". Screens and documents (screenshots,
slides, flat graphics) look alike without being the same moment, so only the tightest
tier may group them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from collections.abc import Sequence

import numpy as np

from ..config import BurstThresholds, CuratorConfig
from ..db.repository import dequantize_embedding

log = logging.getLogger(__name__)

# Fallback until BurstThresholds grows the field of the same name.
DEFAULT_MAX_BURST_DURATION_S = 120.0
# Timestamp sources that say nothing about when the photo was taken.
UNRELIABLE_TIME_SOURCES = frozenset({"mtime", "unknown"})
# Content that looks alike without being the same moment (UI screens, pages, flat art).
FLAT_CONTENT_CLASSES = frozenset({"screenshot", "document", "graphic", "illustration",
                                  "slide", "utility"})
# Rows of the similarity matrix computed at once for photos without a capture time.
_BLOCK = 256


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
    # A screen, page or flat graphic: only the tightest tier may group it.
    flat_content: bool = False
    filename: str = ""
    phash: str = ""


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
    if v.tzinfo is not None:
        # Drop the zone (as naive stamps have none) so both kinds share one wall-clock convention.
        v = v.replace(tzinfo=None)
    return v.replace(tzinfo=timezone.utc).timestamp()


def _unit_rows(items: Sequence[BurstInput], idx: list[int]) -> tuple[list[int], np.ndarray]:
    """Keep the items whose embeddings share the most common size; return unit-length rows."""
    if not idx:
        return [], np.zeros((0, 0), dtype=np.float32)
    dims = [len(items[i].embedding) for i in idx]
    common = max(set(dims), key=dims.count)
    idx = [i for i, d in zip(idx, dims) if d == common]
    embs = np.array([np.asarray(items[i].embedding, dtype=np.float32) for i in idx],
                    dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return idx, embs / norms


class BurstDetector:
    """Cluster near-duplicate burst shots via temporal + cosine constraints."""

    def __init__(self, config: CuratorConfig | None = None,
                 thresholds: BurstThresholds | None = None):
        self.cfg = config or CuratorConfig()
        self.t = thresholds or self.cfg.burst

    # ------------------------------------------------------------------ tiers
    def _tiers(self) -> list[tuple[float, float, float]]:
        """(window_s, cosine threshold, group span limit) from the tightest tier to the widest."""
        t = self.t
        base_w = float(t.window_seconds)
        base_span = float(getattr(t, "max_burst_duration_s", DEFAULT_MAX_BURST_DURATION_S))
        tiers = [(base_w, float(t.similarity_threshold), base_span)]
        for w_name, s_name in (("near_window_seconds", "near_similarity_threshold"),
                               ("far_window_seconds", "far_similarity_threshold")):
            w = float(getattr(t, w_name, 0) or 0)
            thr = getattr(t, s_name, None)
            if thr is None or w <= tiers[-1][0]:
                continue  # tier switched off, or no wider than the one before it
            # The group may grow to the widest gap this tier admits, no further.
            tiers.append((w, max(float(thr), tiers[0][1]), w))
        return tiers

    # --------------------------------------------------------------- clustering
    def cluster(self, items: Sequence[BurstInput]) -> list[list[int]]:
        """Return a list of groups; each group is a list of image_ids.

        Only groups with >= ``min_group_size`` members are returned; singleton
        photos are omitted (they are not bursts).
        """
        if len(items) < self.t.min_group_size:
            return []
        timed = [i for i in range(len(items))
                 if items[i].taken_at is not None and items[i].embedding
                 and items[i].taken_at_source not in UNRELIABLE_TIME_SOURCES]
        timed_set = set(timed)
        untimed = [i for i in range(len(items))
                   if items[i].embedding and i not in timed_set]
        groups = self._cluster_timed(items, timed) + self._cluster_untimed(items, untimed)

        result: list[list[int]] = []
        for members in groups:
            if len(members) < self.t.min_group_size:
                continue
            member_items = sorted(
                members,
                key=lambda gi: (_ts(items[gi].taken_at) or 0.0, items[gi].filename or str(items[gi].image_id))
            )
            result.append([items[gi].image_id for gi in member_items])
        return result

    def _cluster_timed(self, items: Sequence[BurstInput], timed: list[int]) -> list[list[int]]:
        timed.sort(key=lambda i: _ts(items[i].taken_at))
        usable, embs = _unit_rows(items, timed)
        if len(usable) < self.t.min_group_size:
            return []
        times = np.array([_ts(items[i].taken_at) for i in usable], dtype=np.float64)
        flat = np.array([items[i].flat_content for i in usable], dtype=bool)
        phashes = [items[i].phash for i in usable]
        max_phash_dist = int(getattr(self.t, "max_phash_distance", 12))
        tiers = self._tiers()

        n_usable = len(usable)
        uf = _UnionFind(n_usable, times)
        edges: list[list[tuple[int, int]]] = [[] for _ in tiers]
        # Every in-window pair (i, k) is compared once, as one matrix-vector product per i
        # (times are sorted, so the window is a contiguous slice).
        ends = np.searchsorted(times, times + tiers[-1][0], side="right")
        from ..scanner.phash import hamming
        for i in range(n_usable - 1):
            k_end = int(ends[i])
            if k_end <= i + 1:
                continue
            sims = embs[i + 1:k_end] @ embs[i]
            gaps = times[i + 1:k_end] - times[i]
            lower = -1.0
            for ti, (window, thr, _span) in enumerate(tiers):
                ok = (gaps > lower) & (gaps <= window) & (sims >= thr)
                if ti and flat[i]:
                    ok[:] = False
                elif ti:
                    ok &= ~flat[i + 1:k_end]
                for off in np.flatnonzero(ok):
                    k = i + 1 + int(off)
                    if max_phash_dist > 0 and phashes[i] and phashes[k]:
                        if hamming(phashes[i], phashes[k]) > max_phash_dist:
                            continue
                    edges[ti].append((i, k))
                lower = window
        # Tightest tier first, so a true burst is never split by a looser tier's group limit.
        # Cohesive floor check prevents chaining drift into unrelated scenes/subjects.
        cohesive_floor = float(getattr(self.t, "cohesive_floor", 0.0))
        members: dict[int, list[int]] = {k: [k] for k in range(n_usable)}
        for a, b in edges[0]:
            ra, rb = uf.find(a), uf.find(b)
            if ra == rb:
                continue
            ma, mb = members[ra], members[rb]
            if cohesive_floor > 0.0 and float((embs[ma] @ embs[mb].T).min()) < cohesive_floor:
                continue
            if not uf.union_capped(ra, rb, tiers[0][2]):
                continue
            root = uf.find(ra)
            merged = ma + mb
            members.pop(ra, None)
            members.pop(rb, None)
            members[root] = merged
        # Looser tiers join groups only when every cross pair is still alike: a long chain of
        # look-alikes minutes apart must not drift into a different scene.
        floor = float(getattr(self.t, "loose_floor", 0.0))
        for ti in range(1, len(tiers)):
            span = tiers[ti][2]
            for a, b in edges[ti]:
                ra, rb = uf.find(a), uf.find(b)
                if ra == rb:
                    continue
                ma, mb = members[ra], members[rb]
                if float((embs[ma] @ embs[mb].T).min()) < floor:
                    continue
                if not uf.union_capped(ra, rb, span):
                    continue
                root = uf.find(ra)
                merged = ma + mb
                members.pop(ra, None)
                members.pop(rb, None)
                members[root] = merged

        return [[usable[k] for k in grp] for grp in members.values()]

    def _cluster_untimed(self, items: Sequence[BurstInput], untimed: list[int]) -> list[list[int]]:
        """Group photos that cannot be placed in time, on look alone and only when near identical."""
        thr = getattr(self.t, "untimed_similarity_threshold", None)
        if thr is None or len(untimed) < self.t.min_group_size:
            return []
        untimed = [i for i in untimed if not items[i].flat_content]
        cap = int(getattr(self.t, "untimed_max_items", 40000))
        if len(untimed) > cap:
            log.warning("skipping look-alike grouping of %d photos without a capture time "
                        "(limit %d)", len(untimed), cap)
            return []
        usable, embs = _unit_rows(items, untimed)
        n = len(usable)
        if n < self.t.min_group_size:
            return []

        phashes = [items[u].phash for u in usable]
        max_phash_dist = int(getattr(self.t, "max_phash_distance", 12))
        thr = float(thr)
        floor = float(getattr(self.t, "untimed_floor", thr - 0.05))
        max_group = int(getattr(self.t, "untimed_max_group", 12))
        # Cap neighbours per photo: enough candidates to form a group but bounded in RAM.
        top_k = max(max_group * 4, 32)
        edges: list[tuple[float, int, int]] = []
        from ..scanner.phash import hamming
        for start in range(0, n, _BLOCK):
            block = embs[start:start + _BLOCK] @ embs.T
            for r in range(block.shape[0]):
                row = block[r]
                i = start + r
                tail = row[i + 1:]
                cand = np.flatnonzero(tail >= thr)
                if cand.size == 0:
                    continue
                if cand.size > top_k:
                    top = np.argpartition(tail[cand], -top_k)[-top_k:]
                    cand = cand[top]
                for j in cand:
                    k = i + 1 + int(j)
                    if max_phash_dist > 0 and phashes[i] and phashes[k]:
                        if hamming(phashes[i], phashes[k]) > max_phash_dist:
                            continue
                    edges.append((float(tail[j]), i, k))
        # Most alike pairs first; a group takes a new member only when that member is close to
        # every photo already in it, so a chain of look-alikes cannot drift into a stranger.
        # Tie-break by image_id so a rescan with the same photos produces the same groups
        # regardless of the order rows were read from the database.
        iid = [items[u].image_id for u in usable]
        edges.sort(key=lambda e: (-e[0], min(iid[e[1]], iid[e[2]]), max(iid[e[1]], iid[e[2]])))
        parent = list(range(n))
        members: dict[int, list[int]] = {i: [i] for i in range(n)}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for _sim, a, b in edges:
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            ma, mb = members[ra], members[rb]
            if len(ma) + len(mb) > max_group:
                continue
            if float((embs[ma] @ embs[mb].T).min()) < floor:
                continue
            parent[rb] = ra
            members[ra] = ma + mb
            del members[rb]
        return [[usable[k] for k in grp] for grp in members.values() if len(grp) > 1]

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
            flat = False
            sem = getattr(r, "semantic", None)
            if sem is not None and sem.embedding:
                emb = dequantize_embedding(sem.embedding)
            if sem is not None:
                flat = (getattr(sem, "top1_category", "") or "") in FLAT_CONTENT_CLASSES
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
                flat_content=flat,
                filename=getattr(r, "filename", "") or "",
                phash=getattr(r, "phash", "") or "",
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
