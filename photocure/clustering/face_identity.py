"""Pure face-identity clustering and backfill-remap matching: no DB, no OpenCV.

Two distinct matching problems live here. ``assign_cluster``/``update_centroid``
decide which *global* person (across the whole library) a newly-seen face
embedding belongs to. ``remap_person_ids`` instead matches a *single photo's*
previously-detected faces against its freshly re-detected ones, so a forced
re-analysis does not lose the identities already assigned to that photo.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def cosine(a: list[float], b: list[float]) -> float:
    av, bv = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = float(np.linalg.norm(av)), float(np.linalg.norm(bv))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


@dataclass
class ClusterCentroid:
    person_id: int
    centroid: list[float]
    face_count: int


def assign_cluster(embedding: list[float], centroids: Sequence[ClusterCentroid],
                   threshold: float) -> int | None:
    """Id of the closest centroid at or above ``threshold``, else ``None``."""
    best_id, best_score = None, threshold
    for c in centroids:
        score = cosine(embedding, c.centroid)
        if score >= best_score:
            best_id, best_score = c.person_id, score
    return best_id


def update_centroid(old_centroid: list[float], old_count: int,
                    new_embedding: list[float]) -> list[float]:
    """Incremental running mean after adding one more embedding."""
    n = old_count + 1
    return [(c * old_count + e) / n for c, e in zip(old_centroid, new_embedding)]


@dataclass
class OldFace:
    person_id: int
    embedding: list[float]


@dataclass
class NewFace:
    index: int
    embedding: list[float]


def remap_person_ids(old_faces: Sequence[OldFace], new_faces: Sequence[NewFace],
                     threshold: float) -> dict[int, int]:
    """Map ``new_face.index -> person_id`` via a one-to-one greedy match.

    Every old/new pair at or above ``threshold`` is a candidate; pairs are
    claimed highest-similarity first, and a face on either side is never
    reused once claimed.
    """
    pairs = []
    for oi, old in enumerate(old_faces):
        for new in new_faces:
            score = cosine(old.embedding, new.embedding)
            if score >= threshold:
                pairs.append((score, oi, new.index, old.person_id))
    pairs.sort(key=lambda p: p[0], reverse=True)

    used_old: set[int] = set()
    used_new: set[int] = set()
    result: dict[int, int] = {}
    for _score, oi, new_index, person_id in pairs:
        if oi in used_old or new_index in used_new:
            continue
        used_old.add(oi)
        used_new.add(new_index)
        result[new_index] = person_id
    return result
