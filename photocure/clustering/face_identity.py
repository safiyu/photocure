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
    exemplars: list[list[float]] | None = None  # Diverse member embeddings for multi-exemplar matching


def assign_cluster(embedding: list[float], centroids: Sequence[ClusterCentroid],
                   threshold: float,
                   similarity_boosts: dict[int, float] | None = None) -> int | None:
    """Id of the closest centroid/exemplars at or above ``threshold``, else ``None``.

    Supports:
    - Multi-exemplar matching (checks cluster centroid and any stored diverse exemplars).
    - Similarity boosts (e.g. temporal/burst co-occurrence bonus per cluster).
    """
    best_id, best_score = None, threshold
    boosts = similarity_boosts or {}
    for c in centroids:
        # Score against mean centroid
        score = cosine(embedding, c.centroid)
        # Multi-exemplar comparison: check closest exemplar if available
        if c.exemplars:
            for ex in c.exemplars:
                ex_score = cosine(embedding, ex)
                if ex_score > score:
                    score = ex_score
        # Add potential temporal / burst / event boost
        effective_score = score + boosts.get(c.person_id, 0.0)
        if effective_score >= best_score:
            best_id, best_score = c.person_id, effective_score
    return best_id


def update_centroid(old_centroid: list[float], old_count: int,
                    new_embedding: list[float]) -> list[float]:
    """Incremental running mean after adding one more embedding."""
    n = old_count + 1
    return [(c * old_count + e) / n for c, e in zip(old_centroid, new_embedding)]


def update_exemplars(current_exemplars: list[list[float]] | None,
                     new_embedding: list[float],
                     max_exemplars: int = 5,
                     diversity_min_cosine: float = 0.88) -> list[list[float]]:
    """Maintain a small diverse set of exemplars (different poses/angles/lighting).

    Adds ``new_embedding`` if the current pool is not full, or if it represents a sufficiently
    different pose from all existing exemplars (cosine < diversity_min_cosine).
    """
    exemplars = list(current_exemplars or [])
    if not exemplars:
        return [list(new_embedding)]
    # Check similarity to existing exemplars
    max_sim = max(cosine(new_embedding, ex) for ex in exemplars)
    if max_sim < diversity_min_cosine:
        # Novel pose / angle: add it
        if len(exemplars) < max_exemplars:
            exemplars.append(list(new_embedding))
        else:
            # Replace the most redundant pair or oldest non-seed
            exemplars.pop(1 if len(exemplars) > 1 else 0)
            exemplars.append(list(new_embedding))
    elif len(exemplars) < max_exemplars:
        exemplars.append(list(new_embedding))
    return exemplars


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
