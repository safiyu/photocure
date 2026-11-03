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
    temporal_medoids: dict[str, list[float]] | None = None  # Chronological/yearly medoid vectors
    image_ids: set[int] | None = None  # Set of image_ids already in this cluster (co-occurrence veto)


def assign_cluster(embedding: list[float], centroids: Sequence[ClusterCentroid],
                   threshold: float,
                   similarity_boosts: dict[int, float] | None = None,
                   image_id: int | None = None,
                   time_bucket: str | None = None) -> int | None:
    """Id of the closest centroid/exemplars at or above ``threshold``, else ``None``.

    Supports:
    - Sibling Protection / Cannot-Link Veto: strictly forbids assigning if image_id is already in the cluster.
    - Chronological Medoid Chaining: compares against adjacent yearly medoids for childhood growth bridging.
    - Calibrated Operating Point: scales threshold from 0.55 down to 0.38 for 512-D ArcFace vectors.
    """
    dim = len(embedding)
    # Calibrate threshold for 512-D ArcFace vectors vs 128-D SFace
    if dim >= 512:
        # Standard ArcFace operating point: strangers are < 0.20, same person is > 0.38
        eff_threshold = min(threshold, 0.38) if threshold >= 0.50 else threshold
    else:
        eff_threshold = threshold

    best_id, best_score = None, eff_threshold
    boosts = similarity_boosts or {}

    for c in centroids:
        # 1. Co-occurrence Veto (cannot-link constraint)
        if image_id is not None and c.image_ids and image_id in c.image_ids:
            continue

        # 2. Score against mean centroid
        score = cosine(embedding, c.centroid)

        # 3. Multi-exemplar comparison
        if c.exemplars:
            for ex in c.exemplars:
                ex_score = cosine(embedding, ex)
                if ex_score > score:
                    score = ex_score

        # 4. Chronological medoid chaining (yearly / developmental bucket)
        if c.temporal_medoids:
            if time_bucket and time_bucket in c.temporal_medoids:
                med_score = cosine(embedding, c.temporal_medoids[time_bucket])
                if med_score > score:
                    score = med_score
            else:
                for med in c.temporal_medoids.values():
                    med_score = cosine(embedding, med)
                    if med_score > score:
                        score = med_score

        # 5. Add potential temporal / burst / event boost
        effective_score = score + boosts.get(c.person_id, 0.0)
        if effective_score >= best_score:
            best_id, best_score = c.person_id, effective_score

    return best_id


def update_temporal_medoid(current_medoids: dict[str, list[float]] | None,
                           time_bucket: str | None,
                           new_embedding: list[float]) -> dict[str, list[float]]:
    """Update or add a yearly/chronological medoid vector for a cluster."""
    import numpy as np

    if not time_bucket:
        return current_medoids or {}
    meds = dict(current_medoids or {})
    if time_bucket not in meds:
        meds[time_bucket] = list(new_embedding)
    else:
        old_vec = np.array(meds[time_bucket], dtype=np.float32)
        new_vec = np.array(new_embedding, dtype=np.float32)
        blended = old_vec * 0.7 + new_vec * 0.3
        norm = float(np.linalg.norm(blended))
        if norm > 1e-6:
            blended /= norm
        meds[time_bucket] = blended.tolist()
    return meds


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
