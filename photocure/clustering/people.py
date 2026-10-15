"""Person clustering from stored face embeddings (numpy only).

Leader clustering on cosine similarity: each face joins the closest running
centroid when similarity >= ``threshold``, otherwise it starts a new person.
A second pass reassigns every face against the final centroids and merges
centroids that ended up closer than the threshold. Existing person ids (and
their user-given names) are kept when a new cluster overlaps an old one.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from sqlmodel import col, select as sselect

from ..db.models import FaceDetection, PersonCluster
from ..db.repository import dequantize_embedding

# ArcFace-style embeddings: same person is typically above ~0.4-0.5 cosine.
DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_CONFIDENCE = 0.65


def _assign(embs: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return np.argmax(embs @ centroids.T, axis=1)


def _leader_pass(embs: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    cents = np.zeros_like(embs)
    sums = np.zeros_like(embs)
    counts: list[int] = []
    k = 0
    labels = np.empty(len(embs), dtype=np.int64)
    for i, e in enumerate(embs):
        if k:
            sims = cents[:k] @ e
            j = int(np.argmax(sims))
            if sims[j] >= threshold:
                labels[i] = j
                sums[j] += e
                counts[j] += 1
                n = np.linalg.norm(sums[j])
                cents[j] = sums[j] / n if n > 1e-9 else sums[j]
                continue
        labels[i] = k
        sums[k] = e
        cents[k] = e
        counts.append(1)
        k += 1
    return labels, cents[:k].copy()


def _merge_centroids(embs: np.ndarray, labels: np.ndarray, cents: np.ndarray,
                     threshold: float) -> tuple[np.ndarray, np.ndarray, bool]:
    """Union centroids closer than ``threshold``; returns new labels/centroids."""
    k = len(cents)
    if k < 2:
        return labels, cents, False
    sim = cents @ cents.T
    pairs = np.argwhere(np.triu(sim >= threshold, 1))
    if len(pairs) == 0:
        return labels, cents, False
    parent = list(range(k))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        parent[find(int(j))] = find(int(i))
    roots = sorted({find(i) for i in range(k)})
    remap = {r: n for n, r in enumerate(roots)}
    labels = np.array([remap[find(int(lb))] for lb in labels], dtype=np.int64)
    new = np.zeros((len(roots), embs.shape[1]), dtype=embs.dtype)
    for lab in range(len(roots)):
        c = embs[labels == lab].sum(axis=0)
        n = np.linalg.norm(c)
        new[lab] = c / n if n > 1e-9 else c
    return labels, new, True


def cluster_embeddings(embs: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Cluster unit-norm rows; returns a 0-based label per row."""
    if len(embs) == 0:
        return np.empty(0, dtype=np.int64)
    labels, cents = _leader_pass(embs, threshold)
    for _ in range(3):
        labels, cents, merged = _merge_centroids(embs, labels, cents, threshold)
        if not merged:
            break
    # Final reassignment against the settled centroids, then compact the ids.
    labels = _assign(embs, cents)
    uniq = {int(v): n for n, v in enumerate(sorted(set(labels.tolist())))}
    return np.array([uniq[int(v)] for v in labels], dtype=np.int64)


def cluster_people(session, threshold: float = DEFAULT_THRESHOLD,
                   min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> int:
    """Cluster all embedded faces into people; returns the number of people."""
    faces = [f for f in session.exec(
        sselect(FaceDetection).where(col(FaceDetection.embedding_json).is_not(None))
    ).all() if f.confidence >= min_confidence]
    faces.sort(key=lambda f: f.id or 0)

    usable: list[FaceDetection] = []
    vecs: list[np.ndarray] = []
    for f in faces:
        v = dequantize_embedding(f.embedding_json or "")
        if not v:
            continue
        a = np.asarray(v, dtype=np.float32)
        n = float(np.linalg.norm(a))
        if n < 1e-9:
            continue
        usable.append(f)
        vecs.append(a / n)

    # Every previously assigned face is reset, even ones that no longer qualify.
    old_by_face = {f.id: f.person_id for f in session.exec(
        sselect(FaceDetection).where(col(FaceDetection.person_id).is_not(None))).all()}
    for f in session.exec(
            sselect(FaceDetection).where(col(FaceDetection.person_id).is_not(None))).all():
        f.person_id = None

    labels = (cluster_embeddings(np.stack(vecs), threshold)
              if vecs else np.empty(0, dtype=np.int64))
    n_clusters = int(labels.max()) + 1 if len(labels) else 0

    members: dict[int, list[FaceDetection]] = {c: [] for c in range(n_clusters)}
    for f, lab in zip(usable, labels):
        members[int(lab)].append(f)

    # Reuse old person rows (keeps names) for clusters that mostly overlap them.
    existing = {p.id: p for p in session.exec(sselect(PersonCluster)).all()}
    claimed: set[int] = set()
    assignment: dict[int, PersonCluster] = {}
    ordered = sorted(members, key=lambda c: -len(members[c]))
    for c in ordered:
        votes = Counter(old_by_face.get(f.id) for f in members[c]
                        if old_by_face.get(f.id) in existing)
        for pid, cnt in votes.most_common():
            if pid not in claimed and cnt * 2 >= len(members[c]):
                claimed.add(pid)
                assignment[c] = existing[pid]
                break
    for c in ordered:
        if c not in assignment:
            p = PersonCluster(face_count=0)
            session.add(p)
            session.flush()
            assignment[c] = p
    for c, fs in members.items():
        p = assignment[c]
        p.face_count = len(fs)
        for f in fs:
            f.person_id = p.id
    for pid, p in existing.items():
        if pid in claimed:
            continue
        if p.name:
            p.face_count = 0  # keep the user's naming work
        else:
            session.delete(p)
    session.flush()
    return n_clusters
