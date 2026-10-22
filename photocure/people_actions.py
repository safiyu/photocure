"""People actions: list/rename clusters and reassign a single face. Shared by the web app and MCP."""
from __future__ import annotations

from pathlib import Path

from sqlmodel import select

from .db import models as M
from .db.repository import PhotoRepository, dequantize_embedding
from .db.session import get_session


def _best_face_id_for(person_id: int, s) -> int | None:
    """Return the face ID with the highest sharpness for a given person cluster, or any face ID."""
    # Try sharpest face first
    best = s.exec(
        select(M.FaceDetection.id, M.FaceDetection.face_sharpness)
        .where(M.FaceDetection.person_id == person_id)
        .where(M.FaceDetection.x1.isnot(None))  # must have normalised bbox for crop endpoint
        .order_by(M.FaceDetection.face_sharpness.desc())
        .limit(1)
    ).first()
    return best[0] if best else None


class ActionError(ValueError):
    """A refusal with a message fit to show the user and the HTTP status the web app should use."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _as_dict(p: M.PersonCluster, sample_face_id: int | None = None, similar_to: dict | None = None) -> dict:
    d = {"id": p.id, "name": p.name, "face_count": p.face_count, "sample_face_id": sample_face_id}
    if similar_to is not None:
        d["similar_to"] = similar_to
    return d


def _arrange_by_similarity(clusters: list[M.PersonCluster]) -> list[tuple[M.PersonCluster, dict | None]]:
    """Arrange person clusters so that visually similar faces are grouped next to each other.

    Returns a list of tuples: (cluster, match_info), where match_info is either None or
    {'target_id': int, 'target_name': str | None, 'score': float}.
    """
    if len(clusters) <= 1:
        return [(c, None) for c in clusters]

    import numpy as np

    with_emb = []
    without_emb = []
    for c in clusters:
        vec = dequantize_embedding(c.centroid_json) if c.centroid_json else None
        if vec and len(vec) > 0:
            arr = np.asarray(vec, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if norm > 1e-6:
                with_emb.append((c, arr / norm))
                continue
        without_emb.append(c)

    without_emb.sort(key=lambda p: (p.face_count, p.id or 0), reverse=True)

    if not with_emb:
        return [(c, None) for c in without_emb]

    # Pre-sort with_emb by prominence: named first, then most faces
    with_emb.sort(key=lambda x: (x[0].name is not None, x[0].face_count, x[0].id or 0), reverse=True)

    n = len(with_emb)
    dim = len(with_emb[0][1])
    if all(len(v) == dim for _, v in with_emb):
        vectors = np.stack([v for _, v in with_emb])
        sim_matrix = np.dot(vectors, vectors.T)
    else:
        sim_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i, n):
                if len(with_emb[i][1]) == len(with_emb[j][1]):
                    s = float(np.dot(with_emb[i][1], with_emb[j][1]))
                    sim_matrix[i, j] = sim_matrix[j, i] = s

    unvisited = set(range(n))
    ordered: list[tuple[int, dict | None]] = []

    while unvisited:
        # Pick the most prominent unvisited node as seed
        seed = min(unvisited)
        unvisited.remove(seed)
        ordered.append((seed, None))

        group = [seed]
        while unvisited:
            best_cand = None
            best_score = -1.0
            best_ref = None

            for cand in unvisited:
                cand_scores = sim_matrix[cand, group]
                max_idx = int(np.argmax(cand_scores))
                score = float(cand_scores[max_idx])
                if score > best_score:
                    best_score = score
                    best_cand = cand
                    best_ref = group[max_idx]

            # Threshold for linking as similar face: 0.40
            if best_cand is not None and best_score >= 0.40:
                unvisited.remove(best_cand)
                match_info = {
                    "target_id": with_emb[best_ref][0].id,
                    "target_name": with_emb[best_ref][0].name,
                    "score": round(best_score, 2),
                }
                ordered.append((best_cand, match_info))
                group.append(best_cand)
            else:
                break

    result = []
    for idx, match in ordered:
        result.append((with_emb[idx][0], match))
    for c in without_emb:
        result.append((c, None))
    return result


def list_people(db_path: Path | str, sort: str = "similarity") -> list[dict]:
    with get_session(db_path) as s:
        repo = PhotoRepository(s)
        clusters = repo.all_person_clusters()

        if sort == "name":
            # Named first alphabetically, then unnamed by count desc
            clusters.sort(key=lambda p: (0 if p.name else 1, (p.name or "").lower(), -p.face_count))
            return [_as_dict(p, _best_face_id_for(p.id, s)) for p in clusters]
        elif sort == "count":
            clusters.sort(key=lambda p: (p.face_count, p.id or 0), reverse=True)
            return [_as_dict(p, _best_face_id_for(p.id, s)) for p in clusters]
        else:  # default: "similarity"
            arranged = _arrange_by_similarity(clusters)
            return [_as_dict(p, _best_face_id_for(p.id, s), match) for p, match in arranged]


def rename_person(db_path: Path | str, person_id: int, name: str) -> dict:
    name = name.strip()
    if not name:
        raise ActionError("name must not be blank")
    with get_session(db_path) as s:
        repo = PhotoRepository(s)
        p = s.get(M.PersonCluster, person_id)
        if p is None:
            raise ActionError(f"no person with id {person_id}", 404)
        existing = repo.find_person_by_name(name)
        if existing is not None and existing.id != person_id:
            faces = s.exec(select(M.FaceDetection).where(M.FaceDetection.person_id == person_id)).all()
            for f in faces:
                f.person_id = existing.id
            existing.face_count += p.face_count
            s.delete(p)
            s.flush()
            return _as_dict(existing)
        p = repo.rename_person(person_id, name)
        return _as_dict(p)


def reassign_face(db_path: Path | str, face_id: int, *, person_id: int | None = None,
                  new_name: str | None = None) -> dict:
    if (person_id is None) == (new_name is None):
        raise ActionError("exactly one of person_id or new_name is required")
    if new_name is not None:
        new_name = new_name.strip()
        if not new_name:
            raise ActionError("new_name must not be blank")
    with get_session(db_path) as s:
        repo = PhotoRepository(s)
        face = s.get(M.FaceDetection, face_id)
        if face is None:
            raise ActionError(f"no face with id {face_id}", 404)
        old_person_id = face.person_id
        created_fresh = False
        if person_id is not None:
            target = s.get(M.PersonCluster, person_id)
            if target is None:
                raise ActionError(f"no person with id {person_id}", 404)
        else:
            target = repo.find_person_by_name(new_name)
            if target is None:
                embedding = dequantize_embedding(face.embedding_json) or []
                target = repo.create_person_cluster(embedding)
                target = repo.rename_person(target.id, new_name)
                created_fresh = True

        if old_person_id == target.id:
            return {"id": face.id, "person_id": target.id, "person_name": target.name}

        face.person_id = target.id
        if old_person_id is not None:
            old = s.get(M.PersonCluster, old_person_id)
            if old is not None:
                old.face_count = max(0, old.face_count - 1)
        if not created_fresh:
            target.face_count += 1
        s.flush()
        return {"id": face.id, "person_id": target.id, "person_name": target.name}


def merge_people(db_path: Path | str, source_id: int, target_id: int) -> dict:
    """Merge all faces from *source* cluster into *target* cluster, then delete *source*.

    Returns the updated target dict (with fresh sample_face_id).
    """
    if source_id == target_id:
        raise ActionError("cannot merge a cluster into itself")
    with get_session(db_path) as s:
        source = s.get(M.PersonCluster, source_id)
        target = s.get(M.PersonCluster, target_id)
        if source is None:
            raise ActionError(f"no person with id {source_id}", 404)
        if target is None:
            raise ActionError(f"no person with id {target_id}", 404)

        # Re-parent every face from source → target
        faces = s.exec(select(M.FaceDetection).where(M.FaceDetection.person_id == source_id)).all()
        for f in faces:
            f.person_id = target_id
        target.face_count += len(faces)
        s.delete(source)
        s.flush()

        sample_face_id = _best_face_id_for(target_id, s)
        return _as_dict(target, sample_face_id)


def delete_person(db_path: Path | str, person_id: int) -> dict:
    """Delete a person cluster and unassign all its associated faces."""
    with get_session(db_path) as s:
        p = s.get(M.PersonCluster, person_id)
        if p is None:
            raise ActionError(f"no person with id {person_id}", 404)
        faces = s.exec(select(M.FaceDetection).where(M.FaceDetection.person_id == person_id)).all()
        for f in faces:
            f.person_id = None
        s.delete(p)
        s.flush()
        return {"id": person_id, "deleted": True}


