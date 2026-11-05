"""People actions: list/rename clusters and reassign a single face. Shared by the web app and MCP."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlmodel import col, func, select

from .db import models as M
from .db.repository import (PhotoRepository, decode_yearly, dequantize_embedding, encode_yearly,
                            quantize_embedding, stage_clause)
from .db.session import get_session
from .clustering.face_identity import effective_threshold, update_exemplars, update_temporal_medoid


def _cluster_stage(stage: str) -> str:
    """People made inside Trash are kept apart; New and Library share one set of people."""
    return "trash" if stage == "trash" else "library"


def _best_face_id_for(person_id: int, s, stage: str = "all") -> int | None:
    """Return the face ID with the highest sharpness for a given person cluster, or any face ID."""
    # Try sharpest face first
    q = (
        select(M.FaceDetection.id, M.FaceDetection.face_sharpness)
        .join(M.ImageRecord, M.ImageRecord.id == M.FaceDetection.image_id)
        .where(M.FaceDetection.person_id == person_id)
        .where(M.FaceDetection.ignored == False)
        .where(stage_clause(stage))
        .where(M.FaceDetection.x1.isnot(None))  # must have normalised bbox for crop endpoint
    )
    best = s.exec(
        q.order_by(M.FaceDetection.face_sharpness.desc())
        .limit(1)
    ).first()
    return best[0] if best else None


class ActionError(ValueError):
    """A refusal with a message fit to show the user and the HTTP status the web app should use."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _as_dict(p: M.PersonCluster, sample_face_id: int | None = None, similar_to: dict | None = None) -> dict:
    import hashlib
    d = {"id": p.id, "name": p.name, "face_count": p.face_count, "sample_face_id": sample_face_id}
    if sample_face_id is not None:
        d["face_v"] = hashlib.sha256(f"{p.id}:{p.face_count}:{sample_face_id}".encode()).hexdigest()[:16]
    if similar_to is not None:
        d["similar_to"] = similar_to
    return d


def _unit_centroids(clusters: list[M.PersonCluster]) -> list[tuple[M.PersonCluster, "np.ndarray"]]:
    import numpy as np
    out = []
    for c in clusters:
        vec = dequantize_embedding(c.centroid_json) if c.centroid_json else None
        if vec and len(vec) > 0:
            arr = np.asarray(vec, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if norm > 1e-6:
                out.append((c, arr / norm))
    return out


def _best_face_ids_for(person_ids: list[int], s, stage: str = "all") -> dict[int, int]:
    """Sharpest usable face per person in one query per 900 ids."""
    best: dict[int, int] = {}
    ids = [i for i in person_ids if i is not None]
    for k in range(0, len(ids), 900):
        rows = s.exec(
            select(M.FaceDetection.person_id, M.FaceDetection.id)
            .join(M.ImageRecord, M.ImageRecord.id == M.FaceDetection.image_id)
            .where(col(M.FaceDetection.person_id).in_(ids[k:k + 900]))
            .where(M.FaceDetection.ignored == False)  # noqa: E712
            .where(stage_clause(stage))
            .where(M.FaceDetection.x1.isnot(None))
            .order_by(M.FaceDetection.face_sharpness.desc())
        ).all()
        for pid, fid in rows:
            best.setdefault(pid, fid)
    return best


def _cluster_image_sets(s) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for pid, iid in s.exec(select(M.FaceDetection.person_id, M.FaceDetection.image_id)
                           .where(M.FaceDetection.person_id.is_not(None))
                           .where(M.FaceDetection.ignored == False)).all():  # noqa: E712
        out.setdefault(pid, set()).add(iid)
    return out


MERGE_NAMED_MARGIN = 0.05  # a named target wins only when within this of the best score overall


def _find_merge_matches(clusters: list[M.PersonCluster], s=None, stage: str = "all",
                        cluster_images: dict[int, set[int]] | None = None) -> dict[int, dict]:
    """For every unnamed cluster, the best cluster to suggest merging into (one matmul per dimension).

    Named targets are preferred, but only within ``MERGE_NAMED_MARGIN`` of the strongest match;
    clusters that share a photo (two people in one frame) are never suggested to each other.
    """
    import numpy as np
    by_dim: dict[int, list] = {}
    for c, v in _unit_centroids(clusters):
        by_dim.setdefault(len(v), []).append((c, v))
    images = cluster_images or {}
    pending: dict[int, tuple[M.PersonCluster, float]] = {}

    def shares_photo(a: M.PersonCluster, b: M.PersonCluster) -> bool:
        ia, ib = images.get(a.id), images.get(b.id)
        return bool(ia and ib and (ia & ib))

    def pick(row: "np.ndarray", cand: M.PersonCluster, items: list, thresh: float) -> tuple[int | None, float]:
        row = row.copy()
        while True:
            j = int(np.argmax(row))
            score = float(row[j])
            if score < thresh:
                return None, -1.0
            if shares_photo(cand, items[j][0]):
                row[j] = -1.0
                continue
            return j, score

    for dim, items in by_dim.items():
        if len(items) < 2:
            continue
        vectors = np.stack([v for _, v in items])
        sims = vectors @ vectors.T
        np.fill_diagonal(sims, -1.0)
        thresh = 0.35 if dim >= 512 else 0.65
        named = np.array([bool(c.name) for c, _ in items])
        counts = np.array([int(c.face_count or 0) for c, _ in items])
        sims[:, counts <= 0] = -1.0  # a person with no faces is not a merge target
        for i, (cand, _) in enumerate(items):
            if cand.name:
                continue  # named people never show incoming merge pills on themselves
            jn, sn = pick(np.where(named, sims[i], -1.0), cand, items, thresh)
            ju, su = pick(np.where(~named & (counts > cand.face_count), sims[i], -1.0), cand, items, thresh)
            if jn is not None and sn >= su - MERGE_NAMED_MARGIN:
                pending[cand.id] = (items[jn][0], sn)
            elif ju is not None:
                pending[cand.id] = (items[ju][0], su)

    faces = _best_face_ids_for([t.id for t, _ in pending.values()], s, stage) if s is not None else {}
    return {cid: {"target_id": target.id, "target_name": target.name,
                  "target_sample_face_id": faces.get(target.id), "score": round(score, 2),
                  "shared_photos": 0}
            for cid, (target, score) in pending.items()}


def _arrange_by_similarity(clusters: list[M.PersonCluster]) -> list[M.PersonCluster]:
    """Order clusters so visually similar people sit next to each other (no DB, no match info)."""
    if len(clusters) <= 1:
        return list(clusters)
    import numpy as np
    with_emb = _unit_centroids(clusters)
    have = {id(c) for c, _ in with_emb}
    without_emb = sorted((c for c in clusters if id(c) not in have),
                         key=lambda p: (p.face_count, p.id or 0), reverse=True)
    if not with_emb:
        return without_emb
    with_emb.sort(key=lambda x: (x[0].name is not None, x[0].face_count, x[0].id or 0), reverse=True)
    n = len(with_emb)
    dims = np.array([len(v) for _, v in with_emb])
    sims = np.full((n, n), -1.0, dtype=np.float32)
    for dim in np.unique(dims):
        idx = np.flatnonzero(dims == dim)
        vectors = np.stack([with_emb[i][1] for i in idx])
        sims[np.ix_(idx, idx)] = vectors @ vectors.T
    thresholds = np.where(dims >= 512, 0.35, 0.65).astype(np.float32)
    unvisited = np.ones(n, dtype=bool)
    order: list[int] = []
    while unvisited.any():
        seed = int(np.argmax(unvisited))  # most prominent remaining
        unvisited[seed] = False
        order.append(seed)
        best = sims[seed].copy()  # best similarity of every node to the growing group
        while unvisited.any():
            masked = np.where(unvisited, best, -2.0)
            cand = int(np.argmax(masked))
            if float(masked[cand]) < float(thresholds[cand]):
                break
            unvisited[cand] = False
            order.append(cand)
            best = np.maximum(best, sims[cand])
    return [with_emb[i][0] for i in order] + without_emb


def list_people(db_path: Path | str, sort: str = "similarity", stage: str = "all") -> list[dict]:
    with get_session(db_path) as s:
        # Prune any empty unnamed ghost clusters
        has_any_face = s.exec(select(M.FaceDetection.id).limit(1)).first() is not None
        if has_any_face:
            active_pids = select(M.FaceDetection.person_id).where(
                M.FaceDetection.person_id.is_not(None),
                M.FaceDetection.ignored == False,
            )
            empty_unnamed = s.exec(
                select(M.PersonCluster)
                .where(M.PersonCluster.name.is_(None))
                .where((M.PersonCluster.face_count <= 0) | col(M.PersonCluster.id).not_in(active_pids))
            ).all()
        else:
            empty_unnamed = s.exec(
                select(M.PersonCluster)
                .where(M.PersonCluster.face_count <= 0)
                .where(M.PersonCluster.name.is_(None))
            ).all()
        if empty_unnamed:
            for p in empty_unnamed:
                s.delete(p)
            s.commit()

        repo = PhotoRepository(s)
        # Compute scoped face counts
        if stage in ("new", "library", "trash"):
            counts_q = (
                select(M.FaceDetection.person_id, func.count(M.FaceDetection.id))
                .join(M.ImageRecord, M.ImageRecord.id == M.FaceDetection.image_id)
                .where(M.FaceDetection.person_id.is_not(None))
                .where(M.FaceDetection.ignored == False)
                .where(stage_clause(stage))
                .group_by(M.FaceDetection.person_id)
            )
            scoped_counts = dict(s.exec(counts_q).all())

            clusters = []
            for c in repo.all_person_clusters():
                if c.stage != _cluster_stage(stage):
                    continue
                cnt = scoped_counts.get(c.id, 0)
                if cnt > 0:
                    c.face_count = cnt
                    clusters.append(c)
        else:
            # stage == "all": use cluster's face_count, but sync with active faces if FaceDetections exist
            library_clusters = [c for c in repo.all_person_clusters() if c.stage != "trash"]
            if has_any_face:
                counts_q = (
                    select(M.FaceDetection.person_id, func.count(M.FaceDetection.id))
                    .join(M.ImageRecord, M.ImageRecord.id == M.FaceDetection.image_id)
                    .where(M.FaceDetection.person_id.is_not(None))
                    .where(M.FaceDetection.ignored == False)
                    .where(M.ImageRecord.scan_status != "missing")
                    .group_by(M.FaceDetection.person_id)
                )
                scoped_counts = dict(s.exec(counts_q).all())
                clusters = []
                for c in library_clusters:
                    cnt = scoped_counts.get(c.id, 0)
                    if cnt > 0:
                        c.face_count = cnt
                        clusters.append(c)
                    elif c.name:
                        c.face_count = 0
                        clusters.append(c)
            else:
                clusters = [c for c in library_clusters if c.face_count > 0]

        matches = _find_merge_matches(clusters, s, stage=stage, cluster_images=_cluster_image_sets(s))
        faces = _best_face_ids_for([c.id for c in clusters], s, stage)
        if sort == "name":
            # Named first alphabetically, then unnamed by count desc
            ordered = sorted(clusters, key=lambda p: (0 if p.name else 1, (p.name or "").lower(), -p.face_count))
        elif sort == "count":
            ordered = sorted(clusters, key=lambda p: (p.face_count, p.id or 0), reverse=True)
        else:  # default: "similarity"
            ordered = _arrange_by_similarity(clusters)
        items = [_as_dict(p, faces.get(p.id), matches.get(p.id)) for p in ordered]
        return [it for it in items if it["face_count"] > 0]


def rename_person(db_path: Path | str, person_id: int, name: str) -> dict:
    name = name.strip()
    if not name:
        raise ActionError("name must not be blank")
    with get_session(db_path) as s:
        repo = PhotoRepository(s)
        p = s.get(M.PersonCluster, person_id)
        if p is None:
            raise ActionError(f"no person with id {person_id}", 404)
        existing = repo.find_person_by_name(name, stage=p.stage)  # never merge across library and trash
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
                  new_name: str | None = None, separate: bool = False) -> dict:
    if separate:
        if person_id is not None or new_name is not None:
            raise ActionError("when separate is true, person_id and new_name must be omitted")
    elif (person_id is None) == (new_name is None):
        raise ActionError("exactly one of person_id, new_name, or separate=True is required")
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
        if separate:
            image = s.get(M.ImageRecord, face.image_id)
            person_stage = _cluster_stage(image.stage if image is not None else "new")
            embedding = dequantize_embedding(face.embedding_json) or []
            target = repo.create_person_cluster(embedding, stage=person_stage)
            created_fresh = True
        elif person_id is not None:
            target = s.get(M.PersonCluster, person_id)
            if target is None:
                raise ActionError(f"no person with id {person_id}", 404)
        else:
            image = s.get(M.ImageRecord, face.image_id)
            person_stage = _cluster_stage(image.stage if image is not None else "new")
            target = repo.find_person_by_name(new_name, stage=person_stage)
            if target is None:
                embedding = dequantize_embedding(face.embedding_json) or []
                target = repo.create_person_cluster(embedding, stage=person_stage)
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


def delete_person(db_path: Path | str, person_id: int, stage: str = "all") -> dict:
    """Delete a person cluster and mark its faces ignored (limited to one stage when given)."""
    with get_session(db_path) as s:
        p = s.get(M.PersonCluster, person_id)
        if p is None:
            raise ActionError(f"no person with id {person_id}", 404)
        if stage in ("new", "library", "trash"):
            q = (
                select(M.FaceDetection)
                .join(M.ImageRecord, M.ImageRecord.id == M.FaceDetection.image_id)
                .where(M.FaceDetection.person_id == person_id)
                .where(M.ImageRecord.stage == stage)
            )
            pipeline_faces = s.exec(q).all()
            for f in pipeline_faces:
                f.ignored = True
                f.person_id = None
            rem = s.exec(
                select(func.count(M.FaceDetection.id))
                .where(M.FaceDetection.person_id == person_id, M.FaceDetection.ignored == False)
            ).one()
            p.face_count = int(rem or 0)
            if p.face_count <= 0 and not p.name:
                s.delete(p)
            s.commit()
            return {"id": person_id, "deleted": True, "scoped": stage}

        faces = s.exec(select(M.FaceDetection).where(M.FaceDetection.person_id == person_id)).all()
        for f in faces:
            f.ignored = True
            f.person_id = None
        s.delete(p)
        s.commit()
        return {"id": person_id, "deleted": True}


def bulk_delete_people(db_path: Path | str, person_ids: list[int], stage: str = "all") -> dict:
    """Delete multiple person clusters and mark their faces ignored in a single transaction."""
    if not person_ids:
        return {"deleted_count": 0}
    with get_session(db_path) as s:
        if stage in ("new", "library", "trash"):
            q = (
                select(M.FaceDetection)
                .join(M.ImageRecord, M.ImageRecord.id == M.FaceDetection.image_id)
                .where(col(M.FaceDetection.person_id).in_(person_ids))
                .where(M.ImageRecord.stage == stage)
            )
            pipeline_faces = s.exec(q).all()
            for f in pipeline_faces:
                f.ignored = True
                f.person_id = None
            clusters = s.exec(
                select(M.PersonCluster)
                .where(col(M.PersonCluster.id).in_(person_ids))
            ).all()
            count = len(clusters)
            for c in clusters:
                rem = s.exec(
                    select(func.count(M.FaceDetection.id))
                    .where(M.FaceDetection.person_id == c.id, M.FaceDetection.ignored == False)
                ).one()
                c.face_count = int(rem or 0)
                if c.face_count <= 0 and not c.name:
                    s.delete(c)
            s.commit()
            return {"deleted_count": count, "scoped": stage}

        faces = s.exec(
            select(M.FaceDetection)
            .where(col(M.FaceDetection.person_id).in_(person_ids))
        ).all()
        for f in faces:
            f.ignored = True
            f.person_id = None

        clusters = s.exec(
            select(M.PersonCluster)
            .where(col(M.PersonCluster.id).in_(person_ids))
        ).all()
        count = len(clusters)
        for c in clusters:
            s.delete(c)
        s.commit()
        return {"deleted_count": count}


def consolidate_clusters(db_path: Path | str, threshold: float = 0.65, stage: str = "new") -> dict:
    """Post-clustering consolidation: automatically merge visually convergent clusters.

    When multiple clusters have drifted close (cosine similarity >= threshold) and have NO
    overlapping photos (different people cannot be in the same photo as the same person),
    they are safely merged into the more prominent cluster (named or higher face_count).
    Named clusters with conflicting custom names are never auto-merged.
    """
    import numpy as np

    with get_session(db_path) as s:
        cluster_stage = _cluster_stage(stage)
        clusters = s.exec(
            select(M.PersonCluster)
            .where(M.PersonCluster.stage == cluster_stage)
            .where(M.PersonCluster.face_count > 0)
        ).all()

        if len(clusters) <= 1:
            return {"merged_count": 0}

        # Map cluster_id -> set of image_ids
        cluster_images: dict[int, set[int]] = {}
        cluster_centroids: dict[int, np.ndarray] = {}
        for c in clusters:
            if not c.centroid_json:
                continue
            emb = dequantize_embedding(c.centroid_json)
            if not emb:
                continue
            arr = np.asarray(emb, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if norm < 1e-6:
                continue
            cluster_centroids[c.id] = arr / norm
            img_ids = set(s.exec(
                select(M.FaceDetection.image_id)
                .where(M.FaceDetection.person_id == c.id)
                .where(M.FaceDetection.ignored == False)
            ).all())
            cluster_images[c.id] = img_ids

        # Sort clusters by prominence: named first, then higher face_count
        sorted_clusters = sorted(
            [c for c in clusters if c.id in cluster_centroids],
            key=lambda c: (1 if c.name else 0, c.face_count or 0, c.id or 0),
            reverse=True,
        )

        merged_count = 0
        deleted_ids: set[int] = set()

        for i, target in enumerate(sorted_clusters):
            if target.id in deleted_ids:
                continue
            target_emb = cluster_centroids[target.id]
            target_imgs = cluster_images[target.id]

            for j in range(i + 1, len(sorted_clusters)):
                cand = sorted_clusters[j]
                if cand.id in deleted_ids:
                    continue
                # Never auto-merge if candidate has a different custom name
                if cand.name and cand.name != target.name:
                    continue

                # Check same-photo exclusivity: cannot merge if they share any photo
                cand_imgs = cluster_images[cand.id]
                if target_imgs.intersection(cand_imgs):
                    continue

                if len(target_emb) != len(cluster_centroids[cand.id]):
                    continue
                sim = float(np.dot(target_emb, cluster_centroids[cand.id]))
                eff_threshold = effective_threshold(threshold, len(target_emb))
                if sim >= eff_threshold:
                    # Merge candidate into target
                    faces = s.exec(select(M.FaceDetection).where(M.FaceDetection.person_id == cand.id)).all()
                    for f in faces:
                        f.person_id = target.id
                    target.face_count = (target.face_count or 0) + len(faces)
                    # Adopt candidate's name if target was unnamed
                    if not target.name and cand.name:
                        target.name = cand.name

                    # Update target centroid
                    cand_emb = cluster_centroids[cand.id]
                    tot = target.face_count
                    n_cand = len(faces)
                    n_prev = max(1, tot - n_cand)
                    if len(target_emb) == len(cand_emb):
                        new_cent = (target_emb * n_prev + cand_emb * n_cand) / max(1, tot)
                        norm_new = float(np.linalg.norm(new_cent))
                        if norm_new > 1e-6:
                            target_emb = new_cent / norm_new
                            cluster_centroids[target.id] = target_emb
                            target.centroid_json = quantize_embedding(target_emb.tolist())

                    target_imgs.update(cand_imgs)
                    deleted_ids.add(cand.id)
                    s.delete(cand)
                    merged_count += 1

        s.commit()
        return {"merged_count": merged_count}


def recluster_people(db_path: Path | str, threshold: float | None = None, stage: str = "new") -> dict:
    """Recluster faces on New Pipeline photos while anchoring on existing named/library PersonCluster centroids.

    Uses two-stage clustering (core discovery + satellite absorption) with multi-exemplar matching
    and temporal/burst co-occurrence boost, followed by post-clustering consolidation.

    - Preserves existing named clusters and Library photos.
    - Excludes ignored faces.
    - Enforces same-photo exclusivity.
    """
    import bisect
    import json
    from collections import Counter
    import numpy as np

    with get_session(db_path) as s:
        # 1. Fetch all non-ignored faces on photos in the specified stage with valid embeddings
        faces = s.exec(
            select(M.FaceDetection)
            .join(M.ImageRecord, M.ImageRecord.id == M.FaceDetection.image_id)
            .where(M.FaceDetection.embedding_json.isnot(None))
            .where(M.FaceDetection.ignored == False)
            .where(stage_clause(stage))
        ).all()

        if not faces:
            return {"clusters_created": 0, "faces_clustered": 0}

        # Fetch image timestamps and burst_id for temporal co-occurrence boost
        face_img_ids = list({f.image_id for f in faces})
        img_records = s.exec(
            select(M.ImageRecord)
            .where(col(M.ImageRecord.id).in_(face_img_ids))
        ).all()
        img_info = {
            r.id: {
                "burst_id": getattr(r, "burst_id", None),
                "timestamp": r.taken_at.timestamp() if getattr(r, "taken_at", None) else None,
            }
            for r in img_records
        }

        # Keep existing custom names mapped to old person_id
        cluster_stage = _cluster_stage(stage)
        old_clusters = s.exec(select(M.PersonCluster).where(M.PersonCluster.stage == cluster_stage)).all()
        custom_names = {c.id: c.name for c in old_clusters if c.name}

        # Seed anchored clusters from existing clusters with centroids
        clusters: list[dict] = []
        anchored_cluster_ids = set()
        for c in old_clusters:
            if not c.centroid_json or stage == "trash":  # trash people are rebuilt from scratch
                continue
            # Only anchor on clusters that have faces in finalized Library photos
            lib_face_cnt = s.exec(
                select(func.count(M.FaceDetection.id))
                .join(M.ImageRecord, M.ImageRecord.id == M.FaceDetection.image_id)
                .where(M.FaceDetection.person_id == c.id)
                .where(M.FaceDetection.ignored == False)
                .where(M.ImageRecord.stage == "library")
            ).one()
            if not lib_face_cnt or lib_face_cnt <= 0:
                continue
            try:
                emb = dequantize_embedding(c.centroid_json)
                if not emb:
                    continue
                arr = np.asarray(emb, dtype=np.float32)
                norm = float(np.linalg.norm(arr))
                if norm < 1e-6:
                    continue
                u = arr / norm
            except Exception:
                continue

            # Find image_ids for this cluster that are NOT in pipeline (anchored library photos)
            anchored_img_ids = set(s.exec(
                select(M.FaceDetection.image_id)
                .join(M.ImageRecord, M.ImageRecord.id == M.FaceDetection.image_id)
                .where(M.FaceDetection.person_id == c.id)
                .where(M.ImageRecord.stage == "library")
            ).all())

            y_meds, y_counts = decode_yearly(c.yearly_json)
            clusters.append({
                "cluster_obj": c,
                "centroid": u,
                "exemplars": [u],
                "face_ids": [],
                "image_ids": anchored_img_ids,
                "names": [c.name] if c.name else [],
                "is_existing": True,
                "anchor_count": int(lib_face_cnt),  # library faces behind the anchored centroid
                "anchor_centroid": u,
                "yearly_medoids": y_meds,
                "yearly_counts": y_counts,
            })
            anchored_cluster_ids.add(c.id)

        # Clear old pipeline clusters that have no Library faces
        for c in old_clusters:
            if c.id not in anchored_cluster_ids:
                if stage != "trash" and c.name and s.exec(
                        select(func.count(M.FaceDetection.id))
                        .join(M.ImageRecord, M.ImageRecord.id == M.FaceDetection.image_id)
                        .where(M.FaceDetection.person_id == c.id)
                        .where(M.ImageRecord.stage == "library")).one():
                    continue  # a person the user named keeps their finished photos, centroid or not
                s.delete(c)
        s.flush()

        # Map face_id -> old custom name if any
        face_custom_name = {f.id: custom_names[f.person_id] for f in faces if f.person_id in custom_names}

        # Helper: assess if a face is suitable as a "core" seed (frontal, sharp, unblinked)
        # Note: A face is treated as satellite only if it has severe profile yaw or eyes closed.
        def is_core_face(f) -> bool:
            # Check landmarks if available (5 points: left eye, right eye, nose, left mouth, right mouth)
            if f.landmarks:
                try:
                    lm = json.loads(f.landmarks) if isinstance(f.landmarks, str) else f.landmarks
                    if len(lm) >= 5:
                        le, re, nose = lm[0], lm[1], lm[2]
                        # Yaw symmetry: nose should be between eyes horizontally
                        eye_w = abs(re[0] - le[0])
                        if eye_w > 1e-4:
                            yaw_ratio = abs((nose[0] - le[0]) - (re[0] - nose[0])) / eye_w
                            if yaw_ratio > 0.65:  # severe profile shot
                                return False
                except Exception:
                    pass
            return True

        # Pre-process embeddings
        parsed_faces = []
        for f in faces:
            try:
                emb = dequantize_embedding(f.embedding_json)
                if not emb:
                    continue
                arr = np.asarray(emb, dtype=np.float32)
                norm = float(np.linalg.norm(arr))
                if norm < 1e-6:
                    continue
                u = arr / norm
                parsed_faces.append((f, u))
            except Exception:
                continue

        # Pass 1: Core faces (sharp, frontal) sorted by sharpness descending
        # Pass 2: Satellite faces (profile, angled, lower sharpness)
        core_faces = [item for item in parsed_faces if is_core_face(item[0])]
        satellite_faces = [item for item in parsed_faces if not is_core_face(item[0])]

        order_key = lambda item: (-(item[0].face_sharpness or 0.0), item[0].id or 0)  # noqa: E731
        core_faces.sort(key=order_key)
        satellite_faces.sort(key=order_key)

        def cluster_time(c: dict) -> tuple[set, list]:
            """Burst ids and sorted timestamps of a cluster's photos, built once and kept current."""
            if "bursts" not in c:
                c["bursts"], c["ts"] = set(), []
                for img_id in c["image_ids"]:
                    _note_time(c, img_info.get(img_id, {}))
            return c["bursts"], c["ts"]

        def _note_time(c: dict, meta: dict) -> None:
            if meta.get("burst_id") is not None:
                c["bursts"].add(meta["burst_id"])
            if meta.get("timestamp") is not None:
                bisect.insort(c["ts"], meta["timestamp"])

        def nearest_dt(ts_sorted: list, t: float) -> float | None:
            if not ts_sorted:
                return None
            k = bisect.bisect_left(ts_sorted, t)
            cands = [abs(ts_sorted[j] - t) for j in (k - 1, k) if 0 <= j < len(ts_sorted)]
            return min(cands) if cands else None

        def assign_face_item(f, u, allow_new_cluster: bool):
            dim = len(u)
            base_thresh = effective_threshold(threshold, dim)

            best_c = None
            best_sim = base_thresh
            f_meta = img_info.get(f.image_id, {})
            f_burst = f_meta.get("burst_id")
            f_ts = f_meta.get("timestamp")
            f_year = str(datetime.fromtimestamp(f_ts).year) if f_ts else None

            for c in clusters:
                # Dimension guard
                if len(u) != len(c["centroid"]):
                    continue

                # 1. Co-occurrence Veto (cannot-link constraint): siblings in same photo NEVER merge
                if f.image_id in c["image_ids"]:
                    continue

                # 2. Score against mean centroid
                sim = float(np.dot(u, c["centroid"]))

                # 3. Multi-exemplar matching
                for ex in c.get("exemplars", []):
                    if len(ex) == len(u):
                        ex_sim = float(np.dot(u, ex))
                        if ex_sim > sim:
                            sim = ex_sim

                # 4. Chronological medoid chaining (growth bridging across years)
                c_medoids = c.setdefault("yearly_medoids", {})
                if f_year and c_medoids:
                    for y_key in [f_year, str(int(f_year) - 1), str(int(f_year) + 1)]:
                        if y_key in c_medoids and len(c_medoids[y_key]) == len(u):
                            med_sim = float(np.dot(u, c_medoids[y_key]))
                            if med_sim > sim:
                                sim = med_sim
                elif c_medoids:
                    for y_med in c_medoids.values():
                        if len(y_med) == len(u):
                            med_sim = float(np.dot(u, y_med))
                            if med_sim > sim:
                                sim = med_sim

                # 5. Burst / seconds-apart tie-break: never enough to cross the threshold on its own
                c_bursts, c_ts = cluster_time(c)
                if f_burst and f_burst in c_bursts:
                    sim += 0.02
                elif f_ts is not None:
                    min_dt = nearest_dt(c_ts, f_ts)
                    if min_dt is not None and min_dt <= 120.0:
                        sim += 0.01

                if sim >= best_sim:
                    best_sim = sim
                    best_c = c

            c_name = face_custom_name.get(f.id)
            if best_c is not None:
                best_c["face_ids"].append(f.id)
                best_c["image_ids"].add(f.image_id)
                if "bursts" in best_c:
                    _note_time(best_c, f_meta)
                if c_name:
                    best_c["names"].append(c_name)
                # Update running centroid (pipeline faces re-clustered here plus the anchored library faces)
                n = len(best_c["face_ids"]) + best_c.get("anchor_count", 0)
                if len(best_c["centroid"]) == len(u):
                    new_cent = (best_c["centroid"] * max(1, n - 1) + u) / max(1, n)
                    best_c["centroid"] = new_cent / float(np.linalg.norm(new_cent))
                else:
                    best_c["centroid"] = u
                best_c["exemplars"] = [np.asarray(ex, dtype=np.float32) for ex in update_exemplars(
                    best_c.get("exemplars", []), u, best_c["centroid"])]
                best_c["yearly_medoids"], best_c["yearly_counts"] = update_temporal_medoid(
                    best_c.get("yearly_medoids"), f_year, u.tolist(), best_c.get("yearly_counts"))
            elif allow_new_cluster:
                y_meds, y_counts = update_temporal_medoid({}, f_year, u.tolist(), {})
                clusters.append({
                    "cluster_obj": None,
                    "centroid": u,
                    "exemplars": [u],
                    "face_ids": [f.id],
                    "image_ids": {f.image_id},
                    "names": [c_name] if c_name else [],
                    "is_existing": False,
                    "yearly_medoids": y_meds,
                    "yearly_counts": y_counts,
                })
            else:
                # If satellite face didn't match any core at threshold, assign to best non-conflicting cluster
                # if reasonably close (>= 0.45), rather than seeding an isolated singleton cluster
                if best_c is None and clusters:
                    fallback_c = None
                    fallback_sim = 0.45
                    for c in clusters:
                        if len(u) != len(c["centroid"]):
                            continue
                        if f.image_id in c["image_ids"]:
                            continue
                        sim = float(np.dot(u, c["centroid"]))
                        for ex in c.get("exemplars", []):
                            if len(ex) == len(u):
                                ex_sim = float(np.dot(u, ex))
                                if ex_sim > sim:
                                    sim = ex_sim
                        if sim >= fallback_sim:
                            fallback_sim = sim
                            fallback_c = c
                    if fallback_c is not None:
                        fallback_c["face_ids"].append(f.id)
                        fallback_c["image_ids"].add(f.image_id)
                        if "bursts" in fallback_c:
                            _note_time(fallback_c, f_meta)
                        if c_name:
                            fallback_c["names"].append(c_name)
                    else:
                        # Otherwise allow creating a new cluster
                        y_meds, y_counts = update_temporal_medoid({}, f_year, u.tolist(), {})
                        clusters.append({
                            "cluster_obj": None,
                            "centroid": u,
                            "exemplars": [u],
                            "face_ids": [f.id],
                            "image_ids": {f.image_id},
                            "names": [c_name] if c_name else [],
                            "is_existing": False,
                            "yearly_medoids": y_meds,
                            "yearly_counts": y_counts,
                        })

        # Execute Pass 1 (core) and Pass 2 (satellites)
        for f, u in core_faces:
            assign_face_item(f, u, allow_new_cluster=True)
        for f, u in satellite_faces:
            assign_face_item(f, u, allow_new_cluster=False)

        # Pass 3: one reassignment sweep against the final centroids, so the result does not depend on
        # the order faces were seen. A face moves when another cluster is clearly closer (by 0.02).
        u_by_face = {f.id: u for f, u in parsed_faces}
        img_by_face = {f.id: f.image_id for f, _ in parsed_faces}
        owner = {fid: c for c in clusters for fid in c["face_ids"]}
        by_dim: dict[int, list[dict]] = {}
        for c in clusters:
            by_dim.setdefault(len(c["centroid"]), []).append(c)
        moved = False
        for dim, dim_clusters in by_dim.items():
            if len(dim_clusters) < 2:
                continue
            cents = np.stack([c["centroid"] for c in dim_clusters])
            fids = [fid for fid, u in u_by_face.items() if len(u) == dim and fid in owner]
            for k in range(0, len(fids), 2048):
                chunk = fids[k:k + 2048]
                sims = np.stack([u_by_face[fid] for fid in chunk]) @ cents.T
                for row, fid in zip(sims, chunk):
                    cur = owner[fid]
                    thr = effective_threshold(threshold, dim)
                    cur_sim = float(np.dot(u_by_face[fid], cur["centroid"]))
                    order = np.argsort(-row)
                    for j in order[:5]:
                        cand = dim_clusters[int(j)]
                        score = float(row[j])
                        if cand is cur or score < thr or score < cur_sim + 0.02:
                            break
                        if img_by_face[fid] in cand["image_ids"]:
                            continue  # cannot-link: someone else from this photo is already there
                        cur["face_ids"].remove(fid)
                        cur["image_ids"].discard(img_by_face[fid])
                        cand["face_ids"].append(fid)
                        cand["image_ids"].add(img_by_face[fid])
                        owner[fid] = cand
                        moved = True
                        break
        if moved:
            for c in clusters:
                members = [u_by_face[fid] for fid in c["face_ids"] if fid in u_by_face]
                if c.get("is_existing"):
                    total = c["anchor_centroid"] * c.get("anchor_count", 0) + (np.sum(members, axis=0) if members else 0)
                elif members:
                    total = np.sum(members, axis=0)
                else:
                    continue
                norm = float(np.linalg.norm(total))
                if norm > 1e-6:
                    c["centroid"] = total / norm
                base = [c["cluster_obj"].name] if c.get("is_existing") and c["cluster_obj"].name else []
                c["names"] = base + [face_custom_name[fid] for fid in c["face_ids"] if fid in face_custom_name]
            clusters = [c for c in clusters if c["face_ids"] or c.get("is_existing")]

        # 3. Apply assignments only for pipeline faces
        created_count = 0
        total_assigned = 0
        face_by_id = {f.id: f for f in faces}

        existing_to_count: list[M.PersonCluster] = []
        for c_data in clusters:
            if c_data.get("is_existing") and c_data["cluster_obj"] is not None:
                c = c_data["cluster_obj"]
                for fid in c_data["face_ids"]:
                    face_obj = face_by_id.get(fid)
                    if face_obj is not None:
                        face_obj.person_id = c.id
                        total_assigned += 1
                existing_to_count.append(c)
                centroid_list = c_data["centroid"].tolist()
                c.centroid_json = quantize_embedding(centroid_list)
                c.yearly_json = encode_yearly(c_data.get("yearly_medoids") or {}, c_data.get("yearly_counts") or {})
            elif c_data["face_ids"]:
                cluster_name = None
                if c_data["names"]:
                    cluster_name = Counter(c_data["names"]).most_common(1)[0][0]
                centroid_list = c_data["centroid"].tolist()
                new_c = M.PersonCluster(
                    centroid_json=quantize_embedding(centroid_list),
                    yearly_json=encode_yearly(c_data.get("yearly_medoids") or {}, c_data.get("yearly_counts") or {}),
                    face_count=len(c_data["face_ids"]),
                    name=cluster_name,
                    stage=cluster_stage,
                )
                s.add(new_c)
                s.flush()
                for fid in c_data["face_ids"]:
                    face_obj = face_by_id.get(fid)
                    if face_obj is not None:
                        face_obj.person_id = new_c.id
                        total_assigned += 1
                created_count += 1
        # Count only after every face has its final person, or a face still to move counts twice.
        s.flush()
        for c in existing_to_count:
            rem = s.exec(
                select(func.count(M.FaceDetection.id))
                .where(M.FaceDetection.person_id == c.id, M.FaceDetection.ignored == False)
            ).one()
            c.face_count = int(rem or 0)

        # Prune any empty unnamed ghost clusters
        has_any_face = s.exec(select(M.FaceDetection.id).limit(1)).first() is not None
        if has_any_face:
            active_pids = select(M.FaceDetection.person_id).where(
                M.FaceDetection.person_id.is_not(None),
                M.FaceDetection.ignored == False,
            )
            empty_unnamed = s.exec(
                select(M.PersonCluster)
                .where(M.PersonCluster.name.is_(None))
                .where((M.PersonCluster.face_count <= 0) | col(M.PersonCluster.id).not_in(active_pids))
            ).all()
        else:
            empty_unnamed = s.exec(
                select(M.PersonCluster)
                .where(M.PersonCluster.face_count <= 0)
                .where(M.PersonCluster.name.is_(None))
            ).all()
        for p in empty_unnamed:
            s.delete(p)

        s.commit()

    # 4. Run post-clustering consolidation to merge convergent clusters
    try:
        dim_seen = max((len(item[1]) for item in parsed_faces), default=0)
        base = effective_threshold(threshold, dim_seen)
        consolidate_thresh = base if dim_seen >= 512 else max(0.60, base + 0.05)
        consolidate_clusters(db_path, threshold=consolidate_thresh, stage=stage)
    except Exception:
        pass

    return {"clusters_created": created_count, "faces_clustered": total_assigned}





def detach_faces(s, image_id: int) -> int:
    """Take a photo's faces out of their people (it left the library); empty unnamed people go.

    The faces keep their ignored flag: only a person the user deleted marks faces ignored."""
    n = 0
    for f in s.exec(select(M.FaceDetection).where(M.FaceDetection.image_id == image_id)).all():
        if f.person_id is not None:
            c = s.get(M.PersonCluster, f.person_id)
            if c is not None:
                c.face_count = max(0, c.face_count - 1)
                if c.face_count <= 0 and not c.name:
                    s.delete(c)
        f.person_id = None  # detached, not rejected: Reanalyse and Restore may cluster it again
        n += 1
    s.flush()
    return n


def attach_faces_to_library(s, image_id: int, threshold: float = 0.64) -> dict:
    """Match a restored photo's faces to the people already in the library.

    A face joins the closest library person at or above ``threshold``, otherwise it starts a new
    person. Two faces of one photo never share a person. Faces leave their trash person, and a
    trash person left with no faces is deleted. Faces ignored while trashed come back.
    """
    from .clustering.face_identity import ClusterCentroid, assign_cluster, update_centroid

    faces = [f for f in s.exec(select(M.FaceDetection).where(M.FaceDetection.image_id == image_id)).all()
             if f.embedding_json]
    faces.sort(key=lambda f: f.face_sharpness or 0.0, reverse=True)
    centroids = {c.id: ClusterCentroid(c.id, dequantize_embedding(c.centroid_json) or [], c.face_count)
                 for c in s.exec(select(M.PersonCluster).where(M.PersonCluster.stage == "library")).all()
                 if c.centroid_json and c.face_count > 0}
    claimed: set[int] = set()
    matched = created = 0
    for f in faces:
        emb = dequantize_embedding(f.embedding_json)
        if not emb:
            continue
        old_pid = f.person_id
        pid = assign_cluster(emb, [c for c in centroids.values() if c.person_id not in claimed], threshold)
        if pid is not None:
            c = centroids[pid]
            person = s.get(M.PersonCluster, pid)
            new_centroid = update_centroid(c.centroid, c.face_count, emb)
            person.centroid_json = quantize_embedding(new_centroid)
            person.face_count = c.face_count + 1
            centroids[pid] = ClusterCentroid(pid, new_centroid, person.face_count)
            matched += 1
        else:
            person = M.PersonCluster(name=None, face_count=1, stage="library",
                                     centroid_json=quantize_embedding(emb))
            s.add(person)
            s.flush()
            pid = person.id
            centroids[pid] = ClusterCentroid(pid, list(emb), 1)
            created += 1
        claimed.add(pid)
        if old_pid is not None and old_pid != pid:
            old = s.get(M.PersonCluster, old_pid)
            if old is not None:
                old.face_count = max(0, old.face_count - 1)
                if old.stage == "trash" and old.face_count <= 0:
                    s.flush()
                    f.person_id = pid
                    s.delete(old)
        f.person_id = pid
        f.ignored = False
    s.flush()
    return {"matched": matched, "created": created}
