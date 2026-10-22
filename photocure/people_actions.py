"""People actions: list/rename clusters and reassign a single face. Shared by the web app and MCP."""
from __future__ import annotations

from pathlib import Path

from sqlmodel import select

from .db import models as M
from .db.repository import PhotoRepository, dequantize_embedding
from .db.session import get_session


class ActionError(ValueError):
    """A refusal with a message fit to show the user and the HTTP status the web app should use."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _as_dict(p: M.PersonCluster) -> dict:
    return {"id": p.id, "name": p.name, "face_count": p.face_count}


def list_people(db_path: Path | str) -> list[dict]:
    with get_session(db_path) as s:
        repo = PhotoRepository(s)
        clusters = repo.all_person_clusters()
        clusters.sort(key=lambda p: (p.face_count, p.id), reverse=True)
        return [_as_dict(p) for p in clusters]


def rename_person(db_path: Path | str, person_id: int, name: str) -> dict:
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
