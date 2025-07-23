"""People business logic: list, rename (with same-name merge), and single-face reassign."""
from __future__ import annotations

import pytest
from sqlmodel import select

from photocure.db import models as M
from photocure.db.repository import quantize_embedding
from photocure.people_actions import ActionError, list_people, reassign_face, rename_person


def _face(session, image_id: int, person_id: int | None = None,
         embedding: list[float] | None = None) -> M.FaceDetection:
    f = M.FaceDetection(
        image_id=image_id, bbox_x1=0, bbox_y1=0, bbox_x2=1, bbox_y2=1, confidence=0.9,
        x1=0.1, y1=0.1, x2=0.3, y2=0.3, person_id=person_id,
        embedding_json=quantize_embedding(embedding) if embedding else None,
    )
    session.add(f)
    session.flush()
    return f


def _image(session, image_id: int) -> None:
    session.add(M.ImageRecord(id=image_id, path=f"/{image_id}.jpg", filename=f"{image_id}.jpg"))
    session.flush()


def test_list_people_reports_name_and_face_count(cfg, session, repo):
    a = repo.create_person_cluster([1.0])
    repo.rename_person(a.id, "Alice")
    repo.update_person_cluster(a.id, [1.0], 5)
    b = repo.create_person_cluster([0.0, 1.0])
    repo.update_person_cluster(b.id, [0.0, 1.0], 1)
    session.commit()  # people_actions opens its own session, so prior writes must be durable
    items = list_people(cfg.db_path)
    by_id = {it["id"]: it for it in items}
    assert by_id[a.id] == {"id": a.id, "name": "Alice", "face_count": 5}
    assert by_id[b.id] == {"id": b.id, "name": None, "face_count": 1}
    assert items[0]["id"] == a.id  # most faces first


def test_rename_person_merges_into_an_existing_same_name_cluster(cfg, session, repo):
    _image(session, 1)
    _image(session, 2)
    keep = repo.create_person_cluster([1.0])
    repo.rename_person(keep.id, "Safi")
    repo.update_person_cluster(keep.id, [1.0], 1)
    dup = repo.create_person_cluster([0.9])
    dup_id = dup.id
    repo.update_person_cluster(dup_id, [0.9], 2)
    _face(session, 1, person_id=dup_id)
    _face(session, 2, person_id=dup_id)
    session.commit()

    result = rename_person(cfg.db_path, dup_id, "  safi ")

    session.expire_all()
    assert result["id"] == keep.id and result["face_count"] == 3
    assert session.exec(select(M.PersonCluster).where(M.PersonCluster.id == dup_id)).first() is None
    faces = session.exec(
        select(M.FaceDetection).where(M.FaceDetection.image_id.in_([1, 2]))
    ).all()
    assert all(f.person_id == keep.id for f in faces)


def test_rename_unknown_person_raises_404(cfg, repo):
    with pytest.raises(ActionError) as exc:
        rename_person(cfg.db_path, 999, "x")
    assert exc.value.status == 404


def test_rename_person_rejects_a_blank_name(cfg, repo):
    p = repo.create_person_cluster([1.0])
    repo.s.commit()
    with pytest.raises(ActionError, match="blank"):
        rename_person(cfg.db_path, p.id, "   ")


def test_reassign_face_to_an_existing_person_updates_both_face_counts(cfg, session, repo):
    _image(session, 10)
    old = repo.create_person_cluster([1.0])
    repo.update_person_cluster(old.id, [1.0], 1)
    new = repo.create_person_cluster([0.0])
    repo.update_person_cluster(new.id, [0.0], 1)
    f = _face(session, 10, person_id=old.id, embedding=[1.0])
    session.commit()

    result = reassign_face(cfg.db_path, f.id, person_id=new.id)

    session.expire_all()
    assert result["person_id"] == new.id
    assert session.get(M.PersonCluster, old.id).face_count == 0
    assert session.get(M.PersonCluster, new.id).face_count == 2


def test_reassign_face_to_a_new_name_creates_a_person(cfg, session, repo):
    _image(session, 20)
    f = _face(session, 20, person_id=None, embedding=[0.5, 0.5])
    session.commit()

    result = reassign_face(cfg.db_path, f.id, new_name="Mohamed")

    session.expire_all()
    created = session.get(M.PersonCluster, result["person_id"])
    assert created.name == "Mohamed" and created.face_count == 1


def test_reassign_face_requires_exactly_one_of_person_id_or_new_name(cfg, session, repo):
    _image(session, 30)
    f = _face(session, 30)
    with pytest.raises(ActionError):
        reassign_face(cfg.db_path, f.id)
    with pytest.raises(ActionError):
        reassign_face(cfg.db_path, f.id, person_id=1, new_name="x")


def test_reassign_face_rejects_a_blank_new_name(cfg, session, repo):
    _image(session, 31)
    f = _face(session, 31)
    session.commit()
    with pytest.raises(ActionError, match="blank"):
        reassign_face(cfg.db_path, f.id, new_name="   ")
