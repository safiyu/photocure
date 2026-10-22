"""People business logic: list, rename (with same-name merge), and single-face reassign."""
from __future__ import annotations

import pytest
from sqlmodel import select

from photocure.db import models as M
from photocure.db.repository import quantize_embedding
from photocure.people_actions import ActionError, delete_person, list_people, reassign_face, recluster_people, rename_person


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
    assert by_id[a.id] == {"id": a.id, "name": "Alice", "face_count": 5, "sample_face_id": None}
    assert by_id[b.id] == {"id": b.id, "name": None, "face_count": 1, "sample_face_id": None}
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


def test_delete_person_unlinks_faces_and_removes_cluster(cfg, session, repo):
    _image(session, 40)
    p = repo.create_person_cluster([1.0, 0.0])
    pid = p.id
    repo.rename_person(pid, "To Delete")
    f = _face(session, 40, person_id=pid)
    session.commit()

    res = delete_person(cfg.db_path, pid)
    assert res == {"id": pid, "deleted": True}

    session.expire_all()
    assert session.get(M.PersonCluster, pid) is None
    refreshed_face = session.get(M.FaceDetection, f.id)
    assert refreshed_face.person_id is None


def test_arrange_similar_faces_together(cfg, session, repo):
    # Cluster A1 (Alice) and A2 (Alice dup) have high similarity ([1.0, 0.05])
    # Cluster B1 (Bob) has completely different vector ([0.0, 1.0])
    # B1 has higher face count than A2, but A2 should be placed adjacent to A1
    a1 = repo.create_person_cluster([1.0, 0.0])
    repo.rename_person(a1.id, "Alice")
    repo.update_person_cluster(a1.id, [1.0, 0.0], 10)

    b1 = repo.create_person_cluster([0.0, 1.0])
    repo.rename_person(b1.id, "Bob")
    repo.update_person_cluster(b1.id, [0.0, 1.0], 8)

    a2 = repo.create_person_cluster([0.98, 0.02])
    repo.update_person_cluster(a2.id, [0.98, 0.02], 3)
    session.commit()

    items = list_people(cfg.db_path, sort="similarity")
    ids = [it["id"] for it in items]
    # In similarity sort: a1 is followed immediately by similar a2, then b1
    assert ids == [a1.id, a2.id, b1.id]
    # a2 should have a similar_to match pointing to a1
    a2_dict = next(it for it in items if it["id"] == a2.id)
    assert "similar_to" in a2_dict
    assert a2_dict["similar_to"]["target_id"] == a1.id
    assert a2_dict["similar_to"]["score"] >= 0.90


def test_recluster_people_separates_distinct_individuals(cfg, session, repo):
    _image(session, 1)
    _image(session, 2)
    _image(session, 3)

    # Person 1 (Boy) with embedding [1.0, 0.0] in image 1 and 2
    f1 = _face(session, 1, embedding=[1.0, 0.0])
    f2 = _face(session, 2, embedding=[0.95, 0.05])
    # Person 2 (Girl) with embedding [0.0, 1.0] in image 2 and 3
    f3 = _face(session, 2, embedding=[0.0, 1.0])
    f4 = _face(session, 3, embedding=[0.05, 0.95])

    # Suppose an old corrupted cluster had all 4 faces assigned to it
    corrupt = repo.create_person_cluster([0.5, 0.5])
    repo.rename_person(corrupt.id, "Corrupted")
    repo.update_person_cluster(corrupt.id, [0.5, 0.5], 4)
    for f in (f1, f2, f3, f4):
        f.person_id = corrupt.id
    session.commit()

    res = recluster_people(cfg.db_path, threshold=0.64)
    assert res["clusters_created"] == 2
    assert res["faces_clustered"] == 4

    session.expire_all()
    rf1 = session.get(M.FaceDetection, f1.id)
    rf2 = session.get(M.FaceDetection, f2.id)
    rf3 = session.get(M.FaceDetection, f3.id)
    rf4 = session.get(M.FaceDetection, f4.id)

    assert rf1.person_id == rf2.person_id
    assert rf3.person_id == rf4.person_id
    assert rf1.person_id != rf3.person_id


