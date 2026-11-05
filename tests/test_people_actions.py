"""People business logic: list, rename (with same-name merge), and single-face reassign."""
from __future__ import annotations

import pytest
from sqlmodel import select

from photocure.db import models as M
from photocure.db.repository import quantize_embedding
from photocure.people_actions import ActionError, bulk_delete_people, delete_person, list_people, reassign_face, recluster_people, rename_person


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

    # Once a2 is named, it should stop showing merge suggestions
    repo.rename_person(a2.id, "Alice 2")
    session.commit()
    items_after_name = list_people(cfg.db_path, sort="similarity")
    a2_named_dict = next(it for it in items_after_name if it["id"] == a2.id)
    assert "similar_to" not in a2_named_dict


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


def test_list_people_prunes_and_excludes_empty_clusters(cfg, session, repo):
    # Create an active cluster with 1 face
    active = repo.create_person_cluster([1.0, 0.0])
    repo.update_person_cluster(active.id, [1.0, 0.0], 1)
    # Create an unnamed orphan cluster with face_count 0
    orphan = repo.create_person_cluster([0.0, 1.0])
    orphan.face_count = 0
    repo.s.add(orphan)
    session.commit()

    orphan_id = orphan.id
    items = list_people(cfg.db_path)
    assert len(items) == 1
    assert items[0]["id"] == active.id

    # Verify orphan cluster was pruned from DB
    session.expire_all()
    assert session.get(M.PersonCluster, orphan_id) is None


def test_bulk_delete_people_removes_clusters_and_unassigns_faces(cfg, session, repo):
    _image(session, 1)
    _image(session, 2)
    p1 = repo.create_person_cluster([1.0, 0.0])
    p2 = repo.create_person_cluster([0.0, 1.0])
    p3 = repo.create_person_cluster([0.5, 0.5])
    f1 = _face(session, 1, person_id=p1.id)
    f2 = _face(session, 2, person_id=p2.id)
    f3 = _face(session, 2, person_id=p3.id)
    session.commit()

    p1_id, p2_id, p3_id = p1.id, p2.id, p3.id
    res = bulk_delete_people(cfg.db_path, [p1_id, p2_id])
    assert res["deleted_count"] == 2

    session.expire_all()
    assert session.get(M.PersonCluster, p1_id) is None
    assert session.get(M.PersonCluster, p2_id) is None
    assert session.get(M.PersonCluster, p3_id) is not None

    rf1 = session.get(M.FaceDetection, f1.id)
    rf2 = session.get(M.FaceDetection, f2.id)
    rf3 = session.get(M.FaceDetection, f3.id)
    assert rf1.person_id is None
    assert rf2.person_id is None
    assert rf3.person_id == p3.id






def test_recluster_temporal_boost_capped(cfg, session, repo):
    from datetime import datetime, timezone
    from tests.test_face_identity import mix, unit
    when = datetime(2024, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    for iid in (1, 2):
        session.add(M.ImageRecord(id=iid, path=f"/{iid}.jpg", filename=f"{iid}.jpg", burst_id=7, taken_at=when))
    session.flush()
    _face(session, 1, embedding=unit(1))
    _face(session, 2, embedding=mix(unit(1), unit(2), cos=0.30))  # same burst, clearly not the same face
    session.commit()

    res = recluster_people(cfg.db_path, threshold=0.35)

    assert res["clusters_created"] == 2


def _clusters_with_centroids(n: int, dim: int = 512) -> list[M.PersonCluster]:
    import numpy as np
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        v = rng.standard_normal(dim)
        v /= np.linalg.norm(v)
        out.append(M.PersonCluster(id=i + 1, name=("Named %d" % i) if i % 10 == 0 else None,
                                   face_count=1 + i % 7, centroid_json=quantize_embedding(v.tolist())))
    return out


def test_find_merge_matches_is_vectorized(monkeypatch):
    import numpy as np
    from photocure import people_actions
    real_dot, calls = np.dot, []

    def counting_dot(*a, **k):
        calls.append(1)
        return real_dot(*a, **k)

    monkeypatch.setattr(np, "dot", counting_dot)  # numpy is imported inside the function
    people_actions._find_merge_matches(_clusters_with_centroids(300))
    assert len(calls) <= 5, len(calls)


def test_list_people_3000_clusters_under_2s(cfg, session):
    import time
    for c in _clusters_with_centroids(3000):
        session.add(c)
    session.commit()
    t0 = time.perf_counter()
    items = list_people(cfg.db_path, sort="similarity")
    elapsed = time.perf_counter() - t0
    assert len(items) == 3000
    assert elapsed < 2.0, elapsed


def test_arrange_by_similarity_returns_order_only(cfg, session, repo):
    from photocure.people_actions import _arrange_by_similarity
    a = repo.create_person_cluster([1.0, 0.0]); repo.update_person_cluster(a.id, [1.0, 0.0], 10)
    b = repo.create_person_cluster([0.0, 1.0]); repo.update_person_cluster(b.id, [0.0, 1.0], 8)
    c = repo.create_person_cluster([0.98, 0.02]); repo.update_person_cluster(c.id, [0.98, 0.02], 3)
    session.commit()
    ordered = _arrange_by_similarity([a, b, c])
    assert [x.id for x in ordered] == [a.id, c.id, b.id]


def _three_photos_two_people(session, repo, cos_between=0.6):
    """Clusters A and B that look alike (cos ``cos_between``) and both appear in photo 7."""
    from tests.test_face_identity import mix, unit
    va, vb = unit(11), mix(unit(11), unit(12), cos=cos_between)
    for iid in (7, 8, 9):
        _image(session, iid)
    a = repo.create_person_cluster(va); repo.update_person_cluster(a.id, va, 2)
    b = repo.create_person_cluster(vb); repo.update_person_cluster(b.id, vb, 1)
    _face(session, 7, person_id=a.id, embedding=va)
    _face(session, 8, person_id=a.id, embedding=va)
    _face(session, 7, person_id=b.id, embedding=vb)
    session.commit()
    return a, b


def test_cooccurring_clusters_are_never_suggested(cfg, session, repo):
    a, b = _three_photos_two_people(session, repo)
    items = {it["id"]: it for it in list_people(cfg.db_path)}
    assert "similar_to" not in items[b.id]


def test_named_target_loses_to_much_stronger_unnamed(cfg, session, repo):
    from tests.test_face_identity import mix, unit
    base = unit(21)
    alice = repo.create_person_cluster(mix(base, unit(22), cos=0.36)); repo.rename_person(alice.id, "Alice")
    repo.update_person_cluster(alice.id, mix(base, unit(22), cos=0.36), 5)
    big = repo.create_person_cluster(mix(base, unit(23), cos=0.92)); repo.update_person_cluster(big.id, mix(base, unit(23), cos=0.92), 9)
    cand = repo.create_person_cluster(base); repo.update_person_cluster(cand.id, base, 2)
    session.commit()
    items = {it["id"]: it for it in list_people(cfg.db_path)}
    assert items[cand.id]["similar_to"]["target_id"] == big.id


def test_named_target_within_margin_wins(cfg, session, repo):
    from tests.test_face_identity import mix, unit
    base = unit(31)
    alice = repo.create_person_cluster(mix(base, unit(32), cos=0.60)); repo.rename_person(alice.id, "Alice")
    repo.update_person_cluster(alice.id, mix(base, unit(32), cos=0.60), 5)
    big = repo.create_person_cluster(mix(base, unit(33), cos=0.63)); repo.update_person_cluster(big.id, mix(base, unit(33), cos=0.63), 9)
    cand = repo.create_person_cluster(base); repo.update_person_cluster(cand.id, base, 2)
    session.commit()
    items = {it["id"]: it for it in list_people(cfg.db_path)}
    assert items[cand.id]["similar_to"]["target_id"] == alice.id


def test_empty_named_cluster_is_not_a_target(cfg, session, repo):
    from tests.test_face_identity import mix, unit
    base = unit(41)
    _image(session, 70)
    bob = repo.create_person_cluster(mix(base, unit(42), cos=0.9)); repo.rename_person(bob.id, "Bob")
    repo.update_person_cluster(bob.id, mix(base, unit(42), cos=0.9), 0)
    cand = repo.create_person_cluster(base); repo.update_person_cluster(cand.id, base, 2)
    _face(session, 70, person_id=cand.id, embedding=base)
    session.commit()
    items = {it["id"]: it for it in list_people(cfg.db_path)}
    assert items[cand.id].get("similar_to", {}).get("target_id") != bob.id


def _synthetic_identities(session, k=3, per=20, seed=0):
    """k people, ``per`` faces each (within cos ~0.55), one face per photo, plus 5 co-occurring pairs."""
    import numpy as np
    from tests.test_face_identity import mix, unit
    rng = np.random.default_rng(seed)
    truth, faces, next_img = {}, [], 1
    for p in range(k):
        center = unit(100 + p)
        for _ in range(per):
            faces.append((next_img, p, mix(center, unit(int(rng.integers(1000, 10**6))), cos=0.55)))
            next_img += 1
    # co-occurring pairs: two different people in one photo
    for q in range(5):
        img = next_img; next_img += 1
        for p in (q % k, (q + 1) % k):
            faces.append((img, p, mix(unit(100 + p), unit(5000 + q * 7 + p), cos=0.55)))
    for img_id in range(1, next_img):
        _image(session, img_id)
    return faces


def _run_recluster_in_order(cfg, session, repo, faces, order):
    session.exec(M.FaceDetection.__table__.delete())
    session.exec(M.PersonCluster.__table__.delete())
    ids = {}
    for idx in order:
        img, p, emb = faces[idx]
        f = _face(session, img, embedding=emb)
        ids[f.id] = (img, p)
    session.commit()
    recluster_people(cfg.db_path, threshold=0.35)
    session.expire_all()
    out = {}
    for fid, (img, p) in ids.items():
        out[(img, p)] = session.get(M.FaceDetection, fid).person_id
    return out


def _partition(assign):
    groups = {}
    for key, pid in assign.items():
        groups.setdefault(pid, set()).add(key)
    return {frozenset(v) for v in groups.values()}


def test_recluster_is_order_invariant(cfg, session, repo):
    import random
    faces = _synthetic_identities(session)
    order_a = list(range(len(faces)))
    order_b = order_a[:]
    random.Random(3).shuffle(order_b)
    pa = _partition(_run_recluster_in_order(cfg, session, repo, faces, order_a))
    pb = _partition(_run_recluster_in_order(cfg, session, repo, faces, order_b))
    assert pa == pb


def test_recluster_purity_on_synthetic_identities(cfg, session, repo):
    faces = _synthetic_identities(session)
    assign = _run_recluster_in_order(cfg, session, repo, faces, list(range(len(faces))))
    by_cluster = {}
    for (img, p), pid in assign.items():
        by_cluster.setdefault(pid, []).append(p)
    purity = sum(max(v.count(x) for x in set(v)) for v in by_cluster.values()) / len(assign)
    assert purity >= 0.98
    by_person = {}
    for (img, p), pid in assign.items():
        by_person.setdefault(p, []).append(pid)
    completeness = sum(max(v.count(x) for x in set(v)) for v in by_person.values()) / len(assign)
    assert completeness >= 0.95
    # the five co-occurring pairs never share a person
    imgs = {}
    for (img, p), pid in assign.items():
        imgs.setdefault(img, []).append(pid)
    assert all(len(set(v)) == len(v) for v in imgs.values())


def test_recluster_face_count_matches_db_after_faces_move(cfg, session, repo):
    from tests.test_face_identity import mix, unit
    # an anchored library person whose centroid is a; two new faces close to a, one far away
    a = unit(61)
    lib = repo.create_person_cluster(a); repo.rename_person(lib.id, "Lib"); repo.update_person_cluster(lib.id, a, 1)
    session.add(M.ImageRecord(id=90, path="/90.jpg", filename="90.jpg", stage="library")); session.flush()
    _face(session, 90, person_id=lib.id, embedding=a)
    for iid in (91, 92, 93):
        _image(session, iid)
    _face(session, 91, embedding=mix(a, unit(62), cos=0.8))
    _face(session, 92, embedding=mix(a, unit(63), cos=0.8))
    _face(session, 93, embedding=unit(64))
    session.commit()

    recluster_people(cfg.db_path, threshold=0.35)

    session.expire_all()
    for c in session.exec(select(M.PersonCluster)).all():
        actual = session.exec(select(M.FaceDetection).where(M.FaceDetection.person_id == c.id,
                                                            M.FaceDetection.ignored == False)).all()
        assert c.face_count == len(actual), (c.id, c.name, c.face_count, len(actual))
