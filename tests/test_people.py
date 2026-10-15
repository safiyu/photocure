"""Person clustering from stored face embeddings."""
from __future__ import annotations

import numpy as np
from sqlmodel import select

from photocure.clustering.people import cluster_embeddings, cluster_people
from photocure.db.models import FaceDetection, ImageRecord, PersonCluster
from photocure.db.repository import quantize_embedding


def _unit(v):
    return v / np.linalg.norm(v)


def _identities(n=3, dim=128, seed=0):
    rng = np.random.default_rng(seed)
    return [_unit(rng.standard_normal(dim)) for _ in range(n)]


def _noisy(v, rng, eps=0.15):
    return _unit((1 - eps) * v + eps * _unit(rng.standard_normal(v.shape)))


def _face(session, image, vec, conf=0.95):
    f = FaceDetection(image_id=image.id, bbox_x1=0, bbox_y1=0, bbox_x2=10, bbox_y2=10,
                      confidence=conf,
                      embedding_json=quantize_embedding(vec.tolist()) if vec is not None else None)
    session.add(f)
    session.flush()
    return f


def _image(session, i):
    r = ImageRecord(path=f"/lib/{i}.jpg", filename=f"{i}.jpg")
    session.add(r)
    session.flush()
    return r


def _populate(session, counts=(5, 4, 3), seed=1):
    rng = np.random.default_rng(seed)
    ids = _identities(len(counts))
    faces, owner = [], []
    n = 0
    for who, c in enumerate(counts):
        for _ in range(c):
            n += 1
            faces.append(_face(session, _image(session, n), _noisy(ids[who], rng)))
            owner.append(who)
    return faces, owner


def test_cluster_embeddings_separates_identities():
    rng = np.random.default_rng(0)
    ids = _identities(3)
    vecs = np.stack([_noisy(ids[i % 3], rng) for i in range(30)])
    labels = cluster_embeddings(vecs, 0.5)
    assert len(set(labels.tolist())) == 3
    for i in range(30):
        assert labels[i] == labels[i % 3]


def test_cluster_embeddings_is_order_independent_in_outcome():
    rng = np.random.default_rng(1)
    ids = _identities(3)
    vecs = np.stack([_noisy(ids[i % 3], rng) for i in range(24)])
    perm = np.random.default_rng(9).permutation(24)
    a = cluster_embeddings(vecs, 0.5)
    b = cluster_embeddings(vecs[perm], 0.5)
    assert len(set(a.tolist())) == len(set(b.tolist())) == 3


def test_cluster_embeddings_empty_and_single():
    assert len(cluster_embeddings(np.empty((0, 8), dtype=np.float32))) == 0
    assert cluster_embeddings(_unit(np.ones(8))[None, :]).tolist() == [0]


def test_cluster_people_writes_clusters_and_person_ids(session):
    faces, owner = _populate(session)
    assert cluster_people(session) == 3
    people = session.exec(select(PersonCluster)).all()
    assert sorted(p.face_count for p in people) == [3, 4, 5]
    by_owner = {}
    for f, who in zip(faces, owner):
        by_owner.setdefault(who, set()).add(f.person_id)
    assert all(len(v) == 1 for v in by_owner.values())
    assert len({next(iter(v)) for v in by_owner.values()}) == 3


def test_faces_below_confidence_or_without_embedding_are_left_out(session):
    faces, _ = _populate(session, counts=(4,))
    rng = np.random.default_rng(3)
    low = _face(session, _image(session, 100), _noisy(_identities(1)[0], rng), conf=0.30)
    none = _face(session, _image(session, 101), None)
    assert cluster_people(session) == 1
    assert low.person_id is None and none.person_id is None
    assert all(f.person_id is not None for f in faces)


def test_recluster_keeps_person_ids_and_names(session):
    faces, owner = _populate(session)
    cluster_people(session)
    biggest = session.exec(select(PersonCluster).order_by(PersonCluster.face_count.desc())).first()
    biggest.name = "Alice"
    ids_before = {f.id: f.person_id for f in faces}
    # Add one more face of the same person and recluster.
    rng = np.random.default_rng(8)
    extra_owner = owner[0]
    _face(session, _image(session, 200), _noisy(_identities(3)[extra_owner], rng, 0.1))
    cluster_people(session)
    ids_after = {f.id: f.person_id for f in faces}
    assert ids_before == ids_after
    alice = session.get(PersonCluster, biggest.id)
    assert alice.name == "Alice" and alice.face_count == 6


def test_stale_unnamed_clusters_are_removed(session):
    faces, _ = _populate(session, counts=(3, 3))
    cluster_people(session)
    for f in faces[3:]:
        session.delete(f)
    session.flush()
    assert cluster_people(session) == 1
    assert len(session.exec(select(PersonCluster)).all()) == 1


def test_named_cluster_survives_losing_all_faces(session):
    faces, _ = _populate(session, counts=(3, 3))
    cluster_people(session)
    gone = faces[0].person_id
    p = session.get(PersonCluster, gone)
    p.name = "Bob"
    for f in [f for f in faces if f.person_id == gone]:
        session.delete(f)
    session.flush()
    cluster_people(session)
    kept = session.get(PersonCluster, gone)
    assert kept is not None and kept.name == "Bob" and kept.face_count == 0


def test_no_faces_gives_zero_people(session):
    assert cluster_people(session) == 0
