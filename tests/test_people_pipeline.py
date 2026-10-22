"""Pipeline wiring: face embeddings persisted, clustered into people, and
remapped across a re-analysis so backfill never loses an identity."""
from __future__ import annotations

from photocure.clustering.face_identity import ClusterCentroid
from photocure.db import models as M
from photocure.db.repository import quantize_embedding
from photocure.inference.backends import BackendState
from photocure.inference.face_engine import FaceInfo
from photocure.inference.pipeline import InferencePipeline


def _yunet_pipeline(cfg):
    return InferencePipeline(cfg, backend=BackendState(
        device="cpu", fp16=False, torch=None, cuda_available=False,
        memory_fraction=0.75, face_backend="yunet", semantic_backend="stub"))


def test_two_photos_of_the_same_person_land_in_one_cluster(cfg, repo):
    pipeline = _yunet_pipeline(cfg)
    centroids: dict[int, ClusterCentroid] = {}
    f1 = FaceInfo(0.1, 0.1, 0.3, 0.3, confidence=0.9, face_sharpness=50.0, embedding=[1.0, 0.0])
    f2 = FaceInfo(0.1, 0.1, 0.3, 0.3, confidence=0.9, face_sharpness=50.0, embedding=[0.99, 0.01])
    ids1 = pipeline._assign_people(repo, centroids, 101, [f1])
    ids2 = pipeline._assign_people(repo, centroids, 102, [f2])
    assert ids1[0] is not None and ids1 == ids2
    clusters = repo.all_person_clusters()
    assert len(clusters) == 1 and clusters[0].face_count == 2


def test_a_different_persons_face_gets_a_new_cluster(cfg, repo):
    pipeline = _yunet_pipeline(cfg)
    centroids: dict[int, ClusterCentroid] = {}
    f1 = FaceInfo(0.1, 0.1, 0.3, 0.3, confidence=0.9, face_sharpness=50.0, embedding=[1.0, 0.0])
    f2 = FaceInfo(0.1, 0.1, 0.3, 0.3, confidence=0.9, face_sharpness=50.0, embedding=[0.0, 1.0])
    ids1 = pipeline._assign_people(repo, centroids, 201, [f1])
    ids2 = pipeline._assign_people(repo, centroids, 202, [f2])
    assert ids1[0] != ids2[0]
    assert len(repo.all_person_clusters()) == 2


def test_reanalysis_preserves_person_id_via_remap(cfg, repo):
    pipeline = _yunet_pipeline(cfg)
    repo.s.add(M.ImageRecord(id=301, path="/p.jpg", filename="p.jpg"))
    repo.s.flush()
    person = repo.create_person_cluster([1.0, 0.0])
    repo.replace_faces(301, [M.FaceDetection(
        bbox_x1=0, bbox_y1=0, bbox_x2=1, bbox_y2=1, confidence=0.9,
        x1=0.1, y1=0.1, x2=0.3, y2=0.3,
        embedding_json=quantize_embedding([1.0, 0.0]), person_id=person.id,
    )])
    centroids = {person.id: ClusterCentroid(person.id, [1.0, 0.0], 1)}
    new_face = FaceInfo(0.1, 0.1, 0.3, 0.3, confidence=0.9, face_sharpness=50.0, embedding=[0.99, 0.01])
    result = pipeline._assign_people(repo, centroids, 301, [new_face])
    assert result[0] == person.id
    # A remap is a re-detection of the same occurrence, not a new one: the count must not grow.
    assert repo.all_person_clusters()[0].face_count == 1


def test_bumping_schema_version_requeues_already_inferred_photos(cfg, repo):
    repo.s.add(M.ImageRecord(path="/a.jpg", filename="a.jpg", infer_status="inferred",
                             faces_schema_version=1))
    repo.s.flush()
    pipeline = _yunet_pipeline(cfg)
    assert cfg.face.schema_version == 2
    recs = pipeline._select_records(repo, None, False, 0)
    assert len(recs) == 1


def test_reanalysis_that_misses_remap_but_rejoins_the_same_cluster_does_not_inflate_the_count(cfg, repo):
    # Remap only compares against this photo's own prior embedding and can miss (stricter
    # threshold) even when the new embedding still matches the cluster's global centroid.
    # That is still the same occurrence continuing, not a second one for this cluster.
    pipeline = _yunet_pipeline(cfg)
    repo.s.add(M.ImageRecord(id=501, path="/q.jpg", filename="q.jpg"))
    repo.s.flush()
    person = repo.create_person_cluster([1.0, 0.0])
    repo.update_person_cluster(person.id, [0.9, 0.1], 2)  # centroid of 2 prior faces
    repo.replace_faces(501, [M.FaceDetection(
        bbox_x1=0, bbox_y1=0, bbox_x2=1, bbox_y2=1, confidence=0.9,
        x1=0.1, y1=0.1, x2=0.3, y2=0.3,
        embedding_json=quantize_embedding([1.0, 0.0]), person_id=person.id,
    )])
    centroids = {person.id: ClusterCentroid(person.id, [0.9, 0.1], 2)}
    # cos(old photo embedding, new) = 0.30 (< 0.5 remap threshold: remap misses)
    # cos(cluster centroid, new) = 0.40 (>= 0.363 join threshold: fresh clustering still hits)
    new_face = FaceInfo(0.1, 0.1, 0.3, 0.3, confidence=0.9, face_sharpness=50.0, embedding=[0.3, 0.95])
    result = pipeline._assign_people(repo, centroids, 501, [new_face])
    assert result[0] == person.id
    assert repo.all_person_clusters()[0].face_count == 2


def test_an_identity_lost_on_reanalysis_decrements_its_cluster(cfg, repo):
    # The old face's person_id is not reclaimed by any new face (neither remap nor fresh
    # clustering matches it): this photo no longer has that occurrence.
    pipeline = _yunet_pipeline(cfg)
    repo.s.add(M.ImageRecord(id=601, path="/r.jpg", filename="r.jpg"))
    repo.s.flush()
    person = repo.create_person_cluster([1.0, 0.0])
    repo.update_person_cluster(person.id, [1.0, 0.0], 2)
    repo.replace_faces(601, [M.FaceDetection(
        bbox_x1=0, bbox_y1=0, bbox_x2=1, bbox_y2=1, confidence=0.9,
        x1=0.1, y1=0.1, x2=0.3, y2=0.3,
        embedding_json=quantize_embedding([1.0, 0.0]), person_id=person.id,
    )])
    centroids = {person.id: ClusterCentroid(person.id, [1.0, 0.0], 2)}
    # Completely different embedding: no face resolves to `person` this time.
    new_face = FaceInfo(0.1, 0.1, 0.3, 0.3, confidence=0.9, face_sharpness=50.0, embedding=[0.0, 1.0])
    result = pipeline._assign_people(repo, centroids, 601, [new_face])
    assert result[0] != person.id
    clusters = {c.id: c.face_count for c in repo.all_person_clusters()}
    assert clusters[person.id] == 1


def test_two_faces_in_the_same_photo_never_share_a_person_id(cfg, repo):
    pipeline = _yunet_pipeline(cfg)
    existing = repo.create_person_cluster([1.0, 0.0])
    centroids = {existing.id: ClusterCentroid(existing.id, [1.0, 0.0], 1)}
    f1 = FaceInfo(0.0, 0.0, 0.2, 0.2, confidence=0.9, face_sharpness=50.0, embedding=[1.0, 0.0])
    f2 = FaceInfo(0.3, 0.3, 0.5, 0.5, confidence=0.9, face_sharpness=50.0, embedding=[0.99, 0.01])
    result = pipeline._assign_people(repo, centroids, 401, [f1, f2])
    assert result[0] is not None and result[1] is not None
    assert result[0] != result[1]
    assert len(repo.all_person_clusters()) == 2
