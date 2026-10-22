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
