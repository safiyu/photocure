from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from PIL import Image
from sqlmodel import select

from photocure.clustering.burst_detector import BurstDetector, BurstInput
from photocure.clustering.events import cluster_events
from photocure.clustering.face_identity import ClusterCentroid
from photocure.db import models as M
from photocure.db.repository import quantize_embedding
from photocure.inference.backends import BackendState
from photocure.inference.face_engine import FaceInfo
from photocure.inference.pipeline import InferencePipeline
from photocure.quality.image_loader import load_image
from photocure.rules.taxonomy import parse_taxonomy


def _pipe(cfg):
    return InferencePipeline(cfg, backend=BackendState(
        device="cpu", fp16=False, torch=None, cuda_available=False,
        memory_fraction=0.75, face_backend="yunet", semantic_backend="stub"))


def test_transparent_png_is_not_black(tmp_path):
    p = tmp_path / "t.png"
    Image.fromarray(np.zeros((32, 32, 4), dtype=np.uint8), "RGBA").save(p)
    assert load_image(p).bgr.mean() > 100


def test_retry_downgraded_restores_backends():
    bs = BackendState(device="cpu", fp16=False, torch=None, cuda_available=False,
                      memory_fraction=0.75, face_backend="yunet", semantic_backend="clip")
    bs.semantic_backend = "stub"
    bs.face_backend = "stub"
    bs.retry_downgraded()
    assert (bs.semantic_backend, bs.face_backend) == ("clip", "yunet")


def test_event_rebuild_keeps_user_label(session):
    t0 = datetime(2024, 1, 1, 10, tzinfo=timezone.utc)
    for i in range(3):
        session.add(M.ImageRecord(path=f"/{i}.jpg", filename=f"{i}.jpg",
                                  taken_at=t0 + timedelta(minutes=i), taken_at_source="exif"))
    session.flush()
    assert cluster_events(session) == 1
    ev = session.exec(select(M.EventGroup)).one()
    ev.label = "Beach trip"
    session.flush()
    cluster_events(session)
    assert session.exec(select(M.EventGroup)).one().label == "Beach trip"


def test_burst_ragged_embeddings_and_mixed_tz_do_not_crash():
    t0 = datetime(2024, 1, 1, 10)
    items = [BurstInput(image_id=1, taken_at=t0, embedding=[1.0, 0.0], taken_at_source="exif"),
             BurstInput(image_id=2, taken_at=t0.replace(tzinfo=timezone(timedelta(hours=2))) + timedelta(seconds=1),
                        embedding=[1.0, 0.0], taken_at_source="exif"),
             BurstInput(image_id=3, taken_at=t0 + timedelta(seconds=2), embedding=[1.0],
                        taken_at_source="exif")]
    assert BurstDetector().cluster(items) == [[1, 2]]


def test_taxonomy_rejects_rule_only_class_names():
    with pytest.raises(ValueError):
        parse_taxonomy([{"name": "accidental", "prompts": ["a dark photo"]}])


def test_failed_persist_reverts_cluster_and_dead_cluster_is_skipped(cfg, repo):
    pipe = _pipe(cfg)
    person = repo.create_person_cluster([1.0, 0.0])
    repo.create_person_cluster([0.0, 1.0])  # keeps rowid reuse from masking the deleted id
    centroids = {person.id: ClusterCentroid(person.id, [1.0, 0.0], 1)}
    repo.s.delete(repo.s.get(M.PersonCluster, person.id))  # merged away in the UI
    repo.s.flush()
    f = FaceInfo(0.1, 0.1, 0.3, 0.3, confidence=0.9, face_sharpness=50.0, embedding=[1.0, 0.0])
    ids = pipe._assign_people(repo, centroids, 1, [f])
    assert ids[0] != person.id and person.id not in centroids

    live = repo.s.get(M.PersonCluster, 2)
    centroids = {live.id: ClusterCentroid(live.id, [0.0, 1.0], 1)}
    from photocure.inference.pipeline import _PeopleUndo
    undo = _PeopleUndo()
    f2 = FaceInfo(0.1, 0.1, 0.3, 0.3, confidence=0.9, face_sharpness=50.0, embedding=[0.0, 1.0])
    pipe._assign_people(repo, centroids, 2, [f2], undo)
    assert repo.s.get(M.PersonCluster, live.id).face_count == 2
    pipe._undo_people(repo, centroids, undo)
    assert repo.s.get(M.PersonCluster, live.id).face_count == 1


def test_keep_people_carries_ids_when_embeddings_missing(cfg, repo):
    repo.s.add(M.ImageRecord(id=7, path="/k.jpg", filename="k.jpg"))
    repo.s.flush()
    person = repo.create_person_cluster([1.0, 0.0])
    repo.replace_faces(7, [M.FaceDetection(
        bbox_x1=0, bbox_y1=0, bbox_x2=1, bbox_y2=1, confidence=0.9,
        x1=0.1, y1=0.1, x2=0.3, y2=0.3,
        embedding_json=quantize_embedding([1.0, 0.0]), person_id=person.id)])
    f = FaceInfo(0.1, 0.1, 0.31, 0.3, confidence=0.9)
    assert _pipe(cfg)._keep_people(repo, 7, [f]) == [person.id]
    assert f.embedding is not None
